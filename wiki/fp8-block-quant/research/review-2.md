<!-- review-meta
round: 2
page: wiki/fp8-block-quant/index.html
reviewed_content_sha256: f4551c725df52ce4
-->
# FP8 块量化审查记录（第 2 轮）

- 页面版本：048239d5bb8fa3b04df4adc9cee5ff01e8c90b45
- 审查时间：2026-09-13 19:01
- 审查者：独立子代理（未参与写作与前序轮次）
- 页面类型：concept（依据 `<meta name="dojo:type" content="concept">`），适用 `guides/concept/check.md`
- 已完整阅读章节：核心问题（含 5 个解答折叠块）、常见误解、1. E4M3 的位布局（含 SVG 图注、代码折叠块、本章问题）、2. 两种格式与特殊值的取舍（含补充折叠块）、3. 缩放：把张量对准可表示区间、4. 分块：128×128 与 1×128（含 SVG 图注、代码折叠块、2 个本章问题）、5. 配套策略与实证（含 2 个本章问题）、来源与范围说明（C/F/N/构造示例/类比边界/简化条件全部条目）
- 来源核对方式：WebFetch 抓 FP8 论文（arXiv:2209.05433）PDF 与 DeepSeek-V3 报告（arXiv:2412.19437）PDF 逐页阅读；curl 抓 transformers v4.57.6 `quantizer_finegrained_fp8.py` 原文；arXiv 与 GitHub 原文片段见下；页面内两段 Python 代码在本机 torch 2.8.0 实跑核对输出

## 核对结论（已核对无误的部分，供复验参考）

- **FP8 论文 Table 1**：E4M3 bias 7 / E5M2 bias 15；`Max normal S.1111.110₂ = 1.75 * 2^8 = 448`；`Min normal S.0001.000₂ = 2^-6`；`Min subnorm S.0000.001₂ = 2^-9`；`E5M2 Max normal = 57,344`、`Min normal = 2^-14`、NaN 三个位模式、有 Infinities。页 §1、§2 表格、SVG 解码例（$2^{15-7}\times(1+6/8)=448$）全部一致。
- **FP8 论文 §3.1**：`We gain the representation of seven more magnitudes (256, 288, 320, 352, 384, 416, 448)... The maximum representable magnitude without this modification would be 240`、`extends the dynamic range by one extra power of 2, from 17 to 18 binades`、`we could gain one additional representable magnitude, 480`、E5M2 侧 `only 3 additional magnitude values... when E5M2 already provides 32 (compared to E4M3's 17 without the adjustment)`。页 §2「240→448（256 到 448，7 个幅值）」「17 变 18」「480 折叠块」「32 对比」逐条一致。
- **FP8 论文 §2**：`Values that overflow are then saturated to the maximum representable value. Weight update skipping... is not a good choice for FP8 as overflows are much more likely due to the narrower dynamic range`；`unscaling is applied once per dot-product, thus amortized by many multiply-accumulates`。页 [C4]/[C6] 一致。
- **FP8 论文 §4.3**：`even exponent bias of 7 results in perplexity of 12.59 which is significantly higher (worse) than 10.19 for the bfloat16 baseline`；`we achieve 10.29 and 10.44 perplexities for GEMM-only and GEMM+residuals`；`several exponent bias choices in the [7, 10] range lead to results matching the bfloat16 baseline`。页 §3 三个数字与口径（per-channel 权重 / per-tensor 激活）一致。
- **DeepSeek-V3 §3.3 / §3.3.2**：`group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and (2) for weights... 128x128 block basis`；`the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits`、`Taking GEMM operations of two random matrices with K = 4096 for example, in our preliminary test, the limited accumulation precision in Tensor Cores results in a maximum relative error of nearly 2%`、`Once an interval of N_C is reached... copied to FP32 registers on CUDA Cores`、`setting N_C = 128 elements, equivalent to 4 WGEMMAs`；`we adopt the E4M3 format on all tensors... our methodology effectively shares exponent bits among these grouped elements`；`highly consistent with the idea of microscaling formats`；`we calculate the maximum absolute value online for each 1x128 activation tile or 128x128 weight block`；`two model scales similar to DeepSeek-V2-Lite and DeepSeek-V2, training for approximately 1 trillion tokens`、`the relative loss error of our FP8-training model remains consistently below 0.25%`。页 §5、核心问题 5、[C8]–[C12] 一致。
- **DeepSeek-V3 §B.2**：`block-wise quantization of activation gradients leads to model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens`、`activation gradients are highly imbalanced among tokens, resulting in token-correlated outliers`。页 §4 与常见误解第 2 条一致。
- **transformers v4.57.6 `quantizer_finegrained_fp8.py` L124-142**（原文实抓）：L126 `max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))`、L127 `scale = fp8_max / max_abs`、L132 `quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)`、L139 `scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()`、L143 存入 `.weight_scale_inv`。页 §4 公式与 [C14] 行号一致。
- **代码实跑**：两段 Python 在 torch 2.8.0 上输出与页面「预期输出」逐字一致（max=448.0 / tiny=1.562e-02 / 位模式值 448.0；块内 max_abs=6.0、scale=74.6667、绝对误差 max=0.142857 mean=0.029085、最大相对误差 4.632%、scale 形状 (16,32)/(128,12)）。
- **GLM-5.3-Flash 数字**：`glm-5-3-flash-dataflow/index.html` 记录 321.32 B 参数、`total_size` 328,326,771,576 字节（305.78 GiB）、1.022 字节/参数、37,338 个 scale 全符合 $\lceil N/128\rceil\times\lceil K/128\rceil$、34 个 KDA 层被量化投影数为 0、indexer/嵌入/输出头/视觉塔高精度、62 个分片。页 §5 与 [C15]/[N4] 逐条一致；`quantization_config` 的 fp8/e4m3/dynamic/weight_block_size [128,128] 与外部模型卡一致。
- **机械项**：`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 `validation ok`；两页互链；前置链接 `../quantization-basics/`、`../mxfp4-qat/`、`../glm-5-3-flash-dataflow/` 三页均真实存在；无 `<text>` 等宽字符框线图（两图均为内联 SVG，公式走 `<foreignObject>`+KaTeX）；`× — → ±` 属 validate.py 明确排除在 BARE_MATH_CHARS 之外的中文排版字符，合规；15 个 `<details>` 与 15 个 `<summary>` 一一对应，5 个核心问题 + 5 组本章问题全部有解答折叠块。

## 问题

- [重要·技术] §3「缩放：把张量对准可表示区间」第 2 段（`<h2 id="scaling">` 之下、`本章问题` 之前）：离群值示例的机制描述不成立，且与同段后半句自相矛盾｜位置：index.html 第 281 行｜引文依据：页面原文「逐张量对齐时 scale 由 6.0 决定，普通元素 $0.01\times(448/6)\approx0.75$——落在 $0.5$ 到 $1$ 的 binade 里，该 binade 的格距是 $1/16$（$2^{-1}\times1/8$），$0.01$ 的原始信息只剩约 4 位分辨率；没有离群值时普通元素的量级由对齐后的区间决定，FP8 的伤害不是「格距粗」而是动态范围被压缩」。核对：E4M3 对规格化数是乘性缩放，相对精度与 scale 无关、恒受 $2^{-4}$ 约束（paper Table 1 尾数 3 位）；本机实跑 0.01×74.6667=0.7467→0.75，误差 0.446%（约 7.8 位），并未低于格式固有精度，即离群值没有把普通元素的相对精度降到「约 4 位」。真正被离群值伤到的是远小于 max 的值（比值低于 $2^{-6}/448\approx3.5\times10^{-5}$ 时落进非规格化区），而示例里 0.01/6.0 的比值远高于该门限。同一段前半句用「格距是 1/16」论证伤害、后半句又说「伤害不是格距粗」，构成同段自相矛盾｜修复要求：重写该例，使失效机制与出处一致——删除「0.01 只剩约 4 位分辨率」这一由离群值导致的精度损失表述，改为演示「块内远小于 max 的值被推进非规格化区而丢精度」（例如块内 99% 在 $10^{-4}$、1% 在 $10^{0}$），或直接改为定性引用 DeepSeek-V3 §3.3.2 `This method makes low-precision training highly sensitive to activation outliers` 而不给精度位数的因果；同时删去同段「伤害不是格距粗而是动态范围被压缩」的自相矛盾句，统一为单一机制叙述｜修复：｜复验：

- [重要·来源] 「来源与范围说明」多条来源指向已移除的 `research/` 实测产物｜位置：index.html 第 444、462、464、465、466 行（[C2]「torch `finfo` 实测一致（research/）」、[F1]「research/ 实跑」、[F3]「research/ 实跑（构造示例）」、[F4]「research/ 与 GLM 数据流页存档」、[F5]「research/ 实跑（构造示例）」）｜引文依据：本页 `research/measured.md` 记载 `concept_probes.py`、`fp8_page_code.py`、`fp8_page_code.out` 等实测产物「现已从仓库移除」；仓库提交信息亦写明 `research/` 排除部署。故上述路径读者无法定位，无法据此复算｜修复要求：把 [C2]/[F1]/[F3]/[F4]/[F5] 的来源改写为可定位的形式——保留「FP8 论文 Table 1」等外部来源，对实测部分改为在正文/来源直接给出可复算的算式与结果（如 [F3] 的 scale=448/6、[F5] 的 $512\times4/8388608=0.0244\%$），删除对 `research/` 的引用；若仍需保留实测记录，指向已登记的存档说明而非目录名｜修复：｜复验：

- [轻微·表述] 元话语与自我指涉｜位置：index.html 第 65、67、409 行｜引文依据：「本文沿「格式 → 缩放 → 分块 → 配套策略 → 实证」的链条讲清它：」「本页直接进入 FP8 特有的部分。」「注意这是该论文实验条件下的结论，不外推到任意模型。」（第 223、434、494、495 行的「本页只做…」「本页不做…」为范围声明，可保留）｜修复要求：把第 65 行的「本文沿…链条讲清它」改为直接陈述（如「它的链条是：格式 → 缩放 → 分块 → 配套策略 → 实证」），第 67 行删去「本页直接进入 FP8 特有的部分」或改为不点明叙述者的衔接句，第 409 行「注意这是…」改为客观陈述（「该结论限于该论文的实验条件」）｜修复：｜复验：

- [轻微·表述] 把「场景」当术语｜位置：index.html 第 103 行｜引文依据：核心问题 5 解答摘要「分块缩放让 E4M3 覆盖梯度场景」｜修复要求：改为「覆盖梯度张量」或「适用于梯度」｜修复：｜复验：

- [轻微·格式] 词内多余空格｜位置：index.html 第 385 行｜引文依据：「块缩放的思想后来被标准 化为 microscaling（MX）格式」（「标准 化」中间有空格）｜修复要求：改为「标准化」｜修复：｜复验：

- [轻微·技术] 「后来被标准化为 microscaling（MX）格式」的时间顺序与来源不符｜位置：index.html 第 385 行｜引文依据：DeepSeek-V3 §3.3.2 原文只作一致性陈述 `our fine-grained quantization strategy is highly consistent with the idea of microscaling formats (Rouhani et al., 2023b)`，未说块缩放「后来被标准化」；MX/OCP 规范（2023）早于 DeepSeek-V3（2024），「后来」时序倒置｜修复要求：改为与来源一致的表述，如「该思路与 microscaling（MX）格式一脉相承」｜修复：｜复验：

- [轻微·技术] 无来源支持的判断写成结论｜位置：index.html 第 407 行｜引文依据：页面「这个间隔也是块大小 128 的另一个呼应」；DSV3 §3.3.2 原文为 `setting N_C = 128 elements, equivalent to 4 WGEMMAs, represents the minimal accumulation interval`，未把 $N_C=128$ 与块大小 128 关联｜修复要求：删除该推断，或改写为「$N_C$ 取 128 是为覆盖 4 个 WGMMA 的最小间隔（论文原话）」，不暗示与块大小同源｜修复：｜复验：

- [轻微·技术] 「整个 KDA 路径保持 BF16」与所引页面记录不符｜位置：index.html 第 415 行｜引文依据：`glm-5-3-flash-dataflow/index.html` 记录「其中卷积核、$A$ 与 $b_{dt}$ 更进一步留在 FP32」，并有 `_keep_in_fp32_modules_strict`；故 KDA 并非整条路径都是 BF16｜修复要求：改为「KDA 层投影不走 FP8（其中卷积核、$A$、$b_{dt}$ 留在 FP32）」｜修复：｜复验：

- [轻微·表述] 同一格式的精度位数两处写法不一致｜位置：index.html 第 76 行与第 232 行｜引文依据：核心问题 1 解答「3 位尾数意味着相对精度只有约 3 位二进制有效数字」；§1 本章问题解答「规格化数的尾数域 3 位，加上隐含前导 1，有效位共 4 位二进制」｜修复要求：统一口径，例如两处都写「尾数 3 位（加隐含前导 1 共约 4 位有效位），相对精度约 $2^{-4}$」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 8
- 处置：修复（无阻断项；两条重要问题位于「缩放」示例的机制与来源可定位性，修复后进入第 3 轮）
- 说明：事实、公式、数字层面未发现与来源冲突或无法复算之处；页面已核对的两段可运行代码在 torch 2.8.0 上输出与「预期输出」完全一致；`validate.py` 返回成功。