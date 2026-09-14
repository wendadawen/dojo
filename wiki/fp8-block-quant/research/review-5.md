<!-- review-meta
round: 5
page: wiki/fp8-block-quant/index.html
reviewed_content_sha256: 290840801823fb77
-->
# FP8 块量化审查记录（第 5 轮）

- 页面版本：a0ccb4caea87264a7fa38c8f5ca76b3c9c92abc1
- 审查时间：2026-09-14 17:39
- 审查者：独立子代理（编排者派发，未参与写作与前序轮次）
- 已完整阅读章节：核心问题、常见误解、1. E4M3 的位布局（含代码折叠块与本章问题）、2. 两种格式与特殊值的取舍（含补充折叠块与本章问题）、3. 缩放：把张量对准可表示区间（含本章问题）、4. 分块：128×128 与 1×128（含代码折叠块与本章问题）、5. 配套策略与实证（含本章问题）、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 问题

- [轻微·图示] 第 1 章「E4M3 的位布局」结构图：分组标签「指数 4 位」与「尾数 3 位」未在所属分组下方居中，而是对齐到分组最左侧比特之下。中外侧度：4 个指数框跨 x=120–289（中心 204.5），而「指数 4 位」foreignObject 为 x=122、宽 48（文字中心 146，偏左约 58px）；3 个尾数框跨 x=328–445（中心 386.5），而「尾数 3 位」foreignObject 为 x=328、宽 72（文字中心 364，偏左约 22px）。同图各比特标签（S、E_3…M_0）本身是居中的，故分组标签显得像在标注最左一位。｜引文依据：不适用｜修复要求：把「指数<br/>4 位」的 foreignObject x 由 122 改为 180.5（=204.5−48/2），把「尾数<br/>3 位」的 x 由 328 改为 350.5（=386.5−72/2），使两标签居中于各自分组下方。｜修复：｜复验：

- [轻微·格式] 正文以编号引用其他章节：「给 max 对齐缩放（第 3 章）」「第 3 章构造例里那个 6.0 的离群值」「（第 4 章 $4\times4$ roundtrip 的最大相对误差 4.6% 就出在小值上）」「即第 2 章的分工」，以及来源章节「构造示例」小节下的「（代入计算，第 3 章）」「（第 4 章）」「（第 1 章）」；而 style-guide §1.3 规定「不使用 S1/S2/S3 等章节代号。正文引用其他章节时使用章节标题」。同页核心问题解答已用标题形式（「见「分块」与「配套策略与实证」两章」），两种写法并存。｜引文依据：不适用｜修复要求：把上述编号引用改为对应章节标题（第三章→「缩放」、第二章→「两种格式与特殊值的取舍」、第四章→「分块」、第一章→「E4M3 的位布局」）。｜修复：｜复验：

- [轻微·技术] 第 4 章「块不是越大越好」段：「$128\times128$ 的块太粗，只有 $1\times128$ 的逐 token 粒度才隔离得住」置于「论文归因于……<sup>[C13]</sup>」句中，读作 DSV3 的结论。DSV3 §B.2 只说 block-wise 量化激活梯度导致发散、离群值「cannot be effectively managed by a block-wise quantization approach」，并未给出「只有 1×128 可用」的唯一性结论（其因果解释也仅以 hypothesize 提出）；本页本章问题解答写的是「B.2 的实验只否定了激活梯度用 $128\times128$，并没有把粒度推向任意细」，两处强度不一致（[C13] 条目本身也未含此断言）。｜引文依据：DSV3 v2 §B.2 原文「block-wise quantization of activation gradients leads to model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens… We hypothesize that this sensitivity arises because activation gradients are highly imbalanced among tokens, resulting in token-correlated outliers. These outliers cannot be effectively managed by a block-wise quantization approach.」｜修复要求：把「只有 $1\times128$ 的逐 token 粒度才隔离得住」改为与 B.2 一致的表述（如「$128\times128$ 的块无法隔离 token 相关离群值，DSV3 改用逐 token 的 $1\times128$ 片」），去掉唯一性断言。｜修复：｜复验：

- [轻微·技术] 第 5 章 GLM 实证清单写「被量化的只有大矩阵乘权重（专家、MLA 的 $q_a$/$q_b$/$kv_a$/$o$ 投影、稠密 MLP）」；本页引为存档的 GLM-5.3-Flash 前向数据流页写的是「DSA 层的 $q_a$/$q_b$/$kv_a$/$o$ 投影」，checkpoint config.json 的 layer_types 也只含 linear_attention 与 deepseek_sparse_attention，页面被量化的正是不含 MLA 的那批稀疏注意力层。｜引文依据：config.json `"layer_types": {"linear_attention": 34, "deepseek_sparse_attention": 11}`；数据流页「被量化的范围很有选择性——只有大矩阵乘的权重：routed 专家的三个投影、共享专家、DSA 层的 $q_a$/$q_b$/$kv_a$/$o$ 投影、前 3 层稠密 MLP」｜修复要求：把「MLA 的」改为「DSA 层的」（或「稀疏注意力层的」），与所引页面及 checkpoint 一致。｜修复：｜复验：

## 来源核对（本轮核对所用版本与结论）

- FP8 论文 arXiv:2209.05433v2（HTML，2022-09-29）。Table 1：E4M3 1/4/3 bias 7、E5M2 1/5/2 bias 15、Max normal 448 与 57,344、Min normal $2^{-6}$ 与 $2^{-14}$、Min subnorm $2^{-9}$ 与 $2^{-16}$。§2：max 对齐（「choose a scaling factor such that the maximum magnitude… close to the maximum representable magnitude」）、「Values that overflow are then saturated to the maximum representable value」、「Weight update skipping… is not a good choice for FP8… too many skipped updates」、「unscaling is applied once per dot-product, thus amortized by many multiply-accumulates」。§3.1：「Infinities are not represented… retain only one mantissa bit-pattern for NaNs… extents the dynamic range… from 17 to 18 binades… seven more magnitudes (256, 288, 320, 352, 384, 416, 448)… maximum representable magnitude without this modification would be 240… could gain one additional representable magnitude, 480」（480 取舍与整数排序对称性亦在原文）。§4.3：「several exponent bias choices in the [7,10] range lead to results matching the bfloat16 baseline… even exponent bias of 7 results in perplexity of 12.59… 10.19 for the bfloat16 baseline… calibrate… per-channel and per-tensor scaling factors for weights and activations… 10.29 and 10.44」。§3 推荐用法：「E4M3 for weight and activation tensors, and E5M2 for gradient tensors」。以上与页面逐条一致。
- DeepSeek-V3 arXiv:2412.19437v2（HTML，2025-02-18）。§3.3：两规模「similar to DeepSeek-V2-Lite and DeepSeek-V2」、约 1T token、「relative loss error… remains consistently below 0.25%」。§3.3.2：激活「1x128 tile basis (i.e., per token per 128 channels)」、权重「128x128 block basis」；「accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits」；「K = 4096… a maximum relative error of nearly 2%」；「setting $N_C$ = 128 elements, equivalent to 4 WGMMAs」；全张量 E4M3、「effectively share exponent bits」；「Online Quantization」；「highly consistent with the idea of microscaling formats」。§B.2：Dgrad 全张量 block-wise 量化导致「model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens」、token-correlated outliers 假说。以上与页面一致（「只有 1×128 才隔离得住」除外，见上条）。
- transformers v4.57.6 `src/transformers/quantizers/quantizer_finegrained_fp8.py`：`max_abs = torch.amax(torch.abs(param_value), dim=(-1,-2))`、`scale = fp8_max / max_abs`、`torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)`、`scale = scale.reshape(...).squeeze().reciprocal()` 后写入 `<name>.weight_scale_inv`。与 [C14]/[F2] 一致。
- GLM-5.3-Flash checkpoint（HF 仓库 zai-org/GLM-5.3-Flash）。config.json 的 quantization_config = {quant_method: "fp8", fmt: "e4m3", activation_scheme: "dynamic", weight_block_size: [128,128]}；model.safetensors.index.json：76,108 张量、62 分片、metadata.total_size = 328,326,771,576、以 `weight_scale_inv` 结尾的张量恰 37,338 个；model-00001-of-00062 的 safetensors 头中 329 个 scale 全为 F32，且形状逐一等于 (⌈N/128⌉, ⌈K/128⌉)（N、K 取该 scale 对应 `.weight` 的形状），0 例外；layer_types 中 34 个 linear_attention（KDA）层 self_attn 下 scale 数为 0，scale 只出现在 experts/shared_experts/稠密 MLP 与 self_attn 的 q_a_proj、q_b_proj、kv_a_proj_with_mqa、o_proj。与第 5 章清单、[C15]、[N4]、[F4] 一致。
- 代码块：两段「预期输出」在本机 torch 2.8.0 上逐字复现（max=448.0、tiny=1.562e-02、位模式 0.1111.110→448.0；scale=74.6667、绝对误差 max=0.142857 / mean=0.029085、最大相对误差 4.632%、(2048,4096)→(16,32)、(16384,1536)→(128,12)）。另测 `float8_e4m3fn` 在 CPU 上逐元素加法抛 NotImplementedError、矩阵乘可用，与第 1 章「简化条件」表述一致；$0.001\times74.6667\approx0.075$ 反量化后取整到 0.078125，相对误差 4.63% 确出在该小值上，与「观察重点」一致。
- 可复算项复核：$448/6=74.6667$、$512\times4/8{,}388{,}608=0.0244\%$、$328{,}326{,}771{,}576/2^{30}=305.78$ GiB、$328{,}326{,}771{,}576/321.32\times10^{9}=1.022$、（328.327−321.32）/321.32=2.18%≈2.2%、$2^{15-7}\times(1+6/8)=448$，均与页面数字一致。
- 页面功能：`.dojo/scripts/validate.py wiki/fp8-block-quant/index.html` 返回 `validation ok`；官网首页与 overview.html 双向链接有效；三处站内概念链接（量化基础、MXFP4 量化感知训练、GLM-5.3-Flash 前向数据流）对应目录均存在；无「（待生成）」占位；公式定界符外无 Unicode 数学字符；SVG 用 `foreignObject` 承载 KaTeX；aria-label 无 `$...$`。
- 通读未发现：元话语、以「本页」为主语的自我指代（style-guide §12 允许「本页」，且仅出现在「来源与范围说明」的范围声明中）、会话指代、调试叙事、临场评价、AI 拼接腔；两级问题块齐全，核心问题 5 条、每章本章问题各 1–2 条均有解答折叠块，核心问题答案均指明完整论证所在章节。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复
