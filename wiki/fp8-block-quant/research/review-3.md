<!-- review-meta
round: 3
page: wiki/fp8-block-quant/index.html
reviewed_content_sha256: 117659776e05b5d2
-->
# FP8 块量化审查记录（第 3 轮）

- 页面版本：20f72c5be14f480e21044c980dd97b78153775dc
- 审查时间：2026-09-13 19:40
- 审查者：编排者派发的独立审查者（未参与写作，未读取 research/ 下的规划、修复与前序审查记录）
- 已完整阅读章节：核心问题 / 常见误解 / 1. E4M3 的位布局 / 2. 两种格式与特殊值的取舍 / 3. 缩放：把张量对准可表示区间 / 4. 分块：128×128 与 1×128 / 5. 配套策略与实证 / 来源与范围说明（含全部折叠块、两个代码块与全部图注，逐段通读）
- 外部核对来源：arXiv:2209.05433（FP8 Formats for Deep Learning，ar5iv 全文）；arXiv:2412.19437（DeepSeek-V3 Technical Report，ar5iv 全文）；huggingface/transformers v4.57.6 `quantizer_finegrained_fp8.py` 与 main 分支 `integrations/finegrained_fp8.py`；`zai-org/GLM-5.3-Flash` 官方 `config.json`、`model.safetensors.index.json` 与分片 safetensors 张量头（HTTP Range，不下载权重）；本机 torch 2.8.0 实跑。

## 核对通过的关键项（引文依据）

- FP8 论文 Table 1 / §3.1：E4M3 bias 7、Max normal 448、Min normal 2^-6、Min subnorm 2^-9、NaN 仅 `S.1111.111₂`、无 Infinities；E5M2 bias 15、57,344、2^-14、2^-16、inf `S.11111.00₂`、NaN `S.11111.{01,10,11}₂`。原文「This modification extents the dynamic range by one extra power of 2, from 17 to 18 binades」「We gain the representation of seven more magnitudes (256, 288, 320, 352, 384, 416, 448)」「The maximum representable magnitude without this modification would be 240」「we could gain one additional representable magnitude, 480, by having just one encoding for zero and one for NaN, this would require breaking the symmetry of positive and negative representations inherent in the IEEE 754 formats」——与页第 1、2 章及第 2 章折叠块逐条一致。
- FP8 论文 §3.1 末段（紧接「3.2 Exponent bias」之前）原文「the benefit of having fewer representations of special values would be much smaller for E5M2 than it was for E4M3 - only 3 additional magnitude values would be added due to the smaller mantissa, one additional binade is much less impactful when E5M2 already provides 32 (compared to E4M3's 17 without the adjustment)」——页第 2 章「E5M2 已有 32 个 binade，多一个 binade 和 3 个幅值的收益小得多」有直接来源，非页内推断。
- FP8 论文 §2 原文「the general idea is to choose a scaling factor such that the maximum magnitude in the tensor becomes close to the maximum representable magnitude」「Values that overflow are then saturated to the maximum representable value.」「Weight update skipping ... is not a good choice for FP8 as overflows are much more likely due to the narrower dynamic range, resulting in too many skipped updates.」「unscaling is applied once per dot-product, thus amortized by many multiply-accumulates」——[C4]/[C6] 定位（§2）与内容一致。
- FP8 论文 §3 原文「The recommended use of FP8 encodings is E4M3 for weight and activation tensors, and E5M2 for gradient tensors」——[C5] 一致。
- FP8 论文 §4.3 原文「several exponent bias choices in the [7,10] range lead to results matching the bfloat16 baseline」「exponent bias of 7 results in perplexity of 12.59 which is significantly higher (worse) than 10.19 for the bfloat16 baseline」「if instead we calibrate the tensors to have their own scaling factors (following the convention of int8 quantization to use per-channel and per-tensor scaling factors for weights and activations, respectively) we achieve 10.29 and 10.44 perplexities for GEMM-only and GEMM+residuals FP8 inference」——页第 3 章「7~10」「12.59 对基线 10.19」「权重 per-channel、激活 per-tensor → 10.29/10.44」逐字一致；节号 §4.3 正确（§4.3 标题即「Per-tensor scaling factors」）。
- DeepSeek-V3 §3.3.2 原文「for activations, we group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and for weights ... on a 128x128 block basis (i.e., per 128 input channels per 128 output channels)」「we calculate the maximum absolute value online for each 1x128 activation tile or 128x128 weight block」「the Tensor Core only uses the highest 14 bits of each mantissa product ... employs 14-bit precision」「K = 4096 for example, in our preliminary test, the limited accumulation precision in Tensor Cores results in a maximum relative error of nearly 2%」「N_C=128 elements, equivalent to 4 WGMMAs, represents the minimal accumulation interval」「our methodology effectively shares exponent bits among these grouped elements」「our fine-grained quantization strategy is highly consistent with the idea of microscaling formats」——页第 4、5 章各条一致。
- DeepSeek-V3 §3.3「two model scales similar to DeepSeek-V2-Lite and DeepSeek-V2, training for approximately 1 trillion tokens」「the relative loss error of our FP8-training model remains consistently below 0.25%」；§B.2 原文「A straightforward strategy is to apply block-wise quantization per 128x128 elements like the way we quantize the model weights ... block-wise quantization of activation gradients leads to model divergence on an MoE model comprising approximately 16B total parameters, trained for around 300B tokens ... token-correlated outliers」——页第 4 章反面实验与第 5 章数字一致。
- transformers v4.57.6 `quantizer_finegrained_fp8.py` L126-139：`max_abs = torch.amax(torch.abs(param_value), dim=(-1,-2))`、`scale = fp8_max / max_abs`、`torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)`、`scale = scale.reshape(...).reciprocal()` 并写入 `... + ".weight_scale_inv"`——[C14] 的「L124-142」定位在标注的版本（4.57.6）成立。
- GLM-5.3-Flash 官方 `config.json` 的 `quantization_config`（其余键为 1509 条 `modules_to_not_convert`）：`{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128]}`——页第 5 章「fp8 / e4m3 / dynamic / weight_block_size [128,128]」逐字一致。
- `model.safetensors.index.json`：`total_size = 328,326,771,576`、张量 76,108 个、`weight_scale_inv` 37,338 个、分片 62 个——页第 5 章与 [N4] 逐条一致。抽查 3 个分片（model-00001/00010/00030）共 1,574 个 scale 张量头：形状全部等于 `[⌈N/128⌉, ⌈K/128⌉]` 且 dtype 为 `F32`，不吻合数 0；带 `self_attn` 的 scale 只出现在层 3,7,…,43（11 个 DSA 层，各 4 个投影 q_a/q_b/kv_a/o）与第 45 层（MTP），34 个 KDA 层为 0，indexer 为 0——「KDA 层零量化」「只有大矩阵乘权重走 FP8」成立。
- 代码实跑（torch 2.8.0）：两段 Python 输出与页面「预期输出」逐字一致——`max = 448.0`、`tiny = 1.562e-02`、位模式 `0.1111.110` 解码 `448.0`；`scale = 74.6667`、反量化误差 `max = 0.142857 / mean = 0.029085`、最大相对误差 `4.632%`、scale 形状 `(16,32)`/`(128,12)`。算术复算：`1e-5×74.67≈7.5e-4` < `2^-9≈0.00195`；`512×4/8388608 = 0.0244%`；`328,326,771,576/321,323,031,390−1 = 2.18%`；`2048×4096` 权重 512 块——均与页面一致。最大相对误差确落在小值上（W=−0.008 与 W=−0.001，均为 4.632%），“出现在小值上”成立。
- `.dojo/scripts/validate.py wiki/fp8-block-quant/index.html` → `validation ok`；页面内链（quantization-basics / mxfp4-qat / glm-5-3-flash-dataflow / overview）全部存在，无 research/ 路径引用，无占位符。

## 问题

- [轻微·来源] 开头段「GLM-5.3-Flash 有 321.3 B 参数，磁盘占用只有 305.8 GiB——平均每个参数 1.02 字节，不到 BF16（2 字节）的 55%」：对比基准 55% 与实算不符｜引文依据：GLM checkpoint `total_size` 328,326,771,576 字节 ÷ 参数量 321,323,031,390 = 1.022 B/参数，÷ 2 字节 = **51.1%**，页面取的 55% 无出处且明显偏松（「不到」使其在逻辑上仍为真，但读者会以为比值接近 55%）｜修复要求：改为与实算一致的表述，如「约为 BF16 的一半」或「不到 BF16 的 52%」｜修复：｜复验：
- [轻微·来源] 第 5 章「内积维度 $K=4096$ 时（大规模训练的常见规模）」：括注为无来源判断｜引文依据：DSV3 §3.3.2 原文只有「K = 4096 for example, in our preliminary test」，未出现「大规模训练的常见规模」一类表述｜修复要求：删除该括注，或改写为可从来源核对的陈述｜修复：｜复验：
- [轻微·技术] 第 4 章代码折叠块「简化条件」：「……做的是四舍五入到最近可表示值，舍入方式与 GPU 路径相同」｜引文依据：torch 2.8.0 实测 `float8_e4m3fn` 的 cast 为就近取偶（round-half-to-even）：中点 1.0625→1.0、1.1875→1.25，与「四舍五入」（half-away-from-zero）不同；且本页「简化条件及其限制」明写「GPU kernel 的舍入细节……未验证」，与本句「与 GPU 路径相同」相抵｜修复要求：将「四舍五入」改为「就近取偶（round-to-nearest-even）」；对 GPU 侧一致性给出依据或降级为「本页未验证 GPU 侧舍入」｜修复：｜复验：
- [轻微·技术] 第 1 章代码折叠块「简化条件」：「CPU 上 `float8_e4m3fn` 的逐元素算术未实现」不准确｜引文依据：torch 2.8.0 CPU 实测加法抛 `NotImplementedError: "add_stub" not implemented for 'Float8_e4m3fn'`，但逐元素乘法 `a*a` 可运行并返回 `float8_e4m3fn`｜修复要求：改为「逐元素加法等部分算术在 CPU 上未实现」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（仅轻微项）。全部事实性论断、公式、数字均已回源核对，代码实跑输出逐字一致，无来源不支持的核心结论、无构造示例冒充来源事实、无页内自相矛盾；四条轻微项修复后即满足发布条件。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
