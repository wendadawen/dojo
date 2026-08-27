# evidence.md：FP8 块量化

来源缩写：
- [FP8] Micikevicius et al., FP8 Formats for Deep Learning, arXiv:2209.05433v2 (2022)
- [DSV3] DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437 (2024), §3.3 与附录 B.2
- [QZ] transformers 4.57.6 本地安装, src/transformers/quantizers/quantizer_finegrained_fp8.py L124-142
- [实测] research/concept_probes.out 探针 D
- [GLMckpt] GLM-5.3-Flash 62 分片张量头（GLM 数据流页 verify_structure 存档）

## C 论断

- C1 E4M3：1 符号+4 指数+3 尾数、偏置 7；E5M2：1+5+2、偏置 15：[FP8] §3 Table 1。已确认
- C2 E4M3 最大正规数 448、最小正正规数 $2^{-6}$、最小正非规格化数 $2^{-9}$：[FP8] Table 1；实测 torch finfo max=448、tiny=1.562e-2（探针 D1）。已确认
- C3 E4M3 不表示无穷、仅一个 NaN 位模式，回收后最大幅值 240→448（多 7 个幅值、17→18 binade）：[FP8] §3.1。已确认
- C4 E4M3 溢出饱和至最大可表示值（无跳过更新的混合精度策略）：[FP8] §2。已确认
- C5 推荐分工：权重与激活用 E4M3、梯度用 E5M2：[FP8] §3 原句。已确认
- C6 逐张量 max 对齐的缩放实践（scale the maximum absolute value … to the maximum representable value）：[FP8] §2。已确认
- C7 量化残差时无单一指数偏置可保持精度；逐张量校准（权重 per-channel、激活 per-tensor）后恢复（困惑度 10.29/10.44 vs 基线 10.19）：[FP8] §4.3。已确认
- C8 DSV3 粒度：激活 1×128（逐 token 逐 128 通道）、权重 128×128（逐 128 输入×128 输出通道）：[DSV3] §3.3.2 原句。已确认
- C9 DSV3 全张量用 E4M3（对比先前 Fprop E4M3 / Dgrad Wgrad E5M2 的混用），归因于分块缩放共享指数：[DSV3] §3.3.2 Mantissa over Exponents。已确认
- C10 H800 FP8 GEMM 累加精度约 14 位；K=4096 随机矩阵最大相对误差近 2%；每 $N_C=128$ 提升到 FP32 寄存器累加：[DSV3] §3.3.2。已确认
- C11 相对 loss 误差 <0.25%（两规模、~1T token）：[DSV3] §3.3 开头。已确认
- C12 在线计算每个 1×128/128×128 块的 max 并即时量化：[DSV3] §3.3.2 Online Quantization。已确认
- C13 激活梯度按 128×128 块量化导致发散（16B、300B token）；token 相关离群值假说：[DSV3] §B.2。已确认
- C14 scale 计算与存储语义：scale=fp8_max/max_abs（fp8_max=448）、量化 clamp(x·scale) 转 E4M3、存 reciprocal 为 weight_scale_inv：[QZ] L124-142。已确认
- C15 GLM 布局：quantization_config fp8/e4m3/dynamic/[128,128]；37,338 个 scale 形状全部 $\lceil N/128\rceil\times\lceil K/128\rceil$、F32；平均 1.022 B/参数（参数 321.32B vs 磁盘 328,326,771,576 B）；KDA 层量化张量数 0：[GLMckpt]（GLM 数据流页 p7 探针与 verify_structure）。已确认

## F 公式

- F1 E4M3 规格化数值 $=(-1)^S\times2^{E-7}\times(1+M/8)$：[FP8] §3 位编码约定；实测位模式 0.1111.110 解码=448（探针 D2）。已确认
- F2 块量化 scale $s=448/\max|W_{\text{block}}|$、$W_q=\mathrm{clamp}(W\cdot s,\pm448)$、$W\approx W_q\cdot s^{-1}$：组合自 [FP8] §2 max 对齐（常数来源：Table 1 的 448）与 [QZ] L124-135（公式形态）。组合陈述，页面标明两处来源
- F3 4×4 块量化 roundtrip 实测：max=6 的块，scale=74.667，反量化绝对误差 max 0.143、均值 0.029、最大相对误差 4.6%：[实测] D3。构造示例（故意构造宽动态范围块演示精度边界）
- F4 scale 形状公式 $\lceil N/128\rceil\times\lceil K/128\rceil$ 逐例验证：(2048,4096)→(16,32)、(16384,1536)→(128,12)、(12288,4096)→(96,32)：[实测] D4 与 [GLMckpt] 37,338 例全验证。已确认
- F5 存储开销：2048×4096 权重 + 16×32 个 F32 scale → 平均 1.000244 B/参数：[实测] D5。构造示例

## N 数字

- N1 E5M2：最大正规数 57,344、偏置 15、32 binade：[FP8] Table 1。已确认
- N2 240→448（多 7 个幅值）：[FP8] §3.1。已确认
- N3 DSV3 实验条件（两规模 ~DeepSeek-V2-Lite/V2、~1T token、误差<0.25%；B.2 的 16B/300B token 发散实验）：[DSV3] §3.3/§B.2。已确认，引用时带条件
- N4 GLM：321.32B 参数、305.78 GiB 磁盘、1.022 B/参数、37,338 scale、KDA 零量化：[GLMckpt]。已确认

## 冲突与缺口

- F2 是组合公式：max 对齐来自 [FP8]（针对逐张量），分块粒度来自 [DSV3]（文字+图），显式算式来自 [QZ] 实现——页面按「论文定原则、实现定公式」如实拆开标注
- [DSV3] 正文无编号公式（HTML/PDF 提取均无）；所有 DSV3 引用按节号+原句定位
- GLM 的激活量化（dynamic scheme）细节在推理框架里，本页不声称
