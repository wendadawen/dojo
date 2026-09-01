# Sherry 稀疏三值量化：核心论断与证据

来源简写：
- [PAPER] Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification, arXiv:2601.07892v1（2026-01-12，ACL 2026 接收）
- [PR] llama.cpp PR #22836「ggml-cpu: add STQ1_0 ternary quantization with ARM NEON vec_dot kernel」
- [HF] HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（2026-08 读取）
- [HY] 腾讯混元官方技术文章「Hy4 preview 轻量版」（2026-09-01 发布，公众号/新闻转载）

## C 论断

- C1：三值量化把权重约束到 {-1, 0, +1}（乘共享缩放系数后为 {-d, 0, +d}）。来源：[PAPER] 摘要；[PR] 引言。已确认。
- C2：2-bit 对齐打包保持 SIMD 对齐但相对理论下限浪费 bit；1.67-bit（3 权重/5 bit）打包更紧凑但 3 路分组与现代 SIMD 的 2 的幂通道不兼容，实际推理常比 2-bit 还慢。来源：[PAPER] 引言与 Table 4（TL2 116.83 t/s 慢于 I2_S 132.13 t/s，0.7B）。已确认。
- C3：三值的理论存储下限为 log2(3)≈1.585 bit/权重。来源：[PAPER] 引言（"理论下限 log₂3 ≈ 1.58 bit"）。已确认。
- C4：3:4 稀疏约束（每 4 个权重恰好一个零）把 4 权重组合数从 3⁴=81 压到 C(4,3)×2³=32=2⁵，可用 5 bit 零浪费编码，平均 1.25 bit/权重。来源：[PAPER] §3.1（附录 A 给 5-bit 打包结构、附录 C 给 16 模式折算）。已确认。
- C5：5 bit 拆为 4 bit 索引 + 1 bit 符号：利用镜像对称，16 种模式（C(4,3)×2^(3-1)=16）恰好填满 4-bit 索引的 16 项查找表，1 个符号位控制整体翻转。来源：[PAPER] 附录 A/C；[PR] 引言（"4-bit codebook index + 1-bit sign"）。已确认。
- C6：标准 SIMD 查表指令（如 x86 vpshufb、ARM tbl）单指令索引上限为 4 bit（16 项），3:4 格式的 16 模式码本恰好满足该约束；这是选 4 权重一组而非 3 或 8 的硬件理由。来源：[PAPER] 附录 C（M 必须为 2 的幂、M=8 索引超出 LUT 容量、M=4 唯一可行；2:4 仅用 12 状态且 50% 稀疏度过危险阈值）。已确认。
- C7：STQ1_0 文件布局为每 256 权重 42 字节：32 字节 4-bit 索引（64 组）+ 8 字节符号位 + 2 字节 fp16 缩放系数，合 1.3125 bpw。来源：[PR] 引言（"qs[32] + sign[8] + fp16 scale；42 B / 256 = 1.3125 bpw"）；[HF]「2 + 32 + 8 = 42 bytes per 256 weights = 1.3125 bpw」。已确认。
- C8：STQ1_0 内核采用 stride-16 分组：每个 64 权重块内，组取 (w0, w16, w32, w48) 而非连续 4 个，使查表解码后各 lane 与顺序存储的 Q8_K 激活天然对齐，无需重排。来源：[PR]「STQ1_0: Stride-16 Sparsity Layout」节。已确认。注意与 [PAPER]「每连续 4 个权重为一个 block」的表述差异：格式概念不绑定内存顺序，stride-16 是内核实现选择，正文须分开陈述。
- C9：论文训练侧缩放系数用 Sparse-AbsMean：组内置零绝对值最小者，d 取留存权重平均绝对值，论文附录 D 证明其为 L2 重建误差最小化的最优解。来源：[PAPER] 方法节与附录 D。已确认。
- C10：llama.cpp 上游 STQ1_0 量化器面向已在三值网格上的 QAT 输入：d=amax、置零 argmin|x|、忽略 imatrix，对 PTQ 较弱。来源：[HF]「Our encoder」节。已确认。
- C11：AngelSlim 编码器保持格式字节不变，只改两个决策：加权最小二乘 d=Σ(w_sel·x)/Σ(w_sel²)；imatrix 感知零位——置零使 w[j](x[j]²-(|x[j]|-d)²) 最小的 lane；交替 3 轮。1200 行真实专家权重上：仅 LS 缩放使加权平方误差降 89.7%，imatrix 项再降剩余部分的 4.1%。来源：[HF]「Our encoder」节。已确认。
- C12：训练侧存在 weight trapping：硬性 3:4 约束下梯度同质化导致表征坍缩，权重退化为类二值状态（Effective Rank < 750 / 4096）；论文以 Arenas（退火残差通路 Y=XTα+λt·XW，λt 训练末退火为 0）应对，推理零开销。来源：[PAPER] 方法与实验节。已确认。适用条件：仅 QAT 训练场景，Hy4 PTQ 应用不使用。
- C13：M4 Pro 上同一 checkpoint 转三种三值格式实测：STQ1_0 358 MiB、pp512 732.69 t/s、tg128 147.47 t/s；体积比 TQ1_0 小约 11%、比 TQ2_0 小约 20%，tg128 快于 TQ1_0（138.87）；与 1-bit Q1_0 比体积增约 6% 但 tg128 高约 35%。来源：[PR] Performance 节 llama-bench 表。已确认。
- C14：论文在 LLaMA-3.2 上的精度：1B 平均 0.519 与 1.67-bit SOTA Tequila 持平（bit 少 25%），3B 0.567 vs 0.576；CPU（i7-14700HX）0.7B 模型 Sherry 148.27 t/s 快于 TL2（116.83）与 I2_S（132.13），3B 45.55 vs 38.80/41.87。来源：[PAPER] Table 1/4。已确认。适用条件：QAT 训练后的模型、五个零样本任务平均。
- C15：STQ1_0 与常见低比特格式同算子对比中，bit 最低的同时速度与 IQ1_M 基本持平、明显快于 IQ1_S。来源：[HY] 算子速度对比图（具体数值仅以图形式给出）。已确认（仅定性引用，不引用读图数值）。

## F 公式

- F1：4 权重组合数 C(4,3)×2³=32=2⁵ → 5 bit/组 → 1.25 bit/权重。来源：[PAPER] §3.1；组合数学可直接复算。已确认。
- F2：镜像对称拆分 C(4,3)×2^(3-1)=16=2⁴ → 4 bit 索引 + 1 bit 符号。来源：[PAPER] 附录 A/C。已确认。
- F3：文件开销 (32+8+2)×8/256=1.3125 bpw。来源：[PR]/[HF] 布局描述直接换算。已确认。
- F4：Sparse-AbsMean 缩放 α_j=(4/3d_in)·Σ_{i∈S_j}|W_ij|，即留存权重平均绝对值。来源：[PAPER] 方法节。已确认。
- F5：加权最小二乘缩放 d=Σ(w_sel·x)/Σ(w_sel²)，x 为原始权重、w_sel 为 imatrix 权重。来源：[HF]「Our encoder」。已确认。
- F6：imatrix 零位的增量代价 w[j](x[j]²-(|x[j]|-d)²)，取最小者置零。来源：[HF]「Our encoder」。已确认。

## N 数字

- N1：42 字节/256 权重、1.3125 bpw。来源：[PR][HF]。条件：STQ1_0 格式定义。
- N2：M4 Pro llama-bench：STQ1_0 358.00 MiB / pp512 732.69±20.00 / tg128 147.47±1.36；TQ1_0 401.50 MiB / 728.69 / 138.87；TQ2_0 445.00 MiB / 689.25 / 175.06；Q1_0 336.25 MiB / 768.47 / 109.62。来源：[PR] Performance 表。条件：Sherry-1B 社区复现 checkpoint 统一转 bf16 再量化、-ngl 0 纯 CPU、8 线程。
- N3：论文 Table 4（i7-14700HX）：0.7B——BF16 34.01 t/s 1360 MB，I2_S 132.13 / 256.56，TL2 116.83 / 233.44，Sherry 148.27 / 205.50；3B——BF16 7.55 / 6190，I2_S 41.87 / 873.65，TL2 38.80 / 846.01，Sherry 45.55 / 712.40。来源：[PAPER] Table 4。
- N4：论文 Table 1 平均精度：1B BF16 0.558 / Tequila 0.519 / Sherry 0.519；3B BF16 0.636 / Tequila 0.576 / Sherry 0.567；ARC-C Sherry 反超 Tequila（1B 0.309 vs 0.305，3B 0.364 vs 0.346）。来源：[PAPER] Table 1。条件：UltraFineWeb 10B tokens QAT、五个零样本任务（PIQA/ARC-E/ARC-C/HellaSwag/WinoGrande）、3 次平均。
- N5：AngelSlim 编码器误差：LS 缩放 -89.7% 加权 SSD，imatrix 项再降剩余 4.1%；1200 行真实专家权重。来源：[HF]。
- N6：Hy4-preview-STQ1_0.gguf 213.66 GiB、平均 2.38 bpw（混合精度整模，STQ1_0 用于其中 29 层专家 gate/up）。来源：[HF] 文件表。仅应用场景一句引用。
