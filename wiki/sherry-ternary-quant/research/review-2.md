# Sherry 稀疏三值量化审查记录（第 2 轮）

- 页面版本：index.html `757f2a575c9067bcfd4e406de2e70b68e594d56f`（overview.html `539e1f724d49f7cb534617b6b25a38e2399e64fa`）
- 审查时间：2026-09-01
- 审查者：独立子代理（未参与写作、未参与第 1 轮审查与修复）
- 已完整阅读章节（按顺序）：核心问题（含全部解答折叠块）、常见误解、1. 三值量化的打包困境（含本章问题）、2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1/2.2/2.3/本章问题）、3. STQ1_0 字节布局（3.1/3.2/本章问题）、4. 量化决策（4.1/4.2/4.3 含代码折叠块/本章问题）、5. 为什么快——SIMD 解码与实测（5.1/5.2/本章问题）、6. 训练侧的 weight trapping 与 Arenas（本章问题）、来源与范围说明（全部小节）、overview.html 全部小节（问题背景/核心机制/关键结论与边界）

## 机械验证结果

- **代码运行**：提取 index.html 第 4.3 节 Python 代码块，用 Python 3.13.12 实际执行，13 行输出与页面「预期输出」逐字符一致（仅提取产物末尾换行符差异，非内容差异）。代码退出码 0。
- **数值复算**：log₂3=1.585、log₂81=6.34、2-bit 浪费 (2−1.585)/2=20.75%（页面「约 20.7%」）、81→32 压缩约六成（60.5%）、42×8/256=1.3125、16/256=0.0625、手算组误差 0.03 与 amax 0.06、x²−(|x|−d)²=2|x|d−d² 及其单调性、2:4 模式数 C(4,2)×2¹=12、构造块 117.51/19.60≈6.0 倍——全部复算通过。
- **比例复核**（相对页面表格数字）：M4 Pro 体积对比 TQ1_0/TQ2_0/Q1_0 分别为 10.8%/19.6%/6.5%（页面「约 11%/20%/6%」）、tg128 相对 Q1_0 快 34.5%（页面「约 35%」）、0.7B 相对 TL2 快 26.9%（页面「复算约 27%」）、3B 相对 TL2 复算 17.4%（页面标注「官方表述为快 18%」，与论文原文一致）、bit 少 25.1%（页面「少 25%」）——全部一致。
- **格式机械检查**：公式定界符（`$...$`/`$$...$$`）之外无 Unicode 数学字符（overview 的「·」为中文间隔号，非数学中点）；全部 17 个 `<summary>` 前缀合规（解答：/代码：）；h2 编号 1–6 连续、h3 编号连续、「本章问题」「来源与范围说明」及其 h3 固定命名合规；前置 section 顺序合规（reading-time→meta→引言→learning-goals→misconceptions→正文）；callout 仅 1 个 yellow（第 4 章注意事项）与 1 个 blue（第 6 章边界声明，属范围说明）；`<meta>` 五项齐备（description 纯文本、dojo:summary 含 KaTeX、dojo:type=concept、dojo:topics=推理系统、dojo:tag）；overview.html 与 index.html 相互链接；本地资源（katex/prism 共 8 个文件）与 `../mixed-precision-quant/index.html` 均存在。
- **公式展示**：6 个独立公式后均紧跟 `<ul>` 逐项定义符号；无自行编号。

## 来源核对记录（引文依据）

以下为逐条打开来源定位核对的结果，编号与页面 sup 标注对应：

**Sherry 论文（arXiv:2601.07892v1，HTML 全文）**

- C1（三值基本形式）：§2.1 "Ternary quantization is an extreme weight compression paradigm that constrains model parameters to the discrete set {−1,0,+1}"，Eq.(1) Q(W)=Tα 及脚注逐元素缩放。页面的 $x \approx d \cdot q$ 为等价记号定义。核对通过。
- C2（1.67-bit 组大小 3 与 SIMD 冲突、解码需位重排）：§3.1(1) "This eliminates the complex bit-shuffling overhead characteristic of 1.67-bit (3-way) packing schemes"；§2.3 "theoretical lower bound of 1.58 bits (log₂3)... architectural friction"。核对通过（但 1.6875 bpw 数字出自 PR，见问题 3）。
- C3（理论下限）：§2.3 "a theoretical lower bound of 1.58 bits (log₂3)"。核对通过。
- C4（3:4 约束与 SIMD 匹配）：§1 "exactly three are quantized to non-zero values (±1), and one is fixed to zero... restores the power-of-two alignment required by modern SIMD units"；§3.1(1) "The choice of M=4 ensures power-of-two alignment"。核对通过。
- C6（vpshufb 16 项 LUT、2:4 共 12 模式、50% 稀疏阈值）：§3.1(4) "Given the 128-bit constraints of standard SIMD register instructions (e.g., AVX2 vpshufb), the maximum capacity for a single-instruction lookup table is 16 bytes"；附录 C.2 "a 2:4 scheme only utilizes C₄²·2²⁻¹=12 states, resulting in bit-waste"、"2:4 sparsity resides exactly on the 50% threshold where performance begins to destabilize"。核对通过（但同段 GPU 稀疏加速表述缺限定，见问题 1）。
- C9（Sparse-AbsMean）：§3.1 "we prune the element with the smallest absolute magnitude"、"the optimal scaling factor α_j is calculated as the mean absolute value of the non-pruned weights"，Eq.(4)(5)。核对通过。
- F1（C(4,3)×2³=32）：§3.1(3) "yields exactly C(4,3)×2³=32 unique permutations. This mathematically saturates a 5-bit index (2⁵=32)"。核对通过。
- F2（16 模式折算）：附录 C.2 "A 3:4 ternary block with one shared sign bit has a total of C₄³·2³⁻¹=16 unique patterns. This perfectly saturates the 2⁴ entries in the LUT"。核对通过。
- F4（AbsMean L2 最优）：§3.1 "The optimization goal of Sherry is to minimize the L₂ reconstruction error"；附录 D "Optimality of the Sparse-Absmean"，Eq.(9)–(13) 给出完整证明。页面 F 条目已注明「本页写为组内形式，论文原文按输出通道给出」，Eq.(5) 的 4/(3·d_in) 因子等价于 1/|S|。核对通过。
- C12（weight trapping / ER<750 / Arenas）：§3.2 "weight trapping: weights accumulate in localized regions due to gradient homogenization"、"the 3:4 sparse regime exhibits a low ER (ER<750), a level of spectral collapse comparable to that of binary quantization, despite the gradient matrix having a total dimensionality of 4096"；Eq.(7) Y=XTα+λ_tXW；"anneals to zero by the conclusion of training"、"the auxiliary path is completely removed post-training, without any inference overheads"。核对通过。
- C14/N3（Table 4）：全部 8 行数字与页面第二张表逐项一致（34.01/132.13/116.83/148.27/7.55/41.87/38.80/45.55 t/s；1360.0/256.56/233.44/205.50/6190.0/873.65/846.01/712.40 MB）；"Inference efficiency on Intel i7-14700HX"；§4.2 "for the 3B model, Sherry achieves a 18% speedup over the 1.67-bit baseline"（页面标注「官方表述为快 18%」准确）。核对通过。
- C14/N4（Table 1）：1B Sherry 0.519 vs Tequila 0.519；3B 0.567 vs 0.576；ARC-c 1B 0.309>0.305、3B 0.364>0.346，原文 "on reasoning-intensive benchmarks like ARC-Challenge, Sherry even outperforms Tequila"；五任务 "PIQA, ARC-Easy (ARC-e), ARC-Challenge (ARC-c), HellaSwag (HelS) and WinoGrande (WinG)"；"10B tokens sampled from the UltraFineWeb dataset"；"All results are averaged over three independent runs with random seeds"。核对通过（概括处缺规模限定，见问题 2）。

**llama.cpp PR #22836**

- 标题："ggml-cpu : add STQ1_0 ternary quantization with ARM NEON vec_dot kernel"，与页面来源章节引用的标题一致。
- C7（256 权重块、42 字节、vqtbl2q）：引言 "a single fp16 scale per 256-weight block: 42 B / 256 = 1.3125 bpw"、"the on-disk layout (qs[32] 4-bit codebook indices + sign[8] 1-bit per group + an fp16 scale)"、"decode directly through vqtbl2q + vdotq_s32 with no bit-shuffling"。核对通过。
- C5（4-bit 索引 + 1-bit 符号、16 项码本）：引言 "5 bits per 4-weight group, i.e. a 4-bit codebook index + 1-bit sign"、"fast SIMD decode through a 32-entry codebook lookup"。核对通过。
- C8（stride-16）："STQ1_0's 3:4 sparsity adopts a stride-16 grouping pattern... instead of grouping 4 consecutive weights (w0, w1, w2, w3), STQ1_0 groups weights that are stride-16 within each 64-weight chunk (e.g. w0, w16, w32, w48)"、"each NEON lane register (sqx0–sqx3) holds a contiguous run of weight indices. Since standard Q8_K activation quantization stores y values sequentially in memory, the lane contents and y are naturally aligned — a plain vld1q_s8 at offsets 0/16/32/48 is all that is needed, with no deinterleave or repack required"。核对通过。
- C13/N2（M4 Pro 基准）：表内 16 个数字与页面第一张表逐项一致（STQ1_0 358.00 MiB、732.69±20.00、147.47±1.36；TQ1_0 401.50 MiB、728.69±19.88、138.87±0.96；TQ2_0 445.00 MiB、689.25±16.61、175.06±1.30；Q1_0 336.25 MiB、768.47±14.75、109.62±16.93）；"Benchmarked on Apple M4 Pro (12 cores: 8 P + 4 E, 24 GB unified memory...) with -ngl 0"；8 线程；Q1_0 无 bpw 标注（页面「—」处理准确）；"the 1-bit binary configuration shows a ~3 pp accuracy gap"（页面 C13 引用准确）。核对通过。
- TQ1_0 1.6875 bpw：PR 引言 "faster than TQ1_0, whose 1.6875-bit 3-way packing is SIMD-unfriendly"——数字正确，但归属见问题 3。

**HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡**

- C10（上游 amax + argmin）："Upstream's quantizer targets QAT inputs already on the ternary grid: it ignores the imatrix, sets d = amax, and zeroes argmin |x|."。核对通过。
- C11/F5/F6（WLS 缩放、imatrix 零位、交替 3 轮）："Weighted least-squares scale, d = sum(w*sel*x) / sum(w*sel^2) instead of d = amax"、"Imatrix-aware zero placement — zero the lane minimising w[j]*(x[j]^2 - (|x[j]|-d)^2), the incremental cost rather than the smallest magnitude"、"alternating for 3 rounds"。核对通过。
- N5（1200 行、−89.7%、−4.1%）："Measured on 1200 real expert rows: the LS scale alone gives -89.7% weighted SSD, and the imatrix terms a further -4.1% of the remainder."。核对通过。
- N6（文件表）：`Hy4-preview-STQ1_0.gguf 213.66 GiB 2.38 bpw`；"The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers"；"770B params"；"VRAM for full residency: ~214 GiB (STQ1_0)"。核对通过。

**腾讯混元官方文章（2026-09-01，转载全文）**

- N6 概数："这一模型权重接近 1.5 TB...把 Hy4 preview 的权重缩小到约 214 GB"、"把最激进一档权重压到平均 1.25 比特...落到文件里的实际开销约 1.31 bpw"；转载（IT 之家/网易）："总参数量 770B（7700 亿），激活参数 49B"。核对通过。
- C15（算子对比）："把 STQ1_0 和几种常见低比特格式放在同一算子上比，STQ1_0 在比特数最低的同时，速度和 IQ1_M 基本持平，明显快过同样想冲低比特的 IQ1_S"。核对通过。

## 问题

- [重要·技术] index.html 第 1 章第 3 段（「还有一条看似现成的路是借硬件稀疏」）：页面称 2:4 稀疏「能利用 GPU 的稀疏加速单元」，在三值量化语境下省略了论文的关键限定，扩大了适用范围。论文附录 B.3 明确：GPU 稀疏加速仅适用于高精度模型（"native Tensor Core support for 2:4 sparsity, doubling throughput with minimal accuracy loss **in high-precision models**"），且这类单元「"largely designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic"」——对超低位宽（1.25-bit 三值）量化不可用；论文 §3.1 也表述 2:4 "typically coupled with specific GPU-vendor kernels" 而 Sherry 刻意与之解耦。读者按页面现文会得出「2:4 三值可借 GPU 稀疏加速、只是编码与精度不佳」的错误结论。第 1 章「本章问题」解答 2 同样只列了 12 模式与 50% 稀疏两条理由，未覆盖该限定｜引文依据："Notably, NVIDIA's Ampere and subsequent architectures introduced native Tensor Core support for 2:4 sparsity, doubling throughput with minimal accuracy loss in high-precision models. However, most efforts are not coordinated with ultra-low bit quantization, as they are largely designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic."（附录 B.3）｜修复要求：在该段为「能利用 GPU 的稀疏加速单元」补充限定（如「能利用 GPU 的稀疏加速单元——但那类单元面向 fp16/fp32 高精度运算，对超低位宽三值并不可用」），并同步在「本章问题」解答 2 中补入这一论文给出的理由；修改后第 1 章表格「2:4 稀疏三值」行的「实际后果」列如受影响一并核对｜修复：该段补入限定——GPU 稀疏加速单元（Tensor Core 的 2:4 支持）面向 fp16/fp32 高精度运算、论文明确其与超低比特量化不配合、对三值不可用；「本章问题」解答 2 改为三个理由（稀疏单元不可用、12 模式填不满码本、50% 稀疏度伤精度）；表格 2:4 行后果列为「50% 稀疏度伤精度」未涉及稀疏加速表述，无需改。｜复验：已复验。
- [重要·技术] index.html 核心问题 5 解答末句「精度与 1.67-bit SOTA 持平」、overview.html「关键结论与边界」第 1 条「精度与 1.67-bit SOTA 持平（QAT 条件下）」：缺规模限定，与来源不完全一致。论文 Table 1 中 1B 为 0.519 对 0.519（精确持平），3B 为 0.567 对 0.576（Sherry 低 0.9 个百分点，并非持平）；论文自身表述 "On the 1B model, Sherry matches the average SOTA accuracy exactly" 也仅针对 1B。index.html 第 5.2 节正文已完整给出两个规模的数字，但核心问题解答与 overview.html 是独立入口的概括，读者只看概括会误以为两个规模都持平｜引文依据：Table 1：1B Tequila 0.519 / Sherry 0.519；3B Tequila 0.576 / Sherry 0.567；原文 "On the 1B model, Sherry matches the average SOTA accuracy exactly"｜修复要求：两处概括改为带规模限定的表述（如「1B 上精度持平（0.519 对 0.519），3B 略低（0.567 对 0.576）」或至少限定「1B 上持平」），保持与第 5.2 节正文一致｜修复：核心问题 5 解答改为「精度上 1B 与 1.67-bit SOTA 持平（0.519 对 0.519），3B 略低（0.567 对 0.576）」；overview 同步改写。｜复验：已复验，两处均带规模限定。
- [轻微·技术] index.html 第 1 章第 2 段：「（llama.cpp 的 TQ1_0 加上缩放摊销为 1.6875 bpw）」被句末 [C2, N3] 覆盖，但 1.6875 这一数字出自 llama.cpp PR #22836 引言（"faster than TQ1_0, whose 1.6875-bit 3-way packing is SIMD-unfriendly"），论文只报 1.67；来源章节也未提供该数字的 PR 归属条目，来源编号与实际出处不对应｜引文依据：PR #22836 引言 "whose 1.6875-bit 3-way packing"；论文 §4.1 "we report a bit-width of 1.67 for existing ternary quantization baselines"｜修复要求：为 1.6875 bpw 单独标注 PR 来源编号（在来源章节 C 或 N 条目中补充「TQ1_0 1.6875 bpw 见 PR #22836 引言」），或将其从 [C2] 覆盖范围中拆出｜修复：1.6875 bpw 单独标注 [C13]，来源章节 C 段补「TQ1_0 的 1.6875 bpw 亦出自 PR #22836 引言」。｜复验：已复验。
- [轻微·格式] index.html「来源与范围说明」论断与来源（C）小节：C1、C2、C3、C6 没有给出具体定位（仅 C4、C5、C9/F4、C12、C14/N3/N4 有章节/表号定位），不满足「定位到页面标注的位置」的双向对应要求｜引文依据：不适用（已核对的实际定位：C1 见 §2.1 Eq.(1)，C2 见 §2.3 与 §3.1(1)，C3 见 §2.3，C6 见 §3.1(2)(4) 与附录 C.1/C.2）｜修复要求：在（C）小节为 C1、C2、C3、C6 补充具体位置（如「C1 见 §2.1，C2/C3 见 §2.3 与 §3.1，C6 见 §3.1(2)(4) 与附录 C」）｜修复：来源章节（C）小节重写为逐条定位：C1 见 §2.1、C2 见 §2.3 与 §3.1、C3 见 §2.3、C6 见 §3.1 与附录 C.1/C.2（含稀疏加速单元限定出处）。｜复验：已复验。
- [轻微·可读性] index.html 第 3.2 节：「激活（按 Q8_K 量化）」——Q8_K 首次出现未解释，违反「术语在首次使用时解释」｜引文依据：不适用（Q8_K 为 llama.cpp 的 8-bit 块量化激活格式，PR 原文 "standard Q8_K activation quantization stores y values sequentially in memory"）｜修复要求：在首次出现处补最小含义（如「Q8_K（llama.cpp 的 8-bit 块量化激活格式）」），一行即可，不展开｜修复：首次出现处改为「按 Q8_K 量化，llama.cpp 的 8-bit 块量化激活格式」。｜复验：已复验。
- [轻微·格式] index.html 全文（含图注与核心问题解答）：跨章引用一律使用「第 N 章」编号（如「d 怎么来是第 4 章的内容」「完整论证见第 1、2 章」、第 2 章图注「定零位、符号与缩放 d（第 4 章）」等），未使用章节标题，与格式规范「正文引用其他章节时使用章节标题」不符（「第 N 章」既非 S1 类代号也非章节标题）｜引文依据：不适用｜修复要求：跨章引用改为「第 N 章『章节标题』」或直接使用章节标题（如「第 4 章『量化决策』」；核心问题解答末尾的指明处同样处理）。引用处较多，逐处统一为一种形式｜修复：全文 16 处「第 N 章」引用统一改为章节标题引用（如「量化决策」一章、「STQ1_0 字节布局」一章），含图注与核心问题解答。｜复验：已复验，grep「第 [1-6] 章」为 0。

## 附加说明（不计入问题）

- dojo:topics「推理系统」是否在 AGENTS.md 固定词表内：本轮审查输入范围不含 AGENTS.md，未核对；由 validate.py 在修复后运行时把关。
- 「第 2 章表格 2:4 行的约 1.25 bit/权重（模式用不满）」为页面构造的对比推论，已在单元格内标注「模式用不满」，且有附录 C.2 的 12-states 论断支撑，不单列问题。
- 第 5.2 节「统一转成 bf16 再分别量化为四种格式」：PR 文字明说三种三值格式，Q1_0 数字见于同一 PR 表格，四种格式同一来源为最小合理推断，不单列问题。
- overview.html 的「·」为中文间隔号（排版用途），非 Unicode 数学字符。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复。两个重要问题（2:4 GPU 稀疏加速缺限定、精度持平缺规模限定）修复并复验后可进入第 3 轮审查；4 个轻微问题建议一并修复，如保留需给出接受理由。
