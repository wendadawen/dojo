# Sherry 稀疏三值量化审查记录（第 3 轮）

- 页面版本：index.html blob `06622365b17e47175cd05d318249f1c8a1a67cf9`，overview.html blob `e221703b1da268c302d4b0574857aabb5d6d6412`（目录尚未提交，`git status` 显示 `?? wiki/sherry-ternary-quant/`）
- 审查时间：2026-09-01 17:10 CST
- 审查者：独立子代理（未参与本页面写作，也未参与第 1、2 轮审查与修复）
- 已完整阅读章节（按顺序）：
  1. 前置：reading-time / blockquote.meta / 引言 / 核心问题 / 常见误解
  2. §1 三值量化的打包困境
  3. §2 3:4 稀疏三值——4 个权重恰好 5 bit（§2.1 合法组合怎么数、§2.2 5 bit 的内部结构、§2.3 一组权重的编码手算）
  4. §3 STQ1_0 字节布局——1.3125 bpw 的账（§3.1 42 字节的构成、§3.2 stride-16 分组）
  5. §4 量化决策——缩放系数与零位怎么选（§4.1 缩放系数的三种选法、§4.2 零位的两种选法、§4.3 误差实测及代码折叠块）
  6. §5 为什么快——SIMD 解码与实测（§5.1 解码不需要位重排、§5.2 实测数字）
  7. §6 训练侧的 weight trapping 与 Arenas——本页边界
  8. 来源与范围说明（C / F / N / 构造示例 / 简化条件）
  9. overview.html 全文（含首页链接、核心机制、关键结论与边界、混合精度链接）
- 已运行的机械验证：
  - 提取并执行页面 §4.3 完整 Python 代码（`/tmp/review3_sherry_code.py`），使用 `/Users/wendadawen/.workbuddy/binaries/python/versions/3.13.12/bin/python3`。实际输出 14 行（含两条断言）逐字符与页面预期输出一致。
  - 运行 `.dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html`，返回 `validation ok`。
  - 独立复算：`log2(3) ≈ 1.585` ✓，`log2(81) ≈ 6.34` ✓，`C(4,3)·2^3 = 32 = 2^5` ✓，`5/4 = 1.25` ✓，`16 = C(4,3)·2^(3-1) = 16` ✓，`2:4 模式 = C(4,2)·2^(2-1) = 12` ✓，`42·8/256 = 1.3125` ✓，`16/256 = 0.0625` ✓，`(2-1.585)/2 = 20.75% ≈ 20.7%` ✓，`148.27/116.83 ≈ 1.269`（+27%）✓，`147.47/109.62 ≈ 1.345`（+35%）✓，`358/401.5 ≈ 89%`（小 11%）✓，`358/445 ≈ 80%`（小 20%）✓，`358/336.25 ≈ 1.065`（大 6%）✓，`117.5101/19.6010 ≈ 5.99`（约 6 倍）✓。
  - 引用目标检查：`../mixed-precision-quant/index.html` 存在 ✓；`overview.html` ↔ `index.html` 互链 ✓；`../../index.html`（Dojo 首页）存在 ✓。

## 来源核对摘要

- **Sherry 论文 arXiv:2601.07892v1**（提交日期 2026-01-12 已从 abs 页确认 `Submitted on 12 Jan 2026`，但 abs 页 Comments 为空、HTML 全文无会议标注）：
  - C1（§2.1 三值定义 `{−1,0,+1}` + 缩放 α，`Q(W)=Tα`）：论文 §2.1 原文 `Ternary quantization is an extreme weight compression paradigm that constrains model parameters to the discrete set {−1,0,+1}` 与 `Q(W)=Tα`，Tα 按列乘以缩放因子 α；页面使用 `{−d,0,+d}` 与 `x ≈ d·q` 的等价写法（与 PR #22836 引言 `each weight is constrained to {-d, 0, +d}` 一致），数学上同义。✓
  - C2（§2.3 / §3.1 1.67-bit 缺陷 + Table 4 速度）：§2.3 原文 `1.67-bit Strategy: This scheme packs three weights into 5-bit blocks ... severe arithmetic inefficiencies because modern hardware accelerators are optimized for 2^n operand groupings`；§1 摘要 `3-way grouping is fundamentally incompatible with the power-of-two vector lanes`；Table 4 `TL2 116.83 t/s` 与 `I2_S 132.13 t/s` 全部逐数字与页面一致。✓
  - C3（§2.3 理论下限 `log2 3 ≈ 1.58`）：论文 §2.3 原文 `theoretical lower bound of 1.58 bits (log₂3)`；页面用 1.585（更精确），数学正确。✓
  - C4（§3.1 3:4 约束、组合数 32=2^5、§1 定义）：§3.1 原文 `enforcing N=3 non-zero elements yields exactly (⁴₃)×2³=32 unique permutations. This mathematically saturates a 5-bit index (2⁵=32)`。✓
  - C5（附录 A 4-bit index + 1-bit sign、镜像对称、16 模式码本）：附录 A 原文 `4-bit index represents the magnitude/sparsity pattern within the block, and a 1-bit value represents the shared or dominant sign`；附录 C.2 原文 `A 3:4 ternary block with one shared sign bit has a total of C₄³·2³⁻¹=16 unique patterns. This perfectly saturates the 2⁴ entries in the LUT`。✓
  - C6（§3.1 + 附录 C 2:4 12 状态、GPU Tensor Core 面向 fp16/fp32、LUT 16 项 / 4-bit 索引上限）：附录 C.1 原文 `Standard SIMD instructions (e.g., x86 AVX2 vpshufb) utilize a 128-bit (16-byte) register as a lookup table. This constrains the index to 4 bits (2⁴=16 entries), implying B−1≤4`；附录 C.2 原文 `a 2:4 scheme only utilizes C₄²·2²⁻¹=12 states, resulting in bit-waste`；附录 B.3 原文 `designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic` 与 `most efforts are not coordinated with ultra-low bit quantization`。✓
  - C9 / F4（§3.1 Sparse-AbsMean、附录 D 最优性证明）：§3.1 Eq. 5 `αj* = (4/(3·d_in)) Σ_{i∈S_j} |Wi,j|, where S_j = {i | Ti,j ≠ 0}`；附录 D 证明 Sparse-AbsMean 对 L2 重建误差最优（Eq. 10–13）。页面用组内形式 `d = (1/|S|) Σ |x_i|` 并在来源说明注明「论文原文按输出通道给出，二者同义」。✓
  - C12（§3.2 weight trapping、ER<750、维度 4096、Arenas `Y = XTα + λt·XW`、λt 退火到 0）：§3.2 原文 `Specifically, the 3:4 sparse regime exhibits a low ER (ER<750), a level of spectral collapse comparable to that of binary quantization, despite the gradient matrix having a total dimensionality of 4096`；Eq. 7 `Y = XTα + λt·XW`；`As λt→0, the residual vanishes, leaving a pure 3:4 sparse ternary model for inference with zero additional overhead`。✓
  - C14 / N3 / N4（Table 1 精度、Table 4 速度、QAT 条件）：Table 1 1B Sherry 平均 0.519 vs Tequila 0.519（持平）、3B 0.567 vs 0.576、ARC-C 1B 0.309>0.305 / 3B 0.364>0.346 全部逐项与页面一致；Table 4 0.7B 与 3B 全部数字（34.01/132.13/116.83/148.27、7.55/41.87/38.80/45.55、1360.0/256.56/233.44/205.50、6190.0/873.65/846.01/712.40）逐数字核对通过；§4.1 `LLaMA-3.2-1B and LLaMA-3.2-3B`、`10B tokens sampled from the UltraFineWeb dataset`、`three independent runs with random seeds` 均一致。✓
  - F1、F2 已通过 §3.1、附录 C.2 原文复算与核对，公式 `C(4,3)·2^3=32=2^5` 与 `C(4,3)·2^(3-1)=16` 均成立。✓
- **llama.cpp PR #22836**（已通过 PR 页面与 `https://github.com/ggml-org/llama.cpp/pull/22836.diff` 双向核对）：
  - C7 / C8 / C13 / N2：引言原文 `1.3125 bits per weight (5 bits per 4-weight group, i.e. a 4-bit codebook index + 1-bit sign, plus a single fp16 scale per 256-weight block: 42 B / 256 = 1.3125 bpw)`；Performance 表全部 8 行数字与页面 §5.2 表格逐行一致（STQ1_0 358.00 MiB pp512 732.69±20.00 tg128 147.47±1.36；TQ1_0 401.50 MiB 728.69±19.88 138.87±0.96；TQ2_0 445.00 MiB 689.25±16.61 175.06±1.30；Q1_0 336.25 MiB 768.47±14.75 109.62±16.93）；实验条件 `Apple M4 Pro (12 cores: 8 P + 4 E, 24 GB unified memory, macOS 26.3.1)`、`-ngl 0`、`8 threads`、`community-reproduced Sherry-1B reference checkpoint MoraxGeo/Sherry-1B-1.25bit-per-channel`、TQ1_0 `1.6875-bit 3-way packing` 全部支持页面表述。✓
  - C8（stride-16 分组 + 激活对齐）：PR 原文 `STQ1_0 groups weights that are stride-16 within each 64-weight chunk (e.g. w0, w16, w32, w48)` 与 `standard Q8_K activation quantization stores y values sequentially in memory, the lane contents and y are naturally aligned — a plain vld1q_s8 at offsets 0/16/32/48 is all that is needed, with no deinterleave or repack required`。✓
  - C7（`vqtbl2q` ARM NEON 解码）：PR 引言 `decode directly through vqtbl2q + vdotq_s32 with no bit-shuffling`。✓
  - **diff 直接验证字节布局**：PR diff 在 `ggml/src/ggml-common.h` 新增
    ```c
    // 1.3125 bpw
    typedef struct {
        uint8_t qs[QK_K/8];    // 32 B
        uint8_t sign[QK_K/32]; // 8 B
        ggml_half d;           // 2 B
    } block_stq1_0;
    ```
    与页面 §3.1 dg-stack 图示「qs 32B → sign 8B → d 2B 自上而下为文件中的存储顺序」完全一致。traits 中 `.blck_size = QK_K = 256`、`.type_size = 42` 字节、`.vec_dot_type = GGML_TYPE_Q8_K` 与 `blck_size / type_size` 在 `gguf-py/gguf/constants.py` 同样记录为 `(256, 42)`。这反向印证了页面「激活按 Q8_K 量化」「256 权重一块」等机制描述。✓
  - C13「1-bit binary vs 1.25-bit ternary ~3pp 精度差距」：PR Performance 节 `per the Sherry paper (Fig. 6), 1.25-bit ternary matches 1.67-bit ternary accuracy at 25% fewer bits, while the 1-bit binary configuration shows a ~3 pp accuracy gap`。论文 HTML 中 Fig 6 标题为 `Figure 6: Ablation study of Arenas`，内嵌 `<object data="2601.07892v1/ablation.svg">`；本审查环境无 SVG 渲染工具（无 cairosvg / rsvg-convert / qlmanage 受 sandbox 限制），论文正文文本未直接给出 3pp 数字，因此该数字仅经 PR（C13）作为可定位依据，原始出处为 Fig 6 柱状图。✓（经 PR 中转）
- **HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡**：
  - C10 / C11 / F5 / F6 / N5：模型卡英文 §3 原文 `Upstream's quantizer targets QAT inputs already on the ternary grid: it ignores the imatrix, sets d = amax, and zeroes argmin |x|` 与 `Weighted least-squares scale, d = sum(w*sel*x) / sum(w*sel^2) instead of d = amax`、`zero the lane minimising w[j]*(x[j]^2 - (|x[j]|-d)^2), the incremental cost rather than the smallest magnitude`、`alternating for 3 rounds. Measured on 1200 real expert rows: the LS scale alone gives -89.7% weighted SSD, and the imatrix terms a further -4.1% of the remainder`。与页面 F5、F6、N5 逐项一致。✓
  - N6：文件表 `Hy4-preview-STQ1_0.gguf 213.66 GiB 2.38 bpw`；正文 `The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48`；侧栏 `770B params`。与页面 N6 / 开头概览一致。✓
- **腾讯混元官方文章**（通过 WebSearch 取到 IT之家、新浪/快科技、网易等的 2026-09-01 转载全文）：
  - C15：原文 `把 STQ1_0 和几种常见低比特格式放在同一算子上比,STQ1_0 在比特数最低的同时,速度和 IQ1_M 基本持平,明显快过同样想冲低比特的 IQ1_S`；与页面 C15 一致。✓
  - N6 中「1.5 TB → 约 214 GB」与「770B 参数」：原文 `这一模型权重接近 1.5 TB... 把 Hy4 preview 的权重缩小到约 214 GB`、`总参数量770B(7700亿),激活参数49B`。与页面一致。✓
  - 原文亦支持「1.25 比特 = 5 bit/4 权重」「计入分块共享的缩放系数等额外开销,落到文件里的实际开销约 1.31 bpw」等机制描述。✓

## 问题

- [轻微·技术·来源] index.html meta blockquote 与「来源与范围说明」§C 段开头：标注「ACL 2026 接收」为论文元数据。｜引文依据：arXiv abs 页 Comments 为空、论文 HTML 全文无任何会议标注；唯一来源是 PR #22836 引言 `which recently accepted to ACL 2026`。｜修复要求：在 meta blockquote 与「来源与范围说明」中明确「ACL 2026 接收」说法的依据为 llama.cpp PR #22836 引言（不挂在 arXiv 论文本身），例如改为「Sherry 论文（arXiv:2601.07892，2026-01-12；据 llama.cpp PR #22836 称已接收至 ACL 2026）」，并在来源章节把此说法并入 C13 或新增 C16 标注来源。｜修复：blockquote.meta 改为「Sherry 论文（arXiv:2601.07892；llama.cpp PR #22836 述及其被 ACL 2026 接收）」；来源章节（C）开头改为「其被 ACL 2026 接收一事出自 llama.cpp PR #22836 的述及，arXiv 页未标注」。｜复验：已复验。

- [轻微·技术·来源] index.html §1 段落 2（`1. 三值量化的打包困境`）首段对 2-bit 打包的描述：`|引文依据：论文 §2.3 原文`2-bit Strategy: This approach pads each ternary weight into 2 bits (Wei et al., 2025), as shown in Fig. 2 (left). While it preserves computational regularity and aligns with SIMD vector lanes` 与 Fig 2 caption `2-bit strategy packs each weight into 2 bits to maintain alignment, resulting in large bit wastage` —— 论文支持「每权重 2 bit、保持 SIMD 对齐、大量 bit 浪费」三项；但页面该段未挂任何引用编号。｜修复要求：在该段末（`每权重比理论下限多花约两成——2-bit 相对下限的浪费是 (2-1.585)/2 ≈ 20.7%` 句后）加引用 `[C2]`，将「三个取值映射到四个码点中的三个」「解码简单」并入该引用所支撑的事实集合；如需把 `解码简单` 明确化，可在 (Wei et al., 2025) 对应表述 `preserves computational regularity` 上挂同一引用。｜修复：该段末补 [C2] 引用；同段浪费口径统一为「多出 0.415 bit、相对下限约 26%」（同时解决第 5 条口径不一致）。｜复验：已复验。

- [轻微·技术·来源] index.html §5.2 表后段落：「2-bit 解码无需查表，用存储换速度，这正是『三值量化的打包困境』一章的权衡在数字上的重现」。｜引文依据：论文 §2.3 与 Fig 2 描述 2-bit 策略仅说 `preserves computational regularity and aligns with SIMD vector lanes`，未直接出现「无需查表」字样；PR #22836 文本未在 Performance 节前说明 TQ2_0 解码路径。`用存储换速度` 是从表格数字（445 MiB / 175.06 t/s 对 358 MiB / 147.47 t/s）直接可观察，但「无需查表」属于未经定位引用的机制描述。｜修复要求：把「无需查表」改为基于 2-bit 编码规则的直接推断并明确标注，例如改为「2-bit 每权重自含符号与幅度（参见 §1 表与论文 §2.3），解码无需查码本」并在句末保留「用存储换速度」属于数据事实；或直接删去「无需查表」四字，仅保留「用存储换速度，这正是 §1 权衡在数字上的重现」。｜修复：改为「按 TQ2_0 每权重整 2 bit 的编码结构推断，其解码可直接取符号位而无需查表，用存储换速度（推断，非来源结论）」。｜复验：已复验。

- [轻微·格式] index.html「核心问题」第 1 题解答末句：「完整论证见第 1、2 章」。｜引文依据：style-guide §1 规定「正文引用其他章节时不使用 S1/S2/S3 等章节代号。正文引用其他章节时使用章节标题」；同页其他位置（§3.1、§4.1、§5.1、§4.3 折叠块、§6 边界声明）均使用章节标题引用，例如「『量化决策——缩放系数与零位怎么选』一章」「『为什么快——SIMD 解码与实测』一章」。Q1 解答是唯一使用章号的位置。｜修复要求：把「完整论证见第 1、2 章」改为「完整论证见『三值量化的打包困境』与『3:4 稀疏三值——4 个权重恰好 5 bit』两章」。｜修复：已改为「完整论证见『三值量化的打包困境』与『3:4 稀疏三值——4 个权重恰好 5 bit』两章」。｜复验：已复验，全文无裸章号引用。

- [轻微·可读性] index.html §1 段落 2 第 2 句：「每权重比理论下限多花约两成——2-bit 相对下限的浪费是 (2-1.585)/2 ≈ 20.7%」。｜引文依据：`log2(3) ≈ 1.585`，`log2(3) ≈ 1.58`（论文 §2.3），算术 `(2-1.585)/2 = 20.75%` 正确；但「多花约两成」在中文中自然读作「多花 ~20%」，对应 `(2-1.585)/1.585 ≈ 26.2%`；公式分母为 2（实为相对 2-bit 自身位宽的占比），而标签「相对下限的浪费」暗示分母是下限 1.585，文字与公式口径不一致。｜修复要求：统一口径，二选一：（a）把整句改为「2-bit 比理论下限多出约 26.2%（多出的 0.415 bit 占 2 bit 的 20.7%）」并保留公式；或（b）把「相对下限的浪费」改为「相对 2-bit 位宽的浪费占比」并删除「比理论下限多花约两成」。两种修法中 (a) 信息更完整。｜修复：按口径 (a) 改写（多出 0.415 bit、相对下限约 26%），并同步统一表格行、核心问题 1 解答、本章问题解答 1 与 overview 的「约两成」为「约四分之一」。｜复验：已复验，全文口径一致。

- [轻微·可读性] index.html §5.2 表后段落：「相对 1-bit 的 Q1_0，STQ1_0 只大约 6% 的体积，tg128 高约 35%」。｜引文依据：算术 `358.00 / 336.25 ≈ 1.0648`（约大 6%）、`147.47 / 109.62 ≈ 1.345`（约高 35%）正确；但「只大约 6% 的体积」缺少谓语动词，自然解读不通（「只大 6%」或「只多 6%」）。｜修复要求：补出谓语，改为「STQ1_0 体积只大约 6%（358 / 336.25 ≈ 1.065），tg128 高约 35%」或「STQ1_0 只大 6% 的体积，tg128 高约 35%」。｜修复：改为「STQ1_0 的体积只大 6%，tg128 高约 35%」。｜复验：已复验。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：**修复 6 项轻微后发布**（修复后需重新运行 `.dojo/scripts/validate.py` 确认仍通过）。
- 已完成的关键核对：
  - 全部 §1–§6 正文章节与两块图示（§2 dg-flow 流程图、§3 dg-stack 字节布局图）按顺序完整阅读并核对机制描述与数据。
  - 全部 5 个核心问题与 10 个章节问题的「解答：」折叠块均独立可读、与正文结论一致；核心问题答案在末段指明完整论证所在章节。
  - 全部外部来源论断（论文 7 处节段 + 1 个 Table 1 + 1 个 Table 4、PR 引言 + Performance + Stride-16 节 + diff 源码、模型卡英文 §1/§3、中文转载全文）均打开并定位到具体段落/表格/文件，给出原文片段。
  - §4.3 折叠块中的 Python 代码完整提取并执行（`python3 3.13.12`），实际输出 14 行逐字符与页面预期输出一致，包括三组加权 SSD、重要性全 1 时两规则一致（True）、贯穿示例的 q/deq/err/SSD、amax 对照、组合数与 42 字节账。
  - 全部结构图为 HTML 节点（`.dg-flow` / `.dg-stack`）与内联公式标签 `$\times$`、`$\{−d,0,+d\}$`、`$d$`，无 SVG `<text>` 近似，无等宽字符框线图。
  - 全文无 Unicode 数学字符直接出现在标题、summary、正文、列表与表格中（仅代码块按代码原样保留，符合 style-guide §11 第 2 段豁免）。
  - `validate.py wiki/sherry-ternary-quant/index.html` 返回 `validation ok`。
  - 引用目标 `../mixed-precision-quant/index.html` 存在，`overview.html` ↔ `index.html` 互链。
  - 头部 `<meta>` 包含 description（纯文本）、dojo:summary（使用 `$...$` 与 `\approx`、`\binom` 等可被 KaTeX 渲染的 LaTeX）、dojo:type=concept、dojo:topics=推理系统、dojo:tag。

## 发布条件逐项判定（按 §5）

- [x] 三轮审查均已完成，每轮由未参与写作的独立审查者执行（本次为第 3 轮；第 1、2 轮 review-1.md / review-2.md 存在于 research/，由其他独立审查者完成）。**附注**：本次执行者本身是独立子代理，符合"未参与写作也未参与前序轮次"的要求。
- [x] 每条来源论断都有引文依据记录；唯一无法从论文正文直读的数字（1-bit vs 1.25-bit 3pp 差距）以 PR C13 文本为定位依据，并在记录中明示原始出处为论文 Fig 6。
- [x] 所有阻断和重要问题均已关闭（本次发现 0 / 0）。
- [x] 遗留轻微问题具有明确的接受理由（见上 6 条，均为表述层 / 引用挂载 / 格式一致性，不影响核心结论；逐条给出可执行的修复要求）。
- [x] 全部学习目标由正文章节完整回答（核心问题 5 题对应 §1+§2 / §2 / §3 / §4 / §5；章节问题 10 题均作答）。
- [x] 核心问题 5 题与每章本章问题（§1×2、§2×2、§3×2、§4×2、§5×2、§6×1）均有 `<details><summary>解答：…</summary>…</details>` 折叠块，无只列问题未作答的情况。
- [x] 数学符号全部使用 LaTeX 书写（`$...$` / `$$...$$`），结构图为 HTML（`.dg-flow` / `.dg-stack`），无等宽字符框线图。
- [x] `.dojo/scripts/validate.py` 返回成功。
- [x] 可运行代码（§4.3 Python 折叠块）的结果与页面描述一致（逐字符核对 14 行）。
- [x] 关键论断和数字已重新核对来源：组合数 32/16、1.25 bit/权重、42 字节 1.3125 bpw、Sparse-AbsMean 公式、WLS 公式、imatrix 增量代价公式、M4 Pro 8 项速度、Table 4 全部 16 项、Table 1 全部 10 项平均、ER<750/4096、Arenas `Y=XTα+λt·XW`、1200 行 -89.7%/-4.1%、Hy4 213.66 GiB/2.38 bpw/770B、1.5 TB→214 GB、IQ1_M 持平/IQ1_S 更快，全部已在来源核对摘要中给出原文片段。
- [x] 页面 `<head>` 包含有效纯文本 description、可渲染 dojo:summary（用 `$...$`）、dojo:type=concept、dojo:topics=推理系统（validate.py 已校验在固定大类词表内）、dojo:tag=权重量化,三值量化,Sherry,STQ1_0,llama.cpp,GGUF,Hy4。
- [x] `overview.html` 与 `index.html` 相互链接（overview.html 内含 `index.html` 链接与 `../../index.html` 首页链接；index.html 含 `overview.html` 链接）。
- [x] 页面引用的概念页 `../mixed-precision-quant/index.html` 存在（已 `ls` 确认），且为 `MIX-STQ1_0 逐层混合精度量化` 概念页。
- [ ] **递归生成的前置概念页 `mixed-precision-quant` 已完成各自质检** — 本次审查范围仅限 `wiki/sherry-ternary-quant/`，未读取 `wiki/mixed-precision-quant/research/` 下的任何文件。需编排者在发布前确认该前置页已通过自身三轮质检。

**综合判定**：本次审查未发现阻断或重要问题，全部 6 项轻微均已给出可执行的修复要求与引文依据；待 6 项轻微修复并复跑 `validate.py` 确认仍通过后即可发布。最终发布决策需在修复完成后由编排者结合前置页 `mixed-precision-quant` 的质检状态做出。

## 发布记录（编排者，2026-09-01）

- 6 项轻微已全部修复并复验（见各条目「修复/复验」栏）；修复后 `validate.py` 返回 validation ok；无头 Chrome 渲染探针 159 个 .katex 节点、正文无残留 $ 定界符。
- 前置概念页 mixed-precision-quant 的三轮质检由编排者另行完成（见该页 research/ 审查记录），其发布判定不晚于本页。
- 发布决定：三轮完成、阻断与重要问题全部关闭、机械验证通过，本页发布。


## 发布记录（编排者，2026-09-01）

- 6 项轻微已全部修复并复验（见各条目「修复/复验」栏）；修复后 `validate.py` 返回 validation ok；无头 Chrome 渲染探针 159 个 .katex 节点、正文无残留 $ 定界符。
- 前置概念页 mixed-precision-quant 的三轮质检由编排者另行完成（见该页 research/ 审查记录），其发布判定不晚于本页。
- 发布决定：三轮完成、阻断与重要问题全部关闭、机械验证通过，本页发布。
