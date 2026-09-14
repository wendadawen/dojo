<!-- review-meta
round: 9
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: 516c1bee3bd317ea
-->
# Sherry 稀疏三值量化审查记录（第 9 轮）

- 页面版本：3589fa67e60473bcc0dedbb5804e85b52f1f6805（index.html 工作树哈希）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，也未参与前序轮次的审查与修复）
- 适用规范：guides/concept/check.md（页面 head `dojo:type=concept`）
- 已完整阅读章节：核心问题（5 条及其解答）、常见误解、1. 三值量化的打包困境（含本章问题）、2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1/2.2/2.3 及本章问题）、3. STQ1_0 字节布局——1.3125 bpw 的账（3.1/3.2 及本章问题）、4. 量化决策——缩放系数与零位怎么选（4.1/4.2/4.3 及本章问题，含代码折叠块与预期输出）、5. 为什么快——SIMD 解码与实测（5.1/5.2 及本章问题）、6. 训练侧的 weight trapping 与 Arenas（及本章问题）、来源与范围说明（C/F/N、构造示例、简化条件）、两处图注、overview.html
- 核对版本：Sherry 论文 arXiv:2601.07892**v1**（arXiv HTML，2026-01-12 提交，含 §2.1/§2.3/§3.1/§3.2/§4.1/§4.2、附录 A/B.3/C/C.1/C.2/D/E.2/F、Table 1、Table 4）；llama.cpp PR #22836（本轮抓取的 .diff，含 `stq1_0_codebook` 定义）及其 Performance 表与引言；HuggingFace `AngelSlim/Hy4-preview-GGUF` 模型卡（raw/main README，本轮抓取）

## 已核对且无问题

- **代码实跑**：本地执行页面 Python 代码（`random.seed(20260901)`，Python 3），输出与「预期输出」逐行一致：`d=1.1743/加权SSD=117.5101`、`d=0.5152/19.6010`、`d=0.5381/17.7533`、`重要性全 1 时…一致: True`、`q=[1, 0, 1, -1]`、`SSD=0.0300`、`amax d=0.7: SSD=0.0600`、`42B`、`bpw = 1.3125`。
- **数字回源**：M4 Pro 表四行（358.00 MiB / 732.69±20.00 / 147.47±1.36；401.50 / 728.69±19.88 / 138.87±0.96；445.00 / 689.25±16.61 / 175.06±1.30；336.25 / 768.47±14.75 / 109.62±16.93）与 PR Performance 表逐项一致；论文 Table 4（0.7B Sherry 148.27/205.50、TL2 116.83/233.44、I2_S 132.13/256.56、BF16 34.01/1360.0；3B 45.55/712.40、38.80/846.01、41.87/873.65、7.55/6190.0）与 Table 1（1B Tequila 0.519 / Sherry 0.519，ARC-c 0.305 vs 0.309；3B 0.576 vs 0.567，ARC-c 0.346 vs 0.364）一致；复算 148.27/116.83≈27%、45.55/38.80≈17.4% 与 §4.2 原文「Sherry achieves a 18% speedup over the 1.67-bit baseline」(3B) 一致；358/336.25≈+6.5%、147.47/109.62≈+35%、401.5/358≈+11%、445/358≈+20% 均与 PR 引言表述一致。模型卡：213.66 GiB / 2.38 bpw / 770B / 1200 行上 −89.7%、imatrix 再 −4.1% 一致。
- **公式**：`Y = XTα + λ_t XW` 与论文 §3.2 原文一致；有效秩 `ER < 750`、总维度 `4096` 与 §3.2 一致；本页 d 的组内平均形式与附录 D 的 `α* = (4/(3·d_in))·Σ|W|`（按列）同义（已复算：per-column active 数 = 3·d_in/4，故二者相等）；`3^4=81`、`log₂81≈6.34`、`32/81` 降幅≈60%、`2−1.585=0.415`、`0.415/1.585≈26%`、`(32+8+2)×8/256=1.3125`、`16/256=0.0625`、`x²−(|x|−d)²=2|x|d−d²` 均可复算。KaTeX 定界符、符号写法（`d`、`q_i`、`w_i`、`\alpha`）全页单义；`dojo:summary` 内 `$3^4=81$`、`$32=2^5$`、`$1.3125$` 可渲染。
- **机械项**：`.dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html` 返回 `validation ok`；`dojo:topics=推理系统` 在 ALLOWED_TOPICS、`dojo:tag=量化` 在 ALLOWED_TAGS；C/F/N 上标与来源小节双向对应；跨页链接 `../hy4-preview-lite/index.html`（存在，dojo:type=note）与 `../mixed-precision-quant/index.html`（存在，dojo:type=concept）有效，无「（待生成）」；结构图为 `dg-flow`/`dg-stack` HTML 结构（非等宽字符框线图）；无 `<img>`，故无 alt 内 `$...$`；无 Unicode 数学字符（validate.py 通过）。
- **表述**（复核后不构成问题）：「本页依次回答…」（style-guide §7 允许开篇说明文章结构）、「本页用到三个背景概念」「本页的量化演示…」「本页写为组内形式…」（style-guide §12 允许以「本页」自称）；无第一/第二人称、无调试叙事；「验证的机制／观察重点／预期输出」为全站既有代码示例标签（已在多页复用），不记问题；TQ2_0 免查表的说法已显式标注「（推断，非来源结论）」。

## 问题

- [重要·技术] 2.2「5 bit 的内部结构」（另见 2 章核心问题解答、5.1、3.2）与「来源与范围说明」C5 条：页面把码本描述为 16 项、且断言「码本只存每对中的一份」，并在源注中把 C5 同时归给 llama.cpp PR #22836；但该 PR 新增的码本恰是 32 项、且注释明说符号半是预计算出来的，即两半都存，与页面断言相反。｜引文依据：页面 2.2「码本只存每对中的一份，共 16 种模式，用 4 bit 索引」「16 项码本恰好填满 SIMD 查表指令的 4-bit 索引上限」；PR #22836 .diff 原文 `// Each STQ1_0 group has 4 lanes with exactly one 0 and three non-zero (+1/-1) of equal magnitude — 4 zero-positions × 2^3 signs = 32 patterns. They are split into 4-bit slot + 1-bit sign.`、`// The sign=1 half is precomputed so decode is a single load:`、`//   qpack = stq1_0_codebook[(sign << 4) | slot]`、`GGML_TABLE_BEGIN(uint8_t, stq1_0_codebook, 32)`；模型卡 raw/main：「Each group of 4 weights is a 4-bit code plus a 1-bit table-select, indexing a 32-entry codebook」；论文（arXiv:2601.07892v1）附录 C.2「a 3:4 block with one shared sign bit gives C(4,3)·2^(3−1) = 16 patterns, exactly saturating the 2^4 = 16 LUT entries」、附录 A「a 4-bit index encoding the block's sparsity/magnitude pattern plus a 1-bit sign」。即 16 只是论文的设计口径（4-bit slot 数），页面把「16 种模式」写成了「16 项码本」并加了实现层断言「只存每对中的一份」。｜修复要求：把该句改为区分两种口径——论文设计口径为 16 种模式 + 1 位共享符号位；STQ1_0 实现中的码本表为 32 项（符号位那一半预计算后一并存入），用 `(符号位<<4)|4 bit slot` 一次查表，不做运行期翻转；同时删除或限定「码本只存每对中的一份」这一实现层断言，并把 C5 的来源限定为论文附录 A/C（或保留 PR 但补出上述差异）。｜修复：｜复验：
- [轻微·表述] 引言、4.1、4.3：口语化措辞降低文档语域——引言「用的是本页的主角——Sherry 稀疏三值量化」；4.1「$d$ 被 256 个权重中最大的离群值钉死」、4.3「改进的大头在缩放系数」、4.3 折叠块 summary「$d$ 被单个离群值钉死」。｜引文依据：不适用｜修复要求：改为中性技术表述（如「本页的核心对象」「被…锁定／由…决定」「主要改进来自缩放系数」），保留行业术语其余部分不动。｜修复：｜复验：
- [轻微·表述] overview.html「问题背景」：「必须在两条烂路里选一条」为口语化措辞，与索引页中性表述（「二选一」）不一致。｜引文依据：不适用｜修复要求：改为「两种都有明显代价的打包方式」一类中性说法，与 index.html 第 1 章一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（1 条重要问题需在 2.2、5.1 与 C5 源注同步收紧口径；2 条轻微为用词调整）
