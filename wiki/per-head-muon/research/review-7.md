<!-- review-meta
round: 7
page: wiki/per-head-muon/index.html
reviewed_content_sha256: 6446c781b80ce869
-->
# Per-Head Muon 审查记录（第 7 轮）

- 页面版本：index.html b0ed132cffcab9f92cb4ee9275559c6e8d16d06c；overview.html e94462480cbcc130809fd1a89b3e8f7351ef0f54
- 审查时间：2026-09-13 21:52
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 页面类型：concept（依 `<meta name="dojo:type" content="concept">`），按 guides/concept/check.md 审查
- 已完整阅读章节（含全部 details 折叠块与两幅 SVG 图注）：引言与「主要依据」、核心问题（5 条）、1. 整块正交化——为什么对多头不够好（含本章问题 3 条）、2. 按头正交化——切分动量矩阵并逐头正交化（含伪代码折叠块、本章问题 3 条）、3. 效果与开销——均衡更新尺度与开销变化（含 Gram 对比折叠块、本章问题 3 条）、4. 分布式实现——P2P 参数取回（含本章问题 3 条）、来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；另读 overview.html。

## 已核对来源（原文片段／关键数值）

- C1、C2、C3｜K3 技术报告 §2.5 "Per-Head Muon"（arXiv:2607.24653，HTML 全文与 ar5iv 全文两路抓取一致）："full-matrix orthogonalization treats all heads as a single coupled block,"；"heads with larger gradient or momentum scales dominate the shared update direction,"；"smaller-scale heads receive insufficiently normalized updates"；"partition their momentum matrices along the head dimension and orthogonalize each head's block separately"；"per-head orthogonalization equalizes the update scale across heads"；"yields more balanced learning dynamics across heads and improves training stability at larger scales"；"It also slightly reduces optimizer overhead,"；"Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix"。正文与来源章节引文逐句一致，C3 三条定性结论与表格三行逐行对应。
- C4、C5｜K3 报告 §5.2.2「Memory-Efficient Training」下 P2P-based Muon orthogonalization 小节："The distributed optimizer shards parameters evenly across DP ranks"；"the Newton–Schulz orthogonalization in Muon requires the full parameter matrix"；"performs an all-gather over the entire parameter buffer on every rank…making communication the primary bottleneck at scale"；"retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication"；"eliminating the full-parameter buffer"；"at the granularity of model-chunk buffers"。与 C4/C5 引文一致；「均匀分片」「消除全参数缓冲区」「model-chunk 流水化」均有原文对应。
- F2｜Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"（kellerjordan.github.io/posts/muon/，2024-12-08）：目标写作 `Ortho(G) = arg min_O { ||O - G||_F : either O^T O = I or O O^T = I }`，并称 "This is equivalent to replacing the update by UV^T, where USV^T is its singular value decomposition (SVD)"。博客标题、日期（2024-12）与「argmin／等价 UV^T」表述均与 F2 相符。
- N 节自述｜Moonlight 报告 arXiv:2502.16982 摘要原文 "Muon achieves ~2× computational efficiency compared to AdamW with compute optimal training"，页面「不把它归因于 per-head 改动、故不引用」的处理与该摘要一致。
- 构造示例复算（numpy）：M 奇异值 (5.02493781, ~0)，`M M^T = [[25,2.5],[2.5,0.25]]`、迹 25.25、行列式 0 均与正文一致；`u1 ≈ [0.99503719, 0.09950372]`、`v1 = [0.6,0.8]`、`Ortho(M) ≈ [[0.59702,0.79603],[0.05970,0.07960]]`、两行范数 0.99504／0.09950 与正文写的 0.597/0.796/0.0597/0.0796、0.995、0.0995 一致；第 2 章 `[3,4]/5 = [0.3,0.4]/0.5 = [0.6,0.8]` 一致。第 1 章本章问题两例（[1,0]/[0,2]、[1,0]/[1,1]）行满秩判断正确。Gram 开销比 `H d_h^2/(H d_h)^2 = 1/H` 一致。
- 引文编号核对：全文 15 处 `<sup>` 标注（C1×3、C2×3、C3×3、C4×2、C5×2、F1×1、F2×2）与来源章节条目一一对应，无错位。
- 机械项：`dojo:topics=训练与优化`、`dojo:tag=优化器` 在词表内；前置概念链接 `../muon-optimizer/index.html`、`../newton-schulz/index.html` 真实存在且非占位；overview.html 与 index.html 双向互链；`python3 .dojo/scripts/validate.py wiki/per-head-muon/index.html` 返回 `validation ok`（退出码 0）；无「（待生成）」占位、无 Unicode 数学字符裸用、SVG 内公式均在 foreignObject、`<text>` 无 ASCII 近似、alt/aria-label 无 `$...$`。
- 表述通读：全文（含折叠块与图注）未发现元话语套语（"本页将…""下面来看…""需要注意的是"）、第二人称、调试叙事或临场评价；自称统一为「本页／本文」，符合 style-guide 第 12 节；两处类比均就地给出失效边界。核心问题 5 条、四章本章问题各 3 条，均有「解答：」折叠块且答案独立可读、与正文结论一致，核心问题答案均指明完整论证所在章节。

## 问题

- [轻微·技术] 第 1 章开头正交化定义句（含第 114、168 行的 `把所有奇异值都设为 1，得到半正交矩阵 $U V^\top$`）：该定义对页面自己使用的秩亏动量矩阵不成立，与同章示例互相不一致——按字面把两个奇异值都设为 1 会得到两行范数均为 1 的结果（无任何压低），而同章示例实际只把非零奇异值拉平、保留零奇异值，才得出小头被压到约 0.1。页面全程未说明零奇异值为何不参与拉平，读者按定义复算示例会得到「两头都被拉平、不存在压低」的相反结论。｜引文依据：本页第 1 章示例 M 的奇异值为 (5.0249, 0)（`M M^T` 迹 25.25、行列式 0），示例结果 `Ortho(M) ≈ [[0.597,0.796],[0.0597,0.0796]]`、两行范数 0.995／0.0995；按「所有奇异值设为 1」复算 `U V^T` 两行范数均为 1.0（本次复算值）。｜修复要求：把该定义句限定为「把所有非零奇异值拉平为 1，并注明矩阵行满秩时 $U V^\top$ 才是半正交矩阵」，或在示例前补一句「NS 正交化保持矩阵的秩，零奇异值不参与拉平」。｜修复：｜复验：
- [轻微·技术] 引言（第 67 行）、第 2 章（第 235 行）、第 3 章（第 296、333 行）及「简化条件及其限制」简化一（第 474 行）对按头正交化的结论：均无条件断言「按头正交化后每个头块的每一行为单位向量，该值为 $\sqrt{d_h}$」「每个头各自获得正交化本应给的行块范数 $\sqrt{d_h}$」。页面在第 1 章已为全矩阵情形补出前提（`rank(M) = H d_h` 行满秩时各头行块范数为 $\sqrt{d_h}$，秩亏时才被压低），但同一前提未施加于每个头块：当某个 $M_h$ 自身行不满秩（$d_h>1$ 时可能发生）时 $\mathrm{Ortho}(M_h)$ 的行范数不足 1，该头行块范数小于 $\sqrt{d_h}$，各头范数并不一致，「各头更新幅度趋于一致」随之失效。本页构造示例取 $d_h=1$，不触发该情形，故正文与本章问题均未暴露。｜引文依据：本页自证判据（`rank(M) < H d_h` 时 $U V^\top$ 部分行块范数小于 $\sqrt{d_h}$）移植到单头块；反例复算：$M_1=[[1,0],[0,0]]$（$d_h=2$，行不满秩）正交化后块 Frobenius 范数 1.0，$M_2=I_2$ 为 1.4142，二者不等且前者小于 $\sqrt{2}$。｜修复要求：在上述四处补一句前提——「当每个 $M_h$ 行满秩（$d \ge d_h$ 且该头动量非退化）时各头行块范数为 $\sqrt{d_h}$；某头块自身秩亏时该头范数低于 $\sqrt{d_h}$，但不再受其他头尺度耦合」。｜修复：｜复验：

## 结论

- 处置：可发布（阻断与重要均为 0；两条轻微不影响正确性与主线理解，建议后续按修复要求补条件句）
- 统计：阻断 0 / 重要 0 / 轻微 2

> 本轮所列问题的处理结果见 `minor-fixes.md`。
