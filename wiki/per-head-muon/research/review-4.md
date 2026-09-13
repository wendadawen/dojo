<!-- review-meta
round: 4
page: wiki/per-head-muon/index.html
reviewed_content_sha256: 7da0952a0332e2ef
-->
# Per-Head Muon 审查记录（第 4 轮）

- 页面版本：b3ce3261f46e7e12b128c34df9a50abcc60a254c
- 审查时间：2026-09-13 19:46
- 审查者：独立子代理（未参与写作，未读本页 research/）
- 已完整阅读章节：核心问题 / 1. 整块正交化——为什么对多头不够好 / 本章问题 / 2. 按头正交化——切分动量矩阵并逐头正交化 / 本章问题 / 3. 效果与开销——均衡更新尺度与开销变化 / 本章问题 / 4. 分布式实现——P2P 参数取回 / 本章问题 / 来源与范围说明（含全部 `<details>` 折叠块、两幅 SVG 图与图注）
- 机械验证：`.dojo/scripts/validate.py wiki/per-head-muon/index.html` 返回 `validation ok`；`dojo:type=concept`、`dojo:topics=训练与优化`（在 ALLOWED_TOPICS 内）、`dojo:tag=优化器`（在 ALLOWED_TAGS 内）；`../muon-optimizer/`、`../newton-schulz/` 前置概念页均存在；页面无 `research/` 路径引用、无「待生成」占位；正文（含标题、summary、列表、表格）无 Unicode 数学字符，伪代码块内的 `∈ × ←` 属 style-guide §11「代码块按原样保留」豁免；全文无「我/我们/你」。

## 来源核对（K3 技术报告 arXiv:2607.24653v2 全文逐句比对）

- C1 §2.5：原文 "full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates"。页面引文逐字一致。
- C2 §2.5：原文 "instead of applying Newton–Schulz orthogonalization to the full Q, K, and V projection matrices, we partition their momentum matrices along the head dimension and orthogonalize each head's block separately" 与 "per-head orthogonalization equalizes the update scale across heads"。页面引文逐字一致。
- C3 §2.5：原文 "this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix"。页面引文逐字一致。
- C4 §5.2.2：原文 "The naive approach performs an all-gather over the entire parameter buffer on every rank [73], which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale"。页面引文逐字一致（页面两个问题的归纳与原句一致）。
- C5 §5.2.2：原文 "each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead"。页面引文逐字一致。
- §5.2.2 前置句 "The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton–Schulz orthogonalization in Muon requires the full parameter matrix" 支持页面「前置」段。
- 构造示例复算：M=[[3,4],[0.3,0.4]]，M M^T=[[25,2.5],[2.5,0.25]]，迹 25.25、行列式 0，σ1=√25.25≈5.025，u1≈[0.995,0.0995]，v1=[0.6,0.8]，Ortho(M)≈[[0.597,0.796],[0.0597,0.0796]]，两行范数≈0.995 与 0.0995。页面 §1 正文与折叠块全部数值一致。
- 按头正交化复算：Ortho([3,4])=[0.6,0.8]、Ortho([0.3,0.4])=[0.6,0.8]，范数均为 1；本章问题第 1 题 M1=[1,0],M2=[0,2] 行满秩两行范数均为 1、第 2 题 0.4/4 比例 0.1 均正确。
- 开销比值复算：行侧 Gram 下每头 O(d_h^2·d)、H 头 O(H·d_h^2·d)，全矩阵 O((H d_h)^2·d)，比值 1/H，与页面展开一致。

## 问题

- [重要·技术] §3「效果与开销——均衡更新尺度与开销变化」正文（"先说'均衡'均衡的是什么。"一段）：「整块动量秩亏（各头行块线性相关）时，各头之间仍保留原始动量尺度的比例（大尺度头行块范数大、小尺度头行块范数小）」被写成对一切秩亏情形的普遍结论，但该性质只在秩 1（各头平行同向）时成立。反例：H=3、d_h=1、d=2，M=[[1,0],[0,1],[1,1]]（三行线性相关，rank=2<H d_h=3），全矩阵正交化 UV^T 三行范数实测均为 0.8165，原始范数最大的第 3 头（√2≈1.414）并未得到更大的行块范数，比例与排序均不保留。｜引文依据：K3 §2.5 只写 "smaller-scale heads receive insufficiently normalized updates"（未给"比例保留"或排序机制）；本页 §1 自身的准确表述是"UV^T 中部分头块对应的行块范数小于 1"。｜修复要求：将 §3 该分句改为与 §1 一致的表述（秩亏只保证"部分头块行块范数小于 1、幅度不均"），或限定为"各头平行同向（秩 1）时比例保留"，删除"大尺度头行块范数大、小尺度头行块范数小"这一无条件排序断言。｜修复：｜复验：
- [轻微·技术] `<head>` meta description、blockquote.meta 与「论断与来源（C）」C4/C5：把 §5.2.2 的标题写作 "P2P-based Muon orthogonalization"。报告实际层级是 §5.2.2 "Memory-Efficient Training"，"P2P-based Muon orthogonalization" 是其下的无编号小节标题。｜引文依据：报告目录 "5.2.2 Memory-Efficient Training ... P2P-based Muon orthogonalization"；正文中该小节标题位于 §5.2.2 内、§5.2.3 之前。｜修复要求：把出处写成"§5.2.2「Memory-Efficient Training」下的 P2P-based Muon orthogonalization 小节"或等价准确写法。｜修复：｜复验：
- [轻微·技术] §2「代码：按头切分与逐头正交化的流程」折叠块：伪代码把装配后的更新记作 `U`（"正交化后的更新 U ← 零矩阵"、"更新由 U 替换原版 Muon 的 Ortho(M)"），而全文 `U` 指 SVD 左奇异向量矩阵（"X = U S V^T"、"$\mathrm{Ortho}(M_h)=U_h V_h^\top$"、§1 图注"$U$ 混合所有头"），同一符号两义。｜引文依据：不适用。｜修复要求：伪代码中的更新改用其他记号（如 `O` 或 `\Delta`），或注明它与 SVD 的 `U` 无关。｜修复：｜复验：
- [轻微·技术] §3「为什么开销会略降？」段 与 「展开：Gram 矩阵规模对比」：正文把结论附上条件"（当 $d_h \ll d$ 且 $H$ 适中时）"，但同章展开推导得比值恰为 $1/H$（与 $d$ 无关、全程未用该条件），展开末尾又写"当 $H$ 较大时"，正文条件与推导依赖不一致；比值实际依赖的是"行侧 Gram 成立（$H d_h \lesssim d$）"。｜引文依据：页面展开"两者比值（按头 / 全矩阵）$\approx H d_h^2 / (H d_h)^2 = 1/H$"。｜修复要求：删去或改写正文条件，使其与推导一致（或用 $H d_h \lesssim d$ 表达真实条件）。｜修复：｜复验：
- [轻微·技术] §3「本章问题」第 3 题解答："Per-Head Muon 修复的正是幅度不均，不改变各头方向由各自动量决定的机制"一句，把"各头方向只由自身动量决定"当成 per-head "不改变"的既有性质；而该性质恰是 per-head 相对全矩阵正交化带来的改变（§2 已写"全矩阵的 U 混合了所有头，按头正交化的每个 $U_h$ 只属于一个头"）。该句易被读成 per-head 不涉及方向。｜引文依据：不适用。｜修复要求：改为"per-head 使各头更新方向只由各自头块的动量决定"，或在句中点明这是 per-head 相对全矩阵正交化的改变之一。｜修复：｜复验：
- [轻微·表述] §3 段首"先说'均衡'均衡的是什么。"；§1"用一个极小例子把这件事算清楚。"：元话语式引导句（"先说…"属"下面来看…"一类），不含信息、可删。style-guide §8 要求衔接句说明前一结论与下一节问题的关系，不使用固定句式。其余全文（含折叠块与图注）通读未发现会话指代、调试/复现踩坑叙事、临场评价，也未把"场景"当术语。｜引文依据：不适用。｜修复要求：删去或改写为陈述性衔接句。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 来源一致性：C1–C5 五条英文引文与 K3 报告 §2.5、§5.2.2 逐字一致；构造示例与开销推导数值全部复算通过；无来源不支持的机制描述被写成结论（Moonlight ~2× 已明确不引用、K3 定性结论已标注为定性并有边界提醒）。
- 处置：修复（1 条重要问题关闭后可发布）
