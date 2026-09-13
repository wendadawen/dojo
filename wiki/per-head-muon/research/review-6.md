<!-- review-meta
round: 6
page: wiki/per-head-muon/index.html
reviewed_content_sha256: 6413e18f5adf453c
-->
# Per-Head Muon 审查记录（第 6 轮）

- 页面版本：8964fb78b3362373940a1a58fc2c0f07f19eb693（wiki/per-head-muon/index.html）
- 审查时间：2026-09-13 21:18
- 审查者：独立子代理（第 6 轮独立审查者，未参与写作与既往审查）
- 已完整阅读章节：核心问题（5 条）；1. 整块正交化——为什么对多头不够好（含「展开」折叠块与本章问题）；2. 按头正交化——切分动量矩阵并逐头正交化（含「代码」折叠块与本章问题）；3. 效果与开销——均衡更新尺度与开销变化（含「展开」折叠块与本章问题）；4. 分布式实现——P2P 参数取回（含两幅 SVG 图注与本章问题）；来源与范围说明；overview.html
- 核对的外部来源：Kimi K3 技术报告原文 §2.5（Per-Head Muon，报告文本第 667–677 行）、§5.2.2 Memory-Efficient Training 下 P2P-based Muon orthogonalization 小节（第 1393–1400 行）；Keller Jordan "Muon" 博客（2024-12-08，kellerjordan.github.io/posts/muon/）；arXiv:2502.16982（Muon is Scalable for LLM Training）
- 机械核对：`.dojo/scripts/validate.py wiki/per-head-muon/index.html` 返回 `validation ok`；`dojo:type=concept`、`dojo:topics=训练与优化`（词表内）、`dojo:tag=优化器`；前置概念链 ../muon-optimizer/index.html、../newton-schulz/index.html 均存在；`overview.html` 与 `index.html` 双向链接；无「（待生成）」占位；alt/aria-label 中无 `$...$`；代码块为显式标注的伪代码，无需执行。
- 已核对并一致的关键数值/原文（摘要）：C1–C5 五段英文引文与报告逐字一致（含 §2.5 "full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates"；"we partition their momentum matrices along the head dimension and orthogonalize each head's block separately"；"this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix"；§5.2.2 "The naive approach performs an all-gather over the entire parameter buffer on every rank, which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale"；"each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication ... eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers"）。F2 与博客原文一致："equivalent to replacing the update by $UV^\top$, where $USV^\top$ is its singular value decomposition"。构造示例全部复算无误：σ1=√25.25≈5.025、MM^T=[[25,2.5],[2.5,0.25]]（迹 25.25、行列式 0）、u1≈[0.995,0.0995]、v1=[0.6,0.8]、Ortho(M)≈[[0.597,0.796],[0.0597,0.0796]]、两头行块范数 0.995 与 0.0995（比 10:1）；按头结果 [3,4]/5=[0.6,0.8]、[0.3,0.4]/0.5=[0.6,0.8]，范数均为 1；开销比 H·d_h²/(H·d_h)²=1/H 复算正确；Moonlight 的 ~2× 表述与其摘要一致且页面明确不予归因。

## 问题

- [重要·技术] 引言（第 67 行）与 1. 整块正交化（第 188 行）、3. 效果与开销（第 296 行）：引言把 C1 写成无条件论断——"原版 Muon 对整块投影矩阵做正交化，会让小尺度头得到的更新幅度不足"；而第 188、296 行把该效应收窄为条件成立——"被压低现象出现的条件是整块动量为秩亏矩阵……只要 M 行满秩（rank(M)=H d_h），正交化就把它的每一行都变成单位向量、各头行块范数均为 sqrt(d_h)"、"整块动量行满秩时正交化本身已让各头行块范数均为 sqrt(d_h)"。按页面自身这组判据，行满秩时全矩阵正交化已让各头幅度相等，秩亏才是压低出现的必要条件；但页面从未说明真实训练的动量矩阵是否满足秩亏（各头行块线性相关），反而在第 188 行给出"实际训练中各头方向通常不完全平行"。读者据此无法判断该动机性效应在实际训练中是否出现，甚至会得出"行满秩时该问题不存在"的结论，与 2–4 章把该问题当作既定前提展开（"如何修复这一幅度不均"）相抵触；同时该"秩亏条件"由本页自行推导，来源 C1 并不带此条件。｜引文依据：K3 §2.5 原文 "full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates"（无条件）；本页第 188 行"被压低现象出现的条件是整块动量为秩亏矩阵——各头行块线性相关，rank(M) < H d_h……反过来，只要 M 行满秩……各头行块范数均为 sqrt(d_h)"；本页第 296 行"整块动量行满秩时正交化本身已让各头行块范数均为 sqrt(d_h)"｜修复要求：在引言该句补上后文确立的条件限定（改为"当整块动量秩亏、各头行块线性相关时，会让小尺度头得到的更新幅度不足"），并在第 1 章明确标注"秩亏条件"是本页由 SVD/UV^T 推出的分析、来源 C1 只给无条件表述，同时补一句说明该条件与实际训练动量的关系（何种情形下各头动量行块线性相关），使读者能判断该效应在何种训练情形下出现。不得删除该条件（删除会使第 1 章手算示例的结论不成立）。｜修复：｜复验：
- [轻微·表述] 第 120 行："现在看多头注意力的投影权重结构。" 属指向阅读动作的元话语式引导句（与规范列举的"下面来看…"同类），不以内容本身起句。｜引文依据：不适用｜修复要求：改为直接陈述内容的句子，如"多头注意力的投影权重矩阵在实现上是各头块沿行方向堆叠"。｜修复：｜复验：
- [轻微·表述] 第 446 行："回到全文：……开篇的五个问题至此都有了答案。" 以页面自身（"全文""开篇的五个问题"）为叙述对象的元话语。｜引文依据：不适用｜修复要求：删去"回到全文："与"开篇的五个问题至此都有了答案"，保留对四章结论的复述句即可。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复