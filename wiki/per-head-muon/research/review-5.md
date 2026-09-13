<!-- review-meta
round: 5
page: wiki/per-head-muon/index.html
reviewed_content_sha256: 971ab8ad16e1c5b7
-->
# Per-Head Muon 审查记录（第 5 轮）

- 页面版本：595be885bfd1d6ae219eeac0d5c8a665b4aafbaf
- 审查时间：2026-09-13 20:22
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查）
- 已完整阅读章节（按顺序，含折叠块与图注）：核心问题（5 条含解答）；1. 整块正交化——为什么对多头不够好（含「展开：全矩阵正交化后两头行块的范数」、本章问题）；2. 按头正交化——切分动量矩阵并逐头正交化（含「代码：按头切分与逐头正交化的流程」、本章问题）；3. 效果与开销——均衡更新尺度与开销变化（含「展开：Gram 矩阵规模对比」、K3 三元结论表、本章问题）；4. 分布式实现——P2P 参数取回（含两张 SVG 图、本章问题）；来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）；overview.html。

## 来源核对留痕

- Kimi K3 技术报告 arXiv:2607.24653v2（https://arxiv.org/html/2607.24653v2，2026-09-13 抓取，1.52 MB HTML）§2.5 "Per-Head Muon" 原文逐句比对：
  - C1 原文："The intuition is that full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates"——与页面第 453 行引文逐字相符。
  - C2 原文："we partition their momentum matrices along the head dimension and orthogonalize each head's block separately ... per-head orthogonalization equalizes the update scale across heads"——相符（省略号处删去的内容不影响语义）。
  - C3 原文："This design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix"——相符。
  - §5.2.2「Memory-Efficient Training」下的 "P2P-based Muon orthogonalization" 小节（目录层级与正文均已确认）：C4 原文 "The naive approach performs an all-gather over the entire parameter buffer on every rank, which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale"；C5 原文 "each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead."——均逐字相符。该小节前一句 "The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton–Schulz orthogonalization in Muon requires the full parameter matrix" 支持第 358 行表述。
  - §2.5 全节未出现 per-head 单独贡献的消融数字或百分比，支持页面第 464 行「只给定性结论」。
- Keller Jordan "Muon" 博客（https://kellerjordan.github.io/posts/muon/）：正交化定义引文 "arg min_O { ||O - G||_F : either O^T O = I or OO^T = I }"，并 "UV^T, where USV^T is its singular value decomposition"；页面第 461 行 F2 表述与之相符。发布日期 byline 为 "December 8, 2024"，页面标注 "2024-12" 相符。
- Moonlight 技术报告 arXiv:2502.16982：摘要原文 "Scaling law experiments indicate that Muon achieves ∼2× computational efficiency compared to AdamW with compute optimal training."，支持第 464 行的定性归属说明。
- 机械项：`.dojo/scripts/validate.py wiki/per-head-muon/index.html` 返回 "validation ok"；`<head>` 的 description（纯文本）、dojo:summary、dojo:type=concept、dojo:topics=训练与优化（在 AGENTS.md 固定大类内）、dojo:tag=优化器（在 catalog_builder ALLOWED_TAGS 内）均有效；前置概念链接 ../muon-optimizer/index.html、../newton-schulz/index.html 与 overview.html 均真实存在；全文无 research/ 或其他不存在的仓库路径；无「（待生成）」占位；数学符号全部由 KaTeX 渲染（代码块内的 ∈、·、×、← 属 `<pre><code>` 原样保留，符合 style-guide 第 109 行的豁免）。
- 构造示例手算复算（均通过）：M=[[3,4],[0.3,0.4]]，MMᵀ=[[25,2.5],[2.5,0.25]]，迹 25.25、行列式 0，σ₁=√25.25≈5.025；u₁∝[10,1] 归一化 [0.9950,0.0995]；v₁=Mᵀu₁/σ₁=[0.6,0.8]；Ortho(M)≈[[0.597,0.796],[0.0597,0.0796]]，两行范数 0.995 与 0.0995，比值 10:1。按头 Ortho([3,4])=[0.6,0.8]、Ortho([0.3,0.4])=[0.6,0.8]，范数均为 1。本章问题二（[4,3]/[0.4,0.3]）同理得 0.1。Gram 开销比值 H·d_h²·d / (H·d_h)²·d = 1/H，与第 320 行一致。

## 问题

- [重要·技术] 第 2 章末「机制澄清」起、至 §3 与「简化条件」的多处（第 90、97、235、277、280、296、333、474 行）：把「每头块的行块范数」写成「都为 1 / 接近 1 / 单位范数」，但该数值只对 $d_h=1$（构造示例的设定）成立；一般情形下 $\mathrm{Ortho}(M_h)$ 是 $d_h\times d$ 的半正交矩阵，行向量各自单位范数，块 Frobenius 范数为 $\sqrt{d_h}$。｜引文依据：页面第 280 行「每个 $\mathrm{Ortho}(M_h)$ 是对 $M_h$ 单独做 SVD 取 $U_h V_h^\top$，把该头块内部的奇异值拉平为 1，行块范数因此为单位值」——「$d_h$ 个奇异值全为 1」的 $d_h\times d$ 矩阵其 Frobenius 范数为 $\sqrt{d_h}$，与同句结论「为单位值（1）」不符（本页自述典型 $d_h$ 为 64–256，第 474 行；$d_h=128$ 时为 11.31）；第 296 行「每个头块的更新行块范数都接近 1」「整块动量行满秩时正交化本身已让各头行块范数均为 1」、第 188 行与第 296 行「部分头块对应的行块范数小于 1」（一般应为小于 $\sqrt{d_h}$）、第 474 行「按头正交化保证（每头行块范数为 1）」同样受影响。K3 §2.5 原文只写 "per-head orthogonalization equalizes the update scale across heads"，未给出范数数值，故数值 1 无来源支持，系由 $d_h=1$ 的构造示例推广而来。另「行块范数」这一量全文未定义，字面义（行块的 Frobenius 范数）与上述数值不符。｜修复要求：在首次使用处定义该量并改为正确的一般值——「每个头块的每一行为单位向量，块 Frobenius 范数为 $\sqrt{d_h}$（构造示例 $d_h=1$ 时即 1）」；同步改正第 188、296 行的「小于 1」为「小于 $\sqrt{d_h}$」、第 474 行「每头行块范数为 1」为「每头行块范数为 $\sqrt{d_h}$」；overview.html「每个头各自获得单位幅度的更新」同处一并修正。｜修复：｜复验：
- [轻微·技术] 第 67 行（引言）与第 76 行（核心问题 1 解答）：把压低现象的触发条件写成「各头动量尺度差异 10 倍」，而 §1 第 188 行给出的实际条件是「整块动量为秩亏矩阵——各头行块线性相关」。按第 188 行自身的结论，整块动量行满秩时全矩阵正交化不压低任一头，因此尺度差异本身并不足以触发该现象（例如两头方向不同且线性无关时，尺度差 10 倍也不会被压到 0.1）。｜引文依据：不适用（页内两处条件的表述不一致）。｜修复要求：在引言与核心问题 1 解答处点明触发条件为「各头行块线性相关／整块动量秩亏」，或把「约 0.1」明确限定为所给构造示例（$M_1=[3,4]$ 与 $M_2=[0.3,0.4]$ 平行同向）的结果。｜修复：｜复验：
- [轻微·来源] 第 67 行：`<sup>[C1]</sup>` 标在含构造数值「被压到约 0.1」的句尾，易被读成该数值有 C1 支持。｜引文依据：第 453 行 C1 原文 "while smaller-scale heads receive insufficiently normalized updates" 不含任何数值；0.1 出自本页构造示例（第 467 行「构造示例」小节）。｜修复要求：把 `[C1]` 移到只覆盖来源支持表述的位置（如「让小尺度头更新幅度不足」之后），或在句中注明「约 0.1」为本页构造示例的复算结果。｜修复：｜复验：

## 未发现的问题（说明）

- 表述维度逐段通读（含折叠块与两张 SVG 图注）未发现元话语（"本页将…""下面来看…""需要注意的是"）、第二人称／第一人称复数指代、调试与复现踩坑叙事、临场评价或 AI 拼接腔；出现「本文／本页」为自称，符合 guides/concept/style-guide.md 第 119 行「自称使用"本页"或"本文"，不使用第一人称复数」。
- 第 255 行伪代码括注「后续缩放与参数更新与原版 Muon 一致」属实现细节，但与第 263 行「只改变正交化作用对象、NS 算法本身不变」的自述结论一致，未单列。
- 全部 C1–C5、F1–F2 引文均逐字核到来源；图表为内联 SVG，公式经 foreignObject 由 KaTeX 渲染；两级问题块命名、`解答：`／`展开：`／`代码：` 前缀、来源章节 h3 固定命名均符合 style-guide。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复。1 条重要问题（「行块范数＝1」由 $d_h=1$ 示例推广到一般 $d_h$）修复后需重算并回源复核；2 条轻微按上述要求修改。修改范围限于上列位置及 overview.html 对应句。
