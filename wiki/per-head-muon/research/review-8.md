<!-- review-meta
round: 8
page: wiki/per-head-muon/index.html
reviewed_content_sha256: c127b554727b8a6c
-->
# Per-Head Muon 审查记录（第 8 轮）

- 页面版本：1c78fe195e9831f8c225dd28488faff5e004d892（index.html 工作树哈希；overview.html e94462480cbcc130809fd1a89b3e8f7351ef0f54）
- 审查时间：2026-09-14 17:09
- 审查者：独立子代理（未参与写作，未读取本页 research/ 任何文件）
- 已完整阅读章节：页面开头（主要依据 ＋ 引言）→ 核心问题 → 1. 整块正交化——为什么对多头不够好（含折叠块、本章问题）→ 2. 按头正交化——切分动量矩阵并逐头正交化（含折叠块、本章问题）→ 3. 效果与开销——均衡更新尺度与开销变化（含折叠块、本章问题）→ 4. 分布式实现——P2P 参数取回（含本章问题）→ 结语段 → 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 问题

- [轻微·格式] 第 1 章图（index.html:126-166）与第 4 章图（index.html:362-413）：两处 `<figure class="diagram">` 都没有图注——既无 `<figcaption class="diagram-caption">`，也无替代性的图注说明段。｜引文依据：不适用（站点结构约定，非来源论断）。对照：`.dojo/templates/concept/components.html:303` 要求「必须替换：节点标题与说明、箭头方向、图注」，`:329/:344` 给出 `<figcaption class="diagram-caption">【图注：…】</figcaption>` 占位；`guides/concept/content-examples.md` A5 正例每个 `<figure>` 均带 `figcaption`；全站概念页中仅 `wiki/nope/index.html`（用一段 `<p class="diagram-note">` 代替）与本页没有图注。｜修复要求：为两图各补一条图注——图 1 写明「箭头代表对整个动量矩阵 M 做一次 Newton–Schulz 正交化，得到 Ortho(M)」；图 2 写明「箭头代表各 rank 从 owner rank 取回自己负责参数的完整矩阵」（也可对照 nope 页改用一段 diagram-note）。节点与箭头含义已由图内 `dg-label` 给出，故本轮不判为技术缺陷，仅记为格式一致性。｜修复：｜复验：

## 来源核对依据（check.md §2.2.3）

- C1/C2/C3（arXiv:2607.24653v2 §2.5 "Per-Head Muon"）：原文逐字为 "For attention projections, we further refine it into a per-head variant [111, 146]: instead of applying Newton–Schulz orthogonalization to the full Q, K, and V projection matrices, we partition their momentum matrices along the head dimension and orthogonalize each head's block separately. The intuition is that full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates; per-head orthogonalization equalizes the update scale across heads. In practice, this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix." 页面 C1/C2/C3 三条引文为该段的准确截取（省略号处删除完整，无拼接）。✓
- C4/C5（同报告 §5.2.2「Memory-Efficient Training」下的 "P2P-based Muon orthogonalization" 小节）：原文 "The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton–Schulz orthogonalization in Muon requires the full parameter matrix, necessitating a communication step to gather complete parameters before each update. The naive approach performs an all-gather over the entire parameter buffer on every rank [73], which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale. Instead, each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead." 页面 C4/C5 引文与该段逐字一致；第 4 章"前置：分布式优化器（ZeRO 式）把参数分片到 DP rank…NS 需要完整矩阵"（index.html:358）也由该段首句直接支持。✓（小节归属正确：该小节确在 §5.2.2 之内，见报告目录 5.2.2 → P2P-based Muon orthogonalization。）
- F2（Muon 原始博客）：kellerjordan.github.io/posts/muon/ 原文 "$$\mathrm{Ortho}(G) = \arg\min_O \{ \|O - G\|_F : \text{either $O^\top O = I$ or $OO^\top = I$} \}$$" 与 "This is equivalent to replacing the update by $UV^\top$, where $USV^\top$ is its singular value decomposition (SVD)." 博客日期 "December 8, 2024"（页面记 2024-12 ✓）；报告参考文献 [53] 即 K. Jordan 等 "Muon: an optimizer for hidden layers in neural networks" (2024)。页面 F2 的表述与博客定义一致。✓
- N 段所引 Moonlight（arXiv:2502.16982）：原文为 "~2× computational efficiency compared to AdamW with compute optimal training"，页面称其为整体 Muon 的结果、不归因于 per-head，与我核到的原文一致。✓ 本页确未引入任何外部实验数字。

## 复算与机械核对

- 构造示例全部可复算且与页面一致：$MM^\top=\begin{bmatrix}25&2.5\\2.5&0.25\end{bmatrix}$，迹 25.25、行列式 0，$\sigma_1=\sqrt{25.25}\approx5.025$；$u_1\propto[10,1]\to[0.995,0.0995]$；$v_1=[0.6,0.8]$；$\mathrm{Ortho}(M)=u_1v_1^\top=\begin{bmatrix}0.597&0.796\\0.0597&0.0796\end{bmatrix}$，行范数 0.995 / 0.0995（比约 10:1）。按头正交化 $[3,4]/5=[0.6,0.8]$、$[0.3,0.4]/0.5=[0.6,0.8]$，范数 1。本章问题两例（$[1,0],[0,2]$ 与 $[1,0],[1,1]$）经手算确为行满秩、两头行块范数均 1。✓
- 开销量级：全矩阵 Gram $O((Hd_h)^2 d)$ 对按头 $O(H d_h^2 d)$，比值 $1/H$，与页面一致。✓
- 秩亏条件（index.html:188）是正确推论：$UV^\top$ 行范数平方和 = rank(M)，故 rank(M) < Hd_h 时至少一个头块行块范数小于 $\sqrt{d_h}$；反之 M 行满秩（rank = Hd_h）时 $UV^\top$ 行正交归一、每头块范数均为 $\sqrt{d_h}$。页面已注明该条件为本页由 SVD 结构的推导、来源 C1 不带此条件，未把推断写成来源结论。✓
- KaTeX：node 载入 libs/katex.min.js 对页面 210 处方括号/行内公式渲染，0 错误（仅 JS 模板串 `${...}` 被正则误捕获触发的 CJK 警告，非页面公式）。`dojo:summary` 为纯文本、无需数学渲染。✓
- `.dojo/scripts/validate.py wiki/per-head-muon/index.html` → `validation ok`（EXIT=0）。✓
- `alt` 属性无 `$...$`（唯一 alt="" 在 lightbox 占位图）；两图 `aria-label` 为纯文字；`<text>` 内无 ASCII 近似公式，图内公式均在 `<foreignObject>` 的 `dg-label` 中。✓
- 前置概念页 `wiki/muon-optimizer/index.html`、`wiki/newton-schulz/index.html` 均真实存在；overview.html 与 index.html 互相链接。✓
- 正文/核心问题/overview 三处数字与结论一致（小头 ≈0.1、大头 ≈1、比值 ≈10:1、比值 $1/H$）；引用编号 C1–C5、F1–F2 在正文与来源章节一一对应，无悬空编号。
- 表述维度：全文（含折叠块、图注、结语）无「本页将…/下面来看…/需要注意的是」类元话语，无我/我们/你等会话指代，无调试叙事与临场评价；「本页」自称仅为 style-guide §12 允许的自我指代（对照全站高频用法）。「拉平/拉偏/一句话说完/怎么切」等口语见于全站多数页面，属既定文风，本轮不另列。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：修复（仅 1 条轻微——两图补图注；本节其余各项已核对通过，页面主体结论、公式与来源均无差错，该轻微关闭后即可发布）

统计：阻断 0 / 重要 0 / 轻微 1