<!-- review-meta
round: 7
page: wiki/delta-rule/index.html
reviewed_content_sha256: a12d9742d96ef0be
-->
# Delta 规则与 DeltaNet 审查记录（第 7 轮）

- 页面版本：f2725211ec8a2d2beb6465a3956143a774953641
- 审查时间：2026-09-13 21:45
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：head 与 meta、核心问题（5 条及解答）、最容易误解、引言、1. 为什么线性注意力会"记不清"（含 1.1、1.2、本章问题）、2. Delta 规则的紧凑公式与手算（含 2.1、2.2、本章问题、代码折叠块）、3. 等价改写（含补充折叠块、3.1、3.2 与投影 SVG、本章问题）、4. 边界情况与 β_t 的退化（含 4.1–4.4、本章问题）、5. DeltaNet 与相邻模型对比（含 5.1–5.5、代码折叠块、本章问题）、全文总结、来源与范围说明（C1–C10、F1–F3、N1–N2、构造示例、辅助解释与类比边界、简化条件及其限制）、页脚脚本
- 核对方式：arXiv:2406.06484v3、arXiv:2412.06464v3、arXiv:2102.11174、arXiv:2607.24653 原文 PDF 全文提取后逐条定位；fla-org/flash-linear-attention 仓库页面核对；页面代码以 python3 实际执行比对输出；validate.py 通过（validation ok）。

## 问题

- [重要·技术] §3.2 末段（第 434 行）、§4.4 标题与表格（第 495–511 行）、§4.4 本章问题第 3 题解答（第 533 行）、最容易误解第 3 条（第 119 行）、核心问题第 4 条解答（第 101 行）、最容易误解第 2 条（第 118 行）、辅助解释与类比边界失效边界(2)（第 770 行），另见 overview.html 第 50 行：把投影性质成立的条件写成「仅当 $\beta = 1$ 且 $\|k\| = 1$」，并据此把「$\beta \in (0,1)$」整体列为非投影。该条件是充分而非必要：$P = I - \beta kk^\top$ 幂等当且仅当 $\beta(1 - \beta\|k\|^2) = 0$，即 $\beta = 0$（$P=I$，也是投影）或 $\beta\|k\|^2 = 1$。反例：$k = (1,1)^\top$、$\|k\|^2 = 2$、$\beta = 0.5$ 时 $I - 0.5kk^\top = \begin{pmatrix}0.5 & -0.5\\-0.5 & 0.5\end{pmatrix}$，$P^2 = P$ 且 $P^\top = P$，是到 $\mathrm{span}\{(1,-1)^\top\}$ 的正交投影；此时 $S_t k = (1-\beta\|k\|^2)S_{t-1}k + \beta\|k\|^2 v_t = v_t$，「完全擦除」同样成立。同一条件错误也出现在第 118 行（把完全擦除的充要条件写成 $\beta=1$ 且 $\|k\|=1$）与第 434 行（「$\beta_t < 1$ 或 $\|k_t\| \neq 1$ 时……不再是正交投影」）；第 504–506 行表格中「$\beta = 1$ 但 $\|k\| \neq 1$」一行结论正确，「$\beta \in (0,1)$」一行在未归一化 key 下不成立。此外第 511 行「它只在 $\beta = 1$ 且 $\|k\| = 1$ 时成立」与同页第 506 行「$\beta = 0 \to$ 单位矩阵 $I$」在自身页面上也不自洽（$I$ 是正交投影）。｜引文依据：Yang 2024 arXiv:2406.06484v3 §3.3 仅在归一化设定下给出充分方向——"Our key/query vectors are given by $k_t = \frac{\mathrm{SiLU}(W_K x_t)}{\|\mathrm{SiLU}(W_K x_t)\|_2}$… when $\beta_t = 1$, $I - k_t k_t^\top$ becomes a projection matrix, erasing information in one subspace while preserving the other $d-1$ subspaces"；原文未给出「仅当」的双必要条件。｜修复要求：把条件改为 $\beta_t\|k_t\|^2 = 1$（或统一限定在 $\|k_t\| = 1$ 的归一化设定下，把各处「仅当/需要两个前提同时满足」改为「当……时」，并同步修正 §4.4 表格「$\beta \in (0,1)$」行加注 $\|k\|=1$ 前提），同时修正第 118 行的「完全擦除」充要条件与第 434 行的「不再是正交投影」；overview.html 第 50 行同步。｜修复：｜复验：
- [轻微·格式] 第 189、309、436、513、694 行的 5 个 `<h3>本章问题</h3>` 均无 id：｜引文依据：不适用｜问题：目录脚本（第 824–850 行）对无 id 标题按 `textContent` 生成 id，5 个「本章问题」会生成同一个 `id="本章问题"`，产生重复 DOM id，且目录中 5 条「本章问题」子项全部指向第一章的那个锚点（滚动高亮也会 5 条同时点亮）。对照 wiki/linear-attention/index.html 的同类标题写有唯一 id（`chapter-1-questions` 等）。｜修复要求：给 5 个「本章问题」h3 各加唯一 id（如 `chapter-1-questions` … `chapter-5-questions`），与 linear-attention 等页保持一致。｜修复：｜复验：
- [轻微·技术] 第 578 行：｜引文依据：同段随后列出三条退化（$\alpha_t \to 1$、$\beta_t \to 0$、$\alpha_t \to 0$）｜问题：「在两个极端下退化为已知模型」但列了三条互不相同的退化情形（$\alpha_t \to 1$ 与 $\alpha_t \to 0$ 是同一参数的两端，$\beta_t \to 0$ 是另一参数的极端），数量与措辞不符。｜修复要求：改为「在以下极端取值下」或明确写「两个参数的三个极端」。｜修复：｜复验：
- [轻微·表述] 第 291、307 行：｜引文依据：不适用｜问题：口语化措辞「这模拟同一 key 改绑新值的更新」「查询 $k_2$ 时把 $v_1$ 也拽了出来」；「拽了出来」全站仅此页出现（`grep 拽` 仅命中本页），与同页其余正式表述不一致。｜修复要求：改用中性说法，如「把旧值 $v_1$ 一并带入检索结果」「覆写同一 key 的关联」。｜修复：｜复验：

## 核对通过项（记录备查，非问题）

- MAD benchmark 表（第 593–601 行）4 行数字与 arXiv:2406.06484v3 §4.1 Table 1 逐一相符：Transformer 94.1/29.8/86.8/99.6/85.2/74.5；Mamba 90.4/6.7/90.1/86.3/89.5/69.3；GLA 80.8/6.9/81.6/88.6/63.3/60.0；DeltaNet 100/35.7/100/100/52.8/71.8。第 604 行注「Average 含 Compress（DeltaNet 42.2 / Mamba 52.7）」经复算：DeltaNet (42.2+35.7+100+52.8+100+100)/6 = 71.78，Mamba (52.7+6.7+90.4+89.5+90.1+86.3)/6 = 69.28，与表内 71.8 / 69.3 一致，且确不能由可见 5 列算出。
- 第 608 行 1.3B / 100B tokens PPL 与 §4.2 Table 2 一致：DeltaNet 16.87/12.21、Mamba 17.06/13.89、GLA 17.25/14.92、Transformer++ 16.85/13.44；「LMB = LAMBADA」由 §4.2 正文 "including LAMBADA [LMB.; 74]" 佐证。
- 引文核对：§2.2 "a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key "collisions" when $L > d$"；§2.2 "it first retrieves the old value using the current key, $v_t^{\text{old}} = S_{t-1}k_t$"；§2.2 "soft "writing strength""；§3.1 "which can be seen as applying a generalized Householder transformation"；§3.3 "erasing information in one subspace while preserving the other $d-1$ subspaces"；§5.1 Table 4 首行 Linear Attention [47] $S_t = S_{t-1} + v_t k_t^\top$。Schlag 2021 §1 "akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)"、§4.1 "storing more than $d_{dot}$ associations will result in a retrieval error"、"overcapacity regime"、§4.2 Eq. 23 的 write/remove 两项形式、正文自称 "as a Delta Network" 均逐字对上；Widrow & Hoff 1960 书目 "In Proc. IRE WESCON Convention Record, pp. 96–104, 1960" 与第 245 行一致；Bischof & Van Loan 1985 WY 表示与 [11] 条目一致。
- Gated DeltaNet 侧：arXiv:2412.06464v3 §3.1 Eq. 10 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$、§2.1 Mamba2 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$、§2.2 "The delta update rule (Widrow et al., 1960; Schlag et al., 2021b)"、§3.1 "(adaptive) learning rate"、摘要 "gating enables rapid memory erasure while the delta rule facilitates targeted updates" 均逐条对上；第 108 行「DeltaNet 逐 key 串行、$\alpha_t$ 提供一步全局衰减」由 §1 "only modifies a single key-value pair at a time…lacks the ability to rapidly clear outdated or irrelevant information" 支持。
- Kimi K3：arXiv:2607.24653 §2.1.1 标题即 "Kimi Delta Attention"，正文 "KDA extends the delta-rule recurrence [106, 140] with a channel-wise forget gate [64]"，与 C9 的最小事实一致；fla-org/flash-linear-attention README 确含 DeltaNet / Gated DeltaNet 与 "A Triton-Based Library" 表述。
- 全部手算可复算：§1.2 的 $S=\begin{pmatrix}1&1\\1&2\end{pmatrix}$、$Sk_2=(2,3)^\top$、差额 $(2,1)^\top=v_1+v_3$；§2.2 与 §3.1、§4.3（$\beta_2=0.5$ 得 $S_2=\begin{pmatrix}0.5&0\\0.5&0\end{pmatrix}$、$S_2k_2=(0.5,0.5)^\top$）；各章本章问题答案的数字（如 $\begin{pmatrix}0&2\\0&0\end{pmatrix}$）均复算无误。
- 代码：第 619–666 行代码实际执行，输出与第 670–687 行「预期输出」逐行逐字符一致（含 `Matches compact form: True`）。
- 页内一致性：$S \in \mathbb{R}^{d_v \times d_k}$、$o_t = S_t q_t$、$\beta_t \in (0,1)$ 等符号全文单义；正文数字与 overview.html 无冲突；标题、summary、正文、列表、表格内无 Unicode 数学字符（仅第 405 行 HTML 注释与第 874 行脚本内出现 →/·，不渲染）；SVG 用 foreignObject 承载 KaTeX，无 `<text>` ASCII 近似；img/SVG 的 alt 与 aria-label 中无 `$...$`；线性注意力前置链接有效（wiki/linear-attention/index.html 存在）；overview.html 与 index.html 双向互链；validate.py 通过。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（1 条重要问题需按上式改为 $\beta_t\|k_t\|^2 = 1$ 或限定归一化前提；3 条轻微问题一并处理，index.html 与 overview.html 同步后重启一轮复验）