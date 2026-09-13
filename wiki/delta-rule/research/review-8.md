<!-- review-meta
round: 8
page: wiki/delta-rule/index.html
reviewed_content_sha256: 9b0361c100c31b87
-->
# Delta 规则与 DeltaNet 审查记录（第 8 轮）

- 页面版本：d97e815a2375f2959094974dd7ce42e25be64284
- 审查时间：2026-09-13
- 审查者：独立子代理（第 8 轮，未参与写作与前序审查）
- 已完整阅读章节：引言 / 核心问题 / 最容易误解 / 1.（1.1、1.2、本章问题）/ 2.（2.1、2.2、本章问题）/ 3.（3.1、3.2、本章问题）/ 4.（4.1–4.4、本章问题）/ 5.（5.1–5.5，含代码折叠块、本章问题）/ 全文总结 / 来源与范围说明
- 机械项：`python3 .dojo/scripts/validate.py wiki/delta-rule/index.html` → `validation ok`；重复 id 0；本地资源（katex/auto-render/prism/dojo-concept.css/dojo-overview.css）与前置页 `wiki/linear-attention/index.html` 均存在；`[Cx]/[Fx]/[Nx]` 引用与来源小节双向对应；全页无 Unicode 数学字符；SVG 图无 `<text>`，公式均由 `<foreignObject>` 承载。

## 来源核对（本轮已逐条回源，除下列问题外全部通过）

- arXiv:2406.06484 **v3** PDF（`arxiv.org/pdf/2406.06484v3`，pdftotext 抽取）核对：§2.2「eventually leading to key “collisions” when L > d, as pointed out by Schlag et al.」；§2.2「Here βt = σ(Wβ xt ) ∈ (0, 1) is a soft “writing strength”」；§2.2 紧凑式 `S_t = S_{t-1}(I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤`；§2.2 原文「it first retrieves the old value using the current key, vtold = St−1 kt」；§3.1「a generalized Householder transformation」；§3.3「erasing information in one subspace while preserving the other d − 1 subspaces」与特征值「1 with multiplicity d − 1 and 1 − βt‖kt‖2 with multiplicity 1」；§4.1「DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it struggles on the “Memorize” task.」
- Table 1（MAD，v3）逐格核对与页面 5.4 表一致：Transformer 94.1/29.8/86.8/99.6/85.2/Avg 74.5；Mamba 90.4/6.7/90.1/86.3/89.5/69.3；GLA 80.8/6.9/81.6/88.6/63.3/60.0；DeltaNet 100/35.7/100/100/52.8/71.8；Compress DeltaNet 42.2、Mamba 52.7。复算 Average：DeltaNet (42.2+35.7+100+52.8+100+100)/6 = 71.78 ≈ 71.8，Mamba 415.7/6 = 69.28 ≈ 69.3，与页面「6 个任务取均值」注一致。
- Table 2（v3，1.3B params / 100B tokens）核对：DeltaNet (w. conv) 16.87/12.21；Mamba (w. conv) 17.06/13.89；GLA (w. conv) 17.25/14.92；Transformer++ 16.85/13.44 —— 与页面 N1、5.4 正文一致。表号：v3 中 MAD 确为 **Table 1**、LM 为 **Table 2**，页面「§4.1 Table 1 / §4.2 Table 2」标注正确（ar5iv 最新版已把 MAD 改为 Figure 3、LM 改为 Table 1，页面声明按 v3 标注，故不算错）。
- Table 4 第一行确为「Linear Attention [47, 46]  St = St−1 + vt kt^T」，[47] = A. Katharopoulos et al. 2020；[11] = Bischof & Van Loan, WY representation, 1985 —— 与 C1、C6、5.5 节一致。
- arXiv:2102.11174 v3（Schlag 2021，pdftotext 抽取）核对：§1「an improved programming instruction akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960).」；§4.2「we propose a basic instruction that essentially implements the famous error-correcting delta rule (Widrow & Hoff, 1960)」；Eq. 23「…⊗φ(k(i)) − v̄(i)⊗φ(k(i))」标注 write / remove；§4.1「storing more than ddot associations will result in a retrieval error」与「the model might be in such an overcapacity regime」；文中确有命名「Delta Network / Delta Net」。ICML 2021（PMLR v139, pp. 9355–9366）由 Yang v3 参考文献 [97]/[98] 佐证。
- arXiv:2412.06464 v3 核对：§2.1 Mamba2 式 `S_t = α_t S_{t-1} + v_t k_t^⊤`；§3.1 Eq. 10 `S_t = S_{t-1}(α_t(I − β_t k_t k_t^⊤)) + β_t v_t k_t^⊤`；§3.1 称 β_t 为「(adaptive) learning rate」；摘要「gating enables rapid memory erasure while the delta rule facilitates targeted updates」；§1「since this process only modifies a single key-value pair at a time」「the model lacks the ability to rapidly clear outdated or irrelevant information」——支撑第 5 章各处表述。
- arXiv:2607.24653 与 `fla-org/flash-linear-attention` 存在且与 C9/C10 一致：K3 §2.1.1「Kimi Delta Attention」，「KDA extends the delta-rule recurrence」加「channel-wise forget gate」，是 hybrid 中的线性注意力层（3 KDA : 1 Gated MLA）；fla 仓库为 Triton 实现，含 DeltaNet / Gated DeltaNet。
- 代码折叠块实际执行（numpy）：输出与页面「预期输出」逐行一致（Linear S_2 = [[1,0],[1,0]]、Linear S_2@k_2 = [1,1]；Delta S_1 = [[1,0],[0,0]]、Delta S_2 = [[0,0],[1,0]]、Delta S_2@k_2 = [0,1]；Matches compact form: True）。
- 手算复核：1.2 节 S = [[1,1],[1,2]]、S k_2 = (2,3)、差值 (2,1)=v1+v3；2.2 节 S_1、S_2；3.1 节等价形式；4.3 节 β=0.5 得 (0.5,0.5)；第 2 章问题 S_t=[[0,2],[0,0]]；3.2 节 P 幂等/对称 —— 全部复算无误。

## 问题

- [重要·技术] 3.2 节末段（第 434 行）：断言「当 $\beta_t\|k_t\|^2 \neq 1$ 时，$I - \beta_t k_t k_t^\top$ 不再是正交投影……因此不幂等」，漏掉 $\beta_t = 0$ 这一页内明确承认的退化情形，与同页另两处结论直接冲突。｜引文依据：本页第 434 行「当 $\beta_t\|k_t\|^2 \neq 1$ 时……因此不幂等」；第 119 行「它是正交投影当且仅当 $\beta\|k\|^2 = 1$（投影到 $k^\perp$）或 $\beta = 0$（退化为单位矩阵 $I$）」；第 511 行「它是正交投影当且仅当 $\beta\|k\|^2 = 1$ 或 $\beta = 0$（此时退化为单位矩阵 $I$）」。核对：$\beta=0$ 时 $\beta\|k\|^2 = 0 \neq 1$，而 $I-0\cdot kk^\top = I$ 既对称又幂等，故第 434 行的「$\beta\|k\|^2\neq1 \Rightarrow$ 非投影、不幂等」在 $\beta=0$ 处不成立。｜修复要求：在第 434 行补上 $\beta_t \neq 0$（如「当 $\beta_t \neq 0$ 且 $\beta_t\|k_t\|^2 \neq 1$ 时……」），或加「除 $\beta_t = 0$ 的退化情形外」限定，使该处与第 119、511 行一致。｜修复：｜复验：

- [轻微·格式] 全页正文引号样式（第 71、118、123、135、143、214、218、334、392、511、586、719 行等）：弯引号 `“ ”` 与直引号 `"` 混用于同一种「引用术语/强调」功能，页内自相矛盾且偏离站内惯例。｜引文依据：第 214 行用「“只写不删”」（弯），紧邻的第 218 行用「"擦除旧 key 方向"」（直）；第 123 行 h2 用「"记不清"」（直），第 71 行同一短语用「“记不清”」（弯）。全站 98 个 index.html 中仅 8 页出现弯引号（delta-rule 26 个最多），gated-deltanet / linear-attention / kda 均为 0。｜修复要求：统一为全站惯例的直引号（或整页统一为弯引号），同一页面不混用。｜修复：｜复验：

- [轻微·表述] 口语化措辞：第 586 行「$\alpha_t$ 与 $\beta_t$ 各管一摊」、第 475 行「$k_t$ 越长擦得越狠，可能"擦过头"」、第 137 行 h3「1.1 retrieval 时到底发生了什么」。｜引文依据：本页原文（无对应外部来源，不适用来源核对）。｜修复要求：改为中性技术表述，例如「$\alpha_t$ 管全局衰减、$\beta_t$ 管方向擦除强度」「$\|k_t\|>1$ 时沿 $k_t$ 方向的擦除量随 $\|k_t\|^2$ 放大，可能反向过头」「1.1 retrieval 的代数过程」。｜修复：｜复验：

- [轻微·表述] 2.1 节末段（第 269 行）：「Schlag 2021 作为 delta rule 的提出者溯源」把 delta rule 的提出归给 Schlag 2021，与同页与来源均不符。｜引文依据：本页第 68 行「Schlag, Irie & Schmidhuber, ICML 2021……首次将 delta rule 引入线性 Transformer 并提出 Delta Network」，第 241–245 行「Delta 规则的命名与形式类比于经典 Widrow-Hoff 1960 delta rule」；Schlag 2021 §1 原文「an improved programming instruction akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)」。｜修复要求：改为「Schlag 2021 作为把 delta rule 引入线性 Transformer 的溯源」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复
