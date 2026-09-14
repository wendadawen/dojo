<!-- review-meta
round: 9
page: wiki/delta-rule/index.html
reviewed_content_sha256: 93b197a19f151ea0
-->
# Delta 规则与 DeltaNet 审查记录（第 9 轮）

- 页面版本：index.html 工作树哈希 113a7fb348ebc859b293296352f9c3511791f980（overview.html：10b75fcb45884eb84e8d433b558c299a9beeca9f）
- 审查时间：2026-09-14 14:35
- 审查者：独立子代理
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么线性注意力会"记不清"（含 1.1 retrieval 的代数过程、1.2 手算例子、本章问题）/ 2. Delta 规则的紧凑公式与手算（含 2.1 来源与命名、2.2 手算一步、本章问题）/ 3. 等价改写（含 3.1 回到手算例子、3.2 几何直觉、本章问题）/ 4. 边界情况与 β_t 的退化（含 4.1–4.4、本章问题）/ 5. DeltaNet 与相邻模型对比（含 5.1–5.5、代码折叠块、本章问题）/ 全文总结 / 来源与范围说明；并读取 overview.html 全文

## 问题

- [轻微·表述] overview.html「关键结论与边界」第 5 条：以「本页」为主语的自我指代｜引文依据：不适用｜修复要求：把「本页未展开」改写为不以「本页」作主语的表述（如「未展开」或改用具名主语），全套资料（index.html 与 overview.html）不得出现以「本页」为主语的自我指代；index.html 本身无此问题｜修复：｜复验：

## 来源核对记录（本轮实际打开来源逐条核对）

- **C1 / Table 4**（arXiv:2406.06484 v3）：§5.1 存在 Table 4（题注 "Overview of recent linear recurrent models…"，行首为 Linear Attention），Linear Attention 行引 [47]/[46]，文献表 [47] = Katharopoulos et al. 2020 "Transformers are RNNs"。§2.1 加性递归原文为 S_t = S_{t-1} + v_t k_t^T，o_t = S_t q_t。与页面 C1 一致。
- **C2**（§2.2）：原文 "…leading to key 'collisions' when L>d, as pointed out by Schlag et al."；Schlag 2021 §4.1 出现 "overcapacity regime"，正文另有 "storing more than d_dot associations will result in a retrieval error"。与页面 C2/F1 引文一致。
- **C3 / C4 等价性**（§2.2、§3.1）：原文给出 "v_t^old = S_{t-1}k_t" 与 "S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T"。Schlag 2021 §4.2 Eq. 23 为 remove+write 形式（W^(i) = W^(i-1) + v_new ⊗ φ(k) − v̄ ⊗ φ(k)）。与页面 C3/C4 一致；页面 C4↔C3 的代数合并（−1+(1−β)=−β）复算通过。
- **C5**：Schlag 2021 §1 原文 "…akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)"，§4.2 "essentially implements the famous error-correcting delta rule"；文献表 Widrow & Hoff, "Adaptive switching circuits.", IRE WESCON Convention Record, pp. 96–104。Gated DeltaNet §2.2 原文 "the delta update rule (Widrow et al., 1960; Schlag et al., 2021b)"。与页面 §2.1 与 C5 一致。
- **C6**（§3.1/§3.3）：原文 "applying a generalized Householder transformation"；"I − k_t k_t^T becomes a projection matrix, erasing information in one subspace while preserving the other d−1 subspaces"；"We instead apply L2 normalization…"。文献 [11] = Bischof & Van Loan, "The WY representation for products of householder matrices"（1985）。与页面 3.2/4.2/C6 一致。
- **C7**（arXiv:2412.06464 v3）：§2.1 原文 S_t = α_t S_{t-1} + v_t k_t^T（Mamba2）；§3.1 Eq. 10 原文 S_t = S_{t-1}(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T，紧随其后 "where β represents the (adaptive) learning rate"。与页面 5.2 表、5.3、C7 及符号表 β_t 注释一致（含 α_t 位置与括号层级）。
- **摘要引文**：Gated DeltaNet 摘要确有 "gating enables rapid memory erasure" 与 "the delta rule facilitates targeted updates" 两短语。与页面 5.3 引文一致。
- **C8 / N1 / N2**（§4.1 Table 1 "Synthetic Benchmarks"、§4.2 Table 2 "Language Modeling"）：
  - Table 1 逐格核对：Transformer 94.1/29.8/86.8/99.6/85.2/74.5；Mamba 90.4/6.7/90.1/86.3/89.5/69.3；GLA 80.8/6.9/81.6/88.6/63.3/60.0；DeltaNet 100/35.7/100/100/52.8/71.8；Compress（页面脚注）DeltaNet 42.2 / Mamba 52.7。与页面 5.4 表及脚注逐格一致；Average 按 6 任务均值复算通过（74.52/69.28/60.0/71.78）。
  - §4.1 原文确为 "DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it struggles on the 'Memorize' task"，且 §4.1 与 Table 1 题注确未给出模型规模/token 数——页面 N2 关于「未给出规模」的说明成立。
  - Table 2（1.3B/100B）：Transformer++ 16.85/13.44；Mamba (w. conv) 17.06/13.89；GLA (w. conv) 17.25/14.92；DeltaNet (w. conv) 16.87/12.21。与页面 N1、5.4 逐值一致。
- **C9**（arXiv:2607.24653 v2/v3）：§2.1.1 小节标题为 "Kimi Delta Attention"；正文 "KDA extends the delta-rule recurrence … with a channel-wise forget gate"。与页面 5.5、C9 一致。
- **C10**：github.com/fla-org/flash-linear-attention 存在，仓库含 DeltaNet / Gated DeltaNet 与 Triton 实现。与页面 C10 一致。
- **代码**（折叠块）：按页面逐字运行，stdout 与「预期输出」完全一致——Linear S_2 = [[1,0],[1,0]]、S_2@k_2=[1,1]；Delta S_1=[[1,0],[0,0]]、S_2=[[0,0],[1,0]]、S_2@k_2=[0,1]；等价形式 S_2=[[0,0],[1,0]]、Matches compact form: True。
- **图内数值**（3.2 的 SVG）：viewBox 0 0 560 340；(a,b) 点 (360,120)、投影后点 (80,120)、虚线落点 x=360 与 "a" 标签中心对齐（box x=345 w=30），箭头水平指向 y 轴，与图注 "k=(1,0)^T, β=1, P=I−kk^T … 结果落在 y 轴上的 (0,b)" 一致；`<text>` 内无 ASCII 近似公式，公式在 `<foreignObject>`；aria-label 无 `$`。
- **页面功能**：`python3 .dojo/scripts/validate.py wiki/delta-rule/index.html` → "validation ok"；dojo:topics=注意力机制 在 AGENTS.md 固定大类内；libs 下 katex/prism/dojo-concept.css 均存在；linear-attention 页与 overview.html 均存在，index↔overview 互链成立；引用编号 C1–C10、F1–F3、N1–N2 全定义且正文引用无悬空；核心问题 5 条与各章本章问题均有解答折叠块，答案均指明完整论证所在章节。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（唯一轻微项为 overview.html 的「本页」自我指代，不推翻任何结论，可按上述修复要求随手修正）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
