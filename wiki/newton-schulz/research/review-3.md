<!-- review-meta
round: 3
page: wiki/newton-schulz/index.html
reviewed_content_sha256: 05177a3dd055b96f
-->
# Newton-Schulz 迭代审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 7fa6d4cb7ee3060bf0bc2350f0efe68148296b33（overview.html e0277e68ab7a4c07b6ff0998782046635645fcc4）
- 审查时间：2026-09-13 19:08
- 审查者：独立子代理（未参与写作与前序轮次审查）
- dojo:type：concept → 依据 guides/concept/check.md
- 已完整阅读章节：核心问题（learning-goals）→ 1. 不用 SVD 的正交化——为什么需要矩阵乘法方案 → 2. 迭代公式与机制——奇多项式如何把奇异值推向 sign → 3. 收敛条件与 Frobenius 预处理——如何保证落入收敛域 → 4. 零奇异值保持为 0——而不被拉平为 1 → 5. 在 Muon 优化器中的应用——正交化动量矩阵 → 来源与范围说明 → 全文总结（含全部 `<details>` 折叠块、代码块与表格）

## 核对来源（本轮实际打开）

- [S1] https://docs.modula.systems/algorithms/newton-schulz/（已抓取原文）
- [S2] arXiv:2506.10935 §1（Grishina, Smirnov, Rakhuba；已抓取 HTML 版 §1）
- [S3] N. Higham, "What Is the Matrix Sign Function?"（nhigham.com/2020/12/15/…；页面仅写 nhigham.com，已定位正确 URL 并抓取）
- [S4] https://ricojia.github.io/2017/02/07/Newton-Schulz/（已定位并抓取，右乘形式与 `X_0=G/‖G‖_F` 均属实）
- [S5] Keller Jordan et al., "Muon" blog https://kellerjordan.github.io/posts/muon/（已抓取）
- 一阶核对：`git hash-object`、`.dojo/scripts/validate.py`（返回 `validation ok`，EXIT=0）、numpy SVD、执行页面代码。

机械项结果：页面代码实际运行，输出与页面「预期输出」逐字符一致（含 `final diag = [0.999999999997711, 0.0]`、`final Q = [[0.8944271909999159, …]]`）；`diag(0.5,0)` 逐步表、`f(0.5)=0.6875`、`f(√3)=0`、`f'(1)=0`、cursed quintic 系数和 `0.701≠1` 均可复算；`G=[[1,1],[0,1]]` 终态 `Q` 与 numpy SVD 给出的极因子 `UV^T=[[0.8944,0.4472],[-0.4472,0.8944]]` 一致；前置页 `svd`、`per-head-muon`、`muon-optimizer` 均存在；overview.html 与 index.html 双向互链；页面无指向已移除 research/ 的路径。数学符号渲染：正文/折叠块中的 `→`（如「非零奇异值→1」）不在 `.dojo/scripts/validate.py` 的 `BARE_MATH_CHARS` 集合内，按该项目校验器注释「× – → 等在中文技术散文中作为普通排版字符使用，不列入」判定合格，不计为问题。

## 问题

- [阻断·技术] §5 在 Muon 优化器中的应用，第 1 段（index.html:439）：把「约 1.5 倍的样本效率」写成来源结论。〔问题〕原句「在实验中，Muon 相比 AdamW 展现出约 1.5 倍的样本效率<sup>[C7]</sup>」。所引三条来源均不支持该数字：[S5] Muon blog 的实测值是 **1.35x**（训练加速）而非 1.5x 样本效率；[S1] docs.modula.systems 全文无任何 Muon 与 AdamW 的倍数比较；[S2] §1 仅把 Muon 列为应用。「1.5」实为 Bernstein & Newhouse《Modular Duality in Deep Learning》（arXiv:2410.21265）中「1.5 billion parameter transformer」的参数量，被误植为倍数。此外 [C7] 条目（index.html:480）本身只覆盖「Muon 用途与 cursed quintic」，并不覆盖该数字。同页「外部数字与实验条件（N）」又写「本文无外部实验数字」，与本句自相矛盾。｜引文依据：Muon blog「Improved the speed record for training to 3.28 val loss on FineWeb by a factor of 1.35x」「Muon improved the training speed by 35%」；arxiv 2410.21265「scaled up to a 1.5 billion parameter transformer」；页面 index.html:491「本文无外部实验数字。」｜修复要求：删除「约 1.5 倍的样本效率」这一数字与断语，以及 [C7] 中与之对应的部分；若要保留 Muon 效果陈述，须改用来源原文数字（Muon blog 的 1.35x / 35% 训练加速），明确其为训练速度而非样本效率，并在 N 小节登记为外部数字、给出实验条件。｜修复：｜复验：

- [重要·技术] §5 在 Muon 优化器中的应用，第 3 段末（index.html:449）：把 embedding 归为「一维参数」，机制归因错误。〔问题〕原句「一维参数（如 embedding、head bias）仍用 AdamW，因为正交化对一维参数无意义（一维向量只有一个奇异值，正交化退化为标量归一化）」。来源中的 embedding / 分类头是二维输入、输出层，Muon 对其改用 AdamW 的原因是它们属输入/输出层，与其维度无关。原句把 embedding 列为「一维参数」并给出错误理由，会让读者误判 Muon 的参数适用范围与原因。｜引文依据：Muon blog「scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW」「it is also important to optimize input and output parameters using AdamW, even though these are typically 2D, particularly for embedding and classifier head layers」。｜修复要求：改写为——一维（标量/向量）参数，以及输入/输出层（embedding、分类头，通常是二维）仍用 AdamW；前者因正交化对一维无意义，后者因属输入/输出层。不得再把 embedding 称为「一维参数」或用「只有一个奇异值」解释 embedding。｜修复：｜复验：

- [轻微·技术] §3，折叠块「展开：非对角矩阵 G=[[1,1],[0,1]] 的完整计算」（index.html:255）：奇异值归属不明。〔问题〕原句「预处理 $Q_0=G/\sqrt3=\dots$，两个奇异值 $\sigma\approx(0.9342,\ 0.3568)$，都小于 $\sqrt3$」。给出的数值实为预处理后 $Q_0$ 的奇异值，$G$ 自身的奇异值为 $1.6180,\ 0.6180$；句子结构可被读成 $G$ 的奇异值，读者按其自算 $G$ 的 SVD 会得到不同数字。｜引文依据：numpy `linalg.svd`：σ(G)=(1.61803399, 0.61803399)；σ(G)/√3=(0.93417236, 0.35682209)。｜修复要求：明确写为「$Q_0$ 的两个奇异值 $\sigma\approx(0.9342,\ 0.3568)$」。｜修复：｜复验：

- [轻微·技术] 来源与范围说明 · 论断与来源（C），C6（index.html:479）：Higham 引文删去条件，成为无条件论断。〔问题〕原文引作「Newton–Schulz iteration is quadratically convergent」，删去了来源给出的收敛条件，读起来像无条件二次收敛。｜引文依据：nhigham.com「This iteration is quadratically convergent if ||I-A^2|| < 1 for some subordinate matrix norm.」（另：该文 NS 形式写作 $X_{k+1}=\tfrac12 X_k(3I-X_k^2)$）。｜修复要求：补回条件完整引用，或标注为转述并写明「在 … 条件下二次收敛」。｜修复：｜复验：

- [轻微·表述] 引言首段（index.html:81）：元话语。〔问题〕「本页先从一个具体场景出发。」「这一页就讲清楚它怎么做到、何时会失败、以及为什么零必须保持为零。」属于告知本页将做什么，而非直接陈述。/check.md 第 4 项要求排除「本页将…」式元话语。｜引文依据：不适用｜修复要求：改为直接给出问题与范围（例如直接从 $X_0=\mathrm{diag}(0.5,0)$ 与目标切入），删去「本页先…」「这一页就讲清楚…」的自我叙述句。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」之后（index.html:507–509）：未编号 h3「全文总结」。〔问题〕`<hr>` 后的 `<h3>全文总结</h3>` 无编号，不在 `guides/concept/style-guide.md` §1 允许的固定未编号标题（每章末「本章问题」、来源章节下六个固定命名 h3）之列；该 h3 在自动目录中会挂到「来源与范围说明」之下，层级语义错误。｜引文依据：style-guide.md §1「每章末尾固定的 `本章问题` 不编号……来源章节（`来源与范围说明`）下的 h3 使用固定命名、同样不编号」。｜修复要求：删除「全文总结」标题，或将其并入正文/改为符合规范的编号章节；不得以未编号 h3 形式挂在来源章节下。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复。阻断项（Muon 效果数字「1.5 倍样本效率」）必须删除或按来源数字改写并登记外部数字；重要项（embedding 维度归因）必须按 Muon blog 改写；四项轻微一并修复。修复后重跑 `.dojo/scripts/validate.py` 并进入下一轮独立审查。
