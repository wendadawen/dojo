<!-- review-meta
round: 6
page: wiki/newton-schulz/index.html
reviewed_content_sha256: b8f4b96c9912570a
-->
# Newton-Schulz 迭代审查记录（第 6 轮）

- 页面版本：c05d89a4fbc99c9a74b49aa212caa9405f76b018（index.html 工作树哈希）
- 审查时间：2026-09-13 21:16
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：引言与核心问题 / 1. 不用 SVD 的正交化 / 2. 迭代公式与机制 / 3. 收敛条件与 Frobenius 预处理 / 4. 零奇异值保持为 0 / 5. 在 Muon 优化器中的应用 / 来源与范围说明（含全部折叠块、图注、overview.html）

## 核对基线（来源片段）

以下为逐条打开的来源原文，用于确认事实性论断成立（本轮无阻断/重要问题，故不逐条列入问题，仅记录核对依据）：

- [S1] docs.modula.systems/algorithms/newton-schulz："the iterations we consider will actually fix zero singular values at zero"；"p(U Σ V^T) = U p(Σ) V^T"（原文："The important property of an odd matrix polynomial of this form is that it _commutes_ with the singular value decomposition"）；"f(x) = 3/2 x - 1/2 x^3"；"We can achieve this via a simple pre-processing step, mapping X ↦ X / ‖X‖_F"；"the iterations we consider will actually fix zero singular values at zero"；cursed quintic "f(x) = 3.4445x - 4.7750x³ + 2.0315x⁵"，"oscillates and in fact does not converge"；"This operation is sometimes referred to as 'symmetric orthogonalization' because no row or column of the matrix M is treated as special … in contrast to Gram-Schmidt"。
- [S2] arXiv:2506.10935："relies solely on matrix multiplications and thus achieves high computational efficiency on GPU hardware"；"This iteration converges to the orthogonal factor of the polar decomposition if σ₁(X)<√3 and σₙ(X)>0."（Eq.1）；"An important application of the polar decomposition is the orthogonal Procrustes problem: min_{Q:QᵀQ=I} ‖Q−X‖_F, with the solution being Q=W the polar factor of X."；polar decomposition "X=WH … W=UVᵀ"；Eq.(1) "X_{k+1}=3/2 X_k−1/2 X_k X_kᵀX_k"。
- [S3] nhigham.com："This iteration is quadratically convergent if ||I-A²|| < 1 for some subordinate matrix norm."
- [S4] ricojia.github.io：右乘形式 "X_{k+1} = 1/2 X_k(3I - X_k^⊤ X_k)"；初始化 "X_0 = G/||G||_F"。
- [S5] kellerjordan.github.io/posts/muon：NanoGPT speedrun（FineWeb，3.28 val loss）"by a factor of 1.35x"；"we use an ad-hoc gradient based approach and end up with the coefficients (3.4445, -4.7750, 2.0315)"；"Muon is an optimizer for 2D parameters of neural network hidden layers."；"scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW."

复核过的数值与推导（均与页面一致）：
- 嵌入代码（纯 Python）实际运行，输出与页面「预期输出」逐行一致：diag(0.5,0) 六步 0.5 → 0.6875 → 0.8687744 → 0.9752996 → 0.9990924 → 0.9999988，终态 0.999999999997711/0；G=[[1,1],[0,1]] 的 ‖G‖_F=1.7321、正交性误差序列 0.881917/0.737435/0.507949/0.226273/0.041296/0.001297/0.000001/0.000000、终态 Q=[[0.89442719,0.44721360],[-0.44721360,0.89442719]] 全部吻合。
- numpy 旁证：G 奇异值 (1.61803, 0.61803)，极因子 U Vᵀ=[[0.89442719,0.4472136],[-0.4472136,0.89442719]]；Q₀=G/√3 奇异值 (0.934172, 0.356822)，与页面 (0.9342, 0.3568) 一致。
- f(0.5)=0.6875、f(0.6875)=0.868774、f(0.8688)=0.9753、f(1) 不动点、f'(1)=0、f(√3)=0 均可直接代入复算通过；3.4445-4.7750+2.0315=0.701≠1 正确。
- 术语单义：σ 全页指奇异值，W 始终指极分解正交因子，Q_k 指迭代估计；σ_max 与 [S2] 的 σ₁（最大奇异值）一致。
- 链接核查：../svd/index.html、../per-head-muon/index.html、../muon-optimizer/index.html 均真实存在；Per-Head Muon 的交叉引用表述与其页面 dojo:summary 一致。
- `.dojo/scripts/validate.py wiki/newton-schulz/index.html` 返回 "validation ok"；页面无 `$...$` 出现在 alt 中，无 Unicode 数学字符，无「（待生成）」占位。
- dojo:summary/description 与正文一致（无数字不一致、无相互矛盾）；§1 蓝 callout 半正交表述与 §4 秩保持说明自洽。

## 问题

- [轻微·表述] 第 1–5 章末过渡句（第 128/200/344/400/441 行）：五处过渡句采用同一固定句式「本章[动词]了 X。但…——下一章[动词]Y。」，且均以「本章给出了…／本章知道了…」这类以「本章」为主语的自我叙述开头，属元话语。style-guide §8 要求章节衔接「不使用固定句式」，check.md §2.2.12 将元话语列为不合格表述。｜引文依据：不适用｜修复要求：改写五处过渡句，去掉「本章[动词]了…」式自我叙述，避免重复同一句式，只保留对「前一节结论与下一节问题的关系」的具体一句话说明；改后须仍能从本章推出下一章的必要性。｜修复：｜复验：
- [轻微·技术] 第 5 章末段（第 439 行）：「Muon 只对二维矩阵参数用 NS；一维（标量、向量）参数与输入/输出层（embedding、分类头，通常是二维）改用标准方法（如 AdamW）。」——同一句内自相矛盾：既称 Muon 只对二维矩阵参数用 NS，又称「通常是二维」的 embedding、分类头改用 AdamW；本页第 5 章首段「对神经网络的二维参数（如隐藏层权重矩阵）」也隐含了「全部二维参数」的泛化。来源实际限定 Muon 只用于隐藏层的二维参数，embedding 与分类头虽为二维仍用 AdamW。｜引文依据：[S5] "Muon is an optimizer for 2D parameters of neural network hidden layers."；"scalar and vector parameters of the network, as well as the input and output layers, should be optimized by a standard method such as AdamW."｜修复要求：把「Muon 只对二维矩阵参数用 NS」限定为「Muon 只对隐藏层的二维矩阵参数用 NS」（并同步第 5 章首段），使全句与来源及自身括号内容一致。｜修复：｜复验：
- [轻微·格式] 来源与范围说明首个 h3（第 463 行）：「来源清单（S）」不在 style-guide 规定的来源章节固定 h3 命名集合内（论断与来源（C）／公式与来源（F）／外部数字与实验条件（N）／构造示例／辅助解释与类比边界／简化条件及其限制）；全库 72 个概念页以「论断与来源（C）」开头，仅本页出现「来源清单（S）」。｜引文依据：不适用｜修复要求：删除「来源清单（S）」小节，将其 [S1]–[S5] 书目信息并入开篇 blockquote.meta 的「主要依据」或「论断与来源（C）」，或改用 style-guide 列的固定命名。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（三条均为轻微，关闭后即可发布）

本轮页面实质内容干净：全部事实性论断、公式、数字与可运行代码均回源核对通过，未发现阻断或重要问题；三条轻微项均为表述与格式层面，不影响核心结论。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
