<!-- review-meta
round: 4
page: wiki/svd/index.html
reviewed_content_sha256: b106197211b32354
-->
# 奇异值分解（SVD）审查记录（第 4 轮）

- 页面版本：e71b0bf2238c446f7d1cd3afe4b4672e2fe03820
- 审查时间：2026-09-14
- 审查者：编排者派发的独立审查者（独立上下文，未参与写作与前三轮审查）
- 适用规范：guides/note.md（head 中 dojo:type=note）
- 已完整阅读章节：1. 分解形式与维度；2. 奇异值的性质；3. 最优低秩近似；4. 极分解与正交 Procrustes 问题；5. 来源与范围说明（论断与来源（C）／公式与来源（F）／来源清单／范围与简化）

## 来源核对（逐条对照外部来源，写下原文片段）

- [S4] §1 极分解定义：arXiv:2506.10935v2 原文 “Polar decomposition of a matrix X∈R^{m×n}, m≥n is a factorization X=WH”、“where W∈R^{m×n} has orthonormal columns”，支持 C4。
- [S4] §1 SVD 路径：原文 “X=USV^T, which immediately leads to W=UV^T, H=VSV^T”；页面括注「原文以 S 记对角奇异值矩阵、以上标 T 记转置」属实，支持 C4／F4。
- [S4] §1 正交 Procrustes：原文 “min_{Q: Q^T Q=I} ‖Q−X‖_F, with the solution being Q=W”，支持 C5。
- [S4] §1 Newton–Schulz 收敛条件：原文 “This iteration converges to the orthogonal factor of the polar decomposition if σ_1(X)<√3 and σ_n(X)>0”，与 C6 及正文第 4 节「在 σ_1(X)<√3 且 σ_n(X)>0 时收敛到 W」一致。
- [S2] 章节号核实：Golub & Van Loan《Matrix Computations》4th ed. §2.4 = “The Singular Value Decomposition”（理论），§8.6 = “Computing the SVD”（算法）；页面 [S2] §2.4／§8.6 两处引用准确。
- [S1] 第 4 讲 “The Singular Value Decomposition”（定理 4.1 为 SVD 存在定理）、第 5 讲 “More on the SVD”；[S3] 第 8 章 “The Polar Decomposition”；[S5] Schönemann 1966, Psychometrika 31(1):1–10；[S6] Eckart–Young 1936, Psychometrika 1(3):211–218——均与所引版本一致，无编号漂移。
- 公式复算：X=UΣV^T ⟹ X^T X=VΣ²V^T、XX^T=UΣ²U^T；W=UV^T 满足 W^T W=V(U^T U)V^T=I_n（列正交）；X=WH=UV^T·VΣV^T=UΣV^T；σ_i>0 时 Xv_i=σ_i u_i ⟹ u_i=Xv_i/σ_i；‖X−X_k‖_2=σ_{k+1}、‖X−X_k‖_F=√(Σ_{i>k}σ_i²)。全部成立，符号全页单义。
- head 元数据：description 为纯文本无 `$`；dojo:summary 由 KaTeX 可渲染；dojo:topics=数学基础、dojo:tag=数学与数值 均在 ALLOWED_TOPICS／ALLOWED_TAGS 内；`.dojo/scripts/validate.py wiki/svd/index.html` 返回 validation ok。
- 内部链接 ../newton-schulz/index.html、../../index.html 与本地 libs/（katex、auto-render、prism、dojo-note.css 等）均真实存在；无 ASCII 结构图、无 img、无 `$...$` 的 alt、无脚本不可读的交互视图。
- 与相邻页一致性：wiki/newton-schulz/index.html 亦记收敛条件为 σ_max<√3（σ_min>0），与本页 σ_1<√3、σ_n>0 表述相容。

## 问题

- [轻微·表述] §5 论断与来源（C）C4 条括注：以「本页」为主语作自我指代（“……本页统一记作 $\Sigma$ 与 $(\cdot)^{\!\top}$”）｜引文依据：不适用（表述类）｜修复要求：把「本页」改为不以页面为主语的表述（如「全文统一记作」「后续统一记作」），使该括注不出现页面自我指代｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（仅余 1 条轻微表述项待落盘修复；无阻断、无重要）