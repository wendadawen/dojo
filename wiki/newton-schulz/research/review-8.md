<!-- review-meta
round: 8
page: wiki/newton-schulz/index.html
reviewed_content_sha256: 0c0cf07e5d550a59
-->
# Newton-Schulz 迭代审查记录（第 8 轮）

- 页面版本：index.html 工作树哈希 8dd7c2b0ee1c6764d8d67a082618a7621caa98d1（同目录 overview.html 一并通读）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序审查；本轮不读取 research/ 下任何文件）
- 已完整阅读章节：核心问题（5 题折叠块）→ 1. 不用 SVD 的正交化 + 本章问题 → 2. 迭代公式与机制（含「为什么奇多项式与 SVD 可交换」「diag(0.5,0) 前几步手算」两折叠块）+ 本章问题 → 3. 收敛条件与 Frobenius 预处理（含「非对角矩阵 G 完整计算」「代码」两折叠块）+ 本章问题 → 4. 零奇异值保持为 0 + 本章问题 → 5. 在 Muon 优化器中的应用 + 本章问题 → 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件）→ 全文总结
- 核对来源版本：S1 = docs.modula.systems/algorithms/newton-schulz（当前线上版）；S2 = arXiv:2506.10935**v2**（2026-02-24；v1 为 2025-06-12，本轮按 v2 的公式与章节编号核对）；S3 = nhigham.com/2020/12/15/what-is-the-matrix-sign-function；S4 = ricojia.github.io/2017/02/07/Newton-Schulz；S5 = kellerjordan.github.io/posts/muon

## 问题

- [轻微·可读性] §1「不用 SVD 的正交化」第 120 行（并见 §2 第 179 行、本章问题答案第 143 行）：术语「半正交」首次出现即使用，全文未给定义，且同句紧接着把同一对象 W 称为「最接近 $X$ 的正交矩阵」，两个词在同句并置易让读者无法分辨何时是正交、何时是半正交；而 §4 的核心区分（极限可能是「半正交（秩保持）矩阵，而非正交矩阵」）正依赖这一区别｜引文依据：不适用（页面内术语一致性）｜修复要求：在第 120 行首次出现「半正交」处补一句括注定义（如「$W^{\!\top}W=I$，列正交但可为长方/降秩矩阵」），使 §4 第 179 行「半正交……而非正交矩阵」可直接对照｜修复：｜复验：

（未发现阻断级或重要级问题。以下为本轮实际核对到位的项，供复验参考，不计为问题：）

- 代码可运行：将页面 `language-python` 代码块原样执行（`python3`，纯标准库），输出与页面「预期输出」逐行一致——`diag(0.5,0)` 在 k=1/2/3/4/5 为 0.6875 / 0.8687744140625 / 0.9752996308188813 / 0.9990923725928302 / 0.9999987646925808，终态 0.999999999997711；`G` 例 `||Q^T Q - I||_F` = 0.881917 / 0.737435 / 0.507949 / 0.226273 / 0.041296 / 0.001297 / 0.000001 / 0.000000，final Q = [[0.8944,0.4472],[-0.4472,0.8944]]。正文表格（第 388–393 行）与代码输出、章节问题答案（第 366、415 行）三处数字一致。
- 关键数字复算：$f(0.5)=\tfrac12\cdot0.5\cdot(3-0.25)=0.6875$（第 173、222 行）；$f(\sqrt3)=\tfrac12\sqrt3(3-3)=0$（第 231、352 行）；$f'(1)=1.5-1.5=0$、$f(1+e)\approx1-\tfrac32 e^2$（第 241、366 行）；$G$ 的 $\lVert G\rVert_F=\sqrt3\approx1.7321$（第 245、327 行）；$Q_0=G/\sqrt3$ 奇异值 (0.9342, 0.3568) 经 $Q_0^{\!\top}Q_0$ 特征值 $\tfrac{1\pm\sqrt{5/9}}{2}$ 开方复算一致；cursed quintic 系数和 $3.4445-4.7750+2.0315=0.701$（第 436、456 行）。
- 左/右乘等价：$\tfrac12Q(3I-Q^{\!\top}Q)=\tfrac32Q-\tfrac12QQ^{\!\top}Q$ 展开核对成立（第 161 行）；奇多项式可交换推导 $XX^{\!\top}X=U\Sigma^3V^{\!\top}$ 逐步核对成立（第 184 行）。
- 极因子旁证：`numpy.linalg.svd(G)` 的 $UV^{\!\top}=[[0.89442719,0.4472136],[-0.4472136,0.89442719]]$，与页面 §3 折叠块所述「与 SVD 给出的极因子一致」相符。
- 来源逐条核对（含原文片段）：C1 [S2] 摘要 "relies solely on matrix multiplications"；C2 [S2] v2 §1 "An important application of the polar decomposition is the orthogonal Procrustes problem"、"with the solution being Q=W the polar factor of X"、"This iteration converges to the orthogonal factor of the polar decomposition if σ1(X)<√3 and σn(X)>0"，页面在 §3 亦保留「且 $\sigma_{\min}>0$」条件，未扩大适用范围；C3 [S1] "snapping the singular values of M to one" 与 "the iterations we consider will actually fix zero singular values at zero"；C4 [S1] 收敛要求奇异值落在 $[-\sqrt3,\sqrt3]$；C5 [S1] "via a simple pre-processing step, mapping X ↦ X/‖X‖_F"；C6 [S3] "This iteration is quadratically convergent if ||I-A^2|| < 1 for some subordinate matrix norm"；C7 [S5] Muon 用途 + [S1] cursed quintic；F1 右乘形式见 [S4] "X_{k+1} = 1/2 X_k(3I - X_k^T X_k)"，[S2] v2 Eq.(1) "X_{k+1}=(3/2)X_k−(1/2)X_kX_k^T X_k"；F2 [S1] "commutes with the singular value decomposition"、"p(U Σ V^⊤) = U p(Σ) V^⊤"；F3 [S1] "f(x) = 3/2 x - 1/2 x^3"；F6 [S1] "f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5"，"These sum to 0.701, so the iteration oscillates and in fact does not converge"；N1 [S5] "3.28 val loss on FineWeb"、"by a factor of 1.35x"、"improved the training speed by 35%"，页面标注「训练速度，不是样本效率」与来源一致；[S5] 亦支持「Muon 只用于隐藏层二维参数，embedding/分类头等用 AdamW」（第 439 行）。
- 第 126 行「对称正交化 vs Gram-Schmidt 挑基准」经 [S1] 核对确为来源对比：S1 有 "This is sometimes referred to as 'symmetric orthogonalization'" 与 "This is in contrast to Gram-Schmidt orthogonalization, which involves first picking out a certain row or column vector"，未被写成无来源结论。
- 表述维度：全文（含折叠块与图注）无「本页/下面来看/需要注意的是/我、我们、你」等元话语或会话指代、无调试叙事与临场评价、无 Unicode 数学字符裸写；章节间过渡句为显式逻辑衔接（如第 128、200、344、400 行），非固定套话。
- 功能与格式：`python3 .dojo/scripts/validate.py wiki/newton-schulz/index.html` 返回 "validation ok"；前置概念链接 ../svd、../per-head-muon、../muon-optimizer 三页 index.html 均存在，无「（待生成）」占位；无 `<img>`/alt 含 `$...$`；`dojo:topics=训练与优化` 在词表内，`dojo:tag=优化器` 在 ALLOWED_TAGS 内；summary 公式可由 KaTeX 渲染；两级「核心问题/本章问题」均有折叠解答块，核心问题答案均指明完整论证所在章节。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（唯一轻微项为术语「半正交」首用未定义，建议随轮内修复一并补括注；不影响正确性与主线理解）
