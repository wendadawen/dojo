# Newton-Schulz 迭代独立审查

- 审查者：独立上下文（AI 模拟）
- 页面版本：index.html 工作树哈希 e726f1e2e41c7f772779e8e9321ff57f24a60271 / overview.html 47eac67d3aec2b7d97d31fdd16be60b5a04117fc
- 时间：2026-08-09

## 问题

- [重要·技术] index.html 第 5 章 muon-application（第 929 行）：Muon 优化器链接标注"（该页待生成，目前为占位）"，但 wiki/muon-optimizer/index.html 实际已存在（59957 字节）且已在 content.json 中发布。错误标注会误导读者认为该页不可用而不点击查看。修法：删除"（该页待生成，目前为占位）"，改为正常链接文本，如"Muon 的完整讲解见 Muon 优化器。" ｜ 修复：已删除「（该页待生成，目前为占位）」，改为正常链接文本 ｜ 复验：
- [轻微·盲读] index.html 导言段（第 668 行）："奇异值"首次出现时未给出任何解释，定义要到第 2 章（第 676 行"对角元即奇异值"）才出现。小白读者在导言段遇到该词时无依据理解。修法：在导言段首次出现"奇异值"处加括注，如"奇异值（衡量矩阵在各方向上作用强度的非负数，见 SVD 概念页）"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 第 3 章 convergence-and-preprocessing（第 765 行）：使用"谱范数不超过 Frobenius 范数这个标准不等式"，但"谱范数"一词未说明其等同于 σ_max。修法：在"谱范数"后加"（即 $\sigma_{\max}$）"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 第 2 章 why-no-svd（第 676 行）：极分解定义中使用"半正定"一词未解释，小白读者可能不知道含义。修法：加括注"（特征值均非负的对称矩阵）"或直接改为"$H$ 为正定或半正定矩阵"并去掉"半正定"作为术语使用。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 第 2 章 iteration-and-mechanism（第 705 行）：声称右乘形式 $Q_{k+1}=\tfrac12 Q_k(3I-Q_k^{\!\top}Q_k)$ 与左乘形式 $Q_{k+1}=\tfrac32 Q_k-\tfrac12 Q_kQ_k^{\!\top}Q_k$ 等价，称"把右乘形式里的 $Q_k$ 提到前面就得到"，但未展示分配步骤。小白读者可能看不出如何转换。修法：加一行展开 $\tfrac12 Q_k(3I-Q_k^{\!\top}Q_k)=\tfrac32 Q_k-\tfrac12 Q_kQ_k^{\!\top}Q_k$。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 第 3 章 convergence-and-preprocessing（第 767 行）：称 $f'(1)=0$ 意味着"二次收敛——每步误差大约平方地下降"，但未解释为何导数为零导致二次收敛。修法：加一句解释，如"在不动点 $\sigma=1$ 处 Taylor 展开 $f(1+e)\approx f(1)+f'(1)e+\tfrac12 f''(1)e^2=\dots+\tfrac12 f''(1)e^2$，误差首项为 $e^2$"。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html Frobenius 范数 $\lVert A\rVert_F=\sqrt{\sum_{i,j}A_{ij}^2}$ 在第 2 章（第 674 行）和第 3 章（第 763 行）重复定义。修法：第 3 章改为"上文的 Frobenius 范数"或直接删除重复定义只保留公式 $Q_0=X/\lVert X\rVert_F$。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html 第 3 章（第 755 行）："所有奇异值都满足 $\sigma_{\max}(X_0)<\sqrt3$" 措辞冗余，$\sigma_{\max}$ 是单一数值而非每个奇异值分别满足的条件。修法：改为"最大奇异值满足 $\sigma_{\max}(X_0)<\sqrt3$"。 ｜ 修复： ｜ 复验：

## 段 A 盲读小结

按页面顺序通读 index.html 和 overview.html，主线理解无阻断级卡点。学习目标逐题核对：

1. NS 解决什么问题、为何不用 SVD → 第 1 章 why-no-svd 完整回答 ✅
2. 迭代式符号解释与收敛机制 → 第 2 章 iteration-and-mechanism 完整回答 ✅
3. 收敛条件 σ_max<√3 与 Frobenius 预处理 → 第 3 章 convergence-and-preprocessing 完整回答 ✅
4. 零奇异值保持 0 → 第 4 章 zero-singular-values 完整回答 ✅
5. NS 在 Muon 中的作用 → 第 5 章 muon-application 完整回答 ✅

正文主线不依赖任何折叠块即可成立（4 个 details 块均为补充推导/数字例子/代码，非主线必需）。

## 段 B 对照来源小结

### 来源核对

- [S1] docs.modula.systems/algorithms/newton-schulz：全文抓取成功。C1（只用矩阵乘法）、C3（零奇异值固定为零，原文"fix zero singular values at zero"）、C4（迭代趋于 sign）、C5（Frobenius 预处理 X/‖X‖_F）、C7（cursed quintic 系数 3.4445/−4.7750/2.0315、和不等于 1 故不收敛）逐条匹配。F1（奇多项式与 SVD 可交换 p(UΣV^⊤)=Up(Σ)V^⊤）、F3（三次 f(x)=3/2 x−1/2 x³）匹配。
- [S2] arXiv:2506.10935（Grishina et al.）：摘要确认 NS 仅依赖矩阵乘法、应用于 Muon 优化器。收敛条件 σ₁(X)<√3 及极因子 W=UV^⊤ 为标准结果。
- [S3] nhigham.com "What Is the Matrix Sign Function?"：确认"Newton–Schulz iteration is quadratically convergent"（C6 匹配）。
- [S4] ricojia.github.io：确认右乘形式 $X_{k+1}=\tfrac12 X_k(3I-X_k^{\top}X_k)$、Frobenius 预处理 $X_0=G/\|G\|_F$、极因子 $Q=UV^{\top}$（F1、F4、F5 匹配）。

### 公式复算

- $f(0.5)=\tfrac12\times0.5\times(3-0.25)=0.6875$ ✅
- $f(\sqrt3)=\tfrac12\sqrt3(3-3)=0$ ✅（代码验证：3.85e-16，数值零）
- $f'(1)=\tfrac32-\tfrac32\times1=0$ ✅
- cursed quintic 系数和 $3.4445-4.7750+2.0315=0.701\ne1$ ✅（代码验证：0.7010）

### 代码执行

纯 Python 代码提取执行，输出与页面"预期输出"完全一致：
- diag(0.5, 0)：sigma 序列 [0.5, 0.6875, 0.8688, 0.9753, 0.9991, 0.999999] 逐位匹配，零奇异值始终为 0 ✅
- G=[[1,1],[0,1]]：误差序列 [0.8819, 0.7374, 0.5079, 0.2263, 0.0413, 0.0013, 0.000001, 0.000000] 匹配，final Q = [[0.8944, 0.4472], [−0.4472, 0.8944]] 匹配 ✅
- 奇异值 σ≈(0.9342, 0.3568) 经手算验证：G^⊤G 特征值 (3±√5)/2，开方后除 √3 得 (0.9342, 0.3568) ✅

### 前置知识链接

- SVD 链接 ../svd/index.html：文件 642 字节（占位），未在 content.json 中发布。页面标注"该页待生成"正确 ✅
- Muon 链接 ../muon-optimizer/index.html：文件 59957 字节（完整页面），已在 content.json 中发布。页面标注"该页待生成，目前为占位"错误 → 见重要问题
- Per-Head Muon 链接 ../per-head-muon/index.html：文件 54798 字节（完整页面），已在 content.json 中发布。页面未标"待生成"，正确 ✅
- overview.html 与 index.html 互相链接 ✅
- validate.py 退出码 0 ✅

### 教学简化

三处简化（三次型为主、纯 Python 代码、SVD 最小衔接）均明确标注成立条件与不可推出的范围，无核心结论失真。教学示例 N1（diag(0.5,0)）和 N2（G=[[1,1],[0,1]]）均标注"教学示例，构造目的"。无未标记的教学构造混入来源结论。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 7
- 处置：进入修复（1 个重要问题为 Muon 链接占位标注错误，删除错误标注即可；7 个轻微问题均为可读性改进，不影响核心正确性）
