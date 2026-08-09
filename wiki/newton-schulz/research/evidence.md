# Newton-Schulz 迭代：核心论断与证据

来源标记：
- [S1] Bernstein & Newhouse, "Modular duality in deep learning" 配套页 docs.modula.systems/algorithms/newton-schulz（Jeremy Bernstein 为 Muon 提出者之一，主来源）
- [S2] Grishina, Smirnov, Rakhuba, "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials", arXiv:2506.10935
- [S3] N. Higham, "What Is the Matrix Sign Function?" / Functions of Matrices (nhigham.com, 书 SIAM 2008 §5)
- [S4] ricojia.github.io/2017/02/07/Newton-Schulz/（辅助数字例子）
- [S5] Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks", blog 2024-12（Muon 来源，经 [S1] 转引）

## C 论断（事实/结论）

### C1
- 论断：NS 迭代只用矩阵乘法、不用矩阵求逆或显式 SVD，即可近似正交化一个矩阵。
- 来源：[S1] 全文；[S2] §1 "the Newton–Schulz iteration ... only requires matrix multiplication"。
- 条件：输入经预处理使 $\sigma_{\max}<\sqrt3$。
- 置信：已确认。

### C2
- 论断：NS 迭代收敛到极分解 $X=WH$ 的正交因子 $W$，等价于 Procrustes 问题 $\min_{Q^{\!\top}Q=I}\lVert Q-X\rVert_F$ 的解 $Q=W\approx UV^{\!\top}$（$X=U\Sigma V^{\!\top}$ 为 SVD）。
- 来源：[S2] §1 "Polar decomposition ... $W=UV^{\!\top}$ ... the orthogonal Procrustes problem ... solution being $Q=W$"；"This iteration converges to the orthogonal factor of the polar decomposition"。
- 条件：$\sigma_1(X)<\sqrt3$ 且 $\sigma_n(X)>0$（满列秩）。
- 置信：已确认。

### C3
- 论断：NS 把矩阵的非零奇异值拉平为 1，零奇异值保持为 0（不把零奇异值变为 1）。
- 来源：[S1] "This map can be thought of as 'snapping the singular values of M to one'—**with the exception that the iterations we consider will actually fix zero singular values at zero**"。
- 机制依据：[S1] 奇多项式 $p(X)=aX+bXX^{\!\top}X+\cdots$ 满足 $p(U\Sigma V^{\!\top})=Up(\Sigma)V^{\!\top}$，对角元上 $f(0)=0$，故 $\sigma=0$ 是不动点。
- 条件：奇多项式族（含三次 NS）。
- 置信：已确认。

### C4
- 论断：三次 NS 迭代反复施加标量多项式 $f(x)=\tfrac32 x-\tfrac12 x^3$ 时，对 $|x|<\sqrt3$ 收敛到 $\mathrm{sign}(x)$；$x=0$ 不动；$|x|\ge\sqrt3$ 不收敛（$\sqrt3$ 一步到 0，更大则发散/震荡）。
- 来源：[S1] "iterate $f$ an infinite number of times ... obtain precisely the sign function on $[-3,3]$"（区间表述，实际收敛域为 $|x|<\sqrt3$）；[S2] "converges ... if $\sigma_1(X)<\sqrt3$"。
- 阈值依据：$f(\sqrt3)=\tfrac32\sqrt3-\tfrac12\cdot3\sqrt3=0$（计算核对）。
- 置信：已确认。

### C5
- 论断：Frobenius 预处理 $X\leftarrow X/\lVert X\rVert_F$ 保证归一化后 $\sigma_{\max}\le\lVert X\rVert_F/\lVert X\rVert_F\le1<\sqrt3$，落入收敛域。
- 来源：[S1] "ensure all singular values of the initial matrix lie in $[-3,3]$ ... via a simple pre-processing step, mapping $X\mapsto X/\lVert X\rVert_F$"；[S4] 同。
- 依据：$\sigma_{\max}(X)\le\lVert X\rVert_F$（谱范数 ≤ Frobenius 范数，标准不等式）。
- 置信：已确认。

### C6
- 论断：三次 NS 在不动点 $x=1$ 附近二次收敛（$f'(1)=\tfrac32-\tfrac32\cdot1^2=0$）。
- 来源：[S3] "The Newton–Schulz iteration is quadratically convergent if $\lVert I-A^2\rVert<1$"（sign 版本）；机制上 $f'(1)=0$ 给出二次收敛。
- 条件：靠近不动点。
- 置信：已确认。

### C7（应用）
- 论断：Muon 优化器对动量矩阵做 NS 正交化，使更新方向在各奇异方向上强度均衡（非零奇异值→1），提升大尺度训练稳定性；工程上用调过系数的高阶奇多项式（如 "cursed quintic" $f(x)=3.4445x-4.7750x^3+2.0315x^5$）以少步数近似 sign。
- 来源：[S1] Muon 章节；[S5] Muon 博客（经 [S1] 转引系数）；[S2] §1 列 Muon 为应用。
- 条件：Muon 的 cursed quintic 系数和 $3.4445-4.7750+2.0315=0.701\ne1$，故非收敛迭代，工程上以固定少步数（如 5 步）使用，牺牲收敛换速度。
- 置信：已确认（机制）；cursed quintic 的"不收敛却实用"依据 [S1] 明确说明。

## F 公式

### F1（迭代式，三次，右乘形式）
$$Q_{k+1}=\tfrac12 Q_k(3I-Q_k^{\!\top}Q_k).$$
- 来源：[S4] $X_{k+1}=\tfrac12 X_k(3I-X_k^{\!\top}X_k)$；与 [S2] Eq.(1) $X_{k+1}=\tfrac32 X_k-\tfrac12 X_kX_k^{\!\top}X_k$ 代数等价（提取左因子 $X_k$）。
- 验证：$\tfrac12 X(3I-X^{\!\top}X)=\tfrac32 X-\tfrac12 XX^{\!\top}X$。

### F2（奇多项式与 SVD 可交换）
$$p(U\Sigma V^{\!\top})=U\,p(\Sigma)\,V^{\!\top},\qquad p(\Sigma)=\mathrm{diag}(f(\sigma_1),\dots,f(\sigma_n)).$$
- 来源：[S1] "an odd matrix polynomial ... commutes with the singular value decomposition, in the sense that $p(U\Sigma V^{\!\top})=Up(\Sigma)V^{\!\top}$"。
- 推论：迭代退化为标量 $f$ 作用于奇异值。

### F3（标量三次多项式）
$$f(x)=\tfrac32 x-\tfrac12 x^3=\tfrac12 x(3-x^2).$$
- 来源：[S1] cubic iteration；[S2] Eq.(1) 对应标量。
- 不动点：$f(x)=x\Rightarrow x\in\{0,\pm1\}$；$f'(1)=0$（二次收敛）；$f(\sqrt3)=0$。

### F4（Frobenius 预处理）
$$Q_0=\frac{X}{\lVert X\rVert_F},\qquad \lVert X\rVert_F=\sqrt{\textstyle\sum_{ij}X_{ij}^2}.$$
- 来源：[S1][S4]。
- 保证：$\sigma_{\max}(Q_0)\le1<\sqrt3$。

### F5（极分解正交因子）
$$X=WH,\quad W=UV^{\!\top}\ (X=U\Sigma V^{\!\top}),\quad Q_k\xrightarrow{k\to\infty}W.$$
- 来源：[S2] §1。

### F6（cursed quintic，Muon 用）
$$f(x)=3.4445\,x-4.7750\,x^3+2.0315\,x^5,\qquad \sum\alpha_i=0.701\ne1\ \Rightarrow\ \text{非收敛迭代}.$$
- 来源：[S1] cursed quintic 章节；[S5] Muon 博客系数。
- 说明：仅作应用引用，不展开其震荡分析。

## N 数字（教学示例，均经代码验证，见 draft-check.md）

### N1（对角手算例子，演示零奇异值边界）
- 输入：$X_0=\mathrm{diag}(0.5,\,0)$（教学示例，构造目的：σ=0.5 与 σ=0 并列，直接观察多项式作用于奇异值）。
- 迭代（$f(\sigma)=\tfrac12\sigma(3-\sigma^2)$）：
  - $k=0$: $\sigma=(0.5,\ 0)$
  - $k=1$: $\sigma=(0.6875,\ 0)$ — $f(0.5)=0.25\times2.75=0.6875$
  - $k=2$: $\sigma=(0.868774,\ 0)$
  - $k=3$: $\sigma=(0.975300,\ 0)$
  - $k=4$: $\sigma=(0.999092,\ 0)$
  - $k=5$: $\sigma=(0.999999,\ 0)$
- 终态：$\mathrm{diag}(1,0)$。$\sigma=0.5\to1$，$\sigma=0\to0$（保持）。
- 验证方式：numpy 运行，输出一致。

### N2（非对角 2×2 例子，演示预处理与一般矩阵）
- 输入：$G=\begin{bmatrix}1&1\\0&1\end{bmatrix}$，$\lVert G\rVert_F=\sqrt3$。
- 预处理：$Q_0=G/\sqrt3=\begin{bmatrix}0.5774&0.5774\\0&0.5774\end{bmatrix}$，$\sigma(Q_0)=(0.9342,\ 0.3568)<\sqrt3$。
- 迭代 7 步后 $\lVert Q_k^{\!\top}Q_k-I\rVert<10^{-6}$，$\sigma\to(1,1)$。
- 终态：$Q\approx\begin{bmatrix}0.8944&0.4472\\-0.4472&0.8944\end{bmatrix}=UV^{\!\top}$（极因子，代码核对一致）。
- 验证方式：numpy，与 `np.linalg.svd` 的 $UV^{\!\top}$ 一致。

## 置信状态汇总

- C1–C7、F1–F6：已确认（均有可定位来源）。
- N1–N2：教学示例，代码已运行核对。
- 无"存在冲突"或"证据不足"项，可进入生产阶段。
