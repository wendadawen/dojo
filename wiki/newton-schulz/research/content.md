# Newton-Schulz 迭代

- 描述：Newton-Schulz 迭代是一种只用矩阵乘法、反复把矩阵推向其极分解正交因子的数值迭代：把非零奇异值拉平为 1，零奇异值保持为 0。
- 摘要：反复施加奇多项式 $p(X)=\frac32X-\frac12XX^\top X$，把非零奇异值推向 1、零奇异值留在 0，只用矩阵乘法就逼近极分解的正交因子 $UV^\top$。
- 主题：数学基础, 训练与优化
- 标签：数学与数值

## 资料

- Jeremy Bernstein、Laker Newhouse 等，Modula 项目文档「Newton-Schulz」，2024，https://docs.modula.systems/algorithms/newton-schulz/
- E. Grishina、M. Smirnov、M. Rakhuba，Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials，arXiv:2506.10935v2，2026，https://arxiv.org/abs/2506.10935
- Keller Jordan 等，Muon: An optimizer for hidden layers in neural networks，2024-12-08，https://kellerjordan.github.io/posts/muon/
- N. Higham，What Is the Matrix Sign Function?，2020-12-15，https://nhigham.com/2020/12/15/what-is-the-matrix-sign-function/

## 短页

### 只用矩阵乘法逼近正交化

给定一个矩阵 $X$，常常要找和它最接近的矩阵 $Q$，其中 $Q$ 的列两两正交、每列长度为 $1$：这样的矩阵叫半正交矩阵，方阵时就是正交矩阵。这个 $Q$ 是 $X$ 的极分解 $X=WH$ 中的正交因子 $W$（极分解把 $X$ 写成一个半正交矩阵 $W$ 与一个对称半正定矩阵 $H$ 的乘积），它等于 $UV^\top$，其中 $X=U\Sigma V^\top$ 是 $X$ 的[SVD](wiki/svd)，$\Sigma$ 的对角元 $\sigma_i$ 是 $X$ 的奇异值。Newton-Schulz 迭代不直接算 $U$、$\Sigma$、$V$，而是反复施加同一个多项式，把 $X$ 的非零奇异值推向 $1$，使各方向强度拉平。零奇异值保持为 $0$，不会被改变。
依据：blog/modula-newton-schulz.md Problem statement；paper/src-v2/main.tex Introduction

### 为什么不去算 SVD

要求 $W=UV^\top$，直接算 SVD 是最直白的办法，但 SVD 在大矩阵上代价高。Newton-Schulz 迭代的每一步只做几次矩阵乘法，矩阵乘法在 GPU 上有高度优化的实现，因此迭代在大矩阵上更快。它给出的是近似，精度由迭代步数控制。
依据：blog/modula-newton-schulz.md Problem statement；blog/muon-blog.txt L50-L53；paper/src-v2/main.tex L57

### 先归一化再反复代入

先把 $X$ 按 Frobenius 范数归一化，$\lVert X\rVert_F=\sqrt{\sum_{ij}X_{ij}^2}$ 是矩阵全部元素平方和的平方根。再把归一化后的矩阵记作 $Q_0$，代进迭代式

$$Q_{k+1}=\tfrac12 Q_k(3I-Q_k^\top Q_k),$$

把得到的 $Q_{k+1}$ 再代进去，如此重复。$I$ 是与 $Q_k^\top Q_k$ 同阶的单位矩阵。归一化是为了让 $X$ 的奇异值落入收敛区间 $(-\sqrt3,\sqrt3)$。
依据：blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Introduction

## 长页

Newton-Schulz 迭代是一种数值迭代，用来逼近一个矩阵 $X$ 的极分解中的正交因子，也就是与 $X$ 最接近的半正交矩阵。它的做法是反复施加同一个矩阵多项式 $p(X)=\tfrac32X-\tfrac12XX^\top X$，每一步只做几次矩阵乘法（转置、相乘、相加），不必算出 SVD[SVD](wiki/svd)（奇异值分解，把 $X$ 写成 $U\Sigma V^\top$，$\Sigma$ 的对角元是 $X$ 的奇异值）。反复迭代把 $X$ 的非零奇异值逐步推向 $1$，把它们拉平；零奇异值则留在 $0$。迭代前先对 $X$ 归一化，使奇异值落入收敛区间。下面先讲它求的是哪个矩阵，再给出迭代式，然后说明为什么反复迭代就能正交化、在什么条件下收敛，最后手算一个小例子。
依据：blog/modula-newton-schulz.md Problem statement；blog/modula-newton-schulz.md Odd polynomial iterations；blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Introduction

### 从一个矩阵求最接近的正交矩阵

#### 想求的是哪个矩阵

「与 $X$ 最接近的正交矩阵」是指下面这个最小化问题的解：

$$\min_{Q^\top Q=I}\;\lVert Q-X\rVert_F .$$

这里 $X$ 取 $m\times n$、$m\ge n$ 的矩阵，$Q$ 取遍所有满足 $Q^\top Q=I$ 的矩阵，$I$ 是单位矩阵；$\lVert\cdot\rVert_F$ 是 Frobenius 范数，$\lVert A\rVert_F=\sqrt{\sum_{ij}A_{ij}^2}$，即把所有元素平方后求和再开方。$Q^\top Q=I$ 表示 $Q$ 的列两两正交、每列长度为 $1$；$Q$ 是方阵时这样的 $Q$ 就是正交矩阵，$Q$ 是长方形时叫半正交矩阵。这个问题叫正交 Procrustes 问题。它的解与 $X$ 的极分解直接相关：极分解把 $X$ 写成 $X=WH$，其中 $W$ 的列两两正交、每列长度为 $1$（半正交），$H$ 是对称半正定的，Procrustes 问题的解正是这个 $W$。若把 $X$ 写成 $X=U\Sigma V^\top$，则 $W=UV^\top$。Newton-Schulz 迭代逼近的，就是这个 $W=UV^\top$。
依据：blog/modula-newton-schulz.md Problem statement；paper/src-v2/main.tex Introduction

#### 为什么不去算 SVD

要得到 $W=UV^\top$，把 SVD 算出来再取 $U$、$V$ 是最直接的办法。但 SVD 要算出 $U$、$\Sigma$、$V$ 三个矩阵，在大矩阵上代价高。Newton-Schulz 迭代的每一步只做几次矩阵乘法（转置、相乘、相加），不显式做 SVD；矩阵乘法在 GPU 上有高度优化的实现，可并行，因此迭代在大矩阵上更快。
依据：blog/modula-newton-schulz.md Problem statement；blog/muon-blog.txt L50-L53；paper/src-v2/main.tex L57

### 迭代式与符号

最基本的 Newton-Schulz 迭代是

$$X_{k+1}=\tfrac32 X_k-\tfrac12 X_kX_k^\top X_k .$$

这里 $X_k$ 是第 $k$ 步得到的矩阵，$X_k^\top$ 是它的转置，$X_kX_k^\top X_k$ 是三次乘积；从一个初始矩阵出发，把结果反复代回右边。同一个多项式也能按另一种方式分组，写成

$$Q_{k+1}=\tfrac12 Q_k(3I-Q_k^\top Q_k),$$

其中 $Q_k$ 就是归一化后的 $X_k$，$Q_0$ 是归一化后的初始矩阵，$I$ 是单位矩阵。两式只是同一个多项式的不同括号分组，对同一个输入给出相同结果。
依据：blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Introduction

::: 两种写法为什么等价
1. 右乘形式 $\tfrac12 Q(3I-Q^\top Q)$ 先算括号里的 $3I-Q^\top Q$。
依据：自行构造
2. 把 $\tfrac12$ 乘进去：$\tfrac12 Q(3I-Q^\top Q)=\tfrac32 Q-\tfrac12 QQ^\top Q$。
依据：自行构造
3. 这与左乘形式 $\tfrac32 Q-\tfrac12 QQ^\top Q$ 逐项相同，故两式等价。
依据：自行构造
:::

$Q_k^\top Q_k$ 是衡量 $Q_k$ 离正交有多远的量：若 $Q_k$ 已经是半正交矩阵，则 $Q_k^\top Q_k=I$，括号恰好 $3I-Q_k^\top Q_k=2I$，于是 $Q_{k+1}=\tfrac12 Q_k\cdot2I=Q_k$，迭代不再改动它。半正交矩阵因此是不动点。
依据：blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Introduction

### 为什么只看奇异值就够了

#### 奇多项式与 SVD 可交换

把迭代式里出现的形式 $p(X)=aX+bXX^\top X+c(XX^\top)^2X+\cdots$ 叫作奇矩阵多项式：每一项都是 $X$ 乘以若干个 $XX^\top$ 再乘一个常数，幂次都是奇数。这样的多项式与 SVD 可交换，意思是

$$p(U\Sigma V^\top)=U\,p(\Sigma)\,V^\top .$$

$\Sigma$ 是对角矩阵，所以对矩阵 $X$ 施加 $p$，等于对每个奇异值单独施加同一个标量多项式 $f(x)=ax+bx^3+cx^5+\cdots$，再把结果放回对角线上。于是「对矩阵反复施加 $p$」就退化成「对每个奇异值反复施加标量 $f$」。
依据：blog/modula-newton-schulz.md Odd polynomial iterations；paper/src-v2/main.tex Introduction

#### 反复迭代趋于符号函数

对三次迭代，标量多项式是

$$f(x)=\tfrac32 x-\tfrac12 x^3 .$$

在收敛区间 $(-\sqrt3,\sqrt3)$ 内反复施加 $f$，序列 $x,\;f(x),\;f(f(x)),\dots$ 趋于符号函数 $\operatorname{sign}(x)$（$x>0$ 为 $1$、$x<0$ 为 $-1$、$x=0$ 为 $0$）。因此迭代把非零奇异值推向 $1$、把 $X$ 正交化；零奇异值因为 $f(0)=0$ 而保持为 $0$，不被推向 $1$。
依据：blog/modula-newton-schulz.md A cubic iteration；blog/modula-newton-schulz.md Odd polynomial iterations

::: $f$ 的不动点与迭代行为
1. 解 $f(x)=x$：$\tfrac32 x-\tfrac12 x^3=x$，即 $-\tfrac12 x^3+\tfrac12 x=0$，得 $x(x^2-1)=0$，所以不动点为 $x=0,1,-1$。
依据：自行构造
2. $x=1$ 处 $f(1)=\tfrac32-\tfrac12=1$；$x=0$ 处 $f(0)=0$；$x=-1$ 处 $f(-1)=\tfrac32(-1)-\tfrac12(-1)^3=-\tfrac32+\tfrac12=-1$。三个点都是不动点。
依据：自行构造
3. $x=0.5$：$f(0.5)=\tfrac32\cdot0.5-\tfrac12\cdot0.125=0.75-0.0625=0.6875$，比 $0.5$ 更接近 $1$；继续迭代会越来越接近 $1$。
依据：自行构造
4. 因此非零初始值被拉向 $1$ 或 $-1$，零初始值留在 $0$，序列趋于符号函数。
依据：自行构造
:::

### 收敛条件与预处理

#### 为什么阈值是 $\sqrt3$

迭代收敛要求初始矩阵的奇异值满足

$$\sigma_1(X)<\sqrt3,\qquad \sigma_n(X)>0 .$$

$\sigma_1(X)$ 是最大奇异值，$\sigma_n(X)$ 是最小奇异值；$\sigma_n(X)>0$ 等价于 $X$ 满列秩。$\sqrt3$ 是上界：三次多项式的收敛区间是 $(-\sqrt3,\sqrt3)$，端点 $-\sqrt3$ 和 $\sqrt3$ 一步就被映到 $0$，不在收敛区间内；奇异值落在区间外时不再保证被推向 $1$。
依据：paper/src-v2/main.tex Introduction

::: 上界 $\sqrt3$ 的来历
1. 三次多项式 $f(x)=\tfrac12 x(3-x^2)$ 在 $x=\sqrt3$ 处的因子 $3-x^2$ 为 $3-3=0$，故 $f(\sqrt3)=0$：$\sqrt3$ 一步就被映到 $0$。
依据：自行构造
2. $x>\sqrt3$ 时 $3-x^2<0$，$f(x)$ 变号，迭代被带向负值，不再收敛到 $1$。
依据：自行构造
3. 所以落在 $(-\sqrt3,\sqrt3)$ 之外的奇异值不再保证收敛，迭代前必须先把奇异值移进这个区间。
依据：自行构造
:::

#### 用 Frobenius 范数归一化

迭代的第一步是归一化，把奇异值移进收敛区间。最直接的办法是用 Frobenius 范数：

$$X\leftarrow \frac{X}{\lVert X\rVert_F},\qquad \lVert X\rVert_F=\sqrt{\textstyle\sum_{ij}X_{ij}^2}.$$

$\lVert X\rVert_F$ 的定义见上。归一化后最大奇异值不大于 $1$，落在 $(-\sqrt3,\sqrt3)$ 内。
依据：blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Normalization of a matrix prior to Newton-Schulz iterations

::: 归一化后为什么落在收敛区间内
1. 对任意矩阵，最大奇异值不超过 Frobenius 范数：$\sigma_1(X)\le\lVert X\rVert_F$。
依据：自行构造
2. 归一化后得到 $X/\lVert X\rVert_F$，它的最大奇异值为 $\sigma_1(X)/\lVert X\rVert_F\le1$。
依据：自行构造
3. $1<\sqrt3$，故归一化后的奇异值都在 $(-\sqrt3,\sqrt3)$ 内，落入收敛区间。
依据：自行构造
:::

归一化的代价是：用 Frobenius 范数可能把较小的奇异值压得更小，使它们需要更多步才被推向 $1$。理想的做法是按最大奇异值归一化。
依据：blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Normalization of a matrix prior to Newton-Schulz iterations

### 手算一个 2×2 例子

取一个满列秩的小整数矩阵 $X=\begin{bmatrix}3&1\\1&3\end{bmatrix}$，迭代几步，观察它如何趋近正交矩阵。
依据：自行构造

::: 例子的每一步
1. 输入矩阵 $X=\begin{bmatrix}3&1\\1&3\end{bmatrix}$，它是方阵，两个奇异值是 $4$ 和 $2$，都大于 $0$，满足满列秩。
依据：自行构造
2. 归一化：$\lVert X\rVert_F=\sqrt{3^2+1^2+1^2+3^2}=\sqrt{20}\approx4.4721$，得 $Q_0=\begin{bmatrix}0.6708&0.2236\\0.2236&0.6708\end{bmatrix}$（保留四位小数）。
依据：自行构造
3. 此时 $Q_0^\top Q_0=\begin{bmatrix}0.5000&0.3000\\0.3000&0.5000\end{bmatrix}$，偏离单位矩阵，说明还不正交。
依据：自行构造
4. 第一步：$Q_1=\tfrac12 Q_0(3I-Q_0^\top Q_0)=\begin{bmatrix}0.8050&0.1789\\0.1789&0.8050\end{bmatrix}$。
依据：自行构造
5. 第二步：先用 $Q_1$ 算 $Q_1^\top Q_1=\begin{bmatrix}0.6800&0.2880\\0.2880&0.6800\end{bmatrix}$，代入得 $Q_2=\tfrac12 Q_1(3I-Q_1^\top Q_1)=\begin{bmatrix}0.9080&0.0916\\0.0916&0.9080\end{bmatrix}$。
依据：自行构造
6. 第三步同样代入得 $Q_3=\begin{bmatrix}0.9763&0.0237\\0.0237&0.9763\end{bmatrix}$。
依据：自行构造
7. 第四步 $Q_4=\begin{bmatrix}0.9983&0.0017\\0.0017&0.9983\end{bmatrix}$，第五步 $Q_5=\begin{bmatrix}1.0000&0.0000\\0.0000&1.0000\end{bmatrix}$（保留四位小数）。非对角元趋近 $0$，对角元趋近 $1$，矩阵趋近单位矩阵这一正交矩阵。
依据：自行构造
8. 这个例子里 $X$ 满列秩、没有零奇异值，所以它逼近的正交因子是非奇异的；若矩阵含零奇异值，那些方向会停在 $0$ 而不被拉平。
依据：自行构造
:::

这个例子说明迭代的每一步都在把 $Q_k^\top Q_k$ 往 $I$ 推。
依据：自行构造

### 适用条件与边界

迭代要成立，输入需满足：归一化后最大奇异值小于 $\sqrt3$。这样迭代把非零奇异值推向 $1$。若 $X$ 不满列秩（有零奇异值、$\sigma_n(X)=0$），迭代仍会运行，但那些零方向停在 $0$，极限 $UV^\top$ 在零方向的分量为 $0$，就不是满列秩的正交因子。
依据：paper/src-v2/main.tex Introduction；blog/modula-newton-schulz.md Problem statement

代数上的代价是它给的是近似而非精确：迭代有限步后得到的只是接近 $UV^\top$ 的矩阵，不给出 $U$、$\Sigma$、$V$ 本身。用更多步可以更接近，代价是更多矩阵乘；用少步提速则以精度换速度。
依据：blog/modula-newton-schulz.md Problem statement；blog/modula-newton-schulz.md Designing your own iteration；blog/higham-matrix-sign.txt L65-L73；blog/muon-blog.txt L42-L44

常见误解之一是迭代把所有奇异值拉平为 $1$：实际只有非零奇异值被推向 $1$，零奇异值保持 $0$，因为标量多项式满足 $f(0)=0$，$0$ 是不动点。另一误解是 Newton-Schulz 迭代等价于 SVD：它只是用矩阵乘法近似正交因子 $UV^\top$，并不算出 $U$、$\Sigma$、$V$。
依据：blog/modula-newton-schulz.md Problem statement；blog/modula-newton-schulz.md Designing your own iteration
