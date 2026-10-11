# SVD

- 描述：奇异值分解把任意矩阵分解为两个正交矩阵夹一个对角非负矩阵，对角元是奇异值，刻画矩阵在正交方向上的伸缩，并给出最优低秩近似。
- 摘要：任意矩阵都能写成一「旋转、按坐标伸缩、再旋转」的复合，伸缩系数就是奇异值。据此得到最优低秩近似、伪逆与条件数。
- 主题：数学基础
- 标签：数学与数值

## 资料

- Stephen Boyd、Sanjay Lall. Singular Value Decomposition. 斯坦福大学 EE263 第 17 讲，2026 秋季版. [https://ee263.stanford.edu/lectures/26q3/original/17_svd.pdf](https://ee263.stanford.edu/lectures/26q3/original/17_svd.pdf)
- Stephen Boyd、Sanjay Lall. Matrix norm. 斯坦福大学 EE263 第 16 讲，2026 秋季版. [https://ee263.stanford.edu/lectures/26q3/original/16_norm.pdf](https://ee263.stanford.edu/lectures/26q3/original/16_norm.pdf)
- Stephen Boyd、Sanjay Lall. Ellipsoids. 斯坦福大学 EE263 第 15 讲，2026 秋季版. [https://ee263.stanford.edu/lectures/26q3/original/15_ellipsoids.pdf](https://ee263.stanford.edu/lectures/26q3/original/15_ellipsoids.pdf)
- LAPACK Users' Guide. Singular Value Decomposition. SIAM，第三版. [https://www.netlib.org/lapack/lug/node53.html](https://www.netlib.org/lapack/lug/node53.html)
- M. P. Deisenroth、A. A. Faisal、C. S. Ong. Mathematics for Machine Learning. Cambridge University Press, 2020，第 4.5、4.6 节. [https://mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf)

## 短页

### 两组正交方向夹一组伸缩

奇异值分解（SVD）把任意 $m\times n$ 实矩阵 $A$ 写成 $A=U\Sigma V^{\!\top}$。$U$、$V$ 是正交矩阵，即列向量两两正交、每列长度为 $1$ 的方阵。$\Sigma$ 与 $A$ 同形状，只有主对角线可能非零，对角元 $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$ 叫奇异值，$r$ 是 $A$ 的秩，$r$ 以后的为零。$U$ 的列叫左奇异向量，$V$ 的列叫右奇异向量。写成求和，$A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\!\top}$，每一项 $\sigma_i u_i v_i^{\!\top}$ 是一个秩 $1$ 矩阵。丢掉最小的那些奇异值项，只留前几项得到的矩阵，就是 $A$ 的最优低秩近似：它是在所有秩不超过保留项数的矩阵里离 $A$ 最近的那个（距离按谱范数衡量）。
依据：blog/mml-book/mml-ch4-svd.txt L27-L100；blog/mml-book/mml-ch4-svd.txt L668-L683；blog/lapack-lug/svd.txt L9-L39；blog/mml-book/mml-ch4-svd.txt L755-L767

### 特征分解的局限

方阵才有特征分解 $A=PDP^{-1}$，而且只有当方阵有一组线性无关的特征向量时才存在，$P$ 一般也不是正交矩阵。长方形矩阵没有特征值可言。SVD 对任意矩阵都存在，它用两组正交方向（$U$ 与 $V$ 的列）夹一组非负的伸缩系数，把这些系数放到对角线上，等于在一组正交坐标下把矩阵化成了对角。
依据：blog/mml-book/mml-ch4-svd.txt L16-L25；blog/mml-book/mml-ch4-svd.txt L460-L531

### 旋转、伸缩、再旋转

把 $y=Ax$ 看成三步：先按 $V^{\!\top}$ 做一次坐标变换，把输入坐标换到以右奇异向量为轴的坐标系；再按 $\Sigma$ 沿各坐标轴伸缩，第 $i$ 个坐标乘 $\sigma_i$，并按 $A$ 的形状在末尾补零或截断；最后按 $U$ 做第二次坐标变换，转回输出空间的坐标系。奇异值就是这些方向上的放大倍数，最大为 $\sigma_1$，最小为 $\sigma_r$。
依据：blog/ee263/ee263-17_svd.txt L146-L198；blog/ee263/ee263-17_svd.txt L386-L398

## 长页

一个 $m\times n$ 的实矩阵 $A$ 作用在向量 $x$ 上得到 $y=Ax$，把一个 $n$ 维向量映成一个 $m$ 维向量。方阵才有特征分解，而奇异值分解（SVD）对任意矩阵都存在：它把 $A$ 写成 $U\Sigma V^{\!\top}$，其中 $U$、$V$ 是正交矩阵，$\Sigma$ 只有对角线可能非零。它可以读成三个动作的复合——换一组正交坐标、沿坐标轴伸缩、再换回原来这组坐标；伸缩倍数就是奇异值。按奇异值从大到小排列，最大的几个方向刻画了矩阵把向量放大的主要通道，去掉最小的那些方向就得到最优的低秩近似。
依据：blog/ee263/ee263-17_svd.txt L16-L70；blog/ee263/ee263-17_svd.txt L146-L173；blog/mml-book/mml-ch4-svd.txt L9-L25；blog/mml-book/mml-ch4-svd.txt L27-L51；blog/ee263/ee263-17_svd.txt L484-L511

### 分解式与两种写法

设 $A$ 是 $m\times n$ 的实矩阵，$r$ 是它的秩。奇异值分解把 $A$ 写成
$$A=U\Sigma V^{\!\top},$$
其中 $U$ 是 $m\times m$ 的正交矩阵，$V$ 是 $n\times n$ 的正交矩阵，$\Sigma$ 是与 $A$ 同样 $m\times n$ 的矩阵，除主对角线外全是零。$U$ 的列向量 $u_1,\dots,u_m$ 叫左奇异向量，$V$ 的列向量 $v_1,\dots,v_n$ 叫右奇异向量。$\Sigma$ 的对角元 $\sigma_i$ 非负，叫奇异值，按从大到小排列，前面 $r$ 个为正、其余为零。正交矩阵指列向量两两正交、每列长度为 $1$ 的方阵，它的转置等于它的逆。
依据：blog/mml-book/mml-ch4-svd.txt L27-L51；blog/lapack-lug/svd.txt L9-L39

把分解式展开成 $r$ 项之和，
$$A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\!\top},$$
每一项 $\sigma_i u_i v_i^{\!\top}$ 是一个秩 $1$ 矩阵。通常只用前 $r$ 组奇异向量：取 $U$ 的前 $r$ 列、$V$ 的前 $r$ 列、$\Sigma$ 左上角 $r\times r$ 的对角块，得 $A=U_r\Sigma_r V_r^{\!\top}$，此时 $U_r$ 与 $V_r$ 是列正交的（$U_r^{\!\top}U_r=I_r$、$V_r^{\!\top}V_r=I_r$），$\Sigma_r$ 是 $r\times r$ 的对角矩阵，这叫薄 SVD。把 $U$ 补上前 $r$ 列之外的一组正交向量补成 $m\times m$、$V$ 同样补成 $n\times n$、$\Sigma$ 补上零行零列，得到全 SVD。两种写法给出同一组奇异值。
依据：blog/ee263/ee263-17_svd.txt L73-L93；blog/mml-book/mml-ch4-svd.txt L75-L100；blog/ee263/ee263-17_svd.txt L313-L364

前 $r$ 个左奇异向量 $u_1,\dots,u_r$ 张成 $A$ 的列空间（值域）的一组标准正交基，前 $r$ 个右奇异向量 $v_1,\dots,v_r$ 张成 $A$ 的零空间的正交补，也就是 $A$ 的行空间。当 $r<n$ 时（存在零奇异值），其余右奇异向量 $v_{r+1},\dots,v_n$ 张成 $A$ 的零空间，也就是使 $Ax=0$ 的 $x$ 全体。零奇异值对应的正是被 $A$ 映到零的那些方向。
依据：blog/ee263/ee263-17_svd.txt L130-L145；blog/mml-book/mml-ch4-svd.txt L347-L352；blog/mml-book/mml-ch4-svd.txt L27-L51

### 几何：单位球映成椭球

把 $A$ 看成线性映射 $x\mapsto Ax$。$A$ 把 $\mathbb{R}^n$ 里的单位球映成 $\mathbb{R}^m$ 里的一个椭球。这个椭球的半轴长度就是各奇异值，方向就是对应的左奇异向量 $u_i$，$\sigma_1$ 对应最长半轴，$\sigma_r$ 对应最短半轴。
依据：blog/ee263/ee263-17_svd.txt L16-L47；blog/ee263/ee263-15_ellipsoids.txt L18-L55

从分解式看，$y=Ax$ 是三个动作：先按 $V^{\!\top}$ 换一组正交坐标（正交变换不改变长度，只转动坐标轴），再按 $\Sigma$ 沿新坐标轴伸缩、第 $i$ 个坐标乘 $\sigma_i$，并按 $A$ 的形状补零（$m>n$）或截断（$m<n$），最后按 $U$ 转动到输出方向。单位圆经这三步变成椭圆。
依据：blog/ee263/ee263-17_svd.txt L146-L198；blog/ee263/ee263-17_svd.txt L386-L425

::: 单位球映成椭球的小例子
取 $A=\begin{pmatrix}1&2\\2&1\end{pmatrix}$，奇异向量由 $A^{\!\top}A$ 的特征分解得到（推导见「奇异值就是 $A^{\!\top}A$ 特征值的平方根」一节）。
依据：自行构造

1. 右奇异向量：$A^{\!\top}A=\begin{pmatrix}5&4\\4&5\end{pmatrix}$，它的特征值是 $9$ 与 $1$，对应单位特征向量 $v_1=\frac{1}{\sqrt2}(1,1)$、$v_2=\frac{1}{\sqrt2}(1,-1)$。
2. 奇异值：$\sigma_1=\sqrt9=3$、$\sigma_2=\sqrt1=1$，即两个伸缩倍数分别是 $3$ 和 $1$。
3. 左奇异向量：$u_1=Av_1/\sigma_1=\frac{1}{\sqrt2}(1,1)$，$u_2=Av_2/\sigma_2=\frac{1}{\sqrt2}(-1,1)$。
4. 取单位圆上沿 $v_1$ 的点 $\frac{1}{\sqrt2}(1,1)$：按 $V^{\!\top}$ 变成 $(1,0)$，按 $\Sigma$ 变成 $(3,0)$，按 $U$ 变成 $\frac{1}{\sqrt2}(3,3)$，长度 $3$。
5. 取沿 $v_2$ 的点 $\frac{1}{\sqrt2}(1,-1)$：经 $V^{\!\top}$ 变成 $(0,1)$，经 $\Sigma$ 仍是 $(0,1)$，经 $U$ 变成 $\frac{1}{\sqrt2}(-1,1)$，长度 $1$。
依据：自行构造
:::

### 奇异值就是 $A^{\!\top}A$ 特征值的平方根

把 $A=U\Sigma V^{\!\top}$ 代入 $A^{\!\top}A$，利用 $U$ 列正交即 $U^{\!\top}U=I$，
$$A^{\!\top}A=V\Sigma^{\!\top}\Sigma V^{\!\top}.$$
$\Sigma^{\!\top}\Sigma$ 是角元为 $\sigma_i^2$ 的对角矩阵，$V$ 是正交矩阵，右端正是 $A^{\!\top}A$ 的正交对角化。于是右奇异向量 $v_i$ 是 $A^{\!\top}A$ 的特征向量，$\sigma_i^2$ 是对应的特征值，奇异值就是 $A^{\!\top}A$ 特征值的平方根。同样地 $AA^{\!\top}=U\Sigma\Sigma^{\!\top}U^{\!\top}$，$\Sigma\Sigma^{\!\top}$ 的角元同样是 $\sigma_i^2$，左奇异向量 $u_i$ 是 $AA^{\!\top}$ 的特征向量。$A^{\!\top}A$ 与 $AA^{\!\top}$ 的非零特征值相同，都是 $\sigma_i^2$。
依据：blog/ee263/ee263-17_svd.txt L94-L129；blog/ee263/ee263-16_norm.txt L61-L89

::: 由 $A^{\!\top}A$ 的特征分解构造 SVD
依据：blog/mml-book/mml-ch4-svd.txt L230-L359

1. 算 $A^{\!\top}A$。它是对称半正定矩阵，必能正交对角化为 $A^{\!\top}A=V D V^{\!\top}$，其中 $D$ 的对角元 $\lambda_i\ge0$ 降序排列，$V$ 是正交矩阵。
2. $V$ 的列就是右奇异向量 $v_i$，奇异值 $\sigma_i=\sqrt{\lambda_i}$。
3. 对每个 $\sigma_i>0$，取 $u_i=Av_i/\sigma_i$，这些 $u_i$ 两两正交且长度为 $1$，就是左奇异向量。
4. 把 $v_i$ 按 $V$ 排、$\sigma_i$ 放上对角线拼成 $\Sigma$；$u_i$ 只有 $r$ 个，$r<m$ 时再取 $A^{\!\top}$ 的零空间的一组标准正交基补满 $m$ 列，拼成 $U$，得到 $A=U\Sigma V^{\!\top}$。
依据：blog/mml-book/mml-ch4-svd.txt L230-L359

取 $A=\begin{pmatrix}2&0\\0&-1\end{pmatrix}$ 计算如下。
依据：自行构造

1. $A^{\!\top}A=\begin{pmatrix}4&0\\0&1\end{pmatrix}$，特征值 $4$、$1$，对应单位特征向量 $v_1=(1,0)$、$v_2=(0,1)$。
2. 奇异值 $\sigma_1=2$、$\sigma_2=1$。
3. $u_1=Av_1/\sigma_1=(1,0)$，$u_2=Av_2/\sigma_2=(0,-1)$。
4. 于是 $U=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$，$\Sigma=\begin{pmatrix}2&0\\0&1\end{pmatrix}$，$V=\begin{pmatrix}1&0\\0&1\end{pmatrix}$。
依据：自行构造

这个矩阵自身的特征值是 $2$ 与 $-1$，而奇异值是 $2$ 与 $1$：特征值可正可负，奇异值一律非负。
依据：自行构造
:::

### 奇异值刻画矩阵

秩等于非零奇异值的个数。谱范数（最大的放大倍数）等于 $\sigma_1$，Frobenius 范数 $\lVert A\rVert_F=\sqrt{\operatorname{trace}(A^{\!\top}A)}=\sqrt{\sum_i\sigma_i^2}$，最后一步用了 $A^{\!\top}A=V\Sigma^{\!\top}\Sigma V^{\!\top}$：$\operatorname{trace}(A^{\!\top}A)$ 等于它的对角元之和，即 $\sum_i\sigma_i^2$。$A$ 是可逆方阵时，条件数 $\kappa(A)=\sigma_1/\sigma_r$ 度量方程 $Ax=y$ 的解对右端误差的相对放大：$y$ 有小的相对误差时，解的相对误差不超过 $\kappa(A)$ 倍，$\kappa$ 越大方程越病态。
依据：blog/ee263/ee263-17_svd.txt L74-L107；blog/ee263/ee263-16_norm.txt L162-L181；blog/ee263/ee263-16_norm.txt L42-L52；blog/ee263/ee263-17_svd.txt L426-L483；blog/lapack-lug/svd.txt L139-L140

::: 用同一个矩阵算四个量
依据：自行构造

1. $A=\begin{pmatrix}1&2\\2&1\end{pmatrix}$ 的两个奇异值是 $3$ 与 $1$，都非零，所以秩为 $2$。
2. 谱范数 $\lVert A\rVert_2=\sigma_1=3$。
3. Frobenius 范数 $\lVert A\rVert_F=\sqrt{1^2+2^2+2^2+1^2}=\sqrt{10}\approx3.162$，与 $\sqrt{\sigma_1^2+\sigma_2^2}=\sqrt{9+1}=\sqrt{10}$ 一致。
4. 条件数 $\kappa(A)=\sigma_1/\sigma_2=3/1=3$。
依据：自行构造
:::

### 最优低秩近似

按奇异值降序只保留前 $p$ 项，得到截断 SVD（truncated SVD）
$$A_p=\sum_{i=1}^{p}\sigma_i u_i v_i^{\!\top},\qquad p\le r,$$
它的秩为 $p$。Eckart–Young 定理给出最优性：在所有秩不超过 $p$ 的矩阵 $B$ 中，$A_p$ 使谱范数误差 $\lVert A-B\rVert_2$ 达到最小，且最小值等于 $\sigma_{p+1}$；等价地，$\sigma_i$ 等于 $A$ 到最近的秩不超过 $i-1$ 的矩阵的距离。奇异值衰减得越快，越少的前几项就越能代表 $A$，这是用截断 SVD 做低秩压缩的依据。
依据：blog/mml-book/mml-ch4-svd.txt L604-L817；blog/ee263/ee263-17_svd.txt L484-L559

::: Eckart–Young 的证明与一个小例子
依据：blog/ee263/ee263-17_svd.txt L512-L538

1. 设 $m\times n$ 的矩阵 $B$ 的秩不超过 $p$，它的零空间维数是 $n-\mathrm{rank}(B)\ge n-p$。
2. $v_1,\dots,v_{p+1}$ 张成一个 $p+1$ 维子空间，两个子空间的维数之和超过 $n$，必相交，取交里的单位向量 $z$。
3. $z$ 在 $B$ 的零空间里，所以 $Bz=0$，从而 $\lVert A-B\rVert_2\ge\lVert(A-B)z\rVert_2=\lVert Az\rVert_2$。
4. $z$ 又在 $v_1,\dots,v_{p+1}$ 张成的空间里，把它按这组正交基展开；由 $Av_i=\sigma_iu_i$、$u_i$ 两两正交且 $\lVert z\rVert_2=1$，得 $\lVert Az\rVert_2^2=\sum_{i=1}^{p+1}\sigma_i^2(v_i^{\!\top}z)^2\ge\sigma_{p+1}^2$。
5. 于是 $\lVert A-B\rVert_2\ge\sigma_{p+1}$，即没有秩不超过 $p$ 的矩阵能比 $A_p$ 更接近 $A$；而 $A-A_p=\sum_{i=p+1}^{r}\sigma_iu_iv_i^{\!\top}$ 的谱范数正是 $\sigma_{p+1}$，所以 $A_p$ 取到这个下界。
依据：blog/ee263/ee263-17_svd.txt L512-L538

再用 $A=\begin{pmatrix}1&2\\2&1\end{pmatrix}$（$\sigma_1=3,\sigma_2=1$）算一次秩 $1$ 近似。
依据：自行构造

1. $A_1=\sigma_1u_1v_1^{\!\top}=3\cdot\frac{1}{\sqrt2}(1,1)\cdot\frac{1}{\sqrt2}(1,1)^{\!\top}=\begin{pmatrix}1.5&1.5\\1.5&1.5\end{pmatrix}$。
2. $A-A_1=\begin{pmatrix}-0.5&0.5\\0.5&-0.5\end{pmatrix}$，它的谱范数是 $1$，等于被丢弃的 $\sigma_2$，与定理一致。
依据：自行构造
:::

### 伪逆

由 SVD 可以定义 $A$ 的伪逆（Moore–Penrose 逆）：把 $\Sigma$ 里每个非零 $\sigma_i$ 换成 $1/\sigma_i$、并把形状转置，记为 $\Sigma^{+}$，令
$$A^{+}=V\Sigma^{+}U^{\!\top}.$$
当 $A$ 是长矩阵（列满秩）时 $A^{+}=(A^{\!\top}A)^{-1}A^{\!\top}$，$A^{+}y$ 给出最小二乘解，即使 $\lVert Ax-y\rVert_2$ 最小的 $x$；当 $A$ 是宽矩阵（行满秩）时 $A^{+}=A^{\!\top}(AA^{\!\top})^{-1}$，$A^{+}y$ 给出最小范数解，即使 $Ax=y$ 成立且 $\lVert x\rVert_2$ 最小的 $x$。一般的 $A^{+}y$ 是所有最小二乘解中范数最小的那个。
依据：blog/ee263/ee263-17_svd.txt L245-L312

::: 列满秩时伪逆与 $(A^{\!\top}A)^{-1}A^{\!\top}$ 相同
依据：blog/ee263/ee263-17_svd.txt L73-L100；blog/ee263/ee263-17_svd.txt L245-L265

1. 列满秩时 $A=U_r\Sigma_rV_r^{\!\top}$，$A^{\!\top}A=V_r\Sigma_r^{\!\top}\Sigma_rV_r^{\!\top}=V_r\Sigma_r^2V_r^{\!\top}$。
2. 于是 $(A^{\!\top}A)^{-1}A^{\!\top}=V_r\Sigma_r^{-2}V_r^{\!\top}\cdot V_r\Sigma_rU_r^{\!\top}=V_r\Sigma_r^{-1}U_r^{\!\top}=A^{+}$。
依据：blog/ee263/ee263-17_svd.txt L73-L100；blog/ee263/ee263-17_svd.txt L245-L265
:::

### 适用条件与边界

SVD 对任意实矩阵都存在，这是它比特征分解适用范围更广的地方：特征分解只对方阵，且只有当方阵有一组线性无关的特征向量时才存在。奇异值矩阵 $\Sigma$ 是唯一的。
依据：blog/mml-book/mml-ch4-svd.txt L16-L25；blog/mml-book/mml-ch4-svd.txt L46-L51；blog/mml-book/mml-ch4-svd.txt L460-L531

用 $A^{\!\top}A$ 的特征分解去算 SVD 在数值上不稳定，实际计算先把矩阵化为双对角形（只有主对角线和相邻一条对角线可能非零）再迭代求奇异值。Eckart–Young 的最优性和误差界 $\lVert A-A_p\rVert_2=\sigma_{p+1}$ 都是在谱范数意义下说的。
依据：blog/mml-book/mml-ch4-svd.txt L454-L456；blog/lapack-lug/svd.txt L40-L105；blog/mml-book/mml-ch4-svd.txt L743-L774
