# Newton-Schulz 迭代：术语表

| 术语/符号 | 首现位置 | 定义/含义 |
|---|---|---|
| Newton-Schulz 迭代 | 页面开头 | 一类只用矩阵乘法、基于奇多项式把矩阵推向其极分解正交因子的迭代。本文特指三次型 $Q_{k+1}=\tfrac12 Q_k(3I-Q_k^{\!\top}Q_k)$。 |
| 正交矩阵 | S1 | 满足 $Q^{\!\top}Q=I$ 的矩阵 $Q$（各列相互正交且单位长）。半正交指方阵或瘦高型 $Q^{\!\top}Q=I$。 |
| $Q^{\!\top}Q$ | S2 | 矩阵 $Q$ 的转置与自身相乘；等于 $I$ 当且仅当 $Q$（半）正交。NS 用它衡量"离正交多远"。 |
| $I$ | S2 | 单位阵（对角元为 1、其余为 0），维度由上下文决定。 |
| 奇异值分解（SVD） | S1 | $X=U\Sigma V^{\!\top}$，$U,V$ 正交、$\Sigma$ 对角非负。$\Sigma$ 对角元 $\sigma_i$ 称奇异值，按 $\sigma_1\ge\cdots\ge\sigma_n\ge0$ 排列。本文前置概念，无独立概念页，正文给最小衔接 + 占位链接。 |
| 奇异值 $\sigma_i$ | S1 | $\Sigma$ 的对角元，非负，描述矩阵在各正交方向上的"作用强度"。 |
| 极分解 | S1 | $X=WH$，$W$ 半正交、$H$ 半正定。$W=UV^{\!\top}$（SVD 给出）。NS 迭代求的就是 $W$。 |
| 极分解正交因子 $W$ | S1 | $=UV^{\!\top}$，"最接近 $X$ 的正交矩阵"（Procrustes 解）。 |
| Procrustes 问题 | S1 | $\min_{Q^{\!\top}Q=I}\lVert Q-X\rVert_F$，解 $Q=W=UV^{\!\top}$。 |
| Frobenius 范数 $\lVert X\rVert_F$ | S3 | $\lVert X\rVert_F=\sqrt{\sum_{i,j}X_{ij}^2}$。预处理用它归一化。性质 $\sigma_{\max}(X)\le\lVert X\rVert_F$。 |
| 谱范数 $\lVert X\rVert_2$ | S3 | $=\sigma_{\max}(X)$，矩阵的最大奇异值。 |
| 奇多项式 | S2 | 形如 $p(X)=aX+bXX^{\!\top}X+c(XX^{\!\top})^2X+\cdots$（只含奇次项）的矩阵多项式。满足 $p(U\Sigma V^{\!\top})=Up(\Sigma)V^{\!\top}$。 |
| 标量奇多项式 $f(x)$ | S2 | $f(x)=ax+bx^3+cx^5+\cdots$。三次型 $f(x)=\tfrac32 x-\tfrac12 x^3=\tfrac12 x(3-x^2)$。 |
| sign 函数 $\mathrm{sign}(x)$ | S2 | $\mathrm{sign}(x)=1$ 若 $x>0$，$-1$ 若 $x<0$，$0$ 若 $x=0$。NS 反复迭代 $f$ 趋于 $\mathrm{sign}$ 作用于奇异值。 |
| 收敛条件 $\sigma_{\max}<\sqrt3$ | S3 | 三次 NS 收敛到正交因子的前提：所有奇异值 $<\sqrt3$（且满列秩时收敛到满秩正交因子）。 |
| Frobenius 预处理 | S3 | $Q_0=X/\lVert X\rVert_F$，保证 $\sigma_{\max}(Q_0)\le1<\sqrt3$。 |
| 二次收敛 | S3 | 靠近不动点 $\sigma=1$ 时 $f'(1)=0$，误差以平方速度下降。 |
| 零奇异值边界 | S4 | 奇多项式 $f(0)=0$，故 $\sigma=0$ 是不动点：零奇异值保持 0，非零奇异值→1。 |
| 不动点 | S4 | 满足 $f(x)=x$ 的点。三次 NS 不动点 $\{0,\pm1\}$。 |
| Muon 优化器 | S5 | 对二维参数动量矩阵做 NS 正交化的优化器；前置概念，占位链接 ../muon-optimizer/index.html。 |
| 动量矩阵 $G$ | S5 | Muon 累积的梯度动量矩阵，NS 正交化的作用对象。 |
| cursed quintic | S5 | Muon 实用的五阶奇多项式 $f(x)=3.4445x-4.7750x^3+2.0315x^5$；系数和 $0.701\ne1$ 故非收敛迭代，工程上固定少步数使用。 |
| 对称正交化 | S1 | 不挑特殊行/列的正交化（NS 属此），区别于 Gram-Schmidt 选基准向量。 |
| Gram-Schmidt 正交化 | S1 | 另一种正交化，选一个向量做基准再正交化其余；与 NS 对比，不展开。 |

## 符号一致性

- 迭代变量统一用 $Q_k$（右乘形式主用，因任务给定）；左乘等价形式用 $X_k$ 仅在说明等价性时出现，并标注 $Q\equiv X$。
- 奇异值统一 $\sigma$（标量）、$\Sigma$（对角阵）。
- 范数：$\lVert\cdot\rVert_F$（Frobenius）、$\lVert\cdot\rVert_2$（谱）。
- 上下标：$Q^{\!\top}$ 表转置；下标 $k$ 表迭代步。
- 来源编号：[S1]–[S5] 与 evidence.md 一致。
