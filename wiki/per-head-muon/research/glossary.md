# Per-Head Muon 术语表

登记全文首次出现的术语、缩写和符号。保证全文含义一致。

| 名称 | 首现位置 | 定义或含义 |
|---|---|---|
| Muon | S1 | MomentUm Orthogonalized by Newton-Schulz，对二维矩阵参数的优化器，把动量矩阵的正交化版本作为更新方向。本页作为前置概念页引用（占位）。 |
| Newton-Schulz 正交化（NS 正交化） | S1 | 一种用多项式迭代近似把矩阵正交化的方法，等价于把矩阵 SVD 后所有奇异值拉平为 1。本页作为前置概念页引用（占位）。 |
| 正交化 $\mathrm{Ortho}(\cdot)$ | S1 | 把矩阵近似变为半正交矩阵的操作，$\mathrm{Ortho}(X)\approx UV^\top$（$X=USV^\top$）。 |
| 奇异值 | S1 | SVD 中对角矩阵 $S$ 的非负对角元，衡量矩阵在各奇异方向的作用强度。 |
| 半正交矩阵 | S1 | 满足 $O^\top O=I$ 或 $OO^\top=I$ 的矩阵（行或列正交）；正交化的目标。 |
| Q/K/V 投影 | S1 | 注意力中对 query/key/value 的线性投影权重矩阵。本页只用其"沿头维度堆叠"的结构事实。 |
| 头（head） | S1 | 多头注意力中的一个注意力单元，对应投影权重的一个行块。 |
| 头维度（head dimension, $d_h$） | S1 | 一个头的表示维度；投影权重按头切分时每个头块的行数。 |
| 头数（number of heads, $H$） | S1 | 多头注意力的头个数；投影权重沿行方向有 $H$ 个头块。 |
| 模型维度（model dimension, $d$） | S1 | 投影权重的列数（输入维度）。 |
| 动量矩阵 $M$ | S1 | Muon 中累积的梯度动量矩阵；Per-Head Muon 切分与正交化的对象。 |
| 头块 $M_h$ | S1 | 动量矩阵 $M$ 沿头维度切分后第 $h$ 个头的块，$M_h\in\mathbb{R}^{d_h\times d}$。 |
| 全矩阵正交化 | S1 | 对完整 $M=[M_1;\dots;M_H]$ 整体做 NS 正交化；原版 Muon 对每个投影权重的做法。 |
| 按头正交化（per-head orthogonalization） | S2 | 对每个 $M_h$ 单独做 NS 正交化；Per-Head Muon 的核心改动。 |
| Gram 矩阵 $XX^\top$ | S3 | NS 迭代中出现的方阵，其规模决定 NS 每步主要计算开销。 |
| Frobenius 范数 $\|\cdot\|_F$ | S1 | 矩阵所有元素平方和的平方根；NS 迭代前用它归一化以保证奇异值在 $[0,1]$。 |
| all-gather | S4 | 一种集合通信原语，所有 rank 互相收集对方参数拼成全参数缓冲区。 |
| P2P 通信（peer-to-peer communication） | S4 | 点对点通信，一个 rank 直接向另一个特定 rank 收发数据。 |
| DP rank | S4 | 数据并行组中的一个进程/设备；分布式优化器把参数分片到各 DP rank。 |
| 参数分片（parameter sharding） | S4 | 分布式优化器把参数均匀切分到各 rank，每 rank 只持有一部分。 |
| model-chunk | S4 | 模型参数缓冲区的一个粒度单元；P2P 方案在 model-chunk 粒度流水化通信与计算。 |
