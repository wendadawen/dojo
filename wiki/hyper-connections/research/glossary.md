# glossary.md：超连接与 mHC

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| 恒等映射（identity mapping） | 第 1 章 | $\mathbf{x}_l$ 不经修改直达深层的性质；残差递归式中 $\mathbf{x}_l$ 本身 |
| 残差流（residual stream） | 第 1 章 | 残差连接里逐层传递的隐藏状态；HC 中扩展为 $n$ 条并行 |
| 扩张率 $n$ / expansion rate | 第 2 章 | 并行残差流的条数；GLM 取 4 |
| 超隐藏矩阵 $\mathbf{H}\in\mathbb{R}^{n\times d}$ | 第 2 章 | $n$ 条流在宽度 $d$ 上的堆叠表示（HC 论文记号） |
| $\mathcal{T}$ | 第 2 章 | 一个网络层（注意力或 FFN），HC 里作为子层 |
| pre 映射（$\mathbf{A_m}$ / $\mathcal{H}^{\mathrm{pre}}$） | 第 2 章 | 把 $n$ 条流加权求和成单路、喂给子层的读出映射 |
| res 映射（$\mathbf{A_r}$ / $\mathcal{H}^{\mathrm{res}}$） | 第 2 章 | $n$ 条流之间的线性混合矩阵；mHC 约束的对象 |
| post 映射（$\mathbf{B}$ / $\mathcal{H}^{\mathrm{post}}$） | 第 2 章 | 把子层输出分发回 $n$ 条流的写入映射 |
| seesaw effect（跷跷板效应） | 第 2 章 | Pre-Norm/Post-Norm 在梯度消失与表示坍塌之间的权衡 |
| 复合映射 $\prod\mathcal{H}^{\mathrm{res}}$ | 第 3 章 | 多层 res 矩阵的连乘；恒等映射被它取代后产生放大/衰减 |
| 双随机矩阵（doubly stochastic） | 第 3 章 | 非负且每行、每列元素和都为 1 的方阵 |
| Birkhoff polytope | 第 3 章 | 双随机矩阵全体构成的凸多面体=置换矩阵的凸包 |
| 谱范数 $\|\cdot\|_2$ | 第 3 章 | 矩阵的最大奇异值；双随机矩阵的谱范数 ≤1 |
| 置换矩阵（permutation matrix） | 第 3 章 | 每行每列恰一个 1 的矩阵；双随机矩阵的极点 |
| 流形约束（manifold constraint） | 第 3 章 | 把可学习矩阵限制在某集合（此处双随机流形）内的做法 |
| Sinkhorn-Knopp | 第 4 章 | 交替行列归一化、把正矩阵推向双随机的迭代算法 |
| 列归一化 $\mathcal{T}_c$ / 行归一化 $\mathcal{T}_r$ | 第 4 章 | 除以列和/行和，使该方向和为 1 |
| 非扩张（non-expansive） | 第 3 章 | 映射后范数不增大（谱范数≤1） |
| 站点（site） | 第 5 章 | GLM 每层的两个 mHC 使用处：attn_hc 与 ffn_hc，参数独立 |
| $pre$/$post$/$comb$ | 第 5 章 | GLM 实现里三路系数的变量名（comb 即 res 混合矩阵） |
| hc_eps $\varepsilon$ | 第 5 章 | GLM 实现的数值下限 1e-6，加在 σ 输出与归一化分母上 |
