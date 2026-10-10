# Kimi Delta Attention

- 描述：KDA 是线性注意力的一种记忆更新规则：先把 $d_k\times d_v$ 的矩阵状态按特征维逐通道衰减，再按 delta 规则用当前键值做秩一修正，靠固定大小的状态达到接近软注意力的表达力。
- 摘要：给线性注意力的矩阵状态加上逐通道遗忘门 $\operatorname{Diag}(\boldsymbol{\alpha}_t)$ 与 delta 修正，合成一个 Householder 型转移矩阵，并按块并行计算，状态大小不随序列长度增长。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 长页

- 开头｜一处状态更新加一处输出，替掉了软注意力的什么（软注意力链接成页）；KDA 是什么、为何需要、怎么工作｜350｜paper/src-v2/1-intro.tex Introduction；paper/src-v2/3-kda.tex Kimi Delta Attention；paper/src-v2/2-preliminary.tex Linear Attention as Online Learning
- ### 状态：一个矩阵记忆｜状态的形状与含义（键维取行、值维取列的关联表）；它由一个递推逐位更新、由它与当前查询算出输出（线性注意力链接成页）；折叠关联表读写小例子｜250｜paper/src-v2/2-preliminary.tex Linear Attention as Online Learning；paper/src-v2/7-related.tex Linear Attention
- ### 修正：delta 规则的写入｜delta 规则把写入看成对重构目标的在线梯度下降、得到秩一修正，与只加不减的区别；折叠两步手算 2×2 状态的例子｜550｜paper/src-v2/2-preliminary.tex DeltaNet: Online Gradient Descent on Reconstruction Loss；paper/src-v2/3-kda.tex Kimi Delta Attention；paper/src-v2/table/update.tex L14-L54
- ### 门控：逐通道的遗忘｜Gated DeltaNet 的按头标量门（链接）；KDA 换成逐通道对角门后的递推；这个门怎么参数化；折叠门只缩放状态某几行的例子｜600｜paper/src-v2/1-intro.tex Introduction；paper/src-v2/2-preliminary.tex Gated DeltaNet as Weight Decay；paper/src-v2/3-kda.tex Kimi Delta Attention；paper/src-v2/4-model.tex Neural Parameterization
- ### 分块并行｜训练要并行而递推是顺序的；切块展开成两项、化成 WY 表示（一句说明 WY）；给状态与输出公式；说明它是 DPLR 的受限变体；折叠证明、块内数值例、提速数字｜950｜paper/src-v2/3-kda.tex Hardware-Efficient Chunkwise Algorithm；paper/src-v2/appendix/proof.tex Derivations for Chunkwise Parallelism of KDA；paper/src-v2/figures/chunks.tex L45-L94；paper/src-v2/6-discuss.tex Relation to DPLR
- ### 作为可学习位置编码｜转移矩阵的累积乘积体现相对位置、且逐通道数据相关，KDA 可解释为一种乘性位置编码；位置编码链接成页；一句对比 RoPE（链接）；折叠与 RoPE 的通式对照｜250｜paper/src-v2/6-discuss.tex as learnable position embeddings；paper/src-v2/table/parallel.tex L16-L64
- ### 适用条件与边界｜状态定长、与序列长度无关换来的是有限容量，长文检索受限；门控制记忆寿命以缓解干扰；代价｜400｜paper/src-v2/1-intro.tex Introduction；paper/src-v2/7-related.tex Discussion；paper/src-v2/2-preliminary.tex Gated DeltaNet as Weight Decay

## 短页

- ### 一个为键值建的矩阵记忆｜是什么：KDA 不为每个位置保存键值，而是维护一个 $d_k\times d_v$ 的矩阵状态，用它和当前查询算出输出（线性注意力链接成页）｜230｜paper/src-v2/2-preliminary.tex Linear Attention as Online Learning；paper/src-v2/3-kda.tex Kimi Delta Attention
- ### 为什么定长状态要遗忘和纠正｜为什么需要：软注意力要存每个位置的键值、开销随长度增长；线性注意力只留一个定长矩阵、容易互相干扰｜330｜paper/src-v2/1-intro.tex Introduction；paper/src-v2/2-preliminary.tex Gated DeltaNet as Weight Decay
- ### 先衰减，再用键值纠正｜怎么做：每个位置分两步更新那个矩阵状态——先按特征维逐通道衰减（乘一个 0 到 1 的门），再按 delta 规则用当前键值纠正并写入；状态大小固定｜300｜paper/src-v2/3-kda.tex Kimi Delta Attention；paper/src-v2/2-preliminary.tex Gated DeltaNet as Weight Decay
