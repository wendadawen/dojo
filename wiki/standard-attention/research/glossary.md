# 标准缩放点积注意力：术语表

登记全文首次出现的术语、缩写和符号。后续阶段写作和审查以此为准。

## 术语

| 术语 | 首现位置 | 定义或含义 |
|---|---|---|
| 注意力（Attention） | 标题 | 让序列中每个位置动态"查询"其它位置并按匹配程度加权聚合信息的机制 |
| 标准注意力 / 标准 Transformer 注意力 | 标题 | Vaswani et al. 2017 提出的原始缩放点积注意力 + 多头形式 |
| 缩放点积注意力（Scaled Dot-Product Attention, SDPA） | 标题 | 用 $q\cdot k$ 算分数、除 $\sqrt{d_k}$、softmax 归一化、对 $v$ 加权求和的注意力函数 |
| 自注意力（Self-Attention） | S1 | $Q,K,V$ 都来自同一输入序列的线性投影；又称 intra-attention |
| 交叉注意力（Cross-Attention） | S1 | $K,V$ 来自另一序列（如编码器输出），$Q$ 来自当前序列；公式与自注意力相同 |
| 查询（Query, $q$ 或 $Q$） | S1 | 当前位置生成的"我在找什么"向量；与每个 key 做点积算匹配度 |
| 键（Key, $k$ 或 $K$） | S1 | 每个位置生成的"我能被什么找到"向量；与 query 做点积 |
| 值（Value, $v$ 或 $V$） | S1 | 每个位置"被找到后返回的内容"向量；按注意力权重加权聚合 |
| 同源投影 | S1 | 自注意力下 $Q=XW^Q$、$K=XW^K$、$V=XW^V$，三者来自同一输入 $X$ 的不同线性投影 |
| 数据库类比 | S1 | 把 $q/k/v$ 类比成数据库查询/索引/记录的教学解释；列失效边界 |
| 点积（Dot Product） | S2 | $a\cdot b=\sum_i a_i b_i=\|a\|\|b\|\cos\theta$；衡量两向量方向一致性 |
| 相似度矩阵 / 注意力分数矩阵（$QK^\top$） | S2 | $n\times m$ 矩阵，第 $(i,j)$ 元素为 $q_i\cdot k_j$ |
| 缩放因子（Scaling Factor, $\sqrt{d_k}$） | S2 | 除以 $\sqrt{d_k}$ 把点积方差归一到 1，防止 softmax 饱和 |
| $d_k$ | S2 | 查询和键向量的维度（论文设置 $d_k=64$） |
| $d_v$ | S2 | 值向量的维度（论文设置 $d_v=64=d_k$） |
| $d_{model}$ | S2 | 模型的隐藏维度（论文设置 $d_{model}=512$） |
| softmax | S2 | 把任意实数向量转为非负且和为 1 的概率分布的函数 $p_i=e^{z_i}/\sum_j e^{z_j}$ |
| 注意力权重矩阵（$A$） | S2 | softmax 后的 $n\times m$ 矩阵，每行和为 1，表示每个 query 在 key 上的概率分配 |
| 凸组合 | S2 | $AV$ 的每行是 $m$ 个值向量的凸组合（权重非负且和为 1） |
| 路径长度 | S1 | 序列中两个位置之间信息传递需要的层数；自注意力为 $O(1)$、RNN 为 $O(n)$ |
| 顺序操作数 | S1 | 必须串行执行的最少操作数；自注意力 $O(1)$、RNN $O(n)$ |
| 方差（Variance, $\text{Var}$） | S3 | 随机变量取值分散程度的度量；标准差 $\sigma=\sqrt{\text{Var}}$ |
| 雅可比（Jacobian, $\partial p/\partial z$） | S3 | softmax 输出对输入的偏导矩阵；$\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ |
| 饱和（Saturation） | S3 | softmax 输入过大时输出近 one-hot、梯度近零的状态 |
| 加性注意力（Additive Attention, Bahdanau 2014） | S3 | 用前馈网络算"分数"的注意力；与点积对照；不展开机制 |
| 多头注意力（Multi-Head Attention, MHA） | S4 | 把 $Q,K,V$ 投影到 $h$ 个子空间分别做注意力再拼接的标准使用方式 |
| 头（Head, $\text{head}_i$） | S4 | 多头中的一个，在 $d_k=d_{model}/h$ 维子空间做一次缩放点积注意力 |
| 子空间（Subspace） | S4 | 多头中每个头所在的 $d_k$ 维向量空间；论文 $d_k=64$ |
| 投影矩阵 $W^Q_i, W^K_i, W^V_i, W^O$ | S4 | 多头中可学习的线性投影参数；$W^O$ 把拼接结果投影回 $d_{model}$ |
| 因果遮罩（Causal Mask, $M$） | S4 | 自回归解码时把未来位置分数设为 $-\infty$ 的矩阵；softmax 后权重为 0 |
| 自回归（Auto-regressive） | S4 | 生成第 $t$ 个 token 时只能看到 $1..t-1$ 的 token；解码器特性 |
| 时间复杂度（Time Complexity） | S5 | 计算量随序列长度 $n$ 增长的方式；标准注意力 $O(n^2 d_k)$ |
| 空间复杂度（Space Complexity） | S5 | 内存随 $n$ 增长的方式；标准注意力 $O(n^2)$ 存注意力矩阵 |
| 瓶颈（Bottleneck） | S5 | 标准注意力 $QK^\top$ 这步产生 $n\times n$ 矩阵，是后续变体要解决的问题 |
| KV cache | S5 | 自回归推理时缓存的 $K,V$ 矩阵；随 $n$ 与头数 $h$ 线性增长 |
| 位置无关（Permutation-Invariant） | S5 | 注意力对输入行重排不变；位置信息需外接位置编码 |
| Transformer | S5 | Vaswani et al. 2017 提出的架构；标准注意力是其核心子层 |
| RNN（Recurrent Neural Network） | S1 | 处理序列的传统架构；隐状态逐步传递；本文作对照 |
| CNN（Convolutional Neural Network） | S1 | 用固定窗口处理序列的架构；本文作对照 |

## 符号

| 符号 | 首现位置 | 含义 |
|---|---|---|
| $Q$ | S1 公式 F6 | 查询矩阵，$\mathbb{R}^{n\times d_k}$ |
| $K$ | S1 公式 F6 | 键矩阵，$\mathbb{R}^{m\times d_k}$ |
| $V$ | S1 公式 F6 | 值矩阵，$\mathbb{R}^{m\times d_v}$ |
| $q_i$ | S2 | $Q$ 的第 $i$ 行（第 $i$ 个查询向量） |
| $k_j$ | S2 | $K$ 的第 $j$ 行（第 $j$ 个键向量） |
| $K^\top$ | S2 公式 F1 | $K$ 的转置，$\mathbb{R}^{d_k\times m}$ |
| $QK^\top$ | S2 公式 F1 | 相似度矩阵，$\mathbb{R}^{n\times m}$，第 $(i,j)$ 元素 $q_i\cdot k_j$ |
| $s_{ij}$ | S2 | $QK^\top$ 的第 $(i,j)$ 元素，未缩放分数 |
| $d_k$ | S2 | 查询和键的向量维度 |
| $d_v$ | S2 | 值的向量维度（论文 $d_v=d_k$） |
| $d_{model}$ | S2 | 模型隐藏维度（论文 $d_{model}=512$） |
| $\sqrt{d_k}$ | S2 公式 F1 | 缩放因子，把点积方差归一到 1 |
| $n$ | S2 | 查询的序列长度（自注意力下 $n=m$） |
| $m$ | S2 | 键/值的序列长度 |
| $\text{softmax}$ | S2 公式 F1 | 沿每行归一化为概率分布的函数 |
| $p_i$ | S2 | softmax 后第 $i$ 个权重 |
| $A$ | S2 | softmax 后的注意力权重矩阵，$\mathbb{R}^{n\times m}$ |
| $AV$ | S2 | 注意力输出，$\mathbb{R}^{n\times d_v}$ |
| $\text{Var}$ | S3 公式 F3 | 方差 |
| $\text{std}$ | S3 公式 F3 | 标准差，$\sqrt{\text{Var}}$ |
| $\sigma$ | S3 | 标准差的常用符号（本文主要用 $\text{std}$） |
| $\partial p_i/\partial z_j$ | S3 公式 F4 | softmax 雅可比，$=p_i(\delta_{ij}-p_j)$ |
| $\delta_{ij}$ | S3 公式 F4 | Kronecker delta，$i=j$ 时为 1 否则为 0 |
| $X$ | S1 公式 F6 | 输入序列矩阵，$\mathbb{R}^{n\times d_{model}}$ |
| $W^Q, W^K, W^V$ | S1 公式 F6 | 自注意力下的投影矩阵 |
| $h$ | S4 公式 F2 | 头的数量（论文 $h=8$） |
| $\text{head}_i$ | S4 公式 F2 | 第 $i$ 个头的输出 |
| $W^Q_i, W^K_i, W^V_i$ | S4 公式 F2 | 第 $i$ 个头的投影矩阵 |
| $W^O$ | S4 公式 F2 | 多头拼接后的输出投影矩阵 |
| $M$ | S4 公式 F5 | 因果遮罩矩阵，上三角 $-\infty$、对角及以下为 $0$ |
| $\text{Concat}$ | S4 公式 F2 | 沿特征维拼接多个头的输出 |
| $O(\cdot)$ | S5 | 复杂度记号 |

## 缩写

| 缩写 | 全称 | 含义 |
|---|---|---|
| SDPA | Scaled Dot-Product Attention | 缩放点积注意力 |
| MHA | Multi-Head Attention | 多头注意力 |
| RNN | Recurrent Neural Network | 循环神经网络 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| KV | Key-Value | 键值对，常用在"KV cache"中 |
| LLM | Large Language Model | 大语言模型 |
| LN | Layer Normalization | 层归一化（本文只在 BN/LN 后方差接近 1 处提及） |

全文符号含义保持一致：$Q,K,V$ 始终指查询/键/值矩阵，$d_k$ 始终指键维度，$h$ 始终指头数。2×2 手算例子复用同一组符号（$Q,K,V$ 为教学构造的具体数值矩阵）。
