# 标准缩放点积注意力：核心论断与证据

来源优先级：原始论文 > 权威教材/同行评审综述 > 官方文档 > 固定版本源码。本页核心论断全部来自 Vaswani et al. 2017 原论文（NeurIPS 2017, arXiv:1706.03762）及其 §3.2.1 脚注 4 的方差推导。WebSearch 获取的二手资料仅用于交叉确认公式与论文位置，不作为核心论断的唯一依据。

## C 论断（核心机制）

### C1 注意力解决"动态对齐 + 加权聚合"问题

- **论断内容**：序列建模中每个位置需要"看到"其它位置的相关信息，且相关位置随任务变化（指代消解、长程依赖、源-目标对齐）；注意力让每个位置生成查询向量去匹配其它位置的键向量，按匹配程度对值向量加权求和，得到上下文相关的输出。任意两个位置直接交互，路径长度 $O(1)$，可并行计算。
- **来源定位**：Vaswani et al. 2017, "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762, §1 第 3 段（"Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence..."）与 §4.1 Table 1（复杂度对比）。
- **适用条件**：序列长度 $n$、维度 $d$、$n<d$ 时自注意力比 RNN 计算量更小（论文 §4.1 Table 1）。
- **置信状态**：已确认。

### C2 缩放点积注意力是标准定义

- **论断内容**：$\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$，其中 $Q\in\mathbb{R}^{n\times d_k}$、$K\in\mathbb{R}^{m\times d_k}$、$V\in\mathbb{R}^{m\times d_v}$；softmax 沿 $K$ 维度（每行）归一化。
- **来源定位**：Vaswani et al. 2017, §3.2.1, Eq.(1)，论文第 4 页。
- **适用条件**：$Q,K$ 在同一 $d_k$ 维向量空间。
- **置信状态**：已确认。

### C3 $\sqrt{d_k}$ 缩放来自方差推导（不是经验常数）

- **论断内容**：当 $q,k$ 各分量为独立同分布、均值 0、方差 1 时，点积 $q\cdot k=\sum_{i=1}^{d_k}q_i k_i$ 的方差为 $d_k$（$d_k$ 个独立项求和，每项方差 1）；标准差为 $\sqrt{d_k}$。除以 $\sqrt{d_k}$ 把方差归一到 1，使 softmax 输入不饱和。
- **来源定位**：Vaswani et al. 2017, §3.2.1 脚注 4（"We suspect that for large values of $d_k$, the dot products grow large in magnitude... To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Then the dot product $q\cdot k=\sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$."）。
- **适用条件**：(1) $q,k$ 分量近似独立；(2) 均值 0、方差 1（BN/LN 后基本成立）；(3) $d_k$ 足够大使点积方差显著大于 1。
- **置信状态**：已确认。论文同时给出实验对照：未缩放点积注意力在大 $d_k$ 下比加性注意力差（§3.2.1 第 1 段引用未缩放点积与加性注意力的对比结果）。

### C4 不缩放时 softmax 饱和导致梯度消失

- **论断内容**：softmax 雅可比 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$；当某个 $p_j\to 1$ 而其它 $p_i\to 0$ 时，所有非对角项几乎为 0，对角项 $p_i(1-p_i)\to 0$，梯度近零，训练停滞。
- **来源定位**：softmax 雅可比是标准结果；Vaswani et al. 2017 §3.2.1 脚注 4 描述的"pushing the softmax function into regions where it has extremely small gradients"。论文未给出雅可比显式形式，但脚注结论一致。
- **适用条件**：softmax 输入 logit 量级远大于 1（如 $d_k=64$ 时未缩放 logit 标准差 8）。
- **置信状态**：已确认。雅可比为标准推导；论文脚注定性描述一致。

### C5 多头公式与拼接机制

- **论断内容**：$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$，$\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$；$W_i^Q\in\mathbb{R}^{d_{model}\times d_k}$、$W_i^K\in\mathbb{R}^{d_{model}\times d_k}$、$W_i^V\in\mathbb{R}^{d_{model}\times d_v}$、$W^O\in\mathbb{R}^{hd_v\times d_{model}}$。论文用 $d_k=d_v=d_{model}/h$，使总参数量与单头 $d_{model}$ 维等价。
- **来源定位**：Vaswani et al. 2017, §3.2.2, Eq.(2)，论文第 5 页。论文设置 $d_{model}=512$、$h=8$、$d_k=d_v=64$、$W^O\in\mathbb{R}^{512\times 512}$。
- **适用条件**：$d_k=d_v=d_{model}/h$（论文默认设置）；否则总参数量改变。
- **置信状态**：已确认。

### C6 多头子空间分化（设计目标）

- **论断内容**：多头允许模型在不同表示子空间中关注不同位置，例如一个头关注语法关系、另一个头关注共指关系；这是设计目标。论文未保证每个头一定学到可解释的不同关系，仅作设计动机陈述。
- **来源定位**：Vaswani et al. 2017, §3.2.2 第 2 段（"Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."）。
- **适用条件**：训练充分时；不可解释的头冗余已被后续研究（如 Michel et al. 2019 "Are Sixteen Heads Really Better than One?"）报道，但本页只引论文设计目标。
- **置信状态**：已确认（设计目标）；多头冗余的实验结果不在本页核心论断内。

### C7 因果遮罩形式

- **论断内容**：自回归解码时把未来位置的注意力分数设为 $-\infty$（实现中用 $-\infty$ 或足够大的负数），softmax 后权重为 0，防止看到未来 token。形式上 $M$ 为上三角矩阵，对角线及以下为 0、以上为 $-\infty$，$\text{MaskedAttention}(Q,K,V)=\text{softmax}((QK^\top+M)/\sqrt{d_k})V$。
- **来源定位**：Vaswani et al. 2017, §3.2.3 第 2 段（"We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all entries in the input of the softmax which correspond to illegal connections."）。
- **适用条件**：自回归解码；编码器双向注意力不需要遮罩。
- **置信状态**：已确认。

### C8 复杂度与瓶颈

- **论断内容**：单层自注意力时间复杂度 $O(n^2 d_k+n^2 d_v+n\cdot d_k\cdot d_{model})$，主导项 $O(n^2 d_k)$；空间复杂度 $O(n^2)$（注意力矩阵 $n\times n$）。这是 Linear Attention、Flash Attention 等变体要解决的瓶颈。
- **来源定位**：Vaswani et al. 2017, §4.1 Table 1（复杂度与顺序操作数、最大路径长度对比）。
- **适用条件**：标准实现（非分块 IO 优化）。
- **置信状态**：已确认。

### C9 注意力本身位置无关

- **论断内容**：缩放点积注意力对输入序列的顺序无感知——把 $Q,K,V$ 的行重排后输出也对应重排，不改变每行的相对注意力分布。位置信息必须由外接的位置编码（如正弦编码、RoPE）注入。
- **来源定位**：Vaswani et al. 2017, §3.5（位置编码章节）暗示注意力本身无位置感知；论文通过加位置编码说明这一点。
- **适用条件**：标准注意力；带相对位置偏置的变体（如 T5 bias、ALiBi）改变这一性质。
- **置信状态**：已确认。

## F 公式

### F1 缩放点积注意力

- **公式**：$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$
- **来源定位**：Vaswani et al. 2017, §3.2.1, Eq.(1)。
- **适用条件**：$Q\in\mathbb{R}^{n\times d_k}$、$K\in\mathbb{R}^{m\times d_k}$、$V\in\mathbb{R}^{m\times d_v}$；softmax 沿 $m$ 维（每行）归一化。
- **置信状态**：已确认。

### F2 多头注意力

- **公式**：$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$，$\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$
- **来源定位**：Vaswani et al. 2017, §3.2.2, Eq.(2)。
- **适用条件**：$W_i^Q,W_i^K\in\mathbb{R}^{d_{model}\times d_k}$、$W_i^V\in\mathbb{R}^{d_{model}\times d_v}$、$W^O\in\mathbb{R}^{hd_v\times d_{model}}$；论文 $d_k=d_v=d_{model}/h$。
- **置信状态**：已确认。

### F3 缩放因子的方差推导

- **公式**：若 $q_i,k_i\overset{iid}{\sim}$ 均值 0、方差 1，则 $\text{Var}(q\cdot k)=\text{Var}\!\left(\sum_{i=1}^{d_k}q_i k_i\right)=\sum_{i=1}^{d_k}\text{Var}(q_i k_i)=\sum_{i=1}^{d_k}1=d_k$；故 $\text{std}(q\cdot k)=\sqrt{d_k}$，除之得标准差 1。
- **来源定位**：Vaswani et al. 2017, §3.2.1 脚注 4。
- **适用条件**：$q_i,k_i$ 独立同分布、均值 0、方差 1（推导用到独立性和方差可加性）。
- **置信状态**：已确认。脚注只写结论"$\text{Var}(q\cdot k)=d_k$"，本页展开为完整推导（独立性和方差可加性）以服务教学。

### F4 softmax 雅可比

- **公式**：$p_i=e^{z_i}/\sum_l e^{z_l}$，$\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$
- **来源定位**：softmax 雅可比是标准结果，可由商法则直接推出。论文未显式给出，本页用于支撑 C4（梯度消失机制）。
- **适用条件**：softmax 任意输入。
- **置信状态**：已确认（标准推导）。

### F5 因果遮罩注意力

- **公式**：$\text{MaskedAttention}(Q,K,V)=\text{softmax}\!\left(\dfrac{QK^\top+M}{\sqrt{d_k}}\right)V$，$M_{ij}=0$ 当 $j\le i$、$M_{ij}=-\infty$ 当 $j>i$。
- **来源定位**：Vaswani et al. 2017, §3.2.3 第 2 段。
- **适用条件**：自回归解码器；编码器与编码-解码注意力不需要。
- **置信状态**：已确认。论文用文字描述遮罩，本页形式化为公式。

### F6 $Q,K,V$ 投影

- **公式**：自注意力下 $Q=XW^Q$、$K=XW^K$、$V=XW^V$，$X\in\mathbb{R}^{n\times d_{model}}$；$W^Q,W^K\in\mathbb{R}^{d_{model}\times d_k}$、$W^V\in\mathbb{R}^{d_{model}\times d_v}$。
- **来源定位**：Vaswani et al. 2017, §3.2.2 第 1 段（"the queries, keys and values come from the input through linear projections"）。
- **适用条件**：自注意力；交叉注意力下 $K,V$ 来自编码器输出而非同一 $X$。
- **置信状态**：已确认。

## N 数字

### N1 论文超参数

- **数字**：$d_{model}=512$、$h=8$、$d_k=d_v=64$、$W^O\in\mathbb{R}^{512\times 512}$（基础模型）；大模型 $d_{model}=1024$、$h=8$、$d_k=d_v=128$、$d_{ff}=4096$。
- **来源定位**：Vaswani et al. 2017, §3.2.2 与 Table 3。
- **适用条件**：原始 Transformer 基础/大模型设置。
- **置信状态**：已确认。本页正文使用基础模型数字作教学示例。

### N2 复杂度对比

- **数字**：自注意力每层复杂度 $O(n^2\cdot d)$、顺序操作 $O(1)$、最大路径长度 $O(1)$；RNN 复杂度 $O(n\cdot d^2)$、顺序操作 $O(n)$、最大路径长度 $O(n)$；卷积（kernel $k>1$）复杂度 $O(k\cdot n\cdot d^2)$、顺序操作 $O(1)$、最大路径长度 $O(\log_k n)$ 或 $O(n/k)$。当 $n<d$ 时自注意力比 RNN 计算量更小。
- **来源定位**：Vaswani et al. 2017, §4.1 Table 1。
- **适用条件**：标准实现。
- **置信状态**：已确认。本页只引用趋势与量级，不引用具体 BLEU 数字。

### N3 不缩放时 softmax 饱和的对照数字（教学示例）

- **数字**：教学示例（非论文数据）。设 $d_k=64$、$q,k$ 各分量 $N(0,1)$，则 $q\cdot k$ 标准差 $\sqrt{64}=8$；不缩放时 256 个 key 的最大注意力权重均值约 0.75（接近 one-hot），缩放后约 0.04（接近均匀 $1/256\approx 0.004$ 的温和峰值）。
- **来源定位**：教学构造，参考 jethroodeyemi.github.io 2026 的实测验证（非论文一手数据）。
- **适用条件**：随机初始化、$n=256$ key、$d_k=64$；教学用，不作为论文结论。
- **置信状态**：教学示例。本页正文只在折叠块使用，并明确标注为教学构造。

## 来源清单

| 编号 | 引用 | 用途 |
|---|---|---|
| [Vaswani2017] | Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762 | C1, C2, C3, C4, C5, C6, C7, C8, C9, F1, F2, F3, F5, F6, N1, N2 |
| [Bahdanau2014] | Bahdanau, D., Cho, K., Bengio, Y. "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015. arXiv:1409.0473 | 仅在 S5 缩放动机处提及加性注意力作对比，不作为核心论断依据 |
| [Luong2015] | Luong, M.-T., Pham, H., Manning, C. D. "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015 | 仅作未缩放点积注意力对照提及 |
| [Michel2019] | Michel, P., Levy, N., Neubig, G. "Are Sixteen Heads Really Better than One?" NeurIPS 2019. arXiv:1905.10650 | 仅在"多头冗余"边界处提及，不作为核心论断依据 |

无冲突论断。所有核心论断置信状态均为"已确认"，主要依据为 Vaswani et al. 2017 原论文与标准推导。
