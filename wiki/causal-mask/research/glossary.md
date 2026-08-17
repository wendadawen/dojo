# 因果掩码（Causal Mask）术语表

登记全文首次出现的术语、缩写和符号。保证全文含义一致。

| 术语/缩写/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| 因果掩码（Causal Mask） | 页面开头 | decoder 自注意力中位置 $t$ 只能 attend 到位置 $1..t$、看不到未来 token 的遮挡规则；实现上把注意力分数矩阵严格上三角置 $-\infty$ |
| look-ahead mask / subsequent mask / 上三角掩码 | S1 别名 | 因果掩码的常见别名；本文统一使用"因果掩码" |
| 自回归生成（autoregressive generation） | S1 | 逐 token 预测下一个 token、每步只依赖已生成前缀的生成方式 |
| decoder / 解码器 | S1 | Transformer 中负责自回归生成的部分；其自注意力用因果掩码 |
| encoder / 编码器 | S1 对照表 | Transformer 中负责构建输入上下文表示的部分；其自注意力双向、不加因果掩码 |
| decoder-only | S5 | 仅含解码器、自回归预测下一个 token 的 Transformer 结构（如 GPT 系列）；所有自注意力层均用因果掩码 |
| 标准注意力（scaled dot-product attention） | S1 | $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$；因果掩码施加在其分数上，不改变基本结构（前置概念页 `wiki/standard-attention/`） |
| 注意力分数（attention scores） | S2 | $QK^\top/\sqrt{d_k}$ 的结果，$n\times n$ 矩阵；softmax 之前、掩码施加的对象 |
| query / key / value（$Q,K,V$） | S2 公式 | 注意力的三组矩阵；query 发起匹配、key 被匹配、value 被加权取出 |
| $d_k$ | S2 公式 | key 向量的维度；$\sqrt{d_k}$ 用于缩放点积稳定 softmax 数值 |
| $n$ | S2 | 序列长度；掩码矩阵 $M$ 为 $n\times n$ |
| 因果掩码矩阵 $M$ | S2 公式 | $n\times n$ 矩阵，$M_{ij}=-\infty$ 当 $j>i$（严格上三角，未来位置），$M_{ij}=0$ 当 $j\le i$（对角线及以下，可见位置） |
| $-\infty$（负无穷） | S2 | 掩码值；softmax 中 $e^{-\infty}=0$ 使被掩位置权重为 $0$ |
| softmax | S2 | 把分数归一化为权重（和为 $1$）的函数；归一化只对未掩（非 $-\infty$）项求和 |
| 重归一化 | S2 结论 | 掩码使未来项为 $0$ 后，softmax 分母只含可见项，可见项权重重新归一化到和为 $1$ |
| $QK^\top/\sqrt{d_k}+M$ | S2 公式 F1 | 带因果掩码的注意力分数；掩码在 softmax 输入（分数）上施加，不在 softmax 输出（权重）上 |
| $o_t$ | S3 公式 F2 | 位置 $t$ 的注意力输出；因果下 $o_t=\sum_{j=1}^{t}\alpha_{t,j}v_j$ |
| $\alpha_{t,j}$ | S3 公式 F2 | 位置 $t$ 对位置 $j$ 的注意力权重（softmax 归一化后） |
| $s_{t,j}$ | S3 公式 F2 | 位置 $t$ 对位置 $j$ 的注意力分数 $q_t^\top k_j/\sqrt{d_k}$ |
| $v_j$ | S3 | 位置 $j$ 的 value 向量；手算例子中取标量 $v_1=1,v_2=2,v_3=3$ |
| 构造示例 | S3 | 人为构造的数字（分数全 $0$、值 $1,2,3$），便于手算，不代表真实模型 |
| KV-cache | S4 | 缓存过去 token 的 key/value，新 query 直接复用、不重算的推理结构（概念页占位） |
| 顺序操作（sequential operation） | S4 | 完成一次前向所需的串行步数；RNN 为 $O(n)$，自注意力 + 因果掩码训练为 $O(1)$ |
| 排列等变 / 排列对称性 | S5 | 交换输入顺序时每个位置的输出不变的性质；双向注意力无位置编码时退化为排列等变，无法区分顺序（概念页占位） |
| 双向注意力（bidirectional attention） | S5 | 每个位置能看到所有位置的注意力；encoder 自注意力属此类，不加因果掩码 |
| 因果注意力（causal attention） | S5 | 每个位置只看到前缀的注意力；decoder 自注意力属此类，由因果掩码实现 |
| 填充掩码（padding mask） | S5 | 屏蔽填充 token（如 `<pad>`）的掩码；与因果掩码目的不同、可叠加 |
| NoPE（No Position Encoding） | S5 | 不施加任何显式位置编码的做法；以因果掩码为隐式编码位置的结构前提（概念页 `wiki/nope/`） |
| 显式位置编码 | S5 | 人工设计的、直接注入位置信息的机制（位置嵌入、距离偏置、旋转等）；与因果掩码正交 |
