# 位置编码基础 · 术语表

登记全文所有首次出现的术语、缩写和符号：名称、首次出现位置、定义或含义。保证全文含义一致。

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| 位置编码（Positional Encoding / Positional Embedding） | 页面开头 | 给不含位置信息的 Transformer 自注意力补充位置信号的一类方法的总称 |
| 自注意力（Self-Attention） | S1 | 用 query 和 key 做匹配、按匹配程度对 value 加权求和的机制；详见 wiki/standard-attention |
| 排列等变（Permutation-equivariant） | S1 | 交换输入 token 顺序后，注意力输出跟着挪位置但每个位置输出值不变的性质；详见 wiki/nope |
| 绝对位置编码（Absolute Positional Encoding） | S2 | 位置向量与绝对位置 pos 绑定、加在 token 嵌入上的方案 |
| 绝对正弦位置编码（Sinusoidal Positional Encoding） | S2 | Vaswani 2017 §3.5 用 sin/cos 公式生成的绝对位置编码 |
| 可学习绝对位置编码（Learned Absolute Positional Embedding） | S3 | 用 L×d_model 可学习参数表代替固定 sin/cos 的绝对方案 |
| 相对位置编码（Relative Positional Encoding） | S4 | 偏置与两个位置的相对距离 i−j 绑定、加在注意力分数上的方案 |
| T5 相对位置偏置（T5 Relative Position Bias） | S4 | Raffel 2019 提出的分桶相对偏置，加在注意力分数 softmax 之前 |
| 分桶（Bucketing） | S4 | 把连续的相对距离映射到有限个离散桶的函数，每桶一个可学习标量 |
| ALiBi（Attention with Linear Biases） | S4 | 相对位置编码的另一种实现，用与距离成正比的线性负偏置；本页不展开 |
| 旋转位置编码（Rotary Position Embedding, RoPE） | S5 | 按绝对位置旋转 Q/K 使内积只依赖相对位置；详见 wiki/rope |
| NoPE（No Positional Encoding） | S5 | 不施加任何显式位置编码，依赖因果掩码隐式提供位置；详见 wiki/nope |
| 因果掩码（Causal Mask） | S5 | decoder-only 注意力中阻止看到未来 token 的上三角 mask；NoPE 位置信息的来源 |
| KDA | S5 | Kimi K3 中的递归注意力变体，通过递归门控与衰减隐式提供位置信息；详见 wiki/kimi-k3 |
| MLA（Multi-head Latent Attention） | S5 | 通过压缩潜向量减少 KV cache 的注意力变体；矩阵吸收是其推理优化关键；详见 wiki/mla |
| 矩阵吸收 | S5 | MLA 把投影矩阵从 key 侧搬到 query 侧避免显式重建 K 的优化；与 RoPE 冲突 |
| FlashAttention | S5 | 不改公式只改 GPU IO 实现的注意力加速方案；与需物化 n×n 的相对 bias 不兼容 |
| 二进制计数器（类比） | S2 | 教学解释，把多频率 sin/cos 类比成二进制各位翻转速率不同 |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| PE | Positional Encoding / Positional Embedding | S2 |
| APE | Absolute Positional Encoding | S5 对比表 |
| RoPE | Rotary Position Embedding | S5 |
| NoPE | No Positional Encoding | S5 |

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $pos$ | S2 F1 | token 在序列中的位置索引，非负整数 |
| $i$ | S2 F1 | 维度对索引，取 $0, 1, \ldots, d_{model}/2-1$；第 $i$ 对覆盖维度 $2i$（sin）与 $2i+1$（cos） |
| $d_{model}$ | S2 F1 | token 嵌入维度（也是位置向量维度）；Vaswani 2017 base 模型为 512 |
| $10000$ | S2 F1 | base 常数，控制频率几何递减的速率；论文取 10000 |
| $\omega_i$ | S2 F3 | 第 $i$ 对的角频率，$\omega_i = 1/10000^{2i/d_{model}}$ |
| $PE_{(pos, 2i)}$ | S2 F1 | 位置 pos 的位置向量在维度 $2i$ 上的分量（sin 项） |
| $PE_{(pos, 2i+1)}$ | S2 F1 | 位置 pos 的位置向量在维度 $2i+1$ 上的分量（cos 项） |
| $PE_m$ | S2 F2 | 位置 $m$ 的完整位置向量 |
| $x_m$ | S2 F2 | 位置 $m$ 的 token 嵌入 |
| $x'_m$ | S2 F2 | 注入位置信息后的嵌入，$x'_m = x_m + PE_m$ |
| $k$ | S2 F3 | 固定位置偏移量，用于线性性质 $PE(pos+k)$ |
| $L$ | S3 F5 | 可学习方案的最大序列长度，参数表为 $L \times d_{model}$ |
| $i, j$ | S4 F4 | 注意力中 query 位置与 key 位置；相对距离为 $i-j$ |
| $b_{\text{bucket}(i-j)}$ | S4 F4 | T5 中与相对距离分桶绑定的可学习标量偏置 |
| $q_i, k_j$ | S4 F4 | 位置 $i$ 的 query 向量、位置 $j$ 的 key 向量 |
| $\text{score}(i,j)$ | S4 F4 | 位置 $i$ 对位置 $j$ 的注意力分数（softmax 之前） |
| $m, n$ | S5 | RoPE 中 query 与 key 的绝对位置；详见 wiki/rope |

## 符号一致性约定

- $pos$ 与 $m$：绝对正弦方案用 $pos$（Vaswani 原文记法）；RoPE 用 $m$、$n$（Su 2021 原文记法）。两者都指绝对位置，只在各自方案内沿用原文符号，避免混淆。
- $i$：在正弦公式中是维度对索引；在 T5 bias 中是 query 位置。两者在不同章节，首次出现时已明确界定，不跨章复用同一含义。
- 频率与波长：$\omega_i$ 是角频率，波长 $\lambda_i = 2\pi/\omega_i$。
