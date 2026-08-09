# 术语表：低秩分解

登记全文首次出现的术语、缩写和符号。保证全文含义一致。

## 术语

| 术语 | 首现位置 | 含义 |
|---|---|---|
| 低秩分解（Low-Rank Decomposition） | S1 | 把大矩阵 $W$ 写成两个小矩阵 $A, B$ 的乘积 $W \approx AB$，内维 $r \ll \min(m,n)$ |
| 低秩近似（Low-Rank Approximation） | S1 | 用秩更低的矩阵近似原矩阵，低秩分解是其实现形式 |
| 低秩投影（Low-Rank Projection） | S5 | 把高维向量经下投影压到低维子空间，MLA 的侧重 |
| 秩（rank） | S2 | 矩阵列（或行）张成空间的维度，即独立方向数，记 $\mathrm{rank}(W)$ |
| 满秩（full rank） | S2 | $\mathrm{rank}(W) = \min(m,n)$，矩阵无冗余方向 |
| 奇异值（singular value） | S3 | SVD 对角矩阵 $\Sigma$ 的对角元，衡量矩阵在各主方向的强度，按从大到小排序 |
| 奇异值分解（SVD, Singular Value Decomposition） | S3 | $W = U\Sigma V^\top$，把矩阵分解为正交矩阵、对角矩阵、正交矩阵的乘积 |
| 截断 SVD（Truncated SVD） | S3 | 只保留前 $k$ 大奇异值的 SVD 近似，给出 $W_k$ |
| Eckart-Young-Mirsky 定理 | S3 | 截断 SVD 给出的 $W_k$ 是所有秩 $\le k$ 矩阵中对 $W$ 的最优近似 |
| Frobenius 范数 | S3 | 矩阵所有元素平方和的平方根，记 $\|\cdot\|_F$ |
| 谱范数（2-范数） | S3 | 矩阵的最大奇异值，记 $\|\cdot\|_2$ |
| LoRA（Low-Rank Adaptation） | S1 钩子 | 冻结预训练权重 $W_0$，把更新量参数化为 $\Delta W = BA$ 的微调方法 |
| 权重更新（weight update / $\Delta W$） | S4 | 微调前后权重的差值 |
| KV cache | S5 | 推理时为每个 token 缓存的 K 和 V 向量，随序列长度线性增长 |
| MLA（Multi-Head Latent Attention） | S5 | DeepSeek-V2 的注意力机制，用低秩压缩减少 KV cache |
| 下投影（down-projection） | S5 | 把高维隐藏状态 $h_t$ 压成低维潜向量 $c_t^{KV}$ 的矩阵 $W^{DKV}$ |
| 上投影（up-projection） | S5 | 把潜向量 $c_t^{KV}$ 升回高维重建 K/V 的矩阵 $W^{UK}, W^{UV}$ |
| 潜向量（latent vector） | S5 | 下投影输出的低维向量 $c_t^{KV}$，MLA 只缓存它 |
| KDA（Kimi Delta Attention） | S6 | K3 的线性注意力，其门控经历 low-rank → full-rank 的转变 |
| 全量微调（full fine-tuning） | S4 | 训练所有参数的微调方式 |

## 符号

| 符号 | 含义 | 首现 |
|---|---|---|
| $W \in \mathbb{R}^{m \times n}$ | 被近似的大矩阵 | S1 |
| $A \in \mathbb{R}^{m \times r}$ | 分解的左因子（瘦矩阵） | S2 |
| $B \in \mathbb{R}^{r \times n}$ | 分解的右因子（矮矩阵） | S2 |
| $r$ | 分解的内维（低秩近似的秩上界），$r \ll \min(m,n)$ | S2 |
| $\mathrm{rank}(W)$ | 矩阵 $W$ 的秩 | S2 |
| $U, \Sigma, V^\top$ | SVD 的三个因子 | S3 |
| $\sigma_i$ | 第 $i$ 大奇异值 | S3 |
| $u_i, v_i$ | $U, V$ 的第 $i$ 列 | S3 |
| $W_k$ | 保留前 $k$ 大奇异值的截断 SVD 近似 | S3 |
| $\|\cdot\|_F$ | Frobenius 范数 | S3 |
| $\|\cdot\|_2$ | 谱范数（最大奇异值） | S3 |
| $W_0$ | 冻结的预训练权重（LoRA） | S4 |
| $\Delta W$ | 权重更新量（LoRA 中 $= BA$） | S4 |
| $d$ | LLM 的隐藏维度 | S4 |
| $h_t$ | 第 $t$ 个 token 的隐藏状态（MLA） | S5 |
| $c_t^{KV}$ | MLA 压缩后的 KV 潜向量 | S5 |
| $W^{DKV}$ | MLA 的下投影矩阵 | S5 |
| $W^{UK}, W^{UV}$ | MLA 的上投影矩阵 | S5 |
| $d_c$ | MLA 的 KV 压缩维度 | S5 |
| $n_h$ | 注意力头数 | S5 |
| $d_h$ | 每头维度 | S5 |

## 沿用记号说明

- LoRA 论文记 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$（$B$ 在左、$A$ 在右），本文沿用此记号；注意与 S3 的通用分解 $W \approx AB$ 中 $A$ 在左、$B$ 在右不同，LoRA 章节会显式说明对应关系。
- DeepSeek-V2 论文用上标 $C$ 标注"压缩后重建"的量（如 $k_t^C, v_t^C$），本文沿用。
