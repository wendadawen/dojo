# glossary.md：KDA 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致。

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| Kimi Delta Attention（KDA） | 标题 | Kimi K3 的线性注意力变体，delta rule + channel-wise forget gate + lower-bounded decay + full-rank output gate |
| Kimi K3 | S1 | 月之暗面开源的前端大模型，93 层（69 KDA + 24 Gated MLA），1M 上下文 |
| Gated MLA | S1 | Multi-head Latent Attention 的门控版本，K3 中负责全局内容交互，与 KDA 3:1 交替 |
| DeltaNet | S1 | delta rule 注意力的原始形式（Schlag 2021 / Yang 2024），无显式遗忘门 |
| Kimi Linear [63] | S3 | KDA 的直接前身，用 negative-softplus decay + low-rank output gate |
| GDN / Mamba-2 | S3 | delta rule + negative-softplus 的更早出处 [24, 138] |
| softmax 注意力 | S1 | 标准 Transformer 注意力，$O(N^2)$、KV cache 随序列线性增长 |
| 线性注意力 | S1 | 用核函数 $\phi$ 把注意力分解，因果掩码下变成固定大小递归状态 |
| delta rule | S1 | "先擦后写"的递归更新，$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ |
| channel-wise forget gate | S2 | 每通道独立衰减率 $\alpha_t \in (0,1)^{d_k}$，作用于旧状态 |
| lower-bounded decay | S3 | K3 的 decay 参数化，scaled sigmoid 把 log-decay 限在 $(g_{\min}, 0)$ |
| negative-softplus | S3 | Kimi Linear 的 decay 参数化，$g = -e^A \mathrm{Softplus}(z) \in (-\infty, 0)$ |
| scaled sigmoid | S3 | K3 的 decay 映射，$g = g_{\min}\mathrm{Sigmoid}(e^A z) \in (g_{\min}, 0)$ |
| full-rank output gate | S4 | K3 的输出门控，$W_g$ 满秩，$y = W_o[\mathrm{Sigmoid}(W_g x) \odot \mathrm{RMSNorm}(\tilde o)]$ |
| low-rank output gate | S4 | Kimi Linear 的输出门控，$W_g$ 低秩 |
| chunkwise 并行形式 | S5 | chunk 内并行 matmul + chunk 间递归传 $S$ 的执行形式 |
| UT 变换 | S5 | Kimi Linear [63] 的变换，产出 $U, W$，构造 $V_e = U - WS$ |
| ShortConv | S4 | 短卷积（kernel=4），在 q/k/v 投影前做局部时序卷积 |
| L2Norm | S4 | L2 归一化，让 q/k 近单位范数 |
| Swish | S4 | 激活函数 $\mathrm{Swish}(x) = x \cdot \mathrm{Sigmoid}(x)$ |
| RMSNorm | S4 | Root Mean Square Normalization，在门控前稳定 $\tilde o$ 幅度 |
| Sigmoid | S2 | $\sigma(z) = 1/(1+e^{-z})$ |
| Softplus | S3 | $\mathrm{Softplus}(z) = \ln(1+e^z)$ |
| Tensor Core | S3 | GPU 上的 dense matmul 加速单元 |
| BF16 | S3 | Brain Float 16，16-bit 浮点格式，动态范围约 $[-3.4\times10^{38}, 3.4\times10^{38}]$ |
| position-pair diagonal | S3 | Kimi Linear 的对角 tile 实现路径，逐位置对计算，非 dense matmul |
| FlashKDA | S6 | K3 的 CUTLASS-based chunkwise kernel（§5.1.1），本文不展开 |
| KCP（KDA Context Parallelism） | S6 | K3 的跨设备上下文并行（§5.1.2），本文不展开 |
| KV cache | S1 | softmax 注意力缓存的历史 K、V，随序列长度线性增长 |
| key 碰撞 | S1 | vanilla 线性注意力中 $L > d$ 时不同 token 的 key 投影到同一方向 |
| 3:1 混合 | S1 | 每 4 层一组 = 3 KDA + 1 Gated MLA |
| secondary tile | S5 | chunk 内的 16-token 子划分（Kimi Linear [63, 140]） |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| KDA | Kimi Delta Attention | 标题 |
| MLA | Multi-head Latent Attention | S1 |
| Gated MLA | Gated Multi-head Latent Attention | S1 |
| KV | Key-Value | S1 |
| GDN | Gated Delta Network | S3 |
| KCP | KDA Context Parallelism | S6 |
| NoPE | No Position Encoding | S6（仅引用，不展开） |

## 符号

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $S_t \in \mathbb{R}^{d_k \times d_v}$ | 第 $t$ 步的递归状态矩阵（K3 约定：状态在右，$\tilde o = S^\top q$） | S2 |
| $S_{t-1}$ | 上一步的状态 | S2 |
| $q_t, k_t \in \mathbb{R}^{d_k}$ | 第 $t$ 步的 query、key 向量（单头） | S2 |
| $v_t \in \mathbb{R}^{d_v}$ | 第 $t$ 步的 value 向量 | S2 |
| $\beta_t \in (0, 1)$ | delta rule 写入强度，标量，$\mathrm{Sigmoid}(W_\beta x_t)$ | S2 |
| $\alpha_t \in (0, 1)^{d_k}$ | channel-wise 一步留存因子，逐通道 | S2 |
| $g_t \in (g_{\min}, 0)^{d_k}$ | log-decay，$\alpha_t = \exp(g_t)$ | S3 |
| $g_{\min} = -5$ | log-decay 的固定下界 | S3 |
| $A_h$ | per-head 可学习 log-scale，初始化 $A_h = 0$ | S3 |
| $z_t^h \in \mathbb{R}^{d_k}$ | decay logit，$z_t^h = W_\alpha^{\uparrow\downarrow} x_t + b_h^\alpha$ | S3 |
| $W_\alpha^{\uparrow\downarrow}$ | 上、下两个投影矩阵，产出 $z_t^h$ | S3 |
| $b_h^\alpha \in \mathbb{R}^{d_k}$ | per-head bias，与 $A_h$ 一起参数化 decay | S3 |
| $\tilde o_t \in \mathbb{R}^{d_v}$ | 递归输出，$\tilde o_t = S_t^\top q_t$ | S2 |
| $y_t$ | 单层最终输出，$y_t = W_o[\mathrm{Sigmoid}(W_g x_t) \odot \mathrm{RMSNorm}(\tilde o_t)]$ | S4 |
| $W_g$ | output gate 投影矩阵，K3 中满秩 | S4 |
| $W_o$ | output 投影矩阵 | S4 |
| $I \in \mathbb{R}^{d_k \times d_k}$ | 单位矩阵 | S2 |
| $\mathrm{Diag}(\alpha_t)$ | 把 $\alpha_t$ 放在对角线的对角矩阵 | S2 |
| $\odot$ | 逐元素乘（Hadamard 积） | S4 |
| $\Gamma_{1\to r}[t]$ | chunk $t$ 内位置 1 到 $r$ 的累积衰减，$\prod_{r'=1}^r \alpha_{r'}[t]$ | S5 |
| $\gamma_{i\to j}[t]$ | chunk $t$ 内位置 $i$ 到 $j$ 的累积衰减 | S5 |
| $C$ | chunk size | S5 |
| $Q[t], K[t], V[t]$ | chunk $t$ 内堆叠的 query/key/value 矩阵 | S5 |
| $O[t]$ | chunk $t$ 的输出矩阵 | S5 |
| $A[t]$ | chunk $t$ 的 intra-chunk attention 矩阵，$\mathrm{Tril}(\cdots)$ | S5 |
| $U[t], W[t]$ | UT 变换的产物，$V_e = U - WS$ | S5 |
| $V_e[t]$ | pseudo-value，$U[t] - W[t]S[t]$ | S5 |
| $\mathrm{Tril}(\cdot)$ | 保留对角及以下的下三角掩码 | S5 |
| $h$ | 上标，表示 per-head 量（$q_t^h, A_h, b_h^\alpha$ 等） | S2 |
| $d_k$ | key 维度（K3 中 = 128） | S2 |
| $d_v$ | value 维度（K3 中 = 128） | S2 |
| $d$ | 模型 hidden_size（K3 中 = 7168） | S4 |
| $x_t \in \mathbb{R}^d$ | 第 $t$ 个 token 的 hidden state | S4 |

## 约定说明

- **状态布局**：K3 报告 KDA 用 $S \in \mathbb{R}^{d_k \times d_v}$（状态在右乘，$\tilde o = S^\top q$）；delta-rule 概念页用 DeltaNet 形式 $S \in \mathbb{R}^{d_v \times d_k}$（状态在左乘，$V'_i = \phi(Q_i)^\top s_i$）。两者是 $S_{\text{K3}} = S_{\text{DeltaNet}}^\top$ 的转置约定。本文统一按 K3 报告约定。
- **多头**：公式默认单头，多头时加 $h$ 上标并按 head 独立参数化 $A_h, b_h^\alpha$。
