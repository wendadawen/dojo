# Stable LatentMoE 术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| MoE | S1 | Mixture-of-Experts，混合专家层：路由器为每个 token 选若干专家，只有被选中的专家参与计算。本文首次出现时链接 `wiki/moe-serving/index.html`。 |
| 路由专家 / routed expert | S1 | 参与 Top-k 路由选择的专家；每个 token 只激活其中 $k$ 个。 |
| 共享专家 / shared expert | S1 | 不参与路由、所有 token 必经的专家；K3 每层固定 $N_s=2$ 个。 |
| Top-k 路由 | S1 | 路由器为每个 token 选 $k$ 个路由专家；K3 取 $k=16$。 |
| 稀疏度 | S1 | 路由专家总数 ÷ 每 token 激活数 = $N/k$；K3 为 $896/16=56$。 |
| 全模型宽度 $d$ | S1 | 整个 transformer 模型的隐藏维度；K3 $d=7168$。 |
| 隐空间宽度 $\ell$ | S1 | 路由专家操作的紧凑空间的维度；K3 $\ell=3584$（`routed_expert_hidden_size`）。 |
| $W_\downarrow$ | S1 | 下投影矩阵，把 $x \in \mathbb{R}^d$ 投到 $z \in \mathbb{R}^\ell$。 |
| $W_\uparrow$ | S2 | 上投影矩阵，把归一化后的 $u \in \mathbb{R}^\ell$ 投回 $\mathbb{R}^d$。 |
| $z$ | S2 | 路由分支的隐空间表示，$z = W_\downarrow x \in \mathbb{R}^\ell$。 |
| $u$ | S2 | 路由专家的加权聚合，$u = \sum_{i \in T_k(x)} p_i E_i^{\text{routed}}(z) \in \mathbb{R}^\ell$。 |
| $x$ | S2 | MoE 层的输入，$x \in \mathbb{R}^d$。 |
| $y$ | S2 | MoE 层的输出，$y \in \mathbb{R}^d$。 |
| $E_j^{\text{shared}}$ | S2 | 第 $j$ 个共享专家的 FFN，输入输出都在 $\mathbb{R}^d$。 |
| $E_i^{\text{routed}}$ | S2 | 第 $i$ 个路由专家的 FFN，输入输出都在 $\mathbb{R}^\ell$。 |
| $p_i$ | S2 | 路由权重，由路由器给出，对 Top-k 集合内的 $i$ 非零。 |
| $T_k(x)$ | S2 | 对 token $x$ 选中的 Top-k 路由专家下标集合。 |
| $N_s$ | S2 | 共享专家数；K3 $N_s=2$。 |
| $N$ | S2 | 路由专家总数；K3 $N=896$。 |
| $k$ | S2 | 每 token 激活的路由专家数；K3 $k=16$。 |
| LatentMoE | S1 | K3 报告引用 [32] 的原始架构：分离全宽与路由宽度，路由专家在隐空间 $\ell$ 操作。 |
| Stable LatentMoE | S1 | 本文主题 = LatentMoE + 三件稳定化。 |
| DeepSeekMoE | S2 | 提出 shared+routed 组织的 MoE 架构；K3 沿用此组织。 |
| RMSNorm | S2 | Root Mean Square Normalization，按均方根归一化；在 Stable LatentMoE 中插在路由聚合后、上投影前。 |
| Normalized LatentMoE | S4 | Stable LatentMoE 的三件稳定化之一：在 $u$ 与 $W_\uparrow$ 之间插 RMSNorm。 |
| SiTU-GLU | S4 | Sigmoid Tanh Unit GLU，有界激活；上界 $\beta_1\beta_2$，K3 取 $\beta_1=4, \beta_2=25$，上界 100。完整机制见 `wiki/situ-glu/index.html`。 |
| Quantile Balancing / QB | S4 | 从路由分数的 $(1-k/n)$ 分位数设置专家 bias 的负载均衡方法。完整推导见 `wiki/quantile-balancing/`（占位）。 |
| 无辅助损失路由 / auxiliary-loss-free | S4 | DeepSeek-V3 引入：bias 加在 Top-k 选择分数上但不进入 mixture 权重；QB 在其基础上改进 bias 更新规则。 |
| 激活爆炸 | S3 | 路由分支连续矩阵乘法链在 2.8T 规模下产生的内部激活异常增大。 |
| 负载失衡 | S3 | 部分专家被过度使用、部分专家几乎不被使用；近千专家下原方法无法解决。 |
| bfloat16 | S6 | brain float 16，16 位浮点格式；K3 的训练精度。 |
| `latent_moe_use_norm` | S2 | K3 `config.json` 字段，控制是否启用 Normalized LatentMoE；取值 true。 |
| `routed_expert_hidden_size` | S2 | K3 `config.json` 字段，即 $\ell=3584$。 |
| `moe_intermediate_size` | S2 | K3 `config.json` 字段，路由专家 FFN 的中间维度 = 3072。 |
| `intermediate_size` | S2 | K3 `config.json` 字段，dense FFN / 共享专家 FFN 的中间维度 = 33792。 |
