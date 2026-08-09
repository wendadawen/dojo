# LatentMoE 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 术语

| 名称 | 首次出现 | 定义或含义 |
|---|---|---|
| MoE（Mixture-of-Experts，专家混合） | S1 | 一种层结构，把 FFN 换成一排专家，每 token 由 router 选 top-k 个计算后加权求和。完整机制见 `wiki/moe-serving/`。 |
| 标准 MoE / 普通 MoE | S1 | 路由专家直接在模型全宽 $d$ 维空间工作的 MoE，即 LatentMoE 的改造对象。 |
| LatentMoE（隐空间 MoE） | 标题 | 本页主题。在路由分支两端加 down/up-projection，把路由专家放进更窄的隐空间 $\ell$ 计算。 |
| Stable LatentMoE（稳定隐空间 MoE） | S5 | K3 的 LatentMoE 变体 = LatentMoE 主结构 + 三件稳定化（Normalized LatentMoE、SiTU-GLU、Quantile Balancing）。见 `wiki/stable-latent-moe/`。 |
| 专家（expert） | S1 | MoE 层里的小 FFN，分路由专家（参与 top-k）与共享专家（恒激活）两类。 |
| 路由专家（routed expert） | S2 | 参与 router top-k 选择的专家，在 LatentMoE 里工作在隐空间 $\ell$。记号 $E_i^{\text{routed}}$。 |
| 共享专家（shared expert） | S2 | 对所有 token 恒激活、不参与路由的专家，承载通用变换。在 LatentMoE 里仍工作在全宽 $d$。记号 $E_j^{\text{shared}}$。来自 DeepSeekMoE，见 `wiki/deepseek-moe/`。 |
| router（路由器） | S1 | 给所有路由专家打分、选出 top-k 的小模块。在 LatentMoE 里仍在 $d$ 维工作。 |
| top-k | S1 | 每 token 由 router 选出得分最高的 $k$ 个路由专家。 |
| 门控权重 / 路由权重（gating weight, routing weight） | S2 | router 给选中专家的归一化权重。本页记作 $p_i$（K3 记法）；前置页 `wiki/moe-serving/` 用 $g_i$，含义相同。 |
| FFN（前馈网络） | S1 | 专家的具体形式，由两个矩阵构成。 |
| dispatch / combine | S1 | MoE 分布式部署里把 token 激活送到选中专家所在卡（dispatch）、把专家输出送回原卡加权求和（combine）的两步。见 `wiki/moe-serving/` §3。 |
| all-to-all | S1 | dispatch/combine 常用的通信模式，每张卡同时与所有其他卡交换数据。 |
| EP（Expert Parallelism，专家并行） | S1 | 把专家分片到多张 GPU 的并行方式。见 `wiki/moe-serving/` §3。 |
| HBM（High Bandwidth Memory） | S1 | GPU 的高速显存，权重必须放进 HBM 才能算。 |
| down-projection（下投影） | S2 | LatentMoE 路由分支入口的线性投影 $W_\downarrow$，把 $x \in \mathbb{R}^d$ 压到 $z \in \mathbb{R}^\ell$。 |
| up-projection（上投影） | S2 | LatentMoE 路由分支出口的线性投影 $W_\uparrow$，把聚合结果 $u \in \mathbb{R}^\ell$ 投回 $\mathbb{R}^d$。 |
| 隐空间 / 潜在空间（latent space） | S2 | LatentMoE 路由专家工作的低维空间 $\mathbb{R}^\ell$，$\ell < d$。 |
| 压缩比（compression ratio） | S3 | $d/\ell$，衡量路由部分开销缩小的比例。 |
| reinvestment | S3 | 把 LatentMoE 节省的预算同时投入到更大的 $N$ 与更高的 $k$ 的策略。 |
| 非线性容量（nonlinear capacity） | S3 | 每 token 的非线性预算，论文定义为 top-k × 专家中间维度。设计原则 3 要求保持其不变。 |
| 组合数 / 组合多样性（combinatorial sparsity） | S3 | $\binom{N}{k}$，衡量 top-k 路由能组合出的专家搭配数。reinvestment 后显著增长。 |
| 特征秩（feature rank） | S4 | 任务相关的隐空间下限，$\ell$ 低于此下限会损害质量。 |
| RMSNorm | S5 | 按均方根归一化。K3 在 $W_\uparrow$ 前加 RMSNorm 以稳定路由聚合 $u$ 的尺度。属 Stable LatentMoE 的稳定化之一。 |
| SiTU-GLU | S5 | Sigmoid Tanh Unit GLU，K3 的有界激活函数。属 Stable LatentMoE 的稳定化之一。 |
| Quantile Balancing（分位点平衡） | S5 | K3 的负载均衡方法，替代固定步长 bias 更新。属 Stable LatentMoE 的稳定化之一。 |
| Nemotron-3 | S3、S4 | NVIDIA 的模型家族，Super/Ultra 版本采用 LatentMoE 架构。 |
| Kimi K3 / K3 | S5 | Moonshot 的模型，采用 Stable LatentMoE。 |

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $d$ | S1 | 模型隐层宽度（全宽）。K3 取 7168，Nemotron-3 Super 取 4096，Ultra 取 8192。 |
| $\ell$ | S2 | 路由隐空间宽度，$\ell < d$。K3 取 3584，Nemotron-3 Super 取 1024，Ultra 取 2048。 |
| $x$ | S1 | MoE 层的输入 token 向量，$x \in \mathbb{R}^d$。 |
| $y$ | S2 | MoE 层的输出向量，$y \in \mathbb{R}^d$。 |
| $z$ | S2 | 下投影后的隐空间表示，$z = W_\downarrow x \in \mathbb{R}^\ell$。 |
| $W_\downarrow$ | S2 | 下投影矩阵，$W_\downarrow \in \mathbb{R}^{\ell \times d}$。 |
| $W_\uparrow$ | S2 | 上投影矩阵，$W_\uparrow \in \mathbb{R}^{d \times \ell}$。 |
| $E_i^{\text{routed}}$ | S2 | 第 $i$ 个路由专家，$E_i^{\text{routed}}: \mathbb{R}^\ell \to \mathbb{R}^\ell$。 |
| $E_j^{\text{shared}}$ | S2 | 第 $j$ 个共享专家，$E_j^{\text{shared}}: \mathbb{R}^d \to \mathbb{R}^d$。 |
| $N$ | S1 | 路由专家总数。 |
| $k$ | S1 | 每 token 激活的路由专家数（top-k）。 |
| $N_s$ | S2 | 共享专家数。 |
| $T_k(x)$ | S2 | 对 token $x$ 选中的 top-k 路由专家下标集合。 |
| $p_i$ | S2 | 路由权重，由 router 给出，对 $T_k(x)$ 内的 $i$ 非零。 |
| $g_i$ | S1 | 前置页 `wiki/moe-serving/` 用的门控权重记号，与 $p_i$ 含义相同。本页首次提到标准 MoE 时用 $g_i$ 以与前置页一致，引入 LatentMoE 公式后改用 $p_i$ 以与 K3 Eq. 11 一致。 |
| $u$ | S2 | 路由专家的加权聚合，$u = \sum_{i \in T_k(x)} p_i E_i^{\text{routed}}(W_\downarrow x) \in \mathbb{R}^\ell$。 |
| $\alpha$ | S3 | reinvestment 因子，通常取 $\alpha = d/\ell$，把 $N \to \alpha N$、$k \to \alpha k$。 |
| $S_k(x)$ | S1 | 前置页用的 top-k 集合记号，与本页 $T_k(x)$ 含义相同。本页在引用前置页公式时用 $S_k(x)$，引入 LatentMoE 公式后用 $T_k(x)$ 以与 K3 一致。 |

## 记号一致性说明

- 门控权重：前置页用 $g_i$，K3 用 $p_i$。本页在 S1 引用标准 MoE 公式时用 $g_i$（与前置页一致），S2 引入 LatentMoE 公式后改用 $p_i$（与 K3 Eq. 11 一致），并在首次切换时点明两者等价。
- top-k 集合：前置页用 $S_k(x)$，K3 用 $T_k(x)$。同样在 S1 用 $S_k(x)$，S2 起改用 $T_k(x)$，并点明等价。
