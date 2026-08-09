# DeepSeekMoE 术语表

| 术语 / 缩写 / 符号 | 首次出现位置 | 定义或含义 |
|---|---|---|
| MoE | S1 | Mixture-of-Experts，专家混合。把一个 FFN 换成一排专家，每 token 只激活其中几个。前置概念页《MoE 大模型推理与服务基础》已讲。 |
| top-K MoE | S1 | 每 token 由 router 打分后选得分最高的 K 个专家计算的稀疏 MoE。 |
| router / 路由器 | S1 | 给每个 token 对所有专家打分的小模块，决定选哪 K 个。 |
| 专家专门化 | S1 | expert specialization：每个专家获得 non-overlapping and focused knowledge（互不重叠且聚焦的知识）。是评价 MoE 架构好坏的标尺。 |
| 知识混合 | S1 | knowledge hybridity：专家数量有限时，分配给某专家的 token 涵盖多种知识，专家被迫在参数中组装难以同时利用的不同知识。 |
| 知识冗余 | S1 | knowledge redundancy：不同专家收到的 token 需要共有知识时，多个专家各自独立学习同一份通用知识，导致参数重复。 |
| shared expert / 共享专家 | S3 | 对所有 token 恒激活、不参与 router 打分与 top-k 选择的专家，承载通用知识。DeepSeekMoE 中数量记为 K_s。 |
| routed expert / 路由专家 | S3 | 参与 router 打分与 top-k 选择的专家，只承载专门知识。 |
| 细粒度专家分割 | S2 | fine-grained expert segmentation：把每个专家 FFN 中间隐藏维度缩减为 1/m，得到 mN 个更小的专家，每 token 激活 mK 个。 |
| 共享专家隔离 | S3 | shared expert isolation：隔离 K_s 个专家作为共享专家，路由专家数与激活路由专家数各减 K_s。 |
| m | S2 | 细粒度分割的粒度参数。每个专家被切成 m 个小专家，每个小专家 FFN 中间隐藏维度为原来的 1/m。正整数。 |
| N | S2 | 细粒度分割前的专家总数（原 top-K MoE 的专家数）。 |
| K | S2 | 细粒度分割前每 token 激活的专家数（原 top-K 的 K）。 |
| mN | S2 | 细粒度分割后的专家总数。 |
| mK | S2 | 细粒度分割后每 token 激活的专家数。 |
| K_s | S3 | 共享专家数。恒激活，不参与路由。 |
| mN - K_s | S3 | 完整 DeepSeekMoE 的路由专家总数。 |
| mK - K_s | S3 | 完整 DeepSeekMoE 每 token 激活的路由专家数。 |
| FFN_i(·) | S2 | 第 i 个专家的前馈网络。细粒度分割后每个 FFN 的中间隐藏维度为原来的 1/m。 |
| u_t^l | S2 | 第 l 层中第 t 个 token 的输入向量（attention 的输出，FFN/MoE 的输入）。 |
| h_t^l | S2 | 第 l 层中第 t 个 token 的 MoE 层输出。 |
| g_{i,t} | S2 | 第 t 个 token 对第 i 个专家的门控值。稀疏：路由专家中只有被选中的 mK-K_s 个非零，共享专家无门控（恒为 1）。 |
| s_{i,t} | S2 | 第 t 个 token 与第 i 个专家的亲和度（softmax 归一化后的打分）。 |
| e_i^l | S2 | 第 l 层中第 i 个专家的中心点（用于计算亲和度）。 |
| TopK(·, k) | S2 | 从给定集合中选出得分最高的 k 个。 |
| 计算量守恒 | S2 | 三架构（标准 top-K、细粒度、完整 DeepSeekMoE）的专家参数总量与每 token 计算成本保持恒定。 |
| C(n,k) / 二项式系数 | S2 | 从 n 个里选 k 个的组合数，用于度量专家搭配的灵活性。 |
| GShard | S4 | 一种 top-K MoE 架构（Lepikhin et al. 2021），DeepSeekMoE 的对比基线。 |
| DeepSeek-V3 | S5 | DeepSeek 的后续大模型，继承了 DeepSeekMoE 的 shared+routed 组织，但路由打分改为 sigmoid + 偏置、均衡改为 aux-loss-free。 |
| Stable LatentMoE | S5 | K3 的 MoE 架构，继承了 DeepSeekMoE 的 shared+routed 组织，在其上做隐空间路由与稳定化。见概念页《Stable LatentMoE》。 |
| softmax 亲和度 | S2 | DeepSeekMoE 原始论文的路由打分方式：s_{i,t} = Softmax_i((u_t^l)^T e_i^l)。 |
| sigmoid 亲和度 + 偏置 | S5 | DeepSeek-V3 的路由打分方式（非本页范围，仅点明差异）。 |
