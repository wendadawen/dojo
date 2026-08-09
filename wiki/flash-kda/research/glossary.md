# glossary.md：FlashKDA 与 KDA Context Parallelism 术语表

登记全文所有首次出现的术语、缩写和符号：名称、首次出现位置、定义或含义。保证全文含义一致。

## 术语与缩写

| 术语 | 首次出现 | 定义/含义 |
|---|---|---|
| KDA（Kimi Delta Attention） | S1 | K3 的线性注意力变体，递归状态 $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$。完整机制见 KDA 概念页。 |
| 递归状态 / recurrent state | S1 | KDA 的 $S_t \in \mathbb{R}^{d_k \times d_v}$，固定大小，随 token 更新。 |
| 串行依赖 / serial dependence | S1 | $S_t$ 依赖 $S_{t-1}$ 的性质，与 GPU 并行偏好冲突。 |
| regime（执行 regime） | S1 | KDA 运行的四类场景：训练/prefill、长 prefill、跨设备、解码。 |
| chunkwise 形式 | S1 | chunk 内并行 + chunk 间串行传 $S$ 的形式。详见 KDA 页。 |
| intra-chunk 计算 | S2 | chunk 内的位置-位置 attention，chunk 内并行 matmul。 |
| cross-chunk 状态传播 | S2 | chunk 间把 $S[t]$ 传给 $S[t+1]$ 的递归步骤。 |
| FlashKDA | S2 | K3 的 CUTLASS-based chunkwise kernel，重叠 intra-chunk 计算与 cross-chunk 状态传播。 |
| token-parallel stages | S2 | FlashKDA 分解的 chunk 内 token 维度并行阶段。 |
| head-parallel recurrence | S2 | FlashKDA 分解的 head 维度递归阶段。 |
| CUTLASS | S2 | NVIDIA 的 CUDA 模板库，用于构建高性能 GEMM/kernel。本页只点名，不展开。 |
| flash-linear-attention | S2 | 线性注意力 kernel 框架，FlashKDA 作为其后端自动分发。本页只点名。 |
| TP（Tensor Parallelism） | S3 | 把 attention head 切到不同设备的并行方式；不缩短序列方向递归。 |
| 设备内 CP / intra-device context parallelism | S3 | SM 级 CP planner 把序列切到单 rank 的 SMs，无跨设备通信。 |
| SM 级 CP planner | S3 | 自动把序列分配到单 rank 多块 SM 的调度器。 |
| 段转移 / segment transition | S3 | 一段 token 对状态的作用效果，可独立于入状态计算后合并。 |
| KCP（KDA Context Parallelism） | S4 | 跨 device rank 的 KDA context parallelism，用 $M+\tilde S$ 分解 + prefix scan + 一次 all-gather。 |
| vanilla 线性注意力 | S4 | 无 delta rule 擦除、无 forget gate 的线性注意力，递归纯加性 $s_i = s_{i-1} + \phi(k_i)v_i^\top$。 |
| 加性递归 / additive recurrence | S4 | vanilla 线性注意力的递归形式，无 $M_t$ 作用于入状态。 |
| 累积转移 $M_{t\leftarrow 1}$ | S4 | 一段内本地 token 的 $M$ 连乘 $\prod M_r$，作用于入状态。 |
| $\tilde S$（从零生成的状态） | S4 | 同一递归从 $S=0$ 起跑出的状态，与入状态无关。 |
| prefix scan | S4 | 用各 rank 的 $(M累积, \tilde S)$ 片段从 $S=0$ 起依次重组入状态的并行算法。 |
| all-gather | S4 | 所有 rank 交换各自片段的集合通信原语；KCP 只需一次。 |
| softmax CP（Ring Attention 等） | S4 | softmax 注意力的跨设备并行，要交换随序列长度增长的 KV block。仅对照用。 |
| 解码 / decoding | S5 | 自回归逐 token 生成，每步原地更新状态。 |
| MTP（multi-token prediction）投机解码 | S5 | 一次猜测多个 token 后验证接受/拒绝的解码策略。本页最小解释。 |
| draft token | S5 | MTP 猜测的待验证 token。 |
| verify / 验证 | S5 | 检查 draft token 是否接受。 |
| accept / 接受 | S5 | draft token 通过验证，保留。 |
| reject / 拒绝 | S5 | draft token 未通过验证，丢弃。 |
| bonus token | S5 | 验证后额外生成的确认 token。 |
| 状态快照 / state snapshot | S5 | 为每个 draft 位置存的状态副本，支持回滚但流量大。 |
| 投影输入 / projected inputs | S5 | draft token 经投影后的 $q,k,v,\beta,\alpha$ 输入，比状态小。 |
| 片上重建 / on-chip rebuild | S5 | 用缓存的投影输入在 SM 片上重算接受 token 的状态。 |
| ReplaySSM | S5 | 并发提出相同投影输入缓存方案的工作。仅点名。 |
| PD 分离 / prefill–decode disaggregation | S5 | prefill 和 decode 分到不同设备的部署方式。投影缓存不改变其载荷。 |
| 融合 kernel | S5 | 覆盖短卷积、输入归一化、门控、KDA 递归、输出归一化的单一 kernel。 |

## 符号

| 符号 | 首次出现 | 定义/含义 |
|---|---|---|
| $S_t$ | S1 | 第 $t$ 步的递归状态，$S_t \in \mathbb{R}^{d_k \times d_v}$。 |
| $M_t$ | S1 | KDA 的 token-dependent 转移矩阵，$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$。 |
| $\beta_t$ | S1 | delta rule 写入强度，标量 $\in (0,1)$。 |
| $k_t, v_t, q_t$ | S1 | 单步的 key/value/query。 |
| $\alpha_t$ | S1 | channel-wise 一步留存因子，$\in (0,1)^{d_k}$。 |
| $d_k, d_v$ | S1 | key/value 维度，K3 配置 $d_k=d_v=128$。 |
| $S[t]$ | S2 | 进入 chunk $t$ 的状态（chunkwise 形式）。 |
| $T_i$ | S4 | rank $i$ 处理的 token 数。 |
| $S_T^{[i]}$ | S4 | 离开 rank $i$ 进入 rank $i+1$ 的状态。 |
| $\tilde S_t^{[i+1]}$ | S4 | rank $i+1$ 内从 $S=0$ 起经 $t$ 个本地 token 的状态。 |
| $M_{t\leftarrow 1}^{[i+1]}$ | S4 | rank $i+1$ 内前 $t$ 个本地 token 的 $M$ 连乘 $\prod_{r\leftarrow 1}^{t} M_r$。 |
| $P$ | S4 | context-parallel rank 总数。 |

## 约定

- 状态约定与 KDA 页一致：$S \in \mathbb{R}^{d_k \times d_v}$（状态在右，$\tilde o = S^\top q$）。
- $M_t$ 的定义与 K3 报告 §5.1.2 一致：$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$。注意 KDA 页 Eq.1 写作 $(I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)$，两者相同——$M_t$ 是这个整体。
- "rank" 在 S4 指设备 rank（跨设备）；"SM" 在 S3 指 SM 块（设备内）。
