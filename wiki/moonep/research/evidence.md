# MoonEP 完美均衡专家并行 · 核心论断与证据

来源：Kimi K3 Technical Report §5.2.1（约 1305–1354 行）与 §E 附录（约 2941–3022 行）。
文件位置：`/tmp/kimi-k3-research/k3-report.txt`。

## C 论断（概念与机制）

### C1：传统 EP 下 token 负载在 rank 间不均衡，且 routed-expert 激活形状动态变化导致内存碎片

- 来源定位：§5.2.1 开头，报告 1305–1308 行：「In conventional EP schemes, token loads are imbalanced across ranks. The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation.」
- 适用条件：传统 EP（如 DeepEP）下，router 输出决定 token-expert 分布，分布随 micro-batch 和层变化。
- 置信状态：已确认。

### C2：MoonEP 用动态冗余专家实现完美负载均衡，保留 DeepEP 等传统方案的总体计算流

- 来源定位：§5.2.1，报告 1308–1313 行：「MoonEP, an EP scheme that achieves perfect load balance with dynamic redundant experts. MoonEP preserves the overall computation flow of conventional schemes such as DeepEP and additionally introduces online planning and migration of redundant experts. In the forward pass, we plan the redundant experts from the router outputs of the current micro-batch and layer and prefetch them before the routed-expert computation. In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks.」
- 适用条件：训练场景；router 输出给定后做规划。
- 置信状态：已确认。

### C3：MoonEP 要求每 rank 恰好收到 $S\times K$ 个 token

- 来源定位：§5.2.1，报告 1315–1317 行：「MoonEP requires every rank to receive exactly $S\times K$ tokens, where $S$ is the sequence length and $K$ is the number of experts selected per token, so that all ranks perform identical amounts of computation.」
- 适用条件：完美均衡的目标。
- 置信状态：已确认。

### C4：MoonEP 的 forward 流程——planning kernel 从当前 micro-batch、当前层 router 输出规划冗余专家并预取

- 来源定位：§5.2.1，报告 1310–1312 行 + 1325–1327 行「Online planning」段：「Computing the exact optimum at every training step is prohibitively expensive. We therefore compute exact solutions offline with integer linear programming (ILP) for representative cases as references and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the $E/R$ upper bound.」
- 适用条件：训练每步；不每步求精确最优。
- 置信状态：已确认（GPU kernel 具体算法未公开，但存在性与近最优性由报告声明）。

### C5：MoonEP 的 backward 流程——冗余专家梯度暂存本地 reduce buffer，计算完成后再 reduce 回 home rank

- 来源定位：§5.2.1，报告 1312–1313 行：「In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks.」
- 适用条件：冗余专家有梯度需要归还。
- 置信状态：已确认。

### C6：MoonEP 实现了 fused permute/unpermute 算子，planning kernel 预计算每 token 目的地，直接发到远端 expert-grouped 位置，返回 buffer view 免中间拷贝

- 来源定位：§5.2.1，报告 1329–1332 行：「We implement a fused permute/unpermute operator in which the planning kernel precomputes the destination of every token, so tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies.」
- 适用条件：完美均衡下，buffer 形状静态已知。
- 置信状态：已确认。

### C7：完美均衡下 MoonEP 只需固定 $S\times K$ 通信 buffer；DeepEP 在最坏不均衡下支持同样 zero-copy 数据路径需要 $S\times K\times R$ buffer

- 来源定位：§5.2.1，报告 1332–1334 行：「Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size $S\times K\times R$, whereas MoonEP requires only a fixed $S\times K$ buffer owing to the perfect balance.」
- 适用条件：DeepEP 的 $S\times K\times R$ 是**最坏不均衡**下的 buffer 量；非最坏可更小。MoonEP 的 $S\times K$ 因完美均衡而恒定。
- 置信状态：已确认。

### C8：MoonEP 让每 rank 收 $S\times K$ token，所有层计算形状静态已知，消除每层 MoE host 同步、缓解 host-side kernel-launch overhead

- 来源定位：§5.2.1，报告 1336–1346 行：「In conventional MoE implementations, the per-expert token counts vary across steps and layers, and the host must synchronize with the device at every layer to obtain the actual computation shapes before launching the expert computation, stalling the pipeline between layers. With perfect balance, every rank receives exactly $S\times K$ tokens and the computation shapes of all layers are statically known. This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead.」
- 适用条件：完美均衡下；传统 EP 每层 token 数变化时需要 host-device 同步。
- 置信状态：已确认。

### C9：MoonEP 用 workload-aware scheduler 调度 routed-expert GEMM，shared expert 分到独立 stream 重叠

- 来源定位：§5.2.1，报告 1348–1354 行：「Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed, and a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers. We therefore schedule the routed-expert GEMM with a workload-aware scheduler that adapts its parameters to the current token distribution before launch and keeps them fixed during execution. A lightweight heuristic selects these parameters using an analytical cost model of hardware metrics, with key coefficients calibrated through offline autotuning. For the shared experts, we dispatch their GEMMs to a separate stream so that they overlap with other kernels.」
- 适用条件：rank 间总量已均衡后，rank 内 per-expert 仍可能偏斜。
- 置信状态：已确认（scheduler 具体参数与 autotuning 系数未公开）。

### C10：ECHO、UltraEP 预设冗余专家数或施加 per-rank token cap，训练可能因无解而中断，且 cap 需手动调参仍留残余不均衡

- 来源定位：§5.2.1，报告 1320–1323 行：「prior work such as ECHO [137] and UltraEP [132] presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists within the cap, and the cap itself requires manual tuning while still leaving residual imbalance.」
- 适用条件：与 MoonEP 对比的相邻方案。
- 置信状态：已确认（报告对 ECHO/UltraEP 的描述；ECHO/UltraEP 内部机制不在本页范围）。

## F 公式（核心公式与来源）

### F1：MoonEP 的规划目标——最小化任意 rank 上冗余专家数的最大值

- 公式：$M(I)=\min_P\max_r\{m_r(P)\}$
- 来源定位：§E 开头，报告 2942–2943 行：「Let $m_r(P)$ denote the number of redundant experts placed on rank $r$ under plan $P$. For a router output $I$, the planning objective is to minimize the maximum number of redundant experts on any rank, i.e., $M(I)=\min_P\max_r\{m_r(P)\}$.」
- 符号：$m_r(P)$ = 规划 $P$ 下 rank $r$ 上的冗余专家数；$I$ = 当前 router 输出；$P$ 取遍所有可行规划；$M(I)$ = 该 router 输出下的最优（最小化最大冗余数）。
- 适用条件：规划目标是 min-max；不要求每 rank 冗余数相等。
- 置信状态：已确认。

### F2：Theorem 1（一般上界）——$M(I)\le E/R$

- 公式：$M(I)=\min_P\max_r\{m_r(P)\}\le E/R$，对任意 router 输出 $I$ 成立
- 来源定位：§E Theorem 1，报告 2944–2962 行。关键引理：「there exists a plan $P^*$ such that every EP rank receives exactly the same number of tokens ($S\times K$), and the remote tokens of each rank come from only one other EP rank.」构造性证明：每次取一个 underloaded rank 与一个 overloaded rank，迁移 token 直到 underloaded rank 达到 $S\times K$；每次 fill 让一个 underloaded rank 平衡且此后不变，故至多 $R-1$ 次 fill 终止；每 rank 至多被 fill 一次，故其远端 token 全来自同一 rank。结论：「supposing all remote tokens of rank $r$ come from rank $s$; these tokens belong to at most $E/R$ local experts on rank $s$, hence $m_r(P^*)\le E/R$」。
- 符号：$E$ = 专家总数；$R$ = EP size；$P^*$ = 构造的规划。
- 适用条件：每个 home rank 持有 $E/R$ 个本地专家（专家均匀分片）。
- 置信状态：已确认（构造性证明完整复述于报告）。

### F3：Theorem 2（界的基本紧性）——存在 router 输出 $I^*$ 使 $M(I^*)=\lceil E(R-1)/R^2\rceil\approx E/R$

- 公式：$M(I^*)=\lceil E(R-1)/R^2\rceil\approx E/R$（大 $R$）
- 来源定位：§E Theorem 2，报告 3005–3022 行。最坏构造：「the experts on EP rank 0 receive no tokens, while all experts on the other $R-1$ ranks share all tokens evenly. Then all $S\times K\times R$ tokens are evenly divided among $E(R-1)/R$ experts, so each expert receives $SKR^2/(E(R-1))$ tokens. Under any plan $P$, rank 0 must receive $S\times K$ tokens, all of which are remote, and these tokens involve at least $\lceil E(R-1)/R^2\rceil$ distinct experts; taking the ceiling, rank 0 requires at least $\lceil E(R-1)/R^2\rceil$ redundant experts, hence $M(I^*)\ge\lceil E(R-1)/R^2\rceil$. Conversely, by constructing a plan with the filling procedure from the proof of Theorem 1 and migrating tokens expert-wise preferentially, the number of redundant experts on every rank can be kept within this value, so equality holds.」
- 符号：$I^*$ = 最坏 router 输出；$E(R-1)/R$ = 其他 $R-1$ 个 rank 上的专家总数（rank 0 上 $E/R$ 个专家未收到任何 token）；$SKR^2/(E(R-1))$ = 每个被使用的专家收到的 token 数（$S\times K\times R$ 总 token 均分给 $E(R-1)/R$ 个专家）。
- 适用条件：大 $R$ 时 $\lceil E(R-1)/R^2\rceil\approx E/R$；小 $R$ 时两者可能差一个 ceiling。
- 置信状态：已确认（构造与反向填充均给出）。

### F4：每 rank 收到 $S\times K$ token 的总量守恒

- 公式：$R$ 个 rank 各收 $S\times K$ token，总 token = $S\times K\times R$；每 rank 本地有 $S$ 个 token、每个选 $K$ 个专家，共 $S\times K$ 个 token-expert pair 要发出，$R$ 个 rank 共发出 $S\times K\times R$ 个 pair。
- 来源定位：由 C3（每 rank 收 $S\times K$）与 token-expert pair 总量守恒直接推出。
- 适用条件：每个 token 恰好选 $K$ 个专家；所有 token-expert pair 都要被计算。
- 置信状态：已确认（推导链：C3 + 守恒）。

## N 数字（外部数字与实验条件）

本页**不引用任何实验性能数字**。K3 报告 §5.2.1 与 §E 未给出 MoonEP 的吞吐 / 内存 / 通信量的实验数值（这些在 §5.2 的训练系统总览图中可能有，但未在 §5.2.1 文字中给出具体数）。本页只用报告声明的结构性结论（$E/R$、$S\times K$、$S\times K\times R$、$\lceil E(R-1)/R^2\rceil$），这些是机制定义的一部分，不是实验测量。

手算例子中的具体数字（$E=4, R=2, S=4, K=1$ 等）是教学构造，标为教学示例，不来自报告。
