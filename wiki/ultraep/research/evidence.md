# UltraEP 核心论断与证据

论文固定版本：arXiv:2606.04101v3（2026-06-18），TeX 源码包 `sections/*.tex`。
章节号按 v3 编译顺序：§1 Introduction，§2 Background，§3 Expert Load Analysis，§4 UltraEP System Design，§5 Quota-Driven Planning，§6 RSN-Native Balancing Communication，§7 Implementation，§8 Evaluation，§9 Related Work，§10 Conclusion。

## C 论断

| 编号 | 论断 | 来源定位 | 适用条件 | 置信 |
|---|---|---|---|---|
| C1 | 大 EP 把设备级专家负载不均衡放大成计算 straggler、token all-to-all 瓶颈与活跃内存尖峰，且随 EP degree 增大而复合 | §1 第 2 段：“This imbalance manifests in expert computation stragglers, token all-to-all bottlenecks, and activation memory spikes on overloaded devices. As the EP degree increases, these effects compound” | 大 EP（32/64-way 量级） | 已确认 |
| C2 | EPLB 用冗余专家策略，把高负载专家启发式地复制到多个设备；常见部署用近期路由历史周期性重均衡以摊薄开销 | §1 第 3 段：“EPLB, a widely used balancer, adopts a redundant expert strategy, heuristically replicating high-load experts on multiple devices… common deployments use recent routing history and rebalance periodically to amortize the planning and rearrangement overhead” | EPLB 本身与负载估计器无关，周期性是部署选择 | 已确认 |
| C3 | serving prefill 中专家热度随语义域（science / coding / mixed）剧烈变化；同一域内热专家也逐批漂移，mixed 输入叠加多种路由模式使不均衡更难预测 | §3「Serving Prefill」：“expert popularity varies sharply across semantic transitions, including science, coding, and mixed-domain traffic. Even within a single domain, the hot experts drift from one batch to the next” | Qwen3-235B，128 专家取 top-8，EP=64（Figure 4 caption） | 已确认 |
| C4 | 训练中辅助损失的负反馈持续重调专家利用率；DeepSeek 式路由补偿也不能消除振荡，microbatch 间还有采样随机性带来的抖动 | §3「Training」：“auxiliary-loss negative feedback continually re-adjusts expert utilization; even DeepSeek-style router compensation that proactively equalizes experts does not eliminate the oscillation. Inter-microbatch jitter from sampling randomness also remains visible” | GLM4.5-106B（GShard 式辅助损失）与 DeepSeek-V3（DeepSeek 式），EP64 组内（Figure 5 caption） | 已确认 |
| C5 | 大 EP 下每 rank 专家更少，专家级路由抖动直接转成明显的 rank 间倾斜；EPLB 依赖历史统计跟不上快速非平稳漂移，当实际负载偏离统计时甚至会加剧不均衡、制造新尖峰与 straggler | §3「Limitations of History-Based Balancing」：“With fewer experts per EP rank, large-EP directly translates routing dynamics across experts into pronounced inter-rank skew… When the realized load deviates from the statistics used to derive the expert layout, EPLB can even worsen imbalance, creating spikes and new stragglers” | EP=64；EPLB 重均衡间隔 prefill 50 batch、训练 3 个 global batch（Figure 6 caption） | 已确认 |
| C6 | 精确负载只在 gating 之后可得，因此均衡操作被迫进入关键路径；暴露的开销包含在线求解与重的专家重排通信（前向搬权重，训练还要搬梯度或优化器状态） | §1 第 4 段：“exact load becomes available only after gating, forcing balancing operations onto the critical path. The exposed overhead includes online plan solving and heavy expert rearrangement communication, with weight transfers in forward execution and additional gradient or optimizer-state movement in training” | — | 已确认 |
| C7 | 标准 RDMA 集群的高带宽 scale-up 只覆盖单台 4/8 卡服务器，跨机走较慢的 scale-out；大 EP 下跨多机搬大量专家状态在热路径上代价过高、不实用 | §1 第 4 段：“In standard RDMA clusters, high-bandwidth scale-up connectivity is confined to a single 4/8-GPU node, while inter-node traffic relies on slower scale-out networks. Under large-EP, moving substantial expert states across multiple nodes is prohibitively expensive and impractical on the hot path” | — | 已确认 |
| C8 | RSN 把 scale-up 域从单台 4/8 卡服务器扩到整机架、通常 64+ 卡；RSN 内跨服务器的 GPU 仍通过机架级 scale-up fabric 直连；scale-up 提供数百 GB/s 每卡带宽与 load/store 内存语义，scale-out 是包交换、通常每张网卡只有数十 GB/s | §2.1：“a rack-scale node (RSN) expands the scale-up domain from a single 4/8-GPU server to a full rack, typically spanning 64+ GPUs… scale-up offers much higher per-GPU bandwidth (hundreds of GB/s) and load/store-style memory semantics, whereas scale-out uses packet-based networking and typically provides only tens of GB/s per NIC” | — | 已确认 |
| C9 | 两个挑战：控制面要在 gating 与 token dispatch 之间的短窗口内做出高质量决策；数据面要执行静态集合通信支撑不好的不规则、易变专家状态搬运，否则可能吃不满 RSN 带宽。缺乏协同设计时这些开销容易抹掉均衡收益 | §1 第 5 段：“The control plane must make a high-quality balancing decision within the short window between gating and token dispatch. The data plane must then execute irregular, volatile expert state transfers that are poorly backed by static collectives, potentially underutilizing RSN bandwidth. Without careful co-design, these overheads can easily negate the balancing gain” | — | 已确认 |
| C10 | 逻辑专家是模型定义的专家身份，物理专家是某 rank 实体化的副本；每 rank 保留相同数量的主槽与冗余槽，主槽放逻辑专家的原始实例，冗余槽放一个副本或留空，形成一对多映射 | §4.1：“we use logical expert to denote the expert identity defined by the model, and physical expert to denote an expert replica instantiated by a rank. Every rank reserves the same number of main and redundant slots… yields a one-to-many logical-to-physical mapping: each logical expert has one fixed main instance and zero or more redundant replicas” | — | 已确认 |
| C11 | UltraEP 只做复制、从不重排主专家；理由是大 EP 下每 rank 本地主专家已很少（常 2 或 4），重排只能重洗一个很小的本地集合而付出可观的状态迁移、控制复杂度与局部性破坏，边际收益递减 | §4.1「Replication Only」：“It never reorders main experts. This reordering-free design is effective because large-EP reduces the number of local main experts per rank (often two or four). At that point, reordering brings diminishing marginal benefits; it can only reshuffle a tiny local set while incurring substantial state migration, control complexity, and locality disruption” | 大 EP | 已确认 |
| C12 | 冗余槽不保存优化器状态（优化器更新只作用于主专家），权重/梯度 buffer 跨层共享；代价是前向关键路径上出现逐层权重实体化的紧期限 | §4.1「Cross-Layer Buffer Reuse」：“For each redundant slot, it keeps no optimizer state (optimizer updates are applied only on main experts) while sharing weight/gradient buffer across layers… at the cost of a tight, per-layer weight-materialization deadline on the forward critical path” | 训练与 serving 均适用；主专家保持标准布局 | 已确认 |
| C13 | 前向：复用已有的 notify-dispatch 收集全局路由信息；拿到精确负载后每个 rank 确定性地算出同一份复制与 reroute 计划，无需额外同步；两者全在设备上完成、无 host 瓶颈。复制决定后分发主专家权重到远端副本，该同步可与 reroute 重叠，但 token dispatch 必须等它结束以避免带宽争用，因此规划与权重复制都在关键路径上 | §4.2「Forward」：“every rank deterministically computes an identical replication and reroute plan with no extra synchronization… This synchronization can overlap with reroute, but token dispatch should wait for it to finish to avoid bandwidth contention. As a result, planning and weight replication both stay on the critical path” | — | 已确认 |
| C14 | 反向：先把冗余专家权重恢复到与前向相同的状态，该通信可与 Wgrad 计算重叠、到 Dgrad 开始前为止以避免竞态；MoE 反向结束后每个主专家把所有远端副本贡献的梯度聚合进主梯度 buffer，该归约保持与无副本形式的等价性，且必须在下一 MoE 层开始前完成（冗余梯度 buffer 也跨层复用）；反向不重新求解，复用前向缓存的元数据，反向 reroute 是前向分配的 scatter-to-gather 逆操作，开销可忽略 | §4.2「Backward」全段：“This reduction preserves equivalence with the no-replica formulation and must finish before the next MoE layer begins… Backward execution does not solve replication again, and it reuses the cached metadata from the forward pass. The reverse reroute is effectively a scatter-to-gather inversion of the forward assignment with negligible overhead” | 训练 | 已确认 |
| C15 | 约束四条：主专家放置不可变；每 rank 冗余槽预算为 $N_{\mathrm{slot}}$ 且同一逻辑专家不得在同一 rank 出现两次；副本引入的反向通信必须被完全隐藏；每个新建副本至少承载 $u_{\min}$ 配额 | §4.3「Constraints」：“First, main expert placement is immutable. Second, each rank has a redundant-slot budget of $N_{\mathrm{slot}}$, and no logical expert may appear more than once on the same rank. Third, the backward communication introduced by replicas must be fully hidden… Fourth, we additionally require every newly created replica to carry at least a quota of $u_{\min}$” | — | 已确认 |
| C16 | 配额求解流程：二分 $\tau$ 于「目标 rank 负载」与「初始最大 rank 负载」之间，每次探测跑一个贪心可行性 oracle；oracle 按残余超额降序访问超载 rank、按 $\lambda_e$ 降序访问其主专家，把当前最热专家的负载尽量多地转给空闲最大的合法目标 rank，受 $u_{\min}$ 约束；接受的转移同时创建副本并更新其在临时计划 $\tilde U$ 中的配额，因此放置本身已编码有效的 reroute 容量。全部残余超额被排空则记录 $\tilde U$ 并向更小阈值搜索，否则该探测不可行、向上搜索 | §5.1「Quota Construction」全段 + Algorithm 1 | — | 已确认 |
| C17 | 用配额作耦合变量，使每次探测在选副本的同时预留 reroute 容量，避免枚举副本集合或 token 级路由，也避免产生「满足放置启发式但收不到多少流量」的无效副本 | §5.1「Why efficient」：“By using quota as the coupling variable, each probe reserves reroute capacity while choosing replicas. This avoids enumerating replica sets or token-level routes, and prevents ineffective replicas that would satisfy placement heuristics but receive little traffic” | — | 已确认 |
| C18 | reroute 不再回访均衡目标，只把 $U$ 实体化成源侧拆分 $Q$。先让来自同一 host rank 的 token 消费该 host 的配额；本地优先只改变哪个源 rank 消费某个已解出的配额，不改变配额本身，因此在不破坏已解阈值的前提下减少跨 rank 流量。残余源需求按剩余配额的比例分摊，用确定性取整同时保持每源需求与每实例配额 | §5.2「Quota Decomposition with Locality」：“Prioritizing local quota only changes which source rank consumes a solved quota, not the quota itself. Therefore, token locality reduces cross-rank traffic without breaking the solved threshold” | — | 已确认 |
| C19 | token 级分配：每 rank 用轻量前缀扫描把分解结果存成按逻辑专家的物理实例排序的累积配额；dispatch 时 $(r,e)$ 对的第 $j$ 个本地 token 发给累积配额首次覆盖 $j$ 的物理实例，把 token 分配降为 rank 本地的上界查找，与优化过程无关 | §5.2「Token Assignment」全段 | — | 已确认 |
| C20 | 求解全在设备上完成，避免热路径的 CPU 同步与设备-主机元数据搬运。难点在于算法不是简单的数据并行扫描：每次二分探测都会改动超额、空闲、槽位占用与每专家副本集合，可行性 oracle 在已接受转移之间有顺序依赖。做法是用单个 SM 上的一个 cooperative thread block，把负载矩阵与放置状态放进共享内存，跨 warp 评估多个阈值探测，用 warp 级归约在槽位与不重复约束下找可行的高空闲目标；同一 kernel 直接产出槽位映射、配额与累积 reroute 元数据 | §5.3「GPU-Native Solving」全段 | — | 已确认 |
| C21 | 均衡流量是 RSN scale-up fabric 上运行期自适应的稀疏传输图，而不是规则集合通信；服务两条路径：前向权重分发与反向权重重分发（主专家到远端副本），以及反向从副本回到主专家的梯度归约。目标是易变逐层计划下的高有效带宽，而不只是峰值链路速率 | §6 开头全段 | — | 已确认 |
| C22 | persistent tile streaming：权重与梯度切成固定大小 tile，放置计划编译成常驻设备的 tile 搬运任务；不是每副本发一次传输，而是跑一个 persistent kernel，其 thread block 反复从全局任务流取下一个 tile。权重分发时每个源 tile 只暂存进共享内存一次然后写到所有远端副本目的地；梯度归约时从远端副本载入梯度 tile 累加进主专家本地梯度 buffer 并就地清零以供后续层复用。kernel 对共享内存 tile 做双缓冲：tile $i$ 在写出或归约时，同一 block 已取下一个 tile 索引并开始载入 tile $i+1$。于是计划相关的任务查表、地址翻译与同步开销被折进 tile 流水线、被数据搬运掩盖，而不是每次副本传输都暴露成独立控制步骤 | §6.1 全段 | — | 已确认 |
| C23 | overlap-aware footprint：常驻 thread block 数是主要资源旋钮，按重叠窗口调整。前向关键路径上启动足够多 block 以增加在飞 tile 载入与远端写出；重叠的反向路径上用可配置的 SM 占用显式限制 footprint，共享内存用量也限制在活跃流水线缓冲，减少与同 SM 上其他计算密集反向 kernel 的争用 | §6.1「Overlap-Aware Footprint」全段 | — | 已确认 |
| C24 | fan-out 瓶颈是发送侧的：每 rank 最多有 $N_{\text{slot}}$ 个入站副本，但可能要把若干热专家推给很多目的地。副本数超过 relay 阈值（设为 4）的专家走两级中继：源 rank 先播种一个小的中继集合，每个中继再转发给分配给它的叶子。中继前沿取在 $\sqrt{\lvert\mathcal{H}(e)\rvert-1}$ 附近，近似平衡两级并相应降低源的关键 fan-out。中继调度作用在连续 tile 组成的 chunk 上而不是整个专家，中继 rank 收到一个 chunk 就能立刻转发：某个一阶段 chunk 的所有 tile 到达中继后 kernel 写下该 chunk 的 ready flag，二阶段等这个 flag 然后立刻从本地中继缓冲把该 chunk 发给叶子。这种 chunk 级流式让两级流水线化，无需等整个专家、也不引入全局跨阶段屏障 | §6.2 全段 | relay 阈值 4；实测中初始不均衡 4.0/6.0/8.0 时分别有 1/2/3 个主专家启用 relay（Figure 16 横轴标注） | 已确认 |
| C25 | load-aware relay scheduling：kernel 构建平衡各 rank 出向流量的中继拓扑。先按复制计划统计每 rank 被分配的发送字节量，再逐个处理可 relay 的热专家：对每个专家从其副本 rank 中选发送量最小的作一阶段中继，其余副本挂到「接收该叶子后发送量仍最小」的中继上；处理下一个热专家前更新源 rank 与其中继的发送量。该调度只决定中继树，每条边仍用 chunk streaming 传输 | §6.2「Load-Aware Relay Scheduling」全段 | — | 已确认 |
| C26 | UltraEP 是与训练/服务框架、MoE token all-to-all 后端都解耦的独立运行时；核心库约 9.6K 行 C++（含 device kernel）与 Python；集成到 Megatron-LM（训练）与 SGLang（服务）各低于 1K 行；token dispatch/combine 用 DeepEP 的 hybrid-ep 分支（v1.2.1+7febc6e）。全在设备上执行，避免 host-device 传输并保留设备操作的 graph capture | §7 第 1 段全段 | — | 已确认 |
| C27 | RSN 内存语义：用 GPU 发起的单侧 peer memory 访问。初始化时所有 rank 为冗余权重/梯度、放置与负载元数据及各类 flag 分配对称缓冲，然后把 RSN 内的 peer handle 解析成常驻设备的地址表；传输 kernel 消费紧凑任务描述符，通过 load/store 原语访问 peer 缓冲 | §7「RSN Memory Semantics」全段 | — | 已确认 |
| C28 | 冗余专家作为层间共享的内部缓冲由 UltraEP 维护，主专家的持久模型状态由外部框架承载；冗余专家被排除在框架侧参数/梯度桶、优化器状态与 checkpoint 之外。反向时 UltraEP 给每个在飞 MoE 调用分配一个 virtual layer ID，它把放置与 reroute 元数据同具体的（真实层, microbatch）在一个环形缓冲里做哈希；该 ID 经 torch.autograd 传递，使反向的权重重实体化与梯度归约能取回匹配的前向均衡计划。把环大小设为最大在飞 microbatch 数即可容纳 PP 与 virtual PP。因为只在 EP 组内起作用，UltraEP 与 attention 侧 DP、TP 和模型级 DP 正交 | §7「End-to-End Integration」全段 | — | 已确认 |
| C29 | UltraEP 直接优化 reroute 后的负载上界（均衡的真实目标），而 EPLB+ 关注的是 reroute 前的不均衡；EPLB+ 按 reroute 前热度盲目复制，UltraEP 只在副本带来足够均衡收益时才实体化它，这解释了 UltraEP 的资源效率并显著减少专家复制流量 | §8.5「Balancing Quality」：“This comes from UltraEP's direct optimization of the post-reroute load bound, which is the actual objective of balancing, rather than the pre-reroute imbalance that EPLB+ focuses on. Unlike EPLB+, which blindly replicates experts based on pre-reroute hotness, UltraEP only materializes a replica when it brings sufficient balancing gain” | Figure 15 的模拟设置：初始负载用 power-law 合成 | 已确认 |
| C30 | 剩余与强制均衡的差距主要来自真实 MoE 训练中不均匀的路由本身，而不是残余不均衡或热路径均衡开销 | §8.2 末段：“The remaining gap to force-balancing mainly comes from uneven routing in realistic MoE training, instead of residual imbalance or hot-path balancing overhead” | 训练；三模型 | 已确认 |
| C31 | 前向 token all-to-all 相对 ideal 增加的部分来自真实的不均匀 token 路由（区别于 ideal 中的合成均匀 dispatch），对 DeepEP 表现为其内部 token dispatch 流水线中的轻微 stall | §8.3：“This stems from uneven token routing in reality, distinct from synthetic uniform dispatch in the ideal. For DeepEP, this irregularity translates into minor stalls within its internal token dispatch pipelines” | Qwen3-235B-A22B 训练 | 已确认 |
| C32 | 均衡也改变活跃内存峰值，尤其在最热的接收 rank 上；不做均衡时训练与 serving 的 MoE 活跃内存峰值分别是 ideal 的 2 倍与 11 倍，UltraEP 通过压平接收侧热点把 MoE 活跃峰值大幅降低到接近 ideal，直接降低 OOM 风险、提升模型扩展余量，并可避免 activation checkpointing 带来的额外性能损失 | §8.4：“we observe 2× and 11× higher peak memory of MoE activation than the ideal for training and serving, respectively. By flattening receive-side hot spots, UltraEP substantially reduces the MoE activation peak and remains close to the ideal” | 训练侧关闭 activation checkpointing 以暴露层累积上界，观测峰值出现在高倾斜的初期阶段；serving 前向只保留当前层的瞬时活跃 | 已确认 |
| C33 | 生产训练的 loss 曲线符合预期的预训练轨迹，因为 UltraEP 只改物理执行逻辑、保留训练语义 | §8.6：“The loss curve follows the expected pretraining trajectory because UltraEP changes only the physical execution logic while preserving training semantics” | RefMoE-288B-A16B，EP32，内部训练栈 | 已确认 |
| C34 | 算法侧路由正则与系统侧均衡互补而非互换：训练期辅助路由损失主要稳定优化、防止 routing collapse、保留专家特化，但都无法保证逐 microbatch 的实现负载均衡（细粒度 MoE + 大 EP 下尤其如此）；系统侧技术纠正运行时不均衡，也不能替代路由损失，因为二者服务的建模目标不同 | §2.2 末段全段 | — | 已确认 |
| C35 | 预测式设计（历史、跨层相关性、profiling 后预取或重排布局）在高动态细粒度 MoE 的大 EP 与 RSN 场景下缺乏实用性；UltraEP 反应的是已实现负载 | §9「System-Side MoE Load Balancing」：“These predictive designs lack practicality for highly dynamic fine-grained MoE models at large-EP and RSN settings, whereas UltraEP reacts to realized load to achieve near-optimal balancing” | — | 已确认 |
| C36 | MoE 通信库与计算优化（专用 token all-to-all 内核、grouped-GEMM 的 kernel fusion、计算与通信的细粒度重叠调度）与 UltraEP 正交、可叠加以获得累积收益 | §9「MoE Computation and Communication Optimization」末句：“These optimizations are orthogonal to and can be stacked with UltraEP for cumulative gains” | — | 已确认 |
| C37 | 因为 UltraEP 同时覆盖训练与推理，同一抽象可自然延伸到交替这两个过程的 RL pipeline | §10 末句：“the same abstraction can naturally extend to reinforcement learning (RL) pipelines that alternate these two procedures” | 论文展望，未做实验 | 已确认为「论文展望」 |
| C38 | decode 阶段计算侧不均衡的影响被访存延迟大幅稀释；增大 batch 可提高计算强度但与 decode TPOT 的严格 SLO 冲突，所以实际均衡目标是 prefill（吞吐优先以降低 TTFT） | §3「Serving Prefill」：“For memory-bound decode, we discover that the impact of compute-side imbalance is largely diluted by memory access latency. Increasing the batch size can improve compute intensity, but that conflicts with strict SLOs for decode TPOT. Therefore, the practical balancing target is prefill” | — | 已确认 |
| C39 | 二分 + 贪心 oracle 不保证全局最优，只保证在该 oracle 的可行性判据下得到最小可行阈值 | §5.1 明确用词是 “runs a greedy feasibility oracle for each probe”；论文全篇（标题、摘要、§8）用 “near-optimal” 而非 optimal，未给出最优性证明 | — | 标注为推断 |

## F 公式

| 编号 | 公式 | 原文 | 说明 |
|---|---|---|---|
| F1 | $T_{\text{solve\_rep}}^{fwd}+\max(T_{\text{reroute}}^{fwd},T_{w\_\text{distr}}^{fwd})+T_{\text{tok\_a2a}}^{fwd}+T_{\text{moe}}^{fwd}$ | §4.3 Eq.(1) | 前向目标；各项依次为规划求解、reroute、权重分发、token all-to-all、MoE 计算的延迟。reroute 与权重分发取 max 因为二者可重叠（C13） |
| F2 | $\min\ T_{\text{tok\_a2a}}^{bwd}+T_{\text{moe}}^{bwd}$ | §4.3 Eq.(2) | 反向目标；反向复用前向元数据，副本相关通信藏在计算下，只剩这两项暴露 |
| F3 | $T_{\text{moe}}^{fwd/bwd}\propto\max_{r\in\mathcal{R}}\sum_{e\in\mathcal{E}}u_{e,r}$ | §4.3 Eq.(3) | 用 reroute 后最忙 rank 建模 MoE 计算；$T_{\text{moe}}^{bwd}\approx 2T_{\text{moe}}^{fwd}$ 以计入 Wgrad 与 Dgrad |
| F4 | $T_{\text{tok\_a2a}}^{fwd/bwd}\propto\max_{r\in\mathcal{R}}\max\left(\sum_{e\in\mathcal{E}}\lambda_{r,e},\sum_{e\in\mathcal{E}}u_{e,r}\right)$ | §4.3 Eq.(4) | 由最忙发送端或接收端主导。训练中 $\sum_e\lambda_{r,e}$ 由 microbatch 形状与并行配置固定，prefill 中由 chunked-prefill 大小上界约束 |
| F5 | $T_{w\_\text{distr}}^{fwd}\propto\max_{r\in\mathcal{R}}\sum_{e\in\mathcal{E}_{r}}(\lvert\mathcal{H}(e)\rvert-1)$ | §4.3 Eq.(5) | 权重分发延迟由承载最热主专家的 rank 主导；每个专家要发给 $\lvert\mathcal{H}(e)\rvert-1$ 个远端副本 |
| F6 | $\mathrm{exc}_r(\tau)=\max(\ell_r-\tau,0)$，$\mathrm{slk}_r(\tau)=\max(\tau-\ell_r,0)$ | §5.1 Eq.(6) | 阈值 $\tau$ 下 rank $r$ 必须卸掉的超额与还能吸收的空闲。其中 $\lambda_e=\sum_{r}\lambda_{r,e}$，$\ell_r=\sum_{e\in\mathcal{E}_r}\lambda_e$ |
| F7 | $\tau_{\mathrm{lo}}\gets\beta\cdot\left\lceil\frac{1}{R}\sum_r\ell_r\right\rceil$，$\tau_{\mathrm{hi}}\gets\max_r\ell_r$ | Algorithm 1 第 3 行 | 二分区间的初始化；$\beta=1.01$ |
| F8 | $\delta\gets\min(\mathrm{exc}_r,\mathrm{slk}_{t^\star},\mathrm{cap}_e)$，且 $\delta<u_{\min}$ 时 break | Algorithm 1 第 12–13 行 | 单次转移量取三者最小；不足 $u_{\min}$ 则不建这个副本 |
| F9 | $q_{r,e,t}\gets\text{round}\left(\hat{\lambda}_{r,e}\times\frac{\hat{u}_{e,t}}{\sum_{t'}\hat{u}_{e,t'}}\right)$ | Algorithm 1 SolveReroute | 残余源需求按残余配额比例分摊 |
| F10 | $\sum_{t\in\mathcal{H}(e)}q_{r,e,t}=\lambda_{r,e}$，$\sum_{r\in\mathcal{R}}q_{r,e,t}=u_{e,t}$ | §4.3 Table 1（$Q$ 行） | reroute 的双向守恒：按源看凑满该源需求，按目标看凑满该实例配额 |
| F11 | relay 前沿 $\approx\sqrt{\lvert\mathcal{H}(e)\rvert-1}$ | §6.2 | 两级中继的一阶段宽度选择 |

## N 数字

### 主结果

| 编号 | 数值 | 来源 | 实验条件 |
|---|---|---|---|
| N1 | 训练与 serving 平均达到强制均衡 ideal 吞吐的 94.3% | Abstract | 训练与 serving prefill 的总平均；分项见 N2、N3 |
| N2 | 训练平均 94.6% ideal | §1 第 6 段 | 三个模型：GLM4.5-106B（128 GPU / 2 机架）、Qwen3-235B（256 GPU / 4 机架）、DeepSeek-V3（256 GPU / 4 机架），bf16 |
| N3 | serving prefill 平均 93.9% ideal，区间 90%–97% | §1 第 6 段、§8.2 | Qwen3-235B（EP64）与 GLM4.7-358B（EP40），单机架，DP attention |
| N4 | 相对不做均衡平均提升 1.49× | Abstract | 训练与 serving 平均、相对 no-balancing |
| N5 | 训练相对 Megatron-LM 平均 1.42× | §1 第 6 段 | 同 N2 |
| N6 | serving prefill 相对 SGLang 1.56×，相对 EPLB 1.29× | §1 第 6 段、§8.2 | 同 N3 |
| N7 | 均衡后 rank 间不均衡度 1.01–1.04（摘要：从 1.30–4.01 降到 1.01–1.04） | Abstract、§1、§8.2 | 训练侧 1.01–1.03，serving 侧 1.01–1.04 |
| N8 | 专家复制相对主流通信后端加速 3.1×–5.5× | §1 第 6 段、§8.5 | Qwen3-235B，EP64；基线为 torch.distributed batch send/recv 与 DeepEP |
| N9 | 生产训练维持 >92% ideal 吞吐，相对 no-balancing 平均提升 9.6% | §8.6 | RefMoE-288B-A16B，EP32，内部训练栈，多机架 |

### 训练端到端（Figure 11 标注值）

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N10 | GLM4.5-106B-A12B 平均 TFLOPS/GPU：Megatron / EPLB / LPLB / EPLB+ / Ours / Ideal | 545 / 646 / 618 / 695 / 757 / 785.4 | 128 GPU，EP64-DP2，$N_{\text{slot}}=2$，从第 3500 个 global batch 恢复跑 20 个 |
| N11 | Qwen3-235B-A22B 同上 | 315 / 449 / 385 / 474 / 524 / 574.7 | 256 GPU，EP64-DP4，$N_{\text{slot}}=2$ |
| N12 | DeepSeek-V3-671B-A37B 同上 | 509 / 505 / 516 / 553 / 613 / 637.6 | 256 GPU，EP64-PP4，$N_{\text{slot}}=2$ |
| N13 | 三模型平均相对 Megatron-LM 提升：EPLB / LPLB / EPLB+ / Ours | 20% / 12% / 29% / 42% | §8.2 正文；由 N10–N12 反算得 20.1 / 12.3 / 28.9 / 41.9%，一致 |
| N14 | 平均不均衡度（GLM4.5 / Qwen3 / DeepSeek-V3），顺序 M/E/L/E+/O | 2.23,1.34,1.29,1.10,1.02 / 2.86,1.34,1.58,1.17,1.03 / 1.30,1.28,1.09,1.18,1.01 | Figure 11 下排柱状图标注 |
| N15 | DeepSeek-V3 上 EPLB 与 LPLB 表现与 Megatron-LM 相当甚至更差，UltraEP 仍在 96% ideal 以上 | 613/637.6 = 96.1% | §8.2；原因是路由补偿降低整体不均衡但放大短期摆动 |

### serving prefill（Figure 12 标注值）

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N16 | Qwen3-235B 平均不均衡度（SGLang / EPLB / EPLB+ / Ours）：STEM 与 Mixed | STEM 3.68 / 2.59 / 1.11 / 1.04；Mixed 4.01 / 2.56 / 1.11 / 1.03 | EP64，单机架 |
| N17 | GLM4.7-358B 同上 | STEM 3.09 / 2.05 / 1.08 / 1.01；Mixed 2.06 / 1.80 / 1.06 / 1.01 | EP40，$N_{\text{slot}}=4$ |
| N18 | UltraEP 相对 EPLB+ 的 prefill 吞吐增益 | 5%–24% | §8.2 |
| N19 | 均衡质量对照采用同一负载：记录 SGLang 满载下的完整路由 trace 再回放给其他算法 | — | §8.2 方法说明 |

### 延迟分解（Figure 13 标注值，Qwen3-235B-A22B 训练，单层平均 ms）

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N20 | 前向（MoE 计算 / token all-to-all / Others）：Ideal、Megatron-LM、Ours | Ideal 1.89/1.68/2.83；Megatron 5.95/6.94/2.88；Ours 2.11/2.24/3.16 | 单层平均 |
| N21 | 反向同上 | Ideal 4.32/1.62/4.49；Megatron 10.41/4.75/4.52；Ours 4.43/1.79/4.58 | 单层平均 |
| N22 | 非 MoE 部分相对 ideal 的额外延迟：前向 0.33 ms，反向可忽略，占总延迟 1.8% | 3.16−2.83 = 0.33；0.33/(7.51+10.80) = 1.8% | §8.3；由 N20/N21 反算一致 |
| N23 | token all-to-all 相对 ideal 增幅：前向 33%，反向 10% | 2.24/1.68 = 1.333；1.79/1.62 = 1.105 | §8.3；反算一致 |

### 活跃内存（§8.4）

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N24 | 不做均衡时 MoE 活跃内存峰值相对 ideal | 训练 2×，serving 11× | 训练关闭 activation checkpointing，峰值出现在高倾斜的初期；serving 只保留当前层瞬时活跃 |

### 消融（Figure 15 与 Table 3）

| 编号 | 内容 | EPLB+ | Ours | 条件 |
|---|---|---|---|---|
| N25 | 结果不均衡（模拟均值） | 1.19 | 1.03 | Table 3；初始负载用 power-law 合成 |
| N26 | 求解耗时 | 0.153 ms | 0.111 ms | 省 27.4%（反算 27.45%，一致） |
| N27 | $\sum_e\lvert\mathcal{H}(e)\rvert$（消耗的实例总数） | 107 | 45 | 少 57.9%（反算 57.94%，一致） |
| N28 | $\max_e\lvert\mathcal{H}(e)\rvert$（最大 fan-out） | 8.5 | 6.8 | Table 3 |
| N29 | 在飞 token 占比 | 99.9% | 96.0%（关闭本地优先为 98.4%） | 降 3.9 个百分点（正文表述为 "reduces token traffic by 3.9%"） |
| N30 | 紧预算高倾斜下的对照：$(128,64,1)$、初始不均衡 6.0 与 8.0 时 EPLB+ 约 1.40，UltraEP 分别约 1.07、1.09 | — | — | Figure 15 下排左图标注；正文表述为 EPLB+ 高达 1.4、UltraEP 仍低于 1.1 |

### 通信（Figure 16 标注值，Qwen3-235B，EP64，ms）

| 编号 | 初始不均衡 | torch.distributed | DeepEP | Ours w/o Relay | Ours | 启用 relay 的主专家数 |
|---|---|---|---|---|---|---|
| N31 | 1.5 | 0.92 | 0.73 | 0.22 | 0.24 | 0 |
| N32 | 2.0 | 1.17 | 0.86 | 0.27 | 0.28 | 0 |
| N33 | 4.0 | 1.01 | 1.02 | 0.37 | 0.29 | 1 |
| N34 | 6.0 | 1.12 | 1.20 | 0.47 | 0.28 | 2 |
| N35 | 8.0 | 1.53 | 1.27 | 0.52 | 0.28 | 3 |
| N36 | relay 在高倾斜大 fan-out 下的额外增益 1.3×–1.8×；Ours 随 fan-out 增长维持约 0.28 ms 近乎恒定，no-relay 变体线性增长；低倾斜时自适应策略让 relay 不激活，只有可忽略的控制开销 | | | | | §8.5；由 N33–N35 反算 relay 增益 1.28×/1.68×/1.86× |

### 生产训练（Figure 17 标注值）

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N37 | RefMoE-288B-A16B：ideal 504 TFLOPS/GPU，no-balancing 均值 425.0 | 504 / 425.0 | EP32；ideal 取实测强制均衡的最好值以剔除环境波动；no-balancing 从关闭 UltraEP 的续跑区间采样 |
| N38 | UltraEP 吞吐 = 425.0 × 1.096 ≈ 465.8，占 ideal 92.4% | — | 与 §8.6 的 ">92%" 和 "9.6% 平均增益" 一致 |

### 内存与配置

| 编号 | 内容 | 数值 | 条件 |
|---|---|---|---|
| N39 | Qwen3-235B-A22B（94 个 MoE 层、128 专家）单个冗余槽：权重 3.3 GB → 每 rank 36 MB，梯度 6.6 GB → 72 MB | 3.3×1024/94 = 35.95；6.6×1024/94 = 71.9 | §4.1；反算一致 |
| N40 | $u_{\min}=1024$，$\beta=1.01$，relay 阈值 = 4 | — | §4.3 Table 1、§6.2 |
| N41 | 测试床：每机架 64 GPU（16 台服务器）；scale-up 链路带宽是 scale-out RDMA 的 8–10 倍；prefill 用 1 机架，训练用 2 或 4 机架 | — | §8.1 Testbed |
| N42 | 被测模型（专家数 / top-k / 并行 / $N_{\text{slot}}$）：GLM4.5-106B-A12B 128(8) EP64-DP2 / — / 2；Qwen3-235B-A22B 128(8) EP64-DP4 / EP64 / 2；GLM4.7-358B-A32B 160(8) — / EP40 / 4；DeepSeek-V3-671B-A37B 256(8) EP64-PP4 / — / 2 | — | Table 2 |
| N43 | 训练配方：200B token 内部语料子集，约 4500 global batch，batch size 从 1024 ramp 到 5120；GLM4.5 与 Qwen3 用 GShard 损失权重 $10^{-2}$，DeepSeek-V3 与 RefMoE 用 DeepSeek 配方（路由 bias 更新步长 $10^{-3}$、序列级损失权重 $10^{-4}$） | — | §8.1 Training Recipes |
| N44 | 基线版本：Megatron-LM dev 分支 commit e93814b；SGLang main 分支 v0.5.9+bbe9c7e；EPLB 重均衡频率 prefill 50 步 / 训练 3 个 global batch；LPLB 限制每专家至多一个副本 | — | §8.1 Baselines |
| N45 | serving 负载：STEM（Codeforces、SWE-bench、DAPO-Math-17K、GPQA、OpenScience）与 Mixed（额外含 LongBench 的多任务长上下文）；输入长度从数百到数万 token；Poisson 到达 | — | §8.1 Serving Workloads |

### 动机观测（Figure 4/5/6）

| 编号 | 内容 | 条件 |
|---|---|---|
| N46 | prefill 负载分布随 forward step、数据域与层漂移；不均衡比定义为最大「每专家」负载除以均值 | Qwen3-235B，128 专家取 top-8，EP=64（Figure 4 caption） |
| N47 | 训练负载分布取初期（前 25 个）与后期（4500 个 global batch 中的第 3500–3510 个） | GLM4.5-106B（128 专家 top-8，GShard 式辅助损失）与 DeepSeek-V3（256 专家 top-8，DeepSeek 式辅助损失），EP64 组内（Figure 5 caption） |
| N48 | Figure 6 中 EPLB 前后的 rank 级不均衡：Qwen3-235B 第 68 层 prefill 与 DeepSeek-V3 第 57 层训练；EPLB 出现高于 no-balancing 的尖峰 | EP=64；EPLB 间隔 prefill 50 batch / 训练 3 个 global batch；prefill 用 mixed 数据，训练取第 3510 个 global batch（Figure 6 caption） |

## 冲突与证据不足

| 项 | 情况 | 处理 |
|---|---|---|
| 摘要的 "1.49× over no-balancing" 与 §1 的 "1.42× over Megatron-LM / 1.56× over SGLang" | 基线不同，不是冲突 | 页面同时给出两组数字并注明各自基线 |
| Abstract 说 "94.3%"，§1 分开说 94.6%（训练）与 93.9%（prefill） | 前者是两项平均 | 页面以分项为主，注明 94.3% 是总平均 |
| 智源社区等转述版本提到「2560 块 GPU 的生产训练验证」 | v3 的 TeX 源码 §8.6 只说 "at larger scale than prototyping runs"，未给出 2560 这个数字；摘要（0-abstract.tex）也无此句 | 不写入页面。转述可能来自另一版本或站点补充，无法在固定版本中定位 |
| §8.5 正文写 3.1×–5.5×，由 Figure 16 读数反算最小档为 3.0× | 读图精度差异（0.92/0.24 = 3.83，0.73/0.24 = 3.04） | 页面采用论文表述 3.1×–5.5×，不写自行反算值 |
| Figure 编号 | TeX 源码用 label 不用显式编号；编译后编号经 arXiv HTML 版核对：Figure 1 teaser、2 bg-rsn、3 bg-ep、4 motive_serving_load、5 motive_train_load、6 motive_eplb_imbalance、7 tech-expert-layout、8 tech-timeline-fwd、9 tech-timeline-bwd、10 tech-relay、11–17 为 §8 各图 | 页面按此编号标注 |

## 原图候选

| 编号 | 原文 Figure | 内容摘要 | 可说明的结论 | 获取途径 |
|---|---|---|---|---|
| G1 | Figure 1 | 上下对照：Existing 用 stale load 在 global batch 边界做 periodic balancing；Ours 在每层的 Attention→Gate→均衡→A2A→Experts→A2A 中插入 hot-path balancing，标注 exact load 与约 0.3 ms 暴露开销 | 三个维度的差异：负载保真度、决策时机、均衡频率（Q1、Q2） | TeX figs/intro-overview.pdf |
| G2 | Figure 2 | 标准 RDMA 集群 vs RSN 的 scale-up 域范围对比 | RSN 把整个 EP 组装进高带宽域（Q2） | TeX figs/bg-rsn.pdf |
| G3 | Figure 6 | EPLB 前后的 rank 级不均衡曲线，两个子图分别是 prefill 与训练；EPLB 曲线出现高于 no-balancing 的尖峰 | 历史式均衡跟不上、甚至加剧不均衡（Q1，支撑 C5） | TeX figs/motive_eplb_imbalance.pdf |
| G4 | Figure 7 | 8 专家、单层、EP=4、$N_{\text{slot}}=1$ 的槽位布局；下方标注主专家持久保存三类状态、冗余槽只 $1\times N_{\text{slot}}$ 且无优化器状态 | 主槽/冗余槽结构与跨层复用（Q4，支撑 C10、C12） | TeX figs/tech-expert-layout.pdf |
| G5 | Figure 8 | 前向时间线：计算流 Gate→Replication Planning→Reroute，通信流 Notify Dispatch→Weight Distribution→Token Dispatch，标出相对标准 MoE 前向的额外开销区间 | 规划与权重分发都在关键路径上（Q2、Q4，支撑 C13） | TeX figs/tech-timeline-fwd.pdf |
| G6 | Figure 10 | 三种 fan-out 方案的时间线：(a) 无中继，源直发 9 个目的地；(b) 同步两级中继，中间有 Barrier；(c) chunk streaming 两级中继，中继收到 chunk 立即转发 | chunk 级流式如何去掉阶段屏障（Q4，支撑 C24） | TeX figs/tech-relay.pdf |
| G7 | Figure 11 | 三模型 20 个迭代的吞吐曲线 + 平均 TFLOPS/GPU 与平均不均衡度柱状图 | 训练端到端结果与基线对照（N10–N15） | TeX figs/eval_train_e2e.pdf |
| G8 | Figure 13 | 前向与反向的堆叠延迟分解（MoE 计算 / token all-to-all / Others），三组：Ideal、Megatron-LM、Ours | 热路径开销与各项延迟归因（N20–N23，支撑 C31） | TeX figs/eval_latency_breakdown.pdf |
| G9 | Figure 15 | 上排：训练与 prefill 全部评测的不均衡分布箱线图；下排：三种 (专家数, EP, $N_{\text{slot}}$) 配置 × 五档初始不均衡的模拟结果柱状图 | 配额求解的均衡质量优势与预算敏感性（N25、N30） | TeX figs/eval_solver_perf.pdf |
| G10 | Figure 16 | 五档初始不均衡下四种方案的权重分发延迟柱状图，横轴同时标出启用 relay 的主专家数 | 通信优化与 relay 的贡献（N31–N36） | TeX figs/eval_comm_perf.pdf |
| G11 | Figure 12 | prefill 的 RPS–平均 TTFT 曲线 + 平均不均衡度柱状图，两模型 × 两数据域 | serving 端到端结果（N16–N18） | TeX figs/eval_serving_e2e.pdf |
| G12 | Figure 17 | 生产训练的 LM loss 曲线与吞吐箱线图，标注 ideal 504 与 no-balancing 均值 425.0 | 生产可扩展性与收敛保持（N37、N38，支撑 C33） | TeX figs/eval_prod.pdf |

原图选择与位置由 outline.md 决定。图片转换：`pdftoppm -png -r 190` 再转 webp。
