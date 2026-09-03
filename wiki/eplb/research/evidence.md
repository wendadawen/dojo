# EPLB 核心论断与证据

来源缩写：
- [repo] deepseek-ai/EPLB，main 分支（仓库于 2025-02-26 初始提交，eplb.py 最后修改 2025-03-24）：README.md 与 eplb.py。https://github.com/deepseek-ai/EPLB
- [v3] DeepSeek-V3 Technical Report，arXiv:2412.19437。https://arxiv.org/abs/2412.19437
- [vllm-doc] vLLM 文档「Expert Parallel Deployment」页（main 分支 docs/serving/expert_parallel_deployment.md）。https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md
- [vllm-src] vLLM 源码 vllm/distributed/eplb/policy/default.py（main 分支）。https://github.com/vllm-project/vllm/blob/main/vllm/distributed/eplb/policy/default.py

## C 论断

- C1：专家并行下不同专家被分到不同 GPU；专家负载随当前 workload 变化，需要保持各 GPU 负载均衡。来源：[repo] README 首段 "When using expert parallelism (EP), different experts are assigned to different GPUs. Because the load of different experts may vary depending on the current workload, it is important to keep the load of different GPUs balanced." 适用条件：专家并行部署。状态：已确认。
- C2：DeepSeek 采用冗余专家策略复制高负载专家，然后启发式地把复制后的专家打包到 GPU 保证负载均衡；同组专家尽量放进同一节点以减少跨节点流量。来源：[repo] README "we adopt a redundant experts strategy that duplicates heavy-loaded experts. Then, we heuristically pack the duplicated experts to GPUs... we also attempt to place the experts of the same group to the same node to reduce inter-node data traffic, whenever possible." 状态：已确认。
- C3：V3 的节点受限路由：每个 token 至多发给 M 个节点，节点按「该节点上专家的最高 K_r/M 个亲和分数之和」选出；该约束下训练框架几乎实现计算通信完全重叠。来源：[v3] §2.1.2 "we ensure that each token will be sent to at most $M$ nodes, which are selected according to the sum of the highest $\frac{K_r}{M}$ affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap." EPLB README 称该机制为 group-limited expert routing。状态：已确认。
- C4：算法计算「复制 + 排布」方案依据的是估计负载；负载预测方法不在 EPLB 范围内，常见做法是历史统计的移动平均。来源：[repo] README "The algorithm computes a balanced expert replication and placement plan based on the estimated expert loads. Note that the exact method to predict the loads of experts is out of this repo's scope. A common method is to use moving average of historical statistics." 状态：已确认。
- C5：层次策略：当节点数整除组数时使用；先把组均衡打包到节点（保证节点间负载均衡），再在节点内复制专家，最后把物理专家均衡打包到各 GPU；适用于 prefill 阶段较小 EP。全局策略：其余情况全局复制、全局打包到 GPU，不考虑组；适用于 decode 阶段较大 EP。来源：[repo] README The Algorithm 两小节；[repo] eplb.py `rebalance_experts` 分支 `if num_groups % num_nodes == 0`。状态：已确认。
- C6：复制贪心：把 num_log 个逻辑专家复制为 num_phy 个物理副本，使其「所有副本的最大负载」最小；实现为每次给 weight/logcnt（单副本负载）最大的逻辑专家增加一个副本。来源：[repo] eplb.py `replicate_experts` docstring "Replicate num_log experts to num_phy replicas, such that the maximum load of all replicas is minimized." 与实现（循环内 `redundant_indices = (weight / logcnt).max(dim=-1).indices`）。状态：已确认（并用纯 Python 复刻运行核对）。
- C7：打包贪心：把 n 个带权对象装进 m 个包，每包恰好 n/m 个对象、各包权重尽量均衡；实现为按权重降序遍历，每个对象放入「仍有容量且当前最轻」的包。来源：[repo] eplb.py `balanced_packing` docstring "Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs are as balanced as possible." 与实现。状态：已确认（复刻运行核对）。
- C8：物理专家的打包负载按「逻辑负载 ÷ 副本数」计算，即假设发往同一逻辑专家的 token 被各副本均摊。来源：[repo] eplb.py `tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)`。适用条件：分发层把同逻辑专家 token 均摊到副本。状态：已确认（代码事实；均摊是显式假设）。
- C9：接口约束：num_replicas 必须是 num_gpus 的倍数；层次路径还要求 num_logical_experts 整除 num_groups、num_groups 整除 num_nodes、num_gpus 整除 num_nodes。来源：[repo] eplb.py `rebalance_experts` docstring "num_replicas: number of physical experts, must be a multiple of num_gpus" 及 rebalance_experts_hierarchical 内四个 assert。状态：已确认。
- C10：V3 prefill 部署：最小单元 4 节点 32 GPU；attention 用 TP4+SP、DP8；MoE 用 EP32；为均衡负载引入冗余专家部署策略，复制高负载专家；高负载专家由线上部署收集的统计检测、周期性调整（如每 10 分钟）；确定冗余集合后在节点内 GPU 间精心重排，在不增加跨节点 all-toall 通信开销的前提下尽量均衡；prefill 阶段设 32 个冗余专家，每 GPU 除原有 8 个专家外多持有 1 个冗余专家。来源：[v3] §3.4.1。状态：已确认。
- C11：V3 decode 部署：把共享专家当作路由专家对待，每 token 路由时选 9 个专家（共享专家视为始终被选中的高负载专家）；最小单元 40 节点 320 GPU；attention TP4+SP、DP80；MoE EP320；每 GPU 只持有 1 个专家，64 张 GPU 承载冗余专家与共享专家；因每卡只有 1 个专家，无需重排；同样按线上统计周期性确定冗余专家集合。来源：[v3] §3.4.2 "During decoding, we treat the shared expert as a routed one... each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts. ... we do not need to rearrange experts since each GPU only hosts one expert." 状态：已确认。
- C12：V3 训练期均衡：aux-loss-free 策略在每训练步末按过载/欠载把 bias 减/加 γ；由于均衡策略有效，训练全程不丢 token；推理负载均衡由部署策略保证，推理也不丢 token。来源：[v3] §2.1.2 "During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded" 与 "DeepSeek-V3 does not drop any tokens during training. In addition, we also implement specific deployment strategies to ensure inference load balance, so DeepSeek-V3 also does not drop tokens during inference." 推论（由机制直接得出，标注为推断）：推理期没有训练步，bias 不再更新，路由规则固定。状态：已确认（原文）/ 推断（推理期 bias 固定，逻辑必然）。
- C13：vLLM 内置 EPLB：--enable-eplb 启用；启用后每次前向收集负载统计并周期性重排专家分布；default 策略的重排算法改编自 DeepSeek EPLB。来源：[vllm-doc] EPLB 节 "vLLM provides an Expert Parallel Load Balancer (EPLB) to redistribute expert mappings across EP ranks, evening the load across experts... When enabled, vLLM collects load statistics with every forward pass and periodically rebalances expert distribution."；[vllm-src] 模块 docstring "The rearrangement algorithm is adapted from [DeepSeek EPLB]"。状态：已确认。
- C14：vLLM EPLB 参数：window_size（重平衡决策追踪的引擎步数，默认 1000）、step_interval（每 N 个引擎步重排一次，默认 3000；大于窗口时只用最近窗口的统计）、num_redundant_experts（冗余专家数，默认 0）、log_balancedness（记录 avg tokens per expert ÷ max tokens per expert，默认关）、policy（默认 "default"）、communicator（专家权重传输后端）。来源：[vllm-doc] EPLB Parameters 表；vLLM 源码 vllm/config/parallel.py EPLBConfig。状态：已确认。
- C15：vLLB 每 rank 专家数公式：默认 NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS；带冗余时 (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS。内存开销 = NUM_MOE_LAYERS × BYTES_PER_EXPERT × NUM_REDUNDANT_EXPERTS ÷ NUM_EP_RANKS；DeepSeek-V3 每 rank 1 个冗余专家约 2.4 GB；文档建议大规模场景 num_redundant_experts=32。来源：[vllm-doc] Expert Distribution Formula 与 Memory Footprint Overhead 节。状态：已确认。
- C16：vLLM 的 preserve_intragpu_slots：GPU 数与每卡槽位不变时，重排后让留在同一 GPU 的专家保持原槽位，新来的专家填空槽，以避免不必要的权重拷贝。来源：[vllm-src] `preserve_intragpu_slots` docstring "Reorder the new mapping per GPU so that experts that remain on the same GPU keep their previous slot positions when possible. Incoming experts to that GPU fill any remaining available slots... Helps to avoid unnecessary weight copying"。状态：已确认。
- C17：V3 模型配置：61 层（前 3 层 dense，其余 MoE）；每个 MoE 层 1 个共享专家 + 256 个路由专家，每 token 激活 8 个路由专家、至多发给 4 个节点；671B 总参数 / 每 token 37B 激活。训练时路由专家均匀部署在 8 节点的 64 张 GPU（EP64）。来源：[v3] §4.2 与 §3.3（"the routed experts will be uniformly deployed on 64 GPUs belonging to 8 nodes"）。状态：已确认。
- C18：V3 探索中的动态冗余策略：每 GPU 持有更多专家（如 16 个），但每个推理步只激活其中 9 个；每层 all-to-all 开始前在线计算全局最优路由方案；prefill 计算量大，算路由的开销几乎可忽略。来源：[v3] §3.4.1 "we are exploring a dynamic redundancy strategy for experts, where each GPU hosts more experts (e.g., 16 experts), but only 9 will be activated during each inference step. Before the all-to-all operation at each layer begins, we compute the globally optimal routing scheme on the fly." 状态：已确认（报告原文为探索性描述）。
- C19：节点内 NVLink 带宽 160 GB/s，约为跨节点 IB（50 GB/s）的 3.2 倍；V3 限制每 token 至多 4 节点以减少 IB 流量，使 IB 与 NVLink 通信完全重叠。来源：[v3] §3.2.2。状态：已确认。

## F 公式

- F1：副本均摊。设逻辑专家 $e$ 的估计负载为 $w_e$、其物理副本数为 $c_e$，则每个副本承载的打包负载为 $w_e / c_e$。来源：[repo] eplb.py `tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)` 的直接形式化；与 C8 同源。适用条件：分发层把发往 $e$ 的 token 均摊到 $c_e$ 个副本。状态：已确认。
- F2（构造示例工具，非来源公式）：R 张卡的层利用率下界 = 平均负载 ÷ 最大负载。用于量化木桶效应；推导：整层时间由最大负载决定，理想均分时间为平均负载。属构造示例，在页面标注为构造示例计算。

## N 数字

- N1：官方示例：2 层 MoE、每层 12 个逻辑专家、每层引入 4 个冗余专家、共 16 个物理副本、2 节点 × 每节点 4 GPU；层 0 权重 [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86]；官方输出 phy2log 层 0 = [5,6,5,7,8,4,3,4,10,9,10,2,0,1,11,1]、层 1 = [7,10,6,8,6,11,8,9,2,4,5,1,5,0,3,1]。来源：[repo] README Interface and Example。状态：已确认（纯 Python 复刻运行输出与官方逐槽一致）。
- N2：构造例中间量（由 N1 输入按算法计算，已运行验证）：层 0 组负载 G0–G3 = 262/330/116/325；组打包后节点负载 446/587；节点 0 复制 e5、e4（副本数 2），节点 1 复制 e10、e1；层次策略 GPU 负载 = 121.5/86.5/125/113/147.5/131.5/156/152（max/mean ≈ 1.21）；全局策略复制 e10、e5、e1、e4，GPU 负载 = 130.5/95.5/130/138/138.5/134.5/134/132（max/mean ≈ 1.07）；总负载 1033、每卡均值 129.125；不复制 e10 时其所在卡负载 ≥ 187（max/mean ≥ 1.45）。来源：构造示例运行输出（/tmp/eplb-verify/eplb_verify.py，算法复刻自 [repo] eplb.py）。状态：已确认（构造示例，非实测数据）。
- N3：V3 训练配置：EP64 跨 8 节点；M=4；γ=0.001（前 14.3T token）→ 0.0（后 500B token）；序列级平衡损失 α=0.0001。来源：[v3] §3.3、§4.3。状态：已确认。
- N4：V3 prefill：4 节点 32 GPU、TP4+SP+DP8、EP32、32 个冗余专家、每卡 8+1 个专家、周期约每 10 分钟。来源：[v3] §3.4.1。状态：已确认（并入 C10）。
- N5：V3 decode：40 节点 320 GPU、TP4+SP+DP80、EP320、每卡 1 专家、64 卡承载冗余+共享、每 token 9 专家、每专家 batch 通常 ≤ 256 token。来源：[v3] §3.4.2。状态：已确认（并入 C11）。
- N6：vLLM 默认参数：window_size=1000、step_interval=3000、num_redundant_experts=0；示例命令 Qwen/Qwen3-30B-A3B 与 deepseek-ai/DeepSeek-V3-0324；DeepSeek-V3 每 rank 1 个冗余 ≈ 2.4 GB；大规模建议 num_redundant_experts=32。来源：[vllm-doc]。状态：已确认（并入 C14/C15）。

## 置信状态汇总

- 存在冲突：无。
- 证据不足：无。
- 标注为推断的论断：C12 中「推理期 bias 不再更新」（由报告描述的更新机制直接得出，页面明确标注推断链）。
- 标注为构造示例的论断：F2、N2（页面正文以「构造示例」标记）。
