# DeepSeek-V3 技术报告摘录（arXiv:2412.19437v2, 2025-02-18）

来源：https://arxiv.org/html/2412.19437v2

## §3.2 开篇（训练框架总述）

> "The training of DeepSeek-V3 is supported by the HAI-LLM framework, an efficient and lightweight training framework crafted by our engineers from the ground up. On the whole, DeepSeek-V3 applies 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)."

> "Secondly, we develop efficient cross-node all-to-all communication kernels to fully utilize IB and NVLink bandwidths and conserve Streaming Multiprocessors (SMs) dedicated to communication."

## §3.2.1 DualPipe（跨节点 EP 的计算通信比）

> "For DeepSeek-V3, the communication overhead introduced by cross-node expert parallelism results in an inefficient computation-to-communication ratio of approximately 1:1. To tackle this challenge, we design an innovative pipeline parallelism algorithm called DualPipe..."

> "The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks. To be specific, we divide each chunk into four components: attention, all-to-all dispatch, MLP, and all-to-all combine."

> Table 2: DualPipe 气泡 = (PP/2−1)(F&B+B−3W)，参数 2×，激活 PP+1。

## §3.2.2 跨节点 All-to-All 通信的高效实现（核心机制章节）

> "In order to ensure sufficient computational performance for DualPipe, we customize efficient cross-node all-to-all communication kernels (including dispatching and combining) to conserve the number of SMs dedicated to communication. The implementation of the kernels is co-designed with the MoE gating algorithm and the network topology of our cluster. To be specific, in our cluster, cross-node GPUs are fully interconnected with IB, and intra-node communications are handled via NVLink. NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s). To effectively leverage the different bandwidths of IB and NVLink, we limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic. For each token, when its routing decision is made, it will first be transmitted via IB to the GPUs with the same in-node index on its target nodes. Once it reaches the target nodes, we will endeavor to ensure that it is instantaneously forwarded via NVLink to specific GPUs that host their target experts, without being blocked by subsequently arriving tokens. In this way, communications via IB and NVLink are fully overlapped, and each token can efficiently select an average of 3.2 experts per node without incurring additional overhead from NVLink. This implies that, although DeepSeek-V3 selects only 8 routed experts in practice, it can scale up this number to a maximum of 13 experts (4 nodes × 3.2 experts/node) while preserving the same communication cost. Overall, under such a communication strategy, only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink."

> "In detail, we employ the warp specialization technique and partition 20 SMs into 10 communication channels. During the dispatching process, (1) IB sending, (2) IB-to-NVLink forwarding, and (3) NVLink receiving are handled by respective warps. The number of warps allocated to each communication task is dynamically adjusted according to the actual workload across all SMs. Similarly, during the combining process, (1) NVLink sending, (2) NVLink-to-IB forwarding and accumulation, and (3) IB receiving and accumulation are also handled by dynamically adjusted warps. In addition, both dispatching and combining kernels overlap with the computation stream, so we also consider their impact on other SM computation kernels. Specifically, we employ customized PTX (Parallel Thread Execution) instructions and auto-tune the communication chunk size, which significantly reduces the use of the L2 cache and the interference to other SMs."

## §2.1.2 Node-Limited Routing

> "Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most M nodes, which are selected according to the sum of the highest Kr/M affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap."

## §3.3.3 低精度通信

> "Communication bandwidth is a critical bottleneck in the training of MoE models. To alleviate this challenge, we quantize the activation before MoE up-projections into FP8 and then apply dispatch components... A similar strategy is applied to the activation gradient before MoE down-projections. For both the forward and backward combine components, we retain them in BF16 to preserve training precision in critical parts of the training pipeline."

## §3.4 Inference and Deployment

> "We deploy DeepSeek-V3 on the H800 cluster, where GPUs within each node are interconnected using NVLink, and all GPUs across the cluster are fully interconnected via IB. To simultaneously ensure both the Service-Level Objective (SLO) for online services and high throughput, we employ the following deployment strategy that separates the prefilling and decoding stages."

### §3.4.1 Prefilling

> "The minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs. ... For the MoE part, we use 32-way Expert Parallelism (EP32), which ensures that each expert processes a sufficiently large batch size, thereby enhancing computational efficiency. For the MoE all-to-all communication, we use the same method as in training: first transferring tokens across nodes via IB, and then forwarding among the intra-node GPUs via NVLink."

> "...we introduce a deployment strategy of redundant experts, which duplicates high-load experts and deploys them redundantly. ... For the deployment of DeepSeek-V3, we set 32 redundant experts for the prefilling stage. For each GPU, besides the original 8 experts it hosts, it will also host one additional redundant expert."

> "Furthermore, in the prefilling stage, to improve the throughput and hide the overhead of all-to-all and TP communication, we simultaneously process two micro-batches with similar computational workloads, overlapping the attention and MoE of one micro-batch with the dispatch and combine of another."

### §3.4.2 Decoding

> "The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs. The attention part employs TP4 with SP, combined with DP80, while the MoE part uses EP320. For the MoE part, each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts. All-To-All communication of the dispatch and combine parts is performed via direct point-to-point transfers over IB to achieve low latency. Additionally, we leverage the IBGDA technology to further minimize latency and enhance communication efficiency."

> "Additionally, to enhance throughput and hide the overhead of all-to-all communication, we are also exploring processing two micro-batches with similar computational workloads simultaneously in the decoding stage. Unlike prefilling, attention consumes a larger portion of time in the decoding stage. Therefore, we overlap the attention of one micro-batch with the dispatch+MoE+combine of another. In the decoding stage, the batch size per expert is relatively small (usually within 256 tokens), and the bottleneck is memory access rather than computation. Since the MoE part only needs to load the parameters of one expert, the memory access overhead is minimal, so using fewer SMs will not significantly affect the overall performance. Therefore, to avoid impacting the computation speed of the attention part, we can allocate only a small portion of SMs to dispatch+MoE+combine."

## §3.5.1 通信硬件建议（SM 的四项通信任务）

> "However, the current communication implementation relies on expensive SMs (e.g., we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose), which will limit the computational throughput. Moreover, using SMs for communication results in significant inefficiencies, as tensor cores remain entirely under-utilized."

> "Currently, the SMs primarily perform the following tasks for all-to-all communication:
> - Forwarding data between the IB (InfiniBand) and NVLink domain while aggregating IB traffic destined for multiple GPUs within the same node from a single GPU.
> - Transporting data between RDMA buffers (registered GPU memory regions) and input/output buffers.
> - Executing reduce operations for all-to-all combine.
> - Managing fine-grained memory layout during chunked data transferring to multiple experts across the IB and NVLink domain."

## §2.1.2 MoE 定义公式（式 12-16）

> "$\mathbf{h}_{t}^{\prime} =\mathbf{u}_{t}+\sum_{i=1}^{N_{s}}{\operatorname{FFN}^{(s)}_{i}\left(\mathbf{u}_{t}\right)}+\sum_{i=1}^{N_{r}}{g_{i,t}\operatorname{FFN}^{(r)}_{i}\left(\mathbf{u}_{t}\right)}$" (Eq. 12)

> "$g_{i,t} =\frac{g^{\prime}_{i,t}}{\sum_{j=1}^{N_{r}}g^{\prime}_{j,t}}$" (Eq. 13)

> "$g^{\prime}_{i,t} = s_{i,t}$ if $s_{i,t}\in\operatorname{Topk}(\{s_{j,t}\},K_{r})$, else 0" (Eq. 14)

> "$s_{i,t} =\operatorname{Sigmoid}\left({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}\right)$" (Eq. 15)

> "$N_s$/$N_r$: shared/routed expert 数；$K_r$: 每 token 激活的路由专家数；$g_{i,t}$: gating value；$s_{i,t}$: token-to-expert affinity。"

超参（V3 模型设置）：$N_s=1$、$N_r=256$、$K_r=8$、node-limited routing $M=4$。

## §3.2.1 DualPipe 补充摘录（"完全隐藏"的原句）

> "As illustrated in Figure 4, for a pair of forward and backward chunks, we rearrange these components and manually adjust the ratio of GPU SMs dedicated to communication versus computation. In this overlapping strategy, we can ensure that both all-to-all and PP communication can be fully hidden during execution."

> （Figure 4 图注）"Both all-to-all and PP communication can be fully hidden."

> （对比：完整双向调度的表述为 "a significant portion of communications can be fully overlapped"。）
