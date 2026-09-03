# DeepEP 核心论断与证据

来源文件（均在 `wiki/deepep/research/sources/`）：

- R-README = `deepep-readme-v2.md`（DeepEP main 分支 README，commit 01dc3aa，2026-08-04）
- R-LEGACY = `deepep-legacy-v1.md`（DeepEP V1 归档文档 docs/legacy.md）
- R-SRC = `deepep-src-extracts.md`（源码 docstring 摘录：初始 commit ebfe47e 的 deep_ep/buffer.py、V2 deep_ep/buffers/elastic.py 的 docstring、初始 README）
- R-V3 = `v3-report-extracts.md`（DeepSeek-V3 技术报告 arXiv:2412.19437v2 摘录）
- R-REPO = 仓库本地克隆 `/tmp/deepep-research/repo`（commit 01dc3aa），git 历史可查

版本说明：V1 初始 README（2025-02-24）的性能表与 legacy.md 的表数字不同（后者为更新后的测量），本文一律采用 legacy.md 的数字。

## C 论断（定义与机制）

- C1：DeepEP 是面向现代机器学习训练与推理的高性能通信库，当前聚焦专家并行——提供高吞吐与低延迟的 all-to-all GPU 内核（MoE dispatch 与 combine），支持包括 FP8 在内的低精度，另提供 PP、CP、远程内存访问（Engram）的实验性原语，均按零或最小 SM 占用设计。来源：R-README 开头段。条件：无。状态：已确认。
- C2：V1 于 2025-02-24 发布（git initial commit ebfe47e），V2 于 2026-04-30 公开发布（commit b306af0 "[Public release 26/04] Introducing EPv2"）。来源：R-REPO git log。状态：已确认。
- C3：V1 实现与 DeepSeek-V3 论文的描述可能存在细微差异。来源：R-LEGACY "Notice: the implementation in this library may have some slight differences from the DeepSeek-V3 paper."。状态：已确认。
- C4：DeepEP 的高吞吐内核面向非对称域带宽转发（如从 NVLink 域向 RDMA 域转发数据）优化，与 DeepSeek-V3 提出的 group-limited（node-limited）gating 算法对齐。来源：R-LEGACY 开头段。状态：已确认。
- C5：两段转发机制——token 的路由决定后，先经 IB 传到目标节点上与本 rank 同 in-node index 的 GPU；到达目标节点后，立即经 NVLink 转发给持有目标专家的 GPU，不被后到的 token 阻塞；由此 IB 与 NVLink 通信完全重叠。来源：R-V3 §3.2.2（原文见摘录）。条件：跨节点全互联 IB + 节点内 NVLink 的集群。状态：已确认。
- C6：V3 用 node-limited routing 限制每 token 至多发送到 $M$ 个节点（节点按各节点上分数最高的 $K_r/M$ 个专家的亲和分数之和选择）；$M=4$ 用于减少 IB 流量。来源：R-V3 §2.1.2、§3.2.2。状态：已确认。
- C7：在该通信策略下，每 token 平均可选 3.2 个专家/节点而不产生额外 NVLink 开销；V3 实际激活 8 个路由专家，通信成本不变的前提下最多可扩到 13 个（4 节点 × 3.2 专家/节点）。来源：R-V3 §3.2.2。状态：已确认。
- C8：V3 的内核设计只需 20 个 SM 即可跑满 IB 与 NVLink 带宽；用 warp specialization 把 20 SM 分成 10 个通信通道；dispatch 侧 (1) IB 发送 (2) IB→NVLink 转发 (3) NVLink 接收由各自 warp 负责，combine 侧 (1) NVLink 发送 (2) NVLink→IB 转发与累加 (3) IB 接收与累加同样由动态调整的 warp 负责。来源：R-V3 §3.2.2。状态：已确认。
- C9：SM 在 V3 的 all-to-all 通信中承担四类任务：IB↔NVLink 域之间的数据转发（并把发往同节点多 GPU 的 IB 流量从单 GPU 聚合）、RDMA buffer 与输入输出 buffer 之间的搬运、combine 的归约运算、分块传输到多专家时的细粒度内存布局管理。来源：R-V3 §3.5.1。状态：已确认。
- C10：V1 normal 内核支持 SM 数量控制（示例代码 `Buffer.set_num_sms(24)`）；V2 用 `ElasticBuffer.get_theoretical_num_sms` 基于带宽建模解析计算最优 SM 数（输入含 RDMA 带宽、NVLink 带宽、单 SM HBM 读/写带宽，假设均衡 gate 分布），不再需要 auto-tuning。源码注释明确该函数只针对均衡 gate，"For V3.0's group-limited gate, please do not use this function"（标记为待支持）。来源：R-LEGACY 示例代码、R-README EPv2 条目 "Analytical SM & QP count calculation — no more auto-tuning needed"、R-SRC elastic.py get_theoretical_num_sms docstring 与内部注释（见 deepep-src-extracts.md 第 6 节）。状态：已确认。
- C11：V1 normal dispatch 内核内部包含隐式的 CPU 等待——等待 GPU 接收计数信号到达（因为发送侧事先不知道本 rank 将收到多少 token），因此 V1 normal 内核与 CUDA graph 不兼容（除非节点内使用 `num_worst_tokens`）。来源：R-LEGACY "Moreover, inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved" 及示例代码注释。状态：已确认。
- C12：V1 低延迟内核基于 IBGDA 的纯 RDMA 实现：要求所有 rank（无论节点内还是跨节点）都经 RDMA 可见，为简化连节点内也完全禁用 NVLink。V1 源码把低延迟模式的 IBGDA 释义为 "no package forwarding between NVLink and RDMA"，并将 NIC handler 设为 GPU 侧（NVSHMEM_IBGDA_NIC_HANDLER=gpu）。来源：R-SRC buffer.py low_latency_dispatch docstring（"requires all the ranks (no matter intranode or internode) should be visible via RDMA... NVLink are fully disabled for simplicity"）与初始化代码注释（见 deepep-src-extracts.md 第 5 节）、R-V3 §3.4.2 "direct point-to-point transfers over IB... leverage the IBGDA technology"。状态：已确认。
- C13：V1 低延迟 dispatch 返回的接收张量形状为 $[\text{num\_local\_experts},\ \text{num\_max\_dispatch\_tokens\_per\_rank}\times\text{num\_ranks},\ \text{hidden}]$（FP8 e4m3 数据 + 每 128 元素一组的 scale，scale 列主序为 TMA 兼容），槽位不全部有效（不做 CPU 接收计数同步），recv_count 给出每个本地专家的实际接收数。分段规则：接收行按来源 rank 分段，段基址 = 来源 rank × num_max，段内下标为该来源 rank 发往此专家的 token 到达序号——见初始 commit csrc/kernels/internode_ll.cu 的槽位偏移计算（dst_expert_local_idx * num_ranks * num_max + rank * num_max + slot_idx，slot_idx 来自发送 rank 本地 workspace 的按专家原子计数；摘录见 deepep-src-extracts.md 第 7 节）。来源：R-SRC buffer.py docstring 与 internode_ll.cu。状态：已确认。
- C14：V1 低延迟内核与 CUDA graph 兼容（replay 时需恢复部分 buffer 状态）；低延迟 dispatch 隐式做 FP8 转换。来源：R-SRC buffer.py docstring（"not incompatible with CUDA graph"、"with implicit FP8 casting"）、R-LEGACY 代码注释。状态：已确认。
- C15：hook 机制——`return_recv_hook=True` 时，dispatch/combine 内核只发出 RDMA 请求而不实际等待数据到达；调用返回的 hook 后才保证数据就位；该通信-计算重叠方法不占用任何 SM 资源。来源：R-SRC buffer.py docstring（"the kernel will just do the RDMA request issues, but without actually receiving the data. You must call the received hook"）、R-README/R-LEGACY 开头（"a hook-based communication-computation overlapping method that does not occupy any SM resource"）。条件：等待期间 GPU 转去做其他计算；V2 起该形式（0 SM RDMA low-latency EP）不再支持。状态：已确认。
- C16：V1 低延迟模式同一时刻最多持有 2 个内核结果张量（内部只有两个 buffer，返回张量复用 buffer）。来源：R-SRC buffer.py docstring warning（"as there are only two buffers... you can not hold more than 2 low-latency kernels' result tensor at a single moment"）。状态：已确认。
- C17：V1 低延迟 buffer 大小以 `num_max_dispatch_tokens_per_rank`（解码引擎的实际 batch size）预留，官方建议该值小于 256；低延迟模式消耗的显存远大于 normal 模式；QP 数必须等于本地专家数（`num_qps_per_rank = num_experts // group.size()`）。来源：R-LEGACY get_buffer 注释与代码注释。状态：已确认。
- C18：V2 从 NVSHMEM 后端切换到 NCCL Gin 后端（header-only、轻量、可复用现有 NCCL communicator）；EPv2 通信路径走 Gin，但 V2 安装仍依赖 NVSHMEM 以支持 legacy 方法（README "Install NVSHMEM dependency: DeepEP also depends on NVSHMEM to provide support for legacy methods"）。来源：R-README News 与安装节。状态：已确认。
- C19：V2 把高吞吐与低延迟 API 统一进单一 `ElasticBuffer` 接口，带新 GEMM 布局；支持更大 scale-up/scale-out 域（至 EP2048）；hybrid 与 direct 两种模式都保留。来源：R-README EPv2 条目。状态：已确认。
- C20：V2 对 V3 式传统训练把通信 SM 用量从 24 降到 4-6，性能相当或更好；对比 V1 峰值性能最高 1.3 倍、SM 数最多省 4 倍。来源：R-README EPv2 条目与 Performance 节。状态：已确认。
- C21：V2 的代价：buffer 消耗大于 V1；不再支持 0 SM RDMA 低延迟 EP；Engram/PP/CP 为实验特性。来源：R-README Notes。状态：已确认。
- C22：V2 全部内核经轻量 JIT 模块在运行时编译，安装时无需 CUDA 编译。来源：R-README 开头段。状态：已确认。
- C23：V2 hybrid 模式用分层 RDMA+NVLink 通信获得更高带宽，对 multi-plane/multi-rail 网络更友好；direct 模式为非分层直达。来源：R-SRC elastic.py ElasticBuffer docstring（allow_hybrid_mode 字段）。状态：已确认。
- C24：V2 decode 支持 handle 缓存：gating 决策不变时复用缓存 handle 的布局，跳过布局重算与 CPU 同步。来源：R-README decoding 示例注释（"Reusing cached handle: skip layout recomputation and CPU sync"）。状态：已确认。
- C25：网络配置：DeepEP 在 InfiniBand 网络上完整测试，理论上兼容 RoCE；建议用虚拟路（VL）隔离不同类型流量；V2 建议在所有网络负载条件下开启自适应路由（虽引入额外延迟）；拥塞控制被禁用（损害最大带宽）。来源：R-README Network configurations。状态：已确认。
- C26：硬件与依赖门槛：V2 要求 Hopper（SM90）或支持 SM90 PTX ISA 的架构、PyTorch 2.10+、NCCL 2.30.4+、节点内 NVLink、跨节点 RDMA 网络；V1 曾支持 Ampere（SM80，仅节点内）与 PyTorch 2.1+。来源：R-README Requirements、R-LEGACY Requirements。状态：已确认。
- C27：V3 prefill 部署：最小单元 4 节点 32 GPU，MoE 用 EP32；all-to-all 与训练相同（先 IB 跨节点、再节点内 NVLink 转发）；同时处理两个 micro-batch，一个的 attention/MoE 与另一个的 dispatch/combine 重叠；预填充阶段设 32 个冗余专家（每 GPU 原 8 专家 + 1 冗余）。来源：R-V3 §3.4.1。状态：已确认。
- C28：V3 decode 部署：最小单元 40 节点 320 GPU，attention 用 TP4+SP 与 DP80，MoE 用 EP320；每 GPU 恰好 1 个专家，64 个 GPU 承载冗余专家与共享专家；dispatch/combine 的 all-to-all 用 IB 直接点对点传输换低延迟，并用 IBGDA 进一步降延迟；decode 每 expert 的 batch size 较小（通常 256 token 以内）、瓶颈是访存而非计算，因此只给 dispatch+MoE+combine 分配少量 SM 不显著影响整体性能；decode 同样探索两个 micro-batch（一个的 attention 与另一个的 dispatch+MoE+combine 重叠）。来源：R-V3 §3.4.2。状态：已确认。
- C29：DualPipe 的动机：跨节点 EP 使计算通信比约为 1:1；DualPipe 把每个 chunk 分为 attention、all-to-all dispatch、MLP、all-to-all combine 四个组件，在前后向 chunk 对内部重排组件并手工调整通信/计算 SM 配比；论文表述为在该重叠策略下 "both all-to-all and PP communication can be fully hidden during execution"（完整双向调度层面的措辞是 "a significant portion of communications can be fully overlapped"）。来源：R-V3 §3.2.1（含补充摘录）。状态：已确认。
- C30：低精度通信策略：V3 把 MoE up 投影前的激活量化为 FP8 再做 dispatch；反向的激活梯度同样处理；前后向 combine 保留 BF16 以保住训练关键路径的精度。来源：R-V3 §3.3.3。状态：已确认。
- C31：接收侧布局为配套 GEMM 库设计：V1 README 示例注释 "Later, you can use our GEMM library to do the computation with this specific format"；V2 handle 提供 `num_recv_tokens_per_expert_list` 供 GEMM 使用。来源：R-LEGACY 示例代码注释、R-README dispatch_forward 注释。状态：已确认。
- C32：V2 实验性原语：Engram = 基于 RDMA 的远程内存访问（远程条目拉取）；0 SM PP（RDMA）；0 SM CP（Copy Engine）。来源：R-README News、R-SRC elastic.py engram_fetch docstring。状态：已确认。

## F 公式

- F1：MoE 层输出 $\mathbf{h}_{t}^{\prime} =\mathbf{u}_{t}+\sum_{i=1}^{N_{s}}\operatorname{FFN}^{(s)}_{i}(\mathbf{u}_{t})+\sum_{i=1}^{N_{r}}g_{i,t}\operatorname{FFN}^{(r)}_{i}(\mathbf{u}_{t})$，其中 $g_{i,t}$ 为选中专家归一化后的门控值（Eq. 12-14）。combine 的通信语义即式中路由专家加权和：各专家副本返回原 rank 后按 $g_{i,t}$ 加权求和。来源：R-V3 §2.1.2 式 12-14。状态：已确认。
- F2：dispatch 通信量（构造性推导，非外部来源）：一个 token 被路由到 $K$ 个专家即复制为 $K$ 份，每份载荷 $h$ 个元素；BF16 每 token 通信量 $2Kh$ 字节，FP8 为 $Kh$ 字节。由 top-K 路由语义直接得出。状态：推导（服务正文计算示例）。
- F3：等效专家数上限 $4 \times 3.2 = 12.8 \approx 13$。来源：R-V3 §3.2.2（原文 "a maximum of 13 experts (4 nodes × 3.2 experts/node)"）。状态：已确认。
- F4：低延迟接收槽位总数 $\text{num\_local\_experts}\times\text{num\_max\_dispatch\_tokens\_per\_rank}\times\text{num\_ranks}\times h$，按最坏情况预留。来源：R-SRC buffer.py docstring 返回形状。状态：已确认。

## N 数字（外部数字与实验条件）

- N1：H800 节点内 NVLink 最大带宽约 160 GB/s；每 GPU 接 CX7 InfiniBand 400 Gb/s 网卡，RDMA 最大带宽约 50 GB/s；NVLink 约为 IB 的 3.2 倍。来源：R-LEGACY Performance 节、R-V3 §3.2.2。状态：已确认。
- N2：V1 normal 内核实测（H800，4096 tokens/batch、hidden 7168、top-4 组合 top-8 专家、FP8 dispatch + BF16 combine）：节点内 EP8 dispatch 153 GB/s（NVLink）、combine 158 GB/s；跨节点 EP16 43/43 GB/s、EP32 58/57 GB/s、EP64 51/50 GB/s（均 RDMA 瓶颈）。来源：R-LEGACY Performance。状态：已确认。
- N3：V1 低延迟内核实测（H800，128 tokens/batch、hidden 7168、top-8、FP8 dispatch + BF16 combine）：dispatch EP8 77 µs（98 GB/s）→ EP128 192 µs（39 GB/s）、EP256 194 µs（39 GB/s）；combine EP8 114 µs（127 GB/s）→ EP128 369 µs（39 GB/s）、EP256 360 µs（40 GB/s）。来源：R-LEGACY Performance。状态：已确认。
- N4：V2 实测（按 V3 配置：8K tokens/batch、hidden 7168、top-8 专家、FP8 dispatch + BF16 combine，逻辑带宽含本 rank 流量）：SM90 CX7 EP8×2 dispatch 90 / combine 81 GB/s（RDMA 瓶颈）用 12 SM；EP8×4 61/61 GB/s 用 6 SM；SM100 单节点 EP8（NVLink）726/740 GB/s 用 64 SM（最大性能）、643/675 GB/s 用 24 SM（最小 SM 数）。来源：R-README Performance。状态：已确认。
- N5：SM 预算：V3 训练用 20/132 个 H800 SM 做通信（§3.5.1）；V1 normal 示例设 24 SM；V2 对 V3 式训练 SM 24→4-6，V2 相对 V1 峰值最高 1.3 倍、SM 最多省 4 倍。来源：R-V3 §3.5.1、R-LEGACY、R-README。状态：已确认。
- N6：V3 部署规模：prefill 最小单元 4 节点 32 GPU（EP32、32 冗余专家、每 GPU 8+1 专家）；decode 最小单元 40 节点 320 GPU（EP320、每 GPU 1 专家、64 GPU 载冗余+共享专家）。来源：R-V3 §3.4.1/§3.4.2。状态：已确认。
- N7：V3 模型超参：$N_s=1$ 共享专家、$N_r=256$ 路由专家、每 token 激活 $K_r=8$ 个路由专家、node-limited $M=4$ 节点。来源：R-V3 §2.1.2 及模型设置。状态：已确认。

## 冲突与不足

- V1 初始 README 与 legacy.md 的性能数字不一致（EP32 44/47 vs 58/57 GB/s；LL EP8 163 µs vs 77 µs）。处理：采用 legacy.md（V1 归档文档的最终版测量），页面不展示两版差异。
- V2 性能表只有 EP8×2 / EP8×4 / EP8 三种拓扑，更大 EP 配置官方未公布数字（README "We omit results for larger EP configurations"）。页面引用时不外推。
- DeepEP 与推理框架（vLLM/SGLang 等）的集成状态：README 未系统描述，不作为核心论断引用；UltraEP 论文提及自身 token 搬运用 DeepEP hybrid-ep 分支（经 wiki/ultraep 页面转述），只在相邻概念处作定性提及。
