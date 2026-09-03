# DeepEP 内容范围

## 1. 概念歧义处理

状态：已裁定。

DeepEP 在公开语境中特指 DeepSeek 于 2025-02-24 开源（GitHub deepseek-ai/DeepEP）的专家并行通信库，无同名冲突。名称无官方全称展开直到 V2 版 README 才标注 "DeepEP (DeepEveryParallel)"；初始发布时 README 仅自称 "a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP)"。V2 的展开名反映其定位从纯 EP 扩展到 PP/CP/远程内存访问等实验性原语，本文以官方 README 为准。

注意区分：DeepSeek 生态内另有 DeepGEMM（FP8 GEMM 计算库）、DualPipe（流水线算法）、Mooncake（KV cache 传输引擎）等，均非本文对象。社区文献中 "DeepEP" 不会指向其他对象，无消歧负担。

## 2. 概念含义

### 简要定义

DeepEP 是一个 GPU 通信库，为 MoE 模型做专家并行时的 token 搬运（dispatch 与 combine 两方向 all-to-all）提供定制内核：训练与 prefill 用高吞吐内核（NVLink 与 RDMA 分层转发），decode 用低延迟内核（纯 RDMA 与 hook 重叠）。它只做通信，不做路由决策、专家计算与调度。

### 正式定义（与权威来源一致）

- "DeepEP (DeepEveryParallel) is a high-performance communication library for modern machine learning training and inference. The library currently focuses on expert parallelism (EP) — providing high-throughput and low-latency all-to-all GPU kernels (MoE dispatch and combine) with low-precision support including FP8 — while also offering experimental primitives for pipeline parallelism (PP), context parallelism (CP), and remote memory access (Engram), all designed for zero or minimal SM occupation."（V2 README，main 分支 commit 01dc3aa）
- V1 初始 README："DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also as known as MoE dispatch and combine."
- V1 与 DeepSeek-V3 论文的关系："the implementation in this library may have some slight differences from the DeepSeek-V3 paper"（docs/legacy.md）；DeepSeek-V3 论文 §3.2.2 描述了同源设计的跨节点 all-to-all 内核。

### 本文采用的语境

以 V2（2026-04-30 公开发布，commit b306af0 起）为当前状态叙述，机制讲解以 V1 的两套内核设计为主线（V2 保留 hybrid 与 direct 两种模式，机制同源），V2 的接口与工程变化单列一章。

### 包括什么

- MoE dispatch/combine 的通信语义：token 按路由发往专家所在 rank，计算后加权归约回原 rank。属于本概念的核心动作。
- 高吞吐内核的分层转发机制：IB 先传到目标节点同 in-node index 的 GPU，再经 NVLink 转发到持有目标专家的 GPU；两种网络重叠使用。
- 低吞吐延迟权衡的 decode 内核：纯 RDMA、固定槽位接收布局、接收 hook、CUDA graph 兼容。
- SM 占用控制：normal 内核的 SM 数量控制、V1 的 auto-tuning 与 V2 的解析式 SM/QP 计算。
- V2 重构：NCCL Gin 后端、ElasticBuffer 统一接口、EP2048 规模、JIT 编译。
- 与调用方的分工边界：grouped GEMM 做接收侧计算、gating 决定路由、框架层做重叠调度。
- 官方性能数字及其测试条件（V1 normal、V1 low-latency、V2 表）。

### 不包括什么

- 专家计算本身（grouped GEMM、FP8 GEMM）：属于计算库（DeepGEMM 等），DeepEP 只保证接收布局供其直接消费。排除理由：另一独立系统。
- 路由与负载均衡算法（node-limited routing 的训练细节、冗余专家规划、负载感知重路由）：DeepEP 不决定 token 去向。node-limited routing 只作为高吞吐内核的配合条件讲结论，规划类工作引用 ultraep/moonep 页面。
- DualPipe 完整调度算法与推导：V3 报告内容，本页只引用其结论（1:1 计算通信比、四组件分解）说明 DeepEP 的使用场景。
- 安装、环境变量、NCCL/NVSHMEM 依赖安装步骤：工程操作，不影响概念理解。
- PTX 指令级优化（ld.global.nc 等）：实现细节，已在来源中核对但不展开。
- 社区分支（hybrid-ep、antgroup-opt、mori-ep 等）的具体内容：只在相邻概念处提及存在性。

### 相邻概念

- UltraEP（wiki/ultraep）：负载感知的专家冗余与重路由系统，其 token dispatch/combine 底层可用 DeepEP。区别：调度决策系统 vs 传输库。
- MoonEP（wiki/moonep）：冗余专家在线规划 + 静态形状通信方案，保留 DeepEP 的 dispatch/combine 数据路径并扩展。区别：规划与静态化 vs 纯传输。
- NCCL/通用集合通信库（wiki/gpu-communication 通信库章节）：面向通用集合通信原语，不提供 MoE 专家分组布局契约。区别：通用原语 vs MoE 专用 all-to-all。
- DualPipe（V3 §3.2.1）：流水线调度算法，消费 DeepEP 提供的内核实现计算通信重叠。本页引用不展开。
- PD 分离（wiki/moe-serving 第 7 章）：部署形态，DeepEP 的两套内核分别服务 prefill 与 decode 两阶段需求。

## 3. 学习目标

### Q1：MoE 专家并行的 dispatch/combine 通信有什么特殊要求，为什么通用集合通信库不够用？

- 完成答案：读者能说明 dispatch/combine 的通信形态（按路由的动态 all-to-all），指出通用库的四个不适配点——SM 与计算争抢（V3 用 20/132 个 SM 做通信）、接收侧无专家分组布局契约（grouped GEMM 无法直接消费）、低精度（FP8 dispatch）非标准需求、decode 小 batch 的延迟与 CUDA graph 兼容——并说明跨节点 EP 通信与计算之比约 1:1 的背景。
- 为什么是核心目标：不理解"哪里不够用"就无法理解 DeepEP 每个设计决策的动机。
- 依赖内容：专家并行与 all-to-all 语义（moe-serving）、NVLink/RDMA 硬件（gpu-communication）、SM 概念（gpu-execution-model）。

### Q2：高吞吐内核如何利用 NVLink 与 RDMA 两种带宽，把跨节点 dispatch/combine 跑满？

- 完成答案：读者能复述两段转发路径（token 先经 IB 到目标节点上与本 rank 同 in-node index 的 GPU，再经 NVLink 转发给持有目标专家的 GPU），说明两种网络为何能完全重叠、node-limited routing（每 token 至多 4 节点）如何限制 IB 流量（平均 3.2 专家/节点、等效最多 13 专家）、SM 数量如何被控制（warp 专职三类通道），并复述关键带宽数字（NVLink 160 GB/s vs IB 50 GB/s、内核实测接近峰值）。
- 为什么是核心目标：这是 DeepEP 最核心的机制创新，也是"非对称域带宽转发"一词的所指。
- 依赖内容：NVLink/NVSwitch/RDMA/IB 传输路径（gpu-communication）、MoE 路由（deepseek-moe）。

### Q3：decode 场景为什么需要另一套低延迟内核，它的纯 RDMA、固定槽位与 hook 各解决什么问题？

- 完成答案：读者能说明 decode 通信的特点（每步小 batch、每步都发生、TPOT 敏感、CUDA graph 需要静态形状），解释低延迟内核的三个关键取舍——放弃 NVLink 走纯 RDMA（所有 rank 经 RDMA 可见，简化路径）、固定大小槽位布局与 mask（不做 CPU 接收计数同步，兼容 CUDA graph）、返回接收 hook 让 RDMA 在后台完成（通信等待不占 SM）——并用延迟/带宽数字说明它用带宽利用率换延迟。
- 为什么是核心目标：低延迟内核是 decode 推理的关键路径，也是 DeepEP 与纯训练库的分水岭；固定槽位与最坏情况预留是理解 buffer 开销和 MoonEP 等后续工作的基础。
- 依赖内容：prefill/decode 两阶段（moe-serving）、CUDA graph（vllm-cudagraph）。

### Q4：V2 重构改变了什么——为什么换掉 NVSHMEM，统一接口与解析式调参解决什么问题？

- 完成答案：读者能列举 V2 的核心变化（NCCL Gin 后端替换 NVSHMEM、ElasticBuffer 统一高吞吐与低延迟 API、解析式 SM/QP 计算取代 auto-tuning、EP2048 规模、JIT 编译、SM 从 24 降到 4-6），说明各项变化的动机（复用现有 NCCL communicator、调参成本、规模扩展），以及代价（buffer 消耗更大、0 SM RDMA 低延迟 EP 不再支持）。
- 为什么是核心目标：V2 是当前主线版本，读者读到的仓库与文档都以 V2 为准；不理解 V1/V2 差异会把两套文档的机制混为一谈。
- 依赖内容：Q1-Q3 的 V1 机制。

### Q5：DeepEP 解决什么、不解决什么，与相邻系统如何分工？

- 完成答案：读者能划出 DeepEP 的责任边界——只做 token 搬运（dispatch/combine 内核）与接收布局，不做路由决策（gating 决定）、不做负载均衡（冗余专家/重路由是另一层，见 ultraep/moonep 与 V3 报告 redundant experts）、不做专家计算（grouped GEMM 配套）、不做重叠调度（DualPipe/TBO 由框架层做）；能说明硬件门槛（Hopper 及以上、节点内 NVLink、跨节点 RDMA 网络、IB 全测试 RoCE 理论兼容）。
- 为什么是核心目标：边界不清会导致把 DeepEP 当成"MoE 推理框架"或"负载均衡方案"的常见误解。
- 依赖内容：Q1-Q4 全部、相邻概念页面。

## 4. 内容分级

### 核心内容

- dispatch/combine 的通信语义与 all-to-all 形态（Q1）。必须说明：数据流方向、top-k 加权归约的数学形式。
- 通用集合通信库不适配的四个点（Q1）。必须说明：SM 争抢（20/132、tensor core 闲置）、布局契约、FP8、CUDA graph/延迟。
- DeepEP 定位：通信库、两套内核、与 V3 论文实现的关系（Q1）。
- 高吞吐内核两段转发机制与两网重叠（Q2）。必须说明：IB 段与 NVLink 段各自职责、同 in-node index 落点、"不被后到 token 阻塞"。
- node-limited routing 配合与 3.2 专家/节点、等效 13 专家的推导背景（Q2）。
- SM 控制与 warp 专职通道（Q2）。必须说明：dispatch 侧三类 warp（IB 发送/IB→NVLink 转发/NVLink 接收）、combine 侧三类、动态调整。
- 高吞吐内核性能数字与测试条件（Q2）。
- decode 通信特点（小 batch、每步发生、TPOT、CUDA graph）（Q3）。
- 低延迟内核机制：纯 RDMA、固定槽位布局与 mask、接收 hook（Q3）。必须说明：接收张量形状 [num_local_experts, num_max_dispatch_tokens_per_rank × num_ranks, hidden] 的含义、无 CPU 同步为何兼容 CUDA graph、hook 语义（只发请求不等待）。
- 低延迟内核延迟/带宽数字与测试条件（Q3）。
- V2 变化清单：Gin 后端、ElasticBuffer、解析式 SM/QP、EP2048、JIT、SM 4-6、buffer 变大、0 SM LL 移除（Q4）。
- V2 性能数字与测试条件（Q4）。
- 责任边界与相邻系统分工、硬件门槛（Q5）。

### 辅助内容

- V1 normal dispatch 的 CPU 等待（隐式等待 GPU 接收计数信号，导致 V1 normal 与 CUDA graph 不兼容）：澄清"DeepEP 都兼容 CUDA graph"的误解，服务 Q1/Q3。
- FP8 dispatch + BF16 combine 的精度策略（V3 §3.3.3）：服务 Q1 的"低精度"点。
- V3 部署数字（prefill EP32、decode EP320、每 GPU 1 专家、IBGDA）：为两套内核的适用场景提供真实规模，服务 Q2/Q3。
- 低延迟模式 buffer 双缓冲限制（同一时刻最多持有 2 个 LL 结果张量）：服务 Q3 的机制完整性。
- V2 hybrid 与 direct 模式的区别（分层 RDMA+NVLink vs 直达）：服务 Q4。
- 网络配置要点（VL 隔离、adaptive routing、拥塞控制禁用）：服务 Q5 的边界条件。
- 时间线（2025-02-24 V1 发布、2026-04-30 V2 公开）：服务 Q4 的版本语境。

### 扩展内容

- Engram（远程 KV cache 拉取）、0 SM PP、0 SM CP 实验特性：纳入，一句话级提及（Q4 边界内）。
- 社区分支与 fork（hybrid-ep、antgroup-opt、mori-ep、uccl 等）：排除，只在不展开的列表中提及存在。
- PTX undefined-behavior 用法：排除，不影响概念理解。
- NVSHMEM 安装细节：排除。
- Eager/Zero-copy 实验分支：排除。

## 5. 前置知识映射

全部前置概念在 wiki/ 下已有页面，无需递归生成：

| 前置概念 | 页面 | 被哪些学习目标依赖 |
|---|---|---|
| MoE 结构、top-k 路由、共享专家 | wiki/deepseek-moe | Q1（dispatch 语义） |
| 专家并行、all-to-all 搬运、TBO/SBO、prefill/decode、PD 分离 | wiki/moe-serving | Q1、Q3 |
| NVLink/NVSwitch、RDMA/IB/RoCE、GPUDirect、通信库分工 | wiki/gpu-communication | Q1、Q2、Q5 |
| CUDA graph 录制重放 | wiki/vllm-cudagraph | Q1、Q3 |
| FP8（E4M3）块量化 | wiki/fp8-block-quant | Q1（低精度点） |
| TP/PP 基础 | wiki/model-parallelism | Q1（背景）、Q4（0 SM PP 提及） |
| 负载感知冗余/重路由（相邻） | wiki/ultraep | Q5 |
| 冗余专家规划/静态形状（相邻） | wiki/moonep | Q5 |

## 6. 明确不展开的内容

- DualPipe 的完整调度与气泡推导：它是流水线算法，本页只引用"1:1 计算通信比"与"chunk 四组件"作为 DeepEP 的使用场景与动机，完整机制属于 V3 训练系统话题。
- node-limited routing 的亲和分数机制细节：只讲结论（限制目标节点数以省 IB 流量），训练层面的 gating 设计属于 deepseek-moe/V3 论文话题。
- grouped GEMM 的内核实现：接收侧计算属于计算库；本页只讲"接收布局按专家分组连续排布"这一契约。
- NCCL Gin 后端的内部协议（GDAKI 等）：属于 NCCL 内部实现。
- RDMA 硬件细节（QP 状态机、verbs）：gpu-communication 已覆盖 RDMA 传输路径，本页不重复。

## 7. 常见误解和适用边界

### 误解

1. 错误理解："DeepEP 是一个 MoE 推理框架/引擎。" 正确结论：DeepEP 是通信库，只提供 dispatch/combine 及少量实验通信原语；专家计算由 grouped GEMM 类库完成，路由由模型 gating 决定，重叠调度由训练/推理框架完成。形成原因：DeepEP 常与推理框架（vLLM/SGLang 集成）一起被提及。影响 Q1、Q5。
2. 错误理解："低延迟内核就是高吞吐内核的小 batch 版/加速版。" 正确结论：两套内核是不同设计点：高吞吐内核用分层转发最大化带宽利用率（节点内 153/158 GB/s 接近 NVLink 峰值），低延迟内核放弃 NVLink、用固定槽位与纯 RDMA 把延迟压到百微秒级，但 RDMA 带宽利用率显著低于高吞吐模式（EP128 时 39 GB/s 对比 internode 43-58 GB/s）。形成原因：两者都叫 dispatch/combine。影响 Q3。
3. 错误理解："用了 DeepEP，MoE 的负载不均衡就解决了。" 正确结论：DeepEP 不决定 token 去向；路由不均时接收槽位照样空置或拥挤（buffer 按最坏情况预留），负载均衡由 gating 算法、冗余专家（V3 部署）或专门系统（UltraEP/MoonEP）负责。影响 Q5。
4. 错误理解："NCCL 也有 all-to-all，DeepEP 只是快一点。" 正确结论：差异是结构性的：接收侧布局契约（按专家分组连续排布供 grouped GEMM）、FP8 dispatch、SM 占用控制、CUDA graph 兼容与零等待 hook 都是通用库不提供的接口语义，不只是带宽数字。影响 Q1。
5. 错误理解："V2 把 V1 的机制全部推翻了。" 正确结论：V2 保留 hybrid（分层 NVLink+RDMA）与 direct 两种模式，两段转发思想延续；变化主要在通信后端（NVSHMEM→NCCL Gin）、接口统一（ElasticBuffer）、调参方式（解析式）与规模（EP2048）；个别能力被移除（0 SM RDMA 低延迟 EP）。影响 Q4。

### 适用边界

- DeepEP 解决：MoE 专家并行 dispatch/combine 的高效传输、接收侧专家分组布局、通信 SM 占用控制、FP8 dispatch。不解决：路由决策、负载均衡、专家计算、调度重叠（由框架配合完成）。
- 结论成立条件：Hopper（SM90）及以上 GPU（V1 曾支持 Ampere SM80 节点内）、节点内 NVLink、跨节点 RDMA 网络（IB 全测试，RoCE 理论兼容）；低吞吐延迟模式要求所有 rank 经 RDMA 可见。
- 条件不满足时：无 RDMA 网络则跨节点高吞吐/低延迟内核不可用（V1 无 NVSHMEM_DIR 时禁用跨节点与低延迟特性）；非 Hopper 架构不在官方支持列表；RoCE 未经完整测试需自行验证。
- V2 的代价：buffer 消耗大于 V1；0 SM RDMA 低延迟 EP 不再支持。

## 8. 完成状态

- 概念歧义：已裁定，无无法消歧项。
- 学习目标 5 个（Q1-Q5），每个有书面完成答案（上文）。
- 核心内容均映射到学习目标；前置知识 8 项全部有已有页面。
- 误解 5 条、边界条件具体可查。
