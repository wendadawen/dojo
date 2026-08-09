# GPU 执行模型与 kernel 调度 evidence：核心论断与证据

来源缩写：
- [PG] CUDA C++ Programming Guide（docs.nvidia.com/cuda/cuda-c-programming-guide），2026-08 访问的现行版本
- [HTG] NVIDIA Hopper Tuning Guide（docs.nvidia.com/cuda/hopper-tuning-guide，与 12.3/12.5 归档 PDF 一致）
- [H100] NVIDIA H100 Tensor Core GPU Architecture White Paper / NVIDIA Hopper Architecture In-Depth（developer.nvidia.com 官方博客与白皮书一致：132 SM、528 Tensor Core、80GB HBM3、3.35 TB/s、50MB L2）
- [DRV] CUDA Driver API 文档（docs.nvidia.com/cuda/cuda-driver-api）
- [MPS] NVIDIA Multi-Process Service 官方文档（docs.nvidia.com/deploy/mps）
- [MIG] NVIDIA MIG 官方资料（docs.nvidia.com MIG User Guide 及 vGPU 参考文档，与 AWS/Azure 文档交叉一致）

## C 论断

- C1：kernel 以 grid 启动，grid 由若干 CTA（thread block）组成；每个 CTA 在同一时刻整体驻留在同一块 SM 上执行；一个 CTA 最多 1024 个线程。来源：[PG] Programming Model / Thread Hierarchy；Compute Capabilities 表。条件：CUDA 全代际。状态：已确认。
- C2：warp 是 32 个线程一组的执行单位，SM 以 warp 为单位调度发射指令；Hopper 每 SM 最多常驻 64 个 warp。来源：[PG] Hardware Multithreading / SIMT；[HTG] Occupancy（Blackwell Tuning Guide 同句确认 Hopper 为 64）。状态：已确认。
- C3：CTA 被分发到 SM 后运行至完成才释放资源；CUDA 不提供用户可控的运行中 kernel 抢占接口（优先级文档明示不抢占，见 C14）。来源：[PG] + [DRV] cuStreamCreateWithPriority 语义反推。条件：默认执行模型。状态：已确认（"运行至完成"为编程模型语义；硬件调度器细节未公开，页面不展开）。
- C4：同一 stream 内的操作按提交顺序执行；不同 stream 之间无顺序保证、可并发（受资源限制）。来源：[PG] Asynchronous Concurrent Execution / Streams。状态：已确认。
- C5：H100 SXM5 有 132 个 SM、每 SM 4 个第四代 Tensor Core、50MB L2、80GB HBM3 @ 3.35 TB/s。来源：[H100]。状态：已确认。
- C6：Hopper 每 SM shared memory 容量 228KB，单个 CTA 最多可用 227KB。来源：[HTG] §4.1.1（Blackwell Tuning Guide 同句）。状态：已确认。
- C7：TMA（Tensor Memory Accelerator）是 Hopper 引入的异步拷贝引擎：单线程即可发起 1D–5D 张量在全局内存与 shared memory 之间的大块双向传输，传输期间线程可做别的计算；可在 cluster 内不同 SM 的 shared memory 之间传输并支持组播（multicast）。来源：[HTG] §4.1.2；[PG] TMA 章节。状态：已确认。
- C8：Hopper 引入 thread block cluster：同一 cluster 内的 CTA 被共同调度到同一 GPC 内相邻 SM 上，可直接读写彼此 shared memory（DSMEM，Distributed Shared Memory）；可移植 cluster 最大 8 个 CTA，H100 经 opt-in（cudaFuncAttributeNonPortableClusterSizeAllowed）可到 16；更大 cluster 可能降低活跃 block 数。来源：[HTG] §4.1.3；[PG] Thread Block Clusters。状态：已确认。
- C9：TMA 使 warp specialization 成为可能/被官方推荐：部分 warp 专职数据搬运、其余 warp 专职计算。来源：[HTG] §4.1.2（"Enables users to write warp specialized codes…"）。状态：已确认。
- C10：CUDA Graph 把一串 kernel/拷贝操作录制为图，实例化后可整体重复提交（重放），消除逐次 CPU 提交开销；图内节点拓扑在捕获时固定。来源：[PG] CUDA Graphs（"Graphs…enable work to be defined once and launched repeatedly"；捕获约束章节）。状态：已确认。
- C11：stream 优先级（cudaStreamCreateWithPriority / cuStreamCreateWithPriority）"provides a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order"。来源：[DRV] cuStreamCreateWithPriority 描述原文。状态：已确认。
- C12：MPS（Multi-Process Service）让多个进程的 kernel 在同一 GPU 上真正并发执行（Volta 起）；active thread percentage 等限额"does not reserve dedicated resources"，只是限制某客户端可用资源上限，不同客户端的 kernel 可能落到同一 SM。来源：[MPS] Execution Resource Provisioning 原文（"Setting the limit does not reserve dedicated resources for any MPS client context…Kernels launched from different MPS client contexts may execute on the same SM"）。状态：已确认。
- C13：H100（80GB）的 MIG profile 只有以下档位：1g.10gb（最多 7 实例）、1g.20gb（最多 4）、2g.20gb（最多 3）、3g.40gb（最多 2）、4g.40gb（最多 1）、7g.80gb（整卡）；MIG 提供 SM/显存/缓存的硬件级隔离，创建/修改分区需要 GPU 空闲（通常需停负载、甚至重启生效）。来源：[MIG]（vGPU 参考文档 H100 SXM5 表 + NVIDIA MIG 官方页面 "7x 10GB / 4x 20GB / 2x 40GB / 1x 80GB"）。条件：H100 80GB SXM/PCIe。状态：已确认。
- C14：Green Context（CUDA 12.4+）在进程内把 GPU 的 SM 集合切分为空间分区，kernel 经其 stream 只落在分区内的 SM 上；Hopper（CC 9.0+）上每分区最少 8 个 SM 且为 8 的倍数；官方明示"即使分区互不相交，也不保证两个 green context 的 kernel 并发执行或有前进保证"；资源在创建时固定。来源：[DRV] §6.35 Green Contexts（12.8/13.1 版本一致）。状态：已确认。
- C15：H100 的 cluster 上限与 GPC 结构相关；H100 有 8 个 GPC，cluster 内 CTA 被共同调度（co-scheduled）到同一 GPC 内的 SM。来源：[H100]（GPC/TPC/SM 结构）+ [PG]（cluster co-scheduling）。状态：已确认。
- C16：综合推断（本文归纳，正文标注）：stream 优先级、MPS、MIG、Green Context 四种机制中，没有任何一种能在 kernel 运行期间以微秒/亚毫秒粒度重新分配 SM——优先级不抢占（C11）、MPS 不预留（C12）、MIG 重配置需停负载（C13）、Green Context 创建时固定（C14）。依据：C11–C14 的合取。状态：已确认（作为归纳，各分句有源）。

## F 公式

- F1（教学推导，非外部公式）：GEMM C=AB（M×K @ K×N）中，每个输出元素需要 K 次乘加；不缓存复用时每个 A 行元素被读 N 次、每个 B 列元素被读 K 次；切成 T×T 的 tile 后，每个 A 元素从全局内存读取 N/T 次。依据：tile 内数据驻留 shared memory 被复用的直接计数，正文用手算例子展示，属标准教材结论（可参考任一 CUDA 教材 tiled GEMM 章，如 Kirk & Hwu《Programming Massively Parallel Processors》第 5 章的同款推导；页面按教学推导标注）。状态：已确认（教学推导，页面内可复算）。

## N 数字

- N1：H100 SXM5：132 SM、每 SM 4 Tensor Core、50MB L2、80GB HBM3 @ 3.35 TB/s。来源：[H100]（C5 同源）。
- N2：Hopper 每 SM shared memory 228KB；单 CTA 上限 227KB。来源：[HTG]（C6 同源）。
- N3：warp = 32 线程；CTA ≤ 1024 线程；Hopper 每 SM 最多 64 warp（2048 线程）常驻。来源：[PG]/[HTG]（C1/C2 同源）。
- N4：cluster 可移植上限 8 CTA；H100 opt-in 上限 16。来源：[HTG]（C8 同源）。
- N5：H100 MIG 六档 profile 及实例数上限。来源：[MIG]（C13 同源）。
- N6：Green Context 在 CC 9.0+ 上分区粒度为 8 SM 的倍数、最小 8。来源：[DRV]（C14 同源）。
- N7：Volta+ MPS 每设备最多 48 个客户端上下文（pre-Volta 16）——页面若提及则标注，拟不收录进正文。来源：[MPS]。
- N8：kernel 启动的 CPU 侧开销为微秒量级。来源：无单一官方数字，正文只作"微秒级"量级说明并标注为工程经验，不给精确值；正式依据为 [PG] CUDA Graphs 章"kernel 很多很小时启动开销占比显著"的定性表述。状态：量级说明，不进入数字断言。

## 来源与页面映射（写作阶段引用用）

- 正文 [C1]–[C16]/[N1]–[N8] 编号与本文一致；文末"来源与教学说明"逐条对应。
