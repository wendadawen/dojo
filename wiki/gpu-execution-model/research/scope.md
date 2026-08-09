# GPU 执行模型与 kernel 调度 scope：内容范围

## 0. 概念歧义处理

- 「kernel」：CUDA 语境指在 GPU 上由大量线程并行执行的一段函数，与操作系统内核（OS kernel）同名不同义。裁定：已裁定，本页采用 CUDA 语境，开头一句话消歧。
- 「调度」：本文指三层不同对象——硬件把 CTA 分发到 SM、warp 调度器在 SM 内选 warp 发射指令、软件层在多任务间分配 SM。裁定：并列呈现，正文明确区分三层，避免读者把 stream 优先级想象成操作系统式抢占。
- 「tile」：指 GEMM 类 kernel 内把输出矩阵切成的计算小块，非图形学 tiling。裁定：已裁定，采用 GEMM/深度学习语境。
- 无影响核心定义的无法消歧项。

## 1. 概念含义

- 概念名称：GPU 执行模型与 kernel 调度（GPU execution model and kernel scheduling）
- 一句话定义：GPU 把一段并行计算（kernel）组织成「线程→warp→CTA→CTA cluster」的层级，由硬件分发到上百个 SM 上执行；kernel 调度研究这些计算单位如何被启动、放置、共享与切换。
- 正式定义（与 CUDA C++ Programming Guide 一致）：一个 kernel 以 grid 形式启动，grid 由若干 thread block（CTA）组成，每个 CTA 在一块 SM 上执行，CTA 内线程以 32 线程的 warp 为单位被 SM 的 warp 调度器发射执行；Hopper 起 CTA 可组成 cluster，经 DSMEM 直接互访 shared memory。
- 本文语境：NVIDIA CUDA GPU（Hopper/H100 为主要例子），面向大模型推理负载（GEMM 为主）。
- 包括什么：
  - GPU 硬件层级（SM、Tensor Core、shared memory、L2、HBM 全局内存、TMA）：执行的物质基础，理解 tile 与 cluster 的前提。
  - 线程层级（thread、warp、CTA、CTA cluster、DSMEM）：kernel 的组织方式，tile 调度的直接对象。
  - kernel 与 CUDA stream 的启动/执行模型：启动开销、stream 语义、硬件 CTA 分发，是理解「抢占难」的根源。
  - GEMM 与 tile 的关系：为什么长输入只是 tile 更多而不是 tile 更大——ExpertPlex APK 在 tile 边界调度的直接前提。
  - warp specialization 流水线直觉：现代高性能 GEMM kernel 内部的工作划分，APK 兼容它的原因。
  - persistent kernel 与 CUDA Graph：消除启动开销的两种手段，APK 的「常驻」正属此类。
  - GPU 共享机制全景（stream 优先级、MPS、MIG、Green Context）：各自能/不能做什么，是 ExpertPlex 动机章（colocation 为什么不够）的硬件依据。
- 不包括什么：
  - MoE 结构、prefill/decode、EP、goodput 等服务层概念：属另一概念页（moe-serving），本页只在开头点出阅读动机。
  - ExpertPlex APK 本身的机制：属论文解析页；本页只提供它依赖的概念基底。
  - CUDA 语法教学与完整编程模型（内存对齐、bank conflict 等优化细节）：不影响学习目标，属编程教程范围。
  - AMD/其他厂商 GPU：术语体系不同（CU/wavefront 等），纳入会引入术语漂移。
  - 训练侧机制（反向传播、梯度通信）：与本页学习目标无关。
- 相邻概念：
  - CUDA Graph 工程实践（vLLM 视角）：本页讲录制/重放机制本身；`wiki/vllm-cudagraph/index.html`（note）讲 vLLM 中的捕获尺寸与失败模式，作扩展阅读链接，不重复。
  - GPU 虚拟化（vGPU）：与 MIG 相邻但面向 VM，排除，只在对照中一句带过。

## 2. 学习目标

### Q1：一个 kernel 从启动到在 GPU 上跑完，经过了哪些层级？

- 完成答案：读者能沿「CPU 提交 → stream 队列 → 硬件把 CTA 分发到空闲 SM → SM 内 warp 调度器发射指令」描述全过程；能说出 thread/warp/CTA/cluster 各是什么、谁住在哪个硬件上；能说出 kernel 一旦开始，运行期间没有操作系统式的时间片抢占（CTA 跑到完才释放 SM）。
- 为什么是核心目标：不理解执行层级，tile、cluster、persistent kernel、共享机制四章都无依托。
- 依赖内容：SM 定义、内存层级、线程层级、stream 语义、CTA 不可抢占的事实。

### Q2：为什么 GEMM 要切成 tile 来算？为什么输入变长只是 tile 变多、单个 tile 不会变大？

- 完成答案：读者能手算一个小例子说明切 tile 后全局内存读取次数下降；能说出 tile 尺寸由 SM 的 shared memory 容量约束决定，与输入总规模无关；输入变长时 tile 数量按比例增长、单 tile 时间近似不变，因此「tile 边界」是与输入长度无关的天然调度粒度。
- 为什么是核心目标：这是 ExpertPlex「在 tile 边界调度、抢占上界与输入长度无关」论断的全部概念依据。
- 依赖内容：内存层级速度差、GEMM 的数据复用结构、shared memory 容量数字。

### Q3：启动一个 kernel 有哪些开销？persistent kernel 和 CUDA Graph 各自消除哪一部分？

- 完成答案：读者能区分 CPU 侧提交开销与 GPU 侧 CTA 分发/收尾开销；能说出 persistent kernel「一次启动、循环领任务」如何把多次启动开销摊为一次；能说出 CUDA Graph「录制一次、整体重放」如何消除逐次 CPU 提交；能说出两者的共同约束——计算形状/地址基本固定，以及「常驻 kernel 不退出，别人就用不了它的 SM」这一代价。
- 为什么是核心目标：APK 是 persistent kernel 且兼容 CUDA Graph，不理解这两者就无法理解 APK 的设计空间与约束。
- 依赖内容：kernel 启动流程（Q1）、stream。

### Q4：CTA cluster、DSMEM、TMA、warp specialization 各自解决了什么瓶颈？

- 完成答案：读者能说出——TMA 解决「搬运占用计算线程」的问题（单线程发起大块异步拷贝）；warp specialization 解决「搬运与计算互相等待」的问题（不同 warp 分工形成流水线）；cluster+DSMEM 解决「相邻 SM 想交换数据只能绕全局内存」的问题（cluster 内直接互访 shared memory，TMA 可组播）；cluster 大小有上限（可移植 8 个 CTA，H100 可到 16）。
- 为什么是核心目标：ExpertPlex 的抢占决策「沿 DSMEM/shared memory 传播、cluster 内一致切换」直接建立在这四个机制上。
- 依赖内容：内存层级、CTA 与 SM 的对应关系（Q1）、tile 流水线（Q2）。

### Q5：两个任务要共享一张 GPU 时，stream 优先级、MPS、MIG、Green Context 各自能做什么、不能做什么？

- 完成答案：读者能填出对照——stream 优先级只是调度提示、不抢占已运行 kernel；MPS 让多进程 kernel 并发但资源限额不预留、不同客户端可落同一 SM；MIG 硬件级切片、隔离最强但档位固定（H100 只有 1g/1g.20gb/2g/3g/4g/7g 六档）且重配置要停负载；Green Context 在进程内按 SM 切空间分区（Hopper 起最小 8 个 SM、8 的倍数）但创建后固定、且无并发前进保证。读者能总结：四种机制都无法在 kernel 运行期间做细粒度、毫秒以下的资源重分配。
- 为什么是核心目标：这是 ExpertPlex 动机章「colocation 为什么治不好」的硬件事实基底。
- 依赖内容：SM 是竞争的资源（Q1）、kernel 运行期间不可抢占（Q1）。

## 3. 内容分级

- 核心内容：
  - 硬件层级（SM/Tensor Core/shared memory/L2/HBM 及其速度差）→ Q1/Q2
  - 线程层级（warp=32、CTA≤1024 线程、CTA 在单 SM 上执行、CTA 跑完才释放 SM）→ Q1/Q5
  - kernel 启动流程与 stream 语义（提交、队列、硬件分发、无运行期抢占）→ Q1/Q3/Q5
  - GEMM 切 tile 的手算例子 + tile 尺寸由 shared memory 约束 → Q2
  - TMA / cluster / DSMEM / warp specialization 的能力与上限 → Q4
  - persistent kernel / CUDA Graph 的机制与约束 → Q3
  - 四种共享机制的能力边界对照（含 MIG 档位表、Green Context 对齐要求、stream 优先级官方语义、MPS 不预留资源）→ Q5
- 辅助内容：
  - H100 规格数字（132 SM、228KB shared memory/SM、50MB L2、HBM3 3.35TB/s）：为手算例子提供真实尺度感，服务 Q1/Q2
  - 「kernel 启动开销微秒级」的定量直觉：服务 Q3，标注为量级说明
  - 集群内 cluster 占用率代价（大 cluster 降低可并发 block 数）：服务 Q4 的边界理解
- 扩展内容：
  - CUDA 语法细节（<<<>>>、cudaMemcpy 等）：排除，附最小伪代码即可
  - L2 持久化控制、cache hint：排除
  - Blackwell 新特性（CTA pair、tcgen05）：排除，一句话提代际差异即可
  - MPS 客户端连接数上限（48/16）：排除，不影响学习目标

## 4. 前置知识映射

读者设定为完全小白，本页从最小必要前置开始，自含全部基础。逐项检查 `wiki/` 下现有概念页（当前 wiki 无 concept 流程产物，只有 3 篇 note）：

| 前置知识 | 被哪些学习目标依赖 | 概念页状态 | 递归深度 |
|---|---|---|---|
| 矩阵乘法的基本形式（C=AB 的逐元素定义） | Q2 | 无概念页，但属高中数学，正文用 30 字内联说明 | — |
| 「进程/程序」的一般概念 | Q5（MPS 跨进程） | 常识级，内联一句说明 | — |
| vLLM CUDA Graph 工程实践（扩展阅读） | 不依赖 | `wiki/vllm-cudagraph/index.html` 已存在，作链接 | — |
| MoE 服务层概念（prefill/decode 等） | 不依赖（本页只在动机段提及名词，标注由姊妹概念页讲解） | `wiki/moe-serving/` 由编排方并行生成，正文放占位提示 | — |
| ExpertPlex 论文解析页 | 不依赖（本页是其前置） | 页面未生成，不链接，仅文字提及 | — |

无需要递归生成的概念页；depth-2 无缺口。

## 5. 明确不展开的内容

- CUDA 编程语法与 API 细节：不影响五个学习目标，属编程教程；本页用伪代码级描述代替。
- Tensor Core 的指令级细节（mma 形状、FP8 格式）：只影响性能调优，不影响「为什么切 tile」「为什么流水线」的概念理解；只说清「Tensor Core 一次算一小块矩阵乘加」。
- 硬件 CTA 调度器（GigaThread engine）的内部算法：官方未公开细节，且不影响「CTA 分发到 SM、跑完才释放」的结论。
- 抢占式多任务（compute preemption）的硬件细节：驱动级时间片存在但粒度粗、不可编程控制；一句话说明并归入 stream 优先级「不能做什么」的证据，不展开。
- 多 GPU 互联（NVLink 拓扑）：属服务层概念页范围。

## 6. 常见误解与适用边界

- 误解 1：「GPU 线程和 CPU 线程一样，会被操作系统随时切换」。正确结论：GPU 的 CTA 一旦被分发到 SM 就运行到完成，没有用户可控的运行期抢占；stream 优先级只是对未开始工作的调度提示（官方文档原文）。形成原因：用 CPU 多任务直觉套 GPU。影响 Q1/Q5。
- 误解 2：「输入越长，每个 tile 就越大」。正确结论：tile 尺寸由 SM 的 shared memory 容量决定，与输入规模无关；输入变长只是 tile 数量变多、单 tile 时间近似不变。形成原因：把「总量变大」误当成「单元变大」。影响 Q2。
- 误解 3：「stream 优先级高的 kernel 能打断正在运行的低优先级 kernel」。正确结论：官方文档明确优先级只是 hint，不抢占已运行的工作，不保证执行顺序。影响 Q5。
- 误解 4：「MIG 可以按任意比例切 GPU」。正确结论：H100 只有固定档位（1g.10gb/1g.20gb/2g.20gb/3g.40gb/4g.40gb/7g.80gb），切片含显存且改配置需停负载。影响 Q5。
- 误解 5：「persistent kernel 是官方 API」。正确结论：它是一种编程模式（kernel 内循环领任务），不是一个 CUDA API。影响 Q3。
- 适用边界：
  - 解决问题：理解 GPU 上计算如何组织、启动、放置、共享；理解 tile 作为调度粒度的来源。
  - 不解决：写出高性能 kernel 的工程能力；非 NVIDIA GPU；训练侧并行策略。
  - 结论成立条件：以 Hopper（H100）与 CUDA 12.x 文档为准；具体数字（132 SM、227KB、cluster 上限等）随代际变化，页面标明代际。
  - 条件不满足时：跨代际数字可能变化（如 Blackwell SM 数为 148/SM 配置不同），但层级模型与「tile 尺寸由 shared memory 约束」等结构性结论不变。

## 7. 论断分级

- C1–C16、F1、N1–N8 全部为「官方文档/白皮书明确陈述」，定位见 evidence.md
- 「tile 是天然的调度粒度」「四种共享机制都无法做运行期细粒度重分配」为基于官方语义的综合推断，正文标注为本文归纳
- persistent kernel 的收益描述标注为工程惯例/教学解释
- 无论断处于证据不足状态
