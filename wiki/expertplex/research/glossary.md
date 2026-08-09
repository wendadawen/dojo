# ExpertPlex glossary：术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| MoE（Mixture-of-Experts） | 第 1 章 | 把 FFN 换成多个「专家」子网络，每 token 只激活少数几个的稀疏结构 |
| routed expert / shared expert | 第 1 章 | 按路由选择性激活的专家 / 处理所有 token 的共享专家 |
| router / top-k | 第 1 章 | 为每个 token 选择 k 个 routed expert 的门控网络 |
| prefill / decode | 第 1 章 | 推理两阶段：并行处理全部输入 token 产出首 token / 逐 token 自回归生成 |
| KV cache | 第 1 章 | 缓存各层 key-value 张量供后续迭代复用 |
| TTFT / TPOT | 第 1 章 | 首 token 时延 / 每输出 token 时延 |
| SLO | 第 1 章 | 服务级别目标，本文对 TTFT 与 TPOT 分别设定 |
| goodput | 第 1 章 | 有效吞吐：满足 SLO 前提下系统能承接的最高请求速率；本文主指标为 P90 goodput |
| PDD（prefill-decode disaggregation） | 第 1 章 | PD 分离：两阶段部署在不同 GPU 实例上 |
| colocation（PD 合设） | 第 1 章 | 两阶段共享同一实例/GPU |
| EP（expert parallelism） | 第 1 章 | 专家并行：把专家分片到不同 GPU |
| dispatch / combine | 第 1 章 | MoE 中把激活发给选中专家 / 回收专家输出的 all-to-all 通信 |
| TBO / SBO | 第 4 章 | two-batch overlap：微批次间通信计算重叠 / single-batch overlap：同批次内共享专家与通信重叠 |
| DP / TP | 第 2 章 | 数据并行 / 张量并行 |
| SM | 第 1 章 | streaming multiprocessor，GPU 的计算单元 |
| CTA / warp / cluster | 第 3 章 | 线程块（跑在一个 SM）/ 32 线程组 / 可跨 SM 协作的 CTA 组（经 DSMEM 与 TMA 组播） |
| tile | 第 3 章 | GEMM 类 kernel 内最小可独立完成的计算单位；APK 的调度单位 |
| grouped GEMM | 第 1 章 | MoE 专家计算的成组矩阵乘形式 |
| persistent kernel | 第 3 章 | 常驻 kernel：一次启动长期运行、在 GPU 内循环领任务，不反复启动 |
| APK（Adaptive Persistent Kernel） | 第 2 章 | 本文核心机制：带 tile 级调度的自适应常驻 kernel |
| CUDA Graph | 第 3 章 | 把一串 kernel 录制成图整体重放，消除逐次启动开销 |
| Green Context | 第 1 章 | NVIDIA 的 SM 空间分区机制，kernel 期间固定 |
| MPS / MIG | 第 3 章 | NVIDIA 的多进程服务（软共享）/ 硬件级 GPU 切分 |
| head-of-line blocking | 第 1 章 | 队头阻塞：长 prefill kernel 挡住就绪的 decode kernel |
| 资源气泡（resource bubble） | 第 1 章 | 为某阶段保留的 SM 在其无就绪工作时闲置 |
| 一侧通信（one-sided） | 第 4 章 | 由一端直接读写远端 buffer、无需对端配合的通信（push/pull） |
| 两侧通信（two-sided） | 第 4 章 | 收发双方需配合推进的通信（ring buffer + 信用） |
| ring buffer / credit | 第 4 章 | 接收侧环形缓冲 / 防止发送方覆盖未读槽位的信用回传 |
| WaitDone kernel | 第 4 章 | attention 侧单线程 kernel，观察 combine 完成信号后发起 pull |
| NVLink / RDMA / IB / IBGDA | 第 4 章 | 节点内高速互连 / 远程直接内存访问 / InfiniBand / IB GPUDirect Async |
| scale-up / scale-out | 第 4 章 | 节点内互连域 / 跨节点网络 |
| 虚拟通道（virtual lane） | 第 4 章 | IB 的流量优先级隔离通道 |
| ℓ（layout）/ q | 第 5 章 | 服务器布局配置 / decode SM 预算 |
| B_p、B_d、T_p、T_d、Ō | 第 5 章 | 满足 SLO 的最大 prefill/decode batch、两阶段迭代延迟、平均输出长度 |
| x_moe、m_e、M_t | 第 5 章 | MoE tile 足迹、专家 e 的 token 行数、tile 高度 |
| Q_max、⌈·⌉_c | 第 3 章 | 保障 prefill 进度的 decode SM 上限 / 向上取整到 CTA cluster 倍数 |
| 部署单元（deployment unit） | 第 1 章 | 实现目标 P:D 配比的最小整副本组合 |
| AFD（attention-FFN disaggregation） | 第 7 章 | attention 与 expert 计算分离的一类系统（MegaScale-Infer、Step3 等） |
