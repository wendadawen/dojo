# GPU 执行模型与 kernel 调度 glossary：术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| kernel | 页面开头 | 在 GPU 上由大量线程并行执行的一段函数（CUDA 语境，非操作系统内核） |
| 大活 / 小活 | 页面开头 | 贯穿例子：教学构造的大 GEMM（约 2ms）与小 GEMM（约 30μs），数字为教学构造 |
| SM（Streaming Multiprocessor） | 第 1 章 | GPU 的独立计算单元，H100 SXM5 有 132 个 |
| Tensor Core | 第 1 章 | SM 内专门做矩阵乘加的单元，H100 每 SM 4 个（第四代） |
| 寄存器（register） | 第 1 章 | SM 内最快最小的存储，线程私有 |
| shared memory | 第 1 章 | SM 内、CTA 内线程共享的可编程高速存储；H100 每 SM 228KB、单 CTA 最多 227KB |
| L2 cache | 第 1 章 | 全卡共享的二级缓存，H100 为 50MB |
| HBM / 全局内存（global memory） | 第 1 章 | GPU 板载主存，H100 为 80GB HBM3 @ 3.35 TB/s |
| thread / 线程 | 第 2 章 | kernel 的最小执行实体 |
| warp | 第 2 章 | 32 个线程一组，SM 调度发射指令的单位 |
| CTA / thread block | 第 2 章 | 线程块，≤1024 线程，整体驻留一块 SM 执行，跑完才释放 |
| grid | 第 2 章 | 一次 kernel 启动的全部 CTA |
| stream（CUDA stream） | 第 2 章 | GPU 上的命令队列，队内有序、队间无序可并发 |
| warp 调度器 | 第 2 章 | SM 内每个周期挑选就绪 warp 发射指令的硬件 |
| GEMM | 第 3 章 | 通用矩阵乘 C = A·B |
| tile | 第 3 章 | GEMM 输出矩阵被切成的计算小块；kernel 内最小可独立完成单位 |
| fp16 | 第 3 章 | 16 位浮点格式，每元素 2 字节 |
| TMA（Tensor Memory Accelerator） | 第 4 章 | Hopper 的异步拷贝引擎，单线程发起大块张量搬运，不占计算线程 |
| warp specialization | 第 4 章 | 让不同 warp 分别负责搬运（producer）与计算（consumer）的分工写法 |
| CTA cluster | 第 4 章 | 一组被共同调度到相邻 SM 的 CTA；可移植上限 8，H100 可到 16 |
| DSMEM（distributed shared memory） | 第 4 章 | cluster 内 CTA 直接互访彼此 shared memory 的机制 |
| GPC（Graphics Processing Cluster） | 第 4 章 | SM 的分组，cluster 内 CTA 被共同调度到同一 GPC 内 |
| mbarrier | 第 4 章 | 异步搬运的完成屏障（仅在教学简化说明中点名，不展开） |
| persistent kernel / 常驻 kernel | 第 5 章 | 一次启动长期运行、循环领取任务的编程模式（非 API） |
| CUDA Graph | 第 5 章 | 把一串操作录制为图、之后整体重放的机制 |
| 捕获（capture）/ 重放（replay） | 第 5 章 | CUDA Graph 的录制阶段 / 整体提交执行阶段 |
| stream 优先级 | 第 6 章 | 对未开始工作的调度提示，不抢占、不保证顺序 |
| MPS（Multi-Process Service） | 第 6 章 | 让多进程 kernel 在同一 GPU 并发执行的服务；限额是上限而非预留 |
| MIG（Multi-Instance GPU） | 第 6 章 | 把 GPU 硬件级切成固定档位的独立实例；H100 六档 profile |
| Green Context | 第 6 章 | 进程内的 SM 空间分区（CUDA 12.4+）；Hopper 上为 8 的倍数、最小 8 |
| 队头阻塞（head-of-line blocking） | 第 6 章 | 长 kernel 占着 SM，就绪的短 kernel 只能等它跑完 |
| prefill / decode / MoE | 页面开头 | 大模型推理两阶段与模型结构（姊妹概念页 moe-serving 负责，本页只点名不展开） |
| APK | 第 6 章末 | ExpertPlex 的 tile 级自适应常驻 kernel（论文解析页负责，本页只点名） |
