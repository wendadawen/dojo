# DeepEP 术语表

登记全文首次出现的术语、缩写和符号。写作时保证同一对象全页只有一种写法。

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| DeepEP | 页面开头 | DeepSeek 开源的专家并行通信库（GitHub deepseek-ai/DeepEP）；V2 README 标注全称 DeepEveryParallel |
| dispatch | 第 1 章 | token 按路由从原 rank 发往各专家所在 rank 的 all-to-all 方向 |
| combine | 第 1 章 | 专家输出按门控值加权归约送回原 rank 的 all-to-all 方向，dispatch 的反向 |
| all-to-all | 第 1 章 | 每个 rank 都向其他所有 rank 发送数据的集合通信形态（引 gpu-communication） |
| MoE | 第 1 章 | 混合专家模型（引 deepseek-moe） |
| 专家并行（EP） | 第 1 章 | 把不同专家放到不同 rank 的并行方式（引 moe-serving） |
| rank | 第 1 章 | 通信组内的进程编号，本页语境下一个 rank 对应一块 GPU |
| 路由 / gating | 第 1 章 | 门控网络为每个 token 选出 top-K 专家并给出门控值，决定 token 去向 |
| 门控值 $g_{i,t}$ | 第 1 章 | 第 $t$ 个 token 对专家 $i$ 的归一化权重，combine 时乘在专家输出上（V3 式 13） |
| top-$K$ 路由 | 第 1 章 | 每个 token 激活 $K$ 个路由专家（V3 取 $K_r=8$） |
| $t_0, t_1$ | 第 1 章 | 贯穿示例中 rank 0 持有的两个 token |
| $E_e$ / 专家 $e$ | 第 1 章 | 贯穿示例中放在 rank $e$ 上的专家（8 专家每 rank 1 个） |
| hidden / $h$ | 第 1 章 | token 的隐藏维长度（贯穿示例 $h=2$） |
| SM | 第 1 章 | 流式多处理器，GPU 计算与 kernel 执行的调度单位（引 gpu-execution-model） |
| grouped GEMM | 第 1 章 | 对按来源分组的多段矩阵连续做 GEMM 的计算内核，接收侧消费 DeepEP 的布局 |
| FP8 / E4M3 | 第 1 章 | 8 位浮点格式（引 fp8-block-quant）；dispatch 载荷用 FP8 |
| BF16 | 第 1 章 | bfloat16 半精度；combine 载荷保留 BF16 |
| CUDA graph | 第 1 章 | 录制固定形状 GPU 工作并重放的机制（引 vllm-cudagraph） |
| NVLink | 第 2 章 | 节点内 GPU 互联总线（引 gpu-communication） |
| RDMA | 第 2 章 | 绕过远程 CPU 的直接内存访问传输（引 gpu-communication） |
| IB（InfiniBand） | 第 2 章 | 跨节点 RDMA 网络的一种，V3/H800 集群所用 |
| in-node index | 第 2 章 | rank 在自己节点内的序号（0-3）；两段转发的第一跳落点规则 |
| 两段转发 | 第 2 章 | IB 先到目标节点同 in-node index 的 GPU、再 NVLink 转发给专家 GPU 的机制 |
| node-limited routing | 第 2 章 | V3 的路由约束：每 token 至多发送到 $M=4$ 个节点 |
| warp | 第 2 章 | SM 内的线程调度单位；warp specialization 指不同 warp 专职不同任务 |
| 通信通道 | 第 2 章 | V3 内核把 20 SM 分成 10 组，每组负责一条 IB×NVLink 转发通道 |
| 瓶颈带宽 | 第 2 章 | DeepEP 性能表的口径：按最慢一段网络折算的吞吐 |
| prefill / decode | 第 3 章 | 推理两阶段（引 moe-serving）：prefill 处理整段输入算 KV cache，decode 逐 token 生成 |
| TPOT | 第 3 章 | 每输出 token 的时间（引 moe-serving），decode 通信延迟的直接牺牲项 |
| IBGDA | 第 3 章 | 基于 GDR 的 IB 实现，GPU 线程可直接发起 RDMA 操作 |
| 纯 RDMA | 第 3 章 | 低延迟内核的传输形态：所有 rank（含节点内）都经 RDMA 可见，NVLink 禁用 |
| 固定槽位 | 第 3 章 | 低延迟内核按 [本地专家数, 每 rank 最大 token 数 × rank 数, hidden] 预留接收空间的布局 |
| mask | 第 3 章 | 槽位未全部有效时标记有效数据位置的方式；recv_count 给每专家实际数 |
| 接收 hook | 第 3 章 | dispatch/combine 只发 RDMA 请求、由调用方稍后调用 hook 确保数据到达的接口 |
| TBO（two-batch overlap） | 第 3 章 | 两个 micro-batch 交错：一个的 attention 与另一个的 dispatch+MoE+combine 重叠（引 moe-serving） |
| micro-batch | 第 3 章 | 一个 batch 切成的子批，TBO/DualPipe 的交错单位 |
| slot / 槽位 | 第 3 章 | 固定布局中为（本地专家，来源 rank）预留的接收空间单元 |
| NVSHMEM | 第 4 章 | NVIDIA 的对称内存通信库，V1 低延迟/跨节点的底层后端 |
| NCCL Gin | 第 4 章 | V2 的新后端：NCCL 之上的轻量传输层，可复用现有 NCCL communicator |
| ElasticBuffer | 第 4 章 | V2 统一高吞吐/低延迟 API 的缓冲区接口 |
| JIT 编译 | 第 4 章 | 内核在运行时按需编译，安装期无需 CUDA 编译 |
| auto-tuning | 第 4 章 | V1 用实测扫描选配置的方法，V2 改为解析式计算 |
| 解析式 SM/QP 计算 | 第 4 章 | V2 按带宽建模直接算出最优 SM 数与 QP 数 |
| hybrid 模式 / direct 模式 | 第 4 章 | 分层 RDMA+NVLink 转发 vs 不分层直达，两种通信路径 |
| scaleout / scaleup | 第 4 章 | V2 的逻辑域分解：跨节点域 / 节点内域，支撑 EP2048 |
| EP2048 | 第 4 章 | V2 支持的专家并行最大规模（2048 个 EP rank） |
| QP（queue pair） | 第 4 章 | RDMA 的传输端点队列对；V1 低延迟模式 QP 数等于本地专家数 |
| handle | 第 4 章 | dispatch 返回的通信句柄，携带路由元数据供 combine 使用；V2 可缓存复用 |
| Engram | 第 4 章 | V2 实验性的远程内存访问原语（如远程 KV 条目拉取） |
| 冗余专家 | 第 5 章 | 复制高负载专家的部署策略（V3 prefill 32 个/decode 64 GPU 承载） |
| DualPipe | 第 5 章 | V3 的双向流水线算法，把 chunk 拆成 attention/dispatch/MLP/combine 重叠调度 |
| UltraEP / MoonEP | 第 5 章 | 相邻系统：负载感知冗余与重路由 / 冗余专家规划与静态形状（各引页面） |

符号约定：全页 $K$（top-K）、$h$（hidden）、$g_{i,t}$（门控值）、$E_e$（专家）写法唯一；µs 表示微秒；GB/s 按 DeepEP 口径为逻辑带宽。
