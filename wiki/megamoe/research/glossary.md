# MegaMoE 术语表

登记全文首次出现的术语、缩写和符号。写作时保证同一对象全页只有一种写法。

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| MegaMoE / Mega MoE | 页面开头 | DeepSeek 开源在 DeepGEMM 中的融合 MoE kernel 家族；正文统一用 Mega MoE，页面标题用 MegaMoE |
| DeepGEMM | 页面开头 | DeepSeek 开源的高性能 Tensor Core kernel 库（GitHub deepseek-ai/DeepGEMM） |
| MoE（混合专家） | 开头/第 1 章 | 路由网络为每个 token 选择部分专家计算的模型结构（引 deepseek-moe） |
| 专家并行（EP） | 第 1 章 | 把不同专家放到不同 rank 的并行方式（引 moe-serving） |
| rank | 第 1 章 | 通信组内的进程编号；本页语境下一个 rank 对应一块 GPU、一个进程 |
| dispatch | 第 1 章 | token 按路由从源 rank 发往专家所在 rank 的方向（引 deepep） |
| combine | 第 1 章 | 专家输出按路由权重加权聚合送回源 rank 的方向，dispatch 的反向（引 deepep） |
| Linear-1 / $W_1$ | 第 1 章 | 专家 FFN 第一层线性变换，hidden → $2\times$ intermediate（gate/up 拼接） |
| SwiGLU | 第 1 章 | $\mathrm{silu}(\text{gate})\times\text{up}$ 门控激活（引 swiglu） |
| Linear-2 / $W_2$ | 第 1 章 | 专家 FFN 第二层线性变换，intermediate → hidden |
| top-$k$ / $k$ | 第 1 章 | 每个 token 被路由到的专家数 |
| 路由权重 / $g_{k}$ | 第 1 章 | combine 时乘在专家输出上的归一化权重 |
| hidden / $h$ | 第 1 章 | token 隐藏维长度（V4-Pro 为 7168） |
| intermediate hidden | 第 1 章 | FFN 中间维长度（V4-Pro 为 3072） |
| 共享专家 | 第 1 章 | 所有 token 都经过、不参与路由的专家（引 deepseek-moe） |
| $t_0$–$t_3$、$E_0$–$E_3$ | 第 1 章 | mini 贯穿示例的 4 个 token 与 4 个专家 |
| NVLink | 第 1 章 | 节点内 GPU 互联总线（引 gpu-communication） |
| SM | 第 1 章 | 流式多处理器，GPU 的计算与调度单位（引 gpu-execution-model） |
| 持久 kernel（persistent kernel） | 第 2 章 | grid 恰好覆盖全部 SM、从任务开始运行到全部完成的单次 kernel 启动形态 |
| warp | 第 2 章 | 32 线程的 GPU 执行单位（引 gpu-execution-model） |
| warp 专用化 | 第 2 章 | 一个 kernel 内不同 warp 组固定承担不同角色的组织方式 |
| CTA / block | 第 2 章 | kernel 启动的线程块；本页语境下一个 block 占一个 SM |
| 2-SM cluster / 2-CTA | 第 2 章 | 相邻两个 CTA 配对协作的硬件分组（Mega MoE 的 UMMA 形态） |
| leader CTA | 第 2 章 | cluster 中 0 号 CTA，MMA 发射等职责只由它执行 |
| TMA | 第 2 章 | Tensor Memory Accelerator 相关的异步批量搬运单元/指令（加载 A/B tile、1D 拉取与写出） |
| TMEM | 第 2 章 | Blackwell（sm100）的 Tensor Memory，UMMA 累加器与 SF 所在的片上存储 |
| UMMA / tcgen05 | 第 2 章 | sm100 的 Tensor Core 矩阵乘指令族 |
| dispatch 线程组 | 第 2 章 | block 内 kNumDispatchThreads=128 线程的角色组，负责路由读取、计数、源索引、数据拉取 |
| TMA A warp / TMA B warp | 第 2 章 | 分别加载激活（含 SFA）与权重（含 SFB）的生产者 warp |
| MMA warp | 第 2 章 | 发射 UMMA 的 warp（仅 leader CTA） |
| 调度 warp | 第 2 章 | 运行 scheduler.mainloop、发布 task 的 warp |
| epilogue 线程组 | 第 2 章 | kNumEpilogueThreads（128–256）线程，做 SwiGLU、量化、远程写回、combine 归约 |
| 对称内存（symmetric memory） | 第 3 章 | 多进程以相同布局各自分配缓冲、地址差固定的内存约定（需 PyTorch >= 2.9） |
| SymBuffer | 第 3 章 | kernel 侧的对称内存视图：本 rank 基址 + 全 rank 偏移表 |
| $\Delta$ / 偏移表 | 第 3 章 | mini 示例中 $B_1 - B_0$；一般化为 offsets 数组 |
| map（地址平移） | 第 3 章 | `map(ptr, dst_rank) = ptr + offsets[dst_rank]` |
| NVLink barrier | 第 3 章 | kernel 内跨 rank 屏障原语（grid sync + 跨 rank 计数信号） |
| remote store / 远程写 | 第 3 章 | 通过 map 得到的远端地址直接写入 |
| pull / 拉取 | 第 4 章 | 接收侧主动从远端读取数据 |
| 元数据 | 第 4 章 | dispatch 阶段先行的计数与源索引（token-topk 索引等） |
| round-robin 最小剥离 | 第 4 章 | 按 per-rank 计数逐轮取最小值交错选源的策略（源码注释 iterative min-peeling） |
| 环形缓冲 / ring | 第 4 章 | 固定容量循环复用的 L1/L2 数据缓冲（kNumRingTokens） |
| full/empty 计数器 | 第 4 章 | 环形槽位的生产/消费完成计数，自旋等待衔接 |
| task / TaskInfo | 第 4 章 | 调度单元：block_phase + 专家号 + M 块 + N 簇等 |
| BlockPhase | 第 4 章 | task 的四相：Linear1 / Linear2 / SharedLinear1 / SharedLinear2 |
| L1 预热波（warmup waves） | 第 4 章 | 调度器先发布的若干批 L1 task，避免 L1→L2 依赖死锁 |
| pool block / M 块 | 第 4 章 | 按专家接收 token 数切成的 BLOCK_M 大小行块 |
| 源元数据（TokenSrcMetadata） | 第 4 章 | L2 写回依据的 (rank_idx, token_idx, topk_idx) 三元组 |
| combine 缓冲 / combine_token_buffer | 第 4 章 | 各 rank 对称缓冲中按 topk 槽组织的回传结果区 |
| FP8 / E4M3 | 第 4 章 | 8 位浮点格式；本页中激活的格式（引 fp8-block-quant） |
| FP4 / E2M1 / NVFP4 | 第 4 章 | 4 位浮点格式；本页中路由专家权重的格式（引 mxfp4-qat） |
| SF / 缩放因子 | 第 4 章 | scaling factor；本页粒度 32 元素、格式 UE8M0 |
| UE8M0 | 第 4 章 | 8 位指数格式（无尾数）的 SF 编码 |
| BF16 | 第 4 章 | bfloat16；combine 回传与最终输出格式 |
| amax | 第 4 章 | 绝对值最大值，量化 SF 的计算依据 |
| block_m / block_n / block_k | 第 4 章 | GEMM tile 尺寸；block_n 固定 128 |
| 期望 token 数 | 第 4 章 | `num_tokens × num_ranks × num_topk / num_experts`（F2） |
| 基线 / legacy | 第 5 章 | DeepEP dispatch/combine + DeepGEMM grouped GEMM + TileLang SwiGLU 非重叠流水线 |
| DeepEP | 第 5 章 | DeepSeek 开源的专家并行通信库，基线通信部分（引 deepep） |
| V4-Flash / V4-Pro | 第 5 章 | 基准用的 DeepSeek-V4 两档规格（PR #316 定义） |
| batch size | 第 5 章 | 每 rank 的 token 数（官方评论确认口径） |
| sm100 / sm90 | 第 5 章 | Blackwell / Hopper 架构的 compute capability 代号 |
| X-Stage | 第 5 章 | arXiv:2607.23264 论文提出的 post-issue 流水线阶段概念及其对 Mega MoE 的改进分析（第三方） |
| expert wave | 第 5 章（折叠） | X-Stage 论文对 Mega MoE 调度分组的描述术语（官方源码无此词） |
