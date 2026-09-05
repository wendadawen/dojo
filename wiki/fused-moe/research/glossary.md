# FusedMoE 术语表

按首次出现顺序登记。写作与审查以此为准，同一对象全文只用一种写法。

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| MoE 层（Mixture of Experts） | 开头 | Transformer 中用 router 为每个 token 选少数专家计算的前馈层，前置概念页 moe-serving |
| 专家（expert） | 开头 | MoE 层中的一份独立 FFN 权重，本页记专家总数为 $E$ |
| router / gate | 开头 | 为 token 对全体专家打分的线性层，输出路由 logits |
| top-k 路由 | 开头 | 按 router 分数为每个 token 选 $k$ 个专家并给出门控权重 |
| token | 开头 | 进入 MoE 层的一个隐藏状态向量，本页记 token 数为 $M$，隐藏维为 $H$ |
| GEMM | 第 1 章 | 通用矩阵乘；GEMV 是其 M=1 的退化情形，前置概念页 gpu-execution-model |
| tile / 分块 | 第 1 章 | GEMM 把大矩阵切成 BLOCK_SIZE_M×BLOCK_SIZE_N×BLOCK_SIZE_K 的小块逐块计算 |
| SM（流式多处理器） | 第 1 章 | GPU 的计算单元组，前置概念页 gpu-execution-model |
| 内核（kernel）/ 内核启动 | 第 1 章 | GPU 上一次函数调用；启动有固定开销 |
| gather / scatter | 第 1 章 | 按索引收集/散布张量行的操作 |
| grouped GEMM（分组 GEMM） | 第 1 章 | 把若干不同尺寸的矩阵乘合并成一次调用、按组切换权重的 GEMM 形态 |
| 块稀疏（block-sparse） | 第 1 章 | MegaBlocks 对 MoE 计算的表述：按块组织的稀疏矩阵运算 |
| 融合 MoE 算子 / FusedMoE | 开头 | 本页主题：把全部专家的 GEMM 合并成分块 GEMM 内核调用的算子与层实现 |
| 槽位（slot） | 第 2 章 | (token, top-k 选项) 展开后的线性下标 $s=t\cdot k+j$，总数 $M\cdot k$ |
| sorted_token_ids | 第 2 章 | 槽位按专家排序、每段填充到 block_size 倍数后的数组 |
| expert_ids | 第 2 章 | 每个 M-block 的专家索引；EP 下非本 rank 为 $-1$ |
| num_tokens_post_padded | 第 2 章 | 填充后的有效总槽位数 |
| block_size / BLOCK_SIZE_M | 第 2 章 | M 维 tile 的行数，对齐的粒度 |
| 填充（padding） | 第 2 章 | 每个专家段补齐到 block_size 倍数的多余槽位，值为 num_valid，被掩码跳过 |
| 掩码（mask） | 第 2 章 | 按 `offs_token < num_valid_tokens` 屏蔽填充槽读写的条件 |
| Triton | 第 3 章 | 编写 GPU 内核的 Python DSL；本页只需"一个 program 处理一个 tile"的执行模型 |
| program（程序实例） | 第 3 章 | Triton 网格中的一次内核执行，负责一个输出 tile |
| EM | 第 3 章 | 排序填充后的槽位总长（sorted_token_ids 长度），决定 M 维 block 数 |
| GROUP_SIZE_M | 第 3 章 | 网格中把相邻 M-block 分组以提升 L2 复用的组大小 |
| w13 / w1 | 第 4 章 | 专家的 gate+up 融合权重，形状 $(E,\,2I,\,H)$ |
| w2 | 第 4 章 | 专家的 down 投影权重，形状 $(E,\,H,\,I)$ |
| 中间维 $I$ | 第 4 章 | 专家 FFN 的中间宽度 |
| silu_and_mul | 第 4 章 | 门控激活：silu(gate 半) 逐元素乘 up 半，$(\cdot,2d)\to(\cdot,d)$，前置概念页 swiglu |
| topk_ids / topk_weights | 第 4 章 | 路由输出的专家索引与门控权重，形状 $(M, k)$ |
| MUL_ROUTED_WEIGHT | 第 4 章 | 内核参数：是否在 GEMM 输出上乘路由权重（默认在 GEMM2 生效） |
| moe_sum | 第 4 章 | 把 $(M, k, H)$ 中间结果沿 $k$ 维求和的自定义算子 |
| topk_softmax | 第 4 章 | 融合的 softmax+top-k 路由内核，改编自 TensorRT-LLM |
| 张量并行（TP） | 第 5 章 | 按层内矩阵维度切分模型，前置概念页 model-parallelism |
| 列切 / 行切 | 第 5 章 | w13 沿输出维切、w2 沿输入维切的 TP 分片方式 |
| 专家并行（EP） | 第 5 章 | 把不同专家放到不同 rank，前置概念页 moe-serving |
| EPLB | 第 5 章 | 专家并行负载均衡，相邻概念页 |
| 调优配置（tuned config） | 第 5 章 | 按 (E, N, device, dtype) 存放的 BLOCK_SIZE 等内核参数 JSON |
| naive_block_assignment | 第 5 章 | 极小批量下跳过对齐、每 block 单 token 的内核路径 |

符号约定：$M$=token 数，$E$=专家数，$k$=top-k 值（正文用"top-k"表概念、$k$ 表数值），$H$=隐藏维，$I$=中间维，$N$=内核当前 GEMM 的输出宽度，$K$=内核当前 GEMM 的输入宽度，$s$=槽位下标，$b$=block 下标，$e$=专家索引，$g_{t,j}$=token $t$ 第 $j$ 个选项的门控权重，$x_t$=token 输入，$y_t$=token 输出。GEMM1/GEMM2 统一称呼第一次/第二次 dispatch 的内核调用。
