# 模型并行（model-parallelism）术语表

| 术语 | 首次出现 | 含义 |
|---|---|---|
| 模型并行（model parallelism） | 页面开头 | 把单个模型按权重切分到多块 GPU 协同计算的技术总称；狭义历史用法指层内并行（Megatron-LM 语境），本页取广义 |
| 张量并行（Tensor Parallelism, TP） | 页面开头 | 按层内权重矩阵的维度（列/行）切分，所有卡参与每一层的计算 |
| 流水线并行（Pipeline Parallelism, PP） | 页面开头 | 按层堆叠深度切成连续分区，数据像流水线一样依次流过各分区 |
| 专家并行（Expert Parallelism, EP） | 第 3 章 | 按 MoE 专家维度切分，本页只链接 moe-serving 不展开 |
| 数据并行（Data Parallelism, DP） | 第 3 章 | 每卡一份完整模型、处理不同数据；与模型并行相区分 |
| all-reduce | 第 1 章 | 集合通信操作：所有卡各出一份部分结果，规约（如求和）后每卡得到完整结果；语义见 gpu-communication 页 |
| 列切分（column parallel） | 第 1 章 | 把权重矩阵 $A$ 按列切成 $[A_1, A_2]$，每卡算 $XA_i$ 得输出的一段列 |
| 行切分（row parallel） | 第 1 章 | 把权重矩阵按行切成上下两块，每卡算本地输入乘本地块得部分积，需求和 |
| GEMM | 第 1 章 | 通用矩阵乘法（General Matrix Multiply） |
| GeLU | 第 1 章 | 高斯误差线性单元，逐元素非线性激活函数 |
| attention head / 多头注意力 | 第 1 章 | 注意力的多头结构，每 head 一组独立 Q/K/V 投影；见 standard-attention 页 |
| stage / 分区（cell） | 第 2 章 | PP 中连续若干层组成的段，放在一张（组）卡上 |
| micro-batch | 第 2 章 | PP 中把一个小批次切成的更小单元，用于填充流水线 |
| 流水线气泡（pipeline bubble） | 第 2 章 | 一个 step 内各 stage 的空闲等待时间；填充期与排空期 |
| slot | 第 2 章 | 流水线时间图上一个 stage 处理一个 micro-batch 的一个时间格 |
| 填充期 / 排空期 | 第 2 章 | 首个 micro-batch 逐级前进（后面 stage 空闲）与最后 micro-batch 逐级离开（前面 stage 空闲）的阶段 |
| NVLink / NVSwitch / NVLink 域 | 第 3 章 | GPU 高速机内互联；NVLink 域指通过 NVSwitch 直连的一组 GPU；见 gpu-communication 页 |
| TP × PP 组合记号 | 第 3 章 | 总卡数 = TP 维度 × PP 维度；EP × PP = 64 为 DeepSeek-R1 64 卡实例 |
| CPP（Chunked Pipeline Parallelism） | 第 3 章 | PP 与输入分块的组合；完整机制见 chunked-prefill 页 |
| 跨机互联（IB / 以太网） | 第 3 章 | 节点间网络；见 gpu-communication 页 |

## 符号

| 符号 | 含义 |
|---|---|
| $A, B$ | FFN 的两个权重矩阵（构造示例中自设维度） |
| $X$ | 层输入（构造示例中 $1\times 4$） |
| $A_1, A_2$ | $A$ 切分后的两个子矩阵 |
| $Y$ | FFN 第一段输出 |
| $p$ | PP 分区（stage）数 |
| $m$ | micro-batch 数 |
| $m + p - 1$ | 一个流水线 step 的总 slot 数 |
