# PCP 与 DCP 术语表

全文以本表为准，同一对象只用一种记号或术语。

## 缩写与专有名词

| 名称 | 首次出现 | 含义 |
|---|---|---|
| PCP（Prefill Context Parallel，预填充上下文并行） | 引言 | prefill 阶段把序列沿 token 维切到多个 rank 并行计算的机制；扩展进程 world size |
| DCP（Decode Context Parallel，解码上下文并行） | 引言 | decode 阶段把 KV cache 沿 token 维分片存储的机制；不扩展 world size，复用 TP rank |
| CP（Context Parallel，上下文并行） | 引言 | 上位概念：把一条序列沿 token 维切到多张 GPU 处理 |
| SLO（Service Level Objective，服务等级目标） | 第 1 章 | 服务质量的量化指标（如 TTFT、吞吐） |
| TTFT（Time To First Token，首 token 时间） | 第 1 章 | 从请求到达到第一个输出 token 的时间 |
| TP（Tensor Parallelism，张量并行） | 第 2 章 | 把层内权重矩阵按维度切到多卡的并行方式；本文中 $t$ 表示 `tensor_parallel_size` |
| PP（Pipeline Parallelism，流水线并行） | 第 4 章 | 把模型层分段放到不同卡的并行方式 |
| DP（Data Parallelism，数据并行） | 第 5 章 | 复制整个模型组处理不同请求批的并行方式 |
| GQA（Grouped-Query Attention） | 第 2 章 | 每组 query 头共享一组 K/V 的注意力结构，KV 头数少于 query 头数 |
| MLA（Multi-head Latent Attention） | 第 2 章 | 把 K/V 压成跨全部 query 头共享的低秩潜向量的注意力结构；有效 KV 头数为 1 |
| MHA（Multi-Head Attention） | 第 2 章 | query 头与 KV 头一一对应的标准多头注意力 |
| rank | 第 2 章 | 参与分布式计算的一个进程/设备编号；如 rank $i$ |
| world size | 第 4 章 | 一个推理副本占用的总进程（设备）数 |
| AllGather | 第 3 章 | 集合通信原语：每卡持有一份数据的片段，通信后每卡持有完整数据 |
| ReduceScatter | 第 3 章 | 集合通信原语：对各组内数据求和后，每卡只取回自己负责的片段 |
| all-to-all（a2a） | 第 3 章折叠块 | 集合通信原语/后端：各卡把自己的数据按目的地分发，vLLM 中指 DCP 的一种通信后端 |
| LSE（log-sum-exp） | 第 3 章 | 一个卡（或一组分数）的 $\ln \sum_i e^{s_i}$，softmax 分母的对数形式 |
| online-softmax | 第 3 章 | 分块/分组维护运行中的最大值与和、最终合并的 softmax 计算方式 |
| ring attention | 第 4 章 | 各卡持有自己一段 KV、沿环逐块交换 KV 完成 attention 的分布式算法（arXiv:2310.01889） |
| PD 分离（P/D disaggregation） | 第 5 章 | prefill 与 decode 分别部署在不同 GPU 池的部署方式 |
| paged KV cache | 第 1 章 | 把 KV cache 按固定大小块分页管理的存储方式 |
| interleave（交错分配） | 第 3 章 | KV 沿 token 维的分片落位规则：token $i$ 存到 rank $i \bmod d$（token 级对齐） |
| Mooncake trace | 第 3 章 | 一种请求轨迹数据格式，含输入/输出长度与前缀哈希，可回放压测 |
| NVFP4 | 第 3 章 | NVIDIA 4 比特浮点量化格式（实验中 Kimi K2.6 使用的权重量化） |
| MTP（Multi-Token Prediction） | 第 5 章 | 一次预测多个 token 的投机解码相关技术 |

## 数学符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $T$ | 第 1 章 | 一条请求的上下文 token 总数（历史 + 当前输入） |
| $H$ | 第 2 章 | 模型的 KV 头数（有效值：MLA 取 1，GQA/MHA 取 KV 头数） |
| $t$ | 第 2 章 | 张量并行规模 `tensor_parallel_size` |
| $d$ | 第 3 章 | decode 上下文并行规模 `decode_context_parallel_size`（dcp size） |
| $N$ | 第 4 章 | prefill 上下文并行的 rank 数（PCP size，文中也用于泛指卡数，按上下文区分） |
| $b$ | 第 4 章 | prefill 切分时每块的 token 数 |
| $j$ | 第 4 章 | 块的序号（从 0 起） |
| $i$ | 第 3/4 章 | token 位置或 rank 编号（按上下文区分，均在首次出现处说明） |
| $s_i$ | 第 3 章 | query 对第 $i$ 个 key 的注意力分数（点积，未 softmax） |
| $v_i$ | 第 3 章 | 第 $i$ 个 token 的 value 向量 |
| $o_r$ | 第 3 章 | DCP 组内第 $r$ 卡用本地 KV 算出的部分 attention 输出（本地 softmax 归一化） |
| $l_r$ | 第 3 章 | 第 $r$ 卡的本地 LSE：$l_r = \ln \sum_{i \in r} e^{s_i}$ |
| $o$ | 第 3 章 | 合并后的全局 attention 输出 |
| $r$ | 第 3 章 | DCP 组内的卡编号 |
| $\max(1, t/H)$ | 第 2 章 | KV cache 重复因子（$t \le H$ 时为 1，$t > H$ 时为 $t/H$） |

## 术语使用约束

- "头"统一指 KV 头；query 头单独说明。"有效 KV 头数"用于 MLA 情形（潜向量不可再按头分，等价于 1）。
- "卡"与"GPU"同义混用时保持同一段内统一；正式表述用"GPU"，口语化对照用"卡"仅出现在图注和表格。
- "dcp size"写作 $d$，"tp size"写作 $t$，"PCP size"写作 $N$；不混用 `--decode-context-parallel-size` 参数名与数学符号（参数名在代码语境中用等宽字体）。
- "交错分配"（interleaving）只指 token 级/块级交错落位规则，不用于描述其他交错概念。
- "重复因子"专指 KV cache 总副本数相对 1 份的倍数；不与"冗余""浪费"换用。
- 序列、上下文、token 维三种说法指同一维度时，首次出现处说明"序列（token）维"。
