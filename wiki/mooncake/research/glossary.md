# Mooncake 术语表

登记全文首次出现的术语、缩写和符号。保证全文含义一致。

## A

**Algorithm 1**：Mooncake cache-aware prefill 调度算法的伪代码（论文 §6 / Algorithm 1）。全文提到时一律指这一算法。

## B

**batch size**（批大小）：decode 阶段同时处理的请求数（序列数）。Mooncake 调度以增加 batch size 提升 MFU 为目标之一，受 TBT SLO 与 VRAM 容量约束（C13）。

**block**（块）：Mooncake 与 PagedAttention 风格的 KVCache 分页单位。Mooncake 默认块大小 $B=512$ tokens（N3）；每块用一个链式哈希去重。符号：$B$ 表示块大小（tokens），块 ID 用 `block_key` 表示。

## C

**chunked pipeline parallelism (CPP)**：Mooncake 在 prefill 池使用的跨节点流水线并行：把每请求输入切成 ≤ prefill_chunk 的块，不同块由不同节点流水线同时处理；只在阶段边界通信（C18、C19）。

**chunked prefill**：把长 prefill 切成等计算量块以避免 generation stall；decode 可搭 prefill 块的车近乎免费。Mooncake 在 prefill 实例内部仍使用 chunked prefill（见 [Chunked Prefill](../../wiki/chunked-prefill/index.html)）。

**Conductor**：Mooncake 的全局调度器（论文 §3、§6）。负责调度请求、预测 KVCache 块未来使用、执行 swapping 与 replication。

**continuous batching**（连续批处理）：decode 阶段常用调度模式（Orca/vLLM），每次迭代检查状态加入新请求并移除完成请求（C8）。

**CPP**：见 chunked pipeline parallelism。

## D

**decode / decoding stage**（解码阶段）：自回归一次生成一个 token；每批每请求 1 token；受内存带宽约束；时间次线性于 batch size（C7）。

**disaggregated cache**（分离式缓存）：Mooncake 把 GPU 集群内闲置的 CPU/DRAM/SSD/RDMA 资源组织成的分布式 KVCache 池（C3）。

**DRAM**：Dynamic Random-Access Memory。Mooncake 的 KVCache 池主存放在 prefill/decode 实例所在节点的 CPU DRAM 上；新生成的增量 KVCache 异步写入 DRAM。

## E

**Early Rejection**（早拒绝）：Mooncake 在过载场景的过载调度策略；在 prefill 开始前就按 prefill 与 decode 池中较高负载决定是否接受请求（C28）。

**Early Rejection Based on Prediction (ERP)**：基于预测的早拒绝；Mooncake 采用系统级（假设每请求 decode 时长均匀 $t_d$，预测 $t$ 时刻 decode 负载）（C26）。

## F

**false sharing / 拥塞**（网络拥塞）：指传输路径上多个请求同时抢占同一节点的网络带宽，造成尾部延迟放大。Mooncake §6.1 提到 transfer time 取决于「the current network status, especially whether the sending node is under congestion」（§6.1）。

**FFT / FTT**（无相关）：不出现。

## G

**goodput**：在满足 SLO 的前提下完成的请求数；只计完整完成的请求（C27）。Mooncake 把 goodput 作为优化目标。

**GPUDirect RDMA**：跨节点直接 DMA 读/写对端 GPU 显存（绕过 CPU 拷贝）。Mooncake 的 Messenger 组件被描述为「(GPUDirect) RDMA-based」（§3）。注：v4 论文中该组件名为 Messenger。

## H

**hash chain / 块哈希链**：Moonshot/Mooncake 与 vLLM 的前缀复用实现：每块 hash = Hash(本块 tokens, 前块 hash)。一个 ID 全等 ⇒ 块相同且前缀相同（§3 Figure 3、§4.1）。

**hot block replication**：Mooncake 自动把访问频率最高的 KVCache 块复制到多个节点，避免 fetch 拥塞（C11、§6.2、C23）。

**HTTP 429**：Mooncake Conductor 在 SLO 不可达时直接返回给上层（§6.1）。

## K

**kvcache_balancing_threshold**：Mooncake Algorithm 1 中权衡"本地复用"与"远端传输"的超参数（§6.1、Algorithm 1；脚注说明当前手动调节）。全文以 threshold 表示。

**KV cache**（键值缓存）：见 [KV cache](../../wiki/kv-cache/index.html)。每 token 大小公式 $B_{\text{kv}}=2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b$ 字节（F1）。

**KVCache pool**（KVCache 池）：Mooncake 的分布式 KVCache 存储层，按分页块管理，使用哈希链去重（C2、C3、C33）。

## L

**layer-wise prefill**：Mooncake 把 prefill 各层的 KVCache load/store 异步化与计算重叠，使 prefill 执行时间 ≈ max(载入时间, 标准 prefill 时间)（C20、§5.2）。

**LFU**：Least Frequently Used；淘汰频率最低的块。论文 trace 实验中 LRU 表现最优（N5），但 LFU 仍是可选策略。

**LRU**：Least Recently Used；淘汰最近最少使用的块。Mooncake 池默认采用（N5）。

**l_tbt / l_ttft**：TTFT 与 TBT 的 SLO 阈值上限。Algorithm 1 中直接对比预测最大 TTFT/TBT 与这两个阈值（§7.1、§7.4）。

## M

**Messenger**：Mooncake v4 论文中的跨节点 KVCache 传输组件（GPUDirect RDMA）；v4 之前版本的开放源码仓库主要组件曾被称为 Transfer Engine。本文以论文 v4 名称 "Messenger" 为准。注：本文不展开开源仓库内部组件命名。

**MFU**（Model FLOPs Utilization）：模型浮点运算利用率。Mooncake 把 MFU 作为 prefill/decode 调度的目标之一（C13、§1）。

**MoE**：Mixture-of-Experts。本文不展开 MoE 与 Mooncake 的关系。

**Mooncake**：本文档研究对象。论文正式标题：*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*（arXiv:2407.00079v4）。

## P

**PagedAttention**：见 [PagedAttention](../../wiki/paged-attention/index.html)。Mooncake 在 CPU 内存以分页块管理 KVCache，沿用 PagedAttention 风格（§3）。

**PD 分离**（prefill/decode disaggregation）：把 prefill 与 decode 拆到不同集群。Mooncake 是 PD 分离的生产级实现之一（与 Splitwise/DistServe/TetriInfer 并行工作）。

**prefill / prefill stage**（预填充阶段）：处理所有输入 token；计算密集；时间随输入长度超线性增长（C6）。

**prefill_chunk**：Mooncake 中 incremental prefill 的切块阈值；若剩余未缓存 token 数超过该阈值则切块流水线执行；典型 >1,000 tokens（N7）。

**prefix caching**（前缀缓存）：见 [Prefix caching](../../wiki/prefix-caching/index.html)。Mooncake 用哈希链实现。

**prefix_len**：Algorithm 1 中请求在某个 prefill 实例本地可复用的前缀长度（tokens）。

**p_{best}**：Algorithm 1 中全局 prefix 最长的 prefill 实例。

## Q

**query**：用户请求（论文 §2 称为 request）。全文统一用"请求"。

## R

**RDMA**：Remote Direct Memory Access，远程直接内存访问；网卡直接读写对端主机内存，绕过对端 CPU。Mooncake Messenger 基于 RDMA（§3）。

**RPS**：Requests Per Per Second；实验中调节请求到达率的控制参数（§8.1）。

## S

**SLO**（Service Level Objective）：服务水平目标。Mooncake 中主要是 TTFT_P90 与 TBT_P90（§2、N10、N14）。

**SP**（sequence parallelism）：序列并行，把请求输入序列切到多节点并行计算；每层至少 1 次跨节点通信（C17）。与 CPP 区分。

**SSD**：Solid State Disk；Mooncake 的 KVCache 池可选地扩展到 SSD（§3）。

## T

**TBT**（Time Between Tokens）：同一请求连续两个 token 间的延迟；decode 阶段 SLO（C9、N10）。

**Tensor Parallelism**（TP）：见 [Model parallelism](../../wiki/model-parallelism/index.html)。Mooncake 在 prefill 实例内部仍使用 TP；跨节点 TP 每层需 2 次 RDMA all-reduce（C16）。

**trace**：Mooncake 公开的 1 小时回放请求集合，23,608 条（N1、C35）。

**TTFT**（Time To First Token）：请求到达 → 第一个 token 的延迟；prefill 阶段 SLO（C9、N10）。

## V

**VRAM**：GPU 显存。Mooncake 把 decode 实例的聚合 KVCache 容量受 VRAM 约束（C13）；layer-wise prefill 使 prefill 调度可忽略 VRAM（只要装得下单请求）（C20）。

**validation**（验证）：指论文中的实验验证（§8），不指代码验证流程。全文区分两者。

## W

**weight streaming**：不出现。

## 数学符号（全文一致）

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $B$ | 块大小（tokens） | 章节 2 |
| $b$ | 每个权重元素的字节数（fp16 取 2） | F1 |
| $d_{\text{head}}$ | 注意力 head 维度 | F1 |
| $H_{\text{kv}}$ | KV 注意力头数（GQA 下的 KV 头数） | F1 |
| $L$ | Transformer 层数 | F1 |
| $p$ | 选定 prefill 实例 | 章节 3 |
| $p_{\text{best}}$ | 全局 prefix 最长 prefill 实例 | 章节 3 |
| $\text{prefix\_len}$ | 本地可复用前缀长度（tokens） | 章节 3 |
| $\text{best\_prefix\_len}$ | 全局最长可复用前缀（tokens） | 章节 3 |
| $T_{\text{queue}}$ | prefill 队列时间 | 章节 3 |
| $T_{\text{prefill}}$ | prefill 执行时间 | 章节 3 |
| $T_{\text{transfer}}$ | KVCache 跨节点传输时间 | 章节 3 |
| $T_{\text{TBT}}$ | decode 端到端 TBT | 章节 3 |
| $l_{\text{ttft}},\ l_{\text{tbt}}$ | TTFT/TBT SLO 上限 | 章节 5 |
| $t_d$ | 系统级预测中假设的均匀 decode 时长 | 章节 5 |
| $\text{kvcache\_balancing\_threshold}$ | 本地复用/远端传输权衡阈值 | 章节 3 |
| $S$ | 单请求 KVCache 大小 | F2 |
| $T$ | 单请求处理时间 | F2 |
| $\text{PrefixHash}$ | 块哈希链计算函数 | 章节 3 |