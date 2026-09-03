# Mooncake 证据表

来源主论文：Ruoyu Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, FAST '25 / arXiv:2407.00079v4 (3 Sep 2025), 23 pages, 13 figures。引用时一律标注论文小节号（§X）、图号（Figure X）、表号（Table X）、Algorithm X。

冲突与边界：C/F/N 项中标注「冲突 / 估算 / 构造」的条目不进入正文关键论断，仅用于辅助说明或折叠块。

## C 论断（核心事实与机制）

| 编号 | 论断 | 来源定位 | 适用条件 | 置信 |
|---|---|---|---|---|
| C1 | Mooncake 是 Moonshot AI 提供 Kimi 在线服务的承载平台 | Abstract（"Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI"） | — | 已确认 |
| C2 | Mooncake 采用 KVCache-centric 分离式架构，把 prefill 与 decode 集群分离 | Abstract；§3 | — | 已确认 |
| C3 | Mooncake 利用 GPU 集群中闲置的 CPU/DRAM/SSD 资源构成"分离式缓存"（disaggregated cache of KVCache） | Abstract；§3 | — | 已确认 |
| C4 | Mooncake 的核心是 KVCache-centric 调度器，权衡整体有效吞吐量与 TTFT/TBT SLO | Abstract；§1.1；§6 | — | 已确认 |
| C5 | Mooncake 在过载场景下采用基于预测的早拒绝策略 | Abstract；§7 | 过载场景（GPU 弹性不足） | 已确认 |
| C6 | prefill 阶段处理所有输入 token（除短请求外）计算密集；注意力复杂度 O(N²)，MLP O(N)；prefill 计算时间总体随输入长度超线性增长 | §2（"computation time in the prefill stage generally increases superlinearly with input length"） | — | 已确认 |
| C7 | decode 每批每请求一个 token，受内存带宽约束；计算时间随 batch 大小次线性增长 | §2（"memory-constrained and causes computation time to increase sublinearly with batch size"） | — | 已确认 |
| C8 | decode 阶段常用连续批处理（Orca 等）；调度器在每次迭代前检查所有请求状态，把新请求加入批 prefill 阶段并移除已完成请求 | §2（cite [12][13]） | — | 已确认 |
| C9 | MaaS 场景下 prefill 主要衡量 TTFT（请求到达 → 第一个 token），decode 主要衡量 TBT（同一请求连续 token 间） | §2 | — | 已确认 |
| C10 | 请求处理总体三步：① 把可复用 KVCache 传到选定的 prefill 实例；② prefill 实例分块/分层完成 prefill 并流式把 KVCache 传到对应 decode 实例；③ decode 实例载入 KVCache 加入连续批处理 | §1（"the global scheduler (Conductor) needs to select a pair of prefill and decoding instances..."） | — | 已确认 |
| C11 | Conductor 预测 KVCache 块未来使用，执行 swapping 与 replication；最热块复制到多节点以避免 fetch 拥塞，最冷块换出 | §1（"the hottest blocks should be replicated to multiple nodes to avoid fetching congestion, while the coldest ones should be swapped out"） | — | 已确认 |
| C12 | prefill 调度受 prefill 节点 DRAM 可用性约束（特别是大量 DRAM 被全局 KVCache 池预留时） | §1 | — | 已确认 |
| C13 | decode 调度目标是在 TBT SLO 与聚合 KVCache VRAM 容量双重约束下，聚合尽可能多的 token 以提升 MFU | §1（"The aim is to aggregate as many tokens as possible in a decoding batch to improve MFU. However, this objective is restricted not only by the TBT SLO but also by the total size of the aggregated KVCache that can be contained in the VRAM"） | — | 已确认 |
| C14 | Mooncake 显式考虑过载场景下接受/拒绝请求；GPU 供给受限时过载在 peak 时段普遍存在 | §1；§7 | 过载场景 | 已确认 |
| C15 | Mooncake 决定采用彻底分离架构而非纯 chunked prefill；原因：1) prefill 节点需要不同的跨节点并行设置以应对长上下文；2) 提供节省 VRAM 的机会 | §5（"There are two main reasons for this decision: 1) Prefill nodes require different cross-node parallelism settings to handle long contexts (§5.1). 2) It presents a unique opportunity to save VRAM (§5.2)"） | — | 已确认 |
| C16 | TP 跨一个 8×GPU 节点以上时每层需 2 次 RDMA all-reduce，显著降低 prefill 节点 MFU | §5.1（"extending tensor parallelism (TP) across more than one node requires two expensive RDMA-based all-reduce operations per layer, significantly reducing the MFU of prefill nodes"） | — | 已确认 |
| C17 | SP（Ring/Striped Attention 等）每层至少 1 次跨节点通信；理想部署把 prefill 节点分成两组（仅 TP 组、TP+SP 组），仅当 TTFT SLO 需要时把请求路由到 SP 组 | §5.1 | — | 已确认 |
| C18 | CPP（chunked pipeline parallelism）：把每请求输入切成块（每块 ≤ prefill_chunk），不同块由不同节点流水线同时处理，跨节点通信只在流水线阶段边界 | §5.1（"We group every X nodes in the prefill cluster into a pipelined prefill node group. For each request, its input tokens are partitioned into chunks, each no longer than the prefill_chunk. Different chunks of the same request can be processed simultaneously by different nodes"） | 长上下文 prefill | 已确认 |
| C19 | CPP 两点收益：1) 类似训练 PP，通信只在阶段边界可与计算重叠 → MFU 更好，与 KVCache 传输网络竞争少；2) 同时适合短/长上下文，无需频繁动态调整节点划分 | §5.1 | — | 已确认 |
| C20 | layer-wise prefill 把 KVCache load/store 异步化与计算重叠；prefill 执行时间 ≈ max(载入时间, 标准 prefill 时间)；使 prefill 调度只需考虑 DRAM 不必考虑 VRAM（只要装得下单请求） | §5.2（"Transfer overlapping allows the prefill instance's execution time to be roughly equivalent to either the KVCache loading time or the standard prefilling time"…） | — | 已确认 |
| C21 | Algorithm 1（cache-aware prefill 调度）：每个请求输入按块计算链式 key；与各 prefill 实例缓存键匹配得 prefix_len；估计 TTFT；选最短；若不可达 SLO 直接 HTTP 429 拒绝 | Algorithm 1；§6.1 | — | 已确认 |
| C22 | Algorithm 1 的两条分支：若 best_prefix_len / prefix_len < kvcache_balancing_threshold 则用本地 prefix_len 直接算 Tprefill（branch 1，cache-aware）；否则计算 Ttransfer + Tqueue + Tprefill(len, best_prefix_len)，其中 Tprefill 用 best_prefix_len 假设传输差额后复用（branch 2，cache-aware and -balancing） | Algorithm 1（条件分支）；§6 | — | 已确认 |
| C23 | Algorithm 1 末尾：若 best_prefix_len / chosen.prefix_len > kvcache_balancing_threshold，则触发 TransferKVCache(best_matched_instance, p)，把热点缓存迁到 chosen — 这是热点自动迁移机制 | Algorithm 1；§6.2（"Both strategies not only reduce the prefill time for requests but also facilitate the automatic replication of hot-spot caches, allowing for their broader distribution across multiple machines"） | — | 已确认 |
| C24 | prefill 执行时间由离线测试数据拟合的预测模型估计（依请求长度与 prefix_len），队列时间 = 已排队请求 prefill 时间之和 | §6.1 | — | 已确认 |
| C25 | 朴素早拒绝（按当前 decode 负载调度）有滞后，会引发 prefill/decode 负载反相震荡（4 阶段循环） | §7.3；Figure 10a | 过载场景 | 已确认 |
| C26 | 预测早拒绝分两种粒度：请求级（预测每请求输出长度，难度高）与系统级（估计整 batch 数或 TBT 状态，要求低，适合过载）；Mooncake 采用系统级：假设每请求 decode 时长均匀 td，预测 t 时刻 decode 负载 = avg(TBT ratio to l_tbt) | §7.4 | 过载场景 | 已确认 |
| C27 | goodput 只计完整完成的请求；否则之前消耗/生成的 token 不计，资源浪费 | §2（"only requests that fully complete their execution are counted in the measure of goodput"） | — | 已确认 |
| C28 | 过载场景下调度按 prefill 与 decode 池中较高负载决定是否接受请求 | §7.1；§7.2 | 过载场景 | 已确认 |
| C29 | Mooncake 端到端实验以 vLLM 作 baseline；vLLM 采用 PagedAttention + 连续批处理 | §8.1 baseline（"We employ vLLM, one of the state-of-the-art open-source LLM service, as our experimental baseline. vLLM incorporates continuous batching and PagedAttention technologies"） | — | 已确认 |
| C30 | 长上下文请求多时 vLLM 耦合设计会扰动 decode；模拟数据实验中 vLLM 必须单请求处理以避免 TBT 违规 | §8.1.2（"the long-context requests in simulated data significantly disrupt the decoding stage of vLLM. To counteract this, vLLM processes requests individually, rather than in batches"） | 模拟数据实验 | 已确认 |
| C31 | Mooncake 在长上下文场景优势显著：模拟数据 50%–525% 吞吐提升；公共数据集 ArXiv Summarization +20%、L-Eval +40% | §8.1.1；§8.1.2 | 论文实验条件 | 已确认 |
| C32 | 真实工作负载 Mooncake 比 vLLM 多处理约 75% 请求；Mooncake ~100% 满足 TBT SLO，vLLM 仅 57% | §8.1.3；Figure 13 | 真实回放实验 | 已确认 |
| C33 | Mooncake 的本地缓存复用逻辑与 vLLM 一致（块哈希链式匹配），但扩展为全局缓存 | §6.1（"Similar reuse logic is already implemented in vLLM, but the open-source version of vLLM only supports local KVCache caching"） | — | 已确认 |
| C34 | 当前工作负载理论最多 ~50% KVCache 可复用（容量与 TTFT SLO 无限时）；papers.cool 场景可达 ~90% | §9（"Theoretically, up to only 50% of the KVCache can be reused in our current workloads... can be as large as 90% for certain scenarios, such as our chat-to-paper service https://papers.cool/"） | 工作负载依赖 | 已确认 |
| C35 | 论文所有实验结果基于回放 trace + dummy LLaMA2-70B 架构（保护商业信息）；trace 不含真实用户内容 | §1 footnote；§4；§8 | 论文实验条件 | 已确认 |

## F 公式

| 编号 | 公式 | 来源定位 | 用途 | 置信 |
|---|---|---|---|---|
| F1 | 每 token KVCache 字节数 $B_{\text{kv}}=2 \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b$ | [KV cache](../../wiki/kv-cache/index.html) 页面 F1（已发表）；LLaMA2-70B 配置：$L=80,\ H_{\text{kv}}=8,\ d_{\text{head}}=128,\ b=2$（fp16），$B_{\text{kv}}=327680\ \text{B}\approx320\ \text{KiB/token}$（出自 LLaMA2 论文 [11]） | 仅用于正文构造示例估算 KVCache 大小 | 已确认 |
| F2 | 占用代价 $S \cdot T$：单请求 KVCache 大小 $S$ 与处理时间 $T$ 的乘积 | §5.2（"its occupation cost is $S \cdot T$"） | layer-wise prefill 节省 VRAM 的论证 | 已确认 |

## N 数字

| 编号 | 数字 | 来源定位 | 用途 | 置信 |
|---|---|---|---|---|
| N1 | 23,608 条采样请求，1 小时 trace | §4.1（"The trace dataset comprises 23,608 entries"） | trace 规模 | 已确认 |
| N2 | 平均输入 7,590 tokens，平均输出 182 tokens | §4.2（"an average input length of 7,590 tokens and an average output length of 182 tokens"） | trace 统计特征。注：原文 "The average input-output ratio is approximately 720"，按均值算术比为 7590/182≈42，正文采用两独立均值不引用 ratio | 已确认（ratio 数值存疑） |
| N3 | 块大小 512 tokens | §4.1（"with a block size of 512"） | 哈希链粒度 | 已确认 |
| N4 | 缓存容量 1,000 块 → 50,000 块，命中率 30% → 50%（LRU） | Table 1（"Increasing the cache capacity from 1,000 to 50,000 blocks boosts the cache hit ratio from 30% to 50%"） | 容量与命中率 | 已确认 |
| N5 | LRU 在该 trace 上表现最优 | §4.2（"LRUCache performs best under this dataset's patterns"） | 淘汰策略选择依据 | 已确认 |
| N6 | >50% 块零访问，部分块被访问上万次 | §4.2；Figure 6 | 热点复制必要性 | 已确认 |
| N7 | prefill_chunk 阈值通常 >1,000 tokens | §3（"typically larger than 1000 tokens"） | 切块阈值 | 已确认 |
| N8 | 调度实验：8 prefill + 8 decode 实例，回放 23,000 请求；KVCache-centric median TTFT 6.26s，cache-aware 14.36s，load-balancing 60.41s，random 92.07s；SLO 线 ~50s | §6.2；Figure 8（已通过抽取 PDF 第 11 页为图像核对柱状对应） | Q3 调度算法收益 | 已确认 |
| N9 | 过载实验：8 prefill + 8 decode 实例，2× 重放 23,000 请求；Baseline 4183 拒绝、Early Rejection 3771、Early Rejection+Prediction 3589 | Table 3；§8.2 | Q5 早拒绝收益 | 已确认 |
| N10 | 端到端 SLO 设置 TTFT_P90=10×, TBT_P90=5× | §2（"we set TTFTP90=10× and TBTP90=5×"） | SLO 阈值 | 已确认 |
| N11 | ArXiv Summarization：Mooncake-[3P+1D] 比 vLLM-[4M] 多 20% 吞吐；L-Eval 多 40% | §8.1.1（"Mooncake-[3P+1D] achieves throughput improvements of 20% and 40%, respectively"） | 公共数据集吞吐 | 已确认 |
| N12 | 模拟数据（16k/32k/64k/128k 上下文）50%–525% 吞吐提升 | §8.1.2（"Mooncake demonstrates significantly higher throughput, with enhancements ranging from 50% to 525%"） | 长上下文优势 | 已确认 |
| N13 | 真实工作负载：Mooncake-[10P+10D] vs vLLM-[20M]，TTFT 上限 30s、TBT 上限 0.1s/token；Mooncake 多 ~75% 请求；TBT SLO 满足率 ~100% vs vLLM 57% | §8.1.3；Figure 13 | 真实场景收益 | 已确认 |
| N14 | 真实工作负载 TTFT 上限 30s，TBT 上限 0.1s/token | §8.1.3（"the upper limit for the TTFT is set at 30 seconds, while the TBT threshold is capped at 0.1 seconds per token"） | SLO 阈值 | 已确认 |
| N15 | 实验节点配置 8×A800-SXM4-80GB（80GB HBM），NVLink 互联，节点间 RDMA NIC 800 Gbps | §8.1 Testbed | 测试平台背景 | 已确认 |
| N16 | 当前工作负载 KVCache 理论复用上限约 50%；papers.cool 场景约 90% | §9 | 复用率上界 | 已确认 |
| N17 | 论文 v4 发布 3 Sep 2025，23 页 13 图 | arXiv 元数据（"23 pages, 13 figures"） | 文献基础 | 已确认 |

## 构造示例与简化

- **构造示例（手算）**：Algorithm 1 的 TTFT 估计（第四章 5.3 节正文采用 3 个 prefill 实例、阈值=2、固定开销 0.5s + 线性 prefill 时间模型 0.00025 s/token、传输模型 0.0001 s/token）。所有数字为说明算法流程的构造值，**非论文实验数字**。正文将明确标注「构造示例」。
- **F1 的应用估算**：用 LLaMA2-70B 配置（80 层 / 8 KV 头 / 128 head dim / fp16）算得每 token KVCache ≈320 KiB。该数字用于正文说明长上下文 KVCache 的显存压力，标注「以 LLaMA2-70B 配置估算」。

## 不进入正文的边缘论断

- 关于 trace "input-output ratio approximately 720" 的具体含义：原文与按均值的算术比不一致（N2 存疑），正文只引用两独立均值不引用该 ratio。
- 开源仓库 kvcache-ai/Mooncake 的具体组件名（如 Transfer Engine 的多 NIC 带宽聚合、87 GB/s 实测带宽、AttentionStore 性能对比等）— 论文 v4 未给出，无可定位来源。