# Chunked Prefill（chunked-prefill）核心论断与证据

## C 论断

- C1：LLM 推理请求分 prefill（处理输入 prompt、产出第一个 token）与 decode（逐个自回归生成）两阶段；prefill 并行处理全部输入 token、小批量即可饱和 GPU 算力，decode 每次迭代每请求只处理一个 token、利用率低。来源：Sarathi 摘要与 §1（"While the prefill phase effectively saturates GPU compute at small batch sizes, the decode phase results in low compute utilization"）。适用条件：通用自回归 LLM。置信：已确认。
- C2：小批量下 decode 每 token 成本可比 prefill 每 token 成本高约 200 倍（A6000、LLaMA-13B 实测）。来源：Sarathi §1（"at small batch sizes, the decode cost per token can be as high as ∼200 times the prefill cost per token"）。适用条件：A6000 + LLaMA-13B 小批量。置信：已确认。
- C3：LLaMA-13B 在 A6000 上 512 token 的单个 prefill 请求即可饱和 GPU 算力（批大小为 1 时）。来源：Sarathi §1（"a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"）。适用条件：该模型-硬件组合。置信：已确认。
- C4：chunked prefill 把一个 prefill 请求切成等（计算量）大的块。来源：Sarathi §3.1（"chunked-prefills, which splits a prefill request into equal sized chunks"）。适用条件：通用。置信：已确认。
- C5：每个后续块的 attention 必须读取同一 prompt 前面所有块的 KV cache；切成 N 块时第一块的 KV 被重复读 N-1 次（Sarathi-Serve 表述）、Sarathi 表述为 N 次；FFN 计算对每块独立进行、总量不变。来源：Sarathi-Serve §4.3（"if a prefill sequence is split into N chunks, then the first chunk's KV-cache is loaded N − 1 times, the second chunk's KV-cache is loaded N − 2 times, and so on"）；Sarathi §4.2（"the first chunk's KV cache is loaded N times"）。冲突处理：两文一处为 N-1、一处为 N（是否计入第一块自身的首次读取），语义等价（都指后续块的重复读取带来的额外访存），页面采用 Sarathi-Serve 的 N-1/N-2 计数并加注。置信：已确认（含记录的表述差异）。
- C6：块之间不消费对方的层输出：prefill 各 token 位置的输出（KV 与 logits）互相独立，块间唯一传递的是累积的 KV cache。来源：Sarathi-Serve Figure 5 数据流 + prefill 因果结构。性质：基于证据的推断（正文标注），依据：chunk 的定义（等大块独立前向）与注意力因果掩码下 token 间无输出依赖。置信：已确认（作为推断标注）。
- C7：decode-maximal batching（piggybacking）：一个批次由一个 prefill chunk 加尽量多的 decode 组成；prefill chunk 饱和算力，搭车 decode 的成本比纯 decode 批最多低一个数量级。来源：Sarathi §3.2 与摘要（"constructs a batch using a single prefill chunk and populates the remaining slots with decodes ... decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch"）。适用条件：chunk 足够大以饱和算力。置信：已确认。
- C8：chunked prefill 使一个 prefill 请求产生多个批次，扩大 decode 可搭车的覆盖面。来源：Sarathi §1（"Chunked-prefills allows us to construct multiple hybrid batches from a single prefill request, thereby increasing the coverage of decodes that can piggyback"）。置信：已确认。
- C9：generation stall：prefill 优先调度下，进行中 decode 在长 prefill 迭代期间不产出，vLLM 中可见持续数秒的 token 间隔尖刺。来源：Sarathi-Serve §2（"prefill-prioritizing schedulers lead to an undesirable phenomenon that we refer to as generation stalls ... stall in vLLM can last over several seconds"）。适用条件：连续批处理 + prefill 优先调度。置信：已确认。
- C10：stall-free 调度用 token budget 限制每次迭代的 token 总量，新请求以 chunk 形式加入当前批次、不暂停进行中的 decode。来源：Sarathi-Serve §3.2（"restricting the computational load in every iteration, stall-free"）、§4.2 算法 3。置信：已确认。
- C11：token budget 的权衡：小预算降 TBT 但增加切块开销（GPU 利用率降低 + KV 重复读）；预算由 TBT SLO 与开销权衡决定，可用一次性 profiling 设定。来源：Sarathi-Serve §4.3（"From a TBT minimization point of view, a smaller token budget is preferable ... However, smaller token budget can result in excessive chunking of prefills resulting in overheads"）。置信：已确认。
- C12：tile 量化：矩阵维度不对齐 GPU tile 时部分线程块做多余计算；chunk 257 比 256 的 prefill 时间高 32%（实例）。来源：Sarathi-Serve §4.3（"using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256"）。适用条件：具体 GPU 与 kernel。置信：已确认。
- C13：切块的算术强度随 chunk 变小而下降，影响 prefill 效率；kernel 启动等固定开销也随之摊薄变差。来源：Sarathi §4.2（"the arithmetic intensity of chunked-prefills computation decreases as the chunk size becomes smaller"）、Sarathi-Serve §4.3。置信：已确认。
- C14：PP 部署中，chunked prefill 使各 micro-batch 计算时长均匀，显著缩小流水线气泡；Sarathi 在 GPT-3 PP 部署上气泡缩小 6.29×、端到端吞吐提升 1.91×。来源：Sarathi 摘要（"reduces bubbles by 6.29×, resulting in an end-to-end throughput improvement of 1.91×"）。适用条件：GPT-3 PP 实验设置。置信：已确认。
- C15：Sarathi-Serve 相比 vLLM：Mistral-7B 单 A100 服务容量 2.6×，Yi-34B 双 A100 至 3.7×，Falcon-180B PP 至 5.6×。来源：Sarathi-Serve 摘要。适用条件：各实验设置、尾延迟约束下。置信：已确认。
- C16：Beyond the Buzz 论文发现 Chunked Pipeline Parallelism 是 prefill 池在严格 FTL SLA 下兼顾高吞吐的最优策略：DeepSeek-R1、ISL 256K、64 GPU（EP×PP=64）上 FTL 随 PP 增大而降低且吞吐保持。来源：Beyond the Buzz §4 图 5（"Chunked pipeline parallelism during Prefill is an optimal strategy to maximize throughput while complying with strict FTL SLA"）。适用条件：该模拟设置。置信：已确认。
- C17：Beyond the Buzz 论文发现 co-located 部署下 context chunking（piggybacking）的有效性对注意力机制敏感：DeepSeek-R1 的 MLA 在 piggyback 下因每块重复计算 down/up 投影产生额外开销，可临时缓存上投影 KV 缓解。来源：Beyond the Buzz §4.1 Model architecture sensitivity 段。适用条件：MLA 架构 + chunked piggybacking。置信：已确认。

## F 公式

- F1：N 块重复读总量：$\sum_{i=1}^{N}(N-i) = \frac{N(N-1)}{2}$ 次额外 KV 块读取（第 $i$ 块的 KV 被后续块读 $N-i$ 次）。来源：由 C5 计数直接求和（构造推导，正文标注为"由 C5 计数推出"）。置信：已确认（推导自已确认论断）。

## N 数字

- N1：decode 每 token 成本 ≈ 200× prefill 每 token 成本（小批量，A6000，LLaMA-13B）。来源：Sarathi §1。
- N2：512 token 单请求饱和 A6000（LLaMA-13B）。来源：Sarathi §1。
- N3：chunk 257 vs 256：prefill 时间 +32%。来源：Sarathi-Serve §4.3。
- N4：Sarathi PP 气泡缩小 6.29×、吞吐 +1.91×（GPT-3）。来源：Sarathi 摘要。
- N5：Sarathi-Serve vs vLLM 服务容量：2.6×（Mistral-7B/A100）、3.7×（Yi-34B/2×A100）、5.6×（Falcon-180B/PP）。来源：Sarathi-Serve 摘要。
- N6：LLaMA-13B/A6000：Sarathi decode 吞吐至多 10×、端到端吞吐 1.33×；LLaMA-33B/A100：端到端 1.25×、decode 吞吐至多 4.25×。来源：Sarathi 摘要。

## 原文图候选

本页机制图自绘（chunk 数据流示意、piggyback 批构造示意、token 间隔平稳化对比），不内嵌论文原图。

## 构造示例

- 手算重复读：prompt 4096 token 切 8 块（每块 512），额外 KV 读取次数 $8\times 7/2 = 28$ 块次；切 4 块时 $4\times 3/2=6$ 块次——块越多重复读越多。数字自设，标注构造示例。
- piggyback 批构造追踪：批容量 512 token，1 个 512-token prefill chunk + 64 个 decode，展示单次迭代内两类工作如何共存。数字自设，标注构造示例。
