# Chunked Prefill（chunked-prefill）内容范围

## 1. 概念歧义处理

- "chunked prefill"（分块预填充）在文献中机制一致：把一次 prefill 请求切成多个近似等大的块逐块计算。状态：已裁定。
- 相关但需区分的术语：
  - piggybacking / decode-maximal batching（Sarathi, arXiv:2308.16369）：chunked prefill 之上的批构造策略，一个批次装一个 prefill chunk + 尽量多的 decode。本页纳入（两者在文献中成对出现，拆开会造成理解断点）。
  - stall-free scheduling（Sarathi-Serve, arXiv:2403.02310）：不暂停进行中 decode 的调度策略。本页纳入其核心思想。
  - chunked pipeline parallelism（CPP）：chunked prefill 与 PP 的组合以消气泡。本页纳入（Beyond the Buzz 论文的 prefill 池策略）。
  - chunked prefill 与 continuous batching 的关系：chunked prefill 是批内容构造方式，continuous batching（Orca 的 iteration-level scheduling）是批调度方式；Sarathi-Serve 建立在 continuous batching 之上。正文区分，continuous batching 一句话定位不展开。
- "TTFT/TBT/TPOT" 与本仓库 moe-serving 页使用的 "TTFT/TPOT" 对应：Sarathi 文献用 TBT（time-between-tokens），本页跟随文献用 TBT 并在首次出现时说明与 TPOT 同义。状态：已裁定。

## 2. 概念含义

### 2.1 简要定义

chunked prefill 把一个长输入的 prefill 计算切成多个近似等大的块，逐块执行；每个块计算时能利用前面块产生的 KV cache，但不依赖它们的输出。配套的调度策略让 decode 请求与 prefill 块同批执行（piggybacking）且从不因新请求进入而暂停（stall-free）。

### 2.2 正式定义与来源

- chunked prefill 定义：将 prefill 请求切成等计算量的块。来源：Sarathi §3.1（"chunked-prefills, which splits a prefill request into equal sized chunks"）。
- 块间依赖：每个块的 attention 需要读前面所有块的 KV cache；FFN 计算量不变。来源：Sarathi §4.2（"the attention kernel in every subsequent chunk after the first will have to reread all the KV pairs of the prior tokens"）。
- piggybacking / decode-maximal batching：一个批次 = 一个 prefill chunk + 尽量多的 decode。来源：Sarathi §3.2（"constructs a batch using a single prefill chunk and populates the remaining slots with decodes"）。
- stall-free：新请求以 chunk 形式加入当前迭代批次，不暂停进行中的 decode。来源：Sarathi-Serve §3（"stall-free schedules that adds new requests in a batch without pausing ongoing decodes"）。
- generation stall：prefill-prioritizing 调度下进行中 decode 被长 prefill 迭代阻塞数秒的现象。来源：Sarathi-Serve §2（"generation stalls ... can last over several seconds"）。

### 2.3 本文采用的语境

面向 LLM 推理 serving（单卡与 PP 部署）的 chunked prefill 与配套调度。

### 2.4 包括什么

- 为什么长 prefill 伤害 decode 延迟（generation stall 机制）
- chunked prefill 的切块与块间依赖（KV 累积、不看输出）
- piggybacking 批构造与"decode 搭车近乎免费"的原因
- stall-free 调度与 token budget
- chunk 大小的代价（重复读 KV、算术强度、tile 量化）
- 与 PP 组合消气泡（CPP）

### 2.5 不包括什么

- continuous batching 的完整机制：Orca 的贡献，本页一句话定位，不展开调度细节。
- PagedAttention 显存管理：与 chunked prefill 正交，不展开。
- PP 气泡公式推导：model-parallelism 页职责，本页直接引用其结论。
- Prefill/decode 两阶段与 KV cache 的基础定义：moe-serving 页职责，本页引用。

## 2.6 相邻概念

- chunked prefill vs 分离式 serving：chunked prefill 是 co-located 部署内部的批构造优化；PD 分离把两阶段放到不同实例。Beyond the Buzz 论文将 piggybacked co-located 作为分离的对照基线。本页末尾链接论文页。
- speculative decoding：同为延迟优化但机制无关，不纳入。

## 3. 学习目标

### Q1：为什么一个长 prefill 会拖慢同机所有进行中的 decode（generation stall）？

- 完成答案：读者应能说明：连续批处理中 prefill 与 decode 混在同一次迭代；prefill 迭代处理上千 token、耗时远长于 decode 迭代；若调度器优先完成 prefill（vLLM 类做法），进行中的 decode 在整个长 prefill 期间不产出 token，token 间隔被拉长到秒级；decode 本身 memory-bound，与 compute-bound 的 prefill 同批时被 prefill 的时长主导。
- 为什么是核心目标：不理解这个问题就无法理解 chunked prefill 为什么有效。
- 依赖内容：prefill/decode 两阶段（moe-serving 链接）、memory-bound vs compute-bound。

### Q2：chunked prefill 怎么切块，块与块之间的依赖关系是什么？

- 完成答案：读者应能说明：按（近似）等计算量切块；每块的 attention 要读前面所有块的 KV cache（第一块的 KV 被后续 N-1 块重复读）；FFN 对每块独立计算、总量不变；块之间不传递输出（prefill 只算 KV，不生成新 token 依赖）。
- 为什么是核心目标：这是机制的数学核心，也是代价（重复 KV 读）的来源。
- 依赖内容：KV cache 结构（moe-serving 链接）、attention 因果掩码（standard-attention/causal-mask 链接）。

### Q3：piggybacking 为什么让 decode"搭车"近乎免费，stall-free 调度怎么保证 decode 不被卡？

- 完成答案：读者应能说明：decode 是 memory-bound（读权重的时间占主导），与 compute-bound 的 prefill chunk 同批时，权重只需读一次即可同时服务两者，decode token 的边际成本比纯 decode 批低一个数量级；stall-free 用 token budget 限制每次迭代的总 token 数，新请求以 chunk 加入、不暂停进行中的 decode，每次迭代时长有上界，token 间隔不再有秒级尖刺。
- 为什么是核心目标：这是 Sarathi 的两个核心收益（吞吐、延迟）的机制。
- 依赖内容：Q1 Q2、memory-bound 概念。

### Q4：chunk 大小（token budget）怎么选，选小了有什么代价？

- 完成答案：读者应能说明：chunk 越小每次迭代越短、TBT 越稳，但三个代价上升——KV cache 重复读次数增多、算术强度下降（GPU 利用率降低）、kernel 启动等固定开销摊薄变差；tile 量化要求 chunk 对齐 GPU tile 尺寸（257 比 256 慢 32% 的实例）；选择由 TBT SLO 与开销的权衡决定。
- 为什么是核心目标：这是工程落地的关键权衡，也是 Beyond the Buzz 论文"context chunking 有效性依赖场景"结论的前置。
- 依赖内容：Q2 Q3。

### Q5：chunked prefill 与流水线并行组合（CPP）为什么能同时消气泡？

- 完成答案：读者应能说明：PP 气泡源于各 micro-batch 计算时长不均（模型并行页结论）；chunked prefill 把 prefill 切成等计算量的块，使各 micro-batch 时长均匀，流水线 slot 被填满、气泡显著缩小；Beyond the Buzz 论文中 CPP 是 prefill 池在严格 FTL 约束下的最优策略（FTL 随 PP 增大而降低、吞吐保持）。
- 为什么是核心目标：连接 model-parallelism 页与论文页的桥梁。
- 依赖内容：PP 气泡机制（model-parallelism 链接）、Q2。

## 4. 内容分级

核心内容：
- generation stall 机制（Q1）
- 切块规则与块间 KV 依赖（Q2）
- piggybacking 的 memory-bound 论证（Q3）
- token budget 与 stall-free（Q3 Q4）
- chunk 代价三项与 tile 量化（Q4）
- CPP 消气泡机制（Q5）

辅助内容：
- continuous batching 定位（衔接）
- decode 搭车的量化数字（Sarathi：搭车 decode 成本比纯 decode 批低一个数量级）
- 与 PD 分离的关系（末尾衔接论文页）

扩展内容（排除）：
- Sarathi-Serve 的 admission control 细节
- Vidur 自动调参
- tile 量化的 GPU 微架构原理

## 5. 前置知识映射

- prefill/decode 两阶段、KV cache、TTFT/TPOT：moe-serving 页已有，正文引用。
- attention 因果掩码：causal-mask 页已有，Q2 块间依赖处引用。
- memory-bound vs compute-bound：gpu-execution-model 页已有（覆盖算术强度与访存瓶颈），Q1 Q3 引用。
- all-reduce / 流水线气泡：model-parallelism 页（本任务同时递归生成），Q5 引用。
- 标准 attention 结构：standard-attention 页已有。
- MoE/EP：本页不需要。

## 6. 明确不展开的内容

- PagedAttention 与显存碎片：与 chunked prefill 正交，moe-serving/vLLM 文档已有。
- Orca continuous batching 完整算法：一句话定位即可。
- GPU tile 微架构：引用 tile 量化现象，不展开硬件实现。
- 训练场景的 gradient checkpointing：无关。

## 7. 常见误解和适用边界

误解 1
- 错误理解：chunked prefill 让 prefill 变快了。
- 正确结论：总计算量不降反微升（KV 重复读、固定开销）；它改善的是延迟结构（TBT 平稳化）与批内均衡，不是单请求 prefill 速度。
- 形成原因：把"切小"误当"加速"。
- 影响目标：Q2 Q4。

误解 2
- 错误理解：块之间需要串行传递 hidden states/输出。
- 正确结论：块间只通过 KV cache 传递（attention 需要读前面块的 K/V）；每块独立过全部层、不消费前块的层输出（prefill 阶段各 token 的输出即其位置的 KV 与 logits，互相独立）。
- 形成原因：与 RNN 式串行混淆。
- 影响目标：Q2。

误解 3
- 错误理解：piggybacking 意味着 decode 没有任何额外成本。
- 正确结论：搭车 decode 的成本"最多低一个数量级"（up to an order of magnitude less），不是零；批内 decode 数量受批容量限制。
- 形成原因：把数量级近似当成精确零。
- 影响目标：Q3。

适用边界
- chunked prefill 是 co-located 部署内的优化；它的有效性高度依赖场景（Beyond the Buzz：注意力机制 MLA vs GQA、延迟目标、流量模式），不是无条件收益——正文末尾引用论文页结论。
- "搭车免费"的论证依赖 prefill chunk 足够大以饱和 GPU（Sarathi：LLaMA-13B 在 A6000 上 512 token 即饱和）；chunk 远小于饱和点时论证弱化。
- 气泡消除结论基于 micro-batch 时长均匀化；请求长度差异极大时仍有残余不均。

## 8. 论断分级标注

- 切块机制/块间 KV 依赖/piggybacking 定义/generation stall：论文明确声称（Sarathi、Sarathi-Serve，逐条定位见 evidence.md）。
- KV 重复读次数（N 块时第一块被读 N-1 次）：论文明确声称（Sarathi-Serve §4.3；Sarathi §4.2 表述为 N 次，两文有一处出入，以 Sarathi-Serve 为准并记录冲突）。
- tile 量化 257/256 → 32%：论文明确声称（Sarathi-Serve §4.3）。
- decode 搭车低一个数量级：论文明确声称（Sarathi 摘要 "cost up to an order of magnitude less"）。
- 512 token 饱和 A6000（LLaMA-13B）：论文明确声称（Sarathi §4.1 附近 "a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"）。
- CPP 是 Beyond the Buzz 论文 prefill 池最优策略：论文明确声称（Beyond the Buzz §4）。
- "chunked prefill 不生成新 token 依赖"的机制表述：基于证据的推断（由 prefill 的因果结构与 Sarathi 图 6 的数据流推出，标注推断）。
- 手算例子中的自设数字：构造示例。
