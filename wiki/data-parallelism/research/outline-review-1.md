# Outline Review 1

## 读者

1. 顺序错乱｜开头｜「区分张量并行」里的「张量并行」我不懂，本节点只说区分，没有先讲它是什么。｜读到这里我缺张量并行的定义，这是第一个节点，此前没有节点讲过它。｜作为前置概念链接成页，或在区分前先定义。
2. 顺序错乱｜一个 DP rank 怎么处理请求｜「独立 KV cache」里的「KV cache」我不懂，本节点直接要求理解它。｜读到这里我缺 KV cache 是什么，开头没讲它，本节也没讲它。｜作为前置概念链接成页。
3. 顺序错乱｜每个 rank 的缓存容量与并发上限｜「token 量」里的 token 我不懂。｜读到这里我缺 token 这个计量单位是什么，此前节点都没讲它。｜作为前置概念链接成页。
4. 顺序错乱｜分流请求并保留各自状态｜短页「token 负载」里的 token 我不懂。｜读到这里我缺 token 是什么，短页前两节都没讲它，不能借长页。｜作为前置概念链接成页。
5. 顺序错乱｜混合并行时的边界｜「MoE 中 attention 与 FFN」里的 MoE、attention、FFN 我都不懂。｜读到这里我缺这三个前置概念，此前节点都没讲它们。｜作为前置概念链接成页。
6. 多余｜混合并行时的边界｜「MoE 中 attention 与 FFN 的分工、同步和空闲 rank 处理」是混合并行或 MoE 变体细节，不是理解推理 DP 必须的内容。｜读到这里我缺的是数据并行的边界，却被带进了别的架构与分工。｜删去该细节，只保留 DP 与 TP 组合的一句边界。
7. 短页标题｜分流请求并保留各自状态｜标题用「并」把“分流请求”和“保留各自状态”两件事并列，「写什么」也分别讲路由和每个 rank 的 KV cache。｜读到这里我不知道这一节到底只讲哪一件事。｜拆成两节：请求分流，以及每个 rank 的独立状态。
8. 遗漏关键内容｜分流请求并保留各自状态｜我会问：为什么每个 rank 必须保留独立 KV cache，不能共享？｜短页只写了“不同 rank 不共享这些状态”，前两节和本节都没回答这个为什么，且不能借长页。｜补一句请求私有性导致这些状态不能共享。

## 对照资料

1. 资料里没有｜开头｜rank 的含义｜依据缺：所引 blog/tensorrt-llm-parallel-strategy.md L4-L13、L32-L36，blog/vllm-data-parallel-deployment.md L3，blog/sglang-server-arguments.md L1216-L1218 只写模型复制和副本数量，没有说明 rank；sources 中说明 rank 的是 “In vLLM, each DP rank is deployed as a separate "core engine" process...” (blog/vllm-data-parallel-deployment.md L14)。｜补引 blog/vllm-data-parallel-deployment.md L14-L20；短页“完整模型的多个服务副本”的 rank 同项另见。
2. 资料里没有｜一个 DP rank 怎么处理请求｜路由器可用的信号｜位置错：依据中的 blog/sglang-server-arguments.md L786-L797 是 --batch-notify-size、streaming-session 等参数，不是路由器信号；正确信号在 “For online deployments, this balancing can be optimized by taking into account the state of each DP engine...” (blog/vllm-data-parallel-deployment.md L20) 和 “Currently, the internal DP load balancing ... based on the running and waiting queues...” (同文件 L148)。｜删去 L786-L797，或改引 blog/vllm-data-parallel-deployment.md L20、L148 与 blog/sglang-server-arguments.md L1228-L1231。
3. 资料里没有｜每个 rank 的缓存容量与并发上限｜每 rank 能保存的 token 量决定该 rank 的并发｜依据缺：该断言只写“自行构造”，而 sources 的原文只分别定义 “The maximum number of running requests.” (blog/sglang-server-arguments.md L446-L447) 和 “The maximum number of tokens in the memory pool...” (同文件 L458-L459)，没有说后者决定前者。｜补引能建立该关系的原文，或改成只列 --max-running-requests 与 --max-total-tokens 两个配置项。
4. 资料里没有｜每个副本为什么有独立 KV cache｜TP 头数不足时复制 KV cache｜漏条件：原文是 “GQA / MQA / MLA: if num_heads < tensor_parallel_size, the KV-cache is replicated on every GPU.” (blog/tensorrt-llm-parallel-strategy.md L73-L78)，大纲写成一般 TP 情形。｜补上 GQA/MQA/MLA 限定。
5. 资料里没有｜它为什么提高吞吐｜请求的逐步生成｜找不到：在 blog/tensorrt-llm-parallel-strategy.md L4-L6、L34-L36、blog/sglang-server-arguments.md L37 内无对应；最接近的是 “Data parallelism is better for throughput if there is enough memory.” (blog/sglang-server-arguments.md L37)。｜删去“逐步生成”，或另补说明逐步解码机制的来源。
6. 资料里没有｜适用条件与边界｜负载不均的浪费｜自编因果：原文只写 “In all cases, it is beneficial to load-balance requests between DP ranks.” (blog/vllm-data-parallel-deployment.md L20)，没有说负载不均会造成“浪费”。｜改成资料支持的“需要做负载均衡”，不要写成浪费。
7. 资料里没有｜有足量请求时扩大吞吐｜单请求用不满一张卡｜找不到：在 blog/tensorrt-llm-parallel-strategy.md L4-L6、L34-L36、blog/sglang-server-arguments.md L37 内无对应；最接近的是 “Parallelism across multiple GPUs becomes necessary when either the model cannot fit in a single GPU’s memory, or a single GPU cannot deliver the desired performance.” (blog/tensorrt-llm-parallel-strategy.md L4-L6)。｜改成“单卡性能不足”，或补充说明单请求利用率低的来源。
8. 资料里没有｜完整模型的多个服务副本｜解释 rank 和 KV cache｜依据缺：所引 blog/tensorrt-llm-parallel-strategy.md L12-L13 与 blog/vllm-data-parallel-deployment.md L3 只说明模型复制/独立 batch，没有 KV cache 说明；sources 里有 “Each DP engine has an independent KV cache...” (blog/vllm-data-parallel-deployment.md L20) 和 “The KV-cache is partitioned...” (blog/tensorrt-llm-parallel-strategy.md L84-L86)。｜补引 blog/vllm-data-parallel-deployment.md L14-L20 与 blog/tensorrt-llm-parallel-strategy.md L81-L86（rank 同项另见“开头”）。
9. 资料里没有｜分流请求并保留各自状态｜按请求数、token 负载或轮询选择 rank｜漏条件：原文写 “The total_tokens algorithm can only be used when DP attention is applied.” (blog/sglang-server-arguments.md L1229)，短页把 token 负载写成无条件选项。｜补上“仅在启用 DP attention 时可用”的限定。

## 篇幅与冗余

1. 重复｜长页「一个 DP rank 怎么处理请求」与长页「每个副本为什么有独立 KV cache」｜两节都写每个 rank 拥有独立 KV cache。｜“说明完整权重和独立 KV cache”；“解释 KV cache 的请求私有性、DP 的缓存划分”。｜保留「每个副本为什么有独立 KV cache」，从「一个 DP rank 怎么处理请求」删去“独立 KV cache”。
2. 重复｜长页「每个副本为什么有独立 KV cache」与长页「每个 rank 的缓存容量与并发上限」｜父节写 DP 的缓存划分，子节又写 KV cache 容量按 rank 划分。｜“DP 的缓存划分”；“KV cache 容量按 rank 划分”。｜保留「每个 rank 的缓存容量与并发上限」，从父节删去“DP 的缓存划分”。
3. 重复｜长页「它为什么提高吞吐」与长页「适用条件与边界」｜两节都写请求流与内存/权重前提。｜“以及请求流与内存前提”；“写单副本容量、请求流条件、重复权重的代价”。｜保留「适用条件与边界」，从「它为什么提高吞吐」删去“以及请求流与内存前提”。
4. 重复｜长页「开头」与长页「适用条件与边界」｜两节都写 DP/TP 区别。｜“区分张量并行”；“DP/TP 一句对比”。｜保留「适用条件与边界」的“DP/TP 一句对比”，从「开头」删去“区分张量并行”。
5. 重复｜短页「完整模型的多个服务副本」与短页「分流请求并保留各自状态」｜两节都写每个 rank 保存自己的 KV cache。｜“解释 rank 和 KV cache”；“每个 rank 保存自己请求的 KV cache，不同 rank 不共享这些状态”。｜保留「分流请求并保留各自状态」，从「完整模型的多个服务副本」删去“和 KV cache”。
6. 无关｜长页「混合并行时的边界」｜整节写 DP 与 TP 组合、MoE 中 attention/FFN 分工、同步和空闲 rank 处理，属于变体或组合实现。｜“说明 DP 与 TP 怎么组合，MoE 中 attention/FFN 分工、同步和空闲 rank 处理”。｜删除该节点；DP/TP 只保留「适用条件与边界」的一句对比。
7. 短页不合格｜短页「完整模型的多个服务副本」｜一节同时讲 DP 复制、rank 和 KV cache 三件事。｜“是什么：推理里的 DP 把同一模型复制到多个 rank，每个 rank 独立处理不同请求；解释 rank 和 KV cache”。｜删去“解释 rank 和 KV cache”，本节只保留“DP 把同一模型复制到多个 rank、各 rank 独立处理请求”。

## 处理

- 读者 1、篇幅 4：成立。开头不再展开张量并行，只在「适用条件与边界」保留一句并链接现行页。
- 读者 2、8，对照 8，篇幅 1、5、7：成立。长页把 KV cache 集中到独立节并链接 [KV cache](wiki/kv-cache)；短页只在该出现的一节解释它是请求私有的已生成键值状态，并链接现行页。
- 读者 3、4：不按前置页处理。token 可用一句话解释为模型输入/输出的文本片段，符合基线外词在本页解释的要求；独立成页会超出本概念范围。
- 读者 5、6，篇幅 6：成立。删除「混合并行时的边界」，MoE、attention、FFN 和 dummy forward 不进本页。
- 读者 7：部分成立。短页必须固定三节，不能拆成四节；把第 3 节改写成“把请求分到带私有 KV cache 的 rank”这一条落地流程，并补请求私有原因。
- 对照 1、2：成立。rank 补引 vLLM L14-L20；路由信号改引 vLLM L20、L148 和 SGLang L1228-L1231。
- 对照 3：成立。删除单独容量节点；容量数值例子折叠在 KV cache 节，明确标为自行构造，只说明每 rank 缓存容量限制该 rank 可保留状态，不写资料外的普遍因果。
- 对照 4：成立。TP 对照限定为 GQA/MQA/MLA 且注意力头数少于 TP 卡数。
- 对照 5、7：成立。删去“逐步生成”“单请求用不满一张卡”，吞吐只按来源写成并行处理不同请求、大批量/高吞吐场景和足量内存条件。
- 对照 6：成立。负载均衡写成“有益/需要做”，不写“浪费”。
- 对照 9：成立。正文不用 token 负载作为无条件选项；路由信号只写请求数和轮询。
