# Outline Review 2

## 读者

无。

## 对照资料

1. 资料里没有｜一个 DP rank 怎么处理请求｜位置错：路由器信号引了 blog/sglang-server-arguments.md L786-L797，但该范围是 --batch-notify-size、--streaming-session 等，不是路由信号。｜“For online deployments, this balancing can be optimized by taking into account the state of each DP engine - in particular its currently scheduled and waiting (queued) requests, and KV cache state.”（blog/vllm-data-parallel-deployment.md L20）；“internal DP load balancing ... based on the running and waiting queues...”（同文件 L148）；策略选项见 blog/sglang-server-arguments.md L1228-L1231。｜删去 SGLang L786-L797，保留或补引 vLLM L20、L148 和 SGLang L1228-L1231。
2. 资料里没有｜每个副本为什么有独立 KV cache｜位置错并漏条件：“TP 头数不足时复制 KV cache”所引的 TRT L81-L86 不覆盖该条件，原文在更早的 L73-L78，且只适用于 GQA/MQA/MLA。｜“GQA / MQA / MLA: if num_heads < tensor_parallel_size, the KV-cache is replicated on every GPU.”（blog/tensorrt-llm-parallel-strategy.md L73-L78）。｜改为“GQA/MQA/MLA 且 num_heads < tensor_parallel_size 时 KV cache 被复制”，并把依据改为 L73-L78；DP 的 KV-cache 划分仍引 L81-L86。

## 篇幅与冗余

1. 重复｜长页「一个 DP rank 怎么处理请求」与长页「每个副本为什么有独立 KV cache」｜“请求为什么不跨 rank 互通”与“KV cache 的请求私有性”是同一隔离机制。｜“请求为什么不跨 rank 互通”/“KV cache 的请求私有性”。｜保留“每个副本为什么有独立 KV cache”的私有性解释，删去“一个 DP rank 怎么处理请求”中的“请求为什么不跨 rank 互通”。
2. 重复｜长页「每个副本为什么有独立 KV cache」与长页「适用条件与边界」｜“每个 rank 的容量上限”与“单副本容量”是同一容量约束。｜“每个 rank 的容量上限”/“单副本容量”。｜保留“每个副本为什么有独立 KV cache”的容量上限和折叠小例子，删去“适用条件与边界”中的“单副本容量”。
3. 重复｜长页「它为什么提高吞吐」与长页「适用条件与边界」｜“大批量、高吞吐场景和足量内存前提”与边界节的“单副本容量、请求流条件”重复。｜“大批量、高吞吐场景和足量内存前提”/“单副本容量、请求流条件”。｜保留“适用条件与边界”中的条件，删去“它为什么提高吞吐”中的“大批量、高吞吐场景和足量内存前提”。
4. 预算不合理｜长页「它为什么提高吞吐」｜核心机制节预算 380 小于边界节 420。｜“380”/“420”。｜边界节预算由 420 改为 350。
5. 重复｜短页「完整模型的多个服务副本」与短页「有足量请求时扩大吞吐」｜“同一模型复制到多个 rank，每个 rank 独立处理不同请求”与“多份完整副本并行处理不同请求”重复。｜“同一模型复制到多个 rank，每个 rank 独立处理不同请求”/“多份完整副本并行处理不同请求”。｜保留“完整模型的多个服务副本”中的定义，删去“有足量请求时扩大吞吐”中的“多份完整副本并行处理不同请求”。
6. 短页不合格｜短页「把请求分到带私有 KV cache 的 rank」｜同一节同时讲路由选择和 KV cache 私有性两件事。｜“路由器按请求数或轮询选择 rank；说明 KV cache 的请求私有性和每 rank 独立缓存”。｜删去“路由器按请求数或轮询选择 rank”，只保留 KV cache 私有性和每 rank 独立缓存这一件事。

## 处理

- 对照 1：不成立。这是审查到的第一版依据；现行大纲已删去 SGLang L786-L797，只引 vLLM L20、L148 和 SGLang L1228-L1231。
- 对照 2：不成立。这是审查到的第一版“TP 头数不足”表述；现行大纲已删除 TP 对照，KV cache 节只引 TRT L81-L86 讲 DP 的缓存划分。
- 篇幅 1：成立。请求分流节点只写机制和路由信号；私有性原因集中到 KV cache 节。
- 篇幅 2：成立。“每个 rank 的容量上限”与“单副本容量”容易读成重复；KV cache 节保留缓存容量例子，边界节改为模型权重要放进每个 rank。
- 篇幅 3、4：成立。吞吐节只讲机制和场景，条件集中到边界节；边界预算由 420 改为 350。
- 篇幅 5：成立。短页第 2 节只回答为什么需要，不重复第 1 节的定义。
- 篇幅 6：成立。短页第 1 节把“rank 及其私有 KV cache”作为 rank 定义的一部分，第 3 节只讲路由。
