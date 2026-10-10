# 前缀缓存

- 描述：前缀缓存把已处理请求的 KV cache 留下来，新请求若与旧请求共享前缀，就直接复用这段 KV、跳过它的重算。
- 摘要：共享前缀的请求不必重算同一段 KV：把已算过的 KV cache 按前缀存起来，新请求做一次长前缀匹配，命中就取用这段缓存，只算没命中的后缀。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 长页

- 开头｜总览：前缀缓存是什么、为什么、怎么工作；交代 [KV cache](wiki/kv-cache)、预填充/解码、[因果掩码](wiki/causal-mask)｜520｜blog/vllm-automatic-prefix-caching.md Introduction；blog/vllm-automatic-prefix-caching.md Example workloads；paper/src-v2/sec-intro.tex L23-L28
- ### 前缀为什么能复用｜讲因果注意力下每个位置的 key、value 只由它前面的 token 决定，于是共享前缀的 KV 逐位相同、可整段复用；推导折叠｜560｜paper/src-v2/appendix.tex Background on the KV Cache；blog/vllm-prefix-caching-design.md Overview
- ### 在已有缓存里找到可复用的前缀｜讲怎么用 radix tree（一句解释：压缩前缀树）按 token 序列存已算过的 KV 并做长前缀匹配，命中整段取用、未命中后缀算完再插入；小例子折叠｜560｜paper/src-v2/sec-radix-attention.tex L12-L22；blog/vllm-prefix-caching-design.md Hashing；blog/vllm-prefix-caching-design.md Block Allocation
- ### 缓存以块为单位，只复用整块｜讲 KV cache 按固定大小分块存放、索引按块建、只复用装满的整块，以及 vLLM 的块哈希键构成；14 token 例子折叠｜540｜blog/vllm-prefix-caching-design.md Hashing；blog/vllm-prefix-caching-design.md Example；blog/vllm-prefix-caching-design.md Data Structure
- ### 显存有限：LRU 换出与引用计数｜讲缓存与运行请求共用显存池、不够时按 LRU 换出、引用计数保护正在用的块、先换出叶子；块大小 4 的例子折叠｜540｜paper/src-v2/sec-radix-attention.tex L18-L20；blog/vllm-prefix-caching-design.md Eviction (LRU)；blog/vllm-prefix-caching-design.md Free；blog/vllm-prefix-caching-design.md Block Allocation；blog/vllm-prefix-caching-design.md Example
- ### 调度顺序也影响命中率｜讲请求执行顺序影响命中率，按已匹配前缀长度排序、长前缀优先可逼近最优；命中率定义与最优性定理折叠｜540｜paper/src-v2/sec-radix-attention.tex L50-L55；paper/src-v2/sec-radix-attention.tex L61-L63；paper/src-v2/appendix.tex Proof of the
- ### 省下了什么，代价是什么｜只省预填充不省解码；命中率越高、批越大、吞吐越高、延迟越低；管理开销小可默认开着；数字折叠｜470｜blog/vllm-automatic-prefix-caching.md Limits；paper/src-v2/sec-eval.tex End-to-End Performance；paper/src-v2/sec-eval.tex Ablation Study；paper/src-v2/sec-radix-attention.tex L12-L13
- ### 适用条件与边界｜适合共享前缀多的场景，收益小与不复用的情况，缓存要争显存可能落空，以及一条易混：省下的是预填充的计算与显存，缓存本身仍占存储｜420｜blog/vllm-automatic-prefix-caching.md Limits；blog/vllm-prefix-caching-design.md Hashing；paper/src-v2/sec-radix-attention.tex L20；paper/src-v2/appendix.tex Background on the KV Cache

## 短页

- ### 缓存已有的 KV cache，前缀相同就复用｜是什么：链接 [KV cache](wiki/kv-cache)，说前缀缓存把已处理请求的 KV cache 存下来、供后续请求共用｜300｜blog/vllm-automatic-prefix-caching.md Introduction；paper/src-v2/sec-radix-attention.tex L5-L7
- ### 相同的开头被一遍遍重算｜为什么：系统提示、示例、对话历史这类前缀反复出现，不缓存就要每次重算整段，白白花算力与显存｜300｜paper/src-v2/sec-intro.tex L23-L28；blog/vllm-automatic-prefix-caching.md Example workloads；paper/src-v2/sec-radix-attention.tex L5-L6；paper/src-v2/appendix.tex L26
- ### 按相同开头建索引，命中就取用｜怎么做：给已算过的 KV cache 按它对应的开头建一份索引，新请求来时找出最长的相同开头，这段 KV 直接取用、只算没命中的剩下部分｜320｜paper/src-v2/sec-radix-attention.tex L12-L22
