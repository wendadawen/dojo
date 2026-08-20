# prefix-caching 核心论断与证据

## C 论断

- C1：SGLang 指出既有系统在请求完成后丢弃 KV cache，阻止跨调用复用、显著拖慢执行；其系统以 radix tree 维护所有请求 KV cache 的 LRU 缓存。来源：SGLang 论文（arXiv:2312.07104）§1/§3.2（"Existing LLM programming systems typically discard... preventing the KV cache from being reused across multiple calls and significantly slowing down the execution. Instead, our system maintains an LRU cache of the KV cache for all requests within a radix tree"）。
- C2：RadixAttention 在运行时自动、系统地复用 KV cache：保留提示与生成结果的缓存于 radix tree，支持高效前缀搜索、复用、插入、逐出；实现 LRU 逐出与 cache-aware 调度；兼容 continuous batching、paged attention、张量并行；无命中时开销可忽略。来源：SGLang 论文 §3.2。
- C3：radix tree 是经典 trie（前缀树）的空间高效替代，边上可标注变长序列（而非单个元素）；SGLang 用它维护 token 序列到 KV cache 张量的映射，KV cache 以非连续分页布局存储、页大小等于 1 token。来源：SGLang 论文 §3.2（"A radix tree is a data structure that serves as a space-efficient alternative to a classical trie (prefix tree)... In our system, we utilize a radix tree to manage a mapping between sequences of tokens, and their corresponding KV cache tensors. These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token"）。
- C4：GPU 显存被 KV cache 快速填满，SGLang 采用简单 LRU 策略：先逐出最近最少使用的叶子；逐出叶子后公共祖先可继续被复用，直到祖先自身成为叶子也被逐出。来源：SGLang 论文 §3.2（"we introduce a simple LRU eviction policy that evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted"）。
- C5：continuous batching 下不能逐出正在运行的批使用的节点；每节点维护引用计数，计数为 0 才可逐出；系统不预分配固定缓存池，缓存 token 与运行请求共享同一内存池。来源：SGLang 论文 §3.2（"each node maintains a reference counter indicating how many running requests are using it. A node is evictable if its reference counter is zero... we let the cached tokens and the currently running requests share the same memory pool"）。
- C6：上下文/前缀缓存跨请求识别公共前缀，被 OpenAI、Google 等提供商广泛采用（prompt caching / context caching）。来源：Strata 论文（arXiv:2508.18572v2）§2.3（"systems exploit context caching across requests by identifying common prefixes using structures like prefix trees or hash maps, widely adopted by providers such as OpenAI and Google"）。
- C7：SGLang 用 RadixTree，vLLM 与 Mooncake 用哈希机制（由 token ID 与前缀页哈希生成唯一页标识），LMDeploy 用粗粒度混合 trie；Strata 基于 SGLang 把 RadixTree 扩展为 HiRadixTree。来源：Strata 论文 §6（"SGLang employs a RadixTree for tracking shared context. Other serving engines, such as vLLM and Mooncake, utilize hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes. LMDeploy adopts a hybrid approach by constructing coarser-grained tries. Strata builds upon SGLang by extending its RadixTree to a HiRadixTree"）。
- C8：缓存匹配按页进行（per-page basis），因此页大小影响命中率：页越大匹配粒度越粗。实验：SGLang-HiCache 页 512 相比页 32 命中率低 2.4%、最优页 512 的吞吐也只有 Strata-IO 的 93%。来源：Strata 论文 §3.1（"cache matching is performed on a per-page basis"）与 §5.3.2。
- C9：多轮对话场景下，前一轮的完整序列（提示+生成）构成下一轮输入的前缀，因此生成结果也应入缓存。来源：SGLang 论文 §3.2（"retains the cache for prompts and generation results"）+ 推断标注（对话结构直接得出）。

## F 公式

- F1（构造算例用）：页命中数 $=\lfloor \text{共享前缀长度}/\text{页大小}\rfloor$；命中 token 数 = 页命中数 × 页大小。构造示例专用。
- F2（引用 paged-attention 页）：页数 $=\lceil \text{token 数}/\text{页大小}\rceil$。

## N 数字

- N1：SGLang 页大小 = 1 token。来源：SGLang 论文 §3.2。
- N2：Strata 实验：页 512 比 页 32 命中率低 2.4%；SGLang-HiCache 最优页（512）吞吐为 Strata-IO 的 93%。来源：Strata 论文 §5.3.2（条件：Qwen-14B、H200、LooGLE）。
- N3：构造算例：两请求共享 100 token 前缀——页 1 命中 100 token、页 32 命中 96 token（3 页）、页 256 命中 0 token。来源：F1 构造（标注构造示例）。
- N4：分层系统借 CPU 内存达到约 95% 命中率（Strata 实验、LooGLE）。来源：Strata 论文 §5.2.1（本页结尾过渡引用）。

## 冲突与不确定项

- 无实质冲突。"哈希 vs radix tree 优劣"的对比表述标注为分析性推断（两机制描述各自有原文依据，优劣判断为页面分析）。
