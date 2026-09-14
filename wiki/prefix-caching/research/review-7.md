<!-- review-meta
round: 7
page: wiki/prefix-caching/index.html
reviewed_content_sha256: 540019dc1c194296
-->
# 前缀缓存审查记录（第 7 轮）

- 页面版本：247e4cfe284876c5206ee97d49bdc23318c3a951（index.html 工作树哈希）
- 审查时间：2026-09-14 17:43
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：引言与核心问题 → 1. 什么能复用——共享前缀从哪来 → 2. radix tree——组织与匹配 → 3. 缓存满了怎么办——LRU 逐出与引用计数 → 4. 命中率——页大小这一独立变量 → 5. 两种实现——radix tree 与哈希 → 来源与范围说明（含全部 details 折叠块、图注、表格与附录小节）
- 核对所用版本：SGLang 论文 arXiv:2312.07104v2（2024-06-06，NeurIPS 2024）；Strata 论文 arXiv:2508.18572v1（2025-08-26，OSDI 2026）。注意：本轮核对的是 Strata **v1**——arXiv 提交历史确认该文目前只有 v1（无 v2）；另据 USENIX OSDI'26 technical-sessions 页面核实 Strata 确为 OSDI 2026 收录论文（Track 1 KV Cache and Long Context），故页面标注的「OSDI 2026」成立。
- 机械验证：`.dojo/scripts/validate.py wiki/prefix-caching/index.html` → `validation ok`；`dojo:topics`（内存与缓存、推理系统）与 `dojo:tag`（KV cache）均在 AGENTS.md / catalog_builder.py 词表内；页面内引用的概念页 ../kv-cache/、../paged-attention/、../strata/、../mooncake/ 均真实存在；overview.html 与 index.html 互链；无非 LaTeX 数学字符；图内公式走 `<foreignObject>`，`<text>` 内无 ASCII 数学近似；无脚本时正文（含折叠块）仍完整可读。

## 核对要点（来源依据摘录）

- C1 引文：SGLang 论文原文 KV cache "of a request is discarded after processing is completed … preventing the KV cache from being reused across multiple calls … significantly slowing down the execution"——与页面「阻止了跨调用复用、显著拖慢执行」一致。
- C3/N1：原文 KV cache tensors "stored in a non-contiguous, paged layout"、"the size of each page is equivalent to one token"、"a mapping between sequences of tokens, and their corresponding KV cache tensors"；radix tree 为 "a space-efficient alternative to a classical trie"，用于 "efficient prefix search, reuse, insertion, and eviction"——页面表述一致。
- C4/C5/C9：原文 "a simple LRU eviction policy that evicts the least recently used leaf first"、"By evicting leaves first, we enable the re-use of their common ancestors"、"each node maintains a reference counter … A node is evictable if its reference counter is zero"、"retains the cache for prompts and generation results in a radix tree"——一致。
- C8 引文：Strata §3.1 "cache matching is performed on a per-page basis"——与页面逐字一致。
- N2 引文：Strata §5.3.2 原文 "Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate."——页面「页 512 的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%、最优页吞吐为 93%」与原文完全对应；条件 Qwen2.5-14B（§5.1 全名 Qwen2.5-14B-Instruct-1M）、H200（"Qwen-14B model on an H200 platform"）均核对无误；LooGLE 为 §5.3 的分析负载（§5.3.3 "In addition to our original (shuffle) workload, we create two additional workloads based on the LooGLE dataset" 反证原始 workload 即 LooGLE 系），页面标注可接受。
- N5：Strata §2.2 "Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang"；§3.1 页大小 32 为 "a maximum supported size in vLLM for CUDA GPUs"——页面表注一致。
- C6/C7/C10：Strata §2.3（"widely adopted by providers such as OpenAI"、context caching）、§3.2（delay hit 定义："arises when multiple requests for the same data object arrive and queue while an initial cache miss is still being resolved"）、§4.1（"HiRadixTree, which is an extension to SGLang's RadixTree"，作为页表存 per-page metadata）、§6（vLLM/Mooncake "hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes"、LMDeploy "coarser-grained tries"、Mooncake "disaggregated KV Cache"）——逐条对应页面 C7/C10 描述。
- 构造算例复算：R1=200+30=230、R2 命中 200 只算 40、R3=200+30=230，合计 500 vs 无缓存 690，与页面一致；100 token 页表 ⌊100/1⌋=100、⌊100/32⌋=3→96、⌊100/256⌋=0，与页面一致。F1 已声明为按页匹配机制（C8）的直接换算、非论文公式——标注正确。

## 问题

- [轻微·格式] 来源与范围说明 →「外部数字与实验条件（N）」N3 条（index.html 第 321 行）与「构造示例」小节（第 324 行）：N3 是构造算例（"共享 100 token × 页 1/32/256 算例：构造示例"），却列在"外部数字与实验条件"小节下，而同一算例已在下方「构造示例」小节声明为"100 token × 三种页大小的命中表均为构造算例"，同一算例两处声明且归类与小节名不符｜引文依据：不适用｜修复要求：将 N3 从"外部数字与实验条件（N）"移出，仅保留「构造示例」小节的声明（或反之），使该算例只在一处声明并归入正确小节｜修复：｜复验：
- [轻微·技术] 来源与范围说明 → N4 条（index.html 第 321 行）：N4 在 N 小节列出但正文无 `[N4]` 标记（条目自述"本文未正文引用，仅口径备查"），与 style-guide §6「正文使用 `<sup>[Cx]</sup>` …与来源章节双向对应」不符｜引文依据：Strata §5.2.1 "consistently reaching approximately 95% cache hit rate by leveraging CPU memory"（数值本身核对无误，§5.1 "1 TB of system DRAM as pinned memory" 亦一致；问题在编号无正文对应项）｜修复要求：删除 N4，或在正文相应处（如第 4 章命中率讨论）补 `[N4]` 标记使双向对应成立｜修复：｜复验：
- [轻微·技术] 第 5 章「两种实现」正文（index.html 第 292 行）："把这份池化推进到整个集群、并让它反过来成为请求调度中心的设计，即分离式服务架构 Mooncake"——所引 Strata §6 仅称 Mooncake "exploits using resources including CPU, DRAM, SSD and NIC to establish a disaggregated KV Cache"，支持"分离式"但未支持"让缓存池反过来成为请求调度中心/整个集群"这一描述；本页 C7 只覆盖 Mooncake 的哈希实现，未覆盖其调度角色，属无 inline 来源的机制描述｜引文依据：Strata §6 "Mooncake exploits using resources including CPU, DRAM, SSD and NIC to establish a disaggregated KV Cache."｜修复要求：删去"让它反过来成为请求调度中心"或补上支持该描述的来源（如 Mooncake 论文），把结构性描述降为可核对的表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 逐项判定说明：核心论断（C1–C10、F1、N1/N2/N5）全部回源核对成立，无来源不支持、无把实验条件观察写成无条件论断、无推断包装成结论；两处数字（2.4%、93%）与 Strata §5.3.2 原文逐字对应；summary/overview/正文/图注/表格中的数字与引文编号一致；公式可复算且符号单义；summary 内公式可由 KaTeX 渲染；页面无「本页」自称（"本文"为 style-guide §12 明文允许的写法）；无会话指代、无调试叙事、无临场评价、章节衔接无固定句式；图注读数与图内标签一致，标签无压线重叠。故本轮无阻断、无重要问题。
- 处置：可发布（遗留 3 条轻微，建议下一轮顺手修复，均不影响正确性与主线理解）
