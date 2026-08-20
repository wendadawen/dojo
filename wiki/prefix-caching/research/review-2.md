# 前缀缓存 审查记录（第 2 轮）

- 页面版本：index.html 869d74f3f26020f847b94d198ef8e7fe5ebf65a6 / overview.html befe0af60c0737a5a4e92a3e25f6782438d9b8a4
- 审查时间：2026-08-20 11:25
- 审查者：独立子代理（reviewer-prefix-caching-2）
- 已完整阅读章节：核心问题（5 题及解答折叠块）、1. 什么能复用（含本章问题）、2. radix tree——组织与匹配（含 SVG 图与「三个请求走树的完整过程」折叠块、本章问题）、3. 缓存满了怎么办（含本章问题）、4. 命中率（含本章问题）、5. 两种实现（含本章问题）、来源与范围说明全部五个小节；overview.html 全文。

核对来源：SGLang 论文（arXiv:2312.07104v2 全文 txt，§1、§3/RadixAttention）；Strata TeX 源码 1.5_background.tex（§2.3）、2_motivation.tex（§3.1）、3_design.tex（§4）、5_eval.tex（§5.2、§5.3.2）、6_related.tex（§6）。

## 来源核对结果（C1–C9、N1–N4）

- C1 ✓：§1 "the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution"。
- C2 ✓：§1 "automatic reuse of the KV cache across multiple generation calls"、"an LRU eviction policy and a cache-aware scheduling policy"、"compatible with techniques like continuous batching, paged attention, and tensor parallelism"。
- C3 ✓：§3.2 "space-efficient alternative to a classical trie (prefix tree)...sequences of elements of varying lengths"、"manage a mapping between sequences of tokens, and their corresponding KV cache tensors"、"non-contiguous, paged layout, where the size of each page is equivalent to one token"。
- C4 ✓：§3.2 "evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted"。
- C5 ✓：§3.2 "each node maintains a reference counter indicating how many running requests are using it. A node is evictable if its reference counter is zero"、"the cached tokens and the currently running requests share the same memory pool...evict all cached tokens in favor of a larger batch size"。
- C6 ✓：Strata §2.3 "widely adopted by providers such as OpenAI and Google"。
- C7 ✓：Strata §6 "SGLang employs a RadixTree...vLLM and Mooncake, utilize hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes. LMDeploy adopts a hybrid approach by constructing coarser-grained tries...extending its RadixTree to a HiRadixTree"。
- C8 ✓：Strata §3.1 "Smaller pages...can improve cache hit rate, as cache matching is performed on a per-page basis"。
- C9 ✓：SGLang §3 "retains the cache for prompts and generation results in a radix tree"。
- N1 ✓：SGLang §3.2 "the size of each page is equivalent to one token"。
- N2 ✗：见问题 1。§5.3.2 原文为 "Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"——2.4% 的对照对象是 Strata-IO，非"页 32"。实验条件（SGLang-HiCache、Qwen-14B、H200、LooGLE）与 §5.3 "All analyses presented here were conducted using the Qwen-14B model on an H200 platform" 及 loogle_page 图一致。
- N3 ✓：构造算例复算无误（⌊100/1⌋=100、⌊100/32⌋=3×32=96、⌊100/256⌋=0）；TensorRT-LLM 32 / vLLM 16 与 Strata §2.2 "32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang" 一致，"32 是（CUDA GPU 上 vLLM 的）最大支持值"与 §3.1 "a maximum supported size in vLLM for CUDA GPUs" 一致。
- N4 ✓：§5.2.1 "consistently reaching approximately 95% cache hit rate by leveraging CPU memory"、§5.1 "allocate 1 TB of system DRAM as pinned memory"。

三请求走树算例复算 ✓：R1=S+m1=200+30=230 全量 prefill；R2 命中 S 段 200、只算 m2=40；R3=S'+m3=230 全量。总 230+40+230=500，无缓存 230×3=690，R2 省下 200=共享前缀长度。折叠块已给出 S/m1/m2/S'/m3 全部长度定义，可独立复算。

radix tree SVG ✓：4 个 dg-label（$S$、$m_1$、$m_2$、$S'$）经坐标核算均不与节点框或连线重叠（S 标签 y 132–158 位于连线 y=170 上方且 x 128–288 在两框之间；m1 标签底 y=112 低于连线最低点 y≈123；m2 标签顶 y=234 高于连线最高点 y≈217；S' 标签 x 110–270 避开 x 74–85 的连线与 y≥258 的节点框）；公式均在 foreignObject 中由 KaTeX 渲染，<text> 内无 ASCII 近似写法。

问题块 ✓：核心问题 5 题与五个章节各 2 题本章问题均有解答折叠块，答案独立可读、与正文一致，核心问题答案均指明所在章节。

机械项：$ 配对两页均无误；链接 ../kv-cache/index.html、../paged-attention/index.html、../strata/index.html、../../index.html、overview.html↔index.html 均存在有效。

## 问题

- [重要·技术] index.html 核心问题 4 答案（"实测页 512 比页 32 命中率低 2.4%、最优页配置的吞吐也只剩对照的 93%"）、§4 正文（"页 512 配置的命中率比页 32 低 2.4%，把基线系统调到其最优页大小（512）吞吐也只剩对照系统的 93%"）、§4 本章问题 1 答案（"实测页 512 比页 32 命中率低 2.4%"）、来源说明 N2（"页 512 比页 32 命中率低 2.4%……对照为 Strata-IO"）共四处：2.4% 的对照对象写成"页 32"，与论文不符，且 N2 注自相矛盾（前半句称比页 32 低、括号条件又称对照为 Strata-IO）；meta summary 与 overview.html 已改为"比 Strata-IO 低 2.4%"，正文侧未同步，第 1 轮修复不完整。｜引文依据：Strata §5.3.2 "Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"（5_eval.tex，对照对象为 Strata-IO，页 1）｜修复要求：四处统一改为"页 512 配置的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%，最优页配置（512）吞吐也只为 Strata-IO 的 93%"；删除 N2 注中"比页 32"表述，消除其与"对照为 Strata-IO"的自相矛盾；核心问题 4 答案与 §4 正文中的"对照"一词必须落到 Strata-IO 明确名称｜修复：｜复验：
- [重要·来源] index.html 主要依据 blockquote 与来源说明"核心论断与来源"两处：SGLang 论文标题写作 "SGLang: Efficient and Flexible Server-Serving"，与论文实际标题不符。｜引文依据：sglang-paper.txt 首页标题 "SGLang: Efficient Execution of Structured Language Model Programs"（arXiv:2312.07104v2）｜修复要求：两处标题改为 "SGLang: Efficient Execution of Structured Language Model Programs"（作者、会议、arXiv 号不变）｜修复：｜复验：
- [轻微·来源] index.html §5 末段："把 RadixTree 扩展为 HiRadixTree（记录各页所在内存层级等元数据）"——"记录各页所在内存层级"超出原文，属未标注推断。｜引文依据：Strata §4.1 "a HiRadixTree, which is an extension to SGLang's RadixTree, effectively serving as a page table and stores metadata about each KV cache page"（3_design.tex；未说明元数据包含内存层级）｜修复要求：改为与原文一致的"作为页表存储各 KV cache 页元数据"，或对"所在内存层级"加推断标注｜修复：｜复验：
- [轻微·机械] index.html §4 构造示例段（"……的页大小权衡同源）：</p></p>"）：出现重复的 </p> 闭合标签。｜引文依据：不适用｜修复要求：删除多余的 </p>，改后 HTML 校验无 stray closing tag｜修复：｜复验：
- [轻微·机械] index.html 与 overview.html 中 = 与 + 的书写不一致：数学语境裸写（index 核心问题 1 答案"新一轮输入 = 前几轮全部内容 + 新消息"、§1 表格"第 $n$ 轮请求 = 第 $n-1$ 轮完整序列 + 新消息"、§1 正文"提示 + 全部生成输出"、SVG 图注与 §2 本章问题 2 答案"根到节点路径 = 已缓存序列"、§5 本章问题 1 summary"token ID + 前页指纹链式哈希"；overview"每页指纹 = 哈希(token ID, 前页指纹)"、"根到节点路径 = 已缓存序列"、"上一轮全部内容 + 新消息"），而同页 N1 注与 overview 关键结论处又用 $=$、$\times$。｜引文依据：不适用（check.md §2.2 第 9 条：数学符号全部由 KaTeX 渲染、同一符号全页写法一致）｜修复要求：数学语境中的 = 与 + 统一放入 $...$（如"第 $n$ 轮请求 $=$ 第 $n-1$ 轮完整序列 $+$ 新消息"、"每页指纹 $=$ 哈希(token ID, 前页指纹)"、"根到节点路径 $=$ 已缓存序列"）；纯列举连接（如"LRU + cache-aware 调度"）可保留现状但需全页统一口径｜修复：｜复验：
- [轻微·可读性] index.html §3 首段："continuous batching 下正在运行的请求还在使用某些节点"——术语 continuous batching 首次出现，无解释、无链接（本章问题 2 答案的"请求加入或离开运行批"在折叠块内，收起时不可见）。｜引文依据：不适用｜修复要求：首现处加一句括号说明（如"continuous batching（连续拼批：请求可动态加入或退出运行中的批次）"）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（重要问题 1、2 修复并复验通过前不得发布；轻微问题建议随本轮一并修复）


## Round 2 修复记录

| 编号 | 问题 | 修复 | 复验 |
|---|---|---|---|
| 重要 #1 | 四处 2.4% 对照对象仍写"页 32" | 四处统一"页 512 配置的 SGLang-HiCache 命中率比 Strata-IO 低 2.4%、吞吐为 Strata-IO 的 93%"；N2 注消除自相矛盾 | grep 旧写法清零 |
| 重要 #2 | SGLang 标题引错 | 两处改 "SGLang: Efficient Execution of Structured Language Model Programs" | 验证 |
| 轻微 #3 | HiRadixTree "记录内存层级"超原文 | 改"作为页表存储各 KV cache 页的元数据"（§4.1 原文口径） | 验证 |
| 轻微 #4 | 重复 </p> | 删除 | 验证 |
| 轻微 #5 | = 与 + 数学语境裸写 | 七处统一入 $...$（index 与 overview） | 验证 |
| 轻微 #6 | continuous batching 首现无解释 | 加括注"连续拼批：请求可动态加入或退出运行中的批次" | 验证 |
