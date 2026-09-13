<!-- review-meta
round: 4
page: wiki/prefix-caching/index.html
reviewed_content_sha256: 0825d1db0e3f1ebc
-->
# 前缀缓存审查记录（第 4 轮）

- 页面版本：3bd06a10ecebfe7dfec8f9ec09521fb2f1e6de70（sha1，wiki/prefix-caching/index.html）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与本页前序轮次）
- 已完整阅读章节：核心问题；1. 什么能复用——共享前缀从哪来（含本章问题）；2. radix tree——组织与匹配（含 SVG 图、图注、「展开」折叠块与本章问题）；3. 缓存满了怎么办——LRU 逐出与引用计数（含本章问题）；4. 命中率——页大小是隐藏的自变量（含本章问题）；5. 两种实现——radix tree 与哈希（含本章问题）；来源与范围说明。外部来源：SGLang arXiv:2312.07104v2 HTML（§1、§3 全文）与 Strata arXiv:2508.18572v1 HTML（§2.3/§3.1/§3.2/§5.2.1/§5.3.2/§6 及 setup）。

## 问题

- [重要·技术] meta blockquote（第 62 行）与「来源与范围说明」C6/C7 条（第 315 行）：Strata 被标为 `arXiv:2508.18572v2`，但该文在 arXiv 上只有 v1，v2 不存在，读者按此定位会失败｜引文依据：arXiv 提交历史仅 "[v1] Tue, 26 Aug 2025 00:09:03 UTC"；`curl -L https://arxiv.org/abs/2508.18572v2` → 404、`.../html/2508.18572v2` → 404、`.../abs/2508.18572v1` → 200。页面引用的全部 Strata 论断（§2.3 "widely adopted by providers such as OpenAI … and Google"、§3.1 "cache matching is performed on a per-page basis"、§5.3.2 "achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"、§6 "utilize hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes" / "LMDeploy … constructing coarser-grained tries" / "Strata builds upon SGLang by extending its RadixTree to a HiRadixTree"）实际均在 v1 对应章节中，内容与页面一致｜修复要求：将 meta 与 C6/C7 两处的 `arXiv:2508.18572v2` 改为 `arXiv:2508.18572v1`；OSS-2026 会名（"OSDI 2026"）经作者主页核实属实，可保留｜修复：｜复验：

- [轻微·表述] 第 182 行（第 2 章匹配示例首句）：「下面用一个构造示例走一遍匹配过程。」属元话语（"下面用…走一遍"式过程预告），与规范禁止的「下面来看…」同类｜引文依据：不适用｜修复要求：改为直接进入示例，如「匹配过程如下例：设系统提示词为 $S$（200 token）……」，删去对阅读动作的预告｜修复：｜复验：

- [轻微·表述] 第 126 行（第 1 章多轮对话段）：「多轮对话值得单独说一句：」为元话语式作者框架（"值得单独说"），未承载内容信息｜引文依据：不适用｜修复要求：删去引导语，直接以「第 $n$ 轮请求的输入恰好以第 $n-1$ 轮的完整序列……为前缀」起句｜修复：｜复验：

- [轻微·表述] 第 211 行「但逐出顺序有讲究：」与第 235 行「以及一个容易被忽略的变量——<b>页大小</b>。」：口语化临场评价（"有讲究""容易被忽略"），替代了客观陈述｜引文依据：不适用｜修复要求：改为「但逐出顺序影响复用效率：」「以及页大小这一常被忽略的变量」或直接「以及页大小」｜修复：｜复验：

- [轻微·格式] 图注（第 179 行）：「根到任一节点的路径拼接 = 一个已缓存序列」中关系符 `=` 裸露在正文，而本页其他处（第 119、204 行）同一含义写作 `$=$`，全页写法不一致｜引文依据：不适用（style-guide 第 11 节：数学运算符与关系符必须包在 $...$ 中）｜修复要求：改为「根到任一节点的路径拼接 $=$ 一个已缓存序列」，与第 204 行 $=$ 写法统一｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」下 h3（第 314、317 行）：使用「核心论断与来源」「核心公式与来源」，规范固定命名为「论断与来源（C）」「公式与来源（F）」｜引文依据：不适用（style-guide 第 1 节、第 21 条）｜修复要求：改为规范固定命名（注：同级页面也用「核心…」写法，若属仓库级约定可由规范侧统一，否则本页按规范改）｜修复：｜复验：

- [轻微·格式] 第 256 行与第 271 行：delay hit 机制（"前缀正在计算中时到达的同前缀请求不能复用、引发重复计算"）为来源论断，但正文只写「在 Strata 一文中处理」，无编号引注，来源章节也未收录该项｜引文依据：Strata §3.2 "the delay hit phenomenon … arises when multiple requests for the same data object arrive and queue while an initial cache miss is still being resolved"；"when such requests are grouped into the same batch, redundant prefill computation occurs"｜修复要求：为 delay hit 增加一条编号引注（如 C10，Strata §3.2）并在来源章节登记，或删去正文机制表述｜修复：｜复验：

- [轻微·技术] 第 254 行：「SGLang 把页大小定为 1 token，正是让匹配粒度细到 token 级、命中率不受页对齐损失[N1]」——"正是让……"把设计动机写成来源结论，但所引 SGLang 论文只陈述页大小事实，未给出该动机｜引文依据：SGLang §3.2 "These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token."（仅有事实，无动机）；页大小→命中率的机制本身由 C8（Strata §3.1 "cache matching is performed on a per-page basis"）支持｜修复要求：改为不含动机的中性表述，如「SGLang 把页大小定为 1 token，匹配粒度即到 token 级、无页对齐损失[N1]」，或标注为推断｜修复：｜复验：

- [轻微·技术] 第 65 行：「文献与商业 API 中也叫 context caching / prompt caching[C6]」——所引来源只出现 "context caching"，未见 "prompt caching"，属对来源术语的扩大｜引文依据：Strata §2.3 "systems exploit context caching across requests by identifying common prefixes … widely adopted by providers such as OpenAI … and Google"（无 prompt caching）｜修复要求：删去 "prompt caching"，或另补支持该别名的来源后保留｜修复：｜复验：

- [轻微·可读性] 第 254、264、321 行：「SGLang-HiCache」「Strata-IO」首次出现即用于 2.4%／93% 的结论，页面未说明二者是何系统（HiCache 为 SGLang 上的分层缓存基线、Strata-IO 为 Strata 的消融变体），读者难以判断该对比的含义｜引文依据：Strata §5.1 "we implemented SGLang-HiCache which incorporates a … hierarchical caching implementation"；§5.3 "Strata-IO, which incorporates the GPU-assisted I/O mechanism"（来源本身有定义，页面缺）｜修复要求：在第 4 章首次出现处用一句说明二者身份（如「SGLang-HiCache：在 SGLang 上实现分层缓存的对照基线；Strata-IO：Strata 启用 GPU 辅助 I/O 的变体」）｜修复：｜复验：

## 已核对无问题（供复验参考）

- 核心数字与来源一致：SGLang §3.2 "evicts the least recently used leaf first"、"each node maintains a reference counter … evictable if its reference counter is zero"、"we do not preallocate a fixed-size memory pool … share the same memory pool"、"retains the cache for prompts and generation results in a radix tree"、"size of each page is equivalent to one token"；§1 "the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution"。Strata §5.3.2 "achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate"（条件 Qwen-14B/H200/LooGLE，见 §5.3 "All analyses presented here were conducted using the Qwen-14B model on an H200 platform" 与 §5.1 数据集设定，页面 N2 条件标注无误）；§3.1 "cache matching is performed on a per-page basis"。
- 算术可复算：$230+40+230=500$、$230\times3=690$、$\lfloor100/32\rfloor=3$、$3\times32=96$、$\lfloor100/256\rfloor=0$ 均正确。
- 内部一致：正文算例（页 1/32/256 → 100/96/0）与核心问题、表格三处一致；「页 512 低 2.4%／93%」在核心问题、第 4 章、章节问题与来源 N2 四处一致。
- 链接与被引页：kv-cache、paged-attention、strata、mooncake 的 index.html 均已存在；正文无 research/ 路径引用，无"（待生成）"占位。
- `.dojo/scripts/validate.py wiki/prefix-caching/index.html` → "validation ok"。
- 会话指代：全页无"我/我们/你"；自称统一用"本文"。公式定界符内外的数学符号一致，无 Unicode 数学字符。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 9
- 处置：修复。核心结论、公式、数字与来源一致，无需返回规划；重要项（Strata arXiv 版本标注）须在下一轮前改毕，轻微项（表述元话语/临场评价、格式与命名、术语自足性）一并修复后复验。
