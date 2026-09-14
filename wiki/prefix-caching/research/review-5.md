<!-- review-meta
round: 5
page: wiki/prefix-caching/index.html
reviewed_content_sha256: e16b39c5a9ac238f
-->
# 前缀缓存审查记录（第 5 轮）

- 页面版本：461e049ff0ece5d25568e2c02cdfc58912aa4755
- 审查时间：2026-09-13 20:22
- 审查者：独立子代理（编排者派发；未参与写作，也未参与前序轮次审查；仅读本页 index.html、overview.html、页面外部来源与 guides/concept/check.md、style-guide.md）
- 已完整阅读章节：核心问题（5 条，含折叠答案）、1. 什么能复用——共享前缀从哪来、2. radix tree——组织与匹配、3. 缓存满了怎么办——LRU 逐出与引用计数、4. 命中率——页大小是隐藏的自变量、5. 两种实现——radix tree 与哈希、来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件／构造示例／辅助解释与类比边界／简化条件及其限制）

## 已核对来源（引文依据）

- C1：SGLang（arXiv:2312.07104v2）§1「the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution」。
- C2：SGLang §1「RadixAttention is compatible with techniques like continuous batching, paged attention, and tensor parallelism」；§3.2 cache-aware scheduling「prioritize requests with longer matched prefixes instead of using a first-come-first-served schedule」。
- C3／N1：SGLang §3.2「we utilize a radix tree to manage a mapping between sequences of tokens, and their corresponding KV cache tensors. These KV cache tensors are stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token」；「a radix tree ... space-efficient alternative to a classical trie」。
- C4：SGLang §3.2「evicts the least recently used leaf first. By evicting leaves first, we enable the re-use of their common ancestors until those ancestors become leaves and are also evicted」。
- C5：SGLang §3.2「A node is evictable if its reference counter is zero」。
- C9：SGLang「our system retains the cache for prompts and generation results in a radix tree」。
- C6：Strata（arXiv:2508.18572v1）§2.3「widely adopted by providers such as OpenAI ... and Google」。
- C7：Strata §6（SGLang「employs a RadixTree」；vLLM／Mooncake「hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes」；LMDeploy「coarser-grained tries」；Mooncake「disaggregated KV Cache」「large-scale disaggregated KV cache memory pools」）。
- C8：Strata §3.1「cache matching is performed on a per-page basis」。
- C10：Strata §3.2（delay hit：同前缀请求在首个 miss 解决前排队）＋§4.3.1「redundant prefill computation occurs when multiple requests sharing the same cache miss are scheduled into the same batch」。
- N2：Strata §5.3.2「Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate」；Strata-IO＝§4.2 的 GPU-assisted I/O 机制（§5.3.1「Strata-IO, which incorporates the GPU-assisted I/O mechanism from §4.2」）。
- N4：Strata §5.2.1（LooGLE、约 95% 命中率、1TB pinned DRAM）。
- F1／N3／R1-R3 复算：$\lfloor 100/1\rfloor=100$、$\lfloor 100/32\rfloor=3\to96$、$\lfloor 100/256\rfloor=0$ 与表中一致；三请求 prefill 合计 $230+40+230=500$、对照无缓存 $230\times3=690$、R2 省 200＝共享前缀长度，均自洽；符号 $S$、$m_1$、$m_2$、$S'$ 全页单义。
- 页面功能与链接：kv-cache、paged-attention、strata、mooncake 四页均存在，无「（待生成）」占位；`.dojo/scripts/validate.py` 返回 `validation ok`。图为内联 SVG（dg-box/dg-line），公式在 `<foreignObject>` 中，`<text>` 仅纯文字。

## 问题

- [轻微·格式] 来源与范围说明，第三个 h3：标题写作「外部数字与实验条件」，未按 `guides/concept/style-guide.md` §1 固定的来源小节命名「外部数字与实验条件（N）」（本库 65 页采用带「（N）」写法）。｜引文依据：不适用｜修复要求：改为「外部数字与实验条件（N）」｜修复：｜复验：
- [轻微·技术] 4. 章 100 token 命中表的「页大小 32」行括注「32（TensorRT-LLM 默认；vLLM 默认 16，32 是最大支持值）」：该括注是外部系统的真实默认值/上限（非构造值），但本页未附任何 C/F/N 来源编号，且其嵌在声明为「构造示例」的表格中，事实与构造混排无标引。｜引文依据：Strata §2.2「Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang」；§3.1 关于 vLLM 在 CUDA GPU 上 32 为最大支持值。｜修复要求：为该括注补上来源编号并在对应小节列出（如引 Strata §2.2／§3.1），或删除该括注只保留构造性的「32」｜修复：｜复验：
- [轻微·技术] 来源与范围说明 C7 条目：条目文字含「Strata 扩展 HiRadixTree」，但该条目统一标注的章节为「§2.3、§6、§3.2」，均不含 HiRadixTree 的出处（实际在 §4.1），读者按标注定位不到该事实。｜引文依据：Strata §4.1「Strata builds upon SGLang by extending its RadixTree to a HiRadixTree ... effectively serving as a page table and stores metadata about each KV cache page」。｜修复要求：将 §4.1 补入 C7 的章节列表，或把正文该句改标到 §4.1｜修复：｜复验：
- [轻微·表述] 4. 章第二段「这不是理论洁癖：Strata 论文的实验里……」：「理论洁癖」为口语化临场评价。｜引文依据：不适用｜修复要求：改为中性陈述（如「该效应在实测中已出现：」）｜修复：｜复验：
- [轻微·表述] 4. 章标题副标题「4. 命中率——页大小是隐藏的自变量」与同章正文「页大小这一常被忽略的变量」：「隐藏」「常被忽略」是对领域关注度的评价性判断，无来源支持。｜引文依据：不适用｜修复要求：改为可核对的中性表述（如「页大小这一独立变量」），标题副标题同步修改｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布。无阻断与重要问题；核心论断（精确前缀匹配、radix tree 匹配到分叉点、LRU 叶子优先逐出与引用计数、页大小→命中率、哈希指纹实现）逐条回源核对一致，算式与结论相符，术语首次使用即解释，学习目标与两级问题均有独立可读答案。遗留 5 项轻微问题（1 项来源小节命名、2 项来源标引、2 项表述）不影响正确性与主线，可按上表择期修复。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
