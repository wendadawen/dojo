<!-- review-meta
round: 6
page: wiki/prefix-caching/index.html
reviewed_content_sha256: e4a81baa02a5c9b9
-->
# 前缀缓存审查记录（第 6 轮）

- 页面版本：index.html 工作树哈希 git blob 2d07db5e9f9d7f2708ebb482a55302b4611a7678（sha1 7d1dbd21b4cee78f9a02faceae076815098830c4）
- 审查时间：2026-09-14 17:10
- 审查者：独立子代理（未参与写作，未读取 research/ 下任何文件；仅依据本页 index.html、overview.html、外部来源与 guides/concept/check.md、style-guide.md）
- 已完整阅读章节：引言、核心问题（5 条，含解答折叠块）、1. 什么能复用——共享前缀从哪来（含本章问题 2 条）、2. radix tree——组织与匹配（含图 1、折叠块「展开：三个请求走树的完整过程」、本章问题 2 条）、3. 缓存满了怎么办——LRU 逐出与引用计数（含本章问题 2 条）、4. 命中率——页大小这一独立变量（含本章问题 2 条）、5. 两种实现——radix tree 与哈希（含本章问题 2 条）、来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）

## 来源核对依据（本轮实际打开的原文）

- SGLang：arXiv:2312.07104v2 HTML（全文抓取）、arXiv:2312.07104v1 HTML（全文抓取）、NeurIPS 2024 官方 camera-ready PDF（proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf，pdftotext 抽取）。
- Strata：arXiv:2508.18572v1 HTML（全文抓取）、arXiv abs 页（Comments 字段）。

## 问题

- [阻断·技术] 主要依据（index.html L62）、来源与范围说明 C 组（L315）、N1 条目（L321）：SGLang 论文被引的「§3.2」不存在，SGLang 第 3 章没有子节。主要依据把 RadixAttention 定位在「§3.2」，C 组把 C1/C2/C3/C4/C5/C9 全部定位在「§1、§3.2」，N1（页大小 = 1 token）定位在「§3.2」。由于 C2（RadixAttention 自动复用 + LRU + cache-aware 调度 + 兼容性）、C3（radix tree 压缩边 / token-张量映射 / 页 1 token / 非连续分页）、C4（LRU 叶子优先、祖先复用）、C5（引用计数与共享内存池）、C9（生成结果入缓存）除 §1 外唯一的定位点就是 §3.2，而 §1 不含这些内容，这些论断在页面上标注的位置实际都定位不到。｜引文依据：arXiv:2312.07104v2 的 TOC 只有 `1 Introduction / 2 Programming Model / 3 Efficient KV Cache Reuse with RadixAttention / 4 ... / 5 ... / 6 Evaluation（6.1、6.2、6.3）/ 7 / 8` 与附录 `A.1–A.4、B.1–B.3、D.1–D.2`，第 3 章无 3.1/3.2 子节；对同一 PDF 用 pdftotext 抽取后 `grep -E '^\s*(3|3\.1|3\.2|4)\s+[A-Z]'` 只命中 `3   Efficient KV Cache Reuse with RadixAttention` 与 `4   Efficient Constrained Decoding ...`；v1（arXiv:2312.07104v1）子节列表为 `2.1–2.4、4.1–4.3、5.1–5.3、6.1–6.4`，同样无 3.2。页面所引内容本身可在第 3 章原文定位：`"we utilize a radix tree to manage a mapping between sequences of tokens, and their corresponding KV cache tensors"`、`"stored in a non-contiguous, paged layout, where the size of each page is equivalent to one token"`、`"we introduce a simple LRU eviction policy that evicts the least recently used leaf first"`、`"each node maintains a reference counter indicating how many running requests are using it"`、`"we let the cached tokens and the currently running requests share the same memory pool"`、`"our system retains the cache for prompts and generation results in a radix tree"`（以上均属第 3 章）。｜修复要求：把 L62「§3.2（RadixAttention）」、L315「§1、§3.2」、L321「SGLang 论文 §3.2」中的 SGLang「§3.2」一律改为「§3」；同一处 L315 中 Strata 的「§3.2」（delay hit）经核对存在于 Strata 第 3.2 节，保持不变。修改后逐条回源确认新定位点确实含该原文片段。｜修复：｜复验：

- [轻微·技术] 1. 什么能复用（L112–L124 正文表，及 L74 核心问题「解答：精确 token 前缀匹配，四类典型来源」）：把「系统提示词、多轮对话历史、few-shot 示例、RAG 文档」写成确定的「四类典型来源」，全页（正文、核心问题答案、overview.html）均未标注来源，来源与范围说明的 C/N/F 条目也未收录该分类。｜引文依据：SGLang arXiv:2312.07104v2 第 1 章只列 prompting 技术 `"like few-shot learning [5], self-consistency [53], skeleton-of-thought [33], and tree-of-thought [56]"`，图 2 的描述为 `"two chat sessions, a batch of few-shot learning inquiries, and a self-consistency sampling"`，评测任务为 `"agent control, logical reasoning, few-shot learning benchmarks, JSON decoding, retrieval-augmented generation pipelines, and multi-turn chat"`；Strata §2.3 仅说 `"these prefixes and sources are frequently reused across applications"`。两文都没有「四类来源」这一划分（论文列举里含 self-consistency，页面分类里没有；页面分类里含系统提示词 / RAG，论文未按此并列）。｜修复要求：或为该四类来源补一条来源标注（SGLang 第 1 章「共享前缀的多种复用场景」），或在「来源与范围说明」的「构造示例 / 辅助解释」中明确写为对本页典型负载的归纳而非论文给出的划分；核心问题答案与 overview 的对应表述同步处理。｜修复：｜复验：

## 本轮核对通过、未构成问题的项（留档说明）

- C1 丢掉缓存的引文在 SGLang §1/§3 均可定位：`"the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls and significantly slowing down the execution"`，与 L65 中文引号内表述一致。
- C6（context/prefix caching 被 OpenAI、Google 等采用）= Strata §2.3 `"widely adopted by providers such as OpenAI ... and Google"`，L65「文献与商业 API 中也叫 context caching[C6]」成立。
- C7（vLLM+Mooncake 哈希 / LMDeploy 粗粒度 trie / Strata 扩展 HiRadixTree）= Strata §6 `"vLLM ... and Mooncake ... utilize hashing mechanisms that generate unique page identifiers based on token IDs and prefix page hashes. LMDeploy ... adopts a hybrid approach by constructing coarser-grained tries. Strata builds upon SGLang by extending its RadixTree to a HiRadixTree."`；HiRadixTree「作为页表存储各 KV cache 页的元数据」= §4.1 `"effectively serving as a page table and stores metadata about each KV cache page"`。
- C8（按页匹配）= Strata §3.1 `"as cache matching is performed on a per-page basis"`。
- C10（delay hit）= Strata §3.2 `"the delay hit phenomenon ... arises when multiple requests for the same data object arrive and queue while an initial cache miss is still being resolved"`；L256/本章问题答案的「同前缀请求不能直接复用、引发重复计算」与该节 `"multiple requests may target the same (or a prefix of the same) context ... redundant prefill computation occurs"` 一致。
- N1 = SGLang §3 `"the size of each page is equivalent to one token"`。
- N2 = Strata §5.3.2 `"Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate."`；2.4% / 93% / 页 512 三处（正文 L254、核心问题答案 L95、来源说明 N2 与 dojo:summary、overview.html）数值与口径一致。条件中模型在 §5.3 记作 `"Qwen-14B"`（= §5.1 的 Qwen2.5-14B-Instruct-1M），H200 平台一致，包 2.4% 的对照组为 Strata-IO 一致；数据集 §5.3.2 未点名，但 §5.3.3 说明 §5.3 的原始（shuffle）workload 由 LooGLE 生成，与 N2 标注的 LooGLE 不矛盾，不另计问题。
- N4（约 95% 命中率）= Strata §5.2.1 `"systems equipped with hierarchical caching ... consistently reaching approximately 95% cache hit rate by leveraging CPU memory"`（LooGLE）；该条已在页面标注「本文未正文引用，仅口径备查」。N5 = Strata §2.2 `"Typical page sizes are small—e.g., 32, 16, and 1 tokens in TensorRT-LLM, vLLM, and SGLang"` + §3.1 `"a maximum supported size in vLLM for CUDA GPUs"`，与表格注「TensorRT-LLM 默认；vLLM 默认 16，32 是最大支持值」一致。
- 数值复算：折叠块 R1/R2/R3 例（230+40+230=500，无缓存 230×3=690，R2 省 200）与命中表（⌊100/1⌋=100、⌊100/32⌋=3→96、⌊100/256⌋=0）逐项复算正确，且与正文/核心问题答案/overview 一致。
- 页内交叉引用有效且被引页真实存在：../kv-cache/index.html（其第 1 章「注意力为什么需要缓存——K/V 与查询无关」、第 4 章「为什么显存成为瓶颈」与页面所指一致）、../paged-attention/index.html（其第 3 章「页间不连续」、第 4 章「页大小怎么选」与页面所指一致）、../strata/index.html、../mooncake/index.html 均存在；无「（待生成）」占位。
- 格式：h2/h3 编号与固定命名符合 style-guide；dojo:summary 公式由 KaTeX 可渲染（\\left\\lfloor、\\text、\\right\\rfloor）；全页无脚本外的 Unicode 数学字符（Python 扫描 α-ω/×/−/≈ 等集合，0 命中）；SVG 图无等宽框线、<text> 内无 ASCII 近似、公式在 <foreignObject><div class="dg-label"> 内；图内标签与线框按坐标核对无重叠；aria-label 与 img alt 中无 `$...$`。
- 表述：全页无第一人称复数与第二人称；「本文」自称符合 style-guide 第 12 节；无「下面来看」「需要注意的是」式元话语、无调试叙事与临场评价；「注意」仅出现在「注意力计算」一词内（非元话语）。
- `.dojo/scripts/validate.py wiki/prefix-caching/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 1
- 处置：修复（改 3 处 SGLang 章节号 §3.2→§3 并回源复验；四类来源补来源标注或改标为归纳；修复后重跑 validate.py。overview.html 与 dojo:summary 不含 §3.2，无需同步改，但四类来源表述需与正文一致处理）
