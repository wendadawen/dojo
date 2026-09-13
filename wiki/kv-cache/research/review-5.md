<!-- review-meta
round: 5
page: wiki/kv-cache/index.html
reviewed_content_sha256: 94e03fe5a77dcb2e
-->
# KV cache 审查记录（第 5 轮）

- 页面版本：c98cea3a15ef8ddf5e75520fc6ce8b79ee85d072
- 审查时间：2026-09-13 20:16
- 审查者：独立子代理（第 5 轮，未参与写作，未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题（4 条解答折叠块）→ 1. 注意力为什么需要缓存——K/V 与查询无关（含「展开：4 token 例子的逐步 K/V 计算」折叠块、本章问题）→ 2. prefill 与 decode——缓存产生的两个阶段（含流程 figure、本章问题）→ 3. 缓存有多大——每 token 字节数公式（含「展开：128 KB/token 的完整代入与两个检查」折叠块、本章问题）→ 4. 为什么显存成为瓶颈——动态缓存与权重的争夺（含四行对照表、本章问题）→ 来源与范围说明（六个 h3 全部）。

复核的外部来源与结果：vLLM（arXiv:2309.06180）§3 原文「For the 13B parameter OPT model, the KV cache of a single token demands 800 KB of space, calculated as 2 × 5120 × 40 × 2 (bytes per FP16). Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as 1.6 GB.」；§1 原文「Approximately 65% of the memory is allocated for the model weights... Close to 30% of the memory is used to store the dynamic states of the requests.」；§4.3 原文「Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests.」。Strata（arXiv:2508.18572）§1 原文「40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B」；§2.1 原文「During prefill, the model typically processes both (i) new tokens from the user query and (ii) context tokens... In the subsequent decode phase, the model generates tokens autoregressively, continually reusing and extending the KV cache.」。SGLang（arXiv:2312.07104）§3 原文「In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls.」。Llama-3.1-8B config.json：num_hidden_layers=32、num_key_value_heads=8、num_attention_heads=32、head_dim=128、max_position_embeddings=131072、dtype=bfloat16。页面全部算式复算通过：2×32×8×128×2=131072 B=128 KB；20000×131072 B=2.44 GiB；131072×131072 B=16 GiB；2×5120×40×2=800 KB；2048×800 KB=1.56 GB；40×2³⁰/131072≈0.33M；1+2+3+4=10；n(n+1)/2；20000×20001/2≈2 亿。`.dojo/scripts/validate.py` 返回 validation ok；全部内链（standard-attention、prefix-caching、hetero-pd、paged-attention、strata、hisparse）目标页均存在；全页无 Unicode 数学字符（`↑` 为返回顶部按钮）。

## 问题

- [阻断·技术] 第 2 章「多轮对话」段：把「下一轮请求前 20,300 token 的缓存已经存在」写成无条件事实，与第 4 章「既有系统在请求完成即丢弃缓存，这让后续请求无法复用」直接冲突，读者无法判断跨请求前缀复用到底成立与否。｜引文依据：本页「多轮对话时，下一轮请求的输入等于「上一轮全部输入 + 上一轮输出 + 新问题」，其前 20,300 token 的缓存恰好已经存在，这正是前缀复用的基础」；本页第 4 章「既有系统在请求完成即丢弃缓存[C5]，这让后续请求无法复用」；SGLang §3「In existing inference engines, the KV cache of a request is discarded after processing is completed, preventing the KV cache from being reused across multiple calls.」｜修复要求：给该句补成立条件（仅当系统保留缓存/开启前缀缓存时成立），改写为「上一轮这 20,300 个 token 的 K/V 与已完成请求完全相同，只要系统保留这份缓存即可复用」，使其与第 4 章一致｜修复：｜复验：
- [重要·技术] 第 3 章符号表 $H_{\text{kv}}$ 条目：写成「每个注意力头的 $q$ 都要与各 KV 头的 $k,v$ 交互」，把查询头描述成与全部 KV 头交互，与同页 GQA 定义（每组共享一个 KV 头）冲突，也让读者看不出为什么公式里是 KV 头数而非查询头数。｜引文依据：本页「$H_{\text{kv}}$：KV 头（组）数——每个注意力头的 $q$ 都要与各 KV 头的 $k,v$ 交互」与同页「GQA（分组查询注意力）中查询头分成 $H_{\text{kv}}$ 组、每组共享一个 KV 头」｜修复要求：改为「每个查询头的 $q$ 只与该头所属组的那个 KV 头的 $k,v$ 交互（MHA 下每个查询头对应一个 KV 头）」，明确决定 $k,v$ 套数的是 KV 头数｜修复：｜复验：
- [轻微·技术] 第 4 章「量级判断」段：「几十份文档」与本页自身换算不符。｜引文依据：本页「一份手册就 0.02M」与「大约只够缓存 0.3M token」，0.3M÷0.02M=15｜修复要求：改为「十几份文档」，或在给出文档规模的同时给出台数｜修复：｜复验：
- [轻微·技术] 第 3 章换算口径说明：把 $1\,\text{KB}=1{,}024$ 字节称作「存储行业惯例」，该二进制口径通常归属内存/IEC 惯例，存储设备容量行业惯例为 $1\,\text{KB}=1{,}000$ 字节，归属写反。｜引文依据：不适用（术语准确性）｜修复要求：删去行业归属，直接写「本页按 $1\,\text{KB}=1{,}024$ 字节、$1\,\text{GB}=2^{30}$ 字节换算」，或改为「内存计量惯例」｜修复：｜复验：
- [轻微·技术] 第 4 章段末 HiSparse 句：机制描述（全量历史留主机内存、显存每请求每层只保留固定大小缓存、按当前选择拉取缺失条目）属无来源机制论断，「来源与范围说明」未登记 HiSparse 的任何来源。｜引文依据：不适用（本页 C/F/N 三节均未列 HiSparse）｜修复要求：为该句补来源标注（指向 hisparse 页所引原始来源），或降级为「见 HiSparse 一文」的纯导航句、不陈述机制｜修复：｜复验：
- [轻微·技术] 第 3 章 GQA 段末句：「多数现代开源模型（Llama 3、Qwen2.5 等）采用 GQA」是无来源的概括性事实，被写成结论。｜引文依据：不适用｜修复要求：补可定位来源（如各模型 config.json 的 num_key_value_heads），或收窄为已在本页给出算例的 Llama-3.1-8B 采用 GQA｜修复：｜复验：
- [轻微·表述] 第 3 章「两个算例对照」段：连续出现「两个算例对照。」「构造示例。」「文献算例。」三个短标签句，呈拼接式标签堆叠；「文献算例」也不是 style-guide 第 4 节给出的标记词。｜引文依据：不适用｜修复要求：并成一句连贯陈述，如「两个算例对照：Llama-3.1-8B 是按本页公式的构造算例…；OPT-13B 是 vLLM 论文给出的原始算例…」｜修复：｜复验：
- [轻微·表述] 第 4 章「三处常见误解需要在此澄清」：「需要在此澄清」属元话语式提点，与「需要注意的是」同类。｜引文依据：不适用｜修复要求：直接进入内容，改为「KV cache 有三处常被误读的地方。」或直接起「其一，…」｜修复：｜复验：
- [轻微·格式] 第 1 章 $k_i,v_i$ 独立性段末：`<span class="callout-inline">…</span>` 使用了未定义的类——libs/dojo-concept.css 只定义 `.callout` 与 `.callout-blue/red/yellow/purple`，无 `.callout-inline`，该段预期的强调效果不生效。｜引文依据：不适用｜修复要求：改用 style-guide 第 3 节定义的 callout 块（`.callout .callout-blue` 或 `.callout-yellow`），或去掉该类名｜修复：｜复验：
- [轻微·可读性] 引言段「这四个问题依次是」：前文四问为「哪些中间结果／从头再算还是留下／要留多少／够不够放」，随后列出的却是「为什么需要全部历史 K/V…／prefill 与 decode…／有多大／为什么成为瓶颈」，两者不对应——「prefill 与 decode」在前四问中没有对应项。｜引文依据：不适用｜修复要求：改为「本文按顺序回答：…」并与核心问题四条对齐，或在引言直接列出将要回答的四个问题、不再回指前文提问｜修复：｜复验：
- [轻微·格式] 「来源与范围说明」构造示例节：「20,000 token 手册（第 2–4 章）」与实际分布不符——第 1 章正文（「对 20,000 token 的手册，这就是 2 亿组与 2 万组的差别」）已使用该算例。｜引文依据：不适用｜修复要求：改为「第 1–4 章」，或改写为不列举章号的说明｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 9
- 处置：修复（阻断与重要项须关闭后复验；轻微项按上面各条改法直接修正）
