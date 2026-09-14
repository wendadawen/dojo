<!-- review-meta
round: 8
page: wiki/kv-cache/index.html
reviewed_content_sha256: b0abe058f571fc7d
-->
# KV cache 审查记录（第 8 轮）

- 页面版本：44dfa0075ea15d413e396fd14dc11620d80b9975（工作树 wiki/kv-cache/index.html）
- 审查时间：2026-09-14 17:03 CST
- 审查者：独立子代理（未参与写作，未参与前序审查）
- 已完整阅读章节：模板信息头与引言 → 核心问题（4 条含折叠答案）→ 1. 注意力为什么需要缓存——K/V 与查询无关 → 2. prefill 与 decode——缓存产生的两个阶段 → 3. 缓存有多大——每 token 字节数公式 → 4. 为什么显存成为瓶颈——动态缓存与权重的争夺 → 来源与范围说明（含全部 details 折叠块、流程图的图注）。overview.html 一并通读。

## 问题

- [轻微·一致性] 第 2 章「沿用手册场景」段（index.html:185）：「decode 生成回答的每个 token 都让缓存多 1 token——回答 200 token，缓存最终 20,300 token」与本章自身的两阶段设定不符。本章三处均写明 prefill 阶段已经产出第 1 个输出 token，因此生成 200 token 的回答只经历 199 个 decode 步，缓存终点应为 20,100 + 199 = 20,299；20,300 相当于把 prefill 已产出的那个 token 又计入一次，读作「输入 20,100 + 回答 200」的上下文总长口径而非缓存口径。｜引文依据：本章正文「prefill（预填充）阶段……算出它们在每一层的 K/V 并写入缓存，同时产出第一个输出 token」（index.html:158）；流程图节点「并行算全部 20,100 token 的 K/V，写入缓存，产出第 1 个输出 token」（:169）；本章问题解答「prefill 一次性并行处理全部输入 token……并产出第一个输出 token」（:195）。｜修复要求：把「回答 200 token，缓存最终 20,300 token」改为「回答 200 token，缓存最终 20,299 token」，或改写为明确的口径说明（输入 20,100 + 回答 200 = 上下文 20,300 token，缓存含 prefill 已产出的首个输出 token 之前的所有 K/V）。｜修复：｜复验：
- [轻微·格式] 「来源与范围说明 → 论断与来源（C）」（index.html:298）：条目列举顺序为 C3、C2、C4、C5，非升序；且全页正文与来源章节均无 C1，C 编号自 C2 起。同站概念页（prefix-caching、paged-attention 等）的来源章节均为自 C1 起的升序。｜引文依据：不适用｜修复要求：把 C 条目按 C2、C3、C4、C5 升序排列；若确无 C1 内容，需说明编号起点或重排编号，使正文 `[Cx]` 与来源章节一一对应且无缺号。｜修复：｜复验：
- [轻微·一致性] 「来源与范围说明 → 构造示例」（index.html:307）：「「今天/天气/很/好」4 token 序列（第 1、2 章）」标注该示例出现在第 1、2 章，实际它只出现在第 1 章（表格 :121–127、折叠块 :135、本章问题解答 :151）；第 2 章全篇未引用该序列，使用的是「手册 20,000 token + 问题 100 token」场景。｜引文依据：第 2 章正文（index.html:156–205）无「今天/天气/很/好」序列，只有 20,100/20,300 token 的手册场景；第 1 章 :117、:123–126、:135、:151 四次出现该序列。｜修复要求：把「（第 1、2 章）」改为「（第 1 章）」，或在第 2 章补出对该示例的引用。｜修复：｜复验：

## 回源核对摘要（本轮判定为无问题、留作引文依据）

- Llama-3.1-8B 配置（32 层 / num_key_value_heads=8 / num_attention_heads=32 / head_dim=128 / max_position_embeddings=131072 / torch_dtype=bfloat16）：HF config.json，meta-llama 仓库受限，经 NousResearch、unsloth 两个同源镜像取得一致取值。
- vLLM §3「the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)」「the memory required to store the KV cache of one request can be as much as 1.6 GB」（OPT-13B、2048 token）。
- vLLM §1 / Figure 1「memory distribution for a 13B-parameter LLM on an NVIDIA A100 GPU with 40GB RAM. Approximately 65% of the memory is allocated for the model weights…… Close to 30% of the memory is used to store the dynamic states of the requests」。
- vLLM §4.3「Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests.」
- Strata §1「40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B, which can be quickly consumed by a handful of documents or hundreds of conversation turns」；§2.1「LLM inference operates in two phases: prefill and decode……」；Table 1 LooGLE「avg. in 21613」（支撑「20,000 token 与长上下文评测平均规模同量级」）。
- SGLang §1「In existing inference engines, the KV cache of a request is discarded after processing is completed」；§3「Unlike existing systems that discard the KV cache after a generation request finishes, our system retains the cache……」。
- 会议归属：vLLM=SOSP '23（arXiv 页 Conference 字段）、Strata=OSDI '26（USENIX OSDI '26 Technical Sessions，「KV Cache and Long Context」track 收录该文）、SGLang=NeurIPS 2024（NeurIPS 2024 proceedings，pp. 62557–62583）。
- HiSparse arXiv:2608.07009v1：摘要「keeps each request's full KV history in host memory and bounds its decode footprint with a small, fixed-size GPU cache」，图 2「the GPU keeps only compact state (indexer state, per-request-per-layer GPU caches with page-table and LRU metadata)」——支撑「全量历史留主机内存、显存里每请求每层只保留固定大小的缓存、解码显存不随上下文长度增长」。
- 数值复算：2×32×8×128×2=131,072 B=128 KB；20,000×131,072 B≈2.44 GB；2×5120×40×2=819,200 B=800 KB；128K×131,072 B=2^34 B=16 GB；40×2^30/131,072≈0.33M；20100×128 KB≈2.45 GB；20,000²/2≈2 亿组。全部与页面一致，且 summary / description / overview / 图注 / 表格之间的同项数字一致。
- 机制项：页面无「声称可运行的代码」，无代码块；流程图与 KaTeX 均可用，交互内容不依赖脚本即可阅读正文；无 Unicode 数学字符替代 LaTeX（仅正文分隔用的破折号与返回顶部按钮的 ↑）；alt 属性为空、不含 `$...$`；本地 libs 与 6 个前置/后续概念链接（standard-attention、prefix-caching、hetero-pd、paged-attention、strata、hisparse）均存在；`python3 .dojo/scripts/validate.py wiki/kv-cache/index.html` 返回 `validation ok`。
- 表述项：通读全文（含折叠块与图注）未发现元话语固定句式、第一/第二人称、调试叙事、临场评价或 AI 拼接腔；「本文」自称符合 style-guide §12；「构造示例。」为全站通用标记（81 处），非本页问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（3 条轻微项，均为措辞/编号一致性，不影响本页核心结论与来源一致性）