<!-- review-meta
round: 4
page: wiki/kv-cache/index.html
reviewed_content_sha256: 9aabeef3a59eff14
-->
# KV cache 审查记录（第 4 轮）

- 页面版本：505ee3745c2fc26077e8a689f8a01336c5116356
- 审查时间：2026-09-13 19:42
- 审查者：独立子代理（未参与写作，未参与前序轮次，未读取 research/ 下任何文件）
- 已完整阅读章节：核心问题（4 条，含全部解答折叠块）→ 1. 注意力为什么需要缓存——K/V 与查询无关 → 2. prefill 与 decode——缓存产生的两个阶段 → 3. 缓存有多大——每 token 字节数公式 → 4. 为什么显存成为瓶颈——动态缓存与权重的争夺 → 来源与范围说明（六个小节）；含全部折叠块、图 1 图注与表格

## 来源核对（本轮读到并用于判断的片段依据）

- vLLM §1（2309.06180）：「the key and value tensors associated with the attention mechanism, commonly referred to as _KV cache_」；「This sequential generation process makes the workload _memory-bound_, underutilizing the computation power of GPUs」→ 支持 C1/C3 与「decode 访存密集」。
- vLLM §1 Fig.1：「Approximately 65% of the memory is allocated for the model weights… Close to 30% of the memory is used to store the dynamic states of the requests.」→ 支持 N3（65%/近 30%）。
- vLLM §3：「the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)」；「Since OPT can generate sequences up to 2048 tokens, the memory required… can be as much as 1.6 GB」→ 支持 N1 与 F1。
- Strata §1（2508.18572）：「40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B」→ 支持 N4。
- Strata §2.1：「During prefill, the model typically processes both (i) new tokens from the user query and (ii) context tokens… In the subsequent decode phase, the model generates tokens autoregressively, continually reusing and extending the KV cache.」→ 支持 C2。
- SGLang §3（2312.07104）：「Unlike existing systems that discard the KV cache after a generation request finishes, our system retains the cache for prompts and generation results in a radix tree.」→ 支持 C5 的「完成后丢弃缓存」。
- Llama-3.1-8B config.json：num_hidden_layers=32、num_attention_heads=32、num_key_value_heads=8、hidden_size=4096、max_position_embeddings=131072、torch_dtype=bfloat16 → 支持 N2（说明：config.json 无 head_dim 字段，128 由 hidden_size/num_attention_heads 得出，数值无误）。
- Strata 会议归属：作者主页 Zhiqiang Xie, Publications 条目作「_OSDI_ 2026」→ 支持「OSDI 2026」，Venue 论断成立。
- 复算全部通过：2×32×8×128×2=131072 B=128 KB；20,000×128 KB≈2.44 GB；128K×128 KB=2^34 B=16 GB；2×5120×40×2=800 KB；2048×800 KB≈1.6 GB；40×2^30/131072≈0.33M；4 token 例 1+2+3+4=10 组 vs 4 组；20,000 步即 n(n+1)/2≈2 亿组。公式、符号表与全文数字互相自洽。
- 机械项：正文无 `<pre>/<code>`（check.md §2.2-3 代码项不适用）；图仅 1 个且为 HTML 结构图（无等宽框线、无 SVG `<text>`）；链接 ../standard-attention、../prefix-caching、../paged-attention、../strata、../hisparse 均真实存在；无 research/ 路径、无「（待生成）」；`validate.py wiki/kv-cache/index.html` 返回 validation ok；overview.html 与 index.html 互链。

## 问题

- [重要·技术] 来源与范围说明／C5：C5 标注的 vLLM 位置定位不到该论断｜引文依据：arXiv:2309.06180（ar5iv 渲染，§4 小节为 4.1 PagedAttention、4.2 KV Cache Manager、4.3 Decoding with PagedAttention and vLLM、4.4–4.6）；「Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests.」出现在 §4.3，§4.2「KV Cache Manager」不含该句｜修复要求：把来源章 C5 括号内的「vLLM §4.2」改为「vLLM §4.3」｜修复：｜复验：
- [轻微·技术] head／description：description 标注来源「SGLang NeurIPS 2024 §3.2」，该章节不存在，且与 blockquote 的范围不一致｜引文依据：arXiv:2312.07104 目录为 1 Introduction／2 Programming Model／3 Efficient KV Cache Reuse with RadixAttention／4 …，§3 无任何子节，无「3.2」；同一页面 blockquote 作「Zheng et al., …（NeurIPS 2024, arXiv:2312.07104）§1、§3」，description 又作「vLLM SOSP 2023 §1/§2/§3」而 blockquote 作「§1、§3」｜修复要求：description 中 SGLang 与 vLLM 的章节号改为与 blockquote 及正文实际引用一致的「§1、§3」（SGLang）与「§1、§3」（vLLM）｜修复：｜复验：
- [轻微·格式] 来源与范围说明／小节标题：h3 未使用规范固定命名｜引文依据：不适用（style-guide.md §1 规定来源章 h3 固定为「论断与来源（C）」「公式与来源（F）」「外部数字与实验条件（N）」「构造示例」「辅助解释与类比边界」「简化条件及其限制」）｜修复要求：把「核心论断与来源」改为「论断与来源（C）」、把「核心公式与来源」改为「公式与来源（F）」｜修复：｜复验：
- [轻微·格式] 来源与范围说明／C1：C1 只有定义、正文无引用，双向对应不成立｜引文依据：不适用（正文上标为 <sup>[C3]</sup>×2、<sup>[C2]</sup>、<sup>[F1]</sup>、<sup>[N1]</sup>×3、<sup>[N2]</sup>×2、<sup>[N3]</sup>、<sup>[N4]</sup>×2、<sup>[N5]</sup>×2、<sup>[C4, N3]</sup>、<sup>[C5]</sup>，无 <sup>[C1]</sup>；style-guide.md §6 要求正文与来源章双向对应）｜修复要求：在正文「KV cache 定义」处补 <sup>[C1]</sup>，或把来源章「C1/C3」改为仅「C3」｜修复：｜复验：
- [轻微·表述] 引言与来源章：以「本文/本页」为主语的自我指代，且两种自称混用｜引文依据：不适用（位置：引言「本文回答四个问题：…」「本文讨论的是推理阶段的 KV cache」「本文直接使用其结论，不重复推导」；§2「本文不展开部署形态」；§3「本页 KB/GB 按存储行业惯例换算」；来源章「本文未使用类比解释机制」「本文未覆盖」「不在本文范围」，以及「与本页 F1 复算一致」）｜修复要求：删去「本文回答四个问题」这类以自我为主语的引导句，改为直接陈述四问；全页自称统一为「本页」或「本文」其中一种｜修复：｜复验：
- [轻微·可读] §1 与 §3：同一维度两处符号不同且未说明等价｜引文依据：不适用（位置：§1 公式分母 $\sqrt{d_k}$ 与折叠块「每组 K/V 是一个 $d_k$ 维向量」；§3 符号表「$d_{\text{head}}$：每个头的向量维度」）｜修复要求：统一为单一符号，或在 §3 符号表注明 $d_k=d_{\text{head}}$｜修复：｜复验：
- [轻微·技术] §2：同一章内缓存规模一处 20,100、一处 20,000，未说明取整｜引文依据：不适用（位置：§2「prefill 处理 20,100 个输入 token，缓存从 0 涨到 20,100 token」（＝手册 20,000＋问题 100）；同节「注意力要读取全部缓存（$20{,}000\times 128\,\text{KB}\approx 2.44\,\text{GB}$）」）｜修复要求：把读取量算式改为 20,100 token（$2.46$ GB），或在式旁注明此处按手册规模 20,000 取整｜修复：｜复验：
- [轻微·可读] §3 与来源章：GQA 定义句易误读为「多个查询头共用 1 个 KV 头」｜引文依据：不适用（位置：§3「GQA（分组查询注意力）中多组查询头共享一组 KV 头」、来源章「GQA 只给最小定义（多组查询头共享一组 KV 头）」，对比 §3 本章问题答案「GQA 把查询头分成 $H_{\text{kv}}$ 组、每组共享同一个 KV 头」）｜修复要求：两处统一为「查询头分成 $H_{\text{kv}}$ 组，每组共享一个 KV 头」｜修复：｜复验：
- [轻微·技术] §2：一处机制描述无来源上标也无概念页链接｜引文依据：不适用（位置：§2 末「因此优化 prefill 与 decode 的手段不同，部署形态上也衍生出把它们分开调度的方案」）｜修复要求：为该句补来源上标或指向对应概念页的链接，或改写为明确标注的概括｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复（重要项 1 条修复后即可发布；轻微项可按上列要求一并修正）
