<!-- review-meta
round: 7
page: wiki/mooncake/index.html
reviewed_content_sha256: 1bb9eb2d59d4511c
-->
# Mooncake审查记录（第 7 轮）

- 页面版本：c3fe40c769802650c19186bc1e7f458fb0919077
- 审查时间：2026-09-14 14:38
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作与前序轮次）
- 来源获取：arXiv:2407.00079v4 全文（HTML + PDF）+ 官方摘要/版本页；图内数值以 PDF 页面 150/600 dpi 像素测量核对；站内前置概念页直接读取
- 已完整阅读章节：核心问题、常见误解、1. 为什么把 prefill 与 decode 拆开（1.1–1.2、本章问题）、2. Mooncake 架构总览（2.1–2.3、本章问题）、3. KVCache-centric 调度算法（3.1–3.4、本章问题）、4. 长上下文 prefill 的多节点与传输重叠（4.1–4.3、本章问题）、5. 过载场景下的早拒绝与预测（5.1–5.4、本章问题）、来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、简化条件及其限制），含全部折叠块与图注

## 回源核对摘要（本轮逐条核对，均定位到原文片段/数值）

- 架构与组件（C2、C3、C11、C36、§3）：Abstract「splits the prefill and decoding clusters」「reuses spare CPU, DRAM, and SSD capacity」；§3「It also replicates or swaps certain blocks of the KVCache if it is beneficial for future inference」；§6.1「Conductor directly returns the HTTP 429 Too Many Requests response status code」；§3 step 2/3/4 逐句对应页面四步工作流与「Messenger service is deployed in each node... operates as an independent process」。
- trace 数字（N1–N6）：§4「23,608 entries」「1-hour period」；§4.2「average input length of 7,590 tokens and an average output length of 182 tokens」；§4.1「block size of 512」；Table 1 LRU 1000→0.30、50000→0.50；§4.2「from 1,000 to 50,000 blocks boosts the cache hit ratio from 30% to 50%」「over 50% of cache blocks remaining unused while certain blocks are accessed tens of thousands of times」。
- Algorithm 1（C22、C23、C24）：页面伪代码与论文 Algorithm 1 逐行一致（分支阈值 `best_prefix_len/prefix_len < kvcache_balancing_threshold`、`transfer_len ← best_prefix_len − prefix_len`、末尾 `TransferKVCache(best_matched_instance, p)`）；§6.1「a predictive model derived from offline test data」。
- 调度实验（N8、Figure 8）：§6.2「8 prefill instances and 8 decoding instances... replayed 23,000 real-world requests」；Figure 8 数值标签 92.07 / 60.41 / 14.36 / 6.26（KVCache-centric / cache-aware / load-balancing / random）与页面一致；像素测量 SLO 虚线落在 y≈29.5–30.1 s，支持页面「SLO 线（≈30s）」。
- 长上下文 prefill（C16–C20、F2）：§5.1「two expensive RDMA-based all-reduce operations per layer」「cross-node communication at least once per layer... Ring Attention or Striped Attention」「group every X nodes... partitioned into chunks, each no longer than the prefill_chunk」「cross-node communication only at the boundaries of each pipeline stage」；§5.2「its occupation cost is S ∗ T」「Before each layer's attention computation begins, the model waits for the asynchronous loading... After the attention calculation is complete, asynchronous storage of that layer's KVCache is launched」「roughly equivalent to either the KVCache loading time or the standard prefilling time」「disregard the available VRAM size... as long as it can contain a single request」。
- 过载与早拒绝（C25–C28、N9）：§7.1「we use SLO satisfaction as a direct load measurement」；§7.2「based on the greater load between the prefill and decoding pools」；§7.3「significant anti-phase fluctuations」；§7.4 系统级预测「each request's decoding takes a fixed time t_d... the mean TBT ratio... against l_tbt」；Table 3 / §8.2「4183 / 3771 / 3589」「increased the replay speed to 2x」「8 prefill instances and 8 decoding instances」。
- 端到端（C29–C32、N10–N15）：§2「TTFT_P90 = 10× and TBT_P90 = 5×」；§8.1.1「20% and 40%」；§8.1.2「vLLM processes requests individually, rather than in batches」「from 50% to 525%」；§8.1.3「upper limit for the TTFT is set at 30 seconds, while the TBT threshold is capped at 0.1 seconds per token」「only 57% of the requests for vLLM-[20M]」「approximately 75% more requests」；Testbed「8 NVIDIA-A800-SXM4-80GB... up to 800 Gbps」；Table 2 ArXiv ~0%、L-Eval >80%。
- 复用上界与 dummy 模型（C34、C35、N16）：§9「up to only 50% of the KVCache can be reused... as large as 90%... chat-to-paper service」；§1 footnote「dummy model that follows the same architecture as LLaMA2-70B... without any real user content」。
- 公式与算例：F1 `2·L·H_kv·d_head·b` 代入 L=80/H_kv=8/d_head=128/b=2 得 327,680 B = 320 KiB/token，×12,288 = 3.75 GiB，×131,072 ≈ 40 GiB，均复算一致；第 3 章贯穿示例三条分支 TTFT（4.524 / 9.012 / 3.036）与四种策略对比（9.012 / 4.572 / 4.524 / 3.036）逐项复算无误；`1/4000 ≈ 0.00025 s/token ≈ 4000 tok/s`、`T_transfer = t/10000` 与文字一致。
- 机械项：validate.py 返回 `validation ok`；页面引用 paged-attention / kv-cache / pcp-dcp / chunked-prefill / prefix-caching 均真实存在；无「（待生成）」；无 `alt` 内 `$...$`；SVG 仅含纯文字标签、无 ASCII 近似公式；KaTeX 定界符正常。

## 问题

- [重要·技术] 第 3.1 节「块哈希链与 prefix 匹配」首个段落：`<sup>[C10]</sup>` 标注的论断与「论断与来源（C）」表中 C10 条目不对应，属同一引文编号在正文与来源表之间不一致｜引文依据：该句为「对每个到达的请求 R，Mooncake 的 Conductor 第一步是算它能命中多少 KVCache」，其内容出自论文 §6.1「its input tokens are divided into several blocks, and a hash key is computed for each block... The request's block keys are then compared one by one against each prefill instance's cache keys to identify the prefix match length (prefix_len)」（亦见 Algorithm 1 第 1、4 行）；而表中 C10 = 「请求处理三步：传可复用 KVCache → 分块/分层完成 prefill 并流式传 KVCache → decode 加入连续批处理｜§1」，指论文 §1「the global scheduler (Conductor) needs to select a pair of prefill and decoding instances and schedule the request in the following steps: 1) transfer as much reusable KVCache...; 2) complete the prefill stage in chunks/layers and continuously stream the output KVCache...; 3) load the KVCache and add the request to the continuous batching process」｜修复要求：二选一——(a) 把该处标注改为与内容对应的来源（`[C22]` 或 `[§6.1]`），或 (b) 把 C10 的标注移到正文确实陈述「三步工作流」的位置（如第 2 章描述请求步骤处），使标签与条目双向一致｜修复：｜复验：

- [轻微·技术] 来源与范围说明·「外部数字与实验条件（N）」：N1、N2、N14、N15 四条在正文中没有任何 `<sup>[Nx]</sup>` 引用（表→正文单向），不满足 style-guide §6「与来源章节双向对应」｜引文依据：不适用（机械项；四条内容本身回源无误：N1 §4「23,608 entries... 1-hour period」、N2 §4.2「average input length of 7,590 tokens and an average output length of 182 tokens」、N14 §8.1.3「the upper limit for the TTFT is set at 30 seconds, while the TBT threshold is capped at 0.1 seconds per token」、N15 §8.1 Testbed「8 NVIDIA-A800-SXM4-80GB... up to 800 Gbps」）｜修复要求：在正文相应位置补上引用（例如第 3.4 节 SLO 线处补 `[N14]`、第 5 章回放实验处补 `[N1, N2]`、第 5.4 节实验配置处补 `[N15]`），或在不保留对应正文论断时删除该条目，使编号双向对应｜修复：｜复验：

- [轻微·技术] 常见误解第 2 条：对「PD 分离此前已有 Splitwise、DistServe、TetriInfer 等并行工作」这一归因未给出任何来源标注（该论断在本页仅出现于此，正文无对应引文编号）｜引文依据：论文 §9「recent research shares our insight into separating the prefill and decoding stages... The arXiv publication of Splitwise [7]... DistServe [8]... TetriInfer [9]」｜修复要求：为该归因补上正文可见的来源标注（可新增 `[Cx]` 并在表中定位到 §9，或直接写作「论文 §9 列出…」）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复
