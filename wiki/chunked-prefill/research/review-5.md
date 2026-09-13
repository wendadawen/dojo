<!-- review-meta
round: 5
page: wiki/chunked-prefill/index.html
reviewed_content_sha256: 7f0a47b83dcf11c8
-->
# Chunked Prefill 审查记录（第 5 轮）

- 页面版本：ebd5c8399af16d175554f0cebd75e710b01d8ffd
- 审查时间：2026-09-13 20:11
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题（5 条两级问题块）、1. 长 prefill 为什么卡住所有人、2. 把长 prefill 切成块：机制与代价（2.1 块间依赖、2.2 切块的三个代价）、3. decode 搭车与不停止的调度（3.1 piggybacking、3.2 stall-free）、4. 块要多大：token budget 的权衡、5. 与流水线并行合流：CPP、6. 边界与相邻工作、来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）

## 本轮机械验证结果

- `python3 .dojo/scripts/validate.py wiki/chunked-prefill/index.html` → validation ok。
- 头字段：description 纯文本、dojo:summary 含 `$...$`、dojo:type=concept、dojo:topics=`推理系统, 并行与通信`（均在 ALLOWED_TOPICS）、dojo:tag=`推理系统`（在 ALLOWED_TAGS）→ 全部合规。
- 数学符号：全页正文/标题/summary/表格/图注中的非 ASCII 字符仅 `§ ✓ ⌂ ◐ ☀ ↑ ▼`，无 Unicode 数学字符；公式均为 KaTeX。
- 链接：6 个前置概念链接（moe-serving、causal-mask、gpu-execution-model、model-parallelism、pp-load-balancing、beyond-buzz-disaggregation）页面均真实存在，无「（待生成）」占位；overview.html 与 index.html 双向互链。
- 算式复算：8000/512→16 块（末块 320）；8×7/2=28、4×3/2=6、28/6≈4.7；16×15/2=120、120/6=20；τ=512 时 64+448=512、8000/448≈18；均与页面标注一致。
- 来源数字核对（原文片段）：SARATHI 摘要 6.29x/1.91x、LLaMA-13B/A6000 decode ≤10x 与端到端 1.33x、LLaMA-33B/A100 1.25x 与 ≤4.25x；Sarathi-Serve 摘要 2.6x/3.7x/5.6x；SARATHI §1「a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one」「at small batch sizes, the decode cost per token can be as high as ∼200 times the prefill cost per token」「cost up to an order of magnitude less compared to a decode-only batch」；Sarathi-Serve §4.3「the first chunk's KV-cache is loaded N−1 times, the second chunk's KV-cache is loaded N−2 times」「using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256」「even at small chunk sizes attention prefill operation is compute bound operation」「smaller token budget can result in excessive chunking … overheads due to 1) lower GPU utilization and 2) repeated KV-cache access」「fixed overheads of kernel launch」「one-time profiling … without violating TBT SLO」；Beyond the Buzz §4.1「most beneficial under relaxed latency targets and generation-heavy traffic patterns」、Figure 5 caption「Chunked pipeline parallelism during Prefill is an optimal strategy to maximize throughput while complying with strict FTL SLA」、MLA「redundant computation of down and up projections … mitigated by temporarily caching the up-projected KV values」。以上全部与页面表述一致。
- 说明：来源说明中「Sarathi 原文为 N 次、Sarathi-Serve 为 N−1 次」经核对成立（SARATHI §4.2「the first chunk's KV cache is loaded N times」 vs Sarathi-Serve §4.3 的 N−1）；C9 所在位置为 Sarathi-Serve §1/§3.2，来源说明标注的「§3」已覆盖。均无需修改。

## 问题

- [重要·技术] §3.1 第 3 段（正文行 252）：句子「不切块时，批里要么是纯 prefill（decode 停摆）、要么是纯 decode（算力闲置）」与 §1（行 126）自相矛盾，也把「不切块」的基线描述错了。§1 明说连续批处理「把 prefill 与 decode 混在同一个批里迭代」，并描述不切块时长 prefill 的迭代里「同批所有进行中的 decode」被卡住；即不切块时批内是混的，不是「纯 prefill」。Sarathi 的 decode-maximal batching 在不切块时同样能构造一次混合批次（一个 prefill chunk/整个 prefill + 其余槽位填 decode），切块的作用只是把搭车机会从 1 次变成 N 次，而非「不切块就没有混合批次」。｜引文依据：SARATHI §1「decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes」；本页行 126「连续批处理……把 prefill 与 decode 混在同一个批里迭代……期间同批所有进行中的 decode 一个 token 都不产出」。｜修复要求：改写该句，使「不切块」基线与 §1 一致——不切块时一个 prefill 只能构造一次混合批次（该迭代装下整个 prompt，批内 decode 在这个长迭代内停摆），切块把它变成 N 次；不得写成「批里要么是纯 prefill 要么是纯 decode」。｜修复：｜复验：
- [轻微·技术] §5 正文（行 313）与本章问题解答（行 323）：「双重气泡都被压缩<sup>[C14]</sup>」「第一重也被均匀的 micro-batch 流压缩」把「填充期/排空期气泡被压缩」当作 C14 的来源结论引用，但 C14（SARATHI 摘要）只把气泡减少归因于消除 micro-batch 时长不均；填充/排空气泡因「micro-batch 数增多而被摊薄」是本页的推断，来源未如此陈述。｜引文依据：SARATHI 摘要「the uniform compute design of these batches ameliorates the imbalance between micro-batches, significantly reducing pipeline bubbles」。｜修复要求：把「填充期/排空期气泡被压缩」降级为明确标注的推断（如注明由切块后 micro-batch 数增加推出），或让 [C14] 只标注「时长不均」那一重气泡。｜修复：｜复验：
- [轻微·格式] 来源与范围说明 · 构造示例（行 383）：首条把「8000 token prompt / 512 每块 / 16 块（末块 320 token）/ 64 个 decode 请求」合并记为同一个「贯穿示例的整数化设定」，但这四个参数在正文任何一处都不同时成立——「512 每块」只出现在第 2 章（那里未涉及 decode 数），「64 个 decode 请求」只出现在 §3.2 的 τ=512 示例，且该处 chunk 是 448 token 而非 512。来源说明与正文用例对不上。｜引文依据：本页行 168「8000 token 的 prompt 按 512 一块切成 16 块（末块 320 token）」；行 258「批容量按 token 计 512（τ = 512），当前有 64 个进行中的 decode……1 个 448 token 的 prefill chunk」。｜修复要求：把首条构造示例的参数与实际用例对齐——删去「64 个 decode 请求」，或拆成两条（第 2 章 8000/512/16；第 3 章 8000 + 64 decode + 448 chunk），使来源说明可逐项在正文中定位。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（1 条重要、2 条轻微均需逐条修复并复验；核心结论、公式、数字与来源一致，无需返回规划）
