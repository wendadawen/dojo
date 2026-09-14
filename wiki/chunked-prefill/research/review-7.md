<!-- review-meta
round: 7
page: wiki/chunked-prefill/index.html
reviewed_content_sha256: aa8d60b339752b55
-->
# Chunked Prefill 审查记录（第 7 轮）

- 页面版本：be05265ac24158caad22fdc0ed8a42eba1594431（index.html）；overview.html：af72d5b92542c19b057cf67e82cef6867b8f235b
- 审查时间：2026-09-14 16:47
- 审查者：独立子代理（未参与写作与前序轮次）
- 规范：guides/concept/check.md（dojo:type=concept），对照 guides/concept/style-guide.md
- 已完整阅读章节：核心问题（5 条）→ 1. 长 prefill 为什么卡住所有人（含本章问题）→ 2. 把长 prefill 切成块：机制与代价（2.1 块间依赖、2.2 三项代价、本章问题）→ 3. decode 搭车与不停止的调度（3.1 piggybacking、3.2 stall-free、本章问题）→ 4. 块要多大：token budget 的权衡（含本章问题）→ 5. 与流水线并行合流：CPP（含本章问题）→ 6. 边界与相邻工作（含本章问题）→ 来源与范围说明（C/F/N/构造示例/类比边界/简化条件）。
- 机械验证：`.dojo/scripts/validate.py wiki/chunked-prefill/index.html` 返回 `validation ok`；无 Unicode 数学字符落在 `$...$` 之外；无 `（待生成）`/TODO/占位；6 个前置概念链接（moe-serving、causal-mask、gpu-execution-model、model-parallelism、pp-load-balancing、beyond-buzz-disaggregation）均在仓库中真实存在；`dojo:topics=推理系统, 并行与通信` 与 `dojo:tag=推理系统` 均在 AGENTS.md 与 ALLOWED_TAGS 允许集内；index.html 与 overview.html 互链。

## 来源核对（本轮实际定位到的原文）

- N2（512 token 饱和）：Sarathi §1 原文 "a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"（A6000/LLaMA-13B），与正文 line 112/154、表格 line 120 一致。
- N1（200 倍）：Sarathi §1 "at small batch sizes, the decode cost per token can be as high as ~200 times the prefill cost per token"，与正文 line 112/154 一致。
- N6 / N4：Sarathi 摘要 "LLaMA-13B/A6000 decode 至多 10×、端到端 1.33×；LLaMA-33B/A100 端到端 1.25×、decode 至多 4.25×"，"GPT-3 + PP：bubbles by 6.29×, end-to-end throughput 1.91×"，与 line 341、313 一致；GPT-3 结果确为模拟（Table 3 标注 "Simulation"，原文 "we report evaluations in a carefully simulated environment"），页面"模拟实验"表述正确。
- C5（KV 重复读计数）：Sarathi-Serve §4.3 "if a prefill sequence is split into N chunks, then the first chunk's KV-cache is loaded N-1 times … the second chunk's KV-cache is loaded N-2 times"；Sarathi 原文对应句为 "the first chunk's KV cache is loaded N times, the second chunk's KV cache is loaded N−1 times"。页面对两文计数差异的元说明（line 359）与采用 Sarathi-Serve 计数的选择均正确；F1 求和 $\sum_{i=1}^{N}(N-i)=N(N-1)/2$ 复算无误。
- C5 "compute-bound 可容忍"：Sarathi-Serve §4.3 "even at small chunk sizes attention prefill operation is compute bound operation"。
- C9（秒级停顿）：Sarathi-Serve "a generation stall in vLLM can last over several seconds"，与 line 126/145/161 一致。
- N3（257 vs 256 = 32%）：Sarathi-Serve §4.3 "using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256"（原文为 prefill time，页面 line 213/237 写"prefill 计算时间"，量级与来源一致）。
- C10/C11（token budget 与 profiling）：Sarathi-Serve §4.2/§4.3 "compute the maximum chunk size that can be accommodated within the leftover token budget for that batch"、"setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO"；Algorithm 3 将 decode token 与 chunk token 计入同一预算，页面 line 256/273 的"总 token 数 ≤ τ"定义正确。
- C13：Sarathi-Serve §4.3 将切块开销列为 "lower GPU utilization" 与 "repeated KV-cache access"，并有 "fixed overheads of kernel launch"，与 line 209/210 一致。
- C7/C8：Sarathi 摘要 "decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch"、"Chunked-prefills allows constructing multiple decode-maximal batches from a single prefill request"，与 line 248/252 一致。
- C15/N5：Sarathi-Serve 摘要 Mistral-7B 单 A100 2.6×、Yi-34B 双 A100 至 3.7×、Falcon-180B（PP）至 5.6×，与 line 341 一致。
- C16：Beyond the Buzz §4 Figure 5 caption "DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"、"an optimal strategy to maximize throughput while complying with strict FTL SLA"；正文 "how FTL can be reduced as we increase the PP, while keeping throughput high"。与 line 315/330 一致。
- C17：Beyond the Buzz §4.1 "redundant computation of down and up projections in multi-latent attention for each prefill chunk"、"This can be mitigated by temporarily caching the up-projected KV values from earlier chunks"，与 line 315 一致。
- 边界结论：Beyond the Buzz 摘要 "disaggregation is most effective for prefill-heavy traffic patterns and larger models"、结论 "serving small-scale models or generation-heavy traffic"，以及 "chunking … is most beneficial under relaxed latency targets and generation-heavy traffic patterns"，与 line 337/349 一致。
- 构造示例复算：8000/512 → 16 块（末块 320）✓；4096 切 8 块 = 8·7/2 = 28、切 4 块 = 4·3/2 = 6 ✓；256 对应 N=16 → 120、1024 对应 N=4 → 6，比值 20 ✓；τ=512、64 decode + 448 chunk，8000/448 ≈ 18 次迭代 ✓。

## 问题

- [轻微·格式] 2.1 数据流图图注（index.html 图 2，line 194）称连接线为"实线箭头"，但图中 `<line class="dg-line">`（line 190–192）没有箭头端点，`.dojo/scripts/../libs/dojo-concept.css` 的 `.diagram svg .dg-line { stroke: var(--text-light); fill: none; }` 也不绘制 marker，图上实际是无向连线，图注与图上不一致。｜引文依据：不适用｜修复要求：或为三条连线加箭头端点（marker-end），或把图注"实线箭头"改为与图一致的"实线"，并保留方向说明文字。｜修复：｜复验：
- [轻微·表述] 全文口语化措辞若干处，与概念页书面语要求不符：line 65"以几十毫秒的间隔蹦一个 token"、line 168"第 1 章的 stall 结构被拆掉了"、line 244"切块解决'长 prefill 卡人'"、line 252"chunked prefill 与 piggybacking 是相互成就的"、line 315"CPP … 成为 prefill 池的标准答案"。｜引文依据：不适用｜修复要求：逐处替换为中性书面表述（如"逐 token 输出"、"消除了该 stall 结构"、"长 prefill 拖慢同批 decode"、"chunked prefill 与 piggybacking 相互依赖"、"CPP 成为 prefill 池的最优策略"）。｜修复：｜复验：
- [轻微·表述] 同一比值（28/6）在正文两处给出不同数值：line 215"块数翻倍，重复读约 4 倍"，line 230"额外读取从 6 涨到 28（约 4.7 倍）"。｜引文依据：不适用｜修复要求：统一为同一写法（如两处均写"约 4.7 倍"，或在 215 处明确标注为 $N^2$ 渐近的粗估）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（3 条轻微问题按上表修复后不影响发布）
- 说明：本轮逐条回源核对全部 C/F/N 论断与构造示例数值，未发现定位不到、来源不支持、条件被抹去或编号错配的情形；页面核心结论（generation stall 机制、KV 依赖与 $\frac{N(N-1)}{2}$ 代价、piggybacking、token budget、CPP 消气泡）均有可定位原文支持，两级问题块齐备且答案与正文一致。