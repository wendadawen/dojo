<!-- review-meta
round: 6
page: wiki/chunked-prefill/index.html
reviewed_content_sha256: 396a221f017a479e
-->
# Chunked Prefill 审查记录（第 6 轮）

- 页面版本：b2d8b4d7a08455bf61bf549690ffbd451c9aa6c4
- 审查时间：2026-09-13 21:06
- 审查者：独立子代理
- 已完整阅读章节：核心问题 → 1. 长 prefill 为什么卡住所有人（含本章问题）→ 2. 把长 prefill 切成块：机制与代价（2.1 块间依赖、2.2 切块的三个代价、本章问题）→ 3. decode 搭车与不停止的调度（3.1 piggybacking、3.2 stall-free、本章问题）→ 4. 块要多大：token budget 的权衡（本章问题）→ 5. 与流水线并行合流：CPP（本章问题）→ 6. 边界与相邻工作（本章问题）→ 来源与范围说明

## 来源核对（引文依据）

- 200 倍：SARATHI §1/§3.1「the decode cost per-token is 200× ... at batch size of 1」（页面标注"小批量"，与 batch size 1/2/18 = 200×/100×/16.7× 的条件自洽）。✔
- 512 token 饱和：SARATHI §1「a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one」（A6000/LLaMA-13B，页面标注一致）。✔
- 搭车成本：SARATHI 摘要「decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch」——页面"最多低一个数量级（不是零）"一致。✔
- KV 重复读：Sarathi-Serve §4.3「the first chunk's KV-cache is loaded N−1 times, the second chunk's KV-cache is loaded N−2 times」——页面采用 N−1 计数并注明与 SARATHI（N 次）的差异。✔
- 257 vs 256：Sarathi-Serve §4.3「using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256」（原文限定 "in some cases"，页面标注"特定 GPU/kernel 的实例，量级在其他硬件上可能不同"）。✔
- compute-bound 可容忍：Sarathi-Serve §4.3「even at small chunk sizes attention prefill operation is compute bound operation」+「small overhead ... fixed overheads of kernel launch」。✔
- token budget / stall-free：Sarathi-Serve §4.2「budget of maximum number of tokens ... based on user specified SLO」、Algorithm 3「Only after all the running requests have been accommodated, we admit new requests」；§3.3「If we can ensure that each micro-batch performs uniform computation, we can mitigate these pipeline bubbles」。页面"从不暂停进行中的 decode""uniform 时长"表述与之一致（原文明说 mitigate/ameliorate，页面在"简化条件"处也承认残余不均仍存在）。✔
- generation stall：Sarathi-Serve §1「generation stalls lasting over several seconds in vLLM」。✔
- 6.29× / 1.91×：SARATHI 摘要「reduces bubbles by 6.29×, resulting in an end-to-end throughput improvement of 1.91×」（GPT-3 PP）。✔
- Sarathi 吞吐：摘要「LLaMA-13B/A6000 decode up to 10×、e2e up to 1.33×」「LLaMa-33B/A100 1.25× e2e、up to 4.25× decode」。✔
- Sarathi-Serve vs vLLM：摘要「Mistral-7B single A100 2.6×、Yi-34B two A100 up to 3.7×、Falcon-180B PP up to 5.6×」。✔
- Beyond the Buzz §4 Figure 5 caption：「Chunked pipeline parallelism during Prefill is an optimal strategy to maximize throughput while complying with strict FTL SLA」「DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)」「FTL can be reduced as we increase the PP, while keeping throughput high」；§4.1「redundant computation of down and up projections in multi-latent attention for each prefill chunk. This can be mitigated by temporarily caching the up-projected KV values」；§4.1「most beneficial under relaxed latency targets and generation-heavy traffic patterns」。✔

## 复算（全部通过）

- $\sum_{i=1}^{N}(N-i)=\frac{N(N-1)}{2}$；$N=8$→28，$N=4$→6（比值 4.67≈"约 4.7 倍"），$N=16$→120（6→120 = 20 倍）。✔
- 8000/512 = 15 余 320（16 块，末块 320）。✔
- τ=512、64 decode → 448 chunk；8000/448 ≈ 17.86 → "约 18 次迭代"；64+448=512=τ。✔
- 8000 = 15×512+320、4096/1024=4、4096/256=16。✔
- 全页数字与 summary/overview 一致，无正文与折叠块互斥的表述；validate.py 返回 "validation ok"；所有内链（moe-serving/causal-mask/gpu-execution-model/model-parallelism/pp-load-balancing/beyond-buzz-disaggregation）页面真实存在；无代码块（无需运行）；alt/aria-label 无 `$...$`；无 Unicode 数学字符（仅 402 行 ↑ 为返回顶部按钮，非数学符号）。

## 问题

- [轻微·表述] 3.1 节 piggybacking 段：句首"注意"是元话语引导词（check.md 第 12 项明列"需要注意的是"一族，且隐含第二人称命令）｜引文依据：不适用｜修复要求：删除句首"注意"，改为直陈——"'最多低一个数量级'不等于零：搭车 decode 仍占批容量与少量计算，说'完全免费'是错的。"｜修复：｜复验：
- [轻微·可读性] 核心问题第 5 条解答首句："PP 气泡的来源之一是各 micro-batch 计算时长不均（FTL = First Token Latency，首 token 延迟）。"——FTL 定义被贴在"计算时长不均"之后，读起来像 FTL 是对该不均现象的命名，而 FTL 实为该答案后文"严格 FTL 约束"处才使用的术语｜引文依据：不适用｜修复要求：把"（FTL = First Token Latency，首 token 延迟）"移到本答案中 FTL 首次实际出现的位置之前（"严格 FTL 约束"处），首句不再挂该括注｜修复：｜复验：
- [轻微·技术] 来源与范围说明·公式与来源（F）F1 条：称 F1"（本页推导，标注为'由 C5 计数推出'）"，但正文（2.2 节公式、第 2 章本章问题解答）只有 `<sup>[C5]</sup>`、`<sup>[C5, F1]</sup>` 标记，全文检索"由 C5 计数推出"为 0 处，该括注描述不存在｜引文依据：页面第 203、230 行仅有 `[C5]`/`[C5, F1]` 上标，无该字样｜修复要求：删除"标注为'由 C5 计数推出'"，或改为与正文一致的表述——"由 C5 的逐块计数求和得到，正文以 [C5] 标注来源"｜修复：｜复验：
- [轻微·技术] 第 1 章第 2 段把连续批处理的由来归因为"Orca 提出的迭代级调度"，该归因无 [Cx]/[Nx] 引用，来源清单亦无 Orca 条目（同段"vLLM 的做法"则有 [C9]）｜引文依据：不适用（页面未给出可定位的来源位置）｜修复要求：为该归因补一条来源条目，或删除"Orca 提出的"、只保留"连续批处理（continuous batching，迭代级调度：每次迭代可加入新请求、移除完成请求）"的术语解释｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布（四例轻微问题不改变正确性与主线，修复后更佳）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
