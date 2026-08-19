# Chunked Prefill 审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 4751b577a9111e5e6efab9f034afc37fdde49731
- 审查时间：2026-08-19
- 审查者：独立子代理（第 3 轮，未参与写作与前两轮审查）
- 已完整阅读章节（index.html 按序）：导言与学习目标、核心问题（5 题及解答）、1 长 prefill 为什么卡住所有人（含 2 问）、2 切成块（2.1/2.2，含 3 问）、3 搭车与 stall-free（3.1/3.2，含 2 问）、4 token budget 权衡（含 1 问）、5 CPP（含 2 问）、6 边界与相邻工作（含 1 问）、来源与范围说明；overview.html 全文。

## 前两轮修复项复核

- 阻断（2.2 节单 ul、公式在首 li、不重复）：已到位（index.html 886–898 行）。
- 重要 2（[C16,N4]→[C16]）：已到位（1002 行）。
- 轻微已到位：Unicode × 换（正文无裸 ×/≈/→，EP $\times$ PP 用 LaTeX）；TBT/FTL/MLA 首现全称；20ms 改"几十毫秒（示意值）"；"翻平方"改"约 4 倍/4.7 倍"；C6 推断标注；SVG 内 text 无公式；overview 5 项 head meta 与 [1][2] 条件；overview 脚注移至问题背景段后。
- 重要 1（第 4 章 N_0 删除、16 倍→20 倍、L 符号冲突）：**未落地**，见问题 1、2。
- 轻微（ISL/EP 首现全称）：**未落地**，见问题 4。

## 来源核对（引文依据摘录）

C1/C2/N1：Sarathi-Serve §1 "prefill…compute-bound…decode…memory-bound"；Sarathi §1 "decode cost per token can be as high as ∼200 times the prefill cost per token"。C3/N2：Sarathi §1 "a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"。C4：Sarathi-Serve "splits a prefill request into near equal sized chunks"。C5：Sarathi-Serve §4.3 "first chunk's KV-cache is loaded N−1 times, the second N−2…attention prefill operation is compute bound"；Sarathi §4.2 "same number of computations for FFNs"。C7：Sarathi 摘要 "piggyback and cost up to an order of magnitude less compared to a decode-only batch"。C8：Sarathi 摘要 "multiple decode-maximal batches from a single prefill request"。C9：Sarathi-Serve §1 "a generation stall in vLLM can last over several seconds"。C10/C11：Sarathi-Serve §4.2/4.3（Algorithm 3 τ；"one-time profiling…without violating TBT SLO"）。C12/N3：Sarathi-Serve §4.3 "chunk size of 257 can increase prefill time by 32% compared to…256"。C13：Sarathi-Serve §4.3 "lower GPU utilization…fixed overheads of kernel launch"。C14/N4：Sarathi 摘要 "reduces bubbles by 6.29×…throughput improvement of 1.91×"（但见问题 5：GPT-3 为模拟）。C15/N5：Sarathi-Serve 摘要 "2.6×…3.7×…5.6×"。C16：paper §4 "FTL constraints apply only to the prefill (context) pool…CPP to be especially effective"；Fig.5 题注 "optimal strategy to maximize throughput while complying with strict FTL SLA…DeepSeek-R1 with ISL of 256K on 64 GPUs (EP × PP = 64)"。C17：paper §4.1 "redundant computation of down and up projections in multi-latent attention for each prefill chunk…mitigated by temporarily caching the up-projected KV"。N6：Sarathi 摘要 "10×…1.33×…1.25×…4.25×"。F1 复算：N=8→28、N=4→6 ✓；τ=512 示例 64+448、8000/448≈18 ✓。前置链接 5 个目标页均存在。

## 问题

- [重要·技术] index.html §4 本章问题解答（991 行）："从 $\frac{L}{1024}\cdot\frac{N_0-1}{2}$ 量级涨到 16 倍"：$N_0$ 全页未定义（前两轮要求删除，未执行）；"16 倍"与同句括号内数值矛盾——$N=4$ 为 6 块次、$N=16$ 为 120 块次，120/6=20 倍｜引文依据：Sarathi-Serve §4.3 逐块计数 N−1/N−2；F1 复算 120/6=20｜修复要求：删除 $N_0$ 表达式，"16 倍"改"20 倍"｜修复：｜复验：
- [重要·技术] index.html §2.1 图示题注（881 行）与 §4 解答（991 行）符号冲突：题注"过全部 $L$ 层"以 $L$ 为层数，§4"对应 $L=4096$ 的实例"以 $L$ 为 prompt 长度，同页同符两义（前两轮"L 符号冲突解决"未执行到题注）｜引文依据：不适用（check.md §2.2-9 同一变量全页一致）｜修复要求：题注改"过全部层"，$L$ 仅保留一种含义｜修复：｜复验：
- [重要·可读性] overview.html 57 行：`<h2>核心机制</h2>` 后残留脚注段尾重复文本"：on an A6000 GPU…prefill cost per token。"及游离 `</p>`（与 55 行脚注段重复，疑为脚注移位编辑事故），"核心机制"标题下出现孤立英文残句｜引文依据：不适用｜修复要求：删除 h2 结束标签至该 `</p>` 之间的全部残留文本｜修复：｜复验：
- [轻微·可读性] index.html 1002/1017 行：ISL、EP 首次出现未给全称（1002 行另有 SLO 首现于 784 行未展开）｜引文依据：paper 附录 Table 1 "First Token Latency / FTL"同理应有全称｜修复要求：首现改为"ISL（Input Sequence Length，输入序列长度）""EP（Expert Parallelism，专家并行）"｜修复：｜复验：
- [轻微·技术] index.html 1000、1010 行：GPT-3 流水线"实测气泡缩小 6.29 倍"——Sarathi 的 GPT-3 64 GPU 实验为模拟（基于 profiling 回归）｜引文依据：Sarathi Table 3 "GPT-3…Simulation"；§5.3 "we report evaluations in a carefully simulated environment"｜修复要求：两处"实测"改"模拟实验"或同等表述｜修复：｜复验：
- [轻微·可读性] index.html 977 行表格"取 2 的幂等 tile 友好值"易误读为"幂等（idempotent）"｜引文依据：不适用｜修复要求：改"取 2 的幂这类 tile 友好值"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 3
- 处置：修复。三项重要中两项为第 2 轮"重要 1"修复未落地（回归或漏改），一项为 overview 编辑残留；修复后复验并运行 `.dojo/scripts/validate.py`，再核对第 4 章解答与 §2.1 题注。


## 修复记录（追加第 1 次）

按 check.md §4 重要问题追加修复（最多 2 次，本次为第 1 次）：

- 重要 1：§4 本章问题 1 解答重写——删 `\frac{L}{1024}\cdot\frac{N_0-1}{2}`（$N_0$ 全页未定义且 L 与第 2 章层数符号冲突），改为 "KV 重复读按 $\frac{N(N-1)}{2}$ 从 $N=4$ 时的 6 块次涨到 $N=16$ 时的 120 块次（精确比值 20 倍，按 $N^2$ 量级增长；prompt 长度 4096 的实例）"。复算：$4\cdot 3/2 = 6$、$16\cdot 15/2 = 120$、$120/6 = 20$。
- 重要 2：§2.1 图示题注 "过全部 L 层" → "过全部层"（前两轮审查中已修复但被识别为回归位），验证后已到位。
- 重要 3：overview.html 57 行编辑残留——`<h2>核心机制</h2>` 后的"：on an A6000 GPU...prefill cost per token。"段是上一轮脚注移位操作的编辑事故，已删除重复段，恢复 h2 后直接进入机制列表。
- 轻微 1：第 5 章 "ISL 256K" → "ISL（Input Sequence Length，输入序列长度）256K"；"EP $\times$ PP = 64" → "EP（Expert Parallelism，专家并行）$\times$ PP = 64"（ISL/EP 全称补全）。
- 轻微 2：两处 "GPT-3 流水线气泡缩小 6.29 倍" → "GPT-3 流水线模拟气泡缩小 6.29 倍"（Sarathi Table 3 标明 GPT-3 64 GPU 实验为 Simulation，非实测；§5.3 述"carefully simulated environment"）。
- 轻微 3：第 4 章 token budget 权衡表 "取 2 的幂等 tile 友好值" → "取 2 的幂这类 tile 友好值"（避免误读为幂等 idempotent）。

**机械验证：** `validate.py` 通过。Chrome 探针：48 KaTeX、0 foreignObject、0 overlap。
