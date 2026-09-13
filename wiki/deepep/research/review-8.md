<!-- review-meta
round: 8
page: wiki/deepep/index.html
reviewed_content_sha256: 3687bfa4efe5d54f
-->
# DeepEP审查记录（第 8 轮）

- 页面版本：a556c5cf084a32d273ee5f97b6d4e4790e0be38c
- 审查时间：2026-09-13 22:43 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：核心问题；1. 专家并行的 all-to-all 卡在哪（1.1 dispatch 与 combine 在搬什么 / 1.2 通用集合通信库的四个不适配 / 1.3 DeepEP 是什么 / 本章问题）；2. 高吞吐内核（2.1–2.5 / 本章问题）；3. 低延迟内核（3.1–3.5 / 本章问题）；4. V2 重构（4.1–4.4 / 本章问题）；5. DeepEP 不解决什么（5.1–5.3 / 本章问题）；来源与范围说明；overview.html

## 核对方式与来源

- 外部来源：README.md（main）、docs/legacy.md、V1 初始 commit ebfe47e 的 `deep_ep/buffer.py` 与 `csrc/kernels/internode_ll.cu`、main 的 `deep_ep/buffers/elastic.py`、V1 初始 README、DeepSeek-V3 技术报告 arXiv:2412.19437v2（用 WebFetch/curl 抓原文，逐节核对）；commit 哈希与日期用 GitHub API 核对。
- 图内数值：SVG 图（2.2 节）逐元素测量坐标与标注文字，与正文/图注对照。
- 代码：从页面提取 `language-python` 折叠块并实际执行，逐行核对"预期输出"。

## 已核对且与来源一致的要点（抽样）

- 两张实测表逐位一致：V1 normal（intranode EP8 153/158、internode EP16 43/43、EP32 58/57、EP64 51/50）与 V1 low-latency（EP8 77 µs/98、…、EP256 194 µs/39）；V2 表（SM90 CX7 EP8×2 90/81 SM12、EP8×4 61/61 SM6、SM100 726/740 SM64、643/675 SM24）与 README 完全一致；实验条件（4096 tokens/top-4 groups/top-8 experts/FP8+BF16；128 tokens；8K tokens）逐项一致。
- F1 为 V3 报告 §2.1.2 式（12），符号表与式（12）–（14）一致；§3.3.3 原文证实"quantize the activation before MoE up-projections into FP8 and then apply dispatch components / A similar strategy is applied to the activation gradient before MoE down-projections / For both the forward and backward combine components, we retain them in BF16"。
- §3.2.2 原文证实：NVLink 160 GB/s、IB 50 GB/s、3.2 倍、至多 4 节点、同 in-node index、instantaneously forwarded（"we will endeavor to ensure"）、3.2 experts/node、至多 13 experts、only 20 SMs、10 communication channels、warp specialization 三项任务、customized PTX + auto-tune chunk size。§3.5.1 原文证实 20/132 SMs、tensor cores under-utilized、SM 四项任务。§3.4.1/§3.4.2 证实 4 节点 32 GPU、40 节点 320 GPU、32 冗余专家、64 GPU、恒选共享专家→9 专家、IB 点对点 + IBGDA、每专家 ≤256 tokens、访存瓶颈、少量 SM。§3.2.1 证实 1:1 计算通信比与"both all-to-all and PP communication can be fully hidden during execution"。
- 源码核对：`buffer.py` docstring 的返回形状 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`、scales 列主序 TMA、recv_count、双缓冲、return_recv_hook、IBGDA 释义与 NIC handler=gpu；`internode_ll.cu` 第 139–141 行 `local_expert_idx*num_ranks*num_max + rank*num_max + slot_idx`、`atomic_counter_per_expert + dst_expert_idx`；`elastic.py` 第 209–211 行 hybrid = "hierarchical RDMA + NVLink … more friendly to multi-plane/multi-rail networks" 与 get_theoretical_num_sms "assumes a balanced gate distribution"、"For V3.0's group-limited gate, please do not use this function"。
- 代码实际运行输出与页面"预期输出"逐行一致（t_0=(3.6000,4.6000)、t_1=(5.0000,7.0000)、16/16 一致、32/128）。
- commit 核对：01dc3aa=2026-08-04、ebfe47e=2025-02-24（initial commit）、b306af0=2026-04-29/30（Public release）。`validate.py` 返回 success；引用的 11 个站内页面均存在。

## 问题

- [轻微·表述] 核心问题 2 解答（约 87 行）与 overview.html「关键结论与边界」第 1 条：把"含本 rank 流量的逻辑带宽"写成无标注的实测口径，而该口径是页面自己的推断。｜引文依据：legacy.md Performance 节只给出 "Bottleneck bandwidth" 表，未定义口径；V1 初始 README 同样未定义；页面 §2.5（约 331 行）自述"legacy.md 未注明口径，此为页面据数值与 NIC 峰值的推断"，§3.5 亦然，两处摘要却漏掉该限定。｜修复要求：在核心问题 2 解答与 overview 同一括号口径内补"（口径为页面推断，说明见 2.5 节）"，或从摘要中删去该口径，使其与 §2.5/§3.5 的标注一致。｜修复：｜复验：
- [轻微·格式] §3.3「固定槽位布局」（约 393 行）与 §3.3 代码折叠块「验证的机制」（约 429 行）：算式"来源 rank × num_max + 段内序号"以纯文本、Unicode `×` 与 `+` 直接书写，未用 `$...$` 包裹，既不符 style-guide §11「数学运算符…必须包在 $...$」，也与本页对另一处算式（`$4\times 3.2=12.8\approx 13$`）的处理不一致；其中 `num_max` 也未按全文他处写法给出 `<code>num_max_dispatch_tokens_per_rank</code>`。｜引文依据：internode_ll.cu 第 139–141 行偏移为 `dst_expert_local_idx*num_ranks*num_max_dispatch_tokens_per_rank + rank*num_max_dispatch_tokens_per_rank + slot_idx`，源码变量名为 `num_max_dispatch_tokens_per_rank`。｜修复要求：将该算式改为 `$...$` 形式（如 `$\text{rank}\times num\_max + \text{段内序号}$`），并把 `num_max` 统一为 `<code>num_max_dispatch_tokens_per_rank</code>` 或明确标注其为源码变量的简写。｜修复：｜复验：

## 结论

- 处置：可发布（无阻断、无重要；两条轻微为标注与公式书写的细节，不影响正确性与主线理解）
- 说明：本轮对事实性论断、公式、数字逐条回源核对，未发现定位不到、来源不支持、无条件化或推断包装成结论的情形；未发现正文/summary/overview/图注之间的数字或引文编号不一致；图内读数与正文一致；代码输出与页面描述一致；无可执行但输出不符的代码；未发现元话语、会话指代、调试叙事或临场评价。
统计：阻断 0 / 重要 0 / 轻微 2