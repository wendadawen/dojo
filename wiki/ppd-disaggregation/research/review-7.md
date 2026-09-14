<!-- review-meta
round: 7
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: c7ce2ff6abbc2feb
-->
# PPD 分离审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 9717650094ed271c6e3894ba79aac3fa69b12861（sha256 前缀 02fc67a52744e25a）
- 论文版本：arXiv:2603.13358v2（2026-05-05 修订），ICML 2026（原文取自 arxiv.org/html/2603.13358v2，与官方 abs 页信息一致：6 位作者、v1 2026-03-09 / v2 2026-05-05、ICML 2026 接收）
- 审查时间：2026-09-14 17:11
- 审查者：独立子代理（未参与写作，未参与前序轮次审查；未读取 research/）
- 已完整阅读章节：head/meta、题头 meta 块、开篇构造示例、核心问题 1–5（含各自解答折叠块）、术语表、1 多轮对话暴露 PD 分离的两个代价、2 full prefill 与 append-prefill 差一个数量级（2.1/2.2）、3 没有静态最优（3.1–3.4）、4 PPD（4.1–4.4 含两阶段伪代码与 vLLM 原型折叠块）、5 真实负载/慢网络与权重旋钮（5.1–5.5 含失败率全表与延迟注入公式折叠块）、6 方法评价（6.1–6.3）、7 附：PPD 架构概念图、来源与范围说明（核心论断原文定位表 C1–C22、核心公式 F1–F5、外部数字 N1–N21、原图对应、构造示例、简化条件）、全部六张图与图注、全部「本章问题」折叠块

## 核对说明（回源方法与结论）

本轮逐条把页面标注回源到 arXiv:2603.13358v2 原文，核对到具体章节/公式/表/图并在下列问题中给出原文片段或关键数值。数值核对结果（全部与原文一致）：

- §2.2 单向协议原文：“P nodes act as KV producers and D nodes as consumers, with no reverse channel from D back to P.”（对应 C1）；2K 上下文“each transfer is ∼256 MB”（N14）。
- §4.1 干扰：原文“Full prefill causes ∼48% slowdown at batch size 200; append-prefill causes only ∼2%”；4 并发“full reaches +57% while append stays at +21%”；上下文“3–4× at 32K tokens while append-prefill stays below 25% even at 64K”（N1/N2/N3）。与 img-06（原文 Figure 2）像素核对：蓝 decoding-only、红 +48% 标注、绿 +2% 标注均吻合。
- §4.2：原文“we explore 17 configurations, including 7 hybrid configurations … we concentrate on 10 core configurations”；“18 synthetic workloads”由“2 Turn 1 … 9 Turn 2”构成；10 个 QPS 档“0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20”；“17×18×10=3,060 data points”；每点“fixed 10-second duration … Poisson process”（N4/N18）。
- Table 1 逐格核对（页面表 1 完全一致）：1P_3D −57.8%/−65.2%/−73.3%；2P_2D −47.7%/−51.6%/−56.2%；3P_1D −44.3%/−38.1%/−24.9%；列头“Low QPS (0.5–2) / Med QPS (4–8) / High QPS (12–20)”与页面一致。
- §4.4/Table 2 逐格核对（页面 winner 表完全一致）：Replica 63.3/0.6/0/21.3；x=0 0/38.3/4.4/14.2；0<x<1 3.3/33.3/27.8/21.5；x=1 27.2/15.6/38.3/27.0；原文“92.2% of workload-QPS combinations have different optimal configurations…”（N6/N7）。
- §4.3/App.C.3：原文“stable ∼70% Turn 2+ TTFT improvement across 2–16 turns and three model sizes (8B, 14B, 30B)”（N17/C20）。
- §6.2：原文“For the stable 1P_3D configuration on ShareGPT, PPD reduces average query latency by 15–25%”；“at the observed average of 3.1 turns per conversation”；“cutting KV transfer load by ∼75% … a ∼3× difference in network load”；“PPD-enabled configurations achieve 100% success rate”（N8/N9/N19/C15）。
- §6.3/Table 3 逐格核对（页面表完全一致）：x=0 10/0/4；x=1 5/13/27；PPD 12/14/27；“27 test points (3 configs × 9 QPS, WildChat)”（N10）。
- §6.4 Figure 5：原文“PD … rises from 143.7 ms to 170.6 ms (+18.7%)”“PPD remains flat at ∼51 ms”“the relative TTFT reduction widens from 64% to 70%”；右子图 3,028→3,169 ms（+4.7%）/PPD +0.9%（N11）。与 img-03 像素核对：三子图数值点吻合。
- §6.5/Figure 6：原文图注“static PD (0% D-local); w_tpot=6 (20%); w_tpot=3 (50%); balanced w_tpot=1 (95%)”“94–96% TTFT reduction, 7–12% TPOT degradation”；img-02 图内标题为“QPS = 8 (TTFT −96%, TPOT +7.0%)”“QPS = 16 (TTFT −94%, TPOT +12.1%)”（N12）。
- 附录 A：原文“winning only 6.1% of TTFT, 12.2% of TPOT, and 16.1% of throughput test points”（C19）。附录 B.6：“per-token footprint is 128 KiB”；“5,115 tokens … ∼670 MB, about ∼4.5 ms over NVLink, ∼27 ms over InfiniBand HDR (25 GB/s), and ∼67 ms over 100GbE”；Eq.(2) Δt=max(0,B(ψ)/β_target−t_NVLink)（N15/F3/C21）。附录 B.2–B.5：kv_role=kv_producer/kv_consumer、ZeroMQ、prefix cache、session_table[conv_hash]、MD5 of the first user message、60 分钟逐出、10s 心跳/30s 移除、Quart（C13）。附录 C.3：server restart、17 配置 × 180 workload-QPS（N17）。附录 C.4/Table 5 逐格核对（页面折叠表完全一致）：3P_1D 11/44/61/89 与 6/22/44/67；2P_2D 0/6/11/22 与 0/0/6/11；4R 全 0（N16）。附录 C.5：30s→10s 阈值结论、“13/27 test points”失败（N21）。
- §5/§8：两阶段离线建表 + 在线 <1 ms、Turn 1 恒 x=0、P:D 比例与 w 解耦（C6/C10/C14）；§8“cutting Turn 2+ TTFT by an average of ∼68% on real-world workloads … adding <1 ms per-request overhead”（N13/N20）。
- 页面自造算式复算无误：C_full=1250²=1,562,500、C_append=50×1250=62,500、比值 25 与 n/m=1200/50=24 同量级；1250×128 KiB=156.25 MiB；2048×128 KiB=256 MiB；5115×128 KiB≈670.6 MB；F4 的 2×8×128×2×32=131,072 B=128 KiB。
- 机械项：validate.py 返回 “validation ok”；dojo:topics 四项（推理系统/并行与通信/内存与缓存/模型结构）均在 ALLOWED_TOPICS 内，dojo:tag「推理系统」在 ALLOWED_TAGS 内；四处前置概念链接（standard-attention / mqa-gqa / gpu-communication / prefix-caching）对应目录与 index.html 均存在；六张图 alt 无 `$...$`；行内 SVG 为 HTML 结构、无等宽字符框线、无 ASCII 公式；全页无 Unicode 数学字符；代码块为算法伪代码，页面未声称可运行，故按静态审查处理（无 Python 执行项）。

## 问题

- [重要·技术] §7 图注末句（index.html 第 608 行）：图注称「PD 与 PPD 两幅中"Last Turn's KV"（红虚线）始终在 D 上，对 P 不可达」，但该句描述的对象 img-01.webp（原文 Figure 3）里，标注 "Last Turn's KV" 的红线其箭头终止于 P 节点（方向 D→P），且 PD 一幅中该线为实线而非虚线——图注对图的描述与图本身方向相反，读者对照图注会得到相反印象｜引文依据：论文正文、图注与 HTML 文本均不含 "Last Turn's KV"（原文 grep 无命中），该标签只存在于 Figure 3 图内；对 assets/img-01.webp 逐像素核对，PD 幅红线在 x≈640–658、y≈46–66 处收束为顶点朝下的三角箭头落在 P 顶部，PPD 幅（x≈1278–1290、y≈42–53）同样收束于 P 顶部，两幅在 D 一侧均无箭头｜修复要求：改写该句，使其只陈述与图一致的观察（例如说明图中该红线在 PD/PPD 两幅均画成指向 P 的箭头），或删除对图的指认、仅保留已有来源支撑的协议事实（KV 传输单向 P→D、D 上 KV 对 P 不可达，见 C1）｜修复：｜复验：

- [轻微·技术] §5.3 图注前正文（第 477 行）、本章问题解答（第 539 行）及 overview.html 第 44 行：写 NVLink 有效带宽 “~150 GB/s”，而所引 Figure 5（img-03.webp）横轴刻度标注为 “NVLink (~200 GB/s)”，页面文字与同页图的刻度不符｜引文依据：论文 Figure 5 图注文本为 “NVLink (~150 GB/s effective, intra-node)”，而同一张图内横轴标签为 “~200 GB/s”——原文自身图注与图不一致，页面采用了图注文字值｜修复要求：与图保持一致（改用 ~200 GB/s，或两值并列并注明图注文字为 ~150 GB/s），使读者对照横轴时不再产生疑问｜修复：｜复验：

- [轻微·格式] 「核心论断与原文定位」表与「原图与原文对应」列表：来源编号存在缺口与定义未用项——C 编号自 C10 跳至 C13（C11、C12 在表中与正文中均不存在）；C20 在表中定义但正文未引用（该论断在正文用 [N17]）；G1 在「原图与原文对应」中定义为 Fig.2，但该图在正文（第 218 行）的引用是 [N1]，[G1] 全页未出现（G2–G6 均被正文引用）｜引文依据：不适用（页内机械项）｜修复要求：补齐或重排 C 编号使连续（或在说明中写明缺口原因）；为 C20 对应的正文位置补 [C20]，或从表中删除该行；把 G1 与正文 [N1] 的标注统一（二者指同一张 Figure 2）｜修复：｜复验：

- [轻微·可读性] §3.2 Figure 1 图注末句（第 261 行）：「PPD 落在合理的右上/左上边界上」——该图横轴为 TTFT（越大越差）、纵轴为 TPS（越大越好），理想方向是左上（低 TTFT、高 TPS），「右上」与理想区方向相反，两个方向并列造成同句自相矛盾｜引文依据：不适用｜修复要求：收敛为单一方向表述（如「落在左上（低 TTFT、高 TPS）边界上」），去掉「右上」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复。全部事实性论断、公式、数字与实验条件经回源核对与原文一致（含 Table 1/2/3/5 逐格核对与六张图的像素读数核对），无阻断问题；1 项重要问题（Figure 3 图注对图的描述与图不符）需修复后复验，3 项轻微问题可一并处理。
