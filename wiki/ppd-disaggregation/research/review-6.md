<!-- review-meta
round: 6
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: e350b5165089f5dc
-->
# Not All Prefills Are Equal（PPD 分离）审查记录（第 6 轮）

- 页面版本：b9291c9a61f68880f4de8d20d0d04da4ce0cef02（index.html 工作树哈希）
- 论文版本：arXiv:2603.13358v2（ICML 2026，2026-05-05 修订；页头标注 ICML 2026 与 v2 日期与 arXiv 一致）
- 审查时间：2026-09-13 21:20
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题（5 题）；术语表；1 多轮对话暴露 PD 分离的两个代价；2 full prefill 与 append-prefill 差一个数量级；3 没有静态最优：3060 个数据点的扫描证据；4 PPD：把路由变成带权重的逐请求决策；5 真实负载、慢网络与权重旋钮下的表现；6 方法评价；7 附：PPD 架构概念图；来源与范围说明（含全部折叠块、图注、脚注锚点）
- 来源获取：论文原文 arxiv.org/html/2603.13358v2 全文（含 App.A–C）、论文原图 figures/*.png、页面 assets/img-01..06.webp、.validate.py

## 问题

- [轻微·图示] §7 Fig.3 图注（line 608）：把 PD 面板的 KV 通道写成「T1 KV 与 T2+ KV 都走 P→D 通道（红色实线）」，与页面所用原图不符——原图中 T1 KV 为红色实线、T2+ KV 为红色虚线｜引文依据：页面 assets/img-01.webp（论文 Fig.3）PD 面板可见粗实线 T1 KV 与点划线 T2+ KV，T2+ KV 线型与「Last Turn's KV」同为虚线｜修复要求：改为「T1 KV（红实线）与 T2+ KV（红虚线）都走 P→D 通道」，与同句后文「Last Turn's KV（红虚线）」的写法统一｜修复：｜复验：

- [轻微·图示] §3.2 Fig.1 图注（line 261）：「QPS=0.5 时两组曲线在原点附近重合」与图不符——该子图 TTFT 横轴是对数轴（10³–10⁵ ms），两组曲线分布在 TPS≈24–100、TTFT≈2×10³–3×10³ ms 的中段，不经过原点｜引文依据：论文 figures/fig1_pareto.png 左图标题 QPS = 0.5，横轴 TTFT P99 从 10³ ms 起，蓝橙两线在 TPS 50–70 段贴近｜修复要求：改为「QPS=0.5 时两组曲线在 TPS 50–70 段几乎重合」，删去「原点附近」｜修复：｜复验：

- [轻微·图示] §2.2 Fig.2 图注（line 214）：「蓝线（decoding-only）与绿线（+1 append-prefill）几乎重合，全程偏差不超过 2%」对图做了强于原图的定量断言——原图中绿线在 batch≈70–180 全程位于蓝线之上，中段差距已超过 2%｜引文依据：论文 figures/interference_tpot.png，green（decoding + 1 append-prefill）在 batch 100–150 高于 blue（decoding-only）约 1–2 ms，原图仅在 batch 200 处标注绿线 +2%｜修复要求：删去「全程偏差不超过 2%」，改为「蓝线略低于绿线（原图在 batch 200 处标注 +2%）」｜修复：｜复验：

- [轻微·格式] §7 正文（line 606）：「根据 SLO 权重、负载估计、节点配动态决定」缺字｜引文依据：不适用｜修复要求：改为「节点配置」（与同段图注中 Node Config 一致）｜修复：｜复验：

- [轻微·格式] 来源与范围说明 F4（line 645）：「验证 … 与 §2.2 §5.3 的 ~256 MB /WildChat ~670 MB 数字自洽」在标注论文定位的公式表内混用了论文节号与页面节号：~256 MB 出自论文 §2.2，~670 MB 出自论文 App.B.6（页面 §5.3 才是带宽模拟节）｜引文依据：论文 §2.2「For Llama-3.1-8B with a 2K-token context, each transfer is ∼256 MB」；App.B.6「each Turn 2+ transfer in the x=0 path moves ∼670 MB」｜修复要求：改为「与论文 §2.2 的 ~256 MB、App.B.6 的 ~670 MB 数字自洽」，或显式注明哪一个是页面节号｜修复：｜复验：

## 本轮已回源核对、未发现问题的事实与数字

- N1/N2/N3：48% vs 2% @batch 200（论文 §1 贡献(ii)、§4.1、Fig.2）；4 并发 +57% vs +21%（C.1 Fig.7）；full 32K 时 3–4×、append 64K 时 <25%（C.1 Fig.8）——逐条一致
- N5/表 1：1P_3D −57.8/−65.2/−73.3、2P_2D −47.7/−51.6/−56.2、3P_1D −44.3/−38.1/−24.9，与论文 Table 1 逐格一致；「48–73%」与 §4.3 原文一致，页面「48–73% 来自 P 稀缺配置、3P_1D 为反例」的读法与 §4.3 原文相符
- N6（92.2%）、N7/表 2（Replica 63.3/0.6/0/21.3；x=0 0/38.3/4.4/14.2；0<x<1 3.3/33.3/27.8/21.5；x=1 27.2/15.6/38.3/27.0）与论文 Table 2 逐格一致；各列合计<100% 系论文图注「hybrid 配置排除」，页面未把它写成合计 100
- N10/表 3（WildChat 27 点：TPOT 10/5/12、TTFT 0/13/14、SR 4/27/27）与论文 Table 3 一致；「PPD 是唯一三项全竞争」与 §6.3 一致
- N11：143.7→170.6 ms、+18.7%、PPD ~51 ms、E2E 3028→3169 ms（+4.7%/+0.9%）、64%→70%、150/50/25/10 GB/s、1P_3D×QPS=1×WildChat 500 会话，与论文 Fig.5 图注逐项一致
- N12：w_tpot=1/3/6 → 95%/50%/20% D-local、x=1 端 94–96% TTFT 降 / 7–12% TPOT 退化，与 §6.5 一致；Fig.6 两个子图标题 −96%/+7.0%（QPS=8）与 −94%/+12.1%（QPS=16）与页面图注逐字一致
- N13：~68%（abstract 与 §8 结论「on real-world workloads」均支持，页面归为真实负载无误）；N19：3.1 轮/会话、~75% KV 削减、~3× 网络负载差（§6.2）；N15：128 KiB/token、5115 token→670 MB、4.5/27/67 ms（App.B.6）；N16：表 5 失败率 11/44/61/89、6/22/44/67、0/6/11/22、0/0/6/11、4R 全 0（App.C.4）；N14：2K context→~256 MB（§2.2）；N17/C20：2–16 轮、8B/14B/30B 上稳定 ~70%（§4.3 原文即「stable ∼70% … across 2–16 turns and three model sizes (8B, 14B, 30B)」）；N18：10 QPS 档 (0.5/1/2/4/6/8/10/12/16/20)、18 负载=2 Turn1×9 Turn2、decode-heavy 4/balanced 2/prefill-heavy 3（§4.2）；N20：<1 ms（§5、§8）；N21：30s→10s（App.C.5）——全部一致
- C1/C3/C4（单向 P→D、无反向通道、重算占多轮 prefill 成本达 99%，引 Gao et al. 2024）、C5/F2（O((n+m)²) vs O(m(n+m))、m≪n 时约 n/m 倍）、C6/C7/C8/C9、C13（kv_producer/kv_consumer、ZeroMQ、Quart、session_table 三字段、MD5 首消息、60 min TTL、10s 心跳/30s 移除，App.B.2–B.5）、C14/C15/C16/C17/C18/C19（hybrid 6.1%/12.2%/16.1%，App.A）、C21（Eq.2 延迟注入式）、C22（balanced 下 TTFT/TPOT 在 E2E 中抵消，App.C.5）——逐条与原文一致
- 复算：F1 打分函数与三个代入示例（S=0.52 / 0.12 / −0.20 / 0.15 / −0.30）、§2.1 复杂度例（1250²=1,562,500；50×1250=62,500；比 25）、F4 构成（2B×8×128×2×32=131072 B=128 KiB）、F5（2048×128 KiB=256 MiB）、构造示例 1250×128 KiB≈156 MiB——全部正确
- 页面链接 ../standard-attention、../mqa-gqa、../gpu-communication、../prefix-caching 均真实存在；无「待生成」占位；alt 内无 $…$；validate.py 返回 validation ok；无会话指代（我/我们/你）、无「本页」自我指代、无元话语模板句；「场景」均为自然用词未作术语

## 结论

- 处置：修复（仅 5 项轻微，无阻断、无重要；修完轻微项即可发布）
- 统计：阻断 0 / 重要 0 / 轻微 5