<!-- review-meta
round: 6
page: wiki/dualpath/index.html
reviewed_content_sha256: 63dae7202b85aec7
-->
# DualPath 审查记录（第 6 轮）

- 页面版本：b9b17d4d04af5f53064a8763e54464564fca88d8（工作树 wiki/dualpath/index.html）
- 论文版本：arXiv:2602.21548v2（2026-02-26）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 + 贯穿示例 → 1. agentic 推理的存储 I/O 瓶颈由三因素叠加（1.1 工作负载 / 1.2 硬件代际 / 1.3 PD 分离架构 / 本章问题）→ 2. 双路径数据流与块布局（2.1 总览 / 2.2 PE read path / 2.3 DE read path / 2.4 buffer 与块布局 / 本章问题）→ 3. P/D 区间（3.1 记号与假设 / 3.2 四个不等式 / 3.3 汇总区间 / 本章问题）→ 4. CNIC-centric 流量管理（4.1–4.5 / 本章问题）→ 5. Adaptive Request Scheduler（5.1–5.6 / 本章问题）→ 6. 实验（6.1–6.8 / 本章问题）→ 7. 方法评价（7.1–7.4 / 本章问题）→ 来源与范围说明。含全部折叠块、全部 14 张原图图注与 alt、overview.html 交叉核对。
- 机械核对：`.dojo/scripts/validate.py wiki/dualpath/index.html` 返回 `validation ok`；alt 属性中无 `$...$`；页面内嵌 MathJax/KaTeX 行内与展示公式均可渲染。
- 回源方式：arXiv:2602.21548v2 的 HTML 全文与 PDF 全文（pdftotext 抽取）双份互证，并逐张读取本页 assets/ 下的原图（img-01…img-14）核对图注数字。

## 问题

- [阻断·技术] §4.1 集合通信段（index.html:341）："GPU 目前不支持 PCIe QoS（论文 §5 引言段引用 7723586）"——论文中不存在编号 7723586 的引用，属定位不到、来源不支持的引文标识｜引文依据：论文 §5 引言段原文 "…(2) existing GPUs do not support PCIe QoS [39]…"；PDF 参考文献 [39] = "Andre Richter, Christian Herber, Thomas Wild, and Andreas Herkersdorf. 2016. Resolving Performance Interference in SR-IOV Setups with PCIe Quality-of-Service Extensions. In 2016 Euromicro Conference on Digital System Design (DSD). 454–462. https://doi.org/10.1109/DSD.2016.41"；对 HTML 全文与 PDF 全文检索字符串 "7723586" 均 0 命中（论文其余处亦无该编号）｜修复要求：删除 "7723586"，改为论文实际引用（如"论文 §5 引言段引用 Richter et al. 2016（参考文献 [39]）"），不得保留任何论文中不存在的编号｜修复：｜复验：
- [重要·技术] §2.4 PE/DE buffer 段（index.html:218）："论文 §4.1「Prefill PE read path」末段解释：…"——出处段名错误，且句尾把页面推断放进"论文…解释："框架内｜引文依据：论文 §4.1 的 "Prefill PE read path" 段全文只有三步搬运描述与 "This process (3-7) repeats n_layer times"，无任何 TTFT/buffer 论述；TTFT 与 DE buffer 的论述位于 §4.1 的 "Decode Phase" 段："The design of DE buffer imposes bandwidth pressure on DRAM and CNIC (an extra H2D), which could be avoided by directly bypassing it via GPU Direct RDMA. However, since the generation length is typically short in this scenario, time-to-first-token (TTFT) accounts for a non-negligible portion of the total end-to-end request time. Introducing DE buffer helps reduce GPU memory usage." 该段并无"让 H2D/D2H 也走 CNIC 是第五章流量隔离方案的统一设计"的表述（该事实出自 §5 引言段，与本段动机解释无关）。同页核心问题 2 的解答（index.html:84）又标为"见 §4.1 Decode Phase 段"，前后自相矛盾｜修复要求：出处统一改为"§4.1 Decode Phase 段"；"让 H2D/D2H 也走 CNIC 是第五章流量隔离方案的统一设计"须移出"论文…解释"框架，改写为页面推断并显式标注，或删除｜修复：｜复验：
- [轻微·表述] §7.6 大规模段（index.html:664）："需要说明：论文 §7.6 明确写…"——元话语（check.md §2.2 第 14 条禁止的"需要注意的是"类）｜引文依据：不适用｜修复要求：删去"需要说明："，直接引出论文原文引述｜修复：｜复验：
- [轻微·表述] 贯穿示例 callout（index.html:112）与表 1 图注（index.html:138）："本页用 32721 hit + 429 miss 作为代表数字"、"本页以 V3.2 660B 的 22 GB/PFLOP…作为引用依据"——以"本页"为主语的自我指代｜引文依据：不适用｜修复要求：改为不出现"本页"主语的表述（如"该代表数字取 32721 hit + 429 miss"、"引用依据取 V3.2 660B 的 22 GB/PFLOP"）｜修复：｜复验：
- [轻微·技术] §1.2 图注（index.html:145）："5→10 仍有约 16% 提升（读图估计，读自 Fig.3 右）"——读图值与原图不符｜引文依据：对 assets/img-14.webp 右侧子图做像素量测：水平网格线 1x=418、2x=292、3x=165（≈126.5 px/单位），曲线在 batch 5 ≈ 2.59×、batch 10 ≈ 2.92×，相对提升 ≈ 12.5%（论文 Figure 3 右侧同图），非 16%｜修复要求：把"约 16%"改为按图量测值（约 12–13%）或删去该括注｜修复：｜复验：
- [轻微·技术] §2 本章问题解答（index.html:226）："论文 §2 background 明确把 layerwise prefill 与 PD 分离并列为「广泛采用的两项技术」"——出处章节错误｜引文依据：该表述在论文 §4 DualPath System Overview："DualPath adopts two widely-adopted techniques demonstrated in §2: (1) PD Disaggregation … (2) Layerwise prefill …"；论文 §2 标题为 "Background"，无该并列表述｜修复要求：出处改为"§4 System Overview（并回头看 §2）"或"论文 §4"｜修复：｜复验：
- [轻微·技术] §6.5 Figure 11 图注（index.html:617）："DS 660B 上有 SGL(MC) 数据点（论文图上标注为 N/A 或在低 APS 区间）"——"标注为 N/A"与图不符，会误导｜引文依据：论文 Figure 11（assets/img-04.webp）中 SGL(MC) 以数据点绘出在 APS≈0.01–0.06 区间，无 N/A 标注；"N/A"只出现在 Figure 7（"N/A for running into an error before finishing"）｜修复要求：改为"DS 660B 上有 SGL(MC) 数据点，绘于低 APS（≈0.01–0.06）区间"，删去"标注为 N/A"｜修复：｜复验：
- [轻微·技术] §6.5 工作集段（index.html:612）："机时与存储成本 r³ 倍"——把机时的倍率也说成 r³｜引文依据：论文 §8.2 "Such experiments require r times more machine hours and r² times more storage (cost scaling as r³), which we cannot afford given limited resources."（机时 r 倍、存储 r² 倍，只有合计成本按 r³）｜修复要求：改为"机时 r 倍、存储 r² 倍，合计成本按 r³ 增长"｜修复：｜复验：

## 结论

- 已核对一致的关键项（本轮无问题，供复验参考）：Table 1 五行比值与 22 GB/PFLOP 代表值、Table 2 三行统计（Turns/Append/Gen/Total/Context）、Table 3 与 §7.2 testbed、Fig.3 左 28.8×/2.0×/2.4× 与 14.4×、四组不等式 F-1…F-12 的代数复算（含 2Bs/g、Bs/g·(1+D/P)、s/g·(P/D+2)、(3+2P/D)Bs、(M/Bs−3)/2 与 (g=8,s=1,M≈500,Bs≈50) → 1/7 ≤ P/D ≤ 7/2）、§4.2 原文 s ≤ g 与 eq.(1) 不一致的笔误照录、§A.5 两种块与 trie、§A.1 的 qos_max_vls 4 / qos_high_limit 240 / vlarb_high 0:192,1:192,2:0,3:192 / vlarb_low 0:192,1:192,2:64,3:192、§5.1 "approximately 99%" 与 240/255≈94% 的反推、RoCE 四 lossless TC 与 Ultra Ethernet/UnifiedBus、§A.4 α=3s/β=5s/compute quota 300ms、调度器 <10 cores、Algorithm 1 伪代码逐行、§7.3 各加速比（1.87×/1.78×/1.09–1.85×/1.82–1.99×/1.64×/2.46×）、§7.4 APS 1.67×/2.25× 与 SLO、§7.5 17.21%/38.19%/45.62% 与差分 20.98/7.43 个百分点、§7.6 3167s vs 3201s 与 44P88D 22×、§8.2 69 GB→681 GB 与 r/r²/r³、§9 的 Mooncake/Strata/KVPR/TailorKV/LayerKV/PrefillOnly/DistServe/Splitwise 会议归属、Fig.8 三对等效与 Fig.9 append 3× 约 1.85× 的读图值。
- 统计：阻断 1 / 重要 1 / 轻微 6
- 处置：修复（阻断与重要问题须关闭后重跑；轻微项一并处理）
