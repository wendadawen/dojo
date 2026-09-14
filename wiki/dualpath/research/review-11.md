<!-- review-meta
round: 11
page: wiki/dualpath/index.html
reviewed_content_sha256: 4715de9bd5ee4bbf
-->
# DualPath 审查记录（第 11 轮）

- 页面版本：wiki/dualpath/index.html，sha256 28833d15842adc348c9154280c23a3db9fb081539f6645918cf09d12917b138d（工作树与 HEAD a5f9073 一致，页面无未提交改动）
- 论文版本：arXiv:2602.21548v2（2026-02-26，cs.DC）。核对了两份渲染：arXiv HTML v2（LaTeXML）与 arXiv PDF v2（13 页）。页面对「参考文献 [39]」的编号引用只存在于 PDF 版（LaTeXML 版为作者-年份制），故一切编号类引用按 PDF v2 逐条核对。
- 审查时间：2026-09-14 17:42
- 审查者：独立子代理（未参与写作，未参与前序 10 轮审查与修复；未读取 research/ 下任何文件，仅依据本页与来源）
- 已完整阅读章节：核心问题（5 题含解答）、§1 1.1/1.2/1.3/本章问题、§2 2.1/2.2/2.3/2.4/本章问题、§3 3.1/3.2/3.3/本章问题、§4 4.1/4.2/4.3/4.4/4.5/本章问题、§5 5.1/5.2/5.3/5.4/5.5/5.6/本章问题、§6 6.1–6.8/本章问题、§7 7.1–7.4/本章问题、来源与范围说明（含全部折叠块与 14 张内嵌原图）

## 问题

- [轻微·原图] §6.5「图（原文 Figure 12）」左图注：称「Basic 的 Sch. + A. 段在 APS 0.25 时占比绝大部分（约 70-80%，Sch. 约 70%、A. 约 3-5%（读图估计））」；但图中 APS 0.25 组第二根柱（Basic）只有 Sch./R./PF. 三段，无 A. 段。像素测量（img-05.webp：0% 在 y=367、100% 在 y=80，287 px/100%）：Basic 柱顶 y=157 → Sch. = (367−157)/287 ≈ 73%，其上是 R.（y=108–155）与 PF.（y=77–105），Sch. 顶与 R. 底之间仅一条 2 px 分隔线（≈0.7%），没有橙色 A. 色带。A. 段只出现在每组第一根柱（DualPath）：APS 0.25 时 A. = (306−288)/287 ≈ 6%。｜引文依据：原文 Figure 12 图注 "In each pair of pillars, the first is for DualPath and the second is for Basic."；原文 §7.4 "Basic's queuing time grows dramatically due to insufficient storage bandwidth"（与第二根柱 Sch. 随 APS 增长 21%→40%→73% 吻合）。｜修复要求：把「A. 约 3-5%」改为 Basic 柱实测值（约 0-1%），或明确该百分比是 DualPath 柱的 A. 占比；「Sch. + A. 合计约 70-80%」保留（Basic 实测 73%）。｜修复：｜复验：

- [轻微·原图] §6.4「图（原文 Figure 8）」图注：「三列分别是 MaxLen 32K/48K/64K，每列内分 1P1D / 2P1D / 1P2D 三种 P/D 比，每组三根柱（512/1024/2048 agents）」。「每组三根柱」与图不符：每个 #Agents 组实为 6 根柱。像素测量（img-07.png，左面板 y=488 行的柱边界）：512 组为 x=191–228、229–266、267–304、305–343、344–381、382–420，共 6 根等宽（≈37 px）柱，即 Ours 与 Ours(basic) 两色 × 2p1d（斜纹）/1p2d（点纹）/1p1d（实心）三纹样。｜引文依据：原文 §7.3 "Basic 1P1D and Basic 1P2D perform comparably; so do DualPath 1P1D and Basic 2P1D, as well as DualPath 2P1D and DualPath 1P2D."（共 6 个被测配置）；Figure 8 图例 5 项（Ours / Ours(basic) / 1p1d / 1p2d / 2p1d，颜色×纹样组合出 6 柱）。｜修复要求：把「每组三根柱」改为「每个 agents 组六根柱（Ours 与 Ours(basic) 各含 1P1D/2P1D/1P2D）」，并去掉挂在「三根柱」后的「（512/1024/2048 agents）」歧义括号（512/1024/2048 是横轴分组，不是柱数）。｜修复：｜复验：

- [轻微·格式] 正文第 185 行与第 764 行：同一算式在正文中以 Unicode 减号 U+2212 直接书写（「命中率 1 − 429/32721 $\approx 98.7\%$」），而第 170 行同一算式写成 KaTeX（`$1 - 429/32721 \approx 98.7\%$`）。check.md §2.2 第 11 条要求「正文…无 Unicode 数学字符直接出现；同一变量全页写法一致」。｜引文依据：不适用（格式项）。｜修复要求：把第 185、764 行的「1 − 429/32721」改为 `$1 - 429/32721$`，与第 170 行写法统一。｜修复：｜复验：

## 已核对且未发现问题

【论断与数字，均回 PDF v2 原文核对】

- 摘要 1.87× / 1.96×：PDF "improves offline inference throughput by up to 1.87× … online serving throughput by an average factor of 1.96×"，与页面 description、§6.2、§6.5、§7.4、§7 一致。
- 命中率 98.7%：PDF §3 "mean number of rounds is 157 … average context length is 32.7k, while the append length mean is only 429, which means a KV-Cache hit rate of 98.7%"。页面四处（§1.1、§2.1、本章问题、来源说明）口径一致；1 − 429/32721 ≈ 98.69% 可复算。
- cache-compute ratio：PDF §3 "approximately 22 GB/PFLOP for DeepSeek-V3.2"；Table 1 五行 117-267 / 47-95 / 39-60 / 13-36 / 4.8-5.8 与页面表逐格一致。
- Table 2：32K(60/608/148/28639/17183)、48K(106/474/172/42607/25120)、64K(157/429/176/55958/32721) 与页面逐格一致。
- 硬件代际：PDF §3 "the I/O-compute ratio decreases by 14.4×"；Fig.3 图内框 "GPU Compute: 28.8x / PCIe Bandwidth: 2.0x / GPU Memory: 2.4x"（读自 img-14.webp），28.8/2.0 = 14.4 自洽。
- Fig.1（img-13.webp）：可见 "40% GPU util."（左 Prefill）、"100% util."（存储箭头）、右图 "80% GPU util."，页面 §1.3 图注与之相符。
- P/D 区间：§4.2 eq(1) 2×T_p×Dg = 2Bs/g ≤ B；eq(3) P/D ≥ s/(g-s)；eq(5) P/D ≤ (g-2s)/s；eq(7) P/D ≤ (g-s)/(2s)；eq(8) P/D ≤ (M/Bs-3)/2；eq(9) 汇总；"For (g=8,s=1) with M≈500 GB/s and Bs≈50 GB/s, the bottleneck-free range is 1/7 ≤ P/D ≤ 7/2"。页面 F-3–F-12 逐条对应，四组不等式代入 g=8,s=1,M/Bs=10 复算得下界 1/7、上界 {6, 3.5, 3.5} → 3.5，与页面 §3.3 的 1/7 ≤ P/D ≤ 7/2 一致。
- §4.2 原文笔误：PDF "Read operations include PE paths (3) and (5), with total traffic … 2Bs/g ≤ B. Since s ≤ g always holds in practice…"。页面 §3.2 写明原文照录该宽松条件、指出与 eq(1) 不一致、疑为笔误、不据此改写结论，处理正确（未把来源缺陷包装成自己的结论）。
- 记号：PDF §4.2 "each with one compute NIC of bandwidth B. The storage bandwidth per machine is s×B (shared by all engines on that machine)"，与页面 §3.1 的 B、sB 定义一致；§7.2 "eight 400Gbps RDMA NICs … one additional storage NIC" 与 §2.3 "a storage NIC (SNIC) up to 400 Gbps" 支持 s=1、Bs≈50 GB/s。
- §4.1 步骤范围：PDF "(3-7) repeats n_layer times"（PE 路径）与 "read from the DE buffer, also overlapping with computation (3-5). This process repeats n_layer times."（DE 路径）；"H2D transfers (Label 8 and 9 in 4(a); Label 6 and 7 in 4(b))"。页面 §2.2/§2.3 图注的步骤编号与重复范围逐个吻合；DE 路径 per-layer 的 miss-KV 回写「原图未单独标号」经放大 img-11.webp（PE CNIC 与 DE CNIC 之间只有一条带 (4) 的左向细箭头）确认属实。
- 块布局：§A.5 "Layer Block is a byte tensor with shape [1, tokens, bytes] … a Full Block has shape [layer, tokens, bytes] … simply concatenating n Layer Blocks to yield a Full Block … using a trie structure, where each tree node corresponds to a Full Block"，与页面 §2.4 一致。
- 调度：Appendix "α … set to the number of tokens we can read during 3 seconds, and … β … the number of tokens one GPU can process for 5 seconds"；"Compute Quota Threshold is set to 300ms"；Algorithm 1 的 C1/C2/C3 定义、argmin_{C2}/argmin_{C3} 降级、两类皆空则终止并归还 Leader Engine；Z = 1.05×(Σ_{r∈R}len_r + Σ_{e∈E}tok_e)/|E|、tok_e+len(r)>Z。页面 §5.1–§5.5 与 Algorithm 1 伪代码逐条一致（伪代码在「来源与范围说明·构造示例」已声明为照搬论文、非可运行代码，未按可运行代码对待，正确）。
- 消融：PDF §7.5 "adding layerwise prefill reduces JCT by 17.21% … adding Dual-path loading … by 38.19% … scheduling … by 45.62%"；页面 §6.6 与本章问题的差分 20.98 / 7.43 可复算。
- 负载均衡：PDF §7.5 "improves load balance from 1.53 to 1.18 compared to round robin"、"as low as 1.06 during the first 5% of the task"；img-03.webp 图内标注 1.528 / 1.184 与之一致。
- 在线：PDF §7.4 "(1.67× for DS 27B, 2.25× for DS 660B)"、SLO "TTFT ≤ 4 seconds and TPOT ≤ 50ms"；§7.3 "average speedup of 1.64× across all configurations (up to 2.46×)"、"1.82 − 1.99×"、"1.78× … 1.09–1.85× slower than Oracle"、"up to 1.87× over Basic"。页面 §6.3–§6.5、§7 全部一致。
- 大规模：PDF Table 3 与 §7.6 (3,167s / 3,201s；TTFT 1.739s→1.847s、TTST 0.228s→0.194s、TPOT 0.039s→0.036s；22× throughput；"scheduler CPU usage remains below 10 cores"；"do not demonstrate additional JCT or serving capacity gains compared to multiple small-scale units with equivalent cost")，与页面 §6.8、§7.1、§7.2 一致。
- 工作集：PDF §8.2 "ranges from 69 GB at APS 0.1 to 681 GB at APS 0.45"、r / r² / r³ 传导、"Such experiments require r times more machine hours and r² times more storage (cost scaling as r³)"。页面 §6.5 与其一致，且把 r³ 明确限定为「复现该实验的机时+存储开销 scaling，不指部署成本」，未越界。
- 图注读数（像素测量）：
  · Fig.3 右（img-14.webp）：y 轴 4x@y=38.5、3x@165、2x@292、1x@418.5（126.5 px/单位），x 轴 ≈18.5 px/单位。测得 batch1≈1.0x、batch5≈2.58x、batch10≈2.92x、batch20≈3.01x → 5→10 = 13.2%，落在页面所写「约 12-13%」内；10→20 仅 3%，与「接近饱和」一致。
  · Fig.7（img-09.png）中排 DS 660B 64K 面板：以 gridline 966=0、870=5000、774=10000、683=17500、636=20000 标定，测得 2048 agents 组 Ours(basic)/Ours ≈ 10052/5365 = 1.87、4096 组 ≈ 19309/10312 = 1.87，故页面 §6.2「最高 1.87× 取自该配置的 2048 agents 面板」成立。
  · Fig.11（img-04.webp）：SGL(MC) 菱形簇 x=190–263，按 0.01@200.5、0.05@252.5（1300 px/单位）换算 ≈ APS 0.002–0.058，页面「低 APS ≈ 0.01–0.06」为合理近似。
  · Fig.14（img-02.webp）、Fig.15（img-01.webp，TTFT 起点约 22 s、Prompt TPS 带 1e7 系数）与页面图注相符。
- Ours(basic) = Basic：Fig.12 右「Basic」在 1024/2048 agents 下 ≈5300/10200，与 Fig.7 64K 面板 Ours(basic) 的 5365/10052 吻合，页面 §6.2 的等同判断成立（非无据推断）。

【引文编号与文献表（按 PDF v2 编号核对）】

- [39] = "Andre Richter, Christian Herber, Thomas Wild, and Andreas Herkersdorf. 2016. Resolving Performance Interference in SR-IOV Setups with PCIe Quality-of-Service Extensions. In 2016 Euromicro Conference on Digital System Design (DSD). 454–462."，页面 §4.1「引用 Richter et al. 2016，参考文献 [39]」正确。
- venue：Mooncake=23rd USENIX FAST ✓；DistServe=OSDI 24 ✓；Splitwise=51st ISCA ✓；PrefillOnly=SOSP '25 ✓；KVPR/TailorKV=ACL 2025 Findings ✓；Strata（arXiv:2508.18572，无会议字段）✓；LayerKV（arXiv:2410.00428，无会议字段）✓；SGL(MC) commit 19089aa ✓。

【公式与页面功能】

- 用仓库本地 libs/katex.min.js 实跑 renderToString(throwOnError:true) 渲染页面全部 156 处 $…$ / $$…$$（HTML 实体解码后），0 处失败，含 dojo:summary 的 \text/\to/\tfrac/\min 表达式、F-12 汇总式与 F-13 的 Z 式。
- python3 .dojo/scripts/validate.py 对 index.html 与 overview.html 均返回 "validation ok"。
- libs/ 下 8 个本地资源（katex.min.css/js、auto-render.min.js、prism 两主题两脚本、dojo-paper.css）全部存在；14 个 assets 图文件全部存在且与页面 src 一一对应。

【链接、图示与结构】

- 前置概念页 moe-serving（含 #s3-ep-alltoall、#s5-prefill-decode-kvcache、#s7-pdd-colocation）、standard-attention、deepseek-moe、mla、dsa、mqa-gqa、gpu-communication 全部真实存在，锚点在目标页中存在，无「（待生成）」占位。
- index.html ↔ overview.html 相互链接；overview 中「1.96 = (1.67+2.25)/2 为推断、论文未明示均值口径」已显式标注。
- 14 张内嵌原图与「原图与原文对应」清单（motivation=Fig.3 … largescale_rollout=Fig.15）逐一核对，Fig.2/Fig.5 未复用与说明一致。
- alt 属性内无 `$...$`；无等宽字符框线图；无自绘 SVG 图（因此不涉及 SVG `<text>` 公式问题）。
- 两级问题块：页面级「核心问题」5 题、章节级「本章问题」9 题（1/2/1/1/2/1/1），全部有解答折叠块，答案独立可读且与正文结论一致，核心问题答案均指明完整论证所在章节。

【表述】

- 通读全文含折叠块与图注：未见元话语（"本页将…""下面来看…""需要注意的是"）、会话指代（我/我们/你）、调试叙事或临场评价、口语化措辞；章节间有明确衔接。
- 实验事实与分析性判断有区分：§7 整体有 callout 标注「分析性判断，不是论文的结论」；边界性内容（trie/hash 对比、「第二个 storage NIC」「compute NIC 闲置资源」、命名由来、1.05 系数含义、H2D/D2H 走 CNIC 与流量隔离的联系）均在正文就地标注为推断。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（3 条均为轻微：2 条图注读数/柱数描述、1 条公式 Unicode 写法一致性；修正后即可发布，无遗留阻断与重要问题）
