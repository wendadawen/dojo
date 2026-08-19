# DualPath 审查记录（第 1 轮）

- 页面版本：502c4984d7d1a03dfa5108370c15e7e1d618b849（git hash-object，工作树未提交）
- 论文版本：arXiv:2602.21548v2，2026-02-26
- 审查时间：2026-08-19
- 审查者：独立审查者（独立上下文子代理）
- 已完整阅读章节：核心问题 Q1–Q5、贯穿示例、章 1–7、来源与范围说明；TeX 源 01/02/03/04/05/06/08/09/10/11；原图 teaser/motivation/dataflow_peread/dataflow_ceread/660b_rollout_var_append/serving-aps-ttft-tpot/serving-aps-avg-jct/read_lb
- 未读取 research/ 下任何文件

## 问题

- [重要·技术] 1.2 节、Q1 解答、本章问题解答：`HBM 仅 2.4×` 与 Fig.3 motivation 左图不符。｜引文依据：fig:motivation 图内文字框 "GPU Memory: 2.2×"，红线 2020→2024 倍率为 2.2×。｜修复要求：`2.4×` 改为 `2.2×`，同步三处表述。
- [重要·可读性] Q1–Q5 解答末、第 1/2 章开头：章节引用全部偏移一位。Q1 答"第二章"、Q2"第三章"、Q3"第四章"、Q4"第五章"、Q5"第六、第七章"；第 1 章开头"第二章回答 Q1"、第 2 章开头"第三章回答 Q2"（第 3 章起自洽）；Q5 答中"第七章"实为方法评价章，与 Q5 论证无关。｜引文依据：页面 h2 标题编号 1–7 实际为 I/O 瓶颈/双路径/P-D 区间/CNIC/调度/实验/评价。｜修复要求：Q1→第一章、Q2→第二章、Q3→第三章、Q4→第四章、Q5→第五、第六章；第 1/2 章开头改"本章回答 Q"；Q5 答删"第七章"。
- [重要·技术] 4.3 节：`abstract 与 §5.1 文字写「approximately 99%」`。abstract 无 99%，仅 §5.1 出现。｜引文依据：paper.tex:57–65 abstract 全文不含 99%；05_traffic_mng.tex:14 `reserves approximately 99%`。｜修复要求：改为 `§5.1 文字写「approximately 99%」是粗略说法`。
- [重要·技术] 3.2 节 F-3：步骤引用错。页面写"PE path 步骤 3 + DE path 步骤 4"，合计 2 T_p·Dg；原文 PE CNIC 读方向仅含 PE read path 两步 (3)(5)。｜引文依据：04_system-overview.tex:89–93 `Read operations include PE paths (3) and (5), 2×T_p×Dg = 2Bs/g ≤ B`。｜修复要求：F-3 改"PE path 步骤 (3) 与 (5)，合计 2 T_p D g = 2Bs/g"；机制描述对应 PE H2D 与 D2H。
- [重要·技术] 3.2 节 F-6 / F-8：步骤编号与原文差一位。F-6 写"PE path 步骤 7"，原文 PE path 8（H2D 读方向）；F-8 写"PE path 步骤 8/9"，原文 PE paths 7/9（7 为写 DE DRAM、9 为 H2D）。｜引文依据：04_system-overview.tex:104–119 `read operations include PE path 8 and DE paths 3/6`、`Write operations include PE paths 7/9 and DE path 7`；§4.1 Decode Phase 段 `Label 8 and 9 in fig:pe_flow` 为 H2D。｜修复要求：F-6 改"PE path 步骤 8"；F-8 改"PE paths 步骤 7/9"。
- [重要·技术] 3.2 节 F-11 上界第三项数值错：`(M/(Bs)−3)/2 = (500/50−3)/2 = 4.7`，正确值 3.5。｜引文依据：500/50−3=7、7/2=3.5；04_system-overview.tex:135 `(g=8,s=1), M≈500, Bs≈50, 1/7 ≤ P/D ≤ 7/2`。｜修复要求：`4.7` 改为 `3.5`，并删除"取最小上界为 3.5"（实际 min 来自第二项 (g−s)/(2s)=3.5）。
- [重要·技术] 2.4 节、PE/DE buffer 图注（图 4 左）：写"全部 9 步按层重复 n_layer 次"，原文仅"(3-7) repeats n_layer times"；(1)(2) 是 Full Block 读存储、(8)(9) 是 decode 前 H2D。页面正文 2.2 段写"3-7 步对每层重复"与图注矛盾。｜引文依据：04_system-overview.tex:51 `This process (3-7) repeats n_layer times`。｜修复要求：图注改"步骤 (3)–(7) 按层重复 n_layer 次；(1)(2) 按 Full Block 读存储，(8)(9) 为 decode 前 H2D"。
- [重要·技术] 2.3 节、DE read path 图注（图 4 右）：写"(3)–(7) 按层重复"，原文"(3-5) repeats n_layer"；图注把 (6)(7) 解释为"PE 发回 miss KV + DE CNIC 写 DRAM"，原文 (6)(7) 是 decode 前一次性 H2D，miss KV 回传未单独给标签。｜引文依据：04_system-overview.tex:55–59 `(3-5) repeats n_layer times`、`Label 6 and 7 in fig:de_flow = H2D`。｜修复要求：图注改"步骤 (3)–(5) 按层重复；(6)(7) 为 decode 前一次性 H2D"；移除 (6) miss KV 回传归属。
- [重要·技术] 2.4 节末、Q2 解答：写"PE/DE buffer 是...刻意不绕过 buffer...更关键的是让 H2D/D2H 也走 CNIC 是第五章统一设计"。原文只论证 DE buffer，PE buffer 动机未明说；TTFT 论据在 §4.1 Decode Phase 段而非 Prefill PE read path 段；"更关键的是 H2D/D2H 走 CNIC 是第五章统一设计"无原文支持。｜引文依据：04_system-overview.tex:60–62 仅 DE buffer TTFT/显存论证；§5 才提 CNIC 统一调度。｜修复要求：①分清 PE/DE buffer，原文动机仅适用 DE buffer；②章节定位改"§4.1 Decode Phase 段"；③删除"更关键的是 H2D/D2H 也走 CNIC 是第五章统一设计"或标注为页面推断。
- [重要·技术] 1.1 节表注、`page-summary`："DS V3.2 22 GB/PFLOP（论文正文给出的 27B/660B 综合值）"。原文 §3 仅对 V3.2（指 660B）给出 22 GB/PFLOP，27B 是 internal downscaled。｜引文依据：03_motivations.tex:13 `approximately 22 GB/PFLOP for DeepSeek-V3.2`；Table cache-compute-ratio 列 V3.2 660B=13–36，V3.2 27B 行已注释。｜修复要求：改为"DS V3.2 660B 22 GB/PFLOP（论文 §3 代表数字）"。
- [重要·技术] 贯穿示例、Q1 解答、6.2 节、来源说明："64K context、append 429 的 round 含 62976 hit + 512 miss"。512 miss 与 trace 平均 append 429 矛盾；总 63488=62976+512，但 64K MAL Context 平均 32721；命中率 1−512/63488=99.19%，与 trace 98.7% 矛盾；来源说明称"不算构造"不成立。｜引文依据：08_evaluation.tex Table agent-dataset-statistics 64K 行 `Turns 157, Append 429, Context 32721`；03_motivations.tex:12 `KV-Cache hit rate of 98.7%`。｜修复要求：①改为"32721 hit + 429 miss"（或保留 62976/512 并明确"取 64K MAL 单条轨迹尾部近似"）；②来源说明删"不算构造"，改"构造示例：..."。
- [重要·技术] 6.5 节图注（fig:serving + fig:serving_jct）：写"SGL(MC) 在 DS 27B 上数据点稀疏，DS 660B 上 N/A"。事实相反：DS 27B 完全无 SGL(MC) 数据点（论文 Baselines 段说明未在 DS 27B 上跑 SGL(MC)），DS 660B 在两张图上均有 SGL(MC) 数据点。｜引文依据：08_evaluation.tex:59 `We did not run SGL(MC) for DS 27B because SGLang lacks support for this downscaled version`；fig:serving 与 fig:serving-aps-avg-jct 图。｜修复要求：两处图注改为"SGL(MC) 未在 DS 27B 上运行；DS 660B 上有 SGL(MC) 数据点（TTST 异常低，论文归因为 first-two-token 同到达实现 bug）"。
- [重要·技术] 6.3 节："Basic 在 append 3× 时已经接近 DualPath 性能"。图 9 x3 时 Ours≈2000s、Ours(basic)≈3700s，加速比仍 1.85×，"接近"夸大。｜引文依据：08_evaluation.tex:131 `1.82–1.99× speedup at different append scales`；图 9 三组柱。｜修复要求：改为"Basic 在 append 3× 时相对优势收窄至 ~1.85×，但仍未接近 DualPath"。
- [重要·技术] 1.2 节、动机图注："batch size 5 之后收益趋平"。图上 1→5 升至 2.5×，5→10 仍升至 2.9×，10→20 趋平。｜引文依据：fig:motivation 右图曲线，坐标轴 1/5/10/20。｜修复要求：改为"batch size 10 之后收益趋平；5→10 仍有约 16% 提升"。
- [轻微·格式] 7.1 节："1142 GPU 大规模下做到近线性扩展"。同页 6.8 节写"1152 GPU"，论文 §8.5 与来源说明均为 1152。｜引文依据：08_evaluation.tex:265 `up to 1,152 GPUs`。｜修复要求：`1142` 改为 `1152`。
- [轻微·技术] 4.2 节末："论文 §5.1 引言段明确指出..."。该句在 §5 CNIC-Centric Traffic Manager 引言段，非 §5.1 子节。｜引文依据：05_traffic_mng.tex:8 位于 §5 开头。｜修复要求：`§5.1 引言段` 改为 `§5 引言段`。
- [轻微·技术] overview.html："abstract 给出的 average 1.96× 来自这两个数字的算术平均"属分析性判断未标注。｜引文依据：(1.67+2.25)/2=1.96。｜修复要求：改为"abstract 的 average 1.96× 与 (1.67+2.25)/2=1.96 吻合（页面推断）"。
- [轻微·技术] 1.3 节、7.3 节："Strata（USENIX ATC 2024）"。论文 bib 中 Strata 是 @misc，无会议字段。｜引文依据：reference.bib:458 `@misc{Strata,...}`。｜修复要求：删除会议名或注明"ATC'24 系外部信息，bib 为 misc"。

## 结论

- 统计：阻断 0 / 重要 13 / 轻微 4
- 处置：修复
---

## 修复记录

| # | 级别 | 修复 |
|---|---|---|
| 1 | 重要·技术 | HBM 2.4× → 2.2×（Q1 解答、§1.2 描述、Fig.3 图注，三处） |
| 2 | 重要·可读性 | 章节引用：Q1→第一章、Q2→第二章、Q3/Q4→第三章/第四章、Q5→第五章与第六章；§1/§2 章开头改"本章回答 Q"；Q5 解答删"第七章" |
| 3 | 重要·技术 | 99% 表述：abstract 无 99% 字样，改"§5.1 文字写「approximately 99%」是粗略说法" |
| 4 | 重要·技术 | F-3 步骤：PE path 步骤 3 + DE path 4 → PE paths 步骤 (3) 与 (5) |
| 5 | 重要·技术 | F-6/F-8 步骤：F-6 改"PE path 步骤 8"；F-8 改"PE paths 步骤 7/9" |
| 6 | 重要·技术 | F-11 上界第三项 4.7 → 3.5；最小上界描述同步改"3.5（与第二项并列）" |
| 7 | 重要·技术 | PE read path 图注：9 步全重复 → (3)-(7) 按层重复 $n_\text{layer}$ 次；说明 (1)(2) 按 Full Block 读存储、(8)(9) 为 decode 前 H2D |
| 8 | 重要·技术 | DE read path 图注：(3)-(7) 重复 → (3)-(5) 重复；说明 (6)(7) 为 decode 前一次性 H2D |
| 9 | 重要·技术 | PE/DE buffer：分清动机仅适用 DE buffer；定位"§4.1 Decode Phase 段"；删"刻意让 H2D/D2H 走 CNIC"无原文支持的归因 |
| 10 | 重要·技术 | DS V3.2 22 GB/PFLOP：删"27B/660B 综合值"，改为"DS V3.2 660B 22 GB/PFLOP（论文 §3 正文代表数字，Table 1 列 13-36）" |
| 11 | 重要·技术 | 贯穿示例 62976/512 → 32721 hit + 429 miss，命中率 1−429/32721 ≈ 98.7% 与 trace 一致；标注"构造示例" |
| 12 | 重要·技术 | fig:serving 与 fig:serving_jct SGL(MC) 描述：明确"未在 DS 27B 上运行"；DS 660B 上有数据点 |
| 13 | 重要·技术 | 6.3 节"接近"改"收窄到约 1.85×，DualPath 仍明显领先" |
| 14 | 重要·技术 | 1.2 节 batch size "5 之后" → "10 之后趋平；5→10 仍有约 16% 提升" |
| 15 | 轻微·格式 | 1142 GPU → 1152 GPU |
| 16 | 轻微·技术 | §5.1 引言段 → §5 引言段（多处） |
| 17 | 轻微·技术 | overview 1.96× 标注"论文 abstract 表述 + 算术平均推断" |
| 18 | 轻微·技术 | Strata 删"USENIX ATC 2024"，注明"论文 bib 为 @misc，外部资料常称 ATC'24 但本页不引用未在论文 bib 中确认的会议归属" |

## 修复后状态

- validate.py: ok
- 所有 18 条问题已关闭
- 派发第二轮独立审查
