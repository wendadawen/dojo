# PPD 分离审查记录（第 3 轮）

- 页面版本：未跟踪（`?? wiki/ppd-disaggregation/`，2026-08-19 后无新增 commit）
- 论文版本：arXiv:2603.13358v2（TeX 源码 /tmp/ppd-research/src/main.tex）
- 审查时间：2026-08-20
- 审查者：独立子代理
- 已完整阅读章节：index.html 头/导航/CSS（1–686）；核心问题 5 题+术语表+§1 两代价+SVG+§1 本章问题（687–846）；§2 复杂度+干扰微基准+Fig.2+§2 本章问题（848–901）；§3 配置空间+Pareto+winner 表+Tab.1+§3 本章问题（903–985）；§4 Eq.1+伪代码+解耦+vLLM 折叠+§4 本章问题（987–1094）；§5 真实负载+带宽+权重+失败率+§5 本章问题（1096–1205）；方法评价+附图+来源说明 C/F/N/G 表（1207–1330）；尾部 script（1335–1533）；overview.html 全文。已阅 6 张原图。
- validate.py 结果：`python3 .dojo/scripts/validate.py wiki/ppd-disaggregation/index.html` → `validation ok`。
- 链接核对：../standard-attention、../mqa-gqa、../gpu-communication、../prefix-caching、../moe-serving 五个目标 index.html 均存在，无悬挂。overview.html ↔ index.html 互链 ✓。

## 前两轮修复项复验

注：review-1.md 列 8 条（2 阻+4 重+2 轻）、review-2.md 列 8 条（1 阻+4 重+3 轻），合计 16 条（任务说明"17 个"按实际清单 16 条计）。

1. R1-阻断·Eq.1 代入算术（Δ=+0.6, +0.08 三组 S）— ✓ §4.1 L1012-1014：(1,1)→0.52、(1,6)→0.12、(1,10)→-0.20；补例 +0.2/+0.05→(1,1)=0.15、(1,10)=-0.30，算术全对。
2. R1-阻断·Δ 符号与决策规则 — ✓ L1005「S>0 本地、否则走 P」与 L1032 Δ_ttft=(TTFT_base-TTFT_local)/TTFT_base 方向一致；代入示例 Δ=+0.6 与 Eq.1 决策规则方向一致。
3. R1-重要·§3.2 Pareto 图（baseline 含 PD 与 Replica）— ✓ L915 与 L917 双处「橙色方点为 baseline（PD 与 Replica 配置，Turn 2+ 都需 P 节点处理）；蓝色圆点为 D-local capable 配置」，与 paper Fig.1 caption L149 一致。
4. R1-重要·C/F 5 处章节号（C8/C18/C21/F3/C22）— ✓ L1254 C8「§3、§6.3」、L1262 C18「§6.5 末句」、L1265 C21「§6.4、App.B.6」、L1274 F3「App.B.6」、L1266 C22「§6.3、App.C.5」全部正确，N11/N12/N15/N16/N21 同步对齐。
5. R1-重要·N 编号对照表 21 条 — ✓ L1280-1307 N1-N21 全部就位，每条给出原文定位；正文 sup [N1]-[N21] 引用全部能在表内找到。
6. R1-重要·§5.2 2.4 倍 — ✓ L1127「约为 x=1 的 2.4 倍（12 vs 5）」算术 12/5=2.4 正确。
7. R1-轻微·§1 起首连贯 — ✓ L727「构造示例。 一个 5 轮客服对话…」为连贯引子，节奏断裂已消。
8. R1-轻微·伪代码 x∈{0,1} KaTeX — ✓（与第 14 项同问题，第 2 轮重做）L1040 现为「输出：二元决策 x ∈ {0, 1}」纯文本。
9. R2-阻断·L1001 符号定义 +0.6 — ✓ L1001「如 +0.6 表示本地 TTFT 比走 P 低 60%」与代入示例、伪代码、Eq.1 决策规则方向一致；grep 无 "-0.6 表示本地 TTFT" 残留。
10. R2-重要·L917 Fig.1 图注 — ✓ L917「橙色方点为 baseline（PD 与 Replica 配置）；蓝色圆点为 D-local capable 配置」与 L915、paper caption 一致。
11. R2-重要·§2.1 数字 n=1200 — ✓ L860-864 C_full=1,562,500、C_append=62,500、比值 25、n/m=24；L884 同步；grep 无 1150/60,000/26.04 残留。
12. R2-重要·G 表 3 处（G4/G5/G6）— ✓ L1315 G4「§6.4」、L1316 G5「§6.5」、L1317 G6「§1 Introduction，Pareto 总览」全部正确。
13. R2-重要·N13/N20 §6→§8 — ✓ L1296 N13「abstract、§8」、L1303 N20「§5、§8」；grep 无 §6 残留。
14. R2-轻微·伪代码 x∈ 纯文本 — ✓ L1040 纯文本 ∈，与 L1032-1033 伪代码其他 KaTeX 符号无冲突。
15. R2-轻微·overview「约 99%」→「高达 99%」— ✓ overview.html L48「占多轮 prefill 成本高达 99%」与 paper abstract L109「accounts for up to 99%」语义匹配。
16. R2-轻微·§1 SLO 全称 — ✗ **未修复/已回归**：grep 全文无 "Service Level Objectives"；L729 首次出现 SLO 处仍为「带 SLO 权重的逐请求二元决策」无全称展开；术语表 L774-786 未收录 SLO 行。R2 复验记录"grep 验证 L729 已含全称"与现状不符，修复在第 3 轮前已丢失或被覆盖。

## 新增问题

- [重要·技术] §4.4 vLLM 折叠块 L1062 描述 P 节点「仅 Turn 1 触发传输」与 PPD 路由回 P 的语义矛盾，且无原文支持。｜引文依据：main.tex B.2 L665-668 仅说「P nodes run with kv_role=kv_producer, generating KV caches and sending them via ZeroMQ」，未限定仅 Turn 1；main.tex §3 L308「when x=0, D receives KV transfer from P every turn」；§6.5 D-local 比例 0%–95% 随权重变化说明被路由回 P 的 Turn 2+ 仍触发 KV 传输（平衡权重 5% 路由回 P 时削减 75% 而非 100%）。｜修复要求：删除「仅 Turn 1 触发传输」或改为「Turn 1 与被路由回 P 的 Turn 2+ 请求触发 KV 传输」。

- [重要·技术] 来源与范围说明 F4 L1275 原文定位「论文 App.D」错误，论文附录只有 A/B/C。｜引文依据：F4 引用的 s_kv=128 KiB 句位于 main.tex L707（App.B.6 app:bandwidth-sim）；N15 同一数字正确定位为 App.B.6。｜修复要求：把 F4 中「论文 App.D」改为「论文 App.B.6」。

- [重要·技术] 元信息 blockquote L724 代码链接 https://github.com/freelulul/vllm-ppd 无可核实来源。｜引文依据：main.tex 无任何 github 引用；WebFetch arxiv.org/abs/2603.13358 摘要页未列代码仓库；00README.json 仅含主 TeX 文件名。｜修复要求：删除该 URL，或在末尾加「（链接未经论文/官方渠道核实）」明示为推断。

- [重要·技术] 核心问题 5 解答 L766 与 overview.html L65 把「1P_3D ShareGPT 15–25%」泛化为 ShareGPT 与 WildChat 两数据集。｜引文依据：main.tex L491「For the stable 1P_3D configuration on ShareGPT, PPD reduces average query latency by 15--25%」，15–25% 仅就 ShareGPT 1P_3D 给出；§5.1 正文 L1106 与 N8 表项均限定 ShareGPT。｜修复要求：核心问题 5 解答与 overview 关键结论加「ShareGPT」限定。

- [轻微·格式] §4.2 引导文 L989 写「（C/F1）」为纯文本括号引用，与全页 sup 规范不一致且 C 编号不明确。｜引文依据：紧邻 L993 用规范 sup [C18, F1]，本句 C 应为 C18。｜修复要求：把「（C/F1）」改为 sup [C18, F1]。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 1（其中复验失败 1 项按 R2 原分级计为轻微）
- 处置：修复后发布。需处理：① R2-轻微 16 SLO 全称（回归补回 L729）；② §4.4「仅 Turn 1 触发传输」删/改；③ F4「App.D」→「App.B.6」；④ 代码链接删除或加未经核实标注；⑤ 核心问题 5/overview 加 ShareGPT 限定。轻微 1 项（C/F1 引用）可一并修复。修后无需重启审查。

## 修复记录（2026-08-20，主编排者）

全部 6 项已修复，修复前先在 main.tex 复核（github 引用 0 处、SLO 全称在 L106、s_kv=128 KiB 句在 App.B.6 L707、15–25% 仅限 ShareGPT L491）：

1. ① SLO 全称：§1 首次出现处改为「带 SLO（Service Level Objectives，服务等级目标）权重」，术语表新增 SLO 行（原文 L106 "no single fixed routing strategy satisfies all Service Level Objectives (SLOs)"）。
2. ② §4.4 vLLM 折叠块：改为「Turn 1 与被路由回 P（$x{=}0$）的 Turn 2+ 请求都触发 KV 传输」（对齐 §3 "when x=0, D receives KV transfer from P every turn"）。
3. ③ F4：「论文 App.D」→「论文 App.B.6」（引句实位于 app:bandwidth-sim，main.tex L707）。
4. ④ 元信息代码行：删除 github.com/freelulul/vllm-ppd，改为「论文未给出官方代码仓库链接（正文与 arXiv 摘要页均无）」。
5. ⑤ 核心问题 5 解答与 overview 关键结论：15–25% 加「ShareGPT 稳定 1P_3D 配置」限定。
6. 轻微：§4.2 引导文「（C/F1）」→ sup [C18, F1]。

修复后 validate.py：index.html 与 overview.html 均 validation ok。按第 3 轮处置意见，修复后即可发布。

