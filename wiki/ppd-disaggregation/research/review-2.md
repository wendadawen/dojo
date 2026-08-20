# PPD 分离审查记录（第 2 轮）

- 页面版本：未跟踪（`?? wiki/ppd-disaggregation/`，index.html/overview.html 均修改于 2026-08-19 21:32）
- 论文版本：arXiv:2603.13358v2（TeX 源码 /tmp/ppd-research/src/main.tex；ICML 2026, 2026-05-05 修订）
- 审查时间：2026-08-19
- 审查者：独立子代理
- 已完整阅读章节：index.html 头/导航/元信息（1–686）；核心问题 5 题+术语表+§1 两个代价+SVG 示意图+§1 本章问题（687–846）；§2 复杂度对比+25 vs 24 算术+Fig.2 内联+微基准文字+§2 本章问题（848–901）；§3 配置空间+3.2 Pareto+winner 表+Tab.1+§3 本章问题（903–985）；§4 Eq.1 符号定义+代入算术+伪代码+解耦+vLLM 折叠+§4 本章问题（987–1094）；§5 真实负载 Fig.4+Tab.3+带宽 Fig.5+权重 Fig.6+失败率折叠+§5 本章问题（1096–1205）；方法评价+附图+来源与范围说明 C/F/N/G 表（1207–1330）；尾部 script（1335–1533）。overview.html 全篇（1–72）。已读 6 张原图（interference_tpot、fig1_pareto、ppd_converted、real_validation_e2e、scaling_simulation、weight_tradeoff_curve）。`.dojo/scripts/validate.py` 通过。
- 站内链接核对：standard-attention / mqa-gqa / gpu-communication / prefix-caching 四个目标页面均存在，无悬挂链接。

## 第 1 轮修复项复验

1. **§4.1 Eq.1 代入算术与 Δ 符号（阻断 2 条）**：balanced/ (1,6)/ (1,10) 三组 S 算术正确（同段）。**同段 L1001 符号定义 -0.6 与代入示例 +0.6 方向相反**（详见新增问题 1）。
2. **§3.2 Pareto 图描述（重要 1 条）**：L915 已改为「橙色方点为 baseline（PD 与 Replica 配置）」，与论文 caption 一致 ✓。**L917 图注仍写「橙色方点为 PD baseline（x=0）」**（详见新增问题 2）。属部分修复。
3. **C/F 编号章节号（重要 1 条）**：C8/C18/C21/F3/C22 全部修正 ✓；N11/N12/N15/N16/N21 同步已改 ✓。
4. **N 编号对照表 21 条（重要 1 条）**：N1–N21 全部就位，正文 sup 编号对位齐全 ✓。
5. **§5.2 "2.4 倍（12 vs 5）"（重要 1 条）**：L1127「约为 x=1 的 2.4 倍（12 vs 5）」算术 12/5=2.4 ✓。
6. **§1 引言连贯性（轻微 1 条）**：L727「构造示例。 一个 5 轮客服对话…」已为连贯引子 ✓。
7. **伪代码 pure text `x ∈ {0, 1}`（轻微 1 条）**：第 1 轮声称"已修复"但复验发现 L1040 仍含 `x ${\in}$ {0, 1}`（详见新增问题 4）。**未修复**。

## 新增问题（已逐条修复）

- [阻断·技术] §4.1 符号定义列表 L1001「Δ_ttft：本地处理相对走 P 的 TTFT 相对改善（如 -0.6 表示本地 TTFT 比走 P 低 60%）」与同段 L1009 代入示例「Δ_ttft=+0.6」及伪代码 L1032 方向相反。｜引文依据：main.tex L240「where Δ_ttft is the relative TTFT improvement」；index.html L1032 伪代码 Δ_ttft = (TTFT_base - TTFT_local) / TTFT_base。｜修复要求：把 L1001 的 "-0.6" 改为 "+0.6"，使符号定义与代入示例、伪代码、Eq.1 决策规则（S>0 走本地）方向一致。｜修复：已把 "如 -0.6 表示本地 TTFT 比走 P 低 60%" 改为 "如 +0.6 表示本地 TTFT 比走 P 低 60%"；Δ_tpot 列表的 "+0.08" 保持不变。｜复验：grep 显示 "-0.6 表示本地 TTFT" 已消失。

- [重要·技术] §3.2 L917 Fig.1 图注沿用「橙色方点为 PD baseline（x=0）」与 L915 已修复正文、论文 caption（main.tex L149 "Baselines (orange): PD and Replica configurations"）不符。｜引文依据：main.tex L149 caption。｜修复要求：把 L917 改为「橙色方点为 baseline（PD 与 Replica 配置）；蓝色圆点为 D-local capable 配置」。｜修复：L917 图注已改为「橙色方点为 baseline（PD 与 Replica 配置，Turn 2+ 都需 P 节点处理）；蓝色圆点为 D-local capable 配置（允许 D 节点本地处理 Turn 2+）」，与 L915、论文 caption 一致。｜复验：grep 验证 L917/L915 表述一致。

- [重要·技术] §2.1 L860–L864 构造示例数字不自洽：开篇 n=1000 输入+200 回复=1200 已缓存 + m=50 新 = 1250 总长，但页面取 n=1150 致 C_append=60,000（正确 62,500）、比值 26（正确 25）、n/m=23（正确 24）。｜引文依据：index.html L727「1000 token…得到 200 token 回复…追加 50 token」+ L858「C_full=(1000+200+50)²=1250²=1,562,500」。｜修复要求：把 n 改为 1200，重写 C_append=62,500、比值=25、n/m=24。｜修复：三处数字全部更正为 n=1200、C_append=62,500、比值 25、n/m=24；§4.1 章内 LCS 比较章节「开篇示例 m=50, n=1150」同步改为「n=1200」；草稿记录同步。｜复验：grep 无 1150、60,000、26.04 残留。

- [重要·技术] 来源与范围说明 G 表 3 处章节号错（G4 §6.2→§6.4、G5 §6.4→§6.5、G6 §4.2→§1）。｜引文依据：main.tex L145–152（Fig.1 位置）、L532–545（subsec:scaling-sim=§6.4）、L549–564（subsec:weight-tradeoff=§6.5）。｜修复要求：更正为 §6.4/§6.5/§1。｜修复：G4 改为 §6.4、G5 改为 §6.5、G6 改为 §1 Introduction（Pareto 总览）。｜复验：grep 无「§6.2 带宽模拟」「§6.4 权重扫描」「§4.2 Pareto」残留。

- [重要·技术] N 表 2 处章节号错：N13「abstract、§6」、N20「§5、§6」中的 §6 应为 §8（~68% 与 <1 ms 出自 §8 Conclusion）。｜引文依据：main.tex L586「adding <1 ms of per-request overhead」在 §8；abstract 引言里的 ~68% 与 §8 数字对应。｜修复要求：N13/N20 的 §6 改为 §8。｜修复：N13 改为「abstract、§8」；N20 改为「§5、§8」。｜复验：grep 无「abstract、§6」或「§5、§6」（仅 N20、N13）残留。

- [轻微·格式] §4.2 伪代码 L1040 「x ${\in}$ {0, 1}」KaTeX 在 <pre><code> 内不渲染，会显示字面字符。｜引文依据：renderMathInElement 默认 ignoredTags 包含 code。｜修复要求：把 ${\in}$ 改为纯文本 ∈。｜修复：在 build_ppd_page.py 加 <pre> 块保护逻辑（替换 ≥/∈ 之前先把 <pre> 内容替换为标记，替换完成后再恢复），保证伪代码块内的 unicode 字符不被 ${\in}$ 替换；实际输出 L1040 现在是 `输出：二元决策 x ∈ {0, 1}`（纯文本）。｜复验：grep 验证页面无 `x ${\in}$` 残留。

- [轻微·格式] overview.html L48「占多轮 prefill 成本约 99%」弱化原文上界语义。｜引文依据：main.tex L121 abstract「accounts for up to 99% of multi-turn prefill cost」。｜修复要求：把"约 99%"改为"高达 99%"。｜修复：overview L48 已改为"占多轮 prefill 成本高达 99%"。｜复验：grep 验证无 "约 99%" 残留。

- [轻微·可读性] §1 首次出现 SLO 未展开全称。｜引文依据：main.tex L106 abstract「Service Level Objectives (SLOs)」。｜修复要求：L729 首次出现 SLO 处用括号注全称。｜修复：L729「带 SLO 权重」改为「带 SLO（Service Level Objectives，服务等级目标）权重」。｜复验：grep 验证 L729 已含全称。

## 结论

- 统计：阻断 1 / 重要 4 / 轻微 3，全部修复
- 处置：第 2 轮全部修复完成，可进入第 3 轮审查
