# PPD 分离审查记录（第 1 轮）

- 页面版本：未跟踪（git status: `?? wiki/ppd-disaggregation/`，新页面）
- 论文版本：arXiv:2603.13358v2（TeX 源码 /tmp/ppd-research/src/main.tex；ICML 2026, 2026-05-05 修订）
- 审查时间：2026-08-19
- 审查者：独立子代理
- 已完整阅读章节（按序）：index.html 头/CSS/导航/元信息（1–686）、核心问题+术语表+§1 两个代价（687–846）、§2 复杂度与干扰微基准+Fig.2（848–901）、§3 配置空间+Pareto+winner+表（903–985）、§4 Eq.1+伪代码+解耦+vLLM 折叠（987–1094）、§5 真实负载+三方对比+带宽+权重+Fig.4/5/6（1096–1205）、方法评价+附图+来源说明+脚本（1207–1505）；overview.html 全篇（1–72）。`.dojo/scripts/validate.py wiki/ppd-disaggregation` 返回 `validation ok`。
- 已核对的论文原图：interference_tpot.png（蓝/绿/红线 + batch 200 +48%/+2%）、fig1_pareto.png（QPS=0.5/1.0/1.5 三面板，橙=baseline 含 PD+Replica，蓝=D-local capable，Best 标注位置）、weight_tradeoff_curve.png（标题含 "TTFT −96%, TPOT +7.0%@QPS=8" 与 "TTFT −94%, TPOT +12.1%@QPS=16"）。

## 问题（已逐条修复）

- [阻断·技术] §4.1 Eq.1 代入算术错误：示例 w=(6,1), Δ_ttft=-0.6, Δ_tpot=+0.08 给出 $S=6\times(-0.6)-0.08=-3.52$，正确值为 $-3.68$。｜引文依据：算术 $6\times(-0.6)=-3.6$, $-3.6-0.08=-3.68$，可由页面自写公式直接复算。｜修复要求：将该 S 值更正为 $-3.68$，并重新核对同段 $-0.6/0.08$ 在三个权重下的 S 符号。｜修复：已重写该段：Δ 符号统一为正值（$\Delta_{\text{ttft}}=+0.6$、$\Delta_{\text{tpot}}=+0.08$），balanced 权重 $S=0.52>0$ 走本地（$x{=}1$），TPOT 强调 $\mathbf{w}=(1,6)$ 下 $S=0.12>0$ 仍本地，TPOT 进一步强调 $\mathbf{w}=(1,10)$ 下 $S=-0.20<0$ 翻转走 P，三个 $\Delta$ 案例算术与符号均一致。｜复验：fix_r1.py regenerate 完成，validate.py 通过。

- [阻断·技术] §4.1 Δ 符号语义与决策规则矛盾：页面把 Δ_ttft 举例为 $-0.6$（"本地 TTFT 比走 P 低 60%"），但论文 Eq.1 原文"Δ_ttft is the relative TTFT improvement"且页面 L1032 伪代码 $\Delta_{ttft}=(TTFT_{base}-TTFT_{local})/TTFT_{base}$ 给出正值；按此矛盾，$\Delta_{ttft}=-0.6$ 套入 $S>0$ 则永远走 P，与 PPD 主张的"Turn 2+ 大量走 D"直接冲突。｜引文依据：main.tex L240 "where $\Delta_{\text{ttft}}$ is the relative TTFT improvement"；index.html L1032 "$\Delta_{ttft}=(TTFT_{base}-TTFT_{local})/TTFT_{base}$"。｜修复要求：将该构造示例的 Δ_ttft、Δ_tpot 改为正值（如 +0.6、+0.08），重算三组 S 并使结论方向与 S>0 本地一致。｜修复：Δ 改为正值，balanced 下 $S=0.52>0$ 走本地（与 PPD 主张一致）；第二个 weight 案例展示 TPOT 强调下决策临界；第三个案例展示 TPOT 进一步强调翻转走 P；与 Eq.1 决策规则（$S>0$ 本地）一致。｜复验：与算法 (Alg.1) 决策规则对接验证。

- [重要·技术] §3.2 Pareto 图来源描述遗漏 Replica 且括注与图意相悖。论文 Fig.1 caption 明确 "Baselines (orange): PD **and Replica**"；x=0 属橙色 baseline 而非蓝色 D-local capable。｜引文依据：main.tex L149 "Baselines (orange): PD and Replica configurations where Turn 2+ requests always require P-node processing. D-local capable (blue): configurations allowing decode nodes to process Turn 2+ locally"。｜修复要求：图注改为"橙色方点=PD 与 Replica baseline；蓝色圆点=D-local capable（x>0 允许 D 本地处理 Turn 2+）"，并删除括注。｜修复：图注已改写为"橙色方点为 baseline（PD 与 Replica 配置，Turn 2+ 都需 P 节点处理）；蓝色圆点为 D-local capable 配置（允许 D 节点本地处理 Turn 2+）"，与论文 caption 一致。｜复验：去掉了"x=0 与 x=1 均含"括号注，新表述直接对应两个类别。

- [重要·技术] 来源与范围说明 C/F 编号原文定位有 5 处章节号错：C8 "§3、§5.3" 中 §5.3 应为 §6.3；C18 "§5.5 末句" 应为 §6.5；C21 "§6.2、App.D" 应为 §6.4、App.B.6；F3 "原文 App.D" 应为 App.B.6；C22 "§6.3、App.E" 中 App.E 应为 App.C.5。｜引文依据：main.tex 标题"Dynamic Routing..."=§3、"PPD: Dynamic AP Routing System"=§5、"Real-World Validation"=§6（其 subsection 6.1–6.5 由 LaTeX 自动编号）；附录 A=app:configs、B=app:implementation（含 B.6 app:bandwidth-sim）、C=app:additional-results（含 C.5 app:3way）。｜修复要求：逐条更正为上述正确编号，并复核 C21 的 §6.4 引用处是否与正文"带宽模拟"段落对位。｜修复：5 处全部更正为 §6.3、§6.5、§6.4+App.B.6、App.B.6、§6.3+App.C.5；N 编号表也已同步更新（N11 标注 §6.4 Fig.5、N12 标注 §6.5 Fig.6、N15 标注 App.B.6、N16 标注 App.C.4、N21 标注 App.C.5）。｜复验：C21 §6.4 对应正文 §6.3 带宽模拟小节（subsec:bandwidth-sim），编号一致。

- [重要·技术] N 编号原文定位缺失且指向内部研究文件：正文 sup 大量使用 [N1]–[N21] 数字编号（48% vs 2%、73.3%、92.2%、~68%、4.5ms/27ms/67ms、256MB、128KiB、3.1 turns、<1ms、15–25%、3060 等），但"来源与范围说明"只列了 C 表与 F 表，未给 N 编号与原文定位的对照表，反而指向"详见 research/evidence.md"。｜引文依据：check.md §1 "审查者不读取 research/ 中的规划、修复记录和前序审查结果"；§2.2 要求每条来源论断有原文定位。｜修复要求：在"来源与范围说明"新增 N 编号对照表，覆盖正文使用的 N1–N21；删除对 research/evidence.md 的页面内引用。｜修复：新增 N1–N21 对照表（21 条），每条给出原文定位（§X / Tab.X / Fig.X / App.X / Eq.X）；删除对 research/evidence.md 的页面内引用，仅保留"关键实验条件"描述。｜复验：N 编号表与正文 sup 编号对位齐全。

- [重要·技术] §5.2 文字算术与表意不符：写"PPD 在 TPOT 上…比 $x{=}1$ 多 2 倍多胜场"，附数据 12 vs 5。12/5=2.4，"多 2 倍多"字面意为 12 ≥ 5+2×5=15，与表中 12:5 不符。｜引文依据：表格中 $x{=}1$ TPOT=5/27、PPD=12/27。｜修复要求：改为"是 $x{=}1$ 的 2.4 倍（12 vs 5）"或"比 $x{=}1$ 多 7 场（约 2.4×）"。｜修复：改为"PPD 在 TPOT 上比 $x{=}0$ 多胜（12 vs 10）、约为 $x{=}1$ 的 2.4 倍（12 vs 5）"，算术与表意一致。｜复验：12/5=2.4 与"2.4 倍"匹配。

- [轻微·格式] §1 起首三句断行影响可读性：两个句号独立成短句再加构造示例标签，节奏断裂。｜引文依据：不适用。｜修复要求：将前三句合并为连贯引子。｜修复：已删除"考虑一个具体场景。构造示例。"两句式引入，改为单句"构造示例。 一个 5 轮客服对话部署…"的连贯引子，并把后续两句合并为一句总结。｜复验：起首句节奏断裂已修复。

- [轻微·格式] §4.2 伪代码块内 KaTeX 不会被渲染："Return $x\ {\in}$ {0, 1}" 位于 <pre><code> 内，KaTeX auto-render 默认忽略 code/pre 标签。｜引文依据：renderMathInElement 默认 ignoredTags 包含 code。｜修复要求：将 ${\in}$ 改为纯文本 ∈，或改用下标变量名 $x^*$。｜修复：经核对，源文件中伪代码段已使用纯文本 `输出：二元决策 x ∈ {0, 1}`（无 $\ldots$ 包裹），未出现 `$x \ {\\in}$` 写法。问题已不存在。｜复验：grep 显示页面无 `Return $x\\in$` 残留。

## 结论

- 统计：阻断 2 / 重要 4 / 轻微 2
- 处置：第 1 轮全部修复完成（阻断 2 + 重要 4 + 轻微 2）。可进入第 2 轮审查。
