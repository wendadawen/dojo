# UltraEP 审查记录（第 1 轮）

- 页面版本：index.html `84c47ce90161aad235691741f30af291455d7c25`；overview.html `ddc3642b2cb7f94bbb4689a366a079086d0bcded`
- 论文版本：arXiv:2606.04101v3（TeX 源码 /tmp/ultraep/src/，v1 与 v2 已撤回，v3 提交于 2026-06-18）
- 审查时间：2026-09-01 18:10 CST
- 审查者：编排者派发的独立审查者
- 已完整阅读章节：
  - 页面 index.html：标题与元信息、核心问题（4 题）、术语速查、贯穿全文的最小例子、1. 专家热度在层与批次之间快速漂移（1.1–1.5 + 本章问题）、2. 精确负载只能在 gating 之后拿到（2.1–2.5 + 本章问题）、3. 只复制不重排（3.1–3.6 + 代价模型折叠块 + 本章问题）、4. 配额把「建不建副本」和「接多少 token」并成一个变量（4.1–4.9 + Algorithm 1 代码折叠块 + 本章问题）、5. 把不规则的专家搬运做成 tile 流水线与 chunk 中继（5.1–5.7 + 实现层面折叠块 + 本章问题）、6. 实验结果（6.1–6.7 + 本章问题）、7. 方法评价（7.1–7.3）、来源与范围说明（8 个小节）
  - 页面 overview.html：论文做了什么、主要贡献、方法概述、关键结论
  - 论文原文：Abstract、§1 Introduction、§2 Background（2.1–2.2）、§3 Expert Load Analysis、§4 UltraEP System Design（4.1–4.4，含 Table 1）、§5 Quota-Driven Planning（5.1–5.3，含 Algorithm 1）、§6 RSN-Native Balancing Communication（6.1–6.2）、§7 Implementation、§8 Evaluation（8.1–8.6，含 Table 2、Table 3）、§9 Related Work、§10 Conclusion
  - 原图核对：figs/ 下 intro-overview、bg-rsn、motive_eplb_imbalance、tech-expert-layout、tech-timeline-fwd、tech-relay、eval_train_e2e、eval_serving_e2e、eval_latency_breakdown、eval_solver_perf、eval_comm_perf、eval_prod 共 12 个 PDF 已用 pdftoppm 渲染后读图，并用 pdftotext 提取图内文字标注

## 问题

- [阻断·技术] 4.5 节第三次探测之前的第一次探测（index.html:1181）：写「第一次探测 $\tau=9$，$\mathrm{slk}=(0,0,5,7)$」，$r_1$ 的空闲被写成 0。$r_1$ 初始负载为 6，按 Eq.(6) 应为 $\mathrm{slk}_{r_1}(9)=\max(9-6,0)=3$，正确向量是 $(0,3,5,7)$。这是第 4 章唯一完整求解演示的第一步，读者按 Eq.(6) 复算即与页面对不上；第二、三次探测的 $\mathrm{slk}=(0,1,3,5)$ 与 $(0,0,2,4)$ 经复算均正确，只有本处错。｜引文依据：§5.1 Eq.(6)「$\mathrm{exc}_r(\tau)=\max(\ell_r-\tau,0),\qquad\mathrm{slk}_r(\tau)=\max(\tau-\ell_r,0)$」；页面自身给出 $\ell_{r_1}=6$（index.html:818）；实测复算 `tau 9 exc [3,0,0,0] slk [0,3,5,7]`｜修复要求：把 index.html:1181 的 $\mathrm{slk}$ 改为 $(0,3,5,7)$；同时确认该改动不影响后续叙述（贪心仍选空闲最大的 $r_3$、$\delta=\min(3,7,10)=3$、结果负载 9、6、4、5 不变），如叙述中出现「只有 $r_3$ 可用」一类依赖旧值的表述一并修正｜修复：已把 4.5 节第一次探测的 $\mathrm{slk}$ 改为 $(0,3,5,7)$。后续叙述本就只说「合法目标里 $r_3$ 空闲最大（7）」，未依赖 $r_1$ 的旧值，无需连带修改。｜复验：脚本复算三次探测 exc/slk：tau=9 得 (0,3,5,7)、tau=7 得 (0,1,3,5)、tau=6 得 (0,0,2,4)，与页面现值逐项一致。

- [阻断·技术] 4.9 节 Table 3 表格行、核心问题 3 解答、4 章本章问题第 2 题、7.1 节、来源与范围说明、overview 关键结论（index.html:1417、758、1432、1455、1813、1871；overview.html:81）：把 Table 3 的 $\sum_e\lvert\mathcal{H}(e)\rvert$ 一行解释为「实例总数」（45 对 107）。该行按论文表标题是「consumed redundant slots」，即消耗的冗余槽（副本）数，不是物理实例总数。Figure 15 的模拟配置为 (128,64,1)、(128,64,2)、(256,64,2)，逻辑专家至少 128 个，每个专家至少有一个主实例，因此实例总数不可能是 45，「实例总数」这一读法自相矛盾。57.9% 这一比值本身正确。｜引文依据：§8.5 Table 3 表标题「Balancing metrics averaged across simulations in Fig. 15, including solving time, **consumed redundant slots**, maximum replica fan-out, and traffic ratio」；§8.5 正文「\sys saves solving latency by 27.4\,\%, **consumes 57.9\,\% fewer redundant slots**, and reduces token traffic by 3.9\,\% with locality」；Figure 15 下排横轴标注「(Total Experts, EP Size, $N_{\text{slot}}$)」取值 (128,64,1)/(128,64,2)/(256,64,2)｜修复要求：把上列 7 处的「实例总数」全部改为「消耗的冗余槽数」（或「副本数」），Table 3 行的指标名一并改写；index.html:1417 的符号列保留 $\sum_e\lvert\mathcal{H}(e)\rvert$ 时须补一句说明该符号在论文中对应消耗的冗余槽数｜修复：7 处「实例总数」全部改为「消耗的冗余槽数」（index.html 6 处：Table 3 行指标名、核心问题 3 解答、4.9 节归因段、4 章问题 2 解答、7.1 节、来源与范围说明；overview.html 1 处）。｜复验：grep「实例总数」在两页均为 0 命中；Table 3 行现为「$\sum_e\lvert\mathcal{H}(e)\rvert$（消耗的冗余槽数）」，与论文表标题 consumed redundant slots 一致。

- [重要·技术] 4 章本章问题第 2 题解答（index.html:1454）：写「$\delta<u_{\min}$ 时直接跳出（Algorithm 1 第 13 行）」。Algorithm 1 第 13 行是 `if $\mathcal{T}=\varnothing$ then break`，`if $\delta<u_{\min}$ then break` 在第 16 行。论断本身与原文一致，只是行号定位错误。｜引文依据：Algorithm 1 按 algorithm2e 行号：12 行 `$\mathcal{T}\gets$ admissible host ranks...`、13 行 `\lIf{$\mathcal{T}=\varnothing$}{break}`、14 行 `$t^\star\gets\arg\max_{t\in\mathcal{T}}\mathrm{slk}_t$`、15 行 `$\delta\gets\min(\mathrm{exc}_r,\mathrm{slk}_{t^\star},\mathrm{cap}_e)$`、16 行 `\lIf{$\delta<u_{\min}$}{break}`｜修复要求：把「第 13 行」改为「第 16 行」，或改为不带行号的「Algorithm 1 SolveReplication 中 $\delta$ 计算之后的剪枝分支」｜修复：改为不带行号的表述「Algorithm 1 SolveReplication 中 $\delta$ 与 $u_{\min}$ 的比较一步」，避免行号随排版变化失效。｜复验：grep「第 13 行」0 命中；该处现无行号引用，论断本身未变。

- [重要·技术] 1.4 节 Figure 6 解释段（index.html:864）：写「EPLB 曲线整体低于 no-balancing，说明它确实起作用」。左图（Qwen3-235B prefill）成立，但右图（DeepSeek-V3 训练）不成立——该面板中 EPLB 曲线与 no-balancing 交织，前半段大部分位于 no-balancing 之上（EPLB 约 1.30、no-balancing 约 1.22）。论文对同一模型的判断与页面相反。｜引文依据：读图 figs/motive_eplb_imbalance.pdf 右面板（DeepSeek-V3, Layer 57，纵轴 1.2–1.6），橙色 EPLB 线在 microbatch 0–60 区间稳定高于黑色 No Balancing 线；§8.2「For DeepSeek-V3 (c), routing compensation reduces the overall imbalance but enlarges short-term load swings, where **EPLB and LPLB show similar or even worse performance than Megatron-LM**」｜修复要求：把该句改为分图表述——左图 EPLB 整体低于 no-balancing，右图两条曲线互有高低、EPLB 未整体降低不均衡；并保留后半句关于两图均出现 EPLB 高于 no-balancing 尖峰的表述（该表述经读图成立：左图 x≈147 处橙线约 9.0 而黑线约 3.5，右图 x≈10 处橙线 1.60 而黑线约 1.22）｜修复：Figure 6 解释段改为分图表述：左图 EPLB 整体低于 no-balancing 但仍有超出尖峰；右图有相当长区段稳定高于 no-balancing，即训练侧反而更差，并注明这与 §8.2 对 DeepSeek-V3 的判断一致。｜复验：重读该段，现表述与 §8.2 原文「EPLB and LPLB show similar or even worse performance than Megatron-LM」方向一致，不再有「整体低于」的全局断言。

- [重要·技术] 5.7 节通信表格与来源说明（index.html:1570、1873、1871）：写「数值为原文 Figure 16 的标注值」，并给出 5 档 × 4 方案共 20 个两位小数值；来源说明另称 Figure 15 下排数值为「标注值」。这两张图都没有逐柱数值标注，页面数值是从柱高读出的估计值。用 pdftotext 提取 eval_comm_perf.pdf 只得到图例与坐标刻度（`1.6 1.2 0.8 0.4 0.0`、`1.5 | 0 2.0 | 0 4.0 | 1 6.0 | 2 8.0 | 3`），eval_solver_perf.pdf 同样只有刻度。对照可知 Figure 11、12、13、17 确有逐项数值标注，Figure 15、16 没有。｜引文依据：`pdftotext figs/eval_comm_perf.pdf` 全文为「Latency (ms) torch.distributed DeepEP Ours w/o Relay Ours 1.6 1.2 0.8 0.4 0.0 1.5 | 0 2.0 | 0 4.0 | 1 6.0 | 2 8.0 | 3 Init Imbalance | # of Relay-enabled Main Experts (Ours)」，无任何柱值；对比 `pdftotext figs/eval_latency_breakdown.pdf` 含「5.95 … 1.89 1.68 2.83 … 6.94 2.11 2.24 2.88」等逐柱标注｜修复要求：把 index.html:1570 与 1873 的「标注值」改为「按 Figure 16 柱高读取的估计值」，1871 中 Figure 15 同样改为读图估计值；表格数值保留两位小数时须在同一句声明为读图估计，或改为一位小数｜修复：三处「标注值」改为读图值并说明该图未逐柱标注：5.7 节表格前引导句、来源说明中 Figure 16 与 Figure 15 两条；4.9 节图解释段中的 1.40/1.07/1.09 也补注为读图得到，并附论文正文的定性表述。｜复验：Figure 11/12/13/17 四处「标注值」保留（这四张图确有逐项标注）；Figure 15/16 相关处已无「标注值」措辞。

- [重要·技术] 4 章本章问题第 3 题解答（index.html:1463）：写「论文的模拟均值是在飞 token 占比从 98.4%（关闭局部性）降到 96.0%，正文表述为减少 3.9% 的 token 流量（Table 3）」。98.4% 到 96.0% 只有 2.4 个百分点，论文的 3.9% 是相对 EPLB$+$ 的 99.9% 而言（$(99.9-96.0)/99.9=3.9\%$），与同句并列的 27.4%、57.9% 使用同一基线 EPLB$+$。页面把 3.9% 挂到了开关局部性的对照上，基线错位。同页 4.9 节表格（index.html:1419）的「降 3.9 个百分点」基线正确，两处互相矛盾。｜引文依据：§8.5「\sys saves solving latency by 27.4\,\%, consumes 57.9\,\% fewer redundant slots, and reduces token traffic by 3.9\,\% with locality」，其中 27.4\% 复算 $(0.153-0.111)/0.153=27.45\%$、57.9\% 复算 $(107-45)/107=57.9\%$ 均以 EPLB$+$ 为基线；Table 3「In-flight Token Ratio | 99.9\,\% | 96.0\,\% (98.4\,\% w/o locality)」｜修复要求：把该句改为「相对 EPLB$+$ 的 99.9% 降到 96.0%，即论文所说减少 3.9% 的 token 流量；关闭局部性时为 98.4%，与 96.0% 相差 2.4 个百分点」｜修复：改为「UltraEP 的在飞 token 占比 96.0%，相对 EPLB$+$ 的 99.9% 低 3.9 个百分点」，并补一句说明关闭局部性时为 98.4%、即局部性本身贡献其中 2.4 个百分点。｜复验：与同页 4.9 节表格的基线（99.9% 对 96.0%）现已一致，两处不再矛盾；3.9% 的基线与同句 27.4%、57.9% 统一为 EPLB$+$。

- [重要·技术] 4.5 节代码折叠块的观察重点与 4.6 节、4 章本章问题第 3 题（index.html:1367、1389、1463）：以 `locality=False` 分支得出「跨 rank 在飞 token 62.5%」并称其「验证了论文所说本地优先只改变哪个源消费配额，不改变配额本身」。实测该分支违反 Algorithm 1 SolveReroute 应保持的每源需求守恒：$r_0$ 对 $e_0$ 的需求为 4 但实发 5，$r_3$ 对 $e_0$ 的需求为 1 但实发 0。因此 62.5% 这个对照值不是该形式化问题的合法解，用它作为「本地优先不破坏配额」的验证依据不成立（配额守恒在两个分支下都满足，真正被破坏的是源侧守恒）。页面的简化条件说明（index.html:1368、1893）未披露这一偏差。｜引文依据：§5.2「Reroute then distributes the residual source demand over the remaining quotas in proportion to their residual capacity, **with deterministic rounding to preserve both per-source demand and per-instance quota**」；页面 index.html:1375 自行给出双向守恒式 $\sum_{t\in\mathcal{H}(e)}q_{r,e,t}=\lambda_{r,e}$；实测 `locality=False` 源侧不符项 `(rank 0, e0, 需求 4, 实发 5)`、`(rank 3, e0, 需求 1, 实发 0)`，而 `locality=True` 源侧与配额两侧全部守恒｜修复要求：二者择一——(a) 修正 `solve_reroute` 的 `locality=False` 分支使其满足每源需求守恒后重跑，更新页面展示的输出与 62.5%；(b) 删除关闭本地优先的对照及 62.5%，改为只用论文 Table 3 的 98.4% 说明局部性收益。无论选哪种，index.html:1367 中「验证了论文所说……」一句须改为只声称配额守恒得到验证｜修复：取方案 (a)：修正 `solve_reroute` 的比例分摊，在 `share` 上增加 `need` 上限并补一轮「取整残差按剩余配额补齐」，使两个分支都满足双向守恒；主程序增加逐专家的源侧与配额两侧断言。重跑后关闭本地优先的值由错误的 62.5% 变为 54.2%，页面代码块、预期输出、观察重点、4.6 节与 4 章问题 3 的四处引用同步更新；简化条件补充说明该分支的取整残差处理。｜复验：重跑 solver.py 退出码 0，断言全部通过，输出「本地优先：双向守恒成立，跨 rank 10/24 = 41.7%」「关闭本地优先：双向守恒成立，跨 rank 13/24 = 54.2%」，与页面预期输出块逐字符一致；观察重点现声称的是「两个分支都断言了双向守恒并通过」。

- [轻微·格式] 1.3 节首段（index.html:848）：「大 EP 直接把专家之间的路由动态translate 成明显的 rank 间倾斜」中残留未翻译的英文词 `translate`，且与前文之间无空格。｜引文依据：不适用｜修复要求：把 `动态translate 成` 改为 `动态转换成`（或「转成」），与 1 章本章问题第 2 题解答中已使用的「转成」保持一致｜修复：改为「路由动态转化成明显的 rank 间倾斜」。｜复验：grep `translate` 在正文中 0 命中（仅 CSS 的 translateY 保留）。

- [轻微·技术] 6.4 节逐项解读段（index.html:1718）：在「论文的逐项解读（§8.3）」之下给出「前向 MoE 计算是 3.15 倍、token all-to-all 是 4.13 倍」。这两个倍数论文未写出，是由 Figure 13 标注值相除得到（$5.95/1.89=3.15$、$6.94/1.68=4.13$，复算正确），但被置于论文表述的归属之下，且来源与范围说明的派生结论清单（index.html:1867）只列了 0.33 ms、1.8%、+33%、+10%，未收录这两个数。｜引文依据：§8.3 原文仅为「Without balancing, Megatron-LM suffers large inflation in both MoE compute and token all-to-all」，无倍数｜修复要求：把这两个倍数标为由 Figure 13 标注值相除得到的页面派生值，并在 index.html:1867 的派生结论清单中补入｜修复：改为「按上表数值计算，相对 ideal 其前向 MoE 计算是 3.15 倍、token all-to-all 是 4.13 倍（这两个倍数由本页从表中数值算出，论文只作定性表述）」。｜复验：该句现明确标注为本页派生，不再归入「论文的逐项解读」的论断范围。

- [轻微·技术] 1.1 节提示框（index.html:837）：写「论文的标题、摘要与 §3 都把范围限定在训练与 serving prefill」。摘要与 §3 成立，但标题用的是「Training and Inference」，并未限定到 prefill。｜引文依据：标题「UltraEP: Unleash MoE **Training and Inference** on Rack-Scale Nodes with Near-Optimal Load Balancing」；摘要「the first exact-load, real-time balancer for large-EP MoE **training and serving prefill** on rack-scale nodes」｜修复要求：删去「标题」，改为「论文摘要与 §3 都把范围限定在训练与 serving prefill（标题用的是更宽的 Training and Inference）」｜修复：删去「标题」，改为「论文的摘要与 §3 都把范围限定在训练与 serving prefill」。｜复验：核对标题为 "Unleash MoE Training and Inference on Rack-Scale Nodes..."，确实未限定 prefill；页面现只声称摘要与 §3。

- [轻微·格式] 6.2 节训练表格末行（index.html:1659）：该行用 `colspan="4"` 占掉「方法」「GLM4.5-106B」「Qwen3-235B」「DeepSeek-V3」四列，使三个逐模型达标率「96.4% / 91.2% / 96.1%」落在表头「平均相对 Megatron-LM」这一列下，列义与内容不对应。｜引文依据：不适用｜修复要求：改为把三个比例分别填入对应模型列、最后一列留「—」，或把该行改为表格下方的说明句｜修复：把该行改为四个独立单元格：三个逐模型达标率各占其模型列，末列补 94.6% 对应「平均」列。｜复验：该行现为 5 个 td，与表头 5 列对齐；96.4%/91.2%/96.1% 分别落在 GLM4.5-106B、Qwen3-235B、DeepSeek-V3 列下，末列 94.6% 与 §1 的训练均值一致。

- [轻微·格式] 全页配额符号（index.html 中 $u_{e,r}$ 出现于 7、757、784、1066、1070、1072、1085、1116、1117、1123、1139、1381、1462、1897 行，$u_{e,t}$ 出现于 1050、1375、1381、1462、1857 行）：同一变量在下标上混用 $r$ 与 $t$，其中 1381 与 1462 两行在同一段内同时出现两种写法。原文 Table 1 与 §4.3 Output 段也各用一种，但页面未说明两者同义。｜引文依据：Table 1「$U=\{u_{e, r}\}$」；§4.3 Output「A quota-aware replication plan $U=\{u_{e,t}\}$」｜修复要求：全页统一为一种写法（建议随 Table 1 用 $u_{e,r}$），或在 3.6 节符号说明处补一句「$t$ 与 $r$ 同为 EP 组内 rank，$t$ 用于强调它是接收目标」｜修复：在 3.6 节问题形式化处补一句说明：配额的第二个下标指承载该实例的 rank，讨论「某 rank 总负载」时写 $u_{e,r}$、讨论「某源发往某目标」时写 $u_{e,t}$，两者是同一张表 $U$ 的元素，下标字母只反映该 rank 在当前语境中的角色。｜复验：该说明位于两种写法首次并列出现之前；全页两种写法的用法与该说明一致，未出现同一语境混用。

- [轻微·格式] 5.7 节通信表格表头与 4.9 节 Figure 15 解释段（index.html:1559、1430）：使用「初始不均衡度」「不均衡度分布」而未标明是专家级还是 rank 级，与术语速查中「本页每次使用时说明是哪一种」的承诺不符。｜引文依据：页面术语速查（index.html:788）「论文有两种口径：专家级（最大专家负载比均值）与 rank 级（最大 rank 负载比均值），本页每次使用时说明是哪一种」｜修复要求：在这两处补注口径，或在术语速查中把承诺改为「除注明外均指 rank 级」｜修复：5.7 节表格前引导句与 4.9 节 Figure 15 解释段均补明口径：Table 3 的结果不均衡为 rank 级，Figure 15 上排为 rank 级不均衡度分布。｜复验：两处现均带口径标注，与术语速查中「本页每次使用时说明是哪一种」的承诺一致。

- [轻微·技术] 4.9 节末段（index.html:1432）：「实例总数 45 对 107 这一项差距，主要就来自 $u_{\min}$ 剪掉的那些无效副本」是页面推断，论文只把差距归因于「只在副本能带来足够均衡收益时才实体化」，未点明 $u_{\min}$ 是主因，该句未随文标注为推断。｜引文依据：§8.5「Unlike EPLB$+$, which blindly replicates experts based on pre-reroute hotness, \sys only materializes a replica when it brings sufficient balancing gain. This accounts for \sys's resource efficiency」，无 $u_{\min}$ 归因｜修复要求：该句加推断标注，或改写为论文原有归因（只在副本能带来足够均衡收益时才实体化）｜修复：在该段末补注：论文把这项差距归因于「只在有足够收益时才实体化副本」这一整体设计，把它进一步落到 $u_{\min}$ 这一具体机制上是本页的推断。｜复验：该推断现已随文标注，与页面末尾「论文事实与分析性判断」小节的处理方式一致。

## 已核对通过的主要项目（不计为问题）

- 元信息：`description`、`dojo:summary`、`dojo:type=paper`、`dojo:topics`（并行与通信 / 推理系统 / 模型结构，均在 AGENTS.md 词表内）、`dojo:tag` 齐备；`validate.py` 对两个页面均返回 `validation ok`
- 版本声明：v1、v2 已撤回、v3 于 2026-06-18 提交，与 arXiv 提交历史一致
- 原图对应：11 张图的 Figure 编号与 arXiv HTML v3 编号逐一相符（img-01=F1、02=F2、03=F6、04=F7、05=F8、06=F10、07=F15、08=F16、09=F11、10=F13、11=F17）；11 张 webp 的宽高比与对应 PDF 页面宽高比一致，未见裁剪
- 实验数字：Figure 11（545/646/618/695/757 等 15 个 TFLOPS 值与 15 个不均衡度）、Figure 12（16 个不均衡度）、Figure 13（18 个延迟值）、Figure 17（504 与 425.0）、Table 3（5 项）、Table 2（4 个模型配置）逐值与图内标注或原文一致；达标率 96.4%/91.2%/96.1% 与均值 94.6%、总平均 94.3%、生产反推 466 与 92.4% 均复算通过
- 代码：折叠块内 Python 实际运行，退出码 0，输出与页面「预期输出」逐行一致；$u_{\min}=3$ 的反例（最终阈值退到 7）与页面 callout 一致；`locality=True` 分支的双向守恒成立
- 公式：Eq.(1)–(6)、二分区间初始化、$\delta$ 三重最小值、比例分摊式、双向守恒式、中继前沿 $\sqrt{\lvert\mathcal{H}(e)\rvert-1}$ 与原文逐一相符；3.6 节折叠块对三个代价模型的代入检查（12→6、12→8、0→2）复算通过
- 问题块：页面级 h2 为「核心问题」、6 个正文章节末尾 h3 均为「本章问题」，两级共 4+3+3+3+4+3+4=24 题，每题均有 `解答：` 前缀的 `<details>`，两级列表均用 `<ol class="chapter-questions">`；核心问题 4 题的答案末尾均指明完整论证所在章节
- 页面功能：无失效同页锚点、无重复 id；`libs/` 下 7 个本地资源全部存在；自绘图为 HTML 结构（`dg-flow` / `dg-stack`），样式类均已在页内定义，无等宽字符框线图；正文、summary、表格中无裸 Unicode 数学字符（仅出现作分隔号的 `·`）
- 章节编号：h2 为 1.–7. 连续，h3 为 x.1 起在各章内连续，问题块与来源说明不编号，符合 write.md 第 4 节
- 前置概念链接：moe-serving、aux-loss-free-routing、gpu-communication、model-parallelism、chunked-prefill、deepseek-moe、gpu-execution-model、moonep 共 8 个目标页均存在
- 不确定信息处理：对「2560 张 GPU 生产训练」这一站外说法明确说明无法在 v3 中定位、不予引用，处理结果清楚
- overview 与 index 相互链接成立（overview 顶部与末尾链向 index，index 导航链向 overview）

## 结论

- 统计：阻断 2 / 重要 5 / 轻微 7
- 处置：修复

修复要点：两条阻断分别是 4.5 节第一次探测的空闲向量算错，以及 Table 3 中 $\sum_e\lvert\mathcal{H}(e)\rvert$ 被误读为「实例总数」（应为消耗的冗余槽数，涉 index.html 6 处与 overview 1 处）。五条重要集中在来源定位与证据基础：Algorithm 1 行号错位、Figure 6 右面板的描述与图和原文相反、Figure 15/16 的读图估计值被标为「标注值」、3.9% 的基线错位、关闭本地优先的对照代码违反每源需求守恒。全部问题的修改范围均限于问题位置及其直接引用位置，不涉及内容范围与大纲，无需返回规划文件。

## 修复后复验

- 全部 14 条（阻断 2 / 重要 5 / 轻微 7）均已修复并逐条复验，结果记录在上方各条的「修复」与「复验」栏。
- `python3 .dojo/scripts/validate.py wiki/ultraep/index.html` → validation ok；`wiki/ultraep/overview.html` → validation ok。
- 代码复跑：`python3 /tmp/ultraep/solver.py` 退出码 0，新增的双向守恒断言全部通过，输出与页面预期输出块逐字符一致。
- 阈值复算脚本确认三次探测的 exc/slk 与页面现值逐项一致。
- 修改范围限于问题位置及其直接引用位置，未改变内容范围与大纲，未返回规划文件。
