<!-- review-meta
round: 4
page: wiki/ultraep/index.html
reviewed_content_sha256: 6178e9bf7ab105de
-->
# UltraEP 审查记录（第 4 轮）

- 页面版本：54da7e743ded695e4bab53cfc0d2d1aca3feea34（index.html 工作树哈希）
- 论文版本：arXiv:2606.04101v3（2026-06-18 提交；v1/v2 已撤回）
- 审查时间：2026-09-13 21:23
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：术语速查 / 贯穿全文的最小例子 / 1. 专家热度在层与批次之间快速漂移，历史统计撑不住 / 2. 精确负载只能在 gating 之后拿到，机架级节点让这件事第一次划得来 / 3. 只复制不重排：把冗余槽做成跨层复用的临时缓冲 / 4. 配额把「建不建副本」和「接多少 token」并成一个变量 / 5. 把不规则的专家搬运做成 tile 流水线与 chunk 中继 / 6. 实验结果：均衡后逼近强制均衡上界，剩余差距来自路由本身 / 7. 方法评价：优化目标改对了，但硬件前提很强 / 来源与范围说明（含全部折叠块与图注）

## 本轮核对依据（摘要）

- 官方材料：arXiv HTML v3（https://arxiv.org/html/2606.04101v3）与 PDF v3（12 页，pdftotext/pdftoppm 提取），逐条定位 §1–§10、Table 1–3、Figure 1/2/6/7/8/10/11/12/13/15/16/17、Eq.(1)–(6)、Algorithm 1。
- 数字核对（原文片段）：摘要 "94.3 % ... 1.49× ... 1.30–4.01 to 1.01–1.04"；§1 "94.6 % ... 93.9 % ... 1.42× over Megatron-LM ... 1.56× over SGLang ... 3.1×–5.5×"；§4.1 "3.3 GB weights and 6.6 GB gradients to 36 MB and 72 MB per rank"；§8.3 "extra latency is 0.33 ms in forward and negligible in backward, which constitutes only 1.8 % of the total latency ... increases latency by 33 % and 10 %"；§8.4 "2× and 11× higher peak memory"；§8.5 "up to 1.4, while UltraEP still keeps it below 1.1 ... 27.4 % ... 57.9 % ... 3.9 %"；§8.6 "over 92 % ... 9.6 % average gain"。
- 图读数核对：Figure 16 逐柱高度（1.5/2.0/4.0/6.0/8.0 五档）与页面 §5.7 表一致；Figure 15 下排 (128,64,1) 组 6.0/8.0 档 EPLB+ 约 1.40、Ours 约 1.07/1.09 与页面一致；Figure 13 六个标注值与页面 §6.4 表逐格一致；Figure 11/12 柱标注值与页面 §6.2/§6.3 表逐格一致；Figure 17 标注 "Ideal (504)" 与 "w/o balancing (mean: 425.0)" 与页面一致。
- 复算：Figure 11 逐模型 IDEAL 比值与「平均相对 Megatron-LM」+20%/+12%/+29%/+42% 均可由表中数值复算（757/785.4=96.4%、524/574.7=91.2%、613/637.6=96.1%，平均 94.6%；三模型平均比值 1.2009/1.1233/1.2888/1.4189）；§6.4 的 +0.22/+0.33/+0.11/+0.09 ms、+33%、+10% 均由表中差值算出，0.33/(7.51+10.80)=1.80%。
- 代码：抽出页面折叠块中的 Python 原样执行，输出与页面「预期输出」逐行一致；另把 U_MIN 改为 3 重跑，复现页面「阈值从 6 退到 7」的说法。
- 机械项：validate.py 通过；无 alt 含 `$...$`；标题/summary/正文/表格无 Unicode 数学字符；8 个前置概念链接（moe-serving、aux-loss-free-routing、gpu-communication、model-parallelism、chunked-prefill、gpu-execution-model、deepseek-moe、moonep）目标文件均存在；overview.html 与 index.html 互链，两页 94.6%/93.9%/94.3%/1.49×/1.42×/1.56×/1.29×/1.8%/3.1–5.5×/2×/11×/92%/9.6% 全部一致。

## 问题

- [轻微·表述] §2.2 引导句、§2.4、§4.9 图注、§5.7 图注、§6.2 图注、§6.4 图注、§6.5 正文与折叠块、§7.1、§7.2 共约 10 处：成对出现的元话语与读者指代——"需要注意 EPLB 曲线是否有超过 no-balancing 的位置""需要关注的是均衡决策发生在时间轴的哪个位置""需要关注冗余槽预算变紧时……""三个读法值得注意""折线图里需要关注的是……""需要关注的是 UltraEP 与 Ideal 两根柱的差异分布""实验条件需要注意""值得注意的是三个机制之间的依赖关系""与冗余槽预算的关系需要注意"。｜引文依据：不适用｜修复要求：改写成陈述句，直接给出该处要传达的判断（例如 §7.2 改为"两者口径不同：训练侧关闭活跃值重计算……"），删除"需要注意/需要关注的是/值得注意/读数注意"这类对读者的提示语；figure 引导句保留"该图给出……"的事实陈述即可。｜修复：｜复验：
- [轻微·表述] 术语速查「不均衡度」行、§贯穿示例开头、§3.6：以"本页"为主语的自我指代——"本页每次使用时说明是哪一种""本页从这里开始，后续第 3、4、5 章都会回到它""本页在讨论「某 rank 的总负载」时写作 $u_{e,r}$"。｜引文依据：不适用｜修复要求：改为直接陈述（如"后续第 3、4、5 章都会回到这个例子""第 4 章讨论 rank 总负载时写作 $u_{e,r}$，讨论源到目标时写作 $u_{e,t}$"）；"来源与范围说明"里用于区分论文事实与页面推断的"本页的推断""由本页从图中读取"属于该章必需的功能性表述，可保留。｜修复：｜复验：
- [轻微·技术] §6.5 正文末句与同节折叠块末句：页面在引用 §8.4 之后自行补了一句因果解释——"serving 那个更大的倍数部分来自它只有单层活跃值这个基数更小的分母"。｜引文依据：论文 §8.4 原文为 "For training, we disable activation checkpointing to expose the layer-accumulated activation upper bound ... In contrast, serving only keeps transient activation of the current layer during forward. Without balancing, we observe 2× and 11× higher peak memory of MoE activation than the ideal for training and serving, respectively."，只说明两侧口径不同，未给出倍数差异的成因；页面「来源与范围说明」把第 1–6 章内容声明为论文事实，并只列出四处随文标记的推断，不含此处。｜修复要求：把该因果从句标注为"本页推断/辅助解释"并收进来源说明的推断清单，或删除该从句、只保留"2 倍与 11 倍不是同一口径"这一由 §8.4 支持的陈述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（无阻断、无重要；遗留 3 处轻微为表述与推断标注问题，不影响核心结论、原文一致性与阅读连续性。§8.4 的 2×/11×、§8.3 的 0.33 ms 与 1.8%、§8.5 的 1.19/1.03 与 107/45、Figure 11/12/13/15/16/17 的全部标注值、Algorithm 1 的三次探测与 $u_{\min}$ 对照均已在 v3 原文与图中逐条核对无误；页面折叠块代码实际执行，输出与页面描述逐行一致）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
