<!-- review-meta
round: 4
page: wiki/eagle-speculative/index.html
reviewed_content_sha256: 589c3ddcf9d1b230
-->
# EAGLE-3 投机解码 draft 模型审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 b10fe925e8201cd534dfe4aa335eac5ade4413fc
- 审查时间：2026-09-13 19:35
- 审查者：独立子代理（未参与写作与任何前序审查）
- 依据规范：guides/concept/check.md（页面 dojo:type=concept）
- 机械核对：`.dojo/scripts/validate.py wiki/eagle-speculative/index.html` 返回 `validation ok`；head 含 description / dojo:summary / dojo:type=concept / dojo:topics=训练与优化（在 AGENTS.md 固定大类内）/ dojo:tag=推理加速（在 ALLOWED_TAGS 内）；overview.html 与 index.html 互链；页面无指向 research/ 的路径；内链 wiki/speculative-decoding/index.html、wiki/mxfp4-qat/index.html 均真实存在，且被引用的章节（「Draft-then-Verify」「为什么这条规则能保分布」「工程实例与边界」）在对应页真实存在。
- 已完整阅读章节（含折叠块与图注）：核心问题（5 题解答）；1. 为什么独立小模型 draft 有两难——EAGLE 的思路转向；2. EAGLE-1 的核心机制——在 feature 空间做自回归；3. EAGLE-3 的两项架构改变（3.1 直接 token 预测 / 3.2 多层特征融合 / 3.3 架构总览）；4. 推理时的自回归 draft（4.1 prefill / 4.2 步骤 1 / 4.3 步骤 2 起 / 4.4 构造示例 + 中间值折叠块 + 伪代码折叠块）；5. 训练 draft 模型（5.1 TTT / 5.2 两种损失 / 5.3 LK vs KL）；6. 工程实例——K3 的 EAGLE-3 部署与边界（6.1–6.4）；来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）。
- 复算与回源结果（无问题项，摘要）：加速比公式 S=(1−α^{γ+1})/[(1−α)(1+γc)] 与 Leviathan arXiv:2211.17192v2 Theorem 3.8 原文一致；α=Σmin(p,q)=1−TV(p,q) 复算成立；构造示例（prefix="How can"、k=4、词表 5、γ=3）三步的 W_a/W_lm 乘法、tanh、softmax 与 L2 距离（≈0.380）逐行复算一致，分项之和与合计、argmax 与结论均相符（唯一差异是四舍五入末位）；[C1][C2][C3][C4][C5][N1][N2][N4][N5][C7][C8][C9][F1][F2][F6] 的引文均能在来源定位到原文片段（EAGLE-1 §1/摘要、EAGLE-3 摘要/§3.1/§3.2/§4、K3 arXiv:2607.24653 §2.2/§4.1.4）；数字 5.6x、1.8x、3x、6.5x、1.4x、1.38x、2.7x-3.5x、3000B/2-4B tokens、TTT 7 步、Table 1 的 3.07x/5.58x 均与来源核对一致；K3 报告确为 arXiv:2607.24653《Kimi K3: Open Frontier Intelligence》，§2.2 为 Attention Residuals、§4.1.4 含 Draft Model Fine-Tuning，EAGLE-3 确为 NeurIPS 2025（poster）。

## 问题

- [重要·技术] 来源与范围说明 [C10]（约第 729 行）：定位编号「Leviathan et al. 2023 (arXiv:2211.17192) §3 Theorem 2」在来源中不存在——该论文 §3 的定理只有 3.5、3.8、3.11，不存在 Theorem 2；"保分布/lossless" 对应的是 Theorem 3.5 与随后紧随的正确性证明（或附录 A.1 Correctness of Speculative Sampling）。｜引文依据：arXiv:2211.17192v2 全文出现的定理编号仅 `Theorem 3.5. β=1−D_LK(p,q)`、`Theorem 3.8. The expected improvement factor in total walltime by Algorithm 1 is ...`、`Theorem 3.11.`（全文 "Theorem" 共 9 处，全部为 3.x，无 "Theorem 2"）。｜修复要求：把该定位改为 Theorem 3.5（或附录 A.1），或删去该定位、仅保留 [C10] 其余可定位依据（EAGLE-1 摘要 "maintaining the distribution of the generated text"、EAGLE-3 §4 "strict speculative sampling acceptance conditions, ensuring no loss in performance"）。｜修复：｜复验：
- [重要·技术] §1 正文（约第 117 行）与本章问题第 1 题解答（约第 143 行）：「7B 自身单步成本 c 不可忽略（约 0.05-0.1）」这一具体数值在本页任何标注来源中都定位不到，却被与 [C1] 同句呈现，读起来像来源结论。｜引文依据：EAGLE-1 §1 仅有 "Despite the 7B model's potential as a draft model, its high overhead diminishes acceleration gains."；EAGLE-1 全文 "0.05" 只出现在参考文献 NEFTune 标题、无 7B draft 的 c 数值；Leviathan 仅讨论 negligible-cost 模型（"c ... often negligibly close to 0"），未给出 7B→70B 的 c。｜修复要求：为该估值补可定位来源，或明确标注为按参数量比（7/70≈0.1）得到的量级估算/推断，或删除该数值、只保留 [C1] 的定性结论。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 [C6]（约第 721 行）：引文把 K3 报告原文的引用编号抄错。｜引文依据：K3 报告 §4.1.4 原文为 "we directly optimize the likelihood-based LK loss [ 103 ] , the negative logarithm of the acceptance rate itself"，本页写作 "[104]"。｜修复要求：将引文中的 [104] 改回 [103]（其余引文文字已与原文一致）。｜修复：｜复验：
- [轻微·技术] §5.1 TTT 因果 mask 图（约第 523-563 行）与简化条件说明（约第 765 行）：来源说明称该图展示「第 1 轮真实、第 2、3 轮自替代」的时序，但表中只有两行预测（"第 1 轮预测"、"第 2 轮预测"），仅体现一轮自替代；论文 Figure 6 为 1 个 native + 2 个 simulated。｜引文依据：论文 Figure 6 图注 "It sequentially shows a native training step (the first step) and two simulated training steps (the second and third steps)."｜修复要求：补齐一行「第 3 轮预测」使图与说明、与论文 Figure 6 一致，或把简化条件一栏改为「第 1 轮真实、第 2 轮自替代」，使图与文字对齐。｜修复：｜复验：
- [轻微·技术] §3.2（约第 238 行）与本章问题第 2 题解答（约第 303 行）：「丢失了低层句法/词法、中层语义信息」把层的语义分工写成了来源事实。｜引文依据：EAGLE-3 论文 §1 只写 "we integrate and leverage low-, mid-, and high-level features from the target model, capturing rich semantic information from different layers"，并未把 low 层对应「句法/词法」、mid 层对应「语义」。｜修复要求：改用论文口径（不同层承载不同层次的信息），或将「句法/词法 vs 语义」明确标注为解释性推断。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（重要问题须关闭后再复验；两条重要问题分别涉及一条无法定位的定理编号与一个无来源支撑的数值，均可通过补来源/改定位/降级为标注推断修复；本轮未发现阻断问题。）

## 说明（未报为问题、经核对确认合格的项）

- 「本页」在全文仅 4 处，且都在「来源与范围说明」中作记号来源的限定语（"本页对 $L_{E3}$ 的形式化见 [F3]" 等），与站点既有概念页（speculative-decoding 11 处、mxfp4-qat 17 处）用法一致，不属元话语式自我指代；「下面是…」3 处为折叠块/表格的引导语，亦与既有页面写法一致。
- Unicode 字符 γ、←、∈ 仅出现在伪代码代码块内部（及正文对其单行引用 <code>g_seq ← g_seq + [a]</code>），标题/summary/正文散文/列表/表格中的数学符号均由 KaTeX 渲染，无 Unicode 数学字符直接出现。
- 结构图为 HTML（dg-flow/dg-stack div 与 table），非等宽框线图；两处来源图注、图内符号已定义。
- 核心问题 5 题与 6 个章节的「本章问题」均有解答折叠块，核心问题答案均指明完整论证所在章节。
