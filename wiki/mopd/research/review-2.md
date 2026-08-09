# MOPD 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟 / 小白读者视角）
- 页面版本：index.html @ ac5b744（2026-08-09）
- 时间：2026-08-09
- 审查范围：段 A 盲读（index.html + overview.html）+ 段 B 对照来源（K3 报告 §4.1.3 "Multi-Teacher On-Policy Distillation" 行 896–921 + Eq.(15) + §4.1.2 "Reinforcement Learning" 行 825–885）

## 段 A 盲读

按页面顺序阅读 index.html，扮演完全小白读者，记录理解主线上的卡点。

**引言 + context-box**：MOPD 定位清晰（两阶段流程的第二阶段——合并，非训练专家）。九个教师 = 3 领域 × 3 努力程度。Eq.15 在 context-box 预告。

**S1（为什么要把九个专家合成一个统一模型）**：三层成本（九份显存、路由难题、能力不共享）清晰。简单平均不可行的理由（参数空间无对应）清晰。统一学生条件化于 e 的概念引入。三方案对照表清晰。误解 callout（MOPD 不训练专家）清晰。小白可跟上。

**S2（学生自己生成的 token，由教师逐个打分）**：on-policy 含义解释（训练轨迹来自学生当前策略）。前置概念（知识蒸馏、策略梯度/RLHF）用 callout 标注"尚未生成"并给最小结论。Eq.15 完整呈现，逐符号说明（11 个符号各一行）。对数比值拆解 log(π_teacher/π_θ) = log π_teacher − log π_θ 清晰。ASCII 数据流图展示循环。三 token 教学例子（学生 (0.5,0.3,0.2)、教师 (0.7,0.2,0.1)、Rmax=5）逐 token 手算。折叠块含未四舍五入中间值。小白可跟上。

**S3（裁剪和停梯度让奖励信号稳定可控）**：sg 解释（阻断对数比值求导路径，梯度只走 ∇log π_θ）清晰。两个误解 callout（"教师梯度回传到学生"错、"clip 限制实际概率比"错）清晰。clip 裁剪的是奖励信号而非实际概率比——明确区分。极端概率比对照（Rmax=5 vs Rmax=3）手算。折叠块含多行表格展示裁剪阈值行为。小白可跟上。

**S4（九个教师按领域和努力程度轮流指导）**：按 (d,e) 选教师清晰。ASCII 图展示九个教师。学生条件化于 e 但不条件化于 d 的推断明确标注为推断（"K3 报告未明文陈述该部署设计意图"）。RLHF 对比表（奖励来源、粒度、训练栈、能力上限）清晰。"无缝接入 RL 框架"解释（partial rollout、per-token 正则化复用）清晰。折叠块含 mini-batch 配对示例。小白可跟上。

**S5（MOPD 能做什么、不能做什么）**：四条边界（不能凭空产生能力、学生以教师为上限、top-k 无优势、Rmax 两面性）清晰。能力边界对照表清晰。适用条件三条总结。top-k 误解 callout（"更精细不等于更好"）清晰。小白可跟上。

**学习目标核对**：
1. 为什么合并九个专家、简单平均为何不可行 → S1 完整回答 ✓
2. 学生 token 如何变成奖励信号、Eq.15 每个符号 → S2 完整回答 ✓
3. R_max 和 sg 各自解决什么问题、去掉会怎样 → S3 完整回答 ✓
4. 九个教师如何轮流指导、稠密奖励如何接入 RL → S4 完整回答 ✓
5. MOPD 不能做什么 → S5 完整回答 ✓

段 A 未发现阻断或重要卡点。

## 段 B 对照来源

逐条核对页面表述与 K3 报告 §4.1.3（行 896–921）及 §4.1.2（行 825–885）的一致性。

**定义与机制**：
- C1（9 专家 = 3 领域 × 3 努力程度 {low,high,max}）：报告 §4.1.2 行 827–834 "we scale RL across three broad domains ... (i) general tasks ... (ii) general agents ... (iii) coding agents ... Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models" ✓。页面"三领域是通用任务、通用 agent、编码 agent"与报告 (i)(ii)(iii) 逐项对应 ✓
- C2（MOPD 合并九专家进统一学生、按 (d,e) 选教师）：报告 §4.1.3 "We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model" + "for a given domain d and a sampled reasoning effort level e ∈ {low, high, max}, optimization is guided by the corresponding teacher model π_teacher^(d,e) among the nine experts" ✓
- C3（per-token OPD 奖励公式 Eq.15）：报告 §4.1.3 Eq.(15)——逐符号对照：r_opd^d(yt|e,x,y<t) = clip(sg(log(π_teacher^(d,e)(yt|x,y<t) / π_θ(yt|e,x,y<t))), -Rmax, Rmax)。页面公式与报告完全一致，包括教师不含 e 在条件中（已由 (d,e) 选中）、学生含 e 在条件中 ✓
- C4（sg 停梯度、Rmax 裁剪阈值）：报告 §4.1.3 "sg(·) denotes the stop-gradient operator, and Rmax > 0 is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training" ✓
- C5（稠密奖励无缝集成、支持 partial rollout）：报告 §4.1.3 "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks" ✓
- C6（top-k 无优势）：报告 §4.1.3 "we also experimented with more fine-grained top-k distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting" ✓
- C7（专家先由 RL 训练、轨迹联合收集用于 SFT 和 MOPD）：报告 §4.1.2 行 884–885 "Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation" ✓
- partial rollout 机制描述：报告 §4.1.2 行 864–869 "the generation phase pauses as soon as a fraction λ ∈ (0, 1) of trajectories completes ... Paused rollouts are enqueued and prioritized for resumption" ✓
- per-token 正则化描述：报告 §4.1.2 行 872–874 "Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. By constraining policy updates within a localized neighborhood" ✓
- policy optimization follows K2.5：报告 §4.1.2 行 870 "policy optimization, which follows the algorithm in Kimi K2.5 [59]" ✓

**公式与推导**：
- F1（Eq.15 per-token OPD 奖励）：来源 K3 报告 §4.1.3 Eq.(15)，逐符号对照一致 ✓
- F2（对数比值分解 log(π_teacher/π_θ) = log π_teacher − log π_θ）：由对数除法性质直接推出 ✓
- 三 token 手算验证：学生 (0.5,0.3,0.2)、教师 (0.7,0.2,0.1)、Rmax=5
  - A: log(0.7)−log(0.5) = −0.356675−(−0.693147) = +0.336472，clip(0.336,−5,5) = +0.336 ✓
  - B: log(0.2)−log(0.3) = −1.609438−(−1.203973) = −0.405465，clip = −0.405 ✓
  - C: log(0.1)−log(0.2) = −2.302585−(−1.609438) = −0.693147，clip = −0.693 ✓
  - 折叠块未四舍五入中间值逐项复算一致 ✓
- 裁剪阈值对照（π_θ(A)=0.01、π_teacher(A)=0.9、log(0.9/0.01)≈4.50）：
  - Rmax=5: clip(4.50,−5,5)=4.50 ✓
  - Rmax=3: clip(4.50,−3,3)=3 ✓
  - 折叠块表格（π_θ 从 0.5 降到 0.001）逐行复算一致 ✓

**可运行代码**：页面无可运行代码（仅 ASCII 图和手算例子），无需执行 ✓

**事实与推断**：
- N1（9 个专家）、N2（{low,high,max}）：报告 §4.1.2 + §4.1.3 ✓
- 页面明确标注"K3 报告本节未给出 MOPD 的具体性能数字、Rmax 的取值或 top-k 对比的具体数值；本文不构造或引用任何此类数字" ✓
- "部署时学生不需要领域路由器"推断：页面明确标注"基于公式形式 π_θ(yt|e,x,y<t) 不含 d 推出的工程推断，非 K3 报告明文结论" + 失效边界 ✓
- "r_opd 作为优势 A"的教学解释：页面在 callout 中标注为"最小结论"（依赖前置概念页，未展开推导），并在教学说明中标注"实际更新还受 advantage 归一化、K2.5 per-token 正则化等因素影响，本文只描述方向不描述幅度" ✓

**前置知识引用**：知识蒸馏、策略梯度/RLHF 均标注"概念页待生成"并给最小结论 ✓

**教学简化**：三 token 词表标注为教学构造；Rmax=5/3 标注为教学值、报告未披露实际取值；mini-batch 示例省略 advantage 归一化等标注 ✓

**页面功能**：validate.py 退出码 0 ✓

## 问题

无。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布

段 A 盲读未发现阻断或重要卡点，学习目标全部由正文章节完整回答。段 B 对照来源逐条核对，核心论断（C1–C7）与报告 §4.1.3 + §4.1.2 一致——尤其三领域名称（通用任务/通用 agent/编码 agent）与报告 (i)(ii)(iii) 逐项对应、Eq.15 逐符号对照完全一致、partial rollout 与 per-token 正则化描述与 §4.1.2 一致。三 token 手算和裁剪阈值对照逐项复算一致。所有推断（部署设计意图、r_opd 作为 advantage）均明确标注为推断或最小结论并给出失效边界。validate.py 退出码 0。关键论断和数字已重新对照外部来源。
