# Quantile Balancing 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html blob 09f59b51381c1f185fa8b2a1a3a53c8c051dec8e
- 时间：2026-08-09

## 审查方法

段 A 盲读：以小白视角按页面顺序阅读 index.html 和 overview.html，记录理解卡点，不对照来源。
段 B 对照来源：逐条核对页面表述与 K3 技术报告（§2.3.3 L547-604、Appendix C L2784-2895、Appendix D L2897-2939）及 HuggingFace config.json 的一致性。重点核查 Eq.13/14、m=8/n=4/k=1 手算例子、histogram 估计、与 DeepSeek-V3 sign 更新对比。手算例子代码已实际执行，输出与页面预期输出完全一致。

## 问题

- [重要·技术] index.html S2 L739 "专家数从 DeepSeek-V3 的 256 增到 896"：K3 报告 L748 对比表显示 K2.5→K3 为 384→896，而非 256→896。256 是 DeepSeek-V3 的正确专家数，但 K3 报告未做 DeepSeek-V3→K3 的对比，K3 从 K2.5（384 专家）演化而非 DeepSeek-V3（256 专家）。该对比不在来源中且具误导性。S7 来源部分（L1229 N1）仅标注 896 来自 config.json，未标注 256 的来源。：将"专家数从 DeepSeek-V3 的 256 增到 896"改为"K3 将 routed expert 池从 K2.5 的 384 扩大到 896（K3 报告 Table 对比，L748）"，若需保留 DeepSeek-V3 参照则另起一句"DeepSeek-V3 采用 256 专家（arXiv:2412.19437），K3 的 896 进一步扩大了规模"并标注来源。 ｜ 修复：将"专家数从 DeepSeek-V3 的 256 增到 896"改为"专家数从前代 K2 的 384 扩大到 896（K3 报告 Table 1）"（来源 Table 1 实际为 K2→K3 对比，故用 K2 而非 K2.5）；并在 S7 N1 补充"K2 的 384 专家来自 K3 报告 Table 1（K2→K3 对比）"。 ｜ 复验：

- [重要·技术] index.html S6 L1135 r 与 margin 符号不一致：S3（L785）定义 margin = s_{i,j} - α_i，S6（L1135）称 r = α_i - s_{i,j} 为"margin"（原文"所有 margin r_{i,j} = α_i - s_{i,j}"），但 r = -margin。来源 Appendix D（L2907）称 r 为"required bias"而非 margin。伪代码注释（L1160 "r = α - s = -margin"）有澄清，但正文 L1135 将 r 直接等同于 margin，与 S3 定义矛盾，可能造成读者在分位数取 k/n 还是 1-k/n 时混淆。：在 L1135 首次引入 r 时明确写"r_{i,j} = α_i - s_{i,j} = -margin_{i,j}（注意与 S3 的 margin 符号相反）"，并将正文"所有 margin r_{i,j}"改为"所有 r_{i,j}（= -margin_{i,j}）"。 ｜ 修复：将 L1135"所有 margin r_{i,j} = α_i - s_{i,j}"改为"所有 r_{i,j} = α_i - s_{i,j} = -margin_{i,j}（与 S3 的 margin 符号相反；来源称 r 为 required bias）"。 ｜ 复验：

- [重要·盲读] index.html S5 L1098-1102 SignSGD 论断依赖折叠块中的 b=-β：正文论断"DeepSeek-V3 的 sign 更新 b_j ← b_j + γ·sign(ℓ̄ - ℓ_j) 恰好是对这个梯度做 SignSGD"依赖关系 b_j = -β_j，但该关系仅在折叠块（L1093"因为 b_j = −β_j"）中给出。来源（Appendix C L2886）明确说"up to the sign convention b = -β"。读者不展开折叠块无法验证此论断的推导链：∂L/∂β_j = ℓ̄-ℓ_j → SignSGD 对 β 做 β←β-γ·sign(∂L/∂β) → 因 b=-β 得 b←b+γ·sign(ℓ̄-ℓ_j)。：在正文 L1100-1102 补充"（因 b = -β，对 β 的梯度下降等价于对 b 的 sign 更新；推导见折叠块）"。 ｜ 修复：在 L1102"恰好是对这个梯度做 SignSGD"后插入"（因 b = -β，对 β 的梯度下降等价于对 b 的 sign 更新，推导见折叠块；只保留方向，步长固定为 γ）"，将原"只保留方向，步长固定为 γ"并入括号内。 ｜ 复验：

- [重要·技术] overview.html L73 "不解决 sequence-wise 均衡"无来源支持：overview.html 称"不解决 sequence-wise 均衡：QB 只保证 batch 级负载均衡，不替代 sequence-wise auxiliary loss"，但 K3 报告全文（含 §2.3.3、Appendix C/D）未提及 sequence-wise 均衡。此论断无来源支持，且 index.html 完全未讨论此边界。overview 将其列为"关键结论与边界"但无来源标注，属于教学推断越界写成来源结论。：从 overview.html 删除此条，或标注为教学推断"QB 在 batch 级操作，不保证 per-sequence 均衡（教学推断，非来源结论）"并在 index.html S6 或 S7 补充对应说明。 ｜ 修复：从 overview.html 删除"不解决 sequence-wise 均衡"条目（采用最小方案，未在 index.html 补充）。 ｜ 复验：

- [轻微·技术] index.html S2 L737 "50000 步"计算不严格："前者需要约 50000 步才能把 bias 调到合理水平"——50000 = 50/0.001 将负载差（token 数 50）等同于 bias 变化量（50 单位），两者量纲不同。bias 变化量取决于分数分布，不等于负载差。定性结论（sign 慢）正确，但具体数字的隐含计算方法不精确。：改为"若 bias 需要变化约 50 个单位才能纠正该不均衡，则需要约 50000 步（50/0.001）"，或删除具体步数仅保留"需要数万步"的定性描述。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html S4 L832 手算例子与来源 Fig.5 的对应关系未提及：手算例子使用 m=8, n=4, k=1，与 K3 报告 Fig.5（L530）配置完全相同，但页面未提及这一对应关系。读者不知道此配置来自来源，可能误以为纯属虚构。：在 L834"设 m=8、n=4、k=1"后补充"（此配置与 K3 报告 Fig.5 一致；以下分数为人为构造）"。 ｜ 修复： ｜ 复验：

- [轻微·盲读] overview.html L74 "不改变路由稀疏性"未在 index.html 讨论：overview.html 称"不改变路由稀疏性：Top-k 固定，QB 只调 bias 不调 k"，但 index.html 正文未明确提及此边界。两个页面的边界说明不一致。：在 index.html S1（L705 附近）或 S6 补充一句"QB 不改变 k，路由稀疏性固定"。 ｜ 修复： ｜ 复验：

## 段 A 盲读学习目标核对

页面声明的 5 个学习目标（L667-672）逐条核对：

1. "QB 要解决什么问题？DeepSeek-V3 的固定步长 bias 更新为什么在 896 专家时失效？" → S1（aux-loss-free 路由）+ S2（sign 更新失效）完整回答 ✓
2. "QB 如何从一次前向传播推导出下一个 bias？公式 Eq.14 的每一步在做什么？" → S3（四步流水线）完整回答 ✓
3. "如何手算 m=8, n=4, k=1 的 QB 例子（从负载 (4,3,1,0) 变为 (2,2,2,2)）？" → S4（完整手算 + 代码验证）完整回答 ✓
4. "QB 与 DeepSeek-V3 sign 更新有什么本质区别？为什么 QB 不需要学习率类超参数？" → S5（对偶理论 + 对比表）完整回答 ✓
5. "QB 在训练时如何用直方图估计分位数？推理时如何处理？" → S6（直方图机制 + 推理冻结表）完整回答 ✓

全部学习目标由正文章节完整回答，无缺失。

## 段 B 对照来源核查摘要

### Eq.13（路由）— ✓ 一致
页面 L695 公式与来源 L552 Eq.13 完全一致：T_i = argtopk(s_i + b)，p_{i,j} = s_{i,j} / Σ_{r∈T_i} s_{i,r}。bias 只参与 Top-k 选择不参与归一化权重的描述与来源 L555-556 一致。

### Eq.14（QB 更新）— ✓ 一致
页面 L795-813 公式与来源 L586-589 Eq.14 完全一致：b̃_j ← -quantile_{1-k/n}(s_{:,j} - α)，b ← b̃ - mean(b̃)·1。margin = s - α 的定义与来源 L579、L2899 一致。Top-(k+1) 取 cutoff 的机制与来源 L564-566 一致。

### m=8, n=4, k=1 手算例子 — ✓ 验证通过
代码已实际执行，输出与页面预期输出（L1014-1049）完全一致：初始负载 (4,3,1,0) → QB 后 (2,2,2,2)。手算各步（cutoff、margins、3rd largest、mean-centering、新路由）逐项复核正确。分位数推导（(q+1)-th largest = (1-k/n) quantile）与来源 L577-579、L2854-2855 一致。

### histogram 估计 — ✓ 一致（符号问题见重要问题 #2）
分箱范围 [b_min-1, b_max+1]、bin width w=(b_max-b_min+2)/B、scatter-add + all-reduce、从 pooled counts 恢复分位数的线性插值公式、B=1000 误差几个 10^{-3}、通信 nB 整数/layer/step 低于 1%、EMA 平滑——均与来源 Appendix D（L2897-2939）逐条一致。counts 可加性保证全局直方图代表 pooled batch 的描述与来源 L2935-2936 一致。

### 与 DeepSeek-V3 sign 更新对比 — ✓ 一致（推导依赖见重要问题 #3）
sign 更新公式 b_j ← b_j + γ·sign(ℓ̄-ℓ_j) 与来源 L557 一致。对偶目标 Eq.23（页面 L1081）与来源 L2823 完全一致。subgradient ∂L/∂β_j = mk/n - Σ 1[...] = 目标负载 - 实际负载（页面 L1100）与来源 Eq.27（L2882）一致。"sign 是 SignSGD，QB 是 coordinate minimizer"的论断与来源 L2885-2888 一致。"近 10^3 专家时几步内收敛"与来源 L2888-2889 一致。

### config.json 核查 — ✓ 一致
num_experts: 896 ✓，num_experts_per_token: 16 ✓，moe_router_activation_func: "sigmoid" ✓，topk_method: "noaux_tc" ✓。

### 前置知识链接
../moe-serving/index.html（L689）——页面已标注"auxiliary-loss-free routing 的完整概念页待生成，此处为占位"（L705），占位提示到位。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 3
- 处置：进入修复
