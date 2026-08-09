# LatentMoE 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 段B对照来源核验）
- 页面版本：index.html `dd6b412` / overview.html `83636ce`（工作树哈希）
- 时间：2026-08-09 15:20
- 审查依据：`guides/concept/check.md`；来源 = WebSearch「LatentMoE NVIDIA arxiv 2601.18089」（[L] 论文 arXiv:2601.18089 + NVIDIA research 页）+ K3 报告 `/tmp/kimi-k3-research/k3-report.txt` §2.3
- 审查范围：仅 `index.html`、`overview.html` 与页面引用的外部来源；未读 `research/` 其他文件、未读其他概念页、未修改文档

## 段 A 盲读笔记

按页面顺序通读，扮演无领域知识的小白读者，只用页面自身信息判断主线卡点。

主线理解（可跟上）：
- S1 用 $d{=}4096$、$k{=}8/16$ 的 dispatch 量算账，建立"路由通信与专家权重读取随 $k\cdot d$ 线性增长"的痛点。逻辑自洽。
- S2 给出 F2 层公式与数据流图，明确共享支全宽、路由支隐空间、router 留在 $d$、$W_\downarrow/W_\uparrow$ 全局共享。$S_k\to T_k$、$g_i\to p_i$ 的符号切换有显式说明（与 K3 Eq. 11 对齐）。可理解。
- S3 用 $d{=}4096,\ell{=}1024$ 算 dispatch 缩小 4×，用 $N{=}8,k{=}2\to N'{=}32,k'{=}8$ 算组合数 $28\to 10{,}518{,}300$。可复算，与代码预期输出一致。
- S4 三条压缩约束（特征秩下限、投影自身成本、只压不 reinvest 会掉点）可理解。
- S5 给出 K3 配置表与 LatentMoE↔Stable LatentMoE 关系图，三件稳定化点名、机制留待独立页。可理解。

折叠块均不为主线所依赖（MLA 类比、五条设计原则、Nemotron-3 配置、可运行代码都可收起而不打断理解）。章节切换每处有过渡问句。学习目标 5 题均由正文章节完整回答。

盲读卡点：
- "K2-1T" 首现时仅有"万亿参数规模"的括号提示，不知其身份（见问题 1）。

## 段 B 对照来源核验

来源 [L]（arXiv:2601.18089 + NVIDIA research 页）逐条核对：
- C1（LatentMoE 定义：路由支两端投影、路由专家在 $\ell$ 维、共享与 router 仍在 $d$）：[L] Figure 1(b) + research 页"router still computes gating decisions from the model's hidden representation—only the routed payload and routed expert computation move into the latent space. Shared experts, if present, operate in the original hidden dimension." ✓
- C2（标准 MoE 全宽路由开销随 $k,d$ 线性增长）：[L] 设计原则 I/II + "Routing volume scales with tokens × top-k × routed width"。✓
- C3（路由部分开销按 $d/\ell$ 缩小）：[L] Figure 1 caption"reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"。✓
- C4（reinvestment 按 $\alpha{=}d/\ell$ 放大 $N,k$）：[L] Figure 1 caption"increase the total number of experts and the top-k active experts per token by the same factor d/ℓ"。放大 $N,k$ 部分 ✓；但页面据此称"保持非线性容量不变"与来源不符（见问题 2）。
- C5（压缩下限与投影成本）：[L] 设计原则 IV（"task-dependent feature rank"）+ §3 消融（"compression ratios α ≤ 4"质量保持）+ §4.3.1（投影额外开销 ~9%）。✓
- C6（Stable LatentMoE = LatentMoE + 三件稳定化）：[K3] §2.3"Stable LatentMoE addresses these two failure modes with three components: an RMSNorm before the up-projection and SiTU-GLU ... and Quantile Balancing (QB)"。✓
- C7（router 仍在 $d$）：[L] research 页明示。✓
- F2 ↔ [K3] Eq. 11：[K3] 报告 Eq. 11 `y = Σ E_j^shared(x) + W↑ RMSNorm(u)`，去掉 RMSNorm 即页面 F2；$p_i$、$T_k(x)$ 符号与 K3 一致。✓
- N1（K3 配置 $d{=}7168,\ell{=}3584,N{=}896,k{=}16,N_s{=}2$，稀疏度 56）：[K3] 配置表（line 745-750）逐项命中；$N_s{=}2$ 见 line 478"K3 fixes the number of full-width shared experts to Ns = 2"。✓
- N2（Nemotron-3 Super $4096{\to}1024{\to}4096$，120B/12B）：[R] Raschka"For Super, the routed path is 4096 -> 1024 -> 4096""Super has 120 billion total and 12 billion active"。✓
- N3（Nemotron-3 Ultra $8192{\to}2048{\to}8192$，550B/55B，512 路由专家、top-22、1 共享、中间维度 5120/10240）：[R] Raschka 逐项命中。✓
- N4（16B 消融 α≤4 保持质量）：[L] §3"model quality is preserved for compression ratios α ≤ 4"。✓
- N5（投影额外计算 ~9%）：[L] §4.3.1"native Kimi-K2-1T remains close, within up to ~9% of Kimi-K2-1T-LatentMoE"。✓（但"K2-1T"身份未在页面说明，见问题 1）
- 可运行代码：复算 $C(8,2){=}28$、$C(32,8){=}10{,}518{,}300$、比值 $375653.57$、$2{\cdot}4096{\cdot}1024{=}8{,}388{,}608$，与页面预期输出逐行一致。✓
- 教学简化均标记；MLA 类比标注失效边界。✓

## 问题

- [重要·技术] index.html S3「压缩比与 Reinvestment」正文段 + S3 对比表「非线性容量」行 + 来源说明 C4 + overview.html「核心直觉」第 4 条："reinvestment 时要保持非线性容量不变（设计原则 3）"误框定来源 [L] 的设计原则 III。[L] §2：原则 III 指压缩阶段——"Since we compress only the input dimension d to ℓ while keeping the intermediate dimension m constant, the effective nonlinear budget U_eff remains unchanged"（压缩时 $K,m$ 均不变 → $U_\text{eff}{=}K\cdot m$ 不变）；reinvestment（$\ell$-MoE_acc，原则 V）按 $\alpha$ 放大 $K$，$m$ 保持，故 $U_\text{eff}$ 随之提升——[L] 明示"The increased expert diversity and non-linearity budget per token ... lead to superior model accuracy"。页面把"保持非线性容量不变"归给 reinvestment，且与自身解释自相矛盾（"放大 $k$"+"中间维度不能缩小" $\Rightarrow k\cdot m$ 上升，并非"不变"），并掩盖 reinvestment 提升准确率的部分机制（非线性容量提升）。：将 S3 正文改为"压缩 $d\to\ell$ 时保持非线性容量不变（设计原则 III：保持 $K$ 与中间维度 $m$ 不变，$U_\text{eff}{=}K\cdot m$ 不变）；reinvestment（设计原则 V）按 $\alpha$ 放大 $K$ 与 $N$，$m$ 不变，$U_\text{eff}$ 随 $K$ 放大而提升——这是 reinvestment 提升准确率的机制之一"。保留"压缩 $\ell$ 只缩路由专家输入输出宽度、不缩中间维度"的正确要点。S3 对比表「非线性容量（$k\times$ 中间维度）」LatentMoE+reinvestment 列由"保持不变"改为"随 $K$ 放大而提升（$m$ 保持不变）"。来源说明 C4 同步修正为"reinvestment 按 $\alpha$ 放大 $N,K$；$m$ 保持不变，$U_\text{eff}$ 提升"，并区分原则 III（压缩）与原则 V（reinvest）。overview.html「核心直觉」第 4 条同步。 ｜ 修复：已将 S3 正文改为区分压缩（原则 III，$U_\text{eff}$ 不变）与 reinvestment（原则 V，$U_\text{eff}$ 提升）；S3 对比表「非线性容量」行改为"压缩时不变（原则 III）；reinvestment 时随 $K$ 放大而提升（$m$ 不变，原则 V）"；C4 同步修正并区分原则 III/V；五条设计原则列表第 3 条补"压缩时"、映射说明区分原则 3（压缩）与原则 5（reinvest）；S3 检查项与 overview「核心直觉」第 4 条同步 ｜ 复验：

- [轻微·盲读] index.html S4「约束二」+ 来源说明 N5 + overview.html「关键结论」第 1 条："K2-1T"术语首现，仅以括号"万亿参数规模"提示规模，未说明其身份——[L] §4.3.1 中 K2-1T 是论文用作服务性能基准的 Kimi-K2-1T 模型（"We benchmark the native Kimi-K2-1T against our proposed variant, Kimi-K2-1T-LatentMoE"），不是论文自创的投影名。小白读者不知"K2"指何物，易误以为是抽象规模代号。功能性含义（万亿参数规模、9% 投影开销）已传达，影响有限。：首现处补一句身份说明，如"论文以 Kimi-K2-1T（1 万亿参数 MoE 模型）为基准，投影其 LatentMoE 变体的服务性能"。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html S5 末段："这并非 K3 不想压得更窄，而是 2.8T 规模与极端稀疏下，再压窄会放大激活爆炸的风险——这正是三件稳定化要兜住的问题"——呈现为 K3 的设计动机事实，但 [K3] §2.3 未明确陈述此动机（仅说明在所选 2x/2.8T/稀疏度 56 配置下已出现激活爆炸与近千专家负载失衡，故需三件稳定化），未讨论"若压到 4x 风险如何"。属合理推断但未标记为推断，违反段B"教学解释是否越界写成来源结论"。：改为不涉及动机的陈述并标记推断，如"K3 在 2.8T 规模与稀疏度 56 下已出现路由分支激活爆炸与近千专家负载失衡（§2.3），三件稳定化即用于兜住这两个失败模式；更激进的压缩比是否会进一步放大风险，K3 报告未明确讨论"。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html 来源说明 N6："iso-accuracy 下 LatentMoE 投影约 3.5x 加速，标准 MoE 需多约 350B 参数"是对 [L] §4.3.1 的不精确转述。论文原文："Matching its accuracy under standard MoE scaling requires an additional ~350B parameters ... and yields a 1.24×–3.46× projected slowdown across the frontier"——即标准 MoE 需多约 350B 参数且慢 1.24×–3.46×（LatentMoE 快 1.24×–3.46×，是一个区间，3.5× 仅近似上界），且论文用"slowdown"表述，页面反转为"3.5x 加速"并取上界为单值。该数字仅登记未用于正文，影响有限。：改为"标准 MoE 需多约 350B 参数且慢 1.24×–3.46×（[L] §4.3.1，投影值非实测）"，或删除"3.5x 加速"单值表述。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 已核对项：C1–C7、F1–F4、N1–N5、可运行代码输出、教学简化标记、MLA 类比失效边界、K3 Eq. 11 差异、五条设计原则、两份 Nemotron-3 配置——均与来源一致（问题 2 涉及 C4 框定、问题 4 涉及 N6 转述）。
- 未核验项：概念页间链接的目标页面存在性（审查任务包禁止读其他页面），仅确认相对路径层级正确（`../<name>/index.html`、`../../index.html`、`overview.html`）；机械项（validate.py、公式渲染、目录锚点）未在本审查内执行。
- 处置：进入修复。无阻断；1 个重要问题（问题 2）可逐条修复，需重新对照 [L] §2 区分压缩阶段（原则 III）与 reinvestment 阶段（原则 V）；3 个轻微问题可一并修复或逐项写明接受理由。修复完成后建议交回本审查者复验问题 2 及其引用位置。
