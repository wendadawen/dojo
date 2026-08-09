# Kimi Delta Attention（KDA）独立审查（第二次）

- 审查者：独立上下文（AI 模拟，未参与生成与第一次审查）
- 页面版本：index.html `02d443f0a925eab4b2f77034f4dc0bca7cd6763a` / overview.html `66e2d2fd509975029fd5ff2f8ab784e9305290ef`
- 时间：2026-08-09
- 审查范围：段A盲读 index.html + overview.html（按页面顺序，不参考来源）；段B对照 `/tmp/kimi-k3-research/k3-report.txt` §2.1.1（Eq.1–6、Fig.3、正文）与 HuggingFace `moonshotai/Kimi-K3` `config.json`
- 未核对项：(1) §2.1 概述原文未读（来源限定 §2.1.1），69/24/93 的 3:1 交替比例仅由 `config.json` 的 `kda_layers`（69 项）/ `full_attn_layers`（24 项）编号模式印证；(2) §5.1.1 FlashKDA、§5.1.2 KCP 仅被页面引用章节、未展开，未读其原文；(3) 前置概念页 `delta-rule`、`linear-attention` 是否存在未核对（禁止读其他页面），页面未给占位提示；(4) 机械项（`validate.py` 退出码、公式渲染、折叠交互、目录锚点）未在本审查中核对

## 问题

- [重要·技术] index.html §「lower-bounded decay」开头（"$z_t^h = W_\alpha^{\uparrow\downarrow} x_t + b_h^\alpha$，其中 $A_h$ 是 per-head 可学习 log-scale、$b_h^\alpha$ 是 per-head bias"）：$A_h$ 被归入 Eq.2 的 $z_t^h$ 解释，但 $A_h$ 不在 $z_t^h$ 公式里，它属于 Eq.5 的映射 $g = g_{\min}\mathrm{Sigmoid}(e^{A_h} z)$；来源 §2.1.1 Eq.2 只含 $W_\alpha$ 与 $b_h^\alpha$，$A_h$ 在 Eq.5 才出现（"where $A_h$ is a learnable per-head log-scale"）。读者首见 $A_h$ 时找不到它属于哪个公式，造成认知跳步。：从 $z_t^h$ 的解释中删除 $A_h$，只保留 $W_\alpha^{\uparrow\downarrow}$ 与 $b_h^\alpha$；$A_h$ 留到下文 K3 scaled sigmoid 映射处首次引入（该处已有"$A_h$ 可学习 per-head log-scale、初始化 $A_h=0$"，无需重复）。｜ 修复：已从 §「lower-bounded decay」L810 的 $z_t^h$（Eq.2）解释中删除"$A_h$ 是 per-head 可学习 log-scale、"，只保留 $b_h^\alpha$；$A_h$ 在下文 Eq.5 scaled sigmoid 映射处（L831）首次引入并标注"可学习 per-head log-scale、初始化 $A_h=0$"，不再重复。 ｜ 复验：

- [轻微·盲读] index.html §「kda-recurrence」末尾"第 2-4 行变成 (0.8,0.9,0.3) 的差异化留存"：表述可能被误读为某一行变成向量 $(0.8,0.9,0.3)$，实际是第 2 行全 0.8、第 3 行全 0.9、第 4 行全 0.3（每行内部一致、行间不同）。：改为"第 2、3、4 行分别按 0.8、0.9、0.3 的留存率衰减（每行内部一致，行间不同）"。｜ 修复： ｜ 复验：

- [轻微·技术] index.html §「parameterization-and-gate」对 $W_\alpha^{\uparrow\downarrow}$ 的解释"是上、下两个投影（产出 z）"：来源 §2.1.1 Eq.2 正文明确称 $W_\alpha$ 为 "low-rank projection"，页面未说明这是低秩分解，读者不知道 ↑↓ 对应降维-升维的 low-rank 结构。：改为"$W_\alpha^{\uparrow\downarrow}$ 是 low-rank 分解（先 $W^\downarrow$ 降维、再 $W^\uparrow$ 升维），与 $b_h^\alpha$ 共同产出 decay logit $z$"。｜ 修复： ｜ 复验：

- [轻微·技术] index.html §「chunkwise-parallel」符号表"$Q[t], K[t], V[t] \in \mathbb{R}^{C \times d_k}$"：来源 Eq.1 明确 $v_t \in \mathbb{R}^{d_v}$，故 $V[t] \in \mathbb{R}^{C \times d_v}$；页面统一写成 $d_k$，虽 $d_k=d_v=128$ 数值等价，但符号不严格。：把 $V[t]$ 形状改为 $\mathbb{R}^{C \times d_v}$（$Q[t], K[t]$ 保持 $\mathbb{R}^{C \times d_k}$）。｜ 修复： ｜ 复验：

- [轻微·技术] index.html §「lower-bounded decay」details"若 $g_{\min}=-10$，则…$1/\Gamma < e^{160} \approx 10^{69}$"：$e^{160} \approx 3.07\times10^{69}$，写"$\approx 10^{69}$"省略系数 3，量级正确但精度不严，与同 details 内 $e^{80}\approx5.54\times10^{34}$ 的精确写法不一致。：改为"$e^{160} \approx 3 \times 10^{69}$"。｜ 修复： ｜ 复验：

- [轻微·盲读] index.html §「chunkwise-parallel」教学示例取 $\alpha_1=(1,1,1,1)$：K3 的 $\alpha\in(e^{-5},1)$ 是开区间，$\alpha=1$ 不在范围内（对应 $z\to-\infty$、$\mathrm{Sigmoid}\to0$ 的极限），教学构造取边界值未说明，细心读者会卡。：在 S5 示例或 S7 (3) 加注"$\alpha_1=(1,1,1,1)$ 表示第 1 步无衰减，是教学构造的边界近似；K3 实际 $\alpha<1$（开区间，$z\to-\infty$ 时趋近 1 但不等于 1）"。｜ 修复： ｜ 复验：

- [轻微·技术] index.html §「k3-config-and-boundaries」适用边界"若用 FP32 训练，negative-softplus 也能用"：来源 §2.1.1 未讨论 FP32，此为页面推断；negative-softplus 的 $1/\Gamma$ 在 FP32 下仍无界（只是 FP32 动态范围 $\approx3.4\times10^{38}$ 远大于溢出阈值），"也能用"说法过强。：改为"若用 FP32 训练，FP32 动态范围更大，negative-softplus 溢出风险降低，但 $1/\Gamma$ 仍无界，长序列或大 $z$ 下仍可能溢出"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 6
- 学习目标闭环：页首"读完你能回答"5 条目标均由 S1–S6 正文章节完整回答（S1 答"做什么/解决什么"、S2 答"forget gate 如何改写递归"、S3 答"lower-bound 与 negative-softplus 区别与原因"、S4/S5 答"full-rank gate 与 chunkwise 如何可训练"、S6 答"工程配置与边界"）
- 核心论断与来源一致性：C1–C10、F1–F6、N1–N6 逐条对照 §2.1.1 与 `config.json`，全部一致；Eq.1–Eq.6 公式形式与来源一致；config.json 十项数值（`num_hidden_layers=93`、`kda_layers` 69 项、`full_attn_layers` 24 项、`gate_lower_bound=-5.0`、`head_dim=128`、`num_heads=96`、`short_conv_kernel_size=4`、`use_full_rank_gate=true`、`hidden_size=7168`、`max_position_embeddings=1048576`）全部吻合，`kda_layers` 末三项 89/90/91 与 `full_attn_layers` 末两项 92/93 吻合
- 教学示例数字复算：Softplus(1)=ln(1+e)≈1.313 ✓、e^{-1.313}≈0.269 ✓、Sigmoid(1)≈0.731 ✓、g=-5×0.731=-3.655 ✓、e^{-3.655}≈0.0259 ✓、e^{-5}≈6.7×10^{-3} ✓、16-token tile 1/Γ<e^{80}≈5.54×10^{34} ✓、1M KV cache=2^{20}×96×128×2×2≈5.15×10^{10} bytes≈48 GiB ✓；S2 的 4 维 3 步手算（衰减→擦除→写入）逐矩阵复算正确
- 无可运行代码块：页面 S7 教学简化 (5) 明确声明无伪代码与可运行代码，符合"模型层机制"定位，无需执行核对
- 两页一致性：index.html 与 overview.html 的关键数字（69/24/93、g_min=-5、α∈(e^{-5},1)、1/Γ<e^{80}≈5.54×10^{34}、最后两层 Gated MLA、use_full_rank_gate）与公式形式完全一致；两页互相链接（overview→index 深度教学、index→overview 快速阅读）有效
- 处置：进入修复（1 项重要 + 6 项轻微，均不触及研究范围或教学大纲变更）
