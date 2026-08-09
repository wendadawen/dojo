# EAGLE-3 投机解码 draft 模型 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html MD5 b97bb9dcbe3de8eddc5e01a5938a4726 / HEAD d4f9e4ef
- 时间：2026-08-09
- 审查范围：index.html + overview.html
- 来源：EAGLE-1 论文 arXiv:2401.15077（ar5iv 全文）、EAGLE-3 论文 arXiv:2503.01840（arXiv HTML 全文 + WebSearch 摘要）、K3 报告 §4.1.4（/tmp/kimi-k3-research/k3-report.txt:922-954）

## 段 A 盲读小结

按页面顺序通读 index.html + overview.html，扮演无领域知识的小白读者。主线卡点记录如下：

- 开篇段落（index 行 653）使用 α、c 两个变量但未立即定义；下一章（行 672）给出加速比公式并解释。可由上下文推断，不阻断。
- γ 在加速比公式（行 672）中首次出现但未定义，直到推理 pipeline 章（行 805）才定义为"草稿长度"。先用在后定义。
- EAGLE-1 机制章（行 701-740）主线清晰：两个观察 → time-shifted token → draft 输出 feature 再过 lm_head。可跟。
- EAGLE-3 架构改变章（行 742-800）称 EAGLE-1 为"3-4 层 decoder"（行 711、762、783），并称"EAGLE-3 减到 1 层"。对照来源后发现此声明与 EAGLE-1 论文矛盾（详见段 B 问题 1）。
- 推理 pipeline 章（行 803-977）三步时序清晰，手算例子可逐行复算。公式 $[g_{1:t};\, e_{t+1}]$（行 815）记号含义不如 EAGLE-1 图示（行 720）的 `concat(f_i, e_{i+1}) for i=1..t` 清晰。
- 手算例子将 draft 从序列处理简化为单 (g, e) 对（行 853），简化说明列出了省略的组件但未点明"省略 attention 后各位置独立、故可只处理单个位置"这一前提。
- 训练章（行 979-1058）L_E3 公式（行 1026）中求和上限 $k$ 与前文 hidden size 的 $k$（行 756-758）符号碰撞。
- K3 工程章（行 1061-1113）配置项与来源逐条对应，可跟。

学习目标核对（页面"读完你能回答"5 题）：
1. EAGLE 复用 target 隐藏状态 → §why-new-draft + §eagle-1-feature-ar 回答 ✓
2. EAGLE-3 两项架构改变 → §eagle-3-changes 回答 ✓（但对比表含层数错误，见问题 1）
3. 推理 pipeline 自回归 → §inference-pipeline 回答 ✓
4. TTT + LK loss → §training-ttt-loss 回答 ✓
5. K3 MTP fine-tune → §k3-engineering-boundaries 回答 ✓

## 段 B 对照来源核查

逐条核对页面 [C1]-[C10]、[F1]-[F5]、[N1]-[N5] 声明与外部来源：

- [C1] TinyLLaMA 3000B vs EAGLE 2-4B tokens：EAGLE-1 论文 §1 一致 ✓
- [C2] EAGLE-1 feature 层自回归 + time-shifted token：EAGLE-1 摘要一致 ✓
- [C3] EAGLE-3 两项改变：EAGLE-3 摘要 verbatim 一致 ✓
- [C4] 推理 pipeline 自替代：EAGLE-3 §3.1 原文 "we use the output a_I from the draft model in the previous step to replace g_I" 一致 ✓
- [C5] TTT 因果 mask：EAGLE-3 §3.2 + Figure 6 一致 ✓
- [C6] L_E3 / L_LK：L_LK 与 K3 §4.1.4 Eq.(16) verbatim 一致 ✓；L_E3 公式来源为 emergentmind 摘要（非论文原文），公式形式合理但直接来源待确认
- [C7] K3 MTP 初始化：K3 §4.1.4 verbatim 一致 ✓
- [C8] 三层 feature + W_E3=[0 0 I]：K3 §4.1.4 verbatim 一致 ✓
- [C9] QAT 配置：K3 §4.1.4 verbatim 一致 ✓
- [C10] lossless：EAGLE-1/EAGLE-3 摘要 + Leviathan 2023 一致 ✓
- [N1] EAGLE-1 70B 2.7x-3.5x：EAGLE-1 摘要 verbatim 一致 ✓
- [N2] EAGLE-3 6.5x / 1.4x over EAGLE-2 / SGLang 1.38x：EAGLE-3 摘要 verbatim 一致 ✓
- [N3] Vicuna 13B 5.6x / 1.8x over EAGLE-1：与 EAGLE-3 论文 Table 1 (5.58x / 3.07x → 1.82x) 一致 ✓
- [N4] K3 TTT 7 步：K3 §4.1.4 verbatim 一致 ✓
- [N5] EAGLE 2-4B / TinyLLaMA 3000B：同 [C1] ✓
- [F1] 融合公式 g_t = W_fuse·[l;m;h]：EAGLE-3 §3.1 一致 ✓
- [F2] 单步 a_t = DraftLayer([g;e]) → q_t = softmax(W_lm·a)：EAGLE-3 §3.1 一致 ✓
- [F5] α = Σ min(p,q) = 1 - TV(p,q)：数学恒等式正确 ✓

## 问题

- [阻断·技术] index.html 行 711、762、783（overview.html 无此问题）：页面在三处声称 EAGLE-1 draft 模型为"3-4 层 decoder"，并以此构建"EAGLE-3 减到 1 层"的对比叙事。EAGLE-1 论文原文（ar5iv 全文 §3 Implementation）描述 Autoregression Head 为 "an FC layer and **a decoder layer**"（单数），NVIDIA NeMo EAGLE-1 训练配方配置 `draft_num_hidden_layers: 1`，多个二级来源（CSDN 技术博客"单层 decoder"、aiengineeringfromscratch "one or two layers"）均确认 EAGLE-1 为 1 层。来源 [C2]（EAGLE-1 论文）不支持"3-4 层"声明。此错误使对比表（行 783）"draft 层数"行事实错误，且让读者误以为"EAGLE-3 减层"是一项架构改变——实际 EAGLE-1 与 EAGLE-3 均为单层 decoder，两项改变是直接 token 预测 + 多层融合，不含层数变化。：将行 711、762、783 中"3-4 层"改为"单层"（或"1 层"），删除"EAGLE-3 减到 1 层"的对比叙事，对比表"draft 层数"行改为"EAGLE-1 = 单层 / EAGLE-3 = 单层"或删除该行并说明层数非两项改变之一。重新对照 EAGLE-1 论文 §3 原文 "a decoder layer"。 ｜ 修复：已将行 711、719 的"3-4 层"改为"单层"；行 762 删除"EAGLE-1 是 3-4 层 decoder，EAGLE-3 减到 1 层"，改为"EAGLE-1 与 EAGLE-3 均为单层 decoder，层数不是 EAGLE-3 的两项改变之一"；行 783 对比表"draft 层数"行改为"EAGLE-1 = 单层 decoder / EAGLE-3 = 单层 decoder（层数非两项改变之一）"。 ｜ 复验：待 validate.py 通过

- [重要·技术] index.html 行 756-758 vs 行 1026-1028：符号 $k$ 一物两用。行 756 中 $g_t \in \mathbb{R}^k$、$W_{\text{fuse}} \in \mathbb{R}^{k \times 3k}$ 的 $k$ 是 hidden size；行 1026 中 $L_{E3} = -\sum_{i=1}^{k} \log q(\ldots)$ 的 $k$ 是 TTT unroll 长度（行 1028 明确写"其中 $k$ 是 TTT unroll 长度"）。同一符号在同页表示两个不同量，读者在 L_E3 公式中会将求和上限 $k$ 误解为 hidden size。：将 L_E3 公式中的 unroll 长度改用不同符号（如 $K$、$L$ 或 $D$），并在公式下方定义。同步修改行 1028 的"其中 $k$ 是 TTT unroll 长度"为对应新符号。 ｜ 修复：已将 L_E3 公式（行 1026）求和上限 $k$ 改为 $K$，定义句（行 1028）改为"其中 $K$ 是 TTT unroll 长度"并补注"$K$ 表 unroll 长度，与前文 hidden size 的 $k$ 区分"；同步更新来源 [F3]（行 1145）的公式符号为 $K$ 并注明原文用 $k$。 ｜ 复验：待 validate.py 通过

- [重要·技术] index.html 行 1067：页面称"TTT unroll 长度为 7 步 [N4]——即训练时 draft 展开 7 步、与 K3 推理时的 γ 设定一致"。K3 报告 [N4]（§4.1.4 行 934-937）原文只说 "the draft is unrolled for seven steps during training"，未明确说明推理时 γ = 7。"与 K3 推理时的 γ 设定一致"是页面推断，无来源支持。EAGLE-3 TTT 的设计意图确实是让训练 unroll 匹配推理 γ，但 K3 报告未给出推理 γ 的具体值。：删除"与 K3 推理时的 γ 设定一致"，或改为"训练时展开 7 步（K3 报告未给出推理 γ 的具体值）"。如保留"一致"表述，需补充 K3 报告或其他来源中推理 γ = 7 的直接依据。 ｜ 修复：已删除"与 K3 推理时的 γ 设定一致"，改为"训练时 draft 展开 7 步（K3 报告未给出推理时 γ 的具体值，训练 unroll 与推理 γ 的对应关系属设计意图、非报告明示）"。 ｜ 复验：待 validate.py 通过

- [轻微·盲读] index.html 行 672：γ 在加速比公式 $S = (1 - \alpha^{\gamma+1}) / [(1-\alpha)(1 + \gamma c)]$ 中首次出现但未定义。γ 直到行 805（推理 pipeline 章"设当前前缀长度为 $t$，草稿长度 $\gamma$"）才定义。读者在第一章遇到公式时不知道 γ 代表什么。：在行 672 公式后补充"其中 γ 为草稿长度（每轮生成的 draft token 数）"一句。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html 行 887 vs 行 919-920：Step 3 手算中 $a_{\text{it}}$ 第二分量，正文（行 887）写 $a_{\text{it}} \approx [0.026,\, 0.103,\, 0.022,\, 0.493]$，展开块（行 920）写 pre-tanh 值 0.104、$a_{\text{it}} = \tanh([0.026, 0.104, 0.022, 0.540]) \approx [0.026, 0.103, 0.022, 0.493]$。tanh(0.104) ≈ 0.1036，标准四舍五入为 0.104 而非 0.103。正文与展开块间存在 0.001 的舍入不一致。不影响 argmax 结论（仍为 "now"）。：将正文行 887 的 0.103 改为 0.104，使正文与展开块一致；或将展开块的 pre-tanh 值改为 0.103 使正文 0.103 成立（但 0.1·0.084+0.3·0.318=0.1038 更接近 0.104）。统一舍入规则即可。 ｜ 修复： ｜ 复验：

- [轻微·盲读] index.html 行 849-853：手算例子将 draft 从序列处理简化为单 (g, e) 对。简化说明（行 853）列出"省略 attention 子层、MLP、layer norm"但未点明"因省略 attention 后各位置独立计算、故只需处理最后一个位置"这一前提。读者需自行推断"为什么从序列简化到单位置是合理的"。：在行 853 简化说明中补充"省略 attention 后各位置独立计算，故手算只需处理最后一个位置的 (g, e) 对"。 ｜ 修复： ｜ 复验：

- [轻微·技术] index.html 行 813-815：推理 pipeline Step 1 公式 $a_{t+1} = \text{DraftLayer}([g_{1:t};\, e_{t+1}])$ 中 $[g_{1:t};\, e_{t+1}]$ 记号有歧义——可被读作"整个 g 序列与单个 e 拼接成一个长向量"，而实际含义是"每个位置 i 上 concat(g_i, e_{i+1})"（EAGLE-1 图示行 720 用 `concat(f_i, e_{i+1}) for i=1..t` 更清晰）。：将公式改为 $a_{t+1} = \text{DraftLayer}(\{\text{concat}(g_i, e_{i+1})\}_{i=1}^{t})$ 或在公式下注明"$[\cdot;\cdot]$ 表示逐位置拼接，非整体拼接"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 4
- 处置：进入修复

阻断问题（EAGLE-1 层数"3-4 层"错误）涉及核心架构对比的事实性错误，来源明确不支持，必须修复后复验。两个重要问题（符号碰撞、K3 推理 γ 无来源）影响公式可读性和事实准确性，应一并修复。四个轻微问题不影响核心正确性，可在修复阻断和重要问题时顺带处理。

overview.html 与 index.html 内容一致，overview 未引入额外问题（overview 未提及 EAGLE-1 层数、未含 L_E3 公式、未含 K3 γ 声明）。overview 中"单层 transformer decoder"的表述正确。
