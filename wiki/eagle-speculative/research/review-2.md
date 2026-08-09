# EAGLE-3 投机解码 draft 模型独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：wiki/eagle-speculative/index.html（78134 bytes, 2026-08-09 15:33）+ overview.html（9192 bytes, 2026-08-09 13:48）
- 时间：2026-08-09
- 审查依据：guides/concept/check.md（段A盲读 + 段B对照来源）
- 来源：
  - WebSearch "EAGLE speculative decoding arxiv" → EAGLE-1 论文（arXiv:2401.15077, ICML 2024）摘要与 §1、§3；EAGLE-3 论文（arXiv:2503.01840, NeurIPS 2025）摘要与 §3.1、§3.2
  - /tmp/kimi-k3-research/k3-report.txt §4.1.4 Draft Model Fine-Tuning（行 922-954）

## 段 A 盲读笔记（小白读者卡点）

按页面顺序阅读：

- §1（行 672）：加速比公式 $S = (1 - \alpha^{\gamma+1}) / [(1-\alpha)(1 + \gamma c)]$ 直接给出，引用投机解码概念页。α、c、γ 符号在上下文解释（α 接受率、c 单步成本比、γ 草稿长度），可接受。
- §3（行 756-758）：$g_t = W_{\text{fuse}} \cdot [l_t; m_t; h_t] \in \mathbb{R}^k$。$k$ 首现未明确说明是 target 模型的 hidden size，要到行 851 手算例子才见"hidden size $k = 4$"。
- §3（行 770 ASCII 图）："FC (k×2k)"——FC 缩写未展开，但常见术语（全连接层），可接受。
- §4（行 813-815）：DraftLayer 首现于公式 $a_{t+1} = \text{DraftLayer}([g_{1:t}; e_{t+1}])$，未明确说 DraftLayer 即前述"单层 transformer decoder"，读者需回溯推断。
- §4（行 851-893 手算例子）：简化说明（行 853）已涵盖"简化后 draft 步骤为 $a = \tanh(W_a \cdot [g_{\text{prev}}; e_{\text{sampled}}])$"，解释了为何手算只用 $g_{\text{can}}$（前一位置）而非整个序列 $g_1, \ldots, g_t$。读者需推断"线性+tanh 简化使只有最后位置 feature 影响输出"。可接受。
- §5（行 1026-1028）：$L_{E3}$ 公式中 $K$（unroll 长度）与前文 hidden size 的 $k$ 符号冲突，页面行 1028 已注明"$K$ 表 unroll 长度，与前文 hidden size 的 $k$ 区分"。**正确处理**。

段 A 结束时逐题核对学习目标：
- 目标1（EAGLE 复用 target feature + 降低成本/提高接受率）：S1 回答，清晰。
- 目标2（EAGLE-3 两项架构改变）：S3 回答，每项改变对应具体问题，清晰。
- 目标3（推理 pipeline 三步 + 自替代原因）：S4 回答，手算例子完整可复算。
- 目标4（TTT + LK loss vs KL）：S5 回答，机制清晰。
- 目标5（K3 MTP fine-tune）：S6 回答，配置表清晰。

## 段 B 对照来源核查

### 1. 定义与机制

逐条对照 EAGLE-1 论文（arXiv:2401.15077）、EAGLE-3 论文（arXiv:2503.01840）、K3 报告 §4.1.4：

- C1（传统独立小模型 draft 两难 + EAGLE 复用 target feature）——EAGLE-1 论文 §1"TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens" **完全一致**；页面行 1119 [C1] 引用原文准确。
- C2（EAGLE-1 feature 层自回归 + time-shifted token）——EAGLE-1 论文摘要"Firstly, autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level. Secondly, the inherent uncertainty in feature level autoregression constrains its performance... By incorporating a token sequence advanced by one time step, EAGLE effectively resolves the uncertainty" **完全一致**；页面行 1121 [C2] 引用准确。
- C3（EAGLE-3 两项架构改变：直接 token 预测 + 多层 feature 融合）——EAGLE-3 论文摘要"EAGLE-3, which abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test" **完全一致**；页面行 1123 [C3] 引用准确。
- C4（EAGLE-3 推理 pipeline 三步 + 自替代）——EAGLE-3 论文 §3.1"In Step 2... we use the output a_I from the draft model in the previous step to replace g_I" **完全一致**；页面行 1125 [C4] 引用准确。
- C5（TTT 训练机制 + 因果 mask）——EAGLE-3 论文"The training-time test enables the draft model to simulate multi-step autoregressive generation during training, attending to its own previous predictions through custom causal masks" **完全一致**；页面行 1127 [C5] 引用准确。
- C6（$L_{E3}$ 论文版 + K3 LK loss）——K3 报告行 945-950 Eq.(16)"Since minimizing the conventional KL-divergence surrogate does not guarantee maximizing this rate for a capacity-limited draft model, we directly optimize the likelihood-based LK loss... $L_{LK} = -\log \sum_{x \in V} \min(p(x), q(x))$" **完全一致**；页面行 1129 [C6] 引用准确。
- C7（K3 MTP 层初始化 EAGLE-3 draft）——K3 报告行 930-934"Kimi K3 is pre-trained with a multi-token-prediction (MTP) layer that mirrors the structure of a backbone block... we fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model, with the target model frozen and only the draft layer and its feature-fusion projection updated" **完全一致**；页面行 1131 [C7] 引用准确。
- C8（三层 feature 来源 + $W_{E3}=[0\,0\,I]$ 初始化）——K3 报告行 938-942"The draft input fuses low-, mid-, and high-level features of the target model, taken from the outputs of the 1st, 4th, and final AttnRes blocks... initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature $h_h$" **完全一致**；页面行 1133 [C8] 引用准确。
- C9（K3 draft QAT 配置）——K3 报告行 952-954"Draft fine-tuning follows the post-training QAT configuration, with MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision" **完全一致**；页面行 1135 [C9] 引用准确。
- C10（EAGLE 系列 lossless + 不改变投机解码框架）——EAGLE-1 摘要"maintaining the distribution of the generated text"；EAGLE-3 摘要"ensuring lossless performance"；**一致**；页面行 1137 [C10] 引用准确。

### 2. 公式与推导

- F1（多层特征融合 $g_t = W_{\text{fuse}} \cdot [l_t; m_t; h_t] \in \mathbb{R}^k$）——EAGLE-3 论文"concatenates k-dimensional feature vectors from three selected layers into a 3k-dimensional vector, then passes it through a fully connected layer to reduce it back to k dimensions" **一致**；页面行 1141 [F1] 引用 emergentmind 摘要"g_t = W_fuse [f^(1)_t; ...; f^(L)_t] ∈ R^k"，K3 实现命名为 $W_{E3}$。
- F2（单步 draft 计算 $a_t = \text{DraftLayer}([g_{<t}; e_{t-1}])$、$q_t = \mathrm{softmax}(W_{\text{lm}} \cdot a_t)$）——EAGLE-3 论文 §3.1"The concatenated vector is then passed through an FC layer to reduce its dimensionality to k, and subsequently inputted into a single layer decoder, producing the output a. Finally, we input a_I into the LM head" **完全一致**；页面行 1143 [F2] 引用准确。
- F3（$L_{E3} = -\sum_{i=1}^{K} \log q(t_{t+i} | g_{1:t}, a_{t+1:t+i-1})$）——EAGLE-3 论文 token-level NLL；AI Wiki 确认"trained only with a token-level classification (cross-entropy) loss" **一致**；页面行 1145 [F3] 注明改用 $K$ 避免与 hidden size $k$ 碰撞，处理正确。
- F4（$L_{\text{LK}} = -\log \sum_{x \in V} \min(p(x), q(x))$）——K3 报告 Eq.(16) **完全一致**。
- F5（$\alpha = \sum_x \min(p, q) = 1 - \mathrm{TV}(p, q)$）——K3 报告行 943"per-token acceptance rate $\sum_{x \in V} \min(p(x), q(x))$"；$\alpha = 1 - \mathrm{TV}(p, q)$ 由 $\min(a,b)=(a+b-|a-b|)/2$ 推出，数学恒等式正确；页面行 1149 [F5] 引用 Leviathan et al. 2023 §3。

手算例子复算（§4 行 849-926，prefix="How can I"，γ=3，k=4，词表 5 token）：

**Step 1**（输入 $(g_{\text{can}}, e_I)$，target 真实 g）：
- $W_a \cdot [0.4, 0.2, 0.1, 0.6, 0, 1, 0, 0] = [0.08, 0.45, 0.14, 0.13]$——**复算正确**。
- $a_I = \tanh([0.08, 0.45, 0.14, 0.13]) \approx [0.080, 0.422, 0.139, 0.129]$——**正确**。
- $W_{\text{lm}} \cdot a_I = [0.160, 0.844, 0.278, 0.259, 0.385]$——**正确**。
- softmax $\sum \approx 7.585$，$q_1 \approx [0.155, 0.307, 0.174, 0.171, 0.194]$，argmax="do"——**正确**。

**Step 2**（用 $a_I$ 替代 $g_I$，输入 $(a_I, e_{\text{do}})$）：
- $W_a \cdot [0.080, 0.422, 0.139, 0.129, 0, 0, 1, 0] = [0.092, 0.084, 0.329, 0.040]$——**复算正确**。
- $a_{\text{do}} \approx [0.092, 0.084, 0.318, 0.040]$——**正确**。
- $W_{\text{lm}} \cdot a_{\text{do}} = [0.184, 0.167, 0.635, 0.080, 0.267]$（个别位四舍五入差 0.001，可接受）。
- softmax $\sum \approx 6.661$，$q_2 \approx [0.180, 0.178, 0.283, 0.163, 0.196]$，argmax="it"——**正确**。

**Step 3**（用 $a_{\text{do}}$ 替代 $g_{\text{do}}$，输入 $(a_{\text{do}}, e_{\text{it}})$）：
- $W_a \cdot [0.092, 0.084, 0.318, 0.040, 0, 0, 0, 1] = [0.026, 0.104, 0.022, 0.540]$——**复算正确**。
- $a_{\text{it}} \approx [0.026, 0.103, 0.022, 0.493]$——**正确**。
- $W_{\text{lm}} \cdot a_{\text{it}} = [0.052, 0.207, 0.045, 0.985, 0.322]$（个别位四舍五入差 0.001，可接受）。
- softmax $\sum \approx 7.387$，$q_3 \approx [0.143, 0.166, 0.142, 0.363, 0.187]$，argmax="now"——**正确**。

**偏差对照**（折叠块行 925）：$g_I - a_I = [0.370, -0.022, -0.039, 0.071]$，L2 $\approx \sqrt{0.137+0.0005+0.0015+0.005} \approx 0.380$——**复算正确**（$\sqrt{0.143946} \approx 0.3794$，四舍五入 0.380）。

三步手算全部可复算，结果与页面一致；自替代偏差的 L2 距离计算正确。

### 3. 可运行代码

页面伪代码（行 932-964）标记为"不是 Python"，形式化推理 pipeline 三步，包含输入、Prefill、Draft 循环、Verify 阶段、输出，符合 A6 规则。关键行 `g_seq ← g_seq + [a]` 明确体现"自替代"机制。可接受。

### 4. 事实与推断

- EAGLE-1 论文 arXiv:2401.15077，ICML 2024——页面行 656，搜索结果确认 **正确**。
- EAGLE-3 论文 arXiv:2503.01840，NeurIPS 2025——页面行 656，搜索结果（NeurIPS 2025 poster）确认 **正确**。
- N1（EAGLE-1 在 LLaMA2-Chat 70B 上 2.7x-3.5x 加速、吞吐翻倍）——EAGLE-1 论文摘要 **完全一致**。
- N2（EAGLE-3 最高 6.5x + 相比 EAGLE-2 约 1.4x + SGLang batch=64 吞吐 1.38x）——EAGLE-3 论文摘要 **完全一致**。
- N3（EAGLE-3 在 Vicuna 13B 上 5.6x、比 EAGLE-1 快 1.8x）——页面引用 HF 模型卡；AI Wiki 也确认"Peak measured speedup reached 5.6x on Vicuna 13B"；$5.6/3.1 \approx 1.81\text{x}$ 与"1.8x"一致 **支持**。
- N4（K3 TTT unroll 7 步）——K3 报告行 934-935"unrolled for seven steps during training" **完全一致**；页面行 1159 [N4] 引用原文准确。
- N5（EAGLE-1 训练 2-4B tokens、TinyLLaMA 3000B tokens）——EAGLE-1 论文 §1 **完全一致**。
- 手算例子标注为"教学构造...不代表真实模型或工程推荐值"——诚实标注，**正确**。

### 5. 前置知识引用

- 投机解码（../../wiki/speculative-decoding/index.html）——链接有效，页面存在（本审查范围内）。
- MXFP4 量化感知训练（../../wiki/mxfp4-qat/index.html）——链接有效，页面存在。

### 6. 教学简化

- 词表缩小到 5 token + one-hot embedding + draft 简化为"线性+tanh"——均标记为教学简化，说明"不改变自替代机制的数学结构"，列出可推出/不可推出的范围，**正确**。
- TTT 因果 mask 简化为示意图——说明"只展示时序关系，不展示具体 attention 计算细节"，**正确**。
- K3 工程实例只列配置项——说明"K3 报告未给出具体训练数据集、训练步数、学习率等细节"，**正确**。
- 教学解释"TTT 像在带噪声的数据上练答"类比——说明失效边界（不覆盖 mask 实现、不保证消除深度衰减、不等价推理时 fine-tune），**正确**。

### 7. 页面功能

- KaTeX 公式渲染配置正确。
- 折叠块（行 895-926 完整手算、行 928-967 伪代码、行 1042-1049 LK loss 梯度补充）均正确，summary 清晰，收起后正文仍有完整摘要。
- 目录锚点正确，scroll-margin-top 避开顶部导航。
- 来源引用 [C1]-[C10]、[F1]-[F5]、[N1]-[N5] 在文末完整列出，**引用原文准确可定位**。

## 问题

- [轻微·盲读] §3 行 758：$g_t \in \mathbb{R}^k$ 中 $k$ 首现未明确说明是 target 模型的 hidden size，要到行 851 手算例子才见"hidden size $k = 4$"。：首次出现时加"$k$ 是 target 模型的 hidden size（特征维度）"。｜ 修复： ｜ 复验：
- [轻微·盲读] §4 行 815：DraftLayer 首现于公式 $a_{t+1} = \text{DraftLayer}([g_{1:t}; e_{t+1}])$，未明确说即前述"单层 transformer decoder"，读者需回溯推断。：首次出现时加"DraftLayer 即 §3 的单层 transformer decoder"。｜ 修复： ｜ 复验：
- [轻微·技术] §3 行 748：页面引用"EAGLE-3 论文 Figure 1 的 scaling law 曲线"，但根据 OpenReview 评审版 EAGLE-3 论文，scaling law 曲线在 Figure 8（"Figure 8: Scaling law evaluated on the MT-bench"）。Figure 1 通常是架构总览图。图号引用可能与论文最终版不符。：核对 EAGLE-3 论文（arXiv:2503.01840）原图号，若 Figure 1 非 scaling law 则更正为正确图号，或改为"论文 scaling law 曲线"避免具体图号。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：进入修复

来源对照全部通过（C1-C10、F1-F5、N1-N5 与 EAGLE-1 论文 arXiv:2401.15077、EAGLE-3 论文 arXiv:2503.01840、K3 报告 §4.1.4 完全一致）；手算例子三步 100% 可复算（矩阵乘法、tanh、softmax、argmax 全部正确，偏差 L2 距离正确）；arXiv 编号、会议（ICML 2024、NeurIPS 2025）、加速比数字（2.7x-3.5x、5.6x、6.5x、1.38x）均有来源支持。三个轻微问题分别为 $k$ 符号首现未解释、DraftLayer 首现未明确、Figure 1 图号引用可能不准确，均不影响核心论断与主线理解。
