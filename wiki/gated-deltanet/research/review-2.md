# Gated DeltaNet 独立审查（第二次）

- 审查者：独立上下文（AI 模拟小白读者 + 来源对照）
- 页面版本：wiki/gated-deltanet/index.html（1392 行）、overview.html（82 行）
- 时间：2026-08-09
- 来源：[Y] Yang, Kautz & Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025, arXiv:2412.06464；[K3] k3-report.txt §2.1.1（KDA 公式与参数化）

## 段 A 盲读

按页面顺序阅读，记录小白读者理解主线上的卡点。

1. 开篇 callout 用"阅读长文档时需忘掉整段背景"引入问题，直觉且具体。context-box 给出前置概念（Delta 规则、线性注意力）与来源。

2. S1 对比 DeltaNet（缺全局遗忘）与 Mamba2（缺方向擦除），三模型对照表清晰。关键引文"gating enables rapid memory erasure while the delta rule facilitates targeted updates"点出互补性。

3. S2 给出公式 F1：$S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t k_t^\top$。符号表完整，形状检查验证维度自洽。α_t/β_t 职责分工用编号列表明确："α_t 控全局衰减、β_t 控单点覆写"。

4. S2 "先衰减后擦写"等价解读：因 α_t 是标量可与矩阵交换，公式可读作三步（衰减→擦除→写入）。前提条件（α_t 是标量）明确标注，并指出 KDA 的 channel-wise α_t 不能这样交换。

5. S3 手算例子（$d=2$，3 步，正交 key）清晰展示 α_t 衰减非 $k_t$ 方向、β_t 覆写 $k_t$ 方向。三模型对比表（DeltaNet/Mamba2/Gated DeltaNet）在同一序列上并排展示 $S_3$、$S_3 k_3$、$S_3 k_2$，三个关键观察精准。代码复算一致 ✓。

6. S4 退化关系（α→1 退化为 DeltaNet、β→0 退化为 Mamba2 无写入、α→0 清空历史）从公式直接代入。chunkwise 并行算法说明 chunk 内并行 + chunk 间递归，α_t 不破坏 Tensor Core 兼容性。

7. S5 KDA 对比：α_t 标量 vs channel-wise 向量、sigmoid vs scaled sigmoid（lower-bounded $g_{\min}=-5$）、full-rank output gate。公式差异（KDA 的 Diag(α_t) 在左侧、不可交换）正确指出。实验数字（PPL、S-NIAH-2）从论文 Table 2/3 引用。

8. 逐题核对学习目标：
   - "DeltaNet 的 delta 规则在快速遗忘上有什么缺陷？为什么要加 α_t？" → S1 ✓
   - "给定参数手算一步更新，说明 α_t、β_t 职责分工" → S2+S3 ✓
   - "α_t→1 与 β_t→0 退化为哪个模型？为什么？" → S4 ✓
   - "并行训练算法解决什么问题？为什么不能直接串行递归？" → S4 ✓
   - "Gated DeltaNet 与 KDA 在 α_t 设计上有什么区别？" → S5 ✓

   全部学习目标由正文章节完整回答。

## 段 B 对照来源

### 1. 定义与机制

- C1（DeltaNet 缺全局遗忘）：[Y] §2.3 原文 "lacks the ability to rapidly clear outdated information, particularly during context switches"。摘要确认 "Delta rule updates allow precise, targeted memory modifications but lack the ability to quickly clear outdated information" ✓
- C2（Mamba2 标量衰减无差别）：[Y] §2.2 原文 "does not account for the varying importance of different key-value associations"。摘要确认 "Gating mechanisms enable rapid memory erasure but affect all stored information equally" ✓
- C3（Gated DeltaNet 公式 F1）：[Y] §3.1 Eq.8。ArxivLens 来源确认公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$ ✓
- C4（α_t 与 β_t 职责分工）：[Y] §3.1 原文 "the data-dependent gating term α_t ∈ (0,1) controls state decay"。摘要确认互补性 ✓
- C5（"先衰减后擦写"等价）：由 C3 公式 + 标量交换律直接推导。前提条件（α_t 是标量）正确标注 ✓
- C6（chunkwise 并行算法）：[Y] §3.2。ArxivLens 确认 "WY representation" 与 "tensor core GPU optimization"。Bischof & Van Loan 1985 引用正确 ✓
- C7（互补性论断）：[Y] §1 原文 "we observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates"。与摘要一致 ✓
- C8（KDA 三项改动）：[K3] §2.1.1 确认。(1) "KDA extends the delta-rule recurrence with a channel-wise forget gate" — α_t 是向量 ✓；(2) "Kimi K3 instead uses a scaled sigmoid to bound the log-decay from below" + "gmin = −5" — lower-bounded ✓；(3) "Kimi K3 changes KDA's output gate from the low-rank parameterization" — full-rank ✓

### 2. 公式与推导

- F1（Gated DeltaNet $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$）：[Y] §3.1 Eq.8。ArxivLens 确认 ✓
- F2（退化 α→1 = DeltaNet、β→0 = α·S_{t-1}）：由 F1 直接代入，推导正确 ✓
- F3（Mamba2 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$）：[Y] §2.2。标准线性注意力 with gating ✓
- F4（DeltaNet $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$）：DeltaNet 论文 NeurIPS 2024 §2.2。标准 delta rule ✓
- KDA 公式对比：页面写 $S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$。[K3] Eq.1 原文 "$S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$"。逐字符匹配 ✓。注意 KDA 状态 $S_t \in \mathbb{R}^{d_k \times d_v}$（转置于 Gated DeltaNet 的 $d_v \times d_k$），页面正确使用了对应的 $k_t v_t^\top$ 与 $v_t k_t^\top$。

### 3. 可运行代码

S4 代码块已实际执行。输出与页面"预期输出"逐行一致：

```
DeltaNet S_3 = [[0,0],[1,1]], S_3@k_3=[0,1], S_3@k_2=[0,1]
Gated DeltaNet S_3 = [[0,0],[1,0.5]], S_3@k_3=[0,1], S_3@k_2=[0,0.5]
Mamba2 S_3 = [[0.5,0],[0,0.5]], S_3@k_3=[0.5,0], S_3@k_2=[0,0.5]
Degradation alpha=1: True, beta=0: True
```

一致 ✓

### 4. 事实与推断

- N1（1.3B/100B FineWeb-Edu 实验，Table 2 PPL 数字）：页面引用 [Y] §4 Table 2。论文 HTML 不可用，无法逐数字核对；但趋势（Gated DeltaNet < Mamba2 < DeltaNet < Transformer++ 的 PPL）与摘要"consistently surpasses"一致。页面在来源说明中标注了全部数字与"不外推到其他规模或数据"。
- N2（S-NIAH-2 4K：DeltaNet=18.6, Mamba2=56.2, Gated DeltaNet=92.2）：页面引用 [Y] §4 Table 3。论文 HTML 不可用；趋势（Gated DeltaNet >> Mamba2 > DeltaNet）与"number-in-haystack 检索"摘要一致。
- N3（常识推理 7 项平均数字）：页面引用 [Y] §4 Table 2。同 N1，趋势一致。
- K3 配置（93 层中 69 层 KDA）：[K3] Table 1 确认 "#Layers: 93"、"69 KDA + 24 MLA"。69+24=93 ✓
- K3 的 KDA lower bound $g_{\min}=-5$：[K3] §2.1.1 确认 "gmin = −5" ✓
- α_t 下界 $e^{-5} \approx 0.0067$：由 $g_{\min}=-5$ 推导（$\alpha = e^g$，$g > -5 \Rightarrow \alpha > e^{-5}$）✓
- "Qwen3-Next 等用 Gated DeltaNet 作线性层"：页面未给出具体来源引用。此为现实应用陈述，页面来源说明未覆盖（见问题 1）。

### 5. 前置知识引用

- ../delta-rule/index.html — 目录存在 ✓
- ../linear-attention/index.html — 目录存在 ✓
- ../kda/index.html — 目录存在 ✓

### 6. 教学简化

- $d_v = d_k = 2$ 极小维度——已说明 ✓
- α_t 取 0.5 便于手算——已说明 ✓
- 未展开 WY 表示与 chunkwise 推导——已说明 ✓
- 未展开多头实现——已说明 ✓
- 代码用 numpy 标量循环——已说明 ✓
- 未展开混合架构 H1/H2——已说明 ✓
- 未展开在线学习视角——已说明 ✓

### 7. 页面功能

- KaTeX 公式渲染：正确 ✓
- 折叠块（WY 表示、逐步代入、代码）：收起后主线不依赖其内容 ✓
- 目录锚点：h2/h3 带 id ✓
- `<pre class="diagram">` 标签闭合正确 ✓

## 问题

- [轻微·来源] S5 "现实应用"："Qwen3-Next 等混合架构用 Gated DeltaNet 作线性层（多数层 GDN + 少数层全注意力）"——此陈述涉及具体模型的架构选择，但页面来源与教学说明中未给出引用来源（论文 [Y] 未提及 Qwen3-Next，k3-report 也未涉及）。若此为公开已知信息，应标注来源；若无法定位来源，应标注"公开报道"或弱化为"已有混合架构模型采用 Gated DeltaNet 作线性层"。修法：补充 Qwen3-Next 架构选择的来源引用，或弱化为不点名具体模型的通用陈述 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：进入修复（1 条轻微为现实应用陈述缺少来源引用，改动量小，不影响核心结论与主线理解）
