# Gated DeltaNet · 内容范围

## 1. 概念歧义处理

**状态**：已裁定。

概念名称存在两层歧义，必须先裁定：

### 1.1 论文标题与 arXiv 编号的对应

任务描述把两篇论文标为 "Yang et al. 2024 'Gated Delta Networks'（arXiv:2406.06484 NeurIPS 2024）" 和 "Yang et al. 2025 'Gated DeltaNet'（arXiv:2412.06464 ICLR 2025）"。经核对原始来源，实际对应关系为：

- **arXiv:2406.06484** = Yang, Wang, Zhang, Shen & Kim. "Parallelizing Linear Transformers with the Delta Rule over Sequence Length." NeurIPS 2024。这是 **DeltaNet** 论文（提出 DeltaNet 命名与并行训练算法），不是 Gated DeltaNet。
- **arXiv:2412.06464** = Yang, Kautz & Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025。这是 **Gated DeltaNet** 论文（论文标题为 "Gated Delta Networks"，但社区与论文正文均用 "Gated DeltaNet" 指代该模型）。

**裁定**：本页以 arXiv:2412.06464 (ICLR 2025) 为 Gated DeltaNet 的正式来源，论文标题 "Gated Delta Networks" 与模型名 "Gated DeltaNet" 视为同一概念的不同写法。arXiv:2406.06484 (NeurIPS 2024) 作为 DeltaNet 的来源，由前置概念页 `wiki/delta-rule/` 已覆盖，本页只引用其公式作对比。裁定依据：原始 arXiv 论文与已生成的前置概念页 `wiki/delta-rule/index.html` 的来源说明一致。

### 1.2 "Gated DeltaNet" 与 "Gated Delta Networks" 是否同一模型

论文标题为 "Gated Delta Networks"，论文正文 §3.1 用 "Gated DeltaNet" 指代所提模型（Table 1、§4 实验均用 "Gated DeltaNet"）。社区（如 Qwen3-Next、flash-linear-attention 库）也通用 "Gated DeltaNet"。两者为同一模型的不同写法，本页采用 "Gated DeltaNet" 为主名。

### 1.3 Gated DeltaNet 与 KDA 的区别

KDA（Kimi Delta Attention）基于 Gated DeltaNet 但有三项改动：(1) α_t 从标量改为 channel-wise 向量；(2) decay 参数化改为 lower-bounded scaled sigmoid；(3) full-rank output gate。KDA 是独立架构概念，本页只在结尾一句话提及作为现实应用，不展开，引用 `wiki/kda/`。

## 2.1 概念含义

- **概念名称**：Gated DeltaNet（门控 DeltaNet）
- **英文名称**：Gated DeltaNet；论文标题 "Gated Delta Networks"
- **常见缩写**：GDN（论文 §3.3、§4 混合架构用此缩写）
- **一句话定义**：Gated DeltaNet 是一种线性注意力递归模型——在 DeltaNet 的 delta rule 基础上，给每一步的整个记忆状态乘一个数据相关的标量衰减门 α_t，让模型既能用 delta rule 精准覆写单个 key-value 关联，又能用 α_t 快速遗忘整段过时记忆。
- **正式定义**（与权威来源一致，Yang 2025 ICLR §3.1 Eq.8）：

  $$S_t = S_{t-1}\bigl(\alpha_t(I - \beta_t k_t k_t^\top)\bigr) + \beta_t v_t k_t^\top$$

  其中 $S \in \mathbb{R}^{d_v \times d_k}$ 是记忆状态矩阵；$q_t, k_t, v_t$ 由输入 $x_t$ 经线性投影得到；$\alpha_t = \sigma(\cdot) \in (0,1)$ 是数据相关标量衰减门；$\beta_t = \sigma(\cdot) \in (0,1)$ 是数据相关标量写入强度；输出 $o_t = S_t q_t$。

- **本文采用的语境**：作为 DeltaNet（delta rule）与 Mamba2（标量衰减门）的统一递归模型，讲解其公式、动机、与两者的关系、并行训练算法的存在与作用。

### 包括什么

- DeltaNet 缺乏全局遗忘能力的问题（C1）：为什么 delta rule 只能精准覆写单个 key，不能快速清空过时段落
- Mamba2 的标量衰减门机制（C2）：α_t 全局衰减整张状态矩阵
- Gated DeltaNet 公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$（F1，C3）：核心机制
- α_t 与 β_t 的职责分工（C4）：α_t 控全局衰减、β_t 控单点覆写强度，两者都是数据相关标量
- Gated DeltaNet 的退化（F2）：α_t→1 退化为 DeltaNet；β_t→0 退化为 Mamba2 的无写入特殊情形
- "先衰减后擦写"的等价解读（C5）：α_t 先作用整张 S_{t-1}，再 delta rule 擦写
- chunkwise 并行训练算法的存在与作用（C6）：基于 WY 表示扩展，使训练可并行、兼容 Tensor Core
- 与 Mamba2、DeltaNet 的机制对比（C7）
- 与 KDA 的关系（C8）：KDA 把 α_t 改为 channel-wise + lower-bounded decay
- 一个可手算的小维度数字例子（d=2，2-3 步序列），展示 α_t 与 β_t 的协同
- 一段可独立运行的 Python 代码，验证 Gated DeltaNet 递归与 DeltaNet、Mamba2 的对比

### 不包括什么

- WY 表示（Bischof & Van Loan 1985）与 chunkwise 算法的完整推导：属于数值线性代数，Yang 2025 ICLR §3.2 已有完整论证；本页只说明存在与作用，不展开 UT 变换、前向替换等细节
- Triton kernel 实现细节：工程内容，不影响概念理解
- 完整语言建模 benchmark：本页只引用 1.3B/100B tokens 的关键 PPL 数字与 S-NIAH 检索数字说明效果
- KDA 架构详情：独立概念，引用 `wiki/kda/`
- 混合架构（Gated DeltaNet + SWA + Mamba2）的完整设计：属于架构组合，本页只在一句话提及存在
- 反向传播通过递归的梯度计算：训练理论，不影响前向机制理解
- 多头实现、head dimension 选择：工程
- 在线学习视角（Table 1 的目标函数）的完整推导：论文 §3.1 给出 gated delta rule 对应的带自适应权重衰减的在线学习目标，本页只提及存在

### 相邻概念

- **DeltaNet / delta rule**：Gated DeltaNet 的直接基础。`wiki/delta-rule/` 已生成，本页引用其公式与几何性质，不重复推导。
- **线性注意力**：DeltaNet 的母体。`wiki/linear-attention/` 已生成，本页只通过 delta-rule 间接依赖。
- **Mamba2 / 状态空间模型**：Gated DeltaNet 的门控来源。Mamba2 公式 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ 是对比基线，本页引用此式作对比，不展开 SSM 理论。
- **KDA**：基于 Gated DeltaNet 的工程化改型。`wiki/kda/` 已生成，本页结尾引用。

## 2.2 学习目标

### Q1：DeltaNet 的 delta rule 在"快速遗忘整段记忆"上有什么缺陷？Gated DeltaNet 为什么要加 α_t？

- **完成答案**：读者应能说明——delta rule 的 $I - \beta_t k_t k_t^\top$ 只擦除当前 $k_t$ 一个方向的旧值，要清空整段过时记忆需要逐 key 串行擦除；Mamba2 的标量 α_t 一步衰减整张状态但无选择性；Gated DeltaNet 把两者结合，α_t 快速衰减全局、β_t k_t k_t^\top 精准覆写单点。
- **为什么是核心目标**：不理解这个动机就无法理解公式为什么这样设计。
- **依赖内容**：delta rule 公式（前置）、Mamba2 公式、Gated DeltaNet 公式。

### Q2：给定 $S_{t-1}$、$k_t$、$v_t$、$\alpha_t$、$\beta_t$，用 Gated DeltaNet 公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$ 手算一步更新，并说明每个符号含义与 α_t、β_t 的职责分工。

- **完成答案**：读者应能代入数字算出 $S_t$，并说明 α_t 是数据相关标量衰减门（控全局遗忘）、β_t 是数据相关标量写入强度（控单点覆写），两者职责不重叠。
- **为什么是核心目标**：公式是概念的核心，不能手算就没真正理解。
- **依赖内容**：Gated DeltaNet 公式、α_t 与 β_t 的语义、矩阵乘法结合律。

### Q3：Gated DeltaNet 在 α_t→1 与 β_t→0 两种极端下分别退化为哪个模型？为什么？

- **完成答案**：α_t→1 时退化为 DeltaNet（$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$）；β_t→0 时退化为 $S_t = \alpha_t S_{t-1}$（Mamba2 在 v_t=0 时的特殊情形）。读者应能从公式直接代入说明。
- **为什么是核心目标**：退化关系揭示 Gated DeltaNet 是 DeltaNet 与 Mamba2 的统一。
- **依赖内容**：Gated DeltaNet 公式、DeltaNet 公式、Mamba2 公式。

### Q4：Gated DeltaNet 的并行训练算法解决什么问题？为什么不能直接串行递归？

- **完成答案**：读者应能说明——递归形式 $S_t$ 依赖 $S_{t-1}$，串行计算无法利用 GPU 并行；chunkwise 算法把序列切成 chunk，chunk 内用 WY 表示把递归展开为矩阵乘法（兼容 Tensor Core），chunk 间递归传状态；门控项 α_t 只做逐元素乘法不破坏矩阵乘法结构。
- **为什么是核心目标**：不理解并行算法就无法理解为什么 Gated DeltaNet 能实际训练。
- **依赖内容**：chunkwise 形式、WY 表示、Tensor Core。

### Q5：Gated DeltaNet 与 KDA 在 α_t 的设计上有什么关键区别？

- **完成答案**：Gated DeltaNet 的 α_t 是标量（每步一个衰减率作用于整张 S），KDA 的 α_t 是 channel-wise 向量（每个 key 通道一个独立衰减率）且 decay 参数化有下界（scaled sigmoid 限定 $g \in (g_{\min}, 0)$）。读者应能说明这个区别影响什么：KDA 可按通道差异化遗忘，Gated DeltaNet 只能整张状态同速率遗忘。
- **为什么是核心目标**：KDA 是 Gated DeltaNet 的现实应用改型，理解两者的 α_t 差异是理解 KDA 创新的前提。
- **依赖内容**：Gated DeltaNet 公式、KDA 的 channel-wise 门与 lower-bounded decay（引用 `wiki/kda/`）。

## 2.3 内容分级

### 核心内容（缺少后导致至少一个学习目标无法完整回答）

- DeltaNet 缺乏全局遗忘的问题（C1）→ Q1
- Mamba2 标量衰减门机制（C2）→ Q1
- Gated DeltaNet 公式与符号定义（F1, C3）→ Q1, Q2, Q3
- α_t 与 β_t 的职责分工（C4）→ Q2
- Gated DeltaNet 的退化（F2）→ Q3
- chunkwise 并行算法的存在与作用（C6）→ Q4
- 与 KDA 的 α_t 区别（C8）→ Q5

### 辅助内容（消除关键理解障碍或常见误解）

- "先衰减后擦写"的等价解读（C5）：帮助理解公式乘法顺序
- α_t 的数据相关性：说明 α_t = σ(输入) 而非固定常数，这是它与早期 data-independent decay 的区别
- 误解排查：α_t 与 β_t 不是同一回事、α_t 不是 channel-wise

### 扩展内容（相关但不影响学习目标回答）

- 混合架构 Gated DeltaNet-H1/H2（纳入范围，一句话提及，不展开）
- 在线学习视角的目标函数（不纳入，只提及存在）
- 实验数字（纳入范围，引用 1.3B PPL 与 S-NIAH 关键数字说明效果，不展开完整 benchmark）

## 2.4 前置知识映射

| 前置概念 | 依赖的学习目标 | 概念页链接 | 递归深度 |
|---|---|---|---|
| Delta 规则 / DeltaNet | Q1, Q2, Q3, Q5 | `wiki/delta-rule/index.html`（已生成） | 0 |
| 线性注意力 | Q1（间接，通过 delta-rule） | `wiki/linear-attention/index.html`（已生成） | 0 |
| Mamba2 / 状态空间模型 | Q1, Q3 | 无概念页，本页给出最小事实（公式 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$） | — |

Mamba2 无概念页，按用户指令不递归生成（本页是 Gated DeltaNet，不是 Mamba2）。本页正文给出 Mamba2 公式作为对比基线，不展开 SSM 理论。

## 2.5 明确不展开的内容

- **WY 表示与 chunkwise 算法的完整推导**：属于数值线性代数独立概念，Yang 2025 ICLR §3.2 有完整论证。本页只说明存在、作用（使训练可并行、兼容 Tensor Core），不展开 UT 变换、前向替换、$\mathbf{A}_{[t]}^W$/$\mathbf{A}_{[t]}^U$ 等细节。原因：不影响 Gated DeltaNet 概念理解，读者需要时查论文 §3.2。
- **Triton kernel 实现**：工程内容，不影响概念。
- **混合架构 H1/H2 的完整设计**：属于架构组合独立概念，本页一句话提及存在与组成（GDN+SWA、GDN+Mamba2+SWA），不展开。
- **反向传播梯度计算**：训练理论，不影响前向机制。
- **多头实现**：工程，本页只讲单头递归。

## 2.6 常见误解和适用边界

### 常见误解

1. **误解**：Gated DeltaNet 的 α_t 和 KDA 的 α_t 是同一个东西。
   **正确**：Gated DeltaNet 的 α_t 是标量（每步一个衰减率作用于整张 S），KDA 的 α_t 是 channel-wise 向量（每通道独立）且有 lower-bounded decay。两者形式同源但设计不同。
   **形成原因**：两者都叫 α_t、都来自 sigmoid、都作用于 S_{t-1}。
   **影响**：Q5。

2. **误解**：α_t 和 β_t 都是"遗忘门"，职责重叠。
   **正确**：α_t 控全局衰减（整张 S 按比例缩小），β_t 控单点覆写强度（当前 k_t 方向的新旧值混合比例），职责不重叠。论文 §3.1 原文 "gating enables rapid memory erasure while the delta rule facilitates targeted updates"。
   **影响**：Q2。

3. **误解**：Gated DeltaNet 只是 DeltaNet 加了个 α_t 系数，没什么本质区别。
   **正确**：α_t 使 Gated DeltaNet 能一步衰减整段记忆（上下文切换），DeltaNet 要逐 key 擦除；S-NIAH-2/3 检索任务上 Gated DeltaNet 显著优于 DeltaNet（Yang 2025 ICLR Table 3）。
   **影响**：Q1。

4. **误解**：α_t 是固定的常数衰减率。
   **正确**：α_t 是数据相关的（α_t = σ(W_α x_t)），模型根据当前输入决定本步衰减多少。这是它相对早期 data-independent decay（如 RetNet 的固定 γ）的区别。
   **影响**：Q2。

### 适用边界

- **Gated DeltaNet 解决什么**：线性注意力家族中"精准覆写单点"与"快速遗忘全局"的统一。
- **不解决什么**：不解决全局内容交互（任意两个远距离 token 的直接注意力），这仍需标准注意力或混合架构；不解决 BF16 数值稳定性（KDA 才解决）。
- **结论成立条件**：α_t, β_t ∈ (0,1) 数据相关标量；chunkwise 训练需要 chunk size 为 16 的倍数以用 Tensor Core。
- **条件不满足时**：若 α_t 退化为常数，失去数据相关遗忘能力；若用 FP16/BF16 训练且 α_t 无下界，累积衰减倒数可能溢出（这是 KDA 改 lower-bound 的动机）。
