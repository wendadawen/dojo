# Delta 规则与 DeltaNet · 内容范围

## 1. 概念歧义处理

**状态**：已裁定。

"Delta rule" 在机器学习文献中有两个常见含义，必须先裁定本页采用哪个：

1. **经典 Widrow-Hoff delta rule**（1960，又称 LMS / 最小均方）：训练阶段更新参数权重的规则，形式 $w \leftarrow w + \eta (y - \hat y) x$，是感知机和现代梯度下降的早期原型。
2. **DeltaNet 的 delta rule**（Schlag et al. 2021 ICML 起）：前向推理阶段更新线性注意力记忆矩阵 $S$ 的递归规则，形式 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$。

两者形式上同构（都是"误差 × 学习率 × 输入"），但作用对象完全不同：经典 delta rule 修改训练参数，DeltaNet delta rule 修改前向记忆状态。本页以 DeltaNet delta rule 为正题，经典 delta rule 仅作为类比与命名来源在文中明确区分。

"DeltaNet" 在文献中可指：
- Schlag et al. 2021 ICML 论文（首次将 delta rule 引入线性 Transformer，原文称 "Delta Network"，缩写 "Delta Net"）
- Yang et al. NeurIPS 2024（arXiv:2406.06484，正式使用 "DeltaNet" 命名，给出并行训练算法，扩展到 1.3B / 100B tokens）
- Yang et al. ICLR 2025（arXiv:2412.06464，"Gated DeltaNet"，在 delta rule 上加遗忘门 $\alpha_t$）

本页以 **Yang et al. NeurIPS 2024** 的 DeltaNet 公式为正式定义，Schlag 2021 作为 delta rule 的提出者溯源，Gated DeltaNet 作为延伸。

裁定依据：原始论文 ICML 2021 + NeurIPS 2024 + ICLR 2025 三篇正式会议论文，优先级最高。

## 2.1 概念含义

- **概念名称**：Delta 规则（Delta Rule）与 DeltaNet
- **英文名称**：Delta Rule；DeltaNet（Delta Network 的缩写）
- **常见缩写**：DeltaNet（模型名），$\Delta$ rule（公式名）
- **一句话定义**：Delta 规则是一种递归状态更新规则——在向记忆矩阵写入新的 key-value 关联前，先沿当前 key 方向擦除记忆中已存在的旧值，再做加权写入；DeltaNet 是把线性注意力的纯加性累加替换为 delta 规则的模型。
- **正式定义**（与权威来源一致）：
  - Delta 规则（DeltaNet 形式，Yang 2024 NeurIPS §2.2）：$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，其中 $\beta_t = \sigma(W_\beta x_t) \in (0,1)$，$S \in \mathbb{R}^{d_v \times d_k}$，$q_t, k_t, v_t$ 由 $x_t$ 经 $W_Q, W_K, W_V$ 线性投影得到，输出 $o_t = S_t q_t$。
- **本文采用的语境**：作为线性注意力记忆更新的替代规则，以及使用该规则的模型（DeltaNet 与 Gated DeltaNet）。

### 包括什么

- 线性注意力的纯加性累加 $S_t = S_{t-1} + v_t k_t^\top$ 作为动机基线（C1）：说明 delta 规则替代的是什么
- 线性注意力在 $L > d$ 时的"key 碰撞"/retrieval error 问题（C2）：说明为什么需要替代
- Delta 规则的两种等价形式：紧凑形式 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 与"先擦除后写入"形式 $S_t = S_{t-1} - v_t^{\text{old}} k_t^\top + v_t^{\text{new}} k_t^\top$（C3, C4）：核心机制
- $\beta_t$ 的语义（写入强度 / 学习率，sigmoid 输出）与边界情况 $\beta_t \to 0, \beta_t \to 1$（F2）：核心机制
- $I - \beta k k^\top$ 的 Householder / 投影性质（C6）：解释擦除的几何含义
- DeltaNet 的定义（线性注意力 + delta 规则）（C3 + 模型定位）
- 与 Mamba2（$\alpha_t S_{t-1} + v_t k_t^\top$）、线性注意力（$S_{t-1} + v_t k_t^\top$）、Gated DeltaNet（$S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$）的机制对比（C7）
- 一个可手算的小维度数字例子（d=2，2 步序列）
- 一段可独立运行的 Python 代码，实现 delta 规则递归并与加性累加对比
- Widrow-Hoff 1960 经典 delta rule 的简短类比与边界说明（C5）

### 不包括什么

- WY 表示（Bischof & Van Loan 1985）与 chunkwise 并行训练算法的完整推导：属于独立数值线性代数概念，Yang 2024 §3 已有完整论证；本页只在"扩展边界"中提及存在，不展开
- Triton kernel 实现细节：工程内容，不影响概念理解
- 完整语言建模 benchmark（PPL、零样本下游任务）：属于模型评估，本页只引用一两句关键数字说明 delta 规则的实际效果
- Kimi K3 / KDA 架构详情：用户提到作为动机，但 KDA 本身是独立概念，本页只在开头一句话提及作为学习动机，不展开
- Hebbian vs. delta rule 的理论容量分析（Schlag 2021 §5, Gardner 1988）：理论扩展，不影响机制理解
- 反向传播通过递归的梯度计算：属于训练理论，不影响前向机制
- 多头实现、head dimension 选择：工程
- Hybrid 模型（DeltaNet + sliding attention、DeltaNet + global attention）：属于架构组合，不影响 delta 规则本身

### 相邻概念

- **线性注意力（Linear Attention）**：DeltaNet 的母体概念，本页开头即依赖。`wiki/linear-attention/` 不存在，按用户指令标占位（不递归生成）。本页正文给出 delta 规则所需的最小线性注意力事实（加性递归 + retrieval），不重复讲解。
- **Mamba2 / 状态空间模型**：Gated DeltaNet 的对比对象。$\alpha_t S_{t-1} + v_t k_t^\top$ 是 Mamba2 的更新规则，本页只引用此式作对比，不展开 SSM 理论。
- **GLA、RetNet、RWKV-6**：其他线性 RNN 模型，与 DeltaNet 同属"矩阵状态 + 结构化递归"家族，本页只在一句话对比中提及，不展开。
- **Widrow-Hoff 1960 经典 delta rule**：训练阶段的参数更新规则，与 DeltaNet delta rule 形式同构但作用对象不同（C5）。本页作为类比和命名溯源提及，不展开训练理论。
- **LSTM 遗忘门**：Gated DeltaNet 的 $\alpha_t$ 在直觉上类似 LSTM 遗忘门，但作用对象与逐元素性质不同；本页只在边界说明中提及区别。

## 2.2 学习目标

### Q1：为什么线性注意力的"加性累加"在长序列上会出现"key 碰撞"，delta 规则如何解决这个问题？

- **完成答案**：读者能说明——线性注意力把所有历史压成一个矩阵状态 $S = \sum_i v_i k_i^\top$；retrieval 时 $S k_j = v_j + \sum_{i \neq j} (k_i^\top k_j) v_i$，干扰项 $\sum_{i \neq j} (k_i^\top k_j) v_i$ 即 retrieval error；d 维空间最多容纳 d 个正交 key，当 $L > d$ 时必然出现 $k_i^\top k_j \neq 0$，碰撞不可避免；delta 规则在写入新 key-value 前先擦除旧 key 方向的分量，避免无脑堆叠。
- **为什么是核心目标**：不理解动机，就无法理解 delta 规则的设计意图，公式变成无意义符号。
- **依赖内容**：线性注意力的矩阵状态 $S$、key/value/query、外积、retrieval 操作 $S k$。

### Q2：给定 $S_{t-1}$、$k_t$、$v_t$、$\beta_t$，用 delta 规则公式 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 手算一步更新，并说明每个符号含义与 $\beta_t \in (0,1)$ 的作用。

- **完成答案**：读者能逐项代入数字、计算矩阵-向量乘法和外积、说明 $S$ 的形状 $\mathbb{R}^{d_v \times d_k}$、说明 $\beta_t$ 是写入强度（学习率），控制新值覆盖旧值的程度。
- **为什么是核心目标**：公式是 delta 规则的正式定义，不手算无法验证理解。
- **依赖内容**：矩阵-向量乘法、外积 $v k^\top$、单位矩阵、sigmoid 输出范围。

### Q3：把 delta 规则改写为"先擦除后写入"形式 $S_t = S_{t-1} - v_t^{\text{old}} k_t^\top + v_t^{\text{new}} k_t^\top$（其中 $v_t^{\text{old}} = S_{t-1} k_t$，$v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$），并说明两种形式等价。

- **完成答案**：读者能把 $v_t^{\text{new}}$ 代入展开，验证等式两边相等；解释 $v_t^{\text{old}}$ 是"当前记忆对 $k_t$ 的响应"、$v_t^{\text{new}}$ 是新旧值的凸插值。
- **为什么是核心目标**："先擦除后写入"是 delta 规则的核心直觉；它是公式的等价形式而非独立规则，理解等价性才能避免把"擦除"误读为硬覆盖。
- **依赖内容**：矩阵代数展开、凸组合。

### Q4：说明 $\beta_t \to 1$ 和 $\beta_t \to 0$ 时 delta 规则分别退化为哪种更新规则，并指出各自的语义与前提条件。

- **完成答案**：$\beta_t = 0$ 时 $S_t = S_{t-1}$（记忆完全不变）；$\beta_t = 1$ 时 $S_t = S_{t-1}(I - k_t k_t^\top) + v_t k_t^\top$，当 $\|k_t\| = 1$ 时 $I - k_t k_t^\top$ 是沿 $k_t$ 方向的正交投影，意味着完全擦除 $k_t$ 方向的旧关联并写入新值。读者需说明 $\|k_t\| = 1$ 是投影性质的必要前提。
- **为什么是核心目标**：边界情况是验证对 $\beta_t$ 与 $I - k k^\top$ 理解的最简方式，也是误解多发点。
- **依赖内容**：边界代入、正交投影定义、向量范数。

### Q5：DeltaNet 与线性注意力、Mamba2、Gated DeltaNet 在记忆更新机制上的关键差异是什么？为什么 Gated DeltaNet 要在 delta 规则上再加一个 $\alpha_t$ 门？

- **完成答案**：读者能说出——线性注意力是纯加法（不擦除，碰撞不可避免）；DeltaNet 用 delta 规则选择性擦除单个 key 方向（精细但缺少全局遗忘）；Mamba2 用标量 $\alpha_t$ 衰减整个状态（全局遗忘但无选择性）；Gated DeltaNet 把两者结合——$\alpha_t \to 0$ 时清空所有记忆，$\alpha_t \to 1$ 时退化为纯 delta 规则。
- **为什么是核心目标**：相邻概念的区分是理解 delta 规则边界的关键，避免把它误当成万能方案。
- **依赖内容**：三种更新规则公式对比、标量门与方向门的区别。

## 2.3 内容分级

### 核心内容

- 线性注意力加性递归 $S_t = S_{t-1} + v_t k_t^\top$ 与 retrieval $S q$（C1，Q1）
- $L > d$ 时的 retrieval error $S k_j = v_j + \sum_{i \neq j}(k_i^\top k_j) v_i$（F1，Q1）
- Delta 规则紧凑公式 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$（C3，Q2）
- Delta 规则"先擦除后写入"等价形式（C4，Q3）
- $\beta_t$ 语义与边界 $\beta_t \to 0, \beta_t \to 1$（F2，Q2、Q4）
- $I - \beta k k^\top$ 的 Householder / 投影性质与 $\|k\| = 1$ 前提（C6，Q3、Q4）
- DeltaNet 定义：线性注意力 + delta 规则（C3 + 模型定位，Q5）
- Mamba2 / Gated DeltaNet 对比公式（C7，Q5）
- Widrow-Hoff 1960 与 DeltaNet delta rule 的区分（C5，Q1 边界）

### 辅助内容

- 手算例子：d=2 维、2 步序列，对比加性累加与 delta 规则（消除"公式能背但不会算"的理解障碍）
- 可运行 Python 代码：实现 delta 规则递归并与加性累加对比（澄清"实际效果"的疑问）
- 来源溯源时间线：Schlag 2021 → Yang 2024 NeurIPS → Yang 2024 ICLR 2025（澄清"DeltaNet 这个名字到底来自哪篇论文"）

### 扩展内容

- WY 表示与 chunkwise 并行训练算法：不纳入本页（属于数值线性代数与工程）
- flash-linear-attention 库的 Triton kernel：不纳入
- 完整 LM benchmark：不纳入，只引用一句关键数字
- Kimi K3 KDA 架构：不纳入，开头一句话动机
- 状态追踪 / TC^0 复杂度理论：不纳入
- Hybrid 模型：不纳入

## 2.4 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | wiki 页面状态 | 处理方式 |
|---|---|---|---|
| 线性注意力（Linear Attention） | Q1, Q2, Q5 | `wiki/linear-attention/` 不存在 | 按用户指令标占位，不递归生成；正文给出 delta 规则所需的最小事实（加性递归 + retrieval），不展开背景 |
| 矩阵-向量乘法、外积、单位矩阵、sigmoid | Q2, Q3, Q4 | 非 Dojo 概念页范畴 | 假设读者已掌握基础线性代数，公式中首次出现时给出形状与定义 |
| 注意力机制（softmax attention）的一般概念 | Q5（对比） | `wiki/` 下无对应页面 | 假设读者有"注意力 = Q·K·V"的常识；本页不展开 softmax attention |
| Mamba2 / SSM | Q5（对比） | `wiki/` 下无对应页面 | 只引用 Mamba2 的更新公式 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$，不展开 SSM |

递归深度：本页前置知识中的"Dojo 概念页"只有线性注意力一项，按用户指令标占位不生成，不进入递归流程。

## 2.5 明确不展开的内容

- **WY 表示与并行训练算法**：与 delta 规则的概念机制正交，属于"如何高效计算"而非"是什么"。Yang 2024 NeurIPS §3 完整论证。本页只提及"DeltaNet 通过 WY 表示实现了 chunkwise 并行训练"作为延伸说明，不展开推导。
- **Triton kernel 实现**：纯工程，不影响概念理解。
- **完整 LM benchmark**：1.3B/100B 的 PPL 表只引用一句关键数字说明实际效果，不抄录完整表。
- **Kimi K3 KDA**：用户提到作为学习动机，但 KDA 是独立架构概念；本页开头一句话提及作为学习动机，不展开 KDA 自身。
- **Hebbian vs. delta rule 容量分析**：Schlag 2021 §5 引用 Gardner 1988 的理论分析，属于理论扩展，不影响机制理解。
- **反向传播梯度**：训练理论，不影响前向机制。
- **Hybrid 模型**：DeltaNet + sliding attention 等组合属于架构工程，不影响 delta 规则本身。
- **多头 / head dimension 选择**：工程内容。

## 2.6 常见误解和适用边界

### 误解 1：Delta 规则就是经典 Widrow-Hoff delta rule

- **错误理解**：Delta 规则 = 经典 delta rule，是用来训练感知机权重的那个。
- **正确结论**：形式上同构（都是"误差 × 学习率 × 输入"），但作用对象不同——经典 delta rule 修改训练阶段参数权重 $w \leftarrow w + \eta(y - \hat y)x$；DeltaNet delta rule 修改前向推理阶段记忆矩阵 $S \leftarrow S + \beta(v - Sk)k^\top$。Schlag 2021 §4.2 明确说"akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)"，是类比而非同一对象。
- **形成原因**：名称相同，且两者都叫"学习率"。
- **影响**：Q1 边界、Q5。

### 误解 2：Delta 规则"完全擦除"旧 key-value 对

- **错误理解**：写入新值时旧 key-value 关联被完全删除。
- **正确结论**：仅当 $\beta_t = 1$ 且 $\|k_t\| = 1$ 时完全擦除；一般 $\beta_t \in (0,1)$ 是部分擦除——新值是 $v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ 的新旧凸插值。
- **形成原因**："remove"字面理解，忽略 $\beta_t$ 的插值作用。
- **影响**：Q2、Q4。

### 误解 3：$I - \beta k k^\top$ 总是投影矩阵

- **错误理解**：矩阵 $I - \beta k k^\top$ 是正交投影。
- **正确结论**：仅当 $\|k\| = 1$ 且 $\beta = 1$ 时是投影矩阵（沿 $k$ 方向的正交投影）；其他情况下是秩-1 扰动，不保持范数。Yang 2024 NeurIPS §3.3 明确说 DeltaNet 在工程实现中使用 L2 归一化以获得投影性质。
- **形成原因**：论文提及投影性质但读者忽略前提条件。
- **影响**：Q3、Q4。

### 误解 4：DeltaNet 在所有任务上必然优于线性注意力 / Mamba

- **错误理解**：DeltaNet 是"更好的线性注意力"，全面占优。
- **正确结论**：DeltaNet 在合成检索任务（MQAR、MAD recall 类）上显著占优；但在 MAD memorize 任务上反而弱于 Mamba（Yang 2024 NeurIPS Table 2：DeltaNet 52.8 vs. Mamba 89.5）。1.3B LM PPL 上略优于 Mamba 和 GLA，与 Transformer++ 接近。
- **形成原因**：论文摘要强调优势。
- **影响**：Q5。

### 误解 5：Gated DeltaNet 的 $\alpha_t$ 就是 LSTM 遗忘门

- **错误理解**：$\alpha_t$ 等同于 LSTM 的遗忘门 $f_t$。
- **正确结论**：$\alpha_t$ 是数据相关标量，直接乘到整个矩阵状态上 $S_{t-1} \to \alpha_t S_{t-1}$（标量-矩阵乘）；LSTM 遗忘门是逐元素作用于 cell state 向量（向量），且 LSTM 的 cell state 不是外积记忆矩阵。两者直觉上都是"遗忘"，但作用对象与代数结构不同。
- **形成原因**：名词相似。
- **影响**：Q5。

### 适用边界

- **Delta 规则适用的场景**：序列模型中需要"key-value 关联记忆"的递归更新，且 key 维度 $d$ 远小于序列长度 $L$ 的场景；具体用于线性注意力变体的前向递归。
- **不解决**：训练阶段的参数更新（由反向传播 + 优化器负责）；经典监督学习中的样本权重学习。
- **退化条件**：
  - 所有 $\beta_t = 0$：退化为"完全保留旧状态"，等于不更新
  - 所有 $\beta_t = 1$ 且 $\|k_t\| = 1$：退化为"完全覆盖"，每次写入完全擦除旧 key 方向
  - $L \leq d$ 且所有 $k_t$ 两两正交：retrieval error 恒为 0，加性累加与 delta 规则等价（此时 delta 规则的"擦除"操作是冗余的）
- **$L > d$ 时**：delta 规则仍受容量限制约束，但能更好地管理容量——通过擦除旧 key 方向为新 key 腾出空间，避免无脑堆叠导致的灾难性碰撞。
- **数值稳定性**：$\beta_t \in (0,1)$（sigmoid 输出）保证更新有界；$S$ 范数不会无界增长。
