# Gated DeltaNet · 教学大纲

## 1. 页面开头

### 钩子问题

当一个语言模型在阅读一段长文档时，前面写了一段与当前问题无关的背景（比如旧话题的细节），模型需要"忘掉整段"重新聚焦。DeltaNet 的 delta rule 能精准覆写单个 key，但要逐 key 串行擦除整段过时记忆；Mamba2 的标量衰减门能一步衰减整张状态，但无差别衰减所有 key-value 关联。能不能两者都要——既精准覆写单点、又快速遗忘全局？

### 一句话解释

Gated DeltaNet 把 DeltaNet 的 delta rule（精准覆写）与 Mamba2 的标量衰减门 α_t（快速全局遗忘）统一进一个递归公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$。

### 学习承诺（读完能够回答）

见 scope.md Q1–Q5。

### 首个具体场景

沿用 delta-rule 概念页的 2 维小例子：$S_0 = 0$，写入 $(k_1, v_1)$ 后，第 2 步用相同 key 写新值——这次再加一个 α_t < 1 的全局衰减，看 α_t 与 β_t 如何协同。

### 与第一章的过渡

先看 DeltaNet 与 Mamba2 各自缺什么（第 1 章），再看 Gated DeltaNet 怎么把两者结合（第 2 章），再手算（第 3 章），再看退化与并行训练（第 4 章），最后看与 KDA 的关系（第 5 章）。

## 2. 章节设计

### S1：DeltaNet 与 Mamba2 各自缺什么——gating 与 delta rule 的互补性

- **主要教学问题**：DeltaNet 的 delta rule 在"快速遗忘整段记忆"上有什么缺陷？Mamba2 的标量衰减门缺什么？为什么两者互补？
- **对应范围**：Q1；C1, C2, C7。
- **正文要点**：
  - delta rule 只擦单个 k_t 方向，清空整段需逐 key 串行（C1）
  - Mamba2 的 α_t 全局衰减整张 S，不区分 key 重要性（C2）
  - 两者互补：gating 快速遗忘全局、delta rule 精准覆写单点（C7）
- **讲解材料及职责**：
  - DeltaNet 公式（F4，引用 `wiki/delta-rule/`）：复述不重推
  - Mamba2 公式（F3）：给出最小事实
  - 对照表格：方向擦除 vs 标量衰减
- **前置知识安排**：引用 `wiki/delta-rule/index.html`（delta rule 与 DeltaNet）、`wiki/linear-attention/index.html`（线性注意力）；Mamba2 无概念页，本页给出公式。
- **完成检查**：说明 delta rule 为什么不能快速清空整段；说明 Mamba2 为什么不能选择性擦除。
- **过渡**：两者互补——下一章给出把它们结合的公式。

### S2：Gated DeltaNet 的公式与符号

- **主要教学问题**：Gated DeltaNet 的递归公式是什么？α_t 与 β_t 各自的职责是什么？
- **对应范围**：Q2；F1, C3, C4, C5。
- **正文要点**：
  - 公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$（F1, C3）
  - 每个符号定义：S, q_t, k_t, v_t, α_t, β_t, I
  - α_t 与 β_t 的职责分工（C4）：α_t 全局衰减、β_t 单点覆写
  - "先衰减后擦写"等价解读（C5）：α_t 标量可交换，先衰减整张再 delta 擦写
  - 形状检查与边界检查
- **讲解材料及职责**：
  - 公式（F1）：核心
  - 形状检查：验证维度自洽
  - 边界检查：α_t→1, β_t→0, α_t→0 的快速代入
- **前置知识安排**：delta rule 公式（`wiki/delta-rule/`）。
- **完成检查**：说明 α_t 与 β_t 职责不重叠；说明乘法顺序为什么是"先衰减后擦写"。
- **过渡**：公式有了，下一章手算。

### S3：手算 Gated DeltaNet 一步更新

- **主要教学问题**：给定具体数字，怎么代入公式算出 S_t？α_t 与 β_t 如何协同？
- **对应范围**：Q2；F1。
- **正文要点**：
  - 教学示例：d=2 维，2-3 步序列，α_t 与 β_t 取不同值
  - 逐步代入：先算 α_t S_{t-1}（衰减）、再算 (I - β_t k_t k_t^\top)（擦除）、再算 β_t v_t k_t^\top（写入）
  - 对照 DeltaNet（α_t=1）与 Mamba2（β_t=0）在同一序列上的结果
  - 把数字结果翻译成机制结论
- **讲解材料及职责**：
  - 数字例子（教学示例）：展示 α_t 与 β_t 协同
  - 对照表格：Gated DeltaNet vs DeltaNet vs Mamba2 在同序列的结果
- **前置知识安排**：delta rule 手算例子（`wiki/delta-rule/`）。
- **完成检查**：手算 α_t=0.5, β_t=1 的一步更新；说明 α_t 与 β_t 改变各影响什么。
- **过渡**：手算验证了公式，下一章看退化与并行训练。

### S4：退化关系与并行训练算法

- **主要教学问题**：Gated DeltaNet 在极端值下退化为哪个模型？并行训练算法解决什么问题？
- **对应范围**：Q3, Q4；F2, C6。
- **正文要点**：
  - 退化（F2）：α_t→1 退化为 DeltaNet；β_t→0 退化为 Mamba2（v_t=0 情形）；α_t→0 清空历史
  - 并行训练（C6）：递归串行无法用 GPU；chunkwise 算法切 chunk，chunk 内 WY 表示展开为矩阵乘法兼容 Tensor Core，chunk 间递归传状态；α_t 只做逐元素乘法不破坏结构
  - 不展开 WY 表示推导（说明存在与作用）
- **讲解材料及职责**：
  - 退化代入（F2）：直接代入公式
  - 并行算法说明（C6）：说明存在与作用，不展开 UT 变换细节
  - 可运行代码：验证 Gated DeltaNet 递归与 DeltaNet、Mamba2 的对比
- **前置知识安排**：Gated DeltaNet 公式（S2）；delta rule 的 chunkwise 算法（`wiki/delta-rule/` 提及存在）。
- **完成检查**：说明 α_t→1 与 β_t→0 各退化为谁；说明并行算法为什么需要 chunkwise。
- **过渡**：机制讲完，下一章看与 KDA 的关系。

### S5：与 KDA 的关系及现实应用

- **主要教学问题**：Gated DeltaNet 与 KDA 在 α_t 设计上有什么关键区别？实验效果如何？
- **对应范围**：Q5；C8, N1, N2, N3。
- **正文要点**：
  - KDA 的三项改动（C8）：α_t 改 channel-wise、decay 改 lower-bounded scaled sigmoid、full-rank output gate
  - α_t 标量 vs channel-wise 的区别：Gated DeltaNet 整张 S 同速率衰减，KDA 按通道差异化衰减
  - 实验数字（N1, N2, N3）：1.3B PPL、S-NIAH 检索、常识推理平均，Gated DeltaNet 优于 DeltaNet 与 Mamba2
  - 现实应用：Qwen3-Next 等用 Gated DeltaNet 作线性层；K3 的 KDA 基于 Gated DeltaNet
- **讲解材料及职责**：
  - 对照表格：Gated DeltaNet vs KDA 的 α_t 设计
  - 实验数字表格（N1, N2, N3）：说明效果，不展开完整 benchmark
- **前置知识安排**：Gated DeltaNet 公式（S2）；KDA（`wiki/kda/index.html`）。
- **完成检查**：说明 Gated DeltaNet 与 KDA 的 α_t 区别；说明实验上 Gated DeltaNet 在哪类任务占优。
- **过渡**：全文总结。

## 3. 讲解顺序

先讲为什么需要它（S1：DeltaNet 与 Mamba2 各自缺什么）→ 再讲是什么（S2：公式）→ 手算验证（S3）→ 退化与并行（S4）→ 现实应用（S5）。一次只引入一个新变量：S1 复述 delta rule（已知）与 Mamba2 公式（新），S2 引入 α_t（新），S3 手算，S4 退化与并行，S5 KDA。

## 4. 贯穿例子

沿用 delta-rule 概念页的 2 维小例子作为贯穿：

- **输入**：$d_k = d_v = 2$，$S_0 = 0$，$k_1 = (1, 0)^\top$（归一化），$v_1 = (1, 0)^\top$，第 2 步 $k_2 = (1, 0)^\top = k_1$，$v_2 = (0, 1)^\top$（同 key 改值）。
- **S3 推进**：在 delta-rule 例子基础上加 α_t。取 β_1 = β_2 = 1（完全覆写），α_2 = 0.5（全局衰减）。展示 α_t 衰减后 delta rule 仍精准覆写 k_2 方向，但其他方向被 α_t 衰减。
- **对照**：同一序列上 DeltaNet（α_t=1）、Mamba2（β_t=0，v_t 写入）、Gated DeltaNet（α_t=0.5, β_t=1）的结果对比。

数字小到全可手算（2 维、2-3 步、α_t 与 β_t 取 0.5 或 1）。

## 5. 讲解材料职责

- **公式（F1, F2, F3, F4）**：表达递归关系与退化关系
- **形状检查**：验证维度自洽
- **边界检查**：α_t→0,1 与 β_t→0,1 的快速代入
- **数字例子（教学示例）**：展示 α_t 与 β_t 协同，可手算
- **对照表格**：Gated DeltaNet vs DeltaNet vs Mamba2 机制与结果对比；Gated DeltaNet vs KDA 的 α_t 设计对比
- **可运行代码**：验证递归公式与对比
- **不安排**：WY 表示完整推导（不展开）、UT 变换（不展开）

## 6. 正文与折叠块分工

### 必须放正文

- Gated DeltaNet 公式与符号定义（F1）
- α_t 与 β_t 的职责分工（C4）
- 退化关系（F2）
- "先衰减后擦写"等价解读（C5）
- 并行算法的存在与作用（C6，不展开细节）
- 与 KDA 的 α_t 区别（C8）
- 手算例子的关键推进与结论
- 前置概念页链接（delta-rule, linear-attention, kda）

### 可放折叠块

- 形状检查的完整代数展开
- 手算例子的完整逐步代入（正文给结论，折叠给全步骤）
- 退化代入的完整代数
- 可运行代码与完整输出
- 实验数字表格的完整版

折叠块全部收起时，正文仍须回答 Q1–Q5。

## 7. 范围与证据约束

大纲只使用 scope.md 已纳入范围的内容。无缺口。实验数字（N1, N2, N3）与公式（F1–F4）均有来源定位。误解与边界已在 scope.md §2.6 列出，安排在正文相应章节处理。
