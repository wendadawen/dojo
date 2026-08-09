# Quantile Balancing 内容范围

## 1. 概念歧义处理

**状态：已裁定。** "Quantile Balancing"（QB）在 K3 技术报告 §2.3.3 中有明确定义：一种从路由分数分位数推导专家 bias 的 MoE 负载均衡方法。该术语在公开文献中没有同名歧义（"分位数平衡"在统计学中指另一种重加权方法，但 QB 作为 MoE 路由术语是 K3 报告引入的专有名词）。本文采用 K3 报告的定义。

## 2.1 概念含义

- **概念名称**：Quantile Balancing（分位数平衡）
- **英文名称**：Quantile Balancing
- **常见缩写**：QB
- **一句话定义**：QB 是一种 MoE 负载均衡方法，从一次前向传播的路由分数分位数直接推导出每个专家的 bias，替代 DeepSeek-V3 的固定步长 sign 更新。
- **正式定义**（K3 报告 §2.3.3 Eq.14）：给定 m 个 token、n 个专家、Top-k 选择，目标负载 q = mk/n。从 Top-(k+1) 路由得到每个 token 的 cutoff α_i，计算 margins = s_{i,j} - α_i，将每个专家的 bias 设为其 margins 的 (1-k/n) 分位数的负值，再减去均值。
- **本文语境**：MoE（混合专家模型）训练中的负载均衡，特指 auxiliary-loss-free 路由框架下的 bias 更新方法。

### 包括什么

- QB 的 bias 更新公式（Eq.14）及其符号含义
- 从 Top-(k+1) 路由得到 cutoff α_i 的机制
- margin 的定义和计算
- 分位数与目标负载的关系 q = mk/n
- mean-centering 的作用
- 与 DeepSeek-V3 固定步长 sign 更新的对比
- 从平衡分配问题推导 QB（Appendix C 的对偶视角）
- 直方图估计（Appendix D 的工程实现）
- m=8, n=4, k=1 的手算例子（Fig.5）

### 不包括什么

- **MoE 架构的基本原理**（router/FFN/专家并行）：属于前置概念，不内联讲解
- **K3 的完整架构**（LatentMoE、SiTU-GLU 等）：与 QB 无直接关系
- **辅助损失路由**（auxiliary-loss-based routing）：QB 的对比对象，只引用其存在和问题，不展开
- **expert parallelism 的通信细节**（all-to-all dispatch）：QB 的动机之一但不展开
- **训练超参数选择**（学习率、batch size）：不影响 QB 机制理解

### 相邻概念

- **auxiliary-loss-free routing**：QB 的框架，bias 只影响选谁不进权重。前置概念，未生成则占位。
- **DeepSeek-V3 bias 更新**：QB 的直接前身和对比对象，在正文中作为对比引入。
- **Expert Threshold Routing**：维护 EMA 阈值、允许变长专家数，K3 报告 §C 末尾说明与 QB 的区别。扩展内容，不纳入正文。

## 2.2 学习目标

### Q1：QB 要解决什么问题？DeepSeek-V3 的固定步长 bias 更新为什么在 896 专家时失效？

- **完成答案**：读者应能说明 DeepSeek-V3 用 b ← b + γ·sign(ℓ̄ − ℓ_j) 固定步长更新 bias，γ 在慢适应和负载震荡间权衡；专家数增大到 896 时，sign 更新只保留负载误差的方向不保留幅度，收敛慢且震荡，导致负载不均衡、专家并行训练变慢、部分专家训练不足。
- **为什么是核心目标**：不理解 QB 解决的问题，就无法理解 QB 的设计动机和优势。
- **依赖内容**：auxiliary-loss-free 路由的基本框架（bias 加到 router 分数上做 Top-k 选择，不进权重）、DeepSeek-V3 的 sign 更新公式。

### Q2：QB 如何从一次前向传播推导出下一个 bias？公式 Eq.14 的每一步在做什么？

- **完成答案**：读者应能按顺序说明：(1) Top-(k+1) 路由得到 cutoff α_i；(2) 计算 margins = s_{i,j} - α_i；(3) 每个专家的 b̃_j = -quantile_{1-k/n}(margins_{:,j})；(4) b = b̃ - mean(b̃)。能解释为什么用 (1-k/n) 分位数（恰好使 q 个 margin 超过阈值）和 mean-centering 的作用（去掉公共偏移不改变 Top-k 选择）。
- **为什么是核心目标**：这是 QB 的核心机制，不理解公式就无法理解 QB 如何工作。
- **依赖内容**：Top-(k+1) 路由、cutoff、margin、分位数、目标负载 q = mk/n。

### Q3：如何手算 m=8, n=4, k=1 的 QB 例子？

- **完成答案**：读者应能用给定的 8×4 分数矩阵，手动完成：初始路由得到 loads (4,3,1,0)；计算 cutoff α_i 和 margins；找到每个专家的 3rd largest margin（即 (1-1/4)=0.75 分位数）；计算 b̃ 和 mean-centered b；验证新路由得到 (2,2,2,2)。
- **为什么是核心目标**：手算例子是验证理解的最直接方式，也是 concept 页的质量底线。
- **依赖内容**：Q2 的完整公式和符号。

### Q4：QB 与 DeepSeek-V3 sign 更新有什么本质区别？为什么 QB 不需要学习率类超参数？

- **完成答案**：读者应能说明 sign 更新是对同一对偶目标的 (sub)gradient 的 SignSGD 步（只保留方向），而 QB 直接跳到该对偶目标的精确 coordinate minimizer。这解释了为什么 QB 无需 γ 类超参数且在近 10³ 专家时仍能几步内收敛。
- **为什么是核心目标**：理解 QB 的理论优势来源，而非仅记住公式。
- **依赖内容**：平衡分配的对偶问题（Appendix C）、coordinate minimization。

### Q5：QB 在训练时如何用直方图估计分位数？推理时如何处理？

- **完成答案**：读者应能说明：训练时对每个专家维护 margins 的分箱直方图，各 rank 在前向时 scatter-add 本地 bin counts，step 结束时一次 all-reduce 汇总全局 counts，从 pooled counts 恢复分位数（误差 ≤ bin width）。推理时冻结 bias，不做任何分位数计算。
- **为什么是核心目标**：QB 的工程可行性依赖于直方图估计，不理解则无法判断 QB 是否实用。
- **依赖内容**：分位数、histogram、all-reduce。

## 2.3 内容分级

### 核心内容

| 内容 | 对应目标 | 必须讲清的结论 |
|------|---------|--------------|
| auxiliary-loss-free 路由框架：s=σ(W_r x)，Top-k 选 (s+b)，权重 p 不含 b | Q1, Q2 | bias 只影响选谁，不进权重和梯度 |
| DeepSeek-V3 sign 更新公式 b ← b + γ·sign(ℓ̄−ℓ_j) | Q1 | γ 权衡慢适应与震荡 |
| 896 专家时 sign 更新失效 | Q1 | sign 只保留方向不保留幅度 |
| QB 更新公式 Eq.14 全部步骤 | Q2, Q3 | cutoff→margin→quantile→centering |
| Top-(k+1) 得 cutoff α_i | Q2 | 为什么用 Top-(k+1) 而非单独的 token 侧分位数 |
| margins = s_{i,j} - α_i | Q2, Q3 | 旧 bias 通过 cutoff 进入，raw score 减 cutoff |
| b̃_j = -quantile_{1-k/n}(margins_{:,j}) | Q2, Q3 | (1-k/n) 分位数恰好使 q 个 margin 超过阈值 |
| mean-centering | Q2 | 去掉公共偏移不改变 Top-k |
| m=8,n=4,k=1 手算例子 | Q3 | 从 (4,3,1,0) 到 (2,2,2,2) 的完整计算 |
| sign 更新是 SignSGD，QB 是 coordinate minimizer | Q4 | 同一对偶目标，不同求解策略 |
| 直方图估计机制 | Q5 | scatter-add + all-reduce + bin 内插值 |
| 推理冻结 bias | Q5, Q2 | 训练-推理一致性 |

### 辅助内容

| 内容 | 服务的核心/误解 | 说明 |
|------|--------------|------|
| 平衡分配的对偶推导（Appendix C Eq.20-23） | Q4 | 解释 QB 公式的来源 |
| alternating solver（Algorithm 1） | Q4 | QB 是一轮交替更新 |
| 直方图分箱范围 [b_min-1, b_max+1] | Q5 | 为什么范围有界 |
| EMA 平滑 | Q5 | 减少批次间噪声 |
| 因果性：batch 不用自己推导的 bias | Q2 | 更新在下一步生效 |

### 扩展内容

| 内容 | 纳入/排除 | 理由 |
|------|---------|------|
| BIP（Binary-Integer-Programming）对比 | 排除 | 不影响理解 QB 机制 |
| Expert Threshold Routing 对比 | 排除 | 属于另一独立概念 |
| BASE Layers | 排除 | 历史背景，不影响机制理解 |
| K3 的 MoonEP 专家放置 | 排除 | 属于服务系统，与 QB 无关 |

## 2.4 前置知识映射

| 前置知识 | 被哪些目标依赖 | 概念页状态 | 递归深度 |
|---------|-------------|----------|---------|
| auxiliary-loss-free routing | Q1, Q2 | **未生成，占位** | 0（本文直接依赖） |
| MoE 基本路由（router/top-k/专家） | Q1, Q2 | 已有：wiki/moe-serving/index.html | 0 |
| 分位数（quantile） | Q2, Q3 | 无概念页，正文内联最小定义 | — |

auxiliary-loss-free routing 未生成概念页，正文使用占位提示并保留阅读所需最小衔接。MoE 基本路由引用已有页面 wiki/moe-serving/index.html。分位数在正文中给出最小定义（一组数中指定比例以下的阈值）。

## 2.5 明确不展开的内容

- **K3 的 LatentMoE 架构**：QB 是通用的 MoE 负载均衡方法，不依赖 LatentMoE 特性。不展开。
- **辅助损失路由（auxiliary-loss-based）的完整机制**：只在对比中提及其存在和缺点（干扰主目标梯度），不展开公式。属于另一独立概念。
- **expert parallelism 的通信成本分析**：QB 的动机之一（不均衡拖慢 EP 训练），但不展开 EP 通信细节。属于 wiki/moe-serving。
- **K3 的训练超参数配置**：不影响理解 QB 机制。

## 2.6 常见误解和适用边界

### 误解 1

- **错误理解**：QB 是一种新的损失函数或正则化项。
- **正确结论**：QB 是 bias 的更新规则，不引入任何损失项。它替代的是 DeepSeek-V3 中 bias 的更新方式（从 sign 固定步长变为分位数），而不是替代损失函数。
- **形成原因**："负载均衡"常与"均衡损失"关联。
- **影响目标**：Q1, Q2。

### 误解 2

- **错误理解**：QB 一步就能完美均衡任何负载分布。
- **正确结论**：QB 是交替求解器的一轮更新，论文报告在近 10³ 专家时"几步内"收敛。一步更新在给定旧 cutoff 的条件下保证目标负载，但实际路由使用新 cutoff，可能需要多轮。
- **形成原因**：手算例子恰好一步收敛，容易过度推广。
- **影响目标**：Q3, Q4。

### 误解 3

- **错误理解**：bias 参与了 mixture weight（门控值）的计算。
- **正确结论**：bias 只加到 router 分数上用于 Top-k 选择，归一化权重 p 只用原始分数 s 计算（Eq.13）。这是 auxiliary-loss-free 的核心设计。
- **形成原因**：与传统 auxiliary-loss 路由混淆。
- **影响目标**：Q1, Q2。

### 适用边界

- **QB 解决的问题**：auxiliary-loss-free 路由框架下的 bias 更新方式，使 bias 在大规模专家数下仍能快速收敛到均衡负载。
- **QB 不解决的问题**：不替代辅助损失路由的 sequence-wise 均衡（DeepSeek-V3 仍保留 complementary sequence-wise loss）；不解决专家容量溢出（expert capacity）问题；不改变路由的稀疏性（Top-k 固定）。
- **成立条件**：m·k/n 为整数（目标负载 q 是整数）；路由分数无精确并列（tie 概率为零）。
- **条件不满足时**：m·k/n 非整数时取 ⌈q⌉ 做直方图目标 rank；有 ties 时按约定处理（实践中几乎不出现）。
