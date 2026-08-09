# 辅助损失无关路由：内容范围

## 1.1 概念含义

- 概念名称：辅助损失无关路由
- 英文名称：Auxiliary-Loss-Free Routing（亦作 Auxiliary-Loss-Free Load Balancing、Loss-Free Balancing）
- 一句话定义：MoE 训练时给每个专家一个可学习偏置 $b_i$，把它加到路由分数上只参与"选哪些专家"的 top-k 决策、不进入门控权重，并按"过载减小、欠载增大"的固定步长 sign 规则在每步训练后更新 $b_i$，从而不引入任何辅助损失项也能保持专家负载均衡。
- 正式定义：与原始论文（arXiv:2408.15664）和 DeepSeek-V3 技术报告（arXiv:2412.19437 §2.2，Eq.16）一致：
  - 路由分数 $s_{i,t}=\sigma(u_t^\top e_i)$（DeepSeek-V3 用 sigmoid；原始论文一般写作 $G(u_t^\top e_i)$）
  - 带 bias 的选择：$g'_{i,t}=s_{i,t}$ 若 $s_{i,t}+b_i\in\mathrm{Topk}(\{s_{j,t}+b_j\mid 1\leq j\leq N_r\},K_r)$，否则为 $0$
  - 归一化门控权重：$g_{i,t}=g'_{i,t}/\sum_j g'_{j,t}$（注意分母不含 $b$）
  - 每步结束按负载偏差 $e_i=\bar c_i-c_i$ 更新 $b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(e_i)$，其中 $c_i$ 是该步专家 $i$ 收到的 token 数，$\bar c_i$ 是平均负载
- 本文采用的语境：以 DeepSeek-V3 上线的版本为主，必要时与原始 2024 年论文对照。`quantile-balancing/` 是 K3 对其的改进，作为相邻概念引用。

### 包括什么

- "bias 只影响选谁、不影响权重多少"的设计及其公式表现（$g'_{i,t}$、$g_{i,t}$ 的定义）
- 固定步长 sign 更新规则（含负载 $c_i$ 的定义、$\bar c_i$ 的定义、$\gamma$ 的角色）
- 为什么不引入辅助损失梯度（bias 不进 $g_{i,t}$ 的分母、不进反向传播）
- 与辅助损失方法的对照（辅助损失为何产生干扰梯度、本方法为何不产生）
- 一个可手算的 4 专家 top-2 例子，覆盖路由选择、负载统计、bias 更新、下一轮选择变化
- 适用边界：$\gamma$ 是单一全局超参；只在训练时更新 bias；推理时 bias 固定；只保证"批级"专家负载均衡，不保证序列内均衡（V3 配合 sequence-wise balance loss 解决）

### 不包括什么

- DeepSeek-V3 整体架构（MLA、共享专家、训练并行）—— 属 `moe-serving/` 与其它页面
- DeepSeekMoE 的细粒度专家划分 —— 属独立架构概念
- sequence-wise balance loss 的完整推导 —— 属相邻概念，本文只说明它是互补机制，不展开
- K3 Quantile Balancing 的完整推导 —— 属 `quantile-balancing/`，本文只引用
- 推理阶段的负载均衡（部署时复制热门专家等工程实践）—— 属 `moe-serving/`
- bias 的乘法变体、$b_i\leftarrow b_i+\gamma\cdot e_i$ 的连续变体 —— 论文实验过但不采用，本文只在边界处简要提及作为对照

### 相邻概念

- 辅助损失负载均衡（auxiliary-loss load balancing）：在损失函数里加 $L_{bal}=\alpha\sum f_i P_i$ 一类项推动均衡。区别：把均衡写进梯度目标，会产生干扰梯度。本文不展开它，只在"为什么需要 aux-loss-free"处对照。
- sequence-wise balance loss（DeepSeek-V3）：在 aux-loss-free 之外另加的、专门保证单序列内 token 间均衡的小权重损失项。区别：aux-loss-free 解决批级专家负载，sequence-wise 解决序列内 token 维度，二者互补。
- Quantile Balancing（K3）：把 bias 更新从"固定步长 sign"换成"从路由分数分位数一步算出"。区别：仍属 aux-loss-free 路由家族（无辅助损失），只改 bias 的更新规则。本文引用 `quantile-balancing/`。

## 1.2 学习目标

### Q1：为什么 MoE 训练需要负载均衡，传统辅助损失方案有什么缺陷？

- 完成答案：读者应能说明 MoE 路由器可能"负载坍塌"（少数专家包揽多数 token），导致其他专家训不动、专家并行训练出现忙闲不均；传统辅助损失 $L_{bal}=\alpha\sum f_i P_i$ 把均衡写进梯度目标，能压住坍塌但 $\alpha$ 难调——过大压制专家特化、损害主任务性能，过小压不住坍塌；辅助损失的梯度是混进主目标梯度的"干扰梯度"。
- 为什么是核心目标：理解不了"为什么需要它"，就理解不了 bias 这种"绕过损失目标"的设计为什么必须存在。
- 依赖内容：MoE、top-k、router、expert、辅助损失、负载坍塌、专家并行。

### Q2：bias 在前向传播里做什么、不做什么？

- 完成答案：读者应能从公式说明 $s_{i,t}+b_i$ 只用于判断 expert $i$ 是否进入 top-k；进入则 $g'_{i,t}=s_{i,t}$（原始分数，不含 bias），不进入则为 0；归一化门控权重 $g_{i,t}=g'_{i,t}/\sum_j g'_{j,t}$，分母也不含 bias；因此 bias 既不改变 mixture weight 也不出现在反向传播的梯度里。
- 为什么是核心目标：这是"aux-loss-free"名称的来源，是概念的核心定义。
- 依赖内容：路由分数、top-k 选择、门控权重、归一化、反向传播。

### Q3：bias 在训练时怎么被更新，为什么用 sign 而不是直接用误差幅度？

- 完成答案：读者应能写出 $b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(\bar c_i-c_i)$，说明 $c_i$ 是本步专家 $i$ 收到的 token 数、$\bar c_i$ 是平均、$\gamma$ 是固定步长超参；解释 sign 让 bias 走固定小步而非按误差幅度跳变，避免单步剧烈震荡；说明这一更新在反向传播之外发生（非梯度），bias 不是模型参数。
- 为什么是核心目标：bias 的更新规则是本方法的实际工作机制，不写清机制就无法落地。
- 依赖内容：负载统计、sign 函数、固定步长更新、梯度更新与规则更新的区别。

### Q4：完整跑一遍 4 专家 top-2 的训练步，bias 如何影响下一步路由？

- 完成答案：给定 4 个专家对某 token 的打分与当前 bias，读者应能手算哪两个被选中；统计一个 batch（若干 token）后每个专家的 $c_i$ 与 $\bar c_i$；按 sign 规则更新 bias；在新 bias 下重新做一次路由选择，看到冷门专家被拉回。
- 为什么是核心目标：把前三个目标整合成一个可复算的端到端例子，是"真懂了"的检验。
- 依赖内容：Q2 的选择机制、Q3 的更新规则。

### Q5：本方法在 DeepSeek-V3 中的实际配置和边界是什么？

- 完成答案：DeepSeek-V3 有 256 个 routed expert、top-8（来自 `moe-serving/` 已有页）；$\gamma=0.001$ 训练前 14.3T token，最后 500B token 设 $\gamma=0$ 冻结 bias；另外配 $\alpha=0.0001$ 的 sequence-wise balance loss 处理序列内均衡；推理阶段 bias 固定不再更新。读者应能说明：本方法只解决批级负载均衡，不解决序列内均衡，也不能保证单步完美均衡——只能在训练中逐步逼近。
- 为什么是核心目标：把概念落回真实工程配置，并明确边界，避免读者把它误当成"银弹"。
- 依赖内容：DeepSeek-V3 配置数字、$\gamma$ 调度、sequence-wise loss 的角色。

## 1.3 内容分级

### 核心内容（任一缺失则某学习目标无法完整回答）

- MoE 负载坍塌与辅助损失方案 → 服务 Q1
- bias 加到分数上做选择、不进 mixture weight 的公式 → 服务 Q2
- 固定步长 sign 更新规则与负载统计 → 服务 Q3
- 4 专家 top-2 的贯穿手算例子 → 服务 Q4
- DeepSeek-V3 的 $\gamma$ 调度与 sequence-wise loss 边界 → 服务 Q5

### 辅助内容

- "干扰梯度"为什么是干扰（辅助损失梯度与主目标梯度的混合）
- bias 乘法变体、$b_i\leftarrow b_i+\gamma\cdot e_i$ 连续变体为什么不采用（论文实验数据）
- bias 初始化为 0、每步用上一批负载统计（因果约束）

### 扩展内容

- 2026 年的理论分析（arXiv:2512.03915 把 ALF-LB 形式化为 primal-dual，对数期望 regret）—— 排除本页范围，属理论扩展
- 与 Expert Choice 的对比（ Expert Choice 有 future token leakage 问题）—— 排除本页范围，属路由策略族对比

## 1.4 前置知识映射

- MoE、专家、router、top-k、shared/routed expert、专家并行与负载不均 → `wiki/moe-serving/index.html`（已有概念页，正文首次出现这些术语时引用；不再内联展开）
- 反向传播、梯度下降 → 假定读者基础，不引用专门页
- 损失函数、辅助损失、Sigmoid、Topk 选择算子 → 假定读者基础，必要时一句话提示
- Quantile Balancing → `wiki/quantile-balancing/index.html`（已有概念页，作为相邻改进引用）

## 1.5 不展开的内容

- DeepSeek-V3 整体架构与训练并行：与概念无直接依赖，是独立架构页 `moe-serving/` 的范围。
- sequence-wise balance loss 的完整推导：与 aux-loss-free 是互补但独立的两件事，本页只点明它解决什么。
- K3 Quantile Balancing 的推导：属 `quantile-balancing/`，本页只引用对比。
- 推理阶段的负载均衡工程：属 `moe-serving/`。

## 1.6 常见误解与适用边界

### 误解 1：bias 是模型参数，参与反向传播

- 错误理解：bias $b_i$ 像 $W_r$ 一样是模型参数，被梯度更新。
- 正确结论：bias 不是模型参数，不进计算图，不被反向传播更新；它在每步训练后由规则式 sign 更新，更新依据是上一批的负载统计。
- 形成原因：bias 形似可学习参数，且加到分数上像可学习偏置。
- 影响目标：Q2、Q3。

### 误解 2：bias 改变了 mixture weight

- 错误理解：$b_i$ 加到 $s_{i,t}$ 上，所以 $g_{i,t}=s_{i,t}+b_i$ 进入加权。
- 正确结论：bias 只用于判断是否进入 top-k；进入后 $g'_{i,t}=s_{i,t}$（原始分数），归一化分母 $\sum_j g'_{j,t}$ 也不含 $b$。
- 形成原因：公式写作 $s_{i,t}+b_i\in\mathrm{Topk}$，容易误读为"$g$ 用 $s+b$"。
- 影响目标：Q2。

### 误解 3：本方法能保证单步完美均衡

- 错误理解：每步训练后所有专家负载严格相等。
- 正确结论：sign 更新只走固定小步 $\gamma$；负载偏多偏少要经过多步逐步逼近；某步负载不均是正常的。DeepSeek-V3 用 $\gamma=0.001$ 是慢速调度，14.3T token 才稳定。
- 影响目标：Q3、Q5。

### 误解 4：本方法同时解决序列内均衡

- 错误理解：有了 aux-loss-free，序列内 token 间也均衡。
- 正确结论：aux-loss-free 只看批级 $c_i$；序列内某 expert 是否被同序列所有 token 抢占它不管。DeepSeek-V3 另加 $\alpha=0.0001$ 的 sequence-wise balance loss 互补处理。
- 影响目标：Q5。

### 适用边界

- 解决：批级专家负载均衡（防 routing collapse、保证专家并行训练效率）。
- 不解决：序列内 token 间均衡、单步完美均衡、推理阶段路由。
- 条件：训练阶段、批规模足够大使 $c_i$ 统计有意义；$\gamma$ 取得既不过大（震荡）也不过小（收敛慢）。
- 不满足：序列内极端不均衡仍需 sequence-wise loss；$\gamma$ 过大会震荡；专家数极多（如 K3 的 896）时 sign 丢失幅度信息，收敛过慢——这是 `quantile-balancing/` 解决的问题。
