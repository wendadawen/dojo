# Quantile Balancing 核心论断与证据

## C 论断（事实性结论）

### C1
- **论断**：K3 采用 auxiliary-loss-free 路由，router 打分 s_i = Sigmoid(W_r x_i)，Top-k 选择时加 bias b，归一化权重 p 只用 s 不含 b。
- **来源**：K3 报告 §2.3.3 Eq.13；config.json: `moe_router_activation_func: "sigmoid"`, `topk_method: "noaux_tc"`, `moe_renormalize: true`。
- **适用条件**：K3 的 LatentMoE 路由层。
- **置信状态**：已确认。

### C2
- **论断**：DeepSeek-V3 的 bias 更新为 b_j^(t+1) = b_j^(t) + γ·sign(ℓ̄ − ℓ_j)，其中 ℓ̄ 是专家平均负载、ℓ_j 是专家 j 的负载、γ 是固定步长超参数。
- **来源**：K3 报告 §2.3.3（"The original method updates b with the fixed-step rule b_j^(t+1) = b_j^(t) + γ·sign(ℓ̄^(t) − ℓ_j^(t)) [30]"）；DeepSeek-V3 Technical Report (arXiv:2412.19437) §3.3 auxiliary-loss-free 策略。WebSearch 确认原文：overload 则 b_j ← b_j − γ，underload 则 b_j ← b_j + γ，等价于 b_j + γ·sign(ℓ̄ − ℓ_j)。
- **适用条件**：auxiliary-loss-free 路由框架。
- **置信状态**：已确认。

### C3
- **论断**：γ 在慢适应和负载震荡间权衡；896 专家时维持均衡负载变得更具挑战性，不均衡路由拖慢专家并行训练并可能导致部分专家训练不足。
- **来源**：K3 报告 §2.3.3（"γ trades off slow adaptation against load oscillation. Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer. Imbalanced routing slows expert-parallel training and may leave some experts poorly trained [47]."）。
- **适用条件**：大规模专家数。
- **置信状态**：已确认。

### C4
- **论断**：QB 从一次前向传播推导下一个 bias。路由用 Top-(k+1) 选择（而非 Top-k），前 k 个是实际路由，第 (k+1) 个是 cutoff α_i。从 Top-(k+1) 取 cutoff 避免了单独的 token 侧分位数计算。
- **来源**：K3 报告 §2.3.3（"Routing replaces the Top-k selection with Top-(k+1) on the biased score... the (k+1)-th entry is the cutoff α_i that an expert must exceed to enter token i's Top-k. Taking the cutoff from Top-(k+1) routing avoids a separate token-side quantile."）。
- **适用条件**：Top-k 路由框架。
- **置信状态**：已确认。

### C5
- **论断**：margins 用 raw score 减 cutoff：margin_{i,j} = s_{i,j} − α_i。旧 bias 只通过 cutoff 进入更新，raw score 不含旧 bias。
- **来源**：K3 报告 §2.3.3（"The margins subtract the biased cutoff α_i from the raw score s_{i,j}, so the old bias enters the update only through the cutoffs"）。
- **适用条件**：QB 更新公式。
- **置信状态**：已确认。

### C6
- **论断**：设 −b̃_j 为 (q+1)-th largest margin（q = mk/n），则恰好 q 个 margin 超过阈值。由于 q/m = k/n，这是 margins 的 (1 − k/n) 分位数。QB 更新：b̃_j ← −quantile_{1−k/n}(s_{:,j} − α)，b ← b̃ − mean(b̃)。
- **来源**：K3 报告 §2.3.3 Eq.14。
- **适用条件**：无 ties（实践中几乎满足）；q 为整数。
- **置信状态**：已确认。

### C7
- **论断**：mean-centering 去掉一个公共偏移，不改变 Top-k 选择（因为所有 expert 同时加减常数不改变相对排序）。
- **来源**：K3 报告 §2.3.3 Eq.14 第二行（"the second line removes a common offset that leaves Top-k selection unchanged"）。
- **适用条件**：Top-k 选择只取决于相对排序。
- **置信状态**：已确认。

### C8
- **论断**：因因果性，更新只在下一步生效，一个 batch 绝不用自己推导的 bias 路由。
- **来源**：K3 报告 §2.3.3（"For causality, the update takes effect only in the next step [30], i.e., a batch is never routed with a bias derived from itself."）。
- **适用条件**：训练阶段。
- **置信状态**：已确认。

### C9
- **论断**：推理时 bias 冻结，不做任何分位数计算。
- **来源**：K3 报告 §2.3.3（"The final bias is frozen at inference."）；Appendix C（"at deployment, routing is a fixed Top-k selection with a frozen bias, and no quantile computation is needed"）。
- **适用条件**：推理阶段。
- **置信状态**：已确认。

### C10
- **论断**：Fig.5 展示 m=8, n=4, k=1 的 QB 例子，初始负载 (4,3,1,0)，QB 后变为 (2,2,2,2)，每个专家目标负载 q = 8·1/4 = 2。
- **来源**：K3 报告 Fig.5 及其 caption。
- **适用条件**：教学示例。
- **置信状态**：已确认。

### C11
- **论断**：sign 更新是对偶目标的 (sub)gradient 的 SignSGD 步（只保留方向），QB 直接跳到该对偶目标的精确 coordinate minimizer。这解释了 QB 无需学习率类超参数且在近 10³ 专家时仍能几步内收敛。
- **来源**：K3 报告 Appendix C Eq.27（"A SignSGD step on this objective recovers the fixed-step sign update of auxiliary-loss-free balancing [30]... the sign update retains only the direction of the load error in Eq.27, whereas QB jumps directly to the exact coordinate minimizer of the same dual objective."）。
- **适用条件**：平衡分配的对偶视角。
- **置信状态**：已确认。

### C12
- **论断**：QB 的平衡分配推导来自最优分配问题 Eq.20：最大化总分数，约束每个 token 选 k 个专家、每个专家服务 mk/n 个 token。线性松弛后由二部图 b-matching 多面体的整性保证最优解为整数。
- **来源**：K3 报告 Appendix C Eq.20-23。
- **适用条件**：mk/n 为整数。
- **置信状态**：已确认。

### C13
- **论断**：交替求解器（Algorithm 1）交替更新 α（给定 β）和 β（给定 α），每次更新都是 (1-k/n) 分位数。QB 是交替求解器的一轮更新。
- **来源**：K3 报告 Appendix C Algorithm 1, Eq.25-26。
- **适用条件**：—
- **置信状态**：已确认。

### C14
- **论断**：训练时对每个专家维护 margins 的分箱直方图。各 rank 在前向时 scatter-add 本地 bin counts 到 per-expert count matrix H ∈ N^{n×B}，无通信；step 结束时一次 all-reduce 汇总全局 counts；从 pooled counts 恢复分位数。通信量是 nB 个整数 per layer per step，与 m 无关。
- **来源**：K3 报告 §2.3.3 histogram 段及 Appendix D。
- **适用条件**：分布式训练。
- **置信状态**：已确认。

### C15
- **论断**：直方图分箱范围为 [b_min−1, b_max+1]（b_min/b_max 是当前 bias 极值），分 B 个均匀 bin，每步重算范围。分位数误差 ≤ bin width w = (b_max−b_min+2)/B。B=1000 时误差最多几个 10⁻³，观测不到残余负载不均衡。
- **来源**：K3 报告 Appendix D（"Binning range", "Properties" 段）。
- **适用条件**：B 足够大。
- **置信状态**：已确认。

### C16
- **论断**：因为 counts 可加，全局直方图精确不变于 token 如何分布在各 rank 或 accumulation step 上；估计的是 pooled global batch 的分位数而非各 rank 分位数的平均。
- **来源**：K3 报告 Appendix D Properties 第三点。
- **适用条件**：—
- **置信状态**：已确认。

### C17
- **论断**：K3 config 确认：num_experts=896，num_experts_per_token=16（即 k=16），moe_router_activation_func="sigmoid"，topk_method="noaux_tc"（auxiliary-loss-free）。
- **来源**：HuggingFace moonshotai/Kimi-K3 config.json。
- **适用条件**：K3 模型。
- **置信状态**：已确认。

## F 公式

### F1 — Eq.13 路由
$$T_i = \text{argtopk}(s_i + b), \quad p_{i,j} = \frac{s_{i,j}}{\sum_{r \in T_i} s_{i,r}}, \quad j \in T_i$$
- **来源**：K3 报告 §2.3.3 Eq.13。
- **含义**：T_i 是 token i 选中的 k 个专家集合；p 是归一化权重，不含 b。

### F2 — DeepSeek-V3 sign 更新
$$b_j^{(t+1)} = b_j^{(t)} + \gamma \cdot \text{sign}(\bar{\ell}^{(t)} - \ell_j^{(t)})$$
- **来源**：K3 报告 §2.3.3 引用 [30]；DeepSeek-V3 Technical Report。
- **含义**：ℓ̄ 是专家平均负载，ℓ_j 是专家 j 负载，γ 是固定步长。

### F3 — QB 更新 Eq.14
$$\tilde{b}_j^{(t+1)} \leftarrow -\text{quantile}_{1-k/n}\left(s_{:,j}^{(t)} - \alpha^{(t)}\right)$$
$$b^{(t+1)} \leftarrow \tilde{b}^{(t+1)} - \text{mean}\left(\tilde{b}^{(t+1)}\right) \cdot \mathbf{1}$$
- **来源**：K3 报告 §2.3.3 Eq.14。
- **含义**：margins = s_{:,j} − α；α 来自 Top-(k+1) 的 cutoff。

### F4 — 对偶目标 Eq.23
$$\min_{\alpha, \beta} L(\alpha, \beta) := \sum_{i,j} \max\left(0, s_{i,j} - \alpha_i - \beta_j\right) + k \sum_i \alpha_i + \frac{mk}{n} \sum_j \beta_j$$
- **来源**：K3 报告 Appendix C Eq.23。
- **含义**：平衡分配的线性松弛的对偶目标；β_j = −b_j。

### F5 — coordinate minimizer
$$\beta_j^* = \text{quantile}_{1-k/n}\left(s_{:,j} - \alpha\right)$$
$$\alpha_i^* = \text{quantile}_{1-k/n}\left(s_i - \beta\right)$$
- **来源**：K3 报告 Appendix C Eq.25-26。
- **含义**：给定对方固定，各自的最优是 (1-k/n) 分位数。

### F6 — sign (sub)gradient
$$\frac{\partial L}{\partial \beta_j} = \frac{mk}{n} - \sum_{i=1}^{m} \mathbb{1}\left[s_{i,j} - \alpha_i - \beta_j > 0\right]$$
- **来源**：K3 报告 Appendix C Eq.27。
- **含义**：目标负载减实际负载；SignSGD 只保留方向。

### F7 — 直方图分位数恢复
$$\tilde{b}_j = b_{\min} - 1 + \left(\beta_j + \text{clip}\left(\frac{q - c_j}{h_j}, 0, 1\right)\right) w$$
- **来源**：K3 报告 Appendix D。
- **含义**：β_j 是选中 bin 的索引，c_j 是其前累计 count，h_j 是 bin 内 count，w 是 bin width。

## N 数字

### N1
- **数字**：K3 的 routed expert 数为 896 per layer，每个 token 选 16 个专家（k=16）。
- **来源**：config.json: `num_experts: 896`, `num_experts_per_token: 16`。
- **条件**：K3 LatentMoE 层。

### N2
- **数字**：直方图 bin 数 B = 1000 时，分位数误差最多几个 10⁻³，观测不到残余负载不均衡。
- **来源**：K3 报告 Appendix D Properties。
- **条件**：sigmoid router scores ∈ (0,1)。

### N3
- **数字**：直方图通信成本低于 1% of exchanging raw margins over a process group every micro-batch。
- **来源**：K3 报告 Appendix D。
- **条件**：K3 配置。
