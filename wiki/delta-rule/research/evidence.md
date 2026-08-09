# Delta 规则与 DeltaNet · 核心论断与证据

来源选择优先级：原始论文 ICML/NeurIPS/ICLR > 权威教材（暂无）> 官方文档（flash-linear-attention 库）> 官方源码（commit 固定版本）。

## C 论断（概念/机制）

### C1

- **论断内容**：线性注意力的递归更新为 $S_t = S_{t-1} + v_t k_t^\top$，是纯加性累加；输出 $o_t = S_t q_t$。
- **来源定位**：Yang et al. NeurIPS 2024, arXiv:2406.06484 §2.1（"vanilla linear attention"）与 Table 4 第一行（Linear Attention [40]）；Katharopoulos et al. 2020 "Transformers are RNNs"（Yang 引用为 [40]）。
- **适用条件**：标准因果线性注意力，无门控、无归一化。
- **置信状态**：已确认。

### C2

- **论断内容**：当序列长度 $L > d$（key 维度）时，d 维空间无法容纳 $L$ 个正交 key；retrieval $S k_j = v_j + \sum_{i \neq j} (k_i^\top k_j) v_i$ 中的 $\sum_{i \neq j} (k_i^\top k_j) v_i$ 即"retrieval error"（key 碰撞项）。这是线性注意力在长序列上检索能力下降的根本原因。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §2.1（"a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key 'collisions' when $L > d$, as pointed out by Schlag et al. [86]"）；Schlag et al. 2021 ICML arXiv:2102.11174 §4.1（overcapacity regime）；Yang et al. ICLR 2025 arXiv:2412.06464 §1（"the number of orthogonal key-value pairs they can store is bounded by the model's dimensionality. When the sequence length exceeds this dimension, 'memory collisions' become inevitable"）。
- **适用条件**：线性注意力 / 任何使用纯加性累加 key-value 外积的递归记忆。
- **置信状态**：已确认。

### C3

- **论断内容**：Delta 规则（DeltaNet 形式）的紧凑公式为 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，其中 $\beta_t = \sigma(W_\beta x_t) \in (0,1)$ 是写入强度（学习率），$S \in \mathbb{R}^{d_v \times d_k}$ 是矩阵状态，$q_t, k_t, v_t$ 由 $x_t$ 经 $W_Q, W_K, W_V$ 线性投影得到，输出 $o_t = S_t q_t$。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §2.2（"DeltaNet: Linear Transformers with the Delta Update Rule"）与 §3.1 展开式；Table 4（DeltaNet [86] 行）；Yang et al. ICLR 2025 arXiv:2412.06464 §2.3（作为对比基准）。
- **适用条件**：DeltaNet 与 Gated DeltaNet（当 $\alpha_t = 1$ 时）。
- **置信状态**：已确认。

### C4

- **论断内容**：Delta 规则可改写为"先擦除后写入"等价形式 $S_t = S_{t-1} - v_t^{\text{old}} k_t^\top + v_t^{\text{new}} k_t^\top$，其中 $v_t^{\text{old}} = S_{t-1} k_t$ 是当前记忆对 $k_t$ 的响应（检索出的旧值），$v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ 是新旧值的凸插值。代入展开后与 C3 紧凑形式完全等价。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §2.2（展开推导："Letting $v_t^{\text{old}} = S_{t-1} k_t$, ... the above can be simplified"）；Schlag et al. 2021 ICML arXiv:2102.11174 §4.2 Eq. 23（remove + write 形式）。
- **适用条件**：与 C3 同（同一公式的不同写法）。
- **置信状态**：已确认（代数等价，可直接验证）。

### C5

- **论断内容**：Delta 规则形式上借鉴 Widrow-Hoff 1960 经典 delta rule 的"误差 × 学习率 × 输入"结构；但作用对象不同——经典 delta rule 修改训练阶段的参数权重 $w \leftarrow w + \eta (y - \hat y) x$，DeltaNet delta rule 修改前向推理阶段的记忆矩阵状态 $S \leftarrow S + \beta (v - Sk) k^\top$。Schlag 2021 用 "akin to"（类比）描述两者关系，而非"等同"。
- **来源定位**：Schlag et al. 2021 ICML arXiv:2102.11174 §4.2（"we introduce an improved programming instruction akin to the famous error-correcting delta-rule (Widrow & Hoff, 1960)" 与 "our programming instruction or update rule is effectively a delta rule with a dynamic learning rate $\beta^{(i)}$"）；Yang et al. ICLR 2025 arXiv:2412.06464 §2.3（"The delta update rule (Widrow et al., 1960; Schlag et al., 2021b) ..."）；Widrow & Hoff 1960 原文 "Adaptive switching circuits", IRE WESCON Convention Record, pp. 96–104, 1960。
- **适用条件**：概念边界说明，不依赖具体超参。
- **置信状态**：已确认。

### C6

- **论断内容**：$I - \beta_t k_t k_t^\top$ 是广义 Householder 变换（identity plus rank-one matrix）。仅当 $\|k_t\| = 1$ 且 $\beta_t = 1$ 时，$I - k_t k_t^\top$ 是沿 $k_t$ 方向的正交投影——只擦除 $k_t$ 方向的分量，保留与之正交的 $d-1$ 维子空间不变。Yang 2024 NeurIPS §3.3 说明工程实现中通过 L2 归一化 key 向量获得此投影性质。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §3.1（"which can be seen as applying a generalized Householder transformation (i.e., matmul with an identity plus rank-one matrix)"）与 §3.3（"only erases information in one subspace while keeping the other $d-1$ subspace intact"）；Bischof & Van Loan 1985 WY representation 论文（Yang 引用为 [9]）。
- **适用条件**：L2 归一化的 DeltaNet；其他情况下 $I - \beta k k^\top$ 仅为秩-1 扰动，不保持范数与正交投影性质。
- **置信状态**：已确认。

### C7

- **论断内容**：Gated DeltaNet 在 delta 规则上引入标量门 $\alpha_t \in (0,1)$，公式为 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$。$\alpha_t \to 0$ 时清空整个记忆（不论 key 方向），$\alpha_t \to 1$ 时退化为纯 delta 规则（C3）。Mamba2 的更新规则 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ 是 Gated DeltaNet 在 $\beta_t = 0$ 时的退化情形。
- **来源定位**：Yang et al. ICLR 2025 arXiv:2412.06464 §3.1 Eq. 8（gated delta rule 公式）；§2.2（Mamba2 公式 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$）；§1（"gating enables rapid memory erasure while the delta rule facilitates targeted updates"）。
- **适用条件**：Gated DeltaNet 与 Mamba2 对比。
- **置信状态**：已确认。

### C8

- **论断内容**：DeltaNet 在合成检索任务上显著优于线性注意力和 Mamba；但在 MAD benchmark 的 memorize 任务上弱于 Mamba（DeltaNet 52.8 vs. Mamba 89.5）。1.3B / 100B tokens 设置下 DeltaNet 的 Wiki PPL = 16.87，LMB PPL = 12.21，略优于 Mamba（17.06 / 13.89）和 GLA，与 Transformer++（16.85 / 13.44）接近。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §4.1 Table 2（MAD benchmark）与 §4.2 Table 3（1.3B LM PPL）。
- **适用条件**：340M 和 1.3B 模型规模、15B / 100B tokens 训练量；不直接外推到更大规模或不同数据分布。
- **置信状态**：已确认（论文报告值）。

## F 公式（推导/边界）

### F1

- **论断内容**：线性注意力的 retrieval 展开为 $S k_j = v_j + \sum_{i \neq j} (k_i^\top k_j) v_i$，其中 $\sum_{i \neq j} (k_i^\top k_j) v_i$ 是 retrieval error。当所有 $k_i$ 两两正交时此项为 0；当 $L > d$ 时必然存在 $k_i^\top k_j \neq 0$。
- **来源定位**：直接代数展开 $S = \sum_i v_i k_i^\top$（C1）后右乘 $k_j$；C2 来源同 Yang 2024 §2.1。
- **适用条件**：线性注意力（任意 $S$ 形式）。
- **置信状态**：已确认（代数恒等式）。

### F2

- **论断内容**：Delta 规则在 $\beta_t = 0$ 时退化为 $S_t = S_{t-1}$（记忆不变）；在 $\beta_t = 1$ 时退化为 $S_t = S_{t-1}(I - k_t k_t^\top) + v_t k_t^\top$，当 $\|k_t\| = 1$ 时 $I - k_t k_t^\top$ 是沿 $k_t$ 方向的正交投影，意味着完全擦除 $k_t$ 方向的旧关联并写入新值。
- **来源定位**：直接代入 C3 + C6。
- **适用条件**：DeltaNet 边界。
- **置信状态**：已确认（直接代入）。

### F3

- **论断内容**：Gated DeltaNet 在 $\alpha_t = 1$ 时退化为 DeltaNet（C3）；在 $\beta_t = 0$ 且 $\alpha_t$ 任意时退化为 $S_t = \alpha_t S_{t-1}$（无新值写入，即 Mamba2 在 $v_t = 0$ 时的特殊情形）。
- **来源定位**：直接代入 C7。
- **适用条件**：Gated DeltaNet 边界。
- **置信状态**：已确认（直接代入）。

## N 数字（实验/构造）

### N1

- **论断内容**：Yang et al. NeurIPS 2024 在 1.3B 参数 / 100B tokens 设置下报告——DeltaNet (w. conv) Wiki PPL = 16.87, LMB PPL = 12.21；Mamba (w. conv) Wiki PPL = 17.06, LMB PPL = 13.89；GLA (w. conv) Wiki PPL = 17.25, LMB PPL = 14.92；Transformer++ Wiki PPL = 16.85, LMB PPL = 13.44。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §4.2 Table 3。
- **适用条件**：1.3B 模型、100B tokens、标准 LM 评估协议；不外推到其他规模或数据。
- **置信状态**：已确认（论文报告值）。

### N2

- **论断内容**：Yang et al. NeurIPS 2024 在 MAD benchmark（Poli et al. 2024）上报告——DeltaNet 在 In-Context Recall = 100, Fuzzy Recall = 35.7, Noisy Recall = 100, Selective Copy = 100, Memorize = 52.8, Compress = 42.2, Average = 71.8；Mamba 在相同任务上分别为 90.4, 6.7, 90.1, 86.3, 89.5, 52.7, 69.3。
- **来源定位**：Yang et al. NeurIPS 2024 arXiv:2406.06484 §4.1 Table 2。
- **适用条件**：MAD benchmark 标准设置（340M 模型，15B tokens）。
- **置信状态**：已确认（论文报告值）。

### N3（教学示例数字，非外部来源）

- **论断内容**：本页手算例子使用 $d = 2$ 维、$L = 2$ 步序列，key 与 value 取易于手算的具体数值（详见 outline §4.4）。
- **来源定位**：教学示例（人为构造），构造目的是让读者能用纸笔复算每个步骤。
- **适用条件**：仅用于讲解，不代表真实模型推荐超参或实验结果。
- **置信状态**：教学示例（非外部证据）。
