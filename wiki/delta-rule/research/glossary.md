# Delta 规则与 DeltaNet · 术语表

登记全文所有首次出现的术语、缩写和符号。后续阶段写作和审查以此为准。

## 术语

| 名称 | 首次出现 | 定义/含义 |
|---|---|---|
| Delta 规则（Delta Rule） | 第 2 章标题 | 在 DeltaNet 语境下指前向推理阶段更新记忆矩阵 $S$ 的递归规则：写入新 key-value 关联前先沿当前 key 方向擦除旧值，再做加权写入。形式 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$。 |
| 经典 Widrow-Hoff delta rule | 第 2 章类比 | Widrow & Hoff 1960 提出的训练阶段参数更新规则 $w \leftarrow w + \eta(y - \hat y)x$。是 DeltaNet delta rule 的命名来源与形式类比对象，但作用对象（参数权重 vs. 记忆矩阵）不同。 |
| DeltaNet | 第 2 章溯源 | 把线性注意力的纯加性累加替换为 delta 规则的模型。Schlag et al. 2021 ICML 首次提出（原文称 "Delta Network"），Yang et al. NeurIPS 2024 正式命名 "DeltaNet" 并给出并行训练算法。 |
| Gated DeltaNet | 第 5 章对比 | Yang et al. ICLR 2025 提出的 DeltaNet 扩展，在 delta 规则上引入标量遗忘门 $\alpha_t$。公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$。 |
| 线性注意力（Linear Attention） | 第 1 章动机 | 把 softmax 注意力的 softmax 去掉、使 K-Q 内积可重排为递归形式的注意力变体。本页只引用其加性递归 $S_t = S_{t-1} + v_t k_t^\top$ 与 retrieval $o = S q$。`wiki/linear-attention/` 不存在，本页标占位。 |
| Mamba2 | 第 5 章对比 | 基于 SSM 的线性 RNN 模型，递归形式 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$，$\alpha_t$ 是数据相关标量衰减。Gated DeltaNet 在 $\beta_t = 0$ 时退化为 Mamba2 形式。 |
| GLA / RetNet / RWKV-6 | 第 5 章对比 | 其他"矩阵状态 + 结构化递归"线性 RNN 家族成员，本页只在一句话对比中提及，不展开。 |
| Widrow-Hoff 1960 | 第 2 章溯源 | "Adaptive switching circuits"，IRE WESCON Convention Record, pp. 96–104, 1960。经典 delta rule 的原始出处。 |
| Householder 变换 | 第 3 章几何 | 形如 $I - \beta k k^\top$ 的秩-1 扰动矩阵（identity plus rank-one）。Yang 2024 NeurIPS §3.1 称其为"广义 Householder 变换"。 |
| Householder 投影 | 第 3 章几何 | Householder 变换在 $\|k\| = 1, \beta = 1$ 时的特殊情形：$I - k k^\top$ 是沿 $k$ 方向的正交投影，只擦除 $k$ 方向分量，保留正交补 $d-1$ 维不变。 |
| WY 表示 | 第 5 章工程 | Bischof & Van Loan 1985 提出的 Householder 矩阵乘积的紧凑表示。Yang 2024 NeurIPS §3 用其实现 delta 规则的 chunkwise 并行训练。本页不展开。 |
| flash-linear-attention 库 | 第 5 章工程 | 开源的 Triton 实现库（fla-org/flash-linear-attention），提供 DeltaNet、Gated DeltaNet 等线性注意力模型的高效 kernel。 |
| Kimi Delta Attention (KDA) | 页面开头动机 | Kimi K3 模型使用的注意力变体，基于 delta 规则递归。本页作为学习动机提及，不展开 KDA 架构。 |
| MQAR | 第 5 章实验 | Multi-Query Associative Recall，合成 in-context retrieval 评测任务。DeltaNet 在此任务上接近 100% 准确率。 |
| MAD benchmark | 第 5 章实验 | Poli et al. 2024 提出的合成任务套件，包含 Compress / Fuzzy Recall / In-Context Recall / Memorize / Noisy Recall / Selective Copy。 |
| Retrieval error / key 碰撞 | 第 1 章动机 | 线性注意力 retrieval $S k_j = v_j + \sum_{i \neq j}(k_i^\top k_j) v_i$ 中的 $\sum_{i \neq j}(k_i^\top k_j) v_i$ 项。当 $L > d$ 时不可避免。 |
| 凸插值 | 第 3 章 | $v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ 是两个向量的凸组合（系数非负且和为 1）。 |
| 正交投影 | 第 3 章几何 | 矩阵 $P$ 满足 $P^2 = P$ 且 $P^\top = P$。$I - k k^\top$ 在 $\|k\| = 1$ 时是正交投影。 |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| KDA | Kimi Delta Attention | 页面开头 |
| SSM | State Space Model | 第 5 章对比 |
| GLA | Gated Linear Attention | 第 5 章对比 |
| RWKV | Receptance Weighted Key Value（一种线性 RNN 模型家族） | 第 5 章对比 |
| LM | Language Modeling | 第 5 章实验 |
| PPL | Perplexity | 第 5 章实验 |
| RNN | Recurrent Neural Network | 第 1 章动机 |
| FFN | Feed-Forward Network（仅在第 5 章一句话提及，不展开） | 第 5 章对比 |
| WY | WY representation（Bischof & Van Loan 1985） | 第 5 章工程 |

## 符号

| 符号 | 含义 | 首次出现 | 形状/范围 |
|---|---|---|---|
| $S_t$ | 第 $t$ 步的矩阵状态（记忆矩阵） | 第 1 章动机 | $\mathbb{R}^{d_v \times d_k}$ |
| $S_{t-1}$ | 上一步的矩阵状态 | 第 1 章动机 | $\mathbb{R}^{d_v \times d_k}$ |
| $S_0$ | 初始矩阵状态（通常为零矩阵） | 第 2 章手算 | $\mathbb{R}^{d_v \times d_k}$ |
| $d_v$ | value 维度 | 第 1 章动机 | 正整数 |
| $d_k$ | key 维度 | 第 1 章动机 | 正整数 |
| $d$ | 通常 $d_v = d_k = d$，简化记法 | 第 1 章动机 | 正整数 |
| $L$ | 序列长度 | 第 1 章动机 | 正整数 |
| $t$ | 时间步索引 | 第 1 章动机 | $\{1, 2, \ldots, L\}$ |
| $i, j$ | 求和索引 | 第 1 章推导 | $\{1, \ldots, L\}$ |
| $x_t$ | 第 $t$ 步输入向量 | 第 2 章公式 | $\mathbb{R}^{d_{\text{in}}}$ |
| $q_t$ | query 向量 | 第 1 章动机 | $\mathbb{R}^{d_k}$ |
| $k_t$ | key 向量 | 第 1 章动机 | $\mathbb{R}^{d_k}$ |
| $v_t$ | value 向量 | 第 1 章动机 | $\mathbb{R}^{d_v}$ |
| $o_t$ | 输出向量 | 第 1 章动机 | $\mathbb{R}^{d_v}$ |
| $W_Q, W_K, W_V$ | query/key/value 投影权重 | 第 2 章公式 | $\mathbb{R}^{d \times d_{\text{in}}}$（可学习参数） |
| $W_\beta$ | $\beta_t$ 投影权重 | 第 2 章公式 | $\mathbb{R}^{1 \times d_{\text{in}}}$（可学习参数） |
| $\beta_t$ | 写入强度 / 学习率 | 第 2 章公式 | $\sigma(W_\beta x_t) \in (0, 1)$ |
| $\alpha_t$ | Gated DeltaNet 的标量遗忘门 | 第 5 章对比 | $\in (0, 1)$，数据相关 |
| $I$ | 单位矩阵 | 第 2 章公式 | $\mathbb{R}^{d_k \times d_k}$ |
| $\sigma$ | sigmoid 函数 $\sigma(z) = 1/(1+e^{-z})$ | 第 2 章公式 | $\mathbb{R} \to (0, 1)$ |
| $\top$ | 转置 | 第 1 章动机 | 上标 |
| $v_t^{\text{old}}$ | 当前记忆对 $k_t$ 的响应（旧值） | 第 3 章等价形式 | $\mathbb{R}^{d_v}$，$= S_{t-1} k_t$ |
| $v_t^{\text{new}}$ | 新写入值（新旧值凸插值） | 第 3 章等价形式 | $\mathbb{R}^{d_v}$，$= \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ |
| $\|k\|$ | 向量 $k$ 的 L2 范数 | 第 3 章几何 | $\sqrt{k^\top k}$ |
| $\otimes$ | 外积 $u \otimes v = u v^\top$（Schlag 2021 原文记法） | 第 2 章溯源 | 与 $v k^\top$ 等价 |
| $\eta$ | 经典 Widrow-Hoff delta rule 的学习率 | 第 2 章类比 | $\in \mathbb{R}_{>0}$（无 sigmoid 约束） |
| $y, \hat y$ | 经典 delta rule 的真值与预测 | 第 2 章类比 | 标量或向量 |
| $\gamma$ | RetNet 中的固定衰减系数（对比项） | 第 5 章对比 | $\in (0, 1)$ |
