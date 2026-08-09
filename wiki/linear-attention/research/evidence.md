# 线性注意力核心论断与证据

来源优先级：原始论文 > 权威教材 > 官方文档。一般博客仅辅助定位，不作为核心论断唯一依据。

## C 论断

### C1：标准 softmax 注意力公式与复杂度

- 论断内容：标准 softmax 自注意力定义为 $V' = \mathrm{softmax}(QK^T / \sqrt{d})V$，其中 $Q, K, V \in \mathbb{R}^{N \times d}$；计算 $QK^T$ 产生 $N \times N$ 注意力矩阵，因此时间复杂度为 $O(N^2 d)$、空间复杂度为 $O(N^2)$，与序列长度 N 成二次关系
- 来源定位：Katharopoulos et al. 2020, §3, Eq.(2) 给出 softmax attention 公式；§3.2.1 原文 "the computational cost of softmax attention scales with O(N²)"。Choromanski et al. 2020, §1, Eq.(1) 同样给出 $\mathbf{A} = \exp(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})$，时间 $O(L^2 d)$、空间 $O(L^2 + Ld)$
- 适用条件：N 为序列长度、d 为每头维度；不考虑 FlashAttention 等 IO 优化（不改变复杂度阶）
- 置信状态：已确认

### C2：广义注意力定义与 sim 必须非负

- 论断内容：对任意非负相似度函数 sim，可定义广义注意力 $V'_i = \frac{\sum_{j=1}^{N} \mathrm{sim}(Q_i, K_j) V_j}{\sum_{j=1}^{N} \mathrm{sim}(Q_i, K_j)}$；softmax 注意力是 sim(q,k)=exp(q^T k / √d) 的特例。sim 非负是必要条件——分母作归一化用，必须为正
- 来源定位：Katharopoulos et al. 2020, §3, Eq.(3)；§3.2 原文 "the only constraint we need to impose to sim(·), in order for equation 3 to define an attention function, is to be non-negative"
- 适用条件：分母为正（要求 sim 非负且至少有一个 j 使 sim(Q_i, K_j) > 0）
- 置信状态：已确认

### C3：核分解与结合律重排

- 论断内容：当 $\mathrm{sim}(q, k) = \phi(q)^T \phi(k)$ 可分解为某特征映射 $\phi$ 的内积时，广义注意力可改写为 $V'_i = \frac{\phi(Q_i)^T \sum_j \phi(K_j) V_j^T}{\phi(Q_i)^T \sum_j \phi(K_j)}$；矩阵形式 $(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$ 由矩阵乘法结合律得到。先算 $\phi(K)^T V$（与 N 无关的固定大小聚合），再与 $\phi(Q)$ 相乘，避免构造 N×N 矩阵；复杂度降为 $O(N \cdot d \cdot d')$，$d'$ 为 $\phi$ 输出维度
- 来源定位：Katharopoulos et al. 2020, §3, Eq.(4)(5)(6)；§3.2.1 原文 "we can compute ∑φ(K_j)V_j^T and ∑φ(K_j) once and reuse them for every query"
- 适用条件：sim 可分解为有限维 φ 内积（softmax 的 exp 核不满足，需用近似核）
- 置信状态：已确认

### C4：因果递归形式与常数时间推理

- 论断内容：因果掩码下第 i 个 query 只能看到前 i 个 key，线性注意力可写为递归：$s_0=0, z_0=0$；$s_i = s_{i-1} + \phi(K_i) V_i^T$（attention memory，形状 $d' \times d$）；$z_i = z_{i-1} + \phi(K_i)$（normalizer memory，形状 $d'$）；$V'_i = \frac{\phi(Q_i)^T s_i}{\phi(Q_i)^T z_i}$。$s_i$、$z_i$ 形状与 N 无关，因此每步推理时间和内存均为 $O(d \cdot d')$（典型 $d'=d$ 时为 $O(d^2)$），与序列长度 N 无关
- 来源定位：Katharopoulos et al. 2020, §3, Eq.(9)(10)(11)(12)（因果形式）；§3.4, Eq.(16)–(20)（RNN 递归形式，含 W_Q、W_K、W_V 投影与残差连接、激活函数 $f_l$）
- 适用条件：因果掩码（自回归生成）；$\phi(Q_i)^T z_i > 0$ 保证分母非零
- 置信状态：已确认

### C5：表达力代价与核选择约束

- 论断内容：softmax 核 $\exp(q^T k)$ 没有有限维正特征映射（特征映射为无穷维），因此无法精确线性化；线性注意力必须改用近似核或新定义的核。Katharopoulos 选 $\phi(x) = \mathrm{elu}(x) + 1$，利用 $\mathrm{elu}(x) \in (-1, +\infty)$ 保证 $\phi(x) > 0$ 满足非负约束；选 elu 而非 relu 是为了"避免负值区域梯度为 0"。Performer（Choromanski et al. 2020）则用随机特征 $\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}}(\exp(\omega_1^T x), \ldots, \exp(\omega_m^T x))$ 近似 softmax 核，无偏但方差非零。这些替代核的表达力严格弱于 softmax
- 来源定位：Katharopoulos et al. 2020, §3.2.1, Eq.(7)（φ=elu+1）；§3.2.1 原文 "We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative"。Choromanski et al. 2020, §2, Eq.(5)、Lemma 1（正随机特征构造）。softmax 核无穷维：Choromanski §2 原文讨论 trig features 因负值爆炸、需正特征近似
- 适用条件：线性注意力家族通用
- 置信状态：已确认

## F 公式

### F1：标准 softmax 注意力

$$V' = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- 来源：Katharopoulos Eq.(2)、Choromanski Eq.(1)
- 等价写法：$V'_i = \frac{\sum_j \exp(Q_i^T K_j / \sqrt{d}) V_j}{\sum_j \exp(Q_i^T K_j / \sqrt{d})}$

### F2：广义注意力（核形式）

$$V'_i = \frac{\sum_{j=1}^{N} \phi(Q_i)^T \phi(K_j) V_j}{\sum_{j=1}^{N} \phi(Q_i)^T \phi(K_j)} = \frac{\phi(Q_i)^T \sum_j \phi(K_j) V_j^T}{\phi(Q_i)^T \sum_j \phi(K_j)}$$

- 来源：Katharopoulos Eq.(3)(4)(5)
- 关键变换：分子分母同时把 $\phi(Q_i)^T$ 提到求和号外（结合律）

### F3：因果递归形式

$$s_0 = 0, \quad z_0 = 0$$
$$s_i = s_{i-1} + \phi(K_i) V_i^T$$
$$z_i = z_{i-1} + \phi(K_i)$$
$$V'_i = \frac{\phi(Q_i)^T s_i}{\phi(Q_i)^T z_i}$$

- 来源：Katharopoulos Eq.(10)(11)(12)
- 完整 RNN 形式（含投影与残差，Katharopoulos Eq.(16)–(20))：

$$y_i = f_l\left(\frac{\phi(x_i W_Q)^T s_i}{\phi(x_i W_Q)^T z_i} + x_i\right)$$

$$s_i = s_{i-1} + \phi(x_i W_K)(x_i W_V)^T, \quad z_i = z_{i-1} + \phi(x_i W_K)$$

### F4：Katharopoulos 的 φ 选择

$$\phi(x) = \mathrm{elu}(x) + 1$$

- 来源：Katharopoulos Eq.(7)
- $\mathrm{elu}(x) = \begin{cases} x & x \geq 0 \\ e^x - 1 & x < 0 \end{cases}$，故 $\mathrm{elu}(x) \in (-1, +\infty)$，$\phi(x) > 0$

### F5：Performer 的正随机特征 φ

$$\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}}\left(\exp(\omega_1^T x), \ldots, \exp(\omega_m^T x)\right)$$

- 来源：Choromanski Eq.(5) + Lemma 1，$\omega_i \sim \mathcal{N}(0, I_d)$ 独立同分布
- 无偏：$\mathbb{E}[\phi(x)^T \phi(y)] = \exp(x^T y) = \mathrm{SM}(x, y)$

## N 数字

### N1：Katharopoulos 报告的推理加速

- 数字：在 CIFAR-10 自回归图像生成上，线性注意力比 softmax transformer 快 4000×
- 来源：Katharopoulos et al. 2020, Table 2，原文 "up to 4000x faster on autoregressive prediction of very long sequences"
- 适用条件：自回归生成、长序列；论文实验设置
- 置信状态：已确认

### N2：Katharopoulos 表达力对比

- 数字：MNIST 生成 bits/dim：linear 0.644 vs softmax 0.621（线性略差）；语音识别 WSJ PER：linear 8.08 vs softmax 5.12（线性明显差）
- 来源：Katharopoulos et al. 2020, Table 1 与 §4 实验描述
- 适用条件：论文实验模型规模
- 置信状态：已确认

注：本页正文为说明"表达力代价"使用 N2 的定性结论（线性注意力在多数任务上略差或明显差于 softmax），具体数字不写入正文以保持小白可读性，只在文末"来源与教学说明"中登记。
