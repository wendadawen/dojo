# 线性注意力独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：wiki/linear-attention/index.html（57975 bytes, 2026-08-09 00:05）+ overview.html（6933 bytes, 2026-08-09 00:04）
- 时间：2026-08-09
- 审查依据：guides/concept/check.md（段A盲读 + 段B对照来源）
- 来源：WebSearch "linear attention Katharopoulos 2020 arxiv 2006.16236" → Katharopoulos et al. 2020 "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"（ICML 2020, arXiv:2006.16236, §3 Eq.2–20）；Choromanski et al. 2020 "Rethinking Attention with Performers"（ICLR 2021, §1–2, Eq.1–7）

## 段 A 盲读笔记（小白读者卡点）

按页面顺序阅读：

- §1（行 698）：$Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度、$d$ 是每个注意力头的维度——定义清晰。
- §1（行 706）：$QK^T$ 手算 $\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$——复算正确。
- §2（行 762 ASCII 图）：$\phi(Q)$ 形状 $N \times d'$，$d'$ 首现未明确说明是 $\phi$ 的输出维度。行 798"当 $d' = d$ 时为 $O(Nd^2)$"暗示 $d'$ 可不等于 $d$，但未定义。
- §2（行 771）：教学简化 $\phi(x) = x + 1$，已说明"并非 Katharopoulos 论文使用的 $\mathrm{elu}+1$，论文版本在下一章解释"——简化标注清晰。
- §3（行 854-862）：因果递归手算 $s_1, s_2, s_3$ 与 $z_1, z_2, z_3$——全部复算正确，$s_3 = \phi(K)^T V$ 与非因果形式一致（验证递归正确）。
- §3（行 891-892 callout）：完整 RNN 形式含 $W_Q, W_K, W_V$ 投影——$W_Q, W_K, W_V$ 首现未明确说是投影矩阵，但常见记号，读者可推断。
- §4（行 933-942 对比表）：softmax 与 $\mathrm{elu}+1$ 权重对比——全部复算正确（softmax $0.212/0.212/0.576$，elu+1 $0.300/0.300/0.400$）。

段 A 结束时逐题核对学习目标：
- 目标1（softmax $O(N^2)$ 瓶颈在哪一步）：S1 回答，$QK^T$ 这一步构造 $N \times N$ 矩阵，清晰。
- 目标2（核函数 + 结合律 → $O(N)$）：S2 回答，手算完整可复算。
- 目标3（因果递归 + 固定状态）：S3 回答，递归手算完整，$s_3 = \phi(K)^T V$ 验证一致性。
- 目标4（表达力代价从哪来）：S4 回答，softmax 核无穷维 + 对比表展示"指数聚焦 vs 线性放大"。

## 段 B 对照来源核查

### 1. 定义与机制

逐条对照 Katharopoulos et al. 2020（arXiv:2006.16236）：

- C1（softmax 复杂度 $O(N^2)$）——来源 §3.2.1"the computational cost of softmax attention scales with $O(N^2)$"；Choromanski Eq.(1) 给出 $\mathbf{A} = \exp(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})$。页面行 967 [C1] 引用 **完全一致**。
- C2（广义注意力与 sim 非负约束）——来源 §3 Eq.(3) + §3.2 原文"the only constraint we need to impose to sim(·), in order for equation 3 to define an attention function, is to be non-negative. This includes all kernels $k(x,y): \mathbb{R}^{2\times F} \to \mathbb{R}_+$"。页面行 968 [C2] 引用 **完全一致**。
- C3（核分解与结合律重排）——来源 §3 Eq.(4)(5)(6) + §3.2.1 原文"we can compute $\sum_{j=1}^N \phi(K_j) V_j^T$ and $\sum_{j=1}^N \phi(K_j)$ once and reuse them for every query"。页面行 969 [C3] 引用 **完全一致**。
- C4（因果递归与常数时间推理）——来源 §3 Eq.(9)(10)(11)(12) 因果形式 + §3.4 Eq.(16)–(20) 完整 RNN 形式。hugocisneros 笔记确认完整 RNN 形式：$s_i = s_{i-1} + \phi(x_i W_K)(x_i W_V)^T$，$z_i = z_{i-1} + \phi(x_i W_K)$，$y_i = f_l(\frac{\phi(x_i W_Q)^T s_i}{\phi(x_i W_Q)^T z_i} + x_i)$。页面行 891-892 callout **完全一致**。
- C5（表达力代价与核选择）——来源 §3.2.1 Eq.(7) $\phi = \mathrm{elu}+1$ + 原文"We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative"。页面行 971 [C5] 引用 **完全一致**。softmax 核无穷维（泰勒展开 $\exp(q^T k) = \sum_{n=0}^\infty (q^T k)^n/n!$）是数学事实，正确。

### 2. 公式与推导

- F1（softmax attention $V' = \mathrm{softmax}(\frac{QK^T}{\sqrt{d}})V$）——来源 Katharopoulos Eq.(2)（用 $D$ 表示维度，页面用 $d$，数学等价）**完全一致**。
- F2（核分解 $\mathrm{sim}(q,k) = \phi(q)^T\phi(k)$ + 重排 $V'_i = \frac{\phi(Q_i)^T \sum_j \phi(K_j) V_j^T}{\phi(Q_i)^T \sum_j \phi(K_j)}$）——来源 Eq.(3)(4)(5) **完全一致**。
- F3（因果递归 $s_i = s_{i-1} + \phi(K_i) V_i^T$，$z_i = z_{i-1} + \phi(K_i)$，$V'_i = \phi(Q_i)^T s_i / \phi(Q_i)^T z_i$）——来源 Eq.(9)(10)(11)(12) **完全一致**；完整 RNN 形式 Eq.(16)–(20) 在 callout 中给出，与 hugocisneros 笔记确认的形式完全一致。
- F4（$\phi(x) = \mathrm{elu}(x) + 1$）——来源 Eq.(7) **完全一致**。
- F5（Performer 随机特征 $\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}}(\exp(\omega_1^T x), \ldots, \exp(\omega_m^T x))$）——来源 Choromanski Eq.(5) + Lemma 1；页面正确标注这是 Performer 论文（Choromanski）的贡献，非 Katharopoulos。

手算例子复算（$N=3, d=2$，$\phi(x) = x+1$ 教学简化）：

**§1 $QK^T$**（行 706）：
- $\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}\begin{pmatrix}1&0&1\\0&1&1\end{pmatrix} = \begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$——**复算正确**（9 个元素全部验证）。

**§2 $\phi(K)^T V$**（行 777）：
- $\phi(K) = K+1 = \begin{pmatrix}2&1\\1&2\\2&2\end{pmatrix}$——**正确**。
- $\phi(K)^T V = \begin{pmatrix}2&1&2\\1&2&2\end{pmatrix}\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix} = \begin{pmatrix}4&3\\3&4\end{pmatrix}$——**复算正确**。
- $\phi(K)^T \mathbf{1} = \begin{pmatrix}5\\5\end{pmatrix}$——**正确**。
- $V'_1 = \frac{(2,1)\begin{pmatrix}4&3\\3&4\end{pmatrix}}{(2,1)\begin{pmatrix}5\\5\end{pmatrix}} = \frac{(11,10)}{15} \approx (0.733, 0.667)$——**正确**。

**§2 折叠块直接求和验证**（行 791-795）：
- $\phi(Q_1)^T\phi(K_1) = (2,1)\cdot(2,1) = 5$；$\phi(Q_1)^T\phi(K_2) = (2,1)\cdot(1,2) = 4$；$\phi(Q_1)^T\phi(K_3) = (2,1)\cdot(2,2) = 6$——**全部正确**。
- 分母 $5+4+6=15$，分子 $5(1,0)+4(0,1)+6(1,1)=(11,10)$——**与重排结果一致**。

**§3 因果递归**（行 854-862）：
- $s_1 = \begin{pmatrix}2&0\\1&0\end{pmatrix}$，$z_1 = (2,1)$——**正确**。
- $s_2 = \begin{pmatrix}2&1\\1&2\end{pmatrix}$，$z_2 = (3,3)$——**正确**。
- $s_3 = \begin{pmatrix}4&3\\3&4\end{pmatrix}$，$z_3 = (5,5)$——**正确**，且 $s_3 = \phi(K)^T V$ 与非因果形式一致（验证递归正确性）。
- $V'_3 = \frac{(2,2)\begin{pmatrix}4&3\\3&4\end{pmatrix}}{(2,2)\cdot(5,5)} = \frac{(14,14)}{20} = (0.7, 0.7)$——**正确**。
- $V'_1$（折叠块）$= \frac{(2,1)\begin{pmatrix}2&0\\1&0\end{pmatrix}}{(2,1)\cdot(2,1)} = \frac{(5,0)}{5} = (1,0) = V_1$——**正确**（因果掩码下第 1 个 token 只看到自己）。

**§4 对比表**（行 933-942）：
- $Q_3\cdot K_1 = 1$，$Q_3\cdot K_2 = 1$，$Q_3\cdot K_3 = 2$——**正确**。
- softmax 权重：$e/(2e+e^2) \approx 0.212$，$e^2/(2e+e^2) \approx 0.576$——**正确**（$\sum = 2e + e^2 \approx 5.436 + 7.389 = 12.825$，$e/12.825 \approx 0.212$，$e^2/12.825 \approx 0.576$）。
- $\mathrm{elu}+1$ 相似度：$\phi(Q_3)=(2,2)$，$\phi(K_1)=(2,1)$，$\phi(Q_3)^T\phi(K_1) = 6$；$\phi(K_2)=(1,2)$，$= 6$；$\phi(K_3)=(2,2)$，$= 8$——**全部正确**。
- $\mathrm{elu}+1$ 权重：$6/20=0.300$，$8/20=0.400$——**正确**。
- 权重比：softmax $0.576/0.212 \approx 2.72$（即 $e^{2-1}=e$），elu+1 $0.400/0.300 \approx 1.33$——**正确**。

所有手算全部可复算，结果与页面一致。

### 3. 可运行代码

页面无可运行代码（纯数学推导与手算），伪代码以公式与 ASCII 图示呈现。符合概念页性质，可接受。

### 4. 事实与推断

- Katharopoulos et al. 2020, arXiv:2006.16236, ICML 2020——页面行 654，搜索结果确认 **正确**。
- Choromanski et al. 2020, ICLR 2021（Performer）——页面行 654，**正确**。
- N1（CIFAR-10 4000x 加速）——来源摘要"up to 4000x faster on autoregressive prediction of very long sequences"；WebFetch 确认 Table 2 实际 4462x，论文称"4,000× faster" **完全一致**。
- N2（MNIST bits/dim linear 0.644 vs softmax 0.621）——WebFetch 确认 Table 1 **完全一致**。
- N2（WSJ PER linear 8.08 vs softmax 5.12）——WebFetch 确认数字 **完全一致**；但 WebFetch 指出 WSJ PER 在 Table 3，页面行 988 标注"Katharopoulos Table 1"——表号引用有误（见问题 1）。
- 教学例子 $N=3, d=2$ 标注为"教学构造...不代表真实模型输入"——诚实标注，**正确**。

### 5. 前置知识引用

- softmax 注意力（standard-attention 概念页）——页面行 702 标注"standard-attention 概念页（待生成）"，占位提示正确。
- 无其他前置概念页链接（本页是基础概念页）。

### 6. 教学简化

- 略去 $\sqrt{d}$ 缩放——已说明"不影响复杂度分析与机制讲解；正式计算应保留"，**正确**。
- 略去多头机制——已说明"多头只是把 $d$ 维空间分成多个独立子空间"，**正确**。
- 略去位置编码、梯度计算——均说明与线性化主线无关，**正确**。
- 教学 $\phi(x) = x+1$——已说明"仅适用于非负输入；论文实际 $\phi = \mathrm{elu}+1$ 处理一般情况"，并列出失效边界（$x = -2$ 时 $x+1 = -1 < 0$ 违反非负约束），**正确**。
- 略去 Performer FAVOR+ 细节——已说明"与线性注意力主线相关但不影响 Q1–Q4 回答"，**正确**。

### 7. 页面功能

- KaTeX 公式渲染配置正确。
- 折叠块（行 787-796 直接求和验证、行 876-886 $V'_1$ 因果代入）均正确，summary 清晰，收起后正文仍有完整手算摘要。
- 目录锚点正确，scroll-margin-top 避开顶部导航。
- 来源引用 [C1]-[C5]、[F1]-[F5]、[N1]-[N2] 在文末完整列出，**引用原文准确可定位**。

## 问题

- [轻微·技术] §来源行 988：页面标注"WSJ PER linear 8.08 vs softmax 5.12（Katharopoulos Table 1）"，但根据论文实际 WSJ PER 结果在 Table 3（Table 1 是 MNIST bits/dim）。表号引用有误，数字本身正确。：核对 Katharopoulos 论文原表号，将 WSJ PER 的表号更正为 Table 3，或改为"Katharopoulos 实验结果"避免具体表号。｜ 修复： ｜ 复验：
- [轻微·盲读] §2 行 762 ASCII 图 / 行 798：$d'$（$\phi$ 的输出维度）首现未明确说明。读者可从 $N \times d'$ 推断是 $\phi$ 作用于 $d$ 维后的输出维度，但页面未定义；行 798"当 $d' = d$ 时为 $O(Nd^2)$"暗示 $d'$ 可不等于 $d$（如 Performer $d' = m$）。：首次出现时加"$d'$ 是 $\phi$ 的输出维度（对 $\mathrm{elu}+1$ 有 $d'=d$；对 Performer 随机特征 $d'=m$）"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：进入修复

来源对照全部通过（C1-C5、F1-F5、N1-N2 与 Katharopoulos 2020 arXiv:2006.16236 ICML 2020 完全一致）；手算例子 100% 可复算（$QK^T$ 矩阵乘法、$\phi(K)^T V$ 聚合、因果递归 $s_1/s_2/s_3$ 与 $z_1/z_2/z_3$、$V'_1$ 与 $V'_3$ 输出、softmax 与 elu+1 权重对比表全部正确，且 $s_3 = \phi(K)^T V$ 验证了因果与非因果形式的一致性）；4000x 加速、MNIST 0.644/0.621、WSJ PER 8.08/5.12 数字均有来源支持。两个轻微问题分别为 WSJ PER 表号引用错误（Table 1 应为 Table 3）和 $d'$ 符号首现未定义，均不影响核心论断与主线理解。
