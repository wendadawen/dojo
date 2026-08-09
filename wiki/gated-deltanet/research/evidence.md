# Gated DeltaNet · 核心论断与证据

核心论断编号：C 论断 / F 公式 / N 数字。来源优先级：原始论文 > 权威教材 > 官方文档 > 源码。

## C 论断

### C1（DeltaNet 缺乏全局遗忘能力）

- **论断**：DeltaNet 的 delta rule $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 只擦除当前 $k_t$ 一个方向的旧值，要清空整段过时记忆需逐 key 串行擦除，缺乏快速全局遗忘能力。
- **来源定位**：Yang, Kautz & Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025, arXiv:2412.06464 §2.3 与 §1。§2.3 原文 "it lacks the ability to rapidly clear outdated information, particularly during context switches"。
- **适用条件**：DeltaNet 公式成立时（β_t ∈ (0,1)，key 由输入投影得到）。
- **置信状态**：已确认。

### C2（Mamba2 标量衰减门机制）

- **论断**：Mamba2 用 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ 的标量 α_t 全局衰减整张状态矩阵，所有 key-value 关联按同一比例缩小，不区分不同 key 的重要性。
- **来源定位**：Yang 2025 ICLR arXiv:2412.06464 §2.2。原文 "Mamba2 decays all key-value associations uniformly by a dynamic ratio α_t" 与 "this approach does not account for the varying importance of different key-value associations"。
- **适用条件**：Mamba2 公式成立时（α_t ∈ (0,1) 数据相关）。
- **置信状态**：已确认。

### C3（Gated DeltaNet 公式）

- **论断**：Gated DeltaNet 的递归更新为 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$，把 Mamba2 的标量衰减门 α_t 嵌入 DeltaNet 的 delta rule 擦除项前。
- **来源定位**：Yang 2025 ICLR arXiv:2412.06464 §3.1 Eq.8。原文公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$。
- **适用条件**：α_t, β_t ∈ (0,1) 数据相关标量；$S \in \mathbb{R}^{d_v \times d_k}$；$q_t, k_t, v_t$ 由 $x_t$ 经线性投影得到。
- **置信状态**：已确认。

### C4（α_t 与 β_t 的职责分工）

- **论断**：α_t 是数据相关标量衰减门，控制整张 S_{t-1} 的全局衰减；β_t 是数据相关标量写入强度，控制当前 k_t 方向新旧值的混合比例。两者职责不重叠。
- **来源定位**：Yang 2025 ICLR arXiv:2412.06464 §3.1。原文 "the data-dependent gating term α_t ∈ (0,1) controls state decay" 与 "gating enables rapid memory erasure while the delta rule facilitates targeted updates"。
- **适用条件**：Gated DeltaNet 公式成立时。
- **置信状态**：已确认。

### C5（"先衰减后擦写"等价解读）

- **论断**：Gated DeltaNet 公式的乘法顺序 $\alpha_t(I - \beta_t k_t k_t^\top)$ 表示先对 S_{t-1} 整体乘 α_t 衰减，再用 $I - \beta_t k_t k_t^\top$ 擦除 k_t 方向，最后加 $\beta_t v_t k_t^\top$ 写入。即"先衰减、再擦写"。
- **来源定位**：由 C3 公式结合矩阵乘法结合律直接得到：$S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) = (S_{t-1} \alpha_t)(I - \beta_t k_t k_t^\top)$（标量 α_t 可交换）。
- **适用条件**：α_t 是标量（可与矩阵交换）；若 α_t 是 channel-wise 向量则不能这样交换（这是 KDA 的情形）。
- **置信状态**：已确认（代数恒等式）。

### C6（chunkwise 并行训练算法的存在与作用）

- **论断**：Gated DeltaNet 的递归形式 $S_t$ 依赖 $S_{t-1}$，串行计算无法利用 GPU 并行。Yang 2025 ICLR §3.2 给出 chunkwise 并行算法：序列切成 chunk，chunk 内用扩展的 WY 表示（Bischof & Van Loan 1985）把递归展开为矩阵乘法，兼容 Tensor Core；chunk 间递归传状态。门控项 α_t 只做逐元素乘法，不破坏矩阵乘法结构。
- **来源定位**：Yang 2025 ICLR arXiv:2412.06464 §3.2 Eq.9-12 与 §3.2 末 "the gating term only performs element-wise multiplication and does not affect the matrix multiplication structure, thus compatible with tensor core GPU optimization"。WY 表示引自 Bischof & Van Loan 1985（论文引用 [Bischof & Loan, 1985]）。
- **适用条件**：chunk size C 为 16 的倍数（Tensor Core 要求）。
- **置信状态**：已确认。

### C7（与 Mamba2、DeltaNet 的机制对比）

- **论断**：Gated DeltaNet 是 DeltaNet 与 Mamba2 的统一——DeltaNet 只有方向擦除无全局衰减，Mamba2 只有全局衰减无方向擦除，Gated DeltaNet 同时具备两者。
- **来源定位**：Yang 2025 ICLR arXiv:2412.06464 §1 与 §3.1。§1 原文 "we observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates"。
- **适用条件**：三者公式并列对比时。
- **置信状态**：已确认。

### C8（与 KDA 的 α_t 区别）

- **论断**：KDA 基于 Gated DeltaNet 但把 α_t 从标量改为 channel-wise 向量（每 key 通道一个独立衰减率），并把 decay 参数化改为 lower-bounded scaled sigmoid（限定 $g \in (g_{\min}, 0)$）。Gated DeltaNet 的 α_t 是标量、无下界约束。
- **来源定位**：Kimi K3 Technical Report §2.1.1 Eq.1、Eq.5（KDA 公式与 scaled sigmoid）；Gated DeltaNet 公式见 C3。`wiki/kda/index.html` 已生成。
- **适用条件**：对比 Gated DeltaNet 与 KDA 时。
- **置信状态**：已确认。

## F 公式

### F1（Gated DeltaNet 递归公式）

- **公式**：$S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$
- **来源**：Yang 2025 ICLR arXiv:2412.06464 §3.1 Eq.8。
- **符号**：$S \in \mathbb{R}^{d_v \times d_k}$；$q_t, k_t, v_t$ 由 $x_t$ 经 $W_Q, W_K, W_V$ 线性投影得到；$\alpha_t = \sigma(W_\alpha x_t) \in (0,1)$ 标量；$\beta_t = \sigma(W_\beta x_t) \in (0,1)$ 标量；输出 $o_t = S_t q_t$。

### F2（Gated DeltaNet 的退化）

- **公式**：
  - α_t → 1：$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$（DeltaNet）
  - β_t → 0：$S_t = S_{t-1}(\alpha_t \cdot I) + 0 = \alpha_t S_{t-1}$（Mamba2 在 v_t=0 时的特殊情形）
  - α_t → 0：$S_t = 0 + \beta_t v_t k_t^\top$（清空历史只保留当前写入）
- **来源**：由 F1 直接代入。Mamba2 公式见 Yang 2025 ICLR §2.2 $S_t = \alpha_t S_{t-1} + v_t k_t^\top$；DeltaNet 公式见 Yang 2024 NeurIPS arXiv:2406.06484 §2.2 或 `wiki/delta-rule/`。

### F3（Mamba2 递归公式，对比基线）

- **公式**：$S_t = \alpha_t S_{t-1} + v_t k_t^\top$
- **来源**：Yang 2025 ICLR arXiv:2412.06464 §2.2。

### F4（DeltaNet 递归公式，对比基线）

- **公式**：$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$
- **来源**：Yang 2024 NeurIPS arXiv:2406.06484 §2.2；`wiki/delta-rule/`。

## N 数字

### N1（1.3B/100B tokens 语言建模 PPL）

- **数字**：1.3B 参数、100B tokens（FineWeb-Edu）训练。Gated DeltaNet Wiki PPL = 16.42, LMB PPL = 12.17；Mamba2 = 16.56 / 12.56；DeltaNet = 17.71 / 16.88；Transformer++ = 18.53 / 18.32。Gated DeltaNet-H1 = 16.07 / 12.12；Gated DeltaNet-H2 = 15.91 / 12.55。
- **来源**：Yang 2025 ICLR arXiv:2412.06464 §4 Table 2。
- **实验条件**：1.3B 参数，100B tokens，FineWeb-Edu 数据集。不外推到其他规模或数据。

### N2（S-NIAH 检索任务）

- **数字**：S-NIAH-2（真实文本检索）4K 长度：DeltaNet = 18.6，Mamba2 = 56.2，Gated DeltaNet = 92.2。S-NIAH-3 4K（无法外推，仅 1K/2K/4K）：DeltaNet = 22.4，Mamba2 = 4.6，Gated DeltaNet = 27.6。
- **来源**：Yang 2025 ICLR arXiv:2412.06464 §4 Table 3。
- **实验条件**：S-NIAH（Single-Needle in a Haystack）检索基准，不同长度 1K/2K/4K/8K。说明 Gated DeltaNet 的自适应内存管理在真实文本检索上优于 DeltaNet。

### N3（常识推理平均分）

- **数字**：1.3B 模型常识推理 7 项平均：Gated DeltaNet = 55.32，Mamba2 = 54.89，DeltaNet = 52.14，Transformer++ = 52.25。Gated DeltaNet-H1 = 56.40，H2 = 56.18。
- **来源**：Yang 2025 ICLR arXiv:2412.06464 §4 Table 2。
- **实验条件**：PIQA、HellaSwag、WinoGrande、ARC-e、ARC-c、SIQA、BoolQ 七项平均。
