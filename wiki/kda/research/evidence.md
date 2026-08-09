# evidence.md：KDA 核心论断与证据

来源优先级：官方 config.json（数值）> K3 技术报告 §2.1.1 / §5.1（机制与公式）> 官方源码 modeling_kimi_linear.py（数据流）> Kimi Linear 论文 [63]。冲突时记录双方。

## C 类核心论断

### C1：KDA = delta rule + channel-wise forget gate + lower-bounded decay + full-rank output gate

- **论断内容**：KDA 在 delta rule 递归上叠加每通道独立衰减的遗忘门 $\mathrm{Diag}(\alpha_t)$，并把 decay 参数化为有下界的 scaled sigmoid、output gate 改为 full-rank。
- **来源定位**：K3 技术报告 §2.1.1，Eq.1（递归）、Eq.5（lower-bounded decay）、Eq.6（full-rank gate）；Fig.3（与 Kimi Linear 对比）。
- **适用条件**：Kimi K3 实例化下的 KDA。Kimi Linear [63] 的 KDA 变体用 negative-softplus + low-rank gate，与本论断不同。
- **置信状态**：已确认。

### C2：channel-wise forget gate 在 delta 更新前先按 α 衰减旧状态

- **论断内容**：递归为 $S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$，$\mathrm{Diag}(\alpha_t)$ 作用于 $S_{t-1}$ 后才进入 delta 擦写。
- **来源定位**：K3 报告 §2.1.1 Eq.1：`St = (I − βt kt kt⊤) Diag(αt)St−1 + βt kt vt⊤`。
- **适用条件**：单头形式。多头形式把 $A_h$、$b_h^\alpha$ 设为 per-head。
- **置信状态**：已确认。

### C3：K3 用 scaled sigmoid 把 log-decay 限在 (g_min, 0)，g_min = -5

- **论断内容**：$g_t^h = g_{\min}\,\mathrm{Sigmoid}(e^{A_h} z_t^h) \in (g_{\min}, 0)^{d_k}$，$\alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k}$，$g_{\min}=-5$ 固定，$A_h$ 可学习 per-head log-scale，初始化 $A_h=0$。
- **来源定位**：K3 报告 §2.1.1 Eq.5；Fig.3a 对比图；正文 "where $A_h$ is a learnable per-head log-scale and $g_{\min}=-5$ is fixed. We initialize $A_h=0$"。
- **适用条件**：K3 配置。config.json `linear_attn_config.gate_lower_bound = -5.0` 印证。
- **置信状态**：已确认。

### C4：Kimi Linear 用 negative-softplus，log-decay 无下界

- **论断内容**：Kimi Linear [63] 沿用 GDN/Mamba-2 的 $g_t^h = -e^{A_h}\mathrm{Softplus}(z_t^h) \in (-\infty, 0)^{d_k}$，单个 $\alpha$ 可任意接近 0。
- **来源定位**：K3 报告 §2.1.1 Fig.3a 与正文 "Kimi Linear uses the negative-Softplus mapping $g = -e^{A_h}\mathrm{Softplus}(z) \in (-\infty, 0)^{d_k}$ [138, 24, 63]"。
- **适用条件**：Kimi Linear 原版。K3 改了。
- **置信状态**：已确认。

### C5：lower-bound 使 16-token tile 的累积 log-decay ∈ (-80, 0)，1/Γ < e^80 落在 BF16 范围，对角 tile 也能用 Tensor Core

- **论断内容**：$g_{\min}=-5$ 时每个 $\alpha > e^{-5} \approx 6.7\times10^{-3}$；16-token tile 的累积 log-decay $\in (-80, 0)$；chunkwise 形式里 rescale keys 的 $1/\Gamma_{1\to C}$ 因此 $< e^{80}$，BF16 能表示；于是对角 tile 不再需要 position-pair 计算，全部用 dense Tensor Core matmul。
- **来源定位**：K3 报告 §2.1.1 正文 "With $g_{\min}=-5$, every retention factor satisfies $\alpha > e^{-5} \approx 6.7\times10^{-3}$, and the cumulative log-decay over a 16-token tile lies in $(-80, 0)$. The corresponding reciprocal rescaling factor is therefore smaller than $e^{80}$ and remains within the BF16 dynamic range. This finite range allows both diagonal and off-diagonal tiles to use dense Tensor Core matrix multiplications, eliminating the position-pair diagonal path."；Fig.3b。
- **适用条件**：BF16 训练；chunk size $C$ 与 16-token tile 划分按 Kimi Linear [63, 140] 的 secondary tiling。
- **置信状态**：已确认。

### C6：full-rank output gate（K3 改）替代 Kimi Linear 的 low-rank

- **论断内容**：$y_t = W_o[\mathrm{Sigmoid}(W_g x_t) \odot \mathrm{RMSNorm}(\tilde o_t)]$，$W_g$ 满秩；Kimi Linear [63] 用 low-rank。
- **来源定位**：K3 报告 §2.1.1 Eq.6 与正文 "Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear [63] to an input-dependent full-rank projection"；config.json `linear_attn_config.use_full_rank_gate = true`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### C7：chunkwise 并行形式：chunk 内并行 + chunk 间递归

- **论断内容**：chunk 大小 $C$ 下，定义累积衰减 $\Gamma_{1\to r}[t] = \prod_{r'=1}^r \alpha_{r'}[t]$（Eq.3），UT 变换得 $V_e[t] = U[t] - W[t]S[t]$；chunk 内并行 $A[t] = \mathrm{Tril}((Q[t]\odot\Gamma_{1\to C}[t])(K[t]/\Gamma_{1\to C}[t])^\top)$，$O[t] = (\Gamma_{1\to C}[t]\odot Q[t])S[t] + A[t]V_e[t]$（Eq.4）；第一项是 inter-chunk（前序 chunk 的状态），第二项是 intra-chunk。
- **来源定位**：K3 报告 §2.1.1 Eq.3、Eq.4 与正文 "The first term in $O[t]$ carries information from preceding chunks, whereas the second term accounts for interactions within the current chunk"。
- **适用条件**：$C$ 内的位置 $1 \le i \le j \le C$；Tril 保留对角（每个 output 读 current-token update 后的状态）。
- **置信状态**：已确认。UT 变换的完整推导 K3 报告指向 [63]，本文不展开。

### C8：K3 用 69 层 KDA + 24 层 Gated MLA，3:1 混合

- **论断内容**：每 4 层一组 = 3 KDA + 1 Gated MLA，backbone 末尾再加 1 层 Gated MLA 保证最后一层是全局注意力。从 1 到 88 是 22 个完整 3:1 块，89-91 是最后 3 个 KDA，92、93 是末尾 2 个 Gated MLA。共 69 KDA + 24 Gated MLA = 93 层。
- **来源定位**：K3 报告 §2.1 正文 "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio... An additional Gated MLA layer is placed at the end of the backbone, ensuring that the final layer always performs global attention"；config.json `text_config.num_hidden_layers = 93`、`linear_attn_config.kda_layers`（69 个元素）、`linear_attn_config.full_attn_layers`（24 个元素，含 92、93）。
- **适用条件**：K3-Instruct（HF `moonshotai/Kimi-K3`）。
- **置信状态**：已确认。config.json 的 `kda_layers` 列表手动计数 69 项，与 3:1 推算一致。

### C9：K3 KDA 的工程数值

- **论断内容**：head_dim=128，num_heads=96，short_conv_kernel_size=4，gate_lower_bound=-5.0，use_full_rank_gate=true，hidden_size=7168，max_position_embeddings=1048576（1M）。
- **来源定位**：config.json `text_config.linear_attn_config`、`text_config.hidden_size`、`text_config.max_position_embeddings`。
- **适用条件**：K3-Instruct 配置。
- **置信状态**：已确认。

### C10：KDA 状态约定是"状态在右"（S ∈ R^{d_k × d_v}），与 delta-rule 概念页的"状态在左"是转置

- **论断内容**：K3 报告 KDA 用 $S_t \in \mathbb{R}^{d_k \times d_v}$，$\tilde o_t = S_t^\top q_t$（状态在右乘）；delta-rule 概念页用 DeltaNet 形式 $S_t \in \mathbb{R}^{d_v \times d_k}$，$V'_i = \phi(Q_i)^\top s_i$（状态在左乘）。两者是 $S_{\text{K3}} = S_{\text{DeltaNet}}^\top$ 的转置约定，机制等价。
- **来源定位**：K3 报告 §2.1.1 Eq.1 "$\tilde o_t = S_t^\top q_t$"；delta-rule 概念页 wiki/delta-rule/index.html §"Delta 规则的紧凑公式与手算"。
- **适用条件**：引用前置页结论时需注意转置。
- **置信状态**：已确认。

## F 类核心公式

### F1：KDA 单头递归（Eq.1）

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top,\qquad \tilde o_t = S_t^\top q_t.$$

- **来源**：K3 报告 §2.1.1 Eq.1。
- **符号**：$S_t \in \mathbb{R}^{d_k \times d_v}$；$q_t, k_t \in \mathbb{R}^{d_k}$；$v_t \in \mathbb{R}^{d_v}$；$\beta_t \in (0,1)$；$\alpha_t \in (0,1)^{d_k}$（逐通道）。
- **推导链**：直接引用，无中间推导。

### F2：参数化（Eq.2）

$$q_t^h, k_t^h = \mathrm{L2Norm}(\mathrm{Swish}(\mathrm{ShortConv}(W_{q/k}^h x_t))) \in \mathbb{R}^{d_k},$$
$$v_t^h = \mathrm{Swish}(\mathrm{ShortConv}(W_v^h x_t)) \in \mathbb{R}^{d_v},\quad \beta_t^h = \mathrm{Sigmoid}(W_\beta^h x_t) \in (0,1),$$
$$z_t^h = W_\alpha^{\uparrow\downarrow} x_t + b_h^\alpha \in \mathbb{R}^{d_k}.$$

- **来源**：K3 报告 §2.1.1 Eq.2。
- **符号**：$W_\alpha^{\uparrow\downarrow}$ 表示上、下两个投影（产出 decay logit $z_t^h$）；$b_h^\alpha$ 是 per-head bias；$A_h$ 在 Eq.5 出现。
- **推导链**：直接引用。

### F3：累积衰减（Eq.3）

$$\gamma_{i\to j}[t] := \prod_{r=i}^{j} \alpha_r[t],\qquad \Gamma_{1\to r}[t] := \gamma_{1\to r}[t].$$

- **来源**：K3 报告 §2.1.1 Eq.3。
- **符号**：$\Gamma[t] \in \mathbb{R}^{C \times d_k}$ 行向堆叠 $\Gamma_{1\to 1}^t, \ldots, \Gamma_{1\to C}^t$。
- **推导链**：直接引用。

### F4：chunkwise 并行形式（Eq.4）

$$A[t] = \mathrm{Tril}\left((Q[t]\odot\Gamma_{1\to C}[t])\,(K[t]/\Gamma_{1\to C}[t])^\top\right),$$
$$O[t] = \underbrace{(\Gamma_{1\to C}[t]\odot Q[t])\,S[t]}_{\text{inter-chunk}} + \underbrace{A[t]\,V_e[t]}_{\text{intra-chunk}},\qquad V_e[t] := U[t] - W[t]S[t].$$

- **来源**：K3 报告 §2.1.1 Eq.4，UT 变换 $V_e = U - WS$ 来自 [63]。
- **符号**：$\mathrm{Tril}$ 保留对角；$S[t]$ 是进入 chunk $t$ 的状态。
- **推导链**：直接引用；UT 推导不展开。

### F5：lower-bounded decay（Eq.5，K3 关键改动）

$$g_t^h = g_{\min}\,\mathrm{Sigmoid}(e^{A_h}\,z_t^h) \in (g_{\min}, 0)^{d_k},\qquad \alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k},$$

对照 Kimi Linear：$g_t^h = -e^{A_h}\,\mathrm{Softplus}(z_t^h) \in (-\infty, 0)^{d_k}$。

- **来源**：K3 报告 §2.1.1 Eq.5 与 Fig.3a。
- **符号**：$A_h$ per-head log-scale；$g_{\min}=-5$ 固定。
- **推导链**：直接引用。

### F6：full-rank output gate（Eq.6）

$$y_t = W_o\left[\mathrm{Sigmoid}(W_g x_t) \odot \mathrm{RMSNorm}(\tilde o_t)\right].$$

- **来源**：K3 报告 §2.1.1 Eq.6。
- **符号**：$W_g$ 满秩；$\odot$ 逐元素乘；RMSNorm 是 head-wise。
- **推导链**：直接引用。

## N 类外部数字

### N1：g_min = -5

- **数值**：$g_{\min} = -5$（固定常数）。
- **来源**：K3 报告 §2.1.1 Eq.5 正文 "$g_{\min}=-5$ is fixed"；config.json `linear_attn_config.gate_lower_bound = -5.0`。
- **实验条件**：K3 配置。

### N2：e^{g_min} ≈ 6.7 × 10^{-3}

- **数值**：$e^{-5} \approx 6.7379 \times 10^{-3}$。
- **来源**：K3 报告 §2.1.1 正文 "every retention factor satisfies $\alpha > e^{-5} \approx 6.7 \times 10^{-3}$"。
- **实验条件**：$g_{\min}=-5$。

### N3：16-token tile 累积 log-decay ∈ (-80, 0)，1/Γ < e^80

- **数值**：累积 log-decay $\in (-80, 0)$；$1/\Gamma < e^{80} \approx 5.54 \times 10^{34}$。
- **来源**：K3 报告 §2.1.1 正文 "the cumulative log-decay over a 16-token tile lies in $(-80, 0)$. The corresponding reciprocal rescaling factor is therefore smaller than $e^{80}$"。
- **实验条件**：$g_{\min}=-5$、tile size = 16（Kimi Linear [63, 140] 的 secondary tiling）。

### N4：69 KDA + 24 Gated MLA = 93 总层

- **数值**：69 + 24 = 93。
- **来源**：config.json `text_config.num_hidden_layers = 93`、`kda_layers`（69 项）、`full_attn_layers`（24 项）。
- **实验条件**：K3-Instruct。

### N5：head_dim=128, num_heads=96, short_conv=4

- **数值**：head_dim=128, num_heads=96, short_conv_kernel_size=4, hidden_size=7168。
- **来源**：config.json `linear_attn_config.head_dim`、`.num_heads`、`.short_conv_kernel_size`；`text_config.hidden_size`。
- **实验条件**：K3-Instruct。

### N6：max_position_embeddings = 1048576 (1M)

- **数值**：1,048,576 = $2^{20}$。
- **来源**：config.json `text_config.max_position_embeddings`。
- **实验条件**：K3-Instruct 配置上限。

## 冲突记录

无冲突。K3 报告 §2.1.1 的公式与 config.json 的数值完全一致（$g_{\min}=-5$ ↔ `gate_lower_bound=-5.0`，full-rank gate ↔ `use_full_rank_gate=true`）。

## 未确认项

无。所有核心论断均有 K3 报告原文或 config.json 数值支持；UT 变换推导指向 [63]，本文按报告指示不展开。
