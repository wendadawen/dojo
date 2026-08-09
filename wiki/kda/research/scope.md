# scope.md：KDA 内容范围

## 1. 概念歧义处理

**概念名称**：Kimi Delta Attention（KDA）

**歧义检查**："Delta Attention" 在不同语境下可指：(a) Schlag et al. 2021 / Yang et al. 2024 的 DeltaNet delta-rule 注意力；(b) Kimi Linear（[63]）在 DeltaNet 基础上加 channel-wise forget gate 的版本；(c) Kimi K3 在 Kimi Linear 基础上改用 lower-bounded decay 与 full-rank output gate 的版本。

**裁定状态**：已裁定。本文采用 (c)，即 Kimi K3 技术报告 §2.1.1 定义的 KDA。依据：K3 报告 §2.1.1 明确说 "KDA extends the delta-rule recurrence [105, 138] with a channel-wise forget gate [63]"，并在 Eq.5、Eq.6 给出与 Kimi Linear 不同的 decay 参数化与 output gate。当文中需要对比 (b) 时明确称"Kimi Linear 原版"，引用 [63]。

## 2.1 概念含义

- **概念名称**：Kimi Delta Attention
- **英文名称**：Kimi Delta Attention
- **常见缩写**：KDA
- **一句话定义**：KDA 是 Kimi K3 用的一种线性注意力变体，把 delta rule 的"先擦后写"递归加上每通道独立衰减的遗忘门，并用有下界的衰减参数化让长序列训练在 BF16 上数值稳定且能用 Tensor Core 加速。
- **正式定义**（与 K3 报告 §2.1.1 Eq.1, Eq.5, Eq.6 一致）：单头递归
  $S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$，$\tilde o_t = S_t^\top q_t$；
  其中 $\alpha_t = \exp(g_t)$，$g_t = g_{\min}\,\mathrm{Sigmoid}(e^{A_h} z_t) \in (g_{\min}, 0)$，$g_{\min}=-5$；
  输出 $y_t = W_o[\mathrm{Sigmoid}(W_g x_t) \odot \mathrm{RMSNorm}(\tilde o_t)]$。
- **本文采用的语境**：作为 Kimi K3 的长序列混合层，与 Gated MLA 以 3:1 交替。重点讲机制与数值稳定性改动，不展开 K3 的系统实现（FlashKDA、KCP）。

### 包括什么

1. **delta rule 复用与扩展**：KDA 把 delta rule 的擦除项 $I - \beta_t k_t k_t^\top$ 与 channel-wise forget gate $\mathrm{Diag}(\alpha_t)$ 组合。属于 KDA 核心，因为这是 K3 报告 §2.1.1 开篇定义。
2. **channel-wise forget gate**：$\alpha_t \in (0,1)^{d_k}$ 每通道一个衰减率。属于 KDA，是 KDA 相对 DeltaNet 的本质增量。
3. **lower-bounded decay（K3 改动）**：$g = g_{\min}\mathrm{Sigmoid}(e^{A_h} z) \in (g_{\min}, 0)$，$g_{\min}=-5$。属于 KDA，是 K3 相对 Kimi Linear 的关键改动。
4. **full-rank output gate（K3 改动）**：$y = W_o[\mathrm{Sigmoid}(W_g x) \odot \mathrm{RMSNorm}(\tilde o)]$，$W_g$ 满秩。属于 KDA，是 K3 相对 Kimi Linear 的另一关键改动。
5. **参数化**：ShortConv + Swish + L2Norm 的 q/k/v/β/z 投影链（K3 报告 Eq.2）。属于 KDA，因为参数化决定了 decay logit 怎么来、q/k 怎么归一化。
6. **chunkwise 并行形式**：Eq.3、Eq.4 的累积衰减 $\Gamma$、Tril 掩码、UT 变换。属于 KDA，因为这是 KDA 可训练的执行形式，也是 lower-bounded decay 产生数值收益的舞台。
7. **K3 中的层配置**：69 层 KDA + 24 层 Gated MLA、3:1 混合、$g_{\min}=-5$、head_dim=128、num_heads=96、short_conv=4、max_position=1M。属于 KDA，因为这是从 config.json 与报告 §2.1 直接读到的实例化数值。

### 不包括什么

1. **Gated MLA 机制**：MLA 的 KV 压缩、NoPE、content key 重建。排除理由：K3 报告 §2.1.2 单独讲 MLA，是相邻而非本概念；KDA 只提供"位置敏感 + 近因"的序列混合，MLA 提供"全局内容交互"，职责分离。
2. **FlashKDA / KCP 系统实现**：CUTLASS kernel、SM 级 CP、prefix scan。排除理由：K3 报告 §5.1 是工程实现，影响的是吞吐与显存而非概念机制；本文只在边界处说明 lower-bounded decay 的工程收益。
3. **Attention Residuals（AttnRes）**：K3 的深度方向残差。排除理由：K3 报告 §2.2 独立概念，与 KDA 的序列方向混合正交。
4. **MoE 路由**：K3 的 896 专家、sigmoid 路由。排除理由：与注意力机制无关。
5. **训练超参与对齐数据**：排除理由：工程细节，不影响概念理解。
6. **GDN / Mamba-2 的 negative-softplus 推导**：只在对比时引用结论，不展开 GDN 内部。排除理由：属于另一独立概念。

### 相邻概念

- **DeltaNet / delta rule**：已有概念页 `wiki/delta-rule/index.html`。关键区别：DeltaNet 的递归是 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，纯擦写无显式遗忘；KDA 在擦写前再乘 $\mathrm{Diag}(\alpha_t)$ 做通道级衰减。约定差异：DeltaNet 页用 $S \in \mathbb{R}^{d_v \times d_k}$（状态在左），K3 报告 KDA 用 $S \in \mathbb{R}^{d_k \times d_v}$（状态在右，$\tilde o = S^\top q$），两者是转置约定，机制等价。本文按 K3 报告约定，引用 delta-rule 页时不重讲擦除-写入。
- **线性注意力**：已有概念页 `wiki/linear-attention/index.html`。关键区别：vanilla 线性注意力是加性累加 $S_t = S_{t-1} + \phi(k_t)v_t^\top$，无遗忘也无擦除；KDA 在其上加了"先衰减、再擦写"两步。本文引用线性注意力页的"因果递归状态"结论，不重讲核函数与结合律。
- **Gated Linear Attention（GLA）/ Kimi Linear [63]**：KDA 的直接前身。关键区别：Kimi Linear 用 negative-softplus 把 log-decay 映射到 $(-\infty, 0)$、用 low-rank output gate；K3 改为 scaled sigmoid 映射到 $(g_{\min}, 0)$、用 full-rank output gate。本文不生成 Kimi Linear 独立页，只在对比中引用 [63]。
- **GDN / Mamba-2 [24, 138]**：delta rule + negative-softplus 的更早出处。本文不展开，只在 §3 对比时点名。

## 2.2 学习目标

### Q1：KDA 在 Kimi K3 里做什么、解决什么问题？

- **完成答案**：读者应能说明 KDA 是 K3 长序列混合的主力层（69/93 层），用固定大小的递归状态 $S \in \mathbb{R}^{d_k \times d_v}$ 替代 softmax 的 KV cache，在 1M token 上下文下避免 KV 随序列线性增长；同时通过 delta rule 的"先擦后写"和 channel-wise forget gate 解决 vanilla 线性注意力的 key 碰撞与"记不清旧 token"问题。
- **为什么是核心**：不回答它就不知道 KDA 为何存在、为何不直接用 softmax 或 vanilla 线性注意力。
- **依赖内容**：softmax 的 $O(N^2)$ 与 KV cache 增长、线性注意力的固定状态与 key 碰撞、delta rule 的擦写。

### Q2：channel-wise forget gate 怎么改写 delta rule 的递归？

- **完成答案**：读者应能从 KDA 递归 $S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top$ 说出三件事：(1) 先把旧状态 $S_{t-1}$ 按通道乘 $\alpha_{t,j}$ 衰减（$\alpha_{t,j} \in (0,1)$，每通道独立）；(2) 再按 delta rule 用 $I - \beta_t k_t k_t^\top$ 在 $k_t$ 方向擦除；(3) 最后写入 $\beta_t k_t v_t^\top$。能解释为什么 forget gate 在 delta 更新之前而不是之后——擦除和写入都作用于"已经衰减过的旧状态"，保留"先遗忘再擦写"的顺序。
- **为什么是核心**：这是 KDA 相对 DeltaNet 的本质增量，不理解就分不清 KDA 与 DeltaNet。
- **依赖内容**：delta rule 公式、$\mathrm{Diag}(\alpha_t)$ 的通道语义、$\beta_t$ 的写入强度。

### Q3：K3 的 lower-bounded decay 与 Kimi Linear 的 negative-softplus 有什么区别，为什么这么改？

- **完成答案**：读者应能说明：(1) Kimi Linear 用 $g = -e^{A_h}\mathrm{Softplus}(z) \in (-\infty, 0)$，log-decay 无下界；(2) K3 用 $g = g_{\min}\mathrm{Sigmoid}(e^{A_h} z) \in (g_{\min}, 0)$，$g_{\min}=-5$ 固定；(3) 因此 $\alpha = e^g \in (e^{-5}, 1) \approx (6.7\times10^{-3}, 1)$；(4) 16-token tile 的累积 log-decay $\in (-80, 0)$，倒数 $1/\Gamma < e^{80}$，落在 BF16 动态范围内；(5) 于是 chunkwise 形式里对角 tile 也能用 dense Tensor Core 矩阵乘，消除 Kimi Linear 的 position-pair diagonal 路径。
- **为什么是核心**：这是 K3 相对 Kimi Linear 的关键改动，是任务明确要求讲清的点。
- **依赖内容**：chunkwise 形式里 $1/\Gamma$ 的来源、BF16 动态范围、Tensor Core 对 dense matmul 的要求。

### Q4：full-rank output gate 和 chunkwise 并行形式怎么把递归变成可训练的一层？

- **完成答案**：读者应能说明：(1) chunkwise 形式把序列切成大小 $C$ 的 chunk，chunk 内用 $\Gamma$、Tril、$V_e = U - WS$ 做并行 matmul（Eq.4），chunk 间递归传 $S$；(2) full-rank gate $y = W_o[\mathrm{Sigmoid}(W_g x) \odot \mathrm{RMSNorm}(\tilde o)]$ 让每个 token 按输入 $x_t$ 调制 $\tilde o_t$ 的每个通道，$W_g$ 满秩意味着门控维度等于输出维度；(3) ShortConv + Swish + L2Norm 的 q/k 链让 $k_t$ 近单位范数，使 $I - \beta_t k_t k_t^\top$ 接近正交投影。
- **为什么是核心**：递归形式本身不可训练（串行），必须理解并行化与门控才能理解 KDA 为何能扩展到 1M token。
- **依赖内容**：UT 变换、累积衰减 $\Gamma$、RMSNorm、Swish、ShortConv。

### Q5：KDA 在 K3 中的工程配置与边界是什么？

- **完成答案**：读者应能列出：69 层 KDA + 24 层 Gated MLA（3:1 混合，每 4 层一组，最后两层均为 Gated MLA），$g_{\min}=-5$，head_dim=128，num_heads=96，short_conv_kernel_size=4，use_full_rank_gate=true，max_position_embeddings=1048576（1M）。并说明 KDA 不解决什么：全局内容交互由 Gated MLA 负责，KDA 只负责"位置敏感 + 近因"的序列混合；KDA 不等于 K3 全部注意力机制。
- **为什么是核心**：把抽象机制落到具体数值，避免读者把 KDA 误解为 K3 的全部注意力。
- **依赖内容**：config.json 数值、K3 报告 §2.1 的 3:1 描述。

## 2.3 内容分级

### 核心内容（缺则学习目标无法回答）

| 内容 | 对应学习目标 | 必须讲清的结论 |
|---|---|---|
| KDA 在 K3 的角色与动机 | Q1 | 固定状态替代 KV cache、长序列混合、3:1 交替 |
| delta rule + channel-wise forget gate 的递归 | Q2 | $\mathrm{Diag}(\alpha_t)$ 在 delta 更新前先衰减，Eq.1 |
| lower-bounded decay（scaled sigmoid vs negative-softplus） | Q3 | $g \in (g_{\min}, 0)$、$g_{\min}=-5$、BF16 安全、对角 tile 用 Tensor Core |
| 参数化（ShortConv + Swish + L2Norm + β + z） | Q2, Q4 | q/k 近单位范数、decay logit 怎么来 |
| full-rank output gate | Q4 | $W_g$ 满秩、RMSNorm 在门控前 |
| chunkwise 并行形式 | Q4 | $\Gamma$、Tril、$V_e$、inter/intra-chunk 分工 |
| K3 配置与边界 | Q5 | 69+24、3:1、$g_{\min}=-5$、head_dim/heads/short_conv 数值 |

### 辅助内容（消除关键理解障碍）

| 内容 | 服务的核心内容或误解 |
|---|---|
| 状态约定（$S \in \mathbb{R}^{d_k \times d_v}$，状态在右）与 delta-rule 页（状态在左）的转置关系 | 避免读者把 K3 公式与前置页公式误判为矛盾 |
| vanilla 线性注意力 key 碰撞回顾 | 证明 forget gate 必要性 |
| GDN/Mamba-2 的 negative-softplus 出场顺序 | 理解 K3 改动的来源 |
| $1/\Gamma$ 在 chunkwise 形式里出现的具体位置 | 理解 lower-bound 为何直接关联对角 tile |

### 扩展内容

| 内容 | 纳入/排除 |
|---|---|
| FlashKDA / KCP 系统实现 | 排除（属 §5.1 系统设计，独立概念） |
| Kimi Linear 的完整 UT 变换推导 | 排除（K3 报告明示 "refer to Kimi Linear [63]"） |
| KDA 与 Gated MLA 的注意力残差耦合 | 排除（属 §2.2 AttnRes，独立概念） |
| BF16 动态范围的技术细节 | 纳入但只取结论（$1/\Gamma < e^{80}$ 在范围内） |

## 2.4 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页链接 | 递归深度 |
|---|---|---|---|
| delta rule / DeltaNet | Q2, Q3 | `wiki/delta-rule/index.html`（已有） | 0 |
| 线性注意力（核函数 + 因果递归状态） | Q1, Q4 | `wiki/linear-attention/index.html`（已有） | 0 |
| sigmoid / softplus / RMSNorm / Swish | Q3, Q4 | 不生成（基础深度学习构件，超出 concept 流程的递归范围，正文首次出现时用一句话点名） | — |

两条已有概念页在正文首次依赖时给出链接，不内联重复讲解。

## 2.5 明确不展开的内容

1. **Gated MLA 的 KV 压缩与 NoPE**：与 KDA 正交（MLA 负责全局内容，KDA 负责位置敏感混合），属另一独立概念，K3 报告 §2.1.2 单独讲。
2. **FlashKDA 的 CUTLASS kernel 与 SM 级 CP**：工程实现，不影响概念机制；属 K3 报告 §5.1 独立章节。
3. **KCP 的 prefix scan 与 all-gather**：跨设备通信，不影响单层 KDA 的数学形式。
4. **UT 变换的完整推导**：K3 报告明示 "refer to Kimi Linear [63]"，本文只引用 $V_e = U - WS$ 的结论与职责。
5. **K3 的训练超参与对齐数据**：不影响概念理解。
6. **BF16 浮点格式的位级细节**：只取"动态范围足够 $e^{80}$"的结论。

## 2.6 常见误解和适用边界

### 误解 1：KDA 就是 DeltaNet

- **错误理解**：KDA = DeltaNet，只是换个名字。
- **正确结论**：KDA = DeltaNet delta rule + channel-wise forget gate（$\mathrm{Diag}(\alpha_t)$）+ lower-bounded decay（K3 改）+ full-rank output gate（K3 改）。forget gate 和 lower-bound 是 KDA 相对 DeltaNet 的本质增量。
- **形成原因**：名字里都有 "Delta"，且都基于 $I - \beta_t k_t k_t^\top$ 的擦除项。
- **影响**：Q2、Q3。

### 误解 2：K3 的 lower-bounded decay 和 Kimi Linear 的 negative-softplus 等价

- **错误理解**：两者都把 log-decay 限到负值，效果一样。
- **正确结论**：negative-softplus 的 $g \in (-\infty, 0)$ 无下界，单个 $\alpha$ 可以任意接近 0；scaled sigmoid 的 $g \in (g_{\min}, 0) = (-5, 0)$，$\alpha > e^{-5} \approx 0.0067$。这个下界保证 16-token tile 的 $1/\Gamma < e^{80}$，使对角 tile 也能用 dense Tensor Core。Kimi Linear 不行，必须用 position-pair 计算。
- **形成原因**：两者都输出 $\alpha \in (0,1)$，表面看相同。
- **影响**：Q3。

### 误解 3：forget gate 在 delta 更新之后

- **错误理解**：先 delta 擦写，再用 $\alpha$ 衰减。
- **正确结论**：K3 报告 Eq.1 的顺序是 $S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$，$\mathrm{Diag}(\alpha_t)$ 先作用于 $S_{t-1}$，再 delta 更新。即"先衰减旧状态、再擦写"。
- **形成原因**：矩阵乘法顺序在公式里不显眼。
- **影响**：Q2。

### 误解 4：KDA = K3 的全部注意力

- **错误理解**：K3 只用 KDA 做注意力。
- **正确结论**：K3 用 69 层 KDA + 24 层 Gated MLA 交替（3:1），KDA 负责位置敏感 + 近因混合，MLA 负责全局内容交互。最后两层（92、93）均为 Gated MLA。
- **形成原因**：任务背景强调 KDA 是 K3 核心。
- **影响**：Q5。

### 适用边界

- **KDA 解决**：固定状态长序列混合、key 碰撞（相对 vanilla 线性注意力）、数值稳定的 BF16 训练（相对 Kimi Linear）、对角 tile 的 Tensor Core 加速。
- **KDA 不解决**：全局 token-to-token 内容交互（由 Gated MLA 负责）、跨设备状态同步（由 KCP 负责）、kernel 调度（由 FlashKDA 负责）。
- **结论成立条件**：$g_{\min}=-5$、chunk size $C=16$（tile 划分）、BF16 训练。若 $g_{\min}$ 更小或 chunk 更大，$1/\Gamma$ 可能溢出 BF16；若用 FP32 训练，negative-softplus 也能用但代价更高。
- **条件不满足时**：lower-bound 的数值收益消失，退回 Kimi Linear 的 position-pair diagonal 路径。
