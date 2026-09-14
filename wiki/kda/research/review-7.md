<!-- review-meta
round: 7
page: wiki/kda/index.html
reviewed_content_sha256: 2515f2ec956300e0
-->
# Kimi Delta Attention（KDA）审查记录（第 7 轮）

- 页面版本：3c77a6d068046de2e3b814bd8126187c09de6e9a（git hash-object wiki/kda/index.html）
- 审查时间：2026-09-14 17:01
- 审查者：独立子代理（未参与写作与前序轮次，仅读本页 index.html / overview.html、外部来源与本规范）
- 已完整阅读章节：核心问题 / 最容易误解 / 1 为什么 K3 需要 KDA / 2 KDA 的递归核心 / 3 K3 的关键改动——lower-bounded decay（含 3.1–3.4）/ 4 参数化与 full-rank output gate（含 4.1）/ 5 chunkwise 并行形式（含 5.1–5.3）/ 6 KDA 在 K3 中的配置与边界（含 6.1–6.4）/ 来源与范围说明（含全部折叠块、图注、表格与构造示例）
- 机械验证：`.dojo/scripts/validate.py wiki/kda/index.html` → `validation ok`；26 组 `<details>/<summary>` 配对；无「（待生成）」占位；正文/标题/列表/表格无 Unicode 数学字符（`$...$` 均由 KaTeX 渲染）；`alt` 为空字符串、无 `$...$`；锚点、站内概念链接（../../wiki/delta-rule/index.html、../../wiki/linear-attention/index.html）与本地资源均存在

## 来源核对（本轮实际打开核对的条目）

- 报告：Kimi K3 Technical Report（arXiv:2607.24653，v1/v2 HTML 正文）§2.1.1、§2.2、§5.1、§5.1.1、§5.1.2、Table 1/§3.2。
  - Eq.1 原文 `S_t = (I − β_t k_t k_tᵀ)·Diag(α_t)·S_{t−1} + β_t k_t v_tᵀ, õ_t = S_tᵀ q_t`，与页面第 2 章 boxed 式逐字一致；`α_t ∈ (0,1)^{d_k}` 为 per-channel one-step retention factor。
  - Eq.2 `z_t^h = W_α^↑W_α^↓ x_t + b_α^h`；q/k 为 `L2Norm(Swish(ShortConv(W·x_t)))`，v 为 `Swish(ShortConv(W_v x_t))`，β 为 `Sigmoid(W_β x_t)` —— 与页面 §4 一致。
  - Eq.3/4 `γ_{i→j}[t]=∏α`、`A[t]=Tril[(Q[t]⊙Γ)(K[t]/Γ)ᵀ]`、`O[t]=(Γ⊙Q)S[t]+A[t]Ṽ[t]`、`Ṽ[t]=U[t]−W[t]S[t]` —— 与页面 §5.1/§5.2 一致。
  - Eq.5 `g_t^h = g_min·Sigmoid(e^{A_h} z_t^h) ∈ (g_min,0)`、`g_min≡−5`、`A_h` 初值 0；Kimi Linear 的 `g_t^h = −e^{A_h}Softplus(z_t^h) ∈ (−∞,0)` —— 与页面 §3 一致。
  - Fig.3a（log-decay 参数化对照）、Fig.3b（对角 tile：Kimi Linear position-pair vs K3 dense Tensor Core）—— 与页面 §3.4/§5.3 一致；原文「cumulative log-decay over a 16-token tile lies in (−80,0)」「reciprocal … below e^80 … within the BF16 dynamic range」与 [N3] 一致。
  - Eq.6 `y_t = W_o[Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t)]`，full-rank 替换 Kimi Linear 的 low-rank —— 与页面 §4.1 一致。
  - §2.1.1 给出「3 KDA + 1 Gated MLA 重复 + 末层 Gated MLA」的 3:1 结构；69/24 计数与 Table 1/§3.2 及 config.json 一致。
  - §2.2 = Attention Residuals，§5.1.1 = KDA kernels、§5.1.2 = KDA Context Parallelism —— 页面 §6.3 的章节指向正确。
- config.json（HuggingFace moonshotai/Kimi-K3）：num_hidden_layers=93、hidden_size=7168、max_position_embeddings=1048576、linear_attn_config{gate_lower_bound=-5.0, head_dim=128, num_heads=96, short_conv_kernel_size=4, use_full_rank_gate=true}；full_attn_layers=4,8,…,88,92,93（24 项）；kda_layers=1–91 中除 4 的倍数外的 69 项 —— 与页面第 6 章表格、层布局、69/24/93 完全一致。
- 参考文献编号：页面全文以 [63] 指代 Kimi Linear。当前 arXiv HTML v1 正文写作 [56]、v2 写作 [57]（bib key 均为 bib.bib115），而媒体引用的报告摘要版本写作「Kimi Delta Attention [63]」——各版本编号存在漂移，无法据此判 [63] 为错，故不作为问题记录。

## 复算（全部通过）

- 1M KV cache：`1,048,576×96×128×2×2 = 5.154×10^10 bytes`；`/2^30 ≈ 48 GB` —— 与页面一致。
- 单头状态 `128×128×2 = 32,768 B ≈ 32 KB`；单层 `×96 ≈ 3 MB` —— 一致。
- `Softplus(1)=ln(1+e)=1.3133`，`g=−1.3133`，`α=e^{−1.3133}=0.2691`；`Sigmoid(1)=0.7311`，`g=−5×0.7311=−3.6553`，`α=e^{−3.6553}=0.02586` —— 与 §3.3 的 0.269 / 0.0259 一致。
- `e^{−5}=6.738×10^{−3}`；`e^{80}≈5.54×10^{34}`；`e^{160}≈3.06×10^{69}`；BF16 max `≈3.39×10^{38}` —— 与 §3.2/§3.3 表格一致。
- 构造示例：`S_1=k_1v_1ᵀ`（全 1 矩阵）、`Diag(α_2)S_1`、`(I−k_2k_2ᵀ)` 只擦第 1 行、`S_2` 与 `õ_2=(3,3,3,3)ᵀ` —— 逐步复算正确；`Γ_{1→3}=(0.35,0.48,0.855,0.12)` 正确。
- 93 层结构：22×(3K+1G)=88，+89-91 K，+92,93 G ⇒ 69 K + 24 G = 93 —— 与 config.json 逐层吻合。

## 问题

- [轻微·可读性] §4.1「本章问题」第 3 题解答（"解释 RMSNorm 为何在门控之前"）：结论句「若归一化放在门控之后，会把门控刚调制出的通道差异重新抹平，门控失去作用」的机制理由不成立。｜引文依据：不适用（该理由不在来源中，K3 报告 §2.1.1 Eq.6 仅给出 `y_t = W_o[Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t)]`，未说明次序动机）。RMSNorm 是对整个向量除以单一 RMS 标量（再乘可学习 per-channel gain）的全局缩放，保留各通道的相对比例，因此不会"抹平"门控 `Sigmoid(W_g x_t)` 的逐通道调制。｜修复要求：删除或改写该不成立的机制陈述——可保留"先把 $\tilde o_t$ 的幅度稳定住，门控的逐通道调制与 $W_o$ 的输入分布才稳定"这一可核对表述，去掉"抹平通道差异"的因果句。｜修复：｜复验：

- [轻微·技术] §5.2「chunkwise 形式的两项」流程图图注：「intra：chunk 内位置 $i$ 到位置 $j$（$i \le j$）的 attention，$\mathrm{Tril}$ 掩码保留对角」给出的因果方向与本页对 Tril 的描述相反，且未定义 $i$、$j$ 谁是 query。｜引文依据：K3 报告 §2.1.1 Eq.4 `A[t]=Tril[(Q[t]⊙Γ_{1→C}[t])(K[t]/Γ_{1→C}[t])ᵀ]`，行索引为 query、列索引为 key，Tril 保留对角及以下（即行索引 ≥ 列索引）；本页同节第 562 行亦写「$\mathrm{Tril}$ 保留对角及以下（因果掩码）」。若 $i$、$j$ 沿用本页行/列约定，条件应为 $i \ge j$。｜修复要求：把图注条件改为与"对角及以下"一致（query 位置 ≥ key 位置），或在括号内写明 $i$、$j$ 谁是 key、谁是 query。｜修复：｜复验：

## 结论

- 处置：可发布（本条 2 条轻微问题不影响正确性与主线理解，核心结论、来源一致性与公式复算全部通过）
- 统计：阻断 0 / 重要 0 / 轻微 2
