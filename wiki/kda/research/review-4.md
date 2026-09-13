<!-- review-meta
round: 4
page: wiki/kda/index.html
reviewed_content_sha256: 6e7ce0cc9ed86a8e
-->
# Kimi Delta Attention（KDA）审查记录（第 4 轮）

- 页面版本：3fe9e6e9d3a6b50427db85570e2639fc6c874b43（wiki/kda/index.html 工作树 hash）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么 K3 需要 KDA——KV cache 爆炸与线性注意力的"记不清" / 2. KDA 的递归核心——delta rule 加 channel-wise forget gate / 3. K3 的关键改动——lower-bounded decay / 4. 参数化与 full-rank output gate——把递归包成可训练的一层 / 5. chunkwise 并行形式——chunk 内并行 + chunk 间递归 / 6. KDA 在 K3 中的配置与边界——69+24 层的 3:1 交替 / 来源与范围说明（含全部折叠块、图表与本章问题解答）

- 来源获取：官方 HuggingFace `moonshotai/Kimi-K3` 的 `config.json`（WebFetch 抓原始 JSON）；Kimi K3 Technical Report §2.1.1（arXiv:2607.24653v2 HTML）；前置页 `wiki/delta-rule/index.html`、`wiki/linear-attention/index.html`；`overview.html`。

## 来源核对记录（本轮实际核对到的原文片段/数值）

- config.json：`num_hidden_layers=93`、`text_config.hidden_size=7168`、`max_position_embeddings=1048576`；`linear_attn_config`: `gate_lower_bound=-5.0`、`head_dim=128`、`num_heads=96`、`short_conv_kernel_size=4`、`use_full_rank_gate=true`；`full_attn_layers=[4,8,…,88,92,93]`（24 项）、`kda_layers=[1,2,3,5,6,7,…,89,90,91]`（69 项）。页面 §6 表格与 §1 的 69/24/93、末尾两层为 Gated MLA（92,93）逐项一致。
- Report §2.1.1 Eq.1：「S_t=(I−β_t k_t k_tᵀ)Diag(α_t)S_{t−1}+β_t k_t v_tᵀ，õ_t=S_tᵀ q_t」——页面 Eq.1 逐字一致，"先衰减、再擦除、再写入"的乘法顺序与原文一致。
- Eq.2：「z_t^h = W_a↑ W_a↓ x_t + b_a^h」，q/k 走 ShortConv→Swish→L2Norm、v 无 L2Norm、β=Sigmoid(W_β x_t)——与页面 §4 一致。
- Eq.3：「γ_{i→j} := ∏_{r=i}^{j} α_r」；Eq.4：「A[t]=Tril[(Q[t]⊙Γ^{1→C})(K[t]/Γ^{1→C})ᵀ]」「O[t]=(Γ^{1→C}⊙Q[t])S[t]+A[t]Ṽ[t]」——与页面 §5.1/§5.2 一致（页面把 Ṽ 写作 V_e 并注明来自 Kimi Linear [64]）。
- Eq.5：「g_t^h = g_min Sigmoid(e^{A_h} z_t^h) ∈ (g_min,0)^{d_k}，α_t^h = exp(g_t^h)」，g_min=−5，「α > e^{−5} ≈ 6.7×10^{−3}」——与页面 §3.2 一致。
- Eq.6：「y_t = W_o[Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t)]」；正文「After applying head-wise RMSNorm to the recurrent output, KDA applies data-dependent output gating」「Kimi K3 changes KDA's output gate from the low-rank parameterization used by Kimi Linear to an input-dependent full-rank projection」——与页面 §4.1（head-wise RMSNorm、low-rank→full-rank）一致。
- Fig.3 正文：「Kimi Linear evaluates each diagonal tile with an explicit position-pair computation, while the bounded range in Kimi K3 allows all causal tiles to use dense Tensor Core matrix multiplications」；「The corresponding reciprocal rescaling factor is therefore smaller than e⁸⁰ and remains within the BF16 dynamic range」；16-token tile 累积 log-decay ∈(−80,0)——与页面 §3.3/§3.4/§5.3 一致。
- 章节编号：报告 §2.2 = Attention Residuals、§5.1.1 = KDA Kernels across Regimes、§5.1.2 = KDA Context Parallelism——与页面 §6.3 引用一致。
- 前置页：`linear-attention` 页 §3 手算与结论确为 $s_i = s_{i-1} + \phi(K_i)V_i^\top$；`delta-rule` 页 §1 标题确为「为什么线性注意力会"记不清"——key 碰撞与 retrieval error」，状态约定确为 $S_t \in \mathbb{R}^{d_v \times d_k}$、读取式 $o_t = S_t q_t$。

## 问题

- [阻断·技术] §1 折叠块「展开：1M KV cache 估算的完整代入与简化」：中间算式与同段结论不符。原文写下「$2^{20} \times 96 \times 128 \times 2 \times 2 = 2^{20} \times 96 \times 512 = 2^{20} \times 49{,}152 \approx 1.07 \times 10^{5} \times 2^{20}$」，紧接着又说「约 $5.15 \times 10^{10} / 2^{30} \approx 48$ GB」。但 $49{,}152 \times 2^{20} = 51{,}539{,}607{,}552 \approx 5.15 \times 10^{10}$，而 $1.07 \times 10^{5} \times 2^{20} = 112{,}197{,}632{,}000 \approx 1.12 \times 10^{11}$，两者差约 2.2 倍。读者按中间式复算会得到约 104–112 GB，与同页 §1 正文/本章问题解答的 48 GB 互相矛盾。｜引文依据：49,152×2²⁰ = 51,539,607,552；1.07×10⁵×2²⁰ = 112,197,632,000；同页 line 162/216 均为 $\approx 5.15 \times 10^{10} \approx 48$ GB。｜修复要求：把该步系数 "$1.07 \times 10^{5}$" 改为 "$4.92 \times 10^{4}$"，或直接改写为「$= 2^{20} \times 49{,}152 = 48 \times 2^{30}$ bytes $= 48$ GB」；改后重新按 48 GB 结论核对。｜修复：｜复验：

- [轻微·技术] §2「约定提醒」（第 252 行）：把 "$V'_i = \phi(Q_i)^\top s_i$" 归为 delta rule 页的写法，但该页通篇没有 $V'$ 记号，其读取式写作 $o_t = S_t q_t$；$V'_i$、$s_i$、$\phi(Q_i)^\top s_i$ 是 linear-attention 页的记号。转置结论（$S_{K3}=S_{DeltaNet}^\top$）本身正确，仅所引公式的出处错页，指向前置页时读者在该页找不到该式。｜引文依据：delta-rule/index.html 中 "V'" 命中 0 次；同页第 133 行读取式为 $o_t = S_t q_t$；linear-attention/index.html 第 385 行出现 $\phi(x_i W_Q)^T s_i$。｜修复要求：删去该括号内的公式，或改用 delta rule 页的实际读取式 $o_t = S_t q_t$ 描述其"状态在左"约定。｜修复：｜复验：

- [轻微·表述] §1–§6 每章末尾过渡段（共 6 处，第 233、329、447、529、624、697 行）：六处使用同一固定模板"[本章结论]。但[下一问题]——下一章[讲/拆开/落到/是]……"，违反 style-guide §8「不使用固定句式，也不为形式完整而添加过渡」。衔接内容本身成立，问题在句式完全同构。｜引文依据：不适用。｜修复要求：保留每章末的一到两句衔接，但改写为不同句式（至少不要 6 处同构）。｜修复：｜复验：

- [轻微·表述] §3.4（第 403、410 行）：把主观评价写成对来源的判断。「这是 K3 改动最直接的工程收益」「K3 报告 Fig.3b 把这个对比画得很清楚」中的"最直接""很清楚"是对来源内容与制图质量的临场评价，报告原文并无此类表述。｜引文依据：报告 §2.1.1 原句为「Kimi Linear evaluates each diagonal tile with an explicit position-pair computation, while the bounded range in Kimi K3 allows all causal tiles to use dense Tensor Core matrix multiplications」，不含"最直接/清楚"类评价。｜修复要求：改为可核对的陈述，例如"这是 K3 报告为 lower-bound 给出的工程动机"、"K3 报告 Fig.3b 对比两个对角 tile 路径"。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复

（本轮其余核对项通过：Eq.1–Eq.6、Fig.3a/3b、§2.1.1/§2.2/§5.1.1/§5.1.2 编号、config.json 全部数值、69/24/93 层账、32 KB/3 MB/48 GB 量级、α∈(e⁻⁵,1)、e⁸⁰≈5.54×10³⁴、e¹⁶⁰≈3×10⁶⁹、Softplus(1)≈1.313→α≈0.269、Sigmoid(1)≈0.731→g=−3.655→α≈0.0259、Γ₁→₂/Γ₁→₃ 手算、递归章 4 维 3 步手算与 DeltaNet 对照，均与来源或自算一致；`description`/`dojo:summary`/`dojo:type=concept`/`dojo:topics`/`dojo:tag` 齐备，overview↔index 互链，无 research/ 路径引用、无"（待生成）"占位，公式全部 KaTeX、结构图为 HTML，折叠与目录锚点正常，`packages/.dojo/scripts/validate.py` 返回 ok。）