<!-- review-meta
round: 6
page: wiki/kda/index.html
reviewed_content_sha256: 2c0f535fa7703748
-->
# Kimi Delta Attention（KDA）审查记录（第 6 轮）

- 页面版本：6d87d5aaf5805907ff40da0c148cff6c6161aaa8（wiki/kda/index.html）
- 审查时间：2026-09-13 21:12
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、最容易误解、1. 为什么 K3 需要 KDA、2. KDA 的递归核心、3. K3 的关键改动——lower-bounded decay（3.1–3.4）、4. 参数化与 full-rank output gate（4.1）、5. chunkwise 并行形式（5.1–5.3）、6. KDA 在 K3 中的配置与边界（6.1–6.4）、来源与范围说明；并对照 wiki/kda/overview.html。

来源获取方式：Kimi K3 Technical Report §2.1.1（arXiv:2607.24653，HTML 版逐节核对 Eq.1–Eq.6、Fig.3）、HuggingFace 官方 moonshotai/Kimi-K3 的 config.json（raw 抓取，含 linear_attn_config 全字段）、外部 KDA 实现分析（sqliteai/waste docs/KDA.md）、以及 §5.1.1/§5.1.2 与 §2.2 的公开转述。

## 已核对（关键数值与公式，全部一致）

- config.json 逐项核对（raw 原文）：num_hidden_layers=93、hidden_size=7168、max_position_embeddings=1048576、linear_attn_config{head_dim=128, num_heads=96, short_conv_kernel_size=4, use_full_rank_gate=true, gate_lower_bound=-5.0}；kda_layers 列表实为 69 项（1,2,3,5,6,7,…,89,90,91），full_attn_layers 实为 24 项（4,8,…,88,92,93）。页面 6.2 表格与 6.1 层编号、以及 116/140/633/634 行全部与之一致。
- 层布局自洽：22 个完整 3:1 块（层 1–88）= 66 KDA，加 89–91 三层 KDA = 69；MLA = 22 + 92,93 = 24；69+24=93。第 179–206 行流程图与该账一致。
- 递归式：报告 Eq.1 原文 "S_t=(I−β_t k_t k_t^T)Diag(α_t)S_{t−1}+β_t k_t v_t^T, õ_t=S_t^T q_t"，与页面 239 行 boxed 公式逐字一致；独立分析文档给出的等价形式 S'=Diag(exp(g_t))S_{t−1}, S_t=S'+β_t k_t(v_t−S'^T k_t)^T 与该式代数等价，页面 277 行的「先衰减、再擦除、后写入」顺序论断成立。
- 映射式：报告 Eq.5 "g_t^h=g_min·Sigmoid(e^{A_h}z_t^h)∈(g_min,0)^{d_k}, α_t^h=exp(g_t^h)∈(e^{g_min},1)^{d_k}"，与页面 354 行一致；Kimi Linear 侧 g=−e^{A_h}Softplus(z) 与页面 346 行一致。A_h 初始化 0（报告："We initialize A_h = 0"）与页面 356 行一致。
- Eq.2/Eq.3/Eq.4/Eq.6 与页面 453–457、539、557–559、496 行一致（ShortConv→Swish→L2Norm 的 q/k、low-rank W_α^{↑↓}+b_h^α、γ_{i→j}、inter/intra-chunk 与 Tril、y=W_o[Sigmoid(W_g x_t)⊙RMSNorm(õ_t)]）。
- 数值全部可复算：1048576×96×128×2×2=5.15×10^10 B≈48 GiB；128×128×2≈32 KB、×96≈3 MB；Softplus(1)=ln(1+e)≈1.313→α≈0.269；Sigmoid(1)≈0.731→g=−3.655→α≈0.0259；e^{−5}≈6.7×10^−3；e^{80}≈5.54×10^34 < 3.4×10^38(BF16)；e^{160}≈3×10^69。手算例：S_1、Diag(α_2)S_1、擦除后矩阵、S_2、õ_2=(3,3,3,3)^T，以及 Γ_{1→1/2/3}=(1,1,1,1)/(0.5,0.8,0.9,0.3)/(0.35,0.48,0.855,0.12) 全部逐步复算正确。
- 侧记核对：FlashKDA 为 CUTLASS-based chunkwise kernel、覆盖 intra/inter-chunk 重叠（§5.1.1），KCP=KDA Context Parallelism 做 prefix scan + All-Gather（§5.1.2），Attention Residuals 属 §2.2——页面 600/663/664/665 行的归属与章节号均正确。
- 「位置敏感 + 近因」与 KDA 用 ShortConv(kernel=4)+per-channel decay 承载位置信息的公开描述一致（模型整体 NoPE），非无据论断。
- 机械项：页面仅引用 wiki/linear-attention/index.html 与 wiki/delta-rule/index.html，二者均真实存在、无占位；无 img alt 含 $...$；除流程图箭头（结构字符）外无 Unicode 数学字符；dojo:topics=注意力机制、dojo:tag=注意力 均在 ALLOWED_TOPICS/ALLOWED_TAGS 内；overview 与 index 双向互链；`python3 .dojo/scripts/validate.py wiki/kda/index.html`（及 overview）返回 validation ok。页面「简化条件」明确声明不提供可运行代码，故无代码执行项。

## 问题

- [轻微·格式] 来源与范围说明 [C1]、[C10]：两条来源条目在正文中没有任何 `<sup>` 上标引用，与 style-guide §6「正文引用与来源章节双向对应」不符｜引文依据：正文上标全集为 [N6][N5][F1, C2][F2][C4][C3, F5][N1][N2][N3][C5][C6, F6][C9][C7][F3][F4][C8, N4]，无 [C1]/[C10]；[C1]、[C10] 仅出现在第 703、712 行的定义处（其余 C2–C9、F1–F6、N1–N6 均在正文有引用）｜修复要求：在对应论断处补上标——[C1] 补在第 108 行「KDA（Kimi Delta Attention）——delta rule 加每通道遗忘门、并限定衰减下界的线性注意力变体」处，[C10] 补在第 252 行「约定提醒」的转置句「两者是 S_K3=S_DeltaNet^T 的转置，机制等价」处；若某条不再单独成立则删除该定义条目｜修复：｜复验：

- [轻微·表述] 2 章开头（第 241 行，同段第 254 行同类）：出现以篇章编排为主语的元话语引导句｜引文依据：「先说清每个符号，再分三步读。」与「公式分三步读，顺序由矩阵乘法的结合律决定：」｜修复要求：删除第 241 行的编排式引导句或改为直陈（直接进入符号定义列表）；第 254 行若保留须确保「结合律决定作用顺序」这一实质信息不被削弱｜修复：｜复验：

## 结论

- 处置：无阻断、无重要问题；上述 2 项轻微按「修复」处理，修复后即可发布
- 统计：阻断 0 / 重要 0 / 轻微 2

> 本轮所列问题的处理结果见 `minor-fixes.md`。
