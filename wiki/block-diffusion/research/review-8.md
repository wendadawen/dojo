<!-- review-meta
round: 8
page: wiki/block-diffusion/index.html
reviewed_content_sha256: 3f0989723453327d
-->
# 块扩散语言模型（Block Diffusion）审查记录（第 8 轮）

- 页面版本：4d0a2aa6165afc00be76c605e01bfcde7ca5036c（wiki/block-diffusion/index.html 工作树哈希）
- 审查时间：2026-09-14
- 审查者：编排者派发的独立审查者（未参与写作，也未参与前序轮次）
- 适用规范：guides/concept/check.md（页面 dojo:type=concept）
- 已完整阅读章节：标题与元信息；核心问题（5 条，含解答折叠块）；「1. 块间自回归、块内并行——两个范式各让一步」；「2. block-causal 注意力掩码——一张掩码管两件事」（含训练补充折叠块）；「3. KV cache 与灵活长度——从扩散侧保住的东西」；「4. 块内去噪——从全 mask 到整块 token」（含伪代码折叠块、采样补充折叠块）；「5. 当草稿器用——块扩散在 DFlash 里的形态」；「来源与范围说明」全部小节。含所有折叠块。
- 核对所用版本：Arriola et al., Block Diffusion, ICLR 2025（arXiv:2503.09573v3，HTML 全文）；Chen, Liang, Liu, DFlash, ICML 2026（arXiv:2602.06036v2，HTML 全文）。

## 本轮核对依据（关键数值与原文片段）

- 公式编号（F1–F4）：原文 Eq.(1) `log p_θ(x)=∑_{ℓ=1}^{L} log p_θ(x^ℓ∣x^{<ℓ})`；Eq.(4) `log p_θ(x)=∑_{b=1}^{B} log p_θ(x^b∣x^{<b})`；Eq.(6) `−log p_θ(x) ≤ L_BD(x;θ):=∑_{b} L(x^b,x^{<b};θ)`；Eq.(7) `x_logits^b, K^b, V^b ← x_θ^b(x_t^b, K^{1:b−1}, V^{1:b−1})`。与页面 F1–F4 及其来源说明逐一对应，一致。
- C5（每块至少两次前向）：原文「every block has to go through the model at least twice」（§3.2）。与页面正文及训练补充折叠块一致。
- C8（first-hitting、NFE 上界 L）：原文「the first-hitting sampler proposed by Zheng et al. (2024) … (1) the transition probability is independent of the denoising network, (2) the transition probability is the same for all masked tokens for a given t」；§6.2「the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked」。与页面「揭开哪个 token 与预测置信度无关」「NFE 上界为序列长度 L」一致。
- C6（块大小 1）：原文 L′=1 目标「equivalent in expectation to the autoregressive NLL」，但训练方差更高（variance 1.52→0.11）。与页面一致。
- C2（训练掩码）：原文「We design an attention mask for x_noisy ⊕ x such that noisy tokens attend to other noisy tokens in their block and to all clean tokens in preceding blocks」。与页面「带噪块之间互不可见、每个带噪块可见之前块的干净副本」逐字对应。
- N2（数字）：Table 7（OWT，110M，300 样本）：BD3-LM L′=4 → gen PPL 25.7 @1K NFE（L=1024）；SSD-LM L′=25、T=1K → 37.2 @40K NFE；SSD-LM T=25（原文「where NFEs are comparable across methods」）→ 281.3 @1K NFE。页面正文、§4 本章问题答案、来源说明 N2 三处数字与表述完全一致，无互斥。
- 采样 K/V（伪代码）：Algorithm 2 含 `∅, K^b, V^b ← x_θ^b(x^b)`（对干净块单独前向取 K/V）后再 `(K,V) ← (K^{1:b−1}⊕K^b, …)`。与页面「验证的机制」所述一致。
- 掩码图几何（像素核对）：8×8 网格，行 1–4 仅列 1–4 填充（蓝 0.16）、列 5–8 描边无填充；行 5–8 全列填充；块边界矩形 x=82/234/82、w=152/152/304 与图注「两个块内全可见区 + 跨块可见区 + 全遮挡区」一致。
- N1/C9/C10（DFlash）：§5「use a block size of 16 (10 for LLaMA 3.1)」「we set the number of layers to 5 (8 for Qwen3 Coder)」；§5.5.4 消融含块大小 8/16；§4.1「All masked positions within a block are decoded in parallel in a single forward pass」；§3.2「For moderate block sizes, T_draft is therefore largely insensitive to γ」，且「Autoregressive drafters generate tokens sequentially…EAGLE-3 uses a single transformer layer」；§4.2 训练以干净 anchor token 为块首，「directly matches inference-time behavior, where the draft model always conditions on a clean token produced by the target model」(bonus token)。与页面 §5 全部对应。
- 页面功能/格式：`.dojo/scripts/validate.py` 返回 `validation ok`；summary 内 `$L'$`、`$B=L/L'$`、`$B\times$` 为可渲染 LaTeX；目录锚点、内链（causal-mask、standard-attention、speculative-decoding、dflash、dflash2、overview）目标文件均存在；无 alt 含 `$…$`；掩码图为内联 SVG，`.dg-line`/`.dg-accent` 均 `fill:none`，明暗主题可读；正文 Unicode 数学字符仅出现在 `<pre><code>` 伪代码块内（该处非 KaTeX 渲染上下文，属可接受的代码书写）。

## 问题

- [轻微·表述] dojo:summary（head 第 7 行）：summary 末尾句「DFlash 把块内去噪简化为单步并行预测，用作投机解码草稿器。」与同段括号内「DFlash 走极端简化为块内 1 轮去噪」重复陈述同一事实。｜引文依据：不适用（页面内自重复，无需外部来源）｜修复要求：删除或改写其中一处，使 summary 中「DFlash 单步/1 轮简化」只出现一次，其余内容与顺序不变。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（无阻断/重要问题；仅 1 条轻微表述冗余，可修可不修，不影响发布）
