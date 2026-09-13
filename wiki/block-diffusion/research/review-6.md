<!-- review-meta
round: 6
page: wiki/block-diffusion/index.html
reviewed_content_sha256: 5faa2931e0c97352
-->
# 块扩散语言模型（Block Diffusion）审查记录（第 6 轮）

- 页面版本：00023d4cc4b46b0cb4f05bfe4b1ef34fa08fb2b8（wiki/block-diffusion/index.html）
- 审查时间：2026-09-13
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题；1. 块间自回归、块内并行——两个范式各让一步；2. block-causal 注意力掩码——一张掩码管两件事；3. KV cache 与灵活长度——从扩散侧保住的东西；4. 块内去噪——从全 mask 到整块 token；5. 当草稿器用——块扩散在 DFlash 里的形态；来源与范围说明（含全部折叠块、伪代码、两幅图与图注）；overview.html

## 来源核对（关键引文）

Arriola et al., ICLR 2025（arXiv:2503.09573v3，取 PDF 全文核对）：
- Eq.(1) `log pθ(x)=Σ_{ℓ=1}^{L} log pθ(x^ℓ|x^{<ℓ})`（§2.1）；Eq.(4) `log pθ(x)=Σ_{b=1}^{B} log pθ(x^b|x^{<b})`，`B=L/L′`（§3.1）；Eq.(6) `−log pθ(x) ≤ L_BD(x;θ):=Σ_b L(x^b,x^{<b};θ)`，原文「itself a valid NELBO」（§3.1）；Eq.(7) `x_logits^b,K^b,V^b ← x_θ^b(x_t^b,K^{1:b−1},V^{1:b−1})`（§3.1）。页 F1–F4 与之一致。
- §3.1「tokens in block b attend to tokens in blocks 1 to b」；「enable us to compute the loss L_BD(x;θ) in parallel for all B blocks」→ 页 §2 掩码规则与「一次前向并行算所有块损失」一致。
- §3.2「denoising x_t^b requires a forward pass on this noisy input, while denoising the next blocks requires running x_θ on the clean version x^b. Thus every block has to go through the model at least twice.」「Each token passes through x_θ twice.」；向量化实现为 `x^noisy ⊕ x` 拼接上一次前向 → 页 §2 折叠块 [C5] 与正文一致。
- §3.2/§6.2「the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked」；「enables us to sample sequences of arbitrary length, whereas diffusion models are restricted to fixed-length generation」→ 页 [C8]、[C4] 一致。
- §4「the diffusion objective suffers from high variance despite being equivalent to the autoregressive likelihood in expectation」→ 页 L′=1 高方差、L′=L 退化全并行扩散一致。
- 附录 C「BD3-LMs and MDLM use T = 5K diffusion steps」；「the number of generation steps does not exceed the sample length L」；「We adopt the first-hitting sampler proposed by Zheng et al. (2024)… the transition probability is the same for all masked tokens for a given t. Thus, the first timestep where a token is unmasked can be analytically sampled」→ 页 §4「T=5K」「first-hitting」「与预测置信度无关」一致。
- Table 7（OWT，Gen. PPL / NFEs，L=1024）：BD3-LMs L′=4 → 25.7 / 1K；SSD-LM L′=25（T=1K 步）→ 37.2 / 40K；SSD-LM L′=25（T=25 步）→ 281.3 / 1K。§6.2「SSD-LM (Han et al., 2022), an alternative block diffusion formulation. Unlike our discrete diffusion framework, SSD-LM uses Gaussian diffusion」；引言「perform Gaussian diffusion over embeddings (Han et al., 2022; 2023)」→ 页 §4 与本章答案的 25.7 / 37.2 / 281.3 / T=1K / T=5K / T=25 / ≥40K NFE / 「高斯嵌入扩散而非离散掩码」全部一致。

DFlash（Chen, Liang, Liu, ICML 2026，arXiv:2602.06036v2，取 HTML 全文核对）：
- §5「use a block size of 16 (10 for LLaMA 3.1)」「draft models use five layers, eight for Qwen3 Coder」；§5.5.4 消融在 b16 与 b8 训练/测试 → 页 [N1]「16（LLaMA-3.1 为 10）、5 层（Coder 为 8），消融含块大小 8」一致。
- §3.2「Autoregressive drafters generate tokens sequentially… Drafting costs therefore grow linearly with the speculation budget γ」「Diffusion drafters generate all γ tokens in parallel within a single forward pass」「For moderate block sizes, T_draft is therefore largely insensitive to γ」→ 页 [C10] 一致。
- §4.1/§4.2「All masked positions within a block are decoded in parallel in a single forward pass」「We randomly sample anchor tokens from the response, use each anchor as the first position of a block」「always conditions on a clean token produced by the target model… the bonus token from the previous verification step」→ 页 [C9]、§5「以干净 token 为块首」「bonus/修正 token 成为下一轮锚点」一致；训练构块属 §4.2，页脚来源行已列出 §4.2。
- §3.2/§4.1「autoregressive drafters are constrained to very shallow architectures」，EAGLE-3 为「a single transformer layer」→ 页「EAGLE 系列的单层草稿网络」一致。
- 页「块扩散似然仍逊于 AR」：[C11] 对应摘要「they lag in likelihood modeling and are limited to fixed-length generation」；Table 7 同规模（均 110M）AR 14.1 vs BD3-LMs 25.7 → 一致。

机械项：`.dojo/scripts/validate.py` 返回 `validation ok: wiki/block-diffusion/index.html`；F1–F4、C1–C11、N1–N2 均在正文使用处与「来源与范围说明」双向对应，无未定义编号；`../causal-mask/`、`../standard-attention/`、`../speculative-decoding/`、`../dflash/`、`../dflash2/` 目标页均真实存在（dflash2 页确为「不放弃并行」的第二代草稿器，与页内表述一致）；无「（待生成）」占位；`alt=""` 仅 lightbox 占位图一处，SVG `aria-label` 内无 `$...$`；KaTeX 定界符成对（正文 308 个 `$`，含奇数 `$` 的行只在 JS 正则中）；代码块外的 Unicode 数学字符已由 validate.py 清零；掩码图 8×8 网格的可见解/遮挡与「块 b 可见块 1..b」规则逐格相符（行 1–4 只见列 1–4，行 5–8 见全部列），图注读数与网格一致；两级问题（核心问题 5 条、每章本章问题 2 条）均有解答折叠块，核心问题答案均指明完整论证所在章节。

## 问题

- [轻微·格式] §1 自回归分解式、符号表与第 1 章答案（第 118、122、171 行）：`x^{<\ell}` 使用原始 `<`，而同页块分解式与模型签名写作 `x^{&lt;b}`（第 132、136 行），同一变量族混用两种 HTML 转义写法；仓库内其余含 `<` 的公式页（mopd/opd/pretraining/sft）统一用 `&lt;`。两者渲染一致，不影响正确性，但降低源码一致性｜引文依据：不适用｜修复要求：把第 118、122、171 行的 `x^{<\ell}` 统一改写为 `x^{&lt;\ell}`，与页内 `x^{&lt;b}` 写法一致｜修复：｜复验：
- [轻微·可读性] §4 第 3 步（第 166 行）：「揭开哪些位置由 noise schedule 决定，与预测置信度无关」，把「揭开哪个位置」归给 noise schedule；同章折叠块「补充：揭开规则、first-hitting 与『已揭开不遮回』」（第 192 行）与本章答案（第 198 行）却写「揭开『哪个 token』由解析采样的随机时间决定」。两处对同一环节归因不同（实际是：揭开的数量由 noise schedule 决定，具体位置由 first-hitting 的随机时间决定）。简化条件②只声明伪代码那行 `S ← noise_schedule(t) 决定本轮要揭开的 block 位置集合` 不代表真实位置分布，未覆盖正文第 3 步｜引文依据：Arriola 附录 C「the transition probability is the same for all masked tokens for a given t. Thus, the first timestep where a token is unmasked can be analytically sampled」——给定 t 时所有被遮 token 转移概率相同，先揭开哪些由解析采样的随机时间决定，schedule 只决定每轮还剩/揭开多少个被遮位置｜修复要求：把第 166 行改为「揭开的数量由 noise schedule 决定、具体位置由 first-hitting 采样器按随机时间决定」，与同章补充和本章答案对齐｜修复：｜复验：

## 结论

- 处置：可发布（本轮无阻断、无重要问题；上述 2 个轻微问题可按每行的修复要求择机处理）
- 统计：阻断 0 / 重要 0 / 轻微 2