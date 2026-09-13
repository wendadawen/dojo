<!-- review-meta
round: 5
page: wiki/block-diffusion/index.html
reviewed_content_sha256: 7ed50827161ce874
-->
# 块扩散语言模型（Block Diffusion）审查记录（第 5 轮）

- 页面版本：5fa7df881e79c4f885bb9bf0b5ef2f0c963e8149
- 审查时间：2026-09-13 21:45
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题（5 题）；1. 块间自回归、块内并行——两个范式各让一步（含本章问题）；2. block-causal 注意力掩码——一张掩码管两件事（含补充折叠块、本章问题）；3. KV cache 与灵活长度——从扩散侧保住的东西（含本章问题）；4. 块内去噪——从全 mask 到整块 token（含代码折叠块、补充折叠块、本章问题）；5. 当草稿器用——块扩散在 DFlash 里的形态（含本章问题）；来源与范围说明

## 来源核对（关键条目，含原文片段）

- F1=Eq(1)、F2=Eq(4)、F3=Eq(7)、F4=Eq(6)：arXiv:2503.09573v3 逐条对上——Eq(1)「log p_θ(x)=Σ_{ℓ=1}^{L} log p_θ(x^ℓ|x^{<ℓ})」；Eq(4)「=Σ_{b=1}^{B} log p_θ(x^b|x^{<b})」；Eq(6)「−log p_θ(x) ≤ L_BD(x;θ):=Σ_{b=1}^{B} L(x^b,x^{<b};θ)」；Eq(7)「x_logits^b, K^b, V^b ← x_θ^b(x_t^b, K^{1:b−1}, V^{1:b−1})」。
- C2：原文「tokens in block b attend to tokens in blocks 1 to b」；掩码 M_BC 定义「if j belongs in the same block as i, or a block before i」——与本页「块 b 可见块 1..b，跨块因果、块内双向」一致；矢量训练「one forward pass on the concatenation x_noisy ⊕ x」支撑「一次前向并行算所有块」。
- C5：原文「Thus every block has to go through the model at least twice.」
- C6：原文「training is hampered by the high variance of the gradients of the diffusion objective… under-perform autoregression even with a block size of one」。
- C8：原文「the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked」；附录 C.5「We adopt the first-hitting sampler proposed by Zheng et al. (2024)… (1) the transition probability is independent of the denoising network, (2) the transition probability is the same for all masked tokens for a given t」——支撑「揭开与预测置信度无关、由解析采样决定」。
- N2：Table 7 逐行对上——「BD3-LMs L′=4 25.7 1K」「L′=8 30.4 1K」「L′=16 33.4 1K」「SSD-LM L′=25 37.2 40K」「(low-NFE) 281.3 1K」；§6.2「In Table 7, BD3-LMs and MDLM use T=5K diffusion steps」「For SSD-LM, we compare sample quality using T=1K diffusion steps per block, matching their experimental setting (yielding ≥40K NFEs), and T=25 where NFEs are comparable」——正文 420 行与 440 行的 25.7 / 37.2 / 281.3 与 T=5K / T=1K / T=25 全部对上，二者不矛盾。
- C9/C10/N1：DFlash（arXiv:2602.06036v2）§3.2「T_draft = t_parallel」「making t_parallel ≪ γ·t_step for models of comparable size」「For moderate block sizes, T_draft is therefore largely insensitive to γ」；§4.1「All masked positions within a block are decoded in parallel in a single forward pass」；§5「a block size of 16 (10 for LLaMA 3.1)」「the number of layers to 5 (8 for Qwen3 Coder)」；§5.5.4 消融块大小 8；§4.2「use each anchor as the first position of a block, and mask the remaining positions」「the draft model always conditions on a clean token produced by the target model」；§3.1 含 bonus token——第 5 章各论断均对上。
- 机械项：dojo:topics「训练与优化」在允许大类内；validate.py 返回成功；前置概念链接（causal-mask / standard-attention / speculative-decoding）与 dflash / dflash2 页面均存在；无「（待生成）」占位；SVG/alt 内无 $...$；`×`、`⊕` 仅出现在 `<pre>` 代码块（按规范豁免）；问题块 6 组、解答 15 个、details 前缀（补充／代码）均合规；两级问题答案均指向所在章节。

## 问题

- [阻断·技术] `<meta dojo:summary>` 与「1.」表格、构造示例、本章问题答案：同一指标「串行前向次数」在页面内给出互相矛盾的数值。summary 写「串行前向次数从 $L$ 降到 $B=L/L'$」，核心问题 1 答案写「串行步从 $L$ 次降到 $B=L/L'$ 次」，而同一句又承认「BD3LM 一般需多步去噪」；「1.」表格「串行前向次数」行给块扩散为「$B\times$（每块去噪轮数）」，「1.」本章问题 2 答案为「降到 $B=L/L'$ 次乘以每块轮数」，构造示例对同一 8 token/2 块/每块 2 轮配置算出 4 次。即同一术语并存 2 与 4 两个值，且 summary 句内自相矛盾（既说多步去噪又说到 $B$）。｜引文依据：页面第 7 行「…DFlash 走极端简化为 1 步；BD3LM 一般需多步去噪），串行前向次数从 $L$ 降到 $B=L/L'$」；第 80 行「串行步从 $L$ 次降到 $B=L/L'$ 次」；第 147 行「$B\times$（每块去噪轮数）」；第 160 行「每块去噪 2 轮：$2\times2=4$ 次串行前向」；第 177 行「降到 $B=L/L'$ 次乘以每块轮数」。来源侧：Arriola §3.2「every block has to go through the model at least twice」。｜修复要求：统一「串行前向次数」的计法，三处用词一致——或把 summary 与核心问题 1 答案改为与表格/构造示例一致（各块去噪轮数之和），或明确区分「串行步数 $B$」与「串行前向次数 $B\times$每块轮数」两个量并各自命名。｜修复：｜复验：

- [重要·技术] 「4.」正文（第 420 行）与「4.」本章问题答案（第 440 行）：BD3LM 的 gen PPL 25.7 未给出成立条件——对应块大小 $L'=4$ 与数据集 OWT 均未出现，读者会以为任意配置的 BD3LM 都达到 25.7（实际上 $L'=8$ 为 30.4、$L'=16$ 为 33.4）。｜引文依据：Arriola Table 7「BD3-LMs L′=16 33.4 1K … L′=8 30.4 1K … L′=4 25.7 1K」，表题「All models are trained on OWT」。｜修复要求：在第 420 行与第 440 行补出块大小 $L'=4$ 与数据集（OWT）。｜修复：｜复验：

- [重要·技术] 「1.」章节表格「串行前向次数」行、全并行掩码扩散单元格（第 147 行）：单元格写「去噪轮数（与 $L$ 无关的固定长度）」，与该行指标及来源数据不符。掩码扩散的实际串行前向次数由 NFE 决定，而 Table 7 中 MDLM 的 NFE 随 $L$ 增长（$L=1024$ 为 1K，$L=2048$ 为 2K），论文亦明确 NFE 上界为 $L$；「与 $L$ 无关」把配置的扩散步数 $T$ 当成了实际串行前向次数。另「固定长度」与「生成长度」行的「固定」重复，使单元格语义含混。｜引文依据：Arriola Table 7「MDLM 46.8 1K ／ 41.3 2K」；§6.2「the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked」。｜修复要求：把该单元格改为与来源一致（如「去噪轮数（实际 NFE 上界为 $L$）」），并删去与「生成长度」行重复的「固定长度」。｜修复：｜复验：

- [轻微·表述] 开篇 callout（第 66 行）：callout 把块扩散的折中概括为「块内 4 个位置一次并行写出」，把 DFlash 的单步极端写成块扩散的一般形态；与「1.」表格（块内需多轮去噪）及「4.」（同一 8 token/2 块例子用 2 轮）不一致。｜引文依据：不适用。｜修复要求：开篇不预设块内单步——改为「块内 4 个位置并行处理」，把「一次」限定到 DFlash 的单步形态。｜修复：｜复验：

- [轻微·表述] 「5.」正文（第 447、493 行）与「来源与范围说明·辅助解释与类比边界」（第 520 行）：「辅助解释与类比边界」声明「无其他类比」，但正文使用未登记的类比/口语化措辞「换个岗位，短板可以不成问题」「在这个岗位上」。｜引文依据：不适用。｜修复要求：或在「辅助解释与类比边界」登记该类比的作用与边界，或把第 447、493 行改为直述。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 2
- 处置：修复
