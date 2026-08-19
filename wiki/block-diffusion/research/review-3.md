# 块扩散语言模型（Block Diffusion）审查记录（第 3 轮）

- 页面版本：index.html b8752ce5bf727c6b884c6b56d222bbd31f70be1e；overview.html ca010d1715214c14700db50379c96f2707726f7d
- 审查时间：2026-08-19 19:09
- 审查者：独立审查者（未参与写作与前两轮审查，未读取 research/ 下任何文件）
- 已完整阅读章节：index.html 全文（开篇引言、核心问题、1. 块间自回归、块内并行、2. block-causal 注意力掩码、3. KV cache 与灵活长度、4. 块内去噪、5. 当草稿器用、来源与范围说明）；overview.html 全文。
- 核对来源：/tmp/dflash-research/bd-tex/（Arriola et al., ICLR 2025 TeX 全文含附录与算法）；/tmp/dflash-research/tex/（DFlash, ICML 2026 TeX 全文）。
- 机械验证：`.dojo/scripts/validate.py wiki/block-diffusion/index.html` 通过；五个概念链接（../speculative-decoding/、../standard-attention/、../causal-mask/、../dflash/、../dflash2/）目标页面均存在。

## 来源核对结论（通过项摘记）

- F1=Eq.(1)（`\log p_\theta(\x) = \sum_{\ell=1}^L \log p_\theta(\xl \mid \x^{<\ell})`）、F2=Eq.(4)（块分解）、F4=Eq.(6)（`- \log p_\theta(\x) \leq \mathcal{L}_\text{BD}(\x; \theta) := \sum_{b=1}^{B} \mathcal{L}(\x^b, \x^{<b}; \theta)`）、F3=Eq.(7)（模型签名 `\x_\text{logits}^b, \mathbf{K}^b, \mathbf{V}^b \gets \x^b_\theta(\x^b_t, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})`）编号与内容全部核对一致。
- C1–C8、C11 均在 Arriola TeX 定位到支持原文（§3、§3.1、§3.2、§4.2、§6.2、附录 B/C）；C9/C10 在 DFlash TeX 定位到支持原文（见问题 3 的位置勘误）。
- N2 数字复算通过：Table 5（OWT，L=1024）SSD-LM $L'=25$ 37.2@40K NFE、281.3@1K NFE（T=25），BD3LM $L'=4$ 25.7@1K NFE；附录 C "algos{} and MDLM use $T=5$K diffusion steps"。
- N1 数字核对通过：DFlash §5 Implementation "we set the number of layers to 5 (8 for Qwen3 Coder) and use a block size of 16 (10 for LLaMA 3.1)"；消融含块大小 8（"we train two draft models with block sizes 8 and 16"）。
- 边界说法核对通过：$L'=1$ 期望等价于 AR NLL（§4.2、附录 supp:sar-block1-nll "For single-token generation ($L'=1$) we recover the autoregressive NLL"）且高方差（§3 "suffers from high variance despite being equivalent ... in expectation"）。
- 构造示例逐步复算通过（8 token、2×2=4 前向、两轮去噪状态表、两周期 3+1/4 结算、草稿+验证共 4 串行步）。
- 8×8 SVG 掩码网格填色与 block-causal 规则一致；行/列标签、图例、figcaption 均正确。
- 「解答」折叠块每题齐备、答案独立可读；来源说明六小节齐全；C/F/N 双向对应；$[\mathrm{MASK}]$ 全页统一；$\mathbf{K}/\mathbf{V}$ 粗体一致。

## 问题

- [重要·技术] index.html 1075–1091（第 4 章伪代码）：伪代码把去噪轮循环内前向输出的 `K_blk, V_blk` 直接并入缓存，但该前向的输入是仍带 $[\mathrm{MASK}]$ 的块（揭开发生在前向之后），缓存的是带噪输入的 K/V。来源算法中 K/V 来自块采样完成后对干净块的独立前向。｜引文依据：Arriola Alg. 2（Block Diffusion Sampling）第 79–80 行 `x^b ← Sample(x_θ^b, K^{1:b-1}, V^{1:b-1})`、`∅, K^b, V^b ← x_θ^b(x^b)`（输入为干净块 x^b）；§3.2 "denoising the next blocks requires running x_θ on the clean version x^b"。简化条件③（"省略了块内去噪各轮 K/V 的中间态处理"）未覆盖"干净前向"这一差异。｜修复要求：在轮循环结束、并入缓存之前增加一步"对已全部揭开的 block 做一次前向得到 K_blk, V_blk"（与 Alg. 2 第 80 行对应），并同步修改"验证的机制"段中"已揭开不遮回"与 K/V 来源的表述。｜修复：｜复验：
- [重要·技术] index.html 834、846–848、1089（第 1 章表格与构造示例、第 4 章"观察重点"）：串行前向计数均写为 $B\times$每块去噪轮数（"2×2=4 次""每块 1 轮：2 次""串行前向总数等于所有块的轮数之和"），未计入每块采样完成后产出 K/V 的一次干净前向。按来源算法每块为（去噪轮数 + 1）次前向，计数不完整且未声明该省略。｜引文依据：Alg. 2 第 80 行 `∅, K^b, V^b ← x_θ^b(x^b)` 为独立于 Sample 的一次前向；§3.2 "Thus every block has to go through the model at least twice."。｜修复要求：二选一且全文一致：①计数计入干净前向（每块轮数+1）；②维持现计数但在"简化条件及其限制"中明确声明"串行前向计数省略每块完成后的一次干净前向"并说明其对结论（量级对比）的影响。｜修复：｜复验：
- [重要·来源] index.html 1194（来源说明 C 段）：DFlash 章节标注"§1、§2.2、§3.1、§5"与 TeX 实际结构错位。DFlash 的 §2 是 Related Work，§3 是 Preliminaries（§3.2 = Autoregressive vs Diffusion Drafting），§4 是 Method（§4.1 Inference、§4.2 Training）。C10 的支持文在 §3.2 而非 §2.2；C9 的支持文在 §4.1/§4.2 而非 §3.1。按页面标注位置定位不到支持内容。｜引文依据：main.tex input 顺序 intro→related→preliminaries→method→exp；"For moderate block sizes, $T_{\text{draft}}$ is therefore largely insensitive to $\gamma$"（preliminaries，实际 §3.2）；"All masked positions within a block are decoded in parallel in a single forward pass"（method，实际 §4.1）；"we randomly sample anchor tokens from the response, use each anchor as the first position of a block"（method，实际 §4.2）。｜修复要求：把"§2.2"改为"§3.2"，"§3.1"改为"§4.1、§4.2"。｜修复：｜复验：
- [重要·技术] index.html 788（核心问题第 4 题解答）：答案写"按置信度揭开一部分（已揭开的不再遮回）"，与第 4 章正文步骤 3"揭开哪些位置由 noise schedule 决定，与预测置信度无关<sup>[C8]</sup>"及补充折叠块的 first-hitting 描述直接矛盾，来源不支持"按置信度揭开"。｜引文依据：Arriola 附录 C "the transition probability is the same for all masked tokens for a given $t$. Thus, the first timestep where a token is unmasked can be analytically sampled as follows"（与模型置信度无关）。｜修复要求：改为"按 noise schedule 揭开一部分（与预测置信度无关），已揭开的不再遮回"。｜修复：｜复验：
- [重要·技术] index.html 1118–1119（第 4 章本章问题第 1 题）：summary 写"按置信度揭开一部分"；答案写"③按置信度揭开一部分……每轮揭开哪些由预测置信度决定——按 noise schedule 决定揭开顺序……（与按预测置信度高低排序无关）"，同一句内先说由置信度决定、再说由 noise schedule 决定，自相矛盾且与来源不符。｜引文依据：同上（first-hitting，转移概率对所有被遮 token 相同，与置信度无关）。｜修复要求：summary 改为"按 noise schedule 揭开一部分"；答案第 ③ 步及后句统一为"揭开哪些位置由 noise schedule（first-hitting 的随机时间）决定，与预测置信度无关"。｜修复：｜复验：
- [重要·技术] index.html 1090（伪代码"简化条件"段）："真实实现里每轮揭开多少位置由噪声调度决定（置信度阈值随轮数变化）……伪代码只保留「按置信度揭开、不遮回、逐块推进」的骨架"。其一，"置信度阈值随轮数变化"在来源中不存在对应机制（论文用 first-hitting 解析采样随机时间，无置信度阈值）；其二，"按置信度揭开"与正文步骤 3 矛盾。｜引文依据：Arriola 附录 C first-hitting 采样器描述（无置信度阈值机制）；附录 B "Carry-Over Unmasking ... (if a token is unmasked in the reverse process, it is never remasked)"。｜修复要求：删去"（置信度阈值随轮数变化）"，把「按置信度揭开」改为「按 noise schedule 揭开」。｜修复：｜复验：
- [重要·技术] overview.html 58（核心机制列表）："逐轮并行预测、按置信度揭开（已揭开不遮回）"与来源不符（同上）。｜引文依据：同上（first-hitting 与置信度无关）。｜修复要求：改为"按 noise schedule 揭开（与预测置信度无关，已揭开不遮回）"。｜修复：｜复验：
- [轻微·来源] index.html 1203（来源说明·构造示例）："表中置信度高低亦为设定值"——第 4 章表格已无置信度信息（说明列已改为"按 noise schedule 揭开位置 1（示意，非按置信度）"），该句指向不存在的表格内容，是"按置信度揭开"修正的残留。｜引文依据：不适用。｜修复要求：删去该句，或改为"各轮揭开顺序亦为人为设定"。｜修复：｜复验：
- [轻微·格式] index.html 842、850（第 1 章）：两处"见第 4、5 章的权衡""会在第 4、5 章继续推进"使用章节数字引用，违反 style-guide"正文引用其他章节时使用章节标题"；且 842 行针对块大小权衡，第 4 章实际讨论去噪轮数权衡，指代不精确。｜引文依据：不适用。｜修复要求：改为章节标题引用（如「4. 块内去噪——从全 mask 到整块 token」「5. 当草稿器用——块扩散在 DFlash 里的形态」）。｜修复：｜复验：
- [轻微·来源] index.html 1200（来源说明 N2）："Arriola §6.2（LM1B/OWT 实验设置）"——gen PPL 对比表（Table 5）仅在 OWT 上进行，§6.2 不涉及 LM1B。｜引文依据：Table 5 caption "All models are trained on OWT"。｜修复要求：改为"Arriola §6.2（OWT 实验设置）"。｜修复：｜复验：
- [轻微·来源] index.html 1186 与 1200（N1 覆盖度）：本章问题解答写"块大小取 8–16<sup>[N1]</sup>"，但 N1 来源说明只记录 16/10 与层数，未包含消融中的块大小 8（消融见 DFlash §5.5 "we train two draft models with block sizes 8 and 16"）；正文 1135 行"消融中另含 8"亦无编号对应。｜引文依据：DFlash §5 Implementation "use a block size of 16 (10 for LLaMA 3.1)"；§5.5 消融 "block sizes 8 and 16"。｜修复要求：在 N1 中补充"消融含块大小 8（§5.5）"，或把 1186 行改写为仅引用 N1 已覆盖的数值（16/10）。｜修复：｜复验：
- [轻微·格式] index.html 1126（第 4 章本章问题第 2 题解答）：同一变量 $T$ 三种写法并存——"T=5K 步"（纯文本）、"$T=1$K 步"（K 在公式外）、"$T=25$ 步"；1106 行另有"$T=1$K"。字体与基线不一致，违反 style-guide 同一变量写法一致的要求。｜引文依据：不适用。｜修复要求：统一写法，如 $T=5\mathrm{K}$、$T=1\mathrm{K}$、$T=25$。｜修复：｜复验：
- [轻微·格式] index.html 1035（第 3 章）：`<sup>[C4]</sup><sup>[C11]</sup>` 两个上标并列，style-guide 规定可组合写法。｜引文依据：不适用。｜修复要求：合并为 `<sup>[C4, C11]</sup>`。｜修复：｜复验：
- [轻微·格式] index.html 885（SVG 列组标签）：列 5–8 区域 x 范围为 234–386，几何中心在 x=310，"块 2"标签位于 x=329，右偏约 19px（约半格）；对照"块 1"标签（x=158）恰在列 1–4 区域中心（82–234 的中点）。｜引文依据：不适用。｜修复要求：把 x=329 改为 x=310（text-anchor="middle" 不变）。｜修复：｜复验：
- [轻微·格式] index.html 883（SVG 轴标题）：class="dg-axis-label" 在页面样式表中未定义，且该 `<text>` 无 font-size 内联，将以浏览器默认约 16px 渲染，明显大于其余 11–12px 标签。｜引文依据：不适用。｜修复要求：为 .dg-axis-label 定义样式（如 font-size: 11px; fill: var(--text-light)）或在该 text 上内联 font-size。｜修复：｜复验：
- [轻微·图示] index.html 1148、1161（第 5 章流程图）：节点"块扩散草稿器一次前向预测整块 4 个位置"取自贯穿构造示例（块大小 4），而 figcaption 称"DFlash 的草稿—验证周期"、正文刚交代 DFlash 块大小为 16/10，图内未标注"4"为构造示例取值，易被误读为 DFlash 块大小。｜引文依据：DFlash §5 "use a block size of 16 (10 for LLaMA 3.1)"。｜修复要求：在 figcaption 中注明块大小 4 取自贯穿构造示例、DFlash 实际块大小为 16（LLaMA-3.1 为 10）。｜修复：｜复验：
- [轻微·来源] index.html 1194（来源说明 C 段位置列表不完整）：C8 的关键支持在 §6.2 正文（"the number of generation steps (NFEs) is upper-bounded by $L$ since tokens are never remasked"）与附录 C（first-hitting 采样器）；C11 的支持在摘要与 §1（"they lag in likelihood modeling and are limited to fixed-length generation"）——均未列入"§3、§3.1、§3.2、§4、附录 SAR Inference 算法"。且"SAR Inference 算法"（Alg. 2，Block Diffusion Sampling）位于正文 §3.2，不在附录。｜引文依据：Arriola §6.2 原文及附录 C "we adopt the first-hitting sampler proposed by zheng2024masked"；`\input{algs/short_algs}` 位于 §3.2 内。｜修复要求：位置列表补充 §6.2、摘要/§1、附录 C，并把"附录 SAR Inference 算法"改为"§3.2 算法 2（Block Diffusion Sampling）"。｜修复：｜复验：
- [轻微·术语] index.html 995（第 2 章）："掩码扩散的变分下界（NELBO）"——NELBO 为负 ELBO，是负对数似然的上界；"变分下界"通常指 ELBO（log 似然的下界），方向表述易混淆。｜引文依据：Arriola §2 "the Negative ELBO (NELBO)"；Eq.(6) 形式为 $-\log p_\theta(\x) \leq \mathcal{L}_\text{BD}$（上界）。｜修复要求：改为"负变分下界（NELBO，即负对数似然的上界）"或等效表述。｜修复：｜复验：
- [轻微·格式] index.html 1110（第 4 章补充折叠块）：`揭开"哪个 token"由解析采样的随机时间决定`使用英文直引号，页面其余强调统一用「」。｜引文依据：不适用。｜修复要求：改为「哪个 token」。｜修复：｜复验：
- [轻微·技术] index.html 1082（伪代码）：`block[i] ← argmax(logits[i])` 把揭开写为贪心 argmax，而来源采样用（核）采样（附录 C "we employ nucleus sampling for \algos{} ... $p=0.9$"）且附录 C 采用 Gumbel 类分类采样；伪代码"简化条件"未声明这一确定化简化。｜引文依据：Arriola 附录 C "Nucleus Sampling ... For \algos{}, AR and MDLM, we use $p=0.9$"。｜修复要求：在"简化条件"中补一句"揭开时取 argmax 为贪心确定化，真实实现按模型分布（核）采样"，或将该行改为按分布采样并注明。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 7 / 轻微 13
- 处置：仍需修复。七条重要问题集中于两个根因：①推理侧 K/V 来自干净块前向这一来源机制在伪代码与前向计数中被省略且未声明（问题 1、2）；②"按置信度揭开→noise schedule/first-hitting"的修正未贯穿到核心问题答案、本章问题、伪代码简化条件与 overview.html（问题 4–7），另有 DFlash 章节号整体错位（问题 3）。上述修复均为局部定点修改，不涉及大纲与范围调整；修复后运行 `.dojo/scripts/validate.py` 并复验，可发布。
