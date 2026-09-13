<!-- review-meta
round: 4
page: wiki/block-diffusion/index.html
reviewed_content_sha256: 1351c2797d5d9a7d
-->
# 块扩散语言模型（Block Diffusion）审查记录（第 4 轮）

- 页面版本：830ab41963e49ee9a6327b9d99172d71d5550d9b（index.html 工作树哈希）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题（5 题答案折叠块）；1. 块间自回归、块内并行——两个范式各让一步（含本章问题）；2. block-causal 注意力掩码——一张掩码管两件事（含掩码图、补充折叠块、本章问题）；3. KV cache 与灵活长度——从扩散侧保住的东西（含本章问题）；4. 块内去噪——从全 mask 到整块 token（含伪代码折叠块、补充折叠块、本章问题）；5. 当草稿器用——块扩散在 DFlash 里的形态（含流程图、本章问题）；来源与范围说明。overview.html 一并通读。

## 来源核对（本轮实际打开的来源与原片段）

来源一：Arriola et al., "Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models", ICLR 2025，arXiv:2503.09573v3（读全文 HTML/TeX 源）。
- Eq.(1) `log pθ(x)=Σ_{ℓ=1}^{L} log pθ(x^ℓ|x^{<ℓ})`；Eq.(4) `log pθ(x)=Σ_{b=1}^{B} log pθ(x^b|x^{<b})`；Eq.(6) `−log pθ(x) ≤ L_BD(x,θ):=Σ_b L(x^b,x^{<b},θ)`；Eq.(7) `x_logits^b, K^b, V^b ← x_θ^b(x_t^b, K^{1:b-1}, V^{1:b-1})`——与页面 F1/F2/F4/F3 及第 3 章签名逐字一致。
- §3.1："tokens in block b attend to tokens in blocks 1 to b"——与第 2 章掩码规则一致。
- §3："block diffusion decoding algorithm enables us to sample sequences of arbitrary length, whereas diffusion models are restricted to fixed-length generation"——与 C4/C11、第 3 章"长度固定"表述一致。
- §3.2 Algorithm 1："Each token passes through x_θ twice"——与第 2 章补充折叠块"每块至少过两次模型"一致（页面按块表述，含义等价）。
- §4.2："training on the diffusion objective involves estimating loss gradients with 2x fewer tokens and is responsible for higher training variance compared to AR"；"Our block diffusion parameterization (8) is equivalent in expectation to the autoregressive NLL (1) in the limiting case where L′=1"——与 C6（L′=1 期望等价但梯度方差高）一致。
- §6.2 + Table 7（OWT，L=1024）："BD3-LMs L′=4 25.7 / 1K NFEs；SSD-LM L′=25 37.2 / 40K NFEs"，正文"NFEs is upper-bounded by L since tokens are never remasked"，以及"For SSD-LM, we compare sample quality using T=1K diffusion steps per block … and T=25 where NFEs are comparable"（T=25 行 281.3）——与第 4 章正文三个数字 25.7 / 37.2 / 281.3 及条件逐一吻合。
- 附 C.5 "Number of Diffusion Steps"："In Table 7, BD3-LMs and MDLM use T = 5K diffusion steps"——页面 §4 本章问题解答中的"BD3LM 自身在 T=5K 步下达到 gen PPL 25.7"成立，与正文"以不超过序列长度的 NFE 达到 25.7"不矛盾（T=5K 是扩散步数，NFE 上界为 L）。
- 附 C.5 first-hitting："We adopt the first-hitting sampler proposed by Zheng et al. (2024)…(1) the transition probability is independent of the denoising network, (2) the transition probability is the same for all masked tokens for a given t. Thus, the first timestep where a token is unmasked can be analytically sampled"——与 §4 补充折叠块"转移概率相同、由解析采样的随机时间决定、与预测置信度无关"一致。
- §6.2："SSD-LM (Han et al., 2022), an alternative block diffusion formulation. Unlike our discrete diffusion framework, SSD-LM uses Gaussian diffusion"——与第 4 章"另一种块扩散方法，用高斯嵌入扩散而非离散掩码"一致。
- 附 B.3："if a token is unmasked in the reverse process, it is never remasked"——与 C8"已揭开不遮回"一致。
- 摘要："yet they lag in likelihood modeling and are limited to fixed-length generation"；"supporting flexible-length generation and improving inference efficiency with KV caching"——与 C11/C4 一致。
- §3.2 Algorithm 2 "Block Diffusion Sampling" 存在——与来源说明"含算法 2"一致。

来源二：Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026，arXiv:2602.06036v2（读全文 HTML）。
- "we set the number of layers to 5 (8 for Qwen3 Coder) and use a block size of 16 (10 for LLaMA 3.1)"——与第 5 章"块大小 16（Qwen3）/10（LLaMA-3.1）、5 层（Coder 为 8）"及 N1 一致。
- "Diffusion drafters generate all γ tokens in parallel within a single forward pass, yielding T_draft = t_parallel"；"drafting cost no longer scales with the number of generated tokens"——与 C9/C10 及第 5 章"单步并行、延迟对块大小不敏感"一致。
- "the draft model always conditions on a clean token produced by the target model (i.e., the bonus token from the previous verification step)"——与第 5 章"以干净 token 为块首/条件锚点"一致。
- "autoregressive drafters are constrained to very shallow architectures (e.g., a single transformer layer in EAGLE-3)"——与第 5 章"EAGLE 系列的单层草稿网络"一致。
- Table 8 块大小 16 vs 8 的消融——与 N1"消融含块大小 8"一致。

机械项：`python3 .dojo/scripts/validate.py wiki/block-diffusion/index.html` 返回 `validation ok`；页面引用的 5 个前置概念页（causal-mask、standard-attention、speculative-decoding、dflash、dflash2）均真实存在；index.html 与 overview.html 相互链接；`dojo:topics=训练与优化`、`dojo:tag=训练` 均在词表内；alt/aria-label 中无 `$...$`；正文与公式无 Unicode 数学字符（唯一 `×` 在伪代码块内，属代码）；所有本地引用与锚点有效。

## 问题

- [重要·表述] 正文多处元话语与以"本文/本页"为主语的自我指代：§引言第 2、3 段、第 128 行、第 191 行、第 376 行、第 420 行、第 155 行，以及 §2 正文第 308 行与 §2 本章问题解答第 328 行。问题：直接命中规范 §2.2 第 12 条点名的"本页将…""下面来看…""需要注意的是"类不合格表述——"本文讲清它的结构与采样过程：…"、"本文的学习基础是三个已讲解的概念"、"本文只需要它的最小含义"、"本页不推导它，只需要结论"、"本页只引用其可并行计算的结论"、"下面用 8 个 token、2 个块（每块 4 个）画出这张掩码"、"注意第 2 轮的关键差别"、"到这里还差运行时的另一半"、"两个极端值得记住"；同段另有临场评价与松用词"有一个坦率的短板"、"把连贯性问题留给了使用场景去消化"。｜引文依据：不适用（表述类）｜修复要求：改为不带"本文/本页"主语的直接陈述——"它的结构与采样过程包含：…"；"前置概念有三个：…"；"只需它的最小含义：…"；"此处不推导它，只需其结论"；"此处只引用其可并行计算的结论"；"以 8 个 token、2 个块（每块 4 个）为例，掩码如下"；"第 2 轮的关键差别在于：…"；"还剩运行时的另一半：…"；"两个极端的情形是："；"坦率的短板"改"明显的短板"，"留给使用场景去消化"改"由部署方按任务权衡"。§"来源与范围说明"中作为范围声明的"本页"用法属该节功能，可保留。｜修复：｜复验：

- [轻微·可读性] §5 本章问题第 2 题解答折叠块：符号 $\gamma$ 在全页首次且唯一出现，未说明含义（此处指草稿块 token 数），$\gamma\cdot t_{\text{step}}$ 的运算含义需读者自行推断。｜引文依据：不适用｜修复要求：首次出现处补一句定义，如"（$\gamma$ 为每周期草稿 token 数）"。｜修复：｜复验：

- [轻微·可读性] block-causal 掩码图注与 §2 本章问题第 1 题解答：把行 5–8 × 全 8 列的下方横带称作"左下 4×8 区域"，与同句"右下 4×4 子块"并列时方位自相矛盾（4×8 的横带已包含右下 4×4），且"跨块可见区"的称呼与该带含块内列的事实不符。｜引文依据：不适用（图内网格与着色本身正确，$4\times4$ 块内双向、右上全遮、下方全可见均无误）｜修复要求：改为不依赖象限方位的描述，如"查询位置 5–8 的行可见全部 8 列（含块 2 的块内 4×4 与块 2→块 1 的跨块 4×4）"，图注与 §2 解答同步。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（技术事实、公式、数字与两篇来源逐条吻合，无阻断级问题；1 个重要表述问题与 2 个轻微可读性问题需就地修复并复验）