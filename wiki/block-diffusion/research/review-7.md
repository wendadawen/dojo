<!-- review-meta
round: 7
page: wiki/block-diffusion/index.html
reviewed_content_sha256: 42baeb0ea98ccab1
-->
# 块扩散语言模型（Block Diffusion）审查记录（第 7 轮）

- 页面版本：644dda591b88846b3f0f67dba75ff8ec928bc3fa
- 审查时间：2026-09-14 16:49
- 审查者：独立子代理（未参与写作，也未参与前序轮次）
- 已完整阅读章节：核心问题；1. 块间自回归、块内并行——两个范式各让一步（含本章问题）；2. block-causal 注意力掩码——一张掩码管两件事（含「为什么训练时每个块至少要过两次模型」折叠块、本章问题）；3. KV cache 与灵活长度——从扩散侧保住的东西（含本章问题）；4. 块内去噪——从全 mask 到整块 token（含伪代码折叠块、「揭开规则、first-hitting 与『已揭开不遮回』」折叠块、本章问题）；5. 当草稿器用——块扩散在 DFlash 里的形态（含本章问题）；来源与范围说明

## 回源核对已通过的条目（本轮无问题）

- F1/F2/F3/F4：Arriola Eq.(1)「log pθ(x)=∑_{ℓ=1}^{L} log pθ(x^ℓ|x^{<ℓ})」、Eq.(4)「=∑_{b=1}^{B} log pθ(x^b|x^{<b})」、Eq.(6)「−log pθ(x)≤ℒBD(x;θ):=∑_b ℒ(x^b,x^{<b};θ)」、Eq.(7)「x_logits^b,K^b,V^b←x_θ^b(x_t^b,K^{1:b-1},V^{1:b-1})」均与页面写法一致。
- C2/C5：正文「tokens in block b attend to tokens in blocks 1 to b」；「every block has to go through the model at least twice」；算法 2 逐块 Sample 后另做一次 x_θ 取 K/V，与伪代码一致。
- C6：正文「for a block size of one, the diffusion objective suffers from high variance」，且引言称该目标「equivalent to the autoregressive likelihood in expectation」，与页面一致。
- C8：正文采用 Zheng et al. (2024) 的 first-hitting 采样器，「the transition probability is the same for all masked tokens for a given t」「the first timestep where a token is unmasked can be analytically sampled」，位置在附录 C.5（与来源说明「附录 C」相符）；「tokens are never remasked」在 §6.2 与附录 B.3，NFE 上界为 L。
- C11/C4：正文 §1「cannot reuse previous computations with KV caching」「restricted to fixed-length output」/「Unlike prior diffusion models, block diffusion supports variable-length generation and KV caching」。
- N1：DFlash §5「the number of layers to 5 (8 for Qwen3 Coder) and use a block size of 16 (10 for LLaMA 3.1)」，§5.5.4 消融「block sizes 8 and 16」，均相符。
- N2（除下述口径问题外）：Table 7（OWT，L=1024）BD3-LMs L′=4 → 1K NFEs、Gen. PPL 25.7；SSD-LM T=1K per block、≥40K NFEs、37.2；NFE 可比（T=25）时 281.3，数值全部相符。
- C9/C10：DFlash §4.1「All masked positions within a block are decoded in parallel in a single forward pass」；§3.2 T_draft=t_parallel 与 γ·t_step 对比、中等块大小下对 γ 不敏感；§4.2 训练「randomly sample anchor tokens ... use each anchor as the first position of a block」，与「以干净 token 为块首」表述相符。
- 机械项：validate.py 返回成功；无 Unicode 数学字符（`<pre>` 除外）；无 alt 含 `$...$`；无「（待生成）」；索引 causal-mask / standard-attention / speculative-decoding / dflash / dflash2 均真实存在；SVG 掩码图坐标经像素核对（列 x=82+38k、行 y=68+38k，块边界矩形 82/234/220 起点与 152/304 宽度）与图注「块 1 行仅见块 1 列、块 2 行见 8 列」一致；overview.html 与 index.html 互链。

## 问题

- [重要·技术] 位置：第 4 章正文段（第 420 行）与第 4 章「本章问题」第 2 题解答（第 440 行）｜来源：Arriola et al. ICLR 2025 §6.2 Table 7 与附录 C.5；页面来源说明 N2｜问题：同一结果 gen PPL 25.7 的步数条件在正文与解答两处用两个不同数字给出且未区分口径。正文写「BD3LM…以不超过序列长度的 NFE 达到 gen PPL 25.7」，N2 写「NFE 1K，上界为 L」；解答却写「BD3LM 自身在 T=5K 步…达到 gen PPL 25.7」。页面伪代码把 T 定义为「去噪轮数上限 T」，读者会把「T=5K 步」与正文的「不超过序列长度的 NFE（1K）」读成对同一实验的矛盾数字——BD3LM 在 L=1024 时实际 NFE 仅 1K，T=5K 指的是扩散离散化步数，页面全篇未交代二者区别。解答又把「BD3LM T=5K 步」与「SSD-LM T=1K 步、≥40K NFE」并列，读者按 T 比较会得出 BD3LM 步数更多的印象，与正文「BD3LM 用远少的 NFE 取得更好 gen PPL」的结论相反，形成明显误解。｜引文依据：Arriola §6.2 Table 7 列 BD3-LMs L′=4、L=1024、NFEs 1K；附录 C.5「In Table 7, BD3-LMs and MDLM use T = 5K diffusion steps」；§6.2「the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked」｜修复要求：全页对同一实验采用同一口径。或统一用 NFE（正文已如此，则解答改为「NFE 1K（上界为序列长度 L）」），或在正文/来源说明中显式区分扩散步数 T 与生成步数 NFE，并把解答写成「T=5K 扩散步、实际生成 NFE 1K（上界 L）」，同时避免把 BD3LM 的 T 与 SSD-LM 的 T 直接并列比较｜修复：｜复验：
- [轻微·格式] 位置：第 4 章正文末句（第 420 行）｜来源：不适用｜问题：引用其他章节写作「（下一章）」，未使用章节标题，与 style-guide 第 1 节「正文引用其他章节时使用章节标题」及本页其他位置的写法（如「见『4. 块内去噪——从全 mask 到整块 token』」）不一致｜引文依据：不适用｜修复要求：改为章节标题引用，如「见『5. 当草稿器用——块扩散在 DFlash 里的形态』」｜修复：｜复验：
- [轻微·表述] 位置：第 4 章正文末句（第 420 行）｜来源：不适用｜问题：「只是把连贯性问题交由部署方按任务权衡」中「部署方」不是全文定义的术语，表述生硬；同句「轮数从几十到上千都是合法取值」给出具体范围却无来源支持（页面自身只出现 T=25、T=1K、T=5K 三个取值）｜引文依据：不适用｜修复要求：把「交由部署方」改为明确的主语（如「由使用者在质量与延迟之间权衡」），并为「几十到上千」给出依据或改为不指定具体范围的表述｜修复：｜复验：
- [轻微·表述] 位置：开篇 callout（第 66 行）｜来源：不适用｜问题：「想一次前向同时写出 8 个 token？」为面向读者的设问，「各猜各的、凑不成一句话」为口语化措辞，属于会话指代/口语化表述｜引文依据：不适用｜修复要求：改为陈述句与中性描述，如「若一次前向并行写出 8 个 token，各位置看不到彼此的输出，相邻位置容易给出互不衔接的 token」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复