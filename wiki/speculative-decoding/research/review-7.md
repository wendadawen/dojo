<!-- review-meta
round: 7
page: wiki/speculative-decoding/index.html
reviewed_content_sha256: 0db2d9176c4a63ba
-->
# 投机解码审查记录（第 7 轮）

- 页面版本：cad8ec57f6fdeefba9f292d69f024ced658711c6
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未读取本页 research/ 任何文件）
- 已完整阅读章节：核心问题；1. 为什么串行解码慢——内存带宽留下的「免费午餐」；2. Draft-then-Verify——一轮做了什么；3. 为什么这条规则能保分布——单位置证明；4. 能快多少——期望 token 数与加速比；5. 把机制跑一遍——3 个 token 的手算例子；6. 工程实例与边界；来源与范围说明（含全部 details 折叠块与 SVG 流程图图注）
- 已核对来源：arXiv:2211.17192（v2 PDF、PMLR v202 PDF、ar5iv HTML）、arXiv:2302.01318（v1 PDF、ar5iv HTML）、vLLM 官方博客 2024-10-17、wiki/kimi-k3/index.html、wiki/eagle-speculative/index.html
- 机械核对：`.dojo/scripts/validate.py` 返回 validation ok；SVG 内无 `<text>`（全部经 foreignObject 承载并由 KaTeX 渲染）；无带 `$...$` 的 alt（仅有 lightbox 占位 `alt=""`）；无占位/待生成；无第二人称或第一人称复数；overview.html 与 index.html 互链且数字一致（3.07×、1.4×-1.8×、2×-3×、2-2.5×）。

## 问题

- [重要·技术] head `<meta name="dojo:summary">`：摘要称「不匹配处回退到 target 分布」，与正文机制相互矛盾。正文明确说明拒绝后是从残差分布（而非 target 分布 $p$）重采样，并专门解释了不能直接从 $p$ 重采样的原因；overview.html 的对应表述也是「从残差分布重采样」。摘要的措辞恰好是正文所纠正的那种误解。｜引文依据：summary「匹配部分被接受，不匹配处回退到 target 分布」；正文「步骤 4：在<strong>首个</strong>被拒绝的位置 $i^\*$……从残差分布 $p'_{i^\*}(x)=\max(0,p_{i^\*}(x)-q_{i^\*}(x))/\sum_{x'}\max(0,p_{i^\*}(x')-q_{i^\*}(x'))$ 重采样一个 token」「为什么不能直接从 $p$ 重采样？因为接受步骤已经偏向 $q$ 高的 token，必须用残差把被低估的部分补回去」；overview「首个被拒绝处从残差分布 $\mathrm{norm}(\max(0, p-q))$ 重采样」。｜修复要求：把 summary 的「不匹配处回退到 target 分布」改为「不匹配处从残差分布（target 概率大于 draft 的部分归一化）重采样」，与正文与 overview 保持一致；改后 summary 仍为纯文本、无需 KaTeX。｜修复：｜复验：
- [轻微·技术] 来源章节 [F2]：定位「Leviathan et al. 2023 §2.3 Algorithm 1 step 5-6」中的 step 编号在线性来源中不存在，无法据此定位。该论文两版 Algorithm 1 的步骤以注释标签而非数字编号标出。｜引文依据：arXiv:2211.17192v2 §2.3 Algorithm 1（PMLR v202 同）逐行为「. Sample γ guesses x1,...,γ from Mq autoregressively.」「. Run Mp in parallel.」「. Determine the number of accepted guesses n.」「. Adjust the distribution from Mp if needed.」「. Return one token from Mp , and n tokens from Mq .」——均无步骤编号；残差语句位于「Adjust the distribution from Mp if needed」块内：「p′(x) ← norm(max(0, pn+1(x) − qn+1(x)))」。所在小节 §2.3 与 Algorithm 1 本身正确。｜修复要求：把「§2.3 Algorithm 1 step 5-6」改为「§2.3 Algorithm 1『Adjust the distribution from Mp if needed』块」，或改为不带步骤号的「§2.3 Algorithm 1」。｜修复：｜复验：
- [轻微·技术] 「5. 把机制跑一遍」pos 1 单位置验证段：以「pos 1 上 $p_1 < q_1$」作整分布比较为假，该不等关系只对 token $A$ 成立。本页同章折叠块自身即给出反例（$B$ 处 $p_1(B)>q_1(B)$，残差项 $\max(0,0.4-0.3)=0.1\neq 0$）。｜引文依据：正文设定 $p_1=(0.4,0.4,0.2)$、$q_1=(0.5,0.3,0.2)$，故 $p_1(B)=0.4 > q_1(B)=0.3$；同章折叠块「同理 $\Pr[\text{emit } B] = \min(0.3, 0.4) + \max(0, 0.4-0.3) = 0.3 + 0.1 = 0.4 = p_1(B)$ ✓」。｜修复要求：把「pos 1 上 $p_1 < q_1$」改为「pos 1 上 $p_1(A) < q_1(A)$」，或明示该判断只针对 token $A$。｜修复：｜复验：

## 已核对且无误的关键论断（记录依据，非问题）

- [F5] $\mathbb{E}[L]=(1-\alpha^{\gamma+1})/(1-\alpha)$ 确在 §3.1 Equation (1)（两版一致，式 (1) 位于 §3.1 正文右栏）；[F6] $S$ 与 §3.3 Theorem 3.8 推导一致（每轮成本 $c\gamma+1$、产出 $\mathbb{E}[L]$）；[F4] $\alpha$ 见 §3.1 Definition 3.1 与 §3.2 Corollary 3.6；[C7]「argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing.」确在 §2.2 Standardized Sampling；[C2] 引言原句逐字一致。
- [N1] T5-XXL、2X-3X、identical outputs；[N2] Chinchilla 70B、分布式、2-2.5x、不损害 sample quality、不修改模型；[N3] 附录 A.3 Table 4 EnDe/T5-BASE：temp=0 γ=7 α=0.8 c=0.04，temp=1 γ=5 α=0.68 c=0.04；§4.1 原文「T5-base (250M)」（页内「约 250M」与来源一致）与「batch size of 1 on a single TPU-v4」（页内「单机」一致）；[N4] 博客图表：「1.4x slowdown Llama3-70B on ShareGPT with 4xH100」「1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100」。
- 计算复核：$\mathbb{E}[L](\alpha{=}0.8,\gamma{=}5)=3.689$、$S=3.07$；$\alpha{=}0.2\Rightarrow1.04$；$\gamma{=}5\ldots10$ 枚举 $S=3.074,3.187,3.251,3.280,3.282,3.265$，最优 $\gamma\approx9$；手算例 pos1/2/3 的 $a_i=0.8,1.0,0.4$、残差 $(0.2,0.1,0)$、$Z=0.3$、$p'_3=(2/3,1/3,0)$、$\alpha_i=(0.9,0.9,0.7)$、$\mathbb{E}[L]\approx3.11$ 全部可复算且与页面一致。
- 引用概念页均真实存在且互链：gpu-execution-model、standard-attention、kimi-k3、eagle-speculative；kimi-k3 §7 确有「K3 把预训练 MTP 层微调为 EAGLE-3 风格 draft model」。
- 表述维度：通读全文（含折叠块与图注）未发现会话指代（我/我们/你）、调试叙事、临场评价或禁用固定句式；「本页」的使用符合 style-guide §12（自称允许「本页/本文」）。

## 结论

- 处置：修复
- 统计：阻断 0 / 重要 1 / 轻微 2