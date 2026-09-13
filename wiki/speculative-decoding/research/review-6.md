<!-- review-meta
round: 6
page: wiki/speculative-decoding/index.html
reviewed_content_sha256: bec3a78de81f2c7f
-->
# 投机解码审查记录（第 6 轮）

- 页面版本：153a1366496eb3786a6d5f1f19a0d16c85640630
- 审查时间：2026-09-13 22:37（CST）
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题；1. 为什么串行解码慢——内存带宽留下的「免费午餐」；2. Draft-then-Verify——一轮做了什么；3. 为什么这条规则能保分布——单位置证明；4. 能快多少——期望 token 数与加速比；5. 把机制跑一遍——3 个 token 的手算例子；6. 工程实例与边界；来源与范围说明（含全部折叠块、图注、head meta、overview.html 对照）

## 来源核对（逐条打开来源并写下依据）

- Leviathan et al. 2023（arXiv:2211.17192，摘要 + 正文 + 附录全文抽取）：摘要「2X-3X acceleration compared to the standard T5X implementation, with identical outputs」→ 支持 [N1] 与第 6 章表；§1「inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication, so additional computation resources might be available」→ 支持 [C2]；§2.2「argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing」→ 支持 [C7]；§2.3 散文「keeping it if q(x) ≤ p(x) ... reject the sample with probability 1−p(x)/q(x)」+ Algorithm 1（含 p0(x) ← norm(max(0, pn+1(x)−qn+1(x)))）→ 支持 [C3-C6]/[F1]/[F2]；§3.1 Equation (1)「E(# generated tokens) = (1−α^{γ+1})/(1−α)」→ 支持 [F5]；Lemma 3.3「D_LK(p,q) = 1 − Σ min(p,q)」+ Corollary 3.6「α = 1 − E(D_LK(p, q)) = E(min(p, q))」→ 支持 [F4]；§3.3 Theorem 3.8「(1−α^{γ+1})/((1−α)(γc+1))」、Definition 3.7 c 为单步 Mq/Mp 时间比 → 支持 [F6]；§4.1 Setup「a batch size of 1 on a single TPU-v4」→ 支持第 6 章表「单机」；§4.1「T5-large (800M), T5-base (250M), T5-small (77M)」→ 支持「T5-base 约 250M」；附录 A.1 Correctness；附录 A.3 Table 4（ENDE＋T5-BASE：temp0 γ=7 α=0.8 c=0.04；temp1 γ=5 α=0.68）→ 支持 [N3]。提交 2022-11-30、ICML 2023 Oral、作者 Leviathan/Kalman/Matias 核实无异。
- Chen et al. 2023（arXiv:2302.01318，摘要 + 正文 + Supplementary 抽取）：摘要「a 2–2.5× decoding speedup in a distributed setup ... without compromising the sample quality or making modifications to the model itself」「the latency of parallel scoring of short continuations ... is comparable to that of sampling a single token」→ 支持 [N2]/[C2]；§4.2 Modified Rejection Sampling（ar5iv 编号核实为 §4.2）接受规则 min(1, q(x̃)/p(x̃))、残差 x_{n+1} ∼ (q−p)_+、全接受时多采 1 个 token，且该文 q=target、p=draft → 支持 [C3-C6]/[F1]/[F2] 的记号互换说明；Supplementary Materials Theorem 1「Modified Rejection Sampling recovers the target distribution」→ 支持 [F3]。提交 2023-02-02、作者列表核实无异。
- vLLM 官方博客 2024-10-17（vllm.ai/blog/2024-10-17-spec-decode）：高 QPS 段「1.4x slowdown Llama3-70B on ShareGPT with 4xH100」「1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100」→ 支持 [N4] 与正文 1.4×-1.8× 减速。
- 站内前置概念：wiki/gpu-execution-model（H100 80GB HBM3 @ 3.35 TB/s）、wiki/standard-attention（因果遮罩一节）、wiki/kimi-k3 §7「K3 把预训练 MTP 层微调为 EAGLE-3 风格 draft model」、wiki/eagle-speculative 均真实存在且内容支持页面引用；无「（待生成）」占位。

## 复算与机械核对

- 140GB / 3.35 TB/s ≈ 41.8ms ≈ 42ms。α=0.8, γ=5：E[L]=(1−0.8^6)/0.2=3.689，S=3.689/1.2≈3.07。α=0.2：E[L]≈1.25，S≈1.04。γ=5…10 枚举 3.074/3.187/3.251/3.280/3.282/3.265，最优 γ≈9（8 与 9 接近并列）。第 5 章 (0.9,0.9,0.7) 均值 0.833，E[L]≈3.11。第 4 章问题 α=0.7,γ=4,c=0.1 → S≈1.98。全部与页面数字一致。
- 单位置证明（路径 A/B、β=1−Σmin、min+max=b 恒等式）与折叠块五步推导逐步复算无误；第 5 章残差 (0.2,0.1,0)、Z=0.3、p'_3=(2/3,1/3,0)、贪心退化论证均无误。
- 折叠伪代码标为 language-text 且正文声明「不是 Python」，按静态审查核对逻辑正确；页面未声称其可运行。
- .dojo/scripts/validate.py → validation ok。head 含 description/dojo:summary/dojo:type=concept/dojo:topics=训练与优化/dojo:tag；overview.html 与 index.html 双向链接；全文无 Unicode 数学字符替代公式（× · – 为中文散文排版字符，ch5 的 · 在代码块内）；无 alt/aria 含 $...$；SVG 用内联 SVG + foreignObject（非等宽字符框图）。

## 问题

- [轻微·来源] 3. 为什么这条规则能保分布（正文与 [F3]）：本页用 $\beta$ 表示拒绝概率（$\beta = 1-\sum_x\min(p,q)$），与所引 Leviathan 原文中 $\beta$ 的含义相反。｜引文依据：Leviathan §3.1 Definition 3.1「The acceptance rate β_{x<t}, given a prefix x<t, is the probability of accepting x_t」（原文 β 为接受率）；Appendix A.1「Let β be the acceptance probability (Definition 3.1) ... the normalizing constant for the adjusted distribution p′(x) is 1 − β」（原文归一化常数为 1−β，而本页 $\beta$ 恰等于该归一化常数，即原文的 1−β）。｜修复要求：在 [F3] 或 [C3-C6] 注明本页 $\beta$ 为拒绝概率、与 Leviathan 的 $\beta$（接受率）互为补，或改用不与原文冲突的符号，避免对照原文（尤其 [F3] 所引 Appendix A.1）时混淆。｜修复：｜复验：
- [轻微·来源] 2. Draft-then-Verify 首段：括注「每轮（一个 "superstep"）做五件事」。｜引文依据：两篇来源论文正文均不出现 "superstep"（Leviathan §2.3 称 Algorithm 1；Chen §4.2 未使用该词）。引号写法易让读者误以为该术语出自来源。｜修复要求：删去引号/该术语，或注明其外部出处。｜修复：｜复验：
- [轻微·格式] Draft-then-Verify 流程图 SVG：图内公式承载于 `<foreignObject>` 的 `<div xmlns="http://www.w3.org/1999/xhtml">`，未按 `guides/concept/style-guide.md` §11 使用 `class="dg-label"`（站内其它 SVG 页如 prefix-caching 均使用该 class）。｜引文依据：不适用。｜修复要求：按 §11 为 div 加 `class="dg-label"` 并补相应样式（或说明本页自带的 `.flow-svg foreignObject div` 规则为等效替代），使图内公式样式与全站一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。三项轻微均不影响正确性与主线理解（$\beta$ 记号在页内自洽且已就地定义；"superstep" 仅为括注；dg-label 为样式约定，图内公式由 KaTeX 正常渲染），可接受；建议在下一轮随手为 $\beta$ 记号补注一句来源记号差异。
