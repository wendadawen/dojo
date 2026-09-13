<!-- review-meta
round: 4
page: wiki/speculative-decoding/index.html
reviewed_content_sha256: 8bd9cdb821686704
-->
# 投机解码审查记录（第 4 轮）

- 页面版本：19258db94336739be42a03c760d2087428a6c5bf
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题；1. 为什么串行解码慢——内存带宽留下的「免费午餐」；2. Draft-then-Verify——一轮做了什么；3. 为什么这条规则能保分布——单位置证明；4. 能快多少——期望 token 数与加速比；5. 把机制跑一遍——3 个 token 的手算例子；6. 工程实例与边界；来源与范围说明（含全部折叠块、SVG 图注）；overview.html

## 已核对来源（外部链接均 WebFetch 抓原文，记录原文片段）

- [N1] arXiv:2211.17192 摘要原文：'2X-3X acceleration compared to the standard T5X implementation'、'identical outputs'、target=T5-XXL；作者 Leviathan, Kalman, Matias；ICML 2023 Oral；提交 2022-11-30。与页面一致。
- [N2] arXiv:2302.01318 摘要原文：'2-2.5x decoding speedup in a distributed setup'、Chinchilla 70B、'without compromising the sample quality or making modifications to the model itself'；提交 2023-02-02。与页面一致。
- [C2] Leviathan §1 原文：'inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication, so additional computation resources might be available'；Chen 摘要原文：'the latency of parallel scoring of short continuations ... is comparable to that of sampling a single token from the larger target model'。均一致。
- [C3-C6]/[F1]/[F2] Leviathan §2.3 与 Algorithm 1、Chen §4.2 Modified Rejection Sampling：原文 'min(1, q(x̃ₙ₊₁|x₁,…,xₙ)/p(x̃ₙ₊₁|x₁,…,xₙ))'（该文 q=target、p=draft，与页面[F1]所述记号相反一致）、重采样 'xₙ₊₁ ~ (q-p)₊'、'(f)₊ = max(0,f)/Σ max(0,f)'。一致。
- [F3]/[F5]/[F6] Leviathan Appendix A.1（'accepted guesses contribute min(p,q), rejected guesses contribute [p − min(p,q)], yielding p(x′)'）、§3.1 Eq.(1) 'E = (1−α^(γ+1))/(1−α)'、§3.3 Theorem 3.8 '[1−α^(γ+1)]/[(1−α)(γc+1)]'。一致。
- [F4] Leviathan §3.1 Definition 3.1、§3.2 Theorem 3.5 'β = 1 − D_LK(p,q)'、Corollary 3.6 'α = 1 − E(D_LK) = E(min(p,q))'。一致。
- [C7] Leviathan §2.2 原文：'argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing'。页面已明示贪心退化结论为其推论、非论文原句。一致。
- [C8] wiki/kimi-k3/index.html §7 原文确有「K3 把预训练 MTP 层微调为 EAGLE-3 风格 draft model」。一致。
- [N3] Leviathan §4 Table 4 原文：EnDe / T5-base(250M) draft / T5-XXL(11B) target；temp=0 γ=7 α=0.8；temp=1 γ=5 α=0.68；c=0.04；论文另注 T5-XXL 11B、T5-base 250M。与 [N3] 一致。
- [N4] vLLM 官方博客 2024-10-17（blog.vllm.ai 经 301 跳转 vllm.ai/blog/2024-10-17-spec-decode）原文：Llama3-70B on 4xH100，'1.4x slowdown ... on ShareGPT'、'1.8x slowdown ... on CNN Dailymail'；'In high-QPS environments, speculative decoding may introduce performance trade-offs.'。与页面 1.4×-1.8× 一致。
- 内部链接 gpu-execution-model（H100 80GB · 3.35 TB/s 原文确认）、standard-attention、kimi-k3、eagle-speculative 均真实存在，无「（待生成）」占位。
- 复算：E[L](α=0.8,γ=5)=3.6893、S=3.0744（页面 3.69/3.07 一致）；α=0.2→S=1.0416（页面 1.04）；α=0.7,γ=4,c=0.1→E[L]=2.773、S≈1.98；γ 枚举 (5..10) S=3.0744/3.1866/3.2509/3.2795/3.2817/3.2646（页面 3.074/3.187/3.251/3.280/3.282/3.265 一致），最大在 γ=9；「把机制跑一遍」全部残差与接受率数字（0.9/0.9/0.7、ā≈0.833、E[L]≈3.11）复算一致。validate.py 返回 ok。

## 问题

- [阻断·技术] 2 章末段（第 248 行）「因此投机解码从不比纯解码产出更少——它只可能更快，不可能更慢（单流下；高并发的退化见「能快多少」一章）」：该句把「单流下变慢」的唯一原因归于高并发，与同一页第 4 章及页面级核心问题 Q4 的答案直接矛盾。页面自己在第 444 行写「α 再低就会 $S < 1$，反而变慢」，第 453–454 行表格把「α 太低」「c 太大」两条列为单流公式内的减速条件（仅第三条「高负载批处理」才是单流公式不适用者），第 124 行核心问题 Q4 答案亦写「$S$ 趋向 1 甚至小于 1」。按页面自家公式复算，$c=0.04,\gamma=5,\alpha=0.1$ 时 $\mathbb{E}[L]=1.1111$、$S=1.1111/1.2\approx0.926<1$，单流下即已变慢，无需任何高并发。即该无条件论断在页面自身框架内即为假。｜引文依据：页面第 248 行「只可能更快，不可能更慢（单流下…）」对照第 444 行「α 再低就会 S < 1，反而变慢」、第 453–454 行表格「α 太低/C 太大」两条；复算 α=0.1,γ=5,c=0.04 → S≈0.926。｜修复要求：把该句限定为「产出（token 数）不比纯解码少（每次 target 调用至少产出 1 个）」，删除「（单流下）不可能更慢」的无条件论断，并把减速条件改写为「α 过低或 c 过大时单流即会变慢（详见第 4 章），高并发批处理下进一步退化为总吞吐下降」。｜修复：｜复验：
- [轻微·表述] 第 608 行章间过渡「构造示例把机制、证明、加速比公式串通——最后一章把工程实例（K3 与 EAGLE-3）和论文实测做个交代，并说明本页的边界」：只预告下一章覆盖范围，未按 content-examples A9「章间过渡先总结本章已得结论，再指出结论尚不能解决的下一步问题」指出逻辑缺口，形态接近 A9 反例「只报章节名」。同页其余四处章间过渡（第 186、320、400、489 行）均先点出缺口再引出下章，此处不一致。｜引文依据：不适用｜修复要求：改为先总结第 5 章结论、再点明第 6 章要回答的具体缺口（例如「手算例子把 α 当作已知量；工程中 α 由 draft 架构与负载决定，且加速并非无条件」），或删去该预告句。｜修复：｜复验：
- [轻微·技术] 第 626 行「两篇论文独立提出相同机制，是这一思路自然成立的标志」：前半句有来源（[C1]），后半句「是……自然成立的标志」为页面自身判断，无来源支持却写成结论性表述，未以推断口吻标注。｜引文依据：arXiv:2211.17192 与 2302.01318 摘要均无此类论断。｜修复要求：改为明确标注的页面推断（如「本页据此推断该机制是通用有效的推理时优化思路」），或删除该半句。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复