<!-- review-meta
round: 5
page: wiki/speculative-decoding/index.html
reviewed_content_sha256: 2cdff152fff7193c
-->
# 投机解码审查记录（第 5 轮）

- 页面版本：f68840b1ef1cfed8f706e00fb5a01521fccca33e
- 审查时间：2026-09-13 21:53
- 审查者：独立子代理（第 5 轮审查，未参与写作与前四轮审查）
- 已完整阅读章节：核心问题 / 1. 为什么串行解码慢——内存带宽留下的「免费午餐」 / 2. Draft-then-Verify——一轮做了什么 / 3. 为什么这条规则能保分布——单位位置证明 / 4. 能快多少——期望 token 数与加速比 / 5. 把机制跑一遍——3 个 token 的手算例子 / 6. 工程实例与边界 / 来源与范围说明（含全部折叠块、伪代码折叠块与 SVG 图注）

## 已核对通过项（引文依据）

- [N1] arXiv:2211.17192 摘要：「We demonstrate it on T5-XXL」「show a 2X-3X acceleration compared to the standard T5X implementation」「with identical outputs」。与页面「T5-XXL (11B)、2×-3×、输出 identical」一致。
- [N2] arXiv:2302.01318 摘要：Chinchilla「a 70 billion parameter language model, achieving a 2-2.5x decoding speedup in a distributed setup」「without compromising the sample quality or making modifications to the model itself」。与页面一致。
- [C2] Leviathan §1 原句（逐字）：「inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication, so additional computation resources might be available.」；Chen 摘要「the latency of parallel scoring of short continuations ... is comparable to that of sampling a single token from the larger target model」。页面引文准确。
- [C7] Leviathan §2.2（Standardized Sampling）原句：「For example, argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing.」页面已明确标注「逐位 argmax 匹配」是推论而非论文原句，处理正确。
- [F1]/[F2]/[C3-C6] Leviathan §2.3 散文接受规则（「keeping it if q(x)≤p(x)」「reject with probability 1−p(x)/q(x)」）与 Algorithm 1（`p′(x)←norm(max(0,pn+1(x)−qn+1(x)))`）；Chen §4.2 Modified Rejection Sampling（接受 `min(1, q/p)`、残差 `x_{n+1} ∼ (q-p)_+`，且该文 q 记 target、p 记 draft）。页面记号换算说明与之一致。
- [F3] Leviathan Appendix A.1 标题「Correctness of Speculative Sampling」；[F5] §3.1 Equation 1 `E(#generated tokens)=(1−α^{γ+1})/(1−α)`；[F6] §3.3 Theorem 3.8。编号与位置均对应。
- [N3] 数值核对：Leviathan Table 4（EnDe、Mq=T5-base、Mp=T5-XXL）temp0 γ=7 α=0.8 c=0.04、temp1 γ=5 α=0.68 c=0.04；§4.1 列出「T5-base (250M)」。页面 α=0.80/γ=7、γ=5/α=0.68、c=0.04 与 250M 均与来源一致，并已如实标注「α 取 temp=0、γ 取 temp=1，为教学组合，不对应单一实验行」。
- [N4] vLLM 博客 2024-10-17：「we see 1.4x slowdown Llama3-70B on ShareGPT with 4xH100, 1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100」。页面「Llama 3 70B on 4×H100、高查询率下 1.4×-1.8× 减速」与之一致。
- [C8] 指向的 wiki/kimi-k3 §7「把预训练 MTP 层微调为 EAGLE-3 风格 draft model」原文存在；页面引用准确。
- 公式复算全部通过：E[L](α=0.8,γ=5)=3.689、S=3.689/1.2=3.074；α=0.2 反例 E[L]=1.25、S≈1.04；章末题 α=0.7,γ=4,c=0.1 → S≈1.98；最优 γ 枚举 3.074/3.187/3.251/3.280/3.282/3.265（γ≈9）；3-token 例残差 (0.2,0.1,0)→p′=(2/3,1/3,0)、各位置单位置接受率 0.9/0.9/0.7、ᾱ≈0.833→E[L]≈3.11 均无误。
- 一致性：description/dojo:summary/overview.html 与正文的 2×-3×、2-2.5×、1.4×-1.8×、3.07× 一致；无重复矛盾；[C1]–[C8]/[F1]–[F6]/[N1]–[N4] 全部有定义且被引用，无悬挂编号；四个前置概念链接（gpu-execution-model 3.35 TB/s、standard-attention、kimi-k3、eagle-speculative）与首页/overview 链接均真实存在，无「（待生成）」；alt/aria-label 无 `$...$`；无 Unicode 数学字符（伪代码块内 γ 属代码块豁免，× 属 validate.py 明示豁免的中文散文排版字符）；`python3 .dojo/scripts/validate.py wiki/speculative-decoding/index.html` 返回 `validation ok`。
- 表述：全文（含折叠块与图注）无会话指代（无「我们/你」）、无「本页将…」式元话语（style-guide §12 明示自称可用「本页」）、无调试复现叙事、无「（待生成）」等内容占位。

## 问题

- [重要·技术] 来源与范围说明 [N3]（696 行）与第 4 章折叠块（427 行）｜问题：两处把该表标注为「Leviathan et al. 2023 §4 实验部分 Table 4」，但 arXiv:2211.17192 v2 的 Table 4 不在第 4 章——§4 只有 Table 2（§4.1）与 Table 3（§4.2），Table 4 位于附录 A.3。按标注位置（§4 Table 4）定位不到该表，属表号/章节错位（所引数值本身正确，见上）。｜引文依据：Table 4 caption「Expected improvement factor (Exp) vs. empirically measured improvement factor (Emp).」位于 Appendix A.3「Theoretical Predictions vs. Empirical Runtimes」；§4.1 Table 2 caption「Empirical results for speeding up inference from a T5-XXL 11B model」，§4.2 Table 3 为 α 值表；§4 内不含 Table 4。｜修复要求：把两处「§4 实验部分 Table 4」改为「附录 A.3 Table 4」（或等价写法，如「Appendix A.3 Table 4」），使标注章节与表实际所在位置一致；如同时援引 §4.1 的表，写「§4.1 Table 2 / 附录 A.3 Table 4」。数字无需改动。｜修复：｜复验：
- [轻微·格式] 六处 `<h3>本章问题</h3>`（161、295、375、464、583、637 行）与目录生成脚本（768–794 行）｜问题：这六个 h3 都没有显式 id，脚本按相同文本自动生成同一个 `id="本章问题"`，在渲染后的 DOM 里产生 6 个重复 id；目录中出现 6 条同名子项，且都指向第一个「本章问题」，第 2–6 章的该子项锚点失效（第 8 项「目录锚点」）。仓库多数概念页为「本章问题」h3 显式指定了唯一 id（如 `id="boundary-questions"`）。｜引文依据：不适用｜修复要求：为六个「本章问题」h3 分别加上与所在章节呼应的唯一 id（例如 why-sequential-slow-questions、draft-then-verify-questions 等），使目录各子项指向对应章节。｜修复：｜复验：

## 结论

- 处置：修复
- 统计：阻断 0 / 重要 1 / 轻微 1