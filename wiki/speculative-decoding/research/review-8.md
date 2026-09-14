<!-- review-meta
round: 8
page: wiki/speculative-decoding/index.html
reviewed_content_sha256: 531fc446a56fc676
-->
# 投机解码（Speculative Decoding）审查记录（第 8 轮）

- 页面版本：`wiki/speculative-decoding/index.html` 工作树哈希 `2cc87002f2ede8ffe69d0ad792de12e8903049bd`；`overview.html` 哈希 `b639995a47e770f4fa6f511b01e99d99da5029c1`
- 审查时间：2026-09-14 17:43
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；未读取本页 `research/` 下任何文件）
- 页面类型：`dojo:type=concept`，适用 `guides/concept/check.md`
- 已完整阅读章节（含全部折叠块）：核心问题（5 问 5 答）→ 1. 为什么串行解码慢——内存带宽留下的「免费午餐」→ 2. Draft-then-Verify——一轮做了什么（含代码折叠块）→ 3. 为什么这条规则能保分布——单位置证明（含补充折叠块）→ 4. 能快多少——期望 token 数与加速比（含 2 个展开折叠块）→ 5. 把机制跑一遍——3 个 token 的手算例子（含展开折叠块）→ 6. 工程实例与边界 → 来源与范围说明（论断/公式/数字/构造示例/类比边界/简化条件全部小节）

## 来源核对（本轮实际打开并定位，写明所核版本）

核对的版本：arXiv:2211.17192 **v2（2023-05-18 修订版；v1 提交 2022-11-30，Comments: ICML 2023 Oral）**、arXiv:2302.01318 **v1（2023-02-02）**、vLLM 官方博客 **2024-10-17《How Speculative Decoding Boosts vLLM Performance by up to 2.8x》**，以及本页引用的两个站内前置页。

| 编号 | 页面标注位置 | 核对到的原文片段／数值 |
|---|---|---|
| [C1] | 两篇论文命名与提交历史 | Leviathan 摘要 "2X-3X acceleration compared to the standard T5X implementation, with identical outputs"（T5-XXL）；Chen 摘要 "Chinchilla, a 70 billion parameter language model"、2–2.5× decoding speedup、"without compromising the sample quality or making modifications to the model itself"。v1 日期分别 2022-11-30 / 2023-02-02，与页面一致 |
| [C2] | Leviathan §1 引言；Chen 摘要 | "inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth"（并含 communication，"so additional computation resources might be available"）；"the latency of parallel scoring of short continuations … is comparable to that of sampling a single token from the larger target model" |
| [C3-C6] | §2.3 + Algorithm 1；Chen §4.2 | Algorithm 1 名为 "SpeculativeDecodingStep"；Chen §4.2 标题即 "Modified Rejection Sampling"，接受规则 min(1, q/p)，重采样 x ∼ (q−p)₊，"a maximum of K+1 tokens per loop"。Chen 中 draft=*p*、target=*q*，与页面「该文 q 为 target、p 为 draft」的记号说明一致 |
| [C7] | §2.2 Standardized Sampling | "argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing" |
| [F1] | §2.3；Chen §4.2 | 三种边界（p≥q→1、p<q→p/q、p=0→0）与页面表格逐行一致 |
| [F2] | Algorithm 1「Adjust the distribution from Mp if needed」块 | "p′(x) ← norm(max(0, p_{n+1}(x) − q_{n+1}(x)))"；Chen 的 (f)₊ = max(0,f)/Σmax(0,f) |
| [F3] | Appendix A.1；Chen Supplementary Theorem 1 | A.1 标题 "Correctness of Speculative Sampling"，文内 "Let β be the acceptance probability"、"the normalizing constant for the adjusted distribution p′(x) is 1−β"；Chen "Theorem 1 (Modified Rejection Sampling recovers the target distribution)"，在硬件数值精度内恢复目标分布。页面「本页 β 为拒绝概率、与 Leviathan 的 β 互为补」的注记与 A.1 用法吻合 |
| [F4] | §3.1 Definition 3.1、§3.2 Corollary 3.6 | Definition 3.1 定义接受率 β；Corollary 3.6 给出 "α = 1 − E(D_LK(p,q)) = E(min(p,q))"。α = 1 − TV(p,q) 用 min(a,b) = (a+b−|a−b|)/2 复算成立 |
| [F5] | §3.1 Equation (1) | "E(# generated tokens) = (1 − α^{γ+1})/(1 − α)"，capped geometric，cap γ+1 |
| [F6] | §3.3 Theorem 3.8 | 加速比 "(1 − α^{γ+1}) / ((1 − α)(γc + 1))"，c 为 Mq 与 Mp 单次运行时间比 |
| 最优 γ | §3.5 "Choosing γ" | 该小节原文即讨论「在 Theorem 3.8 的加速比上取最优 γ、因 γ 为整数可数值求解」，与页面「最优 γ 的选择见 §3.5」一致 |
| [N1][N2] | 两篇摘要 | 同 [C1]；Chen 为分布式部署（distributed setup） |
| [N3] | 附录 A.3 Table 4 | 各行 target 均为 T5-XXL 11B、draft 均为 T5-base、c = 0.04；EnDe temp=0 → γ=7, α=0.8；EnDe temp=1 → γ=5, α=0.68。§3.6/§4.1 记 "T5-base (250M)"、"T5-XXL (11B)"——页面「T5-base 约 250M」与原文一致 |
| [N4] | vLLM 博客 2024-10-17 Fig.7 caption | "1.4x slowdown Llama3-70B on ShareGPT with 4xH100"、"1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100"；正文 "in high-QPS environments, speculative decoding may introduce performance trade-offs"、"…can sometimes slow down the system when it is already compute-bound" |
| 站内前置页 | GPU 执行模型 / 标准注意力 / Kimi K3 / EAGLE-3 | 四个被引页面均真实存在，无「（待生成）」占位。`gpu-execution-model` 原文记「H100 为 80GB，带宽 3.35 TB/s」，支撑页面的 3.35 TB/s 归因；`kimi-k3` §7 原文记「K3 把预训练 MTP 层微调为 EAGLE-3 风格 draft model」，支撑 [C8] |

### 复算（全部与页面一致）

- 140 GB / 3.35 TB/s = 0.0418 s ≈ 42 ms；70B FP16 = 2 B × 70e9 = 140 GB ✓
- E[L](α=0.8, γ=5) = (1−0.8⁶)/0.2 = 0.737856/0.2 = 3.689；S = 3.689/1.2 = 3.074 ✓
- E[L](α=0.2, γ=5) = 0.999936/0.8 = 1.2499；S = 1.04 ✓
- E[L](α=0.7, γ=4) = 0.83193/0.3 = 2.773；成本 1.4；S = 1.98 ✓
- γ 枚举（α=0.8, c=0.04）：γ=5…10 → 3.074 / 3.187 / 3.251 / 3.280 / 3.282 / 3.265，最大在 γ=9，γ=8 与 9 并列接近 ✓；α=0.5 时最优 γ=3（1.674），确实「显著更小」✓
- 手算例：a₁=0.8、a₂=1、a₃=0.4；max(0,p₃−q₃)=(0.2,0.1,0)，Z=0.3，p′₃=(2/3,1/3,0)；α₁=α₂=0.9、α₃=0.7；三位置逐 token 的 Pr[emit] 均等于 pᵢ ✓
- 补充折叠块的五步推导链与正文一致；Z = β 由 min+max 恒等式求和得出 ✓

### 机械项

- `.dojo/scripts/validate.py wiki/speculative-decoding/index.html` → `validation ok`
- `dojo:topics=训练与优化`、`dojo:tag=推理加速` 均在 AGENTS.md / `catalog_builder.py` 的词表内；`description` 为纯文本、`dojo:summary` 无 `$...$` 且可渲染
- 本页全部公式（含 SVG `<foreignObject>` 内的 `$i^\*$`、`$\mathrm{norm}(\max(0,\,p-q))$`）用 node 调 KaTeX 实测渲染无 `katex-error`；`<text>` 内无 ASCII 近似写法；SVG 各节点/箭头坐标闭合、无越界（viewBox 高 660，最低节点到 640）
- 无 `<img>` 内容图，`alt` 中无 `$...$`；无第一人称复数、无第二人称「你」；折叠块 summary 前缀仅用 `补充：/展开：/代码：` 三种

## 问题

- [轻微·表述] §3 第 2 段（"这一章回答…答案浓缩成一句话…下面把这句话展开成可手算的证明。"）｜元话语：用"下面…"预告行文而非直接给出内容，属 check.md 2.12 列举的元话语句式（同"下面来看…"）｜引文依据：不适用｜修复要求：删去「下面把这句话展开成可手算的证明」，本段以「因为接受步骤偏置了分布，残差重采样恰好把偏置的部分补回去」收尾后直接进入下一段「先固定一个位置 $i$…」；style-guide §12 允许的推导引导语为「关键观察／同理／因此」｜修复：｜复验：
- [轻微·表述] 第 1 章末（"下一章讲 draft 模型如何提供候选…"）、第 2 章末（"下一章证明它们组合起来恰好让输出分布回到 $p$"）、第 3 章末（"下一章用期望 token 数和加速比公式量化"）、第 4 章末（"下一章把整条机制跑一遍…"）、第 5 章末（"最后一章交代工程实例（K3 与 EAGLE-3）…"）｜章节过渡固定句式：五处收尾均复用「现在 X 已（清楚／证明）——但 Y？下一章 Z」同一模板｜引文依据：不适用｜修复要求：style-guide §8 规定章节衔接「不使用固定句式，也不为形式完整而添加过渡」；保留「前一章结论 → 下一章问题」的逻辑关系，但至少改写其中两处为陈述式收尾，不出现"下一章／最后一章"（例如直接以该章结论句结束，把引出问题改为陈述句）｜修复：｜复验：
- [轻微·格式] §1「构造数据（教学计算）：…」与 §5「构造示例。 这是一个为了手算构造的最小例子…」、§5 展开折叠块「构造示例。 上面正文只展示了主路径…」、来源节 h3「构造示例」｜同一类教学手算示例的引入标签不统一（"构造数据" vs "构造示例"），且与 style-guide §4 的固定标签（计算示例／代码示例／构造数据）不一致；「构造示例。」作为独立引入句在两处重复，属该节禁止的固定引入句｜引文依据：不适用｜修复要求：教学手算示例统一改用 style-guide §4 的标签「构造数据」；删去重复的独立引入句「构造示例。」，把该例的用途与非实测性质并入该段首句；来源节 h3 同步改为「构造数据」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（三条均为表述/格式层面的轻微问题，不影响任何来源论断、公式、数字与主线理解；修复后即可发布）

- 核对结论摘要：本页所有事实性论断、公式、引文编号与实验数字均已回到所核版本逐条定位并给出原文片段或数值，未发现定位不到、来源不支持、把实验条件下的观察写成无条件论断或把推断包装成来源结论的情形；正文、summary、overview 与图注之间的同一数字与编号一致；手算示例、∞ 边界与式-结论关系全部可复算；声称的加速比范围与两篇论文摘要及 vLLM 博客数值一致。判为无阻断、无重要。
