<!-- review-meta
round: 8
page: wiki/dflash2/index.html
reviewed_content_sha256: 1e9e35fb05d9da1d
-->
# DFlash 2 概念页审查记录（第 8 轮）

- 页面版本：3b095879b10a2f4856f5b8dc9c1f9327e4bcca36（`git hash-object`，工作树）
- 审查时间：2026-09-14 16:54
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题（5 条）→ 1. 两个剩余问题——候选池里有答案、块末端在漏气 → 2. 路径选择器——在候选之间打分，而不是重新预测（2.1 / 2.2 / 2.3 / 2.4）→ 3. 两抽头卷积——给块末端补上「看见前一位」的通道（3.1 / 3.2 / 3.3，含 SVG 结构图、图注与折叠「补充」）→ 4. 组合效果——每次验证多一个 token 的账（4.1 / 4.2 / 4.3）→ 5. 端到端与边界——哪里收益趋近 1（5.1 / 5.2 / 5.3）→ 来源与范围说明（论断与来源 C1–C17、公式与来源 F1–F2、外部数字与实验条件 N1–N10、构造示例、辅助解释与类比边界、简化条件及其限制）。全部 details 折叠块逐条读完。
- 来源核对方式：抓取 inco.ai/blog/dflash2/ 原文（含 Table 1–5、Figure 2/3/5 数据表、Run It Now 代码块、文末 Citation）；HF 模型卡 incoai/Qwen3.8-27B-DFlash2（YAML license、Acceptance Length 表、Throughput 三档并发表、评测条件、RadixArk DSpark 归属）；HF 模型卡 incoai/Muse-Glimmer-30B-DFlash2（license、Throughput 表）；arXiv:2602.06036（标题/作者/ICML 2026/v2 日期）。
- 机械核对：逐格重算 Table 1–5 与 Figure 2/5 全部数值、均值列、并发倍率（236.1/68.9=3.43 等）与 4.27→6.79、6.79/4.27=1.59 等派生量；`python3 .dojo/scripts/validate.py wiki/dflash2/index.html` → `validation ok`；两处 `href="../dflash/index.html#inference-pipeline"` 对应锚点 `<h2 id="inference-pipeline">2. 推理管线——KV 注入与单步并行起草</h2>` 存在。
- 核对结论（无问题项）：Table 1/2/3/4/5、Figure 2、Figure 5、HF Throughput 三档表与本页表格逐格一致；C4–C17 的引文片段均在原文逐字命中（"Coherence is mostly local…"、"Scoring stays fully parallel…"、"Even the oracle decays… No selector can fix that"、"We trained the DFlash and DSpark drafters ourselves under matched setups, while MTP ships with the model"、"the selector and the convolution together add only 1.3%" 等）；"16–25%" 与 Table 3 相对 DFlash 的逐基准增幅（15.7%–24.7%）吻合；head description / dojo:summary / overview.html / 正文四处数字互不矛盾；页面无 `<pre>` 可运行代码、无 `$...$` 进入 alt、无「本页/本文/我们/你」类自我或会话指代、无占位符、无等宽框线图；SVG 图内公式写在 `<foreignObject>` 并由 KaTeX 渲染。

## 问题

- [轻微·表述] 2. 本章问题「与 DSpark 修正头比，路径选择器便宜在哪？」的解答（第 271 行）：末句以「注意：」起句引导补充说明，属元话语（与 check.md 第 12 项举例的「需要注意的是」同类）。｜引文依据：不适用｜修复要求：删去「注意：」引导词，直接改为「4.61 距 oracle 6.79 仍有空间——选择上限取决于候选池本身，需要『3. 两抽头卷积——给块末端补上「看见前一位」的通道』修骨干。」｜修复：｜复验：
- [轻微·可读性] 核心问题 5 的解答（第 108 行）：缩写 `FA3` 在此首次出现且未展开，首次展开「FlashAttention 3」出现在 5.1（第 465 行），晚于首次使用（第 517、559 行也写 `FA3`）。｜引文依据：不适用｜修复要求：第 108 行改为「FlashAttention 3（FA3）」，或在第 108 行首次出现处展开；其余位置保持 `FA3`。｜修复：｜复验：
- [轻微·格式] 3.2 SVG 结构图 `<text>`（第 342 行）：`位置 1 的前一位 = 已验证 token（上一周期目标模型产出）` 在 `<text>` 内用 ASCII「=」表达等值关系（同图第 343 行 `<text>` 另有「+」连接），与 style-guide §11「`<text>` 只用于不含数学含义的纯文字」不符。｜引文依据：不适用｜修复要求：改为「位置 1 的前一位是已验证 token（上一周期目标模型产出）」与「自身（实线）与前一位（虚线）」；若保留关系符语义，则移入 `<foreignObject>` 交 KaTeX 渲染。｜修复：｜复验：
- [轻微·技术] 4.3 成本总账（第 439 行；同类表述另见 head summary 第 7 行、开篇 callout 第 66 行）：`+0.6%` / `+0.7%` / `+1.3%` 循环延迟在来源中均以五层 Qwen3-4B DFlash 为基准，本页多处按无条件开销陈述，且「简化条件及其限制」未列该基准。｜引文依据：博客 Table 2 注「Overheads are relative to plain DFlash: parameters added to the drafter, and added draft–verify cycle latency」（表题限定 five-layer Qwen3-4B on GSM8K）；Figure 2 注「Its convolutions add 3% parameters and 0.7% cycle latency」；正文原句「the selector and the convolution together add only 1.3% **to the five-layer DFlash draft–verify cycle latency**」——本页 C12 引文截到 "1.3%" 为止，丢掉原文的 "to the five-layer DFlash …" 限定语。｜修复要求：在 4.3 或「简化条件及其限制」补一句限定，例如「四个延迟百分比（选择器 +0.6%、卷积 +0.7%、合计 +1.3%、15 层 +15.2%）均以五层 Qwen3-4B DFlash 草稿器为基准」；C12 引文可补全为 "…add only 1.3% to the five-layer DFlash draft–verify cycle latency"。｜修复：｜复验：
- [轻微·格式] 3.2 卷积定义式（第 295 行）写作 `\operatorname{Conv}_k(x)_t`，而全页其余 9 处同一算子均写作 `\mathrm{Conv}_k`（dojo:summary 第 7 行、3 章解答第 93、94 行、本章问题解答第 384、385 行、SVG 内 3 处标签第 322、326、330 行、F2 第 555 行），同一符号两种写法，与 check.md 第 9 项「同一变量全页写法一致」不符。｜引文依据：不适用｜修复要求：统一为一种写法（建议全页用 `\mathrm{Conv}`，含第 295 行的定义式）。｜修复：｜复验：

## 结论

- 处置：修复（本轮无阻断、无重要；5 条轻微均为单点改动，修完即可发布）

统计：阻断 0 / 重要 0 / 轻微 5