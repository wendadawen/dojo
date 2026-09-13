<!-- review-meta
round: 7
page: wiki/block-attnres/index.html
reviewed_content_sha256: 917a747badb5660c
-->
# Block AttnRes 审查记录（第 7 轮）

- 页面版本：bb4dac8af47911813ef8ae76dd8bd32a156e7560（wiki/block-attnres/index.html 工作树哈希）；overview.html：e656e99b7b4e33c3f99c5dc65fb25b10b24cfc02
- 审查时间：2026-09-13 21:48
- 审查者：编排者派发的独立审查者（未参与写作，未读取 research/）
- 已完整阅读章节：核心问题（5 条）、1. 标准残差在深度上的瓶颈——为什么需要 AttnRes、2. Full AttnRes 的公式——pseudo-query 如何检索前序层、3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$、4. K3 的具体配置——8 块×12 层、9 个候选、加权三次、5. softmax kernel 中的 RMSNorm——为什么不能直接用内积、来源与范围说明（C/F/N 三节、构造示例、辅助解释与类比边界、简化条件及其限制）；含全部 details 折叠块、4 张结构图图注与伪代码块
- 来源获取与核对方式：arXiv 2607.24653 v2 PDF（pdftotext 全文提取，逐句比对）与 v1 PDF（核对 v1→v2 编号映射）；HuggingFace `moonshotai/Kimi-K3` 官方 `config.json`；GitHub `MoonshotAI/nano-kpu`；arXiv abs 版本记录。`.dojo/scripts/validate.py wiki/block-attnres` 返回 success

## 机械核对（无问题，留档）

- 引文原文逐字命中 v2 PDF：`Standard residual connections … compress all prior information into a single state hl over depth — a bottleneck reminiscent of RNNs over time.`；`Since network depth is modest (L < 100), the O(L2 d) arithmetic of this full form is affordable; the practical overhead is the O(Ld) memory (and cross-stage communication under pipeline parallelism) …`；`the RMSNorm prevents layers with large-magnitude outputs from dominating the weights`；`Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(N d)`；`The final output layer then aggregates all N block representations`；`Empirically, N ≈ 8 recovers most of the benefit across model scales [58]; … into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer.`；`Attention Residuals (AttnRes) [58] enable each module to selectively retrieve representations …`；§7 Chip design `hybrid KDA and NoPE-MLA attention, Block AttnRes with a block size of two`。
- 数字核对：Table 1 `#Layers 93`、`Attention-Layer Composition 69 KDA + 24 MLA`、`Hidden Dimension 7,168`、`Attention Heads 96`；config.json `num_hidden_layers=93`、`hidden_size=7168`、`attn_res_block_size=12`、`num_attention_heads=96`、`full_attn_layers` 24 项、`kda_layers` 69 项；93 = 7×12+9；√7168 ≈ 84.7。
- 算式复算一致：6 候选 $h_6\approx[0.703,0.703]$（Σφ=11.8984）、4 候选 $h_6\approx[1.059,1.241]$（Σφ=11.7377）、加 RMSNorm $h_6\approx[1.004,1.056]$（Σφ=9.8583）、3 候选解答 0.274/0.274/0.452、大值敏感性 exp(2)≈7.4 / exp(5)≈148 / exp(4)≈54.6、权重差表 ↑0.065/↓0.120/↓0.027/↑0.082；分项之和均等于合计。
- 引文编号：第 811 行声明本页按 arXiv 2607.24653v2 编号，[58]=原 AttnRes preprint、[148]=RMSNorm；与 v2 PDF 参考文献列表一致（`[58] Kimi Team. Attention Residuals. Preprint. 2026.`、`[148] Biao Zhang and Rico Sennrich. "Root mean square layer normalization"`）；其 v1 映射（[57]/[146]）亦与 v1 PDF 逐条一致。故 index.html 本体编号无误。
- 链接与页面功能：前置页 residual-connection、kimi-k3-dataflow 存在且被引；overview.html 与 index.html 互链；4 张结构图为 HTML/CSS grid（非等宽字符画），无 `<img>` alt 含 `$…$`；无"（待生成）"占位；两级问题块均有解答折叠块且答案独立可读。

## 问题

- [重要·来源一致性] wiki/block-attnres/overview.html 第 49 行（对应 index.html 第 575/811/820/838/840 行）：同一引用（原 AttnRes preprint）在两页给出不同编号——index.html 用 [58]，overview.html 用 [60]，同一文献在两页无法对应到同一编号，读者按 overview 的编号会定位到错误的参考文献。｜引文依据：arXiv 2607.24653v2 PDF 参考文献列表为 `[58] Kimi Team. Attention Residuals. Preprint. 2026.`，而 PDF 中 `[60]` 是 `Kimi Team. Kimi K2.5: Visual Agentic Intelligence`——index.html 与 PDF 一致，overview.html 的 [60] 与 PDF 不符。｜修复要求：将 overview.html 第 49 行的 `[60]` 改为 `[58]`，与 index.html 及 arXiv 2607.24653v2 PDF 参考文献列表一致；若确要改用其他渲染版本的编号，须在两页同时声明编号依据并保持两页编号一致。｜修复：｜复验：
- [轻微·来源] index.html 第 115 行（主要依据 blockquote）：`（arXiv 2607.24653v2，2026-07-28）` 把 v2 标为 2026-07-28，与 arXiv 的版本记录不符。｜引文依据：arXiv abs 页版本记录 `[v1] 27 Jul 2026 / [v2] 7 Aug 2026`；wiki 内 kimi-k3 页亦以 `arXiv:2607.24653v2，2026-08-07` 标注同一版。｜修复要求：将日期改为 v2 的真实日期 2026-08-07；若想标 2026-07-28（v1 公告日），须同时把版本号改为 v1。｜修复：｜复验：
- [轻微·可读性] index.html 第 6/7 行（description / dojo:summary）、第 567 行（h2 第 4 章标题）、第 145/148 行（核心问题第 4 条）：`8 块×12 层` 与 93 层不自洽（8×12=96≠93），同页第 569 行已写明"前 7 个各 12 层、第 8 个 9 层（$93 = 7\times 12 + 9$）"，标题与摘要处的简写易被读成 96 层。｜引文依据：v2 PDF §2.2 `partition its layers into 8 blocks with 12-layer size, giving a partial final block`；config.json `num_hidden_layers=93`。｜修复要求：在标题/摘要的简写后补"（末块 9 层）"，或改写为"7×12+9"以与正文一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（index.html 本体事实、公式、数字、引文编号与来源全部一致；待处理项为 overview.html 与 index.html 的引用编号不一致，以及两处轻微日期/简写问题）

统计：阻断 0 / 重要 1 / 轻微 2