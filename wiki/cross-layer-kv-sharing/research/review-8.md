<!-- review-meta
round: 8
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: 08c2ee813d059c46
-->
# 跨层 KV 复用审查记录（第 8 轮）

- 页面版本：index.html `74ebdd27d71eb578b207db67da54621e9e569cb3`（overview.html `542936fd3b1eb33614a30ad850728f0d5e4af074`）
- 审查时间：2026-09-14 17:38
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已核对来源版本：YOCO = arXiv:2405.05254**v2**（PDF 版 https://arxiv.org/pdf/2405.05254v2 与 arXiv HTML 版 https://arxiv.org/html/2405.05254v2 两版对照，2026-09-14 取）；DeepSeek-V4.1-Flash = 官方仓库 `deepseek-ai/DeepSeek-V4.1-Flash` 的 `DeepSeek_V41_Tech_Report.pdf`、HF 版 `config.json`、推理版 `inference/config.json`、`README.md`（2026-09-14 取）。本记录中的 [`Cx`]/[`Fx`]/[`Nx`] 均按页面自身编号回源核对。
- 已完整阅读章节（按顺序）：核心问题（4 条，含解答）→ 常见误解 → 1. 复用的是哪一层的 KV（含本章问题 2 条）→ 2. 省下多少——层维度被消掉（含补充折叠块、本章问题 2 条）→ 3. 两级复用——共享 KV 与复用索引（含本章问题 2 条）→ 4. 前提与边界——同组才能共用（含补充折叠块、本章问题 2 条）→ 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）→ 页面脚本。

## 回源核对结果（无问题项，作为已核对依据）

- [C1] YOCO Abstract「It consists of two components, i.e., a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」；§2「Specifically, YOCO is stacked with L blocks, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」；§2.2「The KV caches K̂,V̂ are reused by all the L/2 cross-decoder modules」——三处原文均命中。
- [F1] §2.2 Eq.(2)「K̂ = LN(X^{L/2})W_K, V̂ = LN(X^{L/2})W_V」「where W_K,W_V ∈ R^{d×d} are learnable weights」——与页面公式一致（v2 PDF 第 2 页）。
- [F2]/[F3] 表编号按 v2 **PDF** 核对：PDF「Table 1: Inference memory complexity of KV caches. N,L,D are the sequence length, number of layers, and hidden dimension.」（O(LND) vs O((N+L)D)）、「Table 2: Prefilling time complexity of attention modules.」（O(LN²D) vs O(LND)）——页面 [F2]→Table 1、[F3]→Table 2 与 PDF 编号一致（注意 arXiv HTML 版把 Figure 3 误渲染为 Table 1，表号整体后移，页面未采用 HTML 编号）。
- [C2] §2.3「the number of caches is O(N+CL) … so about O(N) caches are required, i.e., you only cache once.」「Transformer decoders have to store N×L keys and values during inference. So YOCO roughly saves L times GPU memory for caches compared to Transformer decoders.」——命中。
- [C3] §2.3「we can exit early before entering the cross-decoder during the prefill stage.」「First, only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」——命中。
- [C4] 技术报告 §2.3.1 三段模式与末句「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」——逐句命中。
- [F4] §2.2「the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^{Z})」——命中。
- [N2]/[N3]/分发分组：HF `config.json` 的 `compress_ratios` = [0,0,2,…(层2–19=2)…,1,…(层20–39=1)…]、`kv_source_layer_ids` = [2,8,14,20]（另 `index_source_layer_ids` = [2,8,14,20,24,28,32,36]）；推理版 `inference/config.json` 的 `kv_source_layers` = [2,8,14,20]、`index_source_layers` = [2,8,14,20,24,28,32,36]。第 4 章表（2–7/8–13/14–19/20–39、source 2/8/14/20、压缩比 2/2/2/1）与配置及报告 §4.2.1 一致。
- [N4] §2.1「This nearly halves prefill computation」——命中（§2.1 与 §2.2 各有一处同义表述）。
- [N5] Abstract 原文逐字命中；§6 Conclusion 有同义重述「…reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」——「（§6 有同义重述）」成立。
- 第 4 章「§4.2.1 另有一套按解码模式划分的组」：报告 §4.2.1「The 20 decoder layers … are divided into five groups of four layers. In the first group, the first layer operates in Full Mode, and the remaining three layers operate in Reuse Mode. The remaining four groups … the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode.」——页面转述与原文一致（结论正确，仅缺上标，见下）。
- 补充块「把主 KV 与索引器键值都量化到 FP4」：§2.4.4「DeepSeek-V4 already uses quantization-aware training (QAT) … for FP4 indexer queries and keys … We now extend QAT to the main KV cache」——两项均 FP4，页面表述成立。
- 机械项：`python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` → `validation ok`；`dojo:topics`（注意力机制、内存与缓存）在 `ALLOWED_TOPICS` 内，`dojo:tag`（KV cache）在 `ALLOWED_TAGS` 内；无「（待生成）」占位；`kv-cache` / `mla` / `dsa` / `deepseek-v4-1` / `standard-attention` 前置页均存在；`overview.html` 与 `index.html` 双向互链；页面所用 CSS 类（dg-stack/dg-layer/dg-layer-label/dg-layer-body/dg-flow/dg-node/dg-node-title/dg-node-note/dg-arrow/diagram-caption/callout-blue/learning-goals/chapter-questions/table-scroll）在 `libs/dojo-concept.css` 中均存在。页面无第三方可运行代码，第 3 节「代码」一项不适用；结构图为 HTML div，无位图数值需像素测量；`alt` 无 `$...$`（唯一的 `img#lightboxImg` alt 为空）。

## 问题

- [重要·来源] 第 4 章正文两处、[C5]、[N2] 与补充块：把配置键名 `kv_source_layer_ids` 归给「官方推理配置」，但该键名属 HF 版 `config.json`，官方推理配置中的键名是 `kv_source_layers`｜来源：DeepSeek-V4.1-Flash 官方仓库｜引文依据：`inference/config.json` → `{"kv_source_layers": [2, 8, 14, 20]}`；`config.json` → `{"kv_source_layer_ids": [2, 8, 14, 20]}`（值相同、键名不同，两文件均无对方的键名）｜修复要求：把承载 `kv_source_layer_ids` 这一键名的来源标注改为官方 HF 版 `config.json`（页面 [N3] 已区分「官方推理配置」与「HF 配置」，勿混用），或全页改用推理配置的键名 `kv_source_layers`；保证「键名—所引配置文件」一一对应｜修复：｜复验：
- [重要·来源] [C6] 记「YOCO 脚注 1」，所引版本 arXiv:2405.05254**v2** 的 PDF 中该脚注编号实为 2，与同页 [F2]/[F3] 采用 v2 PDF 表号的口径不一致｜来源：YOCO v2 PDF｜引文依据：v2 PDF 第 2 页正文「First, because YOCO only caches once², the GPU memory consumption of KV caches is significantly reduced.」（`pdftotext -f 2 -l 2 -layout` 输出为「only caches once2」），同页页脚脚注标记为「2」；arXiv LaTeX 源 `main.tex` 中作者行 `\thanks`/`\footnotemark[1]` 之后该 `\footnote{The word "once" refers to global KV cache…}` 为第 2 个（arXiv HTML 版才编为 1）｜修复要求：把「YOCO 脚注 1」改为「YOCO 脚注 2」（按所引 v2 PDF 编号），或删去编号、只保留脚注原文引文｜修复：｜复验：
- [轻微·来源] 第 4 章「技术报告 §4.2.1 另有一套按解码模式划分的组：20–39 这 20 个 decoder 层被划为五组、每组四层…」未加 `[Cx]` 上标，该论断在「论断与来源（C）」中无对应条目，来源双向对应不完整（style-guide §6）｜来源：技术报告 §4.2.1｜引文依据：§4.2.1「The 20 decoder layers … are divided into five groups of four layers. In the first group, the first layer operates in Full Mode … The remaining four groups … the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode.」（页面转述与之一致，仅缺上标）｜修复要求：为该论断补一个 `[Cx]` 上标，并在「论断与来源（C）」登记对应原文引文｜修复：｜复验：
- [轻微·格式] 主要依据写「§2、Eq.(2)、Table 1–3」，而正文只引用 Table 1、Table 2（v2 PDF 编号），Table 3 为评估结果表且全文未引用｜来源：YOCO v2 PDF｜引文依据：v2 PDF「Table 3: Eval Harness … results compared with previous well-trained Transformer language models」（与 §2 复杂度账目无关）｜修复要求：把「Table 1–3」改为「Table 1–2」｜修复：｜复验：
- [轻微·表述] overview.html 写作「**实测** DeepSeek-V4.1-Flash 的共享分组为 2–7、8–13、14–19、20–39」，index.html 同一事实写「官方**配置声明**」；分组来自配置声明而非实测，两页口径不一致｜来源：官方仓库 `inference/config.json` / `config.json`｜引文依据：index.html「官方配置声明的共享分组为 2-7 / 8-13 / 14-19 / 20-39」；配置 `kv_source_layers` = [2,8,14,20]（声明，非测量）｜修复要求：把 overview.html 的「实测」改为「官方配置声明」，与 index.html 及 [N2]/[N3] 口径一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复（2 条重要问题均为来源标注对不上——配置键名归属与脚注编号；其余为主张与核对项全部通过，核心机制、公式、复杂度、分组表与 v2 PDF / 官方配置逐条一致，未发现核心结论级错误）