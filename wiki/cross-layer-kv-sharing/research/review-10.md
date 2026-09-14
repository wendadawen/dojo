<!-- review-meta
round: 10
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: 399c975f9e0626dd
-->
# 跨层 KV 复用（Cross-Layer KV Sharing）审查记录（第 10 轮）

- 页面版本：fb2ac1bb0f7a54e36416298f6537b7d4b9d7d81f
- 审查时间：2026-09-14 17:57
- 审查者：独立子代理（未参与写作与任何前序轮次）
- 类型判定：`dojo:type=concept` → 规范 guides/concept/check.md
- 已完整阅读章节：核心问题、常见误解、1. 复用的是哪一层的 KV、2. 省下多少——层维度被消掉、3. 两级复用——共享 KV 与复用索引、4. 前提与边界——同组才能共用、来源与范围说明（含全部折叠块与两处图注）
- 核对所用来源与版本：
  - YOCO，arXiv:2405.05254v2（核对 arxiv.org/html/2405.05254v2 与 arxiv.org/pdf/2405.05254v2 抽取文本；PDF 逐句核对）
  - DeepSeek-V4.1-Flash 技术报告（https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash 的 DeepSeek_V41_Tech_Report.pdf，抽取文本逐句核对 Abstract、§2.1、§2.2、§2.3.1、§2.4.4、§4.2.1、§6）
  - 官方 HF 版 config.json 与官方推理配置 inference/config.json（原始 JSON 字段核对）
- 机械验证：`.dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` → validation ok；内部链接 ../kv-cache、../mla、../deepseek-v4-1、../dsa、../standard-attention、overview.html 均实际存在；index↔overview 双向链接成立；diagram 仅用 HTML/CSS（.dg-stack/.dg-flow 等类均在 libs/dojo-concept.css 中存在），无脚本亦可读；无 `<img>` 承载公式，alt 中无 `$...$`；无 Unicode 数学字符（validate 通过）。

## 已回源核对且与本页一致的条目（抽样记录，供复验）

- [C1] YOCO 摘要「a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」；§2「YOCO is stacked with L blocks, where the first L/2 layers are self-decoder...」；「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」——一致。
- [F1] YOCO Eq.(2)「K̂ = LN(X^{L/2})W_K, V̂ = LN(X^{L/2})W_V」及「W_K, W_V ∈ R^{d×d} are learnable weights」——一致（页将 d 记为 D，已注明）。
- [F2]/[F3] 表号核对：arXiv v2 PDF 中 Table 1 = 「Inference memory complexity of KV caches. N, L, D are the sequence length, number of layers, and hidden dimension.」（O(LND) vs O((N+L)D)）；Table 2 = 「Prefilling time complexity of attention modules.」（O(LN²D) vs O(LND)）。页面 [F2] 引 Table 1、[F3] 引 Table 2 与 PDF 版本一致。
- [C2]/[C3] §2.3「Saving GPU Memory...」「Reducing Prefilling Time...」：原文「the number of caches is O(N+CL)... about O(N) caches are required, i.e., you only cache once.」「YOCO roughly saves L times GPU memory for caches compared to Transformer decoders.」「we can exit early before entering the cross-decoder during the prefill stage」「at least half prefilling latency reduction」——一致。
- [N1] §1「the memory of KV caches can be reduced by about 80× for 65B models.」；§4.4「YOCO can serve 128K tokens with 1GB GPU memory」——一致（章节号已核）。
- [C6] PDF 脚注「The word "once" refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches. As the self-decoder utilizes an efficient attention module, the cache size is bounded to a constant, which can be ignored compared to global caches when the sequence length is large.」——一致。
- [F4] 报告 §2.2「the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^{Z}): C^l = H_{L/2}W_l^{KV}, Z^l = H_{L/2}W_l^{Z}」——一致（页据此区分 CED 共享「输入」而非投影结果，正确）。
- [N4] 报告 §2.1「This nearly halves prefill computation」（§2.2 有同义重述）——一致。
- [C4] 报告 §2.3.1 三模式定义（Full/Reindex/Reuse）与「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」——逐句一致。
- [C7] 报告 §4.2.1「The 20 decoder layers use CSA2 with a compression rate of m = 1. These layers are divided into five groups of four layers...」——一致。
- [N5] 报告 Abstract/§6「combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」——一致。
- [C5]/[N2]/[N3] 字段核对：官方 HF 版 config.json → `kv_source_layer_ids=[2,8,14,20]`（无 `kv_source_layers`）；官方推理配置 inference/config.json → `kv_source_layers=[2,8,14,20]`，`compress_ratios` 在索引 2–19 为 2、20–39 为 1；HF 版 `text_config.compress_ratios` 同值——与页面「压缩比 2/2/2/1、source 2/8/14/20」及「两版关键字段取值一致、个别字段名不同」一致。
- 数字一致性：正文、summary（`dojo:summary`）、description、overview 与图注的 O(LND)/O((N+L)D)/O(LN²D)/O(LND)、分组 2–7/8–13/14–19/20–39、source 2/8/14/20、890 字节均一致，无相互矛盾。

## 问题

- [重要·技术] 第 3 章 figcaption（"箭头表示复用的传递方向…复用 KV 不要求复用索引，反之亦然。"）：图注断言两级复用**双向**独立——"复用索引不要求复用 KV"；来源无任何「只复用索引、不复用 KV」的模式，且本页第 3 章正文与 overview 自述相反条件，构成来源不支持的双向断言 + 同页矛盾。｜引文依据：报告 §2.3.1「Reuse Mode. The layer reuses the most recent available main KV and the latest Top-K indices computed against that main KV by a preceding layer in Full or Reindex Mode.」（索引与主 KV 捆绑）；本页第 3 章正文「索引是"对某一份 KV 打分后选出的位置"，因此它必须与它评分的那份 KV 配套」；overview 亦写「成立条件：索引必须与它评分的那份 KV 配套」。｜修复要求：删去「反之亦然」，把图注改为只保留可回源的单向结论——复用 KV 不要求复用索引（Reindex 模式），而复用索引必然连同其配套 KV 一起复用；使图注与第 3 章正文、overview 一致。｜修复：｜复验：
- [轻微·表述] 第 1 章正文与第 3 章正文：段首固定句式「关键在于：」重复出现两次（"关键在于：$\hat K, \hat V$ 只算一次…" 与 "关键在于：<b>复用索引不等于看到一样的东西</b>…"）。｜引文依据：不适用。｜修复要求：其中一处改用非固定句式的直接陈述（如"因此，…"或去掉该引导语），避免同一引导句跨章复用。｜修复：｜复验：
- [轻微·技术] 第 2 章折叠块《补充：跨层复用与 MLA、压缩注意力的分工》末尾："这几个因素缺一不可，890 字节不能只归因于跨层复用。"——把来源的两项归因（跨层复用 + FP4）扩写为三项并断言"缺一不可"，"缺一不可"这一判断在来源中无对应表述。｜引文依据：报告 Abstract「combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching」，仅并列两项；本页 [N5] 亦记「报告把成因列为「跨层复用 + FP4 KV 缓存」两项并列」。｜修复要求：将该句标记为本页的进一步拆分（如"本页把来源的两项再拆开看：…"），或删去"缺一不可"，不把本页判断写成报告结论。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复
