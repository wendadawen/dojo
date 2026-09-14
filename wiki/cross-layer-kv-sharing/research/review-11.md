<!-- review-meta
round: 11
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: 6b9f3502e612f3df
-->
# 跨层 KV 复用（Cross-Layer KV Sharing）审查记录（第 11 轮）

- 页面版本：index.html 工作树哈希 5f3b4544b4471e84bc007414d1b7f495213d8e3b（sha256 前 16 位 c27e09d179b7b645）
- 审查时间：2026-09-14 18:06
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 复用的是哪一层的 KV（含本章问题）/ 2. 省下多少——层维度被消掉（含本章问题与「补充：跨层复用与 MLA、压缩注意力的分工」）/ 3. 两级复用——共享 KV 与复用索引（含本章问题）/ 4. 前提与边界——同组才能共用（含本章问题与「补充：分组与 source 层是怎么定的」）/ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）
- 核对用的来源版本：YOCO arXiv:2405.05254v2（PDF 正文已下载逐段核对，脚注编号按该版）；DeepSeek-V4.1-Flash 技术报告 PDF（HF 仓库 deepseek-ai/DeepSeek-V4.1-Flash 的 DeepSeek_V41_Tech_Report.pdf，1,809,802 字节 / 51 页，LFS 实体已取回）；官方 HF 版 config.json 与官方推理配置 inference/config.json（均直接读取）。overview.html 已读并逐项与正文比对。
- 机械验证：`.dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` → validation ok（exit 0）；页面全部 href/src（含 ../../libs/*、../kv-cache、../mla、../dsa、../deepseek-v4-1、overview.html）逐一解析，无缺失文件；正文无「我们/你/我」，无「下面来看」「需要注意的是」类元话语。

## 已核对并通过的关键条目（引文依据）

- [C1] YOCO Abstract/§2：v2 PDF 原文「We introduce a decoder-decoder architecture, YOCO」「because the cross-decoder reuses the outputs of self-decoder, we can exit early…」；§2.2（PDF 偏移 10588）「The KV caches ˆK, ˆV are reused by all the L/2 cross-decoder modules」。「decoder-decoder」确为论文自称，与正文一致。
- [C2] §2.3（偏移 11499）「the number of caches is O(N +CL)」「aboutO(N) caches are required, i.e., you only cache once.」「Transformer decoders have to store N×L keys and values during inference. So YOCO roughly savesL times GPU memory for caches compared to Transformer decoders.」逐字一致。
- [C3] §2.3（偏移 12283）「we can exit early before entering the cross-decoder during the prefill stage.」「First, only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」逐字一致。
- [C6] 脚注编号：v2 PDF 该脚注标号为「2」（正文「2The word “once” refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches…」），与页面「YOCO 脚注 2（按所引 v2 PDF 编号）」一致。
- [F1] §2.2 Eq.(2)「ˆK = LN(X L/2)WK, ˆV = LN(X L/2)WV (2) whereWK,WV ∈ Rd×d are learnable weights.」与页面公式、量纲说明一致。
- [F2]/[F3] Table 1「Transformer O(LND) / YOCO O((N +L)D)」、Table 2「Transformer O(LN 2D) / YOCO O(LND )」，表注「N,L,D are the sequence length, number of layers, and hidden dimension.」与页面复杂度对照、符号定义一致。
- [N1] §1 Introduction（偏移 6355）「the memory of KV caches can be reduced by about 80× for 65B models.」；§4.4 Inference Advantages（偏移 29717）「YOCO can serve 128K tokens with 1GB GPU memory … at 65B model size.」章节定位正确。
- [C4] 报告 §2.3.1 逐字一致：「Full Mode.The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」「Reindex Mode.The layer reuses the most recent available main KV from a preceding layer together with its corresponding indexer K.」「It performs attention using this selection without computing indexer Q or evaluating index scores.」「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the(L/2)-th layer, i.e. the last layer of the causal encoder.」另核对到「In all three modes, each layer retains its own global Q and SWA KV」，支持页面「各层仍有自己的 query 与本层窗口 KV」。
- [F4] 报告 §2.2（偏移 25342）「the KV entries are not derived from their respective hidden statesHl. Instead, they are projected directly from the hidden state of the(L/2)-th layer,H L/2, using layer-dependent projection weights (W KV l andW Z l )」，章节定位正确。
- [C7] 报告 §4.2.1 Model Setups（偏移 68093，位于 4.2.1 与 4.2.2 之间）「The 20 decoder layers use CSA2 with a compression rate ofm = 1. These layers are divided into five groups of four layers. In the first group, the first layer operates in Full Mode, and the remaining three layers operate in Reuse Mode. The remaining four groups share the same configuration: the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode.」逐字一致。同段还给出编码器「18 encoder layers … compression rate ofm= 2 … three identically configured groups of six layers」，与表中 2–7 / 8–13 / 14–19 三组边界吻合，页面未与之冲突。
- [N4] 报告 §2.1（偏移 20803，位于 2.1 Overview 与 2.2 之间）「This nearly halves prefill computation」，章节定位正确。
- [N5] 报告 Abstract 与 §6 Conclusion（偏移 117748）：「combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」两项并列的因果表述与页面「不能只归因于跨层复用」一致；§2.4.4「We adopt … MXFP4 … We now extend QAT to the main KV cache」「FP4 indexer queries and keys」支持页面「把主 KV 与索引器键值都量化到 FP4」。
- 共享分组：HF 版 config.json `kv_source_layer_ids=[2,8,14,20]`、`num_hidden_layers=40`、`compress_ratios` 43 项（前 2 项 0、中间 18 项 2、中间 20 项 1、末 3 项 0）→ 层 2–19 为 2、层 20–39 为 1；官方推理配置 inference/config.json `kv_source_layers=[2,8,14,20]`、`compress_ratios` 同值。页面表格（2–7/8–13/14–19/20–39，层数 6/6/6/20，压缩比 2/2/2/1，source 2/8/14/20）与之一致，字段名两版差异（kv_source_layer_ids 对 kv_source_layers）与页面说明一致。
- 全页一致性：description、dojo:summary、overview.html、正文与图注四处的复杂度式（O(LND)→O((N+L)D)、O(LN²D)→O(LND)）、分组区间、source 层、890 字节、「约省 L 倍」逐项相同，未发现同页两处矛盾；公式符号（X^{L/2}、H^{L/2}、L、N、D、r、K̂、V̂）全文单义；图（dg-stack、dg-flow）为 HTML 结构，图注读数与图文一致，含公式的 span 由 auto-render 渲染，无 `<text>` ASCII 近似；折叠块（details）不依赖脚本即可展开。

## 问题

- [轻微·可读性] 核心问题 3 的解答（第 90 行）：末句「因此“复用索引”不等于“看到一样的东西”——各层的 query 与本层窗口 KV 仍不同，选出的位置也可能不同。」与第 3 章自身定义冲突。｜引文依据：报告 §2.3.1「Reuse Mode.The layer reuses the most recent available main KV and the latest Top-K indices computed against that main KV by a preceding layer in Full or Reindex Mode.」——复用索引即沿用同一份 Top-K，选中位置按定义相同；页面第 3 章「复用索引：让多层沿用同一份 Top-K 选择结果」及第 3 章本章问题 2 的解答「即便选中的位置集合完全相同，各层的输出也不同」均与末句相反。｜修复要求：删去「选出的位置也可能不同」，或改为限定表述「（Reindex 层会重新打分，故位置可在层间变化）」，使结论与第 3 章定义一致。｜修复：｜复验：
- [轻微·可读性] 第 1 章符号说明（第 123 行 CED 那条 bullet）：「投影出各自的 $C^l, Z^l$」中的 $C^l$、$Z^l$ 全页未定义，读者无法判断二者是主 KV 条目还是别的量。｜引文依据：报告 §2.2 Eq.(1) 下文「whereC andZ represent the KV entries and their corresponding compression weights, respec- tively.」｜修复要求：在该 bullet 内补出「（$C^l$ 为该层主 KV 条目、$Z^l$ 为其对应的压缩权重）」，或把它们加入第 1 章的符号列表。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。阻断与重要均无；两条轻微问题不影响核心结论与来源一致性（一条为核心问题解答的措辞与第 3 章定义不齐，一条为符号未解释），按 check.md §5「遗留轻微问题具有明确的接受理由」可在本轮修复后发布：两者均为删除或补一句即可关闭的局部措辞，不触及公式、数字与来源引用。