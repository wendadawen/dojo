<!-- review-meta
round: 6
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: d2553194463412ff
-->
# 跨层 KV 复用（Cross-Layer KV Sharing）审查记录（第 6 轮）

- 页面版本：bb69587dac5944d86e17e71e37d5a3ad6f0a53e9
- 审查时间：2026-09-13 21:08
- 审查者：独立子代理
- 已完整阅读章节：核心问题、常见误解、1. 复用的是哪一层的 KV、本章问题、2. 省下多少——层维度被消掉、本章问题、3. 两级复用——共享 KV 与复用索引、本章问题、4. 前提与边界——同组才能共用、本章问题、来源与范围说明

## 来源核对（本轮实际打开并定位）

- YOCO（arXiv:2405.05254v2，全文本 arxiv.org/html/2405.05254v2）：Abstract「a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」；§2「YOCO is stacked with L blocks, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」；§2.2 Eq.(2)「K̂ = LN(X^{L/2})W_K, V̂ = LN(X^{L/2})W_V where W_K, W_V ∈ ℝ^{d×d} are learnable weights.」「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」；Table 2（KV Cache Memory：Transformer 𝒪(LND)、YOCO 𝒪((N+L)D)，表注 N/L/D = 序列长度/层数/隐藏维）、Table 3（Prefilling Time：Transformer 𝒪(LN²D)、YOCO 𝒪(LND)）；§2.3「the number of caches is 𝒪(N+CL)… about 𝒪(N) caches are required, i.e., you only cache once」「roughly saves L times GPU memory for caches」「we can exit early before entering the cross-decoder during the prefill stage」「only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction」；§1「the memory of KV caches can be reduced by about 80× for 65B models」；§4.4「YOCO can serve 128K tokens with 1GB GPU memory … at 65B model size」；脚注 1「The word “once” refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches. As the self-decoder utilizes an efficient attention module, the cache size is bounded to a constant, which can be ignored compared to global caches when the sequence length is large.」。全部与页面 [C1][C2][C3][C6][F1][F2][F3][N1] 逐字一致，定位（§2、§2.3、§1、§4.4）正确。
- DeepSeek-V4.1-Flash 技术报告（huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash 的 DeepSeek_V41_Tech_Report.pdf，已下载 pdftotext 取文）：§2.1「This nearly halves prefill computation」（[N4] 正确，属报告宣称）；§2.2「the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^{Z})」，Eq.(1) C^l=H^{L/2}W_l^{KV}, Z^l=H^{L/2}W_l^Z（[F4] 正确）；§2.3.1 三种模式引文与 [C4] 逐字一致（Full/Reindex/Reuse 三段及「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」）；§1（与 Abstract 同）「…reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash」，§6 Conclusion 有同义重述（[N5] 正确）；§2.4.4「DeepSeek-V4 already uses quantization-aware training (QAT) for FP4 indexer queries and keys … We now extend QAT to the main KV cache」（第 2 章补充「把主 KV 与索引器键值都量化到 FP4」有据）。
- 官方配置：HF `config.json` 的 `text_config.compress_ratios` = [0,0,2(×18),1(×20),0,0,0]（层 2–19 为 2、层 20–39 为 1）、`kv_source_layer_ids` = [2,8,14,20]；`inference/config.json` 的字段名正是 `kv_source_layers` = [2,8,14,20] 与 `compress_ratios`（[C5][N2][N3] 的字段名与取值均正确）。报告 §4.2.1「The remaining 18 encoder layers use CSA2 with a compression rate of m = 2 … divided into three identically configured groups of six layers. In each group, the first layer operates in Full Mode …」，据此 Full 层（= 主 KV 生产者）为 2、8、14、20，与 `kv_source_layers` 一致；第 4 章表的层区间（2–7 / 8–13 / 14–19 / 20–39，层数 6/6/6/20，压缩比 2/2/2/1，source 2/8/14/20）与之一致（解码器所有层的主 KV 均来自 Full 层 20，故 20–39 为一个共享组），非误。
- 内部链接：../kv-cache/index.html、../mla/index.html、../dsa/index.html、../deepseek-v4-1/index.html、overview.html、../../index.html 均存在；无指向不存在文件的路径；`dojo:topics`=注意力机制,内存与缓存 与 `dojo:tag`=KV cache 均在词表内；`.dojo/scripts/validate.py` 返回 validation ok。

## 问题

- [轻微·表述] 位置：1. 复用的是哪一层的 KV，符号列表 $X^{L/2}$ 项｜问题：把 $X^{L/2}$ 括注为「整段上下文的摘要」，来源只称其为中间表示，且 $X^{L/2}$ 是逐位置表示（同章因果性问题亦按逐位置处理），"摘要"是未登记的类比，容易被读成单一聚合向量｜引文依据：YOCO Fig.10 说明「M denotes the intermediate representation X^{L/2}, i.e., the output of self-decoder」，全文未见 "summary" 用法｜修复要求：改用来源用词（如"下段的中间表示"），或将该类比登记进「辅助解释与类比边界」｜修复：｜复验：
- [轻微·表述] 位置：核心问题第 2 题解答｜问题：归因句「prefill 阶段上段不必为每个位置计算全局 KV，可以在进入上段前提前退出」不准确——上段（交叉解码器）本就不计算全局 KV，提前退出省下的是上段对全部位置的运算；同页第 2 章「原文：既然上段复用了下段的输出，…提前退出」给出的是正确归因，两处归因不一致｜引文依据：YOCO §2.3「because the cross-decoder reuses the outputs of self-decoder, we can exit early before entering the cross-decoder during the prefill stage.」｜修复要求：将该句归因与第 2 章对齐，去掉"计算全局 KV"的说法｜修复：｜复验：
- [轻微·表述] 位置：4. 前提与边界，本章问题第 2 题解答｜问题：「上段各层仍然能看到同样的全局位置集合（由 Top-K 选择决定）」与第 3 章「Reindex 层用自己的 query 重新打分、选出新的 Top-K」相冲突，会让人误以为上段各层可见位置集合总是一致（实际 Reuse 层沿用、Reindex 层可不同）｜引文依据：报告 §2.3.1「Reindex Mode … produces fresh Top-K indices. This allows the sparse selection to change across layers while main KV and indexer K remain shared.」｜修复要求：改为不暗示集合一致的表述，例如"可见范围不因复用而缩短；各层可见的全局位置集合仍由各自的 Top-K 选择决定"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（0 阻断 / 0 重要；3 项轻微表述问题建议随下一轮修复，或其接受理由记录在案）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
