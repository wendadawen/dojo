# 跨层 KV 复用审查记录（第 2 轮）

- 页面版本：89f6b15f585b81ab79acf7ec2eb8c86cc542f897
- 审查时间：2026-09-10
- 审查者：独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（4 问及解答）、常见误解（4 条）、1. 复用的是哪一层的 KV（含本章问题 2 问）、2. 省下多少：层维度被消掉（含本章问题 2 问与「补充：跨层复用与 MLA、压缩注意力的分工」折叠块）、3. 两级复用：共享 KV 与复用索引（含本章问题 2 问）、4. 前提与边界：同组才能共用（含本章问题 2 问与「补充：运行期怎么确认"谁读了谁的 cache"」折叠块）、来源与范围说明（论断与来源（C）/ 公式与来源（F）/ 外部数字与实验条件 / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；overview.html 全文（问题背景 / 核心机制 / 关键结论与边界 / 前置概念）。

## 机械验证结果

- `validate.py`：`/usr/bin/python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` 返回 `validation ok`，退出码 0。
- 引用编号双向一致性：来源章节定义集合 = {C1–C7, F1–F3, N1–N5}（15 个）；正文 `<sup>` 引用集合 = {C1–C7, F1–F3, N1–N5}（15 个）；定义−引用差集 = ∅，引用−定义差集 = ∅。正文实际 sup 列表：`[C7] [C1] [F1] [C2, F2] [C2] [N1] [C3] [F3] [N4] [N5] [C4] [C4] [C6] [C6, N2] [C5, N2] [N3]`。
- 相邻双上标 `<sup>[X][Y]</sup>` 与 `</sup><sup>`：0 处。
- 残留 Unicode 数学字符（定界符 `$...$` 与 `<pre>` 之外）：`→` 1 处、`×` 4 处（详见问题 3）。
- 占位符（TODO/TBD/待补充/待生成/占位/XXX/FIXME）：无。
- 前置概念链接：`../kv-cache/index.html`、`../standard-attention/index.html`、`../mla/index.html`、`../dsa/index.html` 均存在；index.html 与 overview.html 互相链接。

来源核对摘录（关键片段）：

- YOCO Eq.(2) 与共享权重：「First, the output of the self-decoder X^{L/2} generates global KV caches K̂, V̂ for cross-decoder: K̂ = LN(X^{L/2}) W_K, V̂ = LN(X^{L/2}) W_V (2), where W_K, W_V ∈ ℝ^{d×d} are learnable weights.」「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」。W_K/W_V 为单一共享权重，W_Q^l 逐层独立。
- YOCO §2.3：「the number of caches is O(N+CL)…about O(N) caches are required, i.e., you only cache once」「Transformer decoders have to store N × L keys and values…YOCO roughly saves L times GPU memory for caches」；Table 2 = O(LND) vs O((N+L)D)；Table 3 = O(LN²D) vs O(LND)。
- YOCO 脚注 1：「The word "once" refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches…bounded to a constant, which can be ignored compared to global caches when the sequence length is large.」
- YOCO §1 / §4.4：「the memory of KV caches can be reduced by about 80× for 65B models」「YOCO can serve 128K tokens with 1GB GPU memory, while Transformer with GQA can only support 1.6K tokens at 65B model size」。
- 技术报告 §2.2（行 389–396）：「For the upper half layers (i.e., the decoder, l > L/2), the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H^{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^{Z}): C^l = H^{L/2} W_l^{KV}, Z^l = H^{L/2} W_l^{Z}, l > L/2」。
- 技术报告 §2.1（行 327–330）：「CED constructs the decoder's global KV cache from encoder outputs…This nearly halves prefill computation」。
- 技术报告 §2.3.1（行 493–546）：Full/Reindex/Reuse 三模式定义与「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」。
- 技术报告 §1（行 16–19）：「combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching…reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of…DeepSeek-V4-Flash」。
- 官方配置 `config.json`：`kv_source_layers: [2, 8, 14, 20]`；`compress_ratios` 索引 2–19 全为 2、20–39 全为 1。
- `model.py`：`Compressor.__init__`（行 439 `compress_ratio = args.compress_ratios[layer_id]`）；`Indexer.__init__`（行 500–503 `owns_k`/`is_candidate_source`/`uses_candidates`）；`Attention._compress_kv`（行 739–763 源层发布、其余层读取）。
- `verify_csa2_modes.out`：层 2/8/14/20 为 Full 且读到自己发布的 cache；「每个压缩层读到的 cache 都来自其组内 source 层: True」「源层读到自己刚发布的 cache: True」；「层 2 的 cache 被层 2-7 共享 (6 层)」「层 8 的 cache 被层 8-13 共享 (6 层)」「层 14 的 cache 被层 14-19 共享 (6 层)」「层 20 的 cache 被层 20-39 共享 (20 层)」。

## 问题

- [重要·技术] 第 1 章公式符号列表（index.html:811）：`W_K, W_V` 释义中「DeepSeek-V4.1-Flash 的 CED 上段改用逐层各自的投影权重从同一份 $H^{L/2}$ 投出，共用的是同一份 $H^{L/2}$ 及其派生缓存，而不是同一组权重」把「派生缓存」也说成被共享，与 CED 实际机制不符。｜引文依据：技术报告 §2.2（行 389–396）明确为 layer-dependent weights，`C^l = H^{L/2} W_l^{KV}` 因逐层权重不同而逐层不同，CED 层面共享的只有输入 `H^{L/2}`；派生缓存（`C^l`）的跨层共享是 CSA2（§2.3.1）机制，非 CED 本身。｜修复要求：把该句改为「CED 各层用逐层权重 $W_l^{KV},W_l^{Z}$ 从同一份 $H^{L/2}$ 各自投影出各自的 $C^l,Z^l$，共享的是输入 $H^{L/2}$ 而非投影结果；跨层共享派生缓存是 CSA2 的行为」；改后重新对照 §2.2 与 §2.3.1。｜修复：符号释义改为「CED 各层用逐层权重 W_l^{KV}, W_l^{Z} 从同一份 H^{L/2} 各自投影出各自的 C^l, Z^l，CED 层面共享的是输入 H^{L/2} 而不是投影结果；跨层共享派生缓存是 CSA2 的行为」，并补 <sup>[F4]</sup> 引用｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·技术] 第 4 章与来源章节来源归属矛盾（index.html:983 vs 1034、[N2]/[C5]）：正文 983 称分组表「来自运行期实测<sup>[C5, N2]</sup>」，[N2] 标「来源：运行期实测（verify_csa2_modes.py）」；但「构造示例」小节 1034 称「分组表中的层区间为构造示例（数值取自官方配置，非实测运行）」，同一张表被同时标成实测与"非实测运行的构造示例"。｜引文依据：配置 `kv_source_layers=[2,8,14,20]` 声明分组区间；`verify_csa2_modes.out` 行 53–56 运行期输出「层 2 的 cache 被层 2-7 共享 (6 层)」等，两者一致。真实来源是「配置声明 + 运行期（缩放模型）确认」，非"构造示例"。｜修复要求：统一表述为「分组区间由配置 `kv_source_layers=[2,8,14,20]` 声明，运行期实测（等比缩放模型）确认共享关系与分组边界一致」；删除 1034 对配置真实值的"构造示例/非实测"误标，若保留"构造示例"小节仅用于说明缩放模型的验证性质，须改述为「运行期实测使用等比缩小维度的模型」。｜修复：统一为「层区间由配置 kv_source_layers=[2,8,14,20] 声明，运行期实测（等比缩小维度的模型）确认共享关系与分组边界一致」；正文改为「层区间由官方配置声明 [C6]，共享关系由运行期实测确认 [C5, N2]」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 定界符外残留 Unicode 数学字符（validate.py 未报错，但违反 style-guide §11）：index.html:987「属主 → 层号」的 `→`；index.html:1043「论文的 80×」的 `×`；overview.html:48「层数 × 序列长度」的 `×`；另有 index.html:1014「N × L」、index.html:1027「80×」两处位于逐字引文「」内。｜引文依据：不适用（格式规范）。｜修复要求：非引文处改用 LaTeX 或中文——987 改「属主到层号」，1043 改「80 倍」，overview:48 改「层数乘序列长度」；两处引文内 `×` 若保留原文，需在对应 [C]/[N] 条目注明为逐字引用（或改 `$N\times L$`）。｜修复：非引文处「属主 → 层号」改「属主与层号的对应关系」、「论文的 80×」改「论文的 80 倍」、overview「层数 × 序列长度」改「层数乘序列长度」、「N × L」改 $N\times L$；[N1] 内保留英文逐字引文中的 80× 并注明为原文｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] 来源章节缺少 CED 投影权重的登记条目（index.html:811）：正文「逐层各自的投影权重」的机制论断无对应 [C]/[F] 编号，来源章节只有 [C4]（§2.3.1），未登记 §2.2 Eq.(1)。｜引文依据：技术报告 §2.2（行 389–396）`C^l = H^{L/2} W_l^{KV}, Z^l = H^{L/2} W_l^{Z}, l > L/2`，为「逐层投影权重」的依据。｜修复要求：在「公式与来源（F）」或「论断与来源（C）」新增一条登记 §2.2 Eq.(1) 及原文，并在 index.html:811 标注该编号。｜修复：在「公式与来源（F）」新增 [F4] 登记技术报告 §2.2 Eq.（行 389–396）的逐层投影权重原文，并在符号释义处引用｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·可读性] overview.html:56「共享发生在按压缩比划分的组内」与 index.html 第 4 章「压缩比一致只是必要条件、不是分组依据」矛盾（层 2–19 同为压缩比 2 却拆成三组）。｜引文依据：配置 `compress_ratios` 索引 2–19 全为 2，但 `kv_source_layers=[2,8,14,20]` 把其拆为 2–7 / 8–13 / 14–19 三组。｜修复要求：改为「压缩比一致是共用的必要条件；具体分组由配置的 source 层（`kv_source_layers`）声明」，与 index.html 保持一致。｜修复：overview 改为「压缩比一致是共用的必要条件；具体分组由配置的 source 层（kv_source_layers）声明」，与 index 第 4 章一致｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：修复
