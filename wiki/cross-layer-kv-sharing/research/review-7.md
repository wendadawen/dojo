<!-- review-meta
round: 7
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: 396da4ce3fb48c5b
-->
# 跨层 KV 复用审查记录（第 7 轮）

- 页面版本：3d4a2c269a589c50f6c9b05cd129cc4d8f009c99（git hash-object wiki/cross-layer-kv-sharing/index.html）
- 审查时间：2026-09-14 16:49
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：核心问题 / 常见误解 / 1. 复用的是哪一层的 KV（含本章问题）/ 2. 省下多少——层维度被消掉（含本章问题）/ 3. 两级复用——共享 KV 与复用索引（含本章问题）/ 4. 前提与边界——同组才能共用（含本章问题）/ 来源与范围说明。另读 overview.html 与页面引用的外部来源（YOCO arXiv:2405.05254v2 全文、DeepSeek-V4.1-Flash 技术报告 PDF、HF 仓库 config.json）。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` → `validation ok`（本页无「（待生成）」占位，前置概念页 kv-cache / mla / dsa / deepseek-v4-1 均真实存在；meta description / dojo:summary / dojo:type=concept / dojo:topics / dojo:tag 齐备）。页面无第三方来源可执行代码，第 3 节「代码」一项不适用。

## 问题

- [重要·技术] 来源与范围说明（[C2]、[F2]、[F3]、[C3]）：YOCO 的表号引用与来源不符。[C2] 与 [F2] 把缓存复杂度对照（Transformer $\mathcal{O}(LND)$ / YOCO $\mathcal{O}((N+L)D)$）标注为 YOCO「Table 2」，但 Table 2 是预填充时间复杂度表，该对照在 Table 1；[F3] 把 prefill 复杂度对照（$\mathcal{O}(LN^2D)$ / $\mathcal{O}(LND)$）标注为「Table 3」，Table 3 是 Eval Harness 结果表，该对照在 Table 2；[C3] 标注「Table 1/3」，其两条引文都出自 §2.3 正文，Table 1/3 均不含这两句。｜引文依据：YOCO v2 Table 1 表注「Inference memory complexity of KV caches. N, L, D are the sequence length, number of layers, and hidden dimension.」，表体 Transformer $\mathcal{O}(LN D)$、YOCO $\mathcal{O}((N + L)D)$；Table 2 表注「Prefilling time complexity of attention modules. N, L, D are the same as above.」，表体 Transformer $\mathcal{O}(LN^2D)$、YOCO $\mathcal{O}(LN D)$；Table 3 表注「Eval Harness results compared with previous well-trained Transformer language models」。｜修复要求：把 [C2]、[F2] 的「Table 2」改为「Table 1」，[F3] 的「Table 3」改为「Table 2」，[C3] 的「Table 1/3」改为「§2.3（Figure 3）」或删去表号；正文第 168 行 `[C2, F2]` 与第 180 行 `[C3]`/`[F3]` 的正文引用位置不变。｜修复：｜复验：

- [重要·技术] 正文第 281、295、299 行与来源 [C5]、[N2]（共 5 处 `<code>kv_source_layers=[2,8,14,20]</code>`）：页面以代码体引用的配置字段名不存在。官方配置中生产层列表的键名是 `kv_source_layer_ids`，全文没有 `kv_source_layers` 这个键；数值 [2,8,14,20] 正确。｜引文依据：HF 仓库 deepseek-ai/DeepSeek-V4.1-Flash 的 config.json：「"kv_source_layer_ids": [2, 8, 14, 20]」；同文件另有「"index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]」「"candidate_source_layer_id": 20」，键名中含 "kv" 的仅 `kv_source_layer_ids`。｜修复要求：把 5 处 `kv_source_layers`（含 `kv_source_layers=[2,8,14,20]`）改为 `kv_source_layer_ids`，数值与文字说明不变。｜修复：｜复验：

- [轻微·技术] 来源与范围说明（[N5]）：「每 token 全局 KV 缓存 890 字节」标注为「技术报告 §1」。该句实际出自报告 Abstract；§1 Introduction 只转述了 1/4、1/8 的比例，未出现 890。｜引文依据：技术报告 Abstract「...reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」；全文检索「890」仅命中 Abstract 与 §6 Conclusion；§1 中对应句为「requires only approximately 1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage」。｜修复要求：把「技术报告 §1（§6 Conclusion 有同义重述）」改为「技术报告 Abstract（§6 Conclusion 有同义重述）」。｜修复：｜复验：

- [轻微·表述] 第 76 行（核心问题解答一）与第 161 行（第 1 章本章问题解答二）用「每段一份」，第 186 行表格与来源章节「辅助解释与类比边界」用「每组一份」：同一份数关系在页内出现两种称呼。页面已把「段」定义为「下段/上段」，因此「全局 KV 的份数从『每层一份』变成『每段一份』」会被读成「上下段各一份」（2 份），与实际按生产层分成的 4 组不符。｜引文依据：不适用。｜修复要求：把两处「每段一份」统一为「每组一份」，与第 186 行表格用词一致。｜修复：｜复验：

- [轻微·表述] 第 2 章折叠块（第 196 行）：称 DeepSeek-V4.1-Flash 的跨层复用为「按 ratio 分组共享」，与第 4 章第 295 行「分组边界即这份列表给出的区间——分组由配置给定，不能从压缩比推出」相互矛盾。｜引文依据：不适用（同页两处矛盾）。｜修复要求：改为「按配置声明的分组共享」或「与压缩比一致的层按分组共享」。｜修复：｜复验：

- [轻微·技术] 第 4 章表格（第 287–290 行）与第 97 行：表格把 20 个 decoder 层（20–39）列为单一组（层数 20），页面并称「真正的分组由配置里的生产层列表直接声明」。技术报告 §4.2.1 另把这 20 层划为五组、每组四层。页面未说明表格给的是「主 KV 来源分组」，读者交叉核对报告时会把两者当成同一套「分组」。｜引文依据：报告 §4.2.1「The 20 decoder layers use CSA2 with a compression rate of m = 1. These layers are divided into five groups of four layers. In the first group, the first layer operates in Full Mode, and the remaining three layers operate in Reuse Mode. The remaining four groups share the same configuration: the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode.」；HF config.json 中 `index_source_layer_ids` 为 8 项（[2,8,14,20,24,28,32,36]），与 8 个模式组首层对应。｜修复要求：在表格标题或表下补一句，说明该表列的是主 KV 来源分组（由 `kv_source_layer_ids` 给出的区间），并注明报告另按模式把 20 个 decoder 层分为五组四层（四组首层为 Reindex）。｜修复：｜复验：

## 本轮核对通过的关键项（供复验参考）

- YOCO [C1] 三条引文与 Abstract、§2 原文逐字一致；[C6] 脚注 1 与原文逐字一致；[C2]「the number of caches is O(N+CL) ... i.e., you only cache once.」「Transformer decoders have to store N×L keys and values ... roughly saves L times GPU memory」与 §2.3 原文一致；[C3]「we can exit early before entering the cross-decoder during the prefill stage.」「First, only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」与 §2.3 原文一致。
- [F1] $\hat K=\mathrm{LN}(X^{L/2})W_K$、$\hat V=\mathrm{LN}(X^{L/2})W_V$ 与 YOCO Eq.(2) 及「where $W_K, W_V \in \mathbb{R}^{d\times d}$ are learnable weights」一致；[F2]/[F3] 的两个复杂度式数值本身与 YOCO Table 1 / Table 2 表体一致（仅表号错，见上）。
- [N1] 「the memory of KV caches can be reduced by about 80× for 65B models.」在 §1，「YOCO can serve 128K tokens with 1GB GPU memory ... at 65B model size.」在 §4.4，两处定位正确。
- [C4] 四条引文与报告 §2.3.1 逐字一致；[F4] 引文与 §2.2 逐字一致；[N4]「This nearly halves prefill computation」在 §2.1；[N5] 890 字节句与 Abstract/§6 一致，且与「跨层复用 + FP4 KV 缓存」两项并列的归因一致。
- [N3] compress_ratios 与 HF config.json 一致：`[0, 0, 2×18, 1×20, 0, 0, 0]`，即层 2–19 = 2、层 20–39 = 1（前两个 0 对应纯 SWA 的层 0–1，末尾三个 0 对应 MTP 层）。[C5]/[N2] 的分组区间 2–7 / 8–13 / 14–19 / 20–39 与 `kv_source_layer_ids` 的区间一致，source 层 2 / 8 / 14 / 20 正确。
- 折叠块第 196 行「把主 KV 与索引器键值都量化到 FP4」「890 字节不能只归因于跨层复用」有来源支持（报告 §2.4.4「DeepSeek-V4 already uses QAT ... for FP4 indexer queries and keys ... We now extend QAT to the main KV cache」；Abstract 两因素并列）。
- 数学符号全部由 KaTeX 渲染，未发现 Unicode 数学字符直接出现（validate.py 通过）；`dojo:summary` 的公式为 KaTeX 可渲染语法；结构图为 HTML（`.dg-stack` / `.dg-flow`），图注定义了节点与箭头含义，无 `<text>` ASCII 近似写法；交互视图（目录、折叠、主题）有脚本且折叠用原生 `<details>`；两级问题块命名正确、每题都有解答折叠块且答案独立可读。
- 未发现元话语套语、会话指代（我/我们/你）、调试叙事或临场评价；「本页…」的自称出现在 style-guide 允许的范围说明与固定小节内。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（修完后按 §4 重新核对来源并重跑 validate.py；本轮无阻断项，重要项须全部关闭）
