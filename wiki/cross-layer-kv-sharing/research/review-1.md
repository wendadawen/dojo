<!-- review-meta
round: 1
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: bf92e817ddc1423a
-->
# 跨层 KV 复用（Cross-Layer KV Sharing）审查记录（第 1 轮）

- 页面版本：c4db1b4ae57c4ab07af83614cb7b50773fbb4b0f
- 审查时间：2026-09-10 16:24
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：核心问题、常见误解、1. 复用的是哪一层的 KV（含本章问题）、2. 省下多少：层维度被消掉（含补充分工折叠块、本章问题）、3. 两级复用：共享 KV 与复用索引（含本章问题）、4. 前提与边界：同组才能共用（含补充运行期折叠块、本章问题）、来源与范围说明（核心论断与来源、核心公式与来源、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）；另校对 overview.html 全文。

## 机械验证结果

**1. validate.py**

```
$ /usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py /Users/wendadawen/code/dojo/wiki/cross-layer-kv-sharing/index.html
validation ok: /Users/wendadawen/code/dojo/wiki/cross-layer-kv-sharing/index.html
EXIT=0
```

通过。`dojo:topics`（注意力机制、内存与缓存）、`description`（纯文本）、`dojo:summary`（行内公式、`$` 配对）、`dojo:type=concept`、`dojo:tag` 均在词表内。

**2. 页面内可运行代码块**

本页不含声称可内联运行的代码块（正文代码仅出现在 `<code>` 行内标识，如 `inference/config.json`、`model.py`、`verify_csa2_modes.py`）。因此无可实跑对象。运行期实测来源 `verify_csa2_modes.py` 属外部来源文件，已按其存档输出核对（见 [C5] 条），未在本轮重新执行（本机缺少该脚本所需的缩放 checkpoint 装配环境不属于本页提供物，且脚本位于 deepseek-v4-1 来源目录）。

**3. 链接检查**

- index.html → overview.html：`href="overview.html"` 存在。
- overview.html → index.html：`href="index.html"` 存在。
- overview.html 前置概念链接 `../kv-cache/index.html`、`../standard-attention/index.html`、`../mla/index.html`、`../dsa/index.html`：四个目标文件均存在。
- 本地资源 `../../libs/katex.min.css`、`katex.min.js`、`auto-render.min.js`、`prism*.js/css`：libs 目录均存在。

**4. 来源逐条核对（按 check.md §2.2 四步）**

| 编号 | 页面标注 | 定位结果 | 原文片段 / 数值 |
|---|---|---|---|
| [C1] | YOCO Abstract 与 §2 | 命中 | Abstract：「a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention.」；§2：「YOCO is stacked with L layers, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」 |
| [C2] | YOCO §2.3 与 Table 2 | 命中 | 「the number of caches is O(N+CL) ... about O(N) caches are required, i.e., you only cache once.」；「Transformer decoders have to store N × L keys and values during inference. So YOCO roughly saves L times GPU memory for caches compared to Transformer decoders.」 |
| [C3] | YOCO §2.3 与 Table 1/3 | 命中 | 「we can exit early before entering the cross-decoder during the prefill stage.」；「only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」 |
| [C4] | 技术报告 §2.3.1（tech_report.txt 行 490–545） | 命中，逐字一致 | 「Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices.」；「Reindex Mode. The layer reuses the most recent available main KV from a preceding layer together with its corresponding indexer K.」；「Reuse Mode. ... performs attention using this selection without computing indexer Q or evaluating index scores.」；「When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the (L/2)-th layer」。均在行 497–546 内。 |
| [C5] | 运行期实测 verify_csa2_modes.py / verify_csa2_modes.out | 命中 | 存档输出：「每个压缩层读到的 cache 都来自其组内 source 层: True  (不符: [])」「源层读到自己刚发布的 cache: True」；「层 2 的 cache 被层 2-7 共享 (6 层)」「层 8 … 8-13」「层 14 … 14-19」「层 20 … 20-39 (20 层)」 |
| [C6] | 官方配置 compress_ratios + kv_source_layers | 命中（但语义问题见问题 2） | config.json：`"kv_source_layers": [2, 8, 14, 20]`；`"compress_ratios": [0,0,2×18,1×20,0,0,0]`（层 2–19 为 2、层 20–39 为 1）。model.py `Compressor.__init__` 第 439 行 `compress_ratio = args.compress_ratios[layer_id]`。 |
| [C7] | YOCO 脚注 1 | 命中 | 「The word 'once' refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches. As the self-decoder utilizes an efficient attention module, the cache size is bounded to a constant, which can be ignored compared to global caches when the sequence length is large.」 |
| [F1] | YOCO Eq.(2) 说明 | 命中 | 「First, the output of the self-decoder X^{L/2} generates global KV caches K̂,V̂ for cross-decoder: K̂=LN(X^{L/2})W_K, V̂=LN(X^{L/2})W_V (2) where W_K,W_V∈R^{d×d} are learnable weights.」 |
| [F2] | YOCO Table 2 与 §2.3 | 命中 | Table 2：Transformer `O(LND)`；YOCO `O((N+L)D)`。图注「N,L,D are the sequence length, number of layers, and hidden dimension.」 |
| [F3] | YOCO Table 3 与 §2.3 | 命中 | Table 3：Transformer Prefilling `O(LN²D)`；YOCO `O(LND)`。 |
| [N1] | 称来源为「同 [C2] 论文 §4.4」 | 部分命中：数值对、章节归属错 | 80×：「the memory of KV caches can be reduced by about 80× for 65B models.」实际在 **§1 Introduction**；128K/1GB：「YOCO can serve 128K tokens with 1GB GPU memory ... at 65B model size.」在 §4.4。页面把两者都归到 §4.4。 |
| [N2] | 运行期实测 | 命中 | 分组 2–7 / 8–13 / 14–19 / 20–39，source 2 / 8 / 14 / 20，与 .out 输出一致。 |
| [N3] | 官方配置 compress_ratios | 命中 | 层 2–19 为 2、层 20–39 为 1；与 config.json 一致。 |
| [N4] | 称「§2.1（tech_report.txt 行 322–326）」 | 原文对、行号错 | 「This nearly halves prefill computation」实际在 **行 329**（行 322–326 是「DeepSeek-V4.1-Flash.」与「The Causal Encoder–Decoder (CED) architecture and Compressed Sparse Attention 2 (CSA2)」及页眉 7，不含该句）。 |

**5. 结构计数**

`<details>` 14 个（核心问题 4 + 四章各 2 = 12，加 2 个「补充：」）；`class="chapter-questions"` 5 处（核心问题 + 四章）；2 个 `<figure class="diagram">` 均为 HTML 结构（`.dg-stack`/`.dg-flow`），无内联 SVG，无等宽框线字符；`<summary>` 前缀统计：`补充：`×2、`解答：`×12，全部合规。

## 问题

- [重要·技术] 第 4 章前提段与分组表：页面把分组规则说成「按压缩比划分的组内」，但表中第 1–3 组（层 2–7、8–13、14–19）压缩比全为 2；按页面自述规则这 18 层应合成一组，与表格三组自相矛盾。真正的分组由配置 `kv_source_layers=[2,8,14,20]` 直接声明，压缩比一致只是必要非充分条件。｜引文依据：页面「因此"共享"实际发生在按压缩比划分的组内」；表内第 1/2/3 组「压缩比」列均为 2；config.json `"kv_source_layers": [2, 8, 14, 20]`、`"compress_ratios"` 层 2–19 全为 2。｜修复要求：区分「压缩比一致」为必要条件与「分组由生产层列表声明」，并在第 4 章补充一句说明层 2–19 同为压缩比 2 却分三组的原因是生产层列表指定，不是压缩比推出；本章问题 1 答案同步改为「同压缩比是共用的必要条件，具体分组由生产层列表决定」。｜修复：第 4 章改为「压缩比一致是必要条件、不是分组依据」并说明层 2–19 同为压缩比 2 却分三组、分组由 kv_source_layers=[2,8,14,20] 声明；本章问题 1 答案同步改为「必要而非充分」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·技术] §2 补充折叠块「890 字节」：页面把每 token 缓存 890 字节归于「压缩比 2/1 的压缩条目 + 按 ratio 分组共享」，漏掉报告并列的 FP4 KV 缓存，且该数字无 `<sup>` 引用、来源章节亦无对应 [Nx] 条目。｜引文依据：报告行 16–19「DeepSeek-V4.1-Flash combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash.」；行 1968–1969 同义重述。｜修复要求：在 890 字节句补 `[Nx]` 引用，并把成因改写为「跨层复用 + FP4 KV 缓存」两因并列，删除「因此…才能降到」隐含的单一归因；在「外部数字与实验条件」新增该条目，写明来源为技术报告 §1/§4.5 及「1/4 of DeepSeek-V4-Flash」。｜修复：890 字节句改写为「压缩条目 + 分组共享 + 主 KV 与索引器键值量化到 FP4」三个因素并列，补 <sup>[N5]</sup> 引用，并在来源章节新增 [N5]（技术报告 §1 行 16–19，含「roughly 1/4 of DeepSeek-V4-Flash」）｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·格式] 「来源与范围说明」的「构造示例」与「辅助解释与类比边界」两小节描述的内容在正文中不存在：「『40 份 vs 4 份』的份数对照」与「整栋楼共用一套供水」类比全文各只出现 1 次，即出现在该小节自身的说明里，正文无对应材料。｜引文依据：grep 全文「40 份」命中 1 次、「供水」命中 1 次、「一口井」命中 1 次，均位于来源章节；正文第 1–4 章无份数对照表、无供水类比。｜修复要求：二选一——(a) 在正文补入「40 份 vs 4 份」对照与供水类比并标注构造/辅助性质，或 (b) 删除来源章节中这两小节。不得保留指向不存在正文内容的来源说明。｜修复：删除来源章节中指向不存在正文的「40 份 vs 4 份」与「整栋楼共用一套供水」两处描述，改为与正文一致的口径（层区间为构造示例；辅助解释为份数关系本身）｜复验：已复跑 validate.py 通过并核对修改位置｜

- [重要·格式] 引用编号单向：正文定义了 [C7]、[N1]、[N3]，但正文任何位置都没有 `<sup>[C7]</sup>`、`<sup>[N1]</sup>`、`<sup>[N3]</sup>`。style-guide §6 要求正文上标与来源章节双向对应。｜引文依据：`grep -o '<sup>[^<]*</sup>'` 结果集为 C1/F1/C2F2/C2/C3/F3/N4/C4/C4/C6/C5N2，不含 C7、N1、N3；来源章节有 `<p>[C7]`、`<p>[N1]`、`<p>[N3]`。｜修复要求：三处各补一个正文引用点或删除该条目——[N1] 的 80×/128K 例子应在 §2 长序列收益处引用；[N3] 应在 §4 分组表的「压缩比」列引用；[C7] 应在 §1 或 §2 本章问题「只缓存一次」答案处引用（现该答案未挂编号）。｜修复：[C7] 在第 1 章「只缓存一次」处、[N1] 在第 2 章「约省 L 倍缓存」处、[N3] 在第 4 章分组表说明处各补正文引用，三处编号已双向闭合（脚本核对：定义未引用 0、引用未定义 0）｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·格式] [N4] 的行号定位不准。｜引文依据：页面写「tech_report.txt 行 322–326」，实际「This nearly halves prefill computation」在行 329；行 322–326 为「DeepSeek-V4.1-Flash.」「The Causal Encoder–Decoder (CED) architecture and Compressed Sparse Attention 2 (CSA2)」及页码 7。｜修复要求：把行号改为含该句的区间（如 327–330）；[N1] 同时把「论文 §4.4」的归属修正为「80× 见 §1 Introduction，128K/1GB 见 §4.4」。｜修复：[N4] 行号改为 327–330；[N1] 归属修正为「80 倍见 §1 Introduction，128K/1GB 见 §4.4」｜复验：已复跑 validate.py 通过并核对修改位置｜

- [轻微·可读性] §1 用 YOCO 的单一共享投影 $W_K,W_V$ 描述机制后，未说明 DeepSeek-V4.1-Flash 的 CED 上段全局 KV 是用**逐层各自的**投影权重 $W_l^{KV}$ 从 $H^{L/2}$ 投出，而共享来自 CSA2 模式；读者会把「一份投影权重」误当作 DeepSeek 的实现细节。｜引文依据：报告行 390–395「For the upper half layers ... the KV entries are not derived from their respective hidden states H_l. Instead, they are projected directly from the hidden state of the (L/2)-th layer, H_{L/2}, using layer-dependent projection weights (W_l^{KV} and W_l^Z)」；页面 §1「它的末层输出 $X^{L/2}$ 被投影成全局键值」。｜修复要求：在 §1 末段或 §4 加一句限定——YOCO 用一份共享投影权重，DeepSeek-V4.1-Flash 的 CED 用逐层投影权重，共用的是同一份 $H^{L/2}$ 及其派生缓存；不改变核心结论。｜修复：第 1 章投影权重符号表补限定：YOCO 用一份共享权重，DeepSeek-V4.1-Flash 的 CED 上段改用逐层各自的投影权重从同一份 H^{L/2} 投出，共用的是隐状态与派生缓存而非同一组权重｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 2
- 处置：修复

说明：核心结论（复用对象为全局 KV、缓存复杂度 O(LND)→O((N+L)D)、prefill 提前退出、共享 KV 与复用索引解耦为三模式、分组 2–7/8–13/14–19/20–39 与 source 2/8/14/20）经来源逐条核对全部成立且与运行期实测存档一致，无阻断问题。4 条重要问题均不推翻核心结论，但会形成明显误解（分组规则自相矛盾、890 字节单一归因、来源说明指向不存在内容、引用单向），须修复后复验。修复时同步核对被改动的定义、数字与公式来源。
