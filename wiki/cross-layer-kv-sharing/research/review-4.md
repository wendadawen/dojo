<!-- review-meta
round: 4
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: 13efb90a1120f2bc
-->
# 跨层 KV 复用审查记录（第 4 轮）

- 页面版本：8bf52f5730922d4faff995f9c51e5a1b2183b579（index.html 工作树哈希，与 HEAD 一致）
- 审查时间：2026-09-13 19:36
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、常见误解、1. 复用的是哪一层的 KV、2. 省下多少：层维度被消掉、3. 两级复用：共享 KV 与复用索引、4. 前提与边界：同组才能共用、来源与范围说明（含全部折叠块、图注、表格）
- 机械验证：`python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` → `validation ok`；`dojo:topics`（注意力机制、内存与缓存）在 `catalog_builder.py` 的 `ALLOWED_TOPICS` 内；页面无「（待生成）」占位；`kv-cache` / `mla` / `standard-attention` / `dsa` / `deepseek-v4-1` 前置页均存在；`overview.html` 与 `index.html` 双向互链；`libs/dojo-concept.css` 含 `dg-stack / dg-layer / dg-flow / dg-node / dg-arrow / dg-node-note / dg-node-title / diagram / diagram-caption` 全部类名。（页面无 `pre`/`code` 可运行代码，第 3 项「运行代码」不适用。）

## 问题

- [阻断·技术] 来源与范围说明 [C5]，并波及 `<head>` 的 dojo:summary、`主要依据` 段末句「共享关系的运行期实测见文末来源说明」、第 4 章正文与「构造示例」小节：运行期实测的来源 `wiki/deepseek-v4-1/research/verify_csa2_modes.py` 与其存档 `ckpt/verify_csa2_modes.out` 在仓库中已不存在，来源无法定位；「实测共享分组为 2-7 / 8-13 / 14-19 / 20-39」「实测有 4 个 source 层」是本页核心结论之一（出现在 dojo:summary 与常见误解），其支撑来源缺失。引文依据：`find /Users/wendadawen/code/dojo -name "verify_csa2*"` 无输出；`ls wiki/deepseek-v4-1/research/` 现仅有 md 与 `official/`；`git log --all --diff-filter=A --name-only` 显示该两文件曾被加入后被删除（提交 13ead44「research/ 只保留 md：实测产物先登记进 measured.md 再删除（264 个文件 / 36.5 MB），含实测判定改为读记录」）；全库 `grep -rl "verify_csa2"` 仅命中本页与 review 记录。修复要求：把 [C5]/[N2] 的来源改指向保留的实测登记（`wiki/deepseek-v4-1/research/measured.md` 中该产物的登记条目）并保留「等比缩小维度」的条件说明；若该登记不足以支撑「实测确认共享关系」，则删除实测表述，把分组来源降级为配置声明 [C6] 并同步修改 dojo:summary、常见误解、第 4 章正文与「构造示例」小节中的「实测」措辞。｜修复：｜复验：

- [重要·技术] 来源与范围说明 [C4]（行 490–545）、[F4]（行 389–396）、[N4]（行 327–330）、[N5]（行 16–19）：用「tech_report.txt 行号」定位的四条技术报告论断指向已移除的本地文件，读者无法按此定位。引文依据：`find /Users/wendadawen/code/dojo -name "tech_report*"` 无输出；`wiki/deepseek-v4-1/research/official/` 现只有 `README.md`、`encoding/`、`inference/`；`git log --all --diff-filter=A --name-only | grep tech_report` 显示曾存在 `wiki/deepseek-v4-1/research/official/tech_report.txt`（登记体积 0.2 KB）后被删除。修复要求：去掉对已移除本地文件的行号引用，改为按技术报告章节号定位（[C4] §2.3.1、[F4] §2.2、[N4] §2.1、[N5] §1）并保持英文引文片段；可补官方报告（HuggingFace `deepseek-ai/DeepSeek-V4.1-Flash` 的 `DeepSeek_V41_Tech_Report.pdf`）作为可访问出处。｜修复：｜复验：

- [轻微·技术] 来源与范围说明 [C1]：引文与原文不符。引文依据：页面写「YOCO is stacked with L layers, where the first L/2 layers are self-decoder while the rest modules are cross-decoder.」；arXiv:2405.05254v2 §2 原文为「YOCO is stacked with $L$ blocks, where the first $L/2$ layers are self-decoder while the rest modules are cross-decoder.」（两次独立抓取均为 blocks）。修复要求：把引文中的 "L layers" 改回 "L blocks"，或改为不逐字引用的转述。｜修复：｜复验：

- [轻微·技术] 第 1 章 [F1] 与符号表、第 2 章符号表：隐藏维符号全页不统一。引文依据：第 1 章与 [F1] 写作 $W_K, W_V \in \mathbb{R}^{d\times d}$，第 2 章符号表写作「$D$：隐藏维」，dojo:summary 与正文复杂度式用 $D$（YOCO Table 2 表注本身即「N, L, D are the sequence length, number of layers, and hidden dimension」，Eq.(2) 用 $d$）。修复要求：全页统一为 $D$（并说明该量即 YOCO Eq.(2) 中的 $d$），或保留 $d$ 处加一句「$d=D$」。｜修复：｜复验：

- [轻微·格式] 第 2、3、4 章 h2 标题：副标题分隔符不符合 `guides/concept/style-guide.md` 第 1 节「需要副标题时使用 `1. 主题——副标题`」。引文依据：页面为「2. 省下多少：层维度被消掉」「3. 两级复用：共享 KV 与复用索引」「4. 前提与边界：同组才能共用」；同库概念页（如 `wiki/mla/index.html`「1. MLA 压缩了什么——KV 联合压缩的核心机制」、`wiki/deepseek-v4-1/index.html`「1. 890 字节的账——压缩、共享与 FP4 各贡献多少」）均用破折号。修复要求：三处 h2 副标题分隔符由「：」改为「——」。｜修复：｜复验：

- [轻微·技术] 第 4 章正文、核心问题 4 解答、第 4 章本章问题 2 解答：「表达能力损失由训练补偿」「需要训练补偿」被写成无条件结论，全页无对应来源编号，属无来源支持的归因。引文依据：不适用（页面未给来源；本页「辅助解释与类比边界」自己声明「也不说明复用对质量没有代价」）。修复要求：把该判断降级为明确标注的推断（例如「本页推断…」），或补一条来源/报告章节；无法补来源则删除该归因，只保留「上段各层不再有各自的全局 KV」这一事实陈述。｜修复：｜复验：

## 已核对且有引文依据的来源论断（非问题）

- [C1] arXiv:2405.05254v2 Abstract/§2「a cross-decoder stacked upon a self-decoder」「The KV caches K̂, V̂ are reused by all the L/2 cross-decoder modules」一致（仅 "L layers" 见上条）。
- [C2] §2.3「the number of caches is O(N+CL) ... about O(N) caches are required, i.e., you only cache once」「roughly saves L times GPU memory for caches compared to Transformer decoders」一致。
- [C3]/[F3] Table 3「Prefilling Time: Transformer 𝒪(LN²D) / YOCO 𝒪(LND)」与 §2.3「at least half prefilling latency reduction」一致。
- [F1] Eq.(2)「K̂ = LN(X^{L/2})W_K, V̂ = LN(X^{L/2})W_V where W_K, W_V ∈ ℝ^{d×d} are learnable weights」一致。
- [F2] Table 2「KV Cache Memory: Transformer 𝒪(LND) / YOCO 𝒪((N+L)D)」+ 表注「N, L, D are the sequence length, number of layers, and hidden dimension」一致。
- [N1] §1「the memory of KV caches can be reduced by about 80× for 65B models」、§4.4「YOCO can serve 128K tokens with 1GB GPU memory」一致。
- [C7] 脚注 1「The word 'once' refers to global KV cache. Strictly, self-decoder also needs to store a certain number of caches.」一致。
- [N5] 官方 `official/README.md`「reduce the global KV cache footprint to 890 bytes per token — roughly 1/4 of DeepSeek-V4-Flash」一致（页面把成因写为「跨层复用 + FP4」两项并列，与 README「Combined with FP4 main KV caching」一致）。
- 第 3 章三种模式（Full/Reindex/Reuse）与 README「assigns each attention layer one of three static modes — Full, Reindex, or Reuse — to share main KV and indexer K across layers and reuse Top-K sparse-attention indices」一致；「Reindex 重新打分」有外部二手来源佐证（「只重新计算一次哪些历史位置最重要」）。
- 第 4 章分组表：解码器侧 20–39 共用层 20 的 KV（解码器内部 20–23/24–27/28–31/32–35/36–39 各 1 个生产/重打分层），与页面 `kv_source_layers=[2,8,14,20]` 的「4 个 source 层」表述不冲突；层数换算（2–7 = 6 层、20–39 = 20 层）自洽。
- 图注、折叠块正文与正文结论一致；两级问题块均有 `解答：` 折叠答案，页面级答案均指明完整论证所在章节；符号 $L$（层数）、$N$（序列长度）、$\hat K/\hat V$、$X^{L/2}$ 全文单义；图表为 HTML 结构（非等宽字符框线），类名在 `dojo-concept.css` 中全部存在。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复（[C5] 已移除 research 文件的引用与 [C4]/[F4]/[N4]/[N5] 的 tech_report.txt 行号定位须先关闭；全部轻微项一并处理后再复验）