<!-- review-meta
round: 5
page: wiki/cross-layer-kv-sharing/index.html
reviewed_content_sha256: d3824b38ea070d70
-->
# 跨层 KV 复用审查记录（第 5 轮）

- 页面版本：f45fb47fab211bad6fee70fb495eebe82cc80f9f
- 审查时间：2026-09-13 20:14
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 复用的是哪一层的 KV / 2. 省下多少——层维度被消掉 / 3. 两级复用——共享 KV 与复用索引 / 4. 前提与边界——同组才能共用 / 来源与范围说明（含全部折叠块与两条图注）

核对过的来源：YOCO（arXiv:2405.05254，ar5iv 全文，§1/§2.1/§2.2/§2.3、Eq.(2)、Table 1–3、脚注 1、Figure 7）；DeepSeek-V4.1-Flash 技术报告 PDF（`official/DeepSeek_V41_Tech_Report.pdf`，§1/§2.1/§2.2/§2.3.1/§2.3.2/§2.4.4/§4.2.1/§6）；官方发布说明（`official/README.md`）；官方配置（`official/inference/config.json` 的 `kv_source_layers`/`compress_ratios`/`index_source_layers`，`official/config.json` 的 `text_config.compress_ratios`/`kv_source_layer_ids`）。validate.py 通过。本页无代码块，代码项不适用。

## 问题

- [重要·表述] 常见误解第 1 条（第 106 行）与核心问题第 4 题解答 summary（第 96 行）：「不同压缩比的组各有一份」把"组"与"压缩比"绑定，暗示分组依据是压缩比；这与第 4 章正文（第 281 行）「压缩比一致只是必要条件、不是分组依据——层 2–19 的压缩比全为 2，却被拆成 2–7、8–13、14–19 三个组」以及分组表（第 287–289 行三组同为压缩比 2）冲突：按该说法压缩比只有 2 与 1 两个取值、应只有 2 个组，与表中 4 个组不符。｜引文依据：`inference/config.json` 的 `compress_ratios` 层 2–19 全为 2、层 20–39 全为 1（仅 2 个取值），而 `kv_source_layers=[2,8,14,20]` 给出 4 个组。｜修复要求：改写这两处，使表述不暗示"分组依据=压缩比"，例如「不同压缩比的层无法共用，各自有独立的份；分组由配置的生产层列表给定，压缩比相同的层也可能分属不同组」。｜修复：｜复验：
- [轻微·可读性] 第 3 章表格表头「主 KV 与索引器 K」（第 230 行）、第 3 章图注节点（第 245 行「产出主 KV、索引器 K 与 Top-K」）、第 4 章补充（第 299 行）均使用术语「索引器 K」，但全文未解释它是什么（正文只解释了对侧的「复用索引」，第 239 行「索引是"对某一份 KV 打分后选出的位置"」），也未链接 overview.html 列为前置概念的「稀疏注意力与索引器」（`../dsa/index.html`）。｜引文依据：不适用｜修复要求：在「索引器 K」首次出现处用一句普通技术语言说明它是索引器打分用的键，或链接 `../dsa/index.html`。｜修复：｜复验：
- [轻微·来源] 第 2 章（第 180 行）「论文称至少省一半的层计算」标注 `<sup>[F3]</sup>`，但 [F3]（第 335 行）只登记 prefill 复杂度对照 $\mathcal{O}(LN^2D)$/$\mathcal{O}(LND)$（YOCO Table 3），不含"层数减半"这一论断；该论断的出处是 [C3] 引用的原文。｜引文依据：[C3]（第 327 行）「First, only half the layers are needed for forward computation, i.e., at least half prefilling latency reduction.」｜修复要求：把「至少省一半的层计算」后的上标由 [F3] 改为 [C3]。｜修复：｜复验：
- [轻微·格式] 「来源与范围说明」的「构造示例」小节（第 346 行）没有给出任何构造示例，内容是用官方配置解释分组表来源，与第 4 章正文（第 295 行）重复；本页唯一的人为设定是第 4 章第 310 行「第 $j$ 条对应位置 $j\cdot r$ 起的一段」这一位置换算。｜引文依据：不适用｜修复要求：把「构造示例」小节改写为对上述位置换算这一构造的说明，或按 `guides/concept/style-guide.md` §6「没有内容的小节不保留」删除该小节。｜修复：｜复验：
- [轻微·格式] 第 354 行「论文的 80 倍 与加速数字按原值登记」在「80 倍」与「与」之间多出一个空格。｜引文依据：不适用｜修复要求：删除多余空格。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复

已核对无误的关键项（本轮复算）：YOCO 引文（Abstract、§2「first L/2 layers are self-decoder」、§2.2 Eq.(2) 与「$W_K,W_V\in\mathbb{R}^{d\times d}$ are learnable weights」、§2.3 Table 2 的 $\mathcal{O}(LND)$/$\mathcal{O}((N+L)D)$ 及其表注 N/L/D、Table 3 的 $\mathcal{O}(LN^2D)$/$\mathcal{O}(LND)$、Table 1 的提前退出、§2.3「only cache once」「roughly saves L times」、脚注 1 全文、§1「about 80× for 65B models」、§4.4「128K tokens with 1GB GPU memory … at 65B model size」）逐条与原文一致；DeepSeek-V4.1-Flash 侧（§2.2 Eq.(1) 的 $C^l=H^{L/2}W_l^{KV}$、§2.3.1 三种模式的四段引文、890 字节在 §1 与 §6 的重述、§2.1「This nearly halves prefill computation」、compress_ratios 层 2–19 为 2 / 层 20–39 为 1、kv_source_layers=[2,8,14,20]、分组 2–7/8–13/14–19/20–39）逐条与配置和报告一致；`../deepseek-v4-1/index.html` 中确有 890 = 主 KV 720 + 索引器 K 170 的逐项加总，第 2 章补充的对该页指引成立；页面链接（kv-cache、mla、deepseek-v4-1、overview）均可解析，无指向 research/ 或不存在文件的路径；`dojo:topics`（注意力机制、内存与缓存）与 `dojo:tag`（KV cache）在封闭词表内。