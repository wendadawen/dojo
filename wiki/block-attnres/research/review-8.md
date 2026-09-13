<!-- review-meta
round: 8
page: wiki/block-attnres/index.html
reviewed_content_sha256: 07c8981a351e18ca
-->
# Block AttnRes 审查记录（第 8 轮）

- 页面版本：7d80efedde50bef5dfd083e33a02c456e421ab59
- 审查时间：2026-09-13
- 审查者：独立子代理
- 已完整阅读章节（按顺序，含折叠块、图注与伪代码）：核心问题；1. 标准残差在深度上的瓶颈——为什么需要 AttnRes；本章问题；2. Full AttnRes 的公式——pseudo-query 如何检索前序层；本章问题；3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$；本章问题；4. K3 的具体配置——8 块×12 层（末块 9 层）、9 个候选、加权三次；本章问题；5. softmax kernel 中的 RMSNorm——为什么不能直接用内积；本章问题；来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 核对方式

- 官方材料：arXiv 2607.24653v2 与 v1 的 HTML 全文（含 §2.2、§5.2.2、§5.4.2、§7、参考文献表），逐句比对；HuggingFace 官方 `config.json`（huggingface.co/moonshotai/Kimi-K3/raw/main/config.json）逐字段比对。
- 图内数值：本页图示为 HTML 结构图，无像素刻度，改为核对图内标注与正文公式、手算结果是否一致；三个结构图的节点、候选编号与正文一致。
- 手算复算：Full AttnRes 6 候选、Block AttnRes 4 候选、加 RMSNorm 重算、softmax 大值敏感性四组数字全部逐步复算通过。
- `.dojo/scripts/validate.py`：validation ok。

### 本轮确认无误的来源对照（摘要）

- `config.json`：`attn_res_block_size=12`、`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`linear_attn_config.kda_layers` 69 项、`full_attn_layers` 24 项——全部与页面表格一致。
- §2.2 原文逐句核对通过："Standard residual connections compress all prior information into a single state … a bottleneck reminiscent of RNNs over time"；Eq.(8) `k_i = v_i = { h_1 , f_i(h_i) }`；Eq.(9) `α_{i→l}=φ/Σφ`、`h_l=Σα·v`；`φ(q,k)=exp(qᵀRMSNorm(k))`、"the RMSNorm prevents layers with large-magnitude outputs from dominating the weights"；"Since network depth is modest (L<100), the O(L²d) arithmetic … is affordable"；Eq.(10) 的两种候选集合；"The final output layer then aggregates all N block representations"；"memory and communication overhead drop from O(Ld) to O(Nd)"。
- §2 框架："designed to scale information flow along three complementary dimensions: sequence length, network depth, and model width"，支持页面「三维信息流」表述；Table 1 行 "#Layers 93"、"69 KDA + 24 MLA"。
- §7 Case Studies / Chip design 原文确认："hybrid KDA and NoPE-MLA attention, Block AttnRes with a block size of two"，页面「nano-kpu 原型使用 block size=2」成立。
- §5.2.2「Memory-efficient Attention residual」确认"block representation is generated once at the boundary layer"、"wrapped with checkpointing"、"cache-based pipeline communication"；§5.4.2「Block AttnRes」确认"two-phase schedule: a batched inter-block pass … online-softmax merge"、SP 激活分片、"launch the inter-block kernel on a side stream"。
- arXiv 提交历史：v2 = Fri, 7 Aug 2026，页面标注「2026-08-07」正确。

## 问题

- [阻断·技术] 来源与范围说明（第 811 行）及全页引用：K3 报告的参考文献编号与官方 arXiv 2607.24653v2 不符，导致引文定位失效｜位置：第 286 行（F5 正文）、第 572 行（blockquote 引文内）、第 575 行、第 642 行、第 811 行、第 820 行（C6）、第 838 行（N2）、第 840 行（N4）、第 832 行（F5）｜引文依据：v2 版 §2.2 原文为 "Attention Residuals (AttnRes) [60] applies the same methodology to depth" 与 "Empirically, $N\approx 8$ recovers most of the benefit across model scales [60]"，其参考文献 **[60]** = "Kimi Team (2026) Attention residuals. Note: Preprint"；v2 **[58]** = "Kimi Team (2025) Kimi k2: open agentic intelligence"，**[57]** = "Kimi Team … Kimi linear"；RMSNorm 在 §2.2 的引文为 "[55, 147]"，v2 **[147]** = "B. Zhang and R. Sennrich (2019) Root mean square layer normalization"，**[148]** = "C. Zhao, … DeepEP"。v1 版对应为 AttnRes = **[59]**、RMSNorm = **[145]**（v1 [57] = Kimi K2、[146] = DeepEP）｜问题：页面第 811 行断言「[58] 为该版中的 "Kimi Team. Attention Residuals. Preprint. 2026."（v1 中为 [57]），[148] 为该版中的 RMSNorm 论文（v1 中为 [146]）」——四个编号全部错误；v2 的 [58]/[148] 实际分别指向 Kimi K2 与 DeepEP，AttnRes preprint 是 [60]、RMSNorm 是 [147]。第 572 行把官方引文 "…across model scales [60]" 直引为 "[58]"，属引文失真。全页 8 处（正文、C6、N2、N4、F5、来源说明）沿用错误编号｜修复要求：全页 `[58]` 改为 `[60]`、`[148]` 改为 `[147]`；第 811 行 v1 映射改为「[60]（v1 中为 [59]）」「[147]（v1 中为 [145]）」；同步修改第 572 行直引内的编号；`overview.html` 同处出现 `[58]`，一并同步｜修复：｜复验：

- [轻微·格式] 来源与范围说明「外部数字与实验条件（N）」：N4、N5 两个条目在正文没有任何 `<sup>` 引用，违反 style-guide §6「与来源章节双向对应」｜位置：第 840 行（N4）、第 841 行（N5）｜引文依据：不适用（正文 `<sup>` 集合为 C1–C9、F1–F5、N1–N3，无 `[N4]`/`[N5]`）｜修复要求：在正文对应位置补 `<sup>[N4]</sup>`、`<sup>[N5]</sup>`，或删除这两个条目｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 1
- 处置：修复