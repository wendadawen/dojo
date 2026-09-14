<!-- review-meta
round: 10
page: wiki/block-attnres/index.html
reviewed_content_sha256: e7f4657c31a983cd
-->
# Block AttnRes 审查记录（第 10 轮）

- 页面版本：42b11e9e6969ca694ad0a0cb62d024354344848c（`wiki/block-attnres/index.html` 工作树哈希）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；未读取本页 `research/` 下任何文件）
- 已完整阅读章节（含全部折叠块、图注、伪代码块与 `overview.html`）：核心问题（5 题）；1. 标准残差在深度上的瓶颈；2. Full AttnRes 的公式；3. Block AttnRes 的分块与块间 attention；4. K3 的具体配置；5. softmax kernel 中的 RMSNorm；来源与范围说明；页面脚本文本

## 来源核对（本轮实际打开并定位的来源）

- K3 Technical Report，arXiv 2607.24653（v2 PDF，last-modified 2026-08-10，content-length 1790685；HTML v2；HTML v1；v1 PDF），`pdftotext -layout` 文本
- HuggingFace `moonshotai/Kimi-K3`：`config.json`、`modeling_kimi_linear.py`、文件清单（HF API）
- 逐项核对结果（均通过）：
  - Eq.(8)：`k_i = v_i = { h_1  (i=0); f_i(h_i)  (1 ≤ i ≤ l−1) }`——页面写法一致
  - Eq.(9)：`φ(q,k)=exp(qᵀRMSNorm(k))`，`α_{i→l}=φ(q_l,k_i)/Σ_{j=0}^{l-1}φ(q_l,k_j)`，`h_l=Σα_{i→l}·v_i`——一致；页面引用的原文 "the RMSNorm prevents layers with large-magnitude outputs from dominating the weights." 逐字一致
  - Eq.(10)：`V = [b_0,…,b_{n−1}]ᵀ (i=1)` / `[b_0,…,b_{n−1},b_n^{i−1}]ᵀ (i≥2)`——一致；`b_n=Σ_{j∈B_n}f_j(h_j)`、`b_0=h_1` 一致
  - "Since network depth is modest (L<100), the O(L²d) arithmetic … the practical overhead is the O(Ld) memory (and cross-stage communication under pipeline parallelism) for keeping all layer outputs alive."——逐字一致
  - "The final output layer then aggregates all N block representations." / "Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(Nd)"——逐字一致
  - "Empirically, N≈8 recovers most of the benefit across model scales [·]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."——逐字一致
  - "Standard residual connections compress all prior information into a single state h_l over depth — a bottleneck reminiscent of RNNs over time."——逐字一致
  - §2 概览 "…enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks…"——存在，支持"each module"论断
  - §5.2.2："The block representation is generated once at the boundary layer…"、"…entirely wrapped with checkpointing…"、"cache-based pipeline communication [·]"——与简化条件小节一致
  - §5.4.2："Block AttnRes … follows a two-phase schedule: a batched inter-block pass … online-softmax merge"、SP 激活分片、side stream 重叠——与简化条件小节一致
  - §7 Case Studies / Chip design："…Block AttnRes with a block size of two…"——page 的 nano-kpu 说法成立
  - Table 1 "Attention-Layer Composition" 行：K3 = `69 KDA + 24 MLA`——与页面一致
  - `config.json`（text_config）：`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`attn_res_block_size=12`、`linear_attn_config.full_attn_layers` 24 项、`kda_layers` 69 项——与页面表格逐格一致
  - 官方源码 `modeling_kimi_linear.py`：`self_attention_res_norm/proj`（attention 子层前）、`mlp_res_norm/proj`（MLP 子层前）、`output_attn_res_norm/proj`（`self.norm` 前）三处、三套参数——页面"加权三次"与三个字段名全部成立（页面自标"间接证据/本页未直接复核"，属保守标注，内容正确）
- 手算复核（全部可复算，结果与页面一致）：Full AttnRes 6 候选手算（内积 0.5/0.5/1.0/0.5/0.75/0.75，Σexp=11.8984，权重 0.1386/0.1386/0.2284/0.1386/0.1779/0.1779，h₆≈[0.703,0.703]）；Block AttnRes 4 候选手算（Σexp=11.7377，权重 0.1405/0.3818/0.2974/0.1804，h₆≈[1.059,1.241]）；加 RMSNorm 对比（‖b‖=1.00/2.24/1.80/1.12，权重 0.2058/0.2620/0.2704/0.2620，h₆≈[1.004,1.056]，对比表 ↑↓ 差值均正确）；§2 本章问题 3 的 3 候选 softmax（0.274/0.274/0.452）；大值敏感性推导（exp(2)≈7.4、exp(5)≈148、exp(4)≈54.6）
- 机械项：`validate.py` 返回 `validation ok`；无 `待生成`/`TODO` 占位；`overview.html` 与 `index.html` 互链；`../residual-connection/index.html`、`../kimi-k3-dataflow/index.html` 目标页真实存在；无 `<img alt>`（唯一 img 为 lightbox 空 alt）；标题/摘要/正文/列表/表格中无 Unicode 数学字符（`×` 为站内 50 页通用写法，非公式符号）；`dojo:summary` 内 `$O(Ld)$`/`$O(Nd)$` 可渲染；结构图为 HTML 结构（`flow-diagram`），非等宽字符框线图

## 问题

- [重要·技术] 来源与范围说明（第 811 行）＋全页引注编号：引文依据的版本不统一，且声明版本下存在异于页面的文献表。页面声明"本页引用的 K3 报告参考文献编号依据 arXiv 2607.24653v2"，但：(a) 行号取自 **v1 PDF** 的分页——`pdftotext -layout` v1 PDF 第 209 行 = "(AttnRes) [57] enable each module…"、第 379 行 = "Standard residual connections [43] compress all prior information…"、第 414 行 = "…The final output layer then aggregates all N block"、第 415 行 = "representations. Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(N d)…"，与页面 C1(379-383)/C5(415-417)/C7(414)/C8(209) 逐一吻合（v2 PDF 对应行为 214/384/419/420）；而 C6 的"§2.2 第 423-424 行"却是 **v2 PDF** 行号（v1 PDF 的 423-424 行是 Stable LatentMoE 正文，v1 中"N≈8…12-layer size"在 418-419 行）。(b) 引文编号 [60]/[147] 取自 **arXiv HTML v2** 的文献表（HTML v2：[60] Kimi Team (2026) Attention residuals.；[147] B. Zhang and Sennrich (2019) Root mean square layer normalization.），但 v2 **PDF** 的文献表把同一两篇列为 [58]（Kimi Team. Attention Residuals. Preprint. 2026.）与 [148]（Biao Zhang and Rico Sennrich, "Root mean square layer normalization"），PDF 的 [60] 是 "Kimi K2.5"、[147] 是 "GLM-5"；v1 PDF 则为 [57]/[146]。｜引文依据：v2 PDF 文献表 "[58] Kimi Team. Attention Residuals. Preprint. 2026."、"[148] Biao Zhang and Rico Sennrich. \"Root mean square layer normalization\". In: Advances in NeurIPS 32 (2019)."；v2 HTML "[60] Kimi Team (2026) Attention residuals."、"[147] B. Zhang and Sennrich (2019) Root mean square layer normalization."；v1 PDF 第 209/379/414 行原文见上｜修复要求：把全页行号统一到页面所声明的 v2（或声明行号依据 v1），并在来源说明中注明所依据的 arXiv 渲染（HTML v2 的编号；若读者按 v2 PDF 核对，AttnRes 为 [58]、RMSNorm 为 [148]），或将引注改为不依赖行号（章节号＋公式号）｜修复：｜复验：
- [轻微·技术] 第 118 行（引导段）与核心问题 1 的解答折叠块：把"第 1 层 embedding 在第 93 层只占 1/93"写成无限定的结论，而 §1 同一论断带限定"（如果后续 $f_l$ 不特意放大它）"。$h_L=h_1+\sum_l f_l(h_l)$ 中 $1/L$ 是求和项数占比的量级示意，不是范数占比；不带限定会被读成精确定量断言｜引文依据：不适用（页面自述的示意，源端 §2.2 只给出 "compress all prior information into a single state h_l over depth" 的定性描述）｜修复要求：在引导段与解答块补上与 §1 一致的限定语，或统一改为"早期信息被层层稀释"的定性表述｜修复：｜复验：
- [轻微·技术] 第 689 行（§2 本章问题第 2 题解答）："这也让 pseudo-query 无需随输入重新计算，是该机制开销较低的原因之一。"——"开销较低"是对成本优劣的判断，页面未给来源；源端只说明 pseudo-query 是层参数、Full AttnRes 的算力 $O(L^2d)$ 可承受、瓶颈在 $O(Ld)$ 内存，未就"开销较低"作比较结论｜引文依据：K3 报告 §2.2 只有 "Since network depth is modest (L<100), the O(L²d) arithmetic of this full form is affordable; the practical overhead is the O(Ld) memory…"，无"开销较低"表述｜修复要求：删除"是该机制开销较低的原因之一"，或降级为明确标注的推断并给出依据｜修复：｜复验：
- [轻微·表述] 第 118、290、584、782 行：正文以"本文/本页"为主语的自我指代与元话语句——"本文直接使用…的结论"（118）、"本文只需要这个结论，不展开推导"（290）、"具体槽位分配的实现细节 K3 报告未展开，本文不推测"（584）、"至此本文五个核心问题全部作答：…"（782），其中 782 为收束式元话语。（"来源与范围说明"一节内的同类用法属该节体例，可保留）｜引文依据：不适用｜修复要求：正文四处改为直接陈述（如"此处只用到该结论，不展开推导"、"槽位分配细节报告未展开，不作推测"），删去"至此本文五个核心问题全部作答"这类收束元话语｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（重要项为引注依据的版本统一，不涉及任何技术结论；修复后建议复验全部行号与 [60]/[147] 两处编号）
- 说明：页面全部事实性论断、引用原文、公式与手算数字本轮均逐条回源核对通过，未发现核心结论错误、无来源支持的机制描述、无同页数字互斥、无图注读数与刻度不符；`validate.py` 通过。
