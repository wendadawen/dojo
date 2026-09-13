<!-- review-meta
round: 6
page: wiki/block-attnres/index.html
reviewed_content_sha256: bd7c6b5c387adaec
-->
# Block AttnRes 概念审查记录（第 6 轮）

- 页面版本：011f61fb62590b652658274a2dc7b1c6e8784260（index.html）
- 审查时间：2026-09-13 21:09
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：引言与核心问题 → 1. 标准残差在深度上的瓶颈 → 2. Full AttnRes 的公式 → 3. Block AttnRes 的分块与块间 attention → 4. K3 的具体配置 → 5. softmax kernel 中的 RMSNorm → 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制），含全部 details 折叠块与图注。
- 来源获取：K3 报告 v1 与 v2 的 PDF 原文（arXiv 2607.24653v1 / v2，两版全文逐条核对）；HuggingFace 官方 config.json（moonshotai/Kimi-K3）；MoonshotAI/Kimi-K3 仓库；MoonshotAI/nano-kpu 仓库。不读取本页 research/。

## 问题

- [阻断·技术] 来源与范围说明 C6/N2/N4 及正文 §4：AttnRes 原 preprint 的引用编号写作 [60]，与 K3 报告不符｜引文依据：K3 报告 v2 参考文献 [58] = "Kimi Team. Attention Residuals. Preprint. 2026."；v1 参考文献 [57] = 同一篇。两版中 [60] 都不是该 preprint（v2 [60] = "Kimi K2.5: Visual Agentic Intelligence"，v1 [60] = 报告自身）。同页引文句 "Empirically, N ≈ 8 recovers most of the benefit across model scales [58]"（v2 §2.2）被本页写成 [60]｜修复要求：把 6 处 [60]（正文第 572 行引文、575、642；来源 C6 第 818 行、N2 第 836 行、N4 第 838 行）改为 [58] 并在 blockquote.meta 与来源章节注明所依据的报告版本（v2）；若统一依据 v1 则改为 [57]。两处版本数字不同，须择一并全文一致｜修复：｜复验：

- [阻断·技术] §2 与 F5：RMSNorm 的引用编号写作 [147]，与 K3 报告不符｜引文依据：K3 报告 v2 参考文献 [148] = "Biao Zhang and Rico Sennrich. 'Root mean square layer normalization'. In: Advances in NeurIPS 32 (2019)."；v1 为 [146] = 同一篇。[147] 在 v2 = "GLM-5"、v1 = "DeepEP"，均非 RMSNorm｜修复要求：把第 286 行（"Zhang & Sennrich 2019，K3 报告引用 [147]"）与 F5（第 830 行）的 [147] 改为 [148]（或 v1 的 [146]），与上一条的报告版本保持一致｜修复：｜复验：

## 已回源核对（无问题，供复验参考）

- 引文（K3 报告 v2 原文逐字核对）："Standard residual connections [44] compress all prior information into a single state hl over depth — a bottleneck reminiscent of RNNs over time"（对应页 §1 引文）；"Since network depth is modest (L < 100), the O(L2 d) arithmetic ... the practical overhead is the O(Ld) memory (and cross-stage communication under pipeline parallelism) for keeping all layer outputs alive"（§3 引文）；"Under Block AttnRes, memory and communication overhead drop from O(Ld) to O(N d)"（§3 引文）；"Empirically, N ≈ 8 recovers most of the benefit across model scales [58] ... 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer"（§4 引文，编号除上条错误外文字一致）；"the RMSNorm prevents layers with large-magnitude outputs from dominating the weights"（§5 引文）；"The final output layer then aggregates all N block representations"（C7）；"AttnRes [58] enable each module to selectively retrieve representations from the embedding, the current block, and"（C8 "each module" 措辞在报告中确实存在，页 §4 引文准确）。
- 公式：Eq.(8) k_i=v_i={h_1 (i=0); f_i(h_i) (1≤i≤l−1)}、Eq.(9) α 与 h_l、ϕ=exp(q^T RMSNorm(k))、Eq.(10) 的 V 矩阵（i=1 与 i≥2 两分支）均与报告一致；页内 b_n=Σ f_j、b_n^i partial sum 定义与报告文字一致。
- 数字与配置：config.json 的 attn_res_block_size=12、num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、full_attn_layers=24、kda_layers=69 全部相符；报告 Table 1 的 "#Layers 93 / Attention-Layer Composition 69 KDA + 24 MLA" 相符；93=7×12+9、69+24=93、93/4=23 块×3 KDA+24 Gated MLA 自洽。
- 手算全部复算通过：Full AttnRes 6 候选 h6≈[0.703,0.703]、权重 0.1386/0.2285/0.1779；Block AttnRes 4 候选 h6≈[1.059,1.241]、权重 0.1405/0.3818/0.2974/0.1804；加 RMSNorm 后权重 [0.2058,0.2620,0.2704,0.2620]、h6≈[1.004,1.056]；对比表变化量 ↑0.065/↓0.120/↓0.027/↑0.082 均正确；大值敏感性 exp(2)=7.4、exp(5)=148、exp(4)=54.6 正确。
- 工程优化引用：§5.2.2 "The block representation is generated once at the boundary layer ... entirely wrapped with checkpointing ... cache-based pipeline communication [58]" 支持页 §简化条件第 1/6 条；§5.4.2 "two-phase schedule: a batched inter-block pass ... intra-block partial sum through an online-softmax merge"、"sequence parallelism (SP) for activations"、"launch the inter-block kernel on a side stream" 支持第 6 条；§7 Case Studies(Chip design) "Block AttnRes with a block size of two" 支持第 4 条（页 §4 提到的 nano-kpu block size=2 属实）。
- 表述：全文无第二人称、无第一人称复数、无调试/踩坑叙事、无"需要注意的是/综上所述"式公文连接词、无"场景"当术语；"本文/本页"自称符合 style-guide §12；details 前缀仅用 补充：/展开：/代码：，解答前缀为 解答：。
- 机械项：`.dojo/scripts/validate.py wiki/block-attnres/index.html` 返回 ok；alt 只有 lightbox 的空 alt=""，无 `$...$`；无 Unicode 数学字符直出（"8 块×12 层"为标签式简写，validate.py 不报）；前置链接 ../residual-connection/ 与 ../kimi-k3-dataflow/ 均真实存在，无"（待生成）"；overview.html 与 index.html 互链；代码块为 language-text 伪代码（未声称可运行），静态核对与 §3 候选集合/partial sum 定义一致。

## 结论

- 统计：阻断 2 / 重要 0 / 轻微 0
- 处置：修复（两处引用编号改与报告一致并锁定版本，改后重新核对来源）
