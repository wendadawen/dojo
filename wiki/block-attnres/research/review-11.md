<!-- review-meta
round: 11
page: wiki/block-attnres/index.html
reviewed_content_sha256: 2a5ead20351750e7
-->
# Block AttnRes 审查记录（第 11 轮）

- 页面版本：97f06cb3a03abb44979839a4b23e82160c1530a4（index.html 工作树哈希）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复）
- 已完整阅读章节：head 元信息与「核心问题」；1. 标准残差在深度上的瓶颈——为什么需要 AttnRes；2. Full AttnRes 的公式——pseudo-query 如何检索前序层；3. Block AttnRes 的分块与块间 attention——把内存从 $O(Ld)$ 降到 $O(Nd)$；4. K3 的具体配置——8 块×12 层（末块 9 层）、9 个候选、加权三次；5. softmax kernel 中的 RMSNorm——为什么不能直接用内积；来源与范围说明（含全部折叠块）；overview.html。

## 来源核对

核对版本：arXiv 2607.24653 **v2**（HTML 与 PDF 两种渲染）与 **v1**（HTML 与 PDF），以及 huggingface.co/moonshotai/Kimi-K3 的 `config.json`。行号按页面声明的 v2 PDF（`pdftotext -layout`）。逐条回源结果（均给出原文片段）：

- §2 第 214 行（v2 PDF）：「(AttnRes) [58] enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks」——与正文 C8 引文一致。
- §2.2 第 384-388 行：「Standard residual connections [44] compress all prior information into a single state hl over depth — a bottleneck reminiscent of RNNs over time…」——C1、第 1 章 blockquote 一致。
- 第 390-401 行 Eq.(8)(9)：「ki = vi = h1 (i=0) / fi(hi) (1≤i≤l−1)」「ϕ(q,k)=exp(q⊤RMSNorm(k))」「αi→l=ϕ(q_l,k_i)/Σϕ(q_l,k_j)，h_l=Σαi→l·vi」——F1、F2、全文公式逐符号一致。
- 第 395-397 行：「where the RMSNorm prevents layers with large-magnitude outputs from dominating the weights」——C3、第 5 章引文一致。
- 第 403-404 行：「the O(L2 d) arithmetic of this full form is affordable; the practical overhead is the O(Ld) memory (and cross-stage communication under pipeline parallelism)」——第 3 章 blockquote 一致。
- 第 406-418 行 Eq.(10)：V=[b0..b_{n−1}]⊤ (i=1) / [b0..b_{n−1}, b_n^{i−1}]⊤ (i≥2)——F3、F4、候选集合一致。
- 第 419 行：「The final output layer then aggregates all N block representations.」；第 419-422 行：「memory and communication overhead drop from O(Ld) to O(Nd)」——C7、C5 一致。
- 第 423-424 行：「Empirically, N ≈ 8 recovers most of the benefit across model scales [58]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer.」——C6、第 4 章 blockquote 一致（页面按 HTML v2 记 [60]）。
- 参考文献四种渲染编号逐一核对：HTML v2 `[60] Kimi Team (2026) Attention residuals. Note: Preprint`、`[147] B. Zhang and R. Sennrich (2019) Root mean square layer normalization`；HTML v1 为 [59]、[145]；PDF v2 为 `[58] Kimi Team. Attention Residuals. Preprint. 2026.`、`[148] Biao Zhang and Rico Sennrich. "Root mean square layer normalization". Advances in NeurIPS 32 (2019)`；PDF v1 为 [57]、[146]——与「来源与范围说明」首段声明的四种映射**完全一致**。
- `config.json`：`attn_res_block_size=12`、`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`linear_attn_config.full_attn_layers` 24 个索引、`kda_layers` 69 个索引——与第 4 章两张表一致。
- §7 Case Studies「Chip design」：「Block AttnRes with a block size of two」+ `github.com/MoonshotAI/nano-kpu`——第 4 章末段一致。
- §5.2.2「Memory-efficient Attention residual」：「block representation is generated once at the boundary layer」「entirely wrapped with checkpointing」「cache-based pipeline communication」；§5.4.2「Block AttnRes [60] follows a two-phase schedule: a batched inter-block pass…online-softmax merge」「sequence parallelism (SP) for activations」「we launch the inter-block kernel on a side stream」——「简化条件」两条一致。
- Table 1「Attention-Layer Composition: 69 KDA + 24 MLA」——第 4 章一致。
- 图内数值无像素测量需求（本页无内嵌位图；结构为 HTML/CSS 网格）。
- 手算全部复算通过：6 候选 Full AttnRes $h_6\approx[0.703,0.703]$（权重最大 0.2284）；4 候选 Block AttnRes $h_6\approx[1.059,1.241]$（0.3818/0.2974/0.1804/0.1405）；加 RMSNorm 后 $[0.2058,0.2620,0.2704,0.2620]$、$h_6\approx[1.004,1.056]$；第 2 章 3 候选 0.274/0.274/0.452；大值比 $\exp(2)\approx7.4$、$\exp(5)\approx148$、$\exp(4)\approx54.6$。
- 机械项：`validate.py` 返回 `validation ok`；页面链接（`../residual-connection/`、`../kimi-k3-dataflow/`、`overview.html`、`../../index.html`）目标均存在；无「（待生成）」占位；`alt` 无 `$...$`；`dojo:summary` 的公式均为 KaTeX 可渲染形式；无 元话语/会话指代 命中。

## 问题

- [轻微·技术] index.html 第 7 行 `<meta name="dojo:summary">`：summary 写「复杂度从 $O(Ld)$ 降到 $O(Nd)$」，但 $O(Ld)\to O(Nd)$ 在来源与正文中指**内存与跨 stage 通信开销**，Full 形式的计算量是 $O(L^2d)$，块形式也不把计算复杂度降为 $O(Nd)$；正文第 3 章标题与核心问题 Q3 均写「内存」，summary 用「复杂度」与页面自身及来源不一致。｜引文依据：v2 PDF 第 403-404 行「the $O(L^2 d)$ arithmetic of this full form is affordable; the practical overhead is the $O(Ld)$ memory」、第 420 行「memory and communication overhead drop from $O(Ld)$ to $O(N d)$」。｜修复要求：把 summary 中「复杂度从 $O(Ld)$ 降到 $O(Nd)$」改为「内存（与跨 stage 通信）从 $O(Ld)$ 降到 $O(Nd)$」或等义表述。｜修复：｜复验：
- [轻微·技术] index.html 第 286-290 行（F5）与第 5 章：正文把 RMSNorm 定义为 $\mathrm{RMSNorm}(x)=x/\sqrt{\tfrac1d\sum x_j^2+\epsilon}$，并据此在全文反复断言「缩放到单位均方根，只用方向、不用幅值」。所引来源 Zhang & Sennrich 2019 的定义含可学习增益 $g$（$a_i/\mathrm{RMS}(a)\cdot g_i$），本页定义省略 $g$，「简化条件及其限制」六条中也未说明该省略。该省略不改变「幅值标量被约去、大值不再主导」的核心结论，但「只用方向」的断言在含增益的实际 kernel 中需限定。｜引文依据：Zhang & Sennrich 2019《Root mean square layer normalization》定义式 $\mathrm{RMSNorm}(a_i)=a_i/\mathrm{RMS}(a)\cdot g_i$；本页第 288 行公式无 $g$。｜修复要求：在公式后补一句说明省略了可学习增益 $g$（或按来源补上 $g$），并在「简化条件及其限制」中登记该理想化及其不改变结论的理由。｜修复：｜复验：
- [轻微·技术] index.html 第 654 行表「维度｜Full AttnRes｜Block AttnRes（一般形式）｜K3 实例化」中「末尾聚合」一行：一般 Block 形式一列标为「不强制」。但来源 §2.2 把「The final output layer then aggregates all N block representations.」直接写在 Block AttnRes 的定义段内（与「块内求和 + 块间 attention」并列、Eq.(10) 同段），读作该形式的一部分；页面同表把「加权次数」标为「公式不规定」并在正文 [C8] 明确标注为「本页推断」，本行的「不强制」却未作同样标注。｜引文依据：v2 PDF 第 411-419 行「Across blocks, full attention is applied over only the $N$ block-level representations… The final output layer then aggregates all $N$ block representations.」。｜修复要求：把该格改为「来源将其写入 Block AttnRes 一般形式（[C7]）」，或保留「不强制」但在紧邻处明确标注为「本页推断：公式（Eq.8-10）未强制」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（仅轻微项；三项均不涉及事实错误或主线理解，修复后即可发布）