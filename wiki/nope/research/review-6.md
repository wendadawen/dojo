<!-- review-meta
round: 6
page: wiki/nope/index.html
reviewed_content_sha256: f6786aba8d142612
-->
# NoPE 审查记录（第 6 轮）

- 页面版本：f7a64a51a77ce82da86e79e8ad8e31e98d0c2d36（index.html 工作树哈希）
- 审查时间：2026-09-13 21:17
- 审查者：编排者派发的独立审查者（未参与写作，未读取 research/ 下任何文件）
- 已完整阅读章节：head（description / dojo:summary / dojo:type / topics / tag）、核心问题（5 条及全部解答折叠块）、引言、1. 为什么 Transformer 需要位置编码、2. NoPE 是什么、3. 为什么去掉位置编码仍能区分词序、4. NoPE 在长度泛化上的表现、5. 在 Kimi K3 中怎么用 NoPE、6. NoPE 的适用边界、来源与范围说明（含「论断与来源」「公式与来源」「外部数字与实验条件」「构造示例」「辅助解释与类比边界」「简化条件及其限制」全部小节、两处图注与脚本区）

## 来源回源核对（本轮回源摘录）

- NoPE 论文 arXiv:2305.19466 摘要（WebFetch 抓 arXiv 摘要页原文）逐字确认页面引用的四段：「...in addition to Transformers without positional encoding (NoPE)」；「the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while requiring no additional computation.」；「We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns.」；「Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.」——C1/C2/C3/C4 四条引文与摘要一致，且摘要确为「五种 PE 方案（APE、T5's Relative PE、ALiBi、Rotary、NoPE）+ 推理与数学任务」。
- K3 报告 arXiv:2607.24653v2 §2.1.2（Gated MLA）：「Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to all MLA layers.」；「no explicit positional encoding is applied to their queries or keys. The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction.」；「3 KDA layers followed by 1 Gated MLA layer」（3:1 配比、骨干末尾另加 1 层）；「As a result, the model extrapolates directly to 1M-token contexts without any positional-encoding modification.」——C1、C5、C6 及正文「3:1 堆叠」「末尾另加 1 层」「K2/K2.5 对比」全部有据。
- K3 报告 §3.4（Long-Context Extension）：「Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA.」；「The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase.」——N1 与正文/核心问题/description 的 8K→64K、256K→1M 一致，无同一数字多处不符。
- K3 报告 §2.1.1 Eq.(5)：g_t^h = g_min·Sigmoid(e^{A_h}·z_t^h) ∈ (g_min,0)^{d_k}，α_t^h = exp(g_t^h) ∈ (e^{g_min},1)^{d_k}，g_min = −5；经回源确认该式确编号为 (5)、g_min 确为 −5——F2、N2 与正文 $\lambda=\exp(g)$、$g_{\min}=-5$ 一致。
- 手算可复算：v=(2,4,6)、可见集合均匀加权下 o₁=2、(2+4)/2=3、(2+4+6)/3=4；双向对照三处均为 4；正文、图注、两处解答折叠块数字两两一致。
- 机制类链接 ../positional-encoding、../rope、../causal-mask、../kda、../linear-attention 的 index.html 均真实存在，无「（待生成）」；overview.html 与 index.html 相互链接；`.dojo/scripts/validate.py wiki/nope/index.html` 返回 validation ok。
- alt/aria-label 中无 `$...$`；正文、列表、标题、图注中无 Unicode 数学字符（`→` 仅出现在图内箭头「8K→64K」类范围与 arch-arrow，非数学算子）；全文无「我/我们/你」、无「本页将…」「下面来看…」「需要注意的是」类元话语；「本页/本文」自称符合 style-guide 第 12 节。

## 问题

- [轻微·技术] 核心问题第 3 条解答（第 143 行）：句末 `<sup>[F1]</sup>` 挂在「NoPE 的 query、key、value 都只来自内容、不含与训练长度绑定的位置参数，因此外推时不需要插值或重缩放」这一整句推断上，而来源章节 F1 条目自述只提供「q/k/v 不含位置项」这一前提（公式用于第 3 章构造示例），并不支持「外推无需插值/重缩放」这一结论；同页第 4 章正文对同一推断未加任何上标引用，两处处理不一致，读者顺 [F1] 回查会落到一章手算公式上。｜引文依据：F1 条目原文「F1（因果注意力输出）：标准 scaled dot-product attention 的因果形式（Vaswani et al. 2017 的因果变体），用于『因果掩码的隐式位置信号』一章构造示例，非外部新结论。」｜修复要求：让被引条目与所支持的论断一致——或把该上标移到「query、key、value 都只来自内容」这一前提之后（或改标 [C1]，因「不含位置项」来自 NoPE 定义），或整体删除该上标，使其与第 4 章正文的处理一致；不得改写成含糊表述保留原意。｜修复：｜复验：

（本轮回源未发现阻断级与重要级问题：全部事实性论断、公式与数字均可在 NoPE 论文摘要与 K3 报告对应章节定位，无把实验条件下的观察写成无条件论断之处——第 4 章的「优于显式方法」始终带「在长度泛化的下游任务上」限定，「外推无需调参」在标题、正文与解答三处均显式标注为「本页推断」，且 C2 条目已坦白因果掩码机制系本文自证而非论文理论；构造示例（v=2,4,6）在来源说明中标注为教学构造，未写成来源事实；手算、公式符号（α 作注意力权重、λ 作 KDA 衰减因子、报告记法 α_t^h 已注明）全文单义；无指向仓库中不存在路径的引用。）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（本轮无阻断、无重要问题；1 条轻微为引用归属精确性，不影响正确性与主线理解，按上条修复后即可发布）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
