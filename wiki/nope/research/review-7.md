<!-- review-meta
round: 7
page: wiki/nope/index.html
reviewed_content_sha256: 7e837ac1bb02594d
-->
# NoPE 审查记录（第 7 轮）

- 页面版本：b322b61e69f17512a811e4c8769f4380cd1e5118（`git hash-object wiki/nope/index.html`）
- 审查时间：2026-09-14 17:09
- 审查者：独立子代理（未参与写作，未参与前 6 轮审查与修复；未读取本页 research/）
- 已完整阅读章节：引言 + 核心问题；1. 为什么 Transformer 需要位置编码——内容匹配不关心位置（含本章问题）；2. NoPE 是什么——去掉所有显式位置编码（含本章问题）；3. 为什么去掉位置编码仍能区分词序——因果掩码的隐式位置信号（含本章问题与「展开：同一例子改成双向注意力会怎样」折叠块）；4. NoPE 在长度泛化上的表现——优于显式方法，外推无需调参（本页推断）（含本章问题）；5. 在 Kimi K3 中怎么用 NoPE——MLA 用 NoPE、KDA 提供位置（含本章问题）；6. NoPE 的适用边界——因果掩码是必要前提（含本章问题）；来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）。已连同 overview.html 一并通读。

## 问题

未发现问题。逐项核对结果如下。

## 来源核对记录（引文依据）

说明：本轮所有事实性论断均逐条回源核对，定位到的原文片段记录于此。外部来源获取方式——NoPE 论文经 arXiv（abs/2305.19466、html/2305.19466v2）抓原文；K3 报告经 arXiv（html/2607.24653）抓原文；K3 层布局另与 config.json 派生的同站 kda 页交叉确认。

### NoPE 论文（arXiv:2305.19466，NeurIPS 2023）

- 作者与题名核对：[C1] 所列 Kazemnejad, Padhi, Ramamurthy, Das, Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers" 与 arXiv 条目一致。
- [C1] 括注「将 NoPE 列为‘Transformers without positional encoding’」：摘要原文含 “Transformers without positional encoding (NoPE).” 其前为 “(APE), T5's Relative PE, ALiBi, and Rotary, in addition to”，据此确认论文比较的五种方案为 APE、T5's Relative PE、ALiBi、Rotary、NoPE，与正文第 4 章表述一致。
- [C2] 引文 “explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences” 与 “We theoretically demonstrate that NoPE can represent both absolute and relative PEs”：摘要均含，逐字一致。
- [C3] 引文 “We theoretically demonstrate that NoPE can represent both absolute and relative PEs” 与 “but when trained with SGD, it mostly resembles T5's Relative PE attention patterns”：摘要逐字一致。
- [C4] 引文 “the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. NoPE outperforms other explicit positional encoding methods while requiring no additional computation.”：摘要含 “most commonly used”（非 “widely used”）与 “ALiBi, Rotary, and APE”（未展开为 “Absolute Position Embedding (APE)”），片段一致；“reasoning and mathematical tasks” 亦见摘要，支持「在推理与数学任务上比较」。
- 第 4 章表 T5 相对 PE 行「不如 NoPE」由摘要 “NoPE outperforms other explicit positional encoding methods” 覆盖（T5 相对 PE 属 explicit methods），表头已限定「论文摘要结论」。

### Kimi K3 技术报告（arXiv:2607.24653）

- [C1]/[C5] §2.1.2 引文 “applies No Position Encoding (NoPE) to all MLA layers”“no explicit positional encoding is applied to their queries or keys”“The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction.”：§2.1.2 原文逐字一致。
- [C6] §2.1.2/§3.4 引文 “extrapolates directly to 1M-token contexts without any positional-encoding modification, such as RoPE rescaling or interpolation”：§3.4 原文一致；§2.1.2 另有 “This separation also avoids modifying positional-encoding parameters when extending the context length, such as retuning a RoPE frequency base or applying YaRN”，与正文该段表述一致。
- [F2] §2.1.1 Eq.(5) 引文 “gth = gmin Sigmoid(eAh zth) ∈ (gmin, 0)dk”“αth = exp(gth) ∈ (egmin, 1)dk”“where Ah is a learnable per-head log-scale and gmin = −5 is fixed”：§2.1.1 原文逐字一致。正文用 $\lambda$ 代报告 $\alpha_t^h$、在 [F2] 注明改名，$\lambda$ 全页单义（不与注意力权重 $\alpha_{t,i}$ 混淆）。
- [N1]「预训练 8K→64K、cooldown 256K→1M」：§3.4 原文 “The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase.”（报告该句本身未叙 64K→256K 一段，本页与来源一致，非本页缺漏）。
- [N2] $g_{\min}=-5$：§2.1.1 原文 “gmin = −5 is fixed”，一致。
- §2.1.2「混合注意力 3:1、骨干末尾另加 1 层 Gated MLA」：报告原文 “Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio.” 与 “An additional Gated MLA layer is placed at the end of the backbone”。与 config.json 派生的层布局（w 站 kda 页：93 层 = 69 KDA + 24 Gated MLA，22 个完整 3:1 块 + 末尾块 + 末尾 1 层 Gated MLA）一致，不矛盾。
- §2.1.2「与 Kimi K2 / K2.5 不同」：原文 “Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear”，本页「后两者未对 MLA 层使用 NoPE、未说明其具体方案」为该对比句的合理读法，且明确声明其内容边界，属可核对表述。

### 页内数字与算式

- 构造示例 $v_1=2,v_2=4,v_3=6$，分数两两相等：$o_1=2$、$o_2=(2+4)/2=3$、$o_3=(2+4+6)/3=4$，复算一致；双向对照三处均为 4，复算一致。
- 因果注意力公式 $o_t=\sum_{i=1}^{t}\alpha_{t,i}v_i$、$\alpha_{t,i}=\exp(q_t^\top k_i)/\sum_{j=1}^{t}\exp(q_t^\top k_j)$ 与符号表自洽，「求和上界 $t$ 随位置变化」与公式一致。
- 正文、summary、description、图注中的同一数字（3:1、8K/64K/256K/1M、$g_{\min}=-5$、$o_1..o_3$）互不矛盾；引文编号 C1–C6、F1–F2、N1–N2 均定义完整、正文引用与来源小节一一对应；图注读数与图内标注一致（无坐标图）。

### 表述与格式

- 通读含折叠块与图注，无会话指代（我/我们/你）、无调试叙事与临场评价、无 AI 拼接腔（无抽象名词堆叠、无「场景」当术语）。「本页据此推断」「（本页推断）」属 style-guide §12 明确许可的自称（「自称使用‘本页’或‘本文’」），非元话语缺陷；其作用是把第 4 章的推断与摘要结论分开，正文对应处另有「这个类比只解释…」「不保证…」等边界声明。
- 推断均带标注：第 4 章标题标「（本页推断）」，核心问题第 3 问答案标「本页据此推断」，来源章节设「辅助解释与类比边界」「简化条件及其限制」两小节逐条给出可推出/不可推出结论。未发现把实验条件下的观察写成无条件论断、或把推断包装成来源结论。
- 公式书写：204 个 `$`（成对）、1 个 `$$…$$` 显示式；KaTeX 定界符完整，无未闭合；summary 内 `$t$` 可渲染；标题/summary/正文/列表/表格无 Unicode 数学字符（`validate.py` 通过）。
- 结构图均为 HTML 结构（`.stack-diagram` div 堆叠、`.arch-diagram` span 节点），无等宽字符框线图、无 SVG `<text>` ASCII 近似；两处 `aria-label` 内无 `$...$`（写作 v1、KDA/MLA，未用 `$v_1$`）。
- 问题块：页面级「核心问题」5 条（3–5 条合规）、6 个正文章节末各带「本章问题」，22 个 `<details>` 中 21 个用「解答：」、1 个手算展开用「展开：」，与 style-guide §5/§9 一致；核心问题答案均指明完整论证所在章节。
- h2 编号 1–6 连续 + 未编号「来源与范围说明」；来源章节 h3 用固定命名（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制），合规。
- 链接：positional-encoding、rope、causal-mask、kda、linear-attention 五个前置概念页均存在；overview.html 与 index.html 相互链接；无「（待生成）」占位。
- 无 `<pre>/<code>` 代码块、无交互视图，故不涉及代码执行与脚本缺失可读性。
- `.dojo/scripts/validate.py wiki/nope/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 0
- 处置：可发布