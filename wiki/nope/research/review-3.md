<!-- review-meta
round: 3
page: wiki/nope/index.html
reviewed_content_sha256: c20e10ab4d126349
-->
# NoPE审查记录（第 3 轮）

- 页面版本：fc896ea7b40c516d73f76bcc3e178fc2d72768c1
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前两轮审查与修复）
- 审查输入：wiki/nope/index.html、wiki/nope/overview.html、本规范、页面引用的外部来源（NoPE 论文 arXiv:2305.19466；Kimi K3 技术报告 arXiv:2607.24653 v1/v2）
- 已完整阅读章节：引言、核心问题（5 题）、1. 为什么 Transformer 需要位置编码、2. NoPE 是什么、3. 为什么去掉位置编码仍能区分词序、4. NoPE 在长度泛化上的表现、5. 在 Kimi K3 中怎么用 NoPE、6. NoPE 的适用边界、来源与范围说明（含全部折叠块、图注与「来源与范围说明」六小节）
- 机械项：`.dojo/scripts/validate.py wiki/nope/index.html` 返回 `validation ok`；`dojo:topics=注意力机制` 在 AGENTS.md 固定大类内；5 个前置概念链接（positional-encoding / rope / causal-mask / kda / linear-attention）对应页面均存在，无「（待生成）」占位；正文无 `<pre>` 代码块，无「声称可运行的代码」，代码执行项 N/A。

## 来源核对（已核对通过，无问题）

- NoPE 论文摘要（https://arxiv.org/abs/2305.19466）逐句核对，页面 C3/C4 及第 4 章表述与原文一致，原文：「the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks.」「NoPE outperforms other explicit positional encoding methods while requiring no additional computation.」「We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns.」「explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.」论文比较对象为 APE、T5's Relative PE、ALiBi、Rotary + NoPE 五种，第 4 章表述一致。
- K3 报告 C5 原文核对（§2.1.2 Gated MLA）：「The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction.」；C1 原文核对：「...applies No Position Encoding (NoPE) to all MLA layers...」，一致。
- K3 报告 C6 原文核对（§3.4）：「extrapolates directly to 1M-token contexts without any positional-encoding modification, such as RoPE rescaling or interpolation.」；§2.1.2 亦述「...avoids modifying positional-encoding parameters when extending the context length, such as retuning a RoPE frequency base or applying YaRN.」，一致。
- K3 报告 N1 原文核对（§3.4）：「The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase.」；§3.4 另有「Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA.」，与第 5 章表述一致。
- 算术复算：$v_1=2,v_2=4,v_3=6$，因果掩码下 $o_1=2$、$o_2=(2+4)/2=3$、$o_3=(2+4+6)/3=4$；双向对照三位置均为 $(2+4+6)/3=4$。页面标注值与计算一致，无 a×b 不等于标注积一类错误。符号 $q_t,k_i,v_i,\alpha_{t,i},o_t,g,\alpha,g_{\min}$ 全页写法一致，均由 KaTeX 渲染。
- 结构图：两处（stack-diagram、arch-diagram）为 HTML 结构，非等宽字符框线图；图内 $v_1,v_2,v_3$ 等由 KaTeX 渲染，无 ASCII 近似写法。

## 问题

- [重要·技术] 来源与范围说明 →「公式与来源（F）」F2 与「外部数字与实验条件（N）」N2（第 484、490 行）：KDA 衰减公式的来源章节号标错。含 $g_{\min}$、Sigmoid、$\alpha=\exp(g)$ 的 Eq.(5) 实际位于 K3 报告 §2.1.1「Kimi Delta Attention」的「Lower-bounded decay」小节，而非页面两处标注的 §2.1.2（§2.1.2 标题为「Gated MLA」，内容为 NoPE/Gated MLA，不含该公式）。公式内容与数值本身正确，仅定位错误。｜引文依据：K3 报告 §2.1.1「Lower-bounded decay」小节：「Kimi K3 instead uses a scaled sigmoid to bound the log-decay from below:」其后 Eq.(5)：$g_t^h=g_{\min}\mathrm{Sigmoid}(e^{A_h}z_t^h)\in(g_{\min},0)^{d_k}$、$\alpha_t^h=\exp(g_t^h)\in(e^{g_{\min}},1)^{d_k}$、$g_{\min}=-5$；§2.1.2 标题为「Gated MLA」。v1 与 v2 一致。｜修复要求：将 F2、N2 及正文第 5 章对该公式的引用章节号由 §2.1.2 改为 §2.1.1（公式号 Eq.(5) 不变）；C1/C5/C6 的 §2.1.2 正确，保持不变。｜修复：｜复验：

- [轻微·技术] 第 5 章第 2 段「这和 Kimi K2 / K2.5 不同——后两者用了显式位置编码（具体方案 K3 报告未说明）。」：超出来源的断言。K3 报告只说明 K3 与 K2/K2.5 不同、对 MLA 层用 NoPE，并未说明 K2/K2.5 使用了显式位置编码，也未指明其方案；该断言被放在以 [C5] 支撑的段落中。｜引文依据：K3 报告 §2.1.2：「Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to all MLA layers.」；报告全文未出现 K2 使用 RoPE 或显式位置编码的表述。｜修复要求：改为明确标注的推断或删除，例如「K3 报告只说明 K2/K2.5 未对 MLA 层使用 NoPE，未说明其具体方案」。｜修复：｜复验：

- [轻微·可读性] 第 3 章「因此 $o_t$ 的取值范围依赖于 $t$，位置进入了计算。」：用词不准确。此处随 $t$ 变化的是 $o_t$ 的取值（前 $t$ 个 value 的加权平均），不是取值范围。｜引文依据：不适用｜修复要求：改为「因此 $o_t$ 的取值依赖于 $t$」。｜修复：｜复验：

- [轻微·表述] 第 3 章「用公式把这个机制写清楚。」「现在回到贯穿全文的问题，把它手算出来。」；第 3 章「注意全程没有加任何位置编码——」；第 5 章「这里要澄清一个容易产生的误解：」：元话语与写作过程叙事，属规范要求排除的表述。｜引文依据：不适用｜修复要求：改写为直接陈述句，去掉指向行文过程的「用公式把…写清楚」「现在回到…把它手算出来」「注意…」「这里要澄清…」等说法。｜修复：｜复验：

- [轻微·表述] 第 1、2、3、4、5 章章末过渡句（第 208、237、319、370、423 行）：五处使用同一固定句式「本章…——下一章讲…」，违反概念页格式规范第 8 节「不使用固定句式」。｜引文依据：不适用｜修复要求：逐处改写为表述前后节结论与问题关系的一句话，句式各异，避免模板化。｜修复：｜复验：

- [轻微·表述] 第 6 章段落以「其次，…」「最后，…」起头而无对应「首先」：公文式序号连接词堆叠。｜引文依据：不适用｜修复要求：删除「其次」「最后」等序号连接词，改为内容层面的衔接。｜修复：｜复验：

- [轻微·格式] 第 3 章因果注意力公式 $o_t=\sum_{i=1}^{t}\alpha_{t,i}v_i$：公式后未按概念页格式规范第 11 节「公式后紧跟 `<ul>` 逐项定义每个符号」处理，仅以散文说明 $q_t,k_i,v_i$，且未定义 $o_t$、$\alpha_{t,i}$ 及求和上界 $t$ 的含义。｜引文依据：不适用｜修复要求：公式后紧跟 `<ul>`，逐项定义 $o_t$、$\alpha_{t,i}$、$q_t$、$k_i$、$v_i$，并说明 $i$ 从 1 到 $t$ 的求和范围为因果掩码所致。｜修复：｜复验：

- [轻微·格式] 来源与范围说明 →「外部数字与实验条件（N）」N2（第 490 行）：N2 在正文无 `<sup>[N2]</sup>` 引用，来源条目与正文未双向对应（$g_{\min}=-5$ 在正文第 5 章实际由 [F2] 承载）。｜引文依据：不适用｜修复要求：将正文第 5 章该处引用改为 `<sup>[F2, N2]</sup>`，或删除 N2 条目。｜修复：｜复验：

- [轻微·表述] 引言首句「位置编码有四类主流方案，NoPE 是其中"什么都不做"的那个。」：与第 2 章「NoPE 不是"又一种位置编码方案"」形成轻微自相矛盾（前者把 NoPE 计入"四类位置编码方案"，后者否认它是位置编码方案）。｜引文依据：不适用｜修复要求：改为如「位置编码有三类显式方案，加上'什么都不做'的 NoPE 构成四种选择」的表述，使两处口径一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复（关闭重要项 F2/N2 章节号定位错误及上述轻微项后即可发布）
