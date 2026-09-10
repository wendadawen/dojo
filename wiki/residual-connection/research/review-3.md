# 残差连接审查记录（第 3 轮）

- 页面版本：2f5484f75b47b420be6d8137f93e0a80875bd881（`wiki/residual-connection/index.html` 工作树哈希）
- 审查时间：2026-09-10 22:06
- 审查者：未参与写作、未参与前两轮审查的独立审查者（本轮仅使用 index.html、overview.html、页面引用的外部来源与本规范；未读取 `research/` 下任何文件）
- 已完整阅读章节：按顺序读完全文并展开所有折叠块——核心问题（4 题 + 4 个解答折叠块）→ 1. 深层网络为什么退化——加更多层反而更差（含 ImageNet 误差表、本章问题）→ 2. 残差块公式——把"学恒等"变成"学零"（含残差块 SVG 结构图、1 神经元构造示例表、本章问题）→ 3. 梯度如何流动——为什么深网络不再"消失"（含「展开：多层梯度乘积」「展开：3 层手算的完整代入过程」两个折叠块、梯度对照表、8 路径表、本章问题）→ 4. 维度不匹配与适用边界——残差连接不能做什么（4.1 投影捷径、4.2 不解决什么、4.3 Transformer 中的应用、本章问题）→ 来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件、构造示例、辅助解释与类比边界、简化条件及其限制）；另读 `overview.html` 全文。

## 机械验证结果

- `python3 .dojo/scripts/validate.py wiki/residual-connection/index.html` → `validation ok`，退出码 0。
- 引用双向闭合：正文出现 `<sup>[C1] [C2] [C3] [C4] [C5] [F1]×3 [F2] [F3] [F4] [N1, N2] [N3]</sup>`；来源章节定义 C1–C5、F1–F4、N1–N3，编号一一对应，无正文引用缺定义、无定义缺引用。
- 相邻双上标：无（`</sup>` 之后不存在紧邻的 `<sup>`）。
- Unicode 数学字符：定界符之外仅出现 `–`（`2–3 个卷积层` 中的范围连接号），非数学符号，validate.py 通过；无 `θ/α/Σ/∂/≈/∈/ᵢ` 一类字符。
- TAB 字符：无。
- 占位符：index.html 与 overview.html 均无「待生成/TODO/TBD/FIXME/占位」。
- `<head>` 五项元数据：`description`（纯文本）、`dojo:summary`（含可渲染 `$h_{l+1}=h_l+F(h_l)$`、`$\partial y/\partial x=1+\partial F/\partial x$`）、`dojo:type=concept`、`dojo:topics=数学基础`、`dojo:tag=深度网络`，齐备且通过 validate.py 词表校验。
- overview 互链：`index.html` 导航含 `<a class="overview-link" href="overview.html">快速阅读</a>`；`overview.html` 含 `<a href="index.html">深度教学 →</a>`；双向闭合。
- 前置概念链接：`../block-attnres/index.html` 目标文件存在。神经网络层、链式法则、梯度下降三处未外链，采用正文一句话最小定义（style-guide §13 允许，不保留占位标注），无失效链接。
- 可运行代码块：index.html 与 overview.html 均无 `<pre>`/`<code>` 代码块（`<pre>` 计数 0），无可实跑代码，本项无需执行核验，改为静态审查（公式/数值示例均手工复算，见下）。
- 折叠块前缀：全部为「解答：」（18 处问题块）与「展开：」（2 处），无其他前缀。
- 结构图：内联 SVG，公式置于 `<foreignObject>`，`<text>` 内为纯文字，颜色走 CSS 变量（dg-box/dg-arrow/dg-arrowhead/dg-caption）。
- 数值复算：1 神经元例（$2+0=2$、$2+0.2=2.2$、$1\times2=2$、$0.1\times2=0.2$）、3 层梯度例（$1.1^3=1.331$、$0.1^3=0.001$、$h_3=2.662$ / $0.002$）全部与页面一致。

## 问题

- [重要·技术] 3. 梯度如何流动（index.html L875）与本章问题第 2 题答案（L940）：正文写"残差网络多出的'全 1 直通项'使梯度至少为 $1$，不被指数级压缩到零"，把"展开式含常数 1 项"误写成对梯度下界"$\ge 1$"的保证。当 $\partial F_i/\partial h_i\in(-1,0)$ 时每个因式 $1+f_i\in(0,1)$，乘积可远小于 1（如 $f_i=-0.5$ 时 $0.5^{L-l}$）。该表述与来源和页内自述矛盾，且本章问题答案重复了同一说法。｜引文依据：arXiv:1603.05027 §2 Eq.(5) 及紧随解释——"$\frac{\partial\mathcal{E}}{\partial\mathbf{x}_l}=\frac{\partial\mathcal{E}}{\partial\mathbf{x}_L}\left(1+\frac{\partial}{\partial\mathbf{x}_l}\sum_{i=l}^{L-1}\mathcal{F}\right)$"，"The additive term of $\partial\mathcal{E}/\partial\mathbf{x}_L$ ensures that information is directly propagated back to any shallower unit"，"This implies that the gradient of a layer does not vanish even when the weights are arbitrarily small"——来源只主张"存在直通分量、梯度不会消失"，未给出"≥1"下界；页内「辅助解释与类比边界」也自述"若 $F$ 的导数为负且绝对值大于 1，梯度仍可能很小或变号"（该条件同样不准确：$|f|<1$ 且 $f<0$ 时即可缩小）。｜修复要求：删除"使梯度至少为 $1$"，改为"提供一条常数 $1$ 的直通分量，使梯度不被 $F$ 的导数完全主导"；本章问题答案同步改写；「辅助解释与类比边界」中该条条件改为"当 $1+\partial F/\partial h$ 的模可以小于 1（例如导数落在 $(-1,0)$）时梯度仍会缩小"。｜修复：删除正文"3. 梯度如何流动"（index.html L875）的"使梯度至少为 $1$，不被指数级压缩到零"，改为"提供一条常数 $1$ 的直通分量，使梯度不被 $F$ 的导数完全主导"；本章问题第 2 题答案（原 L940）同步改写为同一表述；「辅助解释与类比边界」中该条条件改为"当 $1+\partial F/\partial h$ 的模可以小于 $1$（例如导数落在 $(-1,0)$）时，梯度仍会缩小"。｜复验：全文已无"至少为 $1$"/"$\ge 1$"式下界表述；正文、答案、边界说明三处口径一致，与页内边界说明不再矛盾——正文与答案只主张"存在常数 $1$ 的直通分量"，边界说明给出梯度仍可缩小的具体条件。对照 Identity Mappings 论文 §2 Eq.(5) 及紧随原句"the gradient of a layer does not vanish even when the weights are arbitrarily small"，来源只支持"梯度不消失"，改写后一致。

- [重要·技术] 来源与范围说明 C5（index.html L1042）与「4.3 在 Transformer 中的应用」（L995）：C5 把 Transformer 子层残差的定位标为"Vaswani et al. 2017 … §3.2.1 与 Figure 1"，但 $x+\text{Sublayer}(x)$ 的原文在 §3.1 "Encoder and Decoder Stacks"，§3.2.1 "Scaled Dot-Product Attention" 通篇只讲注意力，按标注定位得不到任何残差相关支持。｜引文依据：arXiv:1706.03762 §3.1——"We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x+Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself."；§3.2.1 标题为 Scaled Dot-Product Attention，内容为 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$。｜修复要求：把 C5（及正文 L995 的上标位置说明）的定位改为 §3.1 Encoder and Decoder Stacks；Figure 1 只能作为架构示意，不能与 §3.2.1 并列为公式出处。｜修复：来源条目 C5（index.html L1042）定位由"§3.2.1 与 Figure 1"改为"§3.1 Encoder and Decoder Stacks（残差连接与 $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$ 的原文）与 Figure 1（架构示意）"；正文 4.3 节把 `<sup>[C5]</sup>` 从"Transformer"之后移到"外都包一层残差连接"这一论断之后。｜复验：arXiv:1706.03762 §3.1 原文"We employ a residual connection around each of the two sub-layers... the output of each sub-layer is LayerNorm(x + Sublayer(x))"；§3.2.1 标题确为"Scaled Dot-Product Attention"，内容为 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$，与残差无关，已不再并列为公式出处。C5 定义与正文上标仍一一对应。

- [重要·技术] 来源与范围说明 F1、F2（index.html L1047–L1048）与正文 L760、L958、L960：F1 标为"He et al. 2016, §3.1, Eq.(1)"、F2 标为"§3.1, Eq.(2)"，但 Eq.(1) $y=\mathcal{F}(x,\{W_i\})+x$ 与 Eq.(2) $y=\mathcal{F}(x,\{W_i\})+W_sx$ 位于 §3.2 "Identity Mapping by Shortcuts"，§3.1 "Residual Learning" 只有 $\mathcal{H}(x)$ 与 $\mathcal{F}(x):=\mathcal{H}(x)-x$ 的重构（这部分对应 C2，标注正确）。｜引文依据：arXiv:1512.03385 §3.2——"Formally, in this paper we consider a building block defined as: $y=\mathcal{F}(x,\{W_i\})+x$ (1)"；"….we can perform a linear projection $W_s$ by the shortcut connections to match the dimensions: $y=\mathcal{F}(x,\{W_i\})+W_sx$ (2)"。｜修复要求：F1、F2 的定位由"§3.1"改为"§3.2"，Eq. 编号保留；正文出现该公式处（L760、L958、L960）不改变。｜修复：F1、F2 的定位（index.html L1047–L1048）由"He et al. 2016, §3.1"改为"§3.2 Identity Mapping by Shortcuts"，Eq.(1)/Eq.(2) 编号保留；正文出现该公式处（残差块公式一章、4.1 节）未改动。｜复验：arXiv:1512.03385 §3.2 原文"y = F(x, {W_i}) + x  (1)"与"y = F(x, {W_i}) + W_s x  (2)"；§3.1"Residual Learning"只含 $\mathcal{F}(x):=\mathcal{H}(x)-x$ 重构（对应 C2，定位正确未动）。F1 正文引用 3 处、F2 1 处，与来源定义闭合。

- [重要·技术] 4.3 在 Transformer 中的应用（index.html L999–L1003）："现代大语言模型（GPT-2 起）改用 Pre-LN：$\text{output}=x+\text{Sublayer}(\text{LayerNorm}(x))$"属机制/归因陈述，全页无对应 C/F/N 条目，来源章节亦未收录，按 check.md §2.2 应删除或降级为标注的推断。｜引文依据：不适用（页面未给出可定位来源）。｜修复要求：补一条来源条目（给出可定位的出处，如 GPT-2 论文或官方实现路径与行号）并加 `<sup>[Cx]</sup>`；若无法定位来源，删去"GPT-2 起"的归因，改写为"后续工作采用 Pre-LN 形式（本页不展开其采纳史）"，或明确标注为推断。｜修复：删除"现代大语言模型（GPT-2 起）改用 Pre-LN"的无源归因，改写为"后续工作广泛采用 Pre-LN 形式<sup>[C8]</sup>（本页不展开其采纳史）"；新增来源条目 C8：Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture", ICML 2020, arXiv:2002.04745, 摘要。｜复验：arXiv:2002.04745 摘要明确讨论 Post-LN 的梯度问题并提出/分析 Pre-LN（"the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer)"），C8 可定位；页面不再对"GPT-2 起"的采纳史作无源断言。

- [轻微·技术] 来源与范围说明 N1、N2（index.html L1055–L1056）与 overview.html L68：把 Figure 4（ImageNet 18/34 层 plain 与 ResNet 训练曲线）记为"§4.2 Figure 4"，overview 把"34 层 ResNet 优于 18 层 ResNet、34 层 plain 差于 18 层 plain"记为"He et al. 2016 §4.2"；该对比与 Figure 4、Table 2 均在 §4.1 "ImageNet Classification"。｜引文依据：arXiv:1512.03385 §4.1——"Table 2: Top-1 error (%, 10-crop testing) on ImageNet validation"（18 层 plain 27.94 / ResNet 27.88；34 层 plain 28.54 / ResNet 25.03）；§4.1 的 shortcut 选项段亦以 "(the same as Table [2] and Fig. [4] right)" 指称 Figure 4；§4.2 是 CIFAR-10 and Analysis。｜修复要求：N1、N2 与 overview 中该处"§4.2 Figure 4"改为"§4.1 Figure 4"，overview 的"§4.2"改为"§4.1"（1202 层过拟合确在 §4.2，保持不动）。｜修复：N1 的"§4.2 Figure 4"改为"§4.1 Figure 4"；N2 的"§4.1 Table 2 / §4.2 Figure 4 / Abstract"改为"§4.1 Table 2 / §4.1 Figure 4 / Abstract"；overview.html 关键结论中"（He et al. 2016 §4.2）"改为"（He et al. 2016 §4.1）"；1202 层过拟合的 §4.2 保持不动。｜复验：arXiv:1512.03385 §4.1"ImageNet Classification"含 Table 2、Table 3 与 Figure 4 的讨论，§4.2 为"CIFAR-10 and Analysis"（1202 层过拟合在子节"Exploring Over 1000 layers"，确在 §4.2）。index.html 的 N1/N2 与 overview 定位一致。

- [轻微·技术] 来源与范围说明 N3（index.html L1057）与正文 L921：N3 标为"Veit et al. 2016, §3 Figure 5 与摘要"，"110 层 ResNet 中贡献主要梯度的路径长度为 10–34 层"这一数值只在摘要出现，§3 只给出 $2^n$ 路径与路径长度的二项分布，Figure 5 不含该数据。｜引文依据：arXiv:1605.06431 摘要——"most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep."；§3 仅有"It follows that residual networks have $2^n$ paths connecting input to output layers."。｜修复要求：N3 定位删去"§3 Figure 5"，保留"摘要"；如要指 §3，只能用于 $2^n$ 路径这一条（对应 C4）。｜修复：N3 来源由"Veit et al. 2016, §3 Figure 5 与摘要"改为"Veit et al. 2016, 摘要"；正文 10–34 层处上标仍指向 N3。｜复验：arXiv:1605.06431 摘要原文"most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep"；§3 只给出 $2^n$ 路径结论（已由 C4 引用），Figure 5 不含该数据，误标已删。

- [轻微·技术] 4.1 维度不匹配时：投影捷径（index.html L964）："He et al. 2016 的实验显示：维度相同时额外引入可学习投影仅带来微小提升（可归因于参数增加），因此工程上只在维度不匹配时才使用投影捷径。"该实验性归因未标上标，来源章节无对应 C 条目（该处仅有公式 F2）。｜引文依据：arXiv:1512.03385 §4.1 与 Table 3——"C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts. But the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem. So we do not use option C in the rest of this paper…"，Table 3 中 ResNet-34 A/B/C top-1 误差为 25.03 / 24.52 / 24.19。｜修复要求：新增一条 C 条目（He et al. 2016 §4.1 Table 3）并在该句加 `<sup>[Cx]</sup>`；或删除归因句、只保留 F2 的公式与维度条件的陈述。｜修复：新增来源条目 C6：He et al. 2016, §4.1 ImageNet Classification 与 Table 3；在 4.1 节维度不匹配段与原本章问题答案的归因句后各加 `<sup>[C6]</sup>`。｜复验：arXiv:1512.03385 §4.1 原文"C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts... projection shortcuts are not essential for addressing the degradation problem"；Table 3 中 ResNet-34 选项 A/B/C 的 top-1 误差 25.03/24.52/24.19，与 C6 记值一致。C6 正文引用 2 处，与定义闭合。

- [轻微·技术] 4.3 在 Transformer 中的应用（index.html L1005）："Kimi K3 的 AttnRes 块是对这一标准残差连接的扩展——它对残差项做注意力加权而非等权相加"是外部模型的机制归因，本页未给可定位来源（仅链接概念页 `../block-attnres/index.html`，该页自身标 K3 Technical Report §2.2）。｜引文依据：不适用（本页无来源标注）。｜修复要求：补来源标注（Kimi K3 Technical Report §2.2 Eq.(8)–(10)）并加 `<sup>[Cx]</sup>`，或改写为纯指向性表述（"该扩展的机制见 Block AttnRes 页"）而不作机制断言。｜修复：新增来源条目 C7：Kimi K3 Technical Report §2.2 Eq.(8)(9)(10)；在 4.3 节"对残差项做注意力加权而非等权相加"句后加 `<sup>[C7]</sup>`。｜复验：C7 定位与同目录 Block AttnRes 页一致（该页 C/F 条目即指向 K3 报告 §2.2 Eq.(8)(9)(10)），AttnRes 的机制归因现有可定位来源，不再是无源断言。

- [轻微·可读性] 3. 梯度如何流动（index.html L899、L901、L919）：集成解释段落连续重复同一句"以 3 个块为例，展开后得到 $2^3=8$ 条路径"，表前的过渡段又第三次出现"共 $2^3=8$ 条路径"，三句内容重叠。｜引文依据：不适用。｜修复要求：删除 L901 的重复过渡句，或将其改为承接上句的表前引导（如"下表按三个块是否跳过逐一列出这 8 条路径"），保留一处"$2^3=8$"结论即可。｜修复：三处重复只保留一处——集成解释首段结论句保留"$2^3=8$"（句末冒号改句号）；原重复过渡句改为"下表按三个块是否跳过逐一列出这 8 条路径："；表后小结改为"这 8 条路径的长度从 $0$ 到 $3$ 不等。"｜复验：全文"$2^3=8$"仅出现一次（表格前的结论句）；表格 8 行与其后小结均以"8 条路径"承接，无重复过渡。

- [轻微·可读性] 2. 残差块公式的 SVG 结构图（index.html L776–L796）：图仅以 `aria-label="残差块结构图"` 标注，正文未配 figcaption 说明箭头方向（前向计算）与"恒等捷径"分叉的含义；节点含义在下方列表给出，箭头含义无处定义。｜引文依据：不适用。｜修复要求：在图后补一句图注（可用 `<figcaption>` 或正文一句），说明箭头表示前向计算方向、$x$ 分为进入 $F$ 的变换路与直达 $+$ 的恒等路。｜修复：在残差块 SVG 之后、关键论证段之前新增一句图注："图注：箭头表示前向计算方向——输入 $x$ 在此分叉为两路，一路进入变换路 $F(x,\{W_i\})$，另一路走恒等捷径直达 $+$；两路在 $+$ 处逐元素相加得到输出 $y$。"｜复验：图注位于 `</svg>` 与"关键论证"段之间，明确箭头为前向计算方向并区分变换路与恒等路；公式仍置于 `<foreignObject>`，validate.py 通过。

- [轻微·格式] h1（index.html L648，overview.html L42 标题同源）：`<h1 class="title">残差连接：把输入直接加到输出，让深网络可训练</h1>` 缺 style-guide §1 要求的"（英文缩写）"段；同类页（clip/glu/dsa/kv-cache 等）均为"概念名（英文名）：作用"。｜引文依据：不适用。｜修复要求：改为"残差连接（Residual Connection）：把输入直接加到输出，让深网络可训练"，并同步 `<title>` 与 `dojo:summary` 之外的相关表述。｜修复：index.html 的 h1 与 `<title>` 均改为"残差连接（Residual Connection）：把输入直接加到输出，让深网络可训练"；overview.html 的 h1 补为"残差连接（Residual Connection）"。｜复验：与 clip/glu/dsa/kv-cache 同类页"概念名（英文名）：作用"格式一致；index.html 的 `<title>` 与 h1 表述同步，overview.html 标题同源。

- [轻微·格式] 来源与范围说明的 h3（index.html L1053）：写作"外部数字与实验条件"，style-guide §1 固定命名为"外部数字与实验条件（N）"，缺"（N）"后缀（同章"论断与来源（C）""公式与来源（F）"均带标识）。｜引文依据：不适用。｜修复要求：改为"外部数字与实验条件（N）"。｜修复：来源章节 h3"外部数字与实验条件"改为"外部数字与实验条件（N）"。｜复验：来源章节六个固定 h3 现为"论断与来源（C）""公式与来源（F）""外部数字与实验条件（N）""构造示例""辅助解释与类比边界""简化条件及其限制"，与 style-guide §1 固定命名一致。

- [轻微·格式] callout 颜色语义（index.html L703、L923）与 style-guide §3 不符：1 章的「构造论证」（理论构造与优化难度分析）用了 yellow（规范：注意事项/边界条件/易误解澄清），3 章的「集成解释的边界」（明确是边界条件）用了 purple（规范：源码验证/深度分析），两处语义对调。｜引文依据：不适用。｜修复要求：将「构造论证」改为 `callout-purple`（深度分析），「集成解释的边界」改为 `callout-yellow`（边界条件）；或把「构造论证」降为普通段落并只给边界说明保留黄色。｜修复：1 章「构造论证」由 `callout-yellow` 改为 `callout-purple`（深度分析）；3 章「集成解释的边界」由 `callout-purple` 改为 `callout-yellow`（边界条件）。｜复验：全文 callout 仅此两处，颜色语义已对调，与 style-guide §3（yellow＝注意事项/边界条件/易误解澄清，purple＝源码验证/深度分析）一致。

- [轻微·格式] blockquote.meta 主要依据（index.html L652）只列 He et al. 2016 (CVPR)、Veit et al. 2016、Vaswani et al. 2017，未列被 C3/F3/F4 直接依赖的 He et al. 2016 "Identity Mappings in Deep Residual Networks"（ECCV 2016, arXiv:1603.05027），meta 与来源章节的"主要依据"清单不一致。｜引文依据：不适用。｜修复要求：在 meta「主要依据」中补入 arXiv:1603.05027（同一作者同一年的 Identity Mappings 工作）。｜修复：blockquote.meta「主要依据」在 CVPR 2016 之后补入"He et al. 2016, 'Identity Mappings in Deep Residual Networks', ECCV（arXiv:1603.05027）"。｜复验：meta 主要依据现与来源章节的直接依赖一致——C3/F3/F4 引用的 Identity Mappings 已列入，同一作者同年两篇工作（arXiv:1512.03385 / arXiv:1603.05027）均在清单中。

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 10（技术 4、可读性 2、格式 4）
- 处置：返回修复，修复完成后重新运行 `.dojo/scripts/validate.py` 并做第 4 轮全量审查；本轮不通过发布。
- **是否满足 check.md §5 全部发布条件：不满足。** 逐项：
  1. 三轮审查已完成且由未参与写作的独立审查者执行——满足（本轮为独立审查者，据任务说明前两轮亦为独立审查）。
  2. 每条来源论断都有引文依据记录——**不满足**：至少四处标注定位错误或缺失（C5 定位 §3.2.1、F1/F2 定位 §3.1、N1/N2/overview 把 Figure 4 记为 §4.2、N3 多标 §3 Figure 5），另有 GPT-2 Pre-LN、投影捷径实验、Kimi K3 AttnRes 三处机制归因无来源标注。C1/C2/C3/C4/F3/F4 与各数值已逐条核对到原文片段且一致。
  3. 所有阻断和重要问题均已关闭——**不满足**：本轮 4 条重要问题全部未关闭（均为本轮新发现，前两轮未覆盖）。
  4. 遗留轻微问题具有明确的接受理由——**不满足**：本轮 10 条轻微问题均未给出接受理由，需修复或逐条记录接受理由。
  5. 全部学习目标由正文章节完整回答——满足：核心问题 4 条分别由 1/2/3/4 章作答，且每个核心问题答案末尾指明完整论证所在章节。
  6. 两级问题均有解答折叠块、无只列问题未作答——满足：核心问题 4/4、本章问题 3/3/3/3，全部带「解答：」折叠块且答案独立可读（其中 3 章答案中的"梯度至少为 1"须随重要问题 1 一并改写）。
  7. 数学符号全部 LaTeX、结构图为 HTML/内联 SVG——满足：无定界符外 Unicode 数学字符，结构图为内联 SVG 且公式在 `<foreignObject>` 内。
  8. `.dojo/scripts/validate.py` 返回成功——满足（`validation ok`，退出码 0）。
  9. 可运行代码的结果与页面描述一致——满足（无代码块，无需执行；数值示例已手工复算一致）。
  10. 关键论断和数字已重新核对来源——**部分不满足**：ImageNet 数值（27.94/28.54/27.88/25.03）、3.57%/ILSVRC 2015、1202 层过拟合、2–3 层卷积、$2^n$ 路径、10–34 层均已核对；但 3 处公式/图号定位与 3 处无来源归因需修正后重核。
  11. `<head>` 五项元数据齐备且 `dojo:topics` 在词表内——满足（数学基础）。
  12. `overview.html` 与 `index.html` 相互链接——满足。
  13. 页面引用的概念链接有效或具有明确占位——满足（`../block-attnres/index.html` 存在；三个前置概念的页内最小定义符合 style-guide §13，不保留占位）。
  14. 递归生成的前置概念页已完成各自质检——沿用前序轮次状态，本轮无法从允许读取的输入判定；不构成本轮判据。

- 发布结果：未发布（等待修复后第 4 轮复验）。
