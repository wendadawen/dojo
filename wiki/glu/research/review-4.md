<!-- review-meta
round: 4
page: wiki/glu/index.html
reviewed_content_sha256: a57bfad64472c495
-->
# GLU（Gated Linear Unit）审查记录（第 4 轮）

- 页面版本：24bc96ab1f8dc4c53108c000cfad8081aa9ca006
- 审查时间：2026-09-13 19:39
- 审查者：编排者派发的独立审查者（独立上下文子代理，未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 最容易误解的几条 / 1. 为什么需要"门"——一层线性+激活缺了什么 / 2. GLU 的公式与手算——定义、边界与逐维度缩放 / 3. 为什么 GLU 给梯度留了一条线性通路——门值缩放而非导数缩放 / 4. GLU 家族——换激活、去门控、塞进 FFN / 5. 经验结论与边界——GLU 不保证什么 / 来源与范围说明（含全部折叠块与图注，正文与折叠块逐段通读）

## 核对依据（本轮已回源确认项）

回源材料：Dauphin et al. 2017, arXiv:1612.08083（ar5iv 正文）；Shazeer 2020, arXiv:2002.05202（ar5iv 正文）。

- 定义/公式：GLU 定义 `h_l(X) = (X*W+b)⊗σ(X*V+c)`（Dauphin §2 Eq.(1)）；GTU 梯度 `∇[tanh(X)⊗σ(X)] = tanh'(X)∇X⊗σ(X) + σ'(X)∇X⊗tanh(X)`（Dauphin §3 Eq.(2)）；GLU 梯度 `∇[X⊗σ(X)] = ∇X⊗σ(X) + X⊗σ'(X)∇X`（Dauphin §3 Eq.(3)）；Bilinear `h_l(X) = (X*W+b)⊗(X*V+c)`（Dauphin §5.3，原文注 [Mnih & Hinton 2007]）；变体 `ReGLU/GEGLU/SwiGLU`（Shazeer §2 Eq.(5)）；`FFN_GLU(x,W,V,W₂)=(σ(xW)⊗xV)W₂`（Shazeer §2 Eq.(6)）。全部与页面一致。
- 关键引文：`"This can be thought of as a multiplicative skip connection which helps gradients flow through the layers."`（Dauphin §3）——页面"乘性跳连"归因正确；`"has a path ∇X⊗σ(X) without downscaling for the activated gating units in σ(X)"`（Dauphin §3）——页面第 326/343 行"在开门单元上不额外缩放梯度"与"for the activated gating units"引用正确；`"Notice that it gradually vanishes as we stack layers because of the downscaling factors tanh'(X) and σ'(X)"`（Dauphin §3）——页面 GTU 渐变消失论断有据；`"To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units d_ff by a factor of 2/3..."`（Shazeer §2）——2/3 论断有据；`"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."`（Shazeer §4）——逐字一致。
- 数字：Shazeer Table 1（524,288 步列）ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633——页面表格八个数逐一吻合（该表另有 65,536 步列，页面只取 524,288 列并已标注列条件，无冲突）；`"The GEGLU and SwiGLU variants produce the best perplexities"`（Shazeer §2/§3）与页面"GEGLU 与 SwiGLU 最优"吻合；Table 1 caption `"...segment-filling task from [Raffel et al. 2019]. All models are matched for parameters and computation."` 与页面"segment-filling 任务、参数与计算量匹配"吻合；§3.1 `"we pre-train for 524,288 steps on the span-filling objective on the C4 dataset"` 与"524,288 步""T5 base 架构（12 层、d_model=768、h=12、d_ff=3072→2048）"吻合；`d_ff: 3072→2048`（§3.1）吻合。
- 可复算：手算 σ(1.0)=0.7311、σ(−0.5)=0.3775，h=[1.0,0.5]⊗[0.7311,0.3775]≈[0.7311,0.1888] 复算一致；极端例 σ(10)≈0.99995、σ(−5)≈0.00669，h≈[1.0,0] 复算一致；参数量 2·768·3072=4,718,592、3·768·3072=7,077,888（+50%）、3·768·2048=4,718,592 复算一致；3·d·d_ff′=2·d·d_ff ⇒ d_ff′=⅔d_ff，取 d_ff=4d 时 8d/3 一致。符号 ⊗、σ、X、W、V、b、c 全页单义。
- 图与结构：SVG 为内联 SVG，图内公式均在 `<foreignObject>` 中经 KaTeX 渲染，`<text>` 仅含"常规层/GLU/值分支/门分支"等纯文字；节点与箭头含义在图后文字中给出。
- 页面功能：`python3 .dojo/scripts/validate.py wiki/glu/index.html` 返回 `validation ok: wiki/glu/index.html`；无重复 id、无 `【】` 占位、无"（待生成）"；index.html↔overview.html 互链存在；`../situ-glu/index.html`、`../swiglu/index.html` 目标页真实存在。两级问题块（核心问题 5 条、5 个正文章节各 3 条/2 条"本章问题"）均带 `解答：` 折叠块，答案独立可读且与正文一致，核心问题答案末尾均指向对应章节。
- 可执行代码：本页无任何 `<pre><code>` 代码块，"实际执行代码"一项不适用（静态审查）。

## 问题

- [轻微·表述] 第 1 章末（第 154 行）：以"后面会看到，GLU 的梯度里有一条不被导数压没的通路"预告后文，属元话语式表述；同源问题另见第 221 行"结构图只给出分支走向，未给精确公式"（与第 199 行"图只展示结构"重复自述页面自身图内容）｜引文依据：不适用｜修复要求：第 154 行改为无预告的直接陈述（如"这一步把『按维度调节』与『是否杀梯度』分开：GLU 的梯度里有一条不被门导数压没的通路"）；第 221 行删去"结构图只给出分支走向，未给精确公式"，仅保留指向第 2 章的衔接句｜修复：｜复验：
- [轻微·技术] 第 1 章（第 152 行）：GTU 以内联式 $\tanh(\mathbf{X})\otimes\sigma(\mathbf{X})$ 当作定义引入，但该式是论文 §3 梯度分析用的"两分支共享输入"简写，论文的 GTU 功能式带仿射项；页面「简化条件及其限制」只登记了 GLU 的这条简化，未覆盖 GTU，读者可能据此认为 GTU 无学习权重｜引文依据：论文 §3 GTU 写作 "tanh(X*W+b)⊗σ(X*V+c)"（含 W,b,V,c）；简写 "∇[tanh(X)⊗σ(X)]" 仅出现在梯度分析式｜修复要求：在第 152 行 GTU 处补注该式沿用 §3 的共享输入简写、一般形式两支各带 $\mathbf{X}*\mathbf{W}+\mathbf{b}$ 与 $\mathbf{X}*\mathbf{V}+\mathbf{c}$；或在「简化条件及其限制」增列该条｜修复：｜复验：
- [轻微·格式] 第 135 行：误解块标题为「最容易误解的几条」，全站其余 17 个概念页（swiglu、situ-glu、mqa-gqa、latent-moe、clip、muon-optimizer 等）统一为「最容易误解」｜引文依据：不适用｜修复要求：标题改为「最容易误解」｜修复：｜复验：
- [轻微·格式] 第 251、275 行：「构造示例。 取 $x=[1.0,\,0.5]$」「构造示例。 把 $V$ 换成 …」，句号后多一个空格｜引文依据：不适用｜修复要求：删除句号后的多余空格｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（本页核心结论、公式编号、Table 1 数字、引文与手算均已逐条回源核实无误；4 条轻微问题关闭后可发布）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
