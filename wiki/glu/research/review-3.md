<!-- review-meta
round: 3
page: wiki/glu/index.html
reviewed_content_sha256: f4e96677a2bb3f72
-->
# GLU 审查记录（第 3 轮）

- 页面版本：ab74bc628c2bd86079671b590eda89ab75db9898
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题、最容易误解的几条、1. 为什么需要"门"——一层线性+激活缺了什么、2. GLU 的公式与手算——定义、边界与逐维度缩放、3. 为什么 GLU 给梯度留了一条线性通路——门值缩放而非导数缩放、4. GLU 家族（4.1 派生规则 / 4.2 记号差异 / 4.3 塞进 Transformer FFN）、5. 经验结论与边界（5.1 GLU 不解决什么）、来源与范围说明；含全部 `<details>` 折叠块、SVG 图注与 overview.html。

## 来源核对摘要（引文依据汇总）

外部来源：Dauphin et al. 2017, arXiv:1612.08083（ar5iv 全文）；Shazeer 2020, arXiv:2002.05202（ar5iv 全文）。

- 定义：原文 §2 Eq.(1) `h_l(X) = (X * W + b) ⊗ σ(X * V + c)`；`X ∈ R^{N×m}`，`W,V ∈ R^{k×m×n}`，`b,c ∈ R^n`。与页面 C1/F1、第 2 章符号表逐项一致。
- 梯度：原文 §3 Eq.(3) `∇[X ⊗ σ(X)] = ∇X ⊗ σ(X) + X ⊗ σ'(X)∇X`；Eq.(2) `∇[tanh(X) ⊗ σ(X)] = tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X)`。与页面第 3 章两式一致，且式号标注（GLU 梯度 Eq.(3)、GTU 梯度 Eq.(2)）正确。
- "乘性跳连"：原文 §3 "This can be thought of as a multiplicative skip connection which helps gradients flow through the layers" 及 "a path ∇X ⊗ σ(X) without downscaling for the activated gating units in σ(X)"。与页面第 316、343 行一致。
- Bilinear：原文 §5.3 `h_l(X) = (X * W + b) ⊗ (X * V + c)`，归因 Mnih & Hinton (2007)。与页面第 378 行一致。
- 变体与 FFN：原文 §2 Eq.(4)/(5)/(6)：`GLU=σ(xW+b)⊗(xV+c)`、`ReGLU=max(0,xW+b)⊗(xV+c)`、`GEGLU=GELU(xW+b)⊗(xV+c)`、`SwiGLU=Swish_β(xW+b)⊗(xV+c)`、`FFN_GLU=(σ(xW)⊗xV)W₂`；"reduce the hidden units d_ff by a factor of 2/3"；§3.1 `d_model=768`、`h=12`、`d_ff=3072→2048`；§4 "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."；§3.2 "The GEGLU and SwiGLU variants produce the best perplexities."。页面第 384–426、462–488 行逐项一致。
- Table 1（"Heldout-set log-perplexity …"，524,288 步）：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633。页面表格数值与 N1 全部一致。
- 复算：σ(1)=0.7311、σ(−0.5)=0.3775、h=[0.7311,0.1888]；σ(10)=0.99995、σ(−5)=0.00669；2·768·3072=4,718,592、3·768·3072=7,077,888（比基线多 50%）、3·768·2048=4,718,592；3072×2/3=2048、4d×2/3=8d/3。全部与页面标注一致，无算式与结论不符。页面无声明可运行的代码块，无代码执行项。
- 机械项：`python3 .dojo/scripts/validate.py wiki/glu/index.html` → `validation ok`；无 Unicode 数学字符（公式全部 KaTeX）；`dojo:type=concept`、`dojo:topics=模型结构`（词表内）、`dojo:tag=网络结构`（词表内）；`../swiglu/index.html`、`../situ-glu/index.html` 两个前置概念页真实存在；overview.html 与 index.html 互链。

## 问题

- 阻断｜—｜全页｜本轮未发现核心结论错误、来源不符或页面不可用的阻断级问题。｜引文依据：不适用｜修复要求：不适用｜修复：｜复验：无。

- 重要｜Dauphin 2017 / Shazeer 2020｜index.html:201（第 1 章末"一句话定位"）｜把无来源支持的判断写成结论：断言 GLU"成为现代 Transformer 前馈层的事实标准结构"，两篇被引论文均无此表述，且 Shazeer 的结论原文明确限定在 T5 并声明"无理论解释"。｜引文依据：页面第 480 行自述"限定条件：T5 base 架构…segment-filling 任务"，第 485 行引文"We offer no explanation as to why these architectures seem to work"；两篇来源均未出现"事实标准"一类表述。｜修复要求：删除"成为现代 Transformer 前馈层的事实标准结构"，或改为标注为推断并补可核对来源（如给出采用该结构的模型清单及出处）；不得以无来源断言保留原意。｜修复：｜复验：

- 重要｜Page 自述来源清单（主要依据仅列两篇论文）｜index.html:426（4.3 节末）｜来源论断无引文依据：断言"这是现代用 SwiGLU 的大模型（如 LLaMA）内部维度取 8/3 d 的来源"，其中"（如 LLaMA）…取 8/3 d"是对具体外部模型的事实陈述，页面"主要依据"未列出可核对来源，读者无法在页面所给来源中定位。overview.html 同一论断（"这是现代大模型 FFN 内部维度取 8/3 d 的来源"）同此问题。｜引文依据：页面第 88 行"主要依据：Dauphin et al. 2017…；Shazeer 2020…"，仅此两项；第 426 行原文"这是现代用 SwiGLU 的大模型（如 LLaMA）内部维度取 $\tfrac{8}{3}d$ 的来源"。｜修复要求：删除"（如 LLaMA）"或补 LLaMA 论文/官方 config 的出处（含文件路径或表号）；overview.html 同步处理。｜修复：｜复验：

- 轻微·表述｜不适用｜index.html:394、307、406、221、370、458、525、152｜元话语、会话指代与固定过渡句式：全文反复以"下一章看/下一章给出/到这一节起"预告后续章节，并出现对读者的第二人称式指代与临场口语。｜引文依据（原文片段）：第 394 行"这里必须停一下，否则读 SwiGLU 会乱。"；第 307 行"会算了，但…这一章看梯度里发生了什么。"；第 406 行"到这一节起读变体时切到 Shazeer 记法。"；第 221 行"下一章给出 GLU 的正式定义并手算一个例子。"；第 152 行"梯度一章会展开它的梯度结构。"（另见 370、458、525 行同类句式）。｜修复要求：按 guides/concept/style-guide.md §8、§12 改写——过渡句只陈述"前一节结论与下一节问题"的逻辑关系，删除"下一章看/这一章看/必须停一下/读…会乱"一类元话语与第二人称指代；引用他章一律用章节标题（第 152 行"梯度一章"改为第 3 章标题）。｜修复：｜复验：

- 轻微·来源｜不适用｜index.html:532、545（来源与范围说明）｜来源条目与正文未双向对应：C2、F2 在来源章节登记，但正文无任何 `<sup>[C2]</sup>`/`<sup>[F2]</sup>` 引用（正文含 sigmoid 导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$ 处亦未标 F2）。｜引文依据（原文片段）：第 532 行"C2（GLU 提供梯度线性通路、保留非线性、缓解梯度消失）：Dauphin §2 原文。"；第 545 行"F2 Sigmoid 与导数：标准定义（基础记号）。"；全文 grep 无对应上标。｜修复要求：二选一——在正文相应位置补 `<sup>[C2]</sup>`/`<sup>[F2]</sup>`，或从来源清单中删除无正文对应的条目（按 style-guide §6 双向对应）。另 C2 标注的出处"Dauphin §2"需复核，梯度线性通路论述在原文 §3。｜修复：｜复验：

- 轻微·格式｜guides/concept/style-guide.md §1｜index.html:554｜来源章节 h3 命名与固定词表不符，缺"（N）"：页面写 `<h3>外部数字与实验条件</h3>`，规范固定命名为"外部数字与实验条件（N）"（其余主流页面如 block-attnres、attention-sink 等均含"（N）"）。｜引文依据：style-guide §1"来源章节（来源与范围说明）下的 h3 使用固定命名…'外部数字与实验条件（N）'"；页面第 554 行无"（N）"。｜修复要求：改为 `<h3>外部数字与实验条件（N）</h3>`。｜修复：｜复验：

- 轻微·技术｜Dauphin §3｜index.html:152｜表述把"单元"当成"现象"：句中将梯度消失现象直接命名为 GTU（Gated Tanh Unit），GTU 是使用 σ 门与 tanh 激活的门控单元，不是该梯度消失现象本身的名称，易造成概念混淆。｜引文依据：原文 §3 将 GTU 定义为 `tanh(X * W + b) ⊗ σ(X * V + c)`（一个单元），梯度消失表述为"Notice that it gradually vanishes as we stack layers because of the downscaling factors tanh'(X) and σ'(X)"（现象）。｜修复要求：改为"这种梯度消失出现在门控单元 GTU（Gated Tanh Unit，…）中"一类表述，把单元与现象分开陈述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（存在 2 项重要问题，须逐条修复后再进入复验；2 项重要问题均为"无来源支持的论断写成结论"，修复方式为删除或补可核对来源。核心公式、梯度推导、Table 1 数字与手算示例经来源逐条核对与复算，均一致；validate.py 通过。已达发布条件的其余机械项均满足。）
