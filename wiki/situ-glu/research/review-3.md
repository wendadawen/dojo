<!-- review-meta
round: 3
page: wiki/situ-glu/index.html
reviewed_content_sha256: 7c7d931491d6c36e
-->
# SiTU-GLU 审查记录（第 3 轮）

- 页面版本：5768e0b392de7f9a3319f9fcb5d3617de679b510
- 审查时间：2026-09-13 19:10
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 来源获取：Kimi K3 Technical Report（arXiv:2607.24653v2，`https://arxiv.org/html/2607.24653v2`，正文与 Appendix B）；Shazeer 2020《GLU Variants Improve Transformer》（arXiv:2002.05202，ar5iv 全文）。页面未给出报告外链，来源按报告标题+章节号定位。
- 已完整阅读章节（含全部折叠块）：head/description、导航与目录、主要依据 meta、引言、核心问题（5 条）、最容易误解、1 为什么需要给 SwiGLU 加上界、2 SiTU-GLU 的公式与手算、3 近原点像 SwiGLU、远点饱和、4 为什么是 softcap 而不是 clip、5 在 K3 中的使用位置与不解决、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）；并对照阅读 overview.html。
- 机械验证：`.dojo/scripts/validate.py wiki/situ-glu/index.html` 返回 `validation ok`（exit 0）。页面全部数字以 Python 复算通过：$y(0)=0$；$y(2)=1.6281\cdot1.9957=3.2493$，SwiGLU $3.5232$；$y(10)=3.9463\cdot9.4987=37.485$，SwiGLU $99.9955$；$y(50)=4\cdot24.101=96.403$；$y(100)=4\cdot24.983=99.933$，SwiGLU $10000$；$4\tanh(0.125)=0.497412$、$-z^3/(3\beta^2)=-0.002604$；$4e^{-50}=7.7\times10^{-22}$、$4e^{-8}=1.3\times10^{-3}$。数值示例与标注一致，无算式错误。

## 问题

- [重要·技术] 来源章节号错误（K2→K3 架构对比表）：正文第 433 行「（K3 §2.3 + §4 对比表）」、第 437 行「K2→K3 架构对比表（§4）」、第 439 行「K2→K3 架构对比表（§4）」、第 480 行「K2→K3 架构对比表（§4）」、来源章节 C10（第 514 行）「K3 §4 对比表」、N2（第 533 行）「来源：K3 §4 对比表」。｜引文依据：报告中「Table 1: Architectural comparison between Kimi K2 and Kimi K3.」位于 §3 Pre-Training 之下的 §3.2 Scaling Law（引言句为「Table 1 provides a detailed architectural comparison between Kimi K2 and Kimi K3」），表中行「Activation Function｜SwiGLU｜SiTU-GLU」；报告 §4 的标题是「Post-Training」，§4 全节为后训练，无架构对比表。页面把 §3 的 Table 1 标成 §4，指向了错误位置。｜修复要求：把上述 6 处「§4 对比表」统一改为「§3 Table 1」（或「§3 架构对比表」），并核对 C10/N2 的来源描述同步更新。｜修复：｜复验：

- [重要·技术] 局部展开式的 O 阶与来源 Eq.(18) 不一致：F3（第 521 行）、C5（第 509 行），以及正文第 98、282、292、299–300、340、499 行。页面写 $\beta\tanh(z/\beta)=z+O((z/\beta)^3)$。｜引文依据：报告 Appendix B Eq.(18) 的 MathML alttext 为 `\beta\tanh\!\left(\frac{z}{\beta}\right)=z+O\!\left(\frac{z^{3}}{\beta^{2}}\right).`，即 $z+O(z^3/\beta^2)$；页面写成的 $(z/\beta)^3$ 等于 $z^3/\beta^3$，与来源差一个因子 $\beta$。页面第 300 行补的说明「$O((z/\beta)^3)$ 量纲上对应 $\beta\cdot(z/\beta)^3=z^3/\beta^2$」把 $\beta$ 塞进 O 记号，不是标准读法，且与同一段上一行 $\beta\tanh(z/\beta)=z-z^3/(3\beta^2)+O(z^5/\beta^4)$ 所显示的残差阶 $z^3/\beta^2$ 自相矛盾（该行推出的结论与紧随其后的等式写法不符）。｜修复要求：按来源把 F3/C5 及各引用处改为 $z+O(z^3/\beta^2)$；若确要按 $z$ 展开（$\beta$ 取定值时），须显式声明「$\beta$ 为常数」并写成 $z+O(z^3)$，不得写 $(z/\beta)^3$；删除第 300 行把 $\beta$ 塞进 O 标记的解释句。｜修复：｜复验：

- [轻微·表述] 元话语/设问式引导语。位置：第 82 行「本文要回答：它怎么做到…」；第 240 行「读这张表要看到三件事：」；第 278 行「读到这一章，读者可能会有疑问：…」；第 363 行「读者到这里可能又会问：…」；第 402 行「另外要注意：K3 §B 末段那句…」。｜引文依据：不适用（check.md 2.2 第 12 项；style-guide §12 用词）。｜修复要求：改为直接陈述；例如第 82 行直接写「两个乘性因子平滑饱和到固定上界、同时保留近原点近线性响应」，删去「本文要回答」「读这张表要看到」「读者可能会有疑问」「读者到这里可能又会问」「另外要注意」等引导语，需要承接逻辑时用一句结论句过渡。｜修复：｜复验：

- [轻微·表述] 第一人称复数「我们」。位置：第 390 行「…我们观察到这给出更好的训练行为」；第 402 行「只是说"我们观察到"」。｜引文依据：style-guide §12「自称使用"本页"或"本文"，不使用第一人称复数」。｜修复要求：把「我们观察到」改为「报告观察到」或「报告称」，两处同步修改。｜修复：｜复验：

- [轻微·格式] 来源编号未双向对应。位置：来源章节 F7（第 525 行 $\tanh'(z)=1-\tanh^2(z)$）、F8（第 526 行 clip 定义）、F9（第 527 行 clip 导数）在正文中没有任何 `<sup>` 引用（正文全部引用仅含 C1–C10、F1–F6、N1–N2）。｜引文依据：不适用（style-guide §6「正文使用 `<sup>[Cx]</sup>` 上标引用…与来源章节双向对应」）。｜修复要求：在正文第 371–372 行（softcap/clip 定义与导数）补 `<sup>[F7, F8, F9]</sup>`，或从来源章节删除 F7/F8/F9 三条。｜修复：｜复验：

- [轻微·格式] 冗余资源与死 CSS。位置：head 第 23–27 行的 `prism-primer-light.css`/`prism-primer-dark.css`/`prism.min.js`/`prism-python.min.js`，以及第 35–44 行的 `.diagram` 样式；正文与脚本无 `<pre>`/`<code>` 代码块，也无 `class="diagram"` 元素（`.code-block pre` 仅出现在复制按钮脚本里，无对应元素）。｜引文依据：不适用（check.md 2.2 第 8 项 页面功能/第 13 项 格式一致性；仓库既有「清死 CSS」约定）。｜修复要求：删除未被使用的 prism 引用与 `.diagram` 规则，或补上实际使用它们的元素。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复

补充说明（不构成独立问题，供修复参考）：页面核心论断均已在来源中定位并核对一致——SiTU-GLU 定义 Eq.(12)、$\beta_1=4,\beta_2=25$、上界 $\|\mathrm{SiTU\text{-}GLU}\|_\infty\le\beta_1\beta_2=100$（Eq.19，来源用 $\infty$ 范数，页面用逐坐标绝对值 $|\cdot|$，含义等价，可不改）、$\beta_1,\beta_2\to\infty$ 逐点收敛 SwiGLU（Eq.18 后一句）、hard clamping 引文「Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior.」（逐字一致）、「preventing either branch from dominating the product」与「to the linear factor of the Swish gate」（逐字一致）、GLU 不采用的依据（「The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish.」）、2.8T 规模与「a chain of nearly four consecutive matrix multiplications」、§2.3 两条失败模式、RMSNorm 在 §2.3.1、QB 在 §2.3.3、SwiGLU 公式 F6 出自 Shazeer 2020 §2 Eq.(5)（已核对）。全部手算与构造示例数值复算无误。两级问题块、来源小节的六个固定 h3、章节编号与前置 section 顺序均符合 style-guide。本轮无阻断问题；两条重要问题须关闭后方可发布。
