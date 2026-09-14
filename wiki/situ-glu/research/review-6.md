<!-- review-meta
round: 6
page: wiki/situ-glu/index.html
reviewed_content_sha256: e7292bd51e068ccc
-->
# SiTU-GLU 审查记录（第 6 轮）

- 页面版本：d209d59668a740c1356c9e4b6d1866b37cd3ece5
- 审查时间：2026-09-14 17:14
- 审查者：独立子代理
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么需要给 SwiGLU 加上界——激活爆炸从哪里来（含本章问题）/ 2. SiTU-GLU 的公式与手算——定义、符号与三个边界点（含本章问题、展开折叠块）/ 3. 近原点像 SwiGLU、远点饱和——两个性质怎么同时成立（含本章问题、两个补充折叠块）/ 4. 为什么是 softcap 而不是 clip——饱和区里梯度差别（含本章问题、补充折叠块）/ 5. 在 K3 中的使用位置与不解决——两处使用与四条边界（含本章问题）/ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；另对照 overview.html 全文。

## 来源核对（逐条回源，均定位到原文）

来源为 Kimi K3 Technical Report（arXiv:2607.24653v2），本轮通过 arXiv HTML/PDF 抓取正文与附录，PDF 文本提取后逐段核对；SwiGLU 公式回源 Shazeer 2020《GLU Variants Improve Transformer》（arXiv:2002.05202）PDF。核对结果：

- §2.3.2 Eq.(12)：「SiTU-GLU(x) = β1 tanh(Wg x/β1) ⊙ Sigmoid(Wg x) ⊙ β2 tanh(Wu x/β2)」，与页面 F1/C1 完全一致（原文该式写作两括号乘积）。
- §2.3.2：「siTU-GLU applies the smooth cap softcap(x, β) = β tanh(x/β) to the linear factor of the Swish gate and independently to the up branch」，与页面 F2/C2 一致。
- §2.3.2：「both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk in low-precision arithmetic」——页面正文引用的英文片段逐字一致（C3/C9）。
- §2.3.2：「The sigmoid gate of the original GLU avoids unbounded gate growth, but it does not retain the approximately linear positive regime of Swish」——与 C3 一致。
- §2.3.2：「we set the soft-cap hyperparameters to β1 = 4 for the gate branch and β2 = 25 for the up branch」——与 C4/N1 一致，且确认门支 β1=4、值支 β2=25 的对应关系未颠倒。
- §B Eq.(18)：「β tanh(z/β) = z + O(z^3/β^2)」——与 F3 一致；紧接一句「SiTU-GLU therefore matches SwiGLU to first order around the origin. It also recovers SwiGLU pointwise as β1, β2 → ∞」——与 C5/C6/F5 一致。
- §B Eq.(19)：「Since |tanh(z)| < 1 and 0 < Sigmoid(z) < 1, every output coordinate satisfies ∥SiTU-GLU(x)∥∞ ≤ β1β2 = 100, for β1 = 4 and β2 = 25」——与 C7/F4 一致。
- §B 末段：「Unlike hard clamping of gate pre-activations, the smooth cap preserves nonzero gradients away from saturation boundaries, which we find to give better training behavior.」——页面整句引用逐字一致（C8）。
- §B 第二段：「Kimi K3 applies the same construction to the up branch as β2 tanh(Wu x/β2), preventing either branch from dominating the product.」——页面引用的「preventing either branch from dominating the product」原文一致（C2）。
- §2.3 开头：「the routed path composes W↓, a gated multi-branch expert feed-forward network, and W↑ into a chain of nearly four consecutive matrix multiplications... combined with the 2.8-trillion-parameter scale, produces exploding internal activations」——与「近四连矩阵相乘 + 2.8T」一致（C9）。
- §3 Table 1 行「Activation Function | SwiGLU | SiTU-GLU」——与 C10/N2 一致（K2→K3 模型级替换，无性能数字）。
- §2.3.1 RMSNorm「between expert aggregation and the up-projection... reduces the sensitivity of the routed branch to scale variation」——与第 5 章三件套表一致；§2.3.3 QB 与 896 专家池一致。
- 文献编号：K3 报告 [26]=Dauphin 2017、[108]=Shazeer 2020，与页面 meta 的引文归属一致。
- Shazeer 2020 §2 Eq.(5) 确含 SwiGLU 定义「SwiGLU(x, W, V, b, c, β) = Swishβ(xW + b) ⊗ (xV + c)」——F6 的章节/公式号正确。

## 复算与机械核对

- 全部手算点用 Python 复算：y(0)=0；y(2)=3.249323（页面 3.2493）；y(10)=37.484606（页面 37.485）；y(50)=96.402758（页面 96.403）；y(100)=99.932930（页面 99.933）；对照 SwiGLU 0/3.5232/99.9955/2500/10000 全部吻合。
- 分项：g(10)=3.946278（3.9463）、u(10)=9.498724（9.4987）、u(100)=24.983232（24.983）、tanh(2.5)=0.986614（≈0.987）、tanh(0.4)=0.379949（≈0.380）、tanh(4)=0.999329。
- 折叠块：4tanh(0.125)=0.497412 与 z=0.5 差 −0.002588（页面 −0.00259），理论 −z³/(3β²)=−0.002604（页面 −0.00260），残差 1.62×10⁻⁵（页面 ≈1.6×10⁻⁵）——吻合。
- 梯度：4e⁻⁵⁰=7.71×10⁻²²（页面 7.7×10⁻²²）、4e⁻⁸=1.34×10⁻³（页面 1.3×10⁻³）——吻合。
- KaTeX：用 libs/katex.min.js 以 throwOnError 逐个解析页面核心公式（含 `\mathrm{SiTU\text{-}GLU}`、上界式、极限式、导数式、dojo:summary）全部通过，无解析错误。
- `.dojo/scripts/validate.py wiki/situ-glu/index.html` 返回 `validation ok`，退出码 0。
- 公式定界符外的 Unicode 数学字符仅出现在 description（纯文本，规范要求，允许）与「K2→K3 / SwiGLU→SiTU-GLU」的箭头（非数学变量），无违规。
- 链接：wiki/glu/index.html 存在；index.html 与 overview.html 互链；无「（待生成）」占位。
- 无 `<img>`，故无 alt 含 `$...$` 的情形；无结构图，故不涉及 `<foreignObject>` 规则。

## 问题

- [轻微·来源] 来源与范围说明 C2、正文 2 章符号列表（第 195 行：「（K3 §2.3.2 第二段 "to the linear factor of the Swish gate"[C2]）」）与第 160 行、第 297 行：把「softcap 套到门支线性因子」这句话定位为 §2.3.2 **第二段**，实际在 **第三段**｜引文依据：§2.3.2 段序为 P1「Gated Linear Units (GLUs) modulate…remains open.」、P2「However, both multiplicative factors in SwiGLU are unbounded…trade-off [52].」、P3「To satisfy these requirements, we propose Sigmoid Tanh Unit GLU (SiTU-GLU). SiTU-GLU applies the smooth cap softcap(x, β) = β tanh(x/β) to the linear factor of the Swish gate and independently to the up branch:（Eq.12）」——所引句子在 P3，而页面 C3 用「第二段」指 P2（该处正确），同一「第二段」在页面内被同时用于两段，指向不自洽｜修复要求：把 C2 及第 195 行对「to the linear factor of the Swish gate」的出处由「§2.3.2 第二段」改为「§2.3.2 第三段」（或统一写「§2.3.2」）；同时把 C9 的出处由「§2.3 开头」补为「§2.3 开头、§2.3.2 第二段」，与第 66、135 行正文的用法一致｜修复：｜复验：
- [轻微·表述] 全页口语化措辞：第 66 行「直接撑爆低精度训练」「模型进了饱和区就死」、第 365 行「这个坐标"死"了」、第 208 行「负侧因 sigmoid 杀尾」、第 425 行「三件套」（overview 第 48 行同）｜引文依据：不适用（其中「撑爆」对应来源仅作「increase overflow risk in low-precision arithmetic」，措辞强度高于来源；「撑爆/就死/杀尾」全站仅本页出现）｜修复要求：将「撑爆」「就死/死了」「杀尾」换成中性表述（如「溢出」「梯度被截断、无法恢复」「sigmoid 把负侧压到 0」），「三件套」改为「三个组件/三件稳定化」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布。全部事实性论断、公式、数字、手算与引文编号已回源核对一致（含 arXiv HTML/PDF 原文与 Shazeer 2020 原文），KaTeX 渲染与 validate.py 均通过；仅遗留 2 项轻微（一处引文段落号标注偏差、一处口语化措辞），不影响正确性与主线理解，可在本轮顺手修掉或按轻微项接受。