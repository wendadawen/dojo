<!-- review-meta
round: 8
page: wiki/rope/index.html
reviewed_content_sha256: de4ea2aa432433b7
-->
# RoPE 旋转位置编码审查记录（第 8 轮）

- 页面版本：5c59abc0f60ab63be51a2b0e9c436a1eb4733fbe
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 页面类型：concept（依据 head 内 `dojo:type=concept`），适用规范 `guides/concept/check.md`，格式规范 `guides/concept/style-guide.md`
- 已完整阅读章节：核心问题（5 条解答折叠块）→ 1. 为什么 Transformer 需要位置编码——RoPE 与四类方案的差异（含表格、callout、本章问题）→ 2. 2 维 RoPE 是怎么旋转的——最小可手算机制（含内联 SVG 与图注、折叠块「展开：2 维手算例子的完整代入过程」、本章问题）→ 3. 内积为何只依赖 $m-n$——旋转矩阵群的性质（含折叠块「补充：$R_m^T R_n = R_{n-m}$ 的三角恒等式推导」、本章问题）→ 4. d 维推广——分块对角旋转矩阵与 $\theta_i$ 几何级数（含两份表格、折叠块「展开：$d=4$ 多频率手算例子」、本章问题）→ 5. 远程衰减——相位抵消与 QK-only 机制（含折叠块「补充：从内积公式到归一化和的推导」、衰减表、callout、本章问题）→ 6. 适用边界——长度外推、K3 MLA 的 NoPE 选择（含 6.1、6.2、callout、本章问题）→ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）→ overview.html。

## 外部来源核对（逐条，含原文片段/数值）

- arXiv:2104.09864（RoFormer）摘要：确认含 "encodes the absolute position with a rotation matrix"、"incorporates the explicit relative position dependency in self-attention formulation" 与 "flexibility of sequence length"，与 C2、§6.1「常见误解」的引文与解释一致。
- 论文编号定位（ar5iv 全文）：Eq.(11) 在 §3.1（相对位置编码目标 g(x_m,x_n,m−n)）；Eq.(12) 2 维复数解、Eq.(13) 2 维旋转矩阵形式；Eq.(14) d 维推广、Eq.(15) 分块对角旋转矩阵 R^d_Θ,m、Eq.(16) 内积 x^T W_q R^d_Θ,n−m W_k x_n；§3.4.1「Derivation of RoPE under 2D」含 Eq.(20)–(33)、§3.4.2「Computational efficient realization…」含 Eq.(34)、§3.4.3「Long-term decay」含 Eq.(35)–(37)。与页首 blockquote、C1、F3、F4、F6、F8 的编号标注一致。
- Θ 与 θ_i 原文：Eq.(15) 后 "pre-defined parameters Θ={θi=10000−2(i−1)/d,i∈[1,2,...,d/2]}"；§3.3 "Following Vaswani et al. 2017, we set θi=10000−2i/d."。C3/C4/F6/N1 所述「固定常数」「与 Vaswani 2017 同源」「与本页 i=0 写法数学等价」成立（i 起点差 1，值序列相同）。
- §3.4.3 图 2 原文："the value of 1/(d/2)∑|S_i| decay with the relative distance m−n increases … as shown in Figure 2"，Figure 2 caption "Long-term decay of RoPE."；Abel 变换在 Eq.(36)–(37)。C5、F8 的引文与「Abel 变换给出上界」一致。
- 数值复算（Python，$d=128$、base=10000，$\theta_i=10000^{-2i/128}$）：$\theta_0=1$（2π 圈长 6.283）、$\theta_1=0.86596$（7.256）、$\theta_{16}=0.1$（62.83）、$\theta_{63}=1.154782\times10^{-4}$（2π/θ63=54410.14）。第 4 章表格四行与正文「约 6.3 个位置」「约 54410 个位置」「$1.15\times10^{-4}$」全部吻合。
- 衰减表复算：$\frac{1}{64}\sum_{i=0}^{63}\cos(dd\cdot\theta_i)$ 得 dd=0→1.0000、1→0.9702、5→0.7373、10→0.6691、50→0.5462、100→0.4772、500→0.2985、1000→0.1590、5000→−0.0070、10000→−0.0279，与第 5 章表格及核心问题 4 的四点采样值逐项一致；$|m-n|$ 在 4500–5600 区间确穿过 0 并振荡，「5000 附近穿过 0」成立。
- 手算例复算：$q=k=(1,0)$、$\theta=\pi/4$、$m=1,n=2$ → $\sqrt2/2\approx0.7071$；$q=(1,1),k=(0,1),\theta=\pi/6,m=0,n=1$ → $(\sqrt3-1)/2\approx0.3660$；$d=4$、base=4、$q=k=(1,0,1,0)$ → $\cos1+\cos0.5=0.5403+0.8776=1.4179$；F2 展开、$R_m^T R_n=R_{n-m}$ 的逐元素乘法、极坐标形式 $\sum_i r_{q,i}r_{k,i}\cos((n-m)\theta_i+\phi_{k,i}-\phi_{q,i})$ 均复算正确。
- 外部链接：arXiv:2306.15595（PI, Chen et al. 2023）、arXiv:2309.00071（YaRN, Peng et al. 2023）、arXiv:2402.13753（LongRoPE, Microsoft 2024）、arXiv:2305.19466（NoPE, Kazemnejad et al. 2023）编号与作者/年份匹配。
- Kimi K3：技术报告（arXiv:2607.24653 / MoonshotAI 发布）与第三方内核文档确认 K3 为 KDA（Kimi Delta Attention）+ Gated MLA 混合，MLA 层 `mla_use_nope=True`、不施加旋转，位置信息经 KDA 的衰减/门控递归隐式提供，分阶段扩展（8K→64K→256K→1M）后支持 1M token。C8、6.2 与核心问题 5 的表述与之一致。
- 页内链接：`../../wiki/{standard-attention,positional-encoding,causal-mask,nope,mla,kimi-k3}/index.html` 均真实存在；`overview.html` 与 `index.html` 相互链接；无「（待生成）」占位。`.dojo/scripts/validate.py wiki/rope/index.html` 返回 `validation ok`。
- 图：内联 SVG，公式均在 `<foreignObject>` 的 `.dg-label` 中由 KaTeX 渲染，无 `<text>`  ASCII 近似；图注读数（$R_1q=(0.71,0.71)$、$R_2k=(0,1)$、内积 0.707）与刻度/向量端点一致；`aria-label` 为纯文本、不含 `$...$`。

## 问题

- [重要·可读性] 位置：第 1 章「为什么 Transformer 需要位置编码」表格后段落（index.html:151）｜问题：该段称「前三类都在"加"位置信息，区别只是加在哪里、怎么加」，把表格第 3 行的 RoPE 也归入「加」；紧接着的下一句却写「RoPE 那一行的特征是：它**乘**而不是加……」。同段相邻两句对同一对象（RoPE 的注入方式）给出相反表述，与表格第 3 行「按绝对位置 $m$ 旋转 Q 和 K（乘法）」也冲突，会直接混淆本页「加 vs 乘」这一核心区分｜引文依据：不适用（页内两处自相矛盾）｜修复要求：把「前三类」改为「前两类」，或改写为「APE 与相对位置编码两类在"加"位置信息，区别只是加在哪里、怎么加。RoPE 那一行的特征是：它乘而不是加……」，使该段不再声称 RoPE 属于「加」｜修复：｜复验：
- [轻微·格式] 位置：来源与范围说明·外部数字与实验条件（N）（index.html:632–633）｜问题：N1（base 默认 10000）、N2（$d=128$ 时 $\theta_0=1$、$\theta_{63}\approx0.0001$）在来源章节列出，但正文、折叠块、图注与 overview 中均无 `<sup>[N1]</sup>`/`<sup>[N2]</sup>` 引用（全文上标仅有 C1–C8、F1–F8、N3、N4），不符合 style-guide §6「与来源章节双向对应」｜引文依据：不适用｜修复要求：在正文首次出现 $\mathrm{base}=10000$ 处补 `<sup>[N1]</sup>`、在第 4 章 $d=128$ 频率表处补 `<sup>[N2]</sup>`，或删除 N1、N2 两条｜修复：｜复验：
- [轻微·可读性] 位置：第 2 章末尾段（index.html:251）｜问题：「下一章会看到具体数值」所指的下一章是第 3 章「内积为何只依赖 $m-n$」，该章通篇不含任何真实模型 $\theta_i$ 数值（仅用 $m=1,n=2$ 与 $m=5,n=6$ 复算 $\theta=\pi/4$）；真实模型的 $\theta_i$ 具体数值出现在第 4 章「d 维推广」的表格。跨章指路指向错误章节｜引文依据：不适用｜修复要求：改为「第 4 章会看到具体数值」或「d 维推广一章会看到具体数值」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复