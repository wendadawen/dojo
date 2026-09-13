<!-- review-meta
round: 5
page: wiki/hyper-connections/index.html
reviewed_content_sha256: 93f4eeb72bdc0f83
-->
# 超连接与 mHC 审查记录（第 5 轮）

- 页面版本：7780f133785192426bb1a16c27921f4beabd543e（`wiki/hyper-connections/index.html` 工作树 blob）
- 审查时间：2026-09-13 20:18
- 审查者：独立子代理（未参与写作与前序审查；仅依据本页 index.html、overview.html、外部来源与 `guides/concept/check.md`）
- 已完整阅读章节：核心问题、常见误解、1. 恒等映射、2. 超连接、3. 无约束的代价与双随机约束、4. Sinkhorn-Knopp、5. 落地——GLM-5.3-Flash 的 mHC、来源与范围说明

## 问题

- [轻微·表述] §1 首段（第 120 行）：「本页直接取它的结论并指认其中最关键的一条性质」以「本页」作主语叙述本文自身写法，属元话语式自我叙事；与规范允许的「本页推断／本页实测」（guides/concept/style-guide.md §12「自称使用"本页"或"本文"」，用于标注推断或实测来源）用途不同。｜引文依据：不适用｜修复要求：改为直接陈述，删去「本页…取结论并指认」的自我叙事，如「残差连接的背景见残差连接页，关键是其中的恒等映射性质」。｜修复：｜复验：
- [轻微·表述] §2 代码块前引段（第 219 行）与 §4 观察重点（第 421 行）：祈使句直接指挥读者——「该退化由下例验证，并观察 $n=2$ 时两路流如何开始分化。」（主语悬空，以「观察…」指令读者）与「注意「快收敛」是这个矩阵的性质而不是普遍规律：…」。二者与 guides/concept/style-guide.md §12「不直接使用第二人称称呼读者」冲突。｜引文依据：不适用｜修复要求：改为陈述句——「n=2 时两行输出不同：入口虽复制了相同的两路，不同的 pre/post/res 权重立刻让两路分化」；「注意「快收敛」…」改为「「快收敛」是这个矩阵的性质而非普遍规律：…」。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 [N3]（第 575 行）：「$n=1$ 且 $fn=0$ 时退化为普通残差（误差 $3.8\times10^{-6}$，由 $\varepsilon$ 决定）」把整层退化说成普通残差、并把该误差当作与普通残差之差；本条自称「同结论见 wiki/glm-5-3-flash-dataflow/index.html」，但该页对同一实测给出的结论带限定，二者不一致。｜引文依据：wiki/glm-5-3-flash-dataflow/index.html 第 515 行：「实测 $3.8\times10^{-6}$ 这个数度量的是 comb 与单位阵之差经 $h$ 放大后的结果…而不是该构造与 $h+f(h)$ 之差…该构造下的预压缩系数是 pre=σ(0)+hc_eps≈0.5 而非 1，子层输入被减半，与普通残差的 $h+f(h)$ 并不相等…普通残差是这套机制在回写结构上的特例。」｜修复要求：改为「$n=1$ 且 $fn=0$ 时回写结构退化为普通残差形式（comb=1、post=1）；该构造下 pre≈0.5、子层输入减半，整层并不等于 $h+f(h)$」，与所引页一致。｜修复：｜复验：

## 核对记录（回源，§2.2）

- 事实与数字：[N1] 定位到 mHC 论文 §3.1 及图注——「the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1」「HC exhibits an unexpected loss surge around the 12k step」「All results are based on 27B models」；[C6] 定位到 §5.4——「a maximum value of approximately 1.6」。页面 27B／约 3000／约 1.6／约 12k 步的表述与来源一致，且 §3.1 前向增益的定义为「maximum absolute value of the row sums…captures the worst-case expansion in the forward pass」，故正文把行和绝对值最大值称为前向放大（第 340 行）成立。
- 机制论断：[C1] 恒等映射原句、[C7] 三性质（spectral norm $\le1$、closed under matrix multiplication、Birkhoff polytope 为置换矩阵凸包）、[C8]「when n=1, the doubly stochastic condition degenerates to the scalar 1, thereby recovering the original identity mapping」、[C9]「we impose non-negativity constraints…This constrain prevents signal cancellation」、[F4]/[F5]/[F6]（Eq. 6/8/9，含「we choose tmax=20」）——逐句在 §3.1/§4.1/§4.2 定位到，一致。
- 公式：[F1] Eq. 2 原文「$\hat{\mathbf H}=\mathbf B^{\intercal}\mathcal T(\mathbf H^{\intercal}\mathbf A_m)^{\intercal}+\mathbf A_r^{\intercal}\mathbf H$」及 $\mathbf A_m\in\mathbb R^{n\times1}$、$\mathbf A_r\in\mathbb R^{n\times n}$、$\mathbf B\in\mathbb R^{1\times n}$、「when n>1…adjust the strength of residuals but also rearrange layers」——HC 论文 §2.1/§1 逐项一致；Eq. 15 的 Pre-Norm 矩阵确为 $\begin{pmatrix}0&1\\1&1\end{pmatrix}$。
- GLM 规格：[C10]/[C12]/[N2] 经 HF `zai-org/GLM-5.3-Flash` 的 config.json 核对——`architectures=Glm5NextForConditionalGeneration`、`hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-06`、hidden 4096、45 层。算术复核 $24\times16384+24+3=393{,}243$、$\times2\times45=35{,}391{,}870$、$35{,}391{,}870/321.32\text{B}=0.0110\%$；迭代序（先列归一一次、再循环 19 轮行+列归一、末步为列）与所引 glm-5-3-flash-dataflow/index.html 的 [N5]/[N6] 一致。
- 代码实测：三处代码块在本机 torch 复跑，输出与页面「预期输出」逐字一致——n=1 最大差 0.0、n=2 两行 [1.2,2.4,3.6,4.8]/[0.95,1.9,2.85,3.8]、无约束链 8.693e+05、双随机链行和 [1.0,1.0,1.0,1.0]、$3\times3$ 迭代 1 列和 [1.002536,1.035057,0.962407]、迭代 20 行/列偏差 0.0e+00、最大奇异值 1.0。手算 $2\times2$ 链（1.233/0.767 → 0.730/0.130 → 0.860/1.140 → 1.086/0.914）与开篇 $1.5^{45}\approx10^{7.9}$ 均复算通过。
- 机械项：`.dojo/scripts/validate.py` 返回 `validation ok`；正文无 Unicode 数学字符直接出现（仅 meta 纯文本 description 用普通字符，符合规范）；链接 `residual-connection`、`rmsnorm`、`block-attnres`、`glm-5-3-flash-dataflow` 及 `overview.html`↔`index.html` 均有效；正文引用的 `research/measured.md`（本页与交叉页）路径均存在；两级问题均有解答折叠块。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（0 阻断、0 重要；3 项轻微为表述与来源标注一致性打磨，建议顺手修复但不阻断发布）