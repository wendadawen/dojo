<!-- review-meta
round: 6
page: wiki/hyper-connections/index.html
reviewed_content_sha256: 706c65661b8462cc
-->
# 超连接与 mHC 审查记录（第 6 轮）

- 页面版本：9499283287ad760eeb8aa2bcc600a8dbc1255f6a（工作树）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：引言、核心问题、常见误解、1 恒等映射、2 超连接、3 无约束的代价与双随机约束、4 Sinkhorn-Knopp、5 落地、来源与范围说明（含全部折叠块与图注）

## 核对摘要（来源与代码，逐条回源）

- 两个来源的真实性：arXiv:2409.19606v3 标题 Hyper-Connections、作者 Defa Zhu 等、v3 修订 2025-03-18；arXiv:2512.24880v2 标题 mHC: Manifold-Constrained Hyper-Connections、作者 Zhenda Xie 等、v2 修订 2026-01-05。与 meta「主要依据」一致。
- [C1] 引文逐字命中 mHC §1：「The term identity mapping refers to the component x_l itself…maps directly to the deeper layer without any modification」；同处确有「early research (He et al., 2016b)」并指向 Identity Mappings in Deep Residual Networks。
- [F3]/[C5] mHC Eq.(4) 与第 290 行公式逐字一致：x_L=(∏_{i=1}^{L-l}H^{res}_{L-i})x_l+Σ_{i=l}^{L-1}(∏_{j=1}^{L-1-i}H^{res}_{L-j})H^{post⊤}_i F(H^{pre}_i x_i,W_i)（本页把 F 的参数写成 ⋯，属省略）。
- [N1]/[C6] §3.1「yields extreme values with peaks of 3000」「a stark divergence from 1」；§3.1「HC exhibits an unexpected loss surge around the 12k step」；§5.4「reaching a maximum value of approximately 1.6」「maximum gain magnitude of nearly 3000 in HC」「reduces it by three orders of magnitude」。第 65/292 行的 3000、1.6、12k 步全部命中。
- [F4] §4.1 Eq.(6) 三式与第 296 行一致；[C7] 三性质逐字命中（spectral norm bounded by 1 / closed under matrix multiplication / convex hull of permutation matrices）；[C8]「when n=1, the doubly stochastic condition degenerates to the scalar 1…recovering the original identity mapping」；[C9]「we impose non-negativity constraints on the input mappings H^pre and output mappings H^post」「signal cancellation arising from the composition of positive and negative coefficients」。
- [F5] §4.2 Eq.(7)/(8)：H^pre=σ(H̃^pre)、H^post=2σ(H̃^post)、H^res=Sinkhorn-Knopp(H̃^res)；[F6] Eq.(9) M^(t)=T_r(T_c(M^(t-1))) 与「We choose t_max=20」。另确认 §4.2 的输入确为展平向量 x→_l=vec(x_l)∈ℝ^{1×nC} 后过 RMSNorm——第 365 行「展平成 1×nC…过 RMSNorm」有据。
- [F1] HC §2.1：HC 矩阵 Eq.(1)、输出 Ĥ Eq.(2)、h₀=A_mᵀH Eq.(3)、H′=A_rᵀH Eq.(4)，第 152 行公式与 Eq.(2) 逐字一致；[C3] §3.1 Pre-Norm/Post-Norm 的 Eq.(15)/(16) 编号与 2×2 矩阵（Pre-Norm 为 [[0,1],[1,1]]，Post-Norm 为归一化系数版本）命中。
- 全部代码块实跑（python3，torch）：三个代码块的输出与页面「预期输出」逐字符一致——HC 层 n=1 退化（最大差 0.0）、n=2 分化；无约束链 24 层 max_i|Σ_j G_ij|=8.693e+05、双随机链行和 [1,1,1,1]；3×3 五次迭代的行/列和、20 次偏差 0.0e+00、最大奇异值 1.0。
- 手算复算：1.5^45≈10^7.9（7.924）；对称例 [[3,1],[1,3]]；非对称例 [[9,1],[1,2]] 的 0.233→0.770/0.767→0.086 全部可复算（列和 0.860/1.140、行和 1.086/0.914）。参数账 393,243=24×16384+24+3、786,486、35,391,870、占 321.32 B 的 0.011%（实算 0.011014%）均可复算。
- GLM 规格交叉核对 wiki/glm-5-3-flash-dataflow/index.html：hc_mult=4、45 层、hidden_size=4096、321.32 B、35.39 M、单站点 393,243、hc_sinkhorn_iters=20、hc_eps、行和 1.0e-2/列和 1.1e-6、迭代序「先列归一一次再 19 轮行、列、末步列方向」逐项一致。
- 引用编号：正文 [C1]–[C13]、[F1]–[F8]、[N1]–[N4] 均与来源小节双向对应，无悬空引用。链接 ../residual-connection/、../rmsnorm/、../block-attnres/ 与被引用的 overview.html 均存在；`research/measured.md` 存在。validate.py 返回 validation ok。图表为内联 SVG + foreignObject，aria-label 与 img alt 中无 `$…$`。

## 问题

- [轻微·技术] §1 符号定义（第 125 行）与 §2（第 150、155 行）、§4（第 365 行）：同一个「隐藏宽度」在两章用了两个符号，且全页未说明二者相等。｜引文依据：§1「$\mathbf{x}_l\in\mathbb{R}^{C}$：第 $l$ 层的隐藏状态，$C$ 为宽度」；§2「堆叠成超隐藏矩阵 $\mathbf{H}\in\mathbb{R}^{n\times d}$」与「压成单路 $\mathbf{h}_0\in\mathbb{R}^{d}$」；§4「先把 $n$ 条流展平成 $1\times nC$ 的向量」。三处均指同一隐藏宽度。｜修复要求：把 §1、§4 的 $C$ 统一为 $d$（或反之），若确需保留两个来源各自的记号，则在 §2 首次出现 $d$ 处注明 $d$ 即 §1 的 $C$。｜修复：｜复验：
- [轻微·表述] [N3]（来源小节，第 575 行）：条目在「列和偏差 1.1×10⁻⁶、行和 1.0×10⁻²」之后，附加了一段本页正文从不引用的内容——GLM 的 $n=1$ 且 $fn=0$ 构造、以及「实测 $3.8\times10^{-6}$ 度量的是 comb 与单位阵之差…不是该构造与 $h+\mathcal{F}(h)$ 之差」的含义澄清。该段没有对应的正文论断，措辞为辩护式叙述（「该构造下…整层并不等于…——实测…度量的是…不是…」），读起来像审查回应/调试叙事而非页面内容。｜引文依据：本页对 [N3] 的引用仅第 516 行一处，只用其中两个偏差数字；「$3.8\times10^{-6}$」「$fn=0$」经全文检索仅出现在第 575 行的 [N3]，正文（含图注、折叠块）无任何论断引用它。｜修复要求：删除该段；若保留，则须在正文补上它所支撑的论断（例如说明 GLM 回写结构在 $n=1$ 时退化到何种形式）后再保留。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题不改变任何结论与主线理解，可在下一次清理中一并处理或带理由接受）
