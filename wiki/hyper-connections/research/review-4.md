<!-- review-meta
round: 4
page: wiki/hyper-connections/index.html
reviewed_content_sha256: aef30fe75dda7c0a
-->
# 超连接与 mHC 审查记录（第 4 轮）

- 页面版本：5be690bb9175eb353b085e7e18723778aeab8977
- 审查时间：2026-09-13 19:40
- 审查者：独立子代理
- 已完整阅读章节：核心问题、常见误解、1. 恒等映射——残差连接保住了什么、2. 超连接——把一条流加宽成 n 条、3. 无约束的代价与双随机约束、4. Sinkhorn-Knopp——把矩阵投影到双随机流形、5. 落地——GLM-5.3-Flash 的 mHC、来源与范围说明（含全部折叠块与两张图注）

## 来源核对记录（本轮实际打开来源定位的条目）

- HC 论文 arXiv:2409.19606v3 §2.1 Eq. 2 原文 "Ĥ = Bᵀ𝒯(HᵀA_m)ᵀ + A_rᵀH"，形状 A_m: n×1、A_r: n×n、B: 1×n、H: n×d；"Initially, h⁰∈ℝᵈ is replicated n times … Here, n is the expansion rate"。[F1] 一致。
- HC 论文 §3.1 Eq. 15/16：Pre-Norm `(0 1; 1 1)`、Post-Norm 为归一化系数 `1/√(σᵢ²+σₒ²+2σᵢₒ)` 版本。[C3] 与本章问题解答一致。
- HC 论文 §1："residual connections, including both Pre-Norm and Post-Norm variants, predefine the strength of connections between the output and input within a layer"；"vanishing gradient and representation collapse are like two ends of a seesaw"；"the seesaw effect persists when n=1 … does not improve performance"；"when n>1, hyper-connections can not only learn to adjust the strength of residuals but also rearrange layers, either sequentially or in parallel"。[C2][C4] 一致。
- mHC 论文 arXiv:2512.24880v2 §1：恒等映射句 "The term identity mapping refers to the component x_l itself … without any modification"、残差递归式、Eq. 4 复合展开（两项逐字核对，含 ∑ 内的 ∏ ℋ^res · ℋ_i^{post⊤} ℱ(ℋ_i^pre x_i, W_i)）。[C1][F2][F3] 一致。
- mHC 论文 §3.1 "Amax Gain Magnitude"：定义为复合矩阵行和绝对值最大（前向）/ 列和绝对值最大（反向）；"peaks of 3000"；实验对象为 27B 模型；"loss surge around the 12k step"。[N1]、代码块「验证的机制」一致。
- mHC 论文 §4.1 Eq. 6 双随机约束；三条性质原文 "The spectral norm of a doubly stochastic matrix is bounded by 1"、"closed under matrix multiplication"、"forms the Birkhoff polytope, which is the convex hull of the set of permutation matrices"；"when n=1, the doubly stochastic condition degenerates to the scalar 1, thereby recovering the original identity mapping"；非负约束 "we impose non-negativity constraints on the input mappings ℋ_l^{pre} and output mappings ℋ_l^{post}"（均确在 §4.1）。[C7][C8][C9][F4] 一致。
- mHC 论文 §4.2 Eq. 7/8 `ℋ^pre=σ(·)`、`ℋ^post=2σ(·)`、`ℋ^res=Sinkhorn-Knopp(·)`；Eq. 9 `M^(t)=𝒯_r(𝒯_c(M^(t-1)))`、`t_max=20`。[F5][F6] 一致。§5.4 最大增益 "approximately 1.6"。[C6] 一致。
- GLM-5.3-Flash：与 `wiki/glm-5-3-flash-dataflow/index.html` 逐项交叉一致——hc_mult=4、hidden_size=4096、单站点 `24×16384+24+3`、两站点×45 层 = 35,391,870、占 321.32 B 的 0.011%；回写式 `h = post.unsqueeze(-1)*x + comb.transpose(-1,-2) @ residual`（即 `post⊗y + combᵀh`）；Sinkhorn 迭代序「先列归一 1 次，再循环 19 轮行+列，末步列方向」；残差 20 次后列和 1.1×10⁻⁶ / 行和 1.0×10⁻²；`pre=σ+hc_eps`。[C10][C11][C12][F8][N2][N3] 一致。
- 算术复算：1.5⁴⁵≈10^7.9 ✓；24×16384+24+3=393,243 ✓；×2×45=35,391,870 ✓；35,391,870/321.32e9=0.0110% ✓；(2+4)×4=24 ✓。
- 代码：三个代码块均按页面原样执行（torch 2.8.0），输出与「预期输出」逐字一致——n=1 最大差 0.0、n=2 两行 [1.1,2.2,3.3,4.4]/[1.05,2.1,3.15,4.2]；无约束链 8.693e+05、双随机链 [1.0,1.0,1.0,1.0]；3×3 五轮列和与「迭代 20 偏差 0.0e+00」、最大奇异值 1.0。
- 机械项：`python3 .dojo/scripts/validate.py wiki/hyper-connections/index.html` 返回 `validation ok`；`dojo:type=concept`、`dojo:topics=模型结构`（ALLOWED_TOPICS）、`dojo:tag=网络结构`（ALLOWED_TAGS）均在词表内；description 为纯文本、summary 含 `$...$`；前置页 residual-connection / rmsnorm / block-attnres 均存在；overview↔index 互链；页面引用的 `research/measured.md`（本页）与 `wiki/glm-5-3-flash-dataflow/research/measured.md` 均存在，且前者清单中含 `hc_page_code.out`；两级问题块（核心问题 5 条、各章本章问题 1/2/2/2/2 条）均有解答折叠块且核心问题答案指明了论证章节。

## 问题

- [重要·技术] 「2. 超连接」折叠块（代码第 230 行 `return A_r @ H + torch.outer(B, out)` 对应正文第 152 行公式）：代码的 res 项按 `A_r @ H` 实现，同页公式却是 `+ A_rᵀ H`，而代码块下方又声明"代码按公式 Ĥ = Bᵀ 𝒯(HᵀA_m)ᵀ + A_rᵀ H 逐项实现 pre、res、post 三个映射"——符号 A_r 在公式与代码两处含义不一致，且按公式复算得到的数值与「预期输出」不符。｜引文依据：HC 论文 §2.1 Eq. 2 原文 "Ĥ = Bᵀ𝒯(HᵀA_m)ᵀ + A_rᵀH"；取页面给定的非对称 `A_r=[[0.9,0.1],[0.2,0.8]]` 按公式 `A_rᵀH` 复算，n=2 输出行 0/行 1 = [1.2,2.4,3.6,4.8] / [0.95,1.9,2.85,3.8]（实测得到），而页面「预期输出」按代码算得 [1.1,2.2,3.3,4.4] / [1.05,2.1,3.15,4.2]。｜修复要求：让公式与代码逐项一致——把代码 res 项改为 `A_r.T @ H`（并据实更新第 259–260 行的「预期输出」后重跑核对），或在公式与图注中明确 A_r 存储的是转置约定；二者取其一，不得保留"代码按公式逐项实现"这一不成立的断言。｜修复：｜复验：
- [轻微·表述] 「3. 无约束的代价与双随机约束」第 3 段："还有一个漂亮的退化事实：$n=1$ 时…"——"漂亮的"是作者对内容的临场评价，非正文陈述。｜引文依据：不适用｜修复要求：删去评价性修饰，改为中性陈述（如"还有一个退化事实"）。｜修复：｜复验：
- [轻微·表述] 引言第 1 段："这不是假想的危险：把残差流从一条加宽到多条…"——以作者口吻强调意义的临场评价/修辞断言。｜引文依据：不适用｜修复要求：改为直接陈述论文实测事实（如"HC 在 27B 模型上实测过残差通路增益峰值约 3000…"），删去"这不是假想的危险"这一句。｜修复：｜复验：
- [轻微·表述] 「2. 超连接」首段"在引入多路结构之前，先看单路残差长期存在的权衡"与本章问题第 1 题解答"本章会看到，信号爆炸正是从这里发生的"——两处为预告式导航句（"先看…""本章会看到…"），属元话语。｜引文依据：不适用｜修复要求：改为直接给出内容（如"单路残差长期存在一个权衡：…"；"信号爆炸正发生在这里"），删去预告式句式。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（逐条修复后重跑 `.dojo/scripts/validate.py` 与两个代码折叠块；第 1 条须同步更新「预期输出」数值并复算核对）