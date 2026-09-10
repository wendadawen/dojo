# 超连接审查记录（第 2 轮）

- 页面版本：d127e54c9aaf9c496bf47d702c2e39b83f3f5ddc
- 审查时间：2026-09-10 22:51
- 审查者：编排者派发的独立审查者（未参与写作，未参与第 1 轮审查与修复）
- 已完整阅读章节：核心问题（5 题解答）、常见误解、1. 恒等映射：残差连接保住了什么、2. 超连接：把一条流加宽成 $n$ 条、3. 无约束的代价与双随机约束、4. Sinkhorn-Knopp：把矩阵投影到双随机流形、5. 落地：GLM-5.3-Flash 的 mHC、来源与范围说明；含全部 18 个 `<details>` 折叠块（3 个代码块、1 个手算展开、14 个解答）。

## 机械验证结果

`/usr/bin/python3 .dojo/scripts/validate.py wiki/hyper-connections/index.html` → `validation ok`（退出码 0）；overview.html 同命令亦通过。

| 检查项 | 结果 |
|---|---|
| 引用双向闭合 | 通过。正文 28 处 `<sup>` 引用（C1–C5、C7–C14、F1–F8、N1–N4）与来源章节一一对应；无无源引用，无未被引用的来源条目 |
| 相邻双上标 | 通过。全文无 `</sup><sup>` 或 `</sup> …<sup>` 相邻出现 |
| Unicode 数学字符 | 通过。`×≤≥≈∈∑Σ∂√⊙∞⁻ᵀθαβγσεμλ` 逐字符计数均为 0，数学符号全部由 KaTeX 渲染 |
| TAB | 通过。全文制表符计数 0 |
| 占位符 | 通过。无「待生成 / TODO / 待补 / placeholder」 |
| `<details>` 与 `</details>` | 通过。18 / 18；`<summary>` 18 个，前缀只用「解答：/代码：/展开：」三种 |
| `<head>` 五项元数据 | 通过。`description`（纯文本）、`dojo:summary`（含 `$...$`）、`dojo:type=concept`、`dojo:topics=模型结构`（在 AGENTS.md 固定词表内）、`dojo:tag=残差结构` |
| overview 互链 | 通过。overview.html → `index.html`「完整说明 →」；index.html 导航 → `overview.html`「概览」 |
| 前置概念链接 | 通过。`residual-connection`、`rmsnorm`、`block-attnres` 三页均存在；正文无「（待生成）」占位 |
| 本地资源 | 通过。`katex.min.css`、`katex.min.js`、`auto-render.min.js`、`prism-primer-light.css`、`prism-primer-dark.css`、`prism.min.js`、`prism-python.min.js` 七个文件均在 `../../libs/` 下存在 |
| 结构图 | 通过。两张图为内联 SVG，`<text>` 元素计数 0，全部标签在 `<foreignObject>` 内由 KaTeX 渲染；无等宽字符框线图；节点/箭头含义在 figcaption 与正文中定义 |

代码实跑（本机 torch 2.13.0）。三个 Python 代码块用脚本从页面原文提取后直接执行，与页面「预期输出」逐行比对：

| 代码块 | 退出码 | 比对结果 |
|---|---|---|
| n=2 的 HC 层与 n=1 退化验证 | 0 | 逐字符一致（`n=1 输出: [1.1, 2.2, 3.3, 4.4]`、`最大差: 0.0`、`两行是否已不同: True`）；与预期块仅差文件结尾换行 |
| 无约束链与双随机链的 24 层对照 | 0 | 一致，打印 `1.095e+06` 与 `[1.0, 1.0, 1.0, 1.0]` |
| $3\times3$ Sinkhorn 收敛与谱范数 | 0 | 一致，迭代 1–5 列和数值逐位相同，迭代 20 行/列和偏差 `0.0e+00`，最大奇异值 `1.0` |

来源核对（按 check.md §2.2 四步，逐条给出原文片段或代码片段）：

- 论文部分经 `https://arxiv.org/html/2409.19606v3`（HC）与 `https://arxiv.org/html/2512.24880v2`（mHC）核对：
  - [C1] mHC §1 与页面引文逐字一致："The term identity mapping refers to the component $\mathbf{x}_l$ itself, which emphasizes the property that the signal from the shallower layer maps directly to the deeper layer without any modification."
  - [C2] HC §1："The vanishing gradient and the representation collapse are like two ends of a seesaw"；"residual connections, including both Pre-Norm and Post-Norm variants, predefine the strength of connections between the output and input within a layer."
  - [C3] HC §3.1：Eq.(15) $\mathcal{HC}_{PreNorm}=\begin{pmatrix}0&1\\1&1\end{pmatrix}$、Eq.(16) 含 $\sigma_i,\sigma_o,\sigma_{io}$ 的归一化系数版；"Therefore, their hyper-connection matrices are non-trainable."
  - [C4] HC §1："We argue that n (>1) hidden states are necessary. As analyzed in Appendix F, the seesaw effect persists when n=1, and experiments show that it does not improve performance"；"when n>1, hyper-connections can not only learn to adjust the strength of residuals but also rearrange layers".
  - [C5][F3] mHC §1 Eq.(4) 与页面公式逐项一致（含两个 $\prod$ 的上下限）；"the composite mapping … fails to preserve the global mean of the features … unbounded signal amplification or attenuation"。
  - [F1] HC §2.1 Eq.(2) 与页面公式逐项一致（$\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_m})^{\intercal}+\mathbf{A_r}^{\intercal}\mathbf{H}$）。
  - [F2] mHC §1 Eq.(1) $\mathbf{x}_{l+1}=\mathbf{x}_l+\mathcal{F}(\mathbf{x}_l,\mathcal{W}_l)$、Eq.(2) 递归展开，均逐项一致。
  - [F4] mHC §4.1 Eq.(6) 与页面约束式逐项一致。
  - [F5] mHC §4.2 Eq.(7)(8) 与页面参数化式（含 RMSNorm、$\sigma$/$2\sigma$/Sinkhorn-Knopp）逐项一致。
  - [F6] mHC §4.2 Eq.(9) $\mathbf{M}^{(t)}=\mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))$；"We choose $t_{max}=20$ as a practical value in our experiments."
  - [C7][N1] mHC §3.1："the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1"、"HC exhibits an unexpected loss surge around the 12k step"、Fig.2/Fig.3 图注均写明 27B 模型；§5.4："the deviation increases but remains bounded, reaching a maximum value of approximately 1.6. Notably, compared to the maximum gain magnitude of nearly 3000 in HC, mHC significantly reduces it by three orders of magnitude."
  - [C8] mHC §4.1 三条性质原句："the spectral norm of a doubly stochastic matrix is bounded by 1"、"The set of doubly stochastic matrices is closed under matrix multiplication"、"the Birkhoff polytope, which is the convex hull of the set of permutation matrices"。
  - [C9] mHC §4.1："when n=1, the doubly stochastic condition degenerates to the scalar 1, thereby recovering the original identity mapping."
  - [C10] mHC §4.1 末段："we impose non-negativity constraints on the input mappings $\mathcal{H}_l^{pre}$ and output mappings $\mathcal{H}_l^{post}$. This constrain prevents signal cancellation arising from the composition of positive and negative coefficients."
- 本机源码（逐条打开确认路径与行号存在且含所引内容）：
  - `wiki/deepseek-v4-1/research/official/inference/kernel.py`：L407 `def hc_split_sinkhorn_kernel(hc, sinkhorn_iters, eps)`、L474 `return pre, post, comb`（即 [C11] 所称 L407–474 区间）；L427 `pre[i, j] = T.sigmoid(...) + eps`；L429 `post[i, j] = 2 * T.sigmoid(...)`；L430–443 comb 投影与 softmax+$\varepsilon$；L445–448 先做一次列归一（`comb.sum(-2)`）；L450–458 循环 `sinkhorn_iters - 1` 次「行归一 + 列归一」，末步为列；L448/L454 分母 `+ eps`；L409 `mix_hc = (2 + hc) * hc`——全部命中。
  - `.../model.py`：L938 `mix_hc = (2 + hc_mult) * hc_mult`；L941–946 六个 `hc_attn_*`/`hc_ffn_*` 参数（形状 `mix_hc × hc_dim`、`mix_hc`、`3`）；L948–955 `hc_mixes`（flatten → rsqrt 归一 → `F.linear` → `hc_split_sinkhorn`）；L962–966 `hc_post`，函数体 `post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` 即 [F8] 的 $h'=post\otimes y+comb^{\top}h$；L1257–1258 `# Expand to hc_mult copies for Hyper-Connections` + `h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)`——全部命中。
  - `wiki/glm-5-3-flash-dataflow/research/config.json`：L3 `Glm5NextForConditionalGeneration`、L5 `text_config`、L15 `"hc_eps": 1e-06`、L16 `"hc_mult": 4`、L17 `"hc_sinkhorn_iters": 20`，与 [C11] 标注的 L16/L17/L15 一致。
  - `.../p3.out` §3.1–3.8 全部存在：§3.1 `单站点合计 393,243`、`全模型 45 层 → 35,391,870`、`mix = (2+4)*4 = 24`；§3.3 行和最大偏差 `1.007e-02`、列和最大偏差 `1.073e-06`，并写明「循环体最后一步是列归一」；§3.5 `h' = post ⊗ sublayer_out + comb^T @ h`；§3.6 退化最大误差 `3.815e-06`；§3.7「无权重均值」「hc_head 无任何可学习参数」；§3.8 激活张量放大 4×。
  - `.../verify_structure.out` [10] `参数量合计 = 321,323,031,390 (321.32 B)`，与页面 0.011% 的分母一致；复算 35,391,870 / 321,323,031,390 = 1.10e-4 = 0.011%。
  - `wiki/block-attnres/index.html` 存在，其 h2「2. Full AttnRes 的公式——pseudo-query 如何检索前序层」「3. Block AttnRes 的分块与块间 attention」支持 [C14]。
  - 第 1 轮曾出现的「标注源码路径在本机不存在」问题已不复现：本页引用的每一条路径与行号均已逐一打开确认存在且内容相符。

## 问题

- [重要·技术] 4. Sinkhorn-Knopp「本章问题」第 2 题解答（index.html:1128）：解答把近似双随机的偏差方向说成「方向是「略微收缩」而不是无界放大」。mHC 论文 §5.4 报告复合映射增益是「deviation increases but remains bounded, reaching a maximum value of approximately 1.6」，即近似投影下复合增益可以略大于 1；本页 index.html:982 与 overview.html:63 也自述「最大增益降到约 1.6」。同一份偏差在页内被描述成「收缩」与「约 1.6 的增益」两种相反方向。｜引文依据：mHC 论文 §5.4 "In the composite case shown in Fig. 7(b), the deviation increases but remains bounded, reaching a maximum value of approximately 1.6."；`p3.out` §3.3 行和范围 `[0.98993468, 1.00366139]`（含大于 1 的行和）｜修复要求：把「方向是「略微收缩」」改为与来源一致且有界的表述，例如「偏差不再随深度无界累积，复合增益有界（论文实测最大约 1.6），与无约束情形有本质区别」，或删除该分句｜修复：已按 mHC 论文 §5.4 改写 index.html:1128：删除「方向是「略微收缩」而不是无界放大」，改为「有限次迭代只得到近似双随机，偏差意味着行和或列和离 1 有距离，复合后该偏差随深度增大但仍有界——论文实测复合映射的增益偏差最大约 1.6，与无约束时随深度指数累积的情形有本质区别<sup>[C6]</sup>」，并补「偏差不是单向收缩：本页实现的实测中行和最高到 $1.0037$（略大于 1）」，对齐 `p3.out` §3.3 行和范围 $[0.98993468, 1.00366139]$（含大于 1 的行和）。页内矛盾消除：index.html:982 与 overview.html:63 均为「最大增益约 1.6」，与本处「有界、最大约 1.6」方向一致。｜复验：论文 §5.4 原文复读确认 "the deviation increases but remains bounded, reaching a maximum value of approximately 1.6"；新 [C6] 条目与正文 1128 双向闭合；grep「略微收缩」在 index.html、overview.html 均 0 命中；validate.py 通过。
- [轻微·格式] 来源与范围说明·论断与来源（C）（index.html:1234–1246）：编号从 [C5] 直接跳到 [C7]，正文与来源小节均无 [C6]，出现断号。｜引文依据：不适用｜修复要求：把 [C7] 及其后各条顺次改为 [C6]…[C13] 并同步更新正文 5 处引用，或补回 [C6] 条目并写明其内容｜修复：采用「重编号」方案，index.html 正文与来源小节的 [C7]–[C14] 共 16 处引用全部顺次前移为 [C6]–[C13]——正文 8 处（[C6, N1]、[C7]、[C8]、[C9]、[C10, N2]、[C11]、[C12]、[C13]），来源小节 8 条，使 C 系列连续为 [C1]–[C13]。｜复验：脚本统计正文引用集合与来源条目集合均为 {C1…C13}，双向闭合、无缺失、无未被引用条目；全文再无非连续编号；validate.py 通过。
- [轻微·格式] 核心问题第 2 题解答（index.html:771）：写作「隐藏状态从 $\mathbb{R}^{d}$ 变成 $\mathbb{H}\in\mathbb{R}^{n\times d}$」，而正文其余各处该超隐藏矩阵一律写作 $\mathbf{H}$（index.html:840「堆叠成超隐藏矩阵 $\mathbf{H}\in\mathbb{R}^{n\times d}$」、index.html:842 公式、index.html:846、index.html:953）。同一变量两种写法，违反 style-guide §11「同一变量在页面中保持同一种写法」。｜引文依据：不适用｜修复要求：把 index.html:771 的 `$\mathbb{H}$` 改为 `$\mathbf{H}$`，全页只保留一种写法｜修复：index.html:771 的 `$\mathbb{H}\in\mathbb{R}^{n\times d}$` 改为 `$\mathbf{H}\in\mathbb{R}^{n\times d}$`，全页超隐藏矩阵统一为 $\mathbf{H}$。｜复验：grep `\mathbb{H}` 命中 0；index.html:771/840/842/846/953 五处均为 $\mathbf{H}$；validate.py 通过。
- [轻微·格式] 章节标题（index.html:808、836、1053、1133）：「1. 恒等映射：残差连接保住了什么」「2. 超连接：把一条流加宽成 $n$ 条」「4. Sinkhorn-Knopp：把矩阵投影到双随机流形」「5. 落地：GLM-5.3-Flash 的 mHC」四处 h2 的副标题用全角冒号。style-guide §1 规定副标题格式为 `1. 主题——副标题`（同目录 residual-connection、block-attnres 均用「——」）。｜引文依据：不适用｜修复要求：四处 h2 的「：」改为「——」｜修复：四处 h2 副标题的全角冒号改为「——」——index.html:808「1. 恒等映射——残差连接保住了什么」、836「2. 超连接——把一条流加宽成 $n$ 条」、1053「4. Sinkhorn-Knopp——把矩阵投影到双随机流形」、1133「5. 落地——GLM-5.3-Flash 的 mHC」。｜复验：四条 h2 均改为 `N. 主题——副标题`、副标题处再无全角冒号；与同目录 residual-connection、block-attnres 一致；validate.py 通过。
- [轻微·格式] 正文跨章引用（index.html:955、964、978、982、998、1069、1111、1112、1128 共 9 处）：使用「见第 4 章」「第 3 章会看到」「第 2 章末尾指出」「第 5 章 GLM 的实测」等编号式引用。style-guide §1 要求「正文引用其他章节时使用章节标题」，本页核心问题解答已采用「完整论证见「恒等映射」一章」的正确写法，两种方式并存。｜引文依据：不适用｜修复要求：把 9 处「第 N 章」改为对应章节标题（如「无约束的代价与双随机约束」），与核心问题解答的引用方式统一｜修复：9 处编号式引用改为对应章节标题——955「见「Sinkhorn-Knopp」一章」、964「本章会看到」（该处即本节）、978「「超连接」一章末尾指出」、982「对比「恒等映射」一章的递归展开」、998「见「Sinkhorn-Knopp」一章」、1069「「落地」一章 GLM 的实测」、1111「（「落地」一章）」、1112「在「落地」一章讨论」、1128「「落地」一章 GLM 的实测」。｜复验：grep「第 1 章/第 2 章/第 3 章/第 4 章/第 5 章」在 index.html 命中 0；引用方式与核心问题解答的「见「某章」一章」统一；validate.py 通过。
- [轻微·技术] 3. 无约束的代价与双随机约束·代码「无约束链与双随机链的 24 层对照」（index.html:1015–1016 打印语句、index.html:1027 预期输出、index.html:1031 观察重点、1270 行 [N4]）：标签写「复合矩阵行和绝对值最大」，实际计算的是 `G.abs().sum(1).max()`，即 $\max_i\sum_j|G_{ij}|$（矩阵的无穷范数）；而论文 §3.1 定义的 Amax Gain Magnitude 是行和的绝对值的最大值 $\max_i|\sum_j G_{ij}|$。本机同一 seed 复算：前者 `1.095e+06`，后者 `8.693e+05`。｜引文依据：mHC 论文 §3.1 "the maximum absolute value of the row sums of the composite mapping, captures the worst-case expansion in the forward pass"｜修复要求：二选一——把标签改为「行上绝对值之和最大」，或把代码改为 `G.sum(1).abs().max()` 并同步更新预期输出与 [N4]，使打印量名与论文 Amax Gain 定义一致｜修复：采用「改代码对齐论文定义」方案——index.html:1016 的 `G.abs().sum(1).max()`（$\infty$ 范数）改为 `G.sum(1).abs().max()`，即论文 §3.1 的 Amax Gain Magnitude $\max_i|\sum_j G_{ij}|$；预期输出 1027 由 `1.095e+06` 改为 `8.693e+05`；[N4]（1266）同步为「行和绝对值的最大值 $\max_i|\sum_j G_{ij}|=8.693\times10^{5}$」；「验证的机制」1030 补注打印量即论文 §3.1 的 Amax Gain Magnitude（引 [N1]）；1031 观察重点措辞改为「行和绝对值最大」。｜复验：本机 torch 2.8.0 实跑两种度量得 `abs().sum(1).max()=1.095e+06`、`sum(1).abs().max()=8.693e+05`；三个代码块从页面原文抽取后实跑，`block 1/2/3 match=True`；validate.py 通过。
- [轻微·技术] 4. Sinkhorn-Knopp 正文符号说明与「本章问题」第 1 题解答（index.html:1061、1121）：把 post 用 $2\sigma$、pre 不乘 2 写成设计动机（「子层输出写回各路时的平均强度接近普通残差的 1」「pre 不乘 2 是因为它的职责是把 $n$ 路求和成一路，读出权重不需要围绕 1」）。核对 mHC 论文 §4.2 Eq.(8) 只有定义与一句 σ 说明，正文与附录均未给出该系数差异的理由。｜引文依据：mHC 论文 §4.2 Eq.(8) 后仅 "where $\sigma(\cdot)$ denotes the Sigmoid function."，无 $2\sigma$ 动机说明｜修复要求：保留可验证部分（$2\sigma$ 值域 $(0,2)$、中心为 1），把「目的/理由」分句标注为推断（如「本页推断」），不写成论文的设计说明｜修复：保留可验证部分、把动机分句标注为推断——index.html:1061 改为「非负且取值 $(0,2)$、中心为 1（sigmoid 中心 0.5 乘 2）——论文只给出该参数化，本页推断其用意是让写回各路的平均强度接近普通残差的 1」；1120 summary 改为「解答：$2\sigma$ 取值 $(0,2)$、中心为 1（对齐 1 的用意为本页推断）」；1121 正文改为「post 乘 2 后范围变成 $(0,2)$、中心是 1。论文只给出这一参数化，本页推断其用意：……」与「pre 不乘 2（本页推断）：……」。｜复验：mHC 论文 §4.2 Eq.(8) 后仅 "where σ(·) denotes the Sigmoid function."，无 $2\sigma$ 动机说明，页面不再将其写成论文设计说明；$(0,2)$、中心为 1 等可验算术保留；validate.py 通过。
- [轻微·可读性] overview.html:43、51：概览正文出现前置概念「残差网络 / 单路残差 / 恒等映射」但无链接，全文只链接 Dojo 首页与 index.html。write.md §5 要求「概览面向不了解该领域的读者：首次出现的前置概念给出概念页链接（与完整说明页同一份映射）」，而 index.html 已链接 residual-connection 页。｜引文依据：不适用｜修复要求：在 overview.html 首次出现「残差流 / 恒等映射」处加指向 `../residual-connection/index.html` 的链接｜修复：overview.html:43 把首次出现的前置概念「残差网络」、:51 的「恒等映射」分别用 `<a href="../residual-connection/index.html">` 包裹，与完整说明页的前置概念映射一致。｜复验：`wiki/residual-connection/index.html` 在本机存在；overview.html 运行 validate.py 通过（本地引用完整性检查覆盖该链接）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 7
- 处置：修复
