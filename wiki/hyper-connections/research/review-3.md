<!-- review-meta
round: 3
page: wiki/hyper-connections/index.html
reviewed_content_sha256: 249aa2d82661b411
-->
# 超连接与 mHC 审查记录（第 3 轮）

- 页面版本：9135bb916453e26531fd0c54c7d534027b9b77e9（git hash-object）
- 审查时间：2026-09-13 19:06
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、常见误解、1. 恒等映射——残差连接保住了什么（含本章问题）、2. 超连接——把一条流加宽成 $n$ 条（含本章问题）、3. 无约束的代价与双随机约束（含本章问题）、4. Sinkhorn-Knopp——把矩阵投影到双随机流形（含本章问题）、5. 落地——GLM-5.3-Flash 的 mHC（含本章问题）、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）；含全部折叠块与两幅 SVG 图注。

## 核对方法与已核验通过项（供复验参考）

- 三处代码块均在 torch 2.8.0 下实跑：n=1/n=2 HC 层输出（`n=1 输出: [1.1, 2.2, 3.3, 4.4]`、`最大差: 0.0`、`两行是否已不同: True`）、24 层无约束链对照（`8.693e+05`、双随机链行和 `[1.0, 1.0, 1.0, 1.0]`）、$3\times3$ Sinkhorn（迭代 1–5 列和与 20 次后 0.0e+00、最大奇异值 1.0）——全部与页面「预期输出」逐位一致。
- 论文侧经 WebFetch 抓 arXiv HTML 原文逐条核对通过：mHC §1 原句「The term identity mapping refers to the component x_l itself…」[C1]；§3.1「We refer to these metrics as the Amax Gain Magnitude…」且定义为 composite mapping 行和绝对值最大，与代码打印量一致；§4.1「…both the rows and columns sum to 1」「当 n=1, the doubly stochastic condition degenerates to the scalar 1, thereby recovering the original identity mapping」「we impose non-negativity constraints on the input mappings H^pre and output mappings H^post. This constrain prevents signal cancellation arising from the composition of positive and negative coefficients」[C7][C8][C9]；§4.2「flatten it into a vector x→l=vec(xl)∈R^{1×nC}」「x→l′=RMSNorm(x→l)」「M^(0)=exp(H~res)」「M(t)=T_r(T_c(M(t−1)))」「t_max=20」[F5][F6]；§5.4「maximum gain magnitude of nearly 3000 in HC」vs「approximately 1.6」、27B、12k 步损失突增 [C6][N1]；Eq.1/2/4 与页面 [F2][F3] 逐项一致。HC 论文 §1 摘要 seesaw、Pre/Post-Norm 为 $n=1$ 非可训练超连接（系数矩阵 $\begin{pmatrix}0&1\\1&1\end{pmatrix}$）、$n>1$ 才同时调强度与重排层 [C2][C3][C4] 亦核对通过。算术项 $24\times16384+24+3=393{,}243$、$\times2\times45=35{,}391{,}870$、$35{,}391{,}870/321.32\text{B}=0.011\%$、$4\times4096=16384$ 复算无误，且与 `wiki/glm-5-3-flash-dataflow/index.html` 正文数值交叉一致。
- `.dojo/scripts/validate.py wiki/hyper-connections/index.html` → `validation ok`。前置概念页 `residual-connection`、`rmsnorm`、`block-attnres` 均真实存在，无「（待生成）」。

## 问题

- [阻断·技术] 来源与范围说明 [C10]（L555）、[C11]（L556）、[C12]（L557）、[F8]（L570）及 meta 主要依据（L62）：引文依据指向已被移除的 research/ 文件，标注的行号与章节无法定位。页面把 `wiki/deepseek-v4-1/research/official/inference/model.py`（L938、L941–946、L948–955、L962–966、L1257–1258）、`kernel.py`（L407–474、L409、L427、L429、L430–443、L436–443、L445–458、L448、L454）与 `wiki/glm-5-3-flash-dataflow/research/config.json`（L15/L16/L17）、`p3.out`（§3.1–3.8、§3.3、§3.5）、`verify_structure.out`（[10]）当作可查证的来源位置，但这些文件已不在仓库中：`find wiki/deepseek-v4-1 -name "*.py"` 返回空，`wiki/deepseek-v4-1/research/official/inference/` 仅剩 `README.md`；`wiki/glm-5-3-flash-dataflow/research/` 仅剩 `measured.md`、`prereq-audit.md`。同页 [F7]（L569）「research/ 内实跑输出」与 [N4]（L578）「本页 research/ 实跑输出」也指向已移除的实测产物（本页 `research/` 现仅存 md）。依据 check.md §2.2「定位不到、内容不符或来源本身不支持该论断时，删除该论断及其推论，或降级为明确标注的推断」，此类不可定位的来源论断属阻断级。｜引文依据：`ls wiki/deepseek-v4-1/research/official/inference/` 仅有 `README.md`（其内容为安装与自测说明，无行号 938 等）；`wiki/glm-5-3-flash-dataflow/research/measured.md` 原文「本页的实测产物原先存放在本目录下，现已从仓库移除」并逐条登记 `config.json`、`p3.out`、`verify_structure.out` 为已移除；`wiki/deepseek-v4-1/research/measured.md` 同样把 `official/inference/model.py`、`official/inference/kernel.py` 登记为已移除。｜修复要求：把这些引文位置改为当前可定位的来源——引用 GLM / DeepSeek 官方外部仓库（给出仓库与文件路径）或引用对应 `research/measured.md` 的登记条目，并去掉无法定位的本地行号；无法改引的论断按 check.md §2.2 删除或降级为明确标注的推断。｜修复：｜复验：
- [重要·技术] L440（第 4 章本章问题解答）：「本页实现的实测中行和最高到 $1.0037$（略大于 1）」——该数字在本页任何可视化实验中都定位不到，且与本页代码输出矛盾。本页 $3\times3$ 代码的循环以行归一化收尾，打印的行和每次都是 1.000000，不可能出现「行和最高到 1.0037」；$2\times2$ 手算块中行和的最大可见值是 1.086（列归一化之后）。1.0037 既不等于本页任何一步的行和/列和，也没有对应的本页代码块。｜引文依据：本页代码块实跑输出「迭代 1: 行和 [1.0, 1.0, 1.0], 列和 [1.002536, 1.035057, 0.962407] … 迭代 20: 行和最大偏差 0.0e+00, 列和最大偏差 0.0e+00」；展开块「再一轮列归一化后列和回到 1，行和为 (1.086, 0.914)」；逐轮中间量复算得行和最大 1.063930（列归一化后）、列和最大 1.003085（t=4），均非 1.0037。｜修复要求：删除该数字，或替换为本页代码可复现的数值并写清它是哪个量、在哪个方向（行/列、归一化前还是后），例如「本页 $3\times3$ 实测列和偏差最大 1.0031（第 4 轮）」。｜修复：｜复验：
- [轻微·格式] L7 `dojo:summary`：「靠谱范数 $\leq1$」——术语错误，应为「谱范数」。同一性质在正文 L90、L303、L360、L390、L393、L552、L569 一律写作「谱范数」，summary 中出现的「靠谱范数」是无效术语，且会渲染进首页摘要卡片。｜引文依据：不适用（页面内写法比对）。｜修复要求：把 `dojo:summary` 中的「靠谱范数」改为「谱范数」，与正文保持一致。｜修复：｜复验：
- [轻微·技术] L65（开篇）：「一个 45 层的模型，如果每层的残差混合把信号放大 1.5 倍，到第 45 层信号是最初的 $1.5^{44}\approx 10^{7.7}$ 倍」——指数与层数不一致。按本页自己 24 层演示块的约定（`for _ in range(24)` 即 24 次连乘），45 层对应 $1.5^{45}\approx 10^{7.9}$，而非 $1.5^{44}$。｜引文依据：L326 代码 `for _ in range(24): G = G @ torch.randn(4, 4)`（24 层 = 24 次乘）；$1.5^{44}=10^{7.748}$，$1.5^{45}=10^{7.916}$。｜修复要求：把层数与指数对齐（45 层写 $1.5^{45}\approx 10^{7.9}$，或注明计的是 44 个层间跨越），并在「构造示例」中同步该式。｜修复：｜复验：
- [轻微·技术] L343、L598：声称无约束链的实测数值每次运行不同——「单次采样结果的具体数值不稳定……每次运行都在」（L343）、「无约束链的具体放大数值每次运行不同，量级趋势稳定」（L598）；但代码块第 1 行即 `torch.manual_seed(0)`，[N4]（L578）也标注 `seed=0`，实跑两次输出完全相同（`8.693e+05`），数值是可复现的。｜引文依据：L316 `torch.manual_seed(0)`；[N4]「（$4\times4$、seed=0、$\mathcal{N}(0,1)$）」；连续两次运行该代码块均得 `无约束链 24 层, 复合矩阵行和绝对值最大: 8.693e+05`。｜修复要求：删去「每次运行不同 / 数值不稳定」的表述，改为「随机种子固定，数值可复现；放大是无约束连乘的系统性趋势，换种子只改变具体倍数」，或去掉 seed 并同步说明。｜修复：｜复验：
- [轻微·表述] 元话语：L67「本文先讲清普通残差连接保住了什么……再看 HC 怎么加宽残差流、失控点在哪，然后是……最后对照 GLM-5.3-Flash 的实现规格」为路线图式元话语；L221「下面的代码验证这个退化」、L310「下面的构造实验直接对照两种链的行为」、L390「下面是 $3\times3$ 的完整收敛过程与谱范数检验」为「下面来看」类固定引入句；L135「也是本页后续所有讨论的『基准线』」为对后文的元指涉。｜引文依据：不适用。｜修复要求：改为直接陈述结论或机制，不预告下文、不描述本文结构（章节结构由目录承担）；删去「本文先讲…再看…最后…」句，「下面的 X 验证…」改为「该退化由下例验证：…」一类不带元话语的表述。｜修复：｜复验：
- [轻微·技术] L440：「论文实测复合映射的增益偏差最大约 1.6」——论文 §5.4 的原文是 composite mapping 的 maximum gain **magnitude** ≈1.6（增益上界），不是「增益偏差」；若按「偏差」读则应为 0.6。同页 L294 的写法是「最大增益降到约 1.6」，两处术语不一致。前半句「复合后该偏差随深度增大但仍有界」也未给出可定位来源。｜引文依据：论文 §5.4「mHC's bounded maximum of approximately 1.6」（WebFetch 抓 arXiv:2512.24880v2）；本页 L294「加了约束后（mHC）最大增益降到约 1.6」。｜修复要求：改为「论文实测复合映射的最大增益约 1.6」，与 L294 统一；「复合后该偏差随深度增大但仍有界」若无来源支持则删除或标注为推断。｜修复：｜复验：
- [轻微·技术] L6 `description`：「含 HC 无约束放大的 3000 倍实测对照」——把论文对 27B 模型给出的约 3000 写成页面的「实测对照」，而本页自己的实测对照量级是 $8.693\times10^{5}$（24 层构造示例，[N4]），页面从未测到 3000。｜引文依据：本页代码块输出 `无约束链 24 层, 复合矩阵行和绝对值最大: 8.693e+05`；[N1]「增益峰值约 3000（HC）……实验条件为该论文的训练设置，引用时不外推」。｜修复要求：改为「含无约束链放大约 $10^6$ 的实测对照」或「含论文 27B 实测约 3000 与页面 $4\times4$ 链对照」，把论文数字与页面实测分开表述。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 6
- 处置：修复。核心机制论断（HC 三映射、双随机三性质、Sinkhorn-Knopp 迭代、$n=1$ 退化、3000 vs 1.6）经论文原文与实跑代码双重核对全部成立，无核心结论错误；阻断项为引文依据指向已移除文件、行号无法定位（[C10][C11][C12][F8] 及 [F7][N4] 的 research/ 引用），须改引可定位来源后方可发布。修复完成后从完整页面重跑 validate.py 并复验本记录各条。

---

说明（复验要点备忘）：

1. 三处代码块已在本机 torch 2.8.0 实跑，输出与页面「预期输出」逐位一致，修复时若无必要不要改动代码块本身。
2. 1.0037 与 8.693e5/3000 的措辞改动属于数字修改，按 check.md §4 须重新核对来源。
3. 所有对 `research/` 具体文件（含行号）的引用都要换成外部仓库或 measured.md 登记口径。
