<!-- review-meta
round: 3
page: wiki/svd/index.html
reviewed_content_sha256: 1df005ad2ed713af
-->
# 奇异值分解（SVD）审查记录（第 3 轮）

- 页面版本：f4070a9eea3573fad529feb9ecd96ec95e6a5479（`wiki/svd/index.html` 工作树）
- 页面类型：`dojo:type=note`，适用规范 `guides/note.md`；记录格式按 `guides/concept/check.md` 第 3、6 节
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 `research/` 下任何规划、修复或前序审查记录）
- 已完整阅读章节：全文通读（含正文、`page-lead`、`dojo:summary`/`description`、脚注脚本与图注位置）。顺序：导语 → 1. 分解形式与维度 → 2. 奇异值的性质 → 3. 最优低秩近似 → 4. 极分解与正交 Procrustes 问题 → 5. 来源与范围说明（论断与来源（C）、公式与来源（F）、来源清单、范围与简化）
- 机械验证：`.dojo/scripts/validate.py wiki/svd/index.html` 通过；`dojo:topics=数学基础`、`dojo:tag=数学与数值` 均在 `catalog_builder.py` 词表内；页面无 `research/` 链接，无 `（待生成）` 占位，无 `data-*` 注入残留；`../../libs/` 下 katex/auto-render/prism/dojo-note.css 均存在；正文无 Unicode 数学字符（仅 `·` 作分隔符），`$...$` 定界符成对（含 `<`、`>` 比较符所在的段），KaTeX auto-render 可整段解析
- 来源核对（本轮实际抓取）：[S4] arXiv:2506.10935v2 用 WebFetch 抓取 arXiv HTML 原文，§1 五处引文逐字命中（见 C4/C5/C6 各项）；[S5][S6] 的刊名、卷期页码与 [S4] 参考文献表一致；[S1][S2][S3] 为离线书源，按页面自标章节核对主题与编号一致性

## 问题

- [轻微·表述] 5. 来源与范围说明 / 论断与来源（C）C4：引号内标为"§1 原文"，但把原文的符号 `S` 换成了本页通用的 `\Sigma`，引号内不是逐字原文。｜引文依据：arXiv:2506.10935v2 §1 原文 "Polar decomposition can be computed directly using the singular value decomposition $X=US^{T}$, which immediately leads to $W=UV^{T},H=VS^{T}$."；页内写作 "「$X=U\Sigma V^{\!\top}$, which immediately leads to $W=UV^{\!\top}$, $H=V\Sigma V^{\!\top}$」"。｜修复要求：在引文后加一句记号说明（原文用 $S$，本页记作 $\Sigma$），或把该处引号内的 $S$ 按原文写回。｜修复：｜复验：
- [轻微·技术] 5. 来源与范围说明 / 范围与简化 第 2 条：把"SVD 的数值算法（Golub–Kahan 迭代、分治等）与浮点误差分析"的出处标为 [S2] §2.4，与本页 C1/C2 对同一来源的用法冲突——C1、C2 把 [S2] §2.4 当作 SVD 存在性与数学性质（rank、谱范数、Frobenius 范数、$X^{\!\top}X$ 关系）的来源，同一条目下读者按 §2.4 去查找不到算法与误差分析内容。｜引文依据：页内 C1 "…；[S2] Golub & Van Loan 第 4 版 §2.4 'The Singular Value Decomposition'"、C2 "…；[S2] §2.4"，对照范围与简化 "…不展开 SVD 的数值算法（Golub–Kahan 迭代、分治等）与浮点误差分析，这两者不在讨论范围（[S2] §2.4）"。书源离线，未能定位 §2.4 含算法内容，故按同页用法不一致判定而非按内容不符判定。｜修复要求：把该括号来源改到 [S2] 的 SVD 算法章节（§8.6 "The SVD"），或删去该括号来源只保留范围说明。｜修复：｜复验：

## 已核对通过项（不构成问题）

- C1/C3/F1/F3：Eckart–Young 两条误差式经数值复算成立——对 6×4 随机矩阵，k=1,2 时 $\lVert X-X_k\rVert_2=\sigma_{k+1}$、$\lVert X-X_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^2}$ 与 SVD 实算值逐位吻合（2.033539780392609/2.8953276154442134；1.8563264871772776/2.0609798549026754）。
- C4/C5/C6：与 [S4] §1 原文一致——"Polar decomposition of a matrix $X\in\mathbb{R}^{m\times n}, m\ge n$ is a factorization $X=WH$, where $W\in\mathbb{R}^{m\times n}$ has orthonormal columns"；"$\min_{Q:\,Q^{\top}Q=I}\lVert Q-X\rVert_F$, with the solution being $Q=W$ the polar factor of $X$"；"converges to the orthogonal factor of the polar decomposition if $\sigma_1(X)<\sqrt3$ and $\sigma_n(X)>0$"。
- 正文机制"反复把当前矩阵的非零奇异值推向 1"可由 $X_{k+1}=\tfrac32X_k-\tfrac12X_kX_k^{\top}X_k$ 导出（$\sigma\mapsto(3\sigma-\sigma^3)/2$，不动点 $\sigma=1$；数值复算 0.7→1.0），与 C6 结论不矛盾。
- 符号单义：$U,\Sigma,V,\sigma_i,u_i,v_i,W,H,Q,X,k,m,n$ 全文写法一致；薄/完整 SVD 的维度差异（补 $m-n$ 个零行、$U$ 与 $\Sigma$ 维度变化、$V$ 不变）自洽。
- 表述维度：通读全文未出现"本页将…/下面来看…/需要注意的是"类元话语，无以"本页"为主语的自我指代，无我/我们/你等会话指代，无调试复现叙事与临场评价；未见抽象名词堆叠、公文连接词或把"场景"当术语。
- 链接：`../newton-schulz/index.html` 存在（两处），`../../index.html` 有效；无外链、无失效 research 路径。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题均为引用标写层面的瑕疵，不影响核心结论与断言可核性；修复可在下一轮前顺手完成）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
