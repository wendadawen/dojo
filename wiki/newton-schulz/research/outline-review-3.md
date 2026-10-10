# outline-review-3

## 读者

- 顺序错乱｜反复迭代趋于符号函数（长页）｜「非零奇异值推向 1」写成一般结论，但收敛要 $\sigma_1(X)<\sqrt3$、$\sigma_n(X)>0$，前提在下一章才出现｜读到这里不知还有上限，读到阈值节才知道有范围限制｜在该节点写明「在收敛域内」。
- 遗漏关键内容｜先归一化再反复代入（短页第 3 节）｜只给迭代式，没说该拿什么矩阵来代、是否先处理，照它对任意矩阵代入可能不收敛｜短页内无处可查｜补一句适用范围（先按 Frobenius 范数归一化）。
- 多余｜无。短页标题｜无。

## 对照资料

- 资料里没有｜适用条件与边界·「失效条件」「零奇异值被拉平」｜依据缺：该节点列的四处依据（Designing your own iteration、A cursed quintic iteration、higham L65-L73、muon-blog L50-L53）都没有这两句，其原文在 modula Problem statement 与 paper Introduction｜依据：modula「fix zero singular values at zero」；paper「converges ... if $\sigma_1(X)<\sqrt3$ and $\sigma_n(X)>0$」｜把这两句出处补进本节依据。

## 篇幅与冗余

无。

## 仲裁

- 意见 A（篇幅第 1 轮，方向：删短页第 3 节的归一化）：**不必改**。引不出依据：`guides/concept.md` L40 只规定短页三节与字数、L52「同一件事只讲一遍」，没有「一节只讲一件事」；L53 明确把「步骤」列为应写内容，归一化是「怎么做」里的一步，不算另一件事。
- 意见 B（读者第 3 轮，方向：短页第 3 节补归一化适用范围）：**必须改**。依据 blog/modula-newton-schulz.md「A cubic iteration」L74「we need to ensure all singular values of the initial matrix lie in the interval $[-\sqrt3,\sqrt3]$ ... mapping $X \mapsto X/\lVert X\rVert_F$」与 paper/src-v2/main.tex L201「the first step of any algorithm based on Newton-Schulz is to normalize the matrix ...」——归一化是使用前提；只给迭代式缺「怎么做」的必要一步。

## 处理

- 读者·结论先于条件：成立。「反复迭代趋于符号函数」的「写什么」改为「在收敛域内，$f$ 反复迭代趋近 $\mathrm{sign}(x)$」，提前点明适用前提；阈值节仍详述条件，两节分工不变。
- 读者·短页缺适用范围：成立（仲裁判必须改）。短页第 3 节标题由「反复代入迭代式」改为「先归一化再反复代入」，「写什么」改为「先把矩阵按 Frobenius 范数归一化，再反复代进迭代式，每步只做几次矩阵乘」。
- 对照资料·边界节依据缺：成立。把「失效条件」与「零奇异值被拉平」两条的出处（paper Introduction、modula Problem statement）补进该节依据。
- 篇幅·无问题，无需处理。

## 最后一批改动核对

- 本轮改动：①「反复迭代趋于符号函数」加「在收敛域内」；②短页第 3 节标题与「写什么」补归一化；③「适用条件与边界」依据补入 paper Introduction、modula Problem statement。
- 仲裁核对：改动 ①③ 通过。改动 ② 不通过，指出短页第 3 节用了 Frobenius 范数却没带一句释义标注（长页对应节点有），需对齐。
- 作者据核对意见改：短页第 3 节「写什么」补「$\lVert X\rVert_F$ 一句释义」。
- 仲裁再核对：通过，无「必须改」类新问题，未违背前几轮裁定。大纲至此通过。
