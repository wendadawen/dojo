<!-- review-meta
round: 4
page: wiki/newton-schulz/index.html
reviewed_content_sha256: 4e79f5b05491a208
-->
# Newton-Schulz 迭代审查记录（第 4 轮）

- 页面版本：78f43612995b41166ec99e204a7c460e69b815f9
- 审查时间：2026-09-13 19:46 CST
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题；1. 不用 SVD 的正交化；2. 迭代公式与机制；3. 收敛条件与 Frobenius 预处理；4. 零奇异值保持为 0；5. 在 Muon 优化器中的应用；来源与范围说明（含全部折叠块与代码块）

## 问题

- [重要·技术] 3. 收敛条件与 Frobenius 预处理（正文两处：`σ>√3 时 f(σ) 会变成负数并震荡、发散，不再收敛到 1` 与 `所以 √3 是收敛域的边界`；「本章问题／解答：收敛域边界」与核心问题 3 解答复述同句）：把收敛域外的行为写成无条件论断，且与来源和本页代码实测都不符。区间外并非一律发散、也并非一律"不再收敛到 1"。｜引文依据：[S1]（docs.modula.systems/algorithms/newton-schulz）原句为 `if we iterate f an infinite number of times, we will obtain precisely the sign function on the interval [−√3,√3]`，只断言区间 [−√3,√3] 内的结果，未述区间外行为；[S2] §1 原文 `This iteration converges to the orthogonal factor of the polar decomposition if σ1(X)<√3 and σn(X)>0`，只给充分条件。用本页同一迭代式 σ←½σ(3−σ²) 实测：σ0=1.75→−1、σ0=2.0→−1、σ0=2.2→+1（三者均 >√3≈1.732），σ0=2.5→−8.0e34、σ0=3.0→5.1e21 才发散。｜修复要求：改为限定表述——"σ>√3 时迭代不再保证收敛到 +1（可能收敛到 −1，也可能发散）；σ<√3 是保证收敛到 sign(σ) 的充分条件"，即把"√3 是收敛域的边界"限定为"保证收敛到 +1 的条件"；删除无来源支持的"震荡、发散"无条件断言，或明确降级为标注推断。｜修复：｜复验：
- [轻微·格式] 正文与 callout：Unicode 数学符号直接出现，未用 KaTeX。`非零奇异值→1、零保持 0`（"验证的机制"段）、`非零奇异值→1，零奇异值→0`（常见误解 callout）、`奇多项式与 SVD 可交换 → 退化为标量…→ 趋于 sign`（辅助解释与类比边界）三处用了 `→`，来源章节 `N1（Muon 训练加速 1.35×）` 用了 `×`。｜引文依据：不适用｜修复要求：改为 `$\to$`（或"趋于"）、`$1.35\times$`，全文保持同一写法；`.dojo/scripts/validate.py` 当前不拦这两个字符，但不满足 style-guide §11 对数学关系符/运算符必须由 KaTeX 渲染的要求。｜修复：｜复验：
- [轻微·技术] blockquote.meta「主要依据」与「来源与范围说明」的 Higham 出处不一致：meta 写 `Higham《Functions of Matrices》§5`，来源章节的 Higham 来源实为 [S3] nhigham.com 博客《What Is the Matrix Sign Function?》，全书未被引用（C6 只引博客）。｜引文依据：不适用（页面内部不一致）｜修复要求：把 meta 的 `Higham《Functions of Matrices》§5` 改为实际使用的博客出处，或在来源章节补上该书并标注所引章节；两处保持一致。｜修复：｜复验：
- [轻微·技术] 来源章节无可定位的完整地址：全页无一条外部 `href`，[S1]–[S5] 均为纯文本；其中 [S4] 只给域名 `ricojia.github.io`（无博文路径，实际为 `/2017/02/07/Newton-Schulz/`），[S5] 只给 `"Muon" blog 2024-12`。同库 muon-optimizer 页面在来源段给出了完整 URL。｜引文依据：不适用｜修复要求：为 [S4]、[S5] 补全可定位的 URL/路径，或统一在来源章节列出各来源完整地址。｜修复：｜复验：
- [轻微·技术] C1 引号内非原文：`[S2] Grishina et al. arXiv:2506.10935 §1 "only requires matrix multiplication"`，而该文原文（摘要段）为 `as it relies solely on matrix multiplications`。｜引文依据：arXiv:2506.10935 摘要 `the Newton-Schulz iteration has emerged as a particularly effective solution, as it relies solely on matrix multiplications`｜修复要求：把引号内文字改为原文 `relies solely on matrix multiplications`，或去掉引号改为转述并核对所指章节号。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复
- 机械与复算核对（本轮通过项，供复验参照）：
  - 代码：把页面「代码：验证 diag(0.5, 0) 与 G=[[1,1],[0,1]] 的迭代」原样执行（纯 Python，python3），输出与页面「预期输出」逐行一致（diag 六步 0.5/0.6875/0.8688/0.9753/0.9991/0.999999；G 误差 0.881917→0.000000；final Q=[[0.8944…,0.4472…],[-0.4472…,0.8944…]]）。
  - 数值旁证：numpy SVD 得 G 的极因子 UV^T=[[0.89443,0.44721],[-0.44721,0.89443]]、σ(Q0)=(0.93417,0.35682)，与页面一致。
  - 来源逐条核对：[S1] 的 `fix zero singular values at zero`、`commutes with the singular value decomposition`、`pre-processing step, mapping X ↦ X / ‖X‖_F`、cursed quintic 系数 3.4445/−4.7750/2.0315、`coefficients sum to 0.701 ≠ 1`、`oscillates and in fact does not converge`、symmetric orthogonalization 与 Gram-Schmidt 对比句均逐字命中；[S2] Eq.(1) `X_{k+1}=3/2 X_k − 1/2 X_k X_k^T X_k`、`W = UV^T`、`solution being Q = W`、σ1(X)<√3 条件均命中；[S3] nhigham.com `This iteration is quadratically convergent if ||I-A^2|| < 1 for some subordinate matrix norm`（指 Newton-Schulz 迭代）命中；[S5] Muon 博客 `a factor of 1.35x`/`3.28 val loss`/`(3.4445, -4.7750, 2.0315)`/`scalar and vector parameters … as well as the input and output layers … AdamW` 命中；[S4] ricojia.github.io 确为右乘形式与 Frobenius 预处理出处。
  - 公式复算：f(0.5)=0.6875、f(√3)=0、f'(1)=0、系数和 3.4445−4.7750+2.0315=0.701 均正确；左右乘形式与 [S2] Eq.(1) 代数等价。
  - 链接与结构：`../svd/`、`../muon-optimizer/`、`../per-head-muon/`（与 Per-Head Muon 描述一致）、`overview.html` 互链均有效，无"（待生成）"占位，无已移除的 `research/` 路径；`python3 .dojo/scripts/validate.py wiki/newton-schulz/index.html` 返回 `validation ok`；`dojo:topics=训练与优化`、`dojo:tag=优化器` 均在词表内。