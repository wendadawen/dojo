# Newton-Schulz 迭代

- 描述：Newton-Schulz 迭代是一种只用矩阵乘法、反复把矩阵推向其极分解正交因子的数值迭代：把非零奇异值拉平为 1，零奇异值保持为 0。
- 摘要：反复施加奇多项式 $p(X)=\frac32X-\frac12XX^\top X$，把非零奇异值推向 1、零奇异值留在 0，只用矩阵乘法就逼近极分解的正交因子 $UV^\top$。
- 主题：数学基础, 训练与优化
- 标签：数学与数值

## 长页

- 开头｜它是什么、为何需要、怎么工作：用只含矩阵乘法的迭代把矩阵正交化，逼近正交因子 $UV^\top$；SVD 首次出现处链接 [SVD](wiki/svd)｜280｜blog/modula-newton-schulz.md Problem statement；blog/modula-newton-schulz.md Odd polynomial iterations；paper/src-v2/main.tex Introduction
- ### 从一个矩阵求最接近的正交矩阵｜本章回答它解决什么问题、为什么不用 SVD，内容在两个小节｜0｜paper/src-v2/main.tex Introduction
- #### 想求的是哪个矩阵｜定义正交 Procrustes 问题与极分解，解是正交因子 $W=UV^\top$；奇异值、SVD 链接 [SVD](wiki/svd)｜240｜blog/modula-newton-schulz.md Problem statement；paper/src-v2/main.tex Introduction
- #### 为什么不去算 SVD｜SVD 要算出 $U,\Sigma,V$，大矩阵上贵；NS 只用矩阵乘、GPU 上效率高｜260｜blog/modula-newton-schulz.md Problem statement；blog/muon-blog.txt L50-L53；paper/src-v2/main.tex Introduction
- ### 迭代式与符号｜写出迭代式，说明 $X_k$、转置、$I$ 是什么，$X_k^\top X_k$ 何时表示趋近正交；右乘等价形式的推导折叠｜300｜paper/src-v2/main.tex Introduction；blog/modula-newton-schulz.md A cubic iteration
- ### 为什么只看奇异值就够了｜本章回答反复迭代为何能正交化，内容在两个小节｜0｜blog/modula-newton-schulz.md Odd polynomial iterations
- #### 奇多项式与 SVD 可交换｜奇多项式与 SVD 可交换：施加 $p$ 等于对每个奇异值施加标量奇多项式 $f$｜280｜blog/modula-newton-schulz.md Odd polynomial iterations；paper/src-v2/main.tex Introduction
- #### 反复迭代趋于符号函数｜在收敛域内，$f$ 反复迭代趋近 $\mathrm{sign}(x)$：非零奇异值推向 1、$\sigma=0$ 因 $f(0)=0$ 不动；迭代图说明折叠｜280｜blog/modula-newton-schulz.md A cubic iteration；blog/modula-newton-schulz.md Odd polynomial iterations
- ### 收敛条件与预处理｜本章讲迭代在什么条件下收敛、为何先归一化，内容在两个小节｜0｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Normalization of a matrix prior to Newton-Schulz iterations
- #### 为什么阈值是 $\sqrt3$｜收敛要求 $\sigma_1(X)<\sqrt3$ 且 $\sigma_n(X)>0$；超出 $[-\sqrt3,\sqrt3]$ 不再保证收敛｜260｜paper/src-v2/main.tex Introduction；blog/modula-newton-schulz.md A cubic iteration
- #### 用 Frobenius 范数归一化｜预处理 $X\leftarrow X/\lVert X\rVert_F$ 把奇异值落入收敛域；$\lVert X\rVert_F$ 一句释义；结论折叠｜260｜blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Normalization of a matrix prior to Newton-Schulz iterations
- ### 手算一个 2×2 例子｜自己设一个满列秩的小整数 2×2 矩阵，归一化后迭代几步，看它逐步趋近正交矩阵；过程与数字全折叠｜100｜自行构造
- ### 适用条件与边界｜失效条件（未归一化或不满列秩时不再适用）、代价（近似而非精确、精度换步数）；两条误解：零奇异值被拉平、等价于 SVD｜280｜paper/src-v2/main.tex Introduction；blog/modula-newton-schulz.md Problem statement；blog/modula-newton-schulz.md Designing your own iteration；blog/higham-matrix-sign.txt L65-L73；blog/muon-blog.txt L50-L53

## 短页

- ### 只用矩阵乘法逼近正交化｜给定矩阵 $X$，用只含矩阵乘法的迭代逼近最接近它的正交矩阵，即正交因子 $UV^\top$；非零奇异值推向 1；SVD 链接 [SVD](wiki/svd)｜280｜blog/modula-newton-schulz.md Problem statement；paper/src-v2/main.tex Introduction
- ### 为什么不去算 SVD｜SVD 也能给出该正交矩阵，但大矩阵上贵；NS 只用矩阵乘、GPU 上效率高，精度由步数控制｜320｜blog/modula-newton-schulz.md Problem statement；blog/muon-blog.txt L50-L53；paper/src-v2/main.tex Introduction
- ### 先归一化再反复代入｜先把矩阵按 Frobenius 范数归一化（$\lVert X\rVert_F$ 一句释义），再反复代进迭代式，每步只做几次矩阵乘｜320｜blog/modula-newton-schulz.md A cubic iteration；paper/src-v2/main.tex Introduction
