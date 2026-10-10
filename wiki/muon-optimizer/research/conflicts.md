# 冲突记录

- 动量缓冲区的更新式｜位置：`content.md` 长页「完整更新规则」折叠块「一次更新的公式」｜博客定义处（`blog/muon_algo.png`，Algorithm 2）写 $B_t\leftarrow\mu B_{t-1}+G_t$；官方实现（`repo/Muon/muon.py L35` 的 `momentum.lerp_(grad, 1 - beta)`）对应 $B_t=\mu B_{t-1}+(1-\mu)G_t$。两者对梯度的系数不同｜页面采用：不写出该式的具体系数，只写「按系数 $\mu$ 把梯度累积进去」，两种写法都成立。
- 5 步 Newton-Schulz 的输出｜位置：`content.md` 短页「动量、迭代、步长」、长页「正交化是什么」「用迭代算出正交矩阵」｜博客（`blog/muon-blog.txt L69`）称迭代把更新换成最近半正交矩阵，即 $UV^\top$；官方实现注释（`repo/Muon/muon.py L10-L13`）称该迭代不产生 $UV^\top$，而是 $US'V^\top$，$S'$ 的对角元大致落在 $0.5$ 到 $1.5$；博客另处（L122）给的范围是 $[0.7,1.3]$，与实现的 $0.5$ 到 $1.5$ 又不同｜页面采用：正交化的目标写成「最接近它的半正交矩阵（$UV^\top$）」，并说明实现取 5 步是近似。

