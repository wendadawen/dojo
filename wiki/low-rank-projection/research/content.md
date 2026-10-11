# Low-Rank Projection

- 描述：低秩分解把一个大矩阵写成两个小矩阵的乘积，内维远小于原矩阵的两维，用更少的参数表达几乎相同的变换。
- 摘要：$W \approx AB$，把输入先投影到 $r$ 维再投影回原维度。$r$ 远小于原矩阵维度时，参数和存储降一个量级，损失的多少由被丢掉的奇异值决定。
- 主题：数学基础, 模型结构
- 标签：数学与数值

## 资料

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685v2, 2021. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434v5, 2024. [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- Stephen Boyd, Sanjay Lall. Singular Value Decomposition. 斯坦福大学 EE263 第 17 讲, 2026. [https://ee263.stanford.edu/lectures/26q3/original/17_svd.pdf](https://ee263.stanford.edu/lectures/26q3/original/17_svd.pdf)
- Stephen Boyd, Sanjay Lall. Matrix norm. 斯坦福大学 EE263 第 16 讲, 2026. [https://ee263.stanford.edu/lectures/26q3/original/16_norm.pdf](https://ee263.stanford.edu/lectures/26q3/original/16_norm.pdf)
- M. P. Deisenroth, A. A. Faisal, C. S. Ong. Mathematics for Machine Learning. Cambridge University Press, 2020. [https://mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf)
- LAPACK Users' Guide (Third Edition). Singular Value Decomposition. SIAM, 1999. [https://www.netlib.org/lapack/lug/node53.html](https://www.netlib.org/lapack/lug/node53.html)

## 短页

### 用两个小矩阵代替一个大矩阵

神经网络由多层叠成，每一层把输入向量乘上一个矩阵，得到这一层的输出，这个矩阵叫权重矩阵。一个 $m \times n$ 的权重矩阵有 $mn$ 个元素，都要存下来。低秩分解把这样一个矩阵 $W$ 写成两个小矩阵的乘积 $W \approx AB$，其中 $A$ 是 $m \times r$ 的、$B$ 是 $r \times n$ 的，内维 $r$（两个小矩阵共有的那一维）远小于 $m$ 和 $n$。存这两个小矩阵只要 $r(m+n)$ 个数，比 $mn$ 少得多。
依据：paper/lora-2106.09685.txt L145-L147；paper/lora-2106.09685.txt L212-L213；blog/mml-book/mml-ch4-svd.txt L668-L698

### 无损做不到，近似可以

权重矩阵通常满秩，即它的各行（列）互相独立，没有多余方向，所以拆成两个小矩阵再乘回来无法完全还原，$AB$ 只是 $W$ 的一个近似。奇异值是刻画矩阵在各方向上伸缩强度的、按从大到小排列的一组数。按奇异值把 $W$ 分解成一串秩 1 矩阵的和，只保留最大的前 $r$ 项，得到的低秩矩阵在所有秩不超过 $r$ 的矩阵里与原矩阵最接近（按谱范数衡量，谱范数是矩阵能对向量做的最大放大倍数），误差由被丢掉的奇异值决定。
依据：paper/lora-2106.09685.txt L207-L208；blog/mml-book/mml-ch4-svd.txt L668-L698；blog/mml-book/mml-ch4-svd.txt L754-L812

### 先降到低维再升回去

$W \approx AB$ 这一步可以看成对输入做两次投影：先用 $B$ 把输入从 $n$ 维投到 $r$ 维，再用 $A$ 把它投回 $m$ 维。中间那一层只有 $r$ 维，$r$ 取得越小，存的数越少，丢掉的奇异值越多、按谱范数衡量的近似误差越大；$r$ 越大，误差越小，但要存的数也多。
依据：paper/lora-2106.09685.txt L212-L213；blog/mml-book/mml-ch4-svd.txt L754-L812；paper/deepseek-v2-2405.04434.txt L328-L345

## 长页

神经网络由多层叠成，每一层把输入向量乘上一个矩阵得到输出，这个矩阵叫权重矩阵，它的元素在训练中不断更新，训练完要存下来。一个 $m \times n$ 的权重矩阵有 $mn$ 个元素，表示维度越大，这样的矩阵越大，占的存储越多。于是有了用更少的数表达几乎相同的变换的想法，低秩分解是它的一个具体做法。
依据：blog/mml-book/mml-ch4-svd.txt L668-L706；paper/lora-2106.09685.txt L145-L147

低秩分解把一个大矩阵 $W$ 写成两个小矩阵的乘积 $W \approx AB$。它的依据是：当一个矩阵接近低秩时，决定这个变换的只有少数几个方向，把这些方向单独存下来，就能用远少于 $mn$ 个数复原一个几乎相同的变换。
依据：blog/mml-book/mml-ch4-svd.txt L668-L698；paper/lora-2106.09685.txt L212-L213

低秩分解在大模型里有多种用法，例如微调方法 [LoRA](wiki/lora) 把权重的更新量写成低秩形式 $BA$，注意力机制 [MLA](wiki/mla) 把键和值的生成路径换成一次低秩投影、把每个位置的键值压成一个低维向量再缓存。
依据：paper/lora-2106.09685.txt L212-L213；paper/deepseek-v2-2405.04434.txt L328-L345

### 矩阵的秩

一个 $m \times n$ 矩阵的列可以看成 $n$ 个 $m$ 维向量。这些向量张成的空间维度，就是这个矩阵的秩，记作 $\mathrm{rank}(W)$：秩等于列里互相独立的方向数。若每一列都能由其余列线性组合出来，秩就小于 $n$；若 $n$ 列全部独立，秩等于 $n$，此时称矩阵满秩。秩直接决定能不能用低秩分解：若 $W = AB$，$A$ 是 $m \times r$ 的、$B$ 是 $r \times n$ 的，那 $W$ 的每一列都是 $A$ 各列的线性组合，只张成 $r$ 维空间，于是 $\mathrm{rank}(W) \le r$。反过来，若 $\mathrm{rank}(W) = r$，就能找到这样的分解，$A$ 取 $m \times r$、$B$ 取 $r \times n$，且没有更小的 $r$。
依据：blog/mml-book/mml-ch4-svd.txt L668-L698；blog/ee263/ee263-17_svd.txt L484-L535

只有当矩阵真的低秩（秩不超过 $r$）时，$W = AB$ 才无损；满秩矩阵做不到无损压缩，只能近似。参数也因此减少：$A$ 有 $mr$ 个元素、$B$ 有 $rn$ 个，合起来 $r(m+n)$，$r$ 远小于 $\min(m,n)$ 时这个数远小于 $mn$。
依据：blog/mml-book/mml-ch4-svd.txt L668-L698；paper/lora-2106.09685.txt L207-L208

::: 图像矩阵的参数账
一个 $1432 \times 1910$ 的矩阵，元素个数是 $1432 \cdot 1910 = 2735120$。用低秩分解保留秩 5 时，只需存 5 个奇异值，以及 5 个 $1432$ 维的左奇异向量和 5 个 $1910$ 维的右奇异向量，共 $5 \cdot (1432 + 1910 + 1) = 16715$ 个数，略高于原数的 $0.6\%$。
依据：blog/mml-book/mml-ch4-svd.txt L700-L706
:::

### 最优低秩近似：截断 SVD

#### 奇异值分解

任何一个实矩阵 $W$ 都能写成 $W = U \Sigma V^{\!\top}$，这叫奇异值分解（SVD，singular value decomposition）。$U$ 和 $V$ 是正交矩阵（各自列向量两两正交且长度为 1），$\Sigma$ 与 $W$ 同形状，除主对角线外全是 $0$，主对角线上是从大到小排列的非负实数，叫奇异值，记 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$；$U$ 的每一列叫左奇异向量，$V$ 的每一列叫右奇异向量。$\sigma_i$ 衡量矩阵在第 $i$ 个主方向上的伸缩强度。矩阵的秩等于非零奇异值的个数。
依据：blog/mml-book/mml-ch4-svd.txt L27-L100；blog/lapack-lug/svd.txt L9-L32

把 $W = U\Sigma V^{\!\top}$ 按列展开，就得到一串秩 1 矩阵之和 $W = \sum_{i=1}^{r} \sigma_i u_i v_i^{\!\top}$，$u_i, v_i$ 分别是 $U, V$ 的第 $i$ 列，$r$ 是秩。每一项 $\sigma_i u_i v_i^{\!\top}$ 都是一个秩 1 矩阵，权重是第 $i$ 个奇异值。
依据：blog/mml-book/mml-ch4-svd.txt L668-L698

只保留前 $k$ 大的项、丢掉其余，得到 $W_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\!\top}$，这叫截断 SVD。$W_k$ 的秩不超过 $k$。它本身就是一个低秩分解：$W_k = (U_k \Sigma_k^{1/2})(\Sigma_k^{1/2} V_k^{\!\top})$，其中 $U_k, V_k$ 是 $U, V$ 的前 $k$ 列、$\Sigma_k$ 是前 $k$ 个奇异值组成的对角阵，左因子是 $m \times k$ 的、右因子是 $k \times n$ 的。
依据：blog/mml-book/mml-ch4-svd.txt L605-L698

#### 误差由被截掉的奇异值决定

截断 SVD 给出的 $W_k$，是所有秩不超过 $k$ 的矩阵里与 $W$ 最接近的那个。衡量接近程度用矩阵范数：谱范数 $\|W\|_2$ 是矩阵能对向量做的最大放大倍数，它等于最大的奇异值 $\sigma_1$。Eckart-Young 定理说：$W_k$ 使谱范数误差 $\|W - W_k\|_2$ 在所有秩不超过 $k$ 的矩阵里最小，而这个最小误差正好等于被截掉的第一个奇异值 $\sigma_{k+1}$。
依据：blog/mml-book/mml-ch4-svd.txt L743-L812；blog/ee263/ee263-16_norm.txt L37-L53

$\sigma_{k+1}$ 小，秩 $k$ 的近似就准；$\sigma_{k+1}$ 大，截断丢掉的信息就多。
依据：blog/mml-book/mml-ch4-svd.txt L754-L812；blog/ee263/ee263-17_svd.txt L484-L539
::: 最优性从哪来
误差矩阵 $W - W_k = \sum_{i=k+1}^{r} \sigma_i u_i v_i^{\!\top}$ 只剩被截掉的项。它的最大奇异值是 $\sigma_{k+1}$，所以它的谱范数就是 $\sigma_{k+1}$。
依据：blog/mml-book/mml-ch4-svd.txt L782-L812

反过来，任取一个秩不超过 $k$ 的矩阵 $C$（与 $W$ 同形），都有 $\|W - C\|_2 \ge \sigma_{k+1}$。证明用一步维度计数：$C$ 的零空间维度至少是 $n - k$，而前 $k+1$ 个右奇异向量张成一个 $k+1$ 维子空间，两个子空间必有非零公共向量 $z$，于是 $Cz = 0$ 而 $\|Wz\|_2 \ge \sigma_{k+1} \|z\|_2$，合起来即得 $\|W - C\|_2 \ge \sigma_{k+1}$。
依据：blog/mml-book/mml-ch4-svd.txt L790-L812；blog/ee263/ee263-17_svd.txt L512-L539
:::

### 一个手算的低秩近似

一个秩 1 近似丢掉的是除最大方向外的所有方向，谱范数误差等于被丢掉的第一个奇异值。
依据：自行构造

::: 秩 1 近似的误差
1. $W = \mathrm{diag}(3, 1, 0.5)$，取 $U$ 和 $V$ 都是单位阵，则 $\Sigma = \mathrm{diag}(3, 1, 0.5)$，三个奇异值为 $3, 1, 0.5$。
依据：自行构造
2. 截到 $k = 1$：$W_1 = \sigma_1 u_1 v_1^{\!\top} = 3 \cdot e_1 e_1^{\!\top} = \mathrm{diag}(3, 0, 0)$。
依据：自行构造
3. 误差矩阵 $W - W_1 = \mathrm{diag}(0, 1, 0.5)$。
依据：自行构造
4. 谱范数误差等于最大奇异值，即 $\sigma_2 = 1$，与误差公式 $\sigma_{k+1}$ 一致。
依据：自行构造

```python
import numpy as np
W = np.diag([3.0, 1.0, 0.5])
W1 = np.diag([3.0, 0.0, 0.0])
print(np.linalg.norm(W - W1, 2))
```

```text
1.0
```
依据：自行构造
:::

### 适用条件与边界

低秩近似要有用，被截掉的奇异值要足够小；它们偏大时，丢弃的信息就多，近似误差就大。
依据：blog/mml-book/mml-ch4-svd.txt L754-L812

低秩近似是有损压缩：留存前 $k$ 项等于丢掉后面的项，$W_k$ 一般不等于 $W$，只有当 $W$ 的秩本来就不超过 $k$ 时才精确相等。
依据：blog/mml-book/mml-ch4-svd.txt L683-L698；blog/mml-book/mml-ch4-svd.txt L808-L812

把低秩分解用在一个具体对象上时，这个对象要真的低秩才有收益。[LoRA](wiki/lora) 用很低的 $r$，前提是权重更新量具有低内在秩；这个前提不是对所有任务都成立，任务与预训练（训练模型的那次大规模训练）差别越大（例如换了语言），需要的 $r$ 可能越大，小 $r$ 会不够用。
依据：paper/lora-2106.09685.txt L207-L213；paper/lora-2106.09685.txt L600-L603

