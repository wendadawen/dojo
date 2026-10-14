# Hadamard Transform

- 描述：Hadamard Transform 是用归一化正负号组合混合坐标、保持向量长度的正交变换，用于改善低精度表示前的数值分布。
- 摘要：长度不变，坐标分布改变；先分散大值、同步变换权重，再量化，二次幂维度可用递归加减实现。
- 主题：数学基础, 推理系统
- 标签：量化

## 资料

- Saleh Ashkboos 等. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456v2, 2024. [论文](https://arxiv.org/abs/2404.00456v2)
- QuaRot 作者维护的代码仓库. [仓库](https://github.com/spcl/QuaRot)

## 短页

### 保持长度的坐标混合

在[量化](wiki/quantization-basics)中，Hadamard Transform（哈达玛变换）是量化前的坐标变换，不是把实数舍入为整数的量化器。它用归一化的正负号组合混合各坐标，是保持向量欧氏长度的正交变换。权重是模型中可学习的参数，激活是计算产生的中间数值；两者都可以在变换后再量化。
依据：paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations

### 大值会撑宽量化网格

多个坐标共用一个量化步长时，远大于其余坐标的离群值会把覆盖范围拉宽，同样数量的整数格点变得更稀，较小的数更容易被舍成零。Hadamard 把原来集中在少数坐标上的值混合到其他坐标，目的是缩小这种差异，而不是裁掉大值。长度保持不变，不代表每个输入的最大坐标都会下降。
依据：自行构造

### 先换基、配对权重、再量化

线性层把输入向量乘以权重矩阵，再加偏置。设列向量 $x$ 是输入，$W$ 是权重矩阵，$b$ 是偏置，$y=Wx+b$ 是输出。选一个固定的正交变换 $R$，令 $z=Rx$、$W'=WR^\top$，其中 $z$ 是变换后的输入，$W'$ 是变换后的权重，$\top$ 表示转置，则实数运算下 $W'z+b=Wx+b$。权重先离线变换；运行时变换输入，再对新的输入和权重量化，得到近似计算。
依据：自行构造

常见的选择是 $R=HD$：$H$ 是归一化 Hadamard 矩阵，$D$ 是对角线上独立随机取正负一的矩阵，先翻转部分输入符号，再混合坐标；符号选定后与配对权重保持一致。维度为二次幂时，$H$ 乘向量可用递归加减计算。
依据：paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/method.tex Stage 1a: Weight Modification；paper/quarot-v2-src/method.tex Stage 2a: Weight Quantization

## 长页

Hadamard Transform（哈达玛变换）在[量化](wiki/quantization-basics)中用于改变数值的坐标分布，目标是减轻少数异常大值对低精度表示的影响。权重是模型中可学习的参数，激活是计算产生的中间数值；对它们先做变换，再做低精度近似，是两个分开的操作。数学上的 Hadamard 变换不做量化舍入或裁剪，通过与权重配对变换，量化前可以保持原计算。
依据：paper/quarot-v2-src/intro.tex L5-L8；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations

### 正负号组成的正交换基

设 $d$ 为向量维度，$A_d$ 为一个 $d\times d$ 的矩阵，元素取正负一，各行互相正交。它满足 $A_dA_d^\top=dI$，其中 $I$ 是单位矩阵。未归一化的 Hadamard 矩阵 $A_d$ 不是保长度的变换；乘上归一化因子得到 $H_d=A_d/\sqrt d$，才有 $H_d^\top H_d=I$。量化语境中的正交 Hadamard 通常指后者。
依据：自行构造

对列向量 $x$ 做变换 $z=H_dx$，得到的每个坐标都是原坐标的一种带正负号的和，再除以 $\sqrt d$。$z$ 与 $x$ 的欧氏长度相同，原向量可用 $x=H_d^\top z$ 恢复。这里的 Hadamard Transform 不同于 Hadamard product：后者是两个数组对应元素相乘，不混合坐标。
依据：自行构造

::: 四维矩阵与正交性质
1. 取一组四维正交的正负号行，定义

   $$
   A_4=\begin{bmatrix}
   1&1&1&1\\
   1&-1&1&-1\\
   1&1&-1&-1\\
   1&-1&-1&1
   \end{bmatrix},\qquad H_4=\frac12 A_4.
   $$
依据：自行构造

2. 每行与自己点积为 $4$，任意两行点积为 $0$，所以 $A_4A_4^\top=4I$。$H_4$ 每行长度为 $1$，因而是正交矩阵。
依据：自行构造

3. 用 $\|\cdot\|_2$ 表示欧氏范数，即各坐标平方和的平方根；保长度来自

   $$
   \|H_4x\|_2^2=x^\top H_4^\top H_4x=x^\top x=\|x\|_2^2.
   $$
依据：自行构造

4. 这组 $H_4$ 还满足 $H_4^\top=H_4$，因此做两次相同变换即可恢复输入。一般正交变换的逆是转置，不必等于它自身。
依据：自行构造
:::

### 分散大值，让共享网格变细

对多个坐标共用步长的对称均匀量化，设 $s$ 是步长，$M$ 是这一组坐标的最大绝对值，$q_{\max}$ 是整数网格的正端点，取 $s=M/q_{\max}$。离群值把 $M$ 拉大时，正常值之间的差别可能小于一个步长，舍入后就无法区分。位宽不变时，缩小变换后这一组的最大绝对值，才可能让网格变细。
依据：自行构造

若大值只集中在一个坐标，Hadamard 会把它的贡献分到所有输出坐标；多个输入坐标则可能互相抵消，也可能叠加。混合改变的是大值相对于其他坐标的大小，不是向量的总长度。
依据：自行构造

::: 单个大值的贡献如何分散
1. 取 $x=(12,0,0,0)^\top$，用 $H_4$ 相乘，得到 $z=(6,6,6,6)^\top$。
依据：自行构造

2. 最大绝对值从 $12$ 变成 $6$，但原平方和为 $144$，变换后为 $4\times36=144$。没有裁掉能量，只改变了坐标表示。
依据：自行构造
:::

随机化 Hadamard 在混合前加一次符号翻转：令 $D=\operatorname{diag}(\varepsilon_1,\ldots,\varepsilon_d)$，$\varepsilon_i$ 是各自独立、等概率取正负一的随机符号，$\operatorname{diag}$ 表示以这些数为对角元素的矩阵；采用 $R=H_dD$，先算 $Dx$ 再算 $H_d(Dx)$。$R$ 仍然正交，不过它的逆是 $R^\top=DH_d^\top$，不能默认仍按原顺序计算。
依据：自行构造

::: 随机符号为何减少固定方向的叠加
1. 固定输入 $x$，只对符号取随机平均。设 $h_{ji}$ 是 $H_d$ 第 $j$ 行第 $i$ 列的元素，$i$ 和 $j$ 分别是输入与输出坐标下标；输出 $z_j=\sum_{i=1}^d h_{ji}\varepsilon_i x_i$。
依据：自行构造

2. 记 $\mathbb E$ 为对随机符号的期望。因为 $\mathbb E\varepsilon_i=0$、$\mathbb E\varepsilon_i^2=1$，不同符号独立，交叉项期望为零。
依据：自行构造

3. 每个 $h_{ji}^2=1/d$，于是

   $$
   \mathbb E[z_j^2]=\sum_{i=1}^d h_{ji}^2x_i^2=\frac{\|x\|_2^2}{d}.
   $$
依据：自行构造

4. 每个输出坐标的平方在期望上相同，不会固定保留原来那个大坐标。这是对随机符号取平均的结论，不是某一次变换后所有坐标都等大的保证。
依据：自行构造
:::

在带有小数值的算例里，变换后用同一位宽量化，逆变换后的总平方误差更小，但仍然非零。峰值下降改善了这组数的网格分辨率，没有消除舍入。
依据：自行构造

::: 从输入、量化到逆变换的完整算例
1. 取 $x=(7,1,0,0)^\top$，采用 $H_4$，不加随机符号翻转。使用 $3$ 位有符号存储，只取关于零对称的整数范围 $[-3,3]$，舍入到最近整数，不用额外裁剪比例。设 $q$ 为量化整数向量，$\hat x$ 为反量化后的向量。
依据：自行构造

2. 直接量化：$s=7/3$，$x/s=(3,3/7,0,0)^\top$，舍入后 $q=(3,0,0,0)^\top$，反量化 $\hat x=sq=(7,0,0,0)^\top$。误差向量为 $(0,-1,0,0)^\top$，平方和为 $1$。
依据：自行构造

3. 先变换：$z=H_4x=(4,3,4,3)^\top$。新步长 $s'=4/3$，其中 $s'$ 是变换后采用的步长；$z/s'=(3,9/4,3,9/4)^\top$，舍入得 $q'=(3,2,3,2)^\top$，$q'$ 是变换后量化得到的整数向量。
依据：自行构造

4. 反量化得到 $\hat z=s'q'=(4,8/3,4,8/3)^\top$，$\hat z$ 表示还原到变换坐标的近似值；再逆变换，得到原坐标的近似值 $\tilde x$：

   $$
   \tilde x=H_4^\top\hat z=(20/3,4/3,0,0)^\top.
   $$
依据：自行构造

5. 误差 $\tilde x-x=(-1/3,1/3,0,0)^\top$，平方和为 $2/9$。同一位宽下，这个输入的平方误差由 $1$ 降到 $2/9$；比较是在原坐标系下完成的。
依据：自行构造

![自行构造的输入与变换后坐标：最大绝对值下降，平方和相同](assets/coordinates.png)
:::

### 同步变换权重，保持原计算

线性层把输入乘以权重矩阵，再加偏置。用列向量约定：$x$ 为输入，$W$ 为权重矩阵，$b$ 为偏置，$y=Wx+b$ 为输出。输入改成 $z=Rx$ 后，权重也要改成 $W'=WR^\top$；$R$ 是固定的正交变换。量化前、实数运算下有

$$
W'z+b=Wx+b.
$$

只变换输入却继续使用原权重，算到的不是同一个线性层。
依据：自行构造

::: 配对式与逆变换的误差
1. $R^\top R=I$，因此 $W'z+b=WR^\top Rx+b=Wx+b$。输入的逆变换已经包含在新的权重里，不必在矩阵乘前把 $z$ 单独变回去。
依据：自行构造

2. 令 $e$ 表示变换坐标下的量化误差，即近似输入为 $z+e$。只考察输入量化时，线性层的输出误差为 $WR^\top e$，仍会受 $W$ 的作用影响。
依据：自行构造

3. 在输入坐标系下，逆变换本身不会放大欧氏误差：$\|R^\top e\|_2=\|e\|_2$。保长度只作用于已经产生的误差，不能证明量化产生的 $e$ 比直接量化更小。
依据：自行构造
:::

权重固定时，可以在推理前先算出变换后的权重，再对它量化；推理指用训练好的参数计算输出。需要在线变换的输入，则在运行时先经过 Hadamard，再量化、再矩阵乘。
依据：paper/quarot-v2-src/method.tex Stage 1a: Weight Modification；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2a: Weight Quantization；paper/quarot-v2-src/method.tex L6

设 $\widehat W'$ 为量化后还原出的近似权重，$\hat z$ 为近似输入，实际计算的是 $\hat y=\widehat W'\hat z+b$，$\hat y$ 表示近似输出。输入和权重都可能带误差，所以配对的正交性质不等于低精度输出无损。若重新选择随机符号，必须同步更新配对权重，否则量化前的等式也不成立。
依据：自行构造

### 递归加减，不做稠密矩阵乘

维度 $d=2^n$ 时，$n$ 为非负整数，可以使用递归构造的 Hadamard 矩阵，这种二次幂阶数的构造称为 Walsh-Hadamard。每次把一对数 $u,v$ 换成 $u+v,u-v$，再逐级扩大配对范围，最终统一乘上 $1/\sqrt d$。这是对未归一化正负号矩阵的快速乘法，不需要先存出完整矩阵。
依据：自行构造

普通稠密矩阵乘向量的每个输出坐标需要对 $d$ 个输入做乘加，共有 $d$ 个输出，所以需要 $O(d^2)$ 次运算。
依据：自行构造

快速 Hadamard 有 $\log_2 d$ 级，每级处理整个向量，运算量为 $O(d\log_2 d)$。
依据：paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices

::: 递归结构与逐轮手算
1. 对二次幂阶数，令 $A_1=[1]$，未归一化矩阵可按

   $$
   A_{2m}=\begin{bmatrix}A_m&A_m\\ A_m&-A_m\end{bmatrix}
   $$

   递归构造，$m$ 表示当前矩阵阶数。矩阵每扩大一倍，就增加一级和与差。
依据：自行构造

2. 仍取 $x=(7,1,0,0)^\top$。第一级按相邻两个数配对，得到 $(8,6,0,0)^\top$；第二级分别配对第一与第三、第二与第四个数，得到 $(8,6,8,6)^\top$。
依据：自行构造

3. 最终除以 $\sqrt4=2$，得到 $(4,3,4,3)^\top$，与显式乘 $H_4$ 相同。每对先同时读取旧值，再写入和与差，不能用已经更新的和去算差。
依据：自行构造
:::

::: 可执行的快速变换与配对验证
```python
from math import sqrt


def hadamard(values):
    data = [float(value) for value in values]
    size = len(data)
    if size == 0 or size & (size - 1):
        raise ValueError("length must be a positive power of two")
    half = 1
    while half < size:
        for start in range(0, size, 2 * half):
            for offset in range(half):
                left = start + offset
                right = left + half
                u, v = data[left], data[right]
                data[left], data[right] = u + v, u - v
        half *= 2
    return [value / sqrt(size) for value in data]


x = [7, 1, 0, 0]
z = hadamard(x)
print("transform:", z)
print("inverse:", hadamard(z))
weights = [1, 2, 0, 0]
rotated_weights = hadamard(weights)
original = sum(w * value for w, value in zip(weights, x))
rotated = sum(w * value for w, value in zip(rotated_weights, z))
assert abs(original - rotated) < 1e-12
print("outputs:", original, rotated)
```
```text
transform: [4.0, 3.0, 4.0, 3.0]
inverse: [7.0, 1.0, 0.0, 0.0]
outputs: 9 9.0
```
依据：自行构造
:::

### 适用条件与边界

正交变换保证的是长度和内积，不是任意输入的最大坐标下降。固定 Hadamard 也可能把原本等大的坐标合成一个大值；随机符号的期望性质同样不保证每次都减小峰值。因此量化误差是否下降，要结合实际输入、共享步长的范围和舍入结果判断。
依据：自行构造

::: 峰值与量化误差都可能变差的反例
1. 峰值反例取 $x=(2,2,2,2)^\top$，$H_4x=(4,0,0,0)^\top$。最大绝对值由 $2$ 增到 $4$，长度仍保持不变。
依据：自行构造

2. 误差反例另取 $x=(3,3,3,1)^\top$，仍用 $[-3,3]$ 和最大绝对值定步长。直接量化步长为 $1$，各坐标恰好在网格上，误差为零。
依据：自行构造

3. 变换后 $z=(5,1,1,-1)^\top$，步长为 $5/3$，反量化得到 $\hat z=(5,5/3,5/3,-5/3)^\top$。变换坐标的平方误差为 $4/3$，正交逆变换后也是 $4/3$，大于直接量化的零误差。
依据：自行构造
:::

配对式不能穿过任意非线性运算。若中间操作逐坐标改变数值，一般不能把原式中的输入直接换基，再把逆变换吸收到另一侧权重；必须在确实能抵消的位置插入成对变换。
依据：自行构造

::: 非线性运算不一定与 Hadamard 交换
1. 定义 $f$ 为逐坐标把负数截成零的函数，取 $x=(1,-1,0,0)^\top$，则 $H_4x=(0,1,0,1)^\top$。
依据：自行构造

2. 先变换再截负数，$f(H_4x)=(0,1,0,1)^\top$；先截负数再变换，$H_4f(x)=(1/2,1/2,1/2,1/2)^\top$。两者不同，不能把变换直接移过 $f$。
依据：自行构造
:::

二次幂维度的快速实现不能直接处理任意长度。非二次幂阶数的 Hadamard 矩阵是否存在并无普遍保证，需另行核对结构。
依据：paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices

在线 Hadamard 是新增的运算，不会因为权重预先变换就自动消失；输入变换以高精度执行，部署时还需计算这部分开销。
依据：paper/quarot-v2-src/method.tex Stage 1a: Weight Modification；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations

理论配对式按实数运算成立，浮点执行时有限精度加减仍可能产生舍入误差，这与把变换后数值量化到整数网格的误差不同。
依据：自行构造
