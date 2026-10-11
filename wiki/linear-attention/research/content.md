# Linear Attention

- 描述：线性注意力把注意力里查询与键的相似度改写成两个特征向量的内积，用矩阵乘法的结合律交换求和顺序，把计算量从随序列长度平方增长降到线性增长。
- 摘要：把注意力分数写成 $\phi(q)^\top \phi(k)$ 的内积，先算所有键值的外积之和再乘查询，复杂度从 $O(N^2)$ 降到 $O(N)$；因果形式下这个和就是固定大小的状态，逐步更新，推理等价于循环网络。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 资料

- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020. [https://arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
- Idiap Research Institute. fast-transformers: official implementation. [https://github.com/idiap/fast-transformers](https://github.com/idiap/fast-transformers)

## 短页

### 把注意力分数换成内积

标准注意力里，每个位置用查询和每个位置的键做点积、取指数，得到一组分数，归一后对值加权求和；查询、键、值分别是每个位置用来找信息、被比较、被取值的那三个向量（[标准注意力](wiki/standard-attention)）。线性注意力换一种相似度：把查询和键各用一个函数 $\phi$ 映成一个向量，用这两个向量的内积当相似度；用内积表示相似度的函数叫核，把查询和键映成向量的这个 $\phi$ 叫特征映射。$\phi$ 的输出取得非负，相似度就不会是负的。

依据：paper/src-v3/arxiv.tex Transformers；paper/src-v3/arxiv.tex Linearized Attention；paper/src-v3/arxiv.tex Feature Maps and Computational Cost

### 长序列下的平方代价

标准注意力要为每一对位置算一次相似度，得到并保存一张 $N\times N$ 的分数矩阵，$N$ 是序列长度。前向计算（从输入算出全部输出的一次计算）的时间和保存它的内存都随 $N$ 的平方增长。线性注意力把同样的加权求和换一种顺序算出来，前向的时间和内存都随 $N$ 线性增长。

依据：paper/src-v3/arxiv.tex Introduction；paper/src-v3/arxiv.tex Linearized Attention

### 先求和再乘查询

计算顺序是这样的：每个位置先把它的键经过 $\phi$ 得到的特征向量和对应的值按外积相乘（特征向量是列向量、值是行向量，外积得到一个矩阵），把所有位置的这些矩阵加起来，得到一个矩阵；再把所有位置的键的特征向量加起来，得到一个向量。然后对每个位置，用它的查询的特征向量的转置（行向量）左乘那个矩阵，再用查询的特征向量与那个向量做内积，前者除以后者得到该位置的输出。这个矩阵的大小只由特征向量的维度和值的维度决定，与序列长度无关，所以算一次就能给所有位置复用。

依据：paper/src-v3/arxiv.tex Linearized Attention

## 长页

Transformer 是一类按位置并行处理序列的模型，由若干结构相同的层叠成。它的一层里，注意力让每个位置从其他位置取值：每个位置从自己的输入算出查询、键、值三个向量，一个位置用它的查询去和序列里各个位置的键比较，按比较的结果决定从各个位置的值里各取多少信息，取回的信息合起来就是该位置的新表示。标准注意力（[标准注意力](wiki/standard-attention)）用查询与键的点积取指数作为两者的相似度。

序列里共有 $N$ 个位置，每个位置都要和所有位置比一遍，合起来是 $N$ 的平方次比较，保存这些分数还要一张 $N\times N$ 的矩阵。序列越长，计算时间和内存增长越快。线性注意力保留按相似度从各个位置取值的做法，换一种计算顺序：把能预先合并的键和值先合并起来，再用查询去取，时间和内存都只随序列长度线性增长。

依据：paper/src-v3/arxiv.tex Transformers；paper/src-v3/arxiv.tex Introduction；paper/src-v3/arxiv.tex Linearized Attention

### 把相似度换成特征内积

标准注意力给位置 $i$ 的输出是各个位置的值按相似度加权、再除以相似度之和：

$$V'_i = \frac{\sum_{j=1}^N \mathrm{sim}(Q_i,K_j)\, V_j}{\sum_{j=1}^N \mathrm{sim}(Q_i,K_j)}.$$

$Q_i$ 是位置 $i$ 的查询向量，$K_j$、$V_j$ 是位置 $j$ 的键向量和值向量，$\mathrm{sim}(\cdot,\cdot)$ 是相似度函数。这个式子对相似度只要求一件事：非负。相似度非负，分母才是正的，各个位置的权重才都非负、和为 1。标准注意力取 $\mathrm{sim}(q,k)=\exp(q^\top k/\sqrt{D})$，$D$ 是查询和键的维度。

线性注意力把相似度取成一个核：核是这样一个函数，存在一个映射 $\phi$，把查询或键从 $D$ 维映到一个 $C$ 维向量，使 $\mathrm{sim}(q,k)=\phi(q)^\top \phi(k)$；这个 $\phi$ 叫特征映射，它的输出向量就叫特征向量。$\phi$ 的输出取非负，内积就非负，满足上面那个要求。把这个相似度代进上面的式子，求和里的每一项都变成两个特征向量的内积：

$$V'_i = \frac{\sum_{j=1}^N \phi(Q_i)^\top \phi(K_j)\, V_j}{\sum_{j=1}^N \phi(Q_i)^\top \phi(K_j)}.$$

依据：paper/src-v3/arxiv.tex Transformers；paper/src-v3/arxiv.tex Linearized Attention；paper/src-v3/arxiv.tex Feature Maps and Computational Cost

::: 从广义注意力式到特征内积式的推导
把 $\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$ 逐项展开，看求和号里的每一项；「广义式」指那个只要求相似度非负、相似度可以是任何函数的输出式。
依据：paper/src-v3/arxiv.tex Linearized Attention

1. 标准注意力与特征形式共用同一个广义式，区别只在 $\mathrm{sim}$ 的取法；广义式要求 $\mathrm{sim}\ge0$。
2. 取核相似度 $\mathrm{sim}(Q_i,K_j)=\phi(Q_i)^\top\phi(K_j)$，它非负，因为 $\phi(Q_i)$、$\phi(K_j)$ 都是非负向量，它们的内积也非负。
3. 分子里 $\phi(Q_i)$ 与求和下标 $j$ 无关，因此可以把 $\phi(Q_i)^\top$ 提到求和号外。
4. 于是每个位置只用算一组与位置无关的量：$\sum_j \phi(K_j)V_j^\top$ 和 $\sum_j\phi(K_j)$。
依据：paper/src-v3/arxiv.tex L278-L325；paper/src-v3/arxiv.tex L347-L351
:::

### 交换求和顺序

$\phi(Q_i)$ 和求和下标 $j$ 无关，可以用矩阵乘法的结合律把它提到求和号外面，把先算分数再乘值改成先算出所有键值的加权和、再乘查询：

$$V'_i = \frac{\phi(Q_i)^\top \sum_{j=1}^N \phi(K_j)V_j^\top}{\phi(Q_i)^\top \sum_{j=1}^N \phi(K_j)}.$$

分子里的 $\sum_j \phi(K_j)V_j^\top$ 是一个 $C\times M$ 的矩阵，$M$ 是值向量的维度，每一项 $\phi(K_j)V_j^\top$ 是列向量 $\phi(K_j)$ 与行向量 $V_j^\top$ 的外积；分母里的 $\sum_j \phi(K_j)$ 是一个 $C$ 维向量。这里把值写成行向量 $V_j^\top$，$V'_i$ 也写成行向量，与写成列向量只是写法不同，数值相同。两者都只用到键和值，与查询无关，整条序列只算一次，之后每个位置取用同一份。这个矩阵的大小由 $C$ 和 $M$ 决定，与 $N$ 无关，所以缓存它需要的内存也与 $N$ 无关。

两种顺序算出的结果相同。

依据：paper/src-v3/arxiv.tex Linearized Attention；paper/src-v3/arxiv.tex Feature Maps and Computational Cost

::: 两种计算顺序的代价对比
按加法和乘法次数比较，$D$ 是查询和键的维度，$M$ 是值的维度，$C$ 是特征维度，$N$ 是序列长度。
依据：paper/src-v3/arxiv.tex Feature Maps and Computational Cost

1. 标准注意力的代价是 $O(N^2\max(D,M))$：每个位置的查询都要和所有位置的键算分数。
2. 线性注意力先把查询和键映到 $C$ 维，再算输出，代价是 $O(NCM)$。
3. 只要 $N$ 比 $C$ 大得多，线性注意力就更省。
依据：paper/src-v3/arxiv.tex Feature Maps and Computational Cost
:::

#### 特征映射的选取

要让内积当相似度，就得让它非负。取 $\phi(x)=\mathrm{elu}(x)+1$，逐分量作用在查询和键上：elu 是「指数线性单元」这样一个逐分量的函数，$x>0$ 时取 $x$，$x\le 0$ 时取 $e^x-1$；再加 1，输出落在 $(0,+\infty)$ 里，任意两个这样的向量的内积因此都是正的。之所以不用更简单的 relu（$x>0$ 时取 $x$，否则取 $0$）加 1，是因为 relu 对负的输入恒为 0，负的一半上梯度为 0，参数得不到更新；elu 在负半轴不恒定，梯度不为零。

另一类核也有有限维特征映射，比如多项式核 $\mathrm{sim}(q,k)=(q^\top k+c)^p$，$c$ 是常数、$p$ 是次数：它的特征映射维度是有限的，可以写出展开式。标准注意力取的点积取指数是一个无限维的核，写不出有限维的特征映射，只能用有限维的 $\phi$ 近似。

依据：paper/src-v3/arxiv.tex Feature Maps and Computational Cost；paper/src-v3/arxiv.tex Linearized Attention

::: 特征映射对应的计算量
取 $\phi(x)=\mathrm{elu}(x)+1$ 时，特征维度 $C$ 等于查询、键的维度 $D$，计算量是 $O(NDM)$，$M$ 是值向量的维度。取二次多项式核时，特征维度是 $O(D^2)$，计算量是 $O(ND^2M)$。
依据：paper/src-v3/arxiv.tex Feature Maps and Computational Cost

:::

### 因果掩码下的状态形式

生成任务里，位置 $i$ 只能用到它自己和它之前的位置，即下标满足 $j\le i$，遮挡未来位置的做法叫因果掩码（[因果掩码](wiki/causal-mask)）。把所有求和的上限从 $N$ 收到 $i$：

$$V'_i = \frac{\phi(Q_i)^\top \sum_{j=1}^i \phi(K_j)V_j^\top}{\phi(Q_i)^\top \sum_{j=1}^i \phi(K_j)}.$$

把分子和分母里那个与位置 $i$ 一起增长的项记成两个状态：

$$S_i=\sum_{j=1}^i \phi(K_j)V_j^\top,\qquad Z_i=\sum_{j=1}^i \phi(K_j),$$

则位置 $i$ 的输出是 $V'_i=\phi(Q_i)^\top S_i\,/\,\phi(Q_i)^\top Z_i$。$S_i$ 和 $Z_i$ 可以由前一个位置的状态加上当前项得到：$S_i=S_{i-1}+\phi(K_i)V_i^\top$，$Z_i=Z_{i-1}+\phi(K_i)$，每加一个位置只做常数次加法（$S$、$Z$ 各加一次），代价与位置无关。整条序列是逐个位置累加出来的，时间和内存都随 $N$ 线性增长。

依据：paper/src-v3/arxiv.tex Causal Masking；paper/src-v3/arxiv.tex Linearized Attention

#### 小例子：四个位置走一遍

下面用四个位置的小例子把两种计算顺序和状态的累积走一遍。

依据：自行构造

::: 四个位置走一遍
取 4 个位置。特征维度取 2，每个位置的值是一个数，所以值维度 $M=1$，下面用 $S_i=\sum_j\phi(K_j)V_j^\top$ 的约定，$S$ 是一个 2 维向量。特征映射取 $\phi(x)=\mathrm{elu}(x)+1$，逐分量作用，输出是 $e^x$（$x\le0$）或 $x+1$（$x>0$）。
依据：自行构造

查询、键、值是 $Q=[[1,1],[1,0],[0,1],[1,1]]$、$K=[[1,0],[0,1],[1,1],[2,0]]$、$V=[1,2,1,1]$。
依据：自行构造

1. 把每个查询和键过一遍 $\phi$：$\phi(Q)=[[2,2],[2,1],[1,2],[2,2]]$，$\phi(K)=[[2,1],[1,2],[2,2],[3,1]]$。
2. 先乘后求和：位置 1 的输出是 $\big(\phi(Q_1)^\top\phi(K_1)\cdot1+\phi(Q_1)^\top\phi(K_2)\cdot2+\phi(Q_1)^\top\phi(K_3)\cdot1+\phi(Q_1)^\top\phi(K_4)\cdot1\big)$ 除以同理的分母，等于 $1.2143$。
3. 位置 2、3、4 同理得到 $1.1818$、$1.2500$、$1.2143$。
4. 先求和：$S=\sum_j\phi(K_j)V_j^\top=[9,8]$，$Z=\sum_j\phi(K_j)=[8,6]$；用 $\phi(Q_i)^\top S/\phi(Q_i)^\top Z$ 再算一遍，四个位置得到 $1.2143$、$1.1818$、$1.2500$、$1.2143$，与先乘后求和一致。
依据：自行构造

5. 换成因果：只累加到当前位置。$S$ 依次是 $[2,1]$、$[4,5]$、$[6,7]$、$[9,8]$，$Z$ 依次是 $[2,1]$、$[3,3]$、$[5,5]$、$[8,6]$，四个位置的输出是 $1.0000$、$1.4444$、$1.3333$、$1.2143$。
依据：自行构造

```python
import math

def phi(vec):
    return [math.exp(v) if v <= 0 else v + 1 for v in vec]

def fmt(xs):
    return [int(x) if x == int(x) else x for x in xs]

Q = [[1, 1], [1, 0], [0, 1], [1, 1]]
K = [[1, 0], [0, 1], [1, 1], [2, 0]]
V = [1, 2, 1, 1]

def w(p, q):
    return sum(a * b for a, b in zip(p, q))

for i, q in enumerate(Q):
    p = phi(q)
    num = sum(w(p, phi(K[j])) * V[j] for j in range(4))
    den = sum(w(p, phi(K[j])) for j in range(4))
    print("先乘后求和", i + 1, round(num / den, 4))

S = [0, 0]
Z = [0, 0]
for j in range(4):
    pk = phi(K[j])
    for d in range(2):
        S[d] += pk[d] * V[j]
        Z[d] += pk[d]
print("全部 k 的和 S", fmt(S), "Z", fmt(Z))
for i, q in enumerate(Q):
    p = phi(q)
    print("先求和再乘查询", i + 1, round(w(p, S) / w(p, Z), 4))

S = [0, 0]
Z = [0, 0]
for i in range(4):
    pk = phi(K[i])
    for d in range(2):
        S[d] += pk[d] * V[i]
        Z[d] += pk[d]
    print("因果", i + 1, "S", fmt(S), "Z", fmt(Z))
```

```text
先乘后求和 1 1.2143
先乘后求和 2 1.1818
先乘后求和 3 1.25
先乘后求和 4 1.2143
全部 k 的和 S [9, 8] Z [8, 6]
先求和再乘查询 1 1.2143
先求和再乘查询 2 1.1818
先求和再乘查询 3 1.25
先求和再乘查询 4 1.2143
因果 1 S [2, 1] Z [2, 1]
因果 2 S [4, 5] Z [3, 3]
因果 3 S [6, 7] Z [5, 5]
因果 4 S [9, 8] Z [8, 6]
```
:::

#### 训练并行、推理循环

训练时整条序列都已知，各个位置可以同时算，输出可以一次全部算出。推理时第 $i$ 个输出是第 $i+1$ 个的输入，只能逐个位置算。这时一层维护两个固定大小的量 $S$、$Z$，每读入一个位置 $x_i$ 就更新一次再给出输出。写成

$$S_i = S_{i-1} + \phi(x_iW_K)(x_iW_V)^\top,\quad Z_i = Z_{i-1} + \phi(x_iW_K),$$

$$y_i = f\!\left(\frac{\phi(x_iW_Q)^\top S_i}{\phi(x_iW_Q)^\top Z_i} + x_i\right),$$

$W_Q$、$W_K$、$W_V$ 是算查询、键、值的三个矩阵，$f$ 是层里跟在注意力后面的前馈部分（逐位置各算各的两层小网络），$+x_i$ 是把这层的输出加回它的输入。状态的形状不随位置 $i$ 变，每步的代价也不随 $i$ 变，这样一个因果注意力的层读入一个位置、更新内部状态、给出输出，正是循环网络的形式——循环网络逐个处理输入、每步用前一步留下的内部状态算当前这一步，差别在于这里的内部状态由键值外积之和构成。

依据：paper/src-v3/arxiv.tex Training and Inference；paper/src-v3/arxiv.tex Transformers are RNNs

::: 训练时怎样保持常数内存
推理时每步只更新一次状态；训练时的反向传播（从输出往回逐层算梯度的那一步）要按位置保存状态，会随序列长度占内存。把梯度写成累积和可以避开这一点。
依据：paper/src-v3/arxiv.tex Training and Inference；paper/src-v3/arxiv.tex Gradient Computation

1. 分子 $\bar V_i=\phi(Q_i)^\top S_i$ 的梯度对 $\phi(Q_i)$ 是 $\nabla_{\bar V_i}\mathcal L\,S_i^\top$，$\mathcal L$ 是损失（要最小化的那个标量），$\nabla$ 是对它求梯度；这里的 $S_i$ 就是正向已经算过的累积和，不必另存。
2. 对 $\phi(K_i)$ 和 $V_i$ 的梯度要用到从位置 $i$ 往后到 $N$ 的累积和 $\sum_{j\ge i}\phi(Q_j)(\nabla_{\bar V_j}\mathcal L)^\top$，从后向前扫一遍即可算出。
3. 前向和后向各扫一遍、每次只留一个累积和，因此前向和反向都在线性时间、常数内存下完成。
依据：paper/src-v3/arxiv.tex L407-L427
:::

### 适用条件与边界

线性注意力换掉了相似度函数，它不是标准注意力的等价实现，只能用有限维的特征映射近似它，而不是在原算法上做加速。

收益随序列变长而增大。相似度换成有限维特征映射后，计算量由特征维度 $C$ 决定，$C$ 越大，一次计算要做的乘加越多，代价越高。加速的大小随情形不同。最大的加速出现在自回归生成——生成一个位置、再用它生成下一个，这时标准注意力每生成一个新位置都要和全部历史重算一遍，而线性注意力只把新位置的键值加进状态。在非自回归的任务上，线性注意力同样更快，但快的倍数小得多。

依据：paper/src-v3/arxiv.tex Feature Maps and Computational Cost；paper/src-v3/arxiv.tex Training and Inference；paper/src-v3/arxiv.tex Image Generation；paper/src-v3/arxiv.tex Automatic Speech Recognition

::: 两种情形下加速倍数的差别
逐个位置生成图像（自回归）与一次读出整条序列的语音识别（非自回归）上的加速对比。
依据：paper/src-v3/arxiv.tex Image Generation；paper/src-v3/arxiv.tex Automatic Speech Recognition

1. 生成图像时，线性注意力每秒生成 142.8 张图，标准注意力是 0.45 张，快 317 倍。
2. 语音识别时，线性注意力每一轮训练（一个 epoch）824 秒，标准注意力是 2711 秒，快 3 倍多。
依据：paper/src-v3/arxiv.tex Image Generation；paper/src-v3/arxiv.tex Automatic Speech Recognition
:::
