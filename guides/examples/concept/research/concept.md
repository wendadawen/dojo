# RMSNorm：只按均方根缩放的层归一化

- 描述：RMSNorm 去掉 LayerNorm 里的减均值，把一层的加权和除以它们的均方根后再逐维乘增益，加权和的均值为 0 时两者结果相同。
- 摘要：对一层的加权和 $\mathbf{a}$，LayerNorm 先减均值再除以标准差，RMSNorm 不减均值、直接除以均方根 $\mathrm{RMS}(\mathbf{a})=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}$，两者最后都逐维乘增益 $g_i$。$\mathbf{a}$ 的均值为 0 时两者结果相同。
- 主题：模型结构
- 标签：网络结构

## 资料

- Biao Zhang, Rico Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. [arXiv:1910.07467v1](https://arxiv.org/abs/1910.07467v1)

## 范围

- 讲：LayerNorm 对加权和做的两步，只讲到公式，作为对照。
- 讲：RMSNorm 的公式，以及它和 LayerNorm 什么时候结果相同。
- 讲：论文为什么认为可以去掉减均值，以及支持这一点的实验。
- 不讲：pRMSNorm。论文认为它理论上计算更少，但实测提速并不稳定；只在说明论文的提速数字时提到一次。
- 不讲：论文的梯度分析（§4.2）。推导长，不改变本文的结论。
- 不讲：BatchNorm、WeightNorm 等其他归一化，它们各是独立的概念。
- 不讲：各框架和模型里的实现代码。本文只讲定义、与 LayerNorm 的关系，以及论文的论证。

## 依据

| 论断 | 出处 |
|---|---|
| 前面的层更新时，一层输入的分布会跟着变，这可能影响梯度的稳定、拖慢收敛；LayerNorm 固定加权和的均值和方差来减轻这种变化 | 论文 §3，式 (1) 与式 (2) 之间一段 |
| LayerNorm 被广泛用于各种网络结构 | 论文 §1 第一段 |
| 前馈层先算加权和 $a_i=\sum_j w_{ij}x_j$，再加偏置、过激活；归一化的对象是加权和向量 $\mathbf{a}$ | 论文 §3，式 (1) |
| LayerNorm 的公式含减均值、除标准差，再乘初始为 1 的增益 | 论文 §3，式 (2)(3) |
| RMSNorm 的公式不减均值，分母是均方根 | 论文 §4，式 (4) |
| RMSNorm 由 Zhang 和 Sennrich 在 2019 年提出，论文的实验里效果和 LayerNorm 相当，计算比 LayerNorm 少 | 论文标题页、摘要 |
| 加权和的均值为 0 时，RMSNorm 与 LayerNorm 完全相等 | 论文 §4，式 (4) 之后 |
| 一种常见的解释是 LayerNorm 有效来自中心化不变性和缩放不变性；论文假设起作用的是缩放不变性 | 论文 §4 第一段 |
| 均方根对正数倍缩放是线性的；权重矩阵或层的输入整体乘一个正数时，RMSNorm 的输出不变 | 论文 §4.1，式 (6)(7) |
| LayerNorm 和 RMSNorm 都对权重矩阵的整体放大、对层的输入的整体放大保持不变；LayerNorm 对权重矩阵的平移不变，RMSNorm 不是 | 论文 Table 1 |
| 论文认为减均值并不降低隐藏状态（层的输出）或梯度的方差 | 论文 §1 第二段 |
| RNNSearch 是论文里基于 GRU 的循环网络翻译模型；论文在该实验之后写明，结果支持缩放不变性才是关键这个假设 | 论文 §6.1，Figure 2 与 Table 2 之后 |
| Transformer 是论文里基于自注意力的翻译模型。该实验不做归一化则训练失败；两个测试集上 LayerNorm 的 BLEU 是 26.6、27.7，RMSNorm 是 26.8、27.7；每 1000 步 248 秒对 231 秒，快 6.9% | 论文 §6.1，Table 4 |
| 各表里 RMSNorm 一行比 LayerNorm 快 6.9% 到 40.8% | 论文 Table 2、3、4、6、8、10 |
| pRMSNorm 只用前一部分分量估计均方根 | 论文 §5 |
| 论文写的提速范围是 7% 到 64% | 论文摘要、§1、§7 |
| Table 8 里 pRMSNorm 一行是 63.9%；§6.3 把这一组和 RMSNorm 的提速说成 40% 到 64% | 论文 §6.3，Table 8 |
| 实际提速取决于框架、硬件、网络结构和其他部分的计算量 | 论文 §7 |

## 概览

### 是什么

RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）是神经网络里的一步归一化：把一层算出的加权和除以它们的均方根，再逐维乘可学习的增益。它由 Zhang 和 Sennrich 在 2019 年提出。

### 解决什么问题

LayerNorm 对每个向量做两步：减去均值，再除以标准差。RMSNorm 去掉减均值这一步，计算更少；论文的实验里，效果和 LayerNorm 相当。

### 怎么算

对一层的 $n$ 个加权和组成的向量 $\mathbf{a}$，均方根是 $\mathrm{RMS}(\mathbf{a})=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}$。第 $i$ 个分量变成 $\bar{a}_i=\frac{a_i}{\mathrm{RMS}(\mathbf{a})}g_i$。

### 什么时候成立

- $\mathbf{a}$ 的均值为 0 时，RMSNorm 和 LayerNorm 的结果完全相同；均值不为 0 时两者不同。
- 把层的输入或权重矩阵整体放大若干倍，RMSNorm 的归一化结果不变。给 $\mathbf{a}$ 的所有分量加上同一个数，LayerNorm 的结果不变，RMSNorm 的结果一般会变。
- 论文假设 LayerNorm 起作用靠的是缩放不变性而不是减均值，并在一组循环网络翻译实验之后写明结果支持这个假设。这不是证明。

## 引言

一层神经网络把输入向量变成 $n$ 个加权和，再送进激活函数。训练时前面的层不断更新，这一层拿到的输入分布也跟着变，这可能影响梯度的稳定、拖慢收敛。归一化在激活之前加一步计算。LayerNorm 被广泛用于各种网络结构，它固定的是均值和方差：减去这组数的均值，再除以它们的标准差。

RMSNorm 去掉减均值，直接把这组数除以它们的均方根。它由 Zhang 和 Sennrich 在 2019 年提出，计算比 LayerNorm 少，论文的实验里效果和 LayerNorm 相当。

## 核心问题

### RMSNorm 的公式和 LayerNorm 差在哪里？

解答：不减均值，分母从标准差换成均方根

LayerNorm 把每个分量写成 $\frac{a_i-\mu}{\sigma}g_i$：先减均值 $\mu$，再除以标准差 $\sigma$。RMSNorm 写成 $\frac{a_i}{\mathrm{RMS}(\mathbf{a})}g_i$：不减均值，分母换成均方根。标准差衡量各分量离均值多远，均方根衡量各分量离 0 多远。完整说明见「RMSNorm 去掉减均值，分母换成均方根」一章。

### RMSNorm 和 LayerNorm 的结果什么时候相同？

解答：加权和的均值为 0 时相同

均方根和标准差满足 $\mathrm{RMS}(\mathbf{a})^2=\sigma^2+\mu^2$。均值 $\mu=0$ 时两者相等，两个公式逐项相同。完整说明见「加权和的均值为 0 时两者相同」一章。

### 去掉减均值，为什么还能用？

解答：论文假设起作用的是缩放不变性，并在一组实验之后写明结果支持它

LayerNorm 有效的一种常见解释是两种不变性：整体放大时结果不变，全体加上同一个数时结果也不变。RMSNorm 保留前者、放弃后者。论文假设起作用的是前者。完整说明见「去掉减均值为什么仍然有效」一章。

## 归一化作用在一层的加权和上

一层前馈网络把长度为 $m$ 的输入向量 $\mathbf{x}$ 变成长度为 $n$ 的输出 $\mathbf{y}$：先算加权和，再加偏置、过激活函数。

$$a_i=\sum_{j=1}^{m}w_{ij}x_j,\qquad y_i=f(a_i+b_i),\qquad i=1,\dots,n$$

- $x_j$：输入向量的第 $j$ 个分量。
- $w_{ij}$：权重矩阵 $W$ 第 $i$ 行第 $j$ 列的元素。
- $a_i$：第 $i$ 个输出位置的加权和；$n$ 个加权和组成向量 $\mathbf{a}$。
- $y_i$：第 $i$ 个位置的输出；$n$ 个输出组成向量 $\mathbf{y}$。
- $b_i$：第 $i$ 个位置的偏置。
- $f$：激活函数，对每个分量单独计算。

归一化处理的就是向量 $\mathbf{a}$。LayerNorm 先算出 $\mathbf{a}$ 的均值 $\mu$ 和标准差 $\sigma$：

$$\mu=\frac{1}{n}\sum_{i=1}^{n}a_i,\qquad \sigma=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(a_i-\mu)^2}$$

- $\mu$：$n$ 个加权和的平均值。
- $\sigma$：标准差，衡量各个 $a_i$ 离 $\mu$ 有多远。

再用它们改写每个分量，改写后的 $\bar{a}_i$ 代替 $a_i$ 送进激活函数：

$$\bar{a}_i=\frac{a_i-\mu}{\sigma}\,g_i,\qquad y_i=f(\bar{a}_i+b_i)$$

- $\bar{a}_i$：归一化之后的第 $i$ 个分量。
- $g_i$：第 $i$ 个位置的增益，是可学习的参数，训练开始时设为 1。

减去 $\mu$ 叫中心化，让 $a_i-\mu$ 这组数的均值变成 0。再除以 $\sigma$，让这组数的标准差变成 1。最后乘 $g_i$，由模型自己学每个位置放大多少。

## RMSNorm 去掉减均值，分母换成均方根

RMSNorm 不减均值，把分母换成均方根（root mean square，RMS）：

$$\bar{a}_i=\frac{a_i}{\mathrm{RMS}(\mathbf{a})}\,g_i,\qquad \mathrm{RMS}(\mathbf{a})=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}$$

- $\mathrm{RMS}(\mathbf{a})$：各分量平方的平均值再开方，衡量各个 $a_i$ 离 0 有多远。
- $g_i$：增益，和 LayerNorm 里的一样。

<figure class="diagram">
  <div class="dg-flow">
    <div class="dg-node"><div class="dg-node-title">LayerNorm</div><div class="dg-node-note">加权和 $a_i$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">减均值</div><div class="dg-node-note">$a_i-\mu$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">除以标准差</div><div class="dg-node-note">$(a_i-\mu)/\sigma$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">乘增益</div><div class="dg-node-note">$g_i\,(a_i-\mu)/\sigma$</div></div>
  </div>
  <div class="dg-flow">
    <div class="dg-node"><div class="dg-node-title">RMSNorm</div><div class="dg-node-note">加权和 $a_i$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">不减均值</div><div class="dg-node-note">仍是 $a_i$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">除以均方根</div><div class="dg-node-note">$a_i/\mathrm{RMS}(\mathbf{a})$</div></div>
    <div class="dg-arrow">&#8594;</div>
    <div class="dg-node"><div class="dg-node-title">乘增益</div><div class="dg-node-note">$g_i\,a_i/\mathrm{RMS}(\mathbf{a})$</div></div>
  </div>
  <figcaption class="diagram-caption">第一组是 LayerNorm，第二组是 RMSNorm，箭头是计算顺序。RMSNorm 对应减均值的那一格不做减法，分母也从标准差换成了均方根。</figcaption>
</figure>

和 LayerNorm 比，差别有两处。第一，分子是 $a_i$ 本身，没有减 $\mu$。第二，分母衡量的是各分量离 0 的距离，不是离均值的距离。除以均方根之后，$a_i/\mathrm{RMS}(\mathbf{a})$ 这组数平方的平均值恰好是 1，均值变成原来的均值除以 $\mathrm{RMS}(\mathbf{a})$，不像 LayerNorm 那样被移成 0。

构造的例子：取 $n=4$，$\mathbf{a}=(1,2,3,4)$，增益全取 1。

- LayerNorm：$\mu=2.5$，$\sigma=\sqrt{1.25}$。结果是 $\frac{a_i-2.5}{\sqrt{1.25}}$，约为 $(-1.3416,\,-0.4472,\,0.4472,\,1.3416)$。
- RMSNorm：$\mathrm{RMS}(\mathbf{a})=\sqrt{7.5}$。结果是 $\frac{a_i}{\sqrt{7.5}}$，约为 $(0.3651,\,0.7303,\,1.0954,\,1.4606)$。

LayerNorm 的结果有正有负、均值为 0。这个例子里加权和全为正，RMSNorm 的结果也全为正，均值约为 0.9129，即 $2.5/\sqrt{7.5}$，保留了加权和整体偏正这一点。

计算上，RMSNorm 比 LayerNorm 少做两件事：不用求 $\mathbf{a}$ 的均值 $\mu$，也不用给每个分量减 $\mu$。算均方根仍要做各分量平方的平均。

## 加权和的均值为 0 时两者相同

均方根和标准差之间有一个关系：

$$\mathrm{RMS}(\mathbf{a})^2=\sigma^2+\mu^2$$

- 左边是各分量平方的平均值，右边是方差 $\sigma^2$ 加上均值的平方。

::: 补充：$\mathrm{RMS}(\mathbf{a})^2=\sigma^2+\mu^2$ 的推导
从方差的定义出发，把 $(a_i-\mu)^2$ 展开成 $a_i^2-2\mu a_i+\mu^2$，再对 $i$ 求平均：

$$\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(a_i-\mu)^2=\frac{1}{n}\sum_{i=1}^{n}a_i^2-2\mu\cdot\frac{1}{n}\sum_{i=1}^{n}a_i+\frac{1}{n}\sum_{i=1}^{n}\mu^2$$

右边第一项是 $\mathrm{RMS}(\mathbf{a})^2$。第二项里 $\frac{1}{n}\sum_{i=1}^{n}a_i=\mu$，所以第二项等于 $-2\mu^2$。第三项里 $\mu^2$ 与 $i$ 无关，$\frac{1}{n}\sum_{i=1}^{n}\mu^2=\mu^2$。于是

$$\sigma^2=\mathrm{RMS}(\mathbf{a})^2-2\mu^2+\mu^2=\mathrm{RMS}(\mathbf{a})^2-\mu^2$$

移项得到 $\mathrm{RMS}(\mathbf{a})^2=\sigma^2+\mu^2$。用 $(1,2,3,4)$ 这个例子验算：$\sigma^2=1.25$，$\mu^2=6.25$，和为 $7.5$，正是 $\mathrm{RMS}(\mathbf{a})^2$。
:::

所以 $\mu=0$ 时 $\mathrm{RMS}(\mathbf{a})=\sigma$，LayerNorm 的分子 $a_i-\mu$ 也就是 $a_i$，两个公式逐项相同。论文也写明了这一点：加权和的均值为 0 时，RMSNorm 与 LayerNorm 完全相等。$\mu\neq 0$ 时 $\mathrm{RMS}(\mathbf{a})>\sigma$，分子也不同，两者的归一化结果不一样。

构造的例子：$\mathbf{a}=(-3,-1,1,3)$，均值为 0，增益取 1。$\sigma^2=\mathrm{RMS}(\mathbf{a})^2=5$，两种归一化的结果都是 $\frac{a_i}{\sqrt{5}}$，约为 $(-1.3416,\,-0.4472,\,0.4472,\,1.3416)$。

$(1,2,3,4)$ 减去均值 2.5 得到 $(-1.5,-0.5,0.5,1.5)$。这组数均值为 0，它的均方根就是原来的标准差 $\sqrt{1.25}$，逐个相除正是 $(1,2,3,4)$ 做 LayerNorm 的结果。$(-1.5,-0.5,0.5,1.5)$ 又正好是 $(-3,-1,1,3)$ 的一半，分子和均方根同时减半，相除的结果不变。一般地，增益取 1 时，LayerNorm 等于先减均值、再做 RMSNorm。

::: 代码：复算 $(1,2,3,4)$ 和 $(-3,-1,1,3)$ 两个例子
```python
import math

def layer_norm(a):
    n = len(a)
    mu = sum(a) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in a) / n)
    return [(x - mu) / sigma for x in a]

def rms_norm(a):
    n = len(a)
    rms = math.sqrt(sum(x * x for x in a) / n)
    return [x / rms for x in a]

for a in ([1, 2, 3, 4], [-3, -1, 1, 3]):
    print("a =", a)
    print("  LayerNorm:", [round(x, 4) for x in layer_norm(a)])
    print("  RMSNorm:  ", [round(x, 4) for x in rms_norm(a)])
```

```text
a = [1, 2, 3, 4]
  LayerNorm: [-1.3416, -0.4472, 0.4472, 1.3416]
  RMSNorm:   [0.3651, 0.7303, 1.0954, 1.4606]
a = [-3, -1, 1, 3]
  LayerNorm: [-1.3416, -0.4472, 0.4472, 1.3416]
  RMSNorm:   [-1.3416, -0.4472, 0.4472, 1.3416]
```

`mu`、`sigma`、`rms` 对应公式里的 $\mu$、$\sigma$、$\mathrm{RMS}(\mathbf{a})$。增益全取 1，所以代码里没有乘 $g_i$。
:::

## 去掉减均值为什么仍然有效

LayerNorm 为什么有效，一种常见的解释是它有两种不变性：层的输入或权重矩阵按某种方式改变时，归一化之后的结果不变。

缩放不变性。把 $\mathbf{a}$ 整体乘以一个正数 $\delta$，均方根也乘以 $\delta$，即 $\mathrm{RMS}(\delta\mathbf{a})=\delta\,\mathrm{RMS}(\mathbf{a})$，分子分母里的 $\delta$ 约掉，归一化结果不变。权重矩阵 $W$ 整体乘以 $\delta$，或者输入 $\mathbf{x}$ 整体乘以 $\delta$，都会让 $\mathbf{a}$ 整体乘以 $\delta$，所以 RMSNorm 对这两种放大都不变。LayerNorm 也有这个性质。

中心化不变性。给所有 $a_i$ 加上同一个数 $c$，均值变成 $\mu+c$，每个 $a_i-\mu$ 不变，标准差也不变，所以 LayerNorm 的归一化结果不变。RMSNorm 没有减均值，结果一般会变。前面的例子里增益取 1，$(1,2,3,4)$ 每个分量减去 2.5 之后，RMSNorm 的结果从 $(0.3651,\,0.7303,\,1.0954,\,1.4606)$ 变成 $(-1.3416,\,-0.4472,\,0.4472,\,1.3416)$。给权重矩阵的每一行加上同一个向量，所有 $a_i$ 就会加上同一个数。论文的对照表里，LayerNorm 对这种权重平移不变，RMSNorm 不是。

论文的假设是：LayerNorm 起作用靠的是缩放不变性，而不是中心化不变性。论文给出的理由是，减均值并不降低隐藏状态（即层的输出）或梯度的方差。RMSNorm 保留缩放不变性、放弃中心化不变性，正好可以检验这个假设。

论文在 RNNSearch（一种用 GRU 单元的循环网络翻译模型；GRU 是门控循环单元）实验之后写明，该实验的结果支持这个假设。在 Transformer（一种用[自注意力](wiki/standard-attention)代替循环的翻译模型）实验里，不做归一化的模型训练失败。这一次，LayerNorm 在两个测试集上的 BLEU（机器翻译的评测分数，越高越好）分别是 26.6 和 27.7，RMSNorm 是 26.8 和 27.7。每训练 1000 步，LayerNorm 用 248 秒，RMSNorm 用 231 秒，快 6.9%。论文各张表里 RMSNorm 一行的提速在 6.9% 到 40.8% 之间。论文在摘要里写的提速范围是 7% 到 64%，比各表 RMSNorm 一行的区间更宽，其中最高的 63.9% 来自 pRMSNorm（只用一部分分量估计均方根的变体）。论文同时说明，实际提速取决于框架、硬件、网络结构和其他部分的计算量。

这些数字只覆盖论文测过的模型和任务，不是证明。
