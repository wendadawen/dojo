# SwiGLU

- 描述：SwiGLU 是 Transformer 前馈层的一种写法，把前馈层的激活函数换成一道由 Swish 算出的门，逐元素缩放另一条线性投影。
- 摘要：$\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV$，两条线性投影逐元素相乘，一条先过 Swish 当门，用同一参数量换更好的结果。
- 主题：模型结构
- 标签：模型架构

## 资料

- Noam Shazeer. GLU Variants Improve Transformer. arXiv:2002.05202v1, 2020. [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)

## 短页

### 门换成 Swish 的前馈层

Transformer 是主干由多层相同结构堆叠成的网络。前馈层是其中逐个位置各算各的一个子层：输入是这一位置的向量 $x$（上一层传下来的表示），先乘一个权重矩阵、过一个激活函数（逐元素作用的非线性函数），再乘一个权重矩阵。
依据：paper/src-v1/main.tex Introduction

SwiGLU 是前馈层的另一种写法：中间那一步不再是一个激活函数，而是两条线性投影的逐元素乘积，其中一条先过 Swish 当门。Swish 指 $\mathrm{Swish}_1(z) = z\sigma(z)$，下标 1 是系数；$\sigma$ 是 sigmoid，$\sigma(z) = 1/(1+e^{-z})$，把实数压到 0 到 1 之间。两条投影由 $x$ 乘两组权重得到，门的每个分量逐元素缩放另一条投影的对应分量。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

### 为什么换掉 ReLU

前馈层中间通常用的是 ReLU，它对每个维度取正部 $\max(z, 0)$：为正就原样留下，为负就归零。
依据：paper/src-v1/main.tex Introduction

SwiGLU 把这一步换成由 Swish 算出的门。在参数量和计算量对齐的比较里，它的困惑度（模型预测词的准确程度，越低越好）是测量过的几种写法里最好的一档，下游任务（预训练后微调，也就是再在具体任务上训练，如问答、句子分类）也大多更好。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants；paper/src-v1/main.tex Pre-Training and Perplexity Results；paper/src-v1/main.tex Fine-Tuning

### 两道投影相乘，一道过 Swish

SwiGLU 的前馈层中间一步写成

$$\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV.$$
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

$x$ 是输入向量，$W$、$V$ 是两个权重矩阵，把 $x$ 投到同一维数，$\otimes$ 是逐元素相乘。这个结果再过第三个权重矩阵 $W_2$，所以这一层有三块权重，比 ReLU 前馈层多一块。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

## 长页

Transformer 是主干由若干相同结构的层堆叠成的网络。每层里有一个逐个位置各算各的前馈层：输入是这一位置的向量 $x$（上一层传下来的表示），先过一个线性变换，再过一个激活函数（逐元素作用的非线性函数），最后过第二个线性变换。
依据：paper/src-v1/main.tex Introduction

SwiGLU 是一种前馈层的写法：把前馈层里的激活函数换成一道门。它用两组权重把 $x$ 分别投成两个向量，其中一个过 Swish 再与另一个逐元素相乘，最后过第二个线性变换。Swish 是一种逐元素作用的非线性函数，$\mathrm{Swish}_1(z) = z\sigma(z)$，下标 1 是系数；$\sigma$ 是 sigmoid，$\sigma(z) = 1/(1+e^{-z})$，把实数压到 0 到 1 之间。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

前馈层里通常用的是 ReLU，它对每个维度取正部。SwiGLU 的门换成 Swish 后，在同样参数量和计算量下，训练得到的困惑度（模型预测词的准确程度，越低越好）更低。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants；paper/src-v1/main.tex Pre-Training and Perplexity Results

### 前馈层里换上一道门

前馈层对每个位置单独作用。写成不用偏置（每个输出分量上加的可学习常数）的形式是 $\mathrm{FFN}_{\mathrm{ReLU}}(x, W_1, W_2) = \max(xW_1, 0)W_2$，$x$ 是这一位置的向量，$W_1$ 是第一个权重矩阵，$\mathrm{FFN}$ 就是前馈层，$\max(\cdot, 0)$ 是 ReLU，逐元素取正部；中间过完激活函数后乘 $W_2$。
依据：paper/src-v1/main.tex Introduction

要把中间一步改成门控，做法是把第一块权重 $W_1$ 换成两块权重 $W$、$V$，得到两个投影 $xW$ 与 $xV$，再把 $xW$ 过一道门、逐元素乘上 $xV$。这就是[GLU](wiki/glu)：一条投影过 sigmoid 当门，逐元素缩放另一条投影。SwiGLU 只是把这道门从 sigmoid 换成 Swish。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

### SwiGLU 的公式

SwiGLU 用在前馈层的中间一步，公式是

$$\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV.$$
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

$x$ 是这一位置的向量，$W$ 与 $V$ 是两个同形的权重矩阵，把 $x$ 投到同一维数，$\otimes$ 是逐元素相乘。$xW$ 这一支过 Swish 当门，$xV$ 这一支被它逐元素缩放，相乘得到这一层的中间结果。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

整层前馈写成 $\mathrm{FFN}_{\mathrm{SwiGLU}}(x, W, V, W_2) = \mathrm{SwiGLU}(x, W, V)\, W_2$：中间结果再乘第三个权重矩阵 $W_2$。所以一个 SwiGLU 前馈层有三块权重，而 ReLU 前馈层只有两块。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

### 手算一遍 SwiGLU

::: 手算的全过程
取输入 $x = 2$，两个权重 $W = 3$、$V = -1$，输出权重 $W_2 = 1$，维数都是 1。
依据：自行构造

1. $xV$ 这一支：$xV = 2 \times (-1) = -2$。
2. $xW$ 这一支（门）：先算 $xW = 2 \times 3 = 6$。
3. 过 Swish：$\sigma(6) \approx 0.9975$，$\mathrm{Swish}_1(6) = 6 \times 0.9975 \approx 5.985$。
4. 逐元素相乘：中间结果 $= 5.985 \times (-2) = -11.97$。
5. 过第二个线性变换：输出 $= -11.97 \times 1 = -11.97$。
依据：自行构造
:::

门 $= xW \cdot \sigma(xW)$，$\sigma$ 恒正，所以门的正负由 $xW$ 决定，乘积的正负由 $xW$ 与 $xV$ 共同决定。
依据：自行构造

### 隐层宽度为什么要乘 2/3

ReLU 前馈层的隐层维数记 $d_{ff}$，SwiGLU 的记 $d_{ff}'$；隐层指前馈层中间那一层，维数就是第一个线性变换输出的维数。要在相同参数和相同计算量下比较，就要把 SwiGLU 的 $d_{ff}'$ 缩到原来的 $\frac{2}{3}$。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants

::: 2/3 的来历
把 ReLU 前馈层的两块权重记为 $2d \cdot d_{ff}$，SwiGLU 的三块权重记为 $3d \cdot d_{ff}'$，$d$ 是输入 $x$ 的维数。要两者相等，$d_{ff}' = \frac{2}{3} d_{ff}$。测量用的设置里 $d_{ff} = 3072$，$d_{ff}' = \frac{2}{3} \times 3072 = 2048$。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants；paper/src-v1/main.tex Model Architecture
:::

### 适用条件与边界

这套结果的前提是参数量和计算量对齐：被比较的模型参数总数和每 token（词元，序列切成的一个个片段）的计算量相同。
依据：paper/src-v1/main.tex Gated Linear Units (GLU) and Variants；paper/src-v1/main.tex Pre-Training and Perplexity Results

预训练（先在大规模文本上训练）的去噪任务是把输入里被遮住的片段补回来，在它的困惑度上，SwiGLU 是测量过的几种写法里最好的一档。
依据：paper/src-v1/main.tex Pre-Training and Perplexity Results

下游任务（预训练后微调，也就是再在具体任务上训练，如问答、句子分类）上，GLU 变体在多数任务里最好，但这一批结果本身波动较大。
依据：paper/src-v1/main.tex Fine-Tuning

门为什么换成 Swish 而不是别的函数，资料没有给出解释。
依据：paper/src-v1/main.tex Conclusions
