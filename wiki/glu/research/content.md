# GLU

- 描述：GLU 是一种逐维度的乘性门控单元：把输入做两个线性投影，其中一个过 sigmoid 压到 0 到 1 之间当门，再与另一个逐元素相乘，让每个维度各自决定放行多少。
- 摘要：$h(\mathbf{X}) = (\mathbf{X}\ast\mathbf{W}+\mathbf{b}) \otimes \sigma(\mathbf{X}\ast\mathbf{V}+\mathbf{c})$，一条线性投影乘上一道 sigmoid 门，既保留非线性又给梯度留一条不被压的线性通路。
- 主题：模型结构, 训练与优化
- 标签：模型架构

## 资料

- Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier. Language Modeling with Gated Convolutional Networks. ICML 2017. [https://arxiv.org/abs/1612.08083](https://arxiv.org/abs/1612.08083)

## 短页

### 逐维度的乘性门

GLU 是一种逐维度的乘性门控单元。它把输入各做一次线性投影——与一组可学权重相乘——得到两个同样长的向量，其中一个向量的每个分量压到 0 到 1 之间，当作对应位置的门，再与另一个向量逐元素相乘。压到 0 到 1 之间用的是 sigmoid（把任意实数映到 0 到 1 的 S 形函数），记作 $\sigma$。乘出来的结果里，每个维度放行多少由它自己的门决定：门是一个随输入变化的系数，同一个单元在不同输入下放行的维度可以不同。
依据：paper/src-v3/main.tex L166-L178

### 让梯度有一条不被压的路

反向传播要一层层往回传梯度，每一层都会给它乘上一些系数。常见的门控做法是：把输入分成两支，一支过 sigmoid 当门，另一支套一个两端趋平、取值有界的函数，两支逐元素相乘。这类单元的代价是，梯度反向经过时两项都要乘上相应的导数；当输入取值很大、函数的两端变平时，这些导数都趋近 0，梯度被越乘越小，加深层数就越难训练。GLU 只有门这一支过 sigmoid，另一支保持线性、不套函数。这样梯度里就有一项只乘 sigmoid 本身，不带任何会趋近 0 的导数，只要门开着，这一项就能不被缩放地传下去。
依据：paper/src-v3/main.tex L203-L232

### 两道投影相乘，一道当门

给定输入 $\mathbf{X}$，GLU 算

$$h(\mathbf{X}) = (\mathbf{X}\ast\mathbf{W}+\mathbf{b}) \otimes \sigma(\mathbf{X}\ast\mathbf{V}+\mathbf{c})$$

$\mathbf{W}$、$\mathbf{b}$ 与 $\mathbf{V}$、$\mathbf{c}$ 是两组可学的权重和偏置（偏置是先与权重结果相加的常数向量），$\ast$ 是卷积（与权重按位置加权求和），$\otimes$ 是逐元素相乘，$\sigma$ 是 sigmoid。$\mathbf{X}\ast\mathbf{W}+\mathbf{b}$ 是值，$\sigma(\mathbf{X}\ast\mathbf{V}+\mathbf{c})$ 是门。门的分量都在 0 到 1 之间：全趋近 1 时输出接近值，全趋近 0 时输出接近 0，一般情形下每个分量被自己的门按比例缩放。
依据：paper/src-v3/main.tex L166-L178

## 长页

GLU 是一种逐维度的乘性门控单元，它按输入内容逐维度决定放行还是压低哪些信息。用法是把输入算成两个同样长的向量，其中一个的每个分量压到 0 到 1 之间当门，再与另一个逐元素相乘。压到 0 到 1 之间用的是 sigmoid，记作 $\sigma$，它把任意实数映到 0 到 1 之间。需要这么一个单元，是因为门控让网络自己控制哪些信息往下传，也给梯度留了一条不被压的通路。
依据：paper/src-v3/main.tex L107；paper/src-v3/main.tex L166-L178；paper/src-v3/main.tex L203-L232

### 定义

设某一层的输入是 $\mathbf{X}$，形状是「位置数乘特征维」：若干个位置（例如一句话里的若干个词）各带一个特征向量。GLU 把输入分别做两次线性投影，每次用一组可学的权重和偏置：先与权重按位置加权求和，再加上一组可学偏置（常数向量）。这两次投影各得一个同形的张量（多维数组），一个当值，另一个过 sigmoid 得门，两者逐元素相乘，就是这一层的输出：

$$h(\mathbf{X}) = (\mathbf{X}\ast\mathbf{W}+\mathbf{b}) \otimes \sigma(\mathbf{X}\ast\mathbf{V}+\mathbf{c})$$

$\mathbf{W}$、$\mathbf{b}$ 是值那一组的权重和偏置，$\mathbf{V}$、$\mathbf{c}$ 是门那一组；$\ast$ 是卷积，即与权重按位置加权求和，$\mathbf{b}$、$\mathbf{c}$ 分别加在两次卷积的结果上；$\otimes$ 是逐元素相乘，$\sigma$ 是 sigmoid。门的每个分量落在 0 与 1 之间（不含两端），它逐维度缩放值：门趋近 1，对应维度接近原样通过；门趋近 0，对应维度接近被挡掉。
依据：paper/src-v3/main.tex L166-L178

### 一个手算的小例子

取输入 $\mathbf{X} = (1,\ 2)$，两个投影都用简单的 2×2 权重，算一遍 GLU。
依据：自行构造

::: 手算的全过程
输入是 2 维向量 $(1,\ 2)$。这里卷积核（与输入做卷积的那组权重）只覆盖一个位置，投影就是矩阵乘，两组权重都取 2×2 的矩阵，偏置取零。

1. 值投影 $\mathbf{W}$ 取单位阵、$\mathbf{b} = (0,\ 0)$：值 $= \mathbf{X}\mathbf{W}+\mathbf{b} = (1,\ 2)$。
2. 门投影 $\mathbf{V}$ 取 $\begin{pmatrix}1 & 0\\0 & -1\end{pmatrix}$、$\mathbf{c} = (0,\ 0)$：门的线性量 $= \mathbf{X}\mathbf{V}+\mathbf{c} = (1,\ -2)$。
3. 过 sigmoid：$\sigma(1) = 0.7311$，$\sigma(-2) = 0.1192$，门 $= (0.7311,\ 0.1192)$。
4. 逐元素相乘：输出 $= (1 \times 0.7311,\ 2 \times 0.1192) = (0.7311,\ 0.2384)$。
依据：自行构造
:::

第一维的门偏大，值被压低一些；第二维的门很小，值被压得小得多。
依据：自行构造

### 它为什么给梯度留了一条线性通路

sigmoid 门控把它的导数乘进了梯度。GLU 的输出对输入的导数可以拆成两项，其中一项只带 sigmoid 本身、不带它的导数。
依据：paper/src-v3/main.tex L218-L232

::: 导数的推导与两项的来历
把两个投影和偏置都暂时取成恒等（$\mathbf{W}=\mathbf{V}=\mathbf{I}$、$\mathbf{b}=\mathbf{c}=0$），公式缩成 $\mathbf{X}\otimes\sigma(\mathbf{X})$，对它用乘法法则（$\nabla$ 是求导算子，$\nabla\mathbf{X}$ 是上一层回传到本层的梯度）：

$$\nabla[\mathbf{X} \otimes \sigma(\mathbf{X})] = \nabla\mathbf{X} \otimes \sigma(\mathbf{X}) + \mathbf{X} \otimes \sigma'(\mathbf{X}) \nabla \mathbf{X}$$

1. 第一项 $\nabla\mathbf{X} \otimes \sigma(\mathbf{X})$ 只乘 sigmoid 的值，不带 $\sigma'$。门开着（$\sigma$ 接近 1）时，这一项接近 $\nabla\mathbf{X}$，梯度不被缩放地传过这一层。
2. 第二项 $\mathbf{X} \otimes \sigma'(\mathbf{X}) \nabla\mathbf{X}$ 带 $\sigma'$。门饱和到 1 时 $\sigma'$ 趋近 0，这一项消失；梯度仍可通过第一项。
依据：paper/src-v3/main.tex L226-L231
:::

这条通路不带会趋近 0 的导数，梯度经过它时不被缩小。
依据：paper/src-v3/main.tex L218-L232

### 和 tanh 门控的区别

tanh 是把实数映到 $-1$ 到 $1$、两端趋平的函数。把值那一支再套上 tanh、门仍过 sigmoid，得到 tanh 门控；它输出对输入的导数两项都带这种两端趋平的导数，输入很大时都趋近 0，没有 GLU 那条不被缩放的线性通路。
依据：paper/src-v3/main.tex L216；paper/src-v3/main.tex L476-L481

### 适用条件与边界

那条不被缩放的线性通路只在门被激活时成立：门饱和到 0 时，$\sigma$ 也接近 0，通路随之断开，梯度在这一维上同样传不回去。门控本身要增加参数：值、门两支各有一组权重，比不门控的单个线性层多，比较时要按相同参数量对齐。另外，门逐维度、逐位置地缩放输入，本身不把不同位置的信息混起来；在用它搭成的网络里，上下文加长带来的困惑度（衡量语言模型预测下一个词的不确定程度）改善会越来越小。
依据：paper/src-v3/main.tex L177；paper/src-v3/main.tex L231；paper/src-v3/main.tex L476-L499；paper/src-v3/main.tex L542-L553
