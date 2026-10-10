# RMSNorm

- 描述：RMSNorm 是 LayerNorm 的简化：去掉减均值这一步，只把一层的加权和除以它的均方根，再乘增益；训练质量与 LayerNorm 相当而计算更省。
- 摘要：归一化本来要做减均值和除标准差两件事，RMSNorm 提出假设前者可以省掉，只保留 $\bar{a}_i = a_i / \mathrm{RMS}(\mathbf{a})\, g_i$。
- 主题：模型结构
- 标签：网络结构

## 资料

- Biao Zhang, Rico Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
- Biao Zhang. rmsnorm: official implementation. 2019. [https://github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)

## 短页

### 把加权和除以均方根

神经网络由一层一层叠成，每一层拿到一个输入向量，先做线性加权求和，得到一组分量，这组分量叫这一层的加权和。
依据：paper/src-v1/rmsnorm-v1.tex Background

加权和接着要经过归一化：把它的尺度固定住，再交给激活函数变成这一层的输出，激活函数是一个非线性环节。
依据：paper/src-v1/rmsnorm-v1.tex Background

归一化里有一对随训练更新的可学参数：增益用来缩放，偏置用来平移。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background

LayerNorm 是另一种归一化，它对加权和做两步。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background

第一步减掉均值，这一步叫中心化；第二步除以标准差，这一步叫缩放；之后乘增益、加偏置。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background

RMSNorm 去掉中心化，只保留缩放：不再减均值，只按加权和的均方根（各分量平方的平均再开方）把尺度定下来。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background

### 中心化那一步可以省掉

减均值要先对全部加权和求一次均值、再逐个分量减掉它，层一多，这笔开销在每一步训练里累加起来就不可忽略。LayerNorm 同时给了两种不变性：中心化让输出对整体加常数不敏感，缩放让输出对整体乘常数不敏感。RMSNorm 提出一个假设——让 LayerNorm 真正起作用的是缩放不变性，不是中心化不变性，所以中心化可以省掉。去掉它之后，多个任务上质量与 LayerNorm 相当，而每一步的运算更少。
依据：paper/src-v1/rmsnorm-v1.tex Background；paper/src-v1/rmsnorm-v1.tex Introduction；paper/src-v1/rmsnorm-v1.tex RMSNorm

### 除均方根、乘增益

先算这一层所有分量的均方根，把每个分量除以它，再乘一个可学习的增益向量，得到归一化后的输出。LayerNorm 在乘增益后还要加一个偏置，给输出整体加一个常数；RMSNorm 默认不加偏置，因为它不保证重新居中。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background；repo/rmsnorm/rmsnorm_torch.py L13-L20

## 长页

神经网络由一层一层叠成，每一层拿到一个输入向量后，先做线性加权求和：第 $i$ 个分量的加权和是 $a_i = \sum_j w_{ij} x_j$，$w_{ij}$ 是权重，$x_j$ 是输入的第 $j$ 个分量，这些分量组成向量 $\mathbf{a}$。归一化作用在这个加权和上，把它的尺度固定住再交给激活函数（一层里把加权和变成输出的非线性环节），以稳住各层输出的量级。归一化里有一对随训练更新的可学参数：增益 $\mathbf{g}$ 用来缩放，偏置 $\mathbf{b}$ 用来平移。LayerNorm 把加权和先减均值、再除标准差，然后乘增益、加偏置。RMSNorm 的主张是只保留后半段：不减均值，只把加权和除以它的均方根（各分量平方的平均再开方），再乘增益。
依据：paper/src-v1/rmsnorm-v1.tex Background；paper/src-v1/rmsnorm-v1.tex RMSNorm

### 只保留缩放

#### 从 LayerNorm 去掉减均值

LayerNorm 对加权和做两步。第一步减掉它们的均值 $\mu = \frac{1}{n}\sum_i a_i$（$n$ 是分量个数），让结果零均值，这一步叫中心化。第二步除以标准差 $\sigma = \sqrt{\frac{1}{n}\sum_i (a_i - \mu)^2}$，把尺度固定，这一步叫缩放。之后乘增益 $\mathbf{g}$，再进激活函数。
依据：paper/src-v1/rmsnorm-v1.tex Background

RMSNorm 把第一步整个去掉，只保留第二步，但除的不是标准差而是均方根：$\mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_i a_i^2}$，输出 $\bar{a}_i = a_i / \mathrm{RMS}(\mathbf{a}) \cdot g_i$。标准差量的是各分量到均值的偏离，均方根量的是各分量到 $0$ 的距离。去掉减均值后，输出不再保证零均值。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm

它的一般式还带一个偏置 $\mathbf{b}$：$\mathbf{y} = f\left(\frac{\mathbf{Wx}}{\mathrm{RMS}(\mathbf{a})}\odot\mathbf{g}+\mathbf{b}\right)$，$\odot$ 是逐元素相乘，$f$ 是激活函数，$\mathbf{y}$ 是这一层的输出，$\mathbf{W}$ 是权重矩阵、$\mathbf{x}$ 是输入向量。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm

均方根把归一化后的输出推到半径 $\sqrt{n}$ 的球面上：只除均方根、不乘增益时，输出的欧氏长度恰好是 $\sqrt{n}$。欧氏长度与均方根只差一个因子 $\sqrt{n}$。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm

#### 手算对照

取一个小的加权和向量 $\mathbf{a} = (1, 2, 3, 4)$，分量个数 $n = 4$，分别计算 LayerNorm 和 RMSNorm，比较两者的差别。
依据：自行构造

::: 手算全过程
1. 均值 $\mu = (1+2+3+4)/4 = 2.5$。
2. 各分量到均值的偏离是 $(-1.5, -0.5, 0.5, 1.5)$，方差是 $(1.5^2+0.5^2+0.5^2+1.5^2)/4 = 1.25$，标准差 $\sigma = \sqrt{1.25} \approx 1.118$。
3. LayerNorm 的输出（增益取 $1$、偏置取 $0$）是各偏离除以标准差：$(-1.342, -0.447, 0.447, 1.342)$，它的均值是 $0$。
4. 均方根 $\mathrm{RMS}(\mathbf{a}) = \sqrt{(1+4+9+16)/4} = \sqrt{7.5} \approx 2.739$。
5. RMSNorm 的输出（增益取 $1$）是各分量除以均方根：$(0.365, 0.730, 1.095, 1.461)$。
6. 这组输出的均值是 $(0.365+0.730+1.095+1.461)/4 = 0.913$，不为零。同一个加权和，LayerNorm 先把中心平移到 $0$ 再定尺度，RMSNorm 不平移中心，直接把到原点的距离定下来，所以输出整体落在 $0$ 的同一侧。
依据：自行构造
:::

### 缩放不变性与丢失的中心化不变性

#### 整体缩放不变

把权重矩阵整体乘一个正的常数 $\delta$，输出不变。原因是均方根的齐次性：$\mathrm{RMS}(\delta \mathbf{a}) = \delta\, \mathrm{RMS}(\mathbf{a})$，分子分母里的 $\delta$ 约掉。对输入整体缩放同样成立，逐条样本、整个数据集也成立。这就是缩放不变性：不管输入和权重的整体尺度怎么变，归一化后的输出和原来一样，层激活的尺度因此被固定。
依据：paper/src-v1/rmsnorm-v1.tex Invariance Analysis

::: 整体缩放下输出不变的推导
1. 权重矩阵缩放 $\mathbf{W}' = \delta \mathbf{W}$ 后，加权和变成 $\mathbf{a}' = \delta \mathbf{a}$。
2. 由齐次性，$\mathrm{RMS}(\mathbf{a}') = \delta\, \mathrm{RMS}(\mathbf{a})$。
3. 代回输出式，$\bar{a}_i = \dfrac{a'_i}{\mathrm{RMS}(\mathbf{a}')} g_i = \dfrac{\delta a_i}{\delta\, \mathrm{RMS}(\mathbf{a})} g_i = \dfrac{a_i}{\mathrm{RMS}(\mathbf{a})} g_i$，$\delta$ 约掉。
4. 输入缩放 $\mathbf{x}' = \delta \mathbf{x}$ 的推导同理。
依据：paper/src-v1/rmsnorm-v1.tex Invariance Analysis
:::

#### 逐行缩放与平移会怎样

缩放不变性只在整体缩放下成立。如果只把个别权重行各乘一个不同的常数，均方根的线性关系被打破，输出会变。中心化不给缩放不变性，但给另一种：把加权和整体加一个常数，LayerNorm 的输出不变，这叫中心化不变性。RMSNorm 没有这一步，所以对加性扰动不免疫——加权和整体平移，输出跟着变。实践中去掉中心化质量仍与 LayerNorm 相当，隐藏状态（这一层算出的、传给下一层的向量）里未归一化的均值反而比不做归一化的原模型更稳。
依据：paper/src-v1/rmsnorm-v1.tex Invariance Analysis；paper/src-v1/rmsnorm-v1.tex L95；paper/src-v1/rmsnorm-v1.tex Experiments；paper/src-v1/rmsnorm-v1.tex L353-L354

::: 隐藏状态均值与标准差的实测
在一个逐 token 处理文本的翻译模型（RNNSearch）上，取一个翻译评测集（newstest2013），看解码部分 GRU（一种循环单元）的隐藏状态在各 token（文本切分后的片段）位置的均值（M）与标准差（S），位置 1 到 4 及全部位置的平均（ALL）。
- 无归一化基线：M 为 -2.60、-1.19、-1.43、-1.53，ALL 为 -1.60；S 为 7.35、2.33、2.61、2.73，ALL 为 3.04。
- LayerNorm：M 为 -0.43、-0.48、-0.50、-0.50，ALL 为 -0.51；S 为 1.19、1.51、1.51、1.51，ALL 为 1.51。
- RMSNorm：M 为 -0.40、-0.60、-0.69、-0.74，ALL 为 -0.73；S 为 1.27、1.51、1.50、1.49，ALL 为 1.50。
依据：paper/src-v1/rmsnorm-v1.tex L334-L347
:::

### 梯度上的表现

训练靠梯度下降更新参数，学习率是每次更新里决定步长大小的系数，梯度的大小直接影响参数每次移动多少。对增益 $\mathbf{g}$ 和偏置 $\mathbf{b}$ 的梯度，在输入或权重整体缩放时不变；对增益的梯度正比于归一化后的加权和，而不是原始加权和，这稳住了 $\mathbf{g}$ 的量级。
依据：paper/src-v1/rmsnorm-v1.tex Gradient Analysis

对权重矩阵 $\mathbf{W}$ 的梯度多出一个和输入、权重都相关的矩阵项。这个矩阵项本身随缩放负相关，但代回后，梯度对输入缩放不变，只与权重缩放负相关：权重缩放越大，梯度被压得越小。这个负相关像一个自动调节学习率的机制，把梯度范数控制在合适范围，避免权重矩阵的范数过大。
依据：paper/src-v1/rmsnorm-v1.tex Gradient Analysis

::: 梯度公式与缩放后的行为
1. 记 $\mathbf{v}$ 为激活函数里的一整段 $\frac{\mathbf{Wx}}{\mathrm{RMS}(\mathbf{a})} \odot \mathbf{g} + \mathbf{b}$，$\mathcal{L}$ 是损失函数，$\partial \mathcal{L} / \partial \mathbf{v}$ 是反向传播时从 $\mathcal{L}$ 传回 $\mathbf{v}$ 的梯度。
2. 对偏置的梯度是 $\dfrac{\partial \mathcal{L}}{\partial \mathbf{b}} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{v}}$。
3. 对增益的梯度是 $\dfrac{\partial \mathcal{L}}{\partial \mathbf{g}} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{v}} \odot \dfrac{\mathbf{Wx}}{\mathrm{RMS}(\mathbf{a})}$。
4. 对权重的梯度里含矩阵项 $\mathbf{R} = \dfrac{1}{\mathrm{RMS}(\mathbf{a})}\left(\mathbf{I} - \dfrac{(\mathbf{Wx})(\mathbf{Wx})^T}{n\, \mathrm{RMS}(\mathbf{a})^2}\right)$，$\mathbf{I}$ 是单位矩阵。对权重的梯度 $\dfrac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_i \left[\mathbf{x}^T \otimes \left(\mathrm{diag}\left(\mathbf{g}\odot\dfrac{\partial \mathcal{L}}{\partial \mathbf{v}}\right)\mathbf{R}\right)\right]_i$，$\otimes$ 是克罗内克积。
5. 让输入或权重整体缩放 $\delta$，$\mathbf{R}$ 变成 $\mathbf{R} / \delta$；对输入缩放时，输入因子与 $\mathbf{R}$ 里的 $1/\delta$ 相抵，梯度不变；权重缩放时只留下 $1/\delta$，梯度与缩放负相关。
依据：paper/src-v1/rmsnorm-v1.tex Gradient Analysis
:::

### 省下的计算

LayerNorm 在除标准差之前要多做两件事：求一次均值，再把每个分量减掉它。RMSNorm 没有这两步。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background

速度提升的大小随框架、硬件、网络结构和其他组件的相对开销而变。在多种模型与实现上，RMSNorm 实测比 LayerNorm 快 7%~64%。
依据：paper/src-v1/rmsnorm-v1.tex Experiments；paper/src-v1/rmsnorm-v1.tex Conclusion and Future Work

::: 各模型上的速度数字
前四行时间按每 1000 个训练步计，后两行按每 100 步计；每行末尾的百分比是 RMSNorm 相对 LayerNorm 的加速比。完整区间是 7%~64%，覆盖不同模型与实现，下表只列其中若干项。RNNSearch 是一个循环翻译模型，Transformer 是一个基于自注意力（序列各部分互相加权的一种层）的翻译模型，注意力阅读器是一个阅读理解模型，图像-描述检索是一个跨模态（图像与文本两种模态之间）检索模型；Tensorflow、Theano、PyTorch 是三个深度学习框架。
- RNNSearch（Tensorflow）：LayerNorm 665s，RMSNorm 501s，快 24.7%。
- RNNSearch（Theano）：LayerNorm 988s，RMSNorm 652s，快 34.0%。
- RNNSearch（PyTorch）：LayerNorm 857s，RMSNorm 763s，快 11.0%。
- Transformer：LayerNorm 248s，RMSNorm 231s，快 6.9%。
- 注意力阅读器：LayerNorm 392s，RMSNorm 333s，快 15.1%。
- 图像-描述检索：LayerNorm 12.02s，RMSNorm 7.12s，快 40.8%。
依据：paper/src-v1/rmsnorm-v1.tex L254-L264；paper/src-v1/rmsnorm-v1.tex L288-L305；paper/src-v1/rmsnorm-v1.tex L319-L326；paper/src-v1/rmsnorm-v1.tex L384-L394；paper/src-v1/rmsnorm-v1.tex L454-L464
:::

### 适用条件与边界

当输入的加权和均值恰为 0 时，减不减均值没有差别，RMSNorm 与 LayerNorm 相等，这是它退化为 LayerNorm 的条件。
依据：paper/src-v1/rmsnorm-v1.tex RMSNorm

另一个边界在证据的强度：「只保留缩放就够」是一个假设，不是已证明的结论，靠实验支持。
依据：paper/src-v1/rmsnorm-v1.tex L75；paper/src-v1/rmsnorm-v1.tex L133；paper/src-v1/rmsnorm-v1.tex Experiments
