# 标准注意力

- 描述：标准注意力（缩放点积注意力）让每个位置用查询向量与所有位置的键向量点积得到相关度分数，经 SoftMax 归一成权重后对值向量加权求和，得到该位置的输出表示。
- 摘要：每个位置用查询去匹配所有位置的键，按匹配程度对值向量加权求和，一次算出整个序列的表示；除以 $\sqrt{d_k}$ 让分数幅度不随维度增大，多头在多个子空间并行做同一运算再拼接。
- 主题：注意力机制
- 标签：注意力

## 资料

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. Attention Is All You Need. NeurIPS 2017. arXiv:1706.03762v7, 2023. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- TensorFlow. tensor2tensor: the code used to train and evaluate the Transformer. 2023. [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## 短页

### 按相关度从其他位置取值

序列由 token 组成，token 是序列里被处理的最小单位，可以是一个词或词的一部分；模型把每个 token 变成一个向量，叫嵌入。标准注意力给序列里每个位置算出三个向量：查询、键、值。查询代表这个位置要找什么，键代表每个位置能提供什么，值代表每个位置实际给出的内容。一个位置用它的查询去和所有位置的键比较，按比较的结果决定从各个位置的值里各取多少信息，取回的信息合起来就是该位置的新表示。查询和键的向量维度相同，记作 $d_k$。
依据：paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L139-L140

一个位置的查询、键、值都由同一个序列的向量算出，这种把注意力用在序列内部的做法叫自注意力。
依据：paper/src-v7/background.tex L3；paper/src-v7/model_architecture.tex L107

### 为什么不用循环

循环模型按位置顺序一步步算，每个位置依赖前一个位置的结果，远距离的两个位置之间信息要经过中间每一个位置，路径长，依赖难学。
依据：paper/src-v7/introduction.tex L3；paper/src-v7/why_self_attention.tex L8

注意力没有这个限制：任意两个位置用它们的查询和键直接比较，路径长度与序列长度无关；所有位置的比较和取信息可以同时进行。序列不太长、位置数小于向量维度时，注意力每层的计算量比循环模型更小。
依据：paper/src-v7/why_self_attention.tex L26；paper/src-v7/why_self_attention.tex L84；paper/src-v7/why_self_attention.tex L85

### 一次算完整个序列

把 $n$ 个位置的嵌入向量按行排成矩阵 $X$，各自乘三个可学矩阵，得到查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$。计算分三步：相关度，用 $Q$ 乘 $K$ 的转置，得到 $n\times n$ 的分数矩阵，第 $i$ 行是位置 $i$ 与所有位置的相关度；缩放与归一，分数除以 $\sqrt{d_k}$，再对每一行做 SoftMax（每个分数取指数后除以该行指数之和），得到每行和都为 $1$ 的权重；加权求和，用权重矩阵乘 $V$，第 $i$ 行就是位置 $i$ 的输出。
依据：paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L38；paper/src-v7/model_architecture.tex L40-L42；paper/src-v7/model_architecture.tex L82

这三步都是矩阵运算，一次算出所有位置的输出，写成 $\text{softmax}(QK^\top/\sqrt{d_k})V$。
依据：paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L40-L42

## 长页

序列建模要处理一串 token，token 是序列里被处理的最小单位，可以是一个词或词的一部分。模型把每个 token 变成一个向量，叫嵌入；输入的整串 token 因此变成一系列向量。生成第 $t$ 个 token 时，模型只看得到它前面已经生成的 token，这一性质叫自回归。序列里每个位置要根据其他位置的信息算出自己的新表示，标准注意力（缩放点积注意力）就是做这件事的方法。它给每个位置算出三个向量：查询、键、值。查询代表这个位置要找什么，键代表每个位置能提供什么，值代表每个位置实际给出的内容。一个位置的查询与所有位置的键逐一做点积，得到一组相关度分数；分数归一成权重；再按权重对所有位置的值加权求和，所得向量就是该位置的新表示。
依据：paper/src-v7/model_architecture.tex L12；paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L139-L140

循环模型按位置顺序计算，每个位置的状态建立在前一个位置的结果上，远距离的两个位置之间信息要经过中间每一个位置，路径长，依赖难学。注意力让任意两个位置直接比较、直接取信息，路径长度与序列长度无关，所有位置可以同时算。
依据：paper/src-v7/introduction.tex L3；paper/src-v7/why_self_attention.tex L3；paper/src-v7/why_self_attention.tex L8；paper/src-v7/why_self_attention.tex L84

### 查询、键、值：把序列变成可匹配的向量

每个位置的向量先经过三个可学的投影矩阵，得到查询、键、值三个向量。设位置向量的维度是 $d_{model}$，查询和键的维度是 $d_k$，值的维度是 $d_v$。把 $n$ 个位置的向量按行排成矩阵 $X$，三个投影矩阵记作 $W^Q$、$W^K$、$W^V$，则 $Q=XW^Q$、$K=XW^K$、$V=XW^V$，其中 $Q$、$K$、$V$ 的每一行是相应位置的查询、键、值向量。
依据：paper/src-v7/model_architecture.tex L38；paper/src-v7/model_architecture.tex L82；paper/src-v7/model_architecture.tex L90-L93

标准注意力用在同一个序列内部时，查询、键、值都由这个序列自身的向量经过上面三个投影得到，三者同源，这种用法叫自注意力。
依据：paper/src-v7/model_architecture.tex L107；paper/src-v7/background.tex L3

投影矩阵的形状由维度决定：$W^Q$ 和 $W^K$ 是 $d_{model}\times d_k$ 的矩阵，$W^V$ 是 $d_{model}\times d_v$ 的矩阵。
依据：paper/src-v7/model_architecture.tex L93

### 相关度：查询与键的点积

一个位置与另一个位置的相关程度，用它们的查询向量和键向量的点积衡量：把两个向量逐分量相乘再求和，得到一个数，叫分数。分数大表示两者相关。把所有位置的查询、键分别排成矩阵 $Q$、$K$，各行是各位置的查询、键向量。$Q$ 的第 $i$ 行与 $K$ 的第 $j$ 行点积，就是位置 $i$ 与位置 $j$ 的分数；这些分数排成矩阵 $S=QK^\top$，$S$ 的第 $i$ 行是位置 $i$ 与所有位置的相关度。一次矩阵乘法就算完所有位置两两之间的分数。
依据：paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L38；paper/src-v7/model_architecture.tex L40-L42

相关度也可以用别的函数算，不一定用点积。用点积的好处是它可以用高度优化的矩阵乘法实现，在计算时间和空间上都更省。
依据：paper/src-v7/model_architecture.tex L44

### SoftMax 与加权求和

要把一组分数变成一组和为 $1$ 的非负数，用 SoftMax：对每个分数取指数，再除以这些指数之和。指数函数把任意实数变成正数，除以和之后，一组权重之和等于 $1$，可以看成一个分布，数越大分到的权重越多。归一化逐个位置做：位置 $i$ 的权重由分数矩阵的第 $i$ 行算出，这一行权重之和为 $1$。
依据：paper/src-v7/model_architecture.tex L24-L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L41

得到权重后，位置 $i$ 的输出是权重对各位置值向量的加权求和：所有位置的输出一起算，就是把 $QK^\top$ 除以 $\sqrt{d_k}$ 后，逐行做 SoftMax 得到权重矩阵，再乘值矩阵 $V$。
依据：paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L41

自回归生成时，位置 $i$ 不能看到它后面的位置。做法是在 SoftMax 之前，把位置 $i$ 与它后面位置之间的那些分数设成 $-\infty$：取指数后变成 $0$，这些位置分到的权重就是 $0$，位置 $i$ 只从它自己和它前面的位置取值。
依据：paper/src-v7/model_architecture.tex L109

::: SoftMax 的定义
给一组分数 $x_1,\dots,x_m$，第 $i$ 个的 SoftMax 值是
$$\text{SoftMax}(x)_i=\frac{e^{x_i}}{\sum_{j=1}^{m}e^{x_j}}$$
分母是所有分数的指数之和，分子是第 $i$ 个分数的指数。
依据：自行构造
:::

### 缩放：为什么除以 $\sqrt{d_k}$

直接用点积分数做 SoftMax，当 $d_k$ 较大时会有问题。设查询和键的各分量是均值 $0$、方差 $1$ 的独立随机变量，则它们的点积（$d_k$ 项之和）均值 $0$、方差 $d_k$：$d_k$ 越大，分数的绝对值就越可能大。一个推测是，分数过大会把 SoftMax 推到梯度极小的区域，训练变难；为此把分数除以 $\sqrt{d_k}$，让方差回到 $1$ 附近。
依据：paper/src-v7/model_architecture.tex L53

完整的公式是 $\text{softmax}(QK^\top/\sqrt{d_k})V$：先算分数矩阵 $QK^\top$，除以 $\sqrt{d_k}$，再逐行做 SoftMax 得到权重，最后乘值矩阵 $V$。
依据：paper/src-v7/model_architecture.tex L41；paper/src-v7/model_architecture.tex L53

::: 除以 $\sqrt{d_k}$ 的方差推导
设 $q,k$ 是 $d_k$ 维向量，各分量 $q_1,\dots,q_{d_k}$ 和 $k_1,\dots,k_{d_k}$ 相互独立，每个的均值都是 $0$、方差都是 $1$。
依据：paper/src-v7/model_architecture.tex L53

1. 点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$，是 $d_k$ 项之和。
依据：paper/src-v7/model_architecture.tex L53

2. 每一项 $q_ik_i$ 的均值是 $0$（$q_i$、$k_i$ 均值都是 $0$ 且独立），方差是 $E[q_i^2k_i^2]=E[q_i^2]E[k_i^2]=1$，所以每一项均值 $0$、方差 $1$。
依据：自行构造

3. $d_k$ 个独立的项相加，和的均值是 $0$，方差是 $d_k$ 个 $1$ 之和，即 $d_k$。
依据：自行构造

4. 标准差是方差的平方根，为 $\sqrt{d_k}$。把点积除以 $\sqrt{d_k}$，就直接抵消了这个标准差，方差变回 $1$。
依据：自行构造
:::

### 多头：多个子空间并行

单个注意力头把各位置的信息加权求和到一处。多头注意力并行做 $h$ 次注意力：每次用一套自己的投影矩阵，把输入投影到维数较小的子空间，各自算一遍注意力，得到 $h$ 个输出，再拼接起来，用一个输出投影矩阵投回 $d_{model}$ 维。这样模型可以在不同的表示子空间、不同的位置上同时取信息。
依据：paper/src-v7/model_architecture.tex L82；paper/src-v7/model_architecture.tex L83；paper/src-v7/model_architecture.tex L85

取 $h=8$ 个头，每个头里 $d_k=d_v=d_{model}/h=64$。每个头的维数缩小后，多个头加起来的计算量与一个全维的单头相当。
依据：paper/src-v7/model_architecture.tex L98；paper/src-v7/model_architecture.tex L99

::: 多头注意力的公式与投影矩阵
$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$
$$\text{其中}\quad \text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$
依据：paper/src-v7/model_architecture.tex L87-L91

投影矩阵的形状：$W_i^Q$ 和 $W_i^K$ 是 $d_{model}\times d_k$，$W_i^V$ 是 $d_{model}\times d_v$，输出投影 $W^O$ 是 $hd_v\times d_{model}$。
依据：paper/src-v7/model_architecture.tex L93
:::

### 一个可以算完的例子

用一个三位置的小例子说明：与查询最相关的位置得到的权重最大，其余按相关度递减；每个位置只从权重大的位置取到信息。
依据：自行构造

::: 例子的全过程
取三个位置的向量（位置向量维度 $d_{model}=2$）。这个例子不加自回归遮罩，每个位置都能看到包括自己在内的所有位置。
依据：自行构造

- 位置 $0$：$x_0=(1,0)$
- 位置 $1$：$x_1=(0,1)$
- 位置 $2$：$x_2=(1,1)$
依据：自行构造

为手算方便，取三个投影矩阵为单位矩阵，于是 $Q=K=V=X$，$Q$、$K$、$V$ 的第 $i$ 行分别是 $x_i$。
依据：自行构造

1. 相关度：$S_{ij}=x_i\cdot x_j$，得分数矩阵
   依据：自行构造

   $$S=QK^\top=\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$$

2. 缩放：除以 $\sqrt{d_k}=\sqrt{2}\approx1.414$，得
   依据：自行构造

   $$S/\sqrt{d_k}=\begin{pmatrix}0.707&0&0.707\\0&0.707&0.707\\0.707&0.707&1.414\end{pmatrix}$$

3. SoftMax（逐行）：第 $0$ 行 $(0.707,0,0.707)$ 取指数得 $(2.028,1,2.028)$，和为 $5.056$，权重为 $(0.401,0.198,0.401)$；第 $1$ 行 $(0,0.707,0.707)$ 得 $(0.198,0.401,0.401)$；第 $2$ 行 $(0.707,0.707,1.414)$ 取指数得 $(2.028,2.028,4.113)$，和为 $8.169$，权重为 $(0.248,0.248,0.503)$。
   依据：自行构造

4. 加权求和：第 $0$ 行的输出是
   依据：自行构造

   $$0.401\,x_0+0.198\,x_1+0.401\,x_2=(0.802,0.599)$$

   第 $0$ 个位置与位置 $0$、位置 $2$ 的相关度都是 $1$，两处权重相等；与位置 $1$ 的相关度是 $0$，权重最小。
   依据：自行构造
:::

### 适用条件与边界

缩放的理由建立在一个假设上：查询和键的各分量近似独立、均值 $0$、方差 $1$。这个假设不成立时，除以 $\sqrt{d_k}$ 的推导不适用。
依据：paper/src-v7/model_architecture.tex L53

注意力本身不区分位置：分数由查询向量和键向量的内容算出，同一对内容出现在不同位置，分数相同，模型单靠它用不上顺序；要让模型用上顺序，得在输入里另加位置信息。
依据：paper/src-v7/model_architecture.tex L143-L144

代价是计算量随序列长度的平方增长：$n$ 个位置两两比较，分数矩阵有 $n$ 行 $n$ 列，自注意力每层的计算量是 $O(n^2d)$，$d$ 是向量维度（就是前面各位置向量的维度 $d_{model}$）。
依据：paper/src-v7/why_self_attention.tex L14；paper/src-v7/why_self_attention.tex L26

多头不增加计算量：每个头的维数取 $d_{model}/h$，$h$ 个头加起来的计算量与一个全维单头相当；多头增加的是并行的表示子空间，不是计算量。
依据：paper/src-v7/model_architecture.tex L98；paper/src-v7/model_architecture.tex L99
