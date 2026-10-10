# 位置编码

- 描述：位置编码是一段只随位置变化的向量，逐元素加到每个 token（文本切分后的基本单位）的输入嵌入（按 ID 查表得到的向量）上，告诉模型它在序列里的位置；它用固定频率的正弦和余弦生成。
- 摘要：输入嵌入按内容查表得到，本身不带先后，模型里也没有别的东西带来次序，所以要把一段只随位置变的编码加进输入。位置编码用不同频率的正弦和余弦写成，每一对维度一个频率：$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})$，位置 $pos$ 是序列里的下标。
- 主题：模型结构
- 标签：位置编码

## 资料

- Ashish Vaswani, Noam Shazeer, Niki Parmar 等. Attention Is All You Need. NeurIPS 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- TensorFlow. tensor2tensor: 论文正文写明训练与评测所用代码在此仓库. [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## 短页

### 一段随位置变的向量

位置编码是一段只随位置变化的向量：排在前面的 token 和排在后面的 token 得到的编码不同，同一位置不论上面是哪个 token，得到的都一样。token 是文本切分后的基本单位；输入嵌入是按 token 的整数 ID 查一张表得到的向量，位置编码逐元素加到它上面。位置编码与嵌入维数相同，两者相加的结果进入模型的第一层，模型据此知道每个 token 在序列里的位置。
依据：paper/src-v7/model_architecture.tex Positional Encoding

### 注意力看不出先后

注意力对每个位置算一个输出：用该位置的查询去和各个位置的键算权重，再按权重把各位置的值加权求和（见[标准注意力](wiki/standard-attention)）。这个结果与次序无关：把内容相同、排列不同的序列送进去，每个位置算出的表示不随次序改变。模型里又没有循环或卷积这类顺着序列处理的结构，次序没有别的来源。
依据：paper/src-v7/model_architecture.tex Attention；paper/src-v7/model_architecture.tex Positional Encoding

### 一对维度一个频率的正弦

位置编码用固定公式算出，不需要训练。位置 $pos$ 的每一对维度对应一个频率：偶数维取 $\sin(pos/10000^{2i/d_{\text{model}}})$，奇数维取配对的 $\cos(pos/10000^{2i/d_{\text{model}}})$，$d_{\text{model}}$ 是嵌入维度，$i$ 是这对维度的编号。靠前的维度频率高、随位置变化快，靠后的维度频率低、变化慢，波长按几何级数从 $2\pi$ 排到接近 $10000\cdot 2\pi$。
依据：paper/src-v7/model_architecture.tex Positional Encoding

## 长页

一个序列先按分词切成若干 token（文本切分后的基本单位），每个 token 用它在该词表里的整数 ID 去查一张嵌入表，取回一个向量，这就是该位置的输入嵌入。层里主要是自注意力：每个位置用它自己的查询去和所有位置的键做匹配，算出权重，再按权重把各位置的值加权求和，得到这个位置的新表示（见[标准注意力](wiki/standard-attention)）。位置编码是在输入嵌入之上再逐元素加一段向量，这段向量只由位置决定，与 token 是什么无关。它补的是模型缺的次序信息：注意力本身只看内容的匹配，序列的先后没有别的来源。位置编码用一组固定频率的正弦和余弦算出，不带任何可训练参数（可训练参数指随训练一起学出来的参数）。
依据：paper/src-v7/model_architecture.tex Positional Encoding；paper/src-v7/model_architecture.tex Attention

### 注意力看不出 token 的先后

注意力的式子里只出现查询、键、值之间的匹配，不出现位置本身；把同一批内容换一个次序排列，每个位置算出的表示不变。仅看注意力这一层的输出，分不出哪个 token 在前、哪个在后。
依据：paper/src-v7/model_architecture.tex Attention

这个模型里既没有循环也没有卷积（两类沿着序列逐段处理的层），序列的先后没有别的来源。次序要进到表示里，只能另外把位置信息注入输入。
依据：paper/src-v7/model_architecture.tex Positional Encoding

### 加在输入嵌入上

位置编码加到模型最底层的输入嵌入上，方式是逐元素相加：位置编码与嵌入维数相同、形状一致，对应分量直接相加，相加的结果进入第一层。位置编码本身只由两个量决定：这个位置在序列里的下标，和它在这一段向量里的维度下标；所以一段固定长度的位置编码对所有样本通用，输入换一条，位置编码不换。
依据：paper/src-v7/model_architecture.tex Positional Encoding

### 正弦式位置编码

位置编码用一个固定公式生成，每一对维度对应一个不同频率的正弦波：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

$pos$ 是位置在序列里的下标，$i$ 是这一对维度在 $0,1,\dots,d_{\text{model}}/2-1$ 里的编号，$d_{\text{model}}$ 是嵌入维度。偶数维取正弦、奇数维取余弦，每一个维度都是一个正弦波。各维的波长成一个几何级数，最短的一维是 $2\pi$，最长的一维接近 $10000\cdot 2\pi$：靠前的维度频率高，随位置转得快；靠后的维度频率低，转得慢。
依据：paper/src-v7/model_architecture.tex Positional Encoding

固定偏移 $k$ 时，$PE_{pos+k}$ 都能写成 $PE_{pos}$ 的线性函数。这样选是出于一个假设：它可能让模型容易学会按相对位置（两个位置之间的距离）关注。
依据：paper/src-v7/model_architecture.tex Positional Encoding

::: 固定偏移为什么能写成线性函数

1. 把每一对维度 $i$ 的频率记作 $\omega_i = 1/10000^{2i/d_{\text{model}}}$，位置编码可以写成 $PE_{(pos,2i)} = \sin(pos\,\omega_i)$、$PE_{(pos,2i+1)} = \cos(pos\,\omega_i)$：位置 $pos$ 乘上频率 $\omega_i$ 当作一个角度。
依据：自行构造
2. 位置 $pos+k$ 的角度比 $pos$ 多出 $k\omega_i$。用两角和公式展开正弦：$\sin((pos+k)\omega_i) = \sin(pos\,\omega_i)\cos(k\omega_i) + \cos(pos\,\omega_i)\sin(k\omega_i)$。
依据：自行构造
3. 同样展开余弦：$\cos((pos+k)\omega_i) = \cos(pos\,\omega_i)\cos(k\omega_i) - \sin(pos\,\omega_i)\sin(k\omega_i)$。
依据：自行构造
4. 把两式写成矩阵：$\begin{pmatrix} PE_{(pos+k,2i)} \\ PE_{(pos+k,2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} PE_{(pos,2i)} \\ PE_{(pos,2i+1)} \end{pmatrix}$。
依据：自行构造
5. 等号右边的矩阵只由偏移 $k$ 和这一对的频率 $\omega_i$ 决定，与 $pos$ 无关。所以任意两个位置的编码之间差一个固定的线性变换。
依据：自行构造
:::

::: 四位向量走一遍

取嵌入维度 $d_{\text{model}} = 4$，看位置 0、1、2 的编码。维度 0、1 是 $i=0$ 的一对，$10000^{0}=1$，波长是 $2\pi$；维度 2、3 是 $i=1$ 的一对，$10000^{2/4}=100$，波长是 $2\pi\cdot 100$。

1. 位置 0：角度全是 0，$PE_{(0,0)}=\sin 0=0$、$PE_{(0,1)}=\cos 0=1$、$PE_{(0,2)}=\sin 0=0$、$PE_{(0,3)}=\cos 0=1$，编码是 $(0,\,1,\,0,\,1)$。
依据：自行构造
2. 位置 1：维度 0、1 的角度是 1 弧度，$PE_{(1,0)}=\sin 1 \approx 0.84$、$PE_{(1,1)}=\cos 1 \approx 0.54$；维度 2、3 的角度是 $1/100=0.01$，$PE_{(1,2)}=\sin 0.01 \approx 0.01$、$PE_{(1,3)}=\cos 0.01 \approx 1.00$。
依据：自行构造
3. 位置 2：$PE_{(2,0)}=\sin 2 \approx 0.91$、$PE_{(2,1)}=\cos 2 \approx -0.42$，$PE_{(2,2)}=\sin 0.02 \approx 0.02$、$PE_{(2,3)}=\cos 0.02 \approx 1.00$。
依据：自行构造
4. 取两对维度的正弦分量看快慢：维度 0 从 0 到 0.84 再到 0.91，动得快，相邻的位置在这里就能分开；维度 2 从 0 到 0.01 再到 0.02，动得慢，要隔很远才拉开差别。
依据：自行构造
:::

### 适用条件与边界

要注入的位置信息可以针对相对位置，也可以针对每个位置自己的下标（绝对位置）。正弦式给出的是一段固定的绝对位置向量：每个位置一段，不含可训练参数，所有序列共用。
依据：paper/src-v7/model_architecture.tex Positional Encoding

另一种做法是把每个位置的向量当成可训练参数，随模型一起学出来。两种做法的结果几乎一样；选正弦式，是因为固定的形式可能让模型外推到训练时没见过的更长序列。
依据：paper/src-v7/model_architecture.tex L155；paper/src-v7/results.tex L107
