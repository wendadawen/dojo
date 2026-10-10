# Vision Transformer

- 描述：Vision Transformer 把图像切成固定大小的小块，每块线性映射成一个向量排成序列，交给标准的 Transformer 编码器（一层层在位置之间交换信息的网络）做图像分类。
- 摘要：把 $H\times W$ 的图像切成 $N=HW/P^2$ 个 $P\times P$ 的小块当作 token，用标准 Transformer 编码器处理；在 JFT-300M 这类大数据上预训练后，它在多个图像分类基准上匹配或超过当时最好的卷积网络。
- 主题：模型结构, 注意力机制
- 标签：视觉与多模态

## 资料

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. arXiv:2010.11929v2, 2021. [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

## 短页

### 把图片当成一串小块

Vision Transformer 把一张图像看成由小块组成的序列。小块是按边长 $P$ 切出的像素方块，每块被映成一个 $D$ 维向量，$D$ 是模型各层一直保持的向量长度；一张图像因此变成一串长度等于小块个数的向量。序列里每个位置再加一个可学习的向量（取值由训练确定），叫位置嵌入，把这块在图像里的位置信息带进序列。序列最前面还放一个可学习的向量，叫分类 token，它不对应任何小块，专门用来汇总整张图像的信息。这一串向量交给 Transformer 编码器处理：编码器把自注意力（每个位置按与其他位置的相关度加权取值，见[标准注意力](wiki/standard-attention)）和前馈网络（对每个位置各算各的两层小网络）交替堆叠若干层。
依据：paper/src-v2/03_method.tex L23-L37

### 不靠卷积也能做视觉

在计算机视觉里长期占主导的是卷积网络（用卷积核在图像上逐层滑动、提取局部特征的网络）。注意力此前多与卷积网络结合使用，或替换卷积网络里的部分结构，比如用注意力换掉某些卷积层而卷积网络整体保留；纯用 Transformer 处理图像、不在结构里加进图像专有先验的做法没有被大规模训练起来。Vision Transformer 先在大型数据集上训练一个通用模型（预训练），再迁移到具体任务上继续训练（微调），结果是它能在多个图像分类基准上追平甚至超过当时最好的卷积网络，而预训练用的算力还更少。
依据：paper/src-v2/01_introduction.tex L10-L13；paper/src-v2/01_introduction.tex L23-L25；paper/src-v2/05_conclusion.tex L4-L7

### 切块、嵌入、编码、分类

1. 切块并投影：把图像切成 $P\times P$ 的小块，每块展平成一个向量，乘一个可学习矩阵映到 $D$ 维，得到小块嵌入。
2. 加分类 token 与位置嵌入：在序列最前面拼一个可学习的分类 token，再给每个位置加上可学习的位置嵌入。
3. 过编码器：把这串向量送进 $L$ 层 Transformer 编码器，每层由多头自注意力（把自注意力分成多个头并行做）和前馈网络组成，两处都先做 LayerNorm（把每个向量按自己的均值方差归一）再接残差连接（把子层的输入加回它的输出）。
4. 分类：取分类 token 在最后一层的输出，经分类头（把向量映成各类别分数的层）得到分数。微调时把分类头换成一个线性层（一层矩阵变换），并按更高的分辨率对位置嵌入做二维插值（按每块在原图中的坐标，在原来的位置嵌入上重新取值）。
依据：paper/src-v2/03_method.tex L23-L33；paper/src-v2/03_method.tex L35-L44；paper/src-v2/03_method.tex L59-L67

## 长页

Vision Transformer（ViT）是一种图像分类模型。它把一张图像切成大小相同的小块，每块线性映射成一个向量，排成一串后交给标准的 Transformer 编码器处理，得到整张图像的表示，再经一个分类头（把表示映成各类别分数的层）输出类别分数。图像在这里被当成一串小块，和自然语言处理里的一串词等价，编码器就是处理文本用的标准 Transformer（见[标准注意力](wiki/standard-attention)）。它在大数据集上预训练（先在大量数据上训练一个通用模型）后，在多个图像分类基准上匹配或超过当时最好的卷积网络（用卷积核在图像上逐层滑动、提取局部特征的网络）。
依据：paper/src-v2/03_method.tex L12；paper/src-v2/01_introduction.tex L15-L18；paper/src-v2/01_introduction.tex L23-L27；paper/src-v2/05_conclusion.tex L4-L7；paper/src-v2/03_method.tex L27-L29

### 把图像变成一段序列

#### 分块与线性嵌入

标准 Transformer 的输入是一维的向量序列，图像是二维的。ViT 把 $H\times W$ 的图像按 $P\times P$ 切成不重叠的小块，一共 $N=HW/P^2$ 块：$H$、$W$ 是图像的高和宽，$P$ 是小块边长。把每块展平成一个长度 $P^2\cdot C$ 的向量，$C$ 是通道数（灰度图 $C=1$，RGB 图 $C=3$），再乘一个可学习的矩阵 $\mathbf{E}\in\mathbb{R}^{(P^2\cdot C)\times D}$（取值靠训练得到），把它映到固定的 $D$ 维，这一步的输出叫小块嵌入。$N$ 是小块序列的长度，拼上分类 token（一个专门汇总整张图像的可学习向量）后，送进编码器的是 $N+1$ 个向量。
依据：paper/src-v2/03_method.tex L22-L25；paper/src-v2/03_method.tex L39-L40；paper/src-v2/04_experiments.tex L31-L32

::: 一块 4×4 图像走完切块与投影
取一张 $4\times4$ 的灰度图（$H=W=4$、$C=1$），小块边长 $P=2$，于是块数 $N=4\times4/2^2=4$，每块展平后长度 $P^2\cdot C=4$。投影矩阵取 $D=3$：
依据：自行构造

$$\mathbf{E}=\begin{pmatrix}1&0&1\\0&1&1\\1&1&0\\0&0&1\end{pmatrix}$$

按行切块、展平、乘 $\mathbf{E}$：
依据：自行构造

1. 四块按从左到右、从上到下的顺序，展平为 $(1,2,5,6)$、$(3,4,7,8)$、$(9,10,13,14)$、$(11,12,15,16)$。
2. 第 1 块乘 $\mathbf{E}$：第 1 个分量 $1\cdot1+2\cdot0+5\cdot1+6\cdot0=6$，第 2 个分量 $1\cdot0+2\cdot1+5\cdot1+6\cdot0=7$，第 3 个分量 $1\cdot1+2\cdot1+5\cdot0+6\cdot1=9$，得 $(6,7,9)$。
3. 其余三块同样计算，得第 2 块 $(10,11,15)$、第 3 块 $(22,23,33)$、第 4 块 $(26,27,39)$。
依据：自行构造

```python
E = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]]
img = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
patches = []
for i in (0, 2):
    for j in (0, 2):
        patches.append([img[i][j], img[i][j + 1], img[i + 1][j], img[i + 1][j + 1]])
for k, p in enumerate(patches, 1):
    emb = [sum(p[r] * E[r][c] for r in range(4)) for c in range(3)]
    print("第", k, "块", p, "->", emb)
```

```text
第 1 块 [1, 2, 5, 6] -> [6, 7, 9]
第 2 块 [3, 4, 7, 8] -> [10, 11, 15]
第 3 块 [9, 10, 13, 14] -> [22, 23, 33]
第 4 块 [11, 12, 15, 16] -> [26, 27, 39]
```

四个小块嵌入各是一个 $D=3$ 的向量。
依据：自行构造
:::

#### 分类 token 与位置嵌入

位置嵌入加到小块嵌入上，把每块在图像中的位置信息带进序列。ViT 用的是一维位置嵌入，按小块展平的先后编号；它没有用带二维结构的位置嵌入，因为没有观察到明显收益（位置编码的通用做法见[位置编码](wiki/positional-encoding)）。分类则靠一个专门的可学习向量：在小块嵌入序列的最前面拼一个可学习的 $\mathbf{x}_\text{class}$，叫分类 token，作用与文本模型放在句首的特殊标记相同；取它在编码器最后一层的输出作为整张图像的表示。
依据：paper/src-v2/03_method.tex L27-L33

::: 分类 token 与位置嵌入加进序列
接上例。送进编码器的向量数是小块数加一，即 $N+1=5$。分类 token 取 $(1,1,1)$，位置嵌入取 $5\times3$ 的小矩阵：
依据：自行构造

$$\mathbf{E}_{pos}=\begin{pmatrix}0&0&0\\1&0&0\\0&1&0\\0&0&1\\1&1&0\end{pmatrix}$$

把分类 token 拼在最前，再逐位置加上位置嵌入：
依据：自行构造

1. 第一行是分类 token 加它的位置嵌入：$(1,1,1)+(0,0,0)=(1,1,1)$。
2. 第 1 块：$(6,7,9)+(1,0,0)=(7,7,9)$。
3. 第 2 块：$(10,11,15)+(0,1,0)=(10,12,15)$。
4. 第 3 块：$(22,23,33)+(0,0,1)=(22,23,34)$。
5. 第 4 块：$(26,27,39)+(1,1,0)=(27,28,39)$。
依据：自行构造

```python
emb = [[6, 7, 9], [10, 11, 15], [22, 23, 33], [26, 27, 39]]
x_class = [1, 1, 1]
pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
seq = [x_class] + emb
z0 = [[seq[i][c] + pos[i][c] for c in range(3)] for i in range(5)]
for i, row in enumerate(z0):
    print(i, row)
```

```text
0 [1, 1, 1]
1 [7, 7, 9]
2 [10, 12, 15]
3 [22, 23, 34]
4 [27, 28, 39]
```

得到的 $\mathbf{z}_0$ 是 $5\times3$ 的矩阵，就是送进编码器的那串向量。
依据：自行构造
:::

### 编码器与分类头

编码器沿用标准 Transformer：$L$ 层堆叠，每层两部分，多头自注意力（把自注意力分成多个头并行做，见[标准注意力](wiki/standard-attention)）在位置之间交换信息，前馈网络（MLP，对每个位置各算各的）对每个位置单独处理。两个子层都先把输入做 LayerNorm（按每个向量自己的均值方差归一），再经过子层，输出以残差连接加回输入；MLP 是两层，中间用 GELU 激活（一种平滑的非线性函数）。最后一层的输出再取分类 token 那个位置、做一次 LayerNorm，得到图像表示。分类头把图像表示映到类别分数：预训练用的是带一个隐层（两层 MLP 的中间一层）的 MLP。
依据：paper/src-v2/03_method.tex L35-L44；paper/src-v2/03_method.tex L27-L29

::: 编码器一层的计算式
下标 $\ell$ 是层号，$L$ 是层数，$\mathbf{z}_0$ 是嵌入后的序列，$\mathbf{z}'_\ell$、$\mathbf{z}_\ell$ 是第 $\ell$ 层两个子层的输出；MSA 是多头自注意力，LN 是 LayerNorm，MLP 是前馈网络；$\mathbf{z}_L^0$ 的上标 0 指分类 token 所在的那一行（序列里排第一）。
依据：paper/src-v2/03_method.tex L39-L44

$$\mathbf{z}'_\ell = \mathrm{MSA}(\mathrm{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1},\qquad \ell=1\ldots L$$

$$\mathbf{z}_\ell = \mathrm{MLP}(\mathrm{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell,\qquad \ell=1\ldots L$$

$$\mathbf{y} = \mathrm{LN}(\mathbf{z}_L^0)$$
依据：paper/src-v2/03_method.tex L39-L44
:::

### 与卷积网络差在归纳偏置

卷积网络和 ViT 的差别在归纳偏置，也就是模型在结构里预先假定了数据的哪些性质。卷积网络的每一层都自带三条与图像相符的假定：局部性（每个输出只看输入的一小块邻域）、二维邻域结构（相邻的像素在网格上互为邻居）、平移等变（图像平移后输出跟着平移）。ViT 的结构里这些假定大多被去掉了：只有 MLP 仍是局部的、对平移等变，自注意力在每个位置都能看到全图，是全局的。二维结构只在两处被人工加进模型：开头把图像切成小块，以及微调（在目标任务上继续训练）时按分辨率调整位置嵌入。位置嵌入在初始化时不带任何二维位置信息，小块之间的空间关系要靠训练从数据里学。
依据：paper/src-v2/03_method.tex Inductive bias；paper/src-v2/03_method.tex Fine-tuning and Higher Resolution；paper/src-v2/05_conclusion.tex L4

### 预训练与高分辨率微调

ViT 通常先在一个大数据集上预训练，再迁移到目标任务上微调：预训练是在大量图像上学一个通用模型，微调是把这个模型在某个具体数据集上继续训练。微调时把预训练用的分类头整个去掉，换成一个 $D\times K$ 的线性层（一层矩阵变换）、初始化为零，$K$ 是目标任务的类别数。微调常在比预训练更高的分辨率上做，效果更好。保持小块边长不变，分辨率提高会让块数、也就是小块序列长度变大；编码器能接受任意长度（受硬件内存限制），但预训练好的位置嵌入可能不再有意义，所以对位置嵌入做二维插值：按每块在原图中的坐标，在原来的位置嵌入上重新取值。
依据：paper/src-v2/03_method.tex Fine-tuning and Higher Resolution；paper/src-v2/04_experiments.tex Setup；paper/src-v2/03_method.tex L27-L29

### 数据规模决定成败

ViT 的成败取决于预训练数据量。在中等数据集上从头训练、不加强正则（防止模型过拟合的约束）时，它的准确率比同规模卷积网络低几个百分点；这是因为它缺少卷积网络的归纳偏置，数据不够时不容易泛化（在新数据上表现好）。数据量增大后结论相反：用随机抽出的 9M、30M、90M 子集做实验，ViT 在小数据上比同算力的卷积网络更容易过拟合（在训练数据上表现持续变好、在新数据上变差），但在 90M 以上的子集上反超；数据量足够大时，大规模训练弥补了归纳偏置的缺失，ViT 在多个图像识别基准上接近或超过当时最好的卷积网络。
依据：paper/src-v2/01_introduction.tex L20-L27；paper/src-v2/04_experiments.tex Pre-training Data Requirements；paper/src-v2/04_experiments.tex Comparison to State of the Art

::: 预训练数据量越大，同一个模型的准确率越高
同一个模型（ViT-B/16，其中 /16 指小块边长 16 像素）在不同大小的预训练数据上，ImageNet（一个大规模图像分类数据集）准确率随数据量上升：1.3M 张时 77.91，14M 张时 83.97，300M 张时 84.15。
依据：paper/src-v2/06_appendix.tex L110-L131；paper/src-v2/04_experiments.tex Setup
:::

### 计算代价与规模权衡

固定预训练数据后比较不同规模，ViT 在同等性能下用的预训练算力比卷积网络少大约 2 到 4 倍（在五个数据集上平均），在实验的算力范围内还没有看到性能饱和。改变小块大小会同时改变序列长度与算力：用更小的小块（序列更长）稳健地提升性能，但不增加参数，改用更小的小块算力代价更高。
依据：paper/src-v2/04_experiments.tex Scaling Study；paper/src-v2/04_experiments.tex Comparison to State of the Art；paper/src-v2/06_appendix.tex L138-L171；paper/src-v2/06_appendix.tex L227

::: ViT 三个规模的配置与预训练算力
层数、隐藏维度 $D$、MLP 维度、注意力头数和参数量；最后一列是在大规模数据上预训练的算力，单位 exaFLOPs（$10^{18}$ 次浮点运算）。
依据：paper/src-v2/04_experiments.tex Setup；paper/src-v2/06_appendix.tex L138-L171

| 模型 | 层数 | D | MLP | 头数 | 参数 | 预训练算力 |
|---|---|---|---|---|---|---|
| ViT-B/16 | 12 | 768 | 3072 | 12 | 86M | 224 |
| ViT-L/16 | 24 | 1024 | 4096 | 16 | 307M | 783 |
| ViT-H/14 | 32 | 1280 | 5120 | 16 | 632M | 4262 |
依据：paper/src-v2/04_experiments.tex Setup；paper/src-v2/06_appendix.tex L138-L171
:::

### 适用条件与边界

ViT 成立的前提是有大规模预训练数据：数据不够时它缺少卷积网络的归纳偏置，表现不如同规模的卷积网络；从头训练时还需要较强的正则。它的代价是要先在大数据上预训练，才能追赶或超过卷积网络。两种容易出现的误解：一是以为 ViT 把图像的二维结构建进了模型，实际上二维结构只在切块和微调调分辨率两处注入；二是以为 ViT 在任何数据量下都胜过卷积网络，实际上在中等数据集上它更差，只有数据量足够大时才反超。
依据：paper/src-v2/04_experiments.tex Setup；paper/src-v2/01_introduction.tex L20-L25；paper/src-v2/04_experiments.tex Pre-training Data Requirements；paper/src-v2/03_method.tex Inductive bias；paper/src-v2/03_method.tex Fine-tuning and Higher Resolution
