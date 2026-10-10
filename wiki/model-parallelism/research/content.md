# 模型并行

- 描述：模型并行把单个神经网络的权重切分到多块加速器上，由这些加速器协作完成一次计算，两条基本路径是按层内矩阵维度切的张量并行和按层堆叠深度切的流水线并行。
- 摘要：模型参数大到一块卡放不下时，按层内矩阵维度切（TP）或按层堆叠深度切（PP）。TP 每层通信、要求高带宽互联；PP 只在分区边界传激活、但有气泡。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 资料

- Mohammad Shoeybi, Mostofa Patwary, Raul Puri 等. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053v4, 2020. [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- Yanping Huang, Youlong Cheng, Ankur Bapna 等. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. arXiv:1811.06965, 2019. [https://arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965)

## 短页

### 把权重分到多块卡上

神经网络是一层接一层叠起来的计算结构，权重是其中可训练的参数，加速器（下面简称卡）是用来做这些计算的处理器。

模型并行（model parallelism）把单个神经网络的权重切分到多块卡上，每块卡只持有一部分权重，所有卡合起来才是一个完整的模型。它和数据并行不同：数据并行是把同一份完整模型复制到每块卡上，各卡处理不同的数据。

张量并行（TP）按层内矩阵的维度切，每块卡持有每一层的一部分权重、所有卡参与每一层的计算；流水线并行（PP）按层堆叠的深度切，每块卡持有连续的若干整层、数据依次流过。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 一块卡装不下只好切开算

模型训练时，显存（卡的存储空间）里不只有权重，还有优化器状态——训练时为每个参数额外存下的一份量，比如动量。模型大到权重和优化器状态超出单卡显存时，这时可以把权重分到多块卡上。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction

### 按维度切或按层切

张量并行把一层里的权重矩阵按维度切到多块卡上，切完把各卡的部分结果做一次 all-reduce——把所有卡上的张量（多维数组，矩阵是它的二维情形）按元素相加、结果每块卡都拿到一份。流水线并行按层切出若干分区，把一批样本拆成更小的批次，让不同分区同时处理不同的批次。两条切法正交，可以乘起来一起用。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L127-L128；paper/megatron-src-v1/discussion.tex L5

## 长页

神经网络是一层接一层叠起来的计算结构，权重是其中可训练的参数，加速器（下面简称卡）是用来做这些计算的处理器。模型并行（model parallelism）把单个神经网络的权重切分到多块加速器上，由这些加速器协作完成一次前向（逐层算出中间结果，直到得到输出）或反向（逐层算梯度）计算，每块卡只持有整个模型的一部分权重。它和数据并行不同：数据并行是把同一份完整模型复制到每块卡上、各卡处理不同的数据。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning

切分按层内矩阵的维度做，或按层堆叠的深度做。前者叫张量并行（tensor parallelism，TP），每块卡持有每一层的一部分权重、所有卡参与每一层的计算；后者叫流水线并行（pipeline parallelism，PP），每块卡持有连续的若干整层、数据依次流过。
依据：paper/megatron-src-v1/intro.tex Introduction；paper/gpipe-src-v1/main.tex L127-L128

一块前馈子网络也叫多层感知机（MLP），含两个矩阵乘，中间夹一个逐元素的非线性函数 GeLU；多头注意力指同时算多组注意力、每组叫一个头，它的结构见[标准注意力](wiki/standard-attention)；切分后要合并各卡的部分结果，这个操作是 all-reduce——把所有卡上的张量（多维数组，矩阵是它的二维情形）按元素相加、结果每块卡都拿到一份。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 张量并行：把一层里的矩阵切到多卡

一个 Transformer 层（Transformer 是一类由这种层堆叠而成的模型结构）由两块组成：一块多头注意力，一块两层的前馈（MLP）。张量并行把这两块各自的权重矩阵切开，让所有卡一起算同一层。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### MLP 的列切与行切

MLP 的第一段是一个矩阵乘再跟一个 GeLU 非线性：$Y=\mathrm{GeLU}(XA)$，$X$ 是这一层的输入，$A$ 是第一段的权重矩阵。要把这一个矩阵乘切到多卡上，可以从 $A$ 的两个方向切。

一种切法是按 $X$ 的列和 $A$ 的行同时切：把 $A$ 按行分成上下两块 $A^{(1)}$、$A^{(2)}$，把 $X$ 按列分成左右两块 $X^{(1)}$、$X^{(2)}$，每块卡算 $X^{(i)}A^{(i)}$，再把两块的部分和加起来才做 GeLU。因为 GeLU 是非线性函数，$\mathrm{GeLU}(X^{(1)}A^{(1)}+X^{(2)}A^{(2)})$ 不等于 $\mathrm{GeLU}(X^{(1)}A^{(1)})$ 加 $\mathrm{GeLU}(X^{(2)}A^{(2)})$，所以这个加法必须发生在 GeLU 之前，也就是在非线性激活之前要先同步一次（激活指层算出的、要传给下一层的中间结果张量）。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

另一种切法是只按 $A$ 的列切：$A$ 分成左右两块 $[A_1,A_2]$，每块卡拿同一个完整输入 $X$ 算 $XA_i$，各卡得到的是输出 $Y$ 的一半 $[Y_1,Y_2]=[\mathrm{GeLU}(XA_1),\mathrm{GeLU}(XA_2)]$。GeLU 是逐元素函数，可以各自独立地施加到每块卡自己算出的那一半上，激活之前不需要任何同步。这一种因此更省一次通信，是第一段采用的切法。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

第二段的权重矩阵记作 $B$，它把激活后的结果再映射回输出维度。$B$ 按行切成 $B_1$、$B_2$，每块卡用自己那半激活 $Y_i$ 算 $Y_iB_i$。每块卡算出的都是一个完整输出的一部分，两块相加才是最终输出，这一步用一次 all-reduce 完成，发生在第二段之后。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### 小矩阵走一遍切分

用一个最小的例子看这两种切法为什么拼起来还是对的：输入取 4 维、两段的权重都取 $4\times4$、切到两块卡，手算一遍完整计算和切开后的计算，结果相同。
依据：自行构造

::: 4 维输入、两卡切分的手算
1. 输入取 $X=[1,0,-1,2]$，第一段权重按列切成左右两块，每块 $4\times2$；第二段权重按行切成上下两块，每块 $2\times4$。
依据：自行构造
2. 卡 1 算第一段前半 $XA_1=[3,-1]$，卡 2 算后半 $XA_2=[-1,3]$；这里 $A_1$ 取 $\begin{bmatrix}1&0\\1&1\\0&1\\1&0\end{bmatrix}$，$A_2$ 取 $\begin{bmatrix}0&1\\1&0\\1&0\\0&1\end{bmatrix}$。
依据：自行构造
3. 逐元素做 GeLU：$\mathrm{GeLU}(3)\approx2.996$，$\mathrm{GeLU}(-1)\approx-0.159$。卡 1 得 $[2.996,-0.159]$，卡 2 得 $[-0.159,2.996]$。
依据：自行构造
4. 卡 1 用第二段上块 $B_1=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}$ 算得 $[2.996,-0.159,0,0]$，卡 2 用下块 $B_2=\begin{bmatrix}0&0&1&0\\0&0&0&1\end{bmatrix}$ 算得 $[0,0,-0.159,2.996]$。
依据：自行构造
5. 两块按元素相加，得输出 $[2.996,-0.159,-0.159,2.996]$。不切分时，两段权重拼成完整矩阵一次算完，得到同一个向量，两边相等。
依据：自行构造
:::

#### 注意力按 head 切

注意力块里，输入先乘三个权重矩阵分别得到查询、键、值（记作 Q、K、V）——注意力用它们算出每个位置对其它位置的权重。这三处的矩阵乘按列切：每个头各自有一组 Q、K、V 的权重，按列切正好使一个头的全部计算落在同一块卡上，各卡算各的头，不需要为完成注意力本身通信。注意力之后还有一个把各头输出拼回隐维度的线性层，隐维度就是一个位置向量的宽度；这个线性层按行切，直接承接各卡本地那部分注意力输出，本步不需要通信，它的输出再经一次 all-reduce 相加才是这一层注意力的完整结果。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### 一层两次 all-reduce，反向对称

前向的 all-reduce 与反向的 all-reduce 是一对相反算子：前者在前向把各卡张量全部归约、在反向直接放过，后者在前向放过、在反向才归约。这对算子互为反向，可以一起接进自动求导——框架自动算梯度的机制，不必手写通信。整个层前向一共两次 all-reduce，反向也是两次。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 流水线并行：把整摞层切开

模型由许多层堆叠而成。流水线并行按层叠的顺序把这些层切成连续的若干分区，每块卡只负责其中的一段，数据像流水线一样依次流过各段。
依据：paper/gpipe-src-v1/main.tex L136；paper/gpipe-src-v1/main.tex L141

#### 按层切、micro-batch 流水

训练时一次参数更新用的整批样本叫 mini-batch。流水线并行先把一个 mini-batch 拆成更小的 micro-batch，再让这些 micro-batch 依次流过各分区，于是不同分区可以同时处理不同的 micro-batch。
依据：paper/gpipe-src-v1/main.tex L143

各 micro-batch 的梯度不会各自更新参数，而是先在 mini-batch 末尾全部累加起来，再做一次同步更新。因此不管分成几个分区，参数更新都和没切分时一致，扩分区不会改变训练结果。
依据：paper/gpipe-src-v1/main.tex L143；paper/gpipe-src-v1/main.tex L368

相邻分区之间只传每个 micro-batch 的激活，在分区边界处传一次；此外没有别的卡间通信。
依据：paper/gpipe-src-v1/main.tex L181；paper/gpipe-src-v1/main.tex L363

#### 气泡从哪来、占多少

把流水线按时间切成一个个 slot，每个 slot 里每个分区处理一个 micro-batch。最前面和最后面几段里总有些分区处于空闲，这段空闲就是气泡。用 $p$ 表示分区数、$m$ 表示 micro-batch 数，可以数出气泡占一个 step（一次参数更新的时间）的比例是 $(p-1)/(m+p-1)$。
依据：paper/gpipe-src-v1/main.tex L179

::: slot 计数推出气泡占比
1. 一个 step 的时间切成若干个 slot，一个 slot 里每个分区处理一个 micro-batch；分区 $k$ 要等前一个分区把数据传来才能开始计算。
依据：paper/gpipe-src-v1/main.tex L179
2. 第一个 micro-batch 从分区 1 流到分区 $p$，要用掉 $p$ 个 slot；最前面的 $p-1$ 个 slot 里，后几个分区还没有任务。
依据：paper/gpipe-src-v1/main.tex L179
3. 上游分区把它的最后一个 micro-batch 传出去之后，下游分区还要过 $p-1$ 个 slot 才把剩下的 micro-batch 处理完，这段时间上游分区空闲。
依据：paper/gpipe-src-v1/main.tex L179
4. 因此一个 step 的总 slot 数是 $m+p-1$。每个分区在这段时间里正好处理 $m$ 个 micro-batch、忙 $m$ 个 slot，剩下 $p-1$ 个 slot 空着。
依据：paper/gpipe-src-v1/main.tex L179
5. 每个分区空着的 $p-1$ 个 slot 除以总 slot 数 $m+p-1$，得气泡占比 $\dfrac{p-1}{m+p-1}$。
依据：paper/gpipe-src-v1/main.tex L179
:::

#### 手算一遍气泡占比

取 $p=4$，$m$ 分别取 4 和 16，代入比例 $\dfrac{p-1}{m+p-1}$ 算一遍。
依据：自行构造

::: 两组数的气泡占比
1. 取 $p=4$、$m=4$：气泡占比 $\dfrac{3}{4+4-1}=\dfrac{3}{7}\approx0.43$。
依据：自行构造
2. 取 $p=4$、$m=16$：气泡占比 $\dfrac{3}{16+4-1}=\dfrac{3}{19}\approx0.16$。
依据：自行构造
3. 分区数不变、只把 micro-batch 数从 4 提到 16，气泡占比从约 0.43 降到约 0.16。
依据：自行构造
:::

#### micro-batch 要多到多少

micro-batch 越多，气泡占比越小。当 $m$ 至少是分区数 $p$ 的 4 倍时，气泡已经小到可以忽略；反过来，$m$ 等于 1 时没有流水，任一时刻只有一个分区在算，退化为单卡。
依据：paper/gpipe-src-v1/main.tex L179；paper/gpipe-src-v1/main.tex L242

### 切完每块卡的激活显存怎么变

前向算出的激活量会随层数增长。流水线并行下，每块卡只存分区边界处的激活，反向时再各自把这一段重新算一遍（重算），峰值激活显存（卡的存储空间）从与总层数成正比的量，降到与每块卡的层数、以及 micro-batch 大小有关的量。这样单卡能装下的模型更大。
依据：paper/gpipe-src-v1/main.tex L177

::: 分区与重算把单卡能装的模型扩大
1. 不加分区与重算时，单块卡最多训 82M 参数的模型。
依据：paper/gpipe-src-v1/main.tex L217
2. 加重算与拆 micro-batch，把中间激活显存从 6.26GB 降到 3.46GB，单卡能训到 318M 参数。
依据：paper/gpipe-src-v1/main.tex L217
3. 再用 8 块卡做流水线并行，模型扩到 1.8B 参数，是没加这些手段时的 25 倍。
依据：paper/gpipe-src-v1/main.tex L218
:::

### TP 与 PP 各适合什么互联

张量并行属于资料所称的 SPMD 一类（single program multiple data，所有卡跑同一份程序、各自处理不同分片的数据）：每一层都要 all-reduce，量也大。资料指出，这类靠 all-reduce 合并矩阵乘结果的做法，适用场景限于加速器之间用高速互联相连的时候。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L354；paper/gpipe-src-v1/main.tex L358

流水线并行相反：通信量小，即使加速器之间没有高速互联，也能扩展。
依据：paper/gpipe-src-v1/main.tex L181；paper/gpipe-src-v1/main.tex L358；paper/gpipe-src-v1/main.tex L363

### 两者组合起来用

张量并行和流水线并行互相正交，可以乘起来叠用，把两种切法放在不同维度的卡上。当模型大到连一台机器的所有卡都装不下时，还要在层内、层间之外再叠一层跨机器的并行，这套组合更合适。
依据：paper/megatron-src-v1/modelpar.tex L70；paper/megatron-src-v1/discussion.tex L5

### 适用条件与边界

流水线并行有两个前提：每一层要能放进单块卡，而且各分区算量大致均衡——某一层的计算量明显大于其他层时，这一段成为瓶颈，其余分区空等。层大到单卡放不下时，一种办法是把这一层的矩阵乘拆小。
依据：paper/gpipe-src-v1/main.tex L363；paper/gpipe-src-v1/main.tex L183

张量并行每一层都要通信，能用的卡数因此受高速互联域的大小限制。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L358
