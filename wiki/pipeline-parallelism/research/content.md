# Pipeline Parallelism

- 描述：流水线并行把网络的层按先后切成连续的若干分区，每个分区放在一块加速器上，各分区同时处理不同的 micro-batch。
- 摘要：把全部层切成连续的分区放到多卡，再把一个 mini-batch 拆成 micro-batch 流水执行。只在分区边界传激活、通信量小，代价是分区造成的气泡。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 资料

- Yanping Huang, Youlong Cheng, Ankur Bapna 等. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. arXiv:1811.06965, 2019. [https://arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965)
- Mohammad Shoeybi, Mostofa Patwary, Raul Puri 等. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053v4, 2020. [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)

## 短页

### 把全部层切给不同的卡

神经网络是一层接一层叠起来的计算结构，层里可训练的参数是权重；加速器（下面简称卡）是做这些计算的处理器。流水线并行（pipeline parallelism，PP）把这叠层按先后切成连续的若干段（分区），每块卡持有一段，数据从第一个分区依次流到最后一个。
依据：paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L132-L141

按一层里权重矩阵的维度切开的叫张量并行（[张量并行](wiki/tensor-parallelism)），它和按层堆叠深度切的流水线并行是并列的两条路。
依据：paper/megatron-src-v1/intro.tex L7；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 一块卡装不下整个模型

模型参数大到超出单块卡的显存（卡上的存储空间）时，可以把模型按层切开放在多块卡上，突破单卡显存的限制。
依据：paper/gpipe-src-v1/main.tex L110；paper/gpipe-src-v1/main.tex L127

### 拆小批次，缩小气泡

训练时一次参数更新用的整批样本叫 mini-batch。流水线并行把一个 mini-batch 拆成更小的 micro-batch，让它们依次流过各分区，于是不同分区能同时处理不同的 micro-batch。
依据：paper/gpipe-src-v1/main.tex L143；paper/gpipe-src-v1/main.tex L166

各 micro-batch 的梯度（每个参数该往哪个方向调的指示量）不各自更新参数，而是先在 mini-batch 末尾全部累加，再做一次更新；这样参数更新与没切分时一致。分区越多，填满流水线所需的 micro-batch 也越多：micro-batch 数至少是分区数的 4 倍时，气泡（分区轮流空转的时间）已经小到几乎可以忽略。
依据：paper/gpipe-src-v1/main.tex L128；paper/gpipe-src-v1/main.tex L143；paper/gpipe-src-v1/main.tex L179；paper/gpipe-src-v1/main.tex L242

## 长页

神经网络是一层接一层叠起来的计算结构，每一层以上一层的输出为输入做一次计算、再把结果传出；层里可训练的参数是权重。加速器（下面简称卡）是做这些计算、带自己显存的处理器，显存指卡上的存储空间。一次训练一步分前向、反向两步：前向从输入逐层算到输出，反向从输出往回逐层算梯度——梯度是每个参数该往哪个方向调、调多大幅度的指示量。
依据：paper/gpipe-src-v1/main.tex L132-L141

流水线并行（pipeline parallelism，PP）把网络里的层按堆叠的先后切成连续的若干段（分区），每块卡只负责一段，一个分区的输出传给下一个分区当输入，数据依次流过。它和数据并行不同：数据并行把一次参数更新（按梯度把权重改动一次）用的整批样本拆到多块卡上、各卡处理不同的样本（一条训练数据）。
依据：paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L132-L141；paper/megatron-src-v1/background.tex L26

### 把全部层切成连续分区

网络由 $L$ 层叠成。给定分区数 $K$，流水线并行把这 $L$ 层切成 $K$ 个连续分区，第 $k$ 个分区放在第 $k$ 块卡上，分区内部是一段连续的层。
依据：paper/gpipe-src-v1/main.tex L132-L141

相邻分区之间在边界处自动插入通信，只传激活（分区算出的、要传给下一个分区的中间张量——张量是多维数组，矩阵是它的二维情形）；除此之外没有别的卡间通信。分区算法让各分区的预计代价尽量均衡。
依据：paper/gpipe-src-v1/main.tex L141；paper/gpipe-src-v1/main.tex L181

#### 把一个 mini-batch 拆成 micro-batch 流水

一次参数更新用的整批样本叫 mini-batch，其样本数为 $N$。流水线并行先把每个 mini-batch 等分成 $M$ 个更小的 micro-batch，再让这 $M$ 个 micro-batch 依次流过 $K$ 个分区。
依据：paper/gpipe-src-v1/main.tex L128；paper/gpipe-src-v1/main.tex L143

一个 micro-batch 从第一个分区流到第 $K$ 个分区要经过 $K$ 个分区的时间，等它离开第一个分区，第一个分区就能开始处理下一个 micro-batch。于是不同分区在同一段时间里各自处理不同的 micro-batch，除流水线的首尾两端外都在计算。
依据：paper/gpipe-src-v1/main.tex L143；paper/gpipe-src-v1/main.tex L166

各 micro-batch 的梯度不会各自更新参数：在整个 mini-batch 的末尾，$M$ 个 micro-batch 的梯度全部累加起来，再做一次同步更新。因此不管分成几个分区，参数更新都和没切分时一致。
依据：paper/gpipe-src-v1/main.tex L128；paper/gpipe-src-v1/main.tex L143

#### 气泡从哪来、占多少

分区带来空闲：最前和最后一段里，有些分区还没有任务或已经做完，这段时间没有 micro-batch 可处理，它们是空的。这段空转叫气泡。
依据：paper/gpipe-src-v1/main.tex L175-L183

只数前向：把流水线按时间分成一个个长度相同的时隙（每块卡处理一个 micro-batch 的时间）。$K$ 个分区、$M$ 个 micro-batch 时，气泡占一个 step（$M$ 个 micro-batch 全部流过一遍、做一次参数更新的时间）的比例约为 $\dfrac{K-1}{M+K-1}$。
依据：paper/gpipe-src-v1/main.tex L179

::: 时隙计数推出气泡占比
1. 第 $k$ 个分区要等第 $k-1$ 个分区把数据传来才能开始；第一个 micro-batch 要先流过前面所有分区。
依据：自行构造

2. 第一个 micro-batch 从分区 1 流到分区 $K$，要用掉 $K$ 个时隙；最前面的 $K-1$ 个时隙里，后面几个分区还没有任务。
依据：自行构造

3. 第一个分区把它的最后一个 micro-batch 传给第二个分区之后，还要等这个 micro-batch 走完下游 $K-1$ 个分区，这段时间它是空的，共 $K-1$ 个时隙。
依据：自行构造

4. 只数前向，一个 step 的总时隙数是 $M+K-1$；每块卡在这段时间里正好处理 $M$ 个 micro-batch、忙 $M$ 个时隙，剩下 $K-1$ 个时隙空着。
依据：自行构造

5. 空着的 $K-1$ 个时隙除以总时隙数 $M+K-1$，得气泡占比 $\dfrac{K-1}{M+K-1}$。
依据：自行构造
:::

#### 手算一遍气泡占比

分区数固定、只改 micro-batch 数，看气泡占比怎么变。
依据：自行构造

::: 两组数的气泡占比
1. 取 $K=3$、$M=2$：气泡占比 $\dfrac{2}{2+3-1}=\dfrac{2}{4}=0.5$。
依据：自行构造

2. 取 $K=3$、$M=12$：气泡占比 $\dfrac{2}{12+3-1}=\dfrac{2}{14}\approx0.14$。
依据：自行构造

3. 分区数不变、只把 micro-batch 数从 2 提到 12，气泡占比从 0.5 降到约 0.14。
依据：自行构造
:::

#### micro-batch 要多到多少

micro-batch 越多，气泡占比越小。资料在实验里把 $M$ 至少是分区数 $K$ 的 4 倍时的气泡视为可忽略；反过来，$M$ 较小时气泡不再可以忽略，$M$ 等于 1 时没有流水，任一时刻只有一块卡在算，退化为单卡。
依据：paper/gpipe-src-v1/main.tex L179；paper/gpipe-src-v1/main.tex L242

### 切完每块卡的激活显存怎么变

前向算出的激活量会随层数增长，全都留下来供反向使用会占很多显存。流水线并行让每块卡在前向只存分区边界处的激活，反向时各自把这一段重新算一遍（重算）；这样峰值激活显存从与总层数成正比的量，降到与每块卡的层数、以及 micro-batch 大小有关的量。
依据：paper/gpipe-src-v1/main.tex L175-L179

::: 以 AmoebaNet 为例，分区与重算把单卡能装的模型扩大
1. 不加分区与重算时，单块卡最多训 82M（M 指百万）参数的 AmoebaNet（一个图像分类网络）。
依据：paper/gpipe-src-v1/main.tex L217

2. 加重算与拆 micro-batch，把中间激活显存从 6.26GB 降到 3.46GB，单卡能训到 318M 参数。
依据：paper/gpipe-src-v1/main.tex L217-L218

3. 再用 8 块卡做流水线并行，AmoebaNet 扩到 1.8B（B 指十亿）参数，约是没加这些手段时的 22 倍（82M 到 1.8B）。
依据：paper/gpipe-src-v1/main.tex L193-L198；paper/gpipe-src-v1/main.tex L218
:::

### 适用条件与边界

流水线并行要求每一层能放进单块卡。不同层的参数与计算量往往并不均衡，分得不均时会出现负载不均，扩展的速度受到影响。
依据：paper/gpipe-src-v1/main.tex L363；paper/gpipe-src-v1/main.tex L183；paper/gpipe-src-v1/main.tex L218

分区数越多，气泡占比越高，需要更多的 micro-batch 来降低气泡占比。
依据：paper/gpipe-src-v1/main.tex L179

即使卡之间没有高速互联，流水线并行也能扩展。
依据：paper/gpipe-src-v1/main.tex L181

流水线并行是模型并行（把模型的权重切分到多块卡上这一类）里的一路，不是唯一的一路。流水线并行与张量并行（[张量并行](wiki/tensor-parallelism)，按层内矩阵维度切）正交互补，可以组合使用。
依据：paper/megatron-src-v1/intro.tex L7；paper/megatron-src-v1/abs.tex L3；paper/megatron-src-v1/discussion.tex L5
