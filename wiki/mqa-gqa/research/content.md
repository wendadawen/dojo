# MQA 与 GQA

- 描述：MQA 让所有查询头共用一套键和值，GQA 把查询头分成若干组、每组共用一套，两者都缩小生成时反复读取的键值数据量。
- 摘要：生成每个新 token 都要重读全部键、值张量，这一步受显存带宽限制。MQA 让所有查询头共用一套键值，把这份数据量除以头数；GQA 让每组查询头共用一套，组数在 $1$（等于 MQA）与头数（等于多头）之间取值，换来质量与速度的折中。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 资料

- Noam Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150v1, 2019. [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023. arXiv:2305.13245v3, 2023. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

## 短页

### 每个查询头不再各有一套键值

多头注意力（见[标准注意力](wiki/standard-attention)）给每个位置算出多组查询、键和值，每一组叫一个头。每个头有自己的一套投影——输入乘一组可学的参数，得到这个头自己的查询、键和值；算注意力时每个头只用自己那套键和值。MQA 和 GQA 改的都是键和值这一侧，查询头数量不变。MQA 让所有查询头共用同一套键和值；GQA 把查询头分成若干组，每组只算一份键、一份值，这一份叫这组的键头和值头，组数取 1 就是 MQA，取查询头数就是原来的多头。
依据：paper/src-mqa/main.tex L259；paper/src-gqa/main.tex L132-L134

### 生成慢在反复搬键值

生成文本是一个 token 一个 token 往下走的，token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分。写一个新 token 时，要用这一位置的查询和前面全部位置的键各比一次，再按比较结果取前面全部位置的值。前面位置的键和值在各自算出的那一步之后就不再改变，把它们留下来供后面各步复用，就是 KV cache（见 [KV cache](wiki/kv-cache)）。序列越长、头数越多，每步要重读的这份键、值数据越大，而这一步的算术运算很少，瓶颈落在显存带宽上。让所有查询头共用同一套键值，就把这份数据量除以头数。
依据：paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255；paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L136

### 按组共用一套键值

做法是把键头、值头的数量设成查询头的组数 $G$：查询头照旧，键和值的投影只做 $G$ 套，第 $i$ 个查询头取用它所属那一组的键和值。得到这样的模型有两条路，一条是按这个结构从头训练；另一条是把已有的多头模型（它训练好的参数叫检查点）改造过来，把每组内各头的键投影、值投影分别逐元素取平均合成一套，再继续训练一小段，让它适应新结构。
依据：paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L126；paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L228

## 长页

多头注意力（见[标准注意力](wiki/standard-attention)）给每个位置算出多组查询、键和值，每组叫一个头。生成文本时模型一个 token 一个 token 往下写（token 是模型读写文本的最小单位），写每个新 token 都要把前面所有位置已经算出的键和值重读一遍（见 [KV cache](wiki/kv-cache)），这份数据随序列变长而变大，读取它耗掉的显存带宽成了生成的瓶颈。MQA 让所有查询头共用一套键和值，把反复读取的这份键值数据压到一份；GQA 把查询头分成若干组、每组共用一套，组数在 MQA 与多头之间取值。
依据：paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L132-L134；paper/src-mqa/main.tex L259

### 解码慢在反复搬键值

生成是自回归的：模型每次只推进一个位置，因为第 $t$ 个位置的输出决定了这一步生成哪个 token，而这个 token 又成为下一个位置的输入，前后有依赖，没法把多个位置放到一起并行算。但注意力层每一步要做的事没有变：它把当前位置的查询和前面所有位置的键各做一次比较，再按比较结果取前面所有位置的值。这些键和值在前面的步骤里已经算好、也不会再变，只是它们不能一直留在运算单元里，每一步都要从显存再读进来一次。

每步真正要新算的只有当前位置这一份查询、键和值，读进来的却是整段历史的键和值。当序列长度与向量维度接近、或者批大小（一次同时处理的序列条数）为 $1$ 时，访存与计算之比接近 $1$；硬件的运算能力比带宽高出约两个数量级，这个比值接近 $1$ 时，时间几乎都花在搬数据上，受的是显存带宽的限制，不是运算速度。序列越长，或者同时处理的序列越少，要搬的量相对算术量越大，带宽就越紧张。
依据：paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L207；paper/src-mqa/main.tex L209；paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255

::: 访存与计算之比
1. 训练时目标序列是已知的，所有位置的注意力可以并行算，计算能顺着矩阵乘法摊开。
依据：paper/src-mqa/main.tex L209

2. 分析时取几条约定的规模：待比较的位置数与查询位置数相同，都记作序列长度 $n$；每个键、值向量的维度 $k$、$v$ 都取模型维度 $d$ 除以头数 $h$；$n$ 不超过 $d$。
依据：paper/src-mqa/main.tex L194-L199

3. 这 $n$ 步合起来的算术量是 $\Theta(bnd^2)$（由各步的矩阵乘加出），其中 $b$ 是批大小。
依据：paper/src-mqa/main.tex L249

4. 这 $n$ 步合起来的访存量是 $\Theta(bn^2d + nd^2)$：第一项来自反复读入的键、值张量，第二项来自各投影矩阵。
依据：paper/src-mqa/main.tex L251

5. 两者相除，访存与计算之比是 $\Theta(\frac{n}{d} + \frac{1}{b})$。
依据：paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L253
:::

### 多头各有一套键值，MQA 合成一套

多头注意力里每个头各有一套键和值：输入经投影（乘一组可学的参数）得到 $h$ 个查询向量、$h$ 个键向量和 $h$ 个值向量（$h$ 为头数），每个头各占一分，键和值因此都带着一个长度为 $h$ 的头维度。算注意力时，第 $i$ 个头的查询只用第 $i$ 个头的键和值。

MQA 只改一处：让所有头共用同一套键和值。做法是从键、值以及它们的投影里去掉头这个维度——不再是每个头一套键、一套值，而是整个序列只有一套键、一套值，被所有头的查询共用；查询仍按头分成 $h$ 份，输出投影也仍是每个头一份。于是同样的头数下，生成时每步要从显存读的键、值数据随之缩小，而算术量不变。

按 MQA 对增量解码（每步只推进一个 token 的那种生成）的分析，访存与计算之比里有一项随序列长度增长，共用一套键值后这一项被缩小 $h$ 倍，理论上批大小取大时增量生成的性能会改善。
依据：paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L249；paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L277-L279；paper/src-mqa/main.tex L326；paper/src-mqa/main.tex L330

::: 形状与访存比值的对照
1. 多头：查询张量形状 $[b,h,n,k]$（$n$ 个查询位置），键、值形状 $[b,h,m,k]$ 和 $[b,h,m,v]$（$m$ 个历史位置，与 $n$ 相同），头维度 $h$ 在键、值上。
依据：paper/src-mqa/main.tex L182-L184

2. MQA：查询仍是 $[b,h,n,k]$，键变成 $[b,m,k]$、值变成 $[b,m,v]$，头维度从键、值上去掉；投影矩阵 $P_k$、$P_v$ 也从带 $h$ 的形状变成 $[d,k]$、$[d,v]$。
依据：paper/src-mqa/main.tex L270-L272；paper/src-mqa/main.tex L277-L279

3. 增量一步：读入历史键 $[b,m,k]$、值 $[b,m,v]$，把当前位置的一份拼到末尾，用当前查询和全部键比较、对全部值加权求和；这个读入动作在 $n$ 步里各做一次。
依据：paper/src-mqa/main.tex L307-L315

4. 这 $n$ 步合起来的访存量是 $\Theta(bnd + bn^2k + nd^2)$：第一项来自输入、查询与输出张量，第二项来自反复读入的键、值张量，第三项来自各投影矩阵；多头那一式 $\Theta(bn^2d + nd^2)$ 没有第一项。除以同样的 $\Theta(bnd^2)$ 算术量，访存与计算之比从 $\frac{n}{d} + \frac{1}{b}$ 变成 $\frac{1}{d} + \frac{n}{dh} + \frac{1}{b}$。
依据：paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L328-L330
:::

### GQA 把查询头分组

MQA 把键和值压到一套，每步要读的数据确实少了，但键和值只剩一份，模型在这部分上的容量也随之变小。GQA 取两者之间的做法：把查询头分成 $G$ 组，每组只算一份键和一份值，这一份叫这组的键头和值头。$G$ 等于 $1$ 时只有一组，就是 MQA；$G$ 等于查询头数时每个头各成一组，就是原来的多头。

用 $G$ 记组数，则键头、值头各有 $G$ 个。生成时每步要读的键、值数据量随 $G$ 增大而增大：从 MQA 的一套，增到多头那样的查询头数套。
依据：paper/src-gqa/main.tex L117；paper/src-gqa/main.tex L132-L134；paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L188-L190

::: 4 个查询头分 2 组算一遍
设序列有两个位置，查询头数 $h=4$，每个头里查询、键、值向量的维度都是 $2$，组数 $G=2$：第 0、1 个查询头为第 0 组，第 2、3 个为第 1 组。多头要有 4 套键和 4 套值，MQA 只要 1 套，GQA 这里要 2 套。
依据：自行构造

- 第 0 组的键头：位置 0 的键 $(1,0)$、位置 1 的键 $(0,2)$；值头：位置 0 的值 $(2,0)$、位置 1 的值 $(0,1)$。
- 第 1 组的键头：位置 0 的键 $(0,1)$、位置 1 的键 $(1,0)$；值头：位置 0 的值 $(1,1)$、位置 1 的值 $(1,0)$。
依据：自行构造

分数由当前位置的查询与该位置的键做点积得到，再经 SoftMax 归一成一组和为 $1$ 的权重（两者的算法见[标准注意力](wiki/standard-attention)），用权重对该位置的值加权求和就得到输出。
依据：自行构造

1. 第 0 组的两个查询头都只用第 0 组的键和值。设位置 1 处这两个查询头是 $q_a=(1,0)$、$q_b=(0,1)$。
依据：自行构造

2. $q_a$ 的分数是 $(1,0)\cdot(1,0)=1$、$(1,0)\cdot(0,2)=0$；SoftMax 后是 $(0.731,0.269)$，输出约 $0.731\times(2,0)+0.269\times(0,1)=(1.462,0.269)$。
依据：自行构造

3. $q_b$ 的分数是 $(0,1)\cdot(1,0)=0$、$(0,1)\cdot(0,2)=2$；SoftMax 后是 $(0.119,0.881)$，输出约 $0.119\times(2,0)+0.881\times(0,1)=(0.238,0.881)$。
依据：自行构造

4. 两个头共用同一套键和值，查询不同，算出的输出就不同，这正是同组共享键值、各头仍各自算注意力的样子。
依据：自行构造

5. 第 1 组的查询头用第 1 组自己的键和值。设位置 1 处 $q_c=(1,1)$：两个分数是 $(1,1)\cdot(0,1)=1$、$(1,1)\cdot(1,0)=1$，权重为 $(0.5,0.5)$，输出 $0.5\times(1,1)+0.5\times(1,0)=(1,0.5)$。
依据：自行构造
:::

### 从现有多头检查点改造

已有一个训练好的多头模型，它的参数叫检查点。放弃它、按新结构重新训练代价很大。另一条路是把这个检查点改造成 MQA 或 GQA 的结构，再补一小段训练。改结构只需动键和值的投影：把每一组内各头的键投影逐元素取平均，得到这一组的键头；值投影同样处理，得到这一组的值头；查询和输出的投影不动。
依据：paper/src-gqa/main.tex L113；paper/src-gqa/main.tex L115；paper/src-gqa/main.tex L124-L130；paper/src-gqa/main.tex L134

改造方式还有另外两种：只保留组内第一个头，或把新头随机初始化。三种做法里，取平均这一种的平均分最高，只保留组内第一个头的次之，随机初始化最低；这个顺序大体对应它们保留原模型信息的多少。
依据：paper/src-gqa/main.tex L178-L179

改好的模型要再用原来的训练配方、按原预训练步数的一小部分继续训练（这一步叫续训），让它适应新结构。MQA 也先改结构、再续训，而且更需要这段续训：不续训时质量偏低，续训一段后才达标；GQA-8（组数取 8 的 GQA）改造后即可使用。
依据：paper/src-gqa/main.tex L150-L153；paper/src-gqa/main.tex L183-L184；paper/src-gqa/main.tex L228

::: 三种改造做法的对比与续训比例
在同一规模上比较三种改造做法，续训到原预训练步数的 5% 后，平均分是：

- 组内各头取平均：55.55
- 只留组内第一个头：55.45
- 新头随机初始化：55.23
依据：paper/src-gqa/figures/conversion_methods.tex L22-L27；paper/src-gqa/figures/conversion_methods.tex L31

在另一个更大的规模上，续训比例取 $0$、$0.05$、$0.1$ 时的平均分是：

- GQA-8：56.713、57.4、57.56
- MQA：53.93、56.92、57.153
- 多头：57.537（不改造，作对照）
依据：paper/src-gqa/figures/uptraining_steps.tex L26-L40；paper/src-gqa/figures/uptraining_steps.tex L44

这两组分数各自只在一部分代表性任务上算，两个规模也不同。
依据：paper/src-gqa/main.tex L175；paper/src-gqa/figures/conversion_methods.tex L31；paper/src-gqa/figures/uptraining_steps.tex L44
:::

### 质量与速度的取舍

把三种做法放在一起比：跟同一规模的多头相比，MQA 的推理时间更短、平均分更低；GQA-8 以接近 MQA 的推理时间，取得接近多头的平均分。
依据：paper/src-gqa/main.tex L168；paper/src-gqa/tables/main_results.tex L10-L13；paper/src-gqa/figures/results.tex L18-L31

::: 各做法的推理时间与平均分
实验在同一族模型的两个规模上做；两个规模都用多头，再从大那个规模的多头改造成 MQA 和 GQA-8，两者各续训原预训练步数的 5%。每样本推理时间与平均分：

- 多头 Large：0.37 秒，46.0
- 多头 XXL：1.51 秒，47.2
- MQA-XXL：0.24 秒，46.6
- GQA-8-XXL：0.28 秒，47.1
依据：paper/src-gqa/tables/main_results.tex L8；paper/src-gqa/tables/main_results.tex L10-L13；paper/src-gqa/tables/main_results.tex L17；paper/src-gqa/figures/results.tex L18-L31

GQA-XXL 取不同组数时的推理时间（输入长度 2048、输出长度 512）：1 组 0.489 秒、4 组 0.476 秒、8 组 0.514 秒、16 组 0.594 秒、32 组 0.800 秒、64 组 2.531 秒；多头是 2.531 秒，MQA 是 0.489 秒。从 1 组到 8 组耗时变化很小，之后逐档增大，到 64 组回到多头水平。这一组与上面四行来自两次不同的测量。
依据：paper/src-gqa/figures/time_vs_groups.tex L28-L44；paper/src-gqa/figures/time_vs_groups.tex L48；paper/src-gqa/main.tex L190
:::

### 适用条件与边界

MQA 换来带宽上的收益，付出的是质量上的代价。从头训练的 MQA 还可能在训练中不稳定：预训练出现损失尖峰（损失值陡升），在长输入任务上微调（在已训练模型上再训练）时立刻发散（训练无法继续）；改造来的 MQA 稳定一些，但方差仍大。
依据：paper/src-gqa/main.tex L113；paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L228；paper/src-mqa/main.tex L349；paper/src-mqa/main.tex L353

MQA 与 GQA 都只改键和值这一侧：键头、值头的数目由等于查询头数改成组数，查询头数与输出维度不变，所以并不是把查询头也合并起来。
依据：paper/src-gqa/main.tex L134；paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259

适用范围上，两种做法略有不同。注意力会用在三个地方：编码器自注意力处理输入序列，解码器自注意力处理已经生成的部分，编码器-解码器注意力让解码器读取编码器的输出。GQA 不用在编码器自注意力层，因为编码器各个位置可以并行算，带宽一般不是那里的瓶颈；MQA 则把这三处的注意力都换掉了。
依据：paper/src-gqa/main.tex L140；paper/src-mqa/main.tex L339

模型越大，可压的键值头数越多，所以预期其收益越大。
依据：paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L138
