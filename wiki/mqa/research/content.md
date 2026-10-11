# MQA

- 描述：MQA 是一种注意力机制，让所有查询头共用同一套键和值，把生成时反复读取的键值数据缩到一份。
- 摘要：生成每个新 token 都要重读全部键、值张量，这一步受显存带宽限制。MQA 让所有查询头共用一套键值，把这份数据量除以头数。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 资料

- Noam Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150v1, 2019. [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023. arXiv:2305.13245v3, 2023. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

## 短页

### 所有查询头共用一套键值

多头注意力（见[标准注意力](wiki/standard-attention)）给每个位置算出多组查询、键和值，每组叫一个头。每个头有自己的一套投影——输入乘一组可学的参数，得到这个头自己的查询、键和值；算注意力时每个头只用自己那套键和值。MQA（multi-query attention，多查询注意力）改的是键和值这一侧：让所有查询头共用同一套键和值，查询头数量不变。
依据：paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L259

### 生成慢在反复读取键值

生成文本是一个 token 一个 token 往下走的（token 是模型读写文本的最小单位）。写一个新 token 时，要用这一位置的查询和这个位置及前面所有位置的键各比一次，再按比较结果取这些位置的值。前面位置的键和值在各自算出的那一步之后就不再改变，把它们留下来供后面各步复用，就是 KV cache（见 [KV cache](wiki/kv-cache)）。序列越长、头数越多，每步要重读的这份键、值数据越大；当序列长度接近每个向量的维度、或同时处理的序列很少时，读写这份数据相对算术运算量就大，瓶颈落在显存带宽上（读写显存的速度）。让所有查询头共用同一套键值，就把这份数据量按键值头数缩小：多头每个头各一套，MQA 只剩一套。
依据：paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255

### 把键值头数设成一套

做法是把键头、值头的数量设成一套：查询头数量不变，键和值的投影只做一套，所有查询头取用同一套键和值。按这个结构从头训练一个模型，就得到 MQA 模型。它有一个更一般的版本 GQA：把查询头分成若干组、每组一套键值（见 [GQA](wiki/gqa)）。
依据：paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L337；paper/src-gqa/main.tex L132-L134

## 长页

多头注意力（见[标准注意力](wiki/standard-attention)）给每个位置算出多组查询、键和值，每组叫一个头。生成文本时模型一个 token 一个 token 往下写（token 是模型读写文本的最小单位），写每个新 token 都要把前面所有位置已经算出的键和值重读一遍（见 [KV cache](wiki/kv-cache)），这份数据随序列变长而变大，读取它耗掉的显存（存放这些键值的内存）带宽（每秒能读写的数据量）成了生成的瓶颈。MQA（multi-query attention，多查询注意力）让所有查询头共用一套键和值，把反复读取的这份键值数据缩到一份。
依据：paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L259

### 生成慢在反复读取键值

生成是自回归的：模型每次只推进一个位置，因为第 $t$ 个位置的输出决定了这一步生成哪个 token，而这个 token 又成为下一个位置的输入，前后有依赖，没法把多个位置放到一起并行算。但注意力层每一步要做的事没有变：它把当前位置的查询和这个位置及前面所有位置的键各做一次比较，再按比较结果取这些位置的值。这些键和值在前面的步骤里已经算好、也不会再变，但每一步仍要把它们从显存再读进来一次。

每步真正要新算的只有当前位置这一份查询、键和值，读进来的却是整段历史的键和值。当序列长度与每个向量的维度 $d$ 接近、或者批大小（一次同时处理的序列条数）为 $1$ 时，访存与计算之比（这一步要读写的数据量除以算术运算量的比值）接近 $1$；硬件的运算能力比带宽高出约两个数量级，这个比值接近 $1$ 时，时间几乎都花在读写数据上，受的是显存带宽的限制，不是运算速度。序列越长，或者同时处理的序列越少，要读写的量相对算术量越大，带宽就越紧张（批大小取大时比值里的 $\frac{1}{b}$ 项变小）。
依据：paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L207；paper/src-mqa/main.tex L209；paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255
::: 访存与计算之比
1. 训练时目标序列是已知的，所有位置的注意力可以并行算，计算能顺着矩阵乘法展开。
依据：paper/src-mqa/main.tex L209

2. 分析时取几条约定的规模：待比较的位置数与查询位置数相同，都记作序列长度 $n$；每个键、值向量的维度 $k$、$v$ 都取模型维度 $d$ 除以头数 $h$；$n$ 不超过 $d$。
依据：paper/src-mqa/main.tex L194-L199

3. 这 $n$ 步合起来的算术量是 $\Theta(bnd^2)$，其中 $b$ 是批大小：各步的注意力计算都要和 $d$ 维的投影矩阵（把输入变成查询、键和值的那组参数）相乘，每步约 $bd^2$ 次乘加，$n$ 步合起来是这个量级。
依据：paper/src-mqa/main.tex L249

4. 这 $n$ 步合起来的访存量是 $\Theta(bn^2d + nd^2)$：第一项来自反复读入的键、值（每个位置 $k$、$v$ 约 $d$ 个数，共 $n$ 个位置、$b$ 条序列，$n$ 步各读一遍），第二项来自各投影矩阵。
依据：paper/src-mqa/main.tex L251

5. 两者相除，访存与计算之比是 $\Theta(\frac{n}{d} + \frac{1}{b})$。
依据：paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L253
:::

### 多头各有一套键值，MQA 合成一套

多头注意力里每个头各有一套键和值：输入经投影（乘一组可学的参数）得到 $h$ 个查询向量、$h$ 个键向量和 $h$ 个值向量（$h$ 为头数），每个头各占一分，键和值因此都带着一个长度为 $h$ 的头维度。算注意力时，第 $i$ 个头的查询只用第 $i$ 个头的键和值。

MQA 只改一处：让所有头共用同一套键和值。做法是从键、值以及它们的投影里去掉头这个维度——不再是每个头一套键、一套值，而是整个序列只有一套键、一套值，被所有头的查询共用；查询仍按头分成 $h$ 份，各头算出的结果再经输出投影（把每个头的输出合回模型维度的那组参数）汇总，这一步也仍是每个头一份。于是同样的头数下，生成时每步要从显存读的键、值数据随之缩小，而算术量不变。

按 MQA 对增量解码（每步只推进一个 token）的分析，访存与计算之比里有一项随序列长度增长，共用一套键值后这一项被缩小 $h$ 倍，理论上批大小取大时增量生成的性能会改善。
依据：paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L277-L279；paper/src-mqa/main.tex L328-L330；paper/src-mqa/main.tex L337

::: 两个查询头共用一套键值算一遍
设序列有两个位置，查询头数 $h=2$，每个头里查询、键、值向量的维度都是 $2$。多头要有 2 套键和 2 套值，MQA 只要 1 套。取共用的这一套：位置 0 的键 $(1,0)$、位置 1 的键 $(0,2)$；位置 0 的值 $(2,0)$、位置 1 的值 $(0,1)$。
依据：自行构造

分数由当前位置的查询与该位置的键做点积得到，再经 SoftMax 归一成一组和为 $1$ 的权重（两者的算法见[标准注意力](wiki/standard-attention)），用权重对该位置的值加权求和就得到输出。
依据：自行构造

1. 两个查询头都用这同一套键和值。设位置 1 处两个头的查询是 $q_a=(1,0)$、$q_b=(0,1)$。
依据：自行构造

2. $q_a$ 的分数是 $(1,0)\cdot(1,0)=1$、$(1,0)\cdot(0,2)=0$；SoftMax 后是 $(0.731,0.269)$，输出约 $0.731\times(2,0)+0.269\times(0,1)=(1.462,0.269)$。
依据：自行构造

3. $q_b$ 的分数是 $(0,1)\cdot(1,0)=0$、$(0,1)\cdot(0,2)=2$；SoftMax 后是 $(0.119,0.881)$，输出约 $0.119\times(2,0)+0.881\times(0,1)=(0.238,0.881)$。
依据：自行构造

4. 两个头共用同一套键和值，查询不同，算出的输出就不同；共用键值并不把各头合并成一个，只是让它们读同一份键值数据。
依据：自行构造
:::

::: 形状与访存比值的对照
1. 多头：查询形状 $[b,h,n,k]$（$n$ 个查询位置），键、值形状 $[b,h,m,k]$ 和 $[b,h,m,v]$（$m$ 个历史位置），头维度 $h$ 在键、值上。
依据：paper/src-mqa/main.tex L182-L184

2. MQA：查询仍是 $[b,h,n,k]$，键变成 $[b,m,k]$、值变成 $[b,m,v]$，头维度从键、值上去掉；投影矩阵 $P_k$、$P_v$ 也从带 $h$ 的形状变成 $[d,k]$、$[d,v]$。
依据：paper/src-mqa/main.tex L270-L272；paper/src-mqa/main.tex L277-L279

3. 增量一步：读入历史键 $[b,m,k]$、值 $[b,m,v]$，把当前位置的一份拼到末尾，用当前查询和全部键比较、对全部值加权求和；这个读入动作在 $n$ 步里各做一次（分析里取历史位置数 $m$ 与查询位置数 $n$ 相同）。
依据：paper/src-mqa/main.tex L295-L296；paper/src-mqa/main.tex L307-L315；paper/src-mqa/main.tex L326

4. 这 $n$ 步合起来的访存量是 $\Theta(bnd + bn^2k + nd^2)$：第一项来自输入、查询与输出，第二项来自反复读入的键、值，第三项来自各投影矩阵；多头那一式 $\Theta(bn^2d + nd^2)$ 没有第一项。除以同样的 $\Theta(bnd^2)$ 算术量，访存与计算之比从 $\frac{n}{d} + \frac{1}{b}$ 变成 $\frac{1}{d} + \frac{n}{dh} + \frac{1}{b}$。
依据：paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L326；paper/src-mqa/main.tex L328-L330
:::

### 质量与训练稳定性

在 WMT 2014 英德翻译任务上，MQA 的质量与多头接近：dev 集（训练时用来观察、不参与最终评测的留出数据）上的两项质量指标都比多头稍差，测试集上 beam-4 解码（每次保留 4 条候选的搜索解码）的得分略高于多头；解码速度比多头快。
依据：paper/src-mqa/main.tex L349；paper/src-mqa/main.tex L351；paper/src-mqa/main.tex L357；paper/src-mqa/main.tex L359

::: WMT14 英德结果与解码开销
在同一族模型上，多头与 MQA 的 dev 结果（对数困惑度是模型对每个 token 预测分布的不确定度取对数，越低越好；BLEU 是机器译文与参考译文的 n-gram 重合度，越高越好）：

| 注意力 | dev 对数困惑度 | dev BLEU | 测试集 BLEU（beam 4） |
|---|---:|---:|---:|
| 多头 | 1.424 | 26.7 | 28.4 |
| MQA | 1.439 | 26.5 | 28.5 |

依据：paper/src-mqa/main.tex L362-L388

每 token 的解码开销（TPUv2 微秒，即一块 TPUv2 芯片上一微秒计的开销；源序列与目标序列各 128 token，批量 1024 条序列）：

- 多头：编码器 1.7，解码器每步 46
- MQA：编码器 1.5，解码器每步 3.8
依据：paper/src-mqa/main.tex L359；paper/src-mqa/main.tex L393-L412
:::

从头训练 MQA 还可能在训练中不稳定：预训练出现损失尖峰（损失值在若干步里陡升），在长输入任务上微调（在已训练好的模型上再训练）时会立刻发散（损失不再下降、训练无法继续）。
依据：paper/src-gqa/main.tex L228

### 适用条件与边界

MQA 只改键和值这一侧：键、值各只剩一套，查询头数不变，所以并不是把查询头也合并起来。代价是键和值只剩一份，模型在键值这部分能表达的信息量随之变小；头数越多，可去掉的键值头越多，但每减少一个键值头，这部分信息量就更小。
依据：paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259；paper/src-gqa/main.tex L136

用在哪些注意力层上，MQA 替换了模型里全部注意力层——编码器自注意力（编码器处理输入序列，自注意力指同一段序列内部各位置互相看）、解码器自注意力（解码器处理已经生成的部分）、编码器-解码器注意力（也叫交叉注意力，让解码器读取编码器的输出），这三处都换。
依据：paper/src-mqa/main.tex L339

GQA 在多头与 MQA 之间取中间做法：把查询头分成若干组、每组一套键值，组数取 $1$ 就是 MQA、取查询头数就是原来的多头；质量比 MQA 高，速度比多头快（见 [GQA](wiki/gqa)）。
依据：paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L136
