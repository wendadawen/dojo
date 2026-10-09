# Attention Sink

- 描述：Attention sink 是自回归 Transformer 的注意力现象：序列开头少数几个 token 拿走了远高于其语义分量的注意力，模型把它们当作注意力分数的默认去处。
- 摘要：注意力分数必须和为 $1$，没有该关注的东西时，多余的分量被倾倒到序列开头的几个 token 上，这就是 attention sink。窗口注意力在长文本上崩溃，是因为挤掉了它们；保住它们的 KV，就能在远超训练长度的文本上流式生成。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 资料

- Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453v4, 2024. [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- MIT HAN Lab. streaming-llm: official implementation. 2024. [https://github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)

## 短页

### 开头的几个 token 拿走了注意力

大语言模型逐 token 生成文本，token 是文本被切分后的小片段（一个词或词的一部分），模型以它为单位处理；生成第 $t$ 个 token 时只看到它前面的 $t-1$ 个，这一性质叫自回归。每生成一个 token，注意力层让当前位置用它自己的查询向量和前面每个位置的键向量做点积，得到的分数经 SoftMax 归一成一组权重，再用这些权重把对应的值向量加权求和。已生成 token 的键和值被缓存下来供后面的 token 使用，这份缓存叫 KV cache。
依据：paper/src-v4/text/1_introduction.tex L13；paper/src-v4/text/3_method.tex L11-L16

Attention sink 是这套计算里的一个现象：序列开头少数几个 token 分到的注意力权重远高于它们对当前预测的语义分量。把它们换成内容毫无意义的换行符，权重照样落在它们身上。
依据：paper/src-v4/text/3_method.tex L11-L22；paper/src-v4/text/1_introduction.tex L21-L23

### 分数必须和为 1

注意力分数要先经 SoftMax 归一：把每个位置的分数取指数 $e^{x_i}$，再除以所有位置指数之和，得到的权重加起来恒等于 $1$，不能全为零。当前 token 已经包含了预测所需的信息时，仍然要保证总和为 1，模型于是把多余的分量分给特定 token，这些 token 被称为 sink。
依据：paper/src-v4/text/3_method.tex L23-L25

多余的分量落在序列开头的 token 上。一个解释是：自回归模型里，开头的位置对后面每个位置都可见，比中间位置更容易在训练中被固定成这个去处。这样的位置一旦被删掉，SoftMax 分母里就少了一大块，注意力分布随之整体偏移。
依据：paper/src-v4/text/3_method.tex L12；paper/src-v4/text/3_method.tex L26-L27

只保留最近 $K$ 个 token 的键和值、超出的淘汰，这种做法叫窗口注意力（window attention），$K$ 是缓存大小。Llama-2-13B 的缓存只留最近 1024 个 token 时，困惑度（模型对下一个 token 的不确定程度，越低越好）是 5158.07；在这 1024 个里留 4 个开头 token（再加 1020 个最近 token），困惑度是 5.40。
依据：paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17；paper/src-v4/text/3_method.tex L8-L9

### 缓存的头几个位置留给 sink

做法是把 KV cache 分成两段：开头固定留几个初始 token 的键和值，其余位置按先进先出保存最近的 token，中间的被淘汰。初始 token 取 4 个就够，再多收益很小。模型靠位置编号区分 token 的先后，所以缓存里的位置不能沿用原文序号：中间被淘汰后原文序号会断开，要按它们在缓存内的序号重新编号，位置才连续。
依据：paper/src-v4/text/3_method.tex L32-L37；paper/src-v4/figtab/start_size_and_linebreak.tex L26-L42

还可以让模型一开始就依赖更少的 sink：预训练时在每条样本最前面放一个可学习的占位 token，专门承接多余的注意力；训练完成后只留这一个 token，就足以稳定流式生成。
依据：paper/src-v4/text/3_method.tex L40-L47；paper/src-v4/text/3_method.tex L52；paper/src-v4/figtab/pretrain_fix.tex L6-L14

## 长页

大语言模型逐 token 生成文本，token 是文本被切分后的小片段（一个词或词的一部分）；生成第 $t$ 个 token 时只看得到前 $t-1$ 个，这一性质叫自回归。每一步，注意力层拿当前位置的查询向量和前面各位置的键向量算出分数，经 SoftMax 归一成权重，再对各位置的值向量加权求和。已经算过的键和值缓存下来供后面的 token 使用，这份缓存叫 KV cache。
依据：paper/src-v4/text/1_introduction.tex L13；paper/src-v4/text/3_method.tex L11-L16

Attention sink 指这样一个现象：序列开头少数几个 token 分到的注意力权重很大，与它们对当前预测有没有意义无关。这几个 token 就是 attention sink。
依据：paper/src-v4/text/1_introduction.tex L21；paper/src-v4/text/3_method.tex L11-L14

已训练好的模型只要在缓存里始终留着这几个 token 的 KV，就不用改权重、不用微调，能在远超训练长度的文本上持续生成。
依据：paper/src-v4/text/0_abstract.tex L5-L7；paper/src-v4/text/1_introduction.tex L21-L24

### 窗口注意力的崩溃点

#### 滑窗省显存却崩在 cache 上

dense attention 让每个 token 都注意到它前面的所有 token，要缓存全部 KV，显存占用和解码延迟随文本增长。预训练还把注意力窗口固定在某个长度（Llama-2 是 4K），文本超过它，模型本身的性能也在下降。
依据：paper/src-v4/text/1_introduction.tex L2；paper/src-v4/text/1_introduction.tex L13-L14

窗口注意力（window attention）只保留最近 $K$ 个 token 的键和值，$K$ 是缓存大小；缓存填满后显存占用和解码速度都不再变化。但文本一超过缓存大小，最早的开头 token 被挤出，模型就崩。缓存配置写成 $x$+$y$，表示保留 $x$ 个开头 token 和 $y$ 个最近 token，0+1024 就是窗口注意力。Llama-2-13B 在 0+1024 时，困惑度（模型对下一个 token 的不确定程度，越低越好）是 5158.07，换成 4+1020，即把开头 4 个 token 的 KV 放回去，回到 5.40。另一种做法是带重算的滑窗：每生成一个 token，都用最近的一段原文把窗口内的键和值重算一遍，效果好，但很慢。
依据：paper/src-v4/text/1_introduction.tex L17-L18；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17

::: 困惑度的测量条件
测量用 PG19 测试集的第一本书，长约 65K token。
依据：paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17
:::

### 初始 token 上到底发生了什么

#### 证据：除最底两层外各层各头都集中关注开头

把 Llama-2-7B 的注意力分数（SoftMax 之前的值）按层按头画成图，取 256 句、每句 16 个 token 的平均：最底下两层关注附近的位置，从第三层往上，各层各头都集中关注序列开头的 token。
依据：paper/src-v4/text/3_method.tex L11-L14；paper/src-v4/figtab/attention_visualization.tex L5

一个分辨实验：开头 token 起作用，是因为内容，还是因为位置？把开头四个 token 换成换行符，内容上不再有信息。模型仍把大量注意力放在这四个换行符上，把它们放进缓存，Llama-2-13B 的困惑度是 5.60，与放原始开头 token 时相当。可见起作用的是它们处在序列最开头，而不是语义。
依据：paper/src-v4/text/3_method.tex L20-L22；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17

把序列拉长到 4096 个 token 再统计，除最底下两层，各层分给第一个 token 的注意力常常超过总注意力的一半。
依据：paper/src-v4/text/6_appendix.tex L44

::: 统计口径
256 条长 4096 的序列取平均，画的是每条序列第 4096 个 token 分给首个 token 的注意力，取 SoftMax 之后的值。
依据：paper/src-v4/text/6_appendix.tex L44
:::

#### 为什么是初始 token

注意力分数要经 SoftMax 归一，所有被注意到的 token 的权重之和等于 $1$，不能全部为零。即使当前位置的表示已经包含做预测所需的信息，注意力机制仍需从上下文里取回一份不为零的信息，于是模型把这份用不上的注意力分给特定 token，这些 token 就是 sink。它们因此在 SoftMax 的分母里占了很大一块，去掉它们的 KV，注意力分布就整体偏离正常。
依据：paper/src-v4/text/3_method.tex L12；paper/src-v4/text/3_method.tex L23-L28

::: SoftMax 公式与 sink 在分母里的分量
注意力分数写成向量 $x = (x_1, \dots, x_N)$，$N$ 是它能注意到的位置数，$x_i$ 是第 $i$ 个位置的分数。归一化后：
依据：paper/src-v4/text/3_method.tex L11-L16

$$\text{SoftMax}(x)_i = \frac{e^{x_i}}{e^{x_1} + \sum_{j=2}^{N} e^{x_j}}, \quad x_1 \gg x_j,\ j \in 2, \dots, N$$

条件 $x_1 \gg x_j$ 表示首个位置的分数远大于其余，$e^{x_1}$ 因此主导分母，其余各项相对很小，分给其余位置的权重被压低，注意力集中到首个位置。
依据：paper/src-v4/text/3_method.tex L11-L16
:::

多余的注意力落在开头的 token 上，一个解释是：自回归模型里开头的 token 对后面每一个位置都可见，靠后的 token 只对它后面的位置可见，可见次数多，就更容易在训练中被固定成承接多余注意力的 token。
依据：paper/src-v4/text/3_method.tex L27

### 保住 sink 的滚动 KV cache

做法不改模型权重，无需微调。把 KV cache 分成两段：一段是开头 4 个 token 的键和值，充当 attention sink，稳定注意力的计算；另一段叫滚动 KV cache，按先进先出保留最近的 token，供语言建模使用。
依据：paper/src-v4/text/3_method.tex L34

位置编号取 token 在缓存内的序号，而不取它在原文里的位置：淘汰让原文位置出现断档，缓存内的序号始终连续。相对位置编码用 token 之间的距离表达位置，RoPE 和 ALiBi 都属于此类。RoPE 对每层注意力的查询和键做旋转变换，把相对位置带进分数；旋转取决于位置，而缓存内的序号每步都在变，所以缓存存的是旋转之前的键，每个解码步再按当前序号对缓存里的键施加旋转。ALiBi 按查询和键的距离给注意力分数加偏置；位置取缓存内序号，距离连续，加的就是连续的线性偏置，而不是有跳跃的偏置。
依据：paper/src-v4/text/3_method.tex L34-L38；paper/src-v4/text/2_related.tex L6

::: 滚动缓存的三步与位置编号
设缓存大小为 $K=6$，开头段 3 个 token，滚动段最多 3 个（为了手算，开头段取 3 个，不取 4 个）。每生成一个 token：
依据：自行构造

1. 把缓存里的 token 连同当前 token，按在缓存内的序号 $0, 1, 2, \dots$ 编号，据此施加位置编码，算出当前 token 的注意力。
2. 把当前 token 的键和值追加到滚动段末尾。
3. 滚动段超过 3 个时，淘汰其中最早的。
依据：自行构造

缓存里是原文位置 $0, 1, 2$（开头段）和 $5, 6, 7$（滚动段）的 6 个 token，现在生成原文位置 8 的那个 token：

1. 缓存内的 6 个 token 编号 $0, \dots, 5$，当前 token 编号 $6$，所以原文位置 $5, 6, 7, 8$ 依次得到编号 $3, 4, 5, 6$，而不是 $5, 6, 7, 8$。
2. 追加后缓存是原文位置 $0, 1, 2, 5, 6, 7, 8$，共 7 个。
3. 滚动段 $5, 6, 7, 8$ 有 4 个，超过 3，淘汰 $5$，缓存剩下 $0, 1, 2, 6, 7, 8$。

接着生成位置 9 的那个 token：缓存内 $0, 1, 2, 6, 7, 8$ 编号 $0, \dots, 5$，当前 token 编号 $6$，原文位置 $6, 7, 8, 9$ 得到编号 $3, 4, 5, 6$；淘汰 $6$ 后缓存是 $0, 1, 2, 7, 8, 9$。编号始终不超过 $K=6$，不随原文位置增长。
依据：自行构造
:::

### 四个就够

初始 token 取几个，用消融（逐项改一个设置、看结果如何变化）确定。Llama-2-7B 上，缓存总长 4096，只留最近 token 的困惑度是 3359.95；加 1 个初始 token 降到 11.88，加 2 个到 10.51；加到 4 个是 9.59；再加到 8 个是 9.54，几乎不再变化。所以 4 个足够，模型不是只依赖首个 token 一个 sink。
依据：paper/src-v4/figtab/start_size_and_linebreak.tex L26-L42；paper/src-v4/text/4_experiment.tex L37

::: 其他模型上的初始 token 个数消融
初始 token 个数的消融在拼起来的 PG19 测试集 400K token 上测，缓存总长 2048。

| 缓存配置 | 0+2048 | 1+2047 | 4+2044 |
|---|---|---|---|
| Falcon-7B | 17.90 | 12.12 | 12.12 |
| MPT-7B | 460.29 | 14.99 | 14.99 |
| Pythia-12B | 21.62 | 11.95 | 12.09 |

依据：paper/src-v4/figtab/start_size_and_linebreak.tex L26-L37
:::

另一项消融看缓存大小：把缓存里最近 token 的数量从 252 加到 2044，困惑度并没有一路下降，Falcon-7B 在 4+1020 最低（12.34），到 4+2044 反而升到 12.84。
依据：paper/src-v4/figtab/cache_size.tex L5-L18；paper/src-v4/text/4_experiment.tex L39

::: 其他模型上的缓存大小消融
缓存大小的消融同样在 400K token 上测，每个配置都带 4 个开头 token。MPT-7B 随缓存变大反而更差，Pythia-12B 在 4+1020 之后几乎不变。

| 缓存配置 | 4+252 | 4+508 | 4+1020 | 4+2044 |
|---|---|---|---|---|
| MPT-7B | 14.12 | 14.25 | 14.33 | 14.99 |
| Pythia-12B | 13.17 | 12.52 | 12.08 | 12.09 |

依据：paper/src-v4/figtab/cache_size.tex L5-L18
:::


### 训练时留一个 sink

模型要用 4 个开头 token 当 sink，而不是 1 个，一个解释是预训练时没有统一的起始 token：每条样本最前面是各不相同的内容 token，模型只好拿若干个开头位置一起充当 sink。据此，预训练时在每条样本最前面放一个可学习的占位 token（它在所有样本里的位置都相同，像句首的固定标记），让它专门承接多余的注意力，模型就不必再拿内容 token 当 sink。
依据：paper/src-v4/text/3_method.tex L29；paper/src-v4/text/3_method.tex L44-L45

对比三个从零训练的 160M 参数模型：标准 SoftMax（Vanilla）、Zero Sink（见折叠块）和带可学 sink token 的模型。困惑度在 PG19 测试集的第一条样本上测。标准模型缓存里只留最近 1024 个 token（0+1024）时困惑度是 27.87，加 2 个以上开头 token 回到 18.05，只加 1 个是 18.49；带可学 sink token 的模型，1+1023 就是 18.01。
依据：paper/src-v4/text/3_method.tex L52；paper/src-v4/figtab/pretrain_fix.tex L6-L14

::: Zero Sink 的做法与数字
Zero Sink 把 SoftMax 换成 SoftMax-off-by-One：
依据：paper/src-v4/text/3_method.tex L44-L50

$$\text{SoftMax}_1(x)_i = \frac{e^{x_i}}{1+\sum_{j=1}^{N} e^{x_j}}$$

分母多加一项 $1$，各位置的权重之和可以小于 $1$，等价于在注意力里前置一个键和值全为零的 token。它仍要靠别的开头 token：0+1024 时困惑度是 29214，1+1023 是 19.90，2+1022 是 18.27，4+1020 是 18.01。带可学 sink token 的模型去掉这个 token（0+1024），困惑度是 1235。
依据：paper/src-v4/text/3_method.tex L44-L52；paper/src-v4/figtab/pretrain_fix.tex L6-L14
:::

加入 sink token 对模型收敛没有负面影响，各项基准上的成绩相近，各层各头的注意力都集中到这一个 token 上，分给其他开头 token 的变少。
依据：paper/src-v4/text/4_experiment.tex L14-L16；paper/src-v4/text/4_experiment.tex L23-L24

### 换模型、换长度、换任务都成立

在 Llama-2、Falcon、Pythia（都用 RoPE）和 MPT（用 ALiBi）四个模型族上测试。20K token 的文本上，dense attention 一超过预训练窗口就失效，窗口注意力一超过缓存大小就崩，保住 sink 的做法全程稳定，困惑度与带重算的滑窗几乎重合。文本再拉长，各个模型族、各种规模的模型都能稳定处理 400 万 token 以上。
依据：paper/src-v4/text/4_experiment.tex L3；paper/src-v4/text/4_experiment.tex L7-L8；paper/src-v4/figtab/nll_curve.tex L5-L11；paper/src-v4/figtab/long_lm.tex L5

::: 语言建模实验的条件
语言建模用拼起来的 PG19 测试集（100 本书），困惑度的起伏来自书与书之间的切换。Llama-2 的缓存取 2048，其余三个取 1024，都是预训练窗口的一半。
依据：paper/src-v4/text/4_experiment.tex L5；paper/src-v4/figtab/long_lm.tex L5
:::

在指令微调模型（在基础模型上再用问答、指令数据训练过的模型）的流式问答上，dense attention 显存溢出，窗口注意力超过缓存大小后输出变成随机内容，准确率很低，保住 sink 的做法与逐条单独输入的基线准确率相当。
依据：paper/src-v4/text/4_experiment.tex L30；paper/src-v4/figtab/instruct_qa.tex L14-L18

::: 流式问答的条件
把 ARC 的问答对拼成一条不间断的流，喂给 Llama-2 的 7B、13B、70B 对话模型，在每个答案位置用精确匹配判分。
依据：paper/src-v4/text/4_experiment.tex L30
:::

### 每 token 最多快 22.2 倍

保住 sink 的做法每步只在缓存内做注意力，开销随缓存大小线性增长；带重算的滑窗每步要重算窗口内的注意力，开销随缓存大小二次增长。缓存越大，差距越大，每 token 最多快 22.2 倍；显存占用两者相当。
依据：paper/src-v4/text/4_experiment.tex L44-L45；paper/src-v4/figtab/efficiency.tex L4

::: 效率实验的条件
两个方法都用 Huggingface Transformers 实现，在单张 A6000 上跑，模型是 Llama-2-7B 与 Llama-2-13B，横轴是缓存大小。
依据：paper/src-v4/text/4_experiment.tex L44；paper/src-v4/figtab/efficiency.tex L4
:::

### 适用条件与边界

它适用于采用相对位置编码的模型，做的是在缓存内生成连贯的文本。代价是只保留最近的 token：模型看不到被淘汰的中间内容，因此不适合长文档问答和摘要这类需要长期记忆的任务。在 StreamEval（流式问答的测试集，每行 23 个 token，答案在提问之前若干行）上，答案与提问的距离超过缓存大小时，准确率降到 $0$；距离在缓存之内，准确率也大体随距离增大而下降。
依据：paper/src-v4/text/3_method.tex L34；paper/src-v4/text/6_appendix.tex L7；paper/src-v4/text/6_appendix.tex L29；paper/src-v4/figtab/kv_lines.tex L3-L21

一条常见的误解是它扩大了模型的上下文窗口，或增强了长期记忆。都没有：上下文窗口（预训练时的序列长度）仍由预训练决定，Llama-2 仍是 4K。
依据：paper/src-v4/text/6_appendix.tex L7；paper/src-v4/text/1_introduction.tex L2；paper/src-v4/text/1_introduction.tex L31
