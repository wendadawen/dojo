# Causal Mask

- 描述：因果掩码是解码器（逐步生成输出的那部分网络）自注意力（让每个位置看其他位置的一层）里的一张遮罩，它让每个位置只能关注它自己和它之前的位置，之后的位置被遮住。
- 摘要：把解码器自注意力分数矩阵的上三角遮住，位置 $i$ 的输出就只由位置 $\le i$ 的输入决定；自回归模型训练时因此能一次算出整条序列的所有位置。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 资料

- Vaswani, Shazeer, Parmar 等. Attention Is All You Need. NeurIPS 2017, arXiv:1706.03762v7. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

## 短页

### 只留它自己和它之前的位置

解码器是网络中逐步生成输出的那部分。它的一层[自注意力](wiki/standard-attention)（网络里的一层，处理整个序列）让序列的每个位置关注其他位置：每个位置从自己的输入得到查询、键、值三个向量，位置的输出是各位置的值按查询与键的匹配程度加权求和。因果掩码是加在这层自注意力上的一张遮罩，它把位置 $i$ 能看的位置限为它自己和它之前的位置；位置 $i$ 之后的位置，贡献被压到零。
依据：paper/src/model_architecture.tex L12；paper/src/model_architecture.tex Attention；paper/src/model_architecture.tex Scaled Dot-Product Attention；paper/src/model_architecture.tex Applications of Attention in our Model

### 不让模型提前看到答案

自回归生成一次只生成一个位置：先生成第 $1$ 个，再用它生成第 $2$ 个，依此类推。要生成第 $i$ 个，模型只能用已经生成的前面那些位置；如果位置 $i$ 还能看到位置 $i$ 之后的内容，它就等于提前拿到了后面要预测的答案。因果掩码遮住未来位置，保证位置 $i$ 只依赖已经生成的部分。
依据：paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex L12-L14

### 在被遮位置上把分数压到负无穷

每个位置的查询和所有位置的键算出分数，这些分数排成矩阵、每个位置一行；归一化按指数把每行的分数变成一组权重，非负且和为 $1$。因果掩码在归一化之前，把位置 $i$ 之后那些位置的分数设成 $-\infty$；按指数归一时 $-\infty$ 那一项为 $0$，它们的权重就是 $0$，位置 $i$ 的输出只由它自己和它之前的位置的值加权求和得到。
依据：paper/src/model_architecture.tex Scaled Dot-Product Attention；paper/src/model_architecture.tex Applications of Attention in our Model

## 长页

解码器是逐步生成输出的那部分网络，它每一层里都有一层[自注意力](wiki/standard-attention)，让每个位置关注别的位置。自注意力给每个位置一个查询、一个键和一个值，用查询与键的匹配程度当权重，对各位置的值加权求和。生成是自回归的：一次只生成一个位置，后面的位置以已经生成的内容为输入。
依据：paper/src/model_architecture.tex L12；paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention

因果掩码就是加在解码器自注意力上的一张遮罩，它把每个位置能看的位置限为它自己和它之前的位置，之后的都被遮住。需要它，是因为生成时位置 $i$ 只能依赖它之前的已知输出；如果位置 $i$ 能看到它之后的位置，就提前看到了后面还没生成的答案，自回归性质被破坏。它的做法是：先用查询和键算出分数，再把被遮位置的分数设成负无穷，把分数归一化成权重后这些位置的权重为 $0$，最后把各位置的值加权求和。
依据：paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention

### 掩码把哪些位置遮住

#### 一个位置能看哪些位置

把行看成查询所在的位置、列看成键所在的位置，位置 $i$ 的查询可以配位置 $j$ 的键，当且仅当 $j \le i$。满足这个条件的格子允许，其余遮住：允许的格子连成一个下三角（含对角线上那格），被遮的格子连成对角线上方那个不含对角线的上三角。允许集合只由两个位置的先后决定，与它们的内容无关。
依据：paper/src/model_architecture.tex Applications of Attention in our Model

::: 长度 4 的序列里，哪些位置允许
行是查询位置、列是键位置，允许写「是」。
依据：自行构造

| 查询＼键 | 位置 1 | 位置 2 | 位置 3 | 位置 4 |
|---|---|---|---|---|
| 位置 1 | 是 | 遮 | 遮 | 遮 |
| 位置 2 | 是 | 是 | 遮 | 遮 |
| 位置 3 | 是 | 是 | 是 | 遮 |
| 位置 4 | 是 | 是 | 是 | 是 |
:::

### 掩码怎么算进去

#### 把被遮位置的分数设成负无穷

一个位置的查询和所有位置的键作点积，得到它对这些位置的分数；在解码器自注意力里，把落在上三角（非法连接）的分数设为 $-\infty$。归一化按指数把一行分数变成非负、和为 $1$ 的权重，$-\infty$ 那一项归一到 $0$，于是被遮位置的权重为 $0$，只剩位置 $\le i$ 的贡献。
依据：paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention

::: 一个位置上走一遍：长度 4 的序列，看位置 2
设序列有四个位置，四个值分别是 10、20、30、40；查询与键都取作一维的数 1，于是任意查询与键的点积都是 1。
依据：自行构造

1. 位置 2 的查询与四个键各作点积，得到四个分数 1、1、1、1。
2. 掩码：位置 3、4 的分数被设为 $-\infty$，得 1、1、$-\infty$、$-\infty$。
3. 归一化：两个未被遮的分数相等，权重各占一半，即 0.5；被遮两个的权重为 0。
4. 加权求和：$0.5 \times 10 + 0.5 \times 20 = 15$，位置 2 的输出只由前两个位置的值决定。
依据：自行构造

```python
import math

queries = [1.0, 1.0, 1.0, 1.0]
keys = [1.0, 1.0, 1.0, 1.0]
values = [10.0, 20.0, 30.0, 40.0]

for i in range(4):
    scores = [queries[i] * keys[j] for j in range(4)]
    scores = [s if j <= i else float("-inf") for j, s in enumerate(scores)]
    top = max(s for s in scores if s != float("-inf"))
    exps = [0.0 if s == float("-inf") else math.exp(s - top) for s in scores]
    total = sum(exps)
    weights = [round(e / total, 2) for e in exps]
    out = sum((exps[j] / total) * values[j] for j in range(4))
    print(f"位置{i + 1} 权重 {weights} 输出 {out}")
```

```text
位置1 权重 [1.0, 0.0, 0.0, 0.0] 输出 10.0
位置2 权重 [0.5, 0.5, 0.0, 0.0] 输出 15.0
位置3 权重 [0.33, 0.33, 0.33, 0.0] 输出 20.0
位置4 权重 [0.25, 0.25, 0.25, 0.25] 输出 25.0
```
依据：自行构造
:::

#### 负无穷与实际用的有限负数

把被遮位置的分数设成 $-\infty$，归一化后它的权重恰好为 $0$。如果改设为一个很大的负数 $-B$，被遮位置的权重就不是 $0$，而是趋近 $0$：$B$ 越大越接近 $0$。
依据：自行构造

::: 为什么有限负数只让权重趋近 0
设一行里被遮位置之外的分数最大为 $m$，被遮位置设为 $-B$，$B>0$。
依据：自行构造

1. 归一化把第 $j$ 个位置的权重写成 $e^{s_j - m} / \sum_k e^{s_k - m}$，其中 $s_j$ 是它的分数。这一形式对加同一个常数不变，$m$ 只是让指数不溢出。
依据：自行构造
2. 被遮位置的分数是 $-B$，它的权重等于 $e^{-B - m}$ 除以所有位置该项之和。
依据：自行构造
3. 分母里至少有一项是未被遮位置的 $e^{s_j - m} \ge 1$，所以被遮位置的权重不超过 $e^{-B-m}$。
依据：自行构造
4. $B$ 越大 $e^{-B-m}$ 越小，权重趋近 $0$ 但不等于 $0$；只有取 $-\infty$ 时才是 $0$。
依据：自行构造
:::

### 掩码换来什么

#### 训练时一次算出所有位置

训练时解码器的输入是整条目标序列右移一位：位置 $i$ 上放的是正确答案里它前面那部分已经生成的输出，位置 $1$ 之前没有输出。掩码与这个偏移一起，保证位置 $i$ 的预测只依赖它之前的已知输出，各位置之间就没有先后依赖，可以一次并行算出。
依据：paper/src/model_architecture.tex Decoder:；paper/src/why_self_attention.tex L26；paper/src/why_self_attention.tex L84

#### 解码时逐位置追加

解码时还没有目标序列，只能一次生成一个位置（一个词）。已经生成的位置不被遮，还没生成的位置被遮住，新位置的自注意力只用到已经生成的那部分，用它的输出再预测下一个位置。
依据：paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex L12-L14

### 与相邻概念的区别

和因果掩码最容易混淆的是填充掩码（padding mask）：一段补齐到固定长度时，多出来的位置只是占位。填充掩码遮的是这些占位位置，与位置的先后无关；因果掩码遮的是位置之间的先后，也就是未来位置，与内容有没有无关。两者可以用在同一行上。
依据：自行构造

### 适用条件与边界

因果掩码只用在解码器的自注意力里。解码器看编码器输出（编码器把输入序列编码成表示）的那层注意力、以及编码器自己的自注意力（每个位置本来就可以看前一层所有位置），都不加这张遮罩。另有一个容易漏的限定：做到「位置 $i$ 只依赖它之前的已知输出」，靠的是掩码与目标序列右移一位这两件事一起，掩码只遮住同一次计算里排在当前位置之后、越出可见范围的那些位置。
依据：paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex Scaled Dot-Product Attention；paper/src/why_self_attention.tex L26

掩码不改变分数矩阵的规模：它仍是按全部位置对算出来的 $n \times n$，$n$ 是序列长度。代价在于被遮的位置不再把信息传给当前位置，它本身并不减少计算量。常见误解有两条：一是把它当成能减少计算或加快训练的手段；二是以为它只在逐步生成时才需要，其实训练时同样要遮，否则位置 $i$ 会直接看到它要预测的答案，自回归的假设不再成立。
依据：自行构造
