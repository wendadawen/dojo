# 语言模型预训练

- 描述：语言模型预训练是在大规模无标注文本上，用预测下一个 token（文本切分后的基本单位）的目标训练语言模型，得到能续写任意文本的基座模型（尚未做后续训练的模型）。
- 摘要：语言模型把文本序列的概率分解为逐步的条件概率 $p(s_1,\dots,s_n)=\prod_i p(s_i\mid s_{<i})$；预训练在语料上最小化每一步的负对数概率，不需要人工标签。
- 主题：训练与优化
- 标签：训练

## 资料

- Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI, 2019, [链接](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Touvron et al., LLaMA: Open and Efficient Foundation Language Models, arXiv:2302.13971v1, 2023, [链接](https://arxiv.org/abs/2302.13971v1)
- Ouyang et al., Training language models to follow instructions with human feedback, arXiv:2203.02155v1, 2022, [链接](https://arxiv.org/abs/2203.02155v1)
- Shannon, A Mathematical Theory of Communication, 1948, [链接](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

## 短页

### 一个在前文上预测下一个 token 的分布

语言模型在给定一段前文时，给出下一个 token 在整个词表上的概率。token 是文本被切分后的基本单位，可以是一个字，也可以是一个词；词表就是所有可能 token 的集合。前文是已经出现过的一串 token，模型的输出则是词表里每个 token 各有一个概率值，合起来构成一张概率分布。
依据：paper/llama-v1.txt L587-L596；paper/gpt2.txt L185-L194

### 为什么能在大量文本上训练

训练要的是「在这个位置，正确的下一个 token 是哪个」。这个答案文本本身就有：一段文本里，每个位置的下一个 token 就是紧接着的那个 token。因此不需要人工标注，直接使用原文即可训练。
依据：paper/gpt2.txt L106-L109

### 逐步预测、整句打分、最小化损失

一整句的概率被写成逐步条件概率的连乘：先给第一个 token 的概率，再给「在第一个 token 之后第二个 token 的概率」，依次到末尾，这些条件概率连乘就是整句概率。每一步取它的负对数作为该步的损失，概率越低损失越大；整句损失是各步之和，训练就是把语料上这些损失的平均值降低。
依据：paper/gpt2.txt L64-L70；paper/gpt2.txt L229-L231

## 长页

语言模型预训练是在大量文本上训练一个模型，让它在任意一段前文之后给出下一个 token（文本切分后的基本单位）的概率分布；训练不需要人工标注，因为文本里每个位置的下一个 token 本身就是答案。训练完成后得到基座模型，它把语料上的分布建了出来。
依据：paper/gpt2.txt L94-L98；paper/gpt2.txt L106-L114；paper/llama-v1.txt L587-L596；paper/gpt2.txt L569-L578

### 语言模型的输出：前文之上的下一个 token 分布

语言模型给出的是分布，不是某一句话。给定一段前文，它为下一个 token 在整张词表上各给一个概率。这里的 token 是文本被切分后的基本单位，可以是一个字，也可以是一个词；词表是所有可能 token 的集合，模型只在这张表上分配概率。
依据：paper/llama-v1.txt L587-L596；paper/gpt2.txt L185-L194

把前文记作 $s_1,\dots,s_{i-1}$，下一个 token 记作 $s_i$，模型给出的是条件概率 $p(s_i\mid s_1,\dots,s_{i-1})$：在已知前文的条件下，下一个 token 取 $s_i$ 的概率。对词表里每个 token 都有一个这样的值，它们非负且总和为 1，合起来是一张概率分布。这个「给定前文、预测下一个 token」的任务叫下一 token 预测。
依据：paper/llama-v1.txt L587-L596

#### 生成：把预测出的 token 接回前文再预测

模型一次只预测一个 token。要生成一段更长的文本，就把刚预测出的 token 接到前文后面，作为新的前文再预测下一个，如此重复。
依据：paper/gpt2.txt L64-L70；paper/gpt2.txt L374-L377

### 把一条序列的概率分解成逐步预测

一条序列 $s_1,\dots,s_n$ 一起出现的概率 $p(s_1,\dots,s_n)$ 叫联合概率。语言有自然的先后顺序，因此常把联合概率拆成每一步条件概率的连乘：

$$
p(s_1,\dots,s_n)=\prod_{i=1}^{n} p(s_i\mid s_1,\dots,s_{i-1}).
$$

依据：paper/gpt2.txt L64-L70；paper/gpt2.txt L98-L99

右边的每一项只依赖它前面的 token，这样的模型叫自回归模型。这样，把「给整条序列估一个概率」变成「在每个位置估一个条件概率」，后者的输入是前文、输出只是词表上的分布。对整条序列的概率（记作 $p(x)$）的采样和估计因此都变得可行。
依据：paper/gpt2.txt L64-L70；paper/gpt2.txt L73-L74

::: 链式分解的推导
从两个 token 开始，条件概率的乘法公式给出：

$$p(s_1,s_2)=p(s_1)\,p(s_2\mid s_1).$$
依据：自行构造

再加一个 token，分两步：

1. 先把它看成「前两个 token 之后再来一个」：$p(s_1,s_2,s_3)=p(s_1,s_2)\,p(s_3\mid s_1,s_2)$；
2. 把第一步的 $p(s_1,s_2)$ 代入，得 $p(s_1,s_2,s_3)=p(s_1)\,p(s_2\mid s_1)\,p(s_3\mid s_1,s_2)$。
依据：自行构造

照此每加一个 token 就乘上一个新的条件概率，展开到第 $n$ 项就是
$$p(s_1,\dots,s_n)=\prod_{i=1}^{n} p(s_i\mid s_1,\dots,s_{i-1}).$$
依据：paper/gpt2.txt L64-L70
:::

#### 逐步概率连乘的手算

三个 token 的序列，给定每一步的条件概率，整句概率就是它们的连乘。
依据：自行构造

::: 例子的完整过程
取词表里的三个 token 记作 $A,B,C$，一条序列为 $A,B,C$。模型在每一步给出的条件概率是：

1. 第一个 token：$p(A)=0.5$。
依据：自行构造
2. 第二个 token：$p(B\mid A)=0.8$。
依据：自行构造
3. 第三个 token：$p(C\mid A,B)=0.5$。
依据：自行构造

连乘得整句概率：
$$p(A,B,C)=0.5\times 0.8\times 0.5=0.2.$$
依据：自行构造
:::

#### 每一步的损失：负对数概率

连乘取对数后变成相加：$\log p(s_1,\dots,s_n)=\sum_i \log p(s_i\mid s_1,\dots,s_{i-1})$。每一步的条件概率不超过 1，取对数后不为正，概率越接近 1 对数越接近 0。取负号得到单步损失 $-\log p(s_i\mid s_1,\dots,s_{i-1})$：概率越低，这个值越大，也就是这一步「代价」越大。整句损失是各步损失之和。
依据：paper/gpt2.txt L229-L231；paper/shannon48.txt L801-L803

::: 上例各步的损失
```python
import math

ps = [0.5, 0.8, 0.5]
losses = [-math.log(p) for p in ps]
print([round(x, 4) for x in losses])
print(round(sum(losses), 4))
```

```text
[0.6931, 0.2231, 0.6931]
1.6094
```
:::
依据：自行构造

### 预训练目标：在语料上最小化平均负对数概率

预训练要模型对语料里每个位置的下一个 token 都给高概率。等价地，最小化下面的平均负对数概率：

$$
\mathcal{L}(\theta)=-\frac{1}{|D|}\sum_{x\in D}\frac{1}{|x|}\sum_{i=1}^{|x|}\log p_\theta(s_i\mid s_1,\dots,s_{i-1}).
$$

依据：paper/gpt2.txt L229-L231

$\theta$ 是模型的参数，训练调整的就是它。$D$ 是语料，用来训练的全部文本；$x$ 是其中一条序列，$|x|$ 是它的 token 数，$|D|$ 是序列条数。$s_i$ 是 $x$ 的第 $i$ 个 token，$i$ 从 1 数起。$p_\theta(\cdot\mid\cdot)$ 是模型在给定前文时给下一个 token 的概率。内层求和把一条序列各位置的负对数概率相加，再除以 $|x|$；外层对所有序列取平均。这个每步取负对数、越大代表越差的量就是[交叉熵](wiki/cross-entropy)。
依据：paper/gpt2.txt L229-L231

把这个损失最小化，等价于让模型在语料上最大化似然。
依据：paper/gpt2.txt L575-L577

#### 语料平均的口径

上面的损失把一条序列的负对数概率除以它的预测单元数，得到每个预测单元的平均负对数概率；预测单元通常取一个字符、一个字节或一个词。报告时经常把这个量再做缩放或指数化后使用。
依据：paper/gpt2.txt L228-L233

### 训练完成后得到的基座模型

训练做完，得到的是基座模型。它已经把语料上的分布建了出来：给它一段前文，它能给出下一个 token 的分布，因而能续写多种文本。它以尽可能高的似然拟合这份语料，因此在语料覆盖的多种文本上都能继续，而不限于某一个任务。
依据：paper/gpt2.txt L569-L578

基座模型还没有做后续的对齐训练（预训练之后、在人工示范数据上继续微调的那类训练）。
依据：paper/instructgpt-v1.txt L304-L306

### 适用条件与边界

这套目标只优化「预测下一个 token」，学到的行为是照语料往下续写。它与用户想要的「有用、安全地遵循指令」并不是同一件事：给定一个请求，续写出来的文本未必是那种回答，这个目标只是所要行为的一个近似。这个差异由训练目标决定，由后续训练处理。
依据：paper/instructgpt-v1.txt L74-L78；paper/instructgpt-v1.txt L1165-L1167
