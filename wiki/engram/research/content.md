# Engram

- 描述：Engram 是给大语言模型加的条件记忆模块：用最近几个 token 的哈希去查一张大嵌入表，取回静态向量，经门控后加回隐藏状态。
- 摘要：用后缀 $N$-gram 的哈希在 $O(1)$ 时间里查大嵌入表，把实体、固定搭配这类静态模式交给查表，计算层留给推理。
- 主题：模型结构, 内存与缓存
- 标签：模型架构

## 资料

- Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai 等. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372v2, 2026. [https://arxiv.org/abs/2601.07372](https://arxiv.org/abs/2601.07372)
- DeepSeek-AI. Engram: official implementation. 2026. [https://github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

## 短页

### 一张按 N-gram 查的大表

Engram 是一种条件记忆模块。它给大语言模型外挂一张嵌入表，表里每一行是一个可训练的向量，地址由位置 $t$ 前面几个 token 的 ID 算哈希得到：$N$-gram 指连续 $N$ 个 token 组成的片段，以 $t$ 结尾的那一段就叫后缀 $N$-gram。叫「条件」是因为哪几行被取用由当前位置的局部 token 决定，而不是每行都参与计算。取回的向量不随上下文变化，经一个门控后加回这一层的隐藏状态。它不替换主干里任何一层，只是在一部分层上多接一条支路。
依据：paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/arch.tex Overview

### 模型在用计算模拟查表

大语言模型的主干是一叠 Transformer 层，每层传给下一层的是隐藏状态。一层里主要有两部分：注意力在位置之间传信息，让每个位置能看到前面位置的内容；前馈层逐个位置各算各的。前馈部分常换成混合专家（MoE）：前馈层变成多个并行的专家子网络，每个 token 只激活其中几个，参数总量因此可以远大于每个 token 用到的计算量。这种结构扩展的是计算，模型里没有为这类静态模式准备的原生查表操作。于是文本里的实体和固定搭配只能靠计算一点点拼出来：认出「Diana, Princess of Wales」这样一个多 token 实体，要用掉靠前的若干层注意力和前馈网络，逐层把国名、「Princess of Wales」这样的片段拼成一个整体。这些模式本身是静态的，本来可以一次查表取回。
依据：paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/analysis.tex Is Engram functionally equivalent

### 压缩、哈希、门控、预取

检索只用到 token 的 ID。先把词表压缩，把拼写等价、只是大小写或重音不同的 token 合成同一个 ID；再取最近 $n$ 个 token 组成的后缀 $N$-gram，$n$ 就是阶。同一个 gram 会用多个哈希头各算一个地址，每个头是一张独立的表，查到向量后拼成一个记忆向量。静态向量会碰撞，也会一词多义，所以再加一层按当前上下文算的门：用隐藏状态当查询、记忆向量经矩阵 $\mathbf{W}_K$ 投影出的键算出一个 0 到 1 之间的标量门 $\alpha_t$，再乘到记忆向量经 $\mathbf{W}_V$ 投影出的值向量上。门控后的结果过一层短因果卷积（沿序列方向，只对当前位置及其前面几个位置加权求和），再以残差加回隐藏状态。因为地址只由 token 决定，可以在算到这一层之前就把向量取来，把搬运和计算重叠，表因此可以放在显存之外。
依据：paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed；repo/engram/engram_demo_v1.py L72-L77；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/arch.tex System Efficiency: Decoupling

## 长页

大语言模型的主干是一叠结构相同的 Transformer 层，层数常到几十。每层收到一个张量叫隐藏状态 $\mathbf{H}^{(\ell)}$，形状是序列长度乘隐藏维度，每个位置一个向量，$\ell$ 是层的编号。层算出的输出以残差连接加回输入，写成 $\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \text{层的输出}$，所以每层是在已有表示上加一点东西。多数现代模型还会把前馈层换成混合专家（MoE）：前馈层变成多个并行的专家子网络，每个 token 只激活其中少数几个，参数总量因此可以远大于每个 token 实际参与计算的参数量。
依据：paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/intro.tex Introduction

模型里还有一张嵌入表：行数等于词表大小，每行一个可训练向量，token 的整数 ID 就是取哪一行的地址，序列开头就是这么变成向量的。Engram 用同样的形式再挂一张表，地址不再由单个 token 的 ID 给出，而由最近几个 token 的哈希给出，取回的向量经门控后加回隐藏状态。表里的向量不随上下文变化，论文把这类机制叫条件记忆，用来和 MoE 那种按隐藏状态条件激活参数的条件计算并列。
依据：paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/arch.tex Overview

需要这么一个模块，是因为文本里有两类性质不同的子任务。需要推理的部分要深的、动态的计算；而实体名、固定搭配这类文本是局部的、静态的、高度定型的，本来可以一次查表取回。$N$-gram 指连续 $N$ 个 token 组成的片段，$N$ 叫阶；经典 $N$-gram 模型之所以能覆盖这类依赖，就是因为它们把局部上下文当键去索引一张大表。标准 Transformer 没有这种按 ID 取一行的操作，只能模拟：模型用计算一层层重建本该查表得到的静态向量。
依据：paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/analysis.tex L9-L49

Engram 在少数几层的注意力与 MoE 之前加一条支路，先按 token 序列从表里取回静态向量，再按当前上下文把这批向量调节后加回隐藏状态。
依据：paper/src-v2/sections/arch.tex Overview

::: 模型逐层拼出一个实体的过程
让模型处理「Diana, Princess of Wales」，看实体末尾那个 token（Wales）在各层被映回文本时，得到的解释是这样的。映回文本用的是 PatchScope，它为该 token 在各层逐层生成一段文本，读出那一层「已经认出了什么」。
依据：paper/src-v2/sections/analysis.tex L14-L49

1. 第 1 到 2 层：片段只到「Country in the United Kingdom」，知道是英国的一部分。
2. 第 3 层：变成「Country in Europe」，范围收窄到欧洲。
3. 第 4 层：变成「Title held by female sovereigns in their own right or by queens consort」，头衔的一般含义出来了，还不知道指谁。
4. 第 5 层：变成「Title given to the wife of the Prince of Wales」，头衔的含义完整了，指的人仍不确定。
5. 第 6 层：才拼出「Diana, Princess of Wales」这个实体本身。
依据：paper/src-v2/sections/analysis.tex L24-L45
:::

### 检索：从 token 到记忆向量

检索这一步把每个位置的局部上下文映射成表里的静态向量，顺序是：先把等价 token 压成同一个 ID，再取以该位置结尾的后缀 $N$-gram，对每个阶用多个哈希头各算一个表内的地址，最后把各头查到的行拼成一个记忆向量。
依据：paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed

#### 词表压缩

子词分词器优先保证无损还原，同一个词的大小写、空格、重音差异会拿到不同的 ID，例如 `Apple` 和带前导空格的 `apple` 是两个 token，`a` 和 `á` 也是两个。这些 token 语义上是同一个词，分开处理等于把同一行记忆切成多行。压缩的做法是预先算一个满射函数，把原始 ID 按规范化的文本等价折成同一批规范 ID：先做 NFKC 归一，把字形不同但等价的字符、全角与半角、兼容字符折成同一种形式；再把带重音的字母拆开、去掉重音符号；最后转成小写。压缩之后，位置 $t$ 的后缀 $N$-gram 用规范 ID 组成 $g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$，其中 $x'_t$ 是 $x_t$ 压缩后的 ID。
依据：paper/src-v2/sections/arch.tex Tokenizer Compression；repo/engram/engram_demo_v1.py L72-L77

在 128k 词表的分词器上，压缩把有效词表缩小 23%。
依据：paper/src-v2/sections/arch.tex Tokenizer Compression

::: 合并最多的几类 token
按被合并的原 token 数量排前五的规范 token 及其典型来源。
依据：paper/src-v2/main.tex Case Study of Tokenizer Compression

1. 空白类，合并 163 个：制表符、换行符、回车符、单个到多个空格等。
2. 字母 a，合并 54 个：`A`、`a`、带前导空格的写法，以及 á、ä、ã、ą 等带重音的写法。
3. 字母 o，合并 40 个：`O`、`o`、带前导空格的写法，以及 ó、ö、ô、õ 等。
4. 字母 e，合并 35 个：`E`、`e`、带前导空格的写法，以及 é、è、ę、ě 等。
5. 字母 i，合并 30 个：`I`、`i`、带前导空格的写法，以及 í、ì、î、ī 等。
依据：paper/src-v2/main.tex Case Study of Tokenizer Compression
:::

#### 后缀 N-gram 与多头哈希

把所有 $N$-gram 都各自参数化不可行：$N$ 阶 $N$-gram 的组合空间共有 $|V|^N$ 种，$|V|$ 是词表大小，随阶数指数增长。做法是哈希，把每个后缀 $N$-gram 映到若干张固定大小的表里。为了减少碰撞，每个阶 $n$ 用 $K$ 个哈希头，第 $k$ 个头用确定性函数 $\phi_{n,k}$ 把 $g_{t,n}$ 映到表 $\mathbf{E}_{n,k}$ 的一行 $z_{t,n,k}$，这张表的行数 $M_{n,k}$ 取素数。各阶、各头查到的向量拼成这个位置的记忆向量 $\mathbf{e}_t$。论文实现里的 $\phi_{n,k}$ 是一个轻量的乘法-异或哈希；配套的演示代码把「每张表长度取素数」具体化为各头取互不相同的素数。
依据：paper/src-v2/sections/arch.tex Multi-Head Hashing；repo/engram/engram_demo_v1.py L188-L296

::: 小表上走一遍哈希
取一段序列 1、1、1、3（4 个 token），只用 2-gram 和 3-gram，每阶 2 个头，表长取素数：2-gram 的两张表长度 7 和 11，3-gram 的两张表长度 13 和 17。哈希的算法是：gram 里每个 token 乘一个自己的系数（当前 token 乘 5，前一位乘 3，前两位乘 7），从当前 token 起逐个把乘积异或进去，最后对表长取余。序列开头不足的位按 0 补。逐个位置算。
依据：自行构造

1. 位置 1，2-gram 是 (0, 1)：当前 token 是 $1$，$1 \times 5 = 5$，前一位是 $0$，$0 \times 3 = 0$，$5 \oplus 0 = 5$，两个头分别得 $5 \bmod 7 = 5$、$5 \bmod 11 = 5$。
2. 位置 2，2-gram 是 (1, 1)：$1 \times 5 = 5$，$1 \times 3 = 3$，$5 \oplus 3 = 6$，得 $6$、$6$。
3. 位置 3 的 2-gram 与位置 2 相同，哈希仍是 $6$，两个头仍得 $6$，两处取到同一行。
4. 位置 4，2-gram 是 (1, 3)：$3 \times 5 = 15$，$1 \times 3 = 3$，$15 \oplus 3 = 12$，得 $12 \bmod 7 = 5$、$12 \bmod 11 = 1$。位置 1 的 (0, 1) 在头 1 也落在第 5 行，头 2 把两者分到第 5 行和第 1 行，两个头合起来不再相同。
5. 3-gram 在 2-gram 的结果上再异或两个位置之前那个 token 的乘积：位置 1 的 (0, 0, 1) 得 $5 \oplus 0 = 5$，位置 4 的 (1, 1, 3) 得 $12 \oplus 7 = 11$。
依据：自行构造

```python
def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def find_next_prime(start, seen):
    candidate = start + 1
    while candidate in seen or not is_prime(candidate):
        candidate += 1
    return candidate


def head_sizes(size, heads, seen):
    out, start = [], size - 1
    for _ in range(heads):
        p = find_next_prime(start, seen)
        seen.add(p)
        out.append(p)
        start = p
    return out


seen = set()
sizes = {2: head_sizes(7, 2, seen), 3: head_sizes(12, 2, seen)}
mults = [5, 3, 7]
seq = [1, 1, 1, 3]

for t in range(len(seq)):
    print("位置", t + 1)
    for n in (2, 3):
        tokens = [seq[t] if k == 0 else (seq[t - k] if t - k >= 0 else 0) for k in range(n)]
        mix = tokens[0] * mults[0]
        for k in range(1, n):
            mix ^= tokens[k] * mults[k]
        rows = [mix % p for p in sizes[n]]
        print(" ", n, "gram", tuple(reversed(tokens)), "mix", mix, "行", rows)
```

```text
位置 1
  2 gram (0, 1) mix 5 行 [5, 5]
  3 gram (0, 0, 1) mix 5 行 [5, 5]
位置 2
  2 gram (1, 1) mix 6 行 [6, 6]
  3 gram (0, 1, 1) mix 6 行 [6, 6]
位置 3
  2 gram (1, 1) mix 6 行 [6, 6]
  3 gram (1, 1, 1) mix 1 行 [1, 1]
位置 4
  2 gram (1, 3) mix 12 行 [5, 1]
  3 gram (1, 1, 3) mix 11 行 [11, 11]
```

代码里的 $mults$ 是各 token 的乘法系数，$sizes$ 是各阶、各头的表长，$seq$ 是那段 token 序列。位置 1 和位置 4 的两个 2-gram 在第 1 个头撞到同一行，第 2 个头把它们分开。演示代码给每个头取的表长互不相同。
依据：repo/engram/engram_demo_v1.py L181-L296
:::

这个向量只由 token 序列决定，与隐藏状态无关，所以同一个片段在所有位置、所有样本上都取到同一个向量。
依据：paper/src-v2/sections/arch.tex Multi-Head Hashing

### 融合：门控、卷积与残差

#### 上下文门控

查回的 $\mathbf{e}_t$ 是与上下文无关的先验：同一个 $N$-gram 在任何语境下取到同一行。哈希会碰撞，同一个词也可能在该位置有不同含义，所以这批向量需要按当前语境筛一遍。门控借注意力的形式来做：注意力里每个位置用一个查询向量去和各个键做点积，按点积大小把对应的值加权取回；这里当前隐藏状态 $\mathbf{h}_t$ 当查询，它经过前面若干层注意力，已经读到了整段上下文，拿它来问这批记忆合不合当前语境正合适。$\mathbf{e}_t$ 经矩阵 $\mathbf{W}_K$、$\mathbf{W}_V$ 投影出键 $\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t$ 和值 $\mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t$，$\mathbf{k}_t$ 和查询点积大就说明两者方向一致。门是一个标量：
依据：paper/src-v2/sections/arch.tex Context-aware Gating

$$\alpha_t = \sigma\left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right)$$
依据：paper/src-v2/sections/arch.tex Context-aware Gating

$\sigma$ 是 sigmoid，$\sigma(x) = 1/(1+e^{-x})$，$d$ 是隐藏维度，RMSNorm 是均方根归一化，把向量除上自己的均方根。取回的向量与当前语境不合时，点积为负，门趋近 0，噪声被压掉；门控后的输出是 $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$。
依据：paper/src-v2/sections/arch.tex Context-aware Gating

::: 两个向量的小例子
取隐藏维度 $d=4$，隐藏状态 $\mathbf{h} = (1, 2, -1, 0)$，投影后的记忆向量取三种，看门怎么动。
依据：自行构造

1. 与语境同向的 $\mathbf{k} = (2, 3, -1, 1)$：两者的 RMSNorm 点积是 $3.79$，除以 $\sqrt{4}$ 得 $1.90$，$\alpha = \sigma(1.90) = 0.87$，记忆基本通过。
2. 与语境相反的 $\mathbf{k} = (-1, -2, 1, 0)$：点积是 $-4.00$，除以 $2$ 得 $-2.00$，$\alpha = \sigma(-2.00) = 0.12$，记忆被压到八分之一上下。
3. 与语境正交的 $\mathbf{k} = (2, -1, 0, 3)$：点积是 $0$，$\alpha = \sigma(0) = 0.5$，既不放大也不抑制。
依据：自行构造
:::

这个门在两处打开：多 token 的实体名和固定搭配。英文里是「Alexander the Great」「the Milky Way」这类实体，以及「By the way」这类套语；中文里是「四大发明」「张仲景」这样的成语和历史实体。门的值在这类模式结束的位置上明显偏高。
依据：paper/src-v2/sections/analysis.tex Case Study: Gating Visualization

#### 短卷积、残差与训练

门控后每个位置的向量，再过一层沿序列方向的短卷积：取当前位置以及往前按固定间隔跳开的共 4 个位置（例如序列最长用 3-gram 时，位置 $t$ 取 $t$、$t-3$、$t-6$、$t-9$），各乘一个可学习的权重求和，且只看前面的位置（因果），间隔取最大的 $N$-gram 阶数（膨胀），最后过一个 SiLU 激活函数（一种平滑的非线性，$\mathrm{SiLU}(x) = x \cdot \sigma(x)$）。输出写成 $\mathbf{Y}$，再以残差加回，$\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$。卷积扩大感受野，让不相邻位置的门控结果也能互相影响。加回的位置在标准注意力和 MoE 之前，这两部分的输入因此带上检索到的静态信息。
依据：paper/src-v2/sections/arch.tex Context-aware Gating

表里的向量和几个投影矩阵都随模型一起训练，不需要单独的预训练阶段：表的每一行像词嵌入一样跟着任务损失（模型的训练目标）一起更新，$\mathbf{W}_K$、$\mathbf{W}_V$ 也在反向传播里更新。优化方式与主干不同：表的参数用 Adam（一种按梯度动量和二阶矩自适应调整步长的优化器），学习率放大 5 倍，不加权重衰减（一种把参数往零拉的惩罚）。
依据：paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations

::: 初始化与梯度分发
卷积参数初始化为零，训练开头门控支路输出为零，相当于恒等映射，主干的训练不受扰动。
依据：paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations

表的参数量远大于单个设备，训练时按模型并行（把一份参数分到多张卡上）的方式分片放在多张卡上：前向时用 All-to-All 通信（各张卡互相收发自己缺的那部分）取出各卡上被访问的行，反向时用同样的通信把梯度送回各行所在的卡。
依据：paper/src-v2/sections/arch.tex System Efficiency: Decoupling
:::

### 插在哪几层，多分支怎么接

Engram 不插在每一层。插得早，静态模式在主干花深度重建之前就被接走；插得晚，算门用的隐藏状态已经聚了更多上下文，门更准。固定 1.6B 的记忆，把这 1.6B 全部装进某一层的同一个 Engram 模块（单个模块），只改插在第几层，逐层比较验证损失（12 层 3B 的 MoE）。结果是第 2 层最好，第 1 层次之，越往深越差，说明一轮注意力已经够给门提供有意义的上下文。单模块最优时验证损失 1.770，主干基线是 1.808。把同样的 1.6B 拆成两个较小的模块放在第 2 层和第 6 层更好，验证损失 1.768，早插与晚插各自的优势都能拿到。
依据：paper/src-v2/sections/analysis.tex Where should memory be injected

主干这里换成一种多分支结构（论文默认的骨干，Engram 本身与它无关，只是要接上去得改一处）：残差流不再是一条，而是扩成 $M$ 条并行的分支，同一次前向里每条分支各自带着一份隐藏状态往下走，分支之间按可学习的连接权重互相传信息，最后各分支的表达加权求和。接进去时，每个分支用独立的键投影矩阵，各自算一个门（这叫分支专属融合），值投影和那张记忆表由所有分支共享。再各自去掉一个部件比较：去掉分支专属融合、去掉上下文门控、去掉词表压缩，这三种改动带来的验证损失退化同属最大的一档。
依据：paper/src-v2/sections/analysis.tex Which components matter；paper/src-v2/sections/arch.tex Integration with Multi-branch Architecture

::: 共享与合批的细节
第 $m$ 个分支的隐藏状态 $\mathbf{h}_t^{(m)}$ 算自己的门 $\alpha_t^{(m)}$，取回的记忆用同一个值向量调制：$\mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t)$。一个 $\mathbf{W}_V$ 与 $M$ 个 $\mathbf{W}_K^{(m)}$ 可以合成一次矩阵乘法，用 FP8 精度算（一种 8 位浮点数，比常用的 16 位更快）。实验里的多分支取 $M=4$。
依据：paper/src-v2/sections/arch.tex Integration with Multi-branch Architecture
:::

### 稀疏参数怎么分给专家和记忆

Engram 的表只在被查到时参与计算，所以它和专家一样属于稀疏参数：总参数 $P_{\mathrm{tot}}$ 减去每 token 激活的参数 $P_{\mathrm{act}}$，剩下的 $P_{\mathrm{sparse}}$ 是不参与每个 token 计算的那部分预算。在固定 $P_{\mathrm{tot}}$ 和 $P_{\mathrm{act}}$ 的前提下，分配比 $\rho$ 定义为这块预算分给 MoE 专家的比例，$P_{\mathrm{MoE}}^{(\mathrm{sparse})} = \rho P_{\mathrm{sparse}}$，剩下的 $1-\rho$ 归 Engram 表。$\rho = 1$ 就是纯 MoE，$\rho < 1$ 是把专家数减下来、把省出的参数换成表的槽位（槽位就是表的一行，一个可训练向量）。
依据：paper/src-v2/sections/scaling_law_exp.tex Optimal Allocation Ratio

用两个计算预算扫 $\rho$，验证损失对 $\rho$ 呈 U 形：$\rho$ 接近 1 时模型缺少存放静态模式的地方，只能靠深度重建；$\rho$ 接近 0 时模型失去条件计算能力，需要动态推理的任务变差，记忆在这一端替代不了计算。最低点在 $\rho$ 约 75%–80%，并且这个位置在两个计算预算下都稳定。把 $\rho$ 压到约 40%（也就是分给 MoE 专家的稀疏参数只剩四成）时，模型仍能追平纯 MoE，说明两者在功能上互补而不是互相挤占。
依据：paper/src-v2/sections/scaling_law_exp.tex Results and Analysis

::: 两个计算预算下的数字
$C = 2\times10^{20}$ FLOPs 一档：$P_{\mathrm{tot}} \approx 5.7$B，$P_{\mathrm{act}} = 568$M，纯 MoE 有 106 个专家。$C = 6\times10^{20}$ FLOPs 一档：$P_{\mathrm{tot}} \approx 9.9$B，$P_{\mathrm{act}} = 993$M，纯 MoE 有 99 个专家。两档的稀疏度都维持在 $P_{\mathrm{tot}} / P_{\mathrm{act}} \approx 10$。
依据：paper/src-v2/sections/scaling_law_exp.tex Optimal Allocation Ratio

在 10B 那一档，验证损失从 $\rho = 100\%$ 的 1.7248 降到最优附近的 1.7109。
依据：paper/src-v2/sections/scaling_law_exp.tex Results and Analysis
:::

把存储和计算分开之后，记忆可以单独放大。固定主干、只增加表的槽位数，验证损失随槽位数在双对数坐标上线性下降，即每翻一倍槽位带来的损失改善大致固定，这部分不需要额外计算。
依据：paper/src-v2/sections/scaling_law_exp.tex Engram under Infinite Memory Regime

### 确定性地址带来的预取与卸载

MoE 的路由要等隐藏状态算出来才知道把 token 送给哪些专家，Engram 的地址只看输入 token 的 ID，下一个位置要读哪些行在算到这一层之前就能算出来。这件事让表可以放在 GPU 显存之外：推理时由主机内存经 PCIe 异步取向量，与前面那些层的计算重叠，GPU 不必等。插得越深，前面层留下的计算窗口越长，越能掩盖搬运延迟；但建模效果偏好早插。
依据：paper/src-v2/sections/arch.tex System Efficiency: Decoupling

训练时表按模型并行分片放在多张卡上，反向的梯度用 All-to-All 送回各行所在的卡，总容量随加速器数量线性增长。推理侧再按访问频率分层：自然文本的 $N$-gram 服从齐夫分布（少数模式占绝大多数出现次数），少数高频模式占大多数访问，把它们留在快存储（如 GPU 显存或主机内存），长尾放在慢而大的存储（如固态盘）。
依据：paper/src-v2/sections/arch.tex System Efficiency: Decoupling

::: 显存外一张比模型大得多的表，吞吐损失多少
把一张 100B 参数的表（远大于 4B、8B 的模型本身）整个放进主机内存，插进第 2 个 Transformer 块，不做分层缓存，强制每次检索都走 PCIe。测试用 512 条序列、长度在 100 到 1024 之间均匀分布，硬件是 H800。
依据：paper/src-v2/sections/analysis.tex L168-L184

1. 4B 密集模型的基线吞吐是 9031.62 tok/s。
2. 同一个模型加上 100B 的表并卸载到主机内存，吞吐 8858.28 tok/s。
3. 8B 密集模型的基线吞吐是 6315.52 tok/s。
4. 加表之后是 6140.02 tok/s，跌幅 2.8%，是两组里最大的一组。
依据：paper/src-v2/sections/analysis.tex L190-L196

5. 每次取回的通信量只随实际访问的槽位数变化，不随表的规模变化。
依据：paper/src-v2/sections/analysis.tex L202-L206
:::

### 查表在模型里起了什么作用

逐层用 LM Head（把隐藏状态映成词表上分数的输出层）投影隐藏状态、和模型最后一层的输出分布比 KL 散度（两个分布的差距），Engram 模型的散度在靠前的层就更低，说明它更早给出可信预测。用 CKA 比表示相似度，CKA 比较的是两组表示排成的样本间相似度矩阵的一致程度：Engram 靠前的层与基线更深的层对齐，Engram 第 5 层的表示与基准 MoE 大约第 12 层的最接近。查表把早期用于拼接的特征计算省掉，等效于把网络加深。
依据：paper/src-v2/sections/analysis.tex Is Engram functionally equivalent

推理时把 Engram 的输出整体置零、主干不动，两类任务的表现差别明显：事实类任务的得分只保住原来的 29%–44%，阅读理解类保住 81%–93%。事实类严重依赖这张表，而落在上下文里的读懂任务主要靠注意力。
依据：paper/src-v2/sections/analysis.tex Sensitivity Analysis

把局部依赖交给查表之后，注意力可以腾出来处理全局上下文。长上下文任务上，在预训练损失对齐的设定里，Engram 在一批复杂的检索任务上胜过基线：多查询的大海捞针（在长文里找出一处事实，一次问多个）97.0 对 84.2，变量追踪（盯住某个变量，看它后来的取值）87.2 对 77.0。
依据：paper/src-v2/sections/long_context_exp.tex Long Context Training

::: 关掉 Engram 的两端与条件
这个消融是推理时把检索输出压掉，会引入训练与推理的不一致，在混合能力的任务上噪声较大，所以只看功能光谱的两端。
依据：paper/src-v2/sections/analysis.tex Sensitivity Analysis

事实类里 TriviaQA 保留 29%，阅读理解类里 C3 保留 93%。
依据：paper/src-v2/sections/analysis.tex Sensitivity Analysis
:::

### 适用条件与边界

按 1.6B 的固定记忆预算，把容量分给 4-gram 略差于 2、3-gram，原因大概是稀释了更常见的 2、3-gram 的容量；在更大的记忆规模上高阶 $N$-gram 是否会变好，资料没有下结论。记忆也有边界：$\rho$ 接近 0、稀疏参数几乎全给记忆时，需要动态推理的任务会变差，这一端记忆替代不了计算。
依据：paper/src-v2/sections/analysis.tex Which components matter；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis

一条常见的误解是「记忆多了就不需要计算」：把稀疏预算几乎全部给表，效果反而变差，两者是互补关系。另一条是「这类模块只对知识类任务有用」：关掉 Engram 时事实类掉得最狠，说的是这些任务对表的依赖程度；而在参数量和计算量都对齐的对比里，推理、代码与数学领域的提升幅度比知识类更大，说的是它相对基线带来的增益。依赖程度和提升幅度是两件事。
依据：paper/src-v2/sections/scaling_law_exp.tex Results and Analysis；paper/src-v2/sections/large_scale_exp.tex Experimental Results
