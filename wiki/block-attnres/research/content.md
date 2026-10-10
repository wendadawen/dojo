# Block AttnRes

- 描述：Block AttnRes 是把 Transformer 的残差连接换成按深度做注意力的做法：层分成若干块，每层用一个可学查询对前面各块的块级表示做 softmax 加权，块内仍用普通残差相加。
- 摘要：把残差里各层输出的等权相加，换成每层按相关度对前面各块表示加权取回；层分成 $N$ 块，内存与通信从 $O(Ld)$ 降到 $O(Nd)$。
- 主题：模型结构, 注意力机制
- 标签：模型架构

## 资料

- Kimi Team. Attention Residuals. Technical Report, 2026. [https://arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)
- Kimi Team. Attention Residuals: official repository. 2026. [https://github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

## 短页

### 每层在深度上取一次信息

神经网络由很多层叠起来，每层把算出的结果加回传给它的向量，再交给下一层；这种加回叫残差连接，每层收到、又往下传的那个向量叫隐藏状态。大语言模型用的这种逐层结构叫 Transformer。普通残差把各层的结果等权相加，每一项的系数都是 1。Block AttnRes 改的是这个和：它把层分成若干块，块内各层的结果先相加成一个块表示；每一层再用自己的一个查询向量，去和前面各个块表示一一比较（同一块里后面的层还会把本块当前的部分和，也就是本块已算出的各层输出之和，也当作一个来源），比较的结果经 softmax（把一组数变成和为 1 的权重：每个数取指数，再除以所有指数之和）归一成权重，按权重把那些来源加权求和，作为这一层的输入。查询是每层单独学出来的一个向量，不来自这一层的计算，所以这一层从前面各块取多少由它自己决定。
依据：paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Inter-Block Attention；paper/src-v1/3-attnres.tex Full Attention Residuals

### 等权相加把每一层稀释掉

隐藏状态是前面各层结果的累加和。模型在每个子层（子层指一次注意力或一个前馈层的计算）之前做归一化（把向量的分量按同一尺度重新缩放），层数一多，这个和越加越大，其中每一层结果的占比被逐步摊薄；靠前层的信息也被后来的累加掩盖，深层无法再把它们单独取回。另一种做法是让每一层直接对前面所有层做注意，但那样要把前面每一层的输出都保留下来，开销随层数线性增长。把层分块、块内先相加、层间只对块数做注意，就能在保留按内容取用的同时把开销压下来。
依据：paper/src-v1/1-intro.tex Introduction；paper/src-v1/2-preliminary.tex Limitations

### 分块相加，块间注意

$L$ 层分成 $N$ 块。每块里各层的输出直接相加，得到这一块的块表示；注意只在块间做：块里的头一层只看前面已完成的各块表示，同一块里后面的层再额外看本块当前的部分和。算权重用该层自己的查询，来源向量先做一次归一化，再算 softmax。块数取 $L$ 时就是逐层注意，取 1 时接近普通残差；实验里约取 8 块时收益已接近最好。
依据：paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Inter-Block Attention

## 长页

大语言模型的骨干是一叠结构相同的层，每层把算出的结果加回它收到的向量再传给下一层，这种加回叫残差连接。这个逐层结构叫 Transformer。加回的每一项权重都固定为 1，也就是把到目前为止各层的结果等权相加。Block AttnRes 换掉了这个和：它把层分成若干块，每层用一个自己学出来的查询，对前面各块的块表示（一块里各层的输出相加成的那个向量）按相关度加权，取回来的向量作为这一层的输入。需要它，是因为等权相加会让各层的结果随层数增加被摊薄，而按相关度取用能让每一层挑出它需要的早期结果；直接对前面每一层都算一次相关度开销太大，分块把开销压到只跟块数有关。块内仍然等权相加，块间才做加权。
依据：paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/1-intro.tex Introduction；paper/src-v1/3-attnres.tex Block Attention Residuals

### 固定权重残差随深度稀释每一层

模型把序列里每个 token（token 是序列被切分成的单位，可以是一个词或词的一部分）先变成一个向量，这些向量再逐层往上算。把第 $l$ 层收到的向量记作 $\mathbf{h}_l$，$l$ 是层的编号（从 1 数起），这个向量叫隐藏状态，$\mathbf{h}_1$ 是输入 token 对应的那个向量。在 Transformer 里，一次注意力（[标准注意力](wiki/standard-attention)：让每个位置和序列里其他位置互相取信息的计算）或一个前馈层（对每个位置各自做一次变换）各算一层，所以层数 $L$ 等于 Transformer 块数的两倍。
依据：paper/src-v1/2-preliminary.tex Notation

每层算出该往隐藏状态上加的东西，加到传下来的隐藏状态上，再交给下一层：$\mathbf{h}_l = \mathbf{h}_{l-1} + f_{l-1}(\mathbf{h}_{l-1})$，$f_{l-1}$ 是第 $l-1$ 层做的变换。把这条递推展开，第 $l$ 层收到的就是输入向量加上前面所有层输出的和：$\mathbf{h}_l = \mathbf{h}_1 + \sum_{i=1}^{l-1} f_i(\mathbf{h}_i)$。这个连接叫残差连接，它同时是一条梯度通路：训练要压小的那个数叫损失，损失对每一层输入的梯度里始终保留一项恒等映射，可以从最后一层直接传到任意一层。反向传播指从输出往输入方向逐层求梯度。
依据：paper/src-v1/2-preliminary.tex Residual Learning

::: 残差为什么是梯度通路
1. 损失 $\mathcal{L}$ 对第 $l$ 层输入的梯度，等于它对最后一层的梯度乘上一串因子：$\dfrac{\partial \mathcal{L}}{\partial \mathbf{h}_l} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{h}_{L}} \prod_{j=l}^{L-1}\left(\mathbf{I} + \dfrac{\partial f_j}{\partial \mathbf{h}_j}\right)$。
2. 连乘展开后，其中一项的各因子都取恒等矩阵 $\mathbf{I}$，这一项就是纯恒等矩阵，不随层数衰减；其余项里含各层的雅可比（各层输出对输入的偏导）。所以从损失到第 $l$ 层总有一条系数为 1 的路径。
依据：paper/src-v1/2-preliminary.tex Residual Learning
:::

残差还有另一种写法，给两个加数各配一个权重：$\mathbf{h}_l = \alpha_l \mathbf{h}_{l-1} + \beta_l f_{l-1}(\mathbf{h}_{l-1})$。普通残差把两个权重都取 1，也有做法把它们取成一组可学系数。
依据：paper/src-v1/2-preliminary.tex Generalizing Residuals

不管哪一种，第 $l$ 层都只能看到紧邻的上一层 $\mathbf{h}_{l-1}$，也就是把此前所有层的结果先压成了一个向量。这种压缩带来三处限制：一是无法选择性取用，注意力层和前馈层拿到的是同一个和，即使它们想要的加权可能不同；二是聚合不可逆，被加进和里之后，信息不能再在深层单独取回；三是输出增长，靠后的层为了让自己的结果在越加越大的和里保持占比，会把输出算得越来越大。
依据：paper/src-v1/2-preliminary.tex Limitations

现代模型多用 PreNorm（把归一化放在每个子层之前，而不是之后）：在这种安排下，隐藏状态就是各层输出的累加和，它的幅度随层数增长，每一层输出在其中的相对比重被逐步摊薄，这可能让训练不稳。
依据：paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/1-intro.tex Introduction；paper/src-v1/2-preliminary.tex Limitations

### 把深度当成一条序列来注意

残差把之前所有层的信息压进一个向量 $\mathbf{h}_l$，这和循环网络沿时间把之前所有位置的信息压进一个状态是同一种做法。序列方向上的这个瓶颈，是 Transformer 用注意力打破的：注意力让每个位置用一个查询，去和各个位置的键比较，按比出来的相关度对各个位置的值加权取回，于是可以直接取用任意一个前面的位置。
依据：paper/src-v1/3-attnres.tex The Duality of Time and Depth

把同样的做法搬到深度方向，就是让每一层直接对前面各层的输出加权取回：$\mathbf{h}_l = \alpha_{0 \to l}\mathbf{h}_1 + \sum_{i=1}^{l-1} \alpha_{i \to l} f_i(\mathbf{h}_i)$，其中 $\alpha_{i \to l}$ 是第 $l$ 层分给第 $i$ 层输出的权重，$\mathbf{h}_1$ 也当作一个来源（记作第 0 个），所有权重加起来为 1。沿深度做注意力的层数通常不到 1000，$O(L^2)$ 的开销可以承受。
依据：paper/src-v1/3-attnres.tex The Duality of Time and Depth

权重按注意力那样算：$\alpha_{i \to l} = \phi(\mathbf{q}_l, \mathbf{k}_i) \big/ \sum_{j} \phi(\mathbf{q}_l, \mathbf{k}_j)$，$\phi$ 是一个把两个向量映成非负数的函数。取 $\phi(\mathbf{q}, \mathbf{k}) = \exp\left(\mathbf{q}^\top \mathrm{RMSNorm}(\mathbf{k})\right)$ 并做归一，就是对深度做 softmax；这里的 [RMSNorm](wiki/rmsnorm) 是按键向量的均方根（各分量平方的平均再开方）把它归一化。查询是每一层自己的一个可学向量 $\mathbf{w}_l$，叫伪查询，因为它不来自这一层的计算，而是一个单独的参数；键和值取同一个来源向量，$\mathbf{k}_i = \mathbf{v}_i$，$i=0$ 时取 $\mathbf{h}_1$，$i \ge 1$ 时取 $f_i(\mathbf{h}_i)$。
依据：paper/src-v1/3-attnres.tex Full Attention Residuals

键上的 RMSNorm 把每个来源的幅度归一到同一尺度，避免某个输出本来就大的层仅凭幅度就拿到过高的权重。每层只多这一个 $d$ 维向量（$d$ 是隐藏维度）和这一处归一化。训练开始时把所有 $\mathbf{w}_l$ 初始化为零，softmax 就退化成等权平均，训练一开始不抖动。
依据：paper/src-v1/3-attnres.tex Full Attention Residuals；paper/src-v1/5-exp.tex Architecture Details

普通残差给每个来源的权重都是 1。这里的权重由 softmax 算出，一层给前面各来源的权重加起来为 1，各来源因此互相竞争，会选出更少、更集中的来源。
依据：paper/src-v1/6-discuss.tex Residual Connections as Structured Matrices；paper/src-v1/5-exp.tex Training dynamics；paper/src-v1/5-exp.tex Component design

::: 深度混合矩阵
1. 把第 $l$ 层的输入写成对所有前层来源的加权和 $\mathbf{h}_l = \sum_{i=0}^{l-1} \mathbf{M}_{i \to l}\,\mathbf{v}_i$，$\mathbf{M}$ 是一个 $L \times L$ 的下三角矩阵，叫深度混合矩阵，第 $l$ 行是第 $l$ 层给各来源的权重。
2. 普通残差的 $\mathbf{M}$ 是下三角全 1，即各来源等权。
3. 这里 $\mathbf{M}_{i \to l} = \alpha_{i \to l}$，由 softmax 归一得到。
依据：paper/src-v1/6-discuss.tex Residual Connections as Structured Matrices
:::

### 分块：块内相加，块间注意

若让每层对前面每一层都做一次注意，就要把前面每一层的输出都留着，还要跨设备传递，存储和通信都随层数增长。分块的做法是把 $L$ 层切成 $N$ 块，每块 $S = L/N$ 层。
依据：paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Intra-Block Accumulation

块表示不靠注意拼：把一块里各层的输出直接相加，得到一个块表示 $\mathbf{b}_n = \sum_{j \in \mathcal{B}_n} f_j(\mathbf{h}_j)$，$\mathcal{B}_n$ 是第 $n$ 块里的层编号集合。再把这一块里前 $i$ 层输出的部分和记作 $\mathbf{b}_n^i$，于是 $\mathbf{b}_n = \mathbf{b}_n^S$。
依据：paper/src-v1/3-attnres.tex Intra-Block Accumulation

注意只在块间做，输入的来源里既有前面各块的块表示，也有本块当前的部分和。输入向量记作 $\mathbf{b}_0 = \mathbf{h}_1$，它永远算一个来源。对第 $n$ 块里的第 $i$ 层：$i=1$（这一块的头一层）时，来源是 $\mathbf{b}_0, \dots, \mathbf{b}_{n-1}$；$i \ge 2$ 时，再添上本块已经算出的部分和 $\mathbf{b}_n^{i-1}$。键和权重的算法与上一章相同。
依据：paper/src-v1/3-attnres.tex Inter-Block Attention

块数 $N$ 从 1 到 $L$ 变动，两端分别是普通残差与逐层注意：$N = L$（每块一层）就是上一章的逐层注意，$N = 1$（只有一块、只对前面各块的块表示做一次注意）接近普通残差。因为每层只看 $N$ 个块表示而不是 $L$ 个层输出，存储从 $O(L)$ 降到 $O(N)$，块间注意只在 $N$ 个块表示之间做，注意计算从 $O(L^2)$ 降到 $O(N^2)$。
依据：paper/src-v1/3-attnres.tex Block Attention Residuals；paper/src-v1/3-attnres.tex Efficiency

::: 6 层、2 块、隐藏维度 2 上走一遍
取 $L=6$ 层、$N=2$ 块，每块 $S=3$ 层；隐藏维度 $d=2$；每层的伪查询都取 $\mathbf{w} = (1,1)$（真实模型里各层不同，这里为手算取成一样）。输入向量 $\mathbf{b}_0 = (1,0)$。三层的输出取 $f_1=(2,0)$、$f_2=(0,2)$、$f_3=(1,1)$，于是第一块的块表示 $\mathbf{b}_1 = f_1+f_2+f_3 = (3,3)$。第二块的前两层输出取 $f_4=(1,-1)$、$f_5=(0,1)$；下面第 6 步只用到第二块前两层的部分和，所以 $f_6$ 不必写出。
依据：自行构造

1. 第 1 层是第 1 块的头一层，来源只有 $\mathbf{b}_0$，权重为 1，$\mathbf{h}_1 = (1,0)$。
2. 第 2 层的来源是 $\mathbf{b}_0$ 和本块部分和 $\mathbf{b}_1^1 = f_1 = (2,0)$。两者归一后幅度相同，权重各 0.5，$\mathbf{h}_2 = 0.5\cdot(1,0) + 0.5\cdot(2,0) = (1.5, 0)$。
3. 第 3 层的来源是 $\mathbf{b}_0$ 和 $\mathbf{b}_1^2 = f_1+f_2 = (2,2)$。$\mathbf{b}_0$ 归一后是 $(1.41, 0)$，与 $\mathbf{w}$ 点积 $1.41$；$\mathbf{b}_1^2$ 归一后是 $(1,1)$，点积 $2$。指数归一得权重 $0.358$、$0.642$，$\mathbf{h}_3 = 0.358\cdot(1,0) + 0.642\cdot(2,2) = (1.64, 1.28)$。
4. 第 4 层是第 2 块的头一层，来源是 $\mathbf{b}_0$ 和 $\mathbf{b}_1 = (3,3)$。归一后点积仍是 $1.41$ 和 $2$，权重 $0.358$、$0.642$，$\mathbf{h}_4 = 0.358\cdot(1,0) + 0.642\cdot(3,3) = (2.28, 1.93)$。
5. 第 5 层在 $\mathbf{b}_0$、$\mathbf{b}_1$ 之外加上本块部分和 $\mathbf{b}_2^1 = f_4 = (1,-1)$。它归一后是 $(1,-1)$，与 $\mathbf{w}$ 点积 $0$。三个来源的权重是 $0.329$、$0.591$、$0.080$，$\mathbf{h}_5 = 0.329\cdot(1,0) + 0.591\cdot(3,3) + 0.080\cdot(1,-1) = (2.18, 1.69)$。
6. 第 6 层把部分和换成 $\mathbf{b}_2^2 = f_4+f_5 = (1,0)$，归一后点积回到 $1.41$，权重 $0.263$、$0.473$、$0.263$，$\mathbf{h}_6 = 0.263\cdot(1,0) + 0.473\cdot(3,3) + 0.263\cdot(1,0) = (1.95, 1.42)$。
依据：自行构造

```python
import math

def rmsnorm(v):
    r = math.sqrt(sum(x * x for x in v) / len(v))
    return [x / r for x in v]

def weights(w, sources):
    logits = [sum(a * b for a, b in zip(w, rmsnorm(s))) for s in sources]
    exps = [math.exp(x) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

w = [1, 1]
b0, b1 = [1, 0], [3, 3]
f4, f5 = [1, -1], [0, 1]
cases = [(4, [b0, b1]),
         (5, [b0, b1, f4]),
         (6, [b0, b1, [f4[0] + f5[0], f4[1] + f5[1]]])]
for layer, sources in cases:
    print("第", layer, "层权重", [round(x, 3) for x in weights(w, sources)])
```

```text
第 4 层权重 [0.358, 0.642]
第 5 层权重 [0.329, 0.591, 0.08]
第 6 层权重 [0.263, 0.473, 0.263]
```

这段代码里 `rmsnorm` 先把来源向量除掉自己的均方根，`weights` 再和伪查询点积、取指数、归一。第 5、6 层的第三个来源分别是 $f_4$、$f_4+f_5$。
依据：自行构造
:::

### 分块数 N 怎么选

在没有参与训练的数据上算出的损失叫验证损失。实验发现块数 $N$ 约取 8 时收益已接近最好，块数少了接近普通残差，块数多了接近逐层注意。
依据：paper/src-v1/3-attnres.tex Efficiency
块大小在一个范围内变化时，验证损失变化不大；块再大就逐渐接近不加这个改动的基线模型。按缩放律（一条描述模型变大时损失如何变化的经验规律）比不同算力预算下的损失，分块版全程低于基线，且与逐层版的差距随模型变大而收窄。
依据：paper/src-v1/5-exp.tex Cross-layer access；paper/src-v1/5-exp.tex Scaling Behavior

::: 块大小与缩放律的数字
1. 在一个 16 个 Transformer 块（共 32 层）的模型上扫描块大小 $S$：$S$ 取 2、4、8 时验证损失为 1.746、1.746、1.748；取 16、32 时为 1.753、1.757；基线为 1.766。
2. 基线拟合为 $\mathcal{L} = 1.891 \times C^{-0.057}$，分块版拟合为 $\mathcal{L} = 1.870 \times C^{-0.058}$，两者斜率接近（$\mathcal{L}$ 是验证损失，$C$ 是算力预算）。
3. 在算力预算 5.6 PFLOP/s-days（PFLOP/s-days 是算力单位，等于每秒千万亿次浮点运算乘天数）处，分块版 1.692，基线 1.714，相当于基线要多花 1.25 倍算力才能追平。
依据：paper/src-v1/5-exp.tex Cross-layer access；paper/src-v1/5-exp.tex Scaling Behavior；paper/src-v1/5-exp.tex Ablation Study；paper/src-v1/figures/block.tex L45-L115
:::

### 训练时输出与梯度更均匀

分块把块内的累加在块边界处重置，隐藏状态的幅度不再一路上涨，而是呈有界的周期性起伏；靠后的层也就不必再靠放大输出来争取影响。
依据：paper/src-v1/0-abstract.tex L1-L8；paper/src-v1/5-exp.tex Training dynamics

权重经过 softmax 归一，各来源要争同一份概率质量，梯度在各层之间的分布因此更均匀。
依据：paper/src-v1/5-exp.tex Training dynamics

::: 设计上的两处消融
消融指去掉某个设计、看效果变差多少。
依据：paper/src-v1/5-exp.tex Component design

1. 键上不做 RMSNorm：分块版（$S=4$）的验证损失从 1.746 升到 1.750。
依据：paper/src-v1/5-exp.tex Component design；paper/src-v1/5-exp.tex Cross-layer access

2. 把 softmax 换成 sigmoid（一种把任意实数压到 0 到 1 之间的函数）：损失变差。
依据：paper/src-v1/5-exp.tex Component design
:::

### 适用条件与边界

层数不能被块数整除时，最后一块只有不满 $S$ 层，它的块表示取这一块的部分和。
依据：paper/src-v1/3-attnres.tex Intra-Block Accumulation

分块版可以直接替换普通残差，新增的参数量占比可以忽略。
依据：paper/src-v1/5-exp.tex Architecture Details

块级压缩不只是为省开销而做的折中：换成块间注意之后，学到的权重结构基本保留下来，每层仍最偏向紧邻的上一层，输入向量始终分到非零权重，注意力层和前馈层呈现不同的来源偏好。这种压缩似乎还起到一种隐式正则的作用（隐式正则指不另加惩罚项、靠结构本身抑制过拟合，过拟合指模型把训练数据里的偶然细节也记住、到新数据上变差）。
依据：paper/src-v1/5-exp.tex Analyzing Learned AttnRes Patterns

代价是块表示要存下来并跨设备传递。
依据：paper/src-v1/4-infra.tex Pipeline communication

在固定算力和参数预算下扫描深宽配比，最优的隐藏维度与 Transformer 块数之比往更小一侧移动，也就是更偏好深而窄的网络。
依据：paper/src-v1/5-exp.tex Optimal Architecture

::: 深宽配比的最优点
1. 比值 $d_{\text{model}}/L_b$（$d_{\text{model}}$ 是隐藏维度，$L_b$ 是 Transformer 块数）在基线的最优点约 60、损失 1.847；本方法移到约 45、损失 1.802。
依据：paper/src-v1/5-exp.tex Optimal Architecture
:::

推理指训练完成后用模型对外生成 token。偏好深一般不等于部署建议：更深的网络逐层串行，推理延迟一般更高。
依据：paper/src-v1/5-exp.tex Optimal Architecture
