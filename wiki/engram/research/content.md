# Engram

- 描述：Engram 是给 Transformer 加的一张条件记忆表：把以当前位置结尾的几个 token 哈希成嵌入表的行号，取出的向量由当前隐藏状态门控后加回残差流，用一次查表代替早期层对固定搭配的重建。
- 摘要：Engram 把以当前位置结尾的 $n$ 个 token 哈希成嵌入表的行号，取回的向量由当前隐藏状态门控后加回残差流。查表的计算量不随表增大，与 MoE 构成两条互补的稀疏轴；总稀疏参数在两者间分配时，损失随分配比例呈 U 形。行号只由 token 决定，大表可放主机内存并提前预取。
- 主题：模型结构，内存与缓存
- 标签：模型架构

## 资料

- Xin Cheng, Rui Tian, Wangding Zeng, et al. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372v2, 2026. [https://arxiv.org/abs/2601.07372](https://arxiv.org/abs/2601.07372)
- DeepSeek-AI. Engram. GitHub, commit fb7f84a21f91223715394a33a1dc24bbfb7f788e, 2026. [https://github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

## 短页

### 一张按最近几个词寻址的记忆表

Engram 是加在 Transformer 里的一张条件记忆表。Transformer 逐层处理一个 token 序列，token 是分词器输出的基本单位，每个 token 有一个整数 id。一层里包含注意力与前馈两类计算，隐藏状态是每个位置上的一个向量，残差流是每个位置在各层之间传递的通路，每层算出的更新量加回它：$\mathbf{H}^{(\ell)}\leftarrow\mathbf{H}^{(\ell)}+\mathbf{Y}$，$\mathbf{H}^{(\ell)}$ 是这一层的隐藏状态，$\mathbf{Y}$ 是这一层算出的更新量，两者形状相同。Engram 不改这些层本身，只把以当前位置结尾的最近 $n$ 个 token 当作键，去一张嵌入表里查行。每个位置只读固定几行，表再大也不增加这个位置的计算量。
依据：paper/src-v2/sections/intro.tex L7；paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Tokenizer Compression；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation

### 固定搭配不必再靠计算重建

语言里有很多局部而固定的模式，例如多 token 的实体名和固定短语。没有查表能力时，模型要用前面几层的注意力与前馈逐层把它们拼出来，占掉本来可以做推理的深度。它和 MoE 一样是稀疏的：MoE 是每个 token 只激活少数几个专家子网络的结构，Engram 每个位置只读固定几行。两者都只用到参数的一小部分，没被选中的专家和没被读到的行不参与这个 token 的计算，所以这笔参数可以做得很大而不增加每 token 的计算量。总参数固定时，这笔参数分给谁、各分多少会影响损失，把一部分给记忆比全部给专家更好。
依据：paper/src-v2/sections/intro.tex L3-L4；paper/src-v2/sections/intro.tex L7；paper/src-v2/sections/intro.tex L9；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis

### 哈希取行、门控、加回残差

取回时先把最近 $n$ 个 token 的 id 拼成一个键，用哈希算出嵌入表的行号，取出向量。所有可能的 n-gram 组合太多，不可能每种组合都分配一行，所以用哈希；哈希会把不同的键映到同一行，这叫碰撞。为减少碰撞的影响，用多个独立的哈希头分别取行，把各头取出的向量拼起来，某个头里撞上的两个键很少在所有头里都撞上。取回的向量只由 token 决定，与当前上下文无关，于是再由当前位置的隐藏状态算一个门：门是 0 到 1 之间的一个数，乘在取回的向量上，记忆与上下文不符时接近 0。门控后的结果过一层短卷积，再沿残差流加回该层。
依据：paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Multi-Head Hashing；paper/src-v2/sections/arch.tex Context-aware Gating；engram-repo/engram_demo_v1.py L123-L149

## 长页

语言模型要同时做两件事：需要动态计算的组合推理，和基本固定的知识取回。命名实体、固定短语这类文本是局部的、静态的，经典 n-gram 模型用最近几个词就能覆盖它们。标准 Transformer 没有查表这一步，只能靠前几层的注意力与前馈把这类模式重新算出来。Engram 是一张条件记忆表，以每个位置结尾的最近几个 token 为键，用哈希查出嵌入表的行号，取回的向量按当前上下文加权后加回残差流。查表的计算量是常数级，不随表增大。MoE 是每个 token 只激活少数几个专家子网络、用条件计算换容量的结构，两者占的是同一笔不增加每 token 计算量的参数，可以按比例混合。
依据：paper/src-v2/sections/intro.tex L3-L9；paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation

### 背景：token、隐藏状态与残差流

序列先被分词器切成 token，每个 token 有一个整数 id。模型把 id 变成向量后逐层处理，第 $\ell$ 层输出的 $T\times d$ 矩阵叫隐藏状态，$T$ 是序列长度，$d$ 是每个位置的向量维度。一层里包含注意力与前馈两类计算。残差流指每个位置的向量在各层之间传递的那条通路，每层算出的结果加回它，再进入下一层。训练结束后在验证集上报告损失，损失越低表示模型越好。
依据：paper/src-v2/sections/arch.tex Overview；paper/src-v2/sections/arch.tex Tokenizer Compression；paper/src-v2/sections/intro.tex L7；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/large_scale_exp.tex Evaluation Protocol

### 取回：用最近几个 token 决定查哪几行

取回这一步把每个位置映射到嵌入表的若干行。键是以该位置结尾的后缀 n-gram，也就是从这里往前数 $n$ 个 token 的 id 序列。取出的向量是表里的行，这些行是可训练的参数，跟着模型一起更新，和按语料计数得到的 n-gram 概率表不是一回事。每个位置只读固定几行，所以表的行数可以远大于一次前向真正用到的行数：增加行数只增加参数，不增加每 token 的计算量，这部分参数和 MoE 里未被选中的专家处在同一类。
依据：paper/src-v2/sections/arch.tex Sparse Retrieval via Hashed；paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations；paper/src-v2/sections/related_work.tex L3-L5

#### 后缀 N-gram 与分词压缩

$n$ 个连续 token 组成的序列叫 n-gram，以位置 $t$ 结尾的那个记为 $g_{t,n}$。分词器把文本切成 token 时，写法上等价的 token 常拿到不同的 id：大小写不同、前导空格不同，都会分成两个 id。取回前先做一次分词压缩：按大小写、空格、重音等规则归一化，把等价的写法合并到同一个标准 id，把原始 id 映射成标准 id，$x'_t=\mathcal{P}(x_t)$，$\mathcal{P}$ 就是这个映射。键由标准 id 构成：$g_{t,n}=(x'_{t-n+1},\dots,x'_t)$。
依据：paper/src-v2/sections/arch.tex Tokenizer Compression

#### 多头哈希

所有可能的 n-gram 组合太多，直接给每一种组合分配一行不现实，所以把键哈希成行号。哈希是确定性的映射，不同键可能映到同一行，这叫碰撞。为降低碰撞的影响，每个阶数 $n$ 配 $K$ 个独立的哈希头，第 $k$ 个头把同一个键映到自己的表里：$z_{t,n,k}=\phi_{n,k}(g_{t,n})$，$z$ 是行号，$\phi$ 是哈希函数，取出的向量是 $\mathbf{e}_{t,n,k}=\mathbf{E}_{n,k}[z_{t,n,k}]$，$\mathbf{E}_{n,k}$ 是第 $n$ 阶第 $k$ 个头的嵌入表。各头各阶取出的向量拼成一个记忆向量：$\mathbf{e}_t=\Vert_{n=2}^{N}\Vert_{k=1}^{K}\mathbf{e}_{t,n,k}$，符号 $\Vert$ 表示把向量首尾相接，$N$ 是最大阶数。拼接从 $n=2$ 开始，单个 token 本身由输入嵌入层处理，不进记忆。
依据：paper/src-v2/sections/arch.tex Multi-Head Hashing；paper/src-v2/sections/arch.tex L1-L8

### 融合：由当前上下文决定这份记忆用多少

取回的向量只由 token 序列决定，同一个 n-gram 在任何上下文里取到同一个向量。它本身没有上下文适应能力，碰撞和多义词还会带来噪声。所以在写回之前，用当前位置的隐藏状态给这份记忆算一个门，决定这一处用多少。
依据：paper/src-v2/sections/arch.tex Context-aware Gating

#### 门控公式

先把记忆向量投成 key 和 value：$\mathbf{k}_t=\mathbf{W}_K\mathbf{e}_t$，$\mathbf{v}_t=\mathbf{W}_V\mathbf{e}_t$，$\mathbf{W}_K$、$\mathbf{W}_V$ 是可训练的投影矩阵，把记忆向量投到与隐藏状态相同的维度。门由当前隐藏状态 $\mathbf{h}_t$ 与 key 的归一化点积给出：$$\alpha_t=\sigma\left(\frac{\mathrm{RMSNorm}(\mathbf{h}_t)^\top\mathrm{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right)$$ $\sigma$ 是把实数压到 $(0,1)$ 的函数，$\mathrm{RMSNorm}$ 是按均方根归一化向量的操作，$d$ 是向量维度。点积大说明记忆与上下文一致，门接近 1；不一致时门接近 0，把这份记忆压掉。门控后的值写成 $\tilde{\mathbf{v}}_t=\alpha_t\cdot\mathbf{v}_t$。
依据：paper/src-v2/sections/arch.tex Context-aware Gating；engram-repo/engram_demo_v1.py L351-L352

#### 短卷积与写回残差流

门逐位置决定用多少，门用的隐藏状态已经通过前面的注意力聚合了全局上下文。门控后的值再过一层短的深度因果卷积，扩大每个位置能看到的范围，然后加回残差流：$\mathbf{H}^{(\ell)}\leftarrow\mathbf{H}^{(\ell)}+\mathbf{Y}$。Engram 只插在选定的层，在该层的注意力与前馈之前生效。
依据：paper/src-v2/sections/arch.tex Context-aware Gating

#### 短卷积的公式

::: 短卷积与残差相加
设 $\tilde{\mathbf{V}}$ 是门控后的值按序列排列成的矩阵，卷积核大小为 $w$，膨胀系数为 $\delta$，则
$$\mathbf{Y}=\mathrm{SiLU}\left(\mathrm{Conv1D}_{w,\delta}\left(\mathrm{RMSNorm}\left(\tilde{\mathbf{V}}\right)\right)\right)+\tilde{\mathbf{V}}$$
$\mathrm{Conv1D}_{w,\delta}$ 是沿序列方向的卷积，核大小 $w$ 决定单次看几个位置，膨胀系数 $\delta$ 把采样点之间的间隔拉开，使同样大的核覆盖更长的范围；深度指每个通道各自卷，因果指只看当前位置及之前的位置。$\mathrm{SiLU}$ 是激活函数，定义为 $\mathrm{SiLU}(x)=x\cdot\sigma(x)$，$\sigma$ 是前面把实数压到 $(0,1)$ 的函数。$\mathrm{RMSNorm}$ 是按均方根归一化。
依据：paper/src-v2/sections/arch.tex Context-aware Gating；engram-repo/engram_demo_v1.py L138-L149
:::

::: 小词表上的哈希与门控
用三个标准 id：The 与 the 都归为 1，cat 与 Cat 都归为 2，sat 归为 3。序列 The cat sat the Cat 的标准 id 是 $(1,2,3,1,2)$，位置 2 到 5 上的 bigram 分别是 $(1,2)$、$(2,3)$、$(3,1)$、$(1,2)$。两个哈希头都用行号 $((a\cdot x)\oplus(b\cdot y))\bmod M$，第一头取 $(a,b,M)=(1,3,7)$，第二头取 $(1,1,11)$，$\oplus$ 是异或。
依据：自行构造

```python
head1 = (1, 3, 7)
head2 = (1, 1, 11)


def row(head, x, y):
    a, b, m = head
    return ((a * x) ^ (b * y)) % m


for gram in [(1, 2), (2, 3), (3, 1), (1, 2)]:
    print(gram, row(head1, *gram), row(head2, *gram))
```

```text
(1, 2) 0 3
(2, 3) 4 1
(3, 1) 0 2
(1, 2) 0 3
```
依据：自行构造

$(1,2)$ 和 $(3,1)$ 在第一个头里都映到第 0 行，发生了碰撞；在第二个头里它们映到第 3 行和第 2 行，是分开的。两个头给出的行号合起来看，三个不同的 bigram 都落在不同的组合上，一个头里的碰撞被另一个头补上了。
依据：自行构造

门用当前的隐藏状态和 key 的归一化点积。取隐藏状态 $h=(2,2)$，记忆与上下文相符时取 $k=(3,3)$，不符时取 $k=(-3,-3)$，归一化点积再除以向量维度 $d$ 的平方根（本例如 $\sqrt{2}$），得到两个门值。
依据：自行构造

```python
import math


def rmsnorm(v):
    r = math.sqrt(sum(x * x for x in v) / len(v))
    return [x / r for x in v]


def gate(h, k):
    hn, kn = rmsnorm(h), rmsnorm(k)
    s = sum(a * b for a, b in zip(hn, kn)) / math.sqrt(len(h))
    return 1 / (1 + math.exp(-s))


print(round(gate([2.0, 2.0], [3.0, 3.0]), 4))
print(round(gate([2.0, 2.0], [-3.0, -3.0]), 4))
```

```text
0.8044
0.1956
```
依据：自行构造

相符时门约 0.8044，不符时约 0.1956。门随点积连续变化，不符时降到相符时的约四分之一。
依据：自行构造
:::

### 稀疏参数怎么分：与专家争同一笔预算

设 $P_{\mathrm{tot}}$ 是总可训练参数，不含词表嵌入与输出层；$P_{\mathrm{act}}$ 是每个 token 实际参与计算的激活参数；两者之差 $P_{\mathrm{sparse}}=P_{\mathrm{tot}}-P_{\mathrm{act}}$ 是不参与该 token 计算的那部分。MoE 里未被选中的专家、Engram 里未被取回的行都算在这里。固定 $P_{\mathrm{tot}}$ 与 $P_{\mathrm{act}}$，把这笔稀疏参数分给专家和记忆：定义 $\rho$ 是分给专家的比例，专家分到 $\rho P_{\mathrm{sparse}}$，记忆分到 $(1-\rho)P_{\mathrm{sparse}}$，$\rho=1$ 是所有稀疏参数都给专家的纯 MoE。
依据：paper/src-v2/sections/scaling_law_exp.tex Compute-matched formulation；paper/src-v2/sections/scaling_law_exp.tex Allocation ratio

#### 损失随 $\rho$ 呈 U 形

固定总参数与激活参数后，改变 $\rho$ 并训练，验证损失与 $\rho$ 的关系是一条 U 形曲线。$\rho$ 接近 1 时缺少专门存固定模式的地方，模型仍要靠深度和计算重建它们；$\rho$ 接近 0 时条件计算的能力不够，需要动态推理的任务变差，记忆不能替代计算。最优落在偏向专家一侧，只把稀疏参数的约两成到两成半分给记忆，两档算力下这个位置基本一致。
依据：paper/src-v2/sections/scaling_law_exp.tex Results and Analysis

#### 表越大，损失越低

把骨干固定，只放大记忆表，在资料测试的行数范围内，验证损失随行数的对数近似线性下降，行数越多损失越低。表的行数增加不改变每 token 的计算量。
依据：paper/src-v2/sections/scaling_law_exp.tex Engram under Infinite Memory Regime

#### 分配实验的条件

::: 分配实验的条件
| 项目 | 一档 | 另一档 |
|---|---|---|
| 训练算力（FLOPs） | $2\times10^{20}$ | $6\times10^{20}$ |
| 总参数 | 约 5.7B | 约 9.9B |
| 激活参数 | 568M | 993M |
| 纯 MoE 的专家数 | 106 | 99 |
两组都保持总参数与激活参数之比约为 10，改变 $\rho$ 时只调整专家数和记忆的行数，训练流程与超参数不变。
依据：paper/src-v2/sections/scaling_law_exp.tex Experimental protocol
:::

### 地址只由 token 决定：大表放主机内存，提前预取

记忆的行号只由 token id 决定，不需要等隐藏状态算出来。MoE 的路由要看隐藏状态才知道选哪些专家，Engram 不用，因此在前向开始前就能算出整个序列要读的行号。这把参数存储和计算解耦：跑模型的加速器显存容量有限，装不下的记忆表可以放在主机内存，按行号提前取回，与前面层已经在做的计算重叠，等轮到自己时数据已经到了。取回量只与每个 token 激活的行数有关，与表的总行数无关。
依据：paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory；paper/src-v2/sections/analysis.tex System Efficiency

#### 卸载实验的条件与结果

::: 卸载实验的条件与结果
设置：把整张约 100B 参数的记忆表全部放主机内存，骨干用不含专家并行的稠密模型，在 H800 上跑 512 条、长度在 100 到 1024 之间的序列，读取全部走主机内存。
结果：与不带记忆表的同一个稠密骨干相比，4B 骨干的吞吐从 9031.62 降到 8858.28 token/s，8B 骨干从 6315.52 降到 6140.02 token/s，最大降幅 2.8%。
依据：paper/src-v2/sections/analysis.tex System Efficiency
:::

### 放在哪几层

Engram 不是每层都插。放得早，后面的层不用再花深度重建局部模式；放得晚，隐藏状态已经聚合了更多全局上下文，门的判断更准，前面可供重叠读取的计算也更多。把同样大小的记忆拆成早晚两处各放一份小表，可以兼顾早介入和晚门控。
依据：paper/src-v2/sections/analysis.tex Where should memory be injected；paper/src-v2/sections/arch.tex System Efficiency: Decoupling Compute and Memory

### 哪些部件起作用

在固定记忆参数量的对照里，去掉上下文门控或去掉分词压缩，验证损失上升得最多；去掉短卷积只带来很小的退化。把容量分给更高阶的 n-gram 在固定预算下没有更好，一种解释是更高阶的组合占走了更常见低阶模式的行，但在更大的表上是否仍然如此，资料没有下结论。
依据：paper/src-v2/sections/analysis.tex Which components matter

### 效果的证据：整体更好，也更深更省

在同总参数、同激活参数、同训练量的对照里，带 Engram 的模型优于纯 MoE 基线，而且收益不限于知识类任务，通用推理、代码和数学任务上的提升更大。查表接手局部依赖后，注意力可以更多用于全局上下文，长上下文检索类任务随之变好。
依据：paper/src-v2/sections/large_scale_exp.tex Experimental Results；paper/src-v2/sections/intro.tex L14-L15；paper/src-v2/sections/long_context_exp.tex Experimental Results

#### 大规模对比的设置

::: 大规模对比的设置
模型：不含稀疏层的稠密基线，纯 MoE 基线，带记忆的模型，以及只把记忆表放大的版本。
对照：所有模型用同一份数据训练同样多的 token，激活参数相同；总参数与纯 MoE 相同的那一个版本，差别只在把一部分专家换成了记忆表。
任务：语言建模、知识与推理、阅读理解、代码与数学。
依据：paper/src-v2/sections/large_scale_exp.tex Training Data and Model Configurations；paper/src-v2/sections/large_scale_exp.tex Evaluation Protocol
:::

#### 浅层相当于更深的层

LogitLens 把每一层的隐藏状态直接接到输出层，用 KL 散度衡量这一层的预测分布与最终预测分布的差距；CKA 比较两层表示的结构相似度。结果是带 Engram 的模型在浅层就更快接近最终预测；按 CKA，它第 5 层的表示与基线第 12 层最接近。
依据：paper/src-v2/sections/analysis.tex Accelerated Prediction Convergence；paper/src-v2/sections/analysis.tex Representational Alignment and Effective Depth

### 适用条件与边界

Engram 处理的是局部、静态、固定搭配这类规律。它换掉的是用计算重建这类模式的那部分工作：屏蔽记忆输出后，事实类任务的成绩大幅下降，需要看上下文回答的阅读理解基本保持，说明知识主要存在记忆里，上下文理解仍靠骨干的注意力与计算。记忆不能替代计算，稀疏参数全给记忆时，需要动态推理的任务会变差。Engram 是加在中间层上的模块，输入嵌入层与输出层保持原样。
依据：paper/src-v2/sections/intro.tex L7；paper/src-v2/sections/arch.tex L1-L8；paper/src-v2/sections/arch.tex Multi-Head Hashing；paper/src-v2/sections/arch.tex Context-aware Gating；paper/src-v2/sections/scaling_law_exp.tex Results and Analysis；paper/src-v2/sections/analysis.tex Case Study；paper/src-v2/sections/analysis.tex Sensitivity Analysis
