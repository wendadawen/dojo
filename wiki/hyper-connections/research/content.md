# 超连接

- 描述：超连接是一种替代残差连接的连接方式：把网络的初始输入复制成 n 条并排的流（n 叫扩展率），每层的输入由这 n 条流加权求和得到，层的输出再按权重分发回各条流；两组连接权重可训练，也可以由输入算出。
- 摘要：把网络的初始输入扩成 $n$ 条并排的流，用一组可训练的权重决定每条流对当前层输入、以及层输出对每条流的贡献；残差连接是 $n=1$ 时的不可训练特例。
- 主题：模型结构
- 标签：模型架构

## 资料

- Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou. Hyper-Connections. ICLR 2025, arXiv:2409.19606v3, 2025. [https://arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606)

## 短页

### n 条并排的流

[残差连接](wiki/residual-connection)把一层的输入直接加到它的输出上，主干（一层接一层堆起来的网络）因此只有一条通路。超连接换一种做法：把网络的初始输入复制成 $n$ 条并排的流，$n$ 叫扩展率；每一层再按权重把这 $n$ 条流合起来，算完后又按权重把结果分回去。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections

### 连接强度由谁定

连接强度指一个模块（一层里真正做计算的部分）的输入在多大程度上加到它的输出上。归一化指把向量的数值缩放到统一尺度。残差连接用归一化的位置来定连接强度：在模块前归一化能缓解梯度变小，却让深层表示变得彼此相似；在模块后归一化减轻这种相似，又带回梯度变小。无论选哪一种，连接强度都是预先算好的——前归一化固定为 1，后归一化由输入输出的方差与协方差算出——训练不改变它。超连接把一层里两组权重的取值交给网络自己学。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/src-v3/app.tex Post-Norm Residual Connection

### 汇入、过层、分发回各流

先把各条流加权求和，得到这一层的输入；输入过这一层模块，得到输出。输出再按另一组权重分发回各条流：某条流分到多少，由输出在这条流上的权重，加上各条流原值按流间权重加权的结果决定。两组权重可以随训练更新，也可以由当前输入算出来。等价地看，这也是一个更一般的残差连接——给每个模块的输入与输出各配一个权重。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；paper/src-v3/iclr2025_conference.tex Introduction

## 长页

超连接是一种替代[残差连接](wiki/residual-connection)的连接方式。主干网络是一叠结构相同的层，每层拿到一个张量（多维数组，这里每个位置放一个向量），越过这一层的信息靠残差连接原样加到层的输出上。超连接不再只有这一条通路：它把网络的初始输入复制成 $n$ 条并排的流，$n$ 叫扩展率，每层让这 $n$ 条流汇成一份输入，算完再按权重把结果分回各条流。这两组权重可以随模型训练，也可以由当前输入算出。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Method

### 残差连接的取舍与固定的连接强度

连接强度指一个模块的输入在多大程度上加到它的输出上。在模块前归一化，输入先归一化（把向量的数值缩放到统一尺度）再进模块，输出再加回原输入，梯度不容易变小；代价是深层表示越来越相似（后一层几乎不再改动表示，叫表示坍缩），越深的层贡献越小。在模块后归一化，模块输出与输入相加后再归一化，减轻了这种相似；代价是梯度又会变小。
依据：paper/src-v3/iclr2025_conference.tex Introduction

问题出在两种做法都把连接强度预先算好，训练不改变它：前归一化固定为 1，后归一化由输入输出的方差与协方差算出。超连接让这个强度由网络自己学。
依据：paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/src-v3/app.tex Post-Norm Residual Connection

### 把网络的初始输入扩成 n 条：记号与连接矩阵

把传进第 $k$ 层的超隐藏矩阵记成 $\mathbf{H}^{k}$，是一个 $n$ 行 $d$ 列的矩阵，$d$ 是隐藏维度（每个位置向量的长度），每一行叫一条流；网络的初始输入记成 $\mathbf{h}^{0}$，把它复制 $n$ 份得到 $\mathbf{H}^{0}$，这就是第 0 层拿到的矩阵。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections

一层的全部连接权重写成一个 $(n+1)\times(n+1)$ 的矩阵：第一行右边 $n$ 个元素是 $\mathbf{B}$，给出层输出分到各条流的权重；第一列下面 $n$ 个元素是 $\mathbf{A}_m$，给出汇入时各条流的权重；右下 $n\times n$ 块是 $\mathbf{A}_r$，给出各条流原值分别流到新矩阵哪几行。用这三块写一层的更新：
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections

- 汇入：$\mathbf{h}_0^\top = \mathbf{A}_m^\top \mathbf{H}$，把 $n$ 条流按 $\mathbf{A}_m$ 加权求和，得这一层内部使用的输入 $\mathbf{h}_0$。
- 过层：$\mathbf{h}_0' = \mathcal{T}(\mathbf{h}_0)$，$\mathcal{T}$ 是这一层模块，在主干里是自注意力（[标准注意力](wiki/standard-attention)）或前馈网络（逐个位置各自过一遍的小网络）。
- 分发：$\hat{\mathbf{H}} = \mathbf{B}^\top \mathbf{h}_0'^{\,\top} + \mathbf{H}'$，其中 $\mathbf{H}' = \mathbf{A}_r^\top \mathbf{H}$ 是各条流原值按 $\mathbf{A}_r$ 的分配。

最后，把超隐藏矩阵的各行相加，得到网络的输出向量，再过一个归一化层和一个输出层。$\mathbf{A}_m$、$\mathbf{B}$ 和 $\mathbf{A}_r$ 都是可训练的权重。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/app.tex Transformer with Hyper-Connections；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections

#### 一层里走一遍

取 $n=2$、隐藏维度 $d=1$，两条流各是一个数，$\mathbf{H} = (2, 4)^\top$。设 $\mathbf{A}_m = (0.5, 0.5)$、$\mathbf{B} = (1, 0)$、$\mathbf{A}_r$ 是单位阵，层模块 $\mathcal{T}$ 先把输入乘 2。
依据：自行构造

::: 一步计算的完整过程

1. 汇入：$0.5 \times 2 + 0.5 \times 4 = 3$，这一层的输入是 3。
2. 过层：$2 \times 3 = 6$，层输出是 6。
3. 分发：$\mathbf{A}_r$ 是单位阵，各条流原值仍是 2 和 4；$\mathbf{B} = (1, 0)$ 表示输出只分给第一条流，所以新的第一条流是 $1 \times 6 + 2 = 8$，第二条流是 $0 \times 6 + 4 = 4$。
4. 新的超隐藏矩阵是 $(8, 4)^\top$，各行相加得 $12$，作为输出向量。

```python
H = (2, 4)                 # 两条流，每流一个数
Am = (0.5, 0.5)            # 汇入权重
B = (1, 0)                 # 分发权重
Ar = ((1, 0), (0, 1))      # 各条流原值的去向

h0 = Am[0] * H[0] + Am[1] * H[1]
out = 2 * h0
Hp = (Ar[0][0] * H[0] + Ar[0][1] * H[1], Ar[1][0] * H[0] + Ar[1][1] * H[1])
Hnew = (B[0] * out + Hp[0], B[1] * out + Hp[1])
print("汇入得到这一层的输入", h0)
print("过层后输出", out)
print("新的两条流", Hnew)
```

```text
汇入得到这一层的输入 3.0
过层后输出 6.0
新的两条流 (8.0, 4.0)
```

代码里的 `H` 是初始的超隐藏矩阵，`Am`、`B`、`Ar` 是三块权重，`h0`、`out`、`Hnew` 依次是这一层的输入、层输出和新的超隐藏矩阵。
依据：自行构造
:::

取 $n=1$ 时这个三权重写法收成残差连接：$\mathbf{A}_m$、$\mathbf{B}$、$\mathbf{A}_r$ 都固定成 1，这一层传给下一层的值，就是上一层的输出加上本层模块的输出。残差连接因此是超连接在 $n=1$ 时的不可训练特例。
依据：paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections

#### 连成网络以后：层怎么摆

把每一层的更新式代进下一层，当前层汇入得到的输入可以展开成此前各层输出的加权和，权重是各层那个权重矩阵的乘积。把这些权重排成一张矩阵，就得到层与层之间的连接矩阵，从中能看出哪一层用了哪一层的输出。
依据：paper/src-v3/app.tex Unfolding hyper-connections

::: 展开成层间连接矩阵

1. 汇入那一步是 $\mathbf{h}_0^{k} = \mathbf{H}^{k\top} \mathbf{A}_m^{k}$。
2. 这一层的更新式给出 $\mathbf{H}^{(k+1)\top} = \mathbf{z}^{k} \mathbf{B}^{k} + \mathbf{H}^{k\top} \mathbf{A}_r^{k}$，其中 $\mathbf{z}^{k}$ 是第 $k$ 层算出的输出。
3. 代入后逐层展开到第 0 层：$\mathbf{h}_0^{k} = \sum_{j=0}^{k-1} \mathbf{z}^{j} \mathbf{B}^{j} \left(\prod_{t=j+1}^{k-1} \mathbf{A}_r^{t}\right) \mathbf{A}_m^{k}$。求和里 $j=0$ 的一项取 $\mathbf{z}^{0} = \mathbf{h}^{0}$、$\mathbf{B}^{0} = \mathbf{1}_{1\times n}$，即把第 0 层看成原样传出初始输入、并以全 1 权重分给各条流的恒等映射。
4. 连乘的部分就是第 $j$ 层的输出对第 $k$ 层输入的权重 $\mathbf{B}^{j} \left(\prod_{t=j+1}^{k-1}\mathbf{A}_r^{t}\right) \mathbf{A}_m^{k}$，排成矩阵即连接矩阵 $\mathbf{C}$。

依据：paper/src-v3/app.tex Unfolding hyper-connections
:::

一张连接矩阵可以对应两种极端的层排列：顺序排列，即所有层依次相接，层层都用上一层的输出；并行排列，即相邻的 $n$ 层互不依赖、同时算，组里每层各用一条流，算完再汇总。超连接的权重如果是学出来的，网络就能停在介于两者之间、甚至随 token（文本被切分后的基本单位）变化的位置上。
依据：paper/src-v3/iclr2025_conference.tex Sequential-Parallel Duality；paper/src-v3/app.tex Sequential-Parallel Duality

### 动态超连接：权重由输入算出

权重是固定参数、训练时才更新的叫静态超连接；权重由当前输入算出来的叫动态超连接。动态的算法是：先把超隐藏矩阵过一层归一化，再乘一个线性变换，过 $\tanh$ 把结果压到 $-1$ 到 $1$ 之间，乘一个小系数，最后加上对应的静态权重作为偏置（直接加上去的常数项）。$\tanh$、小系数和归一化都是为了训练稳定。
依据：paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections

::: 计算动态权重的公式

1. 归一化：$\overline{\mathbf{H}} = \texttt{norm}(\mathbf{H})$。
2. 分发的动态增量：$\mathcal{B}(\mathbf{H}) = s_\beta \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_{\beta})^\top + \mathbf{B}$，是 $1 \times n$ 的一行。
3. 汇入的动态增量：$\mathcal{A}_m(\mathbf{H}) = s_\alpha \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_m) + \mathbf{A}_m$，是 $n \times 1$ 的一列。
4. 流间分配的动态增量：$\mathcal{A}_r(\mathbf{H}) = s_\alpha \circ \tanh(\overline{\mathbf{H}} \mathbf{W}_r) + \mathbf{A}_r$，是 $n \times n$ 的块。

$\mathbf{W}_\beta$、$\mathbf{W}_m$、$\mathbf{W}_r$ 是线性变换的权重，$s_\beta$、$s_\alpha$ 是那个小系数，$\circ$ 表示逐元素相乘。
依据：paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections
:::

### 初始化：从残差连接出发

让 $\mathbf{B}$ 初始化为全 1、$\mathbf{A}_m$ 初始化为只选中一条流、$\mathbf{A}_r$ 初始化为单位阵。初始时 $n$ 条流完全相同，之后每层的更新对各条流一致，所以它们始终保持相同，每一条都退化成残差连接的那条通路；动态部分的那几个线性变换权重初始化为 0、小系数初始化为一个很小的数，训练一开始等于没有动态。
依据：paper/src-v3/iclr2025_conference.tex Initialization；paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections

### 效果：n 取多大

扩展率 $n$ 是这套机制新增的超参数（训练前定好、不随训练更新的量）。$n=1$ 时动态超连接不如只用残差连接的对照，说明把残差换成超连接并不自动有效；$n$ 取 2 到 4 时强于该对照，再加大到 8 收益已经很小。
依据：paper/src-v3/iclr2025_conference.tex Ablation Study

::: 实验与结果

动态超连接与静态超连接在 $n=2$ 时接近，$n=4$ 时动态的更好。
依据：paper/src-v3/iclr2025_conference.tex Ablation Study
:::

### 学到了什么样的连接

把训练后的模型展开成连接矩阵，每一层主要依赖相邻几层的输出，这个权重随层距衰减；同时最初的几层被后续很多层引用。前者让权重聚在主对角线附近，后者让权重落在最左几列，两者合起来的形状像一个 $\Lambda$。另一处是浅层注意力的输出很少被远处引用，而前馈网络的输出被保留得更久。还有局部出现的锯齿状接法，等价于把相邻一两层改成并行计算。
依据：paper/src-v3/iclr2025_conference.tex Visualization Analysis；paper/src-v3/sec_visualization.tex L25-L34；paper/src-v3/app.tex Visualization for hyper hidden

### 适用条件与边界

代价方面，连接权重的参数量与算力都很小。$n=4$ 时全模型多出 768（静态）到 394048（动态）个额外参数，占模型总量的 0.00007% 到 0.03349%；每 token 的算力增加不到百分之一。训练时每层留下的中间结果会增加显存（GPU 上的内存），$n=2$ 时约多 15.7%，这部分可以在前向计算（从输入算到输出）结束后丢掉，反向传播（训练时把误差从最后一层往回传，用来算梯度）时再算一遍来回收。
依据：paper/src-v3/app.tex Parameters, Computation and Memory Footprint Analysis；paper/src-v3/app.tex Memory Footprint

一个容易犯的误解是「把主干加宽 $n$ 倍，代价就大」。超隐藏矩阵确实把宽度扩大 $n$ 倍，但新增的只是连接权重本身，参数量和算力都几乎不变。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/app.tex Parameters, Computation and Memory Footprint Analysis

$n>1$ 是这套机制起作用的前提。$n=1$ 时连接强度收成同一个标量，各层距的连接只能同时加强或减弱，做不出「早期层被后续很多层引用」和「只依赖相邻几层」并存的那个 $\Lambda$ 形。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/app.tex L330-L330
