# Tensor Parallelism

- 描述：张量并行把一层里的权重（可训练的参数）按维度切到多块加速器上，各卡各算一部分，合起来算出同一层。
- 摘要：单卡放不下一层时，把层里的权重矩阵按列或按行切到多卡。切法让 GeLU 等逐元素运算不必通信，代价是每层前向、反向各两次 all-reduce。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 资料

- Mohammad Shoeybi, Mostofa Patwary, Raul Puri 等. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053v4, 2020. [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- Yanping Huang, Youlong Cheng, Ankur Bapna 等. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. arXiv:1811.06965, 2019. [https://arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965)

## 短页

### 把一层里的矩阵切到多卡

神经网络是一层接一层叠起来的计算结构，层里可训练的参数是权重，排成矩阵。

加速器（下面简称卡）是做这些计算的处理器。张量并行（tensor parallelism，TP）把一层里的权重矩阵按维度切开，每块卡持有每一层的一部分权重，所有卡一起算出同一层。切完要把各卡的部分结果合并，合并用的操作是 all-reduce——把所有卡上的张量（多维数组，矩阵是它的二维情形）按元素相加、结果每块卡都拿到一份。一层前向（从输入逐层算到输出）要两次 all-reduce，反向（从输出往回逐层算梯度）也两次。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

它和数据并行不同：数据并行把同一份完整模型复制到每块卡上，各卡处理不同的数据。按层堆叠的深度切的叫流水线并行（[流水线并行](wiki/pipeline-parallelism)），它和按矩阵维度切的张量并行是并列的两条路。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex L7

### 一块卡装不下整个模型

训练时显存（卡上的存储空间）里不只有权重，还有优化器状态——训练时为每个参数额外存下、用来更新它的一份量（比如记录参数历史更新方向的动量）。模型大到权重和优化器状态超出单卡显存时，可以把权重分到多块卡上。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction

### 按列或按行切，输出处 all-reduce

一层由多头注意力（[标准注意力](wiki/standard-attention)）和两层的前馈子网络（multilayer perceptron，MLP）两块组成，两块各自的权重矩阵都按维度切开。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

MLP 的权重可以按列切，这样逐元素的非线性——对张量每个元素单独施加的函数，如 GeLU——能各卡本地施加；也可以按行切。注意力则按 head 切：每个 head 各有一组查询、键、值（Q、K、V）的权重矩阵，head 是注意力里负责一组查询、键、值的单元；这些权重矩阵按列切，正好让一个 head 的全部计算都落在同一块卡上，完成注意力本身不需要通信。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

合并发生在输出处：注意力之后有一个把各头输出拼回隐维度（一层输出向量的宽度）的线性层，它按行切、直接承接各卡本地那部分注意力输出，本步不通信，它的输出再经一次 all-reduce 相加，这是这一层两次 all-reduce 中的一次；MLP 段在第二段按行切之后也有一次 all-reduce，这是另一次。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

## 长页

神经网络是一层接一层叠起来的计算结构，每一层以上一层的输出为输入做一次计算、再把结果传出；层里可训练的参数是权重，排成矩阵。加速器（下面简称卡）是做这些计算、带自己显存的处理器，显存指卡上的存储空间。一次训练一步分前向、反向两步：前向从输入逐层算到输出，反向从输出往回逐层算梯度——梯度是每个参数该往哪个方向调、调多大幅度的指示量。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

模型并行把单个网络的权重切分到多块卡上，各卡只持一部分。张量并行（tensor parallelism，TP）是其中一种：把每一层里的权重矩阵按维度切开，每块卡持有每一层的一部分权重，所有卡一起算出同一层。它和数据并行不同，数据并行把同一份完整模型复制到每块卡上、各卡处理不同的数据。
依据：paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction

一个由「注意力块 + 两层前馈」这样的层一层层堆起来、每层结构相同的模型是 Transformer。一个 Transformer 层由两块组成：一块多头注意力（[标准注意力](wiki/standard-attention)），一块两层的前馈子网络（multilayer perceptron，MLP）。切分后要把各卡的部分结果合并，这个操作是 all-reduce——把所有卡上的张量（多维数组，矩阵是它的二维情形）按元素相加，结果每块卡都拿到一份。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 把一层的矩阵切到多卡

这两块的权重矩阵各自按维度切开，分别放在多块卡上。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### MLP 的列切与行切

MLP 的第一段是一个矩阵乘再跟一个 GeLU（逐元素非线性函数，对张量每个元素单独施加）：$Y=\mathrm{GeLU}(XA)$，其中 $X$ 是这一层的输入，$A$ 是第一段的权重矩阵。要把这一个矩阵乘切到多卡，可以从 $A$ 的两个方向切。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

一种切法只按 $A$ 的列切：把 $A$ 分成左右两块 $[A_1,A_2]$，每块卡用同一个完整输入 $X$ 计算 $XA_i$，得到输出 $Y$ 的一半 $[Y_1,Y_2]=[\mathrm{GeLU}(XA_1),\mathrm{GeLU}(XA_2)]$。GeLU 是逐元素的，可以各自独立地施加到每块卡自己算出的那一半上，这一步之前不需要任何同步。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

另一种切法按 $A$ 的行和 $X$ 的列同时切：把 $A$ 按行分成上下两块 $A^{(1)}$、$A^{(2)}$，把 $X$ 按列分成左右两块 $X^{(1)}$、$X^{(2)}$，每块卡算 $X^{(i)}A^{(i)}$，两块的部分和要加起来才做 GeLU。因为 GeLU 是非线性函数，$\mathrm{GeLU}(X^{(1)}A^{(1)}+X^{(2)}A^{(2)})$ 不等于 $\mathrm{GeLU}(X^{(1)}A^{(1)})$ 加 $\mathrm{GeLU}(X^{(2)}A^{(2)})$，这个加法必须发生在 GeLU 之前，也就是在非线性这一步之前要先同步一次。第一段因此采用列切，省掉这次同步。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

第二段的权重矩阵记作 $B$，它把 GeLU 之后的激活（层算出的、要传给下一层的中间张量）映射回输出维度。$B$ 按行切成 $B^{(1)}$、$B^{(2)}$，每块卡用自己那半激活 $Y_i$ 算 $Y_iB^{(i)}$，直接接上本地结果，这一步不需要通信。两块的结果相加才是最终输出，这一步用一次 all-reduce 完成。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### 小矩阵切分示例

用一个最小的例子确认这两种切法合起来与不切分得到相同结果。
依据：自行构造

::: 4 维输入、两卡切分的手算
1. 输入取 $X=[1,0,-1,2]$；第一段权重按列切成左右两块，各 $4\times2$。
依据：自行构造

2. 第二段权重是单位矩阵，按行切成上下两块，各 $2\times4$。
依据：自行构造

3. 卡 1 算第一段前半 $XA_1$：取 $A_1=\begin{bmatrix}1&0\\1&1\\0&1\\1&0\end{bmatrix}$，得 $[3,-1]$。
依据：自行构造

4. 卡 2 算第一段后半 $XA_2$：取 $A_2=\begin{bmatrix}0&1\\1&0\\1&0\\0&1\end{bmatrix}$，得 $[-1,3]$。
依据：自行构造

5. 逐元素做 GeLU：$\mathrm{GeLU}(3)\approx2.996$，$\mathrm{GeLU}(-1)\approx-0.159$。卡 1 得 $[2.996,-0.159]$，卡 2 得 $[-0.159,2.996]$。
依据：自行构造

6. 第二段单位矩阵按行切成 $B^{(1)}=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}$、$B^{(2)}=\begin{bmatrix}0&0&1&0\\0&0&0&1\end{bmatrix}$。卡 1 算得 $[2.996,-0.159,0,0]$，卡 2 算得 $[0,0,-0.159,2.996]$。
依据：自行构造

7. 两块按元素相加，得输出 $[2.996,-0.159,-0.159,2.996]$。不切分时把两段拼成完整权重一次算完，得到同一个向量。
依据：自行构造
:::

#### 注意力按 head 切

注意力块里，输入先乘三个权重矩阵分别得到查询、键、值（记作 Q、K、V），注意力用它们算出每个位置对其它位置的权重。这三处的矩阵乘按列切：每个头各自有一组 Q、K、V 的权重，按列切正好让一个头的全部计算落在同一块卡上，各卡算各的头，完成注意力本身不需要通信。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

注意力之后有一个把各头输出拼回隐维度的线性层（拿一个权重矩阵对输入做一次矩阵乘、线性地合成输出的层），隐维度就是一层输出向量的宽度。这个线性层按行切，直接承接各卡本地那部分注意力输出，本步不通信；它的输出再经一次 all-reduce 相加，才是这一层注意力的完整结果。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

#### 一层两次 all-reduce，反向对称

一层前向一共两次 all-reduce，反向也是两次；前向的 all-reduce 在把各卡张量按元素相加、在反向直接放过，反向的 all-reduce 在前向放过、在反向才相加。两者互为反向，可以一起接进自动求导（框架自动算梯度的机制），实现时只需几行代码。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 切分后每块卡的显存

训练时显存里除了权重，还有优化器状态——训练为每个参数额外存下、用来更新它的一份量（比如记录参数历史更新方向的动量）。张量并行把模型并行区里的权重及其优化器状态、以及激活按卡切开，每块卡只存自己那一份，整个模型因此能装下比单卡容量更大的规模。
依据：paper/megatron-src-v1/intro.tex Introduction；paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

模型并行区（被张量并行切分覆盖的那部分模型）之外的部分——dropout（训练时随机丢掉一部分单元的输出）、层归一化（把一个位置的向量归一化）、残差连接（[残差连接](wiki/residual-connection)，把输入直接加到输出上）——不切分，而在每块卡上各自重复算一份，省下它们跨卡通信的开销。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers

### 适用条件与边界

张量并行每一层都要 all-reduce，通信量大。它属于资料所称的 SPMD 一类（single program multiple data，所有卡跑同一份程序、处理输入张量的不同分片，分片指把张量沿某个维度切成的小块），这类靠 all-reduce 合并矩阵乘结果的做法，适用场景限于卡之间用高速互联（带宽高的一层互联）相连的时候。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L358

张量并行与流水线并行（[流水线并行](wiki/pipeline-parallelism)，按层堆叠的深度切）正交互补，可以组合使用。
依据：paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/megatron-src-v1/abs.tex L3；paper/megatron-src-v1/discussion.tex L5
