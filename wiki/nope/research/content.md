# NoPE

- 描述：NoPE 是去掉位置编码的解码器 Transformer：位置信息由网络结构和训练自己形成，在比训练更长的序列上仍能算对，且不增加注意力里的计算。
- 摘要：去掉位置编码的解码器 Transformer。每个位置只能看到它自己和前面的位置，能看到的 token 数不同，位置信息由此隐式进入网络；训练后它的注意力接近相对位置编码，近处远处都看，在比训练更长的序列上优于显式位置编码，且不增加注意力的点积计算。
- 主题：注意力机制, 模型结构
- 标签：位置编码

## 资料

- Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy. The Impact of Positional Encoding on Length Generalization in Transformers. NeurIPS 2023. arXiv:2305.19466v2, 2023. [https://arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466)
- McGill-NLP. length-generalization: official implementation. 2023. [https://github.com/McGill-NLP/length-generalization](https://github.com/McGill-NLP/length-generalization)

## 短页

### 去掉位置编码的解码器

NoPE 是一种解码器 Transformer：只保留 Transformer（靠注意力在位置之间传信息的网络）里逐个往后接 token 的那一半网络。token 指序列里的一个基本单位。它每一层里的自注意力（见[标准注意力](wiki/standard-attention)）按内容算出每个位置对前面各位置的权重，加权求和得到该位置的输出。位置编码让模型知道每个 token 排在序列的第几位；绝对位置编码的做法是把一段只随位置变化的向量加到输入嵌入上。NoPE 就是这样一个解码器，但去掉了一切位置编码：输入的 token 只换成嵌入向量，不另加随位置变化的向量，注意力里也不加与位置有关的项。位置信息不显式给，由结构和训练一起形成。
依据：paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Background: Positional Encoding in Transformers；paper/src-v2/main.tex Background；paper/src-v2/main.tex NoPE

### 不加计算反而更能外推

注意力按内容算权重：若没有位置信息、又能看到所有位置，把同一批 token 换一种顺序、内容不变，每个位置算出的输出也不变，模型看不出谁先谁后，所以原版 Transformer 给每个位置加一段只随位置变化的向量，把次序带进去。用这种显式位置编码时，模型在一批短序列上训练好，换成长序列常常算不准：训练时只见过长度不超过某个值的序列，位置编码在更长的位置上没有学好。这种「训练短、测试长仍要算对」的能力叫长度外推。NoPE 不加这段向量，相比在分数上加位置的编码，它的注意力里没有与位置有关的项。这样的解码器在同一批任务上从短到长外推，表现反而比显式位置编码好。
依据：paper/src-v2/main.tex What Is The Effect of Positional Encoding?；paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Background: Positional Encoding in Transformers

### 位置从可见范围里读出来

做法就是不给位置。注意力只让每个位置看它自己和前面的位置，挡住后面的，这种遮罩叫因果掩码。于是每个位置能看到的 token 数不同：第 1 个位置只看得到它自己 1 个，第 $t$ 个位置看得到 $t$ 个。这个可见数量随位置递增，网络拿它当位置的线索：从自己排在第几、能看到多少 token，就能定出位置。这条线索是训练自己学会用的。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE；paper/src-v2/main.tex Background: Positional Encoding in Transformers

## 长页

解码器指每次根据前面已生成的内容、往后接一个 token 的那一半 Transformer。Transformer 主要靠注意力在位置之间传信息；token 是序列里的一个基本单位。解码器由一叠层组成，每层收到一个叫隐藏状态 $\mathbf{H}^{(\ell)}$ 的多维数组，每个位置一个向量，$\ell$ 是层号；层里的自注意力让每个位置用查询向量和前面各位置的键向量做点积，按点积大小对值向量加权求和（见[标准注意力](wiki/standard-attention)）。让模型知道 token 的先后，靠位置编码：绝对位置编码把一段只随位置变化的向量加到输入嵌入上；相对位置编码按位置差编码（如在分数上加偏置，分数指注意力里归一成权重之前的那组数）。模型在短序列上训练好、换成长序列仍要算对，这种能力叫长度外推；显式位置编码在更长位置常常没学好。NoPE 把位置编码全部去掉，位置信息靠因果掩码（每个位置只看到它自己和前面位置的那种遮罩）和训练自己形成，点积里不额外加与位置有关的项，计算也不增加。
依据：paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Background: Positional Encoding in Transformers；paper/src-v2/main.tex Background；paper/src-v2/main.tex NoPE；paper/src-v2/main.tex T5's Relative PE

### 因果掩码让输出仍与次序有关

若注意力能看到序列里的所有位置、包括后面的，又不加位置编码，那么把序列打乱重排、内容不变，每个位置的输出只是跟着重排，模型分不出原本谁在谁前面，对输入的处理就只与 token 的集合有关、与次序无关。解码器的注意力不同：因果掩码要求每个位置只能看到它自己和前面的位置，看不到后面的。这条限制让解码器的输出仍与次序有关——同一个 token，把它前面的内容换成另一批，它这一位置的输出就变了，模型不会因为看不到位置编码就完全丢掉次序。
依据：paper/src-v2/main.tex Background: Positional Encoding in Transformers

### 位置信息从结构里自己形成

#### 第一层能写出绝对位置

第一层可以用因果掩码和一个固定表示序列起点的特殊 token（下称起点 token），把绝对位置写进隐藏状态 $\mathbf{H}^{(1)}$。做法是让每个位置对前面所有位置的注意力权重都相等，等于可见范围的倒数 $1/t$，其中 $t$ 是当前的位置编号，并让只有起点 token 的值向量非零；这样加权求和后，起点 token 的值被乘上 $1/t$，输出就把 $t$ 带了进去。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE

::: 权重都是 1/t 的构造
1. 起点 token 在词嵌入里单独留一维标出来，其余 token 这一维为 0。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE
2. 键的投影矩阵取成让所有位置算出的键都相同，于是任意位置 $t$ 对所有可见位置算出的注意力分数都相等。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE
3. 分数相等，经 SoftMax（把一组分数归一成和为 1 的权重）后每个可见位置得权重 $1/t$，只与位置编号 $t$ 有关。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE
4. 值的投影让只有起点 token 的值向量非零，加权求和后这一项被乘上 $1/t$；把它写回隐藏状态的某一维，这一维就等于 $1/t$。同层的前馈层（每层里接在注意力之后的那层网络）再从这一维算出位置编号 $t$，写回同一维。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE
:::

::: 小例子：4 个位置的第一层输出
取一条 4 个位置的序列，起点 token 在第 1 位。按上面的构造，只有起点 token 的值向量非零。
依据：自行构造
1. 位置 1：可见范围只有它自己，$t=1$，权重 $1/1=1$，输出那一维为 $1$。
2. 位置 2：可见 2 个位置，权重各 $1/2$，只有起点非零，输出为 $1/2$。
3. 位置 3：可见 3 个位置，输出为 $1/3$。
4. 位置 4：可见 4 个位置，输出为 $1/4$。
注意力输出这一维是 $1/t$，前馈层取它的倒数，同一维就变成绝对位置 $1,2,3,4$。
依据：自行构造
:::

#### 后续层能实现相对编码

如果第一层已经把绝对位置写进了隐藏状态，并且后面各层不覆盖它，那么从第二层起的注意力可以把点积里的位置部分组织成只与位置差有关。构造的办法是让查询和键各自读隐藏状态里的位置维：查询里放绝对位置的一个线性项，键里放另一个，两者点积后位置部分相减，只剩位置差。
依据：paper/src-v2/main.tex Relative Positional Encoding in NoPE

::: 点积里只剩位置差
1. 设第一层已把位置编号 $t$ 写进隐藏状态的某一维，位置 $t$ 处这一维等于 $t$。
依据：paper/src-v2/main.tex Relative Positional Encoding in NoPE
2. 查询投影取成让查询的第 1 维等于 $1$、第 2 维等于 $-t$；键投影取成让键的第 1 维等于它自己的位置 $i$、第 2 维等于 $1$。这两维的点积是 $1\cdot i + (-t)\cdot 1 = i - t$。
依据：paper/src-v2/main.tex Relative Positional Encoding in NoPE
3. 取位置 5 的查询对位置 2 的键，这两维得 $1\times2 + (-5)\times1 = -3$，位置差 $5-2=3$，即 $-(t-i)$；再取位置 4 的查询对位置 1 的键，得 $1\times1 + (-4)\times1 = -3$，位置差同样是 $3$。
依据：自行构造
4. 其余各维照常算内容相关的那部分，加在一起得到 $\langle \mathbf{q}_t, \mathbf{k}_i \rangle = f_{\text{cnt}}(\mathbf{q}_t, \mathbf{k}_i) - (t-i)$：第一项只看内容，第二项只看位置差；只要其余各维的权重不把这两项抵消，点积里就带一个只与相对距离 $t-i$ 有关的项，等价于相对位置编码。
依据：paper/src-v2/main.tex Relative Positional Encoding in NoPE
:::

### 训练后的注意力像相对编码

训练结束后，NoPE 各注意力头（注意力分多个头并行算，每个头有一套自己的查询、键、值投影）对每个位置算出的注意力分布，接近一个用相对位置编码（把相对距离查表变成一个加在分数上的偏置）的模型。
依据：paper/src-v2/main.tex NoPE learns to use relative PE in practice

::: 怎么量出「最接近」
1. 把同一个输入分别输入两个模型，逐层取出各注意力头对每个位置算出的概率分布，用 JS 距离（衡量两个概率分布差异的量）比较。
依据：paper/src-v2/main.tex NoPE learns to use relative PE in practice
2. 两个模型在第 $\ell$ 层里任意一对头的距离取最小值，就是这个层两模型的差异，越小说明越像。
依据：paper/src-v2/main.tex NoPE learns to use relative PE in practice
3. 换不同随机种子训练的 NoPE 互相比，作为距离的参照。
依据：paper/src-v2/main.tex NoPE learns to use relative PE in practice
:::

### 注意力近处远处都看

在加法任务加完整思维链格式（完整思维链指让模型把中间步骤写进输出的解法）上，NoPE 的注意力既落在近处，也落在远处：近处是当前这一步，远处是题目输入。
依据：paper/src-v2/main.tex Which part of the sequence is attended to?

::: 分布的形状
把输出序列各位置「当前查询到被关注键的距离、按当前长度归一化」的值 $\bar{d}$ 统计成分布：接近 0 表示看近处，接近 1 表示看远处。NoPE 的分布呈双峰，近处一峰、远处一峰。
依据：paper/src-v2/main.tex Which part of the sequence is attended to?
:::

### 不增加注意力的计算

NoPE 不在注意力里算任何与位置有关的额外项。它的注意力点积就是查询和键的点积 $\langle \mathbf{q}_t, \mathbf{k}_i \rangle = \mathbf{q}_t^\top \mathbf{k}_i$，没有别的加项；相比要在分数上加位置的编码，这一步少算一项，运行时间和显存占用也就不因为位置编码而增加。
依据：paper/src-v2/main.tex What Is The Effect of Positional Encoding?；paper/src-v2/main.tex NoPE；paper/src-v2/main.tex Model Evaluation

### 适用条件与边界

成立的范围内是解码器 Transformer、带因果掩码、从零训练。因果掩码是前提：没有它，注意力对次序不敏感，位置无从读出。
依据：paper/src-v2/main.tex Model Evaluation；paper/src-v2/main.tex Background: Positional Encoding in Transformers

代价上，在 1.3B（13 亿参数）的代码模型上做过初步实验，NoPE 能外推到训练上下文之外的一定长度，再长则困惑度（衡量模型预测下一个 token 的不确定性的量，越小越好）上升。这是初步结果，不是下游任务精度。
依据：paper/src-v2/main.tex Pretraining at 1.3B Scale；paper/src-v2/main.tex Generalization Evaluation

::: 1.3B 上的困惑度
1. 训练上下文 1024 个 token；测试上下文从 512 到 2560。
依据：paper/src-v2/main.tex Generalization Evaluation
2. 上下文不超过 1024 时，各编码的困惑度接近。
依据：paper/src-v2/main.tex Generalization Evaluation
3. NoPE 与偏好近处的编码（在分数上加一个随距离线性增大的负偏置）能外推到约 1800；再长时两者困惑度都上升，偏好近处的编码相对更稳。
依据：paper/src-v2/main.tex Generalization Evaluation
:::

一处容易误解的是「不加位置编码就等于没有位置信息」：第一层就能把绝对位置写进隐藏状态，位置信息在，只是不靠显式编码给。另一处是「思维链能代替位置编码」：把中间步骤写进输出，只在加法这类任务上有帮助，并不能补上位置编码的缺失。
依据：paper/src-v2/main.tex Absolute Positional Encoding in NoPE；paper/src-v2/main.tex Does Scratchpad Render The Choice of Positional Encoding Irrelevant?
