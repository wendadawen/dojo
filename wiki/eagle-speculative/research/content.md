# EAGLE-3 投机解码 draft 模型

- 描述：EAGLE-3 是投机解码里的一种草稿模型，它只有一层 decoder，输入是 target 模型前向已经算出的低、中、高三层隐藏状态融合成的向量，直接输出下一个 token 的分布，并在训练时把自己多步的输出喂回去对齐推理。
- 摘要：草稿角色不再由另训的独立小模型承担，改由一层 decoder 直接读 target 前向算出的低、中、高三层隐藏状态预测 token，让多步草稿的接受率几乎不衰减。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 资料

- Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. arXiv:2503.01840v3, 2025. [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)
- SafeAILab. EAGLE: official implementation. [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

## 短页

### 只读 target 特征的草稿模型

大模型逐 token 生成文本，每生成一个 token 都要把全部参数过一遍，一个接一个，解码慢。投机解码用一个草稿模型先猜出几个 token，再让大模型一次并行核对，猜对的直接接下，猜错的退回重来。这套框架是[投机解码](wiki/speculative-decoding)，要加速的那个大模型叫 target。token 是模型处理文本的最小单位，文本先被切成 token 序列；隐藏状态是模型每一层对每个位置算出的向量，逐层往后传，已经聚了该位置之前的上下文；decoder 是把向量按上下文一层层算下去的模块。EAGLE-3 是一种草稿模型，它只有一层 decoder，输入是 target 算出的低、中、高三层隐藏状态融合成的向量（再拼上刚采样出的那个 token 的嵌入，嵌入是 token 对应的向量），输出下一个 token 的分布。
依据：paper/src-v3/paper.tex L73-L76；paper/src-v3/paper.tex L98-L121；paper/src-v3/paper.tex Speculative Sampling；paper/src-v3/paper.tex Inference Pipeline

### 另训小模型当草稿的难处

通常的草稿模型是一个独立于 target 的小模型，容量有限，很难精确逼近大模型，猜出的分布偏离大模型较多，被接下的比例就低。EAGLE-3 改用另一条路：target 一次前向（从输入一路算到输出的一遍完整计算）本来就会算出每个位置的隐藏状态，把这些已经算好的表示直接喂给草稿，比只看 token 更容易对齐。
依据：paper/src-v3/paper.tex L121；paper/src-v3/paper.tex EAGLE and EAGLE-2

### 融合三层特征，一层 decoder 出草稿

低、中、高三层隐藏状态在同一位置各出一个 k 维向量，k 是隐藏维度。把这 3 个向量首尾接成一个 3k 维向量，过一个全连接层压回 k 维，得到融合特征 g；全连接层是每个输出由各输入各乘一个权重再相加的一层。把 g 与刚采样出的那个 token 的嵌入（token 对应的向量）拼接，过一层 decoder，再由 LM head（把向量映成下一个 token 分布的那一层）得到分布，从中采样就是一条草稿。
依据：paper/src-v3/paper.tex Inference Pipeline

## 长页

大模型逐 token 生成文本，每生成一个 token 都要把全部参数过一遍，一个接一个，解码慢。投机解码把这一步加快：用一个草稿模型先连着猜出几个 token，再让大模型一次并行核对，猜对的直接接下，猜错的退回重来。这套框架是[投机解码](wiki/speculative-decoding)，要加速的那个大模型叫 target 模型。这里的 token 是模型处理文本的最小单位，文本先被切成 token 序列；隐藏状态是模型每一层对每个位置算出的向量，逐层往后传，已经聚了该位置之前的上下文；前向是一次从输入一路算到输出的完整计算；decoder 是把向量按上下文一层层算下去的模块。
依据：paper/src-v3/paper.tex L73-L76；paper/src-v3/paper.tex L98-L121；paper/src-v3/paper.tex Speculative Sampling

EAGLE-3 是投机解码里的一种草稿模型。它不用一个独立的小模型当草稿，而是读 target 在前向里已经算出的隐藏状态：target 一次前向会算出每一层的隐藏状态，EAGLE-3 取低、中、高三层，拼起来过一个全连接层得一个向量，再喂给草稿；草稿本身只有一层 decoder，输出下一个 token 的分布。这样草稿不必自己从 token 重新推出上下文，比另训的小模型更容易对齐；训练时还把自己多步猜出的结果接回输入，让它在多步里也保持准确。
依据：paper/src-v3/paper.tex L192-L215；paper/src-v3/paper.tex L121；paper/src-v3/paper.tex L219-L238

### 草稿模型为什么要读 target 的隐藏状态

一个独立于 target 的小模型来当草稿，容量有限，难以精确逼近大模型：参数少，拟合能力不够，猜出的分布偏离大模型较多，被接下的比例就低。EAGLE-3 改用另一条路：target 一次前向本来就会为每个位置算出隐藏状态，这是 target 自己算出的表示，不必再让草稿从 token 重新推。把这个表示直接喂给草稿，草稿拿到的就是 target 对这段上下文的表示，只要再补一步，就更容易猜中 target 接下来要出的 token。
依据：paper/src-v3/paper.tex L121；paper/src-v3/paper.tex EAGLE and EAGLE-2

EAGLE-3 不再只用最深的那一层，而是把低、中、高三层一起用。只用最深的一层时，它已经紧邻输出，信息集中在下一位上，用它预测更远的位置效果就差。
依据：paper/src-v3/paper.tex L157；paper/src-v3/paper.tex L162

### 低、中、高三层特征融成一个向量

三层分别取自浅、中、深层。EAGLE-3 取这三层在同一位置的各一个 k 维向量，把 3k 维首尾接起来，过一个全连接层压回 k 维，得到融合特征 g，k 是隐藏维度。全连接层是每个输出由各输入各乘一个权重再相加的一层。这个全连接层的权重是学出来的，三层各占多少由训练决定。
依据：paper/src-v3/paper.tex Inference Pipeline

::: 小例子：三层不同粗细的特征怎么融成一个向量
取隐藏维度 k=2，低、中、高三层在某个位置输出的向量分别是 l=(1,2)、m=(2,1)、h=(3,1)。把它们首尾接成一个 6 维向量 (1,2,2,1,3,1)，再乘一个 2×6 的融合矩阵压回 2 维。
依据：自行构造

1. 融合矩阵取两行为 (0.5,0,0.5,0,0.5,0) 与 (0,1,0,1,0,1)。
2. 第一行与 6 维向量逐项相乘再相加：0.5×1+0.5×2+0.5×3=3，只从 l、m、h 的第一个分量取。
3. 第二行：1×2+1×1+1×1=4，只从三层的第二个分量取。
4. 得到融合特征 g=(3,4)，还是 2 维，可直接进草稿层。
依据：自行构造
:::

### 一步草稿：一层 decoder 加 target 的 LM head

草稿每走一步，输入是一串向量：前缀里每个位置各有一个融合特征 g，把该位置的 g 与该位置下一个 token 的嵌入拼起来，就是该位置的输入。这串向量过草稿的一层 decoder，自注意力让末尾位置看到前面所有位置，在末尾位置算出一个向量 a。嵌入是每个 token 对应的一个向量，把离散的 token 变成可以参与计算的向量。
依据：paper/src-v3/paper.tex Inference Pipeline

a 再进 target 的 LM head（把隐藏状态映成各 token 分数的输出层），得到分数，过 softmax 就是下一个 token 的分布，从中采样得到草稿猜出的 token。softmax 把一个分数向量变成加起来为 1 的概率分布。草稿直接借 target 的 LM head，不另训。
依据：paper/src-v3/paper.tex Inference Pipeline

### 多步草稿：用自己上一步的输出补位

要一次猜出几个 token，草稿要连着走几步。第一步能用到 target 真实的融合特征：前缀里那些位置的隐藏状态，target 在之前的前向里已经算好。
依据：paper/src-v3/paper.tex Inference Pipeline

到了第二步，前缀延长了一个新 token，可这个新位置还没被 target 核对过，target 的融合特征还没算出来，草稿拿不到它，只能拿自己上一步的输出代替：上一步算出的那个向量 a，本来要接 LM head 出分布，这里把它当作那个位置的融合特征 g 来用，再和刚猜出的 token 的嵌入拼接，走下一步。后面每一步都这样填。这样草稿的输入会偏离训练时只喂真实特征的情形。
依据：paper/src-v3/paper.tex Inference Pipeline；paper/src-v3/paper.tex L124-L125

::: 例子：四步草稿里，输入怎么从 target 特征换成草稿自己的输出
设前缀是「山 中 相」三个 token，target 一次前向算出下一个 token 是「送」，并留下各位置的隐藏状态。
依据：自行构造

1. 第 1 步：草稿用 target 在「山」「中」「相」三处的融合特征，再拼上「送」的嵌入，过一层 decoder 和 LM head 采样，猜出「客」。
2. 第 2 步：前缀变成「山 中 相 送」。新位置「送」的 target 特征还没有，草稿改用上一步自己输出的那个向量代替这里的融合特征，再拼上刚猜出的「客」的嵌入，猜出「问」。
3. 第 3 步：同理，用上一步的输出代替融合特征，拼上「问」的嵌入，猜出「童」。
4. 第 4 步：用上一步的输出代替融合特征，拼上「童」的嵌入，猜出「子」。
5. 草稿交出「客 问 童 子」四个 token，再一起送 target 核对；越靠后的 token，输入里掺的草稿自己的输出越多。
依据：自行构造
:::

### 训练：把自己多步的输出喂回去

如果让草稿的输出去拟合 target 在下一个位置的隐藏状态，就多出一项特征预测损失。损失是衡量输出与目标偏差、用来更新权重的一个量。这项损失给草稿加了一道额外约束：它最终要的是猜对 token，特征却要照着某一层的向量重建，两项要求互相冲突，限制表达能力，增加数据量改善也有限。EAGLE-3 去掉这项损失，让草稿直接预测下一个 token。
依据：paper/src-v3/paper.tex Draft Model Training；paper/src-v3/paper.tex L124-L125

去掉之后，输入里那份偏离就成了主要问题，于是训练时构造出这种偏离。做法叫 training-time test：训练中不只喂真实特征，还让草稿走一步、把自己的输出接回输入再走下一步，一步步模拟推理时的情形，让草稿在训练时也见过输入含自己上一步输出的情形，仍能给出可用的分布。草稿层里需要少许改动的只有自注意力：自注意力是 decoder 里让各位置互相参考的模块，它决定每个位置能看到哪些位置；训练时要按模拟的需要重排掩码。掩码是决定哪些位置之间允许相看的开关，正好对应推理时的顺序。
依据：paper/src-v3/paper.tex Draft Model Training

::: 因果掩码怎么模拟多步
注意力里，每个位置各算出一个查询和一个键，查询去和各个位置的键做点积，再按点积取值；掩码决定哪些位置之间允许这样相看。训练时草稿要模拟推理的多步，掩码就得跟着改。
依据：paper/src-v3/paper.tex L230-L236

1. 喂进训练的一段是长度为 3 的 token 序列，位置之间本来是正常的前后关系，掩码取标准的左下三角，每个位置看得到自己和前面的位置。
2. 草稿走一步后，三个位置各输出一个 token，每个输出接在原序列某个位置之后，像一棵树的三个分叉，彼此之间没有前后关系。
3. 这三个输出互不相看，所以重排后掩码取对角，让每个位置只看自己，只有在以原始训练数据作键的那几处才例外。
依据：paper/src-v3/paper.tex L230-L236
:::

### 草稿步数更深也不掉接受率

接受率指草稿猜出的 token 里，被 target 接下的比例，比例越高，一次能多接的 token 越多，加速越多。记 $n$-$\alpha$ 为输入里已经掺了 $n$ 个草稿自己产出的向量时的接受率，前提是前面那些草稿 token 都被 target 接下了。$n$ 越大，输入里靠近 target 真实特征的部分越少，越难猜。
依据：paper/src-v3/paper.tex L306-L307

EAGLE-3 的 $n$-$\alpha$ 随 $n$ 增大几乎不变：多猜几步，后面的草稿 token 仍能被接住，这正是训练时把自己输出喂回去换来的。没有这道训练，输入一偏离训练分布，第二步往后的接受率就会下降。
依据：paper/src-v3/paper.tex L320

### 适用条件与边界

草稿要读 target 的中间层隐藏状态，前提是这些中间层的输出能拿到。草稿不改 target 的任何权重，也不改投机解码的接受规则，输出分布和纯 target 采样一样。
依据：paper/src-v3/paper.tex L301；paper/src-v3/paper.tex L162-L164

代价上，草稿本身只有一层 decoder 和一个融合全连接层。接受率随任务变化：上下文里固定模板多的任务，草稿容易猜中，接受率偏高。
依据：paper/src-v3/paper.tex Effectiveness；paper/src-v3/paper.tex Inference Pipeline

一条容易混的地方：EAGLE-3 丢掉的是「输出拟合隐藏状态」，不是「读隐藏状态」。它仍在读 target 的隐藏状态，只是把输出目标从拟合隐藏状态换成直接预测 token；读进来的三层隐藏状态是输入，不是要拟合的目标。
依据：paper/src-v3/paper.tex L124-L125；paper/src-v3/paper.tex L157
