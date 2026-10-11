# MoE

- 描述：MoE（混合专家）是一种把一部分层的前馈网络换成一排专家的模型结构，路由器为每个 token 只选其中少数几个参与计算，总参数随专家数增长而每 token 计算量不随之成比例增长。
- 摘要：把一部分层的前馈网络换成一排并行的专家，每个 token 由路由器打分、只让分数最高的 $k$ 个专家参与计算，这一层的输出是这些专家输出的加权和。总参数随专家数一起变大，每个 token 的计算量却不随之成比例增长。
- 主题：模型结构, 推理系统
- 标签：MoE

## 资料

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz 等. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017. [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

## 短页

### 一排专家，每个 token 只让几个参与计算

MoE（混合专家）是一种模型结构。大语言模型的主干由许多结构相同的层堆叠而成，每层里有一个前馈网络（FFN），它把每个位置的向量各自做一次变换；这些变换用到的可训练数字叫参数。MoE 把前馈网络换成一排并行的专家，每个专家本身也是一个小前馈网络；另有一个路由器，它以 token（模型读写文本的最小单位，也就是一个位置）的向量为输入，按分数为每个 token 挑选少数几个专家来算，再把这几个专家的结果加权相加。这里 $k$ 是每个 token 选中的专家个数；没有被选中的专家对这个 token 不参与计算。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Architecture；paper/src-dsv3/main.tex Hyper-Parameters

### 参数多、每 token 计算少

稠密模型是每个 token 都过全部参数、没有专家选择的普通模型。它的参数越多，每个 token 要做的计算也越多。MoE 把这两件事分开：总参数随专家数一起增长，而每个 token 只经过被选中的那几个专家，所以每 token 的计算量不随专家总数成比例增长。
依据：paper/src-dsv3/main.tex Introduction；paper/src-shazeer/main.tex Introduction and Related Work

### 怎么选出这几个专家

路由器以 token 的向量为输入，给每个专家算一个分数，只保留分数最高的 $k$ 个（这种选法叫 top-k），其余专家的分数置为零。每个被选中的专家按自己的分数占比分到一个权重，这一层的输出就是它们各自输出按权重的和。用哪几个专家，由路由器按分数决定；这个按分数挑专家的过程叫路由。
依据：paper/src-shazeer/main.tex Gating Network；paper/src-dsv3/main.tex Architecture

## 长页

大语言模型处理文本时，先把文本切成 token，token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分。模型里可以训练的数字叫参数；一次前向计算是让 token 的向量逐层流过网络、和参数做乘加。主干由许多结构相同的层堆叠而成，每层有两部分：注意力让位置之间交换信息，前馈网络（FFN）把每个位置的向量各自做一次变换。混合专家（MoE）是一类稀疏激活的结构：把一部分层的前馈网络换成一排并行的小前馈网络，每个 token 只让其中少数几个参与计算，其余的不算，这种只让一部分参与的做法就是稀疏激活。
依据：paper/src-shazeer/main.tex Introduction and Related Work；paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex Architecture

### 一排专家加一个路由器

MoE 层由 $n$ 个专家 $E_1, \dots, E_n$ 和一个路由器（也叫门控网络）组成。每个专家本身是一个前馈网络，有独立的参数，各专家彼此结构相同。路由器以 token 的输入向量为输入，给这 $n$ 个专家各算一个分数，只保留分数最高的 $k$ 个（这叫 top-k 选择），其余专家的分数置为零。这一层的输出是被选中专家输出的加权和：每个选中的专家按自己的分数占比分到一个权重，用权重乘上它的输出再相加。分数为零的专家，其输出不必计算，稀疏激活省下的就是这部分。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-shazeer/main.tex Gating Network

::: MoE 层的输出公式
记 $G(x)$ 为路由器对输入 $x$ 给出的权重向量，$E_i(x)$ 为第 $i$ 个专家的输出，MoE 层的输出是

$$ y = \sum_{i=1}^{n} G(x)_i E_i(x) $$

$G(x)_i$ 是第 $i$ 个专家分到的权重。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer
:::

有的实现还额外设一个共享专家：它对所有 token 都计算、不经过路由选择，与按分数选出的路由专家并列参与这一层的输出。
依据：paper/src-dsv3/main.tex Architecture

::: 8 个专家选 2 个，走一遍加权
设一层有 8 个路由专家，每个 token 选 top-2（即 $k=2$）。路由器给某个 token 的 8 个分数依次取 0.10、0.70、0.20、0.05、0.40、0.15、0.30、0.20。
依据：自行构造

1. 分数最高的两个是第 2 个（0.70）和第 5 个（0.40），它们是这个 token 选中的专家，其余 6 个不计算。
2. 在这两个分数上归一化：$0.70/(0.70+0.40)=0.64$、$0.40/(0.70+0.40)=0.36$，两个专家的权重就是 0.64 和 0.36。
3. 这一层的专家部分输出是 $0.64\,E_2(x) + 0.36\,E_5(x)$。
依据：自行构造
:::

### 参数多、每 token 计算少

MoE 的用处是把模型的容量和每 token 的计算量分开；容量指模型能表达和记住的东西，随总参数一起变大。稠密模型（每个 token 都过全部参数、不做专家选择的普通模型）要提高容量，就得多加参数，参数和每 token 的计算量一起增长；MoE 增加专家数只增加总参数，每个 token 仍然只算被选中的 $k$ 个，计算量不随专家总数成比例增长。DeepSeek-V3 做到 671B 总参数（B 表示十亿）、每 token 只激活其中 37B，把容量和计算量分开。
依据：paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex Hyper-Parameters；paper/src-shazeer/main.tex Introduction and Related Work

### 路由集中在少数专家上

路由器按各专家当下的分数选出要参与的几个专家。分数高的专家被选中的机会更多、在训练里被更新得更快，路由器自身也在训练、于是更常给这几个专家打高分，下一轮它们被选中的机会更大。这样路由器会逐渐收敛到总是给同样的少数几个专家打高分，各专家分到的样本数（负载）向少数几个集中，这个过程自我强化。负载集中在分布式硬件上会带来内存与性能问题：少数专家收到很多样本，其余专家收到的很少、更新得少。
依据：paper/src-shazeer/main.tex Balancing Expert Utilization

训练时会加一个约束来抵消这个趋势；训练里被最小化的那个目标量叫损失。一种做法是给重要性加一项损失：把每个专家在一批样本上收到的路由分数之和定义为它的重要性，在总损失里加上重要性这组数值的变异系数的平方（乘一个手工调的比例），促使各专家的重要性趋于相等；再加一项负载损失，让各专家收到的样本数也趋于相等。另一种做法是给每个专家设一个偏置项，只用来决定 top-k：某专家过载就把它的偏置调低、低载就调高，偏置不参与最终的加权，加权仍由原始分数给出。
依据：paper/src-shazeer/main.tex Balancing Expert Utilization；paper/src-shazeer/main.tex Load-Balancing Loss；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing

### 适用条件与边界

稀疏激活省下的计算，来自每个 token 只算 $k$ 个专家：$k$ 远小于专家数时节省的计算多，$k$ 接近专家数时节省有限。MoE 是模型结构本身，不含专家放在哪张卡上、token 的向量怎么在卡间搬运——那是[专家并行](wiki/expert-parallelism)的事；一次请求怎么服务，是[LLM 服务](wiki/llm-serving)的事。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Hyper-Parameters
