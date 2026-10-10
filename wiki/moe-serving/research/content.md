# MoE 大模型推理与服务基础

- 描述：MoE 大模型推理与服务基础，讲一种把部分层的前馈网络换成一排专家的模型：每个 token 由路由器选中少数几个专家计算，总参数随专家数增长而每 token 计算量不涨；这些专家要分片到多张 GPU 上，一次请求分 prefill 与 decode 两个阶段执行。
- 摘要：把前馈层换成一排专家，每个 token 只让其中几个参与计算，于是总参数可以很大而每 token 的计算量几乎不涨；代价是专家要分片到多张 GPU 上，token 的激活得在卡间用 all-to-all 搬运。一次请求分 prefill 与 decode 两个阶段，服务快慢要拆成首字延迟与出字速度分别度量。
- 主题：推理系统, 模型结构, 并行与通信
- 标签：MoE

## 资料

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz 等. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017. [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI. DeepSeek-V3/R1 Inference System Overview. 2025. [https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- Yinmin Zhong 等. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI 2024. [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)

## 短页

### 一排专家，每个 token 只让几个参与计算

MoE（混合专家）是一种模型结构。大语言模型的主干由许多结构相同的层堆叠而成，每层里有一个前馈网络（FFN），它把每个位置的向量各自做一次变换。MoE 把一部分层的这个前馈网络换成一排并行的专家，每个专家本身也是一个小前馈网络；另有一个路由器，它给每个 token（模型读写文本的最小单位，也就是一个位置）打一个分数，只留下分数最高的 k 个，让这几个专家算，再按分数把它们的结果加权相加。这里 k 是每个 token 选中的专家个数，这种选法叫 top-k；没有被选中的专家对这个 token 不参与计算。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Architecture；paper/src-dsv3/main.tex Hyper-Parameters

### 存得多、算得少

稠密模型是每个 token 都过全部参数、没有专家选择的普通模型。它的参数越多，每个 token 要做的计算也越多。MoE 把这两件事分开：总参数随专家数一起增长，而每个 token 只经过被选中的那几个专家，所以每 token 的计算量只随选中的个数增长，不随专家总数增长。
依据：paper/src-dsv3/main.tex Introduction；paper/src-shazeer/main.tex Introduction and Related Work

### 切到多卡、分成两段执行

专家总数多，服务时把专家分片到多张卡上，每张卡只放一部分专家。一个 token 该去哪几个专家由路由器决定，它的激活（这个 token 在层间流动的向量）要送到存放对应专家的卡上算、算完再送回来，这种多张卡之间互相收发数据的通信方式叫 all-to-all。一次请求的执行分成两段：prefill 一次并行处理整段输入、把各层算出的 key 和 value（注意力里每个位置存下的两个向量）缓存下来、产出第一个 token；decode 每步只算一个新 token、复用前面的缓存。服务快慢相应地拆成两个指标：TTFT 是首字延迟，由 prefill 决定；TPOT 是出字速度，由 decode 决定。
依据：paper/src-dsv3/main.tex Inference and Deployment；paper/src-distserve/sections/background.tex LLM Inference；paper/src-distserve/sections/intro.tex Introduction

## 长页

大语言模型处理文本时，先把文本切成 token，token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分。模型里可以训练的数字叫参数；一次前向计算是让 token 的向量逐层流过网络、和参数做乘加。主干由许多结构相同的层堆叠而成，每层有两部分：注意力让位置之间交换信息，前馈网络（FFN）把每个位置的向量各自做一次变换。混合专家（MoE）是一类稀疏激活的结构：把一部分层的前馈网络换成一排并行的小前馈网络，每个 token 只让其中少数几个参与计算。
依据：paper/src-shazeer/main.tex Introduction and Related Work；paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex Architecture；paper/src-dsv3/main.tex L722-L727

这样换的好处是总参数可以随专家的数量一起变大，而每个 token 的计算量只随被选中的专家个数增长。代价落在服务侧：全部专家要分片到多张 GPU（图形处理器，做并行数值计算的芯片）上，token 的激活（这个 token 在当前层等待处理的向量）得在卡间搬运。
依据：paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex Inference and Deployment；paper/src-shazeer/main.tex Introduction and Related Work

### 一排专家加一个路由器

MoE 层由 n 个专家 $E_1, \dots, E_n$ 和一个路由器（也叫门控网络）组成。每个专家本身是一个前馈网络，有独立的参数，各专家彼此结构相同。路由器以 token 的输入向量为输入，给这 n 个专家各算一个分数，只保留分数最高的 k 个（这叫 top-k 选择），其余专家的分数置为零。这一层的输出是被选中专家输出的加权和：每个选中的专家按自己的分数占比分到一个权重，用权重乘上它的输出再相加。分数为零的专家，其输出不必计算，稀疏激活省下的就是这部分。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-shazeer/main.tex Gating Network

::: MoE 层的输出公式
记 $G(x)$ 为路由器对输入 $x$ 给出的权重向量，$E_i(x)$ 为第 $i$ 个专家的输出，MoE 层的输出是

$$ y = \sum_{i=1}^{n} G(x)_i E_i(x) $$

$G(x)_i$ 是第 $i$ 个专家分到的权重。这就是「稀疏」的含义：求和里真正参与计算的非零项只有 k 个。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer
:::

有的实现还额外设一个共享专家，它对所有 token 都计算，不经过路由选择。以 DeepSeek-V3（DeepSeek 的 MoE 大语言模型）为例，它的 MoE 层用 1 个共享专家加 256 个路由专家：每个 token 在这一层的输出，等于它自己的输入、加上共享专家的输出、再加上 8 个被选中路由专家输出的加权和。路由用的分数由一个 sigmoid 函数给出（把实数映射到 0 到 1 之间），在选中的 8 个里归一化成权重；未被选中的路由专家权重为零。
依据：paper/src-dsv3/main.tex Architecture；paper/src-dsv3/main.tex L722-L727

::: 8 个专家选 2 个，走一遍加权
设一层有 8 个路由专家，每个 token 选 top-2。路由器给某个 token 的 8 个分数依次取 0.10、0.70、0.20、0.05、0.40、0.15、0.30、0.20。
依据：自行构造

1. 分数最高的两个是第 2 个（0.70）和第 5 个（0.40），它们是这个 token 选中的专家，其余 6 个不计算。
2. 在这两个分数上归一化：$0.70/(0.70+0.40)=0.64$、$0.40/(0.70+0.40)=0.36$，两个专家的权重就是 0.64 和 0.36。
3. 这一层的专家部分输出是 $0.64\,E_2(x) + 0.36\,E_5(x)$。
依据：自行构造
:::

### 存得多、算得少

MoE 的用处是把模型的容量和每 token 的计算量分开。稠密模型要提高容量，就得多加参数，参数和每 token 的计算量一起涨；MoE 增加专家数只增加总参数，每个 token 仍然只算被选中的 k 个，计算量不随专家总数增长。DeepSeek-V3 做到 671B 总参数、每 token 只激活其中 37B；它只把前 3 层的前馈网络保持稠密，其余层换成 MoE。
依据：paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex L722-L727；paper/src-shazeer/main.tex Introduction and Related Work

### 专家并行：把专家切到多张卡

一个模型的专家总数可以很大。服务时把各专家分片到多张卡上，每张卡只放其中一部分，这种把专家分开摆放的方式叫专家并行（expert parallelism，EP）。分片的一个理由和路由性质有关：每层只有少数专家被选中，要让每个专家都收到足够多的 token，就需要很大的整体批；专家分散到多张卡之后，整体批可以随卡数一起扩大，每个专家因此能收到足够多的 token。每张卡只算少数几个专家，单卡的访存也更省。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)；paper/src-shazeer/main.tex Addressing Performance Challenges

分片之后，一个 token 要算的那几个专家可能不在同一张卡上。它的激活要先被送到存放对应专家的卡上，这一步叫 dispatch；专家算完，结果再被送回原来那张卡，这一步叫 combine。两者都用 all-to-all 实现：每张卡都可能和其他所有卡互相收发数据。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)

::: 2 张卡、4 个专家、top-2 的搬运计数
设 2 张卡，卡 A 放专家 1、2，卡 B 放专家 3、4；每个 token 选 top-2。4 个 token 选中的专家是：$t_1$ 选 $\{1,2\}$，$t_2$ 选 $\{1,2\}$，$t_3$ 选 $\{1,3\}$，$t_4$ 选 $\{1,4\}$。
依据：自行构造

1. 每个 token 送 2 个（token, 专家）对，4 个 token 共 8 对。按卡归类：卡 A 的专家 1 收到 $t_1,t_2,t_3,t_4$ 共 4 个 token，专家 2 收到 $t_1,t_2$ 共 2 个，卡 A 合计 6 对；卡 B 的专家 3 收到 $t_3$、专家 4 收到 $t_4$，合计 2 对。
2. combine 把算好的结果按同样的对应送回，也是卡 A 6 对、卡 B 2 对。
3. 这一步的耗时由收到（token, 专家）对最多的卡 A 决定，卡 B 的 2 对算完就得等卡 A 的 6 对，卡 B 空闲等待。
依据：自行构造
:::

### 负载不均：最慢的卡决定整步

专家分片之后，一个 token 由路由器决定去哪些专家，各张卡收到的 token 数因此不一样。一步计算要等所有卡都算完才算完，最慢的那张卡就决定了整步的耗时：一张卡负载过高，整批都要等它，其余卡空闲。服务端把高负载的专家复制成多份、多放几个副本，让负载过高的卡能分走一些 token；再按各卡的实际负载调度，使每张卡收到的 token 数尽量相等。
依据：blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing；paper/src-dsv3/main.tex Prefilling；paper/src-shazeer/main.tex Balancing Expert Utilization

### 通信与专家计算重叠

一次前向不只算一个 token，而是把同时到达的一批 token 合成一批一起算，批更大时矩阵运算的效率更高。EP 的 dispatch 与 combine 是卡间通信，通信进行时卡上的算力空闲。一个做法是把一批拆成两个更小的微批：微批甲在做 dispatch 与 combine 的时候，微批乙在做专家计算，反过来也一样，于是一个微批的通信与另一个微批的计算重叠。通信没有消失，仍占用网络带宽，只有在有计算可并行推进时才能被重叠掉。
依据：blog/deepseek-v3-r1-inference-system-overview.md System Design Principles；blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping；paper/src-dsv3/main.tex Prefilling

### prefill 和 decode：一次请求的两个阶段

一次请求不是一步算完的。模型每算一步产出一个 token，下一步再基于已经有的全部 token 算下一个。服务把执行分成两个阶段。prefill 阶段一次性并行处理用户输入的全部 token，为每一层算出并保存 key 和 value 张量（这就是 [KV cache](wiki/kv-cache)），并产出第一个 token；输入不短时它要算的 token 多，倾向于计算受限。decode 阶段每步只处理一个新 token，复用前缀已缓存好的 key 和 value，逐 token 往下生成；它每步要搬运的权重和缓存与 prefill 相当，却只算一个 token，受限于访存。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex LLM Inference；paper/src-dsv3/main.tex Inference and Deployment

### 服务好不好怎么量：TTFT、TPOT 与 goodput

等第一个 token 出来的时间和后续每个 token 出来的速度是两个不同的量，要分成两个指标。TTFT（time to first token，首字延迟）是 prefill 阶段的时长；TPOT（time per output token，每 token 输出时间）是 decode 阶段平均生成一个 token 的时间，首 token 不计。一次请求的总延迟等于 TTFT 加上 TPOT 乘以 decode 阶段生成的 token 数（也就是总生成 token 数减 1）。不同应用对两者的要求不同：实时对话看重 TTFT，TPOT 只要快过人的阅读速度就够用。SLO（service-level objective，服务级目标）是对 TTFT 与 TPOT 分别设的上限，以及要满足这个上限的请求占比，叫达成率。goodput 是在满足达成率的前提下、每张卡每秒最多能服务的请求数。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Background and Motivation

### 部署：两阶段合设还是分离

prefill 和 decode 共用同一份模型权重，所以默认把两个阶段放在同一批 GPU 上，这叫合设。合设省显存，也不用来回搬缓存，但两个阶段会互相干扰：一个 prefill 步比一个 decode 步长得多，它们排在同一批里，decode 的每个 token 都被 prefill 拖慢；反过来，decode 的加入也把 prefill 拉长。另一个做法是把两个阶段分到不同的 GPU 上，叫分离：prefill 用一组卡、decode 用另一组卡，干扰就没有了，两组卡还能各自按所属阶段的特性配置资源。分离的代价是每组卡都要各存一份完整的模型权重，prefill 算完要把缓存传给 decode 那组卡，并且扩容时以承载一整份模型权重的一组卡为一个最小单位，不能只加其中一张。
依据：paper/src-distserve/sections/intro.tex Introduction；paper/src-distserve/sections/background.tex Problems and Opportunities；paper/src-dsv3/main.tex Decoding

### 适用条件与边界

稀疏激活省下的计算，来自每个 token 只算 k 个专家：k 远小于专家数时节省的计算多，k 接近专家数时节省有限。这里的数字都对应某一份具体配置，换了模型、专家数或 top-k，参数和布局的数字都跟着变，但路由、dispatch 与 combine、两阶段这些机制不变。两条容易混的地方：其一，每 token 只激活 37B 参数，不等于模型只要 37B 的显存，671B 参数都要存放；其二，goodput 不等于吞吐，裸吞吐再高，若很多请求超出 SLO，goodput 仍然低。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Hyper-Parameters；paper/src-distserve/sections/background.tex Background and Motivation；paper/src-distserve/sections/intro.tex Introduction
