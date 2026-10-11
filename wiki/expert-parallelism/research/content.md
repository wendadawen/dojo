# Expert Parallelism

- 描述：专家并行（EP）把模型的专家分片到多张 GPU 上，一个 token 用到的专家不在本卡时，它的激活（token 在层间流动的向量）经卡间通信送过去算再送回来，这两步叫 dispatch 与 combine。
- 摘要：专家总数大，把专家分片到多张卡、每张卡只放一部分，整体批随卡数一起扩大，好让每个专家收到足够多的 token。一个 token 要用的专家常在别的卡上，它的激活得用 all-to-all 搬过去算、再搬回来；各卡收到的 token 数不同，最慢的一张卡会拖慢整步，需要用冗余专家与调度来均衡。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 资料

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz 等. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017. [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI. DeepSeek-V3/R1 Inference System Overview. 2025. [https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

## 短页

### 把专家分到多张卡上算

在一类模型（混合专家，见 [MoE](wiki/moe)）里，每一层有一排专家，每个 token 只交给其中少数几个处理，用哪几个由路由器决定；每个 token 选中的专家个数记 $k$，这种选法叫 top-k。专家并行（expert parallelism，EP）把这些专家分到多张 GPU（图形处理器，做并行数值计算的芯片）上，每张卡只放其中一部分。token 是模型读写文本的最小单位；一次前向（输入经模型算一遍得到输出）不是只算一个 token，而是把同时到达的一批 token 合成一批一起算；token 在层间流动的向量叫它的激活。多张卡之间互相收发数据这种通信方式叫 all-to-all。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)

### 每个专家分到的批太小

一个 token 只选少数几个专家，一层里有 $n$ 个专家时，一批 $b$ 个 token 分给每个专家的只有 $kb/n$ 个，这个数远小于 $b$，分到每个专家头上的运算量很小、效率低。把专家分到多张卡以后，各卡指向同一个专家的 token 会汇到一起喂给它，整体批可以随卡数扩大，每个专家收到的 token 数跟着变多，计算效率才够高；每张卡只算少数几个专家，单卡的访存也更省。
依据：paper/src-shazeer/main.tex Addressing Performance Challenges；blog/deepseek-v3-r1-inference-system-overview.md System Design Principles

### 搬 token、均衡负载、重叠通信与计算

一个 token 要算的那几个专家可能不在同一张卡上。它的激活要先被送到存放对应专家的卡上，这一步叫 dispatch；专家算完，结果再被送回原来那张卡，这一步叫 combine。两者都用 all-to-all 实现。各张卡收到的 token 数不一样，一步计算要等所有卡都算完，最慢的那张卡就拖慢整步，服务端把高负载的专家复制成多份多放几个副本，让负载高的卡分走一些 token。通信进行时卡上的算力空闲，把一批拆成两个更小的微批，微批甲在通信时微批乙在算专家，反过来也一样，于是一个微批的通信被另一个微批的计算掩盖。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping；blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing

## 长页

一个模型的专家总数可以很大。服务时把各专家分片到多张 GPU（图形处理器，做并行数值计算的芯片）上，每张卡只放其中一部分，这种把专家分开摆放的方式叫专家并行（expert parallelism，EP）。token 是模型读写文本的最小单位；每个 token 只有少数几个专家真正参与计算、其余不算，这叫稀疏激活。专家来自混合专家模型（见 [MoE](wiki/moe)），每个 token 用哪几个专家，由路由器给各专家打分、按分数高低选出；token 在层间流动的向量叫它的激活。多张卡之间互相收发数据这种通信方式叫 all-to-all。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)

一次前向（输入经模型算一遍得到输出）会同时处理一批 token，批更大时矩阵运算的效率更高。把专家铺到更多卡上以后，多张卡各自的批可以合起来喂给同一个专家，整体批随卡数一起扩大，每个专家因此能分到足够多的 token；同时每张卡只算少数几个专家，单卡的访存也更省，延迟更低。
依据：paper/src-shazeer/main.tex Addressing Performance Challenges；blog/deepseek-v3-r1-inference-system-overview.md System Design Principles

### 专家分片到多张卡

分片之后，一个 token 要算的那几个专家可能不在同一张卡上。它的激活要先被送到存放对应专家的卡上，这一步叫 dispatch；专家算完，结果再被送回原来那张卡，这一步叫 combine。两者都用 all-to-all 实现：每张卡都可能和其他所有卡互相收发数据。$k$ 记每个 token 选中的专家个数，它与专家总数之比越小，每个专家收到的 token 就越少，越需要靠扩大整体批来补。
依据：paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)；paper/src-shazeer/main.tex Addressing Performance Challenges

::: 2 张卡、4 个专家、top-2 的搬运计数
设 2 张卡，卡 A 放专家 1、2，卡 B 放专家 3、4；每个 token 选 top-2（即 $k=2$）。4 个 token 选中的专家是：$t_1$ 选 $\{1,2\}$，$t_2$ 选 $\{1,2\}$，$t_3$ 选 $\{1,3\}$，$t_4$ 选 $\{1,4\}$。
依据：自行构造

1. 每个 token 送 2 个（token, 专家）对，4 个 token 共 8 对。按卡归类：卡 A 的专家 1 收到 $t_1,t_2,t_3,t_4$ 共 4 个 token，专家 2 收到 $t_1,t_2$ 共 2 个，卡 A 合计 6 对；卡 B 的专家 3 收到 $t_3$、专家 4 收到 $t_4$，合计 2 对。
2. combine 把算好的结果按同样的对应送回，也是卡 A 6 对、卡 B 2 对。
3. 这一步的耗时由收到（token, 专家）对最多的卡 A 决定，卡 B 的 2 对算完就得等卡 A 的 6 对，卡 B 空闲等待。
依据：自行构造
:::

### 负载不均：最慢的卡决定整步

专家分片之后，一个 token 由路由器决定去哪些专家，各张卡收到的 token 数因此不一样。一步计算要等所有卡都算完才算完，最慢的那张卡就决定了整步的耗时：一张卡负载过高，整批都要等它，其余卡空闲。服务端把高负载的专家复制成多份、多放几个副本，让负载过高的卡能分走一些 token；再按各卡的实际负载调度，使每张卡收到的 token 数尽量相等。
依据：blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing；paper/src-dsv3/main.tex Prefilling

### 通信与专家计算重叠

EP 的 dispatch 与 combine 是卡间通信，开销可观。一个做法是把一批拆成两个更小的微批：微批甲在做 dispatch 与 combine 的时候，微批乙在做计算（[注意力](wiki/standard-attention)与专家计算），反过来也一样，于是一个微批的通信与另一个微批的计算重叠，通信的开销被计算掩盖。
依据：blog/deepseek-v3-r1-inference-system-overview.md Computation-Communication Overlapping；paper/src-dsv3/main.tex Prefilling

### 适用条件与边界

稀疏激活省下的计算，来自每个 token 只算 $k$ 个专家：$k$ 远小于专家数时节省的计算多，$k$ 接近专家数时节省有限。EP 处理的是专家放在哪张卡、以及 token 的激活怎么搬到对应卡上算这两件事；一次请求从收到到出字的快慢由服务系统的指标衡量，见 [LLM 服务](wiki/llm-serving)。
依据：paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Inference and Deployment；blog/deepseek-v3-r1-inference-system-overview.md Large-scale Cross-node Expert Parallelism (EP)
