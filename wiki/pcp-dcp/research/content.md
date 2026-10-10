# PCP 与 DCP

- 描述：PCP 与 DCP 是推理时上下文并行的两个阶段形态：预填充上下文并行（PCP）把一条长请求的预填充序列计算切到多张卡上摊开，解码上下文并行（DCP）把解码时不断增长的 KV cache 沿 token 位置切到多张卡上分片存储，两者都用来服务长上下文请求。
- 摘要：同一条长请求，预填充一次算完全部输入、压力在算力和等待首 token 的时间，解码每步只出一个 token 却要读全部历史 KV、压力在显存容量。PCP 把预填充的新 token 切成多段并行算，DCP 把 KV cache 沿 token 位置分片、每步用 LSE 合并各卡的部分注意力结果。
- 主题：推理系统, 并行与通信, 内存与缓存
- 标签：并行与通信

## 资料

- vLLM 项目，Context Parallel Deployment（官方文档），2026，[https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md)
- Seonghee Lee、Sungsoo Ha、Omri Almog、Lucas Wilkinson，Efficient Decode Context Parallelism with vLLM for Long Context Workloads（vLLM 博客），2026-08-07，[https://vllm.ai/blog/2026-08-07-decode-context-parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)
- vLLM 项目，支持上下文并行与全分片 KV cache、Ring Attention 的 RFC 26133，2025，[https://github.com/vllm-project/vllm/issues/26133](https://github.com/vllm-project/vllm/issues/26133)
- vLLM 项目，vLLM 官方仓库，2026，[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

## 短页

### 把一条长请求切到多张卡上

大语言模型推理要先把一条请求的整段 token（模型处理文本的最小单位）序列算一遍。序列很长时，可以把它沿 token 维切成几份，分给多张卡同时处理，这叫上下文并行。它切的是序列的长度方向。同一条请求分两段算：预填充一次读入整段提示、算出结果，解码逐 token 生成。上下文并行因此分成两种形态，预填充上下文并行（PCP）切预填充要算的 token，解码上下文并行（DCP）切解码时要反复读的 KV cache（键和值，每个 token 算出的两个向量，缓存下来供后面每步复用）。前者把预填充的算力摊开，后者把缓存的存放摊开。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction

### 预填充费算力，解码费显存

一条长请求的预填充要一次算完全部输入 token，慢的是首 token 时间，需要把这段计算摊薄。解码每步只出一个 token，却要读全部历史 KV，缓存装得下多少决定能同时服务多少条请求，也就是吞吐。KV 起初沿注意力头（注意力让每个 token 和已有 token 比对、按相关度取值，头是里面负责一组键和值的小单元）切开，而头数由模型结构定死，切到上限后就开始在卡间重复。两个阶段一个受限于算力、一个受限于显存，于是要沿序列长度这个还没用上的方向分开切。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction

### 预填充切计算，解码切缓存

PCP 把预填充的 $T$ 个新 token 切成 $N$ 段，每张卡算一段。每个 token 会算出查询（query）、键（key）、值（value）三个向量，注意力用查询和键打分、再按分数对值加权求和。装得下全部键、值时，把各卡的键、值聚到一起、各卡算自己那段查询的输出；装不下时各卡只留本段，用 Ring Attention（各卡把本地键值块沿环依次传给下一张卡，边收边算，不必一次装下全部键值）逐块收发键、值。DCP 让每张卡只存一段 token 的 KV cache，解码每步先聚齐查询，各卡在本地切片上算出一份部分注意力和它的 LSE（log-sum-exp，记这一份结果总权重的对数值），再用 LSE 把各卡结果加起来，得到和一张卡直接算全部键一致的结果。
依据：blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process

## 长页

大语言模型推理时要把一条请求的整段 token（模型处理文本的最小单位）序列算一遍。序列很长时，可以把它沿 token 维切成几份，分给多张卡同时处理，这叫上下文并行（context parallelism，CP）。它切的是同一条序列的长度方向，不同于把权重矩阵切开的[张量并行](wiki/model-parallelism)。同一条请求分两段算：预填充（prefill）一次读入整段提示、算出结果，解码（decode）逐 token 生成。预填充上下文并行（PCP）切预填充时要算的 token，让预填充更快；解码上下文并行（DCP）切解码时要反复读的 [KV cache](wiki/kv-cache)（键和值，每个 token 算出的两个向量，缓存下来供后面每步复用），让每张卡腾出 KV 空间、增大批大小，从而提高吞吐。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-rfc-26133-context-parallelism.md Motivation

### 预填充算得多，解码存得多

预填充阶段，一条有 $T$ 个新 token 的长请求要一次算出这些 token 的查询、键、值张量（注意力用查询和键打分、再按分数对值加权求和）。[注意力](wiki/standard-attention)里每个 token 要和已有的 token 打分，输入越长这一步越重，所以预填充是算力受限的，用户能感觉到的是首 token 的等待时间（time to first token，TTFT），要缩短它就得把预填充的计算分摊到多个 token 上并行。解码阶段走的是自回归：每步只算少量新 token 的查询，却要访问缓存里大量历史 token 的键和值，所以解码是显存容量受限的，缓存能装多少条请求的 KV 决定了能同时跑多大的批、也就决定了吞吐。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment

### DCP：把解码的 KV cache 沿序列维分掉

#### 沿头维切到头就开始重复

解码时 KV cache 起初沿注意力头（注意力里每个负责一组键和值的头，模型有若干个）分配：模型有 $H$ 个 KV 头，普通[张量并行](wiki/model-parallelism)把 KV cache 沿 $H$ 维切开，每张卡拿一组头。头是能分给一张卡的最小单位，一张卡不能再只拿半个头。模型的 $H$ 由结构定死，所以当张量并行度 $t$ 超过 $H$ 之后，头部不够分，多出来的卡只能拿同一组头的副本，KV cache 就在卡间重复 $t/H$ 份。两类主流注意力的 KV 头数都不多，会先到这条上限：分组查询注意力（GQA）本来只存少量 KV 头；多头潜在注意力（MLA）把键和值压成所有查询头（算查询向量的头）共享的一个潜在向量（键值压缩后得到的低维中间表示），有效 KV 头数就是 $1$，普通张量并行下这份缓存会在每张卡上整份复制，两页分别是 [MQA 与 GQA](wiki/mqa-gqa) 和 [MLA](wiki/mla)。重复的缓存多占一份显存，能留给请求的余地变小，并发度下降。
依据：blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts；blog/vllm-context-parallel-deployment.md Decode Context Parallel

#### 每张卡只存一段 token 的 KV

DCP 换一个方向切：沿 token 位置维把 KV cache 分给多张卡，每张卡持有唯一的一份切片，容量不花在副本上，省出的显存用来增大批大小、接纳更多请求。缓存会在解码中不断增长，落位方式要能让新 token 自然分到各卡：交错大小（interleave size，连续多少 token 归一张卡）取 $1$ 时，全局第 $i$ 个 token 的 KV 交给第 $i \bmod D$ 张卡，$D$ 是 DCP 组大小；取更大的值时按块轮流落卡。注意力依赖序列方向，前馈层（逐个位置各自计算的那部分网络）和逐元素运算对每个 token 各自独立。注意力阶段按序列切分之后，这批卡立刻重组起来执行前馈，前馈权重的加载就被摊到整组卡上。
依据：blog/vllm-dcp-blog.md 4. What is DCP?；blog/vllm-dcp-blog.md 7. Conclusion；blog/vllm-context-parallel-deployment.md Decode Context Parallel；repo/vllm/vllm/config/parallel.py L414-L424；blog/vllm-rfc-26133-context-parallelism.md Proposed Change

#### 一步解码的通信与合并

解码一步里，新 token 的查询要和全部历史键打分，而这些键分散在各卡上，所以先要把完整的查询聚到每张卡。这一步在 DCP 组内做一次 AllGather（集合通信里把所有卡的数据聚齐到每张卡的操作），解码时查询只有一个 token，这次 AllGather 要搬的数据就是这一份查询。有了完整查询，各卡在自己的 KV 切片上做本地注意力，算出一份部分输出和一个本地 LSE（log-sum-exp，用来记这一份部分结果总权重的对数值）。最后共享各卡的部分输出与 LSE，用 LSE 给各份部分结果重新加权再加起来，就得到和一张卡直接算全部键完全相同的结果。合并用的是 online-softmax（在线 softmax——softmax 把一组分数归一化成和为 $1$ 的权重，在线指边算边维护一个累计的缩放量、避免中间值溢出）的技巧：把各卡的 LSE 当权重。MLA 路径下各卡把本地的潜在向量切片上投影（用上投影矩阵把压缩后的潜在向量还原成键和值）回键和值，GQA 路径下共享的 KV 头广播到它的各个查询头。
依据：blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend

::: 部分结果按 LSE 合并的公式与一个小例
设 DCP 组里有 $R$ 张卡，第 $r$ 张卡持有若干键，对这些键算出本地 log-sum-exp $l_r=\ln\sum_{i\in r}e^{s_i}$（$s_i$ 是查询对第 $i$ 个键的分数）和用本地分数归一化后的输出 $o_r$。全局输出就是把各卡按 $e^{l_r}$ 加权平均：
依据：自行构造

$$o=\frac{\sum_r e^{l_r}\,o_r}{\sum_r e^{l_r}}$$
依据：自行构造

它和一张卡直接对全部键做 softmax 的结果恒等，因为把全局 softmax 的分子分母按卡分组，每组的分子正是 $e^{l_r}o_r$。
依据：自行构造

取 $2$ 张卡、$4$ 个键。第 1 张卡持前 $2$ 个键，分数都是 $0$，值取 $1$ 和 $5$；第 2 张卡持后 $2$ 个键，分数都是 $\ln 3$，值取 $2$ 和 $6$。
依据：自行构造

1. 第 1 张卡：$l_1=\ln(e^{0}+e^{0})=\ln 2$，权重 $e^{l_1}=2$，输出 $o_1=(1+5)/2=3$。
依据：自行构造
2. 第 2 张卡：$l_2=\ln(e^{\ln 3}+e^{\ln 3})=\ln 6$，权重 $e^{l_2}=6$，输出 $o_2=(2+6)/2=4$。
依据：自行构造
3. 合并：$(2\times 3+6\times 4)/(2+6)=30/8=3.75$。
依据：自行构造
4. 直接算全部 $4$ 个键：权重为 $1$、$1$、$3$、$3$ 归一化，输出 $(1\times 1+1\times 5+3\times 2+3\times 6)/8=30/8=3.75$，与合并结果相同。
依据：自行构造
:::

#### dcp 的取值与约束

DCP 的度由参数 dcp 控制，它和普通张量并行一起用，只减少 KV 重复，不增加要启动的卡数。dcp 越大，重复越少，但每步解码的通信越多；没开 PCP 时，推荐取值范围是 $[1, t/H]$，$t$ 是张量并行度、$H$ 是 KV 头数。理论上可以把 dcp 设到超过 $t/H$ 继续分下去，但解码时查询 token 很少，多出来的卡在非注意力层没有明确分工，所以上界就取 $t/H$。MLA 有效 KV 头数为 $1$，可以切到整个张量并行度，要求 $t\ge dcp$ 且 $t$ 能被 dcp 整除；GQA 的切分度被头维切分后剩下的重复因子（张量并行度除以 KV 头数）封顶在 $t/H$。开了 PCP 时，dcp 只能取 $1$、pcp（预填充上下文并行的度），或 $t\times pcp$（跨整个张量并行与 PCP 合成的进程组）。
依据：blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend；repo/vllm/vllm/config/parallel.py L587-L600；repo/vllm/vllm/config/model.py L1558-L1587

### PCP：把预填充的序列计算摊到多张卡

预填充这样一条长请求，用 $N$ 张卡，就把请求切成 $N$ 段，每张卡算一段的查询、键、值。第一种是「部分查询、完整键值」：请求长度中等、还装得下完整的键和值张量时，把各卡的键、值聚到一起，每张卡算自己那段查询对应的注意力输出，这样算力就摊到了各段查询上。第二种是「部分查询、部分键值」：长度太长、已经装不下完整键值时，每张卡只算一段的查询、键、值，再用 Ring Attention（各卡把本地键值块沿环依次传给下一张卡，边收边算，不必一次装下全部键值）让各卡逐块收发键、值。两种策略都还在开发中。
依据：blog/vllm-context-parallel-deployment.md Prefill Context Parallel

::: 因果掩码造成的负载不均与首尾配对
[因果掩码](wiki/causal-mask)下第 $j$ 个 token 只能看到它自己和前面的 token，要打分的键有 $j+1$ 个，位置越靠后计算越重。把序列平均切成 $2N$ 块（$N$ 是卡数），第 $i$ 块的注意力计算量正比于 $2i+1$，第 $0$ 块最轻、最后一块最重。若按顺序把相邻块分给同一张卡，拿到末尾重块的卡负载会比别的卡重。
依据：blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension

首尾配对的做法是让第 $i$ 张卡同时拿第 $i$ 块和第 $(2N-1-i)$ 块，一轻一重配平。
依据：blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension

取 $N=2$、共 $4$ 块的例子，四块的权重按 $1$、$3$、$5$、$7$ 算。
依据：自行构造

1. 第 $0$ 张卡拿第 $0$ 块和第 $3$ 块：$1+7=8$。
依据：自行构造
2. 第 $1$ 张卡拿第 $1$ 块和第 $2$ 块：$3+5=8$。
依据：自行构造
3. 两张卡的负载都是 $8$，相同。
依据：自行构造
:::

### 两个机制为什么要分开

PCP 和 DCP 是两个独立的参数，管的是不同的事。PCP 是一个独立的进程维度：它扩展进程总数（总数等于流水线并行度乘张量并行度再乘 PCP 度，流水线并行度是把不同层分给不同卡的并行度），但不增加 KV cache 的分片数，也就是说它只把预填充的计算摊开，缓存还是分成同样多的片。DCP 不扩展进程总数，它落在已有的卡上：没开 PCP 时复用张量并行的这些卡。
依据：repo/vllm/vllm/config/parallel.py L143-L146；repo/vllm/vllm/config/parallel.py L380-L384；repo/vllm/vllm/config/parallel.py L940-L944

### 适用条件与边界

DCP 要求所用的注意力后端（实际实现注意力计算的算子）在解码时能返回 LSE，否则合并不成立，换后端或关掉 DCP 才能运行。容易与 PCP 混淆的是 [Chunked Prefill](wiki/chunked-prefill)：它把一条预填充切成小块、与解码请求同批分多次算完，改的是时间上的切分；PCP 把预填充的 token 一次切到多张卡上同时算，改的是空间上的切分。
依据：repo/vllm/vllm/v1/worker/cp_utils.py L51-L57；repo/vllm/docs/configuration/optimization.md Chunked Prefill；blog/vllm-context-parallel-deployment.md Prefill Context Parallel
