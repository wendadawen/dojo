# DCP

- 描述：DCP（解码上下文并行，decode context parallelism）把解码时不断增长的 KV cache（键和值，每个 token 算出的两个向量）沿 token（模型处理文本的最小单位）位置维切到多张卡上分片存放，每张卡只保留一段，用更小的缓存占用换取更大的批和更高的吞吐。
- 摘要：解码每步只出一个 token 却要读全部历史 KV，缓存能装多少决定并发；DCP 把 KV cache 沿 token 位置分给各卡、每张卡只存一段，从而接纳更多请求。
- 主题：推理系统, 并行与通信, 内存与缓存
- 标签：并行与通信

## 资料

- vLLM 项目，Context Parallel Deployment（官方文档），2026，[https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md)
- Seonghee Lee、Sungsoo Ha、Omri Almog、Lucas Wilkinson，Efficient Decode Context Parallelism with vLLM for Long Context Workloads（vLLM 博客），2026-08-07，[https://vllm.ai/blog/2026-08-07-decode-context-parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)
- vLLM 项目，支持上下文并行与全分片 KV cache、Ring Attention 的 RFC 26133，2025，[https://github.com/vllm-project/vllm/issues/26133](https://github.com/vllm-project/vllm/issues/26133)
- vLLM 项目，vLLM 官方仓库，2026，[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

## 短页

### 每张卡只留一段 KV cache

解码（decode）是模型逐 token 生成、每步都要读全部历史 KV 的阶段。解码上下文并行（DCP）把这份 KV cache 沿 token 位置维切到多张卡，每张卡只存一段，缓存占用随卡数下降。
依据：blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 4. What is DCP?

### 头维切到头就只剩重复

[KV cache](wiki/kv-cache) 起初沿[注意力](wiki/standard-attention)头（注意力让每个 token 和已有的 token 比对、按相关度加权取值，头是其中负责一组键和值的单元，模型有若干个）分配：普通[张量并行](wiki/model-parallelism)把缓存沿头切开，每张卡拿一组头。头数由模型结构定死，是能分给一张卡的最小单位，切到每卡一个头就到上限。分组查询注意力（[GQA](wiki/gqa)）和[多头潜在注意力（MLA）](wiki/mla)的 KV 头都很少，张量并行度超过头数后，多出来的卡只能拿同一组头的重复副本。重复副本占走显存（GPU 的内存），能同时服务的请求数就被压低。
依据：blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts

### 聚查询、算部分注意力、按 LSE 合并

每张卡只存一段 KV，解码一步里新 token 的查询（query，注意力和历史比对用的向量）却要和全部历史键（key）打分，所以先把完整的查询聚到每张卡，各卡在本地切片上算出一份部分注意力和它的 LSE（log-sum-exp，记这一份部分结果总权重的对数值），再用 LSE 给各份结果加权合并，得到和一张卡直接算全部键一致的结果。
依据：blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-context-parallel-deployment.md Decode Context Parallel

## 长页

大语言模型（用大量文本训练、能按上文继续写下去的模型）的一次推理（把一段文字读进来、算出输出，不涉及训练）要把一条请求的整段 token（模型处理文本的最小单位）序列算一遍。序列很长时，可以把它沿 token 位置维切成几份，分给多张卡同时处理，这叫上下文并行（context parallelism，CP）。它切的是同一条序列的 token 位置方向，不同于把权重矩阵切开的[张量并行](wiki/model-parallelism)。同一条请求分两段算：预填充（prefill）一次读入整段提示、算出结果，解码（decode）逐 token 生成。解码上下文并行（DCP）切解码时要反复读的 [KV cache](wiki/kv-cache)（键和值，每个 token 算出的两个向量，缓存下来供后面每步复用），让每张卡只存一段、腾出显存（GPU 的内存），从而增大批大小（一次并行处理的请求数）、提高吞吐。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-rfc-26133-context-parallelism.md Proposed Change

### 预填充算得多，解码存得多

解码阶段走的是自回归（每步用已生成的 token 继续生成下一个）：每步只算少量新 token 的查询（query，[注意力](wiki/standard-attention)和历史比对用的向量），却要读缓存里全部历史 token 的键和值，所以解码是显存容量受限的，缓存能装多少条请求的 KV 决定了能同时跑多大的批、也就决定了吞吐。预填充阶段一次算完全部输入 token、算力受限、慢在首 token 时间。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment

### DCP：把解码的 KV cache 沿 token 位置维分掉

#### 沿头维切到头就开始重复

解码时 KV cache 起初沿[注意力](wiki/standard-attention)头（注意力里每个负责一组键和值的头，模型有若干个）分配：模型有 $H$ 个 KV 头，普通[张量并行](wiki/model-parallelism)把 KV cache 沿 $H$ 维切开，每张卡拿一组头。头是能分给一张卡的最小单位，一张卡不能再只拿半个头。模型的 $H$ 由结构定死，所以当张量并行度 $t$ 超过 $H$ 之后，头部不够分，多出来的卡只能拿同一组头的副本，KV cache 就在卡间重复 $t/H$ 份。两类主流注意力的 KV 头数都不多，会先到这条上限：分组查询注意力（GQA，多个查询头共享少量 KV 头）本来只存少量 KV 头（见 [GQA](wiki/gqa)）；多头潜在注意力（MLA）把键和值压成所有查询头（算查询向量的头）共享的一个潜在向量（键值压缩后得到的低维中间表示），有效 KV 头数就是 $1$，普通张量并行下这份缓存会在每张卡上整份复制（见 [MLA](wiki/mla)）。重复的缓存多占一份显存，能留给请求的余地变小，并发度下降。
依据：blog/vllm-dcp-blog.md 1. Introduction；blog/vllm-dcp-blog.md 3. Challenges of Serving Long Contexts；blog/vllm-context-parallel-deployment.md Decode Context Parallel

#### 每张卡只存一段 token 的 KV

DCP 换一个方向切：沿 token 位置维把 KV cache 分给多张卡，每张卡持有一段唯一的 token 切片，缓存重复随 dcp 增大而减少（dcp 取到 $t/H$ 时才完全消失）。缓存会在解码中不断增长，落位方式要能让新 token 自然分到各卡：交错大小（interleave size，连续多少 token 归一张卡）取 $1$ 时，全局第 $i$ 个 token 的 KV 交给第 $i \bmod D$ 张卡（token 和卡都从 $0$ 起算），$D$ 是 DCP 组大小、也就是参数 dcp；取更大的值时按块轮流落卡。注意力依赖 token 的先后位置，前馈层（逐个位置各自计算的那部分网络）和逐元素运算对每个 token 各自独立。注意力阶段按序列切分之后，这批卡立刻重组起来共同执行前馈：前馈权重分到这批卡上，每张卡只加载一部分，一份权重的加载就摊到整组卡上。
依据：blog/vllm-dcp-blog.md 4. What is DCP?；blog/vllm-dcp-blog.md 7. Conclusion；blog/vllm-context-parallel-deployment.md Decode Context Parallel；repo/vllm/vllm/config/parallel.py L414-L424；blog/vllm-rfc-26133-context-parallelism.md Proposed Change

#### 一步解码的通信与合并

解码一步里，新 token 的查询要和全部历史键打分，而这些键分散在各卡上，所以先要把完整的查询聚到每张卡。这一步在 DCP 组内做一次 AllGather（集合通信里把所有卡的数据聚齐到每张卡的操作，集合通信指多张卡协同的一类通信操作），解码时查询只有一个 token，这次 AllGather 要搬的数据就是这一份查询。有了完整查询，各卡在自己的 KV 切片上做本地注意力，算出一份部分输出和一个本地 LSE（log-sum-exp，用来记这一份部分结果总权重的对数值）。最后共享各卡的部分输出与 LSE，用 LSE 给各份部分结果重新加权再加起来，就得到和一张卡直接算全部键完全相同的结果。合并用的是 online-softmax（在线 softmax——softmax 把一组分数归一化成和为 $1$ 的权重，在线指边算边维护一个累计的缩放量、避免中间值溢出）的技巧：把各卡的 $e^{l_r}$（$l_r$ 是本地 LSE）当权重。MLA 路径下各卡把本地的潜在向量切片上投影（用上投影矩阵把压缩后的潜在向量还原成键和值）回键和值，GQA 路径下共享的 KV 头广播到它的各个查询头。
依据：blog/vllm-dcp-blog.md 4.1 Decode Context Parallelism Process；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend

::: 部分结果按 LSE 合并的公式与一个小例
设 DCP 组里有 $D$ 张卡，第 $r$ 张卡持有若干键，对这些键算出本地 log-sum-exp $l_r=\ln\sum_{i\in r}e^{s_i}$（$s_i$ 是查询对第 $i$ 个键的分数）和用本地分数归一化后的输出 $o_r$。全局输出就是把各卡按 $e^{l_r}$ 加权平均：
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

DCP 的度由参数 dcp 控制，它和普通张量并行一起用，只减少 KV 重复，不增加要启动的卡数。dcp 越大，重复越少，但每步解码的通信越多；未启用 [PCP](wiki/pcp)（预填充上下文并行，把预填充的 token 摊到多张卡）时，推荐取值范围是 $[1, t/H]$，$t$ 是张量并行度、$H$ 是 KV 头数。理论上可以把 dcp 设到超过 $t/H$ 继续分下去，但解码时查询 token 很少，多出来的卡在前馈层（非注意力层）没有明确分工，所以上界就取 $t/H$。MLA 的有效 KV 头数为 $1$，可以切到整个张量并行度，要求 $t\ge dcp$ 且 $t$ 能被 dcp 整除；GQA 的 $H$ 是模型自带的少量 KV 头数，其上界因此也是 $t/H$。
依据：blog/vllm-context-parallel-deployment.md Decode Context Parallel；blog/vllm-dcp-blog.md 5.3 MLA Backend；blog/vllm-dcp-blog.md 5.4 GQA Backend；repo/vllm/vllm/config/parallel.py L587-L600；repo/vllm/vllm/config/model.py L1558-L1587

### 适用条件与边界

DCP 要求所用的注意力后端（实际实现注意力计算的算子）在解码时能返回 LSE，否则合并不成立，换后端或关掉 DCP 才能运行。容易与 DCP 混淆的是[预填充上下文并行（PCP）](wiki/pcp)：PCP 切预填充要算的 token、增加要启动的卡数，用来缩短首 token 时间；DCP 切解码反复读的 KV cache，用来腾显存、提吞吐。常见的两点误解：以为增大 dcp 要额外启动卡，其实它不增加要启动的卡数；以为 dcp 可以随意调大，其实上界由 KV 头数决定。
依据：repo/vllm/vllm/v1/worker/cp_utils.py L21-L57；blog/vllm-context-parallel-deployment.md Context Parallel Deployment；repo/vllm/vllm/config/parallel.py L143-L146
