# PCP

- 描述：PCP（预填充上下文并行，prefill context parallelism）把一条长请求的预填充序列计算切成几段、摊到多张卡上并行算，用来缩短预填充时间、降低用户等待首 token 的时间。
- 摘要：预填充要一次算完全部输入 token、慢在首 token 时间；PCP 把这 $T$ 个新 token 切成 $N$ 段，每张卡算一段的查询、键、值，装得下全量键值就各算各段，装不下用 Ring Attention 逐块收发。
- 主题：推理系统, 并行与通信
- 标签：并行与通信

## 资料

- vLLM 项目，Context Parallel Deployment（官方文档），2026，[https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md)
- Seonghee Lee、Sungsoo Ha、Omri Almog、Lucas Wilkinson，Efficient Decode Context Parallelism with vLLM for Long Context Workloads（vLLM 博客），2026-08-07，[https://vllm.ai/blog/2026-08-07-decode-context-parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)
- vLLM 项目，支持上下文并行与全分片 KV cache、Ring Attention 的 RFC 26133，2025，[https://github.com/vllm-project/vllm/issues/26133](https://github.com/vllm-project/vllm/issues/26133)
- vLLM 项目，vLLM 官方仓库，2026，[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

## 短页

### 每张卡只算一段 token

预填充（prefill）是模型一次读入整段提示、算出结果的阶段。大语言模型推理要把一条请求的整段 token（模型处理文本的最小单位）序列算一遍，序列很长时，可以把它沿 token 维切成几份，分给多张卡同时处理。预填充上下文并行（PCP）把预填充这段序列计算沿 token 维切分、摊到多张卡上并行做。
依据：blog/vllm-context-parallel-deployment.md Prefill Context Parallel

### 首 token 的等待太长

预填充要一次算完全部输入 token，输入越长这段计算越重，用户能感觉到的是首 token 的等待时间（time to first token，TTFT）。把它摊到多张卡上并行算，这段等待才能缩短。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment

### 切成 N 段各算一段

把本次要算的 $T$ 个 token 切成 $N$ 段，每张卡算一段的查询（query）、键（key）、值（value）三类向量（每个 token 各一套；注意力用查询和键打分、再按分数对值加权求和）。装得下完整的键、值时，把各卡的键值聚到一起、各卡算自己那段查询的输出；装不下时每张卡只留本段，用 Ring Attention（各卡把本地键值块沿环传给下一张卡、边收边算，不必一次装下全部键值）逐块收发键、值。
依据：blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-rfc-26133-context-parallelism.md Prefill

## 长页

大语言模型推理时要把一条请求的整段 token（模型处理文本的最小单位）序列算一遍。序列很长时，可以把它沿 token 维切成几份，分给多张卡同时处理，这叫上下文并行（context parallelism，CP）。它切的是同一条序列的长度方向，不同于把权重矩阵切开的[张量并行](wiki/tensor-parallelism)。同一条请求分两段算：预填充（prefill）一次读入整段提示、算出结果，解码（decode）逐 token 生成。预填充上下文并行（PCP）切预填充时要算的 token，把这批序列计算摊到多张卡上并行做，用来缩短预填充的时间。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-rfc-26133-context-parallelism.md Motivation

### 预填充算得多，解码存得多

预填充阶段，一条有 $T$ 个新 token 的长请求要一次算出这些 token 的查询、键、值张量（注意力用查询和键打分、再按分数对值加权求和）。[注意力](wiki/standard-attention)里每个 token 要和已有的 token 打分，输入越长这一步越重，用户能感觉到的是首 token 的等待时间（time to first token，TTFT）。解码阶段每步只算少量新 token 的查询，却要读键值缓存（KV cache，每个 token 算出的键和值缓存下来供后面复用）里全部历史 token 的键和值，缓存装得下多少决定能同时服务多少请求，这段压力由[解码上下文并行（DCP）](wiki/dcp)处理。
依据：blog/vllm-context-parallel-deployment.md Context Parallel Deployment；blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension

### PCP：把预填充的序列计算摊到多张卡

预填充这样一条长请求，用 $N$ 张卡，就把请求切成 $N$ 段，每张卡算一段的查询、键、值。第一种是「部分查询、完整键值」：请求长度中等、还装得下完整的键和值张量时，把各卡的键、值聚到一起，每张卡算自己那段查询对应的注意力输出。第二种是「部分查询、部分键值」：长度太长、已经装不下完整键值时，每张卡只算一段的查询、键、值，再用 Ring Attention（各卡把本地键值块沿环传给下一张卡、边收边算，不必一次装下全部键值）让各卡逐块收发键、值。传哪一侧看查询与键值张量的大小：未分块的整段预填充一般传键值分片，[Chunked Prefill](wiki/chunked-prefill)（把一条预填充切成小块、与解码请求同批分多次算完）且缓存远长于查询时传查询分片更省。
依据：blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-rfc-26133-context-parallelism.md Prefill

#### 因果负载与首尾配对

[因果掩码](wiki/causal-mask)下第 $j$ 个 token 只能看到它自己和前面的 token，要打分的键有 $j+1$ 个，位置越靠后计算越重。把序列平均划分成 $2N$ 个块（$N$ 是卡数；块是比段更细的单位），越靠后的块这一块上的计算越重：第 $0$ 块最轻、最后一块最重。若按顺序把相邻块分给同一张卡，拿到末尾重块的卡负载会比别的卡重。配平的做法叫首尾配对：让第 $i$ 张卡拿第 $i$ 块和第 $(2N-1-i)$ 块、一轻一重相加，这两块一般分处序列的前后两半、位置对称。
依据：blog/vllm-rfc-26133-context-parallelism.md Partition Sequence Dimension

::: 首尾配对的手算例
取 $N=2$、共 $4$ 块的例子，每块的计算量按 $2i+1$ 算，即 $1$、$3$、$5$、$7$。
依据：自行构造

1. 第 $0$ 张卡拿第 $0$ 块和第 $3$ 块：$1+7=8$。
依据：自行构造
2. 第 $1$ 张卡拿第 $1$ 块和第 $2$ 块：$3+5=8$。
依据：自行构造
3. 两张卡的负载都是 $8$。
依据：自行构造
:::

### 适用条件与边界

PCP 是一种独立的并行方式：它增加要启动的卡数，但不增加 KV cache 的分片数，只把预填充的计算摊开。两种策略都还在开发中。与它容易混淆的是[解码上下文并行（DCP）](wiki/dcp)：DCP 不增加要启动的卡数，切的是解码时反复读的 KV cache，用来腾显存、提吞吐；PCP 切的是预填充要算的 token，用来缩短首 token 时间。
依据：repo/vllm/vllm/config/parallel.py L143-L146；blog/vllm-context-parallel-deployment.md Prefill Context Parallel；blog/vllm-context-parallel-deployment.md Context Parallel Deployment
