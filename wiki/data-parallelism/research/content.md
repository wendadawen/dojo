# Data Parallelism

- 描述：数据并行把同一份模型复制到多个推理工作进程，每个进程独立处理不同请求，输入按负载均衡分给各进程。
- 摘要：请求级 DP 用多份完整模型并行处理不同请求来提高吞吐；每个副本保留自己请求的 KV cache，路由器按当前负载分配请求。它是复制品之间的分工，不是把一层切到多卡。
- 主题：推理系统, 并行与通信
- 标签：推理系统

## 资料

- NVIDIA TensorRT LLM. Parallelism in TensorRT LLM. 官方文档. 2026-10-15 快照. [https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html](https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html)
- vLLM. Data Parallel Deployment. 官方文档. 2026-10-15 快照. [https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- SGLang. Server Arguments. 官方文档. 2026-10-15 快照. [https://docs.sglang.ai/advanced_features/server_arguments.html](https://docs.sglang.ai/advanced_features/server_arguments.html)

## 短页

### 完整模型的多个服务副本

推理是让模型从输入生成输出；这里的模型是一组可复制的学习参数。推理系统里的数据并行（Data Parallelism，DP）把同一模型复制到多个 rank。一个 rank 是一个独立工作进程；它持有自己请求的 [KV cache](wiki/kv-cache)。
依据：blog/tensorrt-llm-parallel-strategy.md L12-L13；blog/vllm-data-parallel-deployment.md L3；blog/vllm-data-parallel-deployment.md L14-L20

GPU 是执行计算的卡。
依据：自行构造

### 有足量请求时扩大吞吐

单张 GPU 的性能不足或需要更高吞吐时，DP 让多份完整模型同时服务不同请求，提高单位时间完成量。内存足够时 DP 的吞吐更好，大批量、高吞吐是适用场景。
依据：blog/tensorrt-llm-parallel-strategy.md L4-L6；blog/tensorrt-llm-parallel-strategy.md L32-L36；blog/sglang-server-arguments.md L37

### 把请求分到对应 rank

路由器先选出一个 rank，再把新请求交给它独立处理。vLLM 的一个做法是按当前运行或等待队列选。
依据：blog/vllm-data-parallel-deployment.md L14-L20；blog/vllm-data-parallel-deployment.md L25-L48；blog/vllm-data-parallel-deployment.md L148

轮询是另一种可选策略。
依据：blog/sglang-server-arguments.md L1228-L1231

## 长页

推理是让模型从输入生成输出；这里的模型是一组可复制的学习参数。GPU 是实际执行计算的卡。推理系统里的数据并行（Data Parallelism，DP）把整个模型复制到多个 GPU。这里的请求是用户提交的一次输入和它要的生成结果。纯请求级 DP 中，每个副本处理不同请求，各自独立完成推理；rank 是承载模型副本的一个工作进程。SGLang 用 `--data-parallel-size` 或 `--dp-size` 配置数据并行副本的数量。
依据：blog/tensorrt-llm-parallel-strategy.md L4-L13；blog/tensorrt-llm-parallel-strategy.md L32-L36；blog/vllm-data-parallel-deployment.md L3；blog/vllm-data-parallel-deployment.md L5-L6；blog/vllm-data-parallel-deployment.md L14-L20；blog/sglang-server-arguments.md L1216-L1218

### 一个 DP rank 怎么处理请求

一个 rank 持有整份模型权重。它接到一个或一批请求后，从输入算到输出。不同请求由负载均衡分给不同 rank；例如在线部署中，vLLM 当前的内部负载均衡按各工作进程的运行和等待队列分配新请求。
依据：blog/tensorrt-llm-parallel-strategy.md L34-L36；blog/vllm-data-parallel-deployment.md L14-L20；blog/vllm-data-parallel-deployment.md L20；blog/vllm-data-parallel-deployment.md L25-L48；blog/vllm-data-parallel-deployment.md L148

#### 请求分流的例子

设 2 个 rank 各已有负载为 1 的运行请求。6 个新请求的长度依次是 3、1、4、1、5、9；这里的长度是例子里计量负载的抽象单位，每个请求的负载就是它的长度。
依据：自行构造

::: 两种分流过程与比较
1. 轮询不看长度：6 个请求按 0、1 循环分给两个 rank。rank 0 的新增负载是 $3+4+5=12$，rank 1 是 $1+1+9=11$。
2. 按当前待处理负载分流：两个 rank 起点都是 1。第 1 个请求给 rank 0，累计到 $4,1$；第 2 个请求给更短的 rank 1，累计到 $4,2$；第 3 个请求在 $4$ 和 $2$ 之间给 rank 1，累计到 $4,6$；第 4 个请求在 $4$ 和 $6$ 之间给 rank 0，累计到 $5,6$；第 5 个请求在 $5$ 和 $6$ 之间给 rank 0，累计到 $10,6$；第 6 个请求在 $10$ 和 $6$ 之间给 rank 1，累计到 $10,15$。
3. 6 个请求的负载合计是 $23$。若比较含初始请求的最终总负载，轮询是 $13,12$，按当前负载分流是 $10,15$，最大值从 13 变成 15；若只比较新增负载，则分别是 $12,11$ 和 $9,14$，最大值从 12 变成 14。
依据：自行构造
:::

### 每个副本为什么有独立 KV cache

[KV cache](wiki/kv-cache) 的请求状态不随模型权重复制。不同用户请求被路由到不同 DP rank，因此 KV cache 按 rank 划分。
依据：blog/tensorrt-llm-parallel-strategy.md L81-L86

::: 2 个 rank 的缓存容量例子
1. 设 rank 0 和 rank 1 各有一个独立的 KV cache 池，每个池最多放 4 个 token。
2. rank 0 已放 3 个 token，只能再放 1 个；rank 1 已放 1 个 token，只能再放 3 个。
3. 若把两个池当作一个 8-token 合并池，rank 0 超出自己 4-token 上限的状态可以放进合并池的剩余空间；两个独立池没有这个共用空间。
4. 因此这个例子里的缓存容量属于每个 rank，不是先合并成一池再分。
依据：自行构造

:::

token 是模型输入或输出的文本片段。在 vLLM 中，DP engine 是承载一个 rank 的独立模型工作进程；每个 DP engine 有独立 KV cache。
依据：blog/tensorrt-llm-parallel-strategy.md L81-L86；blog/vllm-data-parallel-deployment.md L14-L20

### 它为什么提高吞吐

在有足量请求且内存足够时，DP 有利于提高吞吐。
依据：blog/tensorrt-llm-parallel-strategy.md L34-L36；blog/sglang-server-arguments.md L37

::: 单位时间完成量的例子
1. 设 rank A、B 各用 4 秒完成 2 个请求。
2. 两 rank 并行在 4 秒内完成 4 个请求，单位时间完成量是 $4/4=1$ 个/秒。
3. 若同一个 rank 按同样速度逐个处理这 4 个请求，每个请求需要 $4/2=2$ 秒，共 $2+2+2+2=8$ 秒，单位时间完成量是 $4/8=0.5$ 个/秒。
4. 相比单 rank，两 rank 的比值是 $1/0.5=2$。
依据：自行构造

:::

### 适用条件与边界

DP 更适合大批量、高吞吐场景。DP 复制整份模型，内存代价随副本数增长。请求量要足以利用多份副本。
依据：blog/tensorrt-llm-parallel-strategy.md L32-L36；blog/sglang-server-arguments.md L37；blog/vllm-data-parallel-deployment.md L39-L43

负载均衡对请求级 DP 有益。
依据：blog/vllm-data-parallel-deployment.md L20；blog/sglang-server-arguments.md L1228-L1231

与张量并行（[Tensor Parallelism](wiki/tensor-parallelism)）不同，TP 把模型权重切开，TP 组内的多卡处理同一输入；请求级 DP 保留完整副本，每个 DP rank 处理不同请求。DP 也可与 TP 组合：rank 是逻辑副本单位；纯 DP 下一个 rank 通常对应一个工作进程，组合 TP 时一个 rank 可由多卡共同持有完整模型。
依据：blog/tensorrt-llm-parallel-strategy.md L24-L36；blog/vllm-data-parallel-deployment.md L14-L20；blog/sglang-server-arguments.md L37
