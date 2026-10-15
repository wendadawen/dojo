# Data Parallelism

- 描述：数据并行把同一份模型复制到多个推理工作进程，每个进程独立处理不同请求，输入按负载均衡分给各进程。
- 摘要：请求级 DP 用多份完整模型并行处理不同请求来提高吞吐；每个副本保留自己请求的 KV cache，路由器按当前负载分配请求。它是复制品之间的分工，不是把一层切到多卡。
- 主题：推理系统, 并行与通信
- 标签：推理系统

## 长页

- 开头｜回答推理 DP 是什么；说明模型复制、请求分流和 rank 的含义｜360｜blog/tensorrt-llm-parallel-strategy.md L4-L13；blog/tensorrt-llm-parallel-strategy.md L32-L36；blog/vllm-data-parallel-deployment.md L3；blog/vllm-data-parallel-deployment.md L14-L20；blog/sglang-server-arguments.md L1216-L1218
- ### 一个 DP rank 怎么处理请求｜说明完整权重、负载均衡和路由器可用的信号｜480｜blog/tensorrt-llm-parallel-strategy.md L34-L36；blog/vllm-data-parallel-deployment.md L14-L20；blog/vllm-data-parallel-deployment.md L20；blog/vllm-data-parallel-deployment.md L25-L48；blog/vllm-data-parallel-deployment.md L148；blog/sglang-server-arguments.md L1228-L1231
- #### 请求分流的例子｜用 2 个 rank、6 个请求和各自请求长度手算轮询与按待处理请求数分流的差别；分流过程折叠｜90｜自行构造
- ### 每个副本为什么有独立 KV cache｜解释 [KV cache](wiki/kv-cache) 的请求私有性和每个 rank 的容量上限；小数字例子折叠｜450｜blog/tensorrt-llm-parallel-strategy.md L81-L86；blog/vllm-data-parallel-deployment.md L20
- ### 它为什么提高吞吐｜说明并行推进不同请求的吞吐来源｜380｜blog/tensorrt-llm-parallel-strategy.md L4-L6；blog/tensorrt-llm-parallel-strategy.md L34-L36；blog/sglang-server-arguments.md L37
- ### 适用条件与边界｜写请求流条件、重复权重的代价、负载均衡有益和 [Tensor Parallelism](wiki/tensor-parallelism) 对比｜350｜blog/tensorrt-llm-parallel-strategy.md L24-L36；blog/vllm-data-parallel-deployment.md L3-L20；blog/vllm-data-parallel-deployment.md L148；blog/sglang-server-arguments.md L37

## 短页

- ### 完整模型的多个服务副本｜是什么：推理里的 DP 把同一模型复制到多个 rank；rank 是一个独立工作进程，持有自己请求的 [KV cache](wiki/kv-cache)｜250｜blog/tensorrt-llm-parallel-strategy.md L12-L13；blog/vllm-data-parallel-deployment.md L3；blog/vllm-data-parallel-deployment.md L14-L20
- ### 有足量请求时扩大吞吐｜为什么需要：单卡性能不足或需要更高吞吐时，DP 提高单位时间完成量，但每份副本占整份模型内存｜240｜blog/tensorrt-llm-parallel-strategy.md L4-L6；blog/tensorrt-llm-parallel-strategy.md L34-L36；blog/sglang-server-arguments.md L37
- ### 把请求分到对应 rank｜怎么做：路由器按当前运行/等待队列或轮询选择 rank，并把新请求交给它独立处理｜280｜blog/vllm-data-parallel-deployment.md L14-L20；blog/vllm-data-parallel-deployment.md L25-L48；blog/vllm-data-parallel-deployment.md L148；blog/sglang-server-arguments.md L1228-L1231
