# Strata 术语表

登记全文首次出现的术语、缩写和符号。写作与审查以此为准，保证全文同一对象同一写法。

| 术语/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| KV cache | 第 1 章开头场景 | 推理时缓存的注意力 key/value 张量，避免重复 prefill 计算；概念页 kv-cache |
| prefill（预填充） | 第 1 章 | 一次性处理输入 token 并生成其 KV cache 的阶段 |
| decode（解码） | 第 1 章 | 自回归逐个生成输出 token 的阶段，持续复用并追加 KV cache |
| TTFT（Time To First Token） | 定位摘要/术语表 | 请求到达至第一个输出 token 的延迟 |
| 上下文缓存 / 前缀缓存（context/prefix caching） | 第 1 章 | 跨请求识别公共前缀并复用其 KV cache；概念页 prefix-caching |
| hit rate（缓存命中率） | 第 1 章 | 命中缓存而免于重算的 token（或请求）比例 |
| 页（page）/ 页大小（page size） | 第 1 章 | KV cache 分页管理的固定单元；页大小以 token 数计；概念页 paged-attention |
| HBM（High-Bandwidth Memory） | 第 1 章 | GPU 高带宽显存 |
| CPU 内存 / host 内存 | 第 1 章 | 服务器主存（DRAM），分层缓存的中间层 |
| pinned memory（锁页内存） | 第 1 章容量账 | 锁定物理地址的主存页，GPU DMA 可直接访问；实验配 1 TB |
| PCIe | 第 1 章 | CPU-GPU 互联总线；实验平台为 PCIe 5.0 x16，单向峰值 64 GB/s |
| DMA（Direct Memory Access） | 第 2 章 | 不经 CPU 干预的内存直传引擎，cudaMemcpyAsync 背后的机制 |
| cudaMemcpyAsync | 第 2 章 | CUDA 异步拷贝 API，Strata 之前分层系统的传输方式 |
| Little's Law / $\lambda$、$C$、$L$、$X$、$S$ | 第 1 章三杠杆 | $C=\lambda L$、$X=\lambda S=C\cdot S/L$：并发数、到达率、延迟、吞吐、单次传输量 |
| loading-bound / compute-bound | 第 1 章 | 批的受限资源是 I/O 带宽 / 计算能力 |
| load/compute 比例 | 第 4 章 | 批内需加载 token 数与需计算 token 数之比，阈值默认 100 |
| I/O stall（I/O 停顿） | 第 1 章 | prefill 执行中等待 KV cache 加载的时间占比 |
| GPU-assisted I/O | 第 2 章 | 用 CUDA kernel（而非 cudaMemcpyAsync）执行数据搬运的机制 |
| SM（Streaming Multiprocessor） | 第 2 章 | GPU 的流式多处理器，block 被调度到 SM 上执行；概念页 gpu-execution-model |
| CUDA block / 线程（thread） | 第 2 章 | CUDA 线程层次：block 由数百至上千线程组成；I/O kernel 用 2 个 1024 线程 block |
| layer-first 布局 | 第 3 章 | GPU 显存中的排布：同一层各 token 连续，与逐层计算对齐 |
| page-first 布局 | 第 3 章 | host/存储中的排布：同一页各层连续，利于整页大块传输 |
| HiRadixTree | 第 4 章 | SGLang RadixTree 的扩展，充当页表并记录各 KV cache 页元数据（含 transient node） |
| RadixTree / 基数树 | 第 4 章 | 前缀压缩树，SGLang 用于前缀匹配；概念页 prefix-caching |
| transient node（暂态节点） | 第 4 章 | HiRadixTree 中不指向内存索引、携带 in-queue / in-flight 标记的节点 |
| in-queue / in-flight | 第 4 章 | 暂态节点两种状态：有请求引用新上下文待执行 / cache 正在计算 |
| delay hit（延迟命中） | 第 1 章末 | 同一数据的多个请求在 cache miss 解析期间到达并排队的现象，此处指同一上下文 |
| bundle hit | 第 4 章 | 与 delay hit 相对：共享上下文的请求编入同一批，共享加载、省显存与片上带宽 |
| bubble filling（空泡填充） | 第 4 章 | 批 loading-bound 时推迟 prefill、插入 decode 批与加载并行的策略 |
| P-D co-location | 第 4 章 | prefill 与 decode 批在同一 GPU 上时间交替执行的部署形态 |
| continuous batching（连续批处理） | 第 4 章 | 请求完成即并入/退出运行批的动态组批方式 |
| Cache Controller | 第 4 章 | Strata 数据面组件：管理分层 KV cache 的布局与传输 |
| Scheduler（调度器） | 第 4 章 | Strata 控制面组件：cache-aware 组批 |
| SGLang / vLLM / TensorRT-LLM | 第 5 章基线 | 三个 LLM serving 引擎；版本 v0.4.5 / v0.8.5 / v0.17.0 |
| LMCache | 第 5 章基线 | vLLM 的分层缓存社区扩展，v0.2.1，chunk 256 |
| SGLang-HiCache（基线） | 第 5 章基线 | 作者自建基线：SGLang + 层间重叠 + cudaMemcpyAsync 分层缓存 |
| LooGLE / NarrativeQA / ReviewMT / ShareGPT | 第 5 章 | 四个评测数据集（长文档问答/小说阅读理解/多智能体评审/短对话） |
| cache distance（缓存距离） | 第 5 章 | 同一上下文的请求在到达流中的间隔模式：最小/shuffle/最大 |
| GH200 / Grace Hopper | 第 5 章 | NVIDIA Grace CPU + Hopper GPU 超级芯片平台，NVLink 替代 PCIe |
| Oracle（带宽无上限模拟） | 第 5 章 | 模拟 CPU-GPU 间无限带宽时 TTFT 的参照系 |
| LPM（最长前缀匹配） | 第 5 章消融 | SGLang 的 cache-aware 调度策略，消融对照之一 |
| $L_{\text{layers}}$、$H_{\text{kv}}$、$d_{\text{head}}$、$b$ | 第 1 章 F3 | 层数、KV 头数、头维度、每元素字节数（KV cache 大小公式） |
| Llama-3.1-8B | 贯穿示例 | 32 层、8 KV 头（GQA）、head dim 128、128K 窗口的模型，KV cache 128 KB/token（bf16） |
