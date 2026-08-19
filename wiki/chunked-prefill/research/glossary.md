# Chunked Prefill（chunked-prefill）术语表

| 术语 | 首次出现 | 含义 |
|---|---|---|
| chunked prefill（分块预填充） | 页面开头 | 把一个 prefill 请求切成多个近似等大块逐块计算的机制 |
| chunk（块） | 页面开头 | prefill 切分后的一个单元，含固定数量（约等）的输入 token |
| piggybacking（搭车） | 页面开头 | decode 请求与 prefill chunk 同批执行的批构造策略（Sarathi 的 decode-maximal batching） |
| generation stall（生成停顿） | 页面开头 | 进行中 decode 因长 prefill 迭代阻塞而数秒不产出的现象 |
| prefill / decode 阶段 | 第 1 章 | 见 moe-serving 页；prefill 处理输入产出首 token，decode 逐 token 生成 |
| 连续批处理（continuous batching / iteration-level batching） | 第 1 章 | 迭代级调度：每次迭代可加入新请求、移除完成请求（Orca 提出） |
| compute-bound / memory-bound | 第 1 章 | 计算瓶颈 / 访存瓶颈；见 gpu-execution-model 页 |
| TBT（time-between-tokens） | 第 1 章 | 相邻两个生成 token 的间隔；与 TPOT/TTL 同义，本文跟随 Sarathi 文献用 TBT，指明等价关系 |
| TTFT | 第 1 章 | 首 token 延迟；对应 Beyond the Buzz 论文的 FTL |
| KV cache | 第 2 章 | 各层缓存的 K/V 张量；见 moe-serving 页 |
| 因果掩码（causal mask） | 第 2 章 | 注意力只看当前位置之前的 token；见 causal-mask 页 |
| 算术强度（arithmetic intensity） | 第 2 章 | 每字节访存对应的计算量；决定 bound 类型 |
| tile 量化（tile quantization） | 第 2 章 | 矩阵维度不对齐 GPU tile 时部分线程块做多余计算的现象 |
| token budget（token 预算） | 第 3 章 | 每次迭代允许的最大 token 总量（Sarathi-Serve 的调度约束） |
| stall-free 调度 | 第 3 章 | 新请求以 chunk 加入当前批次、从不暂停进行中 decode 的调度策略 |
| decode-maximal batching | 第 3 章 | 一个批次 = 一个 prefill chunk + 尽量多的 decode 槽位 |
| 微批（micro-batch） | 第 5 章 | 流水线并行的工作单元；见 model-parallelism 页 |
| 流水线气泡（pipeline bubble） | 第 5 章 | 流水线各 stage 的空闲时间；见 model-parallelism 页 |
| CPP（Chunked Pipeline Parallelism） | 第 5 章 | chunked prefill 与 PP 的组合；块等大使 micro-batch 时长均匀从而缩小气泡 |
| MLA / GQA | 第 5 章 | 两种注意力机制；见 mla、mqa-gqa 页 |
| FTL / TTL | 第 5 章 | Beyond the Buzz 论文的首 token 延迟 / token 间延迟 |
| PD 分离（disaggregated serving） | 第 6 章 | prefill 与 decode 放到不同实例的部署形态；见 beyond-buzz-disaggregation 论文页 |
| co-located（合设） | 第 6 章 | prefill 与 decode 同实例的部署形态 |

## 符号

| 符号 | 含义 |
|---|---|
| $N$ | 一个 prompt 被切成的块数 |
| $i$ | 块序号 |
| $\frac{N(N-1)}{2}$ | N 块的额外 KV 读取总次数（第 $i$ 块被后续块读 $N-i$ 次的求和） |
| $\tau$ | token budget（Sarathi-Serve 算法 3 记号） |
| $T_{max}$ | TBT SLO 上界（确定 $\tau$ 的输入） |
