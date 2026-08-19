# DualPath 解析页 · 术语表

全文所有首次出现的术语、缩写和符号的登记。写作时保证同一对象使用同一种记号。

## 术语与缩写

| 术语 | 全称 / 含义 | 首次出现位置（计划） |
|---|---|---|
| DualPath | 本文系统名，「双路径 KV-Cache 加载」 | 页头 §1 |
| KV-Cache | LLM 推理中为避免 attention 重算而缓存的 K/V 张量 | §2.1 |
| agentic | 「代理式」，指 LLM 自主规划、调用工具、与环境多轮交互的范式 | §2.1 |
| trajectory | agent 单次完整运行的轮次序列 | §2.1 |
| turn / round | 一次提示与一次生成的最小交互单位 | §2.1 |
| append tokens | 每轮新增的提示 token（来自环境/工具/用户） | §2.1 |
| generation tokens | 每轮 LLM 生成的 token | §2.1 |
| context | 跨轮累积的已确认提示 token | §2.1 |
| KV-Cache 命中率 | hit tokens / (hit + miss) | §2.1 |
| cache-compute ratio | KV-Cache 加载量 / 计算量，论文用 GB/PFLOP | §2.1 |
| PD 分离 | prefill-decode disaggregation，把 prefill 和 decode 放在不同引擎 | §2.1 |
| prefill engine (PE) | 负责 prompt 阶段计算的引擎 | §2.1 |
| decoding engine (DE) | 负责自回归生成阶段的引擎 | §2.1 |
| layerwise prefill | 逐层执行 attention 预填充，避免一次性占满 HBM | §2.1 |
| SNIC | storage NIC，节点上对接存储后端的网卡（south-north） | §2.3 |
| CNIC | compute NIC，节点上对等 GPU 计算网络的网卡（east-west） | §2.3 |
| RDMA | Remote Direct Memory Access，远端 DMA 直读对端内存 | §2.3 |
| InfiniBand | 一种高性能计算网络标准 | §5.3 |
| RoCE | RDMA over Converged Ethernet | §5.4 |
| virtual lane (VL) | InfiniBand 链路层 QoS 通道 | §5.3 |
| traffic class (TC) | RoCE 等价 VL 的概念 | §5.4 |
| DSCP | Differentiated Services Code Point，IP 层 QoS 标记 | §5.4 |
| PFC | Priority Flow Control，RoCE 的无损流控 | §5.4 |
| GPUDirect RDMA | NVIDIA 提供的网卡绕过主机内存直接读写 GPU 显存的技术 | §5.2 |
| GPUDirect Storage | NVIDIA 提供的绕过主机内存从存储直读 GPU 显存的技术 | §5.1 |
| CUDA copy engine | GPU 上独立于 SM 的 DMA 引擎，做 host↔device 拷贝 | §5.1 |
| PCIe QoS | PCIe 链路层 QoS 能力 | §5.1 |
| AllToAll | 集合通信原语，每个 rank 向所有 rank 各发一份数据 | §5.1 |
| ReduceScatter / AllGather | 张量并行常用集合通信原语 | §5.1 |
| WRR | Weighted Round Robin 加权轮转仲裁 | §5.3 |
| qos_high_limit | IB 限高优先级仲裁器占用总流量的比例（255 单位） | §5.3 |
| qos_vlarb_high / low | 高/低优先级仲裁器内各 VL 的权重 | §5.3 |
| doorbell batching | 把多次 RDMA 提交请求合并到一次 MMIO 写入 | §5.5 |
| Full Block | [layer, tokens, bytes] 的完整 KV-Cache 块，存储交互用 | §3.4 |
| Layer Block | [1, tokens, bytes] 的单层 KV-Cache 块，层间传输用 | §3.4 |
| block_size | 块内 token 数 | §3.4 |
| trie | 前缀树结构，按 token 前缀寻址 hit KV | §3.4 |
| PE buffer / DE buffer | 各引擎 host DRAM 上划出的 KV-Cache 中转缓存 | §3.4 |
| H2D / D2H | host-to-device / device-to-host 拷贝 | §3.2 |
| leader engine | 每个引擎组中与调度器通信的代表 | §6.1 |
| seq_e | 引擎 e 当前未完成请求数 | §6.1 |
| tok_e | 引擎 e 未完成请求的总 token 数 | §6.1 |
| read_q_{n(e)} | 引擎 e 所在节点 n(e) 的存储读队列长度 | §6.1 |
| α | 短读队列阈值（token 数，对应 3 秒可读量） | §6.2 |
| β | 未完成 token 上限（对应 5 秒单 GPU 可处理量） | §6.2 |
| Z | DE 组内高 token 阈值（1.05 × 组内平均 tok_e + len） | §6.3 |
| compute quota | intra-engine 调度的 attention 层时间上限（论文用 300ms） | §6.5 |
| chunked prefill | 把单请求的 prompt 切成多段分别 prefill 的机制 | §6.5 |
| bsz | 一次前向 batch 中需在 GPU 上计算的 token 数 | §6.5 |
| cached | 一次前向 batch 中可从 KV-Cache 命中的 token 数 | §6.5 |
| JCT | job completion time，全部 agent 完成 rollout 的时间 | §7.1 |
| TTFT | time to first token，提示到首 token 延迟 | §7.1 |
| TTST | time to second token，提示到第二个 token 延迟 | §7.1 |
| TPOT | time per output token，每生成一个 token 的平均时间 | §7.1 |
| APS | agent arrival rate，每秒新到达的 agent 数 | §7.5 |
| SLO | service-level objective | §7.5 |
| HiCache | SGLang 的 prefix cache 机制 | §7.1 |
| Mooncake | Moonshot 提出的分布式 DRAM KV-Cache 池架构 | §7.1 |
| Strata | 层次化存储 + GPU 辅助 I/O 调度的工作 | §1 相关工作 |
| 3FS | DeepSeek 开源的分布式文件系统 | §3 实际部署 |
| io_uring | Linux 内核异步 I/O 接口 | §7.1 |
| FlashMLA / DeepGEMM / DeepEP | DeepSeek 开源的高性能 CUDA kernel | §7.1 |
| SGL(MC) | SGLang + HiCache + Mooncake Store + 3FS + Mooncake Transfer Engine | §7.1 |
| Basic | 论文未优化的内部推理框架 | §7.1 |
| Oracle | 论文假设的零 I/O 上界基线 | §7.1 |
| EP | expert parallel，专家并行 | §5.1 |
| DP | data parallel，数据并行 | §6.5 |
| TP | tensor parallel，张量并行 | §5.1 |
| DSA | DeepSeek Sparse Attention | §2.1 |
| GQA | Grouped-Query Attention | §2.1 |
| MLA | Multi-head Latent Attention | §2.1 |

## 符号

| 符号 | 含义 |
|---|---|
| $P$ | prefill 节点数 |
| $D$ | decode 节点数 |
| $g$ | 单节点 GPU 数 |
| $B$ | 单 GPU 的 CNIC 带宽 |
| $s$ | 单节点存储带宽占 CNIC 带宽的倍数（$sB$ 是单节点存储带宽） |
| $M$ | 单节点内存带宽 |
| $T_p$ | PE read path 下每对 (PE, DE) 的存储侧流量 |
| $T_c$ | DE read path 下每对 (PE, DE) 的存储侧流量 |
| $n_\text{layer}$ | 模型层数 |
| $\alpha$ | 短读队列阈值（token 数） |
| $\beta$ | 未完成 token 上限 |
| $Z$ | DE 组内高 token 阈值 |
| $E$ | DE 组内引擎集合 |
| $R$ | DE 组内当前可调度请求集合 |
| $\text{len}(r)$ | 请求 $r$ 的 token 长度 |
| $\text{tok}_e$ | 引擎 $e$ 未完成 token 数 |
| $\text{seq}_e$ | 引擎 $e$ 未完成请求数 |
| $\text{read}\_q_{n(e)}$ | 节点 $n(e)$ 的存储读队列长度 |
| $\lambda$ | agent 到达率（APS） |
| $\bar{T}$ | 平均 JCT |
| $\text{total}\_\text{len}_{\text{avg}}$ | 平均每条 trajectory 的总 token 数 |

## 写作约束

- 同一概念全文同一种写法：CNIC、SNIC、prefill engine、decoding engine 一致；KV-Cache 不要写作 KV cache（除引用论文原文或保持论文原样的固定词）
- 「命中率」用百分数（如 98.7%），不要写 0.987
- 带宽单位按论文：SNIC/CNIC 用 Gbps，DRAM 用 GB/s，KV-Cache 数据量用 GB
- 数字按 N 编号引用，不直接说「如图所示」而无数字
- 跨章沿用的 64K/append 429/512 miss 贯穿示例：每次出现时复用输入并增加新步骤
