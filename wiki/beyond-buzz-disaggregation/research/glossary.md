# Beyond the Buzz（beyond-buzz-disaggregation）术语表

| 术语 | 首次出现 | 含义 |
|---|---|---|
| 分解 / 推理分解（disaggregation / inference disaggregation） | 页面开头 | 把推理切成计算特征不同的阶段分别优化；LLM 语境下即 prefill/decode 分离 |
| PD 分离（disaggregated serving） | 页面开头 | prefill 与 decode 跑在不同模型实例（可在不同 GPU）的部署形态 |
| co-located（合设） | 页面开头 | prefill 与 decode 同实例；本文的对照基线 |
| 吞吐-交互性 Pareto 前沿 | 页面开头 | 吞吐（摊销成本）与交互性（服务质量）权衡的有效边界；前沿上的点=不被其他点同时超过的配置 |
| 交互性（interactivity） | 页面开头 | tokens/s/user，即 $1/\mathrm{TTL}$ 的代理 |
| piggybacking | 页面开头 | chunked prefill 下的搭车批构造；见 chunked-prefill 页 |
| IFB（in-flight batching） | 第 1 章 | 批内请求完成即补新请求的批处理方式 |
| FTL（First Token Latency） | 第 1 章 | prefill 完成并产出首 token 的延迟（与 TTFT 同义） |
| TTL（Token-to-Token Latency） | 第 1 章 | 生成相邻两个 token 的延迟（与 TPOT/TBT 同义） |
| TPS（tokens per second per user） | 第 1 章 | $1/\mathrm{TTL}$，交互性度量 |
| SLA | 第 1 章 | 约定的 P50 TTL 与 FTL |
| ISL / OSL | 第 1 章 | 输入/输出序列长度 |
| 设计点（design point） | 第 1 章 | 模拟空间中的一个配置组合（切分+批大小+比例等） |
| 高保真模拟器（high-fidelity simulator） | 第 1 章 | 论文使用的专有 GPU 性能模拟器；输入架构/流量/GPU 配置，输出延迟与吞吐 |
| P50 近似 | 第 1 章 | 用 P50 ISL/OSL 的最近 2 的幂做常数流量近似 |
| 模型切分 / 模型映射（model partitioning / mapping） | 第 2 章 | TP/EP/PP/TEP 等并行策略组合；见 model-parallelism 页 |
| rate matching | 第 3 章 | 确定 prefill:decode 实例（GPU）比例使两阶段吞吐平衡 |
| 整数求解器（integer solver） | 第 3 章 | 附录 B 中求近似整数比 α 的工具 |
| 弹性伸缩（elastic scaling） | 第 3 章 | 随需求变化动态调整两池规模的能力 |
| CPP（Chunked Pipeline Parallelism） | 第 3 章 | 见 chunked-prefill 页；论文中为 prefill 池策略 |
| TEP | 第 3 章 | TP 注意力 + EP FFN 的组合并行（论文列举的切分选项之一） |
| NVLink 域（NVLink domain） | 第 3 章 | NVSwitch 直连的 GPU 组；见 gpu-communication 页 |
| 数据并行（DP，注意力语境） | 第 3 章 | 论文 §4 中指注意力在多实例间按数据划分（各实例算不同请求的注意力）；与模型并行的 DP（每卡全模型）区别在上下文注释 |
| egress / ingress 带宽 | 第 4 章 | prefill GPU 的 KV 出口 / decode GPU 的 KV 入口带宽 |
| KV 复制因子 | 第 4 章 | TP 域超过 KV 头数时 KV 被复制的倍数（TP rank 数 / KV 头数） |
| MLA / GQA | 第 2 章 | 见 mla、mqa-gqa 页 |

## 符号（带宽公式与算法）

| 符号 | 含义 |
|---|---|
| $N_{layers}$ | 模型层数 |
| $BS_{prefill}$ / $BS_{decode}$ | prefill / decode 实例的批大小 |
| $ISL$ / $OSL$ | 输入 / 输出序列长度 |
| $d_{head}$ | 注意力头维 |
| $N_{kv\_heads}$ | KV 头数 |
| $bytes_{element}$ | 每 token 每头 KV 的字节数 |
| $FTL$ / $TTL$ | 首 token 延迟 / token 间延迟 |
| $NumGPU_{prefill}$ / $NumGPU_{decode}$ | 唯一切分 KV 的 prefill / decode GPU 数 |
| $BW_{egress}$ / $BW_{ingress}$ | 每 GPU 出口 / 入口带宽需求 |
| $B$ / $G$ | 算法中的批大小 / GPU 数 |
| $\alpha$ | prefill 吞吐与 decode 请求吞吐的整数比（numerator/denominator 决定两池 GPU 数） |
| $tolerance$ | 整数比的容差（附录 B 默认 0.03） |
