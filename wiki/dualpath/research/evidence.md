# DualPath 解析页 · 核心论断与证据

固定版本：arXiv:2602.21548v2（2026-02-26）。以下所有定位均基于此版本的 TeX 源码 `session/*.tex` 与 PDF（v2 423 KB，13 页）。

置信状态标记：✅ 已确认（按 §X / Eq.(X) / Table X / Figure X / Appendix 定位，可核）；⚠️ 存在歧义或简记（需在正文标注）；❌ 证据不足（不纳入正文）。

## C 论断（叙述性）

| 编号 | 论断 | 来源定位 | 适用条件 | 状态 |
|---|---|---|---|---|
| C-1 | agentic LLM 推理从计算密集退化为 I/O 密集，KV-Cache 加载取代矩阵计算成为主导 | §1 Intro、§3 Motivation | agentic、多轮、长上下文、短追加 | ✅ |
| C-2 | agentic 工作负载的 KV-Cache 命中率 ≥95% | §1 Intro 引 Chen et al. 2026；§2 进一步指出「>95% tokens in our traces」 | 生产 coding agent trace | ✅ |
| C-3 | PD 分离架构中 PE 引擎的 storage NIC 100% 饱和，DE 引擎的 storage NIC 完全闲置 | Fig.1 (teaser) | 一般 agentic 场景 | ✅ |
| C-4 | Ampere→Blackwell 期间 GPU FLOPS 增长 28.8×，PCIe 仅 2×，HBM 2.4×；I/O-compute 比下降 14.4× | Fig.3 (motivation) 左 | NVIDIA GPU 代际对比 | ✅ |
| C-5 | cache-compute ratio 在 DS V3.2 27B/660B 约 22 GB/PFLOP | §3 first para、§3 末段 | 64K context、append 长度 429 | ✅ |
| C-6 | 同样的 append 长度下，Qwen2.5-32B（FP16）的 cache-compute ratio 达 117-267 GB/PFLOP | Table 1 | 16K-64K context、append 长度 429、FP16 | ✅ |
| C-7 | PD 分离架构下所有 KV-Cache 读取都经 prefill 引擎，DE 的 storage NIC 全程闲置 | §3 第三因素段、Fig.1 | 一般 PD 分离部署 | ✅ |
| C-8 | DualPath 新增 storage→decode→CNIC-RDMA→prefill 路径 | §4.1 | DS 部署 | ✅ |
| C-9 | 两条路径都按 layer-wise 与计算流水重叠 | §4.1 描述「(3-7) repeats n_layer times」 | DualPath 全配置 | ✅ |
| C-10 | Full Block 形状 [layer, tokens, bytes]、Layer Block 形状 [1, tokens, bytes]；存储用 Full Block，层间传输用 Layer Block；trie 节点对应 Full Block | §4.1「Different Block Layouts」、Appendix §11.4 | DualPath 全配置 | ✅ |
| C-11 | CNIC-centric 流量管理把所有进出 GPU 的流量（含本地 H2D/D2H）都改走 GPU 配对 CNIC 的 GPUDirect RDMA | §5 引言段、§5.2 引言段 | IB / RoCE 部署 | ✅ |
| C-12 | InfiniBand 用 virtual lanes 做流量分级，模型推理通信占高优先级 VL，~99% 带宽；KV-Cache 占低优先级 VL | §5.1 末段、Appendix §11.1 | IB 集群 | ✅ |
| C-13 | qos_max_vls=4、qos_high_limit=240、qlarb_high=192/192/0/192、vlarb_low=192/192/64/192 | Appendix §11.1 | IB 集群实测配置 | ✅ |
| C-14 | RDMA Write 提交约 1 μs（可经 doorbell batching 摊销）；cudaMemcpyAsync 5-7 μs | §5.2 末段 | 实测（CUDA 闭源无法进一步细分） | ✅ |
| C-15 | PE 调度把 PE 分为三类：过载（tok_e > β）、短读队列候选（read_q ≤ α 且 tok_e ≤ β）、长读队列候选；FIFO 优先短读队列候选中 tok_e 最小者 | §6 Inter-Engine Scheduling、Algorithm 1 | DS 27B/660B | ✅ |
| C-16 | DE 调度两级：跨组按总 token 量选最小者；组内设阈值 Z=1.05×平均、低 token DE 优先、同档 min seq_e、退而 min tok_e | §6 DE Scheduling Phase 1/2 | DS 27B/660B | ✅ |
| C-17 | 读路径选择走读队列较短的一侧；请求拆为两半分别读被列为未来工作 | §6 KV-Cache Read Task Scheduling | DS 27B/660B | ✅ |
| C-18 | intra-engine 调度只对 PE 做：FIFO packing、attention 层理论计算量由 profiling 拟合、compute quota=300ms、超限时对 bsz 二分做 chunked prefill | §6 Intra-Engine Scheduling、Appendix §11.3 | DualPath 全配置 | ✅ |
| C-19 | α 设置为 3 秒内可读 token 数；β 设置为 5 秒内单 GPU 可处理 token 数 | Appendix §11.3 | 实测 | ✅ |
| C-20 | 离线 DS 660B 最高 1.87× over Basic；DS 27B 最高 1.78×（但仍比 Oracle 慢 1.09-1.85×，受 1P1D 存储带宽限制） | §8.1 | agentic RL rollout 离线场景 | ✅ |
| C-21 | 离线消融：仅加 layerwise prefill 平均 -17.21% JCT；加 dual-path 后 -38.19%；加 scheduling 后 -45.62%（均相对 Basic，DS 660B 64K MAL，1024/2048 agents） | §8.4 ablation | DS 660B 离线 | ✅ |
| C-22 | 在线 APS 容量相对 Basic：DS 27B 1.67×、DS 660B 2.25×；abstract 写「average factor of 1.96×」 | §8.3、abstract | 在线 serving | ✅ |
| C-23 | 存储 NIC 负载均衡度（Max/Avg）从无 scheduling 的 1.53 提升到 1.18 | §8.4 Load Balance、Fig.13 | DS 27B 1P1D、48K MAL、1024 trajectories | ✅ |
| C-24 | attention 层 Max/Avg 在任务前 5% 低至 1.06 | §8.4、Fig.14 | DS 27B 1P1D、48K MAL | ✅ |
| C-25 | 大规模 1152 GPU 离线 48P96D/48K agents JCT 3201s vs 2P4D/2K agents 3167s 近线性；在线 44P88D 8.8 APS vs 2P4D 0.4 APS = 22× 吞吐；调度器 CPU < 10 cores | §8.5、Table 3 | 大规模 | ✅ |
| C-26 | 大规模 P/D 比例与并行配置未精调，相对多小集群无额外加速收益；但减少碎片、抗突发 | §8.5 末段 | 1152 GPU | ✅ |
| C-27 | DP 模型可工作集 ≈ λ·T̄·total_len_avg/2；DS 660B 在 0.1 APS 69 GB、0.45 APS 681 GB | §9 Working Set Analysis | DS 660B 在线 | ✅ |
| C-28 | 真实场景若 JCT 因交互间隔增加 r 倍，APS 容量增 r 倍，工作集增 r² 倍，资源成本 r³ 倍 | §9 | 一般推导 | ✅ |
| C-29 | SGL(MC) 在大配置报错 N/A；DS 27B 上未跑 SGL(MC) 因为 SGLang 不支持该 downscaled 模型 | §8.1、§8.2 Baselines | 工程实现差异 | ✅ |
| C-30 | DualPath 实现基于 in-house 推理框架（FlashMLA+DeepGEMM+DeepEP）；约 5K 行修改；用 3FS + io_uring-like 接口 | §8.1 Implementation | DS 内部 | ✅ |
| C-31 | 集合通信（EP AllToAll、TP ReduceScatter/AllGather）亚毫秒级突发，GPU 不支持 PCIe QoS，软件 traffic shaper 不可行 | §5 引言段 | 一般 GPU 系统 | ✅ |
| C-32 | Mooncake DRAM 池在 rollout（DRAM 被训练状态占）和超大 working set 场景不适用 | §1 Intro 末段、Related Work | 一般部署 | ✅ |
| C-33 | P/D 无瓶颈区间 s/(g-s) ≤ P/D ≤ min{(g-2s)/s, (g-s)/(2s), (M/Bs-3)/2}；典型 (g=8, s=1, M≈500, Bs≈50) 下为 1/7 ≤ P/D ≤ 7/2 | §4.2 推导汇总段、Summary | 一般 DGX 节点 | ✅ |

## F 公式

| 编号 | 公式 | 含义 | 来源 | 状态 |
|---|---|---|---|---|
| F-1 | $T_p = \dfrac{Bs}{D g^2}$ | 每对 PE-DE 在 PE read path 上的存储侧流量 | §4.2 Traffic per PE-DE pair | ✅ |
| F-2 | $T_c = \dfrac{Bs}{P g^2}$ | 每对 PE-DE 在 DE read path 上的存储侧流量 | §4.2 同上 | ✅ |
| F-3 | $2 T_p D g = \dfrac{2 B s}{g} \le B$ | PE CNIC PCIe 侧读方向总流量 | §4.2 PE CNIC Read | ✅（要求 s ≤ g） |
| F-4 | $(T_p + T_c) D g = \dfrac{Bs}{g}\left(1+\dfrac{D}{P}\right) \le B$ | PE CNIC PCIe 侧写方向总流量 | §4.2 PE CNIC Write | ✅ |
| F-5 | $P/D \ge \dfrac{s}{g-s}$ | 由 F-4 推出 | §4.2 | ✅ |
| F-6 | $(T_p + 2 T_c) P g = \dfrac{s}{g}\left(\dfrac{P}{D}+2\right) B \le B$ | DE CNIC PCIe 侧读方向总流量 | §4.2 DE CNIC Read | ✅ |
| F-7 | $P/D \le \dfrac{g-2s}{s}$ | 由 F-6 推出 | §4.2 | ✅ |
| F-8 | $(2 T_p + T_c) P g \le B$ | DE CNIC PCIe 侧写方向总流量 | §4.2 DE CNIC Write | ✅ |
| F-9 | $P/D \le \dfrac{g-s}{2s}$ | 由 F-8 推出 | §4.2 | ✅ |
| F-10 | $(3 + 2 P/D) B s \le M$ | DE 节点 DRAM 总压力（半双工） | §4.2 DRAM Pressure | ✅ |
| F-11 | $P/D \le \dfrac{M/(Bs) - 3}{2}$ | 由 F-10 推出 | §4.2 | ✅ |
| F-12 | $\dfrac{s}{g-s} \le P/D \le \min\left\{\dfrac{g-2s}{s},\dfrac{g-s}{2s},\dfrac{M/(Bs)-3}{2}\right\}$ | 汇总区间 | §4.2 Summary | ✅ |
| F-13 | $Z = 1.05 \cdot \dfrac{\sum_{r \in R} \text{len}(r) + \sum_{e \in E} \text{tok}_e}{\lvert E \rvert}$ | DE 组内高 token 阈值 | §6 DE Phase 2 | ✅ |
| F-14 | $\text{working\_set} \approx \lambda \bar{T} \cdot \text{total\_len}_{\text{avg}}/2$ | 在线服务 KV-Cache 工作集估算 | §9 | ✅ |

## N 数字

| 编号 | 数字 | 含义 | 来源 | 状态 |
|---|---|---|---|---|
| N-1 | 平均 157 轮、上下文 32.7k、append 429、KV-Cache 命中率 98.7% | agentic trace 统计 | §3 first para | ✅ |
| N-2 | cache-compute ratio（GB/PFLOP，64K context、append 429、FP8 unless specified） | 模型与负载 | Table 1 | ✅ |
|   |  Qwen2.5-32B (FP16) | 117-267 |   |   |
|   |  GPT-OSS-120B | 47-95 |   |   |
|   |  Qwen3-235B-A22B | 39-60 |   |   |
|   |  DeepSeek-V3.2 660B | 13-36 |   |   |
|   |  DeepSeek-V3 660B | 4.8-5.8 |   |   |
|   |  （注：DS V3.2 27B 在原文表格中作为已删行；正文用 22 GB/PFLOP 描述，跨 27B/660B 的口径与 Table 1 不完全一致，写作时需明确正文 vs 表格的差异） |   |   | ⚠️ |
| N-3 | 14.4× I/O-compute ratio 下降 | 硬件代际 | §3 second para、Fig.3 | ✅ |
| N-4 | GPU FLOPS 28.8× / PCIe 2.0× / HBM 2.4×（2020→2024） | 硬件代际 | Fig.3 左 | ✅ |
| N-5 | 28.8× / 2.0× / 2.4× 数字出现在 Fig.3 左上的标注框 | Fig.3 | ✅ |
| N-6 | 8×Hopper GPU/节点、8×400 Gbps RDMA CNIC、1×400 Gbps SNIC、IB 物理隔离 | 实验集群 | §8.2 Testbed | ✅ |
| N-7 | 数据集统计：32K (Turns 60, Append 608, Gen 148, Total 28639, Context 17183)、48K (Turns 106, Append 474, Gen 172, Total 42607, Context 25120)、64K (Turns 157, Append 429, Gen 176, Total 55958, Context 32721) | 500 轨迹/数据集 | Table 2 | ✅ |
| N-8 | P/D 比默认：DS 660B 2P4D、Qwen 32B 1P2D、DS 27B 1P1D | 评估配置 | §8.2 P/D Ratio | ✅ |
| N-9 | DS 660B 离线最高 1.87× over Basic；DS 27B 离线最高 1.78× | 离线加速比 | §8.1 | ✅ |
| N-10 | append 长度缩放 1.82-1.99× | 离线加速比 | §8.1 | ✅ |
| N-11 | DS 27B 不同 P/D 比平均 1.64×（最高 2.46×） | 离线加速比 | §8.1 | ✅ |
| N-12 | DS 27B Basic 1P1D ≈ Basic 1P2D；DualPath 1P1D ≈ Basic 2P1D；DualPath 2P1D ≈ DualPath 1P2D | 存储带宽等效实验 | §8.1 | ✅ |
| N-13 | 在线 APS 容量：DS 27B 1.67×、DS 660B 2.25× over Basic | 在线加速比 | §8.3 | ✅ |
| N-14 | 在线 SLO：TTFT ≤ 4 s、TPOT ≤ 50 ms | 服务目标 | §8.3 | ✅ |
| N-15 | 消融 -17.21% / -38.19% / -45.62% JCT（DS 660B, 64K MAL, 1024/2048 agents） | 组件贡献 | §8.4 | ✅ |
| N-16 | storage NIC Max/Avg：1.53（无 scheduling）→ 1.18（with scheduling） | 负载均衡 | §8.4、Fig.13 | ✅ |
| N-17 | attention Max/Avg：前 5% 低至 1.06 | 负载均衡 | §8.4、Fig.14 | ✅ |
| N-18 | 1152 GPU 离线 2P4D/2K agents 3167s → 48P96D/48K agents 3201s | 大规模可扩展 | Table 3 | ✅ |
| N-19 | 1152 GPU 在线 2P4D/0.4 APS → 44P88D/8.8 APS（22× 吞吐）；44P88D TTFT 1.847s、TTST 0.194s、TPOT 0.036s | 大规模可扩展 | Table 3 | ✅ |
| N-20 | 调度器 CPU < 10 cores | 调度开销 | §8.5 | ✅ |
| N-21 | DS 660B 工作集 0.1 APS 69 GB、0.45 APS 681 GB | 在线部署 sizing | §9 | ✅ |
| N-22 | α = 3 秒内可读 token 数；β = 5 秒内单 GPU 可处理 token 数；compute quota = 300ms | 调度超参数 | Appendix §11.3 | ✅ |
| N-23 | RDMA Write 提交 ~1 μs（doorbell batching 可摊销）；cudaMemcpyAsync 5-7 μs | 流量管理性能 | §5.2 | ✅ |
| N-24 | qos_max_vls 4、qos_high_limit 240、vlarb_high 0:192,1:192,2:0,3:192、vlarb_low 0:192,1:192,2:64,3:192 | IB QoS 配置 | Appendix §11.1 | ✅ |
| N-25 | DS 节点 80 GB DRAM/节点、Qwen 32B 320 GB/节点；SGL(MC) 1.5 TB/节点 | 内存配置 | Appendix §11.3 | ✅ |
| N-26 | 5K 行代码修改 | 实现量级 | §8.1 Implementation | ✅ |

## 原图候选

按 evidence 优先级（TeX 源 > 官方仓库 > arXiv HTML > PDF 截图）。本论文 TeX 源已带 PDF 矢量图，TeX 源等同于官方图。

| 编号 | 原文 Figure | 内容摘要 | 可说明的结论 | 状态 |
|---|---|---|---|---|
| G-1 | Fig.1 (teaser) | 左：现有架构 PE 100% 存储利用率 + 40% GPU 利用率，DE 100% 存储空闲；右：DualPath 80% GPU 利用率 + 两侧 100% 存储利用率 | Q1 动机：单点瓶颈被聚合 | ✅ |
| G-2 | Fig.2 (workload) | agent trajectory 示例：30k token 提示、append + generation、99% 命中率 | Q1：长上下文、短追加、高命中 | ✅（未在页面复用：trajectory 用文字描述，证据保留便于扩展） |
| G-3 | Fig.3 (motivation) | 左：FLOPS/NIC/HBM 三年趋势（28.8×/2.0×/2.4×）；右：相对吞吐 vs batch size | Q1：硬件代际差距 + 批处理收益快速饱和 | ✅ |
| G-4 | Fig.4 (pe_flow) | PE read path 数据流，标签 (1)-(9) 步骤，含 Full Block / Layer Block 区分 | Q2 机制 | ✅ |
| G-5 | Fig.4 (ce_read) | DE read path 数据流，标签 (1)-(7) | Q2 机制 | ✅ |
| G-6 | Fig.5 (intersched) | Inter-engine PE 调度示意图：三类候选（overload / 短读队列 / 长读队列）、α 与 β 阈值 | Q5 调度 | ✅（未在页面复用：调度逻辑由 Algorithm 1 伪代码 + 文字承载；图保留便于扩展） |
| G-7 | Fig.6 (intrasched) | Intra-engine 调度：compute quota + chunked prefill；GPU 时间线对比（original vs ours） | Q5 调度 | ✅ |
| G-8 | Fig.7 (rollout) | 离线 JCT 网格：模型 × MaxLen × agents；Ours / Ours(basic) / Ours(oracle) / SGL(MC) | Q5 实验 | ✅ |
| G-9 | Fig.8 (rollout_diff_pd) | 离线 DS 27B 在 1P1D / 2P1D / 1P2D 三种 P/D 比下的 JCT | Q5 P/D 比例实验 | ✅ |
| G-10 | Fig.9 (rollout_diff_append) | 离线 append/gen 长度缩放 1×/1.5×/2×/3× 下 JCT 变化 | Q5 append/gen 长度实验 | ✅ |
| G-11 | Fig.10 (serving) | 在线 TTFT / TTST / TPOT vs APS，DS 27B 与 DS 660B | Q5 在线实验 | ✅ |
| G-12 | Fig.12 (serving_breakdown) | TTFT 分解（Sch/A/R/PF）+ 消融四档 Basic / +Layer / +DPL / +Sched | Q5 消融 | ✅ |
| G-13 | Fig.13 (read_lb) | 存储 NIC Max/Avg ratio 时间序列：1.528 vs 1.184 | Q5 负载均衡 | ✅ |
| G-14 | Fig.14 (attn_lb) | attention 层 Max/Avg ratio | Q5 负载均衡 | ✅ |
| G-15 | Table 1 | cache-compute ratio 跨模型 | Q1 数字 | ✅ |
| G-16 | Table 2 | agent dataset 统计 | 实验配置 | ✅ |
| G-17 | Table 3 | 大规模实验结果 | Q5 大规模 | ✅ |
| G-18 | Fig.serving_jct | 平均 JCT vs APS | Q5 工作集推理 | ✅ |
| G-19 | Fig.largescale_rollout | 48P96D 离线指标 | Q5 大规模 | ✅ |

## 存在歧义与处理

- **A-1 abstract「1.96× average」与 §8.3「1.67× / 2.25×」**：abstract 写「improve online serving throughput by an average factor of 1.96×」对应 §8.3 给出的 DS 27B 1.67× 与 DS 660B 2.25× 两个数（几何平均 √(1.67·2.25) ≈ 1.94，算术平均 1.96）。正文取算术平均，标 abstract 表述并附 §8.3 两个原始数据
- **A-2 DS V3.2 22 GB/PFLOP（正文）vs Table 1 的 DS V3.2 660B 13-36 GB/PFLOP（表格）**：正文给的 22 是 27B/660B 综合值，Table 1 仅给 660B 13-36（16K-64K 范围）。正文中以 22 GB/PFLOP 描述 DS V3.2 时应注明「跨 27B/660B、跨 16K-64K 的代表值」，Table 1 单独引用时给出 660B 13-36 GB/PFLOP
- **A-3 vlarb_high 192/192/0/192 与「99% 带宽」**：论文文字「approximately 99% of total bandwidth to high-priority VL」对应 qos_high_limit=240/255 ≈ 94%（高优先级占总流量比例）；vlarb_high 0:192,1:192,2:0,3:192 是高优先级仲裁器内三个高优先级 VL（VL0/1/3）按 192 权重轮转，VL2 分配 0 权重（不用于高优先级）。正文中用「高优先级 VL 合计占 ~94% 带宽（qos_high_limit=240/255），低优先级 VL 保留少量以避免饿死」表述，避免说 99%
