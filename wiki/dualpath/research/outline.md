# DualPath 解析页 · 文章大纲

## 页面开头

定位摘要：DualPath 在 PD 分离推理架构中新增「storage→decode→CNIC-RDMA→prefill」第二条 KV-Cache 加载路径，聚合所有节点的 storage NIC 带宽，并配合 CNIC-centric 流量隔离与跨节点调度，解决 agentic LLM 推理中的存储 I/O 瓶颈。

论文解决的具体问题：agentic 工作负载（多轮、长上下文、短追加、命中率 ≥95%）下，PD 分离架构把 KV-Cache 读取全部压在 prefill 引擎的 storage NIC 上，decode 引擎的 storage NIC 闲置；代际硬件差距让 I/O-compute 比持续下降，存储 I/O 取代矩阵计算成为性能主导。

核心问题：见 scope.md Q1-Q5。共 5 个核心问题，对应 5 个正文章节主体（第三章到第七章），加开篇背景章（第二章）与方法评价章（第八章）。

论文元信息：arXiv:2607.24653v2（v2, 2026-02-26），北京大学 / 清华大学 / DeepSeek-AI，DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference。

首个具体场景：64K context、append 429、命中率 98.7% 的 agent round。模型为 DS 660B（MoE + DSA）。在此场景下，cache-compute ratio 达 13-36 GB/PFLOP（Table 1），存储带宽一旦成为瓶颈，GPU 会在等待 KV-Cache 加载时空转。

与第一章的过渡：从「agent round 的真实数据」过渡到「为什么是 storage I/O 而不是矩阵计算成为瓶颈」，引出第二章的三因素分析。

## 章节设计

每章单一职责，标题说明本章解决的问题或得到的结论。编号规则：h2 「N. 标题」，h3 「N.M 标题」。

### 第二章 · agentic 推理的存储 I/O 瓶颈由三因素叠加

职责：回答 Q1，解释为什么 agentic 推理从计算密集退化为 I/O 密集。

- h3 2.1 工作负载：长上下文、短追加、命中率 98.7%
  - trace 数字：157 轮、32.7k context、append 429、命中率 98.7%（N-1）
  - cache-compute ratio 定义与跨模型对比（Table 1，N-2）：DS V3.2 13-36、Qwen2.5-32B 117-267、Qwen3-235B-A22B 39-60、GPT-OSS-120B 47-95、DS V3 4.8-5.8
  - 引用：standard-attention、moe-serving（KV-Cache 概念）

- h3 2.2 硬件代际：FLOPS 与 NIC/HBM 的剪刀差
  - 14.4× I/O-compute 比下降（N-3）
  - 28.8× / 2.0× / 2.4× 三个数字（N-4、N-5）
  - 引用：gpu-communication（PCIe/NIC 速率）、mla（KV 压缩背景）

- h3 2.3 PD 分离架构：PE 饱和、DE 闲置
  - teaser 图：左 PE 100% 存储 / 40% GPU；右 DualPath 80% GPU / 100% 两侧存储
  - 「为什么以前没人加第二条路径」：Mooncake 用 DRAM 池，但 rollout（DRAM 被训练状态占）和超大 working set 不适用；Strata/KVPR/TailorKV 都在单数据路径内优化
  - 引用：moe-serving（PD 分离概念）

- h3 2.4 本章小结

### 第三章 · DualPath 的双路径数据流与块布局

职责：回答 Q2，详细描述两条数据路径的步骤、PE/DE buffer、Full/Layer Block 块布局、trie 寻址。

- h3 3.1 双路径总览
  - PE read path（path A）与 DE read path（path B）对比
  - 贯穿示例的推进：64K agent round 的 62976 hit tokens 走哪条路径、512 miss tokens 怎么算

- h3 3.2 PE read path（path A）
  - 步骤 (1)-(9)，按论文 Fig.4
  - 关键时间：第 k 层的 KV 加载与第 k 层 attention 重叠；第 k+1 层的加载与第 k 层 FFN 重叠

- h3 3.3 DE read path（path B）
  - 步骤 (1)-(7)，按论文 Fig.4
  - 与 path A 的关键区别：DE 的 SNIC 先吃，CNIC 转发，PE 算 miss，miss KV 沿 CNIC 回流到 DE

- h3 3.4 PE/DE buffer、Full/Layer Block、trie 寻址
  - 块形状：Full Block [layer, tokens, bytes]、Layer Block [1, tokens, bytes]
  - 存储交互用 Full Block，层间传输用 Layer Block；n 个 Layer Block 拼一个 Full Block 避免运行时切块
  - trie 节点对应 Full Block，按前缀定位 hit 范围
  - PE/DE buffer：各引擎 host DRAM 上划出的小块缓存，承担 storage↔engine 中转
  - 引用：gpu-communication（RDMA 路径）

- h3 3.5 本章小结

### 第四章 · 双路径不引入新瓶颈的 P/D 区间

职责：回答 Q3，推导 P/D 比例在什么范围内双路径不会在 CNIC、PCIe、DRAM 引入新瓶颈。

- h3 4.1 记号与假设
  - 节点数 P（prefill）、D（decode）；单节点 GPU 数 g；单 GPU CNIC 带宽 B；单节点存储带宽 sB；单节点内存带宽 M
  - 每对 PE-DE 流量：T_p = Bs/(Dg²)、T_c = Bs/(Pg²)（F-1、F-2）
  - 假设：PCIe 拓扑良好、调度均衡、计算网无拥塞、存储读带宽饱和

- h3 4.2 四个不等式
  - PE CNIC 读：2T_p Dg = 2Bs/g ≤ B → s ≤ g（恒成立，方向无瓶颈）
  - PE CNIC 写：(T_p+T_c) Dg = Bs/g(1+D/P) ≤ B → P/D ≥ s/(g-s)（F-4、F-5）
  - DE CNIC 读：(T_p+2T_c) Pg = s/g(P/D+2)B ≤ B → P/D ≤ (g-2s)/s（F-6、F-7）
  - DE CNIC 写：(2T_p+T_c) Pg ≤ B → P/D ≤ (g-s)/(2s)（F-8、F-9）
  - DE DRAM：(3+2P/D) Bs ≤ M → P/D ≤ (M/(Bs)-3)/2（F-10、F-11）

- h3 4.3 汇总区间
  - s/(g-s) ≤ P/D ≤ min{(g-2s)/s, (g-s)/(2s), (M/(Bs)-3)/2}（F-12）
  - 典型 (g=8, s=1, M≈500, Bs≈50)：1/7 ≤ P/D ≤ 7/2
  - 数值示例：把 g=8, s=1, B=50, M=500 代入，逐项验证

- h3 4.4 本章小结

### 第五章 · CNIC-centric 流量管理

职责：回答 Q4，说明 DualPath 如何在 compute NIC 上把 KV-Cache 流量与模型集合通信隔离开。

- h3 5.1 集合通信的特性让传统方案不够用
  - EP AllToAll、TP ReduceScatter/AllGather 亚毫秒级突发（C-31）
  - GPU 不支持 PCIe QoS
  - 软件 traffic shaper 无法在亚毫秒突发间插空 KV-Cache
  - GPUDirect Storage / CUDA copy engine 与集合通信共用 PCIe 但没有共享 QoS 通道

- h3 5.2 CNIC-centric：所有数据走 GPU 配对 CNIC
  - 关键改造：本地 H2D/D2H 也走 CNIC 的 GPUDirect RDMA
  - 让 CNIC 成为统一 PCIe QoS 调度器
  - 引用：gpu-communication（RDMA、GPUDirect）

- h3 5.3 InfiniBand Virtual Lanes（VL）实现分级
  - 高优先级 VL 承载推理通信（qos_vlarb_high 0:192, 1:192, 2:0, 3:192）
  - 低优先级 VL 承载 KV-Cache（qos_vlarb_low 0:192, 1:192, 2:64, 3:192）
  - qos_high_limit=240 → 高优先级占总流量 ~94%（A-3 修正）；vlarb_high 内部 0/1/3 VL 按 192 等权轮转、VL2 留低优先级独占 64
  - 附 qos_max_vls 4

- h3 5.4 RoCE 上的等价方案
  - DSCP → TC → PFC 队列映射
  - 四个 lossless TC，Proportional scheduling 权重分配
  - 与 IB 四 VL 等价语义

- h3 5.5 附带收益：CNIC 辅助的 H2D/D2H 比 CUDA copy engine 更快
  - RDMA Write 提交 ~1 μs（doorbell batching 可摊销）（N-23）
  - cudaMemcpyAsync 5-7 μs（闭源无法进一步细分）
  - 处理小数据块场景的实质优势

- h3 5.6 本章小结

### 第六章 · Adaptive Request Scheduler：inter-engine 与 intra-engine 调度

职责：回答 Q5 上半，描述双层调度算法的具体步骤与参数。

- h3 6.1 引擎分组与 leader
  - 引擎按组（PE 组 / DE 组），同一节点所有 engine 在同一组
  - 每个组一个 leader engine 与调度器通信
  - fetch 时每个 engine 上报 (seq_e, tok_e, read_q_{n(e)})

- h3 6.2 inter-engine PE 调度
  - 三类：过载（tok_e>β）、短读队列候选（read_q≤α 且 tok_e≤β）、长读队列候选
  - FIFO：优先短读队列候选中 tok_e 最小者；空则降级到长读队列候选
  - α：3 秒内可读 token 数；β：5 秒内单 GPU 可处理 token 数（N-22）
  - 算法伪代码（按 Algorithm 1）

- h3 6.3 inter-engine DE 调度（两级）
  - 跨组：按总 token 量（Σ tok_e）选最小者，平衡 NIC 与 GPU
  - 组内：HBM 预算 → 集合 R → 阈值 Z = 1.05 × 平均 → 低 token DE 优先 / 同档 min seq_e / 退而 min tok_e
  - 引用：moe-serving（s5-prefill-decode-kvcache）

- h3 6.4 KV-Cache 读路径选择
  - 取读队列较短的一侧走
  - 请求拆为两半分别读被列为未来工作（C-17）

- h3 6.5 intra-engine PE 调度
  - 只对 PE 端做，DE 端所有请求都进 forward batch
  - FIFO packing、(cached, bsz) 对
  - attention 层理论计算量由 profiling 拟合（参考 PrefillOnly、SarathiServe）
  - compute quota = 300ms；超限时对 bsz 二分做 chunked prefill
  - 解决：attention 层执行时间对齐以最小化 EP/DP 同步时 GPU bubble
  - 引用：moe-serving（chunked prefill、attention 层时间估计）

- h3 6.6 本章小结

### 第七章 · 实验：offline、online、消融、负载均衡、大规模

职责：回答 Q5 下半，给出实验数据并对应到机制。

- h3 7.1 实验配置
  - 集群：8×Hopper GPU/节点、8×400Gbps RDMA CNIC + 1×400Gbps SNIC（N-6）；IB 物理隔离；3FS 无内部 DRAM 缓存
  - 模型：DS 660B (MoE+DSA)、DS 27B（内部缩小版）、Qwen 32B (dense+GQA)
  - 数据集：3 套生产 agentic RL trace，各 500 轨迹，MaxLen 32K/48K/64K（Table 2，N-7）
  - 基线：SGL(MC) (SGLang+HiCache+Mooncake Store+3FS+Mooncake Transfer Engine)、Basic（未修改内部框架）、Oracle（bypass 所有 I/O）
  - P/D 默认：DS 660B 2P4D、Qwen 32B 1P2D、DS 27B 1P1D（N-8）

- h3 7.2 离线 batch inference
  - DS 660B：最高 1.87× over Basic；与 Oracle 性能接近（说明 I/O 已被吃满）
  - DS 27B：最高 1.78×；仍比 Oracle 慢 1.09-1.85×，原因 1P1D 存储带宽受限
  - Qwen 32B：趋势类似 DS 27B
  - SGL(MC) 在大配置报错 N/A；DS 27B 未跑 SGL(MC)
  - 贯穿示例的推进：把 64K/1024 agents 的 1.87× 与 cache-compute ratio 13-36 GB/PFLOP 对应起来，说明「聚合 SNIC 后 cache-compute 瓶颈消失」

- h3 7.3 append / generation 长度敏感性
  - append 越长 Basic 越接近 DualPath/Oracle（瓶颈转向 GPU 计算）
  - DualPath 1.82-1.99× speedup 跨 append 缩放

- h3 7.4 P/D 比例敏感性
  - DS 27B 平均 1.64×，最高 2.46×
  - Basic 1P1D ≈ Basic 1P2D；DualPath 1P1D ≈ Basic 2P1D；DualPath 2P1D ≈ DualPath 1P2D
  - 三对等效直接证明「存储带宽是主导瓶颈」

- h3 7.5 在线 serving
  - SLO：TTFT ≤ 4s、TPOT ≤ 50ms（N-14）
  - APS 容量：DS 27B 1.67×、DS 660B 2.25×（abstract 写 1.96× avg）
  - TTFT 分解：DualPath 各项稳定，Basic 排队时间随 APS 快速恶化（Fig.12 左）
  - TPOT/TTST 不劣化
  - 工作集：DS 660B 0.1 APS 69 GB → 0.45 APS 681 GB（公式 F-14）

- h3 7.6 消融
  - 仅 +Layerwise：平均 -17.21% JCT
  - +Dual-path：平均 -38.19% JCT（主要收益来源）
  - +Scheduling：平均 -45.62% JCT
  - 解读：dual-path 是核心（聚合 SNIC），scheduling 是放大器（避免重新单点过载），layerwise 是前提交件（释放 PE HBM 容量）

- h3 7.7 负载均衡
  - storage NIC Max/Avg：1.53（无 scheduling）→ 1.18（with scheduling）
  - attention 层 Max/Avg：前 5% 低至 1.06
  - 直接对应算法：α/β 阈值、min tok_e、Z=1.05

- h3 7.8 大规模
  - 1152 GPU 离线：2P4D/2K agents 3167s vs 48P96D/48K agents 3201s（近线性）
  - 1152 GPU 在线：44P88D 8.8 APS vs 2P4D 0.4 APS（22× 吞吐）；TTFT/TTST/TPOT 近似
  - 调度器 CPU < 10 cores
  - 未显示额外加速收益的原因：P/D 比例与并行配置未精调

- h3 7.9 本章小结

### 第八章 · 方法评价：DualPath 解决了什么、没解决什么

职责：方法评价章，标记为分析性判断。

- h3 8.1 解决了什么
  - 存储 I/O 不再是 PD 分离架构的瓶颈
  - 通过 compute NIC 复用与 SNIC 聚合把单点瓶颈转成全局池化资源
  - 给出可证明的 P/D 区间让读者判断自己集群的覆盖度
  - 给出可复现的 IB VL / RoCE TC 配置

- h3 8.2 没解决什么
  - 非 agentic 工作负载：命中率低，KV-Cache 加载不构成主导
  - 仍依赖存储后端性能：3FS 无 DRAM 缓存假设在其他存储不成立
  - 大规模 P/D 比例与并行配置的自动调优（论文标为 future work）
  - 单请求拆为两半分别读（future work）
  - 大规模未精调时相对多小集群无明显收益

- h3 8.3 与相邻工作的关系
  - vs Mooncake（DRAM 池）：DualPath 直指 storage 后端，DRAM 池不适合的 rollout/超大 working set 场景正是 DualPath 的目标
  - vs Strata（层次化存储 + GPU 辅助 I/O）：DualPath 走 compute NIC 二次传输，在多节点 PD 分离下做全集群聚合
  - vs KVPR / TailorKV（单路径内压缩 / 重叠）：DualPath 不做 KV 压缩，而是把多路径变两路径
  - vs LayerKV / PrefillOnly（layerwise prefill）：DualPath 复用其机制作为前置
  - vs DistServe / Splitwise（PD 分离开创）：DualPath 在其架构之上引入第二条加载路径

- h3 8.4 适用边界
  - 仅在 agentic 工作负载下收益显著
  - 仅在 3FS + IB 物理隔离 + g=8 s=1 典型配置下保证 P/D 区间成立
  - 仅在 DS 660B / 27B / Qwen 32B 三类模型验证；其他 attention 变体未测
  - 误用：把 1.87× 当作所有模型数字（实际 DS 660B 最高，DS 27B 最高 1.78× 且受 1P1D 存储带宽限制）

- h3 8.5 本章小结

## 贯穿示例

**64K context、append 429、命中率 98.7% 的 agent round。模型：DS 660B，部署 2P4D。**

- 第一次出现（第二章开头，背景之后）：建立读者心中的「典型 agent round」——62976 hit tokens + 512 miss tokens，cache-compute ratio 13-36 GB/PFLOP
- 第二次出现（第三章，描述完 PE read path 后）：说明这个 round 走 PE read path 时，PE 引擎需要从 SNIC 读 ~62.4 GB hit KV、计算 512 miss token 的新 KV、然后通过 RDMA 传给 DE
- 第三次出现（第三章 DE read path 之后）：说明如果走 DE read path，DE 引擎的 SNIC 读 ~62.4 GB，DE 的 CNIC 把 hit KV 转发给 PE，PE 算 miss 之后再回传
- 第四次出现（第四章区间计算）：把 g=8, s=1, B=50, M=500, P=2, D=4 代入，验证 2/4 = 0.5 在 1/7 ≈ 0.14 与 7/2 = 3.5 之间
- 第五次出现（第六章调度）：这个 round 在等待队列里，调度器会先把它分到 read_q 短、tok_e 小的 PE，DE 端走 Z 阈值下的低 token DE
- 第六次出现（第七章离线实验）：在 64K MAL / 1024 agents 的 1.87× 加速中，每 1024 个这样的 round 共节省的时间与 SNIC 聚合后的 8×400Gbps 带宽对应

无法覆盖全文的话用局部例子补充：第五章流量管理用 PCIe 仲裁示意；第六章 intra-engine 用「3 GPU DP 组 + 64K/1K batch + 1 request 超 quota」的最小示例。

## 表达材料职责

| 材料 | 关联解释目标 |
|---|---|
| Table 1（cache-compute ratio） | Q1：不同模型下 I/O 压力的量级 |
| Fig.1 teaser | Q1：单点瓶颈的视觉化与 DualPath 的解 |
| Fig.2 workload | Q1：agentic trajectory 的形态 |
| Fig.3 motivation | Q1：硬件代际剪刀差与 batch 收益饱和 |
| Fig.4 pe_flow | Q2：PE read path 数据流 |
| Fig.4 ce_read | Q2：DE read path 数据流 |
| Fig.5 intersched | Q5：PE 调度三类候选与 α/β 阈值 |
| Fig.6 intrasched | Q5：intra-engine compute quota + chunked prefill |
| F-1/F-2（T_p、T_c） | Q3：每对流量的基本单元 |
| F-3-F-11（四组不等式） | Q3：CNIC/PCIe/DRAM 逐项约束 |
| F-12（汇总区间） | Q3：典型配置的 P/D 可行域 |
| F-13（Z 阈值） | Q5：DE 组内负载均衡的数学表达 |
| F-14（working set） | Q5：在线部署 sizing |
| Fig.7 rollout | Q5：离线多模型 JCT 网格 |
| Fig.8 rollout_diff_pd | Q5：P/D 比例敏感性 |
| Fig.9 rollout_diff_append | Q5：append/gen 长度敏感性 |
| Fig.10 serving | Q5：在线 TTFT/TTST/TPOT |
| Fig.12 serving_breakdown | Q5：消融与 TTFT 分解 |
| Fig.13 read_lb | Q5：SNIC 负载均衡 |
| Fig.14 attn_lb | Q5：attention 层负载均衡 |
| Table 3 largescale | Q5：1152 GPU 可扩展性 |
| Fig.serving_jct | Q5：在线工作集与 JCT |
| 代码：Algorithm 1 | Q5：PE 调度伪代码 |
| 数值示例：64K/512 miss round | 贯穿全文 |

## 正文与折叠块分工

### 必须放正文

- Q1 答案：三个因素的量化与逻辑链
- Q2 答案：PE read path (1)-(9) 与 DE read path (1)-(7) 的步骤表
- Q3 答案：四组不等式的最终形式与汇总区间
- Q4 答案：CNIC-centric 流量管理的核心机制
- Q5 答案：调度算法的关键步骤与实验的关键数字
- 概念页链接：PD 分离、KV-Cache、attention、RDMA、MoE、chunked prefill
- 公式目的与符号说明
- 关键实验数字与配置
- 各章的「本章小结」

### 可放折叠块

- 完整不等式推导（4.2 的中间步骤，可压缩为一步）
- Z 阈值的更一般推导（包含 HBM 碎片化对 R 上界的影响）
- RoCE TC/PFC 队列的硬件级映射细节
- DS 27B 完整模型规格
- 27B model specs 附录表
- agent task structure 完整符号系统
- 在线 SLO 实验稳态判定的滑动窗口细节
- 5K 行代码修改的模块分布

折叠块全部收起时，正文仍要能完整回答 Q1-Q5。

## 范围与证据约束

大纲只使用 scope.md 已纳入范围的内容。发现缺口时回退到 scope.md 修正，不在 outline.md 中临时纳入。
