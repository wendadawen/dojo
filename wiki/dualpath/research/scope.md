# DualPath 解析页 · 内容范围

## 1. 论文定位

- 标题：DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- 作者：Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang
- 单位：北京大学计算机学院、清华大学、DeepSeek-AI
- 版本：arXiv:2602.21548v2（v1 2026-02-25, v2 2026-02-26）；ACM sigconf LaTeX 模板，未注明已投会议；本页以 v2 为准
- 链接：https://arxiv.org/abs/2602.21548
- 代码仓库：论文正文与 abstract 未提供 GitHub 链接（第三方报道 Two Minute Papers 提到「open-sourced this technique」，但论文未给具体地址；本页不引用未核实的仓库）
- 简要：DualPath 在 PD 分离推理架构中新增「storage→decode engine→RDMA→prefill engine」第二条 KV-Cache 加载路径，把分散在所有节点的 storage NIC 带宽聚合到 KV-Cache 读取上，配合 CNIC-centric 流量隔离与跨节点调度，在 agentic 工作负载下把离线吞吐最多提到 1.87×、在线吞吐平均提到 1.96×

### 论文宣称的贡献（与 abstract 一致）

- C-C1：识别 agentic 推理的 I/O-bound 本质——KV-Cache 加载在多层 PD 分离架构中取代矩阵计算成为性能主导
- C-C2：提出 DualPath 双路径加载，把传统 storage→prefill 之外的多余 storage NIC 带宽纳入可用容量
- C-C3：设计与评估 workload-aware 调度算法，跨 PE/DE 联合平衡计算与网络利用率

### 论文没做的事（避免被误认）

- 没有改造 attention 计算本身（KV-Cache 压缩、量化、稀疏化等单路径优化属于 KVPR / TailorKV / Strata 等工作）
- 没有改造 PD 分离是否要分离的命题（PD 分离假设保留；DualPath 在 PD 分离内做优化）
- 没有提出新的存储后端（沿用 DeepSeek 开源的 3FS）
- 没有解决 LLM 推理的另两条经典瓶颈：GPU 计算本身、HBM 容量（layerwise prefill 是组件级引入，未做新机制）
- 没有评估 GPUDirect Storage / CUDA copy engine 在 PCIe 干扰下与非 PCIe-QoS 集合通信的精确退化曲线，只在 motivation 与 §5.2 定性说明

### 相邻工作（只记关键区别；不展开）

- Mooncake（FAST'25）：分布式 DRAM 池 + 亲和性调度；DualPath 显式避开了 DRAM 池不适合的两类场景（rollout 阶段 DRAM 被训练状态占用、在线服务下 working set 远超 DRAM 容量），目标直指 storage 后端
- Strata：层次化存储 + GPU 辅助 I/O 调度，从单数据路径角度优化
- KVPR、TailorKV：PCIe 带宽约束下的层粒度混合量化 / 重叠重算
- DistServe（OSDI'24）、Splitwise（ISCA'24）：PD 分离的开创工作，DualPath 在其架构之上引入第二条加载路径
- LayerKV、PrefillOnly（SOSP'25）：layerwise prefill 的提出者，DualPath 复用其机制
- SGLang + Mooncake：被列为 SGL(MC) 基线，仅作为工程参照，DualPath 论文承认与 SGL(MC) 因实现差异不能完全公平比较

## 2. 核心问题

每个核心问题对应正文一章并配解答折叠块。

**Q1 · Agentic 推理的存储 I/O 瓶颈是什么，由哪些因素叠加形成？**

预期答案：瓶颈是 PD 分离下 prefill 引擎的 storage NIC（SNIC）被 KV-Cache 读取独占，decode 引擎的 SNIC 完全闲置；由三个相互独立的因素叠加产生——（a）agentic 工作负载命中率极高（trace 平均 98.7%），把 prefill 阶段从计算密集退化为 I/O 密集，cache-compute ratio 达 22 GB/PFLOP（DS V3.2），对 Qwen2.5-32B 等非 MLA 模型达 117-267 GB/PFLOP；（b）硬件代际差距——从 Ampere 到 Blackwell，GPU FLOPS 增长 28.8×，PCIe 仅 2×、HBM 仅 2.4×，I/O-compute 比下降 14.4×；（c）PD 分离架构下 SNIC 利用严重不均：所有 KV-Cache 读取经 prefill，DE 的 SNIC 全程闲置。重要性：动机章节必须由这三因素直接推出 DualPath 的存在必要；任何一项缺失，瓶颈就不足以构成。依赖：第二节 background、第三节 motivation。

**Q2 · DualPath 的双路径数据流如何配合 layerwise prefill 在 PD 分离架构下加载 KV-Cache，PE/DE buffer 与 Full Block / Layer Block 块布局起什么作用？**

预期答案：DualPath 在 storage→prefill 之外新增 storage→decode→CNIC-RDMA→prefill 路径，每层一拍的层粒度流水与计算重叠。块布局上，存储交互用 Full Block（[layer, tokens, bytes]）、层间传输用 Layer Block（[1, tokens, bytes]），trie 结构按 Full Block 寻址；n 个相邻 Layer Block 可拼成 Full Block，避免运行时再切块。PE buffer 和 DE buffer 是各引擎 host DRAM 上划出的小块缓存，承担 storage↔engine 和 engine↔engine 的中转，刻意让 H2D/D2H 也走 CNIC，理由见 Q4。重要性：这是论文机制主体，对应 §4.1。依赖：需读者已知 layerwise prefill（背景页即可），不必展开内部切块细节。

**Q3 · 在什么样的 P/D 比例区间内双路径不会在 CNIC、PCIe、DRAM 引入新的瓶颈？**

预期答案：论文给出四组不等式，按节点数 P、D、单节点 GPU 数 g、单节点存储带宽 sB、内存带宽 M 推导——PE CNIC 写入要 P/D ≥ s/(g-s)，DE CNIC 读要 P/D ≤ (g-2s)/s，DE CNIC 写要 P/D ≤ (g-s)/(2s)，DE DRAM 写要 P/D ≤ (M/Bs-3)/2；合并为 s/(g-s) ≤ P/D ≤ min{(g-2s)/s, (g-s)/(2s), (M/Bs-3)/2}。在 g=8、s=1、M≈500 GB/s、Bs≈50 GB/s 的典型配置下，可行区间是 1/7 ≤ P/D ≤ 7/2，覆盖大多数现实部署。重要性：给出理论的「无瓶颈」条件，让读者判断自己的集群是否在覆盖范围内。依赖：§4.2 bottleneck-free analysis。

**Q4 · CNIC-centric 流量管理如何把 KV-Cache 流量与模型集合通信隔离开来，为什么不能直接用 GPUDirect Storage 或 CUDA copy engine？**

预期答案：所有进出 GPU 的数据——包括本地 H2D/D2H——都改走 GPU 配对 CNIC 的 GPUDirect RDMA 数据路径，让 CNIC 成为统一的 PCIe QoS 调度器；InfiniBand 用 virtual lanes（VL）做高/低优先级分流（推理通信占高优先级，~99% 带宽；KV-Cache 占低优先级），配置上 qos_max_vls 4、qos_high_limit 240、vlarb_high/low 按 192/192/0/192、192/192/64/192 分配权重；RoCE 用 DSCP→TC→PFC 队列做等价隔离。原因：（a）模型集合通信（EP AllToAll、TP ReduceScatter/AllGather）亚毫秒级突发，软件 traffic shaper 无法在其间插空 KV-Cache；（b）GPU 不支持 PCIe QoS，无法在 PCIe 层屏蔽干扰；（c）GPUDirect Storage / CUDA copy engine 与集合通信共用 PCIe 但没有共享 QoS 通道，会拉爆推理延迟；（d）附带收益是 RDMA Write 提交约 1 μs（可经 doorbell batching 摊销），优于 cudaMemcpyAsync 的 5-7 μs。重要性：这是双路径之所以能实际部署的工程前提，§5。依赖：需读者已知集合通信和 NIC 基础（已有概念页）。

**Q5 · Adaptive Request Scheduler 在 inter-engine 和 intra-engine 两级如何选 PE/DE/读路径与前向 batch，三个组件（layerwise、dual-path、scheduling）的相对贡献如何？**

预期答案：inter-engine 调度做 PE/DE 配对与读路径选择。PE 调度：引擎分组，Leader Engine 拉取，PE 分为过载（tok_e>β）、短读队列候选（read_q≤α 且 tok_e≤β）、长读队列候选；FIFO 优先短读队列候选中 tok_e 最小者。DE 调度两级：跨组按总 token 量选最小者（平衡 NIC/GPU 负载），组内按剩余 HBM 估算可调度请求集合 R 与阈值 Z=1.05×平均，按低 token DE 优先、在同档内按 min seq_e、再退而求 min tok_e 的层级挑选；HBM 不够则停止 fetch。读路径：取读队列较短的一侧走，论文明确把请求拆为两半分别读作为未来工作。intra-engine 调度只在 PE 端做：FIFO packing，估计每请求的 attention 层理论计算量（拟合 profiling 曲线）并加到前向 batch 上，以 300ms compute quota 为上限；超限时对请求的 bsz 做二分搜索做 chunked prefill。消融显示在 DS 660B、64K MAL、1024/2048 agents 上：仅加 layerwise prefill 平均 -17.21% JCT；再加 dual-path 平均 -38.19%；再叠加 scheduling 平均 -45.62%——dual-path 是主要收益来源，scheduling 进一步把 SNIC 负载均衡度从 1.53 提升到 1.18、attention 层 Max/Avg 控制在 1.06。重要性：§6。依赖：需读者已知 chunked prefill 概念（背景页）。

## 3. 内容分级

### 3.1 核心内容（每条对应至少一个核心问题）

- 三因素量化（98.7% 命中率、22 GB/PFLOP、14.4× 硬件下降）→ Q1
- PE 100% 饱和、DE 闲置的 teaser 图与实测动机图 → Q1
- 两条数据流的具体标签（(1)-(9) 步序、Full/Layer Block、PE/DE buffer、trie 寻址） → Q2
- 瓶颈推导的四组不等式与汇总区间、典型配置的具体范围 → Q3
- VL 配置（qos_max_vls 4、qos_high_limit 240、vlarb 权重）、1 μs vs 5-7 μs 的实证 → Q4
- PE 三类候选、阈值 α/β、DE 跨组/组内、Z=1.05、compute quota 300ms、chunked prefill 二分 → Q5
- 离线 1.87×、在线 1.96× / 1.67×（DS 27B）/ 2.25×（DS 660B）、消融三档（-17.21%/-38.19%/-45.62%）、负载均衡 1.53→1.18、attention Max/Avg 1.06、大规模 1152 GPU 22× 吞吐 → Q5

### 3.2 辅助内容（消除关键理解障碍）

- PD 分离、layerwise prefill、KV-Cache 概念 → 引用概念页，本页不复述
- RDMA/IB/RoCE/GPUDirect 简述 → 引用 gpu-communication 概念页
- MoE/EP 集合通信（AllToAll、ReduceScatter、AllGather）→ 引用 moe-serving、deepseek-moe 概念页
- 3FS 简述、io_uring 简述（仅一句话提及即可）
- agentic trajectory 概念图 → 用 workload 图代替文字描述

### 3.3 扩展内容（不展开）

- DS 27B 完整规格（隐藏维度、中间维度、层数、头数、专家数、indexer 头维等）→ 脚注一次带过，正文不展开
- agent task structure 完整定义（Context_i、G_i 等符号系统）→ 不展开，仅在解释 dataset 统计时引用
- RoCE 详细配置（TC 数、PFC 队列数、八队列等）→ 一句话提示「与 IB 的四 VL 等价」，不展开
- 5K 行代码修改的实现细节 → 提及数量级即可
- Mooncake / Strata / KVPR / TailorKV / TokenLake 等相邻工作的具体机制 → 仅在「相关工作」一句话标注，不构成页面内容
- 大规模并行配置未精调的细节 → 引用实验结果，不复述

## 4. 前置知识

每项标注概念页是否存在与依赖的核心内容。

| 前置概念 | 已存在概念页 | 依赖的核心内容 |
|---|---|---|
| LLM 推理的 prefill/decode 阶段、KV-Cache、TTFT、TPOT | wiki/moe-serving（s5-prefill-decode-kvcache、s6-metrics、s7-pdd-colocation 锚点） | Q1、Q2、Q5 的展开 |
| 标准 attention 机制 | wiki/standard-attention | Q1 中解释 attention 层 |
| MLA、DS Sparse Attention（DSA） | wiki/mla、wiki/dsa | Q1 的 cache-compute ratio、Q5 中 P/D 选型背景 |
| GQA | wiki/mqa-gqa | Q1 中 Qwen 32B 模型背景 |
| RDMA、InfiniBand、RoCE、PCIe、NVLink、GPUDirect | wiki/gpu-communication | Q3、Q4 的链路与流量分析 |
| MoE 与 expert parallel（AllToAll 等集合通信） | wiki/moe-serving（s2/s3 锚点）、wiki/deepseek-moe | Q4 集合通信与 PCIe 干扰 |
| chunked prefill、disaggregated inference | wiki/moe-serving | Q5 intra-engine 调度 |

无新增概念页需求：以上概念页均已存在并覆盖 DualPath 所需的前置知识。DualPath 自身不引入新的基础概念（双路径加载、CNIC-centric 流量管理、按读队列选路径、compute quota 调度都是工程机制，可在正文内首次出现时即时解释，不必单独建概念页）。

## 5. 明确不展开的内容

- DS 27B 模型完整配置：原因——属于具体工程实现，与系统机制无直接关系，仅作为评估的模型存在
- agent 任务完整定义（Context_i、G_i、A_i、g_i 符号系统）：原因——dataset 描述足够，正文不需引入
- Mooncake/Strata/KVPR/TailorKV 内部机制：原因——相邻工作的具体机制不影响 DualPath 主线，相关性在 abstract 与 Related Work 各自一句已足够
- RoCE PFC 队列的硬件级配置、IB 仲裁器内部实现：原因——属于 NIC 厂商侧实现细节，正文给出与 IB 四 VL 等价的语义即可
- 5K 行代码修改的逐文件分布：原因——实现量级用一行量化即可，文件级分布属于工程报告
- 大规模实验未精调 P/D 比例的细节：原因——结果已在表里给出，正文不展开
- 论文 abstract 提到的 1.96× 是「on average over 660B/27B 两个模型」还是「on average over serving 与 offline」：原因——abstract 写作存在歧义，正文严格按 §8.3「average factor of 1.96×」且对应 online serving 平均吞吐提升（DS 27B 1.67×、DS 660B 2.25× 的几何/算术平均约 1.94×）表述，避免歧义

## 6. 常见误解和适用边界

### 误解 1：把 abstract 的 1.87× 当作 DualPath 唯一数字

错误理解：「DualPath 让 agentic 推理提升 1.87×」
正确结论：1.87× 是 DS 660B 离线推理相对于未优化 Basic 的最高 JCT 比；DS 27B 最高 1.78×（且仍比 Oracle 慢 1.09-1.85×，因为 1P1D 配置存储带宽受限）；Qwen 32B 趋势类似；不同模型/不同 P/D 比例/在线场景下数字不同
形成原因：abstract 写「up to 1.87×」常被误读为全局数字
影响 Q：Q1、Q5

### 误解 2：把 1.96× 当作离线平均

错误理解：「DualPath 离线平均提升 1.96×」
正确结论：1.96× 是 abstract 的「improve online serving throughput by an average factor of 1.96×」；在线 APS 容量在 DS 27B 1.67×、DS 660B 2.25×；离线是「up to 1.87×」
影响 Q：Q5

### 误解 3：DualPath 让 GPU 利用率从 40% 提升到 80% 即可推得 1.96×

错误理解：teaser 图左 40% / 右 80% 是端到端加速比
正确结论：teaser 图的 40% / 80% 是 GPU 利用率，端到端加速比来自完整的 JCT/APS 实验；GPU 利用率是相关性指标不是因果链
影响 Q：Q1、Q5

### 误解 4：DualPath 等价于「把 DE 也用来加载 KV-Cache」

错误理解：第二条路径就是把现有方案搬到 DE 端
正确结论：第二条路径的设计核心不是「DE 加载」，而是「DE 加载后经 RDMA 通过 compute NIC 传回 PE 仍然要满足瓶颈条件」，并通过 CNIC-centric 流量管理让 compute NIC 的高优先级 VL 不被低优先级流量拉爆。简单让 DE 端也读但不与 PE 算重叠，仍不能解决 P/D 比例失配时的 DE CNIC 瓶颈
影响 Q：Q2、Q3、Q4

### 误解 5：DualPath 适用于所有 LLM 推理

正确结论：论文明确指出 DualPath 解决的是 agentic 长上下文、短追加、高命中率负载下的存储 I/O 瓶颈；非 agentic 单轮/批量推理命中率低，KV-Cache 加载不构成主导，DualPath 的边际收益会下降
影响 Q：Q1、Q5

### 适用边界

- 实验集群：8×Hopper GPU/节点、8×400 Gbps RDMA CNIC、1 个 400 Gbps SNIC、IB、3FS 无 DRAM 缓存（实验依赖此物理隔离）
- 适用模型：DS 660B（MoE+DSA）、DS 27B、Qwen2.5-32B；未测 dense MLA 之外的非对称 KV 压缩方案
- 适用 P/D 比例：理论上限 1/7 ≤ P/D ≤ 7/2（g=8, s=1 典型配置）；实验覆盖 1P1D、2P1D、1P2D、2P4D
- 适用存储后端：3FS + io_uring-like 接口；其他存储后端未测
- 适用工作负载：RL rollout 与在线 agent serving；非 agentic 长上下文（如长文档一次性 prefill）收益未量化
- 适用 SLO：在线服务 TTFT ≤ 4 s、TPOT ≤ 50 ms；超出 SLO 的实验点未报告

## 7. 论断分级

- **论文明确声称**（附原文定位）：
  - abstract 与 §1 intro 的动机三因素、§3 motivation 的 22 GB/PFLOP 与 14.4×、§4.1 dual-path 数据流定义、§4.2 bottleneck-free 不等式推导、§5 traffic manager 的 VL 配置、§6 调度算法定义、§8 全部实验数字
- **文献已有结论**（附来源）：
  - PD 分离（DistServe OSDI'24、Splitwise ISCA'24）→ 概念页 moe-serving 已覆盖
  - layerwise prefill（LayerKV、PrefillOnly SOSP'25）→ 概念页可引用，本页不复述机制
  - 高命中率阈值（Chen et al. 2026, concur high-throughput agentic batch，命中率 ≥95%）→ 作为支撑论断引用
  - 路由器 + DeepSeek MoE（DeepSeekV2/DeepSeekV3/DeepSeekV3.2）→ 概念页 deepseek-moe、mla、dsa 已覆盖
- **基于证据的推断**（标"推断"）：
  - 「DualPath 在非 agentic 场景收益有限」由 §1 明确动机范围（agentic）推出，是边界推断而非核心论断
  - 「compute NIC VL 高/低优先级 ~99% 带宽分配」是论文 §5.1 文字 + qos_vlarb_high 权重 192/192/0/192 共同推出（192/(192+192)=50% per high VL，但高优先级有 3 个 VL，加权后约 75%？需要核；实际读 vlarb high=192,192,0,192 → 高优先级 VL0/1/3 分配 192 权重，VL2 分配 0；同 VL 在 0/1/3 间轮转；论文文字「approximately 99% of total bandwidth to high-priority VL」对应 qos_high_limit=240/255 限高优先级上限，剩余 15/255 留给低优先级；具体机制详见 IB 规范，不在正文展开），是综合推断
  - 「DualPath 在 1P1D 下 DS 27B 仍受存储带宽限制」由 §8.1「1P1D 存储带宽受限」与图 8 的 P/D 比实验共同推出
- **缺失假设的猜测**（标"未核实"）：
  - 论文未提供代码仓库链接，第三方开源仓库与论文版本对应关系未核实，正文不引用任何外部 GitHub 链接
  - abstract「1.96× on average」与 §8.3「average factor of 1.96×」是同一指标，abstract 中「improve online serving throughput by an average factor of 1.96×」与 1.67×/2.25× 几何平均 √(1.67×2.25)≈1.94 接近，论文未明确算术还是几何均值——正文标"论文 abstract 表述，按 §8.3 各模型 APS 倍数取平均"并附具体模型数
