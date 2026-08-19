# Beyond the Buzz（beyond-buzz-disaggregation）核心论断与证据

论文固定版本：arXiv:2506.05508v1（TeX 源码）。定位缩写：§=正文节，Eq.=公式，Fig.=图，App.=附录。

## C 论断

- C1：本文是对大规模分解式推理的首次系统性研究，评估了跨工作负载与硬件配置的数十万设计点。来源：abstract（"we present the first systematic study of disaggregated inference at scale, evaluating hundreds of thousands of design points across diverse workloads and hardware configurations"）。适用条件：截至论文发表。置信：已确认（"首次"为论文自称，页面引用时注明"论文自称"）。
- C2：分解对 prefill-heavy 流量模式与更大模型（如 >10B 参数）最有效。来源：abstract + introduction（"disaggregation provides the greatest benefits in prefill-heavy traffic scenarios (i.e., ISL >> OSL) and when serving larger models (e.g., >10B parameters)"）。适用条件：模拟范围内。置信：已确认。
- C3：动态 rate matching 与弹性伸缩对完全实现分解优势至关重要。来源：abstract（"Our results highlight the critical role of dynamic rate matching and elastic scaling in achieving Pareto-optimal performance"）。置信：已确认。
- C4：对 co-located serving，context chunking 的有效性高度依赖注意力机制（MLA vs GQA），在宽松延迟目标与生成密集流量下最有利。来源：introduction（"the effectiveness of context chunking is highly sensitive to the attention mechanism (e.g., Multi-Latent Attention (MLA) vs. Group Query Attention (GQA)) and is most beneficial under relaxed latency targets and generation-heavy traffic patterns"）。置信：已确认。
- C5：模拟器为专有高保真 GPU 性能模拟器，输入模型架构/流量模式/GPU 配置，输出各批大小与并行策略的延迟和吞吐，用于构造 Pareto 前沿。来源：§3 Model partitioning 段（"we use a proprietary, high-fidelity GPU performance simulator designed for datacenter-scale inference. The simulator takes as input the model architecture, traffic pattern, and GPU configuration, and outputs the corresponding latency and throughput across different batch sizes and parallelism strategies"）。置信：已确认。
- C6：分析聚焦现代 Blackwell 系统与 FP4 精度。来源：§3（"Our analysis focuses on modern Blackwell systems using FP4 precision, which represent the state of the art in LLM inference infrastructure"）。适用条件：全部实验的硬件语境。置信：已确认。
- C7：搜索空间排除 FTL>10 秒的设计点（宽松但实用的约束）。来源：§3.2（"All design points with an FTL >10 seconds, a relaxed yet practical constraint, are excluded from our search space"）。置信：已确认。
- C8：模拟假设：数据中心有足够 GPU 与到达请求使 rate-matched 部署满载；prefill 池每层产生的 KV cache 在可用时立即传给生成池、与后续层计算重叠。来源：§3.2（"Our simulation assumes a datacenter setting with sufficient GPUs and incoming requests to fully utilize the rate-matched deployment. We further assume the KV cache produced at each layer by the prefill pool is transferred to the generation pool immediately as it becomes available, overlapping with the computation of subsequent layers"）。适用条件：全部结论的前提。置信：已确认。
- C9：结果大多归一化呈现，目的是传达趋势而非具体性能声明。来源：Fig.1 caption（"Most results in this paper are presented in normalized form, as our primary objective is to convey trends rather than make specific performance claims"）。置信：已确认。
- C10：co-located serving 的痛点：单实例同时优化低 FTL（新 prompt）与低 TTL（进行中生成）两个指标，各指标瓶颈不同、资源调度存在固有张力。来源：§2（"co-located serving forces a single model instance to simultaneously optimize for two metrics ... leading to inherent tension in resource scheduling"）。置信：已确认。
- C11：分离使各阶段可独立采用适合自身性能目标的模型切分与批处理策略，并消除严格 TTL SLA 对 prefill 的人为拖慢（piggybacking 中所见）。来源：§2（"This separation enables each phase to independently adopt model partitioning and batching strategies tailored to its performance targets. Moreover, it eliminates artificial slowdowns in prefill caused by strict TTL service-level agreements"）。置信：已确认。
- C12：SLA 由 FTL（数百毫秒到数分钟）与 TTL（数毫秒级）定义；1/TTL 是交互性（tokens/s/user）的代理。来源：§4 开头（"service level agreements (SLA) are typically defined by two latency metrics: (i) FTL, ranging from hundreds of milliseconds to several minutes, and (ii) TTL, typically spanning a few milliseconds. The reciprocal of TTL (i.e., 1/TTL) serves as a proxy for interactivity"）。置信：已确认。
- C13：分离部署中 FTL 约束只作用于 prefill 池；Chunked Pipeline Parallelism（CPP）对在给定 FTL 内处理混合长度序列、维持高吞吐尤其有效；DeepSeek-R1 ISL 256K、64 GPU（EP×PP=64）上增大 PP 降低 FTL 且吞吐保持。来源：§4（"FTL constraints apply only to the prefill (context) pool ... we found Chunked Pipeline Parallelism (CPP) to be especially effective"）+ Fig.4 caption（"Chunked pipeline parallelism during Prefill is an optimal strategy to maximize throughput while complying with strict FTL SLA"）。适用条件：该模拟设置。置信：已确认。
- C14：TTL 收紧时配置转向更小批与更大张量并行；DeepSeek-R1（ISL 16k/OSL 2k）Pareto 上 NVLink 域内 EP 一致占优，注意力计算从高吞吐区的数据并行转向紧 TTL 下的张量并行，批大小从数百递减；Llama-3.1-70B 的 TP 从 2× 扩到 64×。来源：§4（"As TTL constraints tighten, configurations shift toward smaller batch sizes and greater tensor parallelism ... expert parallelism within the NVLink domain is consistently preferred ... tensor parallelism scales from 2× to 64×"）。适用条件：DeepSeek-R1/Llama-3.1-70B 模拟。置信：已确认。
- C15：分离的 decode 池可比合设更激进地采用高 TP（无需兼顾重计算的 prefill），在中延迟区取得更优性能。来源：§4（"disaggregated decode pools can better adapt to tightening latency demands — leading to superior performance in the medium-latency regime"）。适用条件：模拟对比。置信：已确认。
- C16：DeepSeek-R1 在 piggybacked co-located 下因 prefill 分块产生额外开销——MLA 的 down/up 投影在每个 prefill chunk 上重复计算；可通过临时缓存早前块的上投影 KV 缓解；因此基线 Pareto 同时含 piggybacked 与非 piggybacked 配置。来源：§4.1（"DeepSeek-R1 experiences additional overhead in piggybacked co-located serving due to prefill chunking—specifically, redundant computation of down and up projections in multi-latent attention for each prefill chunk. This can be mitigated by temporarily caching the up-projected KV values from earlier chunks"）。置信：已确认。
- C17：模型越大分离收益越显著；机制是更大模型映射到更多 GPU、并行策略组合更丰富，两阶段选不同映射的优势放大。来源：§4.1（"the benefits of disaggregated inferencing become more pronounced with larger models ... larger models are typically mapped across more GPUs, enabling a broader range of parallelization strategies"）+ Fig.6。适用条件：Llama 8B/70B/405B 模拟。置信：已确认。
- C18：流量敏感性：分解收益在 prefill-heavy 下最显著（若映射优先解码速度会显著牺牲 prefill 吞吐）；合设 piggybacking 在 decode-heavy 流量上最有前景。来源：§4.2（"the benefits of disaggregation are most pronounced for prefill-heavy workloads where mappings, if prioritized to balance decoding speed, can significantly compromise prefill processing throughput. Similarly, for co-located serving, piggybacking is most promising on decode-heavy traffic"）+ Fig.7。置信：已确认。
- C19：常数 ISL/OSL 模拟是动态流量的近似：取 P50 ISL/OSL 的最近 2 的幂；附录 D 显示该近似下的 Pareto 与直接模拟动态分布的前沿接近。来源：§4.2（"our simulation with constant ISL and OSL represents an approximation of dynamic traffic, where these values correspond to power-of-two approximations of the 50th percentile ISL and OSL"）+ App.D（"the approximated frontier closely matches the original"）。适用条件：所测的真实部署流量分布（数值脱敏）。置信：已确认。
- C20：最优 ctx-to-gen GPU 比例随模型特性与目标延迟显著变化，通用分离系统应内置动态 rate matching 机制。来源：§4.3（"The optimal context-to-generation GPU ratio exhibits significant variation with model characteristics and target latency ... a versatile disaggregated serving system should incorporate a dynamic rate matching mechanism"）+ Fig.8。置信：已确认。
- C21：固定比例退化：比例 3.5 在最宽松延迟目标下高性能、延迟收紧时退化；比例 0.5 利于紧延迟但在宽松延迟下显著受损；小规模 GPU 部署因资源受限约束 rate matching 搜索空间，预期有类似效应。来源：Fig.9 caption + §4.3（"A similar effect is expected in small-scale GPU deployments, where limited resources can restrict the rate matching search space"）。适用条件：DeepSeek-R1 模拟。置信：已确认。
- C22：更大 NVLink 域一致增强分离性能；收益来自生成阶段可选更宽 EP 与 TP 的自由度（DeepSeek-R1 中延迟区受益于更高 EP 与批、Llama-3.1-70B 低延迟受益于高 TP）。来源：§4.4（"larger NVLink domains consistently enhance disaggregated serving performance. The benefit comes from the flexibility to choose wider expert and tensor parallelism during generation"）+ Fig.10 caption。置信：已确认。
- C23：现存开源实现与先行研究的局限：未提供"何时/如何分离有利"的具体指导，先前研究聚焦小规模测试台与峰值吞吐、未考察完整吞吐-交互性 Pareto 前沿。来源：§6（"they fall short of providing concrete guidance on when and how disaggregation is beneficial ... prior research has largely focused on small-scale testbeds and peak throughput scenarios, without examining the full throughput–interactivity Pareto frontier"）。置信：已确认。
- C24：结论：分离的最优配置取决于模型大小与架构、流量模式、延迟约束、硬件资源的组合；小模型与生成密集流量下收益有限。来源：§8 conclusions（"the optimal configurations for disaggregated serving depend on a combination of factors ... We also highlight scenarios where disaggregation offers limited benefit—such as serving small-scale models or generation-heavy traffic"）。置信：已确认。
- C25：未来方向：KV cache 复用、投机解码、推理时计算技术、模型架构演进的影响。来源：§7 future work。置信：已确认。

## F 公式

- F1（egress 带宽）：$BW_{egress} = \frac{N_{layers} \times BS_{prefill} \times ISL \times d_{head} \times N_{kv\_heads} \times bytes_{element}}{FTL \times NumGPU_{prefill}}$。来源：§5 Eq.(1)（TeX：BW_{egress} = ... / (FTL × NumGPU_{prefill})）。各符号论文逐一定义（$N_{layers}$ 层数、$BS_{prefill}$ prefill 批大小、$ISL$ 输入序列长、$d_{head}$ 头维、$N_{kv\_heads}$ KV 头数、$bytes_{element}$ 每 token 每头 KV 字节数、$FTL$ prefill 完成时间、$NumGPU_{prefill}$ 唯一切分 KV 的 prefill GPU 数）。置信：已确认。
- F2（ingress 带宽）：$BW_{ingress} = \frac{N_{layers} \times BS_{decode} \times ISL \times d_{head} \times N_{kv\_heads} \times bytes_{element}}{TTL \times OSL \times NumGPU_{decode}}$。来源：§5 Eq.(2)。置信：已确认。
- F3（趋势：egress 随 ISL 下降）：prefill 注意力二次成本 → FTL 随 ISL 超线性增长，KV 大小线性增长 → egress 需求随 ISL 增大而降低。来源：§5（"Due to the quadratic cost of attention during prefill, FTL scales superlinearly with ISL, whereas the KV cache size scales linearly. This divergence implies that the egress bandwidth requirement decreases as ISL increases"）。置信：已确认。
- F4（趋势：ingress 与 ISL 无关、随 OSL 降）：decode 侧 KV 大小与 TTL 都随 ISL 线性增长、相互抵消；ingress 与 OSL 成反比；TTL 收紧 → 更多 decode GPU → 每卡 ingress 需求降低。来源：§5（"both the KV cache size and TTL scale linearly with ISL, effectively canceling out their impact on ingress bandwidth. However, ingress bandwidth is inversely proportional to OSL. As TTL constraints tighten, more decode GPUs are required ... which effectively lowers the per-GPU ingress bandwidth requirement"）。置信：已确认。
- F5（趋势：模型规模）：FTL 随活跃参数数线性增长，但 KV 大小不随参数量同比例增长；带优化注意力（MLA）的大模型 egress 需求可能低于注意力架构低效的小模型。来源：§5（"FTL scales linearly with the number of active parameters. However, the KV cache size does not grow proportionally to the number of model parameters. Consequently, larger models with optimized attention (i.e., MLA in DeepSeek-R1) may require less egress bandwidth than smaller models"）。置信：已确认。
- F6（KV 复制因子）：部分并行方案复制而非切分 KV（如 TP 域超过 KV 头数时，KV 在 TP rank 间复制，复制因子 = TP rank 数 / KV 头数）；计算每卡带宽时只应计入真正切分 KV 的 GPU。来源：§5（"when the tensor parallelism domain exceeds the number of KV heads, the KV cache is duplicated across tensor parallel ranks. The duplication factor in this case is equal to the ratio of tensor parallel ranks to KV heads"）。置信：已确认。

## N 数字

- N1：>10B 参数的模型被视为"更大模型"受益区间。来源：introduction。适用条件：论文语境。置信：已确认。
- N2：FTL SLA 范围：数百毫秒到数分钟；TTL：数毫秒。来源：§4 开头。置信：已确认。
- N3：FTL>10s 的设计点被排除。来源：§3.2。置信：已确认。
- N4：DeepSeek-R1 ISL 256K、64 GPU、EP×PP=64 的 CPP 实验。来源：Fig.4 caption。置信：已确认。
- N5：DeepSeek-R1 ISL 16k / OSL 2k 的 Pareto 配置轨迹；Llama-3.1-70B TP 2×→64×。来源：§4。置信：已确认。
- N6：固定比例 3.5（宽松最优、紧延迟退化）与 0.5（紧延迟有利、宽松显著受损）。来源：Fig.9 caption。适用条件：DeepSeek-R1。置信：已确认。
- N7：Llama 8B / 70B / 405B 的模型规模对比。来源：Fig.6 + §4.1。置信：已确认。
- N8：带宽结论的实验对象：DeepSeek-R1、两组序列长度组合、多档 TTL。来源：§5（"Figure 11 shows the maximum of egress and ingress bandwidth requirements for two sequence length combinations on DeepSeek-R1 under varying TTLs"）。数值本身归一化/脱敏未给出绝对值。置信：已确认。
- N9：数十万（hundreds of thousands）设计点。来源：abstract。置信：已确认。

## 原图候选（TeX 源码矢量图，获取途径：TeX 源码包 > arXiv HTML）

| 页面编号 | 原文 Figure | 内容 | 说明的结论 | 纳入 |
|---|---|---|---|---|
| G1 | Fig.1（figure1_no_static.pdf） | DeepSeek-R1 吞吐-交互性 Pareto：左 ISL 16384 OSL 2048（prefill-heavy）vs 右 ISL 1024 OSL 32768（generation-heavy）；蓝线为 Disaggregation Pareto、红线为 Piggybacked co-location Pareto | 分离收益随流量模式与目标 tokens/s/user 显著变化（Q2） | 纳入 |
| G2 | Fig.2（figure2.png） | co-located vs disaggregated 时间分布（prefill 深色/decode 浅色块，按请求着色） | 两种模式的直观对比（Q1 背景） | 纳入 |
| G3 | Fig.4（ctx_pp.pdf） | DeepSeek-R1 Prefill ISL 256K，64 GPU（EP×PP=64），x 轴 PP 维度 1→32；红线 FTL（log2）从约 $2^{6.5}$ 降到 $2^{2}$，蓝线 Tokens/s/GPU Normalized to PP=1 几乎水平≈1 | CPP 是 prefill 池严格 FTL 下的最优策略（Q3） | 纳入 |
| G4 | Fig.6（disagg_model_size_pb.pdf） | LLaMa-3.1-8B / 70B / 405B 的分离与合设对比（ISL 4096 OSL 256）；8B 差距小、70B 差距明显、405B 分离优势最大 | 模型越大收益越显著（Q2） | 纳入 |
| G5 | Fig.7（isl_osl_pb.pdf） | DeepSeek-R1 四种 ISL/OSL 组合的 Pareto：(16k/2k) prefill-heavy、(16k/16k) 平衡 prefill 偏、(2k/2k) 平衡 decode 偏、(2k/16k) generation-heavy；三条线：Disaggregated（蓝）、Co-located Overall（红实）、Piggybacked（红虚） | prefill-heavy 收益最大、piggybacking 在 generation-heavy 更有前景（Q2） | 纳入 |
| G5' | Fig.5（model_arch_pb.pdf） | DeepSeek-R1 vs LLaMa-3.1-70B（ISL 16k OSL 2k context-heavy）；DeepSeek-R1 piggybacked 远低于合设 overall（LLaMa-70B 不明显） | 架构敏感性 + MLA 重复投影开销（Q2） | 纳入 |
| G6 | Fig.8（ctx_gen_ratios.pdf） | ISL 16k OSL 2k 下四个模型的最优 Ctx:Gen GPU 比例随 tokens/s/user 变化：LLaMa-8B 几乎水平≈0.4；LLaMa-70B ~0.95→0.45；LLaMa-405B ~2.1→0.3；DeepSeek-R1 ~3.6→0.05（变化最剧烈） | 比例不可静态固定（Q3） | 纳入 |
| G7 | Fig.9（fixed_ratios.pdf） | DeepSeek-R1 ISL 16k OSL 2k 下四条线：固定比例 0.5（绿，几乎水平≈0.4）、固定比例 3.5（橙，宽松最优紧延迟退化）、Optimal Rate Matching（蓝，全区间最高）、Co-located Overall（红）；0.5 在低 tokens/s/user 端被卡 0.4 | 动态 rate matching 的必要性（Q3） | 纳入 |
| G8 | Fig.11（kv_bw.pdf） | DeepSeek-R1 KV Transfer Requirements：x 轴 Normalized Tokens/s/user，y 轴 Bandwidth/GPU (GBps/GPU)；两条线（ISL 16k OSL 2k 蓝、ISL 1M OSL 2k 红）；蓝 0.4–1.2 GBps/GPU，红 1.0–1.8 GBps/GPU | 现有带宽足够（Q4） | 纳入 |
| G9 | Fig.13（dynamic_vs_static.pdf） | 动态流量模拟 vs P50 近似的 Pareto 对比 | P50 近似可靠（Q1 方法可信度） | 纳入 |
| — | Fig.3（chunked_pipelining.png） | CPP 机制图 | 机制已在 chunked-prefill 页承载 | 排除（避免跨页重复；chunked-prefill 页自绘） |
| — | Fig.10（nvl_domain.pdf） | 两种 NVLink 域大小 | NVLink 域越大越有利 | 纳入（Q5） |
| — | Fig.12（dynamic_dist.png） | ISL/OSL 分布 CDF | App.D 支撑材料 | 排除（G9 已足够） |

G 编号在正文中作为图引用记号；Figure 编号与原文对应。N8 带宽的实验条件补全：DeepSeek-R1、(16k/2k) 与 (1M/2k) 两组序列长度组合、多档 TTL；带宽量级约 0.4–1.8 GBps/GPU（图中读出，归一化已脱敏无绝对值）。

## 构造示例（不入 C/F/N）

- 手算 egress：DeepSeek-R1 架构参数（61 层、$d_{head}$=128、MLA 的 KV 头数按论文公式语境取 1 组、bytes/element=2（bf16））代入 F1，ISL=16k、FTL=2s、单实例 8 GPU，得出每卡 GB/s 量级并与机内/跨机带宽量级对照。数字标注构造示例（论文未给绝对值），参数取自 DeepSeek-R1 公开架构（架构参数本身是外部事实，需另行标注来源：DeepSeek-R1 报告）。
- 手算 ingress：同参数 OSL=2k、TTL=20ms 代入 F2。
