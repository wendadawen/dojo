# ExpertPlex evidence：核心论断与证据

固定版本：arXiv:2607.18002v2 TeX 源码（research/src/）。以下 § 编号对应 PDF 章节：§1 Introduction、§2 Background、§3 Overview、§4 APK、§5 Communication、§6 Optimizer、§7 Evaluation、§8 Related。

## C 论断

- C1：前沿 MoE 模型中专家权重占模型参数的 95% 以上：DeepSeek-V4-Pro 95%、GLM-5.1-FP8 96%、MiniMax-M2.7 98%。来源：§2.1。条件：所述三个模型。状态：已确认。
- C2：instance-level PDD 每个阶段持完整模型副本，P:D 配比的最小部署单元随模型增大；一个已报道的 DeepSeek-V3 部署单元为 32 prefill + 320 decode GPU，另一个为 176 GPU，Kimi-K2 部署在 128 H200 上。来源：§1、§2.4。条件：所述部署报道。状态：已确认。
- C3：Green Context 在 kernel 期间固定资源分配，重配置需 CPU 介入，现有系统只能在 prefill 层边界重切分。来源：§1、§2.5。状态：已确认。
- C4：prefill kernel 可运行数十至数百毫秒，decode kernel 数百微秒，相差若干数量级，造成 head-of-line blocking。来源：§2.5。状态：已确认。
- C5：EP4 下 MiniMax-M2.7 的 decode grouped GEMM（8 个激活专家）耗时 17.7–34.7 μs，匹配的 16K token prefill GEMM 为 1.8–2.9 ms，长 84–101×。来源：§4.1。条件：EP4、MiniMax-M2.7、所述 token 数。状态：已确认。
- C6：ExpertPlex 跨阶段共享 MoE 专家、按阶段分离 attention；每节点 GPU 分为 prefill attention 服务器、decode attention 服务器、MoE 服务器（前两者可为空）。来源：§3、Figure 3。状态：已确认。
- C7：APK 在 tile 边界调度；tile 边界每 2.2–25.3 μs 出现一次、与操作总长度无关；抢占上界 = 一个 tile 执行时间 + 一次本地 cluster 检查 epoch；无需 checkpoint/restore/recompute；兼容 CUDA Graph。来源：§4.2、§4.3、§7.6（Figure 15）。状态：已确认。
- C8：dispatch 由 attention 侧 push（NVLink peer store / 一侧 RDMA write），combine 由 attention 侧 pull（WaitDone 单线程 kernel + NVLink load / 一侧 RDMA read）；取消 MoE 侧 ring buffer、轮询与信用回传；消除跨阶段死锁。来源：§5.1、§5.2。状态：已确认。
- C9：prefill scale-out 流量尽量经同远端节点 prefill attention 服务器中转（RDMA 一次 + NVLink 组播去重），decode 直连 MoE 服务器；无 prefill 服务器时 prefill 用更低优先级 IB 虚拟通道。来源：§5.3。状态：已确认。
- C10：APK 是唯一在微基准中同时进入 decode 低延迟区与保持 prefill 高性能的机制（对照 CUDA stream 优先级、MPS、Green Context）。来源：§7.3、Figure 11。条件：单 GPU、GLM-5.1-FP8 形状 GEMM、decode 128 token / prefill 8192 token / 8 激活专家、decode 晚 10 μs 启动。状态：已确认。
- C11：独立评价用——共享专家 + tile 调度 + 一侧通信三者是互相使能的设计（APK 预分配 buffer 使一侧通信可行；一侧通信移除 MoE 侧 kernel 使 APK 独占调度）。来源：§5.2 明确陈述 APK 预分配是一侧通信的前提。状态：已确认（论文明确声称该依赖关系）。
- C12：MiniMax-M2.7 + ShareGPT：ExpertPlex 11.3 req/s/node（P90 goodput），为 ChunkedPrefill 5.65×、Colocated 2.72×、PDD 2.01×、PDMux 1.41×。来源：§7.2、Figure 7。条件：N3 的 SLO 与硬件、1P1D PDD 布局。状态：已确认。
- C13：MiniMax-M2.7 + LooGLE：对 Colocated 4.12×、对 PDMux 1.28×；ChunkedPrefill 在整个负载范围内无法满足 SLO。来源：§7.2、Figure 8。状态：已确认。
- C14：GLM-5.1-FP8：对 ChunkedPrefill 3.3×（ShareGPT）/5.0×（LooGLE）；对 Colocated 1.5×/2.5×；ShareGPT 上与 PDMux 持平（约 1.5 req/s/node，PDMux 的 TP attention 对短请求 TTFT 有利）；LooGLE 上对 PDMux 1.66×。来源：§7.2、Figure 9/10。条件：多节点、部分基线 16 GPU 布局、按每节点归一。状态：已确认。
- C15：APK 微基准：CUDA stream 优先级使 decode 延迟相对独占执行 +13.79×；MPS / Green Context 使 prefill 慢 3.33× / 4.07×；APK 使 decode 仅 +8%、prefill 仅慢 1.12×。来源：§7.3、Figure 12。条件同 C10。状态：已确认。
- C16：tile 调度开销：prefill contiguous 布局 <12%；decode masked 布局 <20 μs（激活专家多时相对开销 <10%）。来源：§7.4、Figure 13。条件：对照 DeepGEMM、GLM-5.1-FP8 形状。状态：已确认。
- C17：attention 发起通信开销：normal 模式 dispatch/combine 与 DeepEP v1 差约 5%；低延迟模式差约 45 μs 以内。来源：§7.5、Figure 14。条件：16 GPU、GLM-5.1-FP8。状态：已确认。
- C18：MiniMax-M2.7 全部 MoE 操作抢占间隔 <25.3 μs，GEMM <10.7 μs；REEF 最佳报道延迟 35 μs 且需重算被抢占 kernel（REEF 等不面向 MoE 负载、不支持 TMA multicast/CTA cluster/warp specialization/CUDA Graph，仅作参考点）。来源：§7.6、Figure 15。状态：已确认。
- C19：H100 MIG 只有 1g/2g/3g/4g/7g profile，两路切分中唯一能用满整卡算力的是 3g-4g；MIG 无时间复用与抢占，驱动级重配置无法界定重分配延迟。来源：§4.1。状态：已确认。
- C20：DeepSeek-V3 报道 H800 节点内 NVLink 160 GB/s、跨节点 IB 50 GB/s，3.2× 带宽差。来源：§5.1 引 deepseekv3。状态：已确认（文献已有结论，来源 DeepSeek-V3 技术报告）。

## F 公式

- F1：goodput 定义 G(ℓ,q) = min(B_p/T_p, B_d/(T_d·Ō))。来源：§6.1 Eq.(1)。含义：ℓ 布局、q decode SM 预算、B_p/B_d 满足 SLO 的最大 batch、T_p/T_d 迭代延迟、Ō 平均输出长度；min 体现流水线平衡（每请求 1 次 prefill、Ō 次 decode）。状态：已确认。
- F2：组件延迟拟合 t̂_c(x,s) = α_c + β_c·x + γ_c·xs + δ_c·xs²。来源：§6.2 Eq.(2)。x 为本地 batch、 s 为序列长度（MoE 用 x=x_moe、s=1）。状态：已确认。
- F3：MoE tile 足迹 x_moe = Σ_{e|m_e>0} ⌈m_e/M_t⌉。来源：§6.2 Eq.(3)。m_e 专家 e 收到的 token 行数、M_t tile 高度；依据：每个激活专家至少触发一个 tile 的元数据、权重搬运与计算（Figure 6 实证）。状态：已确认。
- F4：在线 SM 重分配 q' = min(Q_max, ⌈q·x_moe/x_moe*⌉_c)。来源：§6.4 Eq.(4)。⌈·⌉_c 向上取整到 CTA cluster 倍数，Q_max 保证 prefill 进度，decode 优先、prefill 得剩余。状态：已确认。

## N 数字（实验设置）

- N1：MiniMax-M2.7：230 GB FP8 部署足迹；每层 256 routed experts、每 token 激活 8 个；每 token 跨层激活约 7.0B routed expert 参数；full attention。来源：§7.1。
- N2：GLM-5.1-FP8：756 GB FP8 足迹、724.8B routed expert 参数、每 token 激活约 22.6B；256 routed experts、top-8；DSA attention。来源：§7.1。
- N3：SLO 表——MiniMax-M2.7+ShareGPT：TTFT 1s / TPOT 50ms；MiniMax-M2.7+LooGLE：10s/100ms；GLM-5.1-FP8+ShareGPT：2s/100ms；GLM-5.1-FP8+LooGLE：20s/100ms。指标为 P90 goodput（≥90% 请求同时满足 TTFT 与 TPOT SLO 的最高到达率）。来源：§7.1。
- N4：硬件——单节点 8×H800（NVLink）；多节点最多 3 台、每台 8×H800、每节点 8×200 Gbps IB NIC；吞吐按 req/s/node 归一；采样长度上限按 PDD KV-cache 容量截断；Poisson 到达。基线：SGLang-Colocated / ChunkedPrefill / PDD（MiniMax 1P1D；GLM 24 GPU 下 PDD OOM 无数据）/ PDMux（基于 MuxWise 改为 MoE 支持，TP attention；GLM 上部分基线用 16 GPU 布局）。来源：§7.1。

## 原图候选

- Figure 1（GPU 执行模型）：SM/CTA/cluster/DSMEM/TMA 结构。教学点：理解 tile 调度的硬件基底。获取：TeX 源码 figures/figure1.pdf。
- Figure 2（PD colocation 局限）：head-of-line blocking 与资源气泡示意。教学点：Q1 的两种失败模式。figures/figure2.pdf。
- Figure 3（ExpertPlex 架构）：三类服务器与数据通路。教学点：Q2 架构总览。figures/figure3.pdf。
- Figure 4（tile 级抢占机制）：抢占决策沿存储层级传播。教学点：Q3 核心机制。figures/figure4.pdf。
- Figure 5（跨阶段重叠）：一阶段通信与另一阶段计算重叠的时间线。教学点：Q4 收益。figures/figure5.pdf。
- Figure 6（MoE GEMM 延迟 vs 激活专家数/token 数）：教学点：x_moe 的实证依据。figures/figure6.pdf。
- Figure 11（GPU 共享机制 Pareto 前沿）：教学点：APK 唯一进入低 decode 延迟区。figures/figure11.pdf。

选择由 outline.md 决定：选用 Figure 2、3、4、5、6、11；Figure 1 与概念页 gpu-execution-model 内容重叠，不选用（页面内引用概念页）。
