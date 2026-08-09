# ExpertPlex scope：内容范围

## 0. 版本固定

- 论文：ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels
- arXiv:2607.18002，固定版本 v2（2026-07-21 修订），TeX 源码包已下载解压至 research/src/
- 作者：Bingyang Wu、Chao Jin、Zili Zhang、Xinming Wei、Yinmin Zhong、Ruidong Zhu（北京大学）；Chengxu Yang、Yuliang Liu（Independent Researcher）
- 无正式会议版（预印本）；无官方代码仓库链接（实现基于 DeepGEMM / DeepEP v1 / SGLang 修改，§7.1）
- 歧义检查：方法名 ExpertPlex 无同名工作；v1→v2 仅一天内小修（279KB→279KB），无实质差异；无影响定位的未消歧项

## 1. 论文定位

- 一句话：ExpertPlex 用「跨阶段共享 MoE 专家 + 按阶段分离 attention」的混合架构，配合 tile 级自适应常驻 kernel（APK）和 attention 发起的一侧通信，解决 MoE 大模型服务中 PD 分离资源粒度太粗、PD 合设资源切分太死的问题，提升 goodput
- 宣称贡献（§1 贡献列表，原文一致）：
  1. 混合架构：消除跨阶段 MoE 权重冗余、复用动态稀疏专家计算、分离 attention 以降低每阶段并行度与通信
  2. APK（tile 级抢占与重分配）+ attention 发起的一侧通信（减少网络干扰、跨阶段通信-计算重叠）
  3. 跨栈联合优化，并在两个前沿 MoE 模型上验证 goodput 提升
- 论文没做什么：
  - 不提出新的 MoE 模型结构或训练方法（纯服务系统论文，依据：全文无模型结构设计）
  - 不做 MoE 负载均衡算法（router 层面），§8 明确说与 MoE load-balancing 正交
  - 不做 attention 侧优化（序列并行、attention offloading），§8 明确正交
  - 不解决单阶段内部的 kernel 效率（Megakernel 类工作），§8 定位不同
- 相邻工作：
  - DistServe/Splitwise/Mooncake（instance-level PDD）：整副本分阶段，纳入范围作主要对比基线
  - MuxWise/Nexus/Semi（Green Context colocation）：GPU 内按阶段切 SM，纳入范围作主要对比基线
  - MegaScale-Infer/Step3-AFD/Janus（attention-expert 分离）：也分离 attention 与 expert，但建在 instance-level PDD 上，继承其局限；纳入范围作定位对比，不作实验基线
  - REEF/GPreempt/PipeSwitch 等 GPU 抢占系统：面向多任务而非单模型两阶段，§7.6 仅作参考点；纳入范围说明区别，不展开其机制

## 2. 核心问题

- Q1：为什么现有的 PD 分离（PDD）和 PD 合设（colocation）都治不好 MoE 大模型的服务效率？
  答案：PDD 给每个阶段配完整模型副本，MoE 权重占参数 95% 以上使副本极贵，P:D 配比的最小部署单元达数百 GPU（DeepSeek-V3 为 32P+320D），小集群配不出、大集群弹性差、故障爆炸半径大；colocation 用 Green Context 在 GPU 内固定切 SM，但 MoE 负载按层、按 rank、按操作类型动态变化，kernel 期间不可重配置（需 CPU 介入），造成 head-of-line blocking（prefill kernel 数十至数百 ms 挡 decode 数百 μs）与资源气泡，且每 GPU 切分迫使更宽并行、加剧网络干扰。
  为什么是核心：论文全部设计动机的来源。依赖：§2 背景、C2/C3/C4。
- Q2：ExpertPlex 的「共享专家 + 分离注意力」架构为什么能同时避开两条路线的死结？
  答案：MoE 权重占 95%+ 参数，共享一份即消除跨阶段冗余并复用动态稀疏负载；attention 权重不足 5%，按阶段分离成整 GPU 后每阶段保留完整本地算力，无需 GPU 内切分，降低并行度与通信、消除 attention 侧跨阶段网络干扰；部署边界与 MoE 权重解耦，粒度更细、故障域更小。
  为什么是核心：论文的中心架构决策。依赖：§3、C1/C6。
- Q3：APK 怎么做到既不被长 prefill kernel 阻塞 decode，又不浪费空闲 SM？
  答案：APK 把 MoE 计算按 tile（kernel 内最小独立完成单位，边界每 2.2–25.3 μs 出现一次、与序列长度无关）调度；抢占决策沿 system→device→DSMEM→shared memory→warp 的存储层级传播，一次协作决策保证 cluster 内一致切换；抢占上界 = 一个 tile 执行时间 + 一次本地检查 epoch；单阶段就绪时用全部 CTA cluster，竞争时按 q' 公式给 decode 分配、其余归 prefill；全程 GPU 内调度，无需 CPU 介入、兼容 CUDA Graph。
  为什么是核心：论文标题机制，共享架构成立的技术前提。依赖：§4、C5/C7/C11/C19。
- Q4：attention 发起的一侧通信如何避免死锁与跨阶段网络干扰？
  答案：APK 常驻执行要求预分配最大 buffer，ExpertPlex 顺势把最终 dispatch/combine buffer 暴露给 attention 侧，dispatch 由 attention push（NVLink peer store / 一侧 RDMA write）、combine 由 attention pull（WaitDone 单线程 kernel 观察完成后拉取），取消 MoE 侧 ring buffer、轮询 SM 与信用回传，从而消除跨阶段等待环（死锁）；prefill 的 scale-out 流量尽量走 prefill attention 服务器之间（分层去重），decode 直连 MoE 服务器，必要时用 IB 虚拟通道优先级隔离。
  为什么是核心：共享专家后两阶段耦合于通信，该机制是架构可行的另一半。依赖：§5、C8/C20。
- Q5：ExpertPlex 的 goodput 提升有多大，结论在什么条件下成立？
  答案：P90 goodput 主指标下，MiniMax-M2.7+ShareGPT 达 11.3 req/s/node，是 PDD 的 2.01×、colocation（PDMux）的 1.41×；MiniMax-M2.7+LooGLE 对 Colocated 4.12×、对 PDMux 1.28×；GLM-5.1-FP8 对 PDMux 在 ShareGPT 持平（约 1.5 req/s/node）、LooGLE 1.66×；成立条件：SGLang 系基线、H800 单节点（MiniMax）与最多 3 节点（GLM）、ShareGPT/LooGLE 采样、论文给定 SLO；GLM 上 PDD 因 OOM 无数、部分基线只能用 16 GPU 布局。
  为什么是核心：论文价值的最终证据与边界。依赖：§7、C12–C18、N1–N4。

## 3. 内容分级

- 核心内容（缺一则核心问题答不全）：PDD 与 colocation 的局限（Q1）；混合架构与权重占比论据（Q2）；GPU 执行模型中 tile/CTA cluster 的最小必要部分、APK tile 调度与有界抢占、在线 SM 重分配（Q3）；两侧通信死锁成因、一侧 push/pull、分层 prefill 路径（Q4）；goodput 定义 Eq.(1)、端到端数字与条件、APK/调度/通信三个微基准（Q5）
- 辅助内容（消除理解障碍）：tile-aware 延迟模型 Eq.(2)(3)（解释为什么优化器需要 x_moe，支撑 Q5 的成立条件理解）；MIG profile 限制 C19（解释为什么硬件切分不够）；TBO/SBO 概念（理解通信重叠对照）；H800 NVLink/IB 带宽差 C20
- 扩展内容：离线搜索算法逐行细节（排除，伪代码主干一句话即可，不影响核心问题）；与 Megakernel 类工作的详细区别（排除，§8 一段概括）；分层 dispatch 的 chunk 机制细节（排除，不影响 Q4 主答案）；DSA vs full attention 差异（排除，仅实验设置事实）

## 4. 前置知识

读者设定为完全小白。逐项检查 wiki/ 下已有概念页（当前 wiki 仅有 3 篇 note，无 concept 流程产物）：

| 前置知识 | 被哪些核心内容依赖 | 概念页状态 | 递归深度 |
|---|---|---|---|
| MoE 大模型推理与服务基础（Transformer 层结构→MoE 替换 FFN、router/top-k、shared/routed expert、EP 与 all-to-all dispatch/combine、TBO/SBO、prefill/decode 两阶段、KV cache、TTFT/TPOT/SLO/goodput） | Q1 全部局限分析、Q2 架构、Q4 通信、Q5 指标定义 | 无 → 递归生成 `wiki/moe-serving/`（depth 1） | 1 |
| GPU 执行模型与 kernel 调度（SM/Tensor Core/TMA/warp/CTA/cluster/DSMEM、stream 与 kernel 启动、GEMM 与 tile、CUDA Graph、persistent kernel、GPU 共享机制 stream 优先级/MPS/MIG/Green Context） | Q3 全部 APK 机制、Q1 的 Green Context 局限、Q5 微基准 | 无 → 递归生成 `wiki/gpu-execution-model/`（depth 1） | 1 |

depth-2 判定：两个概念页均按 concept 流程「从最小必要前置开始」写作，自身不依赖 wiki 级概念页；depth-2 无缺口，无需再递归。辅助引用（非概念页映射）：`wiki/vllm-cudagraph/index.html`（note，CUDA Graph 工程视角）与 `wiki/vllm-parallelism/index.html`（note，EP/TP 工程视角）可作扩展阅读链接。

## 5. 明确不展开的内容

- chunked prefill 的机制细节（§2.5 只作局限性来源，其 rereading 开销一句话即可；不展开原因：不影响核心问题回答，属另一条技术路线）
- MuxWise 的 TP-attention 实现细节（只作基线；其设计约束已在实验条件说明）
- DeepEP 的 ring buffer 协议完整细节（只讲清两侧等待环成因；完整协议属另一独立工作）
- 离线搜索算法的枚举空间大小与搜索耗时（论文未报告；只影响工程规模）
- IBGDA 定制实现细节（§7.1 实现事实，一句话即可；属工程实现）

## 6. 常见误解与适用边界

- 误解 1：「2.01× 是全面碾压」。错误理解：ExpertPlex 在所有设置下都比所有基线快 2 倍。正确结论：2.01× 是 MiniMax-M2.7+ShareGPT 上对 SGLang-PDD 的单点数字；对 PDMux 在 GLM-5.1-FP8+ShareGPT 上持平（约 1.5 req/s/node）。形成原因：abstract 只报最优数字。影响 Q5。
- 误解 2：「共享专家就是没有隔离」。错误理解：两阶段共享 MoE GPU 会互相干扰。正确结论：APK 提供有界抢占（微基准 decode 仅 +8% 开销）与 SM 预算隔离，通信按路径/优先级隔离。形成原因：把共享等同于无管理混跑。影响 Q2/Q3。
- 误解 3：「抢占上界与输入长度有关」。错误理解：长输入 prefill 被抢占要等更久。正确结论：上界 = 一个 tile 时间 + 一次检查 epoch（实测 <25.3 μs），输入变长只是 tile 变多，单个调度间隔不变。影响 Q3。
- 误解 4：「一侧通信会牺牲通信效率」。正确结论：normal 模式与 DeepEP v1 差约 5%，低延迟模式差约 45 μs 以内（§7.5）。影响 Q4。
- 适用边界：
  - 解决问题：单 MoE 模型两阶段服务的资源匹配与隔离。不解决：router 负载不均、attention 侧优化、模型训练
  - 结论条件：SGLang 系实现、H800 硬件（NVLink 160GB/s vs IB 50GB/s 的 3.2× 差距是架构论据之一）、FP8 模型、ShareGPT/LooGLE 长度分布、Poisson 到达、论文 SLO 表
  - 未覆盖：非 SGLang 栈、其他代际 GPU（H100/B200 的 MIG profile 与带宽比不同）、超过 3 节点的规模、多模型混部、prefill/decode 之外的场景（如 embedding）

## 7. 论断分级

- C1–C20、F1–F4、N1–N4 全部为「论文明确声称」，定位见 evidence.md
- 「tile 级调度是这两种死结的自然解法」类表述为基于证据的推断，归入独立评价章节并标记
- 独立评价章节的优点/局限/适用场景判断全部为解读者推断
- 无「缺失假设的猜测」级论断进入核心内容
