# UltraEP 内容范围

## 1. 论文定位

- 标题：UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing
- 作者：Xinming Wei¹*, Chao Jin¹, Tuo Dai², Yinmin Zhong¹, Shan Yu³, Chengxu Yang⁴, Bingyang Wu¹, Zili Zhang¹, Jing Mai¹, Qianchao Zhu⁴, Zhouyang Li⁴, Yuliang Liu⁴†, Guojie Luo¹†
- 单位：¹北京大学计算机学院 ²小红书 ³上海人工智能实验室 ⁴独立研究者（*实习期间完成 †通信作者）
- 发表：arXiv 预印本，2026，v3（2026-06-18 提交；v1/v2 已被作者撤回）
- 链接：https://arxiv.org/abs/2606.04101v3
- 代码仓库：论文未给出。§7 只说明核心库约 9.6K 行 C++（含 device kernel）与 Python，集成到 Megatron-LM 与 SGLang 各低于 1K 行。页面不得声称存在公开仓库。
- 一句话：UltraEP 用「机架级节点内的高带宽 scale-up 域 + GPU 上原生求解的配额规划 + 面向不规则专家搬运的通信内核」，解决「大规模专家并行下按历史负载周期性均衡跟不上真实负载漂移」的问题。

### 1.1 论文宣称的贡献（§1 结尾 contribution 列表）

| 编号 | 贡献 | 原文依据 |
|---|---|---|
| B1 | 刻画大规模 MoE 训练与 serving prefill 中非平稳的专家负载不均衡 | §1 contribution 第 1 条，指向 §3 |
| B2 | 设计首个面向 RSN 上大规模 EP MoE 部署的实时精确负载均衡器 | §1 contribution 第 2 条，指向 §4 |
| B3 | 构建配额式规划与 RSN 原生的专家状态通信，用于热路径均衡 | §1 contribution 第 3 条，指向 §5、§6 |
| B4 | 验证近最优均衡质量、近理想吞吐与生产可扩展性 | §1 contribution 第 4 条，指向 §8 |

### 1.2 论文没做什么

| 项 | 排除依据 |
|---|---|
| 不做 decode 阶段均衡 | §3「Serving Prefill」：decode 是 memory-bound，计算侧不均衡被访存延迟大幅稀释；增大 batch 提高计算强度又与 decode TPOT 的 SLO 冲突，所以实际均衡目标是 prefill。摘要与标题也限定为 "serving prefill"。 |
| 不改路由算法，不替代辅助损失 | §2 最后一段：算法侧路由正则与系统侧均衡互补而非互换；§8.6：loss 曲线符合预期是因为 UltraEP 只改物理执行逻辑、保留训练语义。 |
| 不做主专家重排 | §4.1「Replication Only」：只做复制，从不重排主专家。 |
| 不做跨机架的 EP | §4 开头：每个 EP 组位于一个 RSN 的 scale-up 域内，跨机架扩展靠 PP 和 DP。 |
| 不优化 token all-to-all 内核本身 | §7：token dispatch/combine 用 DeepEP（hybrid-ep 分支，v1.2.1+7febc6e）；§9「MoE Computation and Communication Optimization」：这类优化与 UltraEP 正交、可叠加。 |
| 不做专家预取或负载预测 | §9：预测式设计（history / 跨层相关性 / profiling）在高动态细粒度 MoE + 大 EP + RSN 场景下缺乏实用性，UltraEP 反应的是已实现负载。 |

### 1.3 相邻工作

| 方法 | 关键区别 | 是否纳入本页 |
|---|---|---|
| EPLB | 冗余专家启发式：按近期路由历史周期性重排布局，reroute 交给独立启发式（如 round-robin）。§1、§8.1 | 纳入，作为主要对照 |
| EPLB+（论文构造的强化基线） | 把 UltraEP 的规划换成标准 EPLB + round-robin reroute，通信机制不变，喂精确负载。§8.1 | 纳入，用于隔离配额求解的贡献 |
| LPLB | 线性规划求解器，在 EPLB 之上为每个 microbatch 做 reroute 调整；限制每个专家至多一个副本以降低开销。§8.1 | 纳入，仅作基线 |
| MoonEP | 已有页面 `../moonep/index.html`，追求每个 rank 恰好收到 $S\times K$ 个 token。moonep 页已把 UltraEP 列在对比中。 | 提及并链接，不展开 |
| DeepEP | token all-to-all 通信库，被 UltraEP 用作 dispatch/combine 后端，不是竞品；在 §8.5 中被适配来做专家权重搬运的通信基线。§7、§8.5 | 纳入通信章节的基线对照 |

## 2. 核心问题（4 个）

### Q1 为什么按历史负载周期性均衡在前沿 MoE 上会失效

预期答案：因为专家热度在 microbatch、层与数据域之间快速漂移，历史统计不再代表下一步的真实负载。§3 观测：serving prefill 中专家热度随 science/coding/mixed 语义切换剧烈变化，同一域内热专家也会逐批漂移；训练中辅助损失的负反馈持续重调专家利用率，DeepSeek 式路由补偿降低整体不均衡但放大短期摆动。大 EP 让每个 rank 只剩很少主专家（常 2 或 4），专家级抖动直接转成 rank 级倾斜。Figure 6 显示 EPLB 在两种负载下都跟不上，甚至会制造新的尖峰与 straggler。

重要性：这是全篇动机，决定了为什么必须把均衡搬上热路径。
依赖：MoE 与专家并行基础（引 `../moe-serving/`）、辅助损失均衡（引 `../aux-loss-free-routing/`）。

### Q2 精确负载只在 gating 之后才拿到，为什么这件事在 RSN 上才变得可行

预期答案：精确负载在 gating 后才产生，于是规划与专家状态搬运必然落在关键路径上（前向要搬权重，训练还要搬梯度或优化器状态）。标准 RDMA 集群里高带宽 scale-up 只覆盖单台 4/8 卡服务器，跨机走慢得多的 scale-out；大 EP 下跨多机搬专家状态在热路径上不现实。RSN 把 scale-up 域扩展到整机架 64+ 卡（§2.1），单个 EP 组可以整体装进高带宽域，热路径均衡才物理可行。但 RSN 是必要不充分：控制面要在 gating 与 token dispatch 之间的极短窗口里出高质量决策，数据面要执行静态集合通信支撑不好的不规则、易变的专家状态搬运（§1、§4.4）。

重要性：解释「为什么是现在、为什么是 RSN」，并引出两个挑战对应的两章。
依赖：GPU 通信（引 `../gpu-communication/`）、模型并行（引 `../model-parallelism/`）。

### Q3 配额求解为什么比「先定副本再 round-robin 分流」均衡得更好，还更省资源

预期答案：配额 $u_{e,r}$ 是耦合变量——它同时是「这个副本要不要建」和「建了之后承接多少 token」的答案。UltraEP 二分搜索最小可行阈值 $\tau$，每次探测跑一个贪心可行性 oracle：超载 rank 按残余超额降序、其主专家按 $\lambda_e$ 降序，把负载搬给空闲最多的合法目标 rank，受配额下限 $u_{\min}$ 约束（§5.1）。因为规划直接优化的是 reroute 之后的负载上界，副本只在能带来足够均衡收益时才被实体化。EPLB+ 优化的是 reroute 之前的热度，副本建了却可能收不到多少流量。Table 3 模拟均值：结果不均衡 1.03 vs 1.19，求解耗时省 27.4%，冗余槽少用 57.9%，在飞 token 降 3.9 个百分点（本地优先带来）。

重要性：这是论文控制面的核心创新，也是与 EPLB 家族的分界线。
依赖：Q2 的问题形式化（§4.3 的 Eq.(1)–(5)）。

### Q4 每个 microbatch 每一层都重新搬专家权重，为什么没把均衡收益吃掉

预期答案：靠三件事。① 内存布局：冗余槽不存优化器状态、权重/梯度 buffer 跨层复用，Qwen3-235B-A22B（94 个 MoE 层、128 专家）下单槽从 3.3 GB 权重 + 6.6 GB 梯度降到每 rank 36 MB + 72 MB（§4.1），代价是前向关键路径上多了一条逐层权重实体化的紧期限。② persistent tile streaming：权重与梯度切成固定大小 tile 编译成常驻设备的搬运任务，persistent kernel 的 thread block 反复取下一个 tile，共享内存双缓冲，把任务查表、地址翻译、同步折进 tile 流水线（§6.1）。③ chunk streaming relay：副本数超过阈值 4 的热专家改走两级中继，中继前沿取在 $\sqrt{|\mathcal{H}(e)|-1}$ 附近，按连续 tile 组成的 chunk 流式转发、不设全局阶段屏障（§6.2）。实测（Figure 13，Qwen3-235B 训练）非 MoE 部分前向多 0.33 ms、反向可忽略，占总延迟 1.8%。

重要性：这是数据面的核心创新，也是「实时均衡是否划算」的直接回答。
依赖：GPU 执行模型中的 tile 与 persistent kernel（引 `../gpu-execution-model/`）。

## 3. 内容分级

### 核心内容
- 非平稳负载的三个观测维度与 EPLB 失效（§3）→ Q1
- RSN 的 scale-up 域扩展与热路径可行性（§2.1、§1）→ Q2
- 两个挑战：控制面决策窗口、数据面带宽利用（§4.4）→ Q2
- 专家布局与内存管理：逻辑/物理专家、主槽与冗余槽、只复制不重排、跨层 buffer 复用（§4.1）→ Q4
- 前向与反向流水线（§4.2）→ Q2、Q4
- 问题形式化：Eq.(1)–(5)、约束（§4.3）→ Q3
- 配额求解：阈值形式化 Eq.(6)、二分 + 贪心 oracle、Algorithm 1、GPU 原生求解（§5）→ Q3
- reroute：本地优先 + 按残余配额比例分摊 + 前缀扫描定 token 目标（§5.2）→ Q3
- persistent tile streaming 与 overlap-aware footprint（§6.1）→ Q4
- chunk streaming relay 与 load-aware relay scheduling（§6.2）→ Q4
- 端到端结果、延迟分解、消融、生产训练（§8）→ Q1、Q3、Q4

### 辅助内容
- 为什么不做 decode 均衡（§3）——澄清「MoE 均衡」的常见过度推广
- 只复制不重排的理由（§4.1）——消除「为什么不像 EPLB 那样重排」的疑问
- virtual layer ID 如何让反向找回前向计划（§7）——澄清 PP 下的正确性
- 活跃内存峰值的下降（§8.4）——说明均衡不只影响延迟

### 扩展内容
- RSN 内存语义与 one-sided peer access 实现细节（§7）：纳入，放折叠块
- 端到端集成（参数桶、checkpoint 排除、lazy 注册）（§7）：纳入，压缩到一段
- 相关工作全景（§9）：部分纳入，只保留与本页对照相关的
- 扩展到 RL pipeline 的展望（§10）：纳入一句，标注为论文展望
- Figure 3/4（训练与 serving 负载分布细节）：排除，Figure 6 已足够支撑 Q1
- Figure 14 内存分解：纳入结论数字，不放图

## 4. 前置知识

| 前置概念 | 被哪些核心内容依赖 | 概念页 | 状态 |
|---|---|---|---|
| MoE、router、top-k、专家并行、all-to-all、prefill/decode、TTFT、TPOT | 全篇 | `../moe-serving/index.html` | 已有 |
| 辅助损失式负载均衡与路由 bias 补偿 | Q1 的训练侧观测（§3、§2 末段） | `../aux-loss-free-routing/index.html` | 已有 |
| scale-up 与 scale-out、NVLink/NVSwitch、RDMA、集合通信原语 | Q2 的 RSN 论证、§6 通信章节 | `../gpu-communication/index.html` | 已有 |
| TP / PP / DP 及其组合 | §4 开头的并行分工、§7 的 PP 兼容 | `../model-parallelism/index.html` | 已有 |
| tile、persistent kernel、共享内存、SM 占用、warp | Q4 的 §6.1 | `../gpu-execution-model/index.html` | 已有 |
| chunked prefill 与 token budget | §4.3 中 prefill 侧发送量上界 | `../chunked-prefill/index.html` | 已有 |
| MoonEP 的完美均衡路线 | 方法评价章节的相邻工作定位 | `../moonep/index.html` | 已有 |
| DeepSeekMoE 的细粒度专家 + 共享专家 | §2.2 的 MoE 架构演进 | `../deepseek-moe/index.html` | 已有 |

全部前置概念页已存在，无需递归生成。

## 5. 明确不展开的内容

| 内容 | 与论文的关系 | 不展开原因 |
|---|---|---|
| DeepEP 的 token dispatch 内核实现 | §7 用它作 all-to-all 后端 | 属于另一独立工作；本页只需知道它承担 token 搬运，不影响 Q1–Q4 的回答 |
| Megatron-LM / SGLang 的框架内部结构 | §7 集成目标 | 只影响工程集成规模，不改变均衡机制的结论 |
| 各被测模型的架构差异（GLM4.5/4.7、Qwen3、DeepSeek-V3） | §8.1 Table 2 | 本页只需专家数、top-k、EP 配置与 $N_{\text{slot}}$；架构细节属于各模型自身的工作 |
| 训练语料与 loss 配方细节 | §8.1 Training Recipes | 属实验条件，页面在实验条件处标注即可，不需展开语料构成 |
| SHARP 式交换机内组播 | §4.4 提到 RSN 可把组播下放到交换机 | 论文只用它说明「预设通信组假设不成立」；本页保留这一句判断，不展开 SHARP 机制 |
| 活跃内存重计算（activation checkpointing） | §8.4 脚注 | 论文只用它解释实验为何关掉；机制属于训练内存优化的独立话题 |

## 6. 常见误解与适用边界

### 误解

| 错误理解 | 正确结论 | 形成原因 | 影响 |
|---|---|---|---|
| UltraEP 能给整条推理链路加速，包括 decode | 只覆盖训练与 serving prefill。§3 明确 decode 是 memory-bound、计算侧不均衡被访存延迟稀释，实际均衡目标是 prefill | 「MoE 推理优化」容易被默认为覆盖全流程 | Q1 |
| 1.49× 是相对生产系统的绝对提速 | 摘要的 1.49× 是训练与 serving 平均、相对「不做任何均衡」的基线。相对真实基线的数字是训练相对 Megatron-LM 平均 1.42×、prefill 相对 SGLang 1.56×（§1、§8.2） | 摘要与正文用了不同基线 | Q3、Q4 |
| 94.3% 意味着每个模型都达到 94% 以上 | 94.3% 是训练与 serving 的总平均（训练 94.6%、prefill 93.9%）。训练三模型分别是 96.4%、91.2%、96.1%；prefill 区间 90%–97%（§8.2 及 Figure 11 标注） | 把平均值当成逐配置下界 | Q3 |
| 不均衡度降到 1.01–1.04 意味着吞吐也就差 1%–4% | 残余不均衡不是唯一损耗项。§8.2 明确剩余与强制均衡的差距主要来自真实 MoE 训练中不均匀的路由本身，而非残余不均衡或热路径开销；§8.3 显示 token all-to-all 前向仍比 ideal 高 33% | 把均衡指标线性映射成吞吐 | Q4 |
| 每层每 microbatch 重搬权重必然很贵，论文一定是隔几步搬一次 | 确实是每 microbatch 每层都重新规划并搬运（§1、Figure 1）。可行的原因是冗余槽跨层复用把单槽内存降到 36 MB/72 MB 量级，加上 tile streaming 与 relay 把搬运压到亚毫秒（Figure 16 中 Ours 约 0.28 ms） | 用常规集群的搬运代价直觉外推 | Q4 |
| 系统侧均衡做好了就不再需要辅助损失 | §2 末段明确两者互补不可互换：路由正则的目标是稳定优化、防止 routing collapse、保留专家特化；系统侧只纠正运行时不均衡 | 把两类「均衡」混为一谈 | Q1 |

### 适用边界

- 解决：大 EP（论文实测 EP32/EP40/EP64）下 rank 级专家负载不均衡带来的计算 straggler、token all-to-all 瓶颈与活跃内存尖峰，覆盖训练与 serving prefill。
- 不解决：decode 阶段的均衡；专家内部计算效率；token all-to-all 内核本身；跨机架的 EP。
- 成立条件：每个 EP 组装在一个 RSN 的 scale-up 域内（实验中每机架 64 卡 / 16 台服务器，scale-up 带宽是 scale-out RDMA 的 8–10 倍）；每 rank 有 $N_{\text{slot}}$ 个冗余槽（实验取 2 或 4）；bf16 精度；反向的副本相关通信能被计算完全隐藏（§4.3 约束三）。
- 条件不满足时：若 EP 组跨出 scale-up 域，§1 的判断是跨多机搬专家状态在热路径上代价过高、不可行；若冗余槽预算过紧，Figure 15 显示 $(128,64,1)$ 且初始不均衡 6.0/8.0 时 EPLB+ 升到约 1.4，UltraEP 仍低于 1.1，但两者都会随预算变紧而变差。
- 实验未覆盖：decode；非 bf16 精度；serving 的多机架部署（§8.1 说明 prefill 只用一个机架）；生产训练只报了 RefMoE-288B-A16B 一个内部模型（EP32），且 2560 卡规模的可扩展性结论来自 v3 摘要在智源等站点的转述版本，页面正文不引用该数字。

## 7. 论断分级

| 编号 | 论断 | 分级 | 定位 |
|---|---|---|---|
| A1 | 大 EP 把专家负载不均衡放大为计算 straggler、token all-to-all 瓶颈与活跃内存尖峰 | 论文明确声称 | Abstract、§1 第二段 |
| A2 | 专家热度在 microbatch、层、数据域之间剧烈漂移，历史统计不可靠 | 论文明确声称 | §3、Figure 4/5 |
| A3 | EPLB 在真实负载下不仅跟不上，还可能加剧不均衡、制造新 straggler | 论文明确声称 | §3 末段、Figure 6 |
| A4 | RSN 把 scale-up 域扩到整机架 64+ 卡，使热路径均衡物理可行 | 论文明确声称 | §2.1、§1 第五段 |
| A5 | UltraEP 只做复制、从不重排主专家 | 论文明确声称 | §4.1「Replication Only」 |
| A6 | 冗余槽不存优化器状态、跨层复用 buffer，单槽降到 36 MB + 72 MB | 论文明确声称 | §4.1 末句 |
| A7 | 配额是耦合变量，副本只在承载有效负载时才被实体化 | 论文明确声称 | §1 innovation 第一条、§5.1 |
| A8 | UltraEP 直接优化 reroute 后的负载上界，EPLB+ 优化的是 reroute 前的热度 | 论文明确声称 | §8.5「Balancing Quality」 |
| A9 | 本地优先只改变哪个源 rank 消费配额，不改变配额本身，因此不破坏已解出的阈值 | 论文明确声称 | §5.2「Quota Decomposition with Locality」 |
| A10 | relay 前沿取在 $\sqrt{\lvert\mathcal{H}(e)\rvert-1}$ 附近可近似平衡两级 | 论文明确声称 | §6.2 |
| A11 | 剩余与强制均衡的差距主要来自真实路由的不均匀，不是残余不均衡或热路径开销 | 论文明确声称 | §8.2 末段、§8.3 |
| A12 | 算法侧路由正则与系统侧均衡互补不可互换 | 论文明确声称 | §2.2 末段 |
| A13 | 论文所有实验数字（见 evidence.md 的 N 编号）自洽：由图中标注值反算可复现正文的百分比与倍数 | 基于证据的推断 | 本次核对脚本，17/17 项通过；作为「数字可复算」的依据，不作为新事实 |
| A14 | 只复制不重排在大 EP 下够用，是因为每 rank 主专家数已很少（常 2 或 4），重排边际收益递减 | 论文明确声称 | §4.1 |
| A15 | 二分 + 贪心 oracle 不保证全局最优，只保证在 oracle 的可行性判据下找到最小可行阈值 | 基于证据的推断 | §5.1 用的是 greedy feasibility oracle，论文全篇称 "near-optimal" 而非 optimal；页面须标注为推断 |
| A16 | UltraEP 可与 DeepEP 等 token 通信优化、grouped-GEMM 优化叠加 | 论文明确声称 | §9 末句 |
| A17 | 同一抽象可自然延伸到交替训练与推理的 RL pipeline | 论文明确声称（展望） | §10 末句，页面须标注为论文展望而非已验证结论 |

不存在「缺失假设的猜测」类论断进入页面核心内容。
