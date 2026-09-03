# PCP 与 DCP 内容范围

## 1. 概念歧义处理

状态：已裁定。

- PCP / DCP 存在多个无关领域的同名缩写（网络协议等），但在 LLM 推理语境下指向唯一：vLLM 将 Context Parallel（上下文并行）按推理阶段拆分出的 Prefill Context Parallel（PCP，预填充上下文并行）与 Decode Context Parallel（DCP，解码上下文并行）。用户已确认采用该语境。
- "Context Parallel" 一词在训练框架（如 Megatron-LM）与推理框架中的内涵有差异；本文以 vLLM 推理语境的定义为准，训练语境不纳入。
- 裁定依据：vLLM 官方文档《Context Parallel Deployment》、vLLM 官方博客（2026-08-07）、vLLM 源码 `vllm/config/parallel.py` 三处一致。

## 2. 概念含义

- 概念名称：PCP 与 DCP（上下文并行的两个阶段形态）
- 英文与缩写：Prefill Context Parallel（PCP）、Decode Context Parallel（DCP）；上位概念 Context Parallel（CP，上下文并行）
- 简要定义：把一条长序列（上下文）沿 token 维切到多张 GPU 上处理。PCP 在 prefill 阶段切分序列计算以缩短 TTFT；DCP 在 decode 阶段把 KV cache 沿 token 维分片存储以消除重复、提高可容纳的并发请求数。
- 正式定义（与来源一致）：
  - 官方文档：prefill 与 decode 呈现完全不同的特性并有不同的 SLO，因此上下文并行需要分别实现；长上下文 prefill 要通过把计算摊到多个 query token 上控制 TTFT，长上下文 decode 需要更多 KV cache 空间以增大 batch（从而提高吞吐）。
  - 源码：`prefill_context_parallel_size` 是切分 prefill 序列计算的 rank 数，PCP 扩展进程 world size 但不增加 KV cache 分片数；`decode_context_parallel_size` 是切分 decode KV cache 的 rank 数，DCP 不扩展 world size，无 PCP 时复用 TP rank。
- 本文采用的语境：vLLM 推理框架中的 PCP 与 DCP。

### 包括什么

| 内容 | 属于本概念的理由 |
|---|---|
| prefill/decode 两阶段瓶颈差异与 SLO | PCP/DCP 分化的直接动机（官方文档开篇） |
| TP 沿注意力头切 KV cache 的天花板与重复因子 | DCP 的动机：head 维切尽后的重复存储 |
| DCP 的 token 维分片、交错分配、decode 通信流程 | DCP 的核心机制 |
| LSE 合并（部分 attention 结果的精确合并） | DCP 每步 decode 的正确性核心 |
| PCP 的序列切分、两种 KV 交换策略、causal 负载均衡 | PCP 的核心机制 |
| dcp 取值范围与约束、官方选型建议、三个 case study | 概念的适用边界 |
| 与 PD 分离的概念辨析 | 最易与 PCP/DCP 混淆的相邻部署概念 |

### 不包括什么

| 内容 | 排除理由 |
|---|---|
| Ring Attention 的完整算法与通信调度推导 | 独立概念（有原始论文），本文只引用其角色：PCP partial-KV 策略的实现手段 |
| MoE 层在 PCP 下的 EPLB 约束、专家重均衡 | 属于 MoE 服务领域，只影响工程规模，不影响 PCP/DCP 核心机制 |
| vllm-ascend（昇腾）兼容矩阵的完整复述 | 硬件移植细节，且矩阵随版本变动；只引用与上游一致的支持状态 |
| Helix Parallelism 的 overlap 调度（HOP-B）细节 | 独立系统工作；本文只引用其对 interleaving 分片策略的详细解释 |
| 训练场景的 Context Parallel（Megatron 式） | 语境已裁定为推理 |
| MTP（多 token 预测）与 DCP 的组合机制 | 只作为支持状态一句话提及（有 speculative-decoding 页） |

### 相邻概念

| 概念 | 关键区别 | 是否纳入 |
|---|---|---|
| chunked prefill | 同样面向长 prompt，但按时间串行切块（一个请求分多步算），不增加并行卡数；PCP 按空间把序列切到多卡并行（一步内多卡同时算） | 第 1 章对比一次，链接已有页 |
| PD 分离（disaggregation） | 部署维度：prefill 与 decode 放在不同 GPU 池；PCP/DCP 是池内单请求的执行维度；两者可组合 | 第 5 章辨析，链接 ppd-disaggregation（paper 页） |
| 序列并行（SP） | 训练语境中对激活值的切分；vLLM 的 PCP 文档将其列为相关但独立的功能 | 不纳入，来源章节一句带过即可 |
| TP / DP / PP | 权重与请求维度的并行；PCP/DCP 是序列维度的并行，且 DCP 复用 TP 的卡 | 第 2 章引用 model-parallelism |

## 3. 学习目标

### Q1：同一条长上下文请求，为什么 prefill 和 decode 需要两种不同的上下文并行？

- 完成答案：prefill 一次处理全部输入 token，attention 计算量随长度平方增长，压力在 GPU 算力，用户感知指标是 TTFT；decode 每步只算一个新 token 但要读全部历史 KV cache，压力在显存容量，容量决定并发上限与吞吐。两种瓶颈、两种 SLO，对应两种切法：PCP 切 prefill 计算，DCP 切 decode 的 KV cache。chunked prefill 与 PCP 的区别（时间串行 vs 空间并行）也能说明。
- 为什么是核心目标：不理解两阶段瓶颈差异，就无法理解为什么 CP 要拆成两个而不是一个统一机制。
- 依赖内容：prefill/decode 流程、KV cache 的作用、TTFT/吞吐的含义。

### Q2：张量并行沿注意力头切 KV cache，为什么在 GQA 和 MLA 模型上会重复存储？重复多少倍怎么算？

- 完成答案：TP 把 $H$ 个 KV 头分给 $t$ 张卡；$t \le H$ 时每卡 $H/t$ 个头、无重复；$t > H$ 时每个头复制到多张卡，KV cache 总副本数变为 $t/H$ 倍。GQA 模型 KV 头少（Qwen3-235B 为 4），MLA 把 K/V 压成共享潜向量、有效 KV 头数为 1，潜向量 KV cache 在每个 TP rank 上完整复制。倍数公式 $\text{重复因子} = \max(1, t/H)$，其中 MLA 取 $H=1$。
- 为什么是核心目标：这是 DCP 存在的理由；算不清重复倍数就无法确定 dcp 的有效范围和收益。
- 依赖内容：TP 对注意力权重的切法、KV head 概念、GQA/MLA 结构。

### Q3：DCP 沿 token 维怎么分片 KV cache？每步 decode 的通信怎么走？为什么分片计算合并后与整卡计算等价？

- 完成答案：dcp 组内每卡只存序列中 $1/d$ 的 token 位置的 KV，按交错规则分配（token $i$ 存到 rank $i \bmod d$），使 decode 中逐 token 增长的 KV 自然均匀分布到各卡。每步 decode：AllGather 把 query 聚齐到每卡（query 每步只有一个 token，通信量小）→ 各卡对本地 KV 切片算 attention 得到部分输出与本地 LSE → 用 LSE 作权重合并部分输出（AllGather + ReduceScatter），数学上 $o = \sum_r e^{l_r} o_r / \sum_r e^{l_r}$ 与整卡全局 attention 完全等价（online-softmax 分解）。可手算验证。
- 为什么是核心目标：这是本页机制核心；等价性回答"分片是否改变计算结果"这一根本疑问。
- 依赖内容：softmax attention、KV cache 分页、AllGather/ReduceScatter 通信原语。

### Q4：PCP 怎么切分 prefill 计算？为什么不能把序列按顺序等分？PCP 与 DCP 在资源占用上有什么本质区别？

- 完成答案：$N$ 张卡把 $T$ 个 token 的 prefill 切成段，每卡算自己段的 Q/K/V 与对应 attention 输出；KV 需要全量时用 AllGather 聚齐（partial-Q/full-KV 策略），序列长到放不下全量 KV 时用 ring attention 逐块交换（partial-Q/partial-KV 策略）。causal mask 使靠后 token 的 attention 计算量更大，按顺序等分会导致负载随段位置递增；把序列切成 $2N$ 块、rank $i$ 取第 $i$ 块与第 $2N-1-i$ 块的配对切分使各卡负载相等（等差数列首尾配对和恒定）。资源区别：PCP 扩展 world size（设备数变为 TP × PCP），DCP 不增加设备、复用 TP rank。
- 为什么是核心目标：PCP 的负载均衡是其区别于朴素切分的关键设计；PCP/DCP 的资源差异决定两者的启用成本，是选型依据。
- 依赖内容：causal mask、attention 计算量与序列长度的关系。

### Q5：什么时候该用 DCP/PCP？dcp 取多少合适？它们和 PD 分离是什么关系？

- 完成答案：官方建议先把 tp 加到性能满意，再加 dcp 消除 KV 重复；dcp 有效范围为 $[1, t/H]$（源码校验为 $t \bmod d = 0$，PCP 开启时 $d \in \{1, \text{pcp}, t \cdot \text{pcp}\}$），dcp 越大重复越少但每步通信越多。收益主要在高并发长上下文场景（实测 Kimi K2.6 / 8×B200：吞吐 1,863 → 6,091 tok/s/GPU，可支撑并发 64 → 512，KV 占用 82%），短上下文或互联差时可能负收益。三个官方 case：DeepSeek-R1（H=1）tp8/dcp8；Kimi-K2 tp16/dcp16 或 dcp8（重复降为 2 倍且通信留在节点内）；Qwen3-235B（H=4）tp8/dcp2。与 PD 分离的区别：PD 分离决定 prefill/decode 跑在哪些 GPU 池（部署维度），PCP/DCP 决定池内一条长请求怎么跨卡（执行维度），两者可组合。
- 为什么是核心目标：概念的价值最终落在适用边界；混淆 PCP/DCP 与 PD 分离是最常见的实际错误。
- 依赖内容：前四章全部机制。

## 4. 内容分级

### 核心内容（缺失则学习目标无法完整回答）

| 内容 | 支持的目标 | 必须说明的结论 |
|---|---|---|
| prefill/decode 瓶颈差异与 SLO、CP 分化动机 | Q1 | prefill 撞算力（TTFT）、decode 撞 KV 容量（并发/吞吐） |
| chunked prefill 与 PCP 的对比 | Q1 | 时间串行切块 vs 空间并行切卡 |
| TP 的 head 维分片、$t \le H$ 与 $t > H$ 两种情况 | Q2 | 重复只在 $t > H$ 时出现 |
| 重复因子 $\max(1, t/H)$、KV 条目数 $H \times T$ | Q2, Q5 | 倍数计算与 dcp 范围推导 |
| GQA 头数少、MLA 有效单头 | Q2 | 两类现代模型都撞 head 维天花板 |
| DCP token 维分片与交错分配规则 | Q3 | token $i$ → rank $i \bmod d$；decode 增长均匀 |
| decode 一步的通信流程 | Q3 | AllGather Q → 本地 attention → LSE 合并 |
| LSE 合并公式与等价性 | Q3 | $\sum_r e^{l_r} o_r / \sum_r e^{l_r}$ 等于全局输出 |
| DCP 收益链路与代价 | Q3, Q5 | 重复消除 → 并发上限 → 吞吐；代价是每步通信 |
| PCP 两种策略（full-KV / partial-KV） | Q4 | 按"能否容纳全量 KV"分流 |
| causal 负载不均与 $2N$ 配对切分 | Q4 | 首尾配对和恒定 |
| PCP 扩展 world size、DCP 不扩展 | Q4, Q5 | 设备数 TP × PCP vs 复用 TP rank |
| dcp 取值范围、源码约束、官方建议 | Q5 | $[1, t/H]$；先 tp 后 dcp |
| 三个官方 case study 数值 | Q5 | R1 / K2 / Qwen3-235B 的 tp、dcp 组合 |
| 实测性能数字（Kimi K2.6 / 8×B200） | Q5 | 收益的量化参照 |
| PCP/DCP 与 PD 分离的辨析 | Q5 | 部署维度 vs 执行维度 |

### 辅助内容（消除理解障碍或误解）

| 内容 | 服务的核心内容或误解 |
|---|---|
| "DCP 不是让单请求更快"的澄清 | 误解 1（收益是并发吞吐，非单请求延迟） |
| "DCP 不加卡、PCP 才加卡"的澄清 | 误解 3 |
| blog 的连续区间示意与实现交错分配的关系 | 消除"两种描述矛盾"的困惑：概念上按位置分片，实现用交错分配 |
| MLA 的 k_up / GQA 的 tensor_broadcast 路径差异 | Q3 的两类模型落地方式 |
| a2a 通信后端与 q_replicate 选项 | 源码层面的通信优化存在性（折叠块） |
| dcp 超过 $t/H$ 的理论讨论 | Q5 边界：为什么上界取 $t/H$ |
| 短上下文/弱互联下的负收益边界 | Q5 适用边界 |
| DCP 已支持约一年、PCP 仍在开发的现状 | 概念成熟度，避免读者误以为两者同等可用 |

### 扩展内容

| 内容 | 纳入/排除 |
|---|---|
| Ring Attention 完整机制 | 排除，链接 arXiv:2310.01889 |
| Helix HOP-B overlap 调度 | 排除，链接 arXiv:2507.07120 |
| block table 在 CP 下的 gap 与 compaction | 排除（RFC 细节，只影响实现） |
| MTP + DCP 组合 | 一句话提及 + speculative-decoding 链接 |
| vllm-ascend 兼容矩阵 | 排除 |

## 5. 前置知识映射

| 前置概念 | 被依赖的学习目标 | 页面状态 |
|---|---|---|
| KV cache | Q1, Q2, Q3 | 已有：`../kv-cache/index.html`（concept） |
| 标准注意力（softmax attention） | Q1, Q3 | 已有：`../standard-attention/index.html`（concept） |
| chunked prefill | Q1 | 已有：`../chunked-prefill/index.html`（concept） |
| 模型并行 / TP | Q2, Q3 | 已有：`../model-parallelism/index.html`（concept） |
| MQA 与 GQA | Q2 | 已有：`../mqa-gqa/index.html`（concept） |
| MLA | Q2 | 已有：`../mla/index.html`（concept） |
| GPU 通信原语（AllGather/ReduceScatter） | Q3 | 已有：`../gpu-communication/index.html`（note） |
| causal mask | Q4 | 已有：`../causal-mask/index.html`（concept） |
| PD 分离（延伸阅读） | Q5 | 已有：`../ppd-disaggregation/index.html`（paper） |

无缺失前置页，不需要递归生成。

## 6. 明确不展开的内容

| 内容 | 与概念的关系 | 不展开的原因 |
|---|---|---|
| Ring Attention 的块交换调度 | PCP partial-KV 策略的实现手段 | 独立概念，有原始论文；展开会喧宾夺主 |
| MoE 层与 PCP 的交互（EPLB 约束） | PCP 的 MoE 支持面 | 只影响工程规模，不影响核心机制理解 |
| 昇腾移植的兼容矩阵 | DCP/PCP 的硬件支持面 | 版本相关且与上游 CUDA 版不完全一致 |
| Helix 的通信隐藏调度 | interleaving 策略的兄弟工作 | 独立系统贡献，超出本页范围 |
| 训练语境的 CP | 同名上位概念 | 语境已裁定为推理 |
| prefix caching 与交错分片的 block 交互 | 实现细节 | 不影响机制结论 |

## 7. 常见误解与适用边界

### 常见误解

| 编号 | 错误理解 | 正确结论 | 形成原因 | 影响目标 |
|---|---|---|---|---|
| M1 | DCP 让单个请求的 decode 快 $d$ 倍 | DCP 消除的是 KV 重复存储，收益路径是"显存释放 → 可容纳更多并发请求 → 吞吐提升"；单请求每步 decode 反而多了组内通信，延迟可能不降反升 | 把"吞吐提升 3 倍"误读为"单请求快 3 倍" | Q3, Q5 |
| M2 | PCP/DCP 是 PD 分离的另一种叫法 | PD 分离决定 prefill/decode 放在哪个 GPU 池（部署维度）；PCP/DCP 决定池内一条长请求如何跨卡执行（执行维度）；两者正交可组合 | 缩写相似（都有 P 和 D）且都涉及 prefill/decode 二分 | Q5 |
| M3 | DCP 需要增加 GPU | dcp 不扩展 world size，DCP 组复用已有 TP rank；PCP 才扩展设备数（TP × PCP） | "并行度"直觉上等于"更多设备" | Q4, Q5 |
| M4 | TP 超过 KV 头数后就不能再用 TP | TP 仍然正常切分权重与激活，只是 KV cache 部分开始重复存储，浪费显存 | 把"KV 切不动"扩大为"TP 失效" | Q2 |
| M5 | 交错分配是任意/随机选择 | token $i$ → rank $i \bmod d$ 是确定性规则，目的有二：decode 中逐 token 增长的 KV 轮流落到各卡、增长均匀；任意 token 的归属可由位置直接算出 | 只看到"看起来乱"没看到增长模式 | Q3 |

### 适用边界

- PCP/DCP 解决的是单条长序列的跨卡执行问题；模型权重放不下单卡仍需 TP/PP，请求并发扩展仍需 DP，三者是不同维度。
- DCP 收益成立需要：KV 重复真实存在（$t > H$）、长上下文使显存压力成为主要瓶颈、GPU 间互联带宽足够（NVLink/NVSwitch 级别）；条件不满足时通信开销可能抵消显存收益，短上下文单卡放得下的场景直接用 TP 更简单。
- dcp 上界 $t/H$ 是"消除全部重复"的临界点，不是硬件限制；继续增大只会在非 attention 层留下无法利用的空闲 rank（官方文档的设计取舍）。
- LSE 合并的等价性对任意 attention 精确成立（数学恒等），不依赖近似；但真实实现还叠加分页 KV、量化、kernel 融合等工程层。
- PCP 目前处于开发中状态（官方文档 "under active development"、博客 "longer roadmap"）；DCP 已在 vLLM 支持 MLA 与 GQA 模型约一年。本页 PCP 部分描述的是设计机制，生产可用性以官方文档为准。
