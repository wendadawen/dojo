# Mooncake 文章大纲

## 1. 页面开头

### 1.1 开篇场景（直接进入问题）

长上下文请求涌入时，prefill 阶段要把几万 token 一次性送进模型——这在常规推理集群里常常把同 batch 上所有 decode 序列的 TBT 抬高数十毫秒甚至数秒。MaaS 服务的用户既看首字延迟（TTFT）也看连续解码延迟（TBT），两个 SLO 同时被破坏。Mooncake 围绕 KVCache 重构了整套推理架构：把 prefill 与 decode 拆到不同集群，把 GPU 集群里长期闲置的 CPU/DRAM/SSD/RDMA 拉成"近 GPU 的分布式 KVCache 池"，再用一个全局调度器把请求路由到「缓存复用最多」与「负载最轻」同时成立的实例。本页解释这套架构为什么这样设计、它怎么工作、在哪些条件下收益最大。

### 1.2 学习目标

列出 5 个核心问题（与 scope.md §3 一致；详见正文「核心问题」块）。

### 1.3 贯穿示例

构造一个小型 Mooncake 集群场景贯穿全章：

- 块大小 $B=512$ tokens（论文 §4.1）
- 一个长请求 $R$：输入 12,288 tokens = 24 块，输出长度未知
- 3 个 prefill 实例：
  - A：本地前缀 16 块（8,192 tokens，40% 复用），当前队列预计 3.0s
  - B：本地前缀 20 块（10,240 tokens，83% 复用，全局最佳），当前队列 8.0s
  - C：本地无缓存（0 tokens），当前队列 1.0s
- 简化时间模型（构造值，非论文实测）：
  - $T_{\text{prefill}}(p) = 0.5 + \frac{12{,}288 - p}{4{,}000}$ s
  - $T_{\text{transfer}}(t) = \frac{t}{10{,}000}$ s
  - 阈值 $\text{kvcache\_balancing\_threshold}=2$
- 章节 3 用此例跑 Algorithm 1 两条分支并比较纯负载均衡/纯缓存优先的策略
- 章节 4 用此例说明 layer-wise prefill 在长上下文下的节省
- 章节 5 用 trace 统计与表 3 数据讨论过载下的策略选择

### 1.4 与第一章的过渡

本章不重复讲解 [KV cache](../../wiki/kv-cache/index.html)、[Prefix caching](../../wiki/prefix-caching/index.html)、[Chunked Prefill](../../wiki/chunked-prefill/index.html) 等基础概念的定义；遇到依赖时直接引用。

## 2. 章节设计

### 章节 1：把 prefill 与 decode 拆开（动机）

**章节问题**：为什么 Mooncake 选择 PD 分离？合在一起服务会出什么问题？

**完成答案要点**：
- 解释 prefill 与 decode 的计算特性差异（C6、C7）
- 解释 TTFT 与 TBT 两个独立 SLO（C9）
- 解释连续批处理 (Orca) 仍不能消除耦合阶段的相互干扰（C8）
- 解释 Mooncake 选择彻底分离、仅允许"不分块且不破坏 TBT SLO"的小请求 inline（§5 中给出的两条理由 C15）
- 解释 chunked prefill 的方案与不足——它解决"长 prefill 阻塞同 batch decode"，但不解决"prefill 与 decode 对硬件的不同偏好"（C15）

**对应范围**：C6–C9、C15；N10；S1（前置）

**正文要点**：用简短的并行性对比（prefill 计算密集、decode 带宽密集）建立动机；引出 TTFT 与 TBT SLO。

**表达材料**：
- 一张 HTML 结构图（A 顺序流程 `dg-flow`）展示"单 GPU 集群同时承担两阶段 → 长 prefill 抬高同 batch decode TBT"的现象
- 一张表格对比两阶段的资源偏好与 SLO（用 `table`）
- 引用 [Chunked-prefill](../../wiki/chunked-prefill/index.html) 解释 chunked prefill 解决了什么、没解决什么

**前置知识安排**：首次依赖 [KV cache](../../wiki/kv-cache/index.html) 与 [Chunked Prefill](../../wiki/chunked-prefill/index.html) 时给出链接；不内联讲解。

**章末「本章问题」**（h3）：2 题，给出解答折叠块。

---

### 章节 2：Mooncake 架构总览与四步工作流

**章节问题**：Mooncake 由哪些组件构成？一个请求经过哪几步？

**完成答案要点**：
- 列出五大组件：Conductor、prefill 实例池、decode 实例池、分布式 KVCache 池（CPU DRAM + SSD + RDMA）、Messenger（GPUDirect RDMA 跨节点传输组件）
- 描述 KVCache 池的存储形态：CPU 内存中按分页块存储；块哈希链式去重（哈希 = Hash(本块 tokens, 前块 hash)）；按 LRU/LFU/LengthAware 等策略淘汰（C33、§3）
- 描述请求四步流程：
  1. KVCache 复用：prefill 节点把请求中的可复用前缀块从远端 CPU 内存经 RDMA 载入 GPU（如果存在）
  2. 增量 prefill：用已加载的前缀加上剩余输入做 prefill，新产生的增量 KVCache 存回 CPU；若剩余 token 数 > prefill_chunk，则切块流水线执行（C10、N7）
  3. KVCache 传输：Messenger 把逐层产生的全量 KVCache 流式传到选定 decode 节点的 CPU DRAM，与第 2 步重叠（C10）
  4. 解码：decode 节点收齐 KVCache 后将请求加入下一个连续批处理批次；本地调度器复核 TBT SLO（可能在此时拒绝，导致 prefill 算力浪费）

**对应范围**：C2、C3、C10–C13、C33；N3、N7

**正文要点**：用一张分层结构图（`dg-stack`）展示"请求 → Conductor → prefill + decode → KVCache 池"的总体走向；再用顺序流程图（`dg-flow`）展示四步。

**表达材料**：
- 一张分层结构图（`dg-stack`）展示组件层级
- 一张顺序流程图（`dg-flow`）展示四步
- 一张小表格列 KVCache 池存储参数（块大小、淘汰策略、传输组件）
- 引用 [PagedAttention](../../wiki/paged-attention/index.html)（分页管理）、[Prefix caching](../../wiki/prefix-caching/index.html)（哈希链）

**章末「本章问题」**（h3）：2 题，给出解答折叠块。

---

### 章节 3：KVCache-centric 调度算法

**章节问题**：Mooncake 的调度具体怎么算？为什么不能只按负载均衡或只按缓存最长调度？

**完成答案要点**：
- 描述 Algorithm 1（cache-aware prefill 调度）：
  1. 块哈希链计算：`block_keys = PrefixHash(R.prompt_tokens, B)`，每块 key = Hash(本块 + 前块 key)
  2. 全局最佳匹配：`best_prefix_len, best_matched_instance = FindBestPrefixMatch(P, block_keys)`
  3. 对每个实例估计 TTFT（C21、C22）：
     - 分支 1（cache-aware）：若 `best_prefix_len / prefix_len < kvcache_balancing_threshold` 则 `T_prefill = EstimatePrefillExecutionTime(len, prefix_len)`，`TTFT = T_queue + T_prefill`
     - 分支 2（cache-aware and -balancing）：否则 `transfer_len = best_prefix_len - prefix_len`，`T_transfer = EstimateKVCacheTransferTime(...)`，`T_prefill = EstimatePrefillExecutionTime(len, best_prefix_len)`（按传输后复用全部最佳前缀计算），`TTFT = T_transfer + T_queue + T_prefill`
  4. 选 TTFT 最短的实例 `p`；若 TTFT > TTFT_SLO 或 TBT > TBT_SLO 则直接拒绝（HTTP 429）
  5. 热点自动迁移：若 `best_prefix_len / p.prefix_len > kvcache_balancing_threshold` 则 `TransferKVCache(best_matched_instance, p)`（C23）
- 解释 kvcache_balancing_threshold 的作用：权衡本地复用与跨节点传输；以及它的副作用（热点自动迁移 = 把热点复制到更多节点）
- 解释为何 Mooncake 选 Mooncake-centric 而非单纯负载均衡或单纯缓存优先：
  - 单纯负载均衡 → 把请求路由到无缓存实例，重算量大（trace 显示工作负载理论上限 ~50% 复用）
  - 单纯缓存优先 → 把请求路由到过载的缓存实例，队列拖慢 TTFT
  - KVCache-centric → 选"传输差额可承受且 TTFT 最短"的实例，同时自动复制热点
- 实验支撑：8P+8D 回放 23,000 请求，KVCache-centric median TTFT 6.26s、cache-aware 14.36s、load-balancing 60.41s、random 92.07s（N8）

**对应范围**：C21–C24、C33–C34；N8

**正文要点**：用贯穿示例跑完整 Algorithm 1；列出每个实例的 TTFT 估计；比较四种策略。

**表达材料**：
- Algorithm 1 完整伪代码（`dg-flow` 风格或 `<details>` 包伪代码；按 A6 风格不写为 Python）
- 一张对照表格列出贯穿示例下四种策略的 TTFT 估计与最终选择
- 引用 [Prefix caching](../../wiki/prefix-caching/index.html) 中哈希指纹链实现
- 一个折叠块给出预 fill_chunk 与队列时间的构造值定义（标为"构造示例"）

**章末「本章问题」**（h3）：2 题，给出解答折叠块。

---

### 章节 4：prefill 池：长上下文跨节点与传输重叠

**章节问题**：长上下文 prefill 怎么跨节点加速？KVCache 传输怎么不拖慢 GPU？

**完成答案要点**：
- 解释为什么需要跨节点：可上下文长度从 8k 到 128K 到 1M（C16 引用 [16]），单 8×GPU 节点 TP 不能满足 TTFT SLO
- 解释三种方案的通信代价对比（C16–C17）：
  - 跨节点 TP：每层 2 次 RDMA all-reduce
  - 跨节点 SP（Ring/Striped Attention）：每层至少 1 次跨节点通信；仍比单节点 TP 差
  - 静态分两组（仅 TP、TP+SP）：请求按需路由；动态调整复杂
- 解释 CPP（chunked pipeline parallelism）的优势（C18–C19）：
  - 把每请求输入切成块（≤prefill_chunk），不同节点流水线同时处理不同块；只在阶段边界通信，可与计算重叠
  - 同时适合短/长上下文，无需频繁动态调整
- 解释 layer-wise prefill（C20、F2）：
  - KVCache load/store 异步化与计算重叠（before 每层 attention 等该层 KVCache 载入完成、触发下一层载入；attention 完成后立即启动该层异步 store）
  - prefill 执行时间 ≈ max(载入时间, 标准 prefill 时间) → 调度只需考虑 DRAM 容量而无需考虑 VRAM（只要装得下单请求）

**对应范围**：C16–C20；F1、F2

**正文要点**：用一张对比表展示三种方案的通信代价；用一张时序图（CPP 流水线 vs SP）展示计算/通信重叠方式。

**表达材料**：
- 一张对照表列出 TP 跨节点 / SP / CPP 的通信次数与 MFU 代价
- 一张 SVG 时序图展示 CPP 的流水线计算/通信重叠（用 `dg-stack` 或内联 SVG）
- 引用 [Model parallelism](../../wiki/model-parallelism/index.html)（TP/PP 通信代价）
- 引用 [Chunked Prefill](../../wiki/chunked-prefill/index.html)（CPP 与 chunked prefill 合流）
- 引用 [PCP 与 DCP](../../wiki/pcp-dcp/index.html)（与 SP 对比）

**章末「本章问题」**（h3）：2 题，给出解答折叠块。

---

### 章节 5：过载场景与早拒绝

**章节问题**：过载时 Mooncake 怎么决定拒收哪些请求？为什么朴素早拒绝会引发负载震荡？

**完成答案要点**：
- 解释 goodput（C27）：只计完整完成的请求；否则之前消耗的算力浪费
- 解释 Early Rejection（C28）：在 prefill 开始前就按 prefill 与 decode 池中较高负载决定是否接受；避免 decode 拒收导致 prefill 算力浪费
- 解释朴素早拒绝的震荡（C25、Figure 10a）：
  - 时滞：按当前 decode 负载决策，但 decode 负载在 prefill 完成时才显现
  - 4 阶段循环：Stage1 接受（prefill 满载）→ Stage2 decode 满载拒收 → Stage3 prefill 空载 → Stage4 又接受 → 重复
  - 结果：prefill/decode 负载反相震荡，集群利用率低
- 解释基于预测的早拒绝（C26）：
  - 请求级（预测每请求输出长度）难度高、过载场景下更难
  - 系统级（估计整 batch 数或 TBT 状态）适合过载场景，要求低
  - Mooncake 采用系统级：假设每请求 decode 时长均匀 $t_d$，在 $t$ 时刻预测 decode 负载 = avg(TBT ratio to $l_{\text{tbt}}$)
- 实验支撑：8P+8D 2× 重放 23,000 请求，拒绝数 Baseline 4183 → ER 3771 → ERP 3589（N9）
- 端到端结果：模拟数据 50%–525%、真实工作负载多 ~75%（N11–N13）；vLLM 在长上下文场景 TBT 满足率仅 57% 而 Mooncake ~100%（N13）

**对应范围**：C25–C28、C30–C32；N9、N11–N13

**正文要点**：用 4 阶段震荡的顺序流程图展示朴素早拒绝的问题；用一张系统级预测示意展示 ERP 的逻辑。

**表达材料**：
- 一张顺序流程图（`dg-flow`）展示朴素早拒绝 4 阶段循环
- 一张小表格比较 Baseline / ER / ERP 的拒绝数（N9）
- 引用 Chapter 1（goodput 与 SLO 定义）

**章末「本章问题」**（h3）：2 题，给出解答折叠块。

---

### 章节 6（来源与范围说明）

按 style-guide 固定命名，使用 h3 子标题：
- 论断与来源（C）
- 公式与来源（F）
- 外部数字与实验条件（N）
- 构造示例
- 辅助解释与类比边界
- 简化条件及其限制

内容直接从 evidence.md 抽取（C/F/N 列表 + 来源定位 + 适用条件 + 置信）。

## 3. 讲解顺序

按知识依赖排列：问题与动机（章节 1）→ 架构与流程（章节 2）→ 调度算法（章节 3）→ prefill 池实现（章节 4）→ 过载与早拒绝（章节 5）。每一章只回答一个主要职责；章间过渡先总结本章已得结论再指出下一步问题。

## 4. 表达材料职责

| 材料类型 | 表达目标 | 关联章节 |
|---|---|---|
| 顺序流程图 `dg-flow` | "单 GPU 集群同时承担两阶段导致 TBT 抬高"的现象 | 章节 1 |
| 对照表 | prefill vs decode 的资源偏好与 SLO | 章节 1 |
| 分层结构图 `dg-stack` | Mooncake 五大组件的层级关系 | 章节 2 |
| 顺序流程图 `dg-flow` | 请求四步流程 | 章节 2 |
| 对照表 | 四步流程每步的输入/输出/传输组件 | 章节 2 |
| 完整 Algorithm 1 伪代码 | cache-aware 调度算法流程 | 章节 3 |
| 对照表 | 贯穿示例下四种策略的 TTFT 估计 | 章节 3 |
| 对照表 | TP 跨节点 / SP / CPP 通信代价与 MFU | 章节 4 |
| 内联 SVG 时序图 | CPP 流水线计算/通信重叠 | 章节 4 |
| 顺序流程图 `dg-flow` | 朴素早拒绝 4 阶段震荡 | 章节 5 |
| 对照表 | Baseline / ER / ERP 拒绝数 | 章节 5 |

## 5. 正文与折叠块分工

### 必须放正文

- 两阶段计算特性差异与 PD 分离动机（章节 1）
- Mooncake 五大组件与四步流程（章节 2）
- Algorithm 1 完整流程与贯穿示例手算（章节 3）
- CPP、layer-wise prefill 的核心机制（章节 4）
- Early Rejection、负载震荡、系统级预测（章节 5）

### 可放折叠块（补充）

- 算法 1 的输入/状态/输出逐行伪代码展开（用 `details` 包 `text`，标 A6 风格）
- 预 fill_chunk、队列时间、传输速度的构造模型定义（标记"构造示例"）
- 时间模型逐项代入的逐步化简（标记"构造示例"）
- 关于 trace "input-output ratio 720" 与算术比不一致的提示
- CPP 在短上下文下的微观时序与气泡占比
- layer-wise prefill 在短请求下不值得 overlap 的边界说明

折叠块全部收起时，正文必须仍能完整回答全部学习目标。

## 6. 范围与证据约束

- 只使用 scope.md 中纳入范围的内容
- 不引入新学习目标、新前置知识、新机制描述
- 所有 C/F/N 项均与 evidence.md 一致
- 数字 N8（N8 median TTFT）已通过抽取 PDF 第 11 页为图像核对
- 数字 N2 引用 trace 均值时标注原文 ratio 与算术比的差异
- 涉及 KVCache 大小（320 KiB/token）的估算使用 LLaMA2-70B 配置（80 层 / 8 KV 头 / 128 head dim / fp16），标注「以 LLaMA2-70B 配置估算」

## 7. 章节 h2 编号与标题

按 style-guide 编号 `1. 标题` 至 `5. 标题`，加末尾「来源与范围说明」（不编号）：

1. 为什么把 prefill 与 decode 拆开
2. Mooncake 架构总览：组件、KVCache 池与四步工作流
3. KVCache-centric 调度算法
4. 长上下文 prefill 的多节点与传输重叠
5. 过载场景下的早拒绝与预测

每章 h3 子标题用 `1.1 标题` / `1.2 标题` 形式（如 `1.1 两阶段的计算特性与 SLO`）；来源章节内的 h3 使用固定命名「论断与来源（C）」「公式与来源（F）」「外部数字与实验条件（N）」「构造示例」「辅助解释与类比边界」「简化条件及其限制」。