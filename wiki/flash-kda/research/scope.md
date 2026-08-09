# scope.md：FlashKDA 与 KDA Context Parallelism 内容范围

## 1.1 概念含义

- **概念名称**：FlashKDA 与 KDA Context Parallelism
- **英文名称**：FlashKDA and KDA Context Parallelism
- **常见缩写**：FlashKDA（chunkwise kernel）、KCP（KDA Context Parallelism）
- **一句话定义**：把 KDA 的串行递归状态更新改造成能在 GPU 上高效运行的四套执行方案，分别针对训练/prefill、长 prefill 的设备内并行、跨设备并行、投机解码。
- **正式定义**：KDA 用固定大小的递归状态 $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$（$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$）替代 softmax 的 KV cache，但状态更新是串行的（$S_t$ 依赖 $S_{t-1}$），与 GPU 偏好宽而均匀的并行相冲突；本文讲解 K3 报告 §5.1.1（FlashKDA、设备内 CP）+ §5.1.2（KCP）+ §5.4.2（KDA 解码）四套方案如何在不同执行 regime 下化解这个冲突。来源：K3 报告 §5.1、§5.4.2。
- **本文采用的语境**：把"FlashKDA 与 KDA Context Parallelism"理解为一个统一的工程主题——"让 KDA 的递归状态在 GPU 上高效运行"，覆盖四个执行 regime。不是单指某一个 kernel。
- **包括什么**：
  - KDA 串行状态与 GPU 并行的根本冲突，以及它在四个 regime 下表现成不同瓶颈（§5.1.1 开头）
  - FlashKDA：chunkwise kernel 的 intra-chunk 计算与 cross-chunk 状态传播重叠（§5.1.1）
  - 设备内 context parallelism：SM 级 CP planner，序列切到单 rank 的 SMs，无跨设备通信（§5.1.1）
  - KCP：跨设备，把每段效果分解为"作用于入状态的累积转移 $M$"+"从零生成的状态 $\tilde S$"，prefix scan + 一次 all-gather，通信量固定（§5.1.2，Eq.17）
  - KDA 解码：投机解码拒绝 draft token 时的状态回滚问题，缓存投影输入而非状态（§5.4.2）
- **不包括什么**：
  - KDA 的递归公式本身、channel-wise forget gate、lower-bounded decay、full-rank output gate 的机制——已由 [KDA 概念页](../../wiki/kda/index.html)讲解，本页只引用 Eq.1 与 $M_t$ 的结论
  - softmax 注意力的 context parallelism（Ring Attention 等）——机制不同（要交换 KV block），只在对照时点名
  - Block AttnRes、Stable LatentMoE 的 kernel 优化——同属 §5.4.2 但与 KDA 并行无关
  - 前缀缓存、fleet 调度、PD 分离等服务层——§5.4.1/§5.4.3 内容
  - CUTLASS / Triton / WarpDecode 的具体编程模型细节——超出本页教学职责
- **相邻概念**：
  - **softmax 的 context parallelism**（Ring Attention / CP）：要交换随序列长度增长的 KV block；KCP 只交换固定大小的状态。易混淆，本页在 KCP 章节对照。
  - **vanilla 线性注意力的 context parallelism**：因递归是纯加性 $s_i = s_{i-1} + \phi(k_i)v_i^\top$，各 rank 算本地状态后直接求和即可；KDA 的 delta rule 有 token-dependent $M_t$，不能简单求和。易混淆，本页在 KCP 章节对照。
  - **Tensor Parallelism（TP）**：把 head 切到不同设备；不缩短序列方向的递归。易混淆，本页在设备内 CP 章节点名区别。

## 1.2 学习目标

### Q1：KDA 的串行状态更新为什么与 GPU 并行偏好冲突，为什么这个冲突在不同执行 regime 表现成不同瓶颈？

- **完成答案**：读者应能说出（a）GPU 偏好宽而均匀的并行，KDA 的 $S_t$ 依赖 $S_{t-1}$ 是串行依赖；（b）在训练/prefill，串行依赖表现为 intra-chunk 并行与 cross-chunk 状态传播交替、SM 空转；（c）在长 prefill + 纯 TP，表现为每个 rank 只持少量 head、大部分 SM 空闲；（d）在跨设备，表现为线性注意力的"求和"对 KDA 不成立；（e）在解码，瓶颈从"利用并行"转为"管理原地更新的状态"。
- **为什么是核心目标**：不理解这一点，后面四套方案都失去动机——读者不知道每个方案在化解什么。
- **依赖内容**：KDA Eq.1（$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$）、GPU 执行模型（SM、CTA、并行偏好）。

### Q2：FlashKDA 如何把 intra-chunk 计算与 cross-chunk 状态传播重叠起来？

- **完成答案**：读者应能说出（a）chunkwise 形式是 chunk 内并行、chunk 间串行传 $S$；（b）朴素实现里两者交替、SM 在传播时空转；（c）FlashKDA 把工作分解成 token-parallel stages 和 head-parallel recurrence，各自独立调度，让 intra-chunk 计算与 cross-chunk 状态传播重叠；（d）它是 CUTLASS-based，同时服务训练和 prefill，是 flash-linear-attention 的一个后端。
- **为什么是核心目标**：这是训练和 prefill 的主力 kernel，不理解重叠机制就无法理解 KDA 为何能训练。
- **依赖内容**：KDA chunkwise 形式（chunk 内并行 + chunk 间递归）、GPU 执行模型（SM 调度、warp）。

### Q3：设备内 context parallelism 如何在单 rank 上并行一条长序列，且不产生跨设备通信？

- **完成答案**：读者应能说出（a）TP 只切 head 不缩短递归，纯 TP 下长 prefill 每个 rank 只持少量 head、SM 空闲；（b）关键观察：每段的 state transition 可独立于入状态计算、之后再精确合并；（c）SM 级 CP planner 把序列切到单 rank 的 SMs，并行算 segment transition 再合并恢复每段的精确初始状态；（d）与 KCP 的区别：完全设备内、无跨设备通信。
- **为什么是核心目标**：这是长 prefill 的关键方案，且它的"段转移可独立于入状态计算"的思想是 KCP 的前置。
- **依赖内容**：GPU 执行模型（SM、CTA）、KDA 状态转移的可组合性。

### Q4：KCP 为什么不能像 vanilla 线性注意力那样直接求和本地状态，它的 $M+\tilde S$ 分解如何让通信量与序列长度无关？

- **完成答案**：读者应能说出（a）vanilla 线性注意力递归是纯加性 $s_i = s_{i-1} + \phi(k_i)v_i^\top$，各 rank 算本地 $s$ 后求和即可恢复入状态；（b）KDA 的 delta rule 有 token-dependent $M_t = I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$，作用于入状态后再加写入，所以本地段的效果依赖进入该段的状态，不能从 $S=0$ 单独确定；（c）KCP 把每段效果分解成"作用于入状态的累积转移 $M$"+"从零生成的状态 $\tilde S$"两个本地可算的量；（d）各 rank 一次 all-gather 交换这两个片段，prefix scan 重组入状态；（e）交换的是固定大小的状态，通信量与序列长度无关。
- **为什么是核心目标**：这是跨设备并行的核心机制，也是本页技术含量最高的部分；Eq.17 的分解是 K3 的原创贡献。
- **依赖内容**：KDA Eq.1 与 $M_t$、线性注意力的加性递归、prefix scan 概念。

### Q5：KDA 解码如何处理投机解码拒绝 draft token 时的状态回滚，又不让状态流量爆炸？

- **完成答案**：读者应能说出（a）解码每步原地更新状态，MTP 投机解码若拒绝部分 draft token，状态已越过最后接受 token、无法平凡回滚；（b）为每个 draft 位置存状态快照能支持回滚，但状态流量在大 batch 下占主导；（c）关键观察：任意接受前缀后的状态由 draft token 的投影输入完全决定，而投影输入比状态小得多；（d）方案：只缓存投影输入，在片上重建接受 token 的状态，写回已验证和 bonus token 的状态；与 ReplaySSM 并发提出；（e）投影缓存不离开 decode 阶段，前缀缓存和 PD 分离与非投机服务同载荷。
- **为什么是核心目标**：解码是 KDA 在线服务的最后一道关，投机解码回滚是它独有的难题。
- **依赖内容**：KDA 单步更新、MTP 投机解码概念（验证/接受/拒绝/bonus token）。

## 1.3 内容分级

### 核心内容

- **C1**：KDA 串行状态与 GPU 并行的冲突，四 regime 不同瓶颈（服务 Q1）。必须讲清：冲突根源是 $S_t$ 依赖 $S_{t-1}$；四 regime 瓶颈各自定位。
- **C2**：FlashKDA 的 token-parallel + head-parallel 分解与重叠（服务 Q2）。必须讲清：chunkwise 形式的两阶段、重叠的对象、CUTLASS-based、flash-linear-attention 后端。
- **C3**：设备内 CP 的"段转移可独立于入状态计算"（服务 Q3）。必须讲清：TP 不缩短递归、SM 级切分、无跨设备通信。
- **C4**：KCP 的 $M+\tilde S$ 分解与 prefix scan（服务 Q4）。必须讲清：vanilla 线性注意力可求和、KDA 因 $M_t$ 不可求和、分解的两个量、一次 all-gather、通信量固定。需配 Eq.17 的手算小例子。
- **C5**：KDA 解码的投影输入缓存方案（服务 Q5）。必须讲清：原地更新 vs 回滚、状态快照的代价、投影输入更小、片上重建、不离开 decode 阶段。

### 辅助内容

- chunkwise 形式的回顾（inter/intra-chunk 两项）——服务 C2，消除"两阶段交替"的理解障碍，引用 KDA 页不重讲。
- softmax CP 与 KCP 的通信量对照——服务 C4，澄清"通信量固定"的含义。
- TP 不缩短递归的说明——服务 C3，澄清"为何纯 TP 下 SM 空闲"。

### 扩展内容

- CUTLASS / TMA / warp specialization 的具体编程模型——纳入或排除：**排除**，属 GPU 编程细节，由 gpu-execution-model 页已讲 SM/CTA/warp 层面足够。
- MTP 投机解码的完整机制——纳入或排除：**排除**，属独立概念，本页只用到验证/接受/拒绝/bonus 词汇并最小解释。
- flash-linear-attention 框架的工程细节——**排除**，只点名它是 FlashKDA 的后端宿主。

## 1.4 前置知识映射

| 前置概念 | 被哪些目标依赖 | 概念页链接 | 递归层级 |
|---|---|---|---|
| KDA（递归 Eq.1、$M_t$、chunkwise 形式） | Q1/Q2/Q4/Q5 | [wiki/kda/index.html](../../wiki/kda/index.html) 已有 | 0 |
| GPU 执行模型（SM、CTA、warp、Tensor Core、并行偏好） | Q1/Q2/Q3 | [wiki/gpu-execution-model/index.html](../../wiki/gpu-execution-model/index.html) 已有 | 0 |
| 线性注意力（加性递归 $s_i = s_{i-1}+\phi(k_i)v_i^\top$、固定大小状态） | Q1/Q4 | [wiki/linear-attention/index.html](../../wiki/linear-attention/index.html) 已有 | 0 |

三份前置页均已存在，本页直接引用结论，不内联重讲。delta-rule 概念页也已有，但 KDA 页已覆盖 delta rule 的擦写几何，本页通过 KDA 页间接引用，不单独链接。

## 1.5 明确不展开的内容

- **CUTLASS 模板编程、TMA 指令、warp specialization 的代码层细节**：与"理解四套方案机制"无关，属 GPU 编程实现。gpu-execution-model 页已讲 SM/CTA/warp 层面足够支撑本页。
- **MTP 投机解码的完整算法（draft 模型、接受策略、bonus 生成）**：属独立服务概念，本页只用到"验证拒绝部分 draft token 时状态已越过最后接受 token"这一事实。
- **flash-linear-attention 框架的注册/分发机制**：只点名它是 FlashKDA 的后端宿主，工程细节不展开。
- **具体性能数字（加速比、带宽利用率）**：K3 报告未在 §5.1/§5.4.2 给出可定位的基准表，不引入未经证实的外部数字。
- **前缀缓存、PD 分离、fleet 调度**：§5.4.1/§5.4.3，与 KDA 并行无关。
- **Block AttnRes、Stable LatentMoE 的 kernel 优化**：同属 §5.4.2 但与 KDA 并行正交。

## 1.6 常见误解和适用边界

### 误解

- **M1**："FlashKDA 就是把 chunkwise 形式实现一遍" —— 正确方向：chunkwise 形式 chunk 内并行、chunk 间串行，朴素实现两者交替会让 SM 在传播时空转；FlashKDA 的关键是把 intra-chunk 计算与 cross-chunk 状态传播**重叠**，不是简单实现。形成原因：忽略"重叠"这个动词。影响 Q2。
- **M2**："设备内 CP 和 KCP 是同一回事，都是跨设备" —— 正确方向：设备内 CP 完全在单 rank 的 SMs 上切序列、无跨设备通信；KCP 是跨 device rank、需要一次 all-gather。形成原因：都叫"context parallelism"。影响 Q3/Q4。
- **M3**："KCP 只是把线性注意力的 CP 搬到 KDA" —— 正确方向：vanilla 线性注意力递归是纯加性，各 rank 本地状态可直接求和；KDA 的 delta rule 有 token-dependent $M_t$ 作用于入状态，本地段效果依赖入状态，不能直接求和——必须分解成 $M+\tilde S$。形成原因：忽略 $M_t$ 的作用。影响 Q4。
- **M4**："KDA 解码的回滚就是存状态快照" —— 正确方向：状态快照能支持回滚但流量在大 batch 下占主导；方案是缓存更小的投影输入、片上重建。形成原因：忽略"状态比投影输入大"这个量级差。影响 Q5。
- **M5**："KCP 的通信量随序列长度增长" —— 正确方向：KCP 只 all-gather 固定大小的状态片段（$M$ 和 $\tilde S$），与序列长度无关；softmax CP 要交换随序列长度增长的 KV block。形成原因：把 KCP 与 softmax CP 混淆。影响 Q4。

### 适用边界

- **FlashKDA 的重叠收益前提**：chunkwise 形式成立（chunk 内并行、chunk 间递归）。若递归本身不可分块（如某些非线性 RNN），重叠无从谈起。
- **设备内 CP 的前提**：段转移可独立于入状态计算并可精确合并。这是 KDA 状态转移 $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$ 的线性性赋予的——$M_t$ 和加性项可分离。若递归非线性使段转移不可分离，设备内 CP 不成立。
- **KCP 的固定通信前提**：交换的是固定大小的状态 $S \in \mathbb{R}^{d_k \times d_v}$（及同形状的 $M$ 累积），与序列长度无关。若状态本身随序列增长（如 softmax 的 KV cache），通信量不再固定。
- **解码投影缓存的前提**：接受前缀后的状态由 draft token 的投影输入完全决定。这依赖 KDA 更新的确定性——给定相同输入序列，状态唯一。若更新有随机性，重建不成立。
- **本页不证明**：FlashKDA 的具体 tile/warp 调度策略、KCP 的 prefix scan 通信复杂度证明——这些属实现细节或标准并行算法，本页只陈述结论。
