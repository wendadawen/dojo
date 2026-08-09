# outline.md：FlashKDA 与 KDA Context Parallelism 教学大纲

## 4.1 页面开头

**钩子问题**：KDA 把 1M token 的 KV cache 压成了固定 32KB 的递归状态——省了显存，但这个状态必须一步一步串行更新。GPU 上有 132 块 SM 想同时干活，串行依赖却让它们排队等。一个长 prefill 进来，大部分 SM 闲着，只有少数在推状态。怎么让 132 块 SM 都动起来？

**一句话解释**：本文讲解 K3 报告 §5.1（FlashKDA、设备内 CP、KCP）+ §5.4.2（解码 kernel）四套方案，分别针对训练/prefill、长 prefill、跨设备、投机解码四个执行 regime，化解 KDA 串行状态与 GPU 并行偏好的冲突。

**要解决的具体问题**：（1）训练和 prefill 里 chunk 间传状态时 SM 空转；（2）纯 TP 下长序列每个 rank 只持少量 head、SM 闲；（3）跨设备时线性注意力的"求和"对 KDA 不成立；（4）投机解码拒绝 draft token 时状态已越过最后接受 token。

**学习承诺**：读完能回答 Q1–Q5（见 learning-goals 组件）。

**首个具体场景**：一条长序列 prefill，在纯 TP 部署下大部分 SM 空闲——这是 S1 的入口。

**与第一章过渡**：S1 先把"冲突根源 + 四 regime 瓶颈"摆出来，后续四章各解一个 regime。

## 4.2 章节设计

### S1：串行状态 vs GPU 并行——为什么 KDA 在四个 regime 瓶颈不同

- **主要教学问题**：KDA 的串行状态更新为什么与 GPU 并行偏好冲突？为什么这个冲突在不同 regime 表现不同？
- **对应范围**：Q1；C1。
- **正文要点**：
  - 复用 KDA 页 Eq.1：$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，$S_t$ 依赖 $S_{t-1}$，串行。
  - GPU 偏好宽而均匀的并行（引用 gpu-execution-model 页：132 块 SM、CTA、warp）。串行依赖让 SM 排队等。
  - chunkwise 形式（引用 KDA 页）是"chunk 内并行 + chunk 间串行"——比纯串行快，但 chunk 间传 $S$ 时仍有空隙。
  - 四 regime 瓶颈定位（用对照表）：(a) 训练/prefill：intra-chunk 计算与 cross-chunk 状态传播交替、SM 空转；(b) 长 prefill + 纯 TP：每 rank 持少量 head、SM 闲；(c) 跨设备：线性注意力可求和、KDA 不可；(d) 解码：瓶颈从并行转为管理原地更新的状态。
- **讲解材料及职责**：
  - 公式 Eq.1（F1）：表达串行依赖根源。
  - 对照表：并排四 regime 的瓶颈定位，服务"为什么不同"。
  - 图示：chunk 间传状态的空隙（ASCII）。
- **前置知识安排**：首次依赖 KDA Eq.1 时链接 [KDA 页](../../wiki/kda/index.html)；首次依赖 SM/CTA 时链接 [GPU 执行模型页](../../wiki/gpu-execution-model/index.html)。
- **完成检查**：（1）说出冲突根源（$S_t$ 依赖 $S_{t-1}$）与 GPU 偏好的对立；（2）对照表里指出四 regime 各自瓶颈；（3）说明 chunkwise 形式为何"比纯串行快但仍有空隙"。
- **过渡**：四 regime 定位完，下面四章各解一个。S2 先解训练/prefill。

### S2：FlashKDA——把 chunk 内计算与 chunk 间状态传播重叠

- **主要教学问题**：FlashKDA 如何重叠 intra-chunk 计算与 cross-chunk 状态传播？
- **对应范围**：Q2；C2。
- **正文要点**：
  - 复述 chunkwise 形式的两阶段（引用 KDA 页）：chunk 内并行 matmul（intra-chunk）+ chunk 间递归传 $S$（cross-chunk）。
  - 朴素实现：两阶段交替——算 chunk $t$ 的 intra 时，cross-chunk 的 $S[t]\to S[t+1]$ 没法开始；传 $S$ 时，intra 的 SM 空转。
  - FlashKDA 的重叠：把工作分解成 **token-parallel stages**（chunk 内的 token 维度并行）和 **head-parallel recurrence**（chunk 间的 head 维度递归），各自独立调度。让 chunk $t$ 的 intra-chunk 计算与 chunk $t-1\to t$ 的状态传播同时进行。
  - 工程定位：CUTLASS-based；服务训练和 prefill；flash-linear-attention 的后端，自动分发。
- **讲解材料及职责**：
  - 图示（ASCII）：朴素两阶段交替 vs FlashKDA 重叠的时间线对比。
  - 对照表：token-parallel vs head-parallel 的职责分工。
- **前置知识安排**：chunkwise 形式链接 KDA 页；CUTLASS/Tensor Core 不展开（gpu-execution-model 页已讲 SM/CTA 层面）。
- **完成检查**：（1）说出朴素实现里 SM 空转发生在哪个阶段；（2）说出 token-parallel 与 head-parallel 各自并行什么维度；（3）说明 FlashKDA 服务的两个场景（训练、prefill）和它的后端宿主。
- **过渡**：FlashKDA 解决了 chunk 间的空隙。但如果序列长到一个 rank 的 head 都不够填满 SM 呢？S3 解设备内 CP。

### S3：设备内 context parallelism——单 rank 的 SM 级切序列

- **主要教学问题**：设备内 CP 如何在单 rank 上并行一条长序列，且不产生跨设备通信？
- **对应范围**：Q3；C3。
- **正文要点**：
  - TP 的局限：TP 把 head 切到不同设备，但每个设备上序列长度不变——递归步数没缩短。纯 TP 下长 prefill，每个 rank 只持少量 head，132 块 SM 大部分闲。
  - 关键观察：KDA 的状态转移 $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$ 是**线性**的——一段 token 的转移效果可以分解成"作用于入状态的部分"+"从零生成的部分"，与入状态本身无关地预先算好，之后精确合并。这个性质是 S4 KCP 的前置，也是 S3 设备内 CP 的基础。
  - SM 级 CP planner：把序列切到单 rank 的多块 SM，每块 SM 算一段的 segment transition（独立于入状态），再合并恢复每段的精确初始状态。
  - 与 KCP 的区别（点名）：完全设备内，无跨设备通信。
- **讲解材料及职责**：
  - 图示（ASCII）：TP 切 head vs 设备内 CP 切序列的对比。
  - 图示：段转移的"独立计算 + 合并"流程。
- **前置知识安排**：TP 概念最小解释（切 head）；SM/CTA 链接 gpu-execution-model 页。
- **完成检查**：（1）说出 TP 为何不缩短递归；（2）说出"段转移可独立于入状态计算"这个关键观察依赖 KDA 递归的什么性质（线性性）；（3）说出设备内 CP 与 KCP 在通信上的区别。
- **过渡**：S3 在单 rank 内切序列。如果序列长到一台设备装不下，要跨 device rank 呢？S4 解 KCP。

### S4：KCP——为什么不能直接求和，以及 M+S̃ 分解

- **主要教学问题**：KCP 为什么不能像 vanilla 线性注意力那样直接求和本地状态？$M+\tilde S$ 分解如何让通信量与序列长度无关？
- **对应范围**：Q4；C4；F2（Eq.17）。
- **正文要点**：
  - vanilla 线性注意力的 CP（对照）：递归 $s_i = s_{i-1} + \phi(k_i)v_i^\top$ 是纯加性。各 rank 算本地从 $s=0$ 起的状态，求和前序 rank 即可恢复入状态。引用线性注意力页。
  - KDA 不能直接求和：$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，$M_t$ 作用于入状态后再加写入。本地段的效果依赖进入该段的状态，不能从 $S=0$ 单独确定。
  - KCP 分解（Eq.17）：每段效果 = "作用于入状态的累积转移 $M_{t\leftarrow 1}$"（本地 token 的 $M$ 连乘）+ "从零生成的状态 $\tilde S$"（同一递归从 $S=0$ 起）。两者都可在入状态可用前本地算出。
  - 一次 all-gather + prefix scan：各 rank 本地算 $(M累积, \tilde S)$，一次 all-gather 交换；rank $i+1$ 用前序 rank 的片段从 $S=0$ 起 prefix scan 重组入状态。
  - 通信量固定：交换的是固定大小状态（$S\in\mathbb{R}^{d_k\times d_v}$ 及同形状 $M$），与序列长度无关；对照 softmax CP 要交换随序列长度增长的 KV block。
  - 线性计算扩展：计算量随 rank 数线性扩展。
- **讲解材料及职责**：
  - 公式 F3（vanilla 加性递归）+ F1（KDA $M_t$）对照：表达"为何可求和 vs 不可求和"。
  - 公式 F2（Eq.17）：表达 $M+\tilde S$ 分解。配手算小例子（折叠块放完整计算，正文放结论）。
  - 数字例子（贯穿）：2 rank × 2 token 的 KDA 递归，手算验证分解。教学构造，$\alpha=0.5$ 标量、$\beta=1$、单位 k。
  - 图示（ASCII）：all-gather + prefix scan 的数据流。
  - 对照表：vanilla LA CP vs KDA KCP vs softmax CP 的通信量。
- **前置知识安排**：vanilla 线性注意力加性递归链接 [线性注意力页](../../wiki/linear-attention/index.html)；KDA $M_t$ 链接 KDA 页。
- **完成检查**：（1）说出 vanilla 线性注意力可求和而 KDA 不可的根因（$M_t$ 作用于入状态）；（2）写出 $M+\tilde S$ 分解的两项各自含义；（3）说出 KCP 通信量为何与序列长度无关；（4）手算例子中验证 $S_T^{[2]} = \tilde S_2 + M累积_2 \cdot S_T^{[1]}$ 与顺序计算一致。
- **过渡**：训练/prefill/跨设备都解了。最后是解码——S5。

### S5：KDA 解码——投影输入缓存与状态重建

- **主要教学问题**：解码如何处理投机解码拒绝 draft token 时的状态回滚，又不让状态流量爆炸？
- **对应范围**：Q5；C5。
- **正文要点**：
  - 解码瓶颈转移：训练/prefill 的瓶颈是"利用并行"；解码的瓶颈是"管理原地更新的状态"——每步原地更新 $S$。
  - MTP 投机解码的回滚问题：验证拒绝部分 draft token 时，状态已越过最后接受 token，无法平凡回滚。最小解释 MTP（draft/verify/accept/reject/bonus）。
  - 朴素方案与代价：为每个 draft 位置存状态快照能回滚，但状态流量在大 batch 下占主导。
  - 关键观察：任意接受前缀后的状态由 draft token 的**投影输入**完全决定（KDA 更新的确定性），而投影输入比状态小得多。
  - 方案：只缓存投影输入，片上重建接受 token 状态，写回已验证和 bonus token 状态。与 ReplaySSM 并发提出。
  - 工程定位：投影缓存不离开 decode 阶段，前缀缓存和 PD 分离与非投机服务同载荷。融合 kernel 覆盖短卷积、输入归一化、门控、KDA 递归、输出归一化。验证延迟随验证 token 数亚线性增长。
- **讲解材料及职责**：
  - 图示（ASCII）：状态快照（大流量）vs 投影输入缓存（小流量）的对比。
  - 图示：接受前缀 → 投影输入 → 片上重建状态的流程。
- **前置知识安排**：MTP 投机解码最小解释（不展开完整算法）。
- **完成检查**：（1）说出解码瓶颈与训练/prefill 的不同；（2）说出状态快照方案的代价；（3）说出投影输入比状态小这个关键观察；（4）说出投影缓存不离开 decode 阶段对前缀缓存/PD 分离的意义。
- **过渡**：四 regime 解完。文末来源与教学说明。

### 文末：来源与教学说明（必有）

- 核心论断与来源（C1–C5 + K3 报告定位）
- 核心公式与来源（F1–F3）
- 教学示例（贯穿的 2×2 KDA 递归构造）
- 教学解释与类比边界
- 教学简化及其限制
- 无外部数字小节（本页无外部性能数字）

## 4.3 讲解顺序

S1 先讲冲突根源（为什么需要）与四 regime 定位（是什么的框架）。S2–S5 各解一个 regime，顺序按 K3 报告 §5.1.1→§5.1.2→§5.4.2 的逻辑：先训练/prefill（最常用），再长 prefill 的设备内扩展，再跨设备，最后解码。S4 是技术含量最高、放中间偏后，让前置的"段转移可独立于入状态"（S3）为它铺路。一次只引入一个新变量：S1 引入"冲突"，S2 引入"重叠"，S3 引入"段转移可独立计算"，S4 引入"$M+\tilde S$ 分解"，S5 引入"投影输入 vs 状态"。

## 4.4 贯穿例子

**贯穿问题**：一条长 KDA 序列如何在不同执行 regime 下高效跑起来。

**贯穿数字例子**（服务 S4 KCP，也可在 S1/S3 复用）：2 rank × 2 token 的 KDA 递归。
- 配置：$d_k=d_v=2$，教学构造 $\alpha=0.5$（标量，$\mathrm{Diag}(\alpha)=0.5I$）、$\beta=1$、单位向量 $k$。
  - Rank 1（token 1,2）：$k_1=(1,0), v_1=(1,2)$；$k_2=(0,1), v_2=(3,4)$。
  - Rank 2（token 3,4）：$k_3=(1,0), v_3=(5,6)$；$k_4=(0,1), v_4=(7,8)$。
- $M_t = I - 0.5 k_t k_t^\top$。
- 顺序计算的 ground truth：$S_4 = [[5.5,7],[8.5,10]]$。
- KCP 分解验证：$S_T^{[2]} = \tilde S_2 + M累积_2 \cdot S_T^{[1]}$，$S_T^{[1]} = \tilde S_1$（因 $S_0=0$），算出同值。

数字足够小可手算，$M累积 = 0.5I$ 干净。构造目的：让"不能直接求和"和"$M+\tilde S$ 分解"在 2×2 矩阵上手算可见。不代表真实 K3 数值（K3 的 $\alpha$ 由 scaled sigmoid 产出、$d_k=128$）。

## 4.5 讲解材料职责

- **公式 F1（Eq.1）**：表达串行依赖根源（S1）与 $M_t$ 的存在（S4）。
- **公式 F3（vanilla 加性递归）**：对照 F1，表达"为何可求和"（S4）。
- **公式 F2（Eq.17）**：表达 $M+\tilde S$ 分解（S4）。配手算折叠块。
- **数字例子**：2×2 KDA 递归，展示 $M+\tilde S$ 分解可手算验证（S4 折叠块）。
- **图示（ASCII）**：S1 chunk 间空隙、S2 朴素 vs 重叠时间线、S3 TP vs 设备内 CP、S4 all-gather+prefix scan、S5 状态快照 vs 投影缓存。每个图示服务一个具体机制。
- **对照表**：S1 四 regime 瓶颈、S2 token/head 并行分工、S4 三种 CP 通信量。
- **无伪代码/可运行代码**：四套方案都是 kernel/系统级机制，伪代码会与图示重复；可运行代码需要 CUTLASS/分布式框架，超出教学职责。手算例子已足够验证 KCP 分解。

## 4.6 正文与折叠块分工

**正文必有**：Eq.1 与 $M_t$ 的引用结论；四 regime 瓶颈定位；FlashKDA 的重叠对象；设备内 CP 的"段转移可独立计算"；KCP 的 $M+\tilde S$ 分解与一次 all-gather；解码的投影输入缓存方案；贯穿例子的关键推进（$M累积=0.5I$、$S_4$ ground truth、分解后重组一致）。

**折叠块**：S4 的完整手算（ground truth 4 步 + 分解重组 2 步，正文放结论）；S4 的 Eq.17 完整展开（含跨 rank 求和形式）。

折叠块全收起时，正文用"分解后重组 = ground truth"一句结论 + 关键数字支撑 S4 的 Q4 闭环。

## 4.7 范围与证据约束

大纲只用 scope.md 已纳入内容。无新增学习目标、无新增范围外内容、无改变概念边界的论断。所有论断来自 K3 报告 §5.1.1/§5.1.2/§5.4.2，evidence.md 已逐条定位。
