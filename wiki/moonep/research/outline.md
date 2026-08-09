# MoonEP 完美均衡专家并行 · 教学大纲

## 1. 页面开头

### 1.1 钩子问题（30 秒内知道在解决什么）

讲一个最小场景：MoE 训练里，每个 EP rank 上"本地专家被多少 token 选中"完全由 router 决定，而 router 偏好随每个 micro-batch、每一层都在变。结果：有的 rank 收到 10 倍于其他 rank 的 token，有的 rank 几乎空转；通信 buffer 的形状每次都不一样，每层都得让 CPU 跟 GPU 同步一次才能确定本层到底算多少。MoonEP 解决的就是这件事——让每个 rank 每层都恰好收到 $S\times K$ 个 token，把"动态形状"彻底改成"静态形状"。

### 1.2 一句话解释

MoonEP 是一种 MoE 训练的专家并行方案：通过在线规划和迁移"冗余专家"，让每个 EP rank 恰好收到 $S\times K$ 个 token，从而把负载、通信 buffer 形状和每层计算形状都变成静态已知。

### 1.3 学习承诺

读完这一页，你应该能够：
- 用一句话说清传统 EP 为什么让 rank 间负载不均衡、MoonEP 用什么思路消除它（Q1）；
- 说明冗余专家是什么、为什么 $E/R$ 个槽位足以保证可行解存在、为什么这个界基本紧（Q2）；
- 解释完美均衡怎样让通信 buffer 从 $S\times K\times R$ 降到 $S\times K$、消除每层 host 同步，并描述 forward/backward 的规划与归还流程（Q3）；
- 说清 MoonEP 解决什么、不解决什么，与 ECHO/UltraEP/DeepEP 的关键区别（Q4）。

### 1.4 前置知识引用

本页依赖两个前置概念：MoE 的 router/top-k、shared/routed expert、token-expert pair 与专家并行（EP）的 all-to-all dispatch/combine 数据流，见 [MoE 大模型推理与服务基础](../../wiki/moe-serving/index.html)；本页首次使用 host-device 同步动机时一句话点出（详细见 [GPU 执行模型与 kernel 调度](../../wiki/gpu-execution-model/index.html)）。本页不重复推导 EP 的基础机制。

### 1.5 首个具体场景

贯穿例子在 §2 引入：$E=4$ 个专家、$R=2$ 个 EP rank、每 rank 本地 $S=4$ 个 token、每 token 选 $K=1$ 个专家。这个例子的 router 输出会从"极度不均衡"逐步推到"完美均衡"，后续每章复用同一组数字、每次加一个新概念（冗余专家、界、buffer、forward/backward）。

### 1.6 与第一章的过渡

钩子问题已说明动态形状带来三个工程问题（负载不均、buffer 动态、host 同步），第一章先回答"为什么会不均衡、MoonEP 用什么思路消除"。

## 2. 章节设计

### S1：传统 EP 为什么让 rank 间负载不均衡？MoonEP 的核心思路是什么？

- **主要教学问题**：传统 EP 的不均衡从哪里来？MoonEP 用什么新对象消除它？
- **对应范围**：Q1（C1, C2, C3）。
- **正文要点**：
  1. 复述 EP 基础：每 rank 持有 $E/R$ 个本地专家，本地 $S$ 个 token 各选 $K$ 个专家 → 每 rank 发出 $S\times K$ 个 token-expert pair；远端 rank 收到 pair 数 = 该 rank 上专家被选中的总次数。引用 moe-serving 页。
  2. 不均衡来源：router 偏好随 micro-batch 和层变化，rank 收到的 token 数动态不均；报告原文给出"computational imbalance degrades training throughput"和"memory fragmentation"两个后果（C1）。
  3. 引入冗余专家：把某个 home rank 上的专家**临时复制**到另一个 rank 上，让该 rank 在本地处理原本要发出的 token。这是 MoonEP 的核心新对象（C2）。
  4. 完美均衡目标：每 rank 恰好收到 $S\times K$ token（C3）；总量守恒（F4）——$R$ 个 rank 各收 $S\times K$，等于总发出的 $S\times K\times R$。
  5. 用贯穿例子展示"极度不均衡"：$E=4, R=2, S=4, K=1$，router 输出使 rank 0 的 4 个 token 全选 rank 0 的专家、rank 1 的 4 个 token 也全选 rank 0 的专家 → rank 0 收到 8 个 pair（过载）、rank 1 收到 0 个（空转）。
- **讲解材料及职责**：
  - ASCII 图示（component 13）：展示传统 EP 的 dispatch 数据流与 rank 0 过载、rank 1 空转的对照。
  - 对照表（component 14）：列出"传统 EP"与"MoonEP 完美均衡"在每 rank 收到 token 数、形状是否静态两个维度上的对比。
  - 数字例子：贯穿例子的初始 router 输出，标为教学示例。
- **前置知识安排**：EP dispatch/combine 引用 moe-serving 页链接，正文不展开。
- **完成检查**：
  1. 用一句话说明传统 EP 下 rank 收到 token 数由什么决定。
  2. 说明 MoonEP 引入的"冗余专家"是什么、它为什么能消除不均衡。
  3. 在 $E=4,R=2,S=4,K=1$ 的例子里，验证 rank 0 收到 8、rank 1 收到 0 时总量与每 rank 平衡值 $S\times K=4$ 的关系。
- **过渡**：本章已说明思路——用冗余专家把过载 rank 的 token 迁到空转 rank 上。下一章回答：要迁多少个冗余专家才够？

### S2：冗余专家要多少个才够？$E/R$ 上界与"基本紧"

- **主要教学问题**：每 rank 至多需要多少冗余专家才能保证总有可行解？这个界能不能更小？
- **对应范围**：Q2（F1, F2, F3）。
- **正文要点**：
  1. 规划目标的形式化：$M(I)=\min_P\max_r\{m_r(P)\}$（F1），即最小化"任意 rank 上冗余数的最大值"。说明这个目标是 min-max 不是求平均。
  2. 关键引理（F2）：存在规划 $P^*$ 使每 rank 的远端 token 全来自同一 home rank。构造过程（简化复述）：每次取一个 underloaded rank 和一个 overloaded rank，迁移 token 直到 underloaded rank 达到 $S\times K$；每次 fill 让一个 rank 平衡且此后不变，至多 $R-1$ 次终止；每 rank 至多被 fill 一次 → 远端 token 同源。
  3. 上界结论：远端 token 同源 rank $s$ 上至多 $E/R$ 个本地专家（专家均匀分片），故 $m_r(P^*)\le E/R$，即 $M(I)\le E/R$（F2）。
  4. 界的基本紧性（F3）：构造最坏 router 输出 $I^*$——rank 0 的专家收到 0 个 token，其余 $R-1$ 个 rank 的专家均分所有 token。rank 0 必须从远端收 $S\times K$ token，这些 token 至少涉及 $\lceil E(R-1)/R^2\rceil$ 个不同专家。大 $R$ 下 $\lceil E(R-1)/R^2\rceil\approx E/R$，故上界基本紧。
  5. 工程含义：预留 $E/R$ 个冗余槽位即保证总有可行解，训练不中断；不可能有显著小于 $E/R$ 的通用上界。
- **讲解材料及职责**：
  - 数字例子折叠块（component 10）：在 $E=4, R=2$ 的贯穿例子里手算 $E/R=2$，并构造最坏情况验证 rank 1 需要 2 个冗余专家（即界在该例子下达到）。
  - 补充折叠块（component 09）：Theorem 1 构造性证明的完整复述与终止性论证（关键引理）。
  - 公式：F1、F2、F3 出现在正文，符号首次出现处定义。
- **前置知识安排**：无新前置；引理的"远端 token 同源"用一句话解释。
- **完成检查**：
  1. 说明 $M(I)$ 的定义与 min-max 含义。
  2. 说明为什么"远端 token 同源"推出 $m_r\le E/R$。
  3. 说明 Theorem 2 的最坏构造里 rank 0 为什么需要 $\lceil E(R-1)/R^2\rceil$ 个冗余专家。
  4. 说明"基本紧"对工程预留槽位意味着什么。
- **过渡**：本章给出了界的存在性与紧性。下一章回答：完美均衡成立后，通信 buffer、host 同步和 forward/backward 流程怎样随之改变？

### S3：完美均衡怎样让 buffer 降到 $S\times K$、消除 host 同步？forward/backward 怎样跑？

- **主要教学问题**：完美均衡的工程收益具体怎样落地？MoonEP 的 forward/backward 与传统 EP 有什么不同？
- **对应范围**：Q3（C4, C5, C6, C7, C8）。
- **正文要点**：
  1. 通信 buffer：传统 EP（DeepEP）在最坏不均衡下要支持 zero-copy 数据路径需要 $S\times K\times R$ 的 buffer（C7）。MoonEP 因保证每 rank 收 $S\times K$，固定 $S\times K$ buffer 即可。说明 $S\times K\times R$ 是**最坏情况**限定（误解 7.1 最后一条）。
  2. 静态形状免同步：传统 EP 每层 per-expert token 数变化，host 必须在每层与 device 同步拿真实计算形状再 launch kernel（C8）。MoonEP 让所有层形状静态已知，消除每层 MoE host 同步、缓解 host-side kernel-launch overhead。
  3. forward 流程（C4, C6）：planning kernel 从当前 micro-batch、当前层 router 输出在线规划冗余专家；fused permute/unpermute 算子预计算每 token 目的地，直接发到远端 expert-grouped 位置，返回 buffer view 免中间拷贝。说明 GPU kernel 不每步求精确最优（C4 的 online planning 段）。
  4. backward 流程（C5）：冗余专家的梯度暂存本地 reduce buffer；该 rank 上所有冗余专家计算完成后，把暂存梯度 reduce 回它们 home rank 的梯度 buffer。"reduce" 在这里指跨 rank 的归约（与 all-reduce 同语义，但只针对冗余专家的梯度）。
  5. 用贯穿例子展示 forward 一步：rank 1 从 rank 0 收到 4 个 token，需要 rank 0 上 1 个专家的副本（冗余专家）。
- **讲解材料及职责**：
  - ASCII 图示（component 13）：展示 MoonEP 的 forward（planning + dispatch）与 backward（暂存 + reduce 回 home rank）数据流。
  - 伪代码折叠块（component 11）：forward planning + dispatch 的伪代码（输入、状态、核心步骤、输出），标为伪代码不是 Python。
  - 对照表（component 14）：传统 EP vs MoonEP 在 buffer 大小、形状是否静态、是否需要每层 host 同步、forward 是否有 planning kernel、backward 是否有 reduce-back 五个维度上的对比。
- **前置知识安排**：host-device 同步动机用一句话点出（链接 gpu-execution-model 页）；reduce 语义在正文就地最小解释。
- **完成检查**：
  1. 说明 DeepEP 最坏情况下 buffer 为什么是 $S\times K\times R$，MoonEP 为什么只需 $S\times K$。
  2. 说明传统 EP 为什么每层需要 host 同步，MoonEP 为什么不需要。
  3. 列出 MoonEP forward 的三个关键步骤（planning、prefetch、dispatch 到 expert-grouped 位置）。
  4. 列出 MoonEP backward 的两个关键步骤（暂存本地 reduce buffer、reduce 回 home rank）。
- **过渡**：本章讲清了 MoonEP 的三个工程收益与 forward/backward 流程。最后一章回答：MoonEP 的边界是什么，与相邻方案有什么区别？

### S4：MoonEP 解决什么、不解决什么？与 ECHO、UltraEP、DeepEP 的区别

- **主要教学问题**：MoonEP 的能力边界在哪里？它和相邻方案的关键差异是什么？
- **对应范围**：Q4（C9, C10）。
- **正文要点**：
  1. MoonEP 解决的三个问题：rank 间负载不均衡、通信 buffer 形状动态、每层 host 同步开销（前 3 章已得结论，本章汇总）。
  2. MoonEP 不解决的三个问题：
     - per-expert token 偏斜：完美均衡只保证 rank 间总量相等，rank 内 per-expert token 仍可能偏斜，由 Expert-GEMM workload-aware scheduler 独立处理（C9）。
     - router 本身的训练：router 是否均衡训练由辅助损失负责，MoonEP 只在 router 输出给定后规划（误解 7.1 第一条）。
     - 内存碎片：由 §5.2.2 memory-efficient training 独立处理。
  3. 与 ECHO/UltraEP 区别（C10）：ECHO/UltraEP 预设冗余数或施加 per-rank token cap，训练可能因无解而中断，且 cap 需手动调参仍留残余不均衡。MoonEP 用 $E/R$ 上界保证总有可行解，不需手动调参。
  4. 与 DeepEP 区别：MoonEP 保留 DeepEP 总体计算流但增加冗余专家规划，最坏 buffer 从 $S\times K\times R$ 降到 $S\times K$。MoonEP 不是 DeepEP 的替代，是其上层的扩展。
  5. Expert-GEMM scheduler 简述（C9）：workload-aware scheduler 在 launch 前根据当前 token 分布调参、launch 后固定；shared expert 分到独立 stream 重叠。具体参数与 autotuning 系数报告未公开，本页不展开。
- **讲解材料及职责**：
  - 对照表（component 14）：四列对比 MoonEP、DeepEP、ECHO、UltraEP——是否有界保证、是否需要手动调 cap、最坏 buffer、训练是否可能中断。
- **前置知识安排**：无新前置。
- **完成检查**：
  1. 列出 MoonEP 解决的三个问题与不解决的三个问题。
  2. 说明 ECHO/UltraEP 为什么训练可能中断，MoonEP 为什么不会。
  3. 说明 MoonEP 与 DeepEP 的关系（保留计算流 + 增加规划）。
- **过渡**：本章是边界总结。文末给出"来源与教学说明"。

## 3. 讲解顺序

按 S1 → S2 → S3 → S4。一次只引入一个新变量：
- S1 引入冗余专家（核心新对象）；
- S2 引入 $E/R$ 上界（依赖冗余专家）；
- S3 引入 buffer 与 host 同步的工程后果（依赖完美均衡）；
- S4 引入边界与对比（依赖前三章结论）。

## 4. 贯穿例子

### 4.1 例子设定（首次出现在 S1）

教学示例。 设 $E=4$ 个专家、$R=2$ 个 EP rank、每 rank 本地 $S=4$ 个 token、每 token 选 $K=1$ 个专家。每 rank 持有 $E/R=2$ 个本地专家：rank 0 持专家 {0, 1}，rank 1 持专家 {2, 3}。每 rank 发出 $S\times K=4$ 个 token-expert pair，总 pair 数 $S\times K\times R=8$。完美均衡时每 rank 收到 $S\times K=4$ 个 pair。

### 4.2 router 输出（S1 引入，逐步推进）

初始 router 输出（极度不均衡）：8 个 token 的选择全部落到专家 0（rank 0 上）。结果：rank 0 收到 8 个 pair（过载），rank 1 收到 0 个（空转）。

### 4.3 S2 推进

应用构造性算法：rank 1 underloaded，rank 0 overloaded。迁移 4 个 pair 从 rank 0 到 rank 1。rank 1 需要专家 0 的副本（1 个冗余专家）。验证 $1\le E/R=2$。

再构造最坏情况：rank 0 上的专家 {0, 1} 收到 0 个 token，rank 1 上的专家 {2, 3} 收到全部 8 个 token。rank 0 需要从 rank 1 收 4 个 pair，这些 pair 涉及专家 {2, 3} 共 2 个 = $E/R$ 个冗余专家。验证界在该例下达到。

### 4.4 S3 推进

展示 forward：planning kernel 决定 rank 0 需要从 rank 1 收专家 {2, 3} 的副本（2 个冗余专家）；fused permute 把 4 个 token 直接发到 rank 0 上 expert-grouped 位置。

展示 backward：rank 0 上专家 {2, 3} 副本的梯度暂存本地 reduce buffer；计算完成后 reduce 回 rank 1 的梯度 buffer。

### 4.5 数字便于手算

$E=4, R=2, S=4, K=1$ 全部 $\le 4$，加减乘除可手算。每 rank 总 pair = $S\times K=4$，总 pair = 8，最坏冗余 = $E/R=2$。

## 5. 讲解材料职责汇总

| 材料 | 服务的教学问题 | 出现章节 |
|---|---|---|
| ASCII 图示：传统 EP dispatch 与 rank 过载/空转对照 | S1 的不均衡来源 | S1 |
| 对照表：传统 EP vs MoonEP（每 rank token 数、形状是否静态） | S1 的核心对照 | S1 |
| 数字例子折叠块：$E=4,R=2$ 贯穿例子的最坏构造手算 | S2 的界基本紧 | S2 |
| 补充折叠块：Theorem 1 构造性证明完整复述 | S2 的关键引理 | S2 |
| ASCII 图示：MoonEP forward（planning + dispatch）与 backward（暂存 + reduce 回）数据流 | S3 的流程 | S3 |
| 伪代码折叠块：forward planning + dispatch | S3 的 forward 步骤 | S3 |
| 对照表：传统 EP vs MoonEP（buffer、形状、host 同步、forward、backward 五维） | S3 的工程收益 | S3 |
| 对照表：MoonEP、DeepEP、ECHO、UltraEP 四方案对比 | S4 的边界 | S4 |

## 6. 正文与折叠块分工

### 6.1 必须放正文

- 传统 EP 不均衡来源（C1）、MoonEP 核心思路（C2）、完美均衡目标（C3）。
- 冗余专家定义；$M(I)$ 目标公式（F1）；$E/R$ 上界陈述（F2）与基本紧陈述（F3）。
- 完美均衡 → buffer 降到 $S\times K$（C7）、消除 host 同步（C8）。
- forward/backward 关键步骤（C4, C5, C6）。
- MoonEP 解决/不解决的问题（C9, C10）；与 ECHO/UltraEP/DeepEP 区别。
- 贯穿例子的初始 router 输出与每章关键推进。
- 公式的目的与符号。

### 6.2 可放折叠块

- Theorem 1 构造性证明完整复述与终止性论证（补充折叠块）。
- 贯穿例子的最坏构造完整手算（数字例子折叠块）。
- forward planning + dispatch 的伪代码（伪代码折叠块）。
- Expert-GEMM scheduler 的具体描述（补充折叠块，简短）。

### 6.3 折叠块全收起时正文仍能回答全部学习目标

正文保留：冗余专家定义、$M(I)$ 与 $E/R$ 陈述、buffer 与 host 同步结论、forward/backward 关键步骤、边界与对比。折叠块只承载展开细节。

## 7. 范围与证据约束

本大纲只使用 scope.md 中已纳入范围的内容。所有论断来自 evidence.md 中的 C1–C10、F1–F4。无外部数字。教学例子标为教学示例。

写作中如发现缺口：
- 缺 Theorem 1 完整证明 → 补充折叠块已纳入大纲（S2）。
- 缺 forward/backward 伪代码 → 伪代码折叠块已纳入大纲（S3）。
- 不需要新增学习目标、不需要纳入已排除内容、不需要新增事实。
