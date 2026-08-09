# ExpertPlex outline：教学大纲

## 1. 页面开头

- 一句话：ExpertPlex 让 prefill 和 decode 共享同一份 MoE 专家、各自独占 attention GPU，并用能在微秒级切换的常驻 kernel 管住共享，把 MoE 大模型的有效吞吐（goodput）做到 PD 分离的 2 倍。
- 解决的具体问题：MoE 大模型服务中，PD 分离资源粒度太粗、PD 合设资源切分太死。
- 学习承诺：读完能回答 Q1–Q5。
- 元信息：标题、作者单位、arXiv v2 链接、提交时间。
- 首个具体场景（贯穿例子的引入）：一台 8 卡 H800 服务器跑 MiniMax-M2.7，来了一条 16K token 的长输入（prefill 需要若干毫秒），同时几十个已开始的请求正以每几十毫秒一个 token 的速度解码。同一张 GPU 上，一个 2 ms 的 prefill GEMM 和一个 30 μs 的 decode GEMM 抢资源——这就是全部故事的起点。
- 过渡到第 1 章：两条现有路线各自怎么处理这个冲突，为什么都治不好。

## 2. 章节设计

- 第 1 章「两条现有路线，各自的死结」（负责 Q1）。教学任务：让读者理解 PDD 的粗粒度死结（C2，32+320 GPU 部署单元）与 colocation 的固定切分死结（C3/C4，head-of-line blocking 与资源气泡，Figure 2）。前置知识引用：moe-serving 概念页（prefill/decode、EP、goodput）；gpu-execution-model 概念页（Green Context、SM）。
- 第 2 章「ExpertPlex 的架构：共享专家，分离注意力」（负责 Q2）。教学任务：权重占比论据（C1：95%+）→ 共享专家消除冗余；attention <5% → 整卡分离保住本地算力；三类服务器与数据通路（C6，Figure 3）。误解 2 在此处理。
- 第 3 章「APK：在 tile 边界上调度 GPU」（负责 Q3）。教学任务：为什么需要五个性质（Table 1 逐行）；为什么选 tile 作调度单位（C7，边界 2.2–25.3 μs 与长度无关）；独立检查为什么会死锁、协作决策如何沿存储层级传播（Figure 4）；抢占上界的构成；在线 SM 重分配 F4。贯穿例子推进：16K prefill GEMM 被 decode 抢占，最长挡 decode 多久。前置知识引用：gpu-execution-model 概念页（CTA/cluster/warp/TMA/CUDA Graph/persistent kernel）。
- 第 4 章「通信：让 attention 侧发起一切」（负责 Q4）。教学任务：两侧通信的等待环为何死锁；APK 预分配 buffer 如何使一侧通信成为可能（C11）；push/pull 与 WaitDone（C8）；分层 prefill 路径与流量隔离（C9/C20）；跨阶段重叠（Figure 5）。误解 4 在此处理。
- 第 5 章「跨栈优化器：从 tile 建模到集群」（辅助 Q5 的条件理解）。教学任务：为什么布局/并行度/重叠/共享策略必须联合优化；goodput 定义 F1；为什么 token 数不够、要按 tile 数建模 MoE 延迟（F3 + Figure 6）；离线搜索主干（Algorithm 1 一段概括）+ 在线重分配呼应第 3 章。
- 第 6 章「实验：提升多少，在什么条件下成立」（负责 Q5）。教学任务：设置（N1–N4 压缩为一张设置表）；端到端数字 C12–C14（四组设置全列，含 PDMux 持平的 ShareGPT 反例——误解 1）；三个微基准 C15–C18（Figure 11）。完整基线列表不选择性省略（含 PDD 在 GLM 上 OOM 无数据的事实）。
- 第 7 章「独立评价」（全章解读者推断）。优点：三机制互相使能的设计闭环、对硬件趋势的顺应（权重 vs 算力失衡）。局限：绑定 SGLang/DeepGEMM/DeepEP 栈；≤3 节点验证；router 负载不均正交未解；预分配最大 buffer 的内存代价论文未量化。适用场景与相邻工作位置（AFD 类系统对比）。

## 3. 贯穿例子

贯穿问题：8 卡 H800 节点服务 MiniMax-M2.7，一条 16K token prefill 请求与进行中的 decode 请求并发（§4.1 的真实数字：decode GEMM 17.7–34.7 μs vs prefill GEMM 1.8–2.9 ms）。
- 第 1 章：引入场景与两个数字，展示固定切分下 decode 要等 84–101×。
- 第 2 章：场景中的 GPU 如何被重新划分为 prefill/decode/MoE 三类服务器。
- 第 3 章：decode 到达后 APK 最长多久让出 SM（一个 tile + 一次检查，<25.3 μs 实测上界）。
- 第 4 章：这条请求的 dispatch/combine 数据具体走哪条路。
- 第 6 章：呼应——微基准中 decode 只慢 8%。
局部例子：goodput F1 的代入手算（教学构造数字，标记为教学示例）。

## 4. 讲解材料职责

- Figure 2：呈现 colocation 两种失败模式（第 1 章，回答「固定切分坏在哪」）
- Figure 3：呈现架构与数据通路（第 2 章，回答「GPU 怎么分工」）
- Figure 4：呈现抢占决策传播路径（第 3 章，回答「为什么不会切一半死锁」）
- Figure 5：呈现跨阶段重叠时间线（第 4 章，回答「去掉 MoE 侧 kernel 后 SM 为何不闲着」）
- Figure 6：呈现激活专家数对延迟的影响（第 5 章，回答「为什么按 tile 建模」）
- Figure 11：呈现共享机制 Pareto 前沿（第 6 章，回答「APK 比 MPS/Green Context 好在哪」）
- F1 + 手算代入：理解 goodput 为什么取 min（第 5 章）
- F3/F4：MoE 延迟建模与在线重分配（第 3/5 章）
- Table 1 重制为对照表：五性质 × 六机制（第 3 章）
- 自绘 SVG：两类失败模式简化时间线（第 1 章，辅助 Figure 2 的读图）；APK 调度循环伪代码（language-text，第 3 章）
- 无可运行代码（系统论文，机制无法在本地复现；不放伪代码以外的代码）

## 5. 正文与折叠块分工

- 正文：Q1–Q5 全部答案所需内容；F1/F3/F4 的目的、符号与边界检查；C12–C15 关键数字及条件；概念页链接；五性质表；贯穿例子推进
- 折叠块：F2 拟合式细节与为什么 MoE 用 s=1；离线搜索算法伪代码；C16/C17/C18 微基准细节；MIG profile 讨论（C19）；PDD 部署数字的完整列举（176 GPU、Kimi-K2 128 H200）
- 折叠块全收起时，五章正文仍完整回答 Q1–Q5

## 6. 范围与证据约束

仅使用 scope.md 已纳入内容与 evidence.md 已确认论断；评价章节全部为推断并标记。
