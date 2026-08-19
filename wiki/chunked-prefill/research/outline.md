# Chunked Prefill（chunked-prefill）文章大纲

## 页面开头

具体场景：一位用户正在逐字看着模型输出（每 20ms 一个 token），此刻另一个用户提交了 8000 token 的长 prompt。若调度器优先算完这段 prefill，前一位用户的输出会停顿数秒。开头给出 chunked prefill 的简要定义、5 个学习目标、第一个例子（把 8000 token 切成 16 块 512），过渡到第 1 章。

## 章节设计

### 第 1 章 长 prefill 为什么卡住所有人：generation stall

- 章节问题：
  1. 同一次迭代里 prefill 和 decode 的耗时为什么差那么远？
  2. prefill 优先调度怎么造成秒级 token 间隔尖刺？
- 答案要点：prefill 并行处理全部输入 token、compute-bound、512 token 即饱和（C1 C3）；decode 每迭代每请求一个 token、memory-bound、小批量下每 token 成本约 200 倍（C1 C2）；连续批处理混批 + prefill 优先 → 进行中 decode 停摆数秒（C9）。
- 对应范围：C1 C2 C3 C9、N1 N2。
- 正文要点：两类迭代的负载结构对比；stall 的时序描述。
- 表达材料：迭代耗时对比表（prefill vs decode）；HTML 结构图 A（时间线：长 prefill 迭代阻塞 decode token 输出，dg-flow 顺序结构）。
- 前置知识安排：prefill/decode 与 KV cache 引用 moe-serving；memory-bound vs compute-bound 引用 gpu-execution-model。

### 第 2 章 把长 prefill 切成块：机制与代价

- 章节问题：
  1. 切块规则是什么，块之间靠什么衔接（KV 累积）、不靠什么（不传层输出）？
  2. 切块的三个代价是什么，各自怎么随块数变化？
- 答案要点：等计算量切块（C4）；块间只通过 KV cache 衔接、attention 读前面全部块的 KV（C5 C6）；代价 = KV 重复读 $\frac{N(N-1)}{2}$（F1）、算术强度下降（C13）、tile 量化对齐（C12）；FFN 总量不变（C5）。
- 对应范围：C4 C5 C6 C12 C13 F1、N3、重复读手算示例。
- 正文要点：数据流主线（每块独立过全层、KV 累积）；三代价的机理各一段。
- 表达材料：chunk 数据流结构图（内联 SVG：3 个块 × 层堆叠，KV 累积箭头、无块间输出箭头，dg-box/dg-line）；重复读手算示例（4096 → 8 块 28 块次 vs 4 块 6 块次）。
- 前置知识安排：因果掩码引用 causal-mask；KV cache 结构引用 moe-serving。

### 第 3 章 decode 搭车与不停止的调度：piggybacking 与 stall-free

- 章节问题：
  1. 为什么 decode 搭 prefill chunk 的车近乎免费？
  2. stall-free 调度怎么用 token budget 保证每次迭代时长有上界？
- 答案要点：权重读取一次同时服务 chunk 与 decode，搭车 decode 成本最多低一个数量级（C7）；一个 prefill 切多块 → 多个搭车机会（C8）；token budget 限制每次迭代 token 总量、新请求以 chunk 加入不暂停 decode（C10）；budget 权衡 TBT SLO 与切块开销（C11）。
- 对应范围：C7 C8 C10 C11。
- 正文要点：memory-bound 论证主线；token budget 机制；"一个数量级不是零"的边界声明（误解 3）。
- 表达材料：piggyback 批构造结构图（HTML 结构 A：一个批次 = 1 chunk + N decode 槽位）；token 间隔平稳化示意（内联 SVG：无预算 vs 有预算的 token 间隔曲线对比，dg-line/dg-accent）。
- 前置知识安排：memory-bound 引用 gpu-execution-model。

### 第 4 章 块要多大：token budget 的权衡

- 章节问题：
  1. chunk 变小，哪些量变好、哪些变差？
  2. 工程上怎么定这个数？
- 答案要点：变小 → TBT 更稳、但 KV 重复读/算术强度/固定开销三项变差（C11 C13）；tile 量化要求对齐（C12，257 vs 256 → +32%）；一次性 profiling 定预算（C11）；块必须够大以饱和算力（C3，"免费搭车"论证的前提）。
- 对应范围：C3 C11 C12 C13、N2 N3。
- 正文要点：权衡清单；边界条件（饱和点以下论证弱化）。
- 表达材料：权衡对照表（chunk 小 ↔ 大）。
- 前置知识安排：无新增。

### 第 5 章 与流水线并行合流：CPP

- 章节问题：
  1. PP 部署中 chunked prefill 为什么顺带消掉了气泡？
  2. Beyond the Buzz 论文里 CPP 的角色是什么？
- 答案要点：气泡源于 micro-batch 时长不均（引用 model-parallelism 结论）；等计算量切块使时长均匀、气泡缩小（C14，6.29×/1.91×）；Beyond the Buzz 中 CPP 是 prefill 池严格 FTL 下的最优策略（C16）；MLA 在 piggyback 下的额外开销与缓解（C17，衔接论文页对架构敏感性的完整讨论）。
- 对应范围：C14 C16 C17、N4。
- 正文要点：均匀化论证；论文页链接（完整搜索空间讨论在论文页）。
- 表达材料：无新增图（气泡机制图在 model-parallelism 页，避免重复）；文字引用结论。
- 前置知识安排：PP 气泡引用 model-parallelism；FTL/TTL 引用 moe-serving。

### 第 6 章 边界与相邻工作

- 章节问题：
  1. chunked prefill 改善了什么、没改善什么，它和 PD 分离是什么关系？
- 答案要点：改善延迟结构与批均衡、不加速单请求 prefill（误解 1）；收益依赖场景（chunk 够大饱和算力、架构 MLA/GQA 敏感、延迟目标宽松时更有利——C17 + 论文页结论）；它是 co-located 部署的优化，PD 分离是另一条路线，Beyond the Buzz 把 piggybacked co-located 作为分离的对照基线；Sarathi-Serve 的容量提升数字（C15 N5 N6）作收尾证据。
- 对应范围：C15 C17、N5 N6（评价性综合标分析性判断）。
- 表达材料：无新增。
- 前置知识安排：链接 beyond-buzz-disaggregation 论文页。

## 贯穿示例

贯穿问题：一个 8000 token 的 prompt 到达时，系统里已有 64 个正在 decode 的请求。
- 第 1 章：不切块时这 64 个请求的 token 输出停顿数秒（时序图）；
- 第 2 章：切成 16 块 × 512 token，逐块处理、KV 累积（数据流图）；
- 第 3 章：每块进批时 64 个 decode 搭车（批构造图，token 间隔恢复平稳）；
- 第 4 章：512 的块大小从哪里来（饱和点 + tile 对齐 + SLO）；
- 第 5 章：若部署是 PP 4 stage，等大的块恰好填满流水线（引用气泡公式）。
数字自设（8000/512/64 均为构造示例的整数化设定，标注），比例结构来自已确认论断（N2 的 512 饱和点为论文数字）。

## 表达材料职责汇总

- 迭代耗时对比表：支撑 prefill/decode 负载结构差异。
- stall 时序结构图：展示 generation stall 的产生。
- chunk 数据流 SVG：展示 KV 累积与块独立性的并存。
- piggyback 批构造结构图：展示单批次内的混合负载。
- token 间隔对比 SVG：展示 stall-free 的效果。
- 重复读手算：验证 F1 的数量级。
- 权衡对照表：支撑第 4 章结论。

## 正文与折叠块分工

- 正文：stall 机制、切块与 KV 依赖、piggyback 论证、token budget、三代价、CPP 均匀化论证、边界。
- 折叠块：重复读计数公式的完整求和展开、piggyback 批构造的逐槽位追踪、tile 量化的补充说明、Sarathi/Sarathi-Serve 完整实验数字表。

折叠块全部收起时，正文可完整回答全部学习目标。

## 误解与边界处理位置

- 误解 1（chunked prefill 加速 prefill）：第 2 章代价小节 + 第 6 章集中。
- 误解 2（块间传输出）：第 2 章数据流小节。
- 误解 3（搭车零成本）：第 3 章。
- 适用边界：第 4 章（饱和前提）+ 第 6 章集中。
