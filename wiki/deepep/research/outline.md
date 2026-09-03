# DeepEP 文章大纲

## 贯穿示例（构造）

2 节点 × 4 GPU 的 EP 系统：rank 0-3 在节点 A、rank 4-7 在节点 B；8 个路由专家 E0-E7，专家 $E_e$ 放在 rank $e$（每 rank 恰好 1 个本地专家）；hidden $h=2$；top-$K$ 路由 $K=2$。首次出现于第 1 章（定义 dispatch/combine 语义），第 2 章追踪 token 两段转发路径与按专家分组的接收布局，第 3 章把同一系统切到 decode 场景（每 rank 2 个 token）展示固定槽位与 mask，第 5 章改路由制造不均衡展示最坏情况槽位预留的含义。所有数字便于手算；标记为构造示例，不代表真实部署规模。

具体 token 设定（第 2 章首用）：rank 0 持有 token $t_0$、$t_1$，$t_0$ 路由到 $(E_1, E_5)$、权重 $(0.6, 0.4)$，$t_0 = (1.0, 2.0)$；$t_1$ 路由到 $(E_3, E_3)$ 之外的真实组合 $(E_2, E_6)$、权重 $(0.5, 0.5)$（保证 $t_0$ 的两条路径恰好分别走节点内 NVLink 与跨节点两段转发，$t_1$ 同理）。

## 页面开头

- 场景引入：MoE 模型每层把 token 发给不同专家所在的 GPU，算完再送回来；规模上来后（V3 decode：320 块 GPU 上每 GPU 1 个专家）这一来一回每层都发生，成为系统的主要通信开销。给出 dispatch/combine 的最小定义（引用 moe-serving 页面）。
- 引出问题：这一搬运为什么不能直接用现成集合通信库解决，DeepEP 做了哪些不同的设计。
- 简要回答：DeepEP 是 DeepSeek 开源的专家并行通信库，两套内核分别对准 prefill/训练的高吞吐与 decode 的低延迟；只做搬运，不做路由与计算。
- 学习目标（= 核心问题 Q1-Q5）。
- 过渡：先看通信量本身有多大、卡在哪。

## 第 1 章 专家并行的 all-to-all 卡在哪——为什么不是再写一个 NCCL 调用

- 章节问题：
  1. dispatch/combine 到底在搬什么、量有多大？
  2. 通用集合通信库哪里不够用？
  3. DeepEP 是什么，它在整条链路里负责哪一段？
- 完成答案要点：
  1. dispatch 按 top-K 路由把 token 复制 $K$ 份发往专家所在 rank（每份 $h$ 元素，BF16 $2Kh$ 字节/FP8 $Kh$ 字节 F2）；专家算完，combine 把各副本按门控值加权求和送回原 rank（F1 式 12 的通信侧）。用贯穿示例手算 rank 0 的发送清单。
  2. 四个不适配（C8/C9/C10/C11）：SM 争抢（20/132 SM 做通信、tensor core 闲置 N5）、接收侧无按专家分组的布局契约（grouped GEMM 没法直接消费 C31）、FP8 dispatch 非标准需求（C30）、decode 小 batch 延迟与 CUDA graph 兼容（C11/C13）。V3 背景：跨节点 EP 计算通信比约 1:1（C29）。
  3. DeepEP 定位（C1）：通信库、两套内核（高吞吐/低延迟）、与 V3 论文实现同源但略有差异（C3）、V1 2025-02-24 / V2 2026-04-30（C2）。责任边界一句话预览（详见第 5 章）。
- 表达材料：
  - dg-flow 顺序流程图：MoE 层内 attention 输出 → dispatch → 按专家分组 → expert 计算（grouped GEMM）→ combine 加权归约 → 下一层。职责：给出 DeepEP 在链路中的位置（第 1 章图）。
  - 手算表格：rank 0 两个 token 的发送计划（token、目标专家、目标 rank、路径类型：节点内/跨节点）。职责：把 dispatch 语义落到可复算数字。
  - F1/F2 公式。
- 前置知识：deepseek-moe（top-K 路由）、moe-serving（专家并行与 all-to-all）、gpu-communication（NVLink/RDMA/通信原语）、vllm-cudagraph（CUDA graph）、fp8-block-quant（E4M3）。
- 来源：C1、C3、C8-C11、C13、C29-C31、F1、F2、N5。

## 第 2 章 高吞吐内核——两种网络的两段转发

- 章节问题：
  1. NVLink 和 RDMA 带宽差多少，为什么不能各自直接用？
  2. 两段转发怎么走，为什么两种网络能完全重叠？
  3. 路由算法要配合做什么？
  4. 20 个 SM 怎么把两种网络都跑满？
  5. 实测带宽到什么水平？
- 完成答案要点：
  1. NVLink 160 GB/s vs IB 50 GB/s，3.2 倍（N1）；数据从 NVLink 域去 RDMA 域（或反向）必须有人搬运（C4）。
  2. 两段转发（C5）：token 先 IB 到目标节点同 in-node index GPU，落地即 NVLink 转发给专家 GPU，不被后到 token 阻塞；IB 与 NVLink 重叠。用贯穿示例追踪 $t_0$ 的 E5 副本：rank 0 →（IB）→ rank 4 →（NVLink）→ rank 5。手算每段字节数。
  3. node-limited routing（C6/C7）：每 token ≤4 节点限制 IB 流量；平均 3.2 专家/节点、等效 13 专家（F3）。
  4. SM 与通道（C8/C10）：20 SM、10 通道、warp 专职（dispatch：IB 发送/IB→NVLink 转发/NVLink 接收；combine 反向）；SM 数量可控（V1 set_num_sms(24)，V2 解析式计算）。
  5. 数字（N2）：节点内 153/158 GB/s 逼近 NVLink 峰值；跨节点 43-58 GB/s 逼近 RDMA 峰值；测试条件（4096 tokens、7168 hidden、top-8、FP8/BF16）。V2 更新（N4 前半）：EP8×2 90/81 GB/s 用 12 SM。
- 表达材料：
  - 内联 SVG 图：两节点 8 GPU 拓扑，$t_0$ 的两条副本路径（NVLink 直达段 + IB 段 + NVLink 转发段）用 dg-accent 高亮，图注定义箭头含义。职责：两段转发的空间结构是文字难以说清的。
  - 手算过程（含折叠块展开）：$t_0$ 的 E5 副本在 IB 段与 NVLink 段的流量（FP8 2 字节/份）。
  - 数字表格：V1 normal 带宽表（精简列）。
- 前置知识：gpu-communication（NVLink/NVSwitch/RDMA/IB）。
- 来源：C4-C8、C10、F3、N1、N2、N4（前半）、N7。
- 衔接：第 1 章留下"decode 延迟与 CUDA graph"两个未决点，本章的高吞吐内核用两段转发换带宽、代价是路径长、同步点多——引出第 3 章。

## 第 3 章 低延迟内核——decode 的纯 RDMA 与固定槽位

- 章节问题：
  1. decode 的通信需求跟 prefill 差在哪？
  2. 为什么放弃 NVLink 与两段转发，全部走 RDMA？
  3. 固定槽位布局怎么让接收侧不做 CPU 同步、兼容 CUDA graph？
  4. hook 怎么让通信等待不占 SM？
  5. 低延迟的代价是什么？
- 完成答案要点：
  1. decode 每步都发生、batch 小（V3 生产 128 tokens/batch，N3 条件）、TPOT 敏感；V3 decode 部署 EP320 每 GPU 1 专家、IB 直接点对点 + IBGDA（C28）；每 expert batch ≤256、瓶颈是访存不是计算，少量 SM 即可（C28）。
  2. 纯 RDMA（C12）：所有 rank 经 RDMA 可见（含节点内），NVLink 禁用为简化设计；点对点直达去掉两段转发的中转与同步。
  3. 固定槽位（C13/F4）：接收张量 [本地专家数, 每 rank 最大 token 数 × rank 数, hidden]，按最坏情况预留；不做 CPU 接收计数同步 → CUDA graph 兼容（C14）；recv_count 标记有效槽位；隐式 FP8。用贯穿示例（每 rank 2 token）画槽位占用表：E5 的行里来自 rank 0 与 rank 6 的槽位。
  4. hook（C15）：return_recv_hook=True 只发 RDMA 请求不等待，GPU 转去算另一个 micro-batch（TBO，C28），调 hook 时数据已到；等待期不占 SM。
  5. 代价（N3 数字对比 N2）：RDMA 带宽利用率低（EP128 时 39 GB/s vs normal 跨节点 43-58 GB/s、节点内 153-158）；buffer 按最坏预留显存大（C17、C16 双缓冲限制）。
- 表达材料：
  - HTML 表格：固定槽位布局（行=本地专家，列=来源 rank 的槽位区间），高亮有效槽位与空槽。职责：让"按最坏预留 + mask"一眼可见。
  - dg-flow 时间线：micro-batch 1 dispatch 发请求 → 算 micro-batch 2 attention → 调 hook 收数 → 算 micro-batch 1 MoE → …… 职责：TBO 交错结构。
  - 可运行代码（Python，模拟）：给定 8 rank × 每 rank 2 token 的 top-2 路由与权重，计算每 rank 的发送计划、每本地专家的槽位占用与 recv_count、combine 带权归约结果，验证 round-trip 数值。标记为模拟布局与归约逻辑的最小实现，不是 DeepEP 库本身代码、不含真实通信。职责：把槽位布局与带权归约变成可执行验证。
  - 数字表格：V1 LL 延迟表（精简列）。
- 前置知识：moe-serving（prefill/decode、TPOT）、vllm-cudagraph（CUDA graph）、本章依赖第 2 章的 normal 机制作对照。
- 来源：C12-C17、C15、C28、F1、F4、N3、N6、N11（C17/N11 同源）。
- 衔接：两套内核（V1）讲完，接口是两套、调参靠 auto-tuning——V2 把这些工程面统一重做，引出第 4 章。

## 第 4 章 V2 重构——统一接口与解析式调参

- 章节问题：
  1. V2 换掉 NVSHMEM 的理由是什么？
  2. ElasticBuffer 统一了什么，规模扩到多大？
  3. 解析式 SM/QP 计算取代了什么？
  4. V2 的收益与代价各是什么？
- 完成答案要点：
  1. NCCL Gin 后端（C18）：header-only、轻量、复用现有 NCCL communicator；全 JIT 编译免安装期 CUDA 编译（C22）。
  2. ElasticBuffer（C19）：高吞吐/低延迟 API 统一、新 GEMM 布局、EP2048；hybrid（分层 RDMA+NVLink，multi-plane/multi-rail 友好）与 direct 模式保留（C23）；decode handle 缓存跳过布局重算与 CPU 同步（C24）。
  3. 解析式调参（C10 后半）：get_theoretical_num_sms 以带宽建模（RDMA/NVLink/单 SM HBM 读写带宽）直接算出 SM 数，假设均衡 gate；取代 V1 的 auto-tuning（默认配置优化在 DeepSeek 内部集群）。
  4. 收益（C20/N4/N5）：SM 24→4-6、峰值 1.3 倍、SM 省 4 倍、EP8×2 90/81 GB/s@12SM、EP8×4 61/61@6SM；代价（C21）：buffer 消耗更大、0 SM RDMA 低延迟 EP 移除（第 3 章 hook 形式的纯 0 SM 等待不再支持）、Engram/PP/CP 实验性（C32 一句话提及）。
- 表达材料：
  - 对照表（V1 vs V2：后端、接口、SM、调参、规模、低延迟 0 SM）。职责：一表收拢重构变化。
  - 数字引用（N4 表格已在第 2 章给过前半，此处补全并给 V2 测试条件）。
- 前置知识：本章依赖第 2、3 章机制；NCCL 概念引 gpu-communication 通信库章节。
- 来源：C10、C18-C24、C21、C32、N4、N5。
- 衔接：机制与版本讲完，剩下"哪些问题 DeepEP 不管"——引出第 5 章。

## 第 5 章 DeepEP 不解决什么——与相邻系统的分工

- 章节问题：
  1. 路由与负载均衡为什么不在 DeepEP 里，不均衡时会发生什么？
  2. 计算与重叠调度分别由谁做？
  3. 用 DeepEP 有哪些硬性门槛？
- 完成答案要点：
  1. DeepEP 只搬不改路由（误解 3）：token 去向由 gating 决定；不均衡时接收槽位照样空置或拥挤（F4 最坏预留显存代价，用贯穿示例改路由演示：6 个 token 全涌向 E5，rank 5 槽位吃紧、其余空置）；负载均衡靠 gating 算法（V3 aux-loss-free + bias）、部署层冗余专家（C27/C28 的 32/64 个冗余）或专门系统（UltraEP 配额重路由、MoonEP 冗余规划，链接 wiki 页面）。
  2. 计算：接收侧 grouped GEMM（C31，DeepEP 保证按专家分组连续布局）；调度：DualPipe 在训练侧重排 chunk 四组件隐藏 all-to-all（C29）、TBO 在 decode 侧交错两个 micro-batch（C28）——DeepEP 提供内核，重叠编排是框架的事。
  3. 门槛（C26/C25）：Hopper SM90+、节点内 NVLink、跨节点 RDMA（IB 全测试、RoCE 理论兼容）；网络建议（VL 隔离、adaptive routing 常开、拥塞控制禁用）。
- 表达材料：
  - 责任分工表（路由/搬运/计算/调度 × 谁负责 × 本页哪章）。职责：一表关掉"DeepEP 是框架"的误解。
  - 构造示例改路由的手算（哪个 rank 槽位不足、哪些空置）。
- 前置知识：ultraep、moonep 页面（相邻概念）；moe-serving PD 分离章。
- 来源：C25-C32、F4、N6；相邻系统叙述引用 wiki/ultraep、wiki/moonep 页面链接。

## 表达材料职责汇总

| 材料 | 位置 | 解释目标 |
|---|---|---|
| dg-flow 链路图 | 第 1 章 | DeepEP 在 MoE 层内的位置与边界 |
| 手算发送计划表 | 第 1 章 | dispatch 语义落到数字 |
| SVG 两段转发图 | 第 2 章 | 两段路径的空间结构与两网重叠 |
| 手算流量（折叠块） | 第 2 章 | IB 段与 NVLink 段流量可复算 |
| 带宽表（V1/V2 精简） | 第 2、4 章 | 高吞吐内核逼近硬件峰值 |
| 槽位占用表 | 第 3 章 | 最坏预留 + mask 的固定布局 |
| TBO 时间线 dg-flow | 第 3 章 | hook 与 micro-batch 交错 |
| 可运行 Python 模拟 | 第 3 章 | 槽位布局与带权归约可执行验证 |
| LL 延迟表 | 第 3 章 | 延迟水平与带宽代价 |
| V1/V2 对照表 | 第 4 章 | 重构变化收拢 |
| 责任分工表 | 第 5 章 | 边界与相邻系统分工 |

## 正文与折叠块分工

- 正文：dispatch/combine 定义、F1/F2、四个不适配、两段转发机制、node-limited routing 结论、SM/通道机制、固定槽位布局形状与含义、hook 语义、V2 变化清单、责任边界、全部关键数字及其条件。
- 折叠块：第 2 章完整手算流量过程（展开：）、第 3 章可运行代码与输出（代码：）、V3 训练侧 DualPipe 背景补充（补充：）、V1 normal 的 CPU 等待细节（补充：）、低延迟双缓冲限制（补充：）、F3 等效专家推导细节（展开：）。
- 折叠块全收起时正文仍回答全部学习目标。

## 章节顺序依据

第 1 章建立"要搬什么、为什么难"（问题），第 2、3 章是两套内核对两种场景的答案（机制，prefill 先于 decode 因训练/prefill 是库的起源且机制更基础），第 4 章是版本演进（依赖 2、3 章的 V1 机制作对照），第 5 章收边界（依赖前四章全部）。每章末尾本章问题 + 解答折叠块。
