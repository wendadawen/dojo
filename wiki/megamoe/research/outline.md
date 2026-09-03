# MegaMoE 文章大纲

## 0. 页面参数

- 页面标题 h1：`MegaMoE：把 MoE 一层塞进一个持久 kernel，让通信与计算同时进行`
- lead 导语：DeepSeek 开源在 DeepGEMM 里的融合 MoE kernel；官方写法 Mega MoE。专家并行下 MoE 层的五段执行（dispatch、Linear-1、SwiGLU、Linear-2、combine）在传统实现里是多个独立 kernel 串行，通信时计算单元空转；Mega MoE 把五段融合成一个占满全部 SM 的持久 kernel，靠 warp 专用化让 NVLink 传输与 Tensor Core 计算重叠。EP8 基准下相对非重叠基线 1.50–1.96 倍。
- dojo:type=concept；dojo:topics=推理系统,并行与通信；dojo:tag=moe-kernel,deepgemm,communication-overlap

## 1. 页面开头

以一个具体场景进入：一块 GPU 上的 MoE 层，token 要发给别的卡上的专家去算，算完还要送回来。传统做法里"发送—等待—计算—等待—收回"是串行的：传输时 Tensor Core 停着，计算时 NVLink 闲着。给出五段流程的一句话预告和 mini 路由示例（贯穿示例首秀，只出现 token 与专家的归属，不引入任何 kernel 细节）。

随后核心问题块（= 学习目标 Q1–Q5，措辞与 scope.md 一致）， misconception 块放误解 1（"融合=消灭通信"）与误解 2（"Mega MoE 是通信库替代 DeepEP"），误解 3、4 放对应章节内（yellow callout）。

过渡句：先看清楚传统执行到底慢在哪。

## 2. 章节设计

### 第 1 章 MoE 一层的五段执行——通信为什么让 GPU 空转（负责 Q1）

- 章节问题：
  1. 专家并行下，一个 token 穿过 MoE 层要经过哪五段？每段数据在哪里？
  2. 为什么说传统实现里"通信时计算空闲、计算时互联空闲"？
- 答案要点：
  1. dispatch（token 按路由发往专家所在 rank）→ Linear-1（hidden → 2×intermediate，gate/up 两半）→ SwiGLU（silu(gate)×up，宽度减半）→ Linear-2（intermediate → hidden）→ combine（按路由权重聚合 top-k 送回源 rank）；用 mini 示例逐段指认 t0 从 rank0 出发到 E3（rank1）再回来的路径。
  2. 五段由独立 kernel 依次完成，每段结束到下段开始之间数据要落显存再读回，dispatch/combine 期间 SM 无有效计算，GEMM 期间互联无流量；SVG 时间线图直接展示两种空闲。
- 范围：Q1；C1（五段定义部分）；F1（输出聚合公式，首次给出）；F3（单 token dispatch 通信量，量级手算）。
- 材料：mini 路由表（构造示例标记）；SVG 时间线对比图（图 1：上泳道传统串行、下泳道融合重叠的示意，标注为示意图非实测 trace）；V4-Pro 规格代入的通信量手算（512/rank、top-k 6、hidden 7168 → 每 token 发 6×7168 B）。
- 前置知识引用：deepseek-moe（路由/top-k/共享专家）、moe-serving（EP）、swiglu、deepep（dispatch/combine 最小含义+链接）、gpu-communication（NVLink）。
- 过渡：要同时做传输和计算，前提是"发数据的人"和"算数的人"住在同一个 kernel 里。

### 第 2 章 一个持久 kernel 里的分工——融合与重叠如何成立（负责 Q2）

- 章节问题：
  1. "融合成一个 kernel"和"重叠通信与计算"分别指什么？持久 kernel 是什么形态？
  2. kernel 内部有哪些角色，各自负责什么？
- 答案要点：
  1. 融合=五段不再各自启动 kernel、状态不落回框架；持久=grid 恰好覆盖全部 SM、kernel 从层开始跑到层结束，任务由 kernel 内调度器分发（对照 gpu-execution-model 的常规 kernel 形态）。重叠=通信动作（远端读写、拉取）由专用线程组与其他 warp 的 GEMM 计算同时推进；通信量不变（澄清误解 1，若未放开头则在此收口）。
  2. 六类角色：dispatch 线程组（128 线程：计数、源索引、分块拉取）、TMA A 加载 warp（激活+SFA）、TMA B 加载 warp（权重+SFB）、MMA 发射 warp（leader CTA，2-CTA UMMA、TMEM）、调度 warp（发布 task）、epilogue 线程组（SwiGLU、量化、远程写回、combine 归约）；block 内三段线程布局 + 2-SM cluster 配对；寄存器在角色间再分配。
- 范围：Q2；C1、C8、C9、C6（持久/分工/cluster/TMEM 概念性引用）、C30。
- 材料：dg-stack 图（图 2：一个 CTA block 的三段线程布局，标注各段 warp 数与职责；旁边注明 2-SM cluster 配对与 leader CTA）；三组线程数字来自 heuristics（128/128/128-256）。
- 过渡：这些角色要直接读写别的 rank 的内存，这依赖一块通常不被注意的地基——对称内存。

### 第 3 章 对称内存——kernel 内跨 rank 读写的地基（负责 Q3）

- 章节问题：
  1. 对称内存的布局约定是什么？地址是怎么"翻译"到远端的？
  2. 为什么融合 kernel 必须要它，而独立通信库不需要？
- 答案要点：
  1. 每个 rank 以相同布局分配各自的缓冲区（PyTorch >= 2.9 对称内存，多进程），SymBuffer 保存本 rank 基址 + 全部 rank 偏移表；`map(ptr, dst_rank) = ptr + offsets[dst_rank]`，一次加法把本地地址变成远端等价地址；mini 示例代入（rank0 基址 $B_0$、rank1 基址 $B_1$、$\Delta = B_1 - B_0$，t0 在 rank0 的输入槽地址加 $\Delta$ 即 rank1 视角）。最大 72 rank。
  2. 独立通信库由框架在 kernel 之间编排 all-to-all，kernel 本身只见本地指针；融合 kernel 在执行中途就要发起远端读写，必须让每个线程都能算出远端地址，且跨 rank 的到达顺序需要 kernel 内屏障（NVLink barrier）协调。使用形态：多进程 spawn、分配对称缓冲、权重布局变换、调用（官方 API 三步，折叠块放 README 代码）。
- 范围：Q3；C10、C22、C23、C29；C3（PyTorch >= 2.9）。
- 材料：SVG 对照图（图 3：rank0/rank1 同布局缓冲并排，地址差 Δ 标注，一条 map 箭头）；官方 API 代码块（折叠块，标注来自 README、需 8×sm100 GPU 才能运行，本页不运行）。
- 过渡：地基就位后，看一个 token 从进入 kernel 到写出 y 的完整路径。

### 第 4 章 一个 token 的完整旅程——拉取、两层 GEMM、远程写回与归约（负责 Q4）

- 章节问题：
  1. dispatch 侧怎么把 token 集合到专家手里？（元数据先行、数据拉取）
  2. 两层 GEMM 的任务怎么组织？环形缓冲怎么衔接？
  3. L1 epilogue 做了什么？L2 结果怎么回到源 rank、最后怎么归约成 y？
- 答案要点：
  1. 三步：dispatch 线程组读本地路由、每专家计数并远端原子累加；把每个 (token,topk) 的源索引写往专家所在 rank；NVLink barrier 后接收侧按专家分组、round-robin（最小剥离）选源 rank，TMA 按 ≤4096 字节的块把 token 从远端拉进本地环形缓冲（元数据小先 push、数据大后 pull 的理由：数据只有接收侧知道何时要用）。mini 示例代入：rank1 为 E2 拉取 t1（源 rank0）、为 E3 拉取 t0、t1。
  2. 调度单元是 task（BlockPhase 四相：Linear1/Linear2/SharedLinear1/SharedLinear2 + 专家号 + M 块 + N 簇）；先发 L1 预热波防死锁，之后 L1/L2 交替；各 SM 调度 warp 原子认领。数据缓冲按环形容量循环，full/empty 计数器自旋衔接生产消费；L1 输出与 L2 输入同张量（SwiGLU 后宽度减半）。共享专家任务不依赖 dispatch 可先行。启发式：按每专家期望 token 数（公式 F2）分六档选 block_m——手算 V4-Pro batch 512/rank：512×8×6/384 = 64 → block_m=96 档。
  3. L1 epilogue：TMEM 读出 → clamp → silu(gate)×up → 乘路由权重 → amax → FP8 E4M3 + UE8M0 SF → TMA store 进 L2 输入。L2 epilogue：TMEM 读出转 BF16 → 查源元数据（rank、token、topk 槽）→ remote store 写入源 rank 的 combine 缓冲。全部写回后 NVLink barrier → combine：每 token 遍历 top-k 槽（共享专家占第 $k$ 槽），float 累加、转 BF16、写出 y。mini 示例代入：E3（rank1）的两行输出分别写进 t0、t1 在 rank0 的 combine 缓冲 2 号槽；rank0 归约 t0 = E0 槽 + E3 槽之和。X-Stage 的 wave 术语与后续改进（1.18x/1.62x）作为折叠块补充，标注第三方。
- 范围：Q4；C7、C11–C21、F1（权重乘法位置的澄清）、F2、F4（可选折叠）。
- 材料：SVG 全景数据流图（图 4：两 rank 并列，token 从 input → pull 箭头 → L1 ring → GEMM 方块 → SwiGLU → L2 → remote store 箭头 → combine 缓冲槽 → 归约 → y，mini 示例的具体 token 标注在图上）；角色协作伪代码（折叠块，text 围栏：主循环骨架按角色分行）；F2 手算表；误解 4 的 yellow callout（dispatch pull/combine push 不对称）。
- 过渡：机制说完，看官方数字与它成立的前提。

### 第 5 章 收益与边界——基准数字、正确性与适用条件（负责 Q5）

- 章节问题：
  1. 官方基准测了什么、加速多少？对照的基线是什么？
  2. 哪些条件下 Mega MoE 用不了或收益存疑？怎么确认它算得对？
- 答案要点：
  1. PR #316：V4-Flash（256 专家、top-k 6、hidden 4096、inter 2048）与 V4-Pro（384 专家、top-k 6、hidden 7168、inter 3072），EP8、batch=每 rank token 数（1/512/8192/32768）、8 rank 平均；加速 1.50–1.96x，batch=1 收益最大（1.96x/1.61x）；互联带宽列展示通信流量可观（batch 512 时 266/182 GB/s）。基线 = DeepEP dispatch/combine + DeepGEMM grouped GEMM + TileLang SwiGLU 非重叠流水线。正确性：无共享专家逐位一致、有共享专家 <1e-8、接收统计逐位一致。
  2. 边界：仅 sm100 实现且用 TMEM/tcgen05（`__CUDA_ARCH__ >= 1000` 守卫）；PR #304 时点仅 FP8×FP4（后续 main 有 bf16 变体）；需 PyTorch >= 2.9、多进程对称内存；top-k ≤ 32、top-k+共享 ≤ 32；仍在开发（官方声明）；Hopper 无官方收益数据（第三方评论：无 TMEM 不显著，可用 FP4 EP v2 + FP8 DeepGEMM + PDL 替代——标注第三方）；X-Stage 学术改进未进 main。误解 3 的 yellow callout 收口。
- 范围：Q5；N1–N12、C5、C3、C4、C21、C25、C26、C27、C28。
- 材料：基准表（HTML 表格，两模型各一张或合并）；时间线小表（2026-04-16 News / PR #304 / PR #316）；边界清单。
- 收尾：回扣核心问题——五段融合、warp 分工、对称内存、数据流、边界。

### 来源与范围说明（固定 h2，不编号）

- 论断与来源（C）：C1–C30 摘要表（编号、论断、来源定位）
- 公式与来源（F）：F1–F4
- 外部数字与实验条件（N）：N1–N12
- 构造示例：mini EP2 路由示例、通信量手算、F2 代入手算、时间线示意图
- 辅助解释与类比边界：时间线图为示意非实测；"传送带 vs 接力"类比（若使用）标注边界
- 简化条件及其限制：mini 示例维度极小；图 2 只画单 CTA；伪代码省略 SF 布局/屏障细节；只讲 FP8×FP4 主路径

## 3. 讲解顺序依赖

问题（第 1 章）→ 总体方案与角色（第 2 章）→ 跨 rank 读写地基（第 3 章）→ 完整数据流（第 4 章）→ 结果与边界（第 5 章）。第 4 章依赖第 2 章的角色与第 3 章的 map 原语；第 5 章依赖全部机制。前置概念页链接在首次依赖处给出（见 scope.md 第 4 节）。

## 4. 贯穿示例

mini EP2 配置（构造示例，全程同一份）：

- 2 个 rank（rank0、rank1），各持 2 个专家（E0、E1 在 rank0；E2、E3 在 rank1），top-k=2，每 rank 2 个 token（t0、t1 在 rank0；t2、t3 在 rank1）。
- 路由：t0→{E0,E3}、t1→{E2,E3}、t2→{E1,E2}、t3→{E0,E1}。
- 第 1 章：指认 t0 的五段路径（本地 E0、远端 E3），展示跨 rank 数据流与串行空闲。
- 第 3 章：SymBuffer 偏移 $\Delta$ 的 map 手算（t0 输入槽地址在两个 rank 视角间的换算）。
- 第 4 章：rank1 视角完整走一遍——为 E2 拉 t1、为 E3 拉 t0/t1；E3 的两行输出 remote store 回 rank0 的 t0/t1 combine 槽；rank0 对 t0 做 E0+E3 两槽归约。
- 复用规则：每次出现只加一个新概念层（路由→地址翻译→拉取/写回→归约），数字保持可手算。

局部例子：V4-Pro 规格的通信量手算与 F2 分档手算（服务量级感与启发式理解），明确标注服务目标。

## 5. 表达材料及职责

| 材料 | 位置 | 职责 |
|---|---|---|
| SVG 时间线对比图（图 1） | 第 1 章 | 展示串行双空闲 vs 重叠占满——只此一图能直观回答 Q1.2；标注示意图 |
| mini 路由表 | 第 1 章首次，后续复用 | 提供可手算的具体对象 |
| 通信量手算 | 第 1 章 | 把"通信不可忽略"变成数字 |
| dg-stack 线程布局图（图 2） | 第 2 章 | 展示 block 内三段线程与 cluster 配对的结构 |
| SVG 对称内存对照图（图 3） | 第 3 章 | 展示地址平移这一句加法 |
| 官方 API 代码块（折叠） | 第 3 章 | 使用形态（多进程+对称缓冲+三步调用），标注不可本地运行 |
| SVG 全景数据流图（图 4） | 第 4 章 | 一张图回答 Q4 的数据流主干；mini token 标注 |
| 角色协作伪代码（折叠） | 第 4 章 | 展示主循环骨架与角色并发关系 |
| F2 手算表 | 第 4 章 | 启发式分档的具体代入 |
| 基准表 | 第 5 章 | 官方数字的完整呈现 |
| 时间线小表 | 第 5 章 | 版本脉络 |

图内公式一律 `<foreignObject>`；SVG 元素用 dg-box/dg-line/dg-accent 类，不写死颜色。公式 F1 在第 1 章给出（符号逐项定义），第 4 章引用并澄清权重乘法位置。

## 6. 正文与折叠块分工

必须放正文：五段定义与 mini 路径、融合/重叠/持久 kernel 定义、六角色分工表、SymBuffer 模型与 map 公式、dispatch 三步、task 调度与 L1/L2 交替、环形缓冲衔接、L1/L2 epilogue 职责、combine 归约、基准表完整数字、全部边界条件、误解澄清。

可放折叠块：官方 API 完整代码、角色协作伪代码、X-Stage 补充（wave 术语与改进数字）、F4 容量公式推导、"补充：为什么元数据 push 数据 pull"的展开论证、完整手算过程（展开：xxx）。

折叠块全部收起时，正文仍完整回答 Q1–Q5。
