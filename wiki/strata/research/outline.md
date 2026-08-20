# Strata 文章大纲

## 1. 页面开头

- 元信息（组件 02）：标题、作者与单位、OSDI 2026 / arXiv:2508.18572v2、论文链接；无官方代码仓库（删代码行）。
- 定位摘要：长上下文 serving 的矛盾——KV cache 放不进显存所以要分层存，但搬回 GPU 的路（PCIe）被小页传输和迟钝的调度器堵住；Strata 用 GPU 自己的线程搬数据（GPU-assisted I/O）、两端各用各的内存布局、调度器把加载当一等资源（cache-aware scheduling），在长上下文基准上把响应时间做到现有分层缓存系统的 1.9–5 倍吞吐水平而短上下文不退化。
- 贯穿场景引入（完整说明输入、变量、目标）：一个文档问答服务，产品手册 20,000 token，客服系统每次请求 = 手册全文（上下文）+ 一个新问题（约 100 token）。三个客服请求 A、B、C 在一分钟内先后到达。Llama-3.1-8B 下手册 KV cache 约 2.5 GB（20,000 × 128 KB），单一 40 GB 显存同时放手册缓存与运行中的其他请求已很紧张，多份文档必然溢出到 CPU 内存。该场景在全文复用：第 1 章算容量与传输账，第 3 章演 delay hit 与 bundle hit，第 4 章对回实验负载。
- 核心问题（组件 03）：scope.md Q1–Q5，每题配解答折叠块。
- 术语速查表（组件 04）：prefill / decode、TTFT、KV cache、页（page）与页大小、hit rate、HBM、PCIe、pinned memory、DMA、loading-bound / compute-bound、P-D co-location、delay hit / bundle hit。

## 2. 章节设计

### 第 1 章 瓶颈定位：分层缓存为什么救不了长上下文的响应时间（答 Q1）

- 章节问题：为什么把缓存从 CPU 搬回 GPU 会成为主要瓶颈？为什么不能靠加大页解决？
- 答案要点：分层缓存解决"存得下"，不解决"搬得快"。（a）容量账：KV cache 每 token 128 KB（Llama-3.1-8B），40 GB 只放 0.3M token（N1、F3 手算）；分层后搬运成为必经之路。（b）碎片化传输：页 1–32 token（C2）导致序列数据散布在不连续页、单次传输 KB 级（C3）；Little's Law $X=C\cdot S/L$（F1/F2）说明小 S 打不满带宽——打满 PCIe 5.0 需 1–2 MB；实测 8192 token 加载仅 22% 带宽、GH200 上 5%（N3，图 B）。（c）大页的代价：按页匹配的缓存粒度变粗、hit rate 下降，TTFT 最多 2x/2.9x 恶化（C6、N4，图 C）——传输效率与缓存收益是根本权衡。（d）调度器假设失效：现有系统假设 prefill 计算能掩盖加载（C4），长上下文下加载时间超过计算，74% prefill 时间 stall、吞吐降 4x（N2，图 A）；优化 I/O 后仍剩 24% stall，说明调度问题独立存在。
- 正文要点：F3 先于 N1 给出并复算；Little's Law 从 $C=\lambda L$、$X=\lambda S$ 两式合并推导（一步变换），随后代入论文数字解释三杠杆；C7 的 delay hit 在此只点出现象（"重复 prefill 的另一个来源"），完整机制留第 3 章。
- 表达材料：F1/F2/F3 公式（变量关系）；容量与传输量手算（数值示例，正文内小账）；原图 Figure 3（B，带宽利用率）、Figure 2（C，页大小代价）、Figure 1（A，stall CDF）。
- 前置知识：kv-cache 页链接（首次引入 KV cache 存储量时）、paged-attention 页链接（首次引入页时）、prefix-caching 页链接（首次引入 hit rate 时）。
- 折叠块：展开：20,000 token 手册的完整容量与传输账（含 2.5 GB 与按页 32 拆成 625 页、单页 4 MB 的计算——注意单页 4 MB 已够大，说明碎片化发生在"页分布不连续"而非单页过小；此算术留给折叠块展开）。

### 第 2 章 GPU-assisted I/O：让 GPU 自己的线程搬运碎片数据（答 Q2 前半）

- 章节问题：不用 cudaMemcpyAsync，I/O kernel 怎么把小页传输的带宽提上去？代价如何控制？
- 答案要点：（a）机制：启动 CUDA kernel，数千线程各搬一小块（源/目的可以是 GPU 显存或 CPU pinned 内存）（C8）；三优势——并发 C 提到数千、128 字节粒度即可高效（S 不再受限）、kernel 内轻量计算免费（C9、N6）。（b）代价与控制：I/O kernel 与计算 kernel 抢寄存器/执行周期/cache（C10）；Strata 用少量大 block 让硬件调度器把 I/O 圈进极少数 SM、绕过 cache 的指令减污染；2 block × 1024 线程实测 50 GB/s、prefill <5%/decode 10% 损失（N5，图 E）；默认 2 block 加载 + 1 block 备份。
- 正文要点：先回顾第 1 章结论"增大 S 不可行"，指出 $X=C\cdot S/L$ 里被忽略的杠杆 C；解释 cudaMemcpyAsync 的并发受限（CPU 侧并行度与驱动/硬件队列容量，C5 的论述）；I/O kernel 与计算 kernel 的关系用 gpu-execution-model 页的 block/SM 语言描述。
- 表达材料：原图 Figure 5（E，干扰-配额曲线）；微基准数字表（组件 14：配额、吞吐、prefill/decode 损失）。
- 前置知识：gpu-execution-model 页链接（block/SM/thread 首次用于解释时）。
- 折叠块：补充：为什么少量大 block 能把 kernel 圈进少数 SM（结合 GPU 硬件调度器按 block 分配 SM 的行为，标注为基于 gpu-execution-model 页的分析性解释）。

### 第 3 章 布局解耦与存储层：两端各用各的最优布局（答 Q2 后半）

- 章节问题：GPU 要 layer-first、传输要 page-first，冲突怎么解？代价多大？
- 答案要点：（a）两种布局的职责：layer-first 与逐层计算对齐（计算友好）、page-first 把一页各层连续排（传输友好，单页变一次大传输）（C11，图 F）；page-first 若用于 GPU 计算需要一层间接寻址、kernel 复杂化。（b）解法：I/O kernel 线程对偏移多做一次算术运算即可在途转换（C11）；GPU 保持 layer-first、host/存储用 page-first；磁盘微基准 8192 token 延迟最多降 4x（N9，图 L）。（c）存储层（辅助内容，一段）：存储命中后机会性预取到 host 内存、与排队延迟重叠，best-effort（C12）；端到端实验未用磁盘（基线不支持，N19 附注）。
- 正文要点：用贯穿示例的手册 KV 讲"一个 token 的 K/V 在每一层都有份"，从而 layer-first 与 page-first 的差别可直接看出；先说明若强行统一布局两端各损失什么，再给转换解法。
- 表达材料：原图 Figure 6（F，布局对比）；原图 Figure 12（L，磁盘延迟）。
- 折叠块：展开：2 层 × 4 token 的最小布局例子，逐字节算 layer-first 与 page-first 下"搬第 1 个 token 全部数据"需要几次传输（构造示例，说明碎片差别的最小算例）。

### 第 4 章 Cache-aware 调度：把 PCIe 带宽当一等资源（答 Q3）

- 章节问题：调度器怎么避免 loading-bound、避免 delay hit 的重复计算？
- 答案要点：（a）系统全景（先立骨架）：请求队列 → Scheduler 组批 → GPU 执行器 + Cache Controller 加载，HiRadixTree 作页表；P-D co-location 交替执行 prefill/decode 批、沿用 SGLang prefill 优先（C17/C18，图 D）。（b）delay hit 延迟执行：现象复述（C7）→ transient node 两种标记（in-queue/in-flight）→ 命中即推迟一轮并置于队首 → 完成后转标准节点；阈值 100 token（C14、N7）。用贯穿示例演示：A miss 计算期间 B、C 到达并命中 A 的 transient node → 推迟 → A 完成后 B、C 直接复用。（c）均衡组批：load/compute 比例阈值 100（N8）；不 bound 则加入并吸收 bundle hit 请求（B、C 与 D 共享手册 → 同批共享加载）；bound 则降级列表、批不满再补；防饿死条款（C15，Algorithm 1 伪代码）；手算贯穿示例：A（load 20,000、compute 100）单请求 ratio 200 已超阈值 → 需配平。（d）bubble filling：仍 bound 时推迟 prefill、插 decode 批并行；decode 饱和 HBM、加载饱和 PCIe，资源不冲突（C16）。
- 正文要点：图 G（Figure 7）在 (b) 或 (c) 处给出，作为三策略全景；Algorithm 1 以伪代码折叠块呈现（组件 11），正文只讲三个决策点（loading_bound 判断、AddBundleHit、防饿死）。
- 表达材料：原图 Figure 4（D，系统架构）、Figure 7（G，调度策略）；伪代码（Algorithm 1 中译）；自绘 HTML 结构图（结构 A）：贯穿示例三请求在 HiRadixTree 中 transient node 的状态流转（in-queue → in-flight → standard 节点），原图未覆盖此细节。
- 前置知识：prefix-caching 页链接（radix tree 首次用于解释 HiRadixTree 时）。
- 折叠块：伪代码（Algorithm 1）；展开：三请求 delay hit 时间线的手算（A/B/C 到达时刻、匹配 token 数 20,000 > 100 阈值、推迟与复用的计算量对比——B、C 各省 20,000 token 的 prefill 计算）。

### 第 5 章 实验验证：收益数字与成立条件（答 Q4）

- 章节问题：Strata 到底快多少？这些数字在什么条件下成立？各机制分别贡献多少？
- 答案要点：（a）设置：H200（8 卡、PCIe 5.0 x16 64 GB/s）与 GH200（384 GB/s LPDDR5X）、三模型（8B/14B/70B TP4）、四数据集（LooGLE/NarrativeQA/ReviewMT/ShareGPT，Table 1 统计）、基线版本与页配置（N16–N19；SGLang-HiCache 为作者自建基线这一事实必须写明）。（b）端到端：分层必要性（C19、N20）；同 TTFT 吞吐倍数全套（N10）与预热稳态（N11），以对照表呈现（组件 14，基线不删减）；abstract 的 5x/3.75x 与 Llama-70B 行对应，标注"up to"。（c）消融：IO 单独 2.3x、调度单独 1.8x；低请求率调度收益大、高请求率 I/O 主导（C20、N12，图 I）；LPM 对照说明显式带宽感知的必要性。（d）页大小：SGLang-HiCache 最优页 512 也只有 93%（C21、N13，图 J）——Strata 免调参。（e）cache distance：各机制贡献随负载模式变化的全套数字（C22、N14，图 K）。（f）GH200：带宽 40→150 GB/s；仅 I/O 机制仍不敌 H200 完整版——调度是吃满新硬件的必要条件；接近 Oracle（C23、N15，图 M/N）。（g）短上下文：无退化，注明 SGLang 底层引擎本身略弱的对照事实（C24）。
- 正文要点：数字全部带 N 编号与条件；先给读者"看图读法"（横轴请求率、纵轴吞吐/TTFT，曲线如何读出"同 TTFT 吞吐倍数"），再报数字——图 H 信息密度高，需引导句。
- 表达材料：原图 Figure 8（H，端到端全景）、9（I）、10（J）、11（K）、13（M）、14（N）；对照表两张（端到端倍数表、消融表）。
- 折叠块：补充：NarrativeQA 预热稳态实验的设计（为什么要预热、TensorRT-HiCache 为何缺席）与 ShareGPT 的 60 秒思考时间、500K token 显存限制等负载构造细节（N19）。

### 第 6 章 方法评价（分析性判断，答 Q5）

- 章节问题：这套设计好在哪、代价是什么、什么场景该用/不该用？
- 答案要点（分析性判断章，开头声明与论文结论区分）：优点——把 I/O 路径与调度同时纳入带宽资源模型（三杠杆框架贯穿设计）、布局解耦免费午餐式地化解两难、机制间正交可组合；局限——I/O kernel 占用 SM 与算力（<5% 但非零）、依赖 tuning 的阈值（ratio 100、token 100，硬件/模型相关需 profiling）、单机范围（不含全局池化）、disk 未端到端验证、AMD 仅声明；适用——长上下文 prefill 主导、分层缓存已成必需的场景；相邻关系——与 Mooncake/MemServe（全局池）正交可集成，与近似缓存（CacheGen/CacheBlend）方向不同，替代了"层间重叠够用"假设的一系工作。
- 表达材料：无新图；一段与相邻工作的位置描述。

### 文末 来源与范围说明（组件 18，必有）

- C/F/N 与原文定位双向对应；外部来源（vLLM/SGLang/Llama 3.1 模型卡）版本与位置；分析性判断位置（第 6 章及随文标记）；构造示例清单；辅助解释与类比边界；简化条件。

## 3. 贯穿问题或例子

贯穿示例（单一）：文档问答服务，手册 20,000 token，请求 = 手册 + 约 100 token 新问题，请求 A、B、C 相继到达。

- 第 1 章首次出现：完整说明输入（手册 20,000 token、Llama-3.1-8B、128 KB/token）与目标（算容量与传输账）；手算 2.5 GB 缓存、显存放不下多份。
- 第 4 章复用 + 新增：B、C 在 A 计算期间到达 → transient node 匹配 20,000 token > 100 → 推迟；A 完成 → B、C 复用；配平计算（A 单独 ratio 200 > 100）。
- 第 5 章回扣：该场景与 LooGLE/NarrativeQA 的 RAG 负载同构（avg in 21,613/54,797 token），说明实验数字对应读者刚才手算的量级。
- 手册 20,000 token 与 LooGLE avg 21,613 接近是构造时的有意选择，页首标注构造示例性质。

## 4. 表达材料职责

- F1/F2（Little's Law）：解释"为什么小传输必然打不满带宽"，变量关系。
- F3（KV cache 每 token 字节）：支撑容量账，来源为前置页与 Llama 3.1 配置。
- 容量/传输手算（数值示例）：展示代入与中间结果（2.5 GB、625 页、单次传输量）。
- 原图 14 张：见 evidence.md 原图候选；每张按引导句-图-解释段组织。
- 自绘结构图（HiRadixTree 状态流转）：展示 transient node 生命周期，原图未覆盖。
- 伪代码（Algorithm 1）：展示均衡组批的可执行决策流程。
- 对照表 ×2（端到端、消融）：多基线多模型数字并排，基线不删减。

## 5. 正文与折叠块分工

必须正文：三杠杆框架与 Little's Law；两个瓶颈来源与全部关键数字（74%、24%、22%/5%、1–2MB、2x/2.9x）；GPU-assisted I/O 机制与三优势；干扰控制策略与 50 GB/s/<5%/10%；两种布局的职责与解法；调度三阶段机制（transient node 标记语义、ratio 阈值、bundle hit、decode/PCIe 资源区分）；P-D co-location；端到端倍数表与全部条件；消融归因与"低请求率调度/高请求率 I/O"；页大小 93% 结论；cache distance 全套数字；GH200 40→150 GB/s 与"调度是必要条件"；短上下文无退化及引擎差异说明；评价章全部。

可折叠：20,000 token 完整账目展开；SM 圈禁的分析性解释；2 层 × 4 token 最小布局算例；Algorithm 1 伪代码；delay hit 三请求时间线手算；实验负载构造细节（预热、思考时间、显存限制）。

## 6. 范围与证据约束

- 大纲全部内容来自 scope.md 纳入范围；全部论断对应 evidence.md 已确认条目。
- 第 3 章把原 §4.2.1 存储层预取作为辅助内容一段处理（scope.md 扩展内容边界内）。
- 评价章为分析性判断，随文标注。
