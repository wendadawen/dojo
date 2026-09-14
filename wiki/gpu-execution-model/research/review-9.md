<!-- review-meta
round: 9
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: ce733f84e16ff39a
-->
# GPU 执行模型与 kernel 调度审查记录（第 9 轮）

- 页面版本：index.html 工作树哈希 c935ed4f861a6a38d2160e0054b188803001aa5d（overview.html b55b131c33d8f72ff02afe8bbb3fe2ad3fcaad2e）
- 审查时间：2026-09-14 18:00
- 审查者：编排者派发的独立审查者（独立子代理，未参与写作与前序轮次审查）
- 已完整阅读章节：1. 先认识硬件——一块 GPU 里有什么；2. kernel 是怎么跑起来的——线程、warp、CTA 与 stream；3. 为什么矩阵乘要切成 tile——一次能搬多少数据说了算（3.1/3.2/3.3）；4. Hopper 的协作工具箱——TMA、warp 分工与 CTA cluster（4.1/4.2/4.3）；5. 反复启动太贵——persistent kernel 与 CUDA Graph（5.1/5.2）；6. 分一张卡的四种机制——能力边界对照（6.1/6.2/6.3/6.4/6.5）；来源与范围说明；核心问题（5 题）与各章「本章问题」全部折叠块、常见误解块，以及 overview.html 全文

## 问题

- [轻微·技术] 「3.3 输入变长时，什么变、什么不变」本章问题第 2 题解答（index.html 第 352 行）：末句「$128\times128$ 一档双缓冲后约 128KB，已占用上限的一半以上，是常用的档位。」中「是常用的档位」把工程实践判断写成事实，无来源支持；同一表述已在本轮修改中从 3.2 正文删除（3.2 现为「所以 tile 的尺寸是由 shared memory 容量、数据类型和缓冲策略共同决定的固定值」，不再有「实际高性能 GEMM 常用 $128 \times 128$ 这类档位」），解答块残留该判断，造成同页前后口径不一致。｜引文依据：不适用（该判断无对应来源；所引 Hopper Tuning Guide v13.4 §1.4.1.1 只给出「the maximum shared memory per thread block is 227 KB」，未推荐 tile 档位）。｜修复要求：删除「，是常用的档位」，或改写为明确标注的本页推断（如「（本页选的示例档位）」），使 3.2 正文与该解答口径一致。｜修复：｜复验：

- [轻微·技术] 6.2 节（第 501 行）、页面级核心问题第 5 条解答（第 152 行）、6.5 对照表 MPS 行（第 537 行）、第 6 章末注（第 548 行）：四处声明 MPS 静态 SM 分区「是 CUDA 13.1 才引入的特性（CUDA 12.x 驱动识别不了该开关）」，该版本归属在所引 [C12]（NVIDIA Multi-Process Service 官方文档）中定位不到。｜引文依据：MPS 官方文档 latest(615)/610 分支「Static SM Partitioning」一节原文为「On NVIDIA Ampere architecture and newer GPUs, users can create static SM partitions for MPS clients. This feature provides deterministic resource allocation and spatial isolation between clients by allowing users to explicitly control which SMs each client can access.」；开关出处为 Legacy MPS v2 Interface：「-S or --static-partitioning – Start daemon with static partitioning mode enabled.」；错误隔离出处为 Common Tasks：「Starting with Driver version r610, partial error isolation is supported when static SM partitioning is enabled.」——三处均未出现「CUDA 13.1」或「12.x 不支持」字样；CUDA Toolkit 12.4/13.0/13.1 发行说明（archive 版）亦无 MPS 静态分区条目（13.1 全新特性列表不含 MPS）。｜修复要求：为该版本归属补一条可定位的具体出处（发行说明/博客的链接与原文），否则删去版本判定、只保留官方文档可支持的事实（Ampere 及更新架构支持、-S/--static-partitioning 开关、r610 起部分错误隔离）。｜修复：｜复验：

## 核对记录（本轮逐条回源，写明核对版本）

已核对且与来源一致的条目（未列出的正文论断均在第 1–6 章通读中逐句对照过对应来源）：

- H100 SXM5 规格：132 SM、每 SM 4 个第四代 Tensor Core、50MB L2、80GB HBM3、8 GPC 结构 — 与 Hopper Architecture In-Depth 博文（developer.nvidia.com）逐字一致：「The H100 SXM5 GPU has 132 SMs」「Tensor Cores / SM 4」「8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU … 4 fourth-generation Tensor Cores per SM」「A 50 MB L2 cache in H100 …」「supporting 80 GB (five stacks) of fast HBM3 memory」；该博文记带宽为「over 3 TB/sec」，与 [C5] 注记「In-Depth 博客记作 3 TB/sec」一致，3.35 TB/s 归 H100 白皮书亦无误。
- shared memory 228KB/SM、227KB/块、每 SM 64 warp — Hopper Tuning Guide v13.4 §1.4.1.1 原文：「shared memory capacity per SM is 228 KB, a 39% increase compared to A100's capacity of 164 KB」「the maximum shared memory per thread block is 227 KB」「The maximum number of concurrent warps per SM remains the same as in NVIDIA Ampere GPU architecture (that is, 64)」。
- CTA ≤ 1024 线程、warp=32 — CUDA Programming Guide v13.4（2026-09-09 更新）Table 30：「Maximum number of threads per block 1024」「Warp size 32」。
- TMA：1 维到 5 维、单线程发起、跨 SM/集群传输、组播 — Hopper Tuning Guide §1.4.1.2「TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster」「a single thread can issue large data movement instructions to the TMA unit. The whole block can then continue working on other instructions」；CUDA Programming Guide §4.12「when in a cluster, a bulk-asynchronous tensor operation can be specified as being multicast … data can be transferred from global memory to the shared memory of multiple blocks within the cluster」。
- cluster 可移植上限 8、H100 opt-in 16、cluster 越大同时活跃 block 越少 — Hopper Tuning Guide §1.4.1.3 原文：「The maximum portable cluster size supported is 8; however, NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in」「Using larger cluster sizes may reduce the maximum number of active blocks across the GPU」。
- cluster 落在单个 GPC — CUDA Programming Guide（01-introduction/programming-model）：「all thread blocks in a cluster are executed in a single GPC」「A GPC is a group of SMs in the hardware hierarchy that are always physically close together」。
- stream 优先级不抢占 — CUDA Driver API，cuStreamCreateWithPriority 原文逐字一致：「Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.」
- Green Context：最小 8 SM、8 的倍数、不保证并发与前进、创建时固定 — CUDA Programming Guide §4.6「where the minimum number of SMs per partition is 8 and the SM count has to be a multiple of 8, if useFlags is zero」；Table 30 CC 9.0 列 min SM partition size=8、SM co-scheduled alignment=8；CUDA Driver API（group__CUDA__GREEN__CONTEXTS）：「On Concurrency Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees.」；CUDA 12.4 发行说明「1.2.1. General CUDA」把 Green contexts 列为新增特性（「Green contexts are a lightweight alternative to traditional contexts…」），「CUDA 12.4 引入」成立。
- MPS：动态限额不预留资源、不同客户端 kernel 可落同一 SM、-S 开关、r610 起部分错误隔离、分区 SM 独占不可借用 — MPS 官方文档 When to Use MPS：「Setting the limit does not reserve dedicated resources for any MPS client context.」「Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing.」；Legacy MPS v2 Interface：「-S or --static-partitioning – Start daemon with static partitioning mode enabled.」；Common Tasks：「Starting with Driver version r610, partial error isolation is supported when static SM partitioning is enabled.」「SMs reserved for a partition remain exclusive to the clients assigned to that partition; other clients cannot automatically borrow idle SMs from it.」
- MIG H100 80GB 全表 — MIG User Guide（最新版，页面注明 Last updated on Sep 11, 2026）Table 10「GPU Instance Profiles on H100」逐行核对：1g.10gb=7、1g.10gb+me=1、1g.20gb=4、2g.20gb=3、3g.40gb=2、4g.40gb=1、7g.80gb=1，与页面表格逐格一致；Fraction of SMs 为 1/7、2/7、3/7、4/7、7/7，与「7 个 SM 切片、每片约占 1/7」一致；+me 档 Hardware Units 为「1 NVDEC /1 JPEG /1 OFA」（页面「额外包含至少一个媒体引擎」成立）。
- CUDA Graph 的 CPU 提交开销 — NVIDIA 博文《Employing CUDA Graphs in a Dynamic Environment》（developer.nvidia.com，标题逐字一致）：「When those kernels are many and of short duration, launch overhead sometimes becomes a problem.」
- ExpertPlex — arXiv:2607.18002 实际存在（HTTP 200），标题「ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels」，摘要含「adaptive persistent kernels to schedule dynamic expert computation at tile granularity」，与页面「自适应常驻内核（Adaptive Persistent Kernel，APK）」「tile 级调度」一致。
- 内部链接 — ../expertplex/index.html、../moe-serving/index.html、../vllm-cudagraph/index.html 均真实存在；链接文字与目标页 <title> 一致（「torch.compile 图捕获与 CUDA Graph」「MoE 大模型推理与服务基础」）；overview.html 与 index.html 双向互链；全文无「（待生成）」占位。
- 公式与复算 — $C[i][j]=\sum_{k=1}^{K}A[i][k]\cdot B[k][j]$ 与符号表（$M\times K$、$K\times N$、$M\times N$）一致；4×4 手算逐条复算成立：不切 tile 每元素读 4 次→A 16×4=64、B 64、合计 128；切 2×2 tile 每元素读 $N/T=2$ 次→A 16×2=32、B 32、合计 64；折叠块清单 4×2×4=32 与正文一致。容量估算复核：$128\times128\times2=32768$B=32KB、双缓冲 2×(32+32)=128KB（>227/2，成立）、$256\times256\times2=131072$B=128KB、$512\times512\times2=512$KB 均正确；$8192/128=64$→$16384/128=128$ 正确。
- 机械项 — `.dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 `validation ok`；无 `<img>` 的 alt 含 `$...$`（页面无实际图片）；无 Unicode 数学字符直接出现（第 327 行「1000×1000」为尺寸写法，validate.py 未判为数学字符）；[C1]–[C17]、[N1]–[N7]、[F1] 的定义与正文上标引用一一对应、无悬空编号；两级问题块（核心问题 5 条、各章本章问题）均命名正确且每题都有独立可读的解答折叠块，核心问题答案均指明完整论证所在章节。
- 未发现：数字与官方材料不符、同页两处互相矛盾、算式与结论不符、图注读数与图上刻度不符（本页结构图为 HTML flow 块，无刻度图）、参考文献编号与所引版本不符。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题按修复要求处理即可，不影响核心结论与主线理解；本轮未发现阻断或重要问题）

统计：阻断 0 / 重要 0 / 轻微 2