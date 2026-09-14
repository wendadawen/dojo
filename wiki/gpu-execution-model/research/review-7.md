<!-- review-meta
round: 7
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: a529ebb4c42c7155
-->
# GPU 执行模型与 kernel 调度审查记录（第 7 轮）

- 页面版本：`wiki/gpu-execution-model/index.html` 工作树 sha1 `01dceb076ea97fd28b6ca25c1e05f380c491913f`（git HEAD a5f9073，页面无未提交改动）
- 审查时间：2026-09-14 17:39
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；未读取本页 `research/`）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 先认识硬件——一块 GPU 里有什么（含本章问题）/ 2. kernel 是怎么跑起来的——线程、warp、CTA 与 stream（含本章问题）/ 3. 为什么矩阵乘要切成 tile——一次能搬多少数据说了算（3.1、3.2、3.3 与本章问题，含折叠块）/ 4. Hopper 的协作工具箱——TMA、warp 分工与 CTA cluster（含本章问题）/ 5. 反复启动太贵——persistent kernel 与 CUDA Graph（含本章问题与伪代码折叠块）/ 6. 分一张卡的四种机制——能力边界对照（6.1–6.5 与本章问题）/ 来源与范围说明。另完整阅读 `overview.html`。
- 机械项：`python3 .dojo/scripts/validate.py wiki/gpu-execution-model/index.html` → `validation ok`；正文引用 [C1]–[C17]、[F1]、[N1]–[N7] 与文末来源表逐条双向对应，无缺号、无多号；正文/折叠块/overview 之间数字一致（132 SM、228KB/227KB、64 warp、1024 线程、50MB、80GB、3.35 TB/s、8 GPC、8/16 cluster、7 档 MIG、8 的倍数 SM 分区、CUDA 13.1 静态分区）；`$...$` 之外无 Unicode 数学字符；无图片故无 alt 公式问题；图均为 HTML 结构（`.flow`），无脚本也可读。

## 核对所用来源版本（本轮逐条回源所依据的版本）

- arXiv:2607.18002v1（HTML 全文，§1、§4/§4.1、目录）
- CUDA Driver API v13.3.1 文档（Stream Management / cuStreamCreateWithPriority；Green Contexts）
- CUDA C++ Programming Guide 文档站 v13.4 索引页 + 归档 12.6.0（Thread Block Clusters 2.2.1）
- NVIDIA Hopper Tuning Guide（现行版，§1.4.1.1 Occupancy / §1.4.1.2 TMA / §1.4.1.3 Thread Block Clusters）
- NVIDIA Hopper Architecture In-Depth（developer.nvidia.com 博客）
- NVIDIA H100 产品规格页（nvidia.com/en-us/data-center/h100）
- NVIDIA Multi-Process Service 文档 latest（index / architecture / common-tasks / when-to-use-mps / mpsv3-interface / mpsv3-memory-partitioning）
- NVIDIA MIG User Guide（现行版：concepts / supported-mig-profiles / getting-started-with-mig / deployment-considerations）
- CUDA Toolkit 12.4 Release Notes（归档 12.4.0）
- NVIDIA 开发者博客《Employing CUDA Graphs in a Dynamic Environment》
- PTX ISA（现行版发布说明条目，用于核对 TMA multicast 指令限定符）

## 问题

- [重要·技术] 引言第 3 段（第 113 行）：把 ExpertPlex 的动机写成"这篇系统论文的**全部设计**，就建立在「小活到底要等多久、能不能只等一个极小的刻度」这个问题上"，属对来源动机的过度归因，来源并未这样定位自己的设计｜引文依据：arXiv:2607.18002v1 §1 的动机陈述是权重重复与分配粒度——"As MoE weights grow, each instance may span tens to hundreds of GPUs, making resource allocation increasingly coarse."；队头阻塞只是对共置方案的一条批评——"Coarse reconfiguration therefore creates head-of-line blocking and resource bubbles."；论文三个机制中 attention-initiated MoE communication（§5）解决的是共享网络上的通信并发，与"小活等多久"无关｜修复要求：把该句限定到论文的自适应常驻内核（APK）机制（如"它的自适应常驻内核机制正是为回答这个问题而设计"），或删除"全部设计"的表述；不得把队头阻塞问题写成整篇论文的动机｜修复：｜复验：
- [轻微·技术] 6.2（第 501 行末）："而且分区在使用期间不能移除"在 MPS 官方文档中定位不到对应表述｜引文依据：MPS v3 Interface 页对 sm-partition 仅写 "delete – Deletes an SM partition."，未附客户端条件；同一页对 server/namespace 的删除反而明确写了 "Without --force, refuses if active clients remain" 与 "--force allows deletion even with active clients"；Common Tasks 页的示例注释是 "After the application completes, remove the partition"｜修复要求：删除该分句，或改为文档可直接支持的表述（静态分区须在客户端创建前配置、分区内 SM 在分区存续期间为该客户端独占）｜修复：｜复验：
- [轻微·技术] 3.2（第 329 行）："（实际高性能 GEMM 常用 $128 \times 128$ 这类档位）"是无来源的工程实践判断，未标注为归纳，且与来源说明把 128×128 估算定位为"教学构造、不代表推荐工程参数"略有张力｜引文依据：不适用（无对应来源可核对；[F1] 只覆盖 tile 化读取计数的教学推导）｜修复要求：补可定位来源（如 CUTLASS tile 配置文档）或改写为明确标注的工程经验，不得以事实陈述出现｜修复：｜复验：
- [轻微·技术] 5.2（第 462 行）："两种手段是正交的，可以叠加：一个常驻 kernel 内部，也可以把它周期性执行的那串小 kernel 录成图来重放"是无来源的可行性判断｜引文依据：不适用（所引 [C10] 只覆盖 CUDA Graph 的一般描述与录制/重放语义，未涉及"常驻 kernel 内重放图"这一组合）｜修复要求：标注为推断，或补来源（device graph launch）；不得写成已验证结论｜修复：｜复验：
- [轻微·可读性] 第 3 章末（第 335 行）与第 6 章末（第 546 行）：两处提到"ExpertPlex 论文解析页"，但均未给出链接，而 `wiki/expertplex/index.html` 真实存在，同页对 `../moe-serving/`、`../vllm-cudagraph/` 都给了链接｜引文依据：不适用｜修复要求：在这两处补上指向 `../expertplex/index.html` 的链接，或改用不加"页"的表述以消除悬空指代｜修复：｜复验：

## 已核对且未发现问题的要点（本轮回源通过）

- 硬件数字：132 SM / 每 SM 4 个第四代 Tensor Core / 50MB L2 / 80GB / 8 GPC —— Hopper In-Depth："8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs per GPU"、"4 fourth-generation Tensor Cores per SM"、"50 MB L2 cache"、"80 GB HBM3"；带宽 3.35 TB/s 取自 nvidia.com H100 规格页 "3.35TB/s"，与 [C5] 自述"In-Depth 博客记作 3 TB/sec"一致，无矛盾。
- shared memory：228KB/每 SM 上限、227KB/每 CTA 上限 —— Hopper Tuning Guide："shared memory capacity per SM is 228 KB"、"the maximum shared memory per thread block is 227 KB"；64 warp —— 同指南 "The maximum number of concurrent warps per SM remains the same as in NVIDIA Ampere GPU architecture (that is, 64)"。
- TMA：1–5 维、单线程发起、cluster 内 SM 间互传 —— 同指南 "TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory"、"a single thread can issue large data movement instructions to the TMA unit"。
- cluster：可移植上限 8、H100 opt-in 16、越大活跃 block 越少 —— 同指南 "The maximum portable cluster size supported is 8"、"NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in"、"Using larger cluster sizes may reduce the maximum number of active blocks across the GPU"；共同调度到同一 GPC —— In-Depth "A cluster is a group of thread blocks that are guaranteed to be concurrently scheduled"、"Clusters execute across SMs inside a GPC"；TMA 组播 —— PTX ISA 发布说明列出 `cp.async.bulk.tensor` 支持 `.multicast::cluster`（限定符本体说明页被截断，未取到完整句，标注为此项证据强度较低）。
- warp specialization —— [C9] 所指 Hopper Tuning Guide §1.4.1.2 有 "Enables users to write warp specialized codes, where specific warps specialize on data movement"（§1.4.1.1–§1.4.1.3 的区间引用成立）；正文已声明"正式机制以官方文档的 TMA 与 pipeline 章节为准"。
- stream 优先级原文 —— Driver API："Priorities provide a hint to preferentially run work with higher priority when possible"、"do not preempt already-running work or provide any other functional guarantee on execution order"，与页面第 2、6 章引文一致。
- MPS 动态限额两句直引 —— MPS 文档 "Dynamic Execution Resource Provisioning" 节："Setting the limit does not reserve dedicated resources for any MPS client context."、"Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing."（逐字吻合）。
- MPS 静态分区 —— 同文档集："-S" 开关、Ampere 及更新架构、"[Creating] Static partitioning is rejected if the target device already has active clients"（支持"须在客户端创建前配置"）、"SMs reserved for a partition remain exclusive to the clients assigned to that partition"、"other clients cannot automatically borrow idle SMs from it"、"Starting with Driver version r610, partial error isolation is supported"；CUDA 13.1 引入与 12.x 不识别 `-S` 由 13.1 发布材料与 12.6 实测报告佐证。
- MIG —— supported-mig-profiles 表与页面表格逐档一致（1g.10gb 7、1g.10gb+me 1、1g.20gb 4、2g.20gb 3、3g.40gb 2、4g.40gb 1、7g.80gb 1）；"每片约占全卡 SM 数的 1/7"对应 concepts 页 "A GPU SM slice is roughly one seventh of the total number of SMs available"；"改配置需 GPU 空闲/无运行中调整"对应 concepts 页对比表 "Reconfigure — When Idle"。
- Green Context —— Driver API："The smCount must be a multiple of 8"（CC 9.0+，min 8 由此成立）、"Even if the green contexts have disjoint SM partitions"、"it is not guaranteed that the kernels launched in them will run concurrently"、"or have forward progress guarantees"；CUDA 12.4 引入由 12.4 Release Notes 1.2.1 "Green contexts are a lightweight alternative to traditional contexts" 佐证。
- CUDA Graph 启动开销 —— 所引博客原文 "When those kernels are many and of short duration, launch overhead sometimes becomes a problem."，与页面"量级说明、无单一官方数字"的自我限定一致。
- 可复算项：3.1 的 128→64 次读取（不切 tile：16×4×2=128；切 2×2：每元素读 N/T=2 次，16×2×2=64，与折叠块 4×2×4=32 一致）；3.2 的 fp16 容量估算（128×128×2=32768B=32KB，双缓冲 128KB > 227KB 的一半；256×256×2=131072B=128KB 已接近上限、双缓冲超限；512×512×2=512KB）；3.3 的 8192/128=64 与 16384/128=128；228KB 与 80GB 差约 5.5 个数量级（"约五个数量级"成立）。均无算错。
- 符号单义：$C[i][j]=\sum_{k=1}^{K}A[i][k]\cdot B[k][j]$ 后在 `<ul>` 中逐项定义 $A,B,C,i,j,k$ 及取值范围，全文未复用为其他含义；summary 内公式（`dojo:summary` 无公式）与页内 `$...$` 均可由 KaTeX 渲染。
- 表述与结构：无"本页将/下面来看/需要注意的是"类元话语，无第一/第二人称，无调试叙事；「本页」仅出现在来源说明与"（本页归纳）"这一归属标记中，符合 style-guide §12；折叠块 summary 前缀均为「补充：」「代码：」与「解答：」；核心问题 5 条、各章本章问题共 17 条，22 个解答折叠块无遗漏，页面级答案均指明完整论证所在章节；核心问题 5 条对应的章节标题逐字匹配实际 h2 标题；无"（待生成）"占位；两个外部概念链接（moe-serving、vllm-cudagraph）目标页存在且链接文字与目标页标题一致。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复。本轮未发现阻断级问题；1 条重要（对 arXiv:2607.18002 动机的过度归因）与 4 条轻微需按上表修复要求逐条处理后，从修复后的完整页面重新复验。
- 统计：阻断 0 / 重要 1 / 轻微 4