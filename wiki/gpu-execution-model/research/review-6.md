<!-- review-meta
round: 6
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 2b988ec9aa2dda13
-->
# GPU 执行模型与 kernel 调度审查记录（第 6 轮）

- 页面版本：a09e064c5d3f4235fce6a0c1f7012324ed3e9c50
- 审查时间：2026-09-14 16:59
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：引言与范围说明 → 核心问题 → 最容易误解 → 1. 先认识硬件（含本章问题）→ 2. kernel 是怎么跑起来的（含本章问题）→ 3. 为什么矩阵乘要切成 tile（3.1–3.3、本章问题）→ 4. Hopper 的协作工具箱（4.1–4.3、本章问题）→ 5. 反复启动太贵（5.1–5.2、本章问题）→ 6. 分一张卡的四种机制（6.1–6.5、本章问题）→ 来源与范围说明；并读 overview.html。

## 问题

- [重要·技术] §6.2（index.html:501）+ §6.5 范围说明（index.html:548）+ 来源条 [C12]（index.html:578）：页面把「静态 SM 分区」当作 MPS 的常规两套资源机制之一介绍（「另一套是静态 SM 分区（static SM partitioning）……该模式在 Ampere 及更新的架构（含 Hopper）上支持」），没有标注它的版本条件；同一章末尾却声明「本章数字以 H100（Hopper）与 CUDA 12.x 文档为准」，[C12] 又引用了 r610 驱动与新一代 MPS 文档。MPS 静态 SM 分区是 CUDA 13.1 才引入的特性（由 MPS 控制守护进程的 -S/--static-partitioning 开关开启），CUDA 12.x 驱动识别不了该开关；页面按「CUDA 12.x 文档为准」定位，会让 CUDA 12.x 读者误判这一能力在自己的环境里可用。本章讲的正是「能做什么、不能做什么」的能力边界，这是重要条件缺失。｜引文依据：NVIDIA MPS 官方文档《When to Use MPS》：「Static SM partitioning mode allows users to create exclusive SM partitions for MPS clients … on NVIDIA Ampere architecture and newer GPUs.」「Starting with Driver version r610, partial error isolation is supported when static SM partitioning is enabled.」；CUDA 13.1 版本资料与开发者答复：「static partitioning is a new feature introduced in CUDA 13.1」（CUDA 12.6 下 -S 不被识别）；r570 版 MIG 用户指南仍把 MPS 的 SM 隔离记为「SM Performance Isolation: Yes (by percentage, not partitioning)」。｜修复要求：在 §6.2 或 §6.5 标注 MPS 静态 SM 分区的版本/驱动条件（CUDA 13.1 引入；r610 起在静态分区下提供部分错误隔离），并让 §6.5「以 H100（Hopper）与 CUDA 12.x 文档为准」的范围声明与之一致（改为覆盖 13.x，或注明该条超出 12.x 范围）。｜修复：｜复验：
- [轻微·技术] 来源条 [C5]（index.html:578）与 §1 数值（index.html:178、184、208）：H100 带宽 3.35 TB/s 与「132 SM / 每 SM 4 个第四代 Tensor Core / 50MB L2 / 80GB HBM3」一并归给「H100 白皮书 及 NVIDIA Hopper Architecture In-Depth（developer.nvidia.com）」，但 In-Depth 博客原文只给 3 TB/sec，并未给出 3.35 TB/s；3.35 TB/s 只由白皮书/官方规格页支持。数字本身正确，属引文精度问题。｜引文依据：Hopper Architecture In-Depth 原文「The H100 SXM5 GPU is the world's first GPU with HBM3 memory delivering a class-leading 3 TB/sec of memory bandwidth.」（另作「over 3 TB/sec」）；NVIDIA H100 官方页 SXM 带宽「3.35TB/s」。｜修复要求：把 3.35 TB/s 只归给 H100 白皮书/官方规格页，或据实注明博客为 3 TB/sec。｜修复：｜复验：
- [轻微·可读性] §6.5 结尾（index.html:546）：「ExpertPlex 的 APK 走的就是这条路」中 APK 为该缩写首次、也是全页唯一一次出现，未给出全称 Adaptive Persistent Kernel，读者无法确定所指。｜引文依据：不适用｜修复要求：首次出现处展开全称，或改写为「ExpertPlex 的自适应常驻内核（APK）」。｜修复：｜复验：

## 已核对来源（本轮，未发现问题）

- H100 SXM5 规格：132 SM、每 SM 4 个第四代 Tensor Core、50MB L2、80GB HBM3、3.35 TB/s、8 GPC（官方规格页 + Hopper Architecture In-Depth）。
- Hopper Tuning Guide：每 SM shared memory 228KB、单 CTA 上限 227KB、每 SM 最多 64 warp、可移植 cluster 上限 8、H100 opt-in 16、cluster 越大同时活跃 block 越少。
- CUDA Driver API cuStreamCreateWithPriority：「do not preempt already-running work or provide any other functional guarantee on execution order」——与页面第 2、6 章的两处引文一致。
- MPS 文档：动态限额「Setting the limit does not reserve dedicated resources for any MPS client context.」「Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing.」；静态分区「SMs reserved for a partition remain exclusive to the clients assigned to that partition」「other clients cannot automatically borrow idle SMs from it」「Partitions cannot be removed while clients are actively using them.」「all MPS clients must set the CUDA_MPS_SM_PARTITION environment variable before creating a CUDA context.」——页面 §6.2 各句均有对应原文。
- MIG 用户指南 H100 表：1g.10gb(7) / 1g.10gb+me(1) / 1g.20gb(4) / 2g.20gb(3) / 3g.40gb(2) / 4g.40gb(1) / 7g.80gb(1)，SM 占比 1/7；+me 含至少一个媒体引擎、每卡限 1 个。页面表格逐一相符。
- Green Context（CUDA Driver API）：CUDA 12.4 引入；CC 9.0+ smCount 为 8 的倍数；「Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees」；创建时固定、无运行期重分区接口。
- TMA / 线程块簇（Programming Guide）：tensorRank ≤ 5（支持 1 到 5 维）；簇内 block 共同调度于同一 GPC；TMA 组播把一份数据送给簇内多个 block 的 shared memory。
- 手算复算：4×4 不切 tile 为 128 次读取、切 2×2 tile 为 64 次（每元素被读 N/T=2 次），与正文及折叠块一致；128×128 fp16=32KB、256×256=128KB、512×512=512KB 复算一致；双缓冲约 128KB > 227KB 的一半。
- ExpertPlex（arXiv:2607.18002）确为真实论文，§4.1「GPU Sharing Design Space」给出 prefill GEMM 1.8–2.9ms / decode 17.7–34.7μs，与页面「大活约 2 毫秒 / 小活约 30 微秒」教学构造同构（页面已声明不引用其测量值）；[N7] 所引 NVIDIA 博客《Employing CUDA Graphs in a Dynamic Environment》真实存在。
- 机械项：validate.py 返回 validation ok；页内 $ 配对完整、唯一 display 公式为 $C[i][j]=\sum_{k=1}^{K}A[i][k]\cdot B[k][j]$ 可渲染；全页无 img（lightboxImg 的 alt 为空）；无缺失本地引用；overview.html 与 index.html 互链；moe-serving、vllm-cudagraph 两页均存在且标题与页面链接文字相符；[C1]–[C17]、[F1]、[N1]–[N7] 均双向对应。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复