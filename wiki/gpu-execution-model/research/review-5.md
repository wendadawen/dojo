<!-- review-meta
round: 5
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 0afa1e7b0a909530
-->
# GPU 执行模型与 kernel 调度审查记录（第 5 轮）

- 页面版本：9f74be29fe6c1997efe2e2d360a8d205f68a6ec1
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 先认识硬件 / 2. kernel 是怎么跑起来的——线程、warp、CTA 与 stream / 3. 为什么矩阵乘要切成 tile（3.1–3.3）/ 4. Hopper 的协作工具箱（4.1–4.3）/ 5. 反复启动太贵（5.1–5.2）/ 6. 分一张卡的四种机制（6.1–6.5）/ 来源与范围说明；含全部折叠块、flow 图注、表格与内联脚本。`dojo:type=concept`，按 `guides/concept/check.md` 审查。

## 问题

- [轻微·表述] 引言段与各章首段的元话语与「本页」自我指代（index.html:113、115、222、285、366、390、431、493）：「本页是它的前置知识页：不讲论文本身，只讲看懂它所需的 GPU 执行模型」「本页约定：kernel 指…」「真实的实现还要处理同步…本页省略」，以及每章开头的「上一章是静态的硬件。这一章回答动态的问题」「上一章说计算被切成 CTA…这一章用一个可以手算的例子给出答案」「这一章讲的都是 Hopper 架构…引入的机制」「前几章都在讲一个 kernel 内部怎么跑。这一章换一个角度」「最后一章回答开头问题的完整版」。前者是以「本页」为主语的自我指代，后者是与「本页将…」同型的章节预告式元话语。｜引文依据：不适用｜修复要求：改为直接陈述内容。例：「kernel 指在 GPU 上由大量线程并行执行的一段函数（CUDA 用法，与操作系统内核同名无涉）」「真实的实现还要处理同步与多级缓冲，正式机制以官方 TMA 与 pipeline 章节为准」；章节开场去掉「这一章…」预告，直接进入内容（如第 2 章由「CPU 上一句『启动』，怎么变成 132 块 SM 上成千上万个并行执行的线程？」起句）。｜修复：｜复验：

- [轻微·技术] 3.1 节手算例（index.html:306）：「$A_{11}$ 的 4 个元素服务了 4 次乘加」主谓数量歧义——按整块读为「4 个元素合计 4 次乘加」与算术不符（$A_{11}$ 为 $2\times2$ 块，在 $C_{11}$、$C_{12}$ 两处被用，每元素被用 $2\times2=4$ 次，整块合计 16 次乘加），只有按「每个元素服务 4 次」才自洽。同段随后「而不是读一次用一次」以 4 对 1 作比，说明原意为每元素。｜引文依据：不适用（可复算：$A_{11}$ 参与 $C_{11}=A_{11}B_{11}+A_{12}B_{21}$ 与 $C_{12}=A_{11}B_{12}+A_{12}B_{22}$，每元素在每处被用 2 次，合计 4 次）｜修复要求：明确为「$A_{11}$ 的每个元素服务了 4 次乘加」或「$A_{11}$ 整块被复用，每个元素被用 4 次」。｜修复：｜复验：

- [轻微·来源] 6.2 节（index.html:501）：「该模式在 Ampere 及更新的架构（含 Hopper）、r610 及以上驱动上支持」把官方原文中 r610 所修饰的对象改了。官方 Common Tasks / Using Static SM Partitioning 一节中，r610 只关联「部分错误隔离」，不构成该模式的可用性门槛。｜引文依据：MPS 官方文档（docs.nvidia.com/deploy/mps/615/common-tasks.html，Using Static SM Partitioning）原文：「Static SM partitioning mode allows users to create exclusive SM partitions for MPS clients on NVIDIA Ampere architecture and newer GPUs…」「Starting with Driver version r610, partial error isolation is supported when static SM partitioning is enabled.」｜修复要求：改为「该模式在 Ampere 及更新的架构（含 Hopper）上支持；自 r610 驱动起，静态分区模式下还提供部分错误隔离（同一分区独占 SM，驱动可将 SM 错误状态归属到出错分区/客户端）」。｜修复：｜复验：

## 已核对来源（本轮实际打开并定位，均无问题）

- [C1] CTA ≤ 1024 线程、CTA 驻留单 SM：CUDA C++ Programming Guide，Programming Model / Thread Hierarchy。
- [C2] warp=32、每 SM 64 warp：Hopper Tuning Guide §1.4.1.1 Occupancy 原文「The maximum number of concurrent warps per SM remains the same as in NVIDIA Ampere GPU architecture (that is, 64)」。
- [C5] 132 SM / 每 SM 4 个第四代 Tensor Core / 50MB L2 / 80GB HBM3 @ 3.35 TB/s：H100 白皮书与公开规格一致（132 SM、4 Tensor Core/SM、50MB L2、80GB HBM3、3350 GB/s）。
- [C6] 228KB shared memory/SM、227KB/CTA：Hopper Tuning Guide §1.4.1.1 原文「shared memory capacity per SM is 228 KB…」「the maximum shared memory per thread block is 227 KB」。
- [C7] TMA：Hopper Tuning Guide §1.4.1.2 原文「TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster」「a single thread can issue large data movement instructions to the TMA unit」——与「1 到 5 维」「相邻 SM 的 shared memory 之间直接传输」「只需要一个线程」逐条吻合。
- [C8] cluster 上限：Hopper Tuning Guide §1.4.1.3 原文「The maximum portable cluster size supported is 8; however, NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in.」「Using larger cluster sizes may reduce the maximum number of active blocks across the GPU.」
- [C11] stream 优先级：CUDA Driver API, cuStreamCreateWithPriority 原文「Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.」——与页面引文逐字一致。
- [C12] MPS：Dynamic Execution Resource Provisioning 原文「Setting the limit does not reserve dedicated resources for any MPS client context. It simply limits how much resources can be used by a client context. Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing.」；静态分区原文「SMs reserved for a partition remain exclusive to the clients assigned to that partition; other clients cannot automatically borrow idle SMs from it.」「Static partitioning is rejected if the target device already has active clients.」——与页面三处引用一致（r610 一处见上）。
- [C13] MIG profile 与实例数：H100 80GB 的 1g.10gb(7)/1g.10gb+me(1)/1g.20gb(4)/2g.20gb(3)/3g.40gb(2)/4g.40gb(1)/7g.80gb(1) 与页面表格逐行一致；「+me」档含媒体引擎故每卡仅一个、改配置需 GPU 无运行负载均与 MIG 用户指南/参考文档一致。3g+4g 用满 7 切片的算术成立。
- [C14] Green Context：CUDA Driver API §Green Contexts 原文「On Compute Architecture 9.0+: The smCount must be a multiple of 8…The alignment (and default value of coscheduledSmCount) is 8」「Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees.」；CUDA 12.4 引入（12.4.0 归档页存在）。
- [C15] H100 8 GPC：H100 白皮书与 NV 论坛（H100 SXM5 8 GPC / 66 TPC / 132 SM）一致。
- [C17] TMA 组播：Programming Guide, Thread Block Clusters（bulk async 操作可指定 multicast，把数据从全局内存送进 cluster 多个 block 的 shared memory）。
- [C10, N8] 「kernel 又多又短时逐次启动开销显著」：NVIDIA 开发者博客《Employing CUDA Graphs in a Dynamic Environment》原文「When kernels are many and of short duration, launch overhead sometimes becomes a problem.」——页面标注为量级说明、未给精确值，处理得当。
- 构造例的可复算性：$4\times4$ 不切 tile 读取 $16\times4=64$（A）+64（B）=128；切 $2\times2$ 后每元素 $N/T=4/2=2$ 次、$16\times2=32$（A）+32（B）=64——表格与正文一致。$128\times128$ fp16 tile $=32768$ B $=32$KB，双缓冲 $2\times(32+32)=128$KB $>227/2$KB；$256\times256\times2=131072$ B $=128$KB，双缓冲 256KB $>227$KB；$512\times512\times2=524288$ B $=512$KB——三处容量估算均正确。$M$ 8192→16384 时 $8192/128=64$、$16384/128=128$ 正确。
- 前置链接：`../moe-serving/index.html`、`../vllm-cudagraph/index.html` 均存在且标题与正文引用相符；`overview.html` 与 `index.html` 双向链接；ExpertPlex（arXiv:2607.18002，v2）真实存在，其 §4.1「GPU Sharing Design Space」讨论 prefill/decode 资源冲突，支持来源说明中「与 ExpertPlex §4.1 的场景同构但不引用其测量值」。
- 引用编号：正文用到的 [C1]–[C17]、[F1]、[N1]–[N6]、[N8] 与文末来源说明逐一对应，无悬空引用（[N7] 未使用亦未定义，无影响）。
- 机械项：`python3 .dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 `validation ok`；无 Unicode 数学字符直接出现在标题/summary/正文；图示为 HTML `.flow` 结构非等宽框线图；alt 仅出现于空 lightbox 占位 `alt=""`，无 `$...$`；两级问题均有解答折叠块且核心问题答案指明所在章；正文/来源说明未指向不存在的仓库路径。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（3 处轻微记录在案，均为表述与来源精度问题，不影响任何核心结论与数字正确性；建议后续顺带修正）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
