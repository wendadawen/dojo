<!-- review-meta
round: 3
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 26f36bb19a85d2d9
-->
# GPU 执行模型与 kernel 调度审查记录（第 3 轮）

- 页面版本：edfd6eb885926dcabe82f1f878cbe9a3bb9acefa
- 审查时间：2026-09-13 19:40
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 先认识硬件——一块 GPU 里有什么 / 2. kernel 是怎么跑起来的——线程、warp、CTA 与 stream / 3. 为什么矩阵乘要切成 tile——一次能搬多少数据说了算 / 4. Hopper 的协作工具箱——TMA、warp 分工与 CTA cluster / 5. 反复启动太贵——persistent kernel 与 CUDA Graph / 6. 分一张卡的四种机制——能力边界对照 / 来源与范围说明（含全部折叠块、图注、伪代码块与脚本）

## 来源核对记录（本轮实际打开并定位到的原文/数值）

- Hopper Tuning Guide：shared memory “per SM is 228 KB”“the maximum shared memory per thread block is 227 KB”；“The maximum portable cluster size supported is 8; however, NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in”；“Using larger cluster sizes may reduce the maximum number of active blocks across the GPU”；TMA “transfer 1D and up to 5D tensors between global memory and shared memory”“between the shared memory regions of different SMs in the same cluster”；小节号 1.4.1.1 Occupancy / 1.4.1.2 Tensor Memory Accelerator / 1.4.1.3 Thread Block Clusters，与页面 [C6][C7][C8]、[C2] 引用的节号一致。
- CUDA Driver API（cuStreamCreateWithPriority）原文：“Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.”——与页内 [C11] 中文引用逐句一致。
- CUDA Driver API（Green Contexts）：CC 9.0+ “The smCount must be a multiple of 8”；“Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees.”；CUDA 12.4 引入（Release Notes）——与 [C14] 一致。
- MIG User Guide（H100 80GB SXM5 表）：七档 profile 与每卡最大实例数 1g.10gb 7 / 1g.10gb+me 1 / 1g.20gb 4 / 2g.20gb 3 / 3g.40gb 2 / 4g.40gb 1 / 7g.80gb 1；“compute resources divided into seven slices”、memory “eight slices”；改配置需 GPU 无运行中进程（“In use by another client”）——与页内表 [C13] 逐行一致。
- MPS 文档：“Setting the limit does not reserve dedicated resources for any MPS client context… Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing.”——与 [C12] 一致。
- NVIDIA 开发者博客《Employing CUDA Graphs in a Dynamic Environment》（developer.nvidia.com，2021-11-03）：标题存在，“many short-running kernels…launch overhead becomes significant”——与 [N8] 一致。
- arXiv:2607.18002 = “ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels”，摘要含 “adaptive persistent kernels to schedule dynamic expert computation at tile granularity”——与页面把 ExpertPlex 当作 tile 级调度动机的定位一致。
- 手算复核：4×4 不切 tile A 读 16×4=64、合计 128；切 2×2 后每元素读 N/T=2、合计 64；128×128×2=32768B=32KB、双缓冲≈128KB（>227KB 之半）；256×256×2=131072B=128KB；512×512×2=512KB；8192/128=64、16384/128=128——全部与页面一致。
- 代码：5.1 节代码块显式标注「伪代码，不可运行」，按 check.md 2.2.3 改静态审查，未执行。
- 机械项：`.dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 “validation ok”，退出码 0；页面链接 ../moe-serving/index.html、../vllm-cudagraph/index.html 目标均存在；页面级「核心问题」5 条、每章「本章问题」均有解答折叠块且 summary 前缀合规；无「（待生成）」占位。

## 问题

- [重要·技术] 第 5 章「本章问题」第 1 题解答 summary（"一个摊 GPU 侧启动，一个砍 CPU 侧提交"）：把 persistent kernel 的收益窄写成"摊 GPU 侧启动"，与同页另两处口径不一——核心问题第 3 题答案写"把启动开销从每任务一次摊成全程一次"，5.1 正文写"启动开销从「每任务一次」摊成「全程一次」"，均不加侧别。常驻内核只启动一次，CPU 侧"准备参数、提交到 stream"同样从每任务一次降为一次，并非只省 GPU 侧。｜引文依据：页内 5.1 正文"persistent kernel 只启动一次、在内部循环领任务…启动开销从「每任务一次」摊成「全程一次」"；核心问题第 3 题答案同义；CUDA Graph 消除的才是"逐个 kernel 的 CPU 提交开销"（[C10]）。｜修复要求：将 summary 与解答改为对 persistent kernel 不加侧别限定的表述（如"一个摊启动、一个砍提交：常驻内核让 CPU 提交与 GPU 分发都只付一次，CUDA Graph 消除逐次 CPU 提交"），或补一句说明为何把它归为 GPU 侧，使三处口径一致。｜修复：5 章本章问题 Q1 的 summary 改为「一个摊启动，一个砍提交」，解答正文把摊掉的开销明确写成「CPU 侧的准备参数与提交、GPU 侧的 grid 分发与收尾」，与核心问题 Q3 及 5.1 正文口径一致｜复验：
- [重要·技术] 5.2 节末（"两种手段是正交的…共同点是都偏好「形状固定」的负载"）：把"偏好形状固定"同时归给 persistent kernel 与 CUDA Graph。后半句对 persistent kernel 不成立：同页 5.1 描述它"不断从全局内存的任务队列里领取下一个任务"，任务可变；其价值恰在处理形状可变/动态的负载，把"偏好形状固定"写成共同点会让读者误以为常驻内核只适配固定形状。｜引文依据：ExpertPlex 摘要 arXiv:2607.18002 "adaptive persistent kernels to schedule dynamic expert computation at tile granularity"（动态、tile 粒度）；页内 5.1"从 Q 领取下一个任务…按任务描述搬入数据 → 计算 → 写回结果"。｜修复要求：删去或改写"共同点是都偏好「形状固定」的负载"，限定为"CUDA Graph 要求形状固定"；persistent kernel 的约束写成"需软件自行管理任务队列与让出时机"。｜修复：删去「共同点是都偏好形状固定」，改为「形状必须固定」是 CUDA Graph 独有的约束，persistent kernel 不受此限、任务形状可变正是其用武之地，代价是软件自行管理任务队列与让出时机｜复验：
- [轻微·技术] 6.3（"带 +me 的一档额外包含至少一个媒体引擎（NVDEC、NVENC、NVJPEG 或 OFA）"）：枚举含 NVENC，但 H100 没有 NVENC（0 个），把该型号不具备的引擎列为可能项。｜引文依据：H100 80GB SXM 媒体引擎为 0 NVENC / 7 NVDEC / 7 NVJPEG / 1 OFA（NVIDIA MIG User Guide H100 表；NVIDIA Dynamo 视频解码文档同述数据中心卡无 NVENC）。｜修复要求：从枚举中删去 NVENC，或限定为"该档在 H100 上包含 NVDEC、NVJPEG 或 OFA"。｜修复：枚举限定为「H100 上为 NVDEC、NVJPEG 或 OFA」，删去该型号不具备的 NVENC｜复验：
- [轻微·技术] 第 2 章末（"小活的 CTA 此刻提交到另一个 stream，只能排队等空位"）：说"CTA 提交到 stream"。提交到 stream 的是 kernel，CTA 是启动后由硬件分发器分配到 SM；此处把两级动作并成一句。｜引文依据：同页 2 章图注"① 提交到一个 stream…② 硬件分发器把 grid 拆成一个个 CTA"。｜修复要求：改为"小活的 kernel 提交到另一个 stream，其 CTA 只能排队等空位"。｜修复：改为「小活的 kernel 此刻提交到另一个 stream，它的 CTA 只能排队等空位」｜复验：
- [轻微·技术] 「简化条件及其限制」（"驱动级时间片（compute preemption，粒度粗且不可编程控制）"）：把 compute preemption 描述为"粒度粗"。NVIDIA 的 compute preemption 是指令级（自 Pascal 起，CILP），粒度细；粗粒度的是上下文/时间片级抢占。此处把两者混同且粒度描述相反。｜引文依据：NVIDIA 驱动/架构说明：compute preemption 为 instruction-level，上下文切换约 0.1 ms 量级。｜修复要求：拆开表述——"驱动按时间片/上下文切换抢占（粗粒度、不可编程控制）"与"compute preemption（指令级、不可编程控制）"分开，或删去对其粒度的描述。｜修复：拆为「驱动按时间片/上下文切换抢占（粗粒度、不可编程控制）」与「compute preemption（指令级、不可编程控制）」，不再混同｜复验：
- [轻微·格式] 5.2 节（链接文字"torch.compile 与 CUDA Graph 笔记"，指向 ../vllm-cudagraph/index.html）：链接文字与目标页标题不一致。｜引文依据：wiki/vllm-cudagraph/index.html 的 title 为"torch.compile 图捕获与 CUDA Graph · Dojo"，站内 deepep、vllm-mm-unified-embeds-cudagraph 两页均用该规范名。｜修复要求：链接文字改用目标页规范名"torch.compile 图捕获与 CUDA Graph"。｜修复：链接文字改为「torch.compile 图捕获与 CUDA Graph」，与目标页 title 一致｜复验：
- [轻微·格式] 第 1 章（"这就是 GPU「 massively parallel 」的全部含义"）：引号内有前后多余空格，与全页其他「」（「大活」「搬数据」「每任务一次」等）写法不一致。｜引文依据：不适用。｜修复要求：改为「massively parallel」。｜修复：改为「massively parallel」，去掉引号内前后空格｜复验：
- [轻微·格式] 全文：自称混用"本页"（7 处）与"本文"（第 3 章、6.5 节、"来源与范围说明"共 3 处）。｜引文依据：不适用。｜修复要求：统一为"本页"（或统一为"本文"）。｜修复：3 处「本文」全部改为「本页」，与其余 7 处统一｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 6
- 处置：修复（2 条重要问题修复并复验后方可发布；6 条轻微问题随轮修复）。核心结论（CTA 运行至完成、tile 尺寸由容量决定、四种共享机制无运行期细粒度重分配）与全部数字经回源核对无误，两处问题均属第 5 章机制归因的表述精度，未推翻主线。