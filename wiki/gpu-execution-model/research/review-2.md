<!-- review-meta
round: 2
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 3a0c798387f6ec14
-->
# GPU 执行模型与 kernel 调度审查记录（第 2 轮）

- 页面版本：359b6e3a186b4f2da57e31abd9b41c23e7115e79007b654ad6d1cb3349fe4a15
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理（未参与写作，未参与第 1 轮）
- 已完整阅读章节：引言 / 核心问题 / 最容易误解 / 主要依据 · 1. 先认识硬件：一块 GPU 里有什么（含本章问题）· 2. kernel 是怎么跑起来的：线程、warp、CTA 与 stream（含本章问题）· 3. 为什么矩阵乘要切成 tile：一次能搬多少数据说了算（3.1–3.3，含折叠块与本章问题）· 4. Hopper 的协作工具箱：TMA、warp 分工与 CTA cluster（4.1–4.3，含本章问题）· 5. 反复启动太贵：persistent kernel 与 CUDA Graph（5.1–5.2，含代码折叠块与本章问题）· 6. 分一张卡的四种机制：能力边界对照（6.1–6.5，含两张表与本章问题）· 来源与范围说明（C/F/N/构造示例/类比边界/简化条件）
- 独立核对的外部来源：arXiv:2607.18002（ExpertPlex，摘要与正文关键词）、docs.nvidia.com/cuda/hopper-tuning-guide、docs.nvidia.com/cuda/cuda-driver-api（CUDA__STREAM / CUDA__GREEN__CONTEXTS）、docs.nvidia.com/deploy/mps、docs.nvidia.com/datacenter/tesla/mig-user-guide（concepts）、developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth、H100 SXM5 规格多源交叉。
- 机械校验：`python3 .dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 `validation ok`；`overview.html`↔`index.html` 互链有效；页内引用的 `../moe-serving/index.html`、`../vllm-cudagraph/index.html` 两个概念页均真实存在，页面无「（待生成）」占位；页内无指向已移除 `research/` 路径的链接。已复算：4×4/2×2 tile 例子的 128→64、每元素 $N/T=2$、$16\times2=32$、折叠块 $4\times2\times4=32$；$128\times128\times2=32768$ B $=32$ KB；双缓冲 $64\times2=128$ KB；$512\times512\times2=512$ KB；$8192/128=64\to16384/128=128$。全部算式自洽。

## 问题

- [阻断·技术] 6.3 MIG 段：`H100 有 7 个可用的 GPU 计算切片（对应 7 个 GPC，另留 1 个 GPC 给管理用）`——把「切片＝GPC、另留 1 个 GPC 管理」这一结构性推断写成 [C13][N5] 来源事实，而官方来源既不支持也与官方定义相反。｜引文依据：NVIDIA MIG User Guide（Concepts 节）原文 `A GPU SM slice is the smallest fraction of the SMs on the GPU. A GPU SM slice is roughly one seventh of the total number of SMs available in the GPU when configured in MIG mode.`；该节全文不含 GPC 一词，也没有任何「保留给管理」的表述（检索确认：`The document does not mention GPC ... anywhere`）。切片＝SM 的 1/7，不是 GPC；页面同章 4.3 又称「H100 有 8 个 GPC」，两处解释缺乏一致依据。｜修复要求：删除 `（对应 7 个 GPC，另留 1 个 GPC 给管理用）` 整句括注，或改写为有原文可定位的表述（按官方定义写「7 个 GPU SM 切片，各约占全卡 SM 的 1/7」）；同时修正 [C13] 的括注，去掉无从核对的「对应 GPC」字样。｜修复：｜复验：

- [重要·技术] 3.2 节 tile 容量估算：`再加输出累加等开销，总量在百 KB 量级——已经逼近单 CTA 227KB 的上限`——算式算出的 128KB 与「逼近 227KB」不符（约 56%），且把「输出累加」计入 shared memory 与实现不符（GEMM 累加器驻留寄存器，不占 shared memory）。｜引文依据：同段自述 `一个 128×128 的 A tile 占 128×128×2 = 32768 字节 = 32KB；B tile 同样 32KB；……搬运缓冲翻倍约 128KB`；Hopper Tuning Guide 原文 `the maximum shared memory per thread block is 227 KB`（128 KB ≠ 逼近 227 KB）。｜修复要求：三者取一——(a) 删除「已经逼近单 CTA 227KB 的上限」；(b) 把缓冲深度写清（如「三段缓冲共约 192KB」）使数字真能逼近上限；(c) 明确累加器在寄存器、不占 shared memory，并把「逼近上限」的依据改为寄存器/累加器开销。改后所有数字须与该节其余算式（$128\times128\times2$、$512\times512\times2$）保持同一套口径。｜修复：｜复验：

- [轻微·表述] 3.2 节正文：`跟你要算的矩阵总共有多大毫无关系`——直接使用第二人称称呼读者，违反 style-guide §12「不直接使用第二人称称呼读者」。｜引文依据：不适用（全文仅此 1 处「你」）。｜修复要求：改为无主语表述，如「与所算矩阵的总规模毫无关系」。｜修复：｜复验：

- [轻微·表述] 通读全页（含折叠块与图注）发现的元话语与固定句式：(a) 6.5 节末 `本页的任务到此为止。`；(b) 3.3 节 `记住这个结论，第 6 章末尾和 ExpertPlex 论文页都会用到它。`（面向读者的指令）；(c) 4.2 节 `教学解释。`；(d) `贯穿例子推进：` 作为固定引导句在第 2、3、4、5 章各出现一次。｜引文依据：不适用（对应 style-guide §8「不使用固定句式，也不为形式完整而添加过渡」）。｜修复要求：删除/改写 (a)(b)(c) 三处元话语为陈述句；(d) 四次重复的固定引导句改为各不相同、直接承接上文的过渡句（或直接并入段落，不加引导语）。｜修复：｜复验：

- [轻微·格式] 前置 section 顺序：`blockquote.meta`（主要依据）位于 `<section class="misconceptions">` 之后、第 1 章之前（第 162 行），不符合 style-guide §2 固定顺序「reading-time → blockquote.meta → 引言 → learning-goals → misconceptions → 正文」。同仓库 `wiki/mla/index.html` 的 meta 块在第 76 行、紧随 reading-time。｜引文依据：不适用。｜修复要求：把 `<blockquote class="meta">` 整块移到 `reading-time`（第 105 行）与引言首段（第 107 行）之间。｜修复：｜复验：

- [轻微·格式] 正文来源引用未按 style-guide §6 写成 `<sup>[Cx]</sup>` 上标，全页一律写成裸文本 `[C5][N1]`、`[C11]` 等（如第 170、176、178、252、359、383、486、494、514、530 行）。对比 `wiki/mla/index.html` 使用 `<sup>[C2]</sup>`。｜引文依据：不适用。｜修复要求：将正文（含折叠块与表格单元格）中所有 `[Cx]/[Nx]/[Fx]` 引用改为 `<sup>[Cx]</sup>` 形式，来源章节中的定义列表保持现状。｜修复：｜复验：

- [轻微·格式] 标题格式：h1 为 `GPU 执行模型与 kernel 调度`，缺 style-guide §1 要求的「概念名（英文缩写）：核心作用或结论」中的括注与冒号副标题（同仓库 mla/kv-cache/chunked-prefill 的 h1 均带 `（缩写）：副标题`）；6 个章级 h2 的副标题一律用全角冒号 `：`（如 `1. 先认识硬件：一块 GPU 里有什么`），而 §1 规定副标题用 `——`（mla 的 h2 为 `1. MLA 压缩了什么——KV 联合压缩的核心机制`）。｜引文依据：不适用。｜修复要求：h1 补为 `GPU 执行模型（英文缩写）：<核心结论>` 形式（与 `<title>` 中已有副标题一致即可）；6 个 h2 的 `：` 改为 `——`，章节编号与锚点 id 不变。｜修复：｜复验：

- [轻微·格式] 来源编号缺口：`主要依据` 括注声明 `正文中的 [C1]–[C16]、[N1]–[N8] 编号在文末「来源与范围说明」逐条对应`，但 [N1]–[N6]、[N8] 有定义、[N7] 既无定义也未被引用（全文 `[N7]` 出现 0 次，`[N/c]` 计数确认 [N1] 5 次、[N2] 4 次、[N3] 2 次、[N4] 2 次、[N5] 2 次、[N6] 2 次、[N8] 3 次）。｜引文依据：不适用。｜修复要求：或在「外部数字与实验条件（N）」补上 [N7] 的实际条目，或把声明范围改为 `[N1]–[N6]、[N8]`，使声明与来源章节一致。｜修复：｜复验：

- [轻微·格式] 分类词表取值与内容不符：`dojo:topics="数学基础"`、`dojo:tag="数学与数值"`，但本页讲的是 GPU 硬件层级与 kernel 调度（同目录同类页如 `wiki/chunked-prefill` 取 `推理系统, 并行与通信`、`wiki/vllm-cudagraph` 取 `推理系统`、`wiki/fused-moe` 取 `推理系统`）。取值本身在 AGENTS.md 允许列表内，故非硬性违规，但分类明显偏离内容。｜引文依据：不适用。｜修复要求：把 `dojo:topics` 改为 `推理系统`（可加 `并行与通信`），`dojo:tag` 改为封闭词表中的 `推理系统`（值须取自 `.dojo/scripts/catalog_builder.py` 的 `ALLOWED_TAGS`）。｜修复：｜复验：

## 已核对通过的条目（供复验参照）

- 1 章数字：132 SM / 每 SM 4 个第四代 Tensor Core / 50MB L2 / 80GB HBM3 @ 3.35 TB/s——与 `NVIDIA Hopper Architecture In-Depth`（`8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs per GPU`、`50 MB L2 cache`、`80 GB HBM3`）及多源 H100 SXM5 规格一致。
- 228KB/227KB：Hopper Tuning Guide 原文 `shared memory capacity per SM is 228 KB`、`the maximum shared memory per thread block is 227 KB`。
- warp=32、每 SM 最多 64 warp：`The maximum number of concurrent warps per SM ... 64`。
- cluster 上限 8 / H100 opt-in 16：`The maximum portable cluster size supported is 8; however, NVIDIA Hopper H100 GPU allows for a nonportable cluster size of 16 by opting in.`
- TMA 1–5 维、跨 SM 传输、单线程发起：`TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory`、`as well as between the shared memory regions of different SMs in the same cluster`、`a single thread can issue large data movement instructions to the TMA unit`。
- stream 优先级引文：`Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.`
- MPS 引文：`Setting the limit does not reserve dedicated resources for any MPS client context. It simply limits how much resources can be used by a client context. Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing.`
- Green Context：CUDA 12.4 引入（CUDA 12.4/12.4.1 Release Notes）；`For Compute Architecture 9.0+: The smCount must be a multiple of 8 ... The alignment ... is 8.`；`Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees.`
- MIG profile 表（1g.10gb×7、1g.20gb×4、2g.20gb×3、3g.40gb×2、4g.40gb×1、7g.80gb×1）与多来源 H100 80GB 表一致；「用满 7 切片即整卡」「改配置需 GPU 空闲」与 MIG 语义一致。
- 三级问题块齐备：页面级「核心问题」5 条、6 个正文章节各「本章问题」均有 `解答：` 折叠块，答案独立可读，核心问题答案均指明完整论证所在章节；折叠块 summary 前缀只用 `解答：`/`补充：`/`代码：`，符合 §5。
- 代码块自述「伪代码，不可运行」并说明省略项，符合 §2.2 第 3 条对不可运行代码的要求。
- 结构图为 HTML `div.flow`，无等宽字符框线图；KaTeX 定界符只用于行内/独立公式，`validate.py` 未报公式字符错误（`×`、`→` 按 `.dojo/scripts/validate.py` 注释属中文散文普通排版字符，不列入）。
- ExpertPlex 定位成立：arXiv:2607.18002 标题 `ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels`，确以 APK 在 tile 粒度调度，与页面「ExpertPlex 的 APK 走的就是这条路」相符。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 7
- 处置：修复（阻断 1 条须先删除或改写 6.3 的 GPC 归因并同步 [C13] 括注；重要 1 条须使 3.2 的容量数字与算式同口径）
