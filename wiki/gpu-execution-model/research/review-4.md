<!-- review-meta
round: 4
page: wiki/gpu-execution-model/index.html
reviewed_content_sha256: 6d4df158ba60d430
-->
# GPU 执行模型与 kernel 调度审查记录（第 4 轮）

- 页面版本：652add5841c44d3a06525973bee6b8e5b24bf9ac（工作树 index.html）
- 审查时间：2026-09-13 20:16
- 审查者：独立子代理（未参与写作，也未参与前三轮审查；仅使用本页、overview.html、页面引用的外部来源与本规范）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. 先认识硬件——一块 GPU 里有什么 → 2. kernel 是怎么跑起来的——线程、warp、CTA 与 stream → 3. 为什么矩阵乘要切成 tile——一次能搬多少数据说了算 → 4. Hopper 的协作工具箱——TMA、warp 分工与 CTA cluster → 5. 反复启动太贵——persistent kernel 与 CUDA Graph → 6. 分一张卡的四种机制——能力边界对照 → 来源与范围说明（含全部折叠块、图注、每个「解答：」与 [C]/[F]/[N] 编号）

## 问题

- [重要·技术] 第 501 行 6.2 节、第 537 行 6.5 对照表 MPS 行、第 152 行核心问题 5 答案：把「MPS 不为客户端预留专用资源」写成 MPS 的整体属性（对照表「关键短板」栏直接写「不预留资源」），据此得出「MPS 提供的是『可以挤在一起』，不是『各自圈地』」。｜引文依据：NVIDIA MPS 官方文档同一册中除被引的 Dynamic Execution Resource Provisioning 外，另有 Static SM Partitioning 一节：「Static SM partitioning mode allows users to create exclusive SM partitions for MPS clients on NVIDIA Ampere architecture and newer GPUs, providing deterministic resource allocation and improved isolation.」以及「SMs reserved for a partition remain exclusive to the clients assigned to that partition; other clients cannot automatically borrow idle SMs from it.」（该节同时注明 Ampere 及更新架构、r610+ 驱动支持，Hopper 在支持范围内）。页面所引「Setting the limit does not reserve dedicated resources for any MPS client context」只适用于动态资源限额这一种机制。｜修复要求：限定 MPS 结论的适用口径——说明「资源限额不预留专用资源」限于动态资源供应；补充 MPS 另有静态 SM 分区模式，可在客户端创建上下文前划分独占 SM 分区、使用期间不能移除；同步修正对照表 MPS 行「关键短板」与核心问题 5 答案，并相应扩充 [C12] 的来源说明。｜修复：§6.2 改写为「MPS 有两套资源机制」——动态资源供应（限额「不为任何客户端预留专用资源」）与静态 SM 分区（Ampere 及更新架构含 Hopper、r610+ 驱动；在客户端创建上下文前划分独占 SM 分区，分区内 SM 为该分区客户端独占、其他客户端不能自动借用空闲 SM，使用期间不能移除），并明确「不预留专用资源」只对动态限额成立、默认形态是共用 SM、只有事先显式配置静态分区才谈得上圈地。§6.2 标题、§6.5 对照表 MPS 行（切分对象补「另有静态 SM 分区」、运行期栏补「静态分区须在客户端创建前配置」、关键短板改为「动态限额不预留资源」）、核心问题 5 答案、§6.5 归纳（「MPS 不预留」改为「MPS 动态限额不预留资源、静态分区须提前配置且创建后固定」）、§6 本章问题 1 答案同步限定口径；[C12] 来源说明补静态 SM 分区一节及其条件；overview.html 关键结论同步限定。｜复验：

- [轻微·技术] 第 396 行正文「配合 TMA 的组播能力，同一份数据可以一次发给 cluster 内多个 CTA [C7]」与第 578 行 [C7] 来源说明「TMA 异步拷贝与组播」：把组播能力归到 Hopper Tuning Guide §1.4.1.1–§1.4.1.3，但该指南不含此内容。｜引文依据：Hopper Tuning Guide（CUDA 13.4 版，页脚 Last updated on Aug 14, 2026）全文检索 "multicast"、"broadcast" 均为 0 次，§1.4.1.2 只写到「TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster」；组播实际载于 CUDA C++ Programming Guide（Thread Block Clusters 章）：「can be specified as being multicast. In this case, data can be transferred from global memory to the shared memory of multiple blocks within the cluster.」｜修复要求：把「组播」的引文锚点改指 CUDA C++ Programming Guide 的 thread block clusters / TMA 小节，或另立编号；[C7] 括注中删去「组播」。论断本身有官方依据，不需要删除。｜修复：正文「配合 TMA 的组播能力……<sup>[C7]</sup>」改为 <sup>[C17]</sup>；新增编号 [C17]（CUDA C++ Programming Guide，Thread Block Clusters：TMA 可将数据从全局内存组播到 cluster 内多个 block 的 shared memory），§4 本章问题 2 答案的同一论断同样补 <sup>[C17]</sup>；[C6][C7][C8][C9] 括注删去「与组播」；主要依据行的编号范围 [C1]–[C16] 改为 [C1]–[C17]。｜复验：

- [轻微·功能] 六个章节的「本章问题」标题（第 202、258、339、404、466、550 行）写作不带 id 的 `<h3>本章问题</h3>`：目录脚本（页面内联 JS）对无 id 的标题按 textContent 生成锚点，六处生成同一个 id，产生重复 id，且侧边目录中六个「本章问题」子项全部指向第 1 章的锚点。｜引文依据：不适用（页面内联 JS：`if (!h.id) { h.id = h.textContent.trim()... }`；仓库中 98 个概念页里 79 个已给「本章问题」写入各自唯一的 id，如 `id="accounting-and-conditions-questions"`）。｜修复要求：为每个「本章问题」`<h3>` 补一个唯一 id（沿用仓库多数页面的 `<章名>-questions` 形式），使目录子锚点各自指向本卷。｜修复：六处「本章问题」h3 补唯一 id：hardware-questions、launch-questions、tile-questions、hopper-tools-questions、persistent-questions、sharing-questions（沿用 < 章 h2 的 id >-questions 形式）。｜复验：

- [轻微·表述] 正文存在指向页面自身结构的元话语与对读者的指令，非章节衔接所必需：第 200 行「回到开头的大小活例子：……」、第 254 行「另一个需要记住的语义：……」、第 113 行「上面的 2 毫秒与 30 微秒……」「本页只在需要动机时点名」。｜引文依据：不适用。｜修复要求：删去或改写这三处，使正文直接陈述内容本身（例如「大活和小活的输入矩阵此刻都躺在 80GB 的 HBM 里」「同一 stream 内的操作严格按提交顺序执行」），不叙述页面结构、不向读者下「记住」的指令。｜修复：§1 末段删去「回到开头的大小活例子：」，改为直陈「大活和小活的输入矩阵，此刻都躺在 80GB 的 HBM 里……」；§2 删去「另一个需要记住的语义：」，直接以「同一个 stream 里的操作严格按提交顺序执行……」开句；开篇段「上面的 2 毫秒与 30 微秒是……」改为「2 毫秒与 30 微秒都是……」，并删去句末「本页只在需要动机时点名」。｜复验：

## 本轮已核对且未发现问题的项

- 硬件数字：132 SM、每 SM 4 个第四代 Tensor Core、8 GPC、50MB L2、80GB HBM3 @ 3.35 TB/s（NVIDIA Hopper Architecture In-Depth 与 NVIDIA H100 数据中心页规格表「Memory 80GB 94GB GPU Memory Bandwidth 3.35TB/s 3.9TB/s」）；228KB/SM 与单 CTA 227KB、每 SM 64 warp（Hopper Tuning Guide §1.4.1.1）；CTA ≤ 1024 线程、warp=32、cluster 保序落同一 GPC（CUDA C++ Programming Guide）。
- 三、四、五、六章机制：cluster 可移植上限 8、H100 非移植 opt-in 16 且「may reduce the maximum number of active blocks across the GPU」（Hopper Tuning Guide §1.4.1.3）；TMA 1–5 维、单线程发起、可在同 cluster 内不同 SM 的 shared memory 之间搬（同上 §1.4.1.2）；stream 优先级原文「do not preempt already-running work or provide any other functional guarantee on execution order」（CUDA Driver API）与编程指南「Higher-priority tasks do not preempt already running lower-priority tasks」一致；MIG 七档 profile 与实例数（1g.10gb=7、1g.10gb+me=1、1g.20gb=4、2g.20gb=3、3g.40gb=2、4g.40gb=1、7g.80gb=1）与 +me 含媒体引擎（MIG User Guide 支持 profile 表）；Green Context CUDA 12.4 引入（CUDA 12.4 release notes「Green contexts are a lightweight alternative to traditional contexts」）、CC 9.0 上 smCount 为 8 的倍数、「Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees」（CUDA Driver API）。
- 公式与手算：第 3 章 $4\times4$ 切 $2\times2$ tile 的读取计数（不切 128 次、切后 64 次、每元素 $N/T$ 次）、$C_{11}=A_{11}B_{11}+A_{12}B_{21}$、$C_{12}=A_{11}B_{12}+A_{12}B_{22}$ 与折叠块逐 tile 清单一致；3.2 节 $128\times128\times2=32$KB、双缓冲 128KB 占 227KB 上限一半以上、$256\times256\times2=128$KB、$512\times512\times2=512$KB 全部可复算；3.3 节 8192/128=64、16384/128=128 正确。
- 构造示例与来源分离：开篇 2 毫秒/30 微秒标注为教学构造；4×4 例、fp16 容量估算在「构造示例」中声明；两处推断（第 335、544 行「本页归纳」、第 402 行「（推断）」）均已显式降级；ExpertPlex（arXiv:2607.18002）确为真实论文《ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels》。
- 格式与功能：`.dojo/scripts/validate.py wiki/gpu-execution-model/index.html` 返回 `validation ok`；dojo:type=concept、dojo:topics=推理系统、dojo:tag=推理系统 均在词表内；overview.html 与 index.html 互链；页内链接 ../moe-serving/index.html、../vllm-cudagraph/index.html 均存在且链接文字与目标 h1 一致；无「（待生成）」占位，正文与来源说明未指向 research/ 下路径；结构图为 HTML 块，公式全部由 KaTeX 渲染（`×`、`→` 按 validate.py 与 style-guide 属中文技术散文的普通排版字符，不计入违规）。
- 表述已复核的其余部分：无第一人称复数、无第二人称称呼读者、无调试/复现叙事；章节间过渡为跨章引用，属规范鼓励的衔接，未计为元话语。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（先关第 1 条重要项，再处理三条轻微项；修复后重跑 validate.py 并从修复后的完整页面开始下一轮，如仍需追加轮次则按规范计入）