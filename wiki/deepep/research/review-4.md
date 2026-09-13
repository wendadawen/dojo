<!-- review-meta
round: 4
page: wiki/deepep/index.html
reviewed_content_sha256: 137703c41238dafd
-->
# DeepEP 审查记录（第 4 轮）

- 页面版本：df08099f7eb4ba66f74f916bc293b87938668d09（wiki/deepep/index.html）
- 审查时间：2026-09-13 19:36
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复；本轮未读取 research/ 下任何规划、修复或前序审查文件）
- 已完整阅读章节：核心问题（5 条及全部解答）；1. 专家并行的 all-to-all 卡在哪（1.1 dispatch 与 combine 在搬什么 / 1.2 通用集合通信库的四个不适配 / 1.3 DeepEP 是什么，负责哪一段 / 本章问题）；2. 高吞吐内核（2.1–2.5 / 本章问题）；3. 低延迟内核（3.1–3.5 / 本章问题，含固定槽位模拟代码块及其预期输出）；4. V2 重构（4.1–4.4 / 本章问题）；5. DeepEP 不解决什么（5.1–5.3 / 本章问题）；来源与范围说明。全部折叠块与图注均已通读。

## 本轮核对方法

外部来源均直取原文核对：DeepEP 仓库 main 分支 README.md 与 docs/legacy.md、commit ebfe47e 的 deep_ep/buffer.py 与 csrc/kernels/internode_ll.cu、deep_ep/buffers/elastic.py、三个 commit 的 .patch 头（日期）；DeepSeek-V3 技术报告 arXiv:2412.19437v2 全文 HTML；页面声明的 `research/sources/` 快照目录真实存在（未读取其内容）。页面 Python 代码已实际执行。以下关键项已逐条回源并留证（除本节列出的问题外，未发现其他不一致）：

- 版本与日期：ebfe47e 作者日期 `Mon, 24 Feb 2025 21:56:14 +0800`（Initial commit）、b306af0 `Thu, 30 Apr 2026 02:37:17 +0800`（"[Public release 26/04] Introducing EPv2"）、01dc3aa `Tue, 4 Aug 2026 13:31:31 +0800`——与正文「V1 2025-02-24、V2 2026-04-30」及主要依据「main 分支 commit 01dc3aa，2026-08-04」一致。
- V1 normal 内核表：legacy.md `Intranode 8 153/158 (NVLink)`、`Internode 16 43/43`、`32 58/57`、`64 51/50` 与页面 2.5 表逐格一致；V1 低延迟表 `8 77 us/98 GB/s...256 194 us/39 GB/s` 六行逐格一致。
- V2 表：README `SM90 CX7 EP 8 x 2 90/81 12`、`EP 8 x 4 61/61 6`、`SM100 CX7 EP 8 x 2 90/91 12`、`SM100 N/A EP 8 726/740 64 (Max perf)`、`643/675 24 (Min #SM)` 与页面 4.4 表逐格一致；`up to 1.3x peak performance, while saving up to 4x SM count`、`from 24 to 4 - 6`、`EP2048`、`Buffer size consumption is larger than V1`、`0 SM RDMA low-latency EP is no longer supported`、`depends on NVSHMEM to provide support for legacy methods` 均有原文。
- 网络与门槛：README `Adaptive routing ... we still recommend enabling it under all network load conditions`、`Congestion control is disabled because it hurts maximum bandwidth`、`PyTorch 2.10 and above`、`NCCL 2.30.4 and above`；legacy.md `Ampere (SM80), Hopper (SM90)`、A100 路线图 `[x] A100 support (intranode only)`——与 5.3 一致（V2 的自适应路由与拥塞控制两处均取 README，非 legacy.md 的旧表述）。
- V3 报告：§3.2.2 `NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)`；`only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink ... partition 20 SMs into 10 communication channels`；`(1) IB sending, (2) IB-to-NVLink forwarding, and (3) NVLink receiving`（combine 侧三项同源）；`a maximum of 13 experts (4 nodes × 3.2 experts/node)`；`an average of 3.2 experts per node`；§3.5.1 `we allocate 20 out of the 132 SMs available in the H800 GPU`、`Forwarding data between the IB ... and NVLink domain while aggregating IB traffic destined for multiple GPUs within the same node from a single GPU`；§3.2.1 `divide each chunk into four components: attention, all-to-all dispatch, MLP, and all-to-all combine`、`computation-to-communication ratio of approximately 1:1`、`both all-to-all and PP communication can be fully hidden during execution`；§3.4.1 `4 nodes with 32 GPUs`、`32 redundant experts for the prefilling stage`；§3.4.2 `40 nodes with 320 GPUs`、`each GPU hosts only one expert`、`64 GPUs are responsible for hosting redundant experts and shared experts`、`also exploring processing two micro-batches`；§3.3.3 `quantize the activation before MoE up-projections into FP8 and then apply dispatch components`、`we retain them in BF16 to preserve training precision`——页面公式与各处引用全部对得上。
- V1 源码：buffer.py `Enable IBGDA for the low latency mode, which refers to "no package forwarding between NVLink and RDMA"`、`NVSHMEM_IBGDA_NIC_HANDLER = 'gpu'`、`This kernel requires all the ranks ... should be visible via RDMA ... Even for ranks in the same node, NVLink are fully disabled for simplicity`、返回形状 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` 与 `hidden // 128` 缩放、`last-two-dimension of the scaling tensors are in column-major for TMA compatibility`、`we do not synchronize CPU received count`、`return_recv_hook ... without actually receiving the data`、`only two buffers ... can not hold more than 2`——C12–C17 全部有出处；internode_ll.cu 偏移式 `dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank + rank * num_max_dispatch_tokens_per_rank + slot_idx`（slot_idx 取自 `atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)`）与正文「来源 rank × num_max + 段内序号」一致。
- V2 源码：elastic.py `get_theoretical_num_sms ... based on bandwidth modeling ... This assumes a balanced gate distribution`、`For V3.0's group-limited gate, please do not use this function`、`Hybrid mode uses hierarchical RDMA + NVLink communication ... more friendly to multi-plane/multi-rail networks`——4.1–4.3 一致；README 的 handle 缓存注释 `reusing routing metadata across iterations when the gating decisions remain unchanged, avoiding redundant CPU synchronization` 与 4.2 一致。
- 代码：页面 Python 模拟已在本机执行，输出与页面「预期输出」逐字节一致（含 `slot 0/4/8/12`、`recv_count = [4,4,4,4,4,4,4,4]`、`t_0: (3.6000, 4.6000)`、`t_1: (5.0000, 7.0000)`、`16/16 一致`、`32 / 128`）；手算 0.6·(2,3)+0.4·(6,7)=(3.6,4.6) 复算相符。
- 机械项：`.dojo/scripts/validate.py wiki/deepep/index.html` 返回 `validation ok`；`dojo:topics`「并行与通信,推理系统」与 `dojo:tag`「并行与通信」在 AGENTS.md/校验脚本允许集内；正文 10 个前置概念链接与 3 个内部页面链接（fused-moe、megamoe、ultraep、moonep、aux-loss-free-routing）全部真实存在，无「（待生成）」占位；overview.html 与 index.html 互链；全文无"我们/你/本页"，无「下面来看」「需要注意的是」类元话语，无调试叙事，无 Unicode 数学字符（校验脚本通过）。

## 问题

- [重要·技术] 4.4 节 V2 性能表下方（line 713）：把 README 拓扑记法「EP 8×2」解读为「8 块 GPU 分布于 2 个节点」、「EP 8×4」为「分布于 4 个节点」，并据此断言卡数不随乘号变化。该解读与 README 同一张表自相矛盾，且给出的依据无法支撑它。｜引文依据：README 表中单节点行为 `| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf) |`（NIC 列 N/A 表示单节点，此时「EP 8」= 单节点 8 卡），而跨节点行为 `| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |`、`| SM90 | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |`；据此「EP 8」表示每节点 8 卡、乘号表示节点数（EP 8×2 = 16 卡、EP 8×4 = 32 卡）的自然读法至少同等成立，本页读法会让乘号失去意义（三种记法都只有 8 卡）。页面自称「本文解读为」并给出「依据是这两种拓扑的瓶颈带宽都标注为 RDMA」，但该依据只说明是跨节点，区分不了 8 卡与 16 卡。｜修复要求：或改用乘法读法（每节点 8 卡 × M 节点），或删去卡数解读、只保留「跨节点部署」；若确实无法从官方材料确定，须同时给出两种读法并注明不可判定，不得只给一种读法作为定论。｜修复：｜复验：

- [轻微·技术] 2.5 节末段（line 331）：「V2 的实测……在 EP 8×2 拓扑用 12 个 SM 达到 90/81 GB/s」未标出架构。README 中 EP 8×2 有两行，combine 值不同（SM90 为 81 GB/s、SM100 为 91 GB/s），读者对照 4.4 节同表时无法确认该句取自哪一行。｜引文依据：README `| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |` 与 `| SM100 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |`。｜修复要求：在本句补出架构与网卡（如「SM90 + CX7，EP 8×2」），使 90/81 与 4.4 节表格的对应行唯一。｜修复：｜复验：

- [轻微·可读性] 3.3 节（line 398）与 5.3 节（line 801）：`TMA`（line 398「列主序排布以兼容 TMA」）、`RoCE`（line 801「理论上兼容 RoCE」）、`PTX`（line 313「定制的 PTX 指令」、line 801「SM90 PTX ISA」）在首次出现处均未给全称或一句说明，而页面其余缩写（SM、warp、IBGDA、QP、TPOT、VL）都在首次使用时作了解释。其中 TMA 承载「列主序缩放因子布局」这一机制的动机，未解释会影响该结论的可读性。｜引文依据：不适用。｜修复要求：在各自首次出现处补全称或一句最小说明（TMA = Tensor Memory Accelerator；RoCE = RDMA over Converged Ethernet；PTX = NVIDIA 的 GPU 汇编中间表示）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复
- 说明：本轮未发现来源不支持的事实性论断、编号错配、数字与官方材料不符、分项之和≠合计或同页互相矛盾之处；公式（F1 与 V3 式（12）逐项一致，F3 的 4×3.2=12.8≈13 可复算）、代码（实际执行输出与页面一致）、问题块（核心问题与各章本章问题均有解答折叠块且答案独立可读）、图示（SVG 用 foreignObject 承载 KaTeX，`<text>` 仅纯文字，路径与构造示例一致）均通过。第 4 轮仅剩上述 1 个重要问题与 2 个轻微问题；重要问题关闭后可发布。
