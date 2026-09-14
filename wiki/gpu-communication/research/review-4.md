<!-- review-meta
round: 4
page: wiki/gpu-communication/index.html
reviewed_content_sha256: 1535e35ce13df381
-->
# GPU 通信审查记录（第 4 轮）

- 页面版本：916f381eefa518d4e14b82fd65cdb03ee99ad65c
- 审查时间：2026-09-14 16:54
- 审查者：独立子代理（未参与写作与前序审查）
- 适用规范：guides/note.md（dojo:type = note）
- 已完整阅读章节：分层与主线：谁调度、谁搬运 → 通信原语：搬、算、换三类动作（含「八个原语的完整定义」）→ 环形 all-reduce：带宽最优的两阶段算法（含阶段一 reduce-scatter、阶段二 all-gather、通信量、其他算法与选择）→ 机内互联：总线、直连与中心交换 → 跨机传输：CPU 参与与绕过 CPU 两条路径（含路径一 socket/TCP/IP/以太网、路径二 RDMA 家族）→ 网卡：一个角色，多个名字与用法 → 通信库：集合通信库与传输库的分工 → 来源

## 核对依据（来源各段已逐条打开核对）

- 环形 all-reduce 通信量：Baidu all-reduce 说明原文「Data Transferred = 2(N−1)K/N」，两阶段各 N−1 步。页面公式 $2(N-1)/N$、4/8/128 卡的 1.5/1.75/1.98 倍、两阶段各 3 步、朴素方案 N−1 倍（3/7/127 倍）均可复算吻合。
- NCCL：NVIDIA 博客《Fast Multi-GPU collectives with NCCL》原文 "three primitives: Copy, Reduce, and ReduceAndCopy" 与 "NCCL implements ring-style collectives"，与来源条一致。
- PyTorch torch.distributed 文档（docs.pytorch.org/docs/2.14/distributed.html）：后端正交表 gloo 列 CPU 支持、NCCL 列 GPU 支持；"PyTorch distributed package supports Linux (stable), macOS (stable), and Windows (prototype)"。页面 gloo「CPU 训练、跨平台」表述成立。
- NVIDIA GPUDirect RDMA 文档：「a direct path for data exchange between the GPU and a third-party peer device」、nvidia-peermem 让 HCA 直读显存「without needing to copy data to host memory」。页面「GPU 与网卡等对端设备之间建立直接路径」一致。
- NIXL：ai-dynamo/nixl 仓库 README 确认 "NVIDIA Inference Xfer Library"、后端含 UCX/GDS/UCCL/LIBFABRIC；NVIDIA Dynamo 文档确认 TensorRT-LLM KV 缓存传输默认走 NIXL + UCX 后端。页面「UCX 是默认传输后端，另有 LIBFABRIC、GDS、UCCL 等可选后端」成立。
- TCCL：腾讯云《TCCL 使用说明》确认「API 与 NCCL 完全兼容」「无法通过替换共享库的方式使用 TCCL」；星脉网络官方披露（多家媒体报道）确认「TCCL 基于开源 NCCL 代码扩展优化」及 AllReduce/AllGather/ReduceScatter 常用通信模式约 40% 性能提升。页面 40% 数值与兼容性/接入方式表述均可定位到来源。
- 机内互联：C(8,2)=28 条两两直连、中心交换每卡一条计 8 条，均正确。「出网必经 PCIe」与页面 PCIe/NVLink 定位一致。
- 内部链接 ../deepep/index.html、../model-parallelism/index.html 与 ../../index.html 均真实存在。
- 机械项：`.dojo/scripts/validate.py wiki/gpu-communication/index.html` 返回 "validation ok"；全页无元话语/会话指代/口语化检索命中；无 img 承载公式；aria-label、<title> 内无 $...$；summary 公式 $\frac{2(N-1)}{N}$ 与正文 $\log_2 N$ 均为合法 KaTeX。

## 问题

- [轻微·技术] 网卡章节「为什么需要网卡这个设备而不是让 CPU 直接处理网络」段：把机器内部信号描述为「PCIe 总线信号（短距离、并行）」、外部为「线缆信号（长距离、串行、带协议）」，以「并行 vs 串行」区分内外信号不准确——PCIe 物理层本身是串行差分链路（并行性来自多 lane，与多对线缆以太网同理）；且该段两条归因（信号转换、协议卸载）在本页来源清单中无对应来源，属未标注的自述机制。｜引文依据：原文「机器内部是 PCIe 总线信号（短距离、并行），外部是线缆信号（长距离、串行、带协议），两套规格需要硬件转接」；来源清单无对应条目。｜修复要求：删除「并行」这一与 PCIe 串行特性不符的限定（或改为「多 lane 短距总线」），并在来源清单补充该段归因的依据，或将其标注为通用工程推断。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（本页无阻断与重要问题；来源论断、公式、数值、内部链接与渲染均核对通过，仅存 1 条不影响主线理解的轻微表述/来源问题）
