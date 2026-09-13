<!-- review-meta
round: 2
page: wiki/gpu-communication/index.html
reviewed_content_sha256: 93aab27ab955f1bf
-->
# GPU 通信审查记录（第 2 轮）

- 页面版本：609905d5664032d94aef34a681ea2d23e5defb7e
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 适用规范：guides/note.md（页面 `dojo:type` = note）
- 已完整阅读章节：分层与主线：谁调度、谁搬运 / 通信原语：搬、算、换三类动作 / 环形 all-reduce：带宽最优的两阶段算法 / 机内互联：总线、直连与中心交换 / 跨机传输：CPU 参与与绕过 CPU 两条路径 / 网卡：一个角色，多个名字与用法 / 通信库：集合通信库与传输库的分工 / 来源（含两处内联 SVG、全部表格与折叠按钮）
- 机械验证：`.dojo/scripts/validate.py wiki/gpu-communication/index.html` → `validation ok`；无头 Chrome 实测渲染，2 处 KaTeX 公式、2 张内联 SVG、5 张表格均正常渲染，无 `katex-error`；页内链接 `../deepep/index.html`、`../model-parallelism/index.html` 均存在；页面未指向任何已移除的 `research/` 路径。

## 问题

- [阻断·技术] 来源：腾讯云官方文档《GPU 型实例安装 TCCL》｜位置：通信库章节「集合通信库与传输库的分工」表格 TCCL 行，及其后正文末段｜问题：页面称 TCCL「接口与 NCCL 完全一致、替换即用」，与所引官方来源直接矛盾——腾讯官方明确说明 TCCL 不能通过替换共享库使用。该论断把「接口兼容」扩写成「库级替换即用」，是来源不支持的机制描述。｜引文依据：腾讯云文档原文 "As the community pytorch connects to the NCCL communication library statically by default, TCCL cannot be used by replacing the shared libraries."（腾讯云文档 1236/62821）；页面原文「接口与 NCCL 完全一致、替换即用」。｜修复要求：删除「替换即用」，改为与来源一致的表述，例如「接口与 NCCL 完全兼容，需通过重编 PyTorch、torch_tccl 插件或 NCCL 插件三种方式接入，不能直接替换共享库」。｜修复：｜复验：

- [重要·技术] 来源：腾讯云官方文档《GPU 型实例安装 TCCL》｜位置：通信库章节正文末段「TCCL 存在的理由是……定制版按相同接口重写后无需改应用代码」｜问题：把 TCCL 描述为「按相同接口重写」的定制实现，与官方定性相反——官方说明 TCCL 是在开源 NCCL 代码基础上扩展优化而来，属派生实现而非重写。｜引文依据：腾讯云文档原文 "TCCL has been expanded and optimized based on open source NCCL code, ensuring full compatibility with NCCL's features and usage methods."（腾讯云文档 1236/62821）；页面原文「定制版按相同接口重写后无需改应用代码」。｜修复要求：将「按相同接口重写」改为「在开源 NCCL 代码基础上扩展优化」，并保留「接口兼容」的表述（接入方式差异见上一条）。｜修复：｜复验：

- [重要·技术] 来源：NVIDIA Dynamo / NIXL 官方文档｜位置：通信库章节表格 NIXL 行，及正文末段「UCX 与 NIXL」对比｜问题：页面称「UCX 只是 NIXL 的可选后端之一」，弱化并歪曲了官方事实——UCX 是 NIXL 默认传输后端，未显式指定后端时即使用 UCX。｜引文依据：NIXL/Dynamo 文档 "UCX ... is the default transport backend for NIXL"、"By default, TensorRT-LLM uses NIXL with UCX as the backend for KV cache transfer"（docs.dynamo.nvidia.com、github.com/ai-dynamo/nixl）；页面原文「UCX 只是 NIXL 的可选后端之一」。｜修复要求：改为与来源一致的表述，例如「UCX 是 NIXL 的默认传输后端，NIXL 亦支持 LIBFABRIC、UCCL 等其它后端」。｜修复：｜复验：

- [轻微·表述] 来源：不适用｜位置：「分层与主线：谁调度、谁搬运」末段（第 119 行）｜问题：元话语——以「后文所有术语都围绕这条主线展开」说明页面的组织方式，属于对读者讲解编排而非陈述内容。｜引文依据：不适用｜修复要求：删除该句，或改为不含「后文」的正文陈述；正文在「网卡等硬件直接搬运、CPU 只下指令，传输就快。」处收束即可。｜修复：｜复验：

- [轻微·来源] 来源：不适用｜位置：「来源」章节全节｜问题：来源仅以名称罗列（如「NCCL 官方文档与 NVIDIA 博客《Fast Multi-GPU collectives with NCCL》」「PCIe 与 NVLink 带宽量级：NVIDIA 公开硬件规格」），未给出可定位的链接、版本或章节号，读者无法据以核对关键论断。｜引文依据：不适用｜修复要求：为每条来源补可定位的 URL（或标题+版本+章节），使「环形通信量公式 2(N-1)/N」「TCCL 定位」「GPUDirect 官方表述」「NIXL 后端」等关键论断可被复核。｜修复：｜复验：

- [轻微·技术] 来源：NVIDIA HGX / NVSwitch 公开规格｜位置：「机内互联：总线、直连与中心交换」表格 NVSwitch 行的「特性」列｜问题：「8 卡两两直连需 28 条线」的组合数计数正确，但紧随的「中心交换用 8 条线实现全互联」是理想星形拓扑的连线计数，放在 NVSwitch 行内作为硬件事实陈述会误导（真实 NVSwitch 系统每张 GPU 有多条 NVLink，由多颗交换芯片组成）。｜引文依据：NVIDIA HGX H100 规格——每颗 H100 有 18 条 NVLink 链路、连到 4 颗 NVSwitch，8 卡聚合带宽 7.2 TB/s。｜修复要求：把「中心交换用 8 条线」限定为拓扑示意的连线计数（如「中心交换只需每卡一条连线，共 8 条」并注明为拓扑示意），不与真实 NVSwitch 链路规格混同。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 3
- 处置：修复
- 说明：核心内容（8 原语定义与示例数值、ring all-reduce 两阶段与通信量 2(N-1)/N 及其 4/8/128 卡取值、树形算法步数、PCIe/NVLink/NVSwitch 分工、GPUDirect CPU 与主机内存不在数据路径上、gloo 的 CPU 训练定位）逐条回源核对后一致，可复算；本轮阻断与重要问题集中在 TCCL 与 NIXL 两处对来源的过度/弱化表述，需按上表逐条修复后复验，复验通过前不满足发布条件。