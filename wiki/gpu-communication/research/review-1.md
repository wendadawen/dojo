<!-- review-meta
round: 1
page: wiki/gpu-communication/index.html
reviewed_content_sha256: 93aab27ab955f1bf
-->
# GPU 通信审查记录（第 1 轮）

- 页面版本：609905d56640
- 页面类型：note（规范：guides/note.md）
- 审查时间：2026-09-13 18:47
- 审查者：独立子代理（未参与写作，未读取 research/ 内规划、修复与前序审查记录）
- 已完整阅读：全文（标题、导语、meta、7 个 h2 章节及其 h3 与两张内联 SVG、来源清单、页脚脚本）；页面无折叠块（`<details>` 计数 0），无 `<pre>` 代码块，无「核心问题/本章问题」块（note 类规范不要求）。
- 机械验证：`.dojo/scripts/validate.py wiki/gpu-communication/index.html` → `validation ok`；内链 `../deepep/index.html`、`../model-parallelism/index.html` 均真实存在；head 含纯文本 `description`、可渲染 `dojo:summary`、`dojo:type=note`、`dojo:topics`、`dojo:tag`。

## 问题

- 阻断｜来源｜「通信库」表 TCCL 行与正文「TCCL 存在的理由」（第 249、254 行）｜把 TCCL 写成「接口与 NCCL 完全一致、替换即用」，与腾讯云官方 TCCL 文档直接冲突：官方明确说明不能通过替换共享库使用 TCCL。｜引文依据：腾讯云《TCCL 使用说明》（cloud.tencent.cn/document/product/1573/93190）：「由于社区 Pytorch 默认采用静态方式连接 NCCL 通信库，所以无法通过替换共享库的方式使用 TCCL」；同文档确认「API 与 NCCL 完全兼容」，但接入必须「重新编译安装 PyTorch」（方法一）或「安装 PyTorch/NCCL 通信插件」（方法二/三），无「替换即用」一路。｜修复要求：删除「替换即用」，改为「接口/API 与 NCCL 兼容，但需重编 PyTorch 或安装通信插件接入，不能直接替换共享库」；「来源」清单该条出处补上《TCCL 使用说明》。
- 重要｜来源｜「来源」清单第 1 条（第 258 行）｜把「环形 all-reduce 两阶段、通信量公式」记为 NVIDIA 博客《Fast Multi-GPU collectives with NCCL》的内容；核对该博客全文，只有 ring 概念与内核原语，没有两阶段分解，也没有 2(N−1)/N 公式。｜引文依据：博客原文「NCCL currently supports the all-gather, all-reduce, broadcast, reduce, and reduce-scatter collectives.」「Internally, NCCL implements each collective in terms of three primitives: Copy, Reduce, and ReduceAndCopy.」「ring algorithms provide near optimal bandwidth for nearly all of the standard collective operations」；全文无 `2(N-1)/N` 或 reduce-scatter→all-gather 两阶段推导。｜修复要求：把「环形 all-reduce 两阶段、通信量公式」的出处改成确实含该推导的来源（如 NCCL 开发者指南相应页或 ring all-reduce 专门说明），或把该条拆成两条、分别标注各自覆盖内容。
- 轻微｜表述｜第 98 行｜出现第一人称会话指代「我」。｜引文依据：不适用｜修复要求：改为非会话指代，如「（如『发起 all-reduce』）」。
- 轻微｜表述｜第 119 行｜「后文所有术语都围绕这条主线展开。」属元话语式前指。｜引文依据：不适用｜修复要求：删除该句，或改为对内容的直接陈述。
- 轻微｜表述｜第 238、240 行｜「同一个角色，两个艺名」「一块坏了另一块秒切」为口语化、临场评价式表述。｜引文依据：不适用｜修复要求：改为「同一角色的两种名称」「另一块立即接管」等书面表述。
- 轻微｜表述｜第 142、236、254 行｜「两个容易混淆的对比：」「一组容易混淆的术语」「两组容易混淆的对比：」以读者临场反应组织段落，属元话语式框架。｜引文依据：不适用｜修复要求：直接给出对比内容，如「broadcast 与 scatter 的区别是发同一份还是切开每人一份」。
- 轻微｜来源｜第 232 行｜「（NVIDIA 官方表述：CPU 与主机内存完全不在数据路径上）」以「官方表述：」的直引形式呈现，但该句非 NVIDIA 原文。｜引文依据：NVIDIA GPUDirect RDMA 官方文档原文为「GPUDirect RDMA is a technology ... that enables a direct path for data exchange between the GPU and a third-party peer device using standard features of PCI Express.」；NVIDIA/Mellanox 白皮书用「eliminating the need for CPU involvement in the communication loop」；均无「CPU 与主机内存完全不在数据路径上」这句。｜修复要求：去掉「官方表述：」的直引包装，改为陈述句并标注为对官方文档的概括。
- 轻微｜技术｜第 195 行｜「代价是延迟随卡数线性增长（步数 N-1）」：环形 all-reduce 两阶段各 N-1 步、总步数 2(N-1)，括注只给了 N-1。｜引文依据：同页第 148、150 行「沿环传 N-1 步」分别为阶段一、阶段二，两阶段合计 2(N-1) 步。｜修复要求：改为「步数 2(N-1)」或「每阶段 N-1 步、合计 2(N-1) 步」。
- 轻微｜技术｜第 254 行｜「UCX 只是 NIXL 的可选后端之一」低估了 UCX 在 NIXL 中的地位。｜引文依据：NVIDIA Dynamo/TensorRT-LLM 文档「TensorRT-LLM uses NIXL with UCX as the default method」；NIXL 后端说明将 UCX 列为默认传输后端。｜修复要求：改为「UCX 是 NIXL 的默认后端，另有 GDS、libfabric 等可选后端」。
- 轻微｜技术｜第 205、207 行｜NVSwitch「中心交换用 8 条线实现全互联」为理想化简化，按每卡 1 条链路计，未说明该假设。｜引文依据：不适用（拓扑计数 C(8,2)=28 成立；「8 条」仅在每卡 1 链路假设下成立，实际 NVSwitch 每卡有多条 NVLink 链路）｜修复要求：补充简化条件（如「按每卡一条链路计」）或删除具体线数。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 8
- 处置：修复（阻断与重要问题须全部关闭；轻微问题建议按上表逐条改，或说明接受理由）
- 说明：本页无 `<pre>` 代码块，无可运行代码，代码执行一项不适用；无折叠块，折叠内容可读性一项不适用。核心结论（通信原语三类、ring all-reduce 两阶段与 2(N−1)/N 通信量、CPU 是否参与搬运主线、PCIe/NVLink/NVSwitch 分工、RDMA vs socket 路径）经复算与来源核对成立，故未列阻断。