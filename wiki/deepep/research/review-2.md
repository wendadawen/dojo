# DeepEP 审查记录（第 2 轮）

- 页面版本：index.html `b6f2cd51f59750d2cf561264d2052c64cbd4b1d6`（修复前）
- 审查时间：2026-09-03
- 审查者：独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：引言、核心问题（5 条含解答）、第 1 章（1.1-1.3、本章问题）、第 2 章（2.1-2.5、手算折叠块、本章问题）、第 3 章（3.1-3.5、双缓冲补充块、代码折叠块、本章问题）、第 4 章（4.1-4.4、本章问题）、第 5 章（5.1-5.3、本章问题）、来源与范围说明；overview.html 全文

## 问题

- [轻微·技术] index 2.2 节 SVG 图注｜"蓝色加粗边框标出路径经过的三块 GPU"与路径事实不符：$t_0$ 两条路径经过四块 GPU（rank 0、1、4、5），IB 落地中转的 rank 4 未加粗，而它恰是两段转发机制的关键节点｜引文依据：图中 accent 加粗框仅 rank 0、rank 1、rank 5 三处；IB 虚线箭头终点与第二条 NVLink 箭头起点均为 rank 4 未加粗｜修复要求：为 rank 4 加粗并把图注改为"四块"，或改为"标出出发与两块目标 GPU"｜修复：已为 rank 4 增加 accent 加粗框（共 4 处），图注改为"蓝色加粗边框标出两条路径经过的四块 GPU：出发的 rank 0、IB 落地中转的 rank 4、两块目标 rank 1 与 rank 5"｜复验：通过（headless Chrome 探针：accentRects=4，标签两两无重叠）
- [轻微·技术] index 4.3 节｜"V2 用 ElasticBuffer.get_theoretical_num_sms…<sup>[C10]</sup>"的引用编号与来源章节分组不对应：C10 在来源章节归入"高吞吐内核"组（legacy.md 与 V3 报告），该句实际来源是 README EPv2 条目与 elastic.py docstring（C18-C24 组）｜引文依据：来源章节 C 组第 2 条不含 elastic.py；第 4 条才含 get_theoretical_num_sms docstring｜修复要求：改引用编号或在来源章节 C10 条目补入 elastic.py 与 README EPv2 条目｜修复：来源章节 C4-C11 组补注"其中 C10 的 V2 部分（解析式 SM 计算与 group-limited 限制）另见 README EPv2 条目与 elastic.py 的 get_theoretical_num_sms docstring 及内部注释（deepep-src-extracts.md 第 6 节）"，编号与来源恢复双向对应｜复验：通过（引用差集为空）
- [轻微·可读性] index 3.4｜TBO 一词首次出现于解答折叠块，未给出全称与定义，且 TBO 字样不在引用来源中（来源只写 "processing two micro-batches… overlapping"）｜引文依据：§3.4.2 原文 "we overlap the attention of one micro-batch with the dispatch+MoE+combine of another"，无 TBO 字样｜修复要求：正文首次描述处给出定义｜修复：3.4 正文改为"两个 micro-batch 交错（two-batch overlap，TBO）"，后续简称沿用｜复验：通过
- [轻微·格式] index 全文｜跨章引用几乎全部只用编号，不带被引章节标题（style-guide 第 1 节要求）｜引文依据：不适用｜修复要求：跨章引用处附带被引章节标题，至少每处首次引用带标题｜修复：28 处跨章/节引用全部改为"第 X 章「主标题」"形式（引言主线段、5 个学习目标答案、1.1 贯穿示例句、各章正文与衔接句、来源章节两处）；残余无标题章引用 0 处｜复验：通过
- [轻微·技术] index 5.1 节｜UltraEP/MoonEP 两句概述无法在审查者输入限制内完成四步核对，需编排者用 wiki/ultraep、wiki/moonep 页面核对并记录引文依据｜引文依据：README 快照仅支持 Hybrid-EP 分支存在｜修复要求：编排者补充核对并记录｜修复：已由编排者核对——wiki/ultraep 页面含"hybrid-ep 分支（该分支针对机架内通信优化）"（另含"用配额驱动的实时规划…做精确负载专家均衡"标题表述）；wiki/moonep 页面含"静态形状消除 host 同步"（4 处）、"dispatch/combine 数据路径"、DeepEP 提及 25 处。页面概述与被引页面表述一致，正文均附链接，来源章节已声明内部来源｜复验：通过（引文依据记录于本条）
- [轻微·可读性] index 引言第 2 段｜"只做 token 搬运"与 1.3 节"另提供 PP、CP、Engram 实验性原语"存在表述张力（PP/CP 原语搬运的并非 MoE token）｜引文依据：README "currently focuses on expert parallelism (EP)… while also offering experimental primitives for PP, CP, and Engram"｜修复要求：加限定表述｜修复：引言改为"当前聚焦专家并行，核心只做 token 搬运"（与 C1 的 README 措辞对齐）｜复验：通过

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：6 条全部闭环（5 条修复 + 1 条由编排者补充核对记录）。修复后复验：validate.py 通过、页面代码运行输出逐字符一致、C/F/N 双向引用差集为空、headless Chrome 渲染探针（135 个 KaTeX 零错误、SVG 公式标签 9 个全渲染、4 个高亮框、标签无重叠）。
