# DeepEP 审查记录（第 1 轮）

- 页面版本：index.html `b5fa1bb6ce6cf1dc7ca9469a392b8392d6dc823a`；overview.html `7e4c1fa651618608112a0903f7745d84b01baaa0`
- 审查时间：2026-09-03
- 审查者：独立审查者（未参与写作，首条消息只含页面路径、来源获取方式、规范路径与指令）
- 已完整阅读章节：引言 → 核心问题 → 1.1-1.3 → 第 1 章本章问题 → 2.1-2.5（含 SVG 图、手算折叠块）→ 第 2 章本章问题 → 3.1-3.5（含槽位表、代码折叠块、TBO 图）→ 第 3 章本章问题 → 4.1-4.4（含两表）→ 第 4 章本章问题 → 5.1-5.3 → 第 5 章本章问题 → 来源与范围说明；overview.html 全文

## 问题

- [重要·技术] index.html 5.2 节（第 5 章本章问题 2 解答同）｜DualPipe 论断"使 all-to-all 与流水线通信可被完全隐藏[C29]"超出快照支持范围：快照 §3.2.1 只有"overlap（重叠）"没有"完全隐藏"｜引文依据：快照 §3.2.1 原文 "The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks. To be specific, we divide each chunk into four components: attention, all-to-all dispatch, MLP, and all-to-all combine."，无 fully hidden 表述；Table 2 气泡公式 $(PP/2-1)(F\&B+B-3W)$ 表明气泡仍存在｜修复要求：核对 V3 报告 §3.2.1 原文是否含 "fully hidden" 原句，有则补充快照与引文依据，无则改写｜修复：已核对 arXiv 原文——论文确有 "In this overlapping strategy, we can ensure that both all-to-all and PP communication can be fully hidden during execution"（限"该重叠策略"语境；完整调度层面措辞为 "a significant portion"）。快照 v3-report-extracts.md 已补该两句原文；evidence C29 更新；页面 5.2 与第 5 章本章问题 2 解答改为"在前后向 chunk 对内部重排组件并手工调整通信/计算的 SM 配比；论文的表述是，在该重叠策略下 all-to-all 与 PP 通信可以在执行中被完全隐藏"，保留限定语境｜复验：通过（validate ok；引文依据已在快照中可定位）
- [轻微·技术] index.html 2.3 节（核心问题、第 2 章本章问题 3 解答同）｜node-limited 节点选择规则丢失"每节点最高的 $K_r/M$ 个"限定｜引文依据：§2.1.2 原文 "selected according to the sum of the highest Kr/M affinity scores of the experts distributed on each node"｜修复要求：补限定｜修复：2.3 正文、第 2 章本章问题 3 解答均改为"按各节点上分数最高的 $K_r/M$ 个专家的亲和分数之和选择节点"；evidence C6 同步更新｜复验：通过
- [轻微·技术] index.html 3.3 节（第 3 章本章问题 5 解答同）｜"QP 数必须等于本地专家数"丢失"为了最佳性能"目的状语｜引文依据：legacy.md 原文 "for the best performance, the QP number **must** be equal to the number of the local experts"｜修复要求：补限定｜修复：3.3 改为"为了最佳性能，QP（RDMA 队列对）数应与本地专家数一致"；第 3 章本章问题 5 解答改为"'QP 数与本地专家数一致'的性能建议"｜复验：通过
- [轻微·技术] index.html 4.4 节｜"EP 8×2 指 2 节点共 8 GPU"是页面推断但未标注｜引文依据：README Performance 表 "Topo" 列原文 "EP 8 x 2"，无记法定义｜修复要求：标注为本文解读或删去｜修复：改为"拓扑列记法的含义 README 未定义，本文解读为：EP 8×2 指 8 块 GPU 分布于 2 个节点、EP 8×4 指分布于 4 个节点——依据是这两种拓扑的瓶颈带宽都标注为 RDMA，与跨节点部署一致"｜复验：通过
- [轻微·技术] index.html 4.3 节与"简化条件及其限制"｜未覆盖解析式 SM 计算对 group-limited gate 的限制｜引文依据：elastic.py 内部注释 "NOTES: this is for balanced gate / For V3.0's group-limited gate, please do not use this function"｜修复要求：补充说明｜修复：已在仓库克隆中核实该注释（elastic.py 750-758 行）；快照 deepep-src-extracts.md 增第 6 节、evidence C10 更新；4.3 正文补"源码注释明确该函数只针对均衡 gate，V3.0 的 group-limited gate 暂不适用（标记为待支持）"；简化条件对应条目扩充｜复验：通过
- [轻微·技术] index.html 5.1 节｜UltraEP/MoonEP 机制描述无来源编号｜引文依据：C 列表无对应条目｜修复要求：补来源或降级｜修复：经核对 wiki/ultraep（"hybrid-ep 分支（该分支针对机架内通信优化）"）与 wiki/moonep（"静态形状消除 host 同步"、"dispatch/combine 数据路径"、DeepEP 提及 25 处）页面，一句话概述均有出处；来源章节 C 组末尾新增"相邻系统描述……取自本仓库 wiki/ultraep、wiki/moonep、wiki/aux-loss-free-routing 页面（内部来源，正文附链接）"条目｜复验：通过
- [轻微·技术] index.html 2.2 节（核心问题 2 解答、overview.html 核心机制同）｜"立即经 NVLink 转发"把设计目标写成无条件保证｜引文依据：§3.2.2 原文 "we will endeavor to ensure that it is instantaneously forwarded via NVLink"｜修复要求：改为设计目标语气｜修复：2.2 改为"到达目标节点后，即经 NVLink 转发……论文把'瞬时转发'表述为设计目标（力求确保），而非协议层面的保证"；核心问题 2 解答与 overview 改为"落地后经 NVLink 转发"（去掉"立即"）；第 2 章本章问题 2 解答补"（论文把瞬时转发表述为设计目标）"｜复验：通过
- [轻微·可读性] index.html 引言/1.2/2.4/3.1｜SM、TPOT、IBGDA、warp specialization 首次出现未解释｜引文依据：不适用｜修复要求：首现处给最小解释｜修复：SM 补"（streaming multiprocessor，GPU 上执行 kernel 的基本单元，通信与计算都要占用它）"；TPOT 补"（每输出一个 token 的耗时）"；IBGDA 补"（InfiniBand GPUDirect Async，IB 的异步 GPUDirect 扩展）"；warp specialization 补"——让 warp（SM 内的线程调度单元）各自专职一类任务——"｜复验：通过
- [轻微·格式] index.html 全文｜"第 2 章"等纯编号章节引用｜引文依据：不适用｜修复要求：使用章节标题｜修复：接受不改。理由：style-guide 该条针对的是 S1/S2 式代号；已发布页面（wiki/ultraep 22 处、wiki/moe-serving 4 处）均采用"第 X 章"自然语序引用，编号 h2 使引用自定位；改为长标题会显著增加行文负担｜复验：接受（记录理由）
- [轻微·格式] index.html 1.2/3.2/4.4/5.2｜4 处"常见误解"分散为 yellow callout 而非前置 misconceptions section｜引文依据：不适用｜修复要求：集中或记录接受理由｜修复：接受不改。理由：outline.md 明确把误解分配到各章（误解 4→第 1 章、2→第 3 章、5→第 4 章、1/3→第 5 章）；组件库 04 条目明写"仅当 outline.md 明确安排在页面开头给出误解提示时使用本组件；否则常见误解放在大纲指定的章节处理"；write.md 要求按 outline 落实"误解和边界的处理位置"｜复验：接受（记录理由）

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 9
- 处置：修复完成。8 条已修复（含 1 条重要）；2 条记录接受理由（章节引用风格、误解分散放置）。修复后复验：validate.py 两页通过、页面代码运行输出与预期逐字符一致、C/F/N 双向引用两个差集为空。
