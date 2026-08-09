# GPU 执行模型与 kernel 调度独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照 NVIDIA 官方文档事实核查）
- 页面版本：index.html 工作树哈希 31bcde63227511039c0a3710b869d854debc79c2
- 时间：2026-08-07

## 问题

- [重要·盲读] index.html §cluster（第 834 行，"cluster 内的 CTA 被硬件共同调度到同一组相邻的 SM（同一个 GPC）上"）：GPC 作为缩写首次出现，正文未给出全称或任何解释；小白不知道 GPC 是什么、为什么 cluster 受 GPC 约束，无法理解"共同调度到同一个 GPC"这一 cluster 核心约束。来源 [C15] 引用了"8 GPC 结构"但正文未用到该数字。修法：在首次出现 GPC 处加括号注释，如"GPC（Graphics Processing Cluster，图形处理集群，是 GPU 硬件层级中一组物理相邻的 SM 集合，H100 有 8 个 GPC）"。 ｜ 修复：已改。第 834 行括注补为"同一个 GPC——Graphics Processing Cluster，图形处理集群，是 GPU 硬件层级中一组物理相邻的 SM，H100 有 8 个 GPC），并保留 [C15] 引用 ｜ 复验：通过。第 834 行括注已补全称、中文译名与"一组物理相邻的 SM""H100 有 8 个 GPC"，[C15] 引用保留；小白可理解 cluster 与 GPC 的约束关系
- [重要·盲读] index.html §sharing-mig（第 909-925 行，MIG profile 表与"3g + 4g 正好用满"例子）：MIG profile 名称中的"g"未解释其含义。小白看到"1g.10gb""7g.80gb（整卡）""3g + 4g 正好用满"时，无法理解"g"代表什么单位、为什么 3+4=7 就是"用满"、7 这个总数从何而来。虽然可从"7g.80gb（整卡）"勉强推断 7g=整卡，但页面未明确说明 H100 有 7 个可用计算切片（GPC）这一算术基础。修法：在 MIG profile 表前加一句解释，如"profile 名称中的数字 g 表示 GPU 计算切片数（H100 有 7 个可用切片，对应 7 个 GPC，留 1 个 GPC 给管理用），数字后的 GB 表示该实例独占的显存量；7 个切片用满即整卡"。 ｜ 修复：已改。表前段落补一句"H100 有 7 个可用的 GPU 计算切片（对应 7 个 GPC，另留 1 个 GPC 给管理用）；profile 名称里的数字 g 表示该实例占用的计算切片数，数字后的 GB 表示它独占的显存量，7 个切片用满即整卡"，[C13][N5] 移到该句末 ｜ 复验：通过。第 909 行已补"7 个可用 GPU 计算切片（对应 7 个 GPC，另留 1 个 GPC 给管理用）"及"g 表示计算切片数、GB 表示独占显存量、7 个切片用满即整卡"，[C13][N5] 已移至该句末；小白可理解"g"单位与"3g+4g 用满"的算术基础
- [轻微·盲读] index.html §launch（第 750 行，"这就是开头问题的第一层答案，也是「队头阻塞」（head-of-line blocking）的来源"）："队头阻塞"作为调度/网络领域术语首次出现，正文未解释其含义。小白可能不知道"队头阻塞"指什么。修法：在首次出现处加括号注释，如"队头阻塞（即队列前面的长任务挡住后面的短任务，使短任务即使已就绪也无法开始）"。 ｜ 修复：已改。括注补为"head-of-line blocking，即队列前面的长任务挡住后面的短任务，使短任务即使已就绪也无法开始" ｜ 复验：通过。第 750 行括注已补"即队列前面的长任务挡住后面的短任务，使短任务即使已就绪也无法开始"，术语含义自明
- [轻微·盲读] index.html §hardware diagram（第 699 行，diagram 中"HBM3 全局内存 80 GB · 3.35 TB/s"）：diagram 中首次出现"HBM3"，但正文（第 694 行）只说"HBM 全局内存"，未说明 HBM3 是什么、与 HBM 的关系。小白可能不知道 HBM3 是第几代显存。修法：在正文首次提到 HBM 处加一句"HBM3 是 H100 使用的第三代高带宽显存（High Bandwidth Memory）"，或将 diagram 中"HBM3"改为"HBM"以与正文措辞一致。 ｜ 修复：已改。正文 HBM 条目补"H100 用的是 HBM3，第三代高带宽显存" ｜ 复验：通过。第 694 行正文已补"H100 用的是 HBM3，第三代高带宽显存"，与 diagram 第 699 行"HBM3"措辞一致，小白可知 HBM3 是第几代显存
- [轻微·技术] index.html §sharing-green（第 929 行，"两条边界由官方写明：其一，资源在创建时固定，运行中的 kernel 期间不能重新划分"）：官方文档（CUDA Driver API §Green Contexts，CUDA 12.8/13.1）明确写了最小 8 SM、8 的倍数、不保证并发与前进，但未明确写"运行中的 kernel 期间不能重新划分"——此句是从 API 行为推断（Green Context API 无运行期重分区接口，改变分区需销毁并重建上下文），页面标注为"由官方写明"略有越界。来源 [C14] 的括注中也把"创建时固定"列为官方文档内容，与实际文档措辞不完全吻合。修法：将"由官方写明"改为"由 API 设计决定"，或补充一句"Green Context API 不提供运行期重分区接口，要改变 SM 分区需销毁并重建 Green Context"，并把 [C14] 括注中的"创建时固定"改为"创建时通过 split API 指定分区"。 ｜ 修复：已改。改写为"两条边界由 API 设计决定：其一，SM 分区在创建 Green Context 时通过 split API 指定，Green Context API 不提供运行期重分区接口，要改变分区需销毁并重建 Green Context，所以运行中的 kernel 期间不能重新划分" ｜ 复验：通过。第 929 行已将"由官方写明"改为"由 API 设计决定"，并补"split API 指定分区""不提供运行期重分区接口""需销毁并重建 Green Context"，来源越界问题已消除。注：[C14] 括注（第 961 行）仍为"创建时固定"，与新正文语义一致（创建时确定、运行期不可变），修法中"同步改括注"未落实但不影响正确性

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 3
- 处置：进入修复。5 个问题均为局部补充或措辞修正，不涉及研究范围或教学大纲调整。两处"重要"问题（GPC 未解释、MIG "g" 未解释）均为术语首现解释不足，导致小白在 cluster 章节和 MIG 章节出现理解卡点，需补一句解释即可关闭。三处"轻微"问题为阅读质量与来源标注精度的小幅改善。

### 事实核查摘要（段 B 对照来源结果）

已逐条核对页面 [C1]–[C16]、[N1]–[N8] 引用的 NVIDIA 官方文档，结果如下：

- H100 SXM5 规格：132 SM、每 SM 4 个第四代 Tensor Core、每 SM 228KB shared memory、单线程块上限 227KB、50MB L2、80GB HBM3 @ 3.35 TB/s——均与 NVIDIA Hopper Tuning Guide §4.1.1、§4.2.1–§4.2.2 及 H100 架构白皮书一致。
- CTA ≤1024 线程、warp=32、Hopper 每 SM 最多 64 warp——与 CUDA Programming Guide / Hopper Tuning Guide §4.1.1 一致。
- CTA cluster 可移植上限 8、H100 opt-in 16、占用率代价——与 Hopper Tuning Guide §4.1.3 一致。
- TMA 支持 1–5 维张量、异步搬运、warp specialization——与 Hopper Tuning Guide §4.1.2 一致。
- DSMEM（cluster 内 SM 互访 shared memory）——与 Hopper Tuning Guide §4.1.3 及 NVIDIA Hopper Architecture In-Depth 一致。
- cluster 共同调度到同一 GPC、H100 有 8 GPC——与 NVIDIA Hopper Architecture In-Depth 官方博客一致。
- MIG H100 六档 profile（1g.10gb/1g.20gb/2g.20gb/3g.40gb/4g.40gb/7g.80gb）及每卡最多实例数（7/4/3/2/1/1）——与 NVIDIA vGPU 参考文档 H100 SXM5 表一致。
- Green Context CUDA 12.4 引入、CC 9.0+ 最小 8 SM 且为 8 的倍数、不保证并发与前进——与 CUDA Driver API §Green Contexts（CUDA 12.8/13.1）一致。
- cuStreamCreateWithPriority 原文"Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order"——页面翻译"不会抢占已经在运行的工作，也不对执行顺序提供任何功能性保证"准确。
- MPS 原文"Setting the limit does not reserve dedicated resources for any MPS client context... Kernels launched from different MPS client contexts may execute on the same SM, depending on load-balancing"——页面翻译"不为任何客户端预留专用资源""不同客户端的 kernel 可能执行在同一块 SM 上"准确。
- 4×4 tile 例子复算：不切 tile 合计 128 次读取、切 2×2 tile 合计 64 次读取、一般规律"tile 边长 T 时每个 A 元素被读 N/T 次"——复算无误。
- 学习目标闭环：页面"读完你能回答"5 条学习目标均由正文第 1–6 章完整回答。
- validate.py：index.html 与 overview.html 均退出码 0。
- KaTeX 渲染：页面 15 处公式在 node + 本地 libs/katex.min.js 下全部渲染成功（1 处 unicodeTextInMathMode 警告但不影响渲染）。
- 前置引用：index.html → overview.html 互相链接有效；`../vllm-cudagraph/index.html` 目标存在；"姊妹概念页（生成中）"已标注占位。
