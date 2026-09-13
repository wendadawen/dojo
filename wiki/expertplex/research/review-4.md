<!-- review-meta
round: 4
page: wiki/expertplex/index.html
reviewed_content_sha256: 91e8e13a088ca44e
-->
# ExpertPlex 审查记录（第 4 轮）

- 页面版本：a4eb4614f1ef02e9bf51b8c21e0851a2a56d41fc（wiki/expertplex/index.html 工作树哈希；overview.html 305c085388099d454c64ff513ce680b0cf3fbcdc）
- 论文版本：arXiv:2607.18002v2 [cs.DC]（2026-07-21 修订）
- 审查时间：2026-09-13 20:09 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：页面级「核心问题」（5 题）→ 1. 两条现有路线，各自的死结（1.1 路线一：PD 分离、1.2 路线二：PD 合设、本章问题）→ 2. ExpertPlex 的架构：共享专家，分离注意力（本章问题）→ 3. APK：在 tile 边界上调度 GPU（3.1 共享机制需要五个性质、3.2 为什么选 tile 当调度单位、3.3 有界抢占、3.4 在线 SM 重分配、本章问题）→ 4. 通信：让 attention 侧发起一切（4.1 两侧通信为什么会死锁、4.2 一侧通信、4.3 分层 prefill 路径与流量隔离、本章问题）→ 5. 跨栈优化器：从 tile 建模到集群（5.1 goodput、5.2 tile 感知延迟模型、5.3 离线搜索＋在线重分配、本章问题）→ 6. 实验：提升多少，在什么条件下成立（6.1 设置、6.2 端到端、6.3 四个微基准、本章问题）→ 7. 独立评价：三机制互相使能，但验证边界要看清（7.1 优点、7.2 局限、7.3 适用场景与位置、本章问题）→ 来源与范围说明。另核对 6 张配图（assets/img-01…img-06）内容与图注。

核对方式：下载 arXiv HTML 版全文（arxiv.org/html/2607.18002v2，arXiv:2607.18002v2 [cs.DC] 21 Jul 2026）逐句比对照片段与数值；读取 assets/ 原图核对图注；运行 `.dojo/scripts/validate.py wiki/expertplex/index.html`（返回 `validation ok`）。本轮未读取本页 research/ 下任何文件。

## 问题

- [重要·技术] 「1.2 路线二：PD 合设」Figure 2 图注（index.html 第 153 行）：图注把原文 Figure 2 的两个失败模式指派给了图片的左右两半，写成「左侧是 head-of-line blocking……右侧是 resource bubble」，与图片实际版式不符——Figure 2 左半面板是「每卡被切成 Prefill 2/3 + Decode 1/3」的总览，标注为「Larger DoP! (1.5x for prefill, 2x for decode)」与「Network Interference!」，并不画队头阻塞；Blocking! 与 Bubble! 两个标注都在右侧那条单卡时间线面板内。照现有文字看图的读者会建立错误的面板对应。｜引文依据：assets/img-06.webp 图内文字——左面板「Larger DoP! (1.5x for prefill, 2x for decode)」「Network Interference!」，右面板时间线「Launch decode → Blocking!」与右端斜纹区「Bubble!」；论文 Figure 2 标题「Limitations of prefill-decode colocation solutions.」，正文 §2.5「As Figure 2 illustrates, a poor partition creates two failure modes.」｜修复要求：改写图注，按图片真实版式描述：左侧面板说明每卡切分导致每阶段本地资源减少、并行度变高与跨阶段网络干扰；右侧单卡时间线面板中，prefill 占位时 decode 的启动点被挡（Blocking!），右侧斜纹区是 decode 预留 SM 闲置的 resource bubble。删除「左侧是 head-of-line blocking／右侧是 resource bubble」的左右指派。｜修复：｜复验：

- [轻微·技术] 「3.1 共享机制需要五个性质」与「逐个对照现有机制」（第 235、248 行）＋来源说明 C7（第 562 行）：五条性质与机制对照表出自论文 §4.1 与 Table 1，但 [C7] 的来源标注范围写的是「§4.2/§4.3/§6.4」，不含 §4.1；这两段论断无法落到标注的章节。｜引文依据：论文 §4.1「Efficient sharing therefore requires five complementary properties, summarized in Table 1.」；Table 1「Comparison of ExpertPlex with existing GPU sharing solutions.」；来源说明原文「C7（APK tile 调度、边界 2.2–25.3 微秒、有界抢占上界、$q'$ 公式）§4.2/§4.3/§6.4」｜修复要求：把 C7 的来源范围补为「§4.1/§4.2/§4.3/§6.4」，或将五性质与机制对照段另立来源编号（如 C7a）标注 §4.1/Table 1。｜修复：｜复验：

- [轻微·技术] 「3.1 逐个对照现有机制」（第 248 行）：列举 API 拦截缺失的性质时写「缺图兼容、空间复用和有界抢占」，漏掉「有界重分配」，字面上使 API 拦截具备「有界重分配」，与同段末句「APK 是上述机制中唯一五条全占的」相抵触（若 API 拦截已占有界重分配，则 APK 并非唯一全占）。｜引文依据：论文 §4.1「API interception offers only launch-boundary time multiplexing. It lacks CUDA Graph compatibility, spatial multiplexing, and bounded preemption and reallocation.」；Table 1 中 API Interception 一列仅在 Temporal Multiplexing 有勾，其余四条（含 Bounded Fast Reallocation）为空｜修复要求：改为「缺图兼容、空间复用、有界抢占和有界重分配」，与 Table 1 的勾选一致。｜修复：｜复验：

- [轻微·技术] 「1.2 路线二：PD 合设」正文与本章问题解答（第 147、172 行）：「Green Context 的 SM 分区在创建时固定，kernel 运行期间不能重新划分，要改得销毁重建」中的「要改得销毁重建」在论文中找不到对应表述。｜引文依据：论文 §2.5 只写到「Changing an allocation requires CPU coordination, and kernel-completion waits make per-kernel changes impractical.」「reconfiguration is limited at the prefill layer level because it requires CPU intervention.」；未出现 destroy/recreate 的说法｜修复要求：删去「销毁重建」，或改写为论文支持的口径：「改配额需要 CPU 介入，且要等 kernel 结束，因此只能在 prefill 层边界重划分」。｜修复：｜复验：

- [轻微·表述] 页面级「核心问题」第 5 题解答（第 116 行）与 description（第 6 行）：把 GLM-5.1-FP8 + LooGLE 的 1.66× 表述为「PD 合设口径下……为 1.66×」，而同一页表格（第 460–461 行）把该设置拆成「vs Colocated 2.5×」与「vs PDMux 1.66×」两列，且「PD 合设（colocation）」在 1.2 节被定义为含 Colocated 在内的一整条路线。读者按页内定义读「PD 合设口径」，会与 Colocated 一列的 2.5× 冲突。｜引文依据：论文 §7.2「On LooGLE, this advantage fades and ExpertPlex improves goodput over SGLang-PDMux by 1.66×.」（论文 Abstract 作「1.66× over prefill-decode colocation」，本页 6.2 节表格也作 vs PDMux 1.66×）｜修复要求：把「PD 合设口径下（GLM-5.1-FP8 + LooGLE）为 1.66×」改为「对 PDMux（GLM-5.1-FP8 + LooGLE）为 1.66×」，description 中的「PD 合设 1.66×」同步改为「对 PDMux 1.66×」，避免与页内「vs Colocated」列碰撞。｜修复：｜复验：

## 已核对无问题（本轮逐条回源，仅列结论）

- 全部数值论断均与原文一致：权重占比 95%/96%/98%（§2.1）；DeepSeek-V3 32P+320D、176 GPU 单元、Kimi-K2 128×H200（§1/§2.4）；EP4 MiniMax decode 17.7–34.7 μs vs prefill 16K 1.8–2.9 ms、84–101×（§4.1，比值复算 1.8ms/17.7μs=101.7、2.9ms/34.7μs=83.6，与 84–101 一致）；tile 边界 2.2–25.3 μs、GEMM <10.7 μs（§4.2/§7.6）；NVLink 160 GB/s vs IB 50 GB/s、3.2×（§5.1，160/50=3.2）；端到端 11.3 req/s/node 与 5.65×/2.72×/2.01×/1.41×、4.12×/1.28×、3.3×/1.5×、5.0×/2.5×、GLM+ShareGPT≈1.5 持平、GLM+LooGLE 1.66×（§7.2）；CUDA stream +13.79×、MPS 3.33×、Green Context 4.07×、APK +8%/1.12×（§7.3）；调度开销 <12%/<20 μs（§7.4）；通信 ~5%/~45 μs（§7.5）；REEF 35 μs（§7.6）；H100 MIG 1g/2g/3g/4g/7g、3g–4g（§4.1）。
- 四个公式与原文 Eq.(1)–(4) 逐字一致（goodput、延迟拟合、tile 足迹、$q'$），符号 $B_p/B_d/T_p/T_d/\bar O/x_{\mathrm{moe}}/x_{\mathrm{moe}}^\star/Q_{\max}/M_t$ 全文单义；goodput 取 min 的手算（8/0.5=16、64/(0.05×20)=64、min=16；32/0.5=64）可复算且已标注为构造示例。
- 机制描述逐段核对支持：Green Context kernel 期间固定与三维动态负载（§2.5）、两侧通信死锁环与常驻轮询 SM（§5.1）、一侧 push/pull 与 WaitDone（§5.2）、分层 prefill 路径与虚拟通道优先级（§5.3）、tile 边界无 live 累加器/TMA/共享内存、抢占上界=一个 tile + 一次 cluster 检查 epoch（§4.2/§4.3）、$P/p_i$/DSMEM/mbarrier 传播链（§4.3 与 img-04）、离线 FitsMemory + 二分搜索（§6.3）。
- 6 张配图（img-01→Figure 11、img-02→Figure 6、img-03→Figure 5、img-04→Figure 4、img-05→Figure 3、img-06→Figure 2）编号与图内内容除上述 Figure 2 图注外均对得上；均为论文原图位图，无等宽字符框线图。
- 结构项合格：页面级「核心问题」（5 题）与 7 个章节的「本章问题」（3/3/4/4/3/3/3 题）均有解答折叠块，核心问题答案均指明完整论证所在章节；overview.html 与 index.html 互链；前置概念页 ../moe-serving/index.html、../gpu-execution-model/index.html 真实存在；无「（待生成）」占位；正文与来源说明无指向 research/ 下非 .md 文件的路径。
- `<head>`：description 为纯文本、dojo:summary 可渲染、dojo:type=paper、dojo:topics=推理系统（在 AGENTS.md 固定大类内）、dojo:tag=MoE（在 ALLOWED_TAGS 内）。
- 表述维度通读（含折叠块与图注）：无「本页将…」「下面来看…」「需要注意的是」式元话语，无以「本页」为主语的自我指代，无「我/我们/你」式会话指代，无调试与复现踩坑叙事，无临场评价；第 7 章「独立评价」整章以 callout 明确标注为解读者推断，与论文事实分开。
- `python3 .dojo/scripts/validate.py wiki/expertplex/index.html` 返回 `validation ok`（含数学字符与结构图检查；× → 属规范排除的排版字符，不计入）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（1 条重要、4 条轻微，修复后按第 4 节复验；无阻断项，无需返回规划的范围或大纲问题）