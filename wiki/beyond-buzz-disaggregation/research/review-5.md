<!-- review-meta
round: 5
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: adef0216218a5bcd
-->
# Beyond the Buzz 审查记录（第 5 轮）

- 页面版本：beddb223e814abc710c10eb8f738f07c54024713（index.html 工作树哈希，2026-09-13）
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；核对来源 https://arxiv.org/html/2506.05508v1 全文 + 页面 assets/ 内 10 张原图逐图核对）
- 审查时间：2026-09-13 20:12
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：题头/meta → 术语速查 → 核心问题（Q1–Q5）→ 1. 方法：模拟器与设计空间（含本章问题）→ 2. 什么条件下分离收益最大（2.1 流量 / 2.2 模型大小 / 2.3 架构，含本章问题）→ 3. 配套机制：切分策略与动态 rate matching（3.1 / 3.2 / 3.3，含本章问题）→ 4. 分离的代价：KV cache 传输带宽（4.1 / 4.2 / 4.3，含本章问题）→ 5. 方法评价：可操作结论与边界（5.1 / 5.2，含本章问题）→ 来源与范围说明（C / F / N / 原图 / 构造示例 / 类比边界 / 简化条件）。折叠块全部展开阅读。

## 核查结论（通过项，供复验参考）

- 抽象结论逐字一致：abstract "hundreds of thousands of design points"、"prefill-heavy traffic patterns and larger models"、"dynamic rate matching and elastic scaling"；introduction "prefill-heavy traffic scenarios (i.e., ISL >> OSL) and when serving larger models (e.g., > 10B parameters)"（N1 核对通过）。
- C8/F1、F2 两条带宽公式与论文 §5 Eq.(1)(2) 逐符号一致；egress 手算复核通过：61×32×16384×128×1×1/(2×8)=255,852,544 B/s≈0.256 GB/s/卡，与页面标注 0.256 吻合，且已标注为构造示例。
- rate matching 算法与 Appendix B 一致：Algorithm 1 throughput=B/(FTL×G)；Algorithm 2 decode_request_throughput=decode_throughput/(OSL−1)、α=round(prefill/decode_request, tolerance=0.03)、num_prefill_gpus=numerator(α)×G_dec、num_decode_gpus=denominator(α)×G_prefill，页面复述无误。
- 原图逐图核对（像素测量）：Fig.5 FTL 端点 log2≈6.58→2.15（页面"约 90 秒/2^6.5 → 约 4 秒/2^2"成立）；Fig.9 R1 3.64→0.06（页面 3.6→0.05）、70B 0.95→0.46（页面 0.95→0.45）、8B≈0.41（页面 ≈0.4）均吻合；Fig.12 蓝 0.4–1.23 / 红 0.97–1.82（页面 0.4–1.2 / 1.0–1.8）吻合；Fig.1/2/6/7/8/14 标题、曲线关系与正文描述一致。
- 直接引文逐字核对通过：§5 "existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck"、§6 "fall short of providing concrete guidance on when and how disaggregation is beneficial"、"largely focused on small-scale testbeds and peak throughput scenarios..."、§8 "we also highlight scenarios where disaggregation offers limited benefit..."。
- 机械项：17 个 <details>/<summary> 与 17 处问题块配对；6 个 chapter-questions 块（1 核心 + 5 章）；全部本地资源（libs/ 8 个、assets/ 10 张图、6 个前置概念页、overview.html）存在；无"待生成"；无 research/ 路径引用；.dojo/scripts/validate.py 返回 validation ok。

## 问题

- [阻断·技术] §5.2「模拟器性质」段（index.html:387）：「论文承认缺乏细节的"小细节"（如显存碎片、调度异常）可能被忽略」——论文从未承认此事；这是页面自己的推断被包装成论文的表述。这是"推断被包装成来源结论"，按分级为阻断。｜引文依据：论文全文对模拟器只有一句定性 "we use a proprietary, high-fidelity GPU performance simulator designed for datacenter-scale inference"（§3.1）；以 fragment|schedul|limitation|ignore|drawback|caveat 检索全文（abstract→§8、Appendix A–C）无任何关于显存碎片、调度异常或"细节被忽略"的表述；论文亦无 limitations 小节。｜修复要求：删除"论文承认…"整句，改写为明确标注的页面推断（如"论文未说明模拟器如何处理显存碎片、调度异常等真实系统细节；这些细节可能未被建模"），或直接删除该分句。｜修复：｜复验：
- [重要·技术] §1（index.html:135）：「论文给出的理由隐含在研究目标的两个特征中：①…②…」——论文没有给出"为何用模拟器而非真实集群"的理由，这是页面的重构被写成论文的理由（同一推断在 §5.2 复现）。｜引文依据：§3.1 仅陈述使用模拟器的事实，abstract/introduction/§8 均无该理由的任何表述。｜修复要求：把"论文给出的理由隐含在…"改为"论文未说明理由；从研究目标可推断…"，明确这是页面推断。｜修复：｜复验：
- [重要·技术] §3.3（index.html:258）：「变化剧烈的原因是不同模型的 TP/EP 调整能力差异——DeepSeek-R1 的 EP 可调性高，所以比例跨度大」——无来源支持的因果机制描述，且未标注为推断。｜引文依据：§4.3 仅写 "The optimal context-to-generation GPU ratio exhibits significant variation with model characteristics and target latency"，未给任何归因；§4.4 只讨论 NVLink 域大小对分离性能的影响，未把比例跨度归因于 EP 可调性。｜修复要求：删除该归因，或补 [推断] 标记并说明论文未给出该机制。｜修复：｜复验：
- [轻微·表述] §5 本章问题（index.html:409）：「论文的方法有什么关键限制，结论能直接外推到我的系统吗？」——使用第一人称会话指代"我的系统"，属规范列举应排除的会话指代。｜引文依据：不适用｜修复要求：改为无人称表述，如"…结论能否直接外推到其他系统？"。｜修复：｜复验：
- [轻微·技术] §3.3 正文与 Fig.9 图注（index.html:254、256）：「LLaMa-405B 从约 2.1 降到约 0.3」——末端读数与图不符。｜引文依据：Fig.9 中 LLaMa-3.1-405B（绿线）像素测量 x≈0.65→0.33、x≈0.70→0.245、x≈0.80→0.22、x≈0.84→0.21，末端稳定在约 0.2；起点 2.07（页面"约 2.1"正确），其余三个模型端点（3.6/0.05、0.95/0.45、≈0.4）均与图吻合，仅此值偏高。｜修复要求：将 405B 末端改为"约 0.2"，正文与图注（alt）同步。｜修复：｜复验：
- [轻微·技术] §4.3（index.html:327）：「论文的模拟结果（图 12）：…egress/ingress 带宽需求约在 0.4–1.8 GB/s/GPU 之间」——图 12 是 egress 与 ingress 两者的最大值，页面未说明，读者会把 0.4–1.8 误读为单独的 egress 或 ingress 需求。｜引文依据：Fig.12 caption 原文 "Bandwidth requirements for KV cache transfer: Maximum of egress and ingress bandwidth across various TTLs"。｜修复要求：补一句说明图 12 纵轴取 egress 与 ingress 的较大值，并把"egress/ingress 带宽需求"改为"两者较大者"。｜修复：｜复验：
- [轻微·表述] 核心问题 5 的解答（index.html:123）与 §5.1（index.html:369-377）：解答清单编号 ①–⑦，正文 §5.1 编号 ①–⑨，两者不一一对应（解答②合并了正文②③，解答⑦对应正文⑧，正文⑨无对应），而解答又写"完整论证见第 5 章"，交叉引用会对不上；解答另含"宽松延迟 + 生成密集时 chunking 最有利"一句，该句出自 introduction，§5.1 无对应内容。｜引文依据：不适用｜修复要求：解答编号与 §5.1 对齐（或改为不带编号的摘要），并把"宽松延迟 + 生成密集时 chunking 最有利"落进 §5.1 或从解答删除。｜修复：｜复验：
- [轻微·技术] 核心问题 4 的解答（index.html:116）：「DeepSeek-R1 模拟显示约 0.4–1.8 GBps/GPU，远低于现代 NVLink」——该外部比较无来源，且参照物选择不当：论文结论的比较对象是"existing provisioned datacenter bandwidth"，跨实例 KV 传输走的是网络（IB/RoCE），NVLink 是节点内互联，拿它作参照会高估余量。｜引文依据：论文 §5.1 结论句仅涉及 "existing provisioned datacenter bandwidth"，全文无 NVLink 与 KV 传输带宽的对比（overview.html 中该数据已标注"外部数据，非论文提供"，本页未标注）。｜修复要求：改为与论文口径一致的"远低于现有数据中心已配置带宽"，或删除该比较；若保留外部数字须标注为非论文来源并改用网络互联口径。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 5
- 处置：修复（阻断与重要项须逐条修复并复验，轻微项按上列要求处理）
