<!-- review-meta
round: 7
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: 8868a9d56ed4bd44
-->
# Beyond the Buzz 审查记录（第 7 轮）

- 页面版本：0ca6f33cb017a328d38a2fdfa945530658925d30（wiki/beyond-buzz-disaggregation/index.html 工作树哈希）
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；本轮以 https://arxiv.org/html/2506.05508v1 全文 + 页面本地原图 assets/*.webp 为准）
- 审查时间：2026-09-13 21:48
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（页面级）、术语速查、1 章方法：模拟器与设计空间（含本章问题）、2 章什么条件下分离收益最大（2.1/2.2/2.3 含本章问题）、3 章配套机制：切分策略与动态 rate matching（3.1/3.2/3.3 含本章问题）、4 章分离的代价：KV cache 传输带宽（4.1/4.2/4.3 含本章问题）、5 章方法评价：可操作结论与边界（5.1/5.2 含本章问题）、来源与范围说明；含全部折叠块、表与图注。

## 核对方法

逐条回源：把页面每个数字与论断对照 arXiv:2506.05508v1 HTML 全文（节号/图号/引文），图内数值用 PIL 对页面本地原图做像素测量定位曲线端点。机械项（锚点、链接、数学符号、结构）由 .dojo/scripts/validate.py 校验。

## 问题

- [重要·技术] §3.1（index.html 第 230 行正文、第 232 行图注 alt）：图 5 中 FTL 终点的读数与图不符。页面写「FTL 从约 90 秒（log2 2^6.5）降到约 4 秒（log2 2^2）」；对 assets/img-04.webp 像素测量，右轴刻度 2^6/2^5/2^4/2^3/2^2 分别位于 y=163/286/407/528/649（等距 121.5 px），红线起点 y≈102.5 → log2 FTL≈6.50（即约 90 秒，与页面一致），终点 y≈704.5 → log2 FTL≈1.54（约 2.9 秒，明显低于 2^2 刻度线）。页面给的精确 log2 值 2^2 是图上的一条曲线未触及的刻度线。｜引文依据：图 5 原图 DeepSeek-R1 Prefill ISL 256K，红线 FTL SLA (log2) 终点像素 y≈704.5，2^2 刻度线在 y=649。｜修复要求：把「约 4 秒（log2 2^2）」改为「约 3 秒（log2 2^1.5）」，并同步第 232 行 alt 中的「从 6.5 降到 2」。｜修复：｜复验：
- [轻微·技术] §5.2 未覆盖方向（第 392 行）：「推理时计算技术（chain-of-thought、search-augmented generation）」把两个具体技术作为论文 future work 的举例并列，但论文未出现 chain-of-thought 或 search-augmented generation。｜引文依据：论文 §7 Future work「the impacts of KV cache reuse, speculation, inference-time compute techniques, and model architecture evolution appear to be particularly promising directions to pursue.」｜修复要求：删除括号内两例，或明确标注为页面举例（非论文所列）。｜修复：｜复验：
- [轻微·技术] 来源与范围说明（第 421 行）：C24–C25 的定位写为「§8 conclusions」，但 future work 内容在论文 §7（Future work），§8 为 Conclusions，二者错节。｜引文依据：论文节标题 1 Introduction … 6 Related work / 7 Future work / 8 Conclusions；future work 句在 §7。｜修复要求：C25 的定位改为 §7 Future work。｜修复：｜复验：
- [轻微·技术] §1（第 133 行、第 162 行）：「数据中心有足够 GPU 与请求使 rate-matched 部署满载」标注 [C8]（来源表定义为 §5.1），该假设实际出自论文 §3.2。｜引文依据：论文 §3.2「a datacenter setting with sufficient GPUs and incoming requests to fully utilize the rate-matched deployment」。｜修复要求：改引 §3（C5–C7）或注明 §3.2。｜修复：｜复验：
- [轻微·格式] 来源与范围说明（第 421–431 行、第 461 行）：来源表列出 C4、C11、C12 但正文从未引用（正文只出现 C1–C3、C5–C10、C13–C25）；Fig.6 的图号标签写作「G5P」，与其余 G1–G9 命名体系不一致且正文未使用该标签。｜引文依据：不适用｜修复要求：删除未引用编号或补齐相应引用；将「G5P」并入统一的 G 编号。｜修复：｜复验：

## 已核对无问题项（本轮）

- 公式：F1/F2 与论文 §5 Eq.(1)/(2) 逐符号一致；egress 手算 61×32×16384×128×1×1/(2×8)=0.2559 GB/s/卡，与页面 0.256 一致，且明确标注为构造示例。
- 数字回源：数十万设计点、>10B、FTL 数百毫秒–数分钟 / TTL 数毫秒、FTL>10s 排除、DeepSeek-R1 ISL 256K + 64 GPU（EP×PP=64）、Llama-70B TP 2×→64× 均与论文一致。
- 图内读数像素复核：图 9 四条曲线端点 DeepSeek-R1 3.63→0.059、70B 0.949→0.461、405B 2.07→0.213、8B≈0.41（页面 3.6→0.05 / 0.95→0.45 / 2.1→0.2 / ≈0.4 全部吻合）；图 12 蓝 1.23→0.38、红 1.82→1.0→1.42（页面 0.4–1.2、1.0–1.8、合计 0.4–1.8 吻合）；图 5 蓝线（吞吐归一化）全程 y≈294 平坦在 1.0。
- 图 1、6、7、8、10、14 的图号、模型、ISL/OSL 组合与正文解释一致；图 8 的「前两张分离优势大、后两张收窄」「generation-heavy 下 piggybacking 更有利」与图一致；图 6 的「DeepSeek-R1 piggybacked 远低于合设 overall、LLaMa-70B 接近」与图一致。
- 引文核对：related work「fall short of providing concrete guidance…」「largely focused on small-scale testbeds…」、§8「scenarios where disaggregation offers limited benefit」、§4.3「A similar effect is expected in small-scale GPU deployments」、§4.4「larger NVLink domains consistently enhance…」均与页面表述一致。
- 附录 B rate matching 两步算法（B/(FTL×G)、decode_throughput/(OSL−1)、α=best_prefill/decode_request、tolerance=0.03、num_prefill=numerator(α)×G_dec、num_decode=denominator(α)×G_prefill）与页面复述一致。
- 边界与推断标注：§2 的 compute-bound/memory-bound 分类标注为 [C10, 推断] 并注明论文 §2 仅说「different bottlenecks」；带宽数字与 0.256 示例均标注为读图/构造；summary/overview 与正文无矛盾。
- 结构：validate.py 返回 validation ok；dojo:type=paper、dojo:topics=并行与通信/推理系统/数学基础、dojo:tag=推理系统 均在词表内；6 个前置概念页链接目标真实存在；alt 无 $…$；公式均 KaTeX；无「待生成」占位。
- 表述：通读全文（含折叠块与图注）未见「本页将…」「下面来看…」类元话语、会话指代、调试叙事或临场评价；推断与构造示例均以「页面推断」「构造示例」显式标注。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（无阻断；1 条重要（图 5 FTL 终点读数）需修正后复验，4 条轻微见上）
