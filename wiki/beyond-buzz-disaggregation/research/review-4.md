<!-- review-meta
round: 4
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: 9fd9606f332eb77c
-->
# Beyond the Buzz 审查记录（第 4 轮）

- 页面版本：233678b6c679ff996829d73716b93955f364ed00
- 论文版本：arXiv:2506.05508v1（2025-06-05 18:47 UTC 提交；e-print 内 tex 成员日期 2025-06-06，与页面标注一致）
- 审查时间：2026-09-13 19:35
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（5 题）、术语速查、1. 方法：模拟器与设计空间（含本章问题）、2. 什么条件下分离收益最大（2.1/2.2/2.3，含本章问题）、3. 配套机制：切分策略与动态 rate matching（3.1/3.2/3.3，含本章问题）、4. 分离的代价：KV cache 传输带宽（4.1/4.2/4.3，含本章问题）、5. 方法评价：可操作结论与边界（5.1/5.2，含本章问题）、来源与范围说明。核对材料：arXiv:2506.05508v1 全文 HTML、e-print 源 tex（introduction / design_principles / disaggregation_in_practice / system_considerations / appendixA / appendixB / appendixD / related_work / future_work / conclusions）、页面全部 10 张原图。`.dojo/scripts/validate.py` 返回 validation ok。

## 问题

- [阻断·原图] 全页 8 张图的 `alt`（img-01/02/03/04/05/06/07/08）：alt 首部的 Figure 编号整体错位一位（写成 Fig.(N−1)），与同段正文的「图 N」及本页「来源与范围说明」的 Fig 编号表直接矛盾；点击图片时 lightbox 标题由 `lightboxCaption.textContent = img.alt` 生成（第 596 行），用户看到的图注与正文图号不一致。｜引文依据：论文源 tex 的 figure label 出现顺序为 fig:overview(Fig.1)、fig:figure2(Fig.2)、fig:rate_matching(Fig.3)、fig:chunked_pipelining(Fig.4)、fig:ctx_pp(Fig.5, caption "Chunked pipeline parallelism during Prefill is an optimal strategy… DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP")、fig:disagg_model_arch(Fig.6)、fig:disagg_model_size(Fig.7)、fig:traffic_sensitivity(Fig.8)、fig:figure6_ctx_gen_ratio(Fig.9)、fig:fixed_ratios(Fig.10)、fig:nvlink_sensitivity(Fig.11)、fig:bandwidth_requirements(Fig.12)、fig:dynamic_isl_osl_distribution(Fig.13)、fig:dynamic_vs_static(Fig.14)。页面实际写法：img-04 alt "Fig.4 DeepSeek-R1 Prefill ISL 256K…"（正文第 3.1 节写「图 5」，实为 Fig.5）；img-05 alt "Fig.5…架构敏感性对比"（2.3 节写「图 6」，实为 Fig.6）；img-06 alt "Fig.6 LLaMa-3.1-8B、70B、405B…"（2.2 节写「图 7」，实为 Fig.7）；img-07 alt "Fig.7…四种组合下的 Pareto"（2.1 节写「图 8」，实为 Fig.8）；img-03 alt "Fig.8…最优 Ctx:Gen GPU 比例"（3.3 节写「图 9」，实为 Fig.9）；img-02 alt "Fig.9…固定 Ctx:Gen Ratio 0.5"（3.3 节写「图 10」，实为 Fig.10）；img-01 alt "Fig.11…KV Transfer Requirements"（4.3 节写「图 12」，实为 Fig.12）；img-08 alt "Fig.13 动态流量模拟…与 P50 近似"（第 1 章写「图 14」，实为 Fig.14）。正文的「图 N」经逐条回源全部正确，alt 是唯一错源（img-10 的 Fig.1、img-09 的 Fig.2 正确，未列）。｜修复要求：把上述 8 处 alt 首部编号分别改为 Fig.5 / Fig.6 / Fig.7 / Fig.8 / Fig.9 / Fig.10 / Fig.12 / Fig.14，其余 alt 文字（内容描述本身与正确图相符）不动；改后全页「图 N」「Fig.N」「（Gx）」三套编号须一一对应。｜修复：｜复验：

- [重要·技术] 4.3 节末括注与 4.3 节「本章问题」第 3 题答案：NVLink / IB 绝对带宽数字无来源支持，却以来源结论口吻给出。｜引文依据：对论文 e-print 源 tex 全量检索 `GB/s|GBps|Tbps|Gb/s` 仅命中 system_considerations.tex:38 "Our analysis indicates that existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck."；§5.1 与 Fig.12 通篇未出现任何 NVLink 或 IB/RoCE 的每卡绝对带宽。页面第 331 行写「NVLink 单向带宽约 50–100 GB/s/卡，IB/RoCE 跨机约 10–25 GB/s/卡……$^{[C8]}$」，第 355 行重复一次。C 表与 N 表均未登记该数字来源；且该量级与现代 NVLink 不符（H100/Blackwell 每卡单向为数百 GB/s 量级），反而与 PCIe Gen5 量级接近。｜修复要求：或删除该组数字、只保留论文原话（现有供给足够），或移到明确的外部来源并给出量级正确的数值，同时在「外部数字与实验条件（N）」登记来源；不得继续挂 [C8] 且不加说明。｜修复：｜复验：

- [重要·表述] 第 69、71、236、380、467 行：存在元话语与以「本页/本文」为主语的自我指代，且同段内「本文」与「论文」指两个不同对象。｜引文依据：不适用（表述类）。原文：第 69 行「本文页面讲述论文做了什么、关键机制是什么、结论的边界在哪。本文讨论的"分离"是论文语境下 PD 分离……的简称」；第 71 行「本文的学习目标包括：说明论文用什么方法研究分离……」；第 236 行「需要注意的边界：图 5 是 ISL 256K 极端长序列的设置……」；第 380 行「MoE Serving 讲分离是什么……本页讲分离在什么条件下值得、怎么配、有什么代价」；第 467 行「Fig.11/13 与本文核心结论弱相关」。｜修复要求：删去「本文页面讲述…」「本文的学习目标包括…」两句元话语（或其改写为不带自我指代的正文句）；「需要注意的边界」改为「边界」类直陈标题式表述（如「该结论的边界在于……」）；「本页讲…」改为「本页内容侧重…」之外的直陈式（如「本文侧重分离的适用条件与代价」）或直接与前句并列陈述对象；同一段内「本文/论文」不得混指两个对象，须统一为「本文（页面）」或「论文（原文）」并各段一致。｜修复：｜复验：

- [重要·格式] img-04（第 232 行）、img-03（第 256 行）、img-02（第 262 行）的 alt：属性值内直接写裸 LaTeX 分隔符，KaTeX 不渲染属性，lightbox 图注会按字面显示。｜引文依据：不适用（渲染事实）。原文片段：`alt="Fig.4 DeepSeek-R1 Prefill ISL 256K，x 轴 PP 维度 1$\to$32。蓝线 Tokens/s/GPU 归一化到 PP=1 几乎水平$\approx$1；红线 FTL（log2）从 $2^{6.5}$ 降到 $2^{2}$。"`；img-03 alt 含 `$\approx$0.4`、`0.95$\to$0.45`、`2.1$\to$0.3`、`3.6$\to$0.05`；img-02 alt 含 `$\approx$0.4`。页面 KaTeX 通过 `renderMathInElement(document.body, …)` 只处理 body 文本节点，属性值不在其中；第 596 行 lightbox 用 `textContent` 赋值，点图后用户读到的是字面 `$2^{6.5}$`。｜修复要求：将三处 alt 中的数学记号改写为可直接显示的中文/纯文本（如「PP 维度从 1 增到 32」「归一化吞吐几乎保持 1」「log2 FTL 从 6.5 降到 2」「Ctx:Gen 比例从 0.95 降到 0.45」等），alt 内不得出现 `$…$`；改后点图核对图注无裸 LaTeX。｜修复：｜复验：

- [重要·技术] 「来源与范围说明」→「构造示例」第 2 条：列出页面中并不存在的构造示例。｜引文依据：不适用（页面内一致性）。原文：「8000 token / 512 一块 / 16 块、64 个 decode 贯穿的决策链：说明分与不分、怎么配、带宽预算的决策逻辑，人为设定的画像，标注构造示例。」全文检索「8000」「512 一块」「16 块」「64 个 decode」仅命中第 472 行本身，正文与折叠块中无此示例及其任何片段。｜修复要求：删除该条，或把该示例完整补写进正文章节并保留「构造示例」标注；不得在说明区引用正文不存在的内容。｜修复：｜复验：

- [轻微·技术] 「论断与来源（C）」分组与 C23 的实际出处不符。｜引文依据：related_work.tex（论文 §6）"they fall short of providing concrete guidance on when and how disaggregation is beneficial"、"prior research has largely focused on small-scale testbeds and peak throughput scenarios"。页面第 397 行以 [C23] 支撑这两句 §6 引文，但 C 表写「C1–C3、C9、C23–C25：abstract 与 introduction、§8 conclusions」，未列 §6 related work。｜修复要求：把 C23 的出处改为「§6 related work」单列，或将 §6 并入相应分组，使 C 表与正文引用一一对应。｜修复：｜复验：

- [轻微·技术] 第 333 行括注与 N8：称原图「不附绝对值」，与图 12 的绝对值纵轴矛盾。｜引文依据：kv_bw.pdf 纵轴标注 "Bandwidth/GPU (GBps/GPU)"，刻度 0.0–1.8（绝对值）；归一化的是横轴 "Normalized Tokens/s/user"。页面第 333 行「原文以归一化形式呈现不附绝对值」、第 450 行 N8「原文归一化无绝对值」均不成立。｜修复要求：改为「横轴为归一化 tokens/s/user，纵轴为 GBps/GPU 绝对值；曲线数值按图读出」，删除「不附绝对值」。｜修复：｜复验：

- [轻微·技术] 4.1 节 egress 手算示例的 MLA 参数与标注不符。｜引文依据：附录 A 与 DeepSeek-R1 公开架构：MLA 每 token 每层 KV 为潜向量 512 + rope 64 = 576 维，按论文 F1 应取 $d_{head}\times N_{kv\_heads}=576$；页面取 $d_{head}=128,N_{kv\_heads}=1$（乘积 128）并称「参数取自 DeepSeek-R1 公开架构」。算式本身无误：$61\times32\times16384\times128\times1\times1/(2\times8)=255{,}852{,}544$ B/s $\approx0.256$ GB/s。｜修复要求：把 $d_{head}=128$、$N_{kv\_heads}=1$ 的取法说明为「以单头 128 维近似表示 MLA 的 KV 维度，仅为量级示意」，或改用 576 维重算示例数值。｜修复：｜复验：

- [轻微·技术] 4.1 节第二段末的 `[C6 补充 F6]`。｜引文依据：system_considerations.tex:27-28 "when the tensor parallelism domain exceeds the number of KV heads, the KV cache is duplicated across tensor parallel ranks… only the GPUs that actually shard the KV cache should be considered in the normalization"（论文 §5）；而 C 表把 C5–C7 归为「§3 模拟方法」。｜修复要求：标注改为 `[C8, F6]` 或仅 `[F6]`。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 4 / 轻微 4
- 处置：修复。阻断与重要问题全部关闭前不得发布；本页其余部分核对结论为——正文「图 N」与论文 Figure 编号一致，标题/作者/日期（2025-06-05 提交，tex 打包 2025-06-06）、「数十万设计点」、>10B、FTL>10s 排除、FTL/TTL 量程、§3.2 满载与逐层即时传输假设、附录 B 两步算法与 tolerance=0.03、`decode_throughput/(OSL-1)`、附录 C 的 P50 / 最近 2 的幂、egress/ingress 公式与四条趋势、Fig.9/10/12 的读出数值（3.6→0.05、0.95→0.45、2.1→0.3、≈0.4、0.4–1.8 GBps/GPU）、CPP 的 FTL 90s→4s 与吞吐≈1、"existing provisioned datacenter bandwidth is sufficient…" 引文、§6/§7 引文、前置概念页存在性与 chunked-prefill 第 5 章引用均经回源核对无误；`.dojo/scripts/validate.py` 通过。