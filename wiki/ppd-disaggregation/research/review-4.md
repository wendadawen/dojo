<!-- review-meta
round: 4
page: wiki/ppd-disaggregation/index.html
reviewed_content_sha256: 316958d6213d438f
-->
# PPD 分离审查记录（第 4 轮）

- 页面版本：df36e452b31acb0c8e28c8adc464c24b614be81a
- 论文版本：arXiv:2603.13358v2（ICML 2026；v2 2026-05-05）
- 审查时间：2026-09-13 19:50
- 审查者：独立子代理（未参与写作，未读取 research/ 规划、修复与前序审查记录）
- 已完整阅读章节：核心问题、术语表、1. 多轮对话暴露 PD 分离的两个代价、2. full prefill 与 append-prefill 差一个数量级、3. 没有静态最优、4. PPD：把路由变成带权重的逐请求决策、5. 真实负载、慢网络与权重旋钮下的表现、6. 方法评价、7. 附：PPD 架构概念图、来源与范围说明、全部脚本与折叠块
- 核对方式：WebFetch/curl 拉取 arXiv HTML 原文（LaTeXML 版），逐条比对 §1–§8、App.A/B/C、Eq.(1)(2)、Table 1–5、Fig.1–8；六张原图与 assets 逐张目视核对；自绘 SVG 用 headless Chrome 渲染后核对标签重叠。
- 机械项：`.dojo/scripts/validate.py wiki/ppd-disaggregation/index.html` → `validation ok`；4 个前置概念链接（standard-attention、mqa-gqa、prefix-caching、increase-kv、gpu-communication）页面均存在；页面无 research/ 路径引用；无「（待生成）」占位。
- 已核对通过的关键数字（与原文一致）：S 公式与 Eq.(1) 逐字符一致；s_kv=128 KiB 及 2048×128 KiB=256 MiB、5115×128 KiB≈670 MB 换算自洽；batch200 +48%/+2%、4 并发 +57%/+21%、32K 3–4×/64K <25%；3060=17×18×10、10 QPS 档、18 负载（4+2+3 类）；Table 1/2/3/5 全表数值；92.2%、3.1 轮、≥75% KV 削减、3× 网络差；带宽模拟 143.7→170.6 ms(+18.7%)、约 51 ms、64%→70%、E2E +4.7%/+0.9%；w_tpot 1/3/6→95%/50%/20%、94–96%/7–12%（并已用 Fig.6 图面核对 −96%/+7.0%、−94%/+12.1%）；ψ=(t,n_in,n_out,n_ctx,q)、MD5/60 min TTL/10s–30s 心跳/Quart/ZeroMQ/kv_role；混合配置 6.1%/12.2%/16.1%；timeout 30s→10s 结论。

## 问题

- [重要·技术] 3.4 节（表 1 与紧随的「本章问题」答案）：页面在同一段内自相矛盾——表 1 列出 3P_1D 三档 −44.3%/−38.1%/−24.9%，答案却写「整段合成扫描范围 48–73%」。分项最小值（−24.9%）与所述合计区间（48–73%）不符，读者按页面自身数据无法复现该区间。｜引文依据：论文 Table 1 行 3P_1D = −44.3%/−38.1%/−24.9%；论文 §1 与 §4.1 又写「reduces Turn 2 TTFT by 48–73%」（该 48–73% 实为 P 稀缺配置 1P_3D/2P_2D 的下界，不含 3P_1D）。｜修复要求：明确标注 48–73% 是论文给出的汇总区间（对应 P 稀缺配置），或直接以表 1 的反例限定其成立条件，使正文/答案与表内数值不冲突。｜修复：｜复验：
- [重要·技术] 2.2 节「把这套数字翻成机制结论」段落：把 batch 200 处 full prefill 的 TPOT 抬升描述为「让同卡 decode 的延迟近乎翻倍」。图中实际为 +48%（≈1.46×，非 ×2，「翻倍」需 +100%），与同页图注「+48%」直接冲突，属夸大图中信息。｜引文依据：论文 §4.1「Full prefill causes ∼48% slowdown at batch size 200」；页面图注与图面标注均为「+48%」（decode-only ≈28 ms → +48% ≈41 ms）。｜修复要求：改为「抬升约 48%（约 1.5 倍）」等与 +48% 相符的表述，删除「近乎翻倍」。｜修复：｜复验：
- [重要·技术] 4.4 节 D 节点折叠条目：「而当请求被路由回 P（x=0）时，D 侧已命中的头部块可免于重传——把命中块数回传、P 侧只发增量即可」。这是页面新增的机制描述，论文实现章节未支持，且被表述为 PPD 原型的既有行为而非标注推断。｜引文依据：论文 App.B.2 全文仅三句——「P nodes run with kv_role=kv_producer … via ZeroMQ」「Decode servers run with kv_role=kv_consumer, receiving KV caches and storing them in local prefix cache for Turn 2+ processing when x>0」「The transfer protocol follows vLLM's standard disaggregated serving format, requiring no custom modifications」，通篇未提命中块免重传/回传块数/增量发送。｜修复要求：删除该句，或明确降级为「增量 KV 传输」这一通用机制（引用概念页为来源）并注明非论文所述 PPD 行为。｜修复：｜复验：
- [轻微·格式] 5.5 节失败率折叠块 summary：「展开：高 QPS 下失败率全表（附录 Table A2）」。论文该表编号为 Table 5，无「Table A2」；同页 N16 又标「App.C.4」，两处标号互不一致。｜引文依据：论文 App.C.4「Table 5: Failure rates at high QPS. x=1 configurations consistently show lower failure rates…」。｜修复要求：summary 与 N16 统一为「附录 Table 5 / App.C.4」。｜修复：｜复验：
- [轻微·图示] 第 1 章自绘 SVG（x=0 多轮请求路径）：渲染后客户端节点框（x 200–320, y 40–88）只包住「客户端」，其下「第 2 轮请求」「（追加 50 token）」两行文字溢出框外；顶部注释「D 上明明已有第 1 轮回复的 KV…」横向穿过该节点框，与框线重叠。｜引文依据：不适用（headless Chrome 渲染 assets 内联 SVG 实测）。｜修复要求：扩大或下移客户端节点框以容纳三行标签，并将顶部注释移出节点框区域，消除标签压框/重叠。｜修复：｜复验：
- [轻微·表述] 全文表述维度：6.2 节「（与 prefix caching 本身的逐出约束相关，本页写作范围内未展开）」以「本页」为主语的自我指代＋元话语；3.3 节「三组对比读起来很清晰：」为临场评价；4.1 节「应注意：吞吐不在打分函数内」为元话语（与规范禁止的「需要注意的是」同类）；2.2 节「图中需要关注的是 batch 200 处的标注」为图注重复的导读腔；来源与范围说明「第 4 章 interp 文字…」中「interp」为写作工具痕迹。｜引文依据：不适用。｜修复要求：逐处改为客观陈述（如「本页写作范围内未展开」→ 直接删除括注或改为对论文范围的陈述；「读起来很清晰」→ 直接给结论；「应注意」→ 直陈；「interp 文字」→「该段文字」）。｜修复：｜复验：
- [轻微·格式] 全页存储量单位标签不统一：正文写「约 256 MB」「约 156 MB」（对应 256 MiB / 156.25 MiB 的 1024 口径），来源说明 F5 却写「256 MiB」；而「约 670 MB」为论文的十进 MB 口径（5115 token 实为 639 MiB）。同一页对同一量纲混用 MB/MiB 两种口径。｜引文依据：论文 §2.2「each transfer is ∼256 MB」（1024 口径）；App.B.6「each Turn 2+ transfer … moves ∼670 MB」（十进口径）；页面 F5「2K×128 KiB = 256 MiB」。｜修复要求：统一正文与 F5 的口径标注（建议对页面自算值统一用 MiB，对直接引自论文的 256 MB / 670 MB 保留原值并注明口径）。｜修复：｜复验：
- [轻微·图示] 第 7 章 Fig.3 图注称「三种架构里『Last Turn's KV』（红虚线）始终在 D 上」，但 Replica 面板无 D 节点、也无该标注，该红虚线仅出现在 PD 与 PPD 两幅。｜引文依据：assets/img-01.webp（Fig.3）Replica 面板仅含「First-turn Prompt→R」「Turn-2+ Prompts→R」，无「Last Turn's KV」。｜修复要求：将「三种架构里」改为「PD 与 PPD 两幅中」或等价限定。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 5
- 处置：修复（重要问题须全部关闭后方可发布；本轮无需返回规划，问题均限于局部表述与标注）
- 附：本轮未发现阻断级问题；核心方法、公式（Eq.1）、复杂度、s_kv 换算、Table 1–5 全表数值、带宽公式与结论、实现细节（B.2–B.5）均已逐条回源核对一致。