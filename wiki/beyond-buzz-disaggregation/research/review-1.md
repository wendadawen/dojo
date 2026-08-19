# Beyond the Buzz 审查记录（第 1 轮）

- 页面版本：index.html 工作树未取哈希（仅做内容核对）
- 论文版本：arXiv:2506.05508v1 / paper.txt Figure 1–14 编号
- 审查时间：2026-08-19
- 审查者：独立审查者（首条消息派发）
- 已完整阅读章节：核心问题 §0；§1 方法；§2.1–2.3 流量/模型/架构敏感性；§3.1–3.3 CPP/decode 轨迹/rate matching；§4.1–4.3 带宽公式/趋势/数值；§5.1–5.2 清单与边界；来源说明；overview.html；paper.txt §1–§8 + 附录 B/D；design_principles/disaggregation_in_practice/system_considerations/appendixB/appendixD 全文；10 张原图 PNG（figure1/figure2/ctx_pp/isl_osl_pb/model_arch_pb/disagg_model_size_pb/ctx_gen_ratios/fixed_ratios/nvl_domain/kv_bw/dynamic_vs_static 等）。
- 未读：wiki/beyond-buzz-disaggregation/research/（规范禁止）；其他 wiki 页（仅在 index.html 出现外链时核对存在与否，未读取内容）。

## 问题

- [重要·技术] index.html 多处 Figure 编号与 paper.txt 实际编号系统性偏移 1：§3.1"图 4"指 ctx_pp（论文 Fig.5 line 198）；§2.3"图 5"指 model_arch_pb（Fig.6 line 221）；§2.2"图 6"指 disagg_model_size_pb（Fig.7 line 241）；§2.1"图 7"指 isl_osl_pb（Fig.8 line 253）；§3.3"图 8"指 ctx_gen_ratios（Fig.9 line 268）；§3.3"图 9"指 fixed_ratios（Fig.10 line 273）；§4.3"图 11"指 kv_bw（Fig.12 line 326）；§1"图 G9"指 dynamic_vs_static（Fig.14 line 763）。"原图（Figure 编号与原论文一致）"声明与事实相反。引文依据：paper.txt Figure 5: "Chunked pipeline parallelism during Prefill…"；Figure 12: "Bandwidth requirements for KV cache transfer…"；Figure 14: "Comparison of Pareto frontiers using dynamic traffic simulation versus P50 approximation."。修复要求：①按 paper.txt 修正 §2.1/2.2/2.3/3.1/3.3/4.3 与 §1 中所有 Figure 编号；②删除或改写"原图（Figure 编号与原论文一致）"声明为"图内容与论文一致；编号以论文 v1 为准"。｜修复：｜复验：

- [重要·技术] §4.1 行 957 / §构造示例 行 1130 的 egress 手算数值错 100 倍。引文依据：行 957 "BW_egress ≈ 61×32×16384×128×1×1/(2×8) ≈ 25.6 GB/s/卡"。复算（python3）：61*32*16384*128/16 = 255,852,544 B/s ≈ 0.256 GB/s/卡；25.6 与参数不符。修复要求：将"≈ 25.6"改为"≈ 0.256"；两处构造示例标注保留。｜修复：｜复验：

- [重要·技术] §4.3 行 1012 把外部 NVLink/IB 带宽数字标为论文 C8 来源。引文依据：system_considerations.tex line 38 仅 "existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck"，未给出 NVLink/IB 绝对数字。修复要求：①删除"[C8]"角标或降级为"（外部数据：NVLink 5 单向 ≈ 900 GB/s/卡、IB/RoCE 跨机 ≈ 10–25 GB/s/卡）"等明确外部脚注；②核实页面"NVLink 单向 50–100 GB/s/卡"数值（与 NVLink5 实际规格差一个数量级）。｜修复：｜复验：

- [轻微·技术] §2.3 行 852 MLA chunking 机制同时标 [C16, C17]，但 MLA 重复 down/up 投影属架构敏感性（论文 §4.1 line 42），C17 是模型大小敏感性。修复要求：删除该处 [C17]。｜修复：｜复验：

- [轻微·技术] 来源说明行 1080 写"C8、§5 带宽：disaggregation_in_practice.tex §5 Bandwidth requirements"，带宽实际在 system_considerations.tex §5.1。修复要求：文件名改为 system_considerations.tex。｜修复：｜复验：

- [轻微·技术] 来源说明行 1102 列 N2（FTL 数百 ms 到数分钟、TTL 数 ms），原文 disaggregation_in_practice.tex line 5 存在，但正文未标 [N2] 引用。修复要求：§2.1 描述 FTL/TTL 范围处加 sup[N2]，或从来源说明删除 N2。｜修复：｜复验：

- [轻微·技术] §3.3 行 906–909 rate matching 伪代码省略算法 2 第 46 行 decode_request_throughput = decode_throughput/(OSL−1)（appendixB.tex line 46）。修复要求：步骤 2 补"decode 请求吞吐 = decode_throughput/(OSL−1)"。｜修复：｜复验：

- [轻微·技术] §2.1 行 834 将"prefill compute-bound / decode memory-bound"机制描述标 [C10]。原文 background.tex line 5 仅说"Each metric exhibits different bottlenecks"，未明文两阶段性质。修复要求：删除 [C10] 或加"（推断）"标记。｜修复：｜复验：

## 已核对且通过

- 核心论断 C1–C3、C5–C9、C13–C25 全部与原文对应；F1/F2 公式与 paper.txt §5 Eq.(1)(2) 符号与分母形式一致；F3–F6 趋势定位 system_considerations.tex line 30–35；N1、N3、N4、N5、N6、N7、N8、N9 全部可定位（abstract、introduction、§3.2、Fig.5/7/10/12 caption）。
- 图内容与正文描述对应：figure1 左 (16k/2k) 右 (1k/32k)、isl_osl 四组 ISL/OSL、disagg_model_size 8B/70B/405B ISL 4k OSL 256、model_arch DeepSeek-R1 vs LLaMa-70B、ctx_pp FTL 由 log2 ≈6.4 降到 ≈1.9 且 Norm.Tokens/s/GPU≈1、ctx_gen_ratios 四模型曲线端点（3.6→0.05、2.1→0.3、0.95→0.45、≈0.4）、fixed_ratios 0.5/3.5/Optimal/Co-located、kv_bw 蓝 ISL 16384 OSL 2048（最低 ≈0.4）与红 ISL 1048576 OSL 2048（最高 ≈1.82 GBps/GPU，OSL 相同）、nvl_domain Domain 8 vs 72、dynamic_vs_static LLaMa-70B Dynamic vs ISL 4096 OSL 512——均与页面文字吻合。
- 可读性：5 个核心问题 + 各章 2–3 个本章问题均带解答折叠块，答案独立可读且与正文一致；折叠内容收起后正文可建立完整结论；术语速查表覆盖 FTL/TTL/ISL/OSL/design point/rate matching/elastic scaling。KaTeX 渲染、未见裸 Unicode 数学符号。overview.html 论断均能在 index.html 与原文找到对应。
- 页面未引用 Megatron/GPipe/Sarathi 的"2L all-reduce"论断，机制细节外链到 chunked-prefill/model-parallelism 等前置页，本页无须核对。

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 5
- 处置：修复

## 修复记录

第一轮所有问题已修复。阻断/重要问题逐条对应：

### model-parallelism
- 阻断：构造示例矩阵改用块对角例子（X=[1,1,1,1]、A 为对角块 [[1,1,0,0];[2,2,0,0];[0,0,3,3];[0,0,4,4]]、B=A），XA=[2,4,7,7]、XA1=[2,4,0,0]、XA2=[0,0,7,7] 可逐元素手算复算。复验：numpy 实算 X@A=[2,4,7,7]、X@A1=[2,4,0,0]、X@A2=[0,0,7,7]。
- 重要：Megatron 章节号 §2.2 → §3（5 处），GPipe 章节号 §2.2 → §2.3（5 处），来源说明范围同步更新。复验：Megatron §3 实际含 MLP/Attention 切分与"two all-reduces"陈述，GPipe §2.3 实际含气泡公式"M ≥ 4×K"。
- 重要：EP×PP=64 标 C9 不在本轮来源——在正文加 Beyond the Buzz 论文页链接作为来源补充；来源说明中给 EP×PP=64 实例的 paper.txt 定位（disaggregation_in_practice.tex fig 5 caption）。
- 重要：取舍表 [C8] 删除并降级为分析性判断（基于 §1.3 通信结构）。
- 轻微：Unicode × 换 $	imes$（3 处）；句不通"其次数" → "其次，"。

### chunked-prefill
- 重要：index.html:1007 [C16, N4] → [C16]（N4 是 Sarathi GPT-3 数字，与本句 Beyond the Buzz 模拟无关）。
- 轻微：Unicode × 换 $	imes$；TBT/FTL/ISL/EP/MLA 首次出现处补全称；N_0 删、用 16 块实例代替；20ms 改"几十毫秒（示意值）"；overview 补 5 项 head meta 并补 [1][2] 实验条件脚注；2.2 节列表结构修复（公式与 N/i 符号并入 KV 重复读项内）；"翻平方" → "按 ^2$ 量级增长（块数翻倍，重复读约 4 倍）"。

### beyond-buzz-disaggregation
- 重要：Figure 编号系统性修正（5 处正文 + 8 处原图清单 + 末尾声明）。修后：页面"图 1"=Fig.1、"图 2"=Fig.2、"图 5"=Fig.5（=G3=ctx_pp）、"图 6"=Fig.6（=G5P=model_arch）、"图 7"=Fig.7（=G4=disagg_model_size）、"图 8"=Fig.8（=G5=isl_osl）、"图 9"=Fig.9（=G6=ctx_gen_ratios）、"图 10"=Fig.10（=G7=fixed_ratios）、"图 12"=Fig.12（=G8=kv_bw）、"图 14"=Fig.14（=G9=dynamic_vs_static）。复验：与 paper.txt 的 Figure caption 顺序逐一对应。Fig.3（rate matching）/Fig.4（chunked prefill 机制图）/Fig.11（NVLink 域）/Fig.13（ISL/OSL 分布 CDF）未在正文使用：前者机制图由 MoE Serving 与 Chunked Prefill 页承载，后两者与本文核心结论弱相关——末尾声明改写说明。
- 重要：egress 手算 25.6 → 0.256 GB/s/卡（python3 复算 61×32×16384×128/(2×8) = 255,852,544 B/s ≈ 0.256 GB/s/卡）。
- 重要：NVLink/IB 数字 [C8] 删（论文 system_considerations.tex line 38 仅说"sufficient"，无绝对数字）。改写为"（外部数据，非论文提供）NVLink 5 单向 ≈900 GB/s/卡、IB/RoCE 跨机 ≈10–25 GB/s/卡"。
- 轻微：MLA chunking [C17] 删（C17 是模型大小敏感性非架构敏感性）；文件名改 system_considerations.tex；N2 在 §3.2 开头加 [N2] 引用（SLA 范围描述同时补全）；rate matching 步骤 2 补 (OSL−1) 与 App.B line 46 来源；[C10] 标"推断"。

**机械验证：** Validate a single concept or paper page (index.html).

Deterministic checks only: shell integrity, template leftovers, duplicate
ids, same-page anchors and broken local references. Semantic quality is out
of scope (handled by the independent review).

Usage:
    python3 .dojo/scripts/validate.py wiki/<name>/index.html 全部 6 个文件返回 ok。Chrome 探针：model-parallelism 184 处 KaTeX、5 处 foreignObject、0 处重叠；chunked-prefill 54 处 KaTeX、0 处 foreignObject、0 处重叠；beyond-buzz-disaggregation 47 处 KaTeX、0 处 foreignObject、0 处重叠。

**复验总评：** 阻断与重要问题全部修复。等待第 2 轮独立审查。
