# 模型并行审查记录（第 1 轮）

- 页面版本：d096de12ced8513b51773af98adf46a2e945374d（index.html 工作树哈希）
- 审查时间：2026-08-19 21:12
- 审查者：独立子代理（未参与写作与规划，未读取 research/ 下任何文件）
- 已完整阅读章节：标题与引言 → 核心问题 → 1（1.1/1.2/1.3）→ 本章问题 → 2（2.1/2.2）→ 补充折叠块 → 本章问题 → 3（3.1/3.2/3.3）→ 本章问题 → 4 → 本章问题 → 来源与范围说明；overview.html 全文

## 来源论断逐条核对（C1–C10、F1–F2、N1–N2）

- C1 通过：Megatron §1 "we implement a simple and efficient model parallel approach using intra-layer model-parallelism…This approach is orthogonal to pipeline-based model parallelism as advocated by approaches such as GPipe"。
- C2 通过（章节号错误见问题 2）：§3 "Another option is to split A along its columns A=[A1,A2]…allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM"；行切需同步亦有原文 "will require a synchronization point before the GeLU function"。
- C3 通过（章节号错误见问题 2）：§3 "partitioning the GEMMs associated with key (K), query (Q), and value (V) in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU…the subsequent GEMM from the output linear layer…parallelized along its rows"。
- C4 通过（章节号错误见问题 2）：§3 "using only two all-reduces in the forward path and two in the backward path"；Figure 4 标题 "4 total communication operations in the forward and backward pass of a single model parallel transformer layer"。
- C5 通过：GPipe §2.2 "partitions the network into K cells…Communication primitives are automatically inserted at partition boundaries"；结论节 "Inter-device communication only takes place at partition boundaries for every micro-batch"。
- C6/F2 通过（章节号错误见问题 3）：GPipe §2.3 "This bubble time is O((K−1)/(M+K−1)) amortized over the number of micro-steps M"。页面 p↔K、m↔M 映射一致；slot 推导复算正确（3/7≈0.43、3/19≈0.16、2L=122、d/dp[(p−1)/(m+p−1)]=m/(m+p−1)²>0 均验证）。
- C7 通过（章节号错误见问题 3）：GPipe §2.3 "we only need to pass activation tensors at the partition boundaries between accelerators. Therefore, we can achieve efficient scaling performance even on accelerators without high-speed interconnects"。
- C8 通过：Sarathi §1 "In servers with high bandwidth connectivity such as NVIDIA DGX A100, tensor-parallelism can enable deployment of an LLM on up to 8 GPUs…Pope et al. show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods. However, tensor-parallelism at such a large scale can result in poor performance when hyper-clusters are unavailable"。但 §4 表格中 TTL 处的 [C8] 用法不成立，见问题 5。
- C9 部分通过：正交性有 Megatron §1 "orthogonal to pipeline-based model parallelism" 直接支持（页面标为推断，属保守标注，可保留）。EP × PP = 64 实例无法核对，见问题 4。
- C10 通过：Megatron §2.3 "data parallelism…where a training minibatch is split across multiple workers, and model parallelism in which the memory usage and computation of a model is distributed across multiple workers"。
- F1 通过（章节号错误见问题 2）：§3 Eq.(2)(3)、Figure 3a 内容与页面公式一致；但页面构造示例数值错误，见问题 1。
- N1 通过（章节号错误见问题 3）：GPipe §2.3 "In our experiments, we found the bubble overhead to be negligible when M ≥ 4 × K"。
- N2 通过（章节号错误见问题 2）：§3 "two all-reduces in the forward path"。

## 机械项检查

- 前置概念链接 standard-attention / gpu-communication / moe-serving / beyond-buzz-disaggregation / chunked-prefill / mla 的 index.html 均存在；libs 本地资源均存在。
- head 含 description、dojo:summary（KaTeX 可渲染）、dojo:type=concept、dojo:topics、dojo:tag；overview ↔ index 互链正常。
- 页面级「核心问题」4 题与各章「本章问题」（3/3/2/1 题）均有解答折叠块，答案独立可读、指明论证章节。
- 时间槽图（p=4, m=4）逐格核对正确。

## 问题

- [阻断·技术] index.html §1.1 及"完整手算"折叠块：构造示例数值不可复算。取 X=[1,2,3,4] 与页面给定的 A，实测 XA=[4,5,6,5]，页面写 [4,4,4,8]；XA1=[4,5]≠[4,4]，XA2=[6,5]≠[4,8]；"单卡直接算 XB=[4,4,4,8]"同错。等价性论证本身成立，但作为验证载体的数字全错，学习者手算必得不同结果｜引文依据：numpy 复算 X@A=[4 5 6 5]、X@A1=[4 5]、X@A2=[6 5]｜修复要求：更换 A 或 X 的取值使命中页述结果，或按实际结果改写全部数字，保证"单卡路径 = 两卡部分积之和"逐元素可复算；修改后重算验证｜修复：｜复验：
- [重要·技术] index.html 多处 Megatron 章节号错误：切分方案与 all-reduce 计数位于论文 §3 "Model Parallel Transformers"，页面在「来源与范围说明」（C1–C4 标 "§1–§2.2"）、F1、N2、第 1 章本章问题 Q1/Q2 答案（两处 "Megatron-LM §2.2"）均标为 §2.2；实际 §2.2 为 "Transformer Language Models and Multi-Head Attention" 背景节；C10 位于 §2.3｜引文依据：§3 原文 "split the second GEMM along its rows…only a single all-reduce operation in the forward pass (g operator)"；§2.2 标题原文｜修复要求：上述全部 §2.2 改为 §3，来源说明范围改为 §1–§3（C10 标 §2.3）｜修复：｜复验：
- [重要·技术] index.html F2、N1、C7 的 GPipe 章节号错误：气泡公式、M ≥ 4×K 阈值、低通信量论证位于 §2.3 "Performance Optimization"，页面标为 §2.2（§2.2 "Algorithm" 仅含 C5 的分区与 micro-batch 内容，C5 标 §2 正确）｜引文依据：§2.3 原文 "This bubble time is O((K−1)/(M+K−1))…negligible when M ≥ 4 × K" 与 "pass activation tensors at the partition boundaries"；§2.2 标题原文 "Algorithm"｜修复要求：F2、N1、C7 的章节改为 §2.3｜修复：｜复验：
- [重要·技术] index.html §3.2、核心问题 Q4 答案、第 3 章本章问题 Q2 答案及 overview.html："DeepSeek-R1 在 64 GPU 上用 EP × PP = 64"（来源说明标 Beyond the Buzz §4 图 5）——本轮允许来源不含该论文，无法核对｜引文依据：不适用（来源未提供）｜修复要求：补充 Beyond the Buzz（arXiv:2506.05508）原文供下轮核对；或降级为明确标注的推断；或删除该具体实例（涉及 4 处）｜修复：｜复验：
- [重要·技术] index.html §4 取舍表第一行 "对 TTL 类约束直接有效<sup>[C8]</sup>" 及第 4 章本章问题 Q1 答案的 [C4, C8] 引标：Sarathi §1 只论及 TP 支持大 batch 与高效 decode、可达 8 GPU，未论及 TP 缩短单层计算时间、降低单请求延迟或 TTL 约束；该行为综合判断，加 [C8] 引标即构成来源论断｜引文依据：Sarathi §1 原文 "increase batch size using model parallelism…supporting large batch sizes and efficient decode"，无延迟/TTL 论述｜修复要求：删除该两处 [C8] 引标，改为与表首一致的分析性判断表述（可保留 [C4] 对通信计数的引用）｜修复：｜复验：
- [轻微·格式] index.html 标题/正文/表格/description/summary：Unicode 乘号 "×" 直接出现（"TP × PP"、"TP 4 × PP 2 = 8"、"EP × PP = 64" 等多处）｜引文依据：不适用｜修复要求：统一改为 KaTeX `$\times$`（summary 为可渲染 LaTeX，允许）｜修复：｜复验：
- [轻微·可读性] index.html §2.1："其次数每个 stage 的忙碌时间。"语句不通｜引文依据：不适用｜修复要求：改为"其次，数每个 stage 的忙碌时间。"｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 4 / 轻微 2
- 处置：修复（数值示例与引标就地修正，不涉及范围与大纲变更；问题 4 需补充来源或降级处理）


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
