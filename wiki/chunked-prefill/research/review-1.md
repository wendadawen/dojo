# Chunked Prefill 审查记录（第 1 轮）

- 页面版本：index.html 28b506acb93fd004d04f1226533a9258e1215125；overview.html 1c7ec4ec64dbcfe0059ef71fe2a5f5f924719d70
- 审查时间：2026-08-19
- 审查者：独立子代理（未参与写作与规划，未读 research/ 既有文件）
- 已完整阅读章节：核心问题 → 1 长 prefill 为什么卡住所有人（含本章问题） → 2 切成块：机制与代价（2.1/2.2/本章问题） → 3 搭车与 stall-free（3.1/3.2/本章问题） → 4 token budget 权衡（含本章问题） → 5 CPP（含本章问题） → 6 边界与相邻工作（含本章问题） → 来源与范围说明；overview.html 全文

## 来源论断核对（§2.2 步骤 1-4，均定位到原文）

- C1 ✓ Sarathi §3："the decoding phase is memory-bound"；"Prefill iterations have high latency but saturate GPU compute"（Sarathi-Serve §2 同）。
- C2 ✓ Sarathi §1（txt L431）："Decode under-utilizes GPU compute and costs as much as 200× prefill for batch size 1"；L72 "as high as ∼ 200 times the prefill cost per token"。
- C3 ✓ Sarathi §1（txt L56）："on an A6000 GPU, for the LLaMA13B model, a prefill with a sequence length of 512 tokens saturates GPU compute even at a batch size of just one"。
- C4 ✓ Sarathi-Serve 摘要："chunked-prefills which splits a prefill request into near equal sized chunks"；§4.2 "near-uniform compute requirement"（支持"等计算量"转述）。
- C5 ✓ Sarathi-Serve §4.2（txt）："the first chunk's KV-cache is loaded N − 1 times, the second chunk's KV-cache is loaded N − 2 times"；"even at small chunk sizes attention prefill operation is compute-bound"；"the computational cost is unchanged"；FFN 同量：Sarathi §3 "will perform the same number of computations for FFNs"。页面按 N-1 表述，与 Sarathi-Serve 一致；来源说明正确记录了 Sarathi 原文（"the first chunk's KV cache is loaded N times"，txt L~860 区域）为 N 次的差异。
- C6 ✓ 页面正文（2.1 节）与来源说明均明确标注为推断。补充：Beyond the Buzz Figure 4 caption 实有直接支持 "processing each chunk independently, using the KV cache from previous chunks but not their outputs"（paper.txt），可作为可选增强，非问题。
- C7 ✓ Sarathi 摘要："decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes… the decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch"。
- C8 ✓ Sarathi 摘要："Chunked-prefills allows constructing multiple decode-maximal batches from a single prefill request, maximizing coverage of decodes that can piggyback"。
- C9 ✓ Sarathi-Serve Figure 1 caption："one of the many generation stalls lasting over several seconds in vLLM"；prefill 优先归因："Prioritizing prefills optimizes throughput but sacrifices TBT (time-between-tokens) tail latency"。
- C10 ✓ Sarathi-Serve 摘要 "stall-free schedules that adds new requests in a batch without pausing ongoing decodes"；§4.2 "first calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO"，"first pack all the running decodes… After that, we include any partially completed prefill"。
- C11 ✓ Sarathi-Serve §4.3："This can be handled with a one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO"。
- C12 ✓ Sarathi-Serve §4.3：tile-quantization 段，"using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256"。
- C13 ✓ Sarathi-Serve §4.3："token budget can lead to higher overhead due to lower arithmetic intensity and other fixed overheads"；§4.2 末："small overhead associated with chunking due to fixed overheads of kernel launch, etc."。
- C14 ✓ Sarathi 摘要："The varying prefill and decode times also lead to imbalance across micro-batches when using pipeline-parallelism… the uniform compute design of these batches ameliorates the imbalance between micro-batches, significantly reducing pipeline bubbles"。
- C15 ✓ Sarathi-Serve 摘要："Mistral-7B on single A100… 2.6× higher serving capacity and up to 3.7× for Yi-34B on two A100… up to 5.6× gain… for Falcon-180B deployed with pipeline parallelism"。
- C16 ✓ Beyond the Buzz §4（disaggregation_in_practice.tex，main.tex input 顺序第 4 节）：paper.txt "In disaggregated serving, FTL constraints apply only to the prefill (context) pool… we found Chunked Pipeline Parallelism (CPP) to be especially effective"；Figure 5 caption："Chunked pipeline parallelism during Prefill is an optimal strategy to maximize throughput while complying with strict FTL SLA… DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"。
- C17 ✓ Beyond the Buzz tex-src/disaggregation_in_practice.tex:42："DeepSeek-R1 experiences additional overhead in piggybacked co-located serving due to prefill chunking—specifically, redundant computation of down and up projections in multi-latent attention for each prefill chunk. This can be mitigated by temporarily caching the up-projected KV values from earlier chunks"。页面表述与原文一致。
- F1 ✓ $\sum_{i=1}^{N}(N-i)=\frac{N(N-1)}{2}$ 复算正确；符号 N、i 已定义；标注为页面推导。
- N1 ✓ 见 C2（Sarathi §1）。N2 ✓ 见 C3（Sarathi §1）。N3 ✓ 见 C12（§4.3）。N4 ✓ Sarathi 摘要："reduces bubbles by 6.29×, resulting in an end-to-end throughput improvement of 1.91×"（GPT-3）。N5 ✓ 见 C15。N6 ✓ Sarathi 摘要："LLaMA-13B on A6000… decode throughput by up to 10×… end-to-end throughput by up to 1.33×. For LLaMa-33B on A100… 1.25× higher end-to-end-throughput and up to 4.25× higher decode throughput"。
- 第 6 章场景结论 ✓ introduction.tex："disaggregation provides the greatest benefits in prefill-heavy traffic scenarios (ISL >> OSL) and when serving larger models"；"context chunking is highly sensitive to the attention mechanism (e.g., MLA vs. GQA) and is most beneficial under relaxed latency targets and generation-heavy traffic patterns"。
- 构造示例：8000 token（第 1 章图 caption、第 2 章、来源说明均已标注）、4096/28/6（2.2 节与本章问题标注）、τ=512/64+448（3.2 节标注"构造示例"）；8000/448≈18 次迭代复算正确；64+448=512=τ 复算正确。
- 链接：moe-serving、causal-mask、gpu-execution-model、model-parallelism、beyond-buzz-disaggregation、首页、libs/katex 均存在；index/overview 互链正常。核心问题 5 题与各章本章问题均配解答折叠块且指明章节，学习目标均有对应章节论证。

## 问题

- [重要·技术] index.html:1007（第 5 章正文）：CPP 段引用标 `<sup>[C16, N4]</sup>`，但该句内容（DeepSeek-R1、ISL 256K、EP × PP = 64 模拟、PP 增大 FTL 降吞吐保持）完全来自 Beyond the Buzz，N4 是 Sarathi 的 GPT-3 气泡 6.29×/1.91× 数字，与本句无关；第 1022 行同内容正确标注 [C16]｜引文依据：Beyond the Buzz Fig.5 caption "DeepSeek-R1 with ISL of 256K on 64 GPUs using EP and PP (EP × PP = 64)"；Sarathi 摘要 "reduces bubbles by 6.29×"（GPT-3）｜修复要求：将 index.html:1007 的 [C16, N4] 改为 [C16]｜修复：｜复验：
- [轻微·格式] index.html:1007、1022（第 5 章正文与本章问题解答）："EP × PP = 64" 两处使用 Unicode 乘号 ×，违反公式书写规则（数学符号须 KaTeX 渲染）｜引文依据：不适用｜修复要求：改为 `$EP \times PP = 64$`｜修复：｜复验：
- [轻微·可读性] index.html:784（核心问题 4 解答）及第 5 章：TBT/SLO 首次出现于核心问题 4 解答，未解释（解释在第 1 章本章问题 2）；FTL、ISL、EP、MLA 在第 5 章首次出现，均未给全称或中文释义｜引文依据：不适用｜修复要求：核心问题 4 解答首次出现处写"TBT（相邻 token 间隔）"；第 5 章首次出现处给"FTL（首 token 延迟）""ISL（输入序列长度）""EP（专家并行）""MLA（Multi-head Latent Attention，多头潜在注意力）"｜修复：｜复验：
- [轻微·可读性] index.html:996（第 4 章本章问题解答）：公式 $\frac{L}{1024}\cdot\frac{N_0-1}{2}$ 中 $N_0$ 全页未定义；"若 256 仍在该硬件的饱和点以上（如 A6000/LLaMA-13B 的 512 以下则弱化），搭车论证仍大体成立；256 已低于该实例饱和点"括号插入位置错误，条件分支前后缠绕，需读两遍才能判定 256<512 时结论｜引文依据：不适用｜修复要求：删去 $N_0$ 表达式，用具体数字（1024 切 4 块 → 256 切 16 块等价改写，或直接引用 4096 例的 6→28）；饱和点句改写为"256 低于该实例饱和点 512，搭车'免费'论证在该硬件组合上弱化"｜修复：｜复验：
- [轻微·可读性] index.html:752（导语）："每 20 毫秒蹦一个 token"为构造叙事数字，未像 8000/4096/τ=512 一样标注为示意值｜引文依据：不适用｜修复要求：加"（示意值）"或改为"以几十毫秒的间隔"｜修复：｜复验：
- [轻微·技术] overview.html:48（问题背景）："一个 512 token 请求即可饱和单卡算力""每 token 成本可差约 200 倍"省略实验条件（LLaMA-13B/A6000 实例；200 倍为 batch size 1 小批量），与 index.html 的标注条件不一致｜引文依据：Sarathi §1 "on an A6000 GPU, for the LLaMA13B model… saturates GPU compute"；"costs as much as 200× prefill for batch size 1"｜修复要求：补"（LLaMA-13B/A6000 实例）"与"小批量下"｜修复：｜复验：
- [轻微·格式] overview.html `<head>`：缺 `description`、`dojo:summary`、`dojo:type`、`dojo:topics`、`dojo:tag` meta（check.md §5 发布条件要求页面 head 具备）｜引文依据：不适用｜修复要求：按 index.html 方式补齐五项 meta｜修复：｜复验：
- [轻微·可读性] index.html 2.2 节（约 L889-903）："切块的三个代价"列表被公式段、符号说明段、补充段拆成 4 个不连续块，第一条（KV 重复读）与其余两条（算术强度、固定开销）之间插入两段正文，列表结构破碎｜引文依据：不适用｜修复要求：将公式与 N/i 符号说明并入第一条 li（或移到三代价列表之后统一给出），"计算量本身不变…"补充段移到列表之后，使三个代价的 li 连续｜修复：｜复验：
- [轻微·可读性] index.html:907（2.2 节正文）："块数翻倍，重复读次数接近翻平方地增长"中"翻平方"为生造词，本章问题解答里"按 $N^2$ 量级增长"才是清楚表述｜引文依据：不适用｜修复要求：改为"按 $N^2$ 量级增长（块数翻倍，重复读约 4 倍）"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复。全部来源论断（C1–C17、F1、N1–N6）均已在三篇来源中定位到支持原文；KV 重复读按 N-1（Sarathi-Serve）表述正确且差异已记录；C17 在 Beyond the Buzz TeX 源码 disaggregation_in_practice.tex:42 有直接对应陈述；构造示例均已标注。仅需修复 1 条引用编号错误（N4 误标）及 8 条轻微问题后进入第 2 轮审查。


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
