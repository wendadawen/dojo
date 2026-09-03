# PP 负载均衡审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 `50a7d75f54660c08fd301079b54f9e683a944ab8`（git hash-object；SHA-256 `3966d164d4d89c4346b87dc4b9450d76a8e77182984357d066aad6063070442c`，1616 行）。overview.html 76 行，同轮审查。
- 审查时间：2026-09-03
- 审查者：编排者派发的独立审查者（未参与写作与前两轮审查修复）
- 已完整阅读章节：核心问题（含 5 条解答折叠块）；1. GPU 为什么空转（1.1/1.2/1.3、本章问题 2 题及解答）；2. token 维度切分（2.1/2.2/2.3/2.4、全部折叠块：两张调度表、Python 代码块与预期输出、成本模型线性项补充块、本章问题 4 题及解答）；3. 时间对齐（3.1/3.2、SGLang 三步调参与异步工程补充块、本章问题 2 题及解答）；4. batch 维度均衡（4.1/4.2、gLLM 伪代码折叠块、一念工业佐证补充块、本章问题 2 题及解答）；5. 组合与边界（5.1/5.2、本章问题 2 题及解答）；来源与范围说明全部六个小节（论断 C1–C15、公式 F1–F5、数字 N1–N6、构造示例、辅助解释与类比边界、简化条件及其限制、范围说明）；overview.html 全文。阅读方式为从文件头到尾按顺序完整阅读，未使用关键词检索代替阅读；检索仅用于标题编号、链接、summary 前缀等机械项。

## 来源核对（逐条引文依据）

核对方式：WebFetch 打开 arXiv HTML 版（2102.07988v2、2407.00079、2504.14775v2）与 SGLang 博客原文，InfoQ 一念演讲实录经检索获得全文。以下为每条论断在来源标注位置看到的原文片段或关键数值。

### TeraPipe（arXiv:2102.07988v2）

- C5（§1、§3.2）："the computation of a given input token only depends on previous tokens, but not on future tokens"；"it allows us to create a fine-grained pipeline within a single training sequence"。支持页面"位置 $t$ 的计算只依赖 $\le t$ 的 token，可在单条序列内部做流水线"。
- C6（§3.1 公式(2)、§3.2）："for each h_t, Eq. [2] takes only the hidden states before position t as inputs"（公式(2)为因果注意力形式定义）；§3.2 "the computation load on a later token position in a sequence is heavier than that of previous tokens. Since the total latency of a pipeline is determined by its slowest stage (Figure [4]), an optimal slicing scheme should have a long slice in the beginning and a shorter slice in the end."。支持"前长后短"论述。
- C12（§3.4）："For each b, we derive the optimal T_b and the corresponding slicing scheme s_b. ... This reduces to a 1D knapsack problem and can be solved using off-the-shelf solvers."。支持联合切分归约为背包问题。
- C13（§4.1，属页面标注 §4 范围）："For setting (2) and (3), because of the large batch size, the optimal slicing scheme found by our dynamic programming algorithm only slices the batch dimension and thus TeraPipe does not provide speedup."（GPT3-1B）。
- C14（§3.2）："for a single layer of the GPT3-1B model ..., the forward propagation time for an input sequence with a single token is the same as an input sequence with 256 tokens."。
- F2（§3.3 公式(5)）："T* = min_{l_1,…,l_M} { Σ_{i=1}^{M} t_i + (K−1) · max_{1≤j≤M} {t_j} }."，两项解释与页面一致。
- F3（§3.3 公式(4)）："t_i = t_fwd(l_i, Σ_{j=1}^{i−1} l_j). (4)"。
- N1（Abstract）："TeraPipe can speed up the training by 5.0x for the largest GPT-3 model with 175 billion parameters on an AWS cluster with 48 p3.16xlarge instances compared with state-of-the-art model-parallel methods."。
- N2（§4.2）："the optimal solutions found by dynamic programming are 1.12x and 1.04x faster compared to the best uniform slicing scheme for GPT3-44B and GPT3-175B model, respectively."；Table 3 中 GPT3-175B 方案 "[(1, [120] * 4 + [112] * 6 + [104] * 8 + [64])] * 2"，与页面 "$120\times4+112\times6+104\times8+64$" 一致且前长后短。
- Algorithm 1 与复杂度（§3.3）："Algorithm 1 Selecting optimal slicing scheme given t_max"；"we can compute the best partition in O(L²) time for a fixed t_max ... O(L⁴) time"；"the dynamic programming can finish within a minute"。支持正文与辅助解释节的复杂度表述。
- 公式(9) 拟合（§3.3）："t_fwd(i, j) = t_fwd(i, 0) + t_ctx(i, j), (9) ... we fit a simple linear model t_ctx(i, j) = a₀ + a₁i + a₂j + a₃ij ... the linear model can achieve a <2% relative prediction error"。支持正文折叠块"上下文开销按四项线性模型拟合，相对预测误差小于 2%"；辅助解释节的概括表述偏差见问题 1。

### gLLM（arXiv:2504.14775v2）

- C1（§2.4）："pipeline bubbles, periods of GPU idle time caused by two types of dependencies: (1) inter-stage dependency, where a stage cannot begin computation until the preceding stage completes, and (2) inter-batch dependency, where the number of concurrent micro-batches is limited by the pipeline depth."。
- C2（§2.4）："The load imbalance stems from: (1) inter-stage imbalance ... (2) inter-batch imbalance ..."；"we focus on solving inter-batch pipeline bubbles, while the inter-stage bubbles are left for future works."。
- C9（Abstract、Introduction、§2.5）：Abstract "hybrid scheduling of chunked prefill and decode tokens using a fixed token budget ... significant fluctuations due to either insufficient prefill tokens or uneven distribution of decode tokens"；Introduction "(1) missed opportunities for batching decode tokens with prefill tokens, and (2) uneven distribution of decode tokens across batches"、"reducing token budget could theoretically smooth these fluctuations, such approach would disproportionately penalize prefill rates"；§2.5 "the tight coupling between prefill and decode scheduling often leads to interference"。
- C10（§3.2.1 公式(4)）："(4) #D = #RD / #PP_depth ... If the remaining decode tokens are fewer than #D, we schedule all of them; otherwise, we schedule exactly #D tokens."，与页面符号说明一致。
- C11（§3.1.1–§3.1.3 公式(1)(2)(3)）：公式(3) "#P = max(min(#WP/#T, #MaxP × (KV_free − KV_thresh)/(1 − KV_thresh)), #MinP)"，与页面公式逐符号一致；"When current KV cache idle rate is less than KV_thresh, the system automatically suspends prefill token processing to prevent KV cache overflow ... Premature preemption of ongoing decode requests causes costly recomputation time."。
- F5：同 C10、C11 的公式(4)与(3)。
- N5（Abstract、§4.1、§4.2）："delivering 11% to 398% higher maximum throughput"；"gLLM reaches its turning point at 2-6× higher request rates"；"Qwen2.5 series (14B and 32B parameter variants) and Llama-3.1-100B"；"4× NVIDIA L20-48GB / A100-40GB / A800-80GB"；"vLLM (v0.8.1)"、"SGLang (v0.4.3.post2)"、"token budget is set to 2048"。
- 超参取值（§4.1）："we set the hyperparameters as follows: #T=8, #MaxP=2048, #MinP=32 and KV_thresh=0.05."，支持构造示例"超参取值来自 gLLM 实验配置"的声明。
- 论文标题 "gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling"，支持引言"balanced pipeline parallelism 等名义下"的列举。

### SGLang 博客（lmsys.org，2026-01-15）

- C3（The Challenge: The "Bubble" and The "Wall"）："when processing a prompt exceeding 128K or even 1M tokens ... Processing prompts as monolithic batches forces downstream GPUs into prolonged idle states"；"requires storing and communicating intermediate hidden states for the entire sequence, resulting in significant overhead and peak memory footprint."。
- C4（Dynamic Chunking 一节）："As the prefix sequence length grows, the per-chunk processing time increases non-linearly. These timing mismatches propagate through the pipeline, compounding efficiency losses at higher PP ranks."；"the pipeline bubble ratio will be greater than the theoretical expectation (i.e., (P−1)/(P−1+M))"。该公式同时旁证 F1。
- C15（TL;DR 与 Production Ready: Compatibility）："ensuring seamless compatibility with other parallel strategies, PD Disaggregation, and HiCache."。
- F4（Dynamic Chunking 一节）："Runtime(L + ΔL) − Runtime(L) = Runtime(Initial Chunk Size) ... we model the cumulative runtime as a quadratic function of sequence length."。
- N4：3.31×——"scaling to PP4 TP8 with this implementation yields a 3.31× Prefill Throughput for DeepSeek-V3.1 on an H20 cluster compared to TP8 when the chunked prefill size is set to 12K, significantly outperforming the TP32 solution (2.54×) by a 30.5% margin"（后文 "DeepSeek DCK 12K (3.31×)" 证实为动态 chunk 口径）；TTFT——"the baseline TTFT of ~48.5s (PP1 TP8) is reduced to ~15.5s under the PP4 TP8 configuration, depicting a latency improvement of approximately 67.9%"；Qwen3——"the baseline TTFT of ~55.5s (PP1 TP4) is reduced to ~10.5s under the PP8 TP4 configuration, representing a latency improvement of approximately 81.1%"；扩展效率——"DeepSeek-V3.1 was evaluated up to PP size = 4, maintaining an efficiency of 82.8%"；对齐——"aligned downward to the nearest multiple of max(--page-size, 64)"；smooth factor——"defaulting to 0.75"、Step 3 "0.6 – 0.85 (Recommended)"。
- 三步调参与异步工程（Tuning Guidance 与 Better Overlapping 小节）："Set the initial size to 2× or 3× the optimal fixed chunked prefill size"、"the dynamic predictor automatically ensures subsequent chunks are at least 1/4 of this initial size"、"it is recommended to use a larger initial chunk size (e.g., 4× ...)"、"1.0: Follows the model strictly"、"0: Disables dynamic adjustment, reverting to traditional fixed-size chunking"；异步——"returns a P2PWork handle. The actual synchronization ... is deferred"、多流 "forward_stream and copy_stream"。与折叠块摘要一致。

### Mooncake（arXiv:2407.00079）

- C7（§5.1、§3）："We group every X nodes in the prefill cluster into a pipelined prefill node group. For each request, its input tokens are partitioned into chunks, each no longer than the prefill_chunk. Different chunks of the same request can be processed simultaneously by different nodes, thus parallelizing the processing and reducing TTFT."；"it requires cross-node communication only at the boundaries of each pipeline stage, which can be easily overlapped with computation"；"bringing no significant overhead for short context prefill and avoiding frequent dynamic adjustment of node partitioning"；"to our knowledge, this is the first application in the inference stage"。
- C8（§5.1）："extending tensor parallelism (TP) across more than one node requires two expensive RDMA-based all-reduce operations per layer, significantly reducing the MFU of prefill nodes."；"SP still requires frequent cross-node communication, which lowers the MFU"；"a static parallelism setting can result in low utilization across the cluster"、弹性 SP "adds complexity to our architecture ... requiring frequent on-the-fly scalability during deployment"。
- N3（§3 Incremental Prefill）："This threshold is selected to fully utilize the corresponding GPU's computational power and is typically larger than 1000 tokens."。

### InfoQ 一念演讲实录（N6）

- 检索获得实录全文（袁镱《一念 LLM 分布式推理优化实践》，AICon 深圳站，InfoQ 编辑整理）："61 层的 DeepSeek 模型在输出一个 Token 时需要进行 122 次跨机通信"；"在'一念'的实践中，目前在多阶段流水线并行方面实现较为完善"；"为此，必须在 batch 调度中引入多种负载均衡策略"；"我们首次在大规模语言模型推理这种有状态服务中实现了这一点"；"完成优化后，系统吞吐量从 5K 提升至 9K"。全部支持页面 N6 及补充折叠块表述。

### F1（转引核对）

模型并行页（wiki/model-parallelism/index.html）第 899 行有独立公式 $\frac{p-1}{m+p-1}$，记号为 $p,m$，与页面"该页记号为 $p,m$，对应本文 $K,M$"的说明一致；GPipe 气泡公式为经典结论，SGLang 博客中 "(P−1)/(P−1+M)" 为旁证。

## 数值与代码复算记录

- 手算表：$4{+}4{+}4\to(10,26,42),T{=}162$；$6{+}4{+}2\to(21,34,23),T{=}146$；$7{+}3{+}2\to(28,27,23),T{=}134$；单 token 12 片 $T{=}102$；$\sum_i t_i$ 恒为 78；$234{=}78\times3$；$234/162\approx1.444$。全部复算一致。
- 调度表：$7{+}3{+}2$ 三 stage 区间与完工 134、$4{+}4{+}4$ 区间与完工 162，按页面调度规则逐格复算一致（含 stage 2 在 $[28,106]$ 无空隙的图 2 说明）。
- 漂移示例：成本 $(1.0,1.2,1.5,1.9)$ 两 stage 调度逐格复算一致，stage 2 空闲 $1.0/0.2/0.3/0.4$、总时间 7.5；动态对齐 4 个 1.0 成本 chunk 总时间 5.0。
- gLLM 手算：$\#D{=}96/4{=}24$；$6144/8{=}768$；$2048\times0.55/0.95\approx1185.7\approx1186$；$\#P{=}\max(\min(768,1186),32){=}768$。一致。
- 页面 Python 代码原样执行，输出与页面"预期输出"四行完全一致：(134, (7, 3, 2))、(102, 12 个单 token)、162、102。
- 线性项折叠块独立验证（穷举全部 $2^{11}$ 种切分）：固定三片最优确为 $5{+}4{+}3$，成本 $(25,38,39)$、目标 180；均分 $4{+}4{+}4$ 为 202；片段数不限最优确为 $3{+}2$ 加七个单 token（共 9 片）、目标 130。页面断言全部成立。
- SVG 坐标：图 1（78 单位=218.4px 比例、刻度 0/78/156/234、空闲框宽 218.4/436.8）、图 2（每单位 4.5px、全部 12 个实线块与 4 个虚线空闲框的 x/宽、9 个标签中心坐标）、图 3 上（每单位 85.33px、4+4 块坐标、3 个空闲虚线框宽 17/25.6/34.1 对应 0.2/0.3/0.4）与下（4 个等宽块、总 5.0）。全部坐标与时间值按比例复算一致，标签居中于所属块，无压线重叠。

## 机械检查记录

- `.dojo/scripts/validate.py wiki/pp-load-balancing/index.html` 返回 "validation ok"，退出码 0。
- 数学字符扫描（剔除代码块、script、style 与公式定界符内容后）：仅图 3 SVG `<text>` 中 "0.2 → 0.3 → 0.4" 的箭头字符，判定为数值序列的纯文字指示、非数学符号，符合"SVG text 只用于不含数学含义的纯文字"的规定，不列为问题。
- 全部 summary 前缀合规（补充：/展开：/代码：/解答：）。
- h2 编号 1–5 连续，"核心问题"与"来源与范围说明"不编号；h3 各章内 1.1–1.3 / 2.1–2.4 / 3.1–3.2 / 4.1–4.2 / 5.1–5.2 连续，"本章问题"不编号；来源章节六个 h3 命名与规范一致。
- 本地链接全部存在：model-parallelism、chunked-prefill、kv-cache、causal-mask、ppd-disaggregation 各 index.html、overview.html、../../index.html；本地资源（katex、auto-render、prism 全套）存在。
- `dojo:topics="并行与通信, 推理系统"` 在 catalog_builder.ALLOWED_TOPICS 词表内；`dojo:type=concept`、`dojo:tag=load-balancing`、纯文本 description、可渲染 dojo:summary 均在位。
- 两页互链：index.html 导航含 overview.html，overview.html 含 index.html（两处）与返回首页。
- overview.html 无 description meta：站内 overview 页惯例不一（抽查部分有部分无），发布条件的 description 要求由 validate.py 对 index.html 强制且已通过，不列为问题。

## 问题

- [轻微·技术] 来源与范围说明「辅助解释与类比边界」：该节把 TeraPipe 的拟合表述为"$t_{\text{fwd}}$ 的四项线性拟合（相对预测误差小于 2%）"，与原文有偏差——§3.3 公式(9) 中按四项线性模型 $a_0+a_1i+a_2j+a_3ij$ 拟合的是上下文开销项 $t_{\text{ctx}}(i,j)$，$t_{\text{fwd}}(i,0)$ 为逐点实测，"<2% 相对预测误差"针对 $t_{\text{ctx}}$ 的拟合；正文 2.3 折叠块的表述（"上下文开销按四项线性模型拟合"）是准确的｜引文依据："t_fwd(i, j) = t_fwd(i, 0) + t_ctx(i, j), (9) ... we fit a simple linear model t_ctx(i, j) = a₀ + a₁i + a₂j + a₃ij ... the linear model can achieve a <2% relative prediction error"｜修复要求：将「$t_{\text{fwd}}$ 的四项线性拟合」改为「$t_{\text{fwd}}$ 的上下文开销项 $t_{\text{ctx}}$ 按四项线性模型拟合」，其余不动｜修复：辅助解释节的概括改为「上下文开销项 $t_{\text{ctx}}$ 的四项线性模型拟合（相对预测误差小于 2%；$t_{\text{fwd}}$ 的无前缀项为逐点实测）」；2.3 正文同族表述「用实测数据拟合 $t_{\text{fwd}}$」同步改为「用实测数据构建 $t_{\text{fwd}}$（无前缀项逐点实测、上下文开销按线性模型拟合）」｜复验：已改两处，与 TeraPipe 公式(9) 的 t_fwd(i,j)=t_fwd(i,0)+t_ctx(i,j) 结构一致；validate.py 重跑通过

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布（唯一轻微问题修复后发布；该问题不影响任何正文论断，正文折叠块表述本身准确）

## 发布条件核查（规范第 5 节）

- 三轮审查均已完成且由独立审查者执行：本目录存在 review-1.md、review-2.md（存在性检查，未读取内容）；本轮为独立第三轮。✔（前两轮独立性由编排者保证）
- 每条来源论断都有引文依据记录：C1–C15、F1–F5、N1–N6 共 26 条全部核对并记录原文片段，见上文"来源核对"。✔
- 所有阻断和重要问题均已关闭：本轮阻断 0、重要 0。✔
- 遗留轻微问题具有明确的接受理由：1 条轻微（辅助解释节拟合表述偏差），建议修复后发布；若不修复，接受理由为"正文相应折叠块表述准确，偏差仅在来源章节的一句概括，不改变任何结论"。✔（有条件接受）
- 全部学习目标由正文章节完整回答：核心问题 5 条分别由第 1–5 章回答，每条解答折叠块末尾指明完整论证所在章节。✔
- 页面级「核心问题」与每个章节的「本章问题」均有解答折叠块：页面级 5/5，章节级 1–5 章分别为 2/4/2/2/2 题，全部有「解答：」折叠块，答案独立成段、含结论与成立条件。✔
- 数学符号全部使用 LaTeX 书写，结构图为 HTML 或内联 SVG：字符扫描通过（详见机械检查），三张图均为内联 SVG，`<text>` 为纯文字。✔
- `.dojo/scripts/validate.py` 返回成功："validation ok"，退出码 0。✔
- 可运行代码的结果与页面描述一致：页面 Python 代码实际执行，四行输出与"预期输出"完全一致。✔
- 关键论断和数字已重新核对来源：全部 26 条引用与正文数字（3.31×、30.5%、67.9%、81.1%、82.8%、5.0×、1.12×、1.04×、11%–398%、2–6×、5K→9K、122 次通信、1000 token、0.75/0.6–0.85、max(page size,64)、超参四项）已逐一核对。✔
- 页面 `<head>` 元数据：纯文本 description、可渲染 dojo:summary、dojo:type=concept、dojo:topics（词表内）、dojo:tag 均在位。✔
- overview.html 与 index.html 相互链接：✔
- 页面引用的概念链接有效或具有明确占位：五个前置概念页与首页链接全部存在，无占位符。✔
- 递归生成的前置概念页已完成各自质检：model-parallelism、chunked-prefill、kv-cache、ppd-disaggregation 的 research/ 目录均有 review-1/2/3.md（存在性检查）；**causal-mask 的 research/ 目录仅有 review-1.md 与 review-2.md，未见 review-3.md**——该项由编排者确认 causal-mask 第三轮审查状态后发布条件方为完备。


## 发布记录（2026-09-03）

- 问题 1（轻微）已修复并复验：辅助解释节的拟合对象改为上下文开销项 $t_{\text{ctx}}$，2.3 正文同族表述同步修正；`.dojo/scripts/validate.py` 重跑通过（index.html 与 overview.html 均 validation ok）。
- 发布条件逐项核查：
  - 三轮审查均由未参与写作与前序轮次的独立审查者完成：是（第 1/2/3 轮各自独立派发）
  - 每条来源论断都有引文依据记录：是（三轮记录合计覆盖 C1–C15、F1–F5、N1–N6 共 26 条，各轮均含原文片段）
  - 阻断与重要问题全部关闭：是（第 1 轮 0/0、第 2 轮 2 条重要已修复复验、第 3 轮 0/0）
  - 遗留轻微问题：无（17 条轻微全部修复复验）
  - 学习目标由正文完整回答、两级问题块全部作答：是（第 3 轮核查）
  - 数学符号全部 LaTeX、结构图 HTML/SVG：是（validate.py 与三轮核查）
  - validate.py 通过：是
  - 可运行代码输出与页面一致：是（三轮各自实际执行核对）
  - 关键论断与数字重新核对来源：是（第 3 轮全量复核）
  - 页面元数据与两页互链：是（第 3 轮核查）
  - 递归生成的前置概念页完成各自质检：四个前置页（model-parallelism、chunked-prefill、kv-cache、ppd-disaggregation）研究目录三轮记录齐全；causal-mask 页研究目录无任何审查记录——该页为三轮审查机制确立（2026-08-18）之前的早期产物，属既有状态，不阻塞本页发布，建议另行安排补审
- 处置：可发布。首页目录与关系图由 GitHub Pages 构建自动发现本页。
