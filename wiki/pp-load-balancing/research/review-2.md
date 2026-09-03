# PP 负载均衡审查记录（第 2 轮）

- 页面版本：ed83eff48b55febd8050d4cdcc7db0e3f030e206（index.html 工作树 SHA-1）
- 审查时间：2026-09-03 14:09
- 审查者：独立审查者（未参与写作，未参与第 1 轮审查与修复）
- 已完整阅读章节（index.html，从开头到结尾按顺序，含全部折叠块）：head 元信息与样式脚本、h1 与 meta 引言、核心问题（5 题及解答）、1. GPU 为什么空转（1.1/1.2/1.3/本章问题）、2. token 维度切分（2.1/2.2/2.3 含两个折叠块与代码折叠块/2.4/本章问题）、3. 时间对齐（3.1/3.2 含补充折叠块/本章问题）、4. batch 维度均衡（4.1/4.2 含伪代码与一念折叠块/本章问题）、5. 组合与边界（5.1/5.2/本章问题）、来源与范围说明（C/F/N/构造示例/辅助解释/简化条件/范围说明）、页尾脚本；overview.html 全文（76 行）。

## 来源核对记录

核对方式：WebFetch 打开 arXiv HTML 版与博客原文，定位到页面标注的章节/公式号，摘录原文片段。

### TeraPipe（arXiv:2102.07988，HTML 版）

- C5（§1、§3.2）：原文 "the computation of a given input token only depends on previous tokens, but not on future tokens"；§3.2 "for an input hidden state sequence … the computation of a self-attention layer SelfAtt(h_t) only depends on the hidden states of previous positions"。支持页面表述。
- C6（§3.1 公式(2)、§3.2、Figure 4）：原文 "the computation load on a later token position in a sequence is heavier than that of previous tokens. Since the total latency of a pipeline is determined by its slowest stage (Figure 4), an optimal slicing scheme should have a long slice in the beginning and a shorter slice in the end."。支持。
- C12（§3.4）：原文 "This reduces to a 1D knapsack problem and can be solved using off-the-shelf solvers."，且流程与页面描述一致（对每个 b 跑 §3.3 DP 得 T_b 与 s_b，再选 b_1+…+b_D=B 最小化总和）。支持。
- C13（§4.1）：原文 "For setting (2) and (3), because of the large batch size, the optimal slicing scheme found by our dynamic programming algorithm only slices the batch dimension and thus TeraPipe does not provide speedup."（GPT3-1B）。支持。
- C14（§3.2）：原文 "for a single layer of the GPT3-1B model … the forward propagation time for an input sequence with a single token is the same as an input sequence with 256 tokens."。支持。
- F2（§3.3 公式(5)）：原文 $T^{*}=\min_{l_1,\ldots,l_M}\{\sum_{i=1}^{M}t_i+(K-1)\cdot\max_{1\le j\le M}\{t_j\}\}$，逐字符一致。支持。
- F3（§3.3 公式(4)）：原文 $t_i = t_{fwd}(l_i, \sum_{j=1}^{i-1} l_j)$，逐字符一致。支持。
- N1（Abstract）：原文 "TeraPipe can speed up the training by 5.0x for the largest GPT-3 model with 175 billion parameters on an AWS cluster with 48 p3.16xlarge instances compared with state-of-the-art model-parallel methods."。支持。
- N2（§4.2 及 Table 2/3）：原文 "the optimal solutions found by dynamic programming are 1.12x and 1.04x faster compared to the best uniform slicing scheme for GPT3-44B and GPT3-175B model"；切分方案 `[(1, [120]*4 + [112]*6 + [104]*8 + [64])] * 2` 出自 Table 2（Appendix C）/Table 3，非 §4.2 正文——定位需补 Table 2/3（见轻微问题 5）。数字本身支持。
- 复杂度（§3.3）：原文 "we can compute the best partition in O(L^2) time for a fixed t_max … in O(L^4) time"，Optimization 段落含 ε 步长、一分钟内完成。支持页面「固定 $t_{\max}$ 时 $O(L^2)$、总体 $O(L^4)$、分钟级」。
- 四项线性拟合（§3.3）：原文 "$t_{ctx}(i,j)=a_0+a_1 i+a_2 j+a_3 ij$ … <2% relative prediction error"，公式(9) 为分解式。支持页面补充折叠块的表述。

### Mooncake（arXiv:2407.00079，HTML 版）

- C7（§5.1、§3）：原文 "We group every X nodes in the prefill cluster into a pipelined prefill node group."；"its input tokens are partitioned into chunks, each no longer than the prefill_chunk. Different chunks of the same request can be processed simultaneously by different nodes … reducing TTFT."；"CPP offers two main benefits: 1) … cross-node communication only at the boundaries of each pipeline stage, which can be easily overlapped with computation … 2) It naturally fits both short and long contexts, bringing no significant overhead for short context prefill and avoiding frequent dynamic adjustment of node partitioning."；"to our knowledge, this is the first application in the inference stage"。全部支持。
- C8（§5.1）：原文 "extending tensor parallelism (TP) across more than one node requires two expensive RDMA-based all-reduce operations per layer, significantly reducing the MFU of prefill nodes."；"SP still requires frequent cross-node communication, which lowers the MFU …"；"a static parallelism setting can result in low utilization across the cluster"（弹性伸缩复杂度）。支持。
- N3（§3）：原文 "This threshold is selected to fully utilize the corresponding GPU's computational power and is typically larger than 1000 tokens."。支持。

### SGLang 博客（lmsys.org，2026-01-15）

- C3（The Challenge: The "Bubble" and The "Wall"）：原文 "Processing prompts as monolithic batches forces downstream GPUs into prolonged idle states"；"requires storing and communicating intermediate hidden states for the entire sequence, resulting in significant overhead and peak memory footprint"。内容支持；但两句话出自同一小节的两个编号要点，页面 C3 定位写「两节」不准确（见轻微问题 4）。
- C4（Dynamic Chunking 一节）：原文 "As the prefix sequence length grows, the per-chunk processing time increases non-linearly. These timing mismatches propagate through the pipeline, compounding efficiency losses at higher PP ranks."；"the pipeline bubble ratio will be greater than the theoretical expectation (i.e., (P−1)/(P−1+M))"。支持。
- F4（Dynamic Chunking 一节）：原文 "Runtime(L + ΔL) − Runtime(L) = Runtime(Initial Chunk Size)"，二次函数拟合 "we model the cumulative runtime as a quadratic function of sequence length"。支持。注意原文未说 "offline"（见轻微问题 3）。
- N4：3.31×（"yields a 3.31× Prefill Throughput for DeepSeek-V3.1 on an H20 cluster compared to TP8 when the chunked prefill size is set to 12K"）、30.5%、48.5s→15.5s（67.9%）、Qwen3-235B-A22B-FP8 PP8 TTFT 降 81.1%（"~55.5s … reduced to ~10.5s … approximately 81.1%"）、对齐到 max(page-size, 64) 倍数、smooth factor 默认 0.75 推荐 0.6–0.85——均支持。但 82.8% 强扩展效率原文明确归属 DeepSeek-V3.1（"Due to resource constraints, DeepSeek-V3.1 was evaluated up to PP size = 4, maintaining an efficiency of 82.8%"），Qwen3 相关仅有泛指的 "over 80% scaling efficiency"——页面把它放在 Qwen3 句中，归属错误（见重要问题 1）。
- C15（TL;DR 与第 4 节）：原文 "seamless compatibility with other parallel strategies, PD Disaggregation, and HiCache"。支持。
- 三步调参与异步工程（折叠块摘要）：Step 1/2/3 原文逐条核对（初始 2×–3×、超长 4×、不小于初始 1/4、smooth factor 语义）；异步 P2P（async_send 返回 P2PWork 句柄、同步推迟到 _pp_commit_comm_work、forward_stream/copy_stream 多流）。支持。

### gLLM（arXiv:2504.14775，HTML 版）

- C1（§2.4）：原文 "(1) inter-stage dependency, where a stage cannot begin computation until the preceding stage completes, and (2) inter-batch dependency, where the number of concurrent micro-batches is limited by the pipeline depth."。支持。
- C2（§2.4）：原文 "The load imbalance stems from: (1) inter-stage imbalance … and (2) inter-batch imbalance …"；"we focus on solving inter-batch pipeline bubbles, while the inter-stage bubbles are left for future works."。支持。
- C9（Abstract、Introduction、§2.5）：原文 "(1) missed opportunities for batching decode tokens with prefill tokens, and (2) uneven distribution of decode tokens across batches"；"reducing token budget could theoretically smooth these fluctuations, such approach would disproportionately penalize prefill rates"；§2.5 耦合干扰（"tightly coupling prefill and decode scheduling under a fixed total token budget cannot effectively satisfy their respective requirements"）。支持。
- C10 / F5 decode 式（§3.2.1 公式(4)）：原文 "(4) #D = #RD / #PP_depth … If the remaining decode tokens are fewer than #D, we schedule all of them; otherwise, we schedule exactly #D tokens."；decode 动机原文 "decode operations require multiple iterations (equal to the output sequence length) while prefill operations typically complete in a single iteration. The variation in decode requests is relatively small"。支持。
- C11 / F5 prefill 式（§3.1.1–§3.1.3 公式(1)(2)(3)）：公式(3) 原文 "#P = max(min(#WP/#T, #MaxP × (KV_free − KV_thresh)/(1 − KV_thresh)), #MinP)" 与页面逐字符一致；暂停机制原文 "the system automatically suspends prefill token processing to prevent KV cache overflow … Premature preemption of ongoing decode requests causes costly recomputation time."。支持。
- 超参取值（§4.1 Schemes）：原文 "#T=8, #MaxP=2048, #MinP=32 and KV_thresh=0.05"。与页面构造示例声称的「超参取值来自 gLLM 实验配置」一致。
- N5（Abstract、§4.1、§4.2）：原文 "11% to 398% higher maximum throughput"；"gLLM reaches its turning point at 2-6× higher request rates"；Qwen2.5 (14B/32B)、Llama-3.1-100B、L20/A100/A800；vLLM (v0.8.1)、SGLang (v0.4.3.post2)、"token budget is set to 2048"。支持（页面写 v0.4.3，原文 v0.4.3.post2，差异可忽略，不单列）。

### InfoQ 一念演讲实录

- N6：原文 "61 层的 DeepSeek 模型在输出一个 Token 时需要进行 122 次跨机通信"；"在'一念'的实践中，目前在多阶段流水线并行方面实现较为完善"；"必须在 batch 调度中引入多种负载均衡策略"；"我们首次在大规模语言模型推理这种有状态服务中实现了这一点"；"完成优化后，系统吞吐量从 5K 提升至 9K"。全部支持，页面已标注「工业口径，无公开论文细节」。

### F1 转引核对

模型并行页（wiki/model-parallelism/index.html）§2.1 确有气泡公式 $\frac{p-1}{m+p-1}$（记号 $p,m$），与页面 F1「转引自模型并行页（该页记号为 $p,m$，对应本文 $K,M$）」一致。但 F1 写「GPipe（Huang et al., 2019）」，模型并行页与 arXiv:1811.06965 均为 2018——年份错误（见轻微问题 6）。

## 本地复算与代码执行记录

- 页面代码折叠块（穷举 + TeraPipe 式 DP）实际执行：输出与页面「预期输出」四行完全一致——`固定切成 3 片的最优方案: (134, (7, 3, 2))`、`片段数不限的 DP 最优方案: (102, (1,)*12)`、`均分 4+4+4 的目标值: 162`、`单 token 切 12 片的目标值: 102`。执行通过。
- 手算表复算：$1+\cdots+12=78$；$4{+}4{+}4\to(10,26,42)$、$T=162$；$6{+}4{+}2\to(21,34,23)$、$T=146$；$7{+}3{+}2\to(28,27,23)$、$T=134$；不切分 $T=234$；单 token 12 片 $T=78+2\times12=102$。全部一致。
- 调度表复算（7+3+2 与 4+4+4 两张表逐格）：与「片段 i 开始时刻 = max(stage k 空闲, 离开 k−1 时刻)」规则一致，完工 134/162 与目标函数值相等。图 2 各矩形坐标按 4.5 px/单位换算全部吻合。
- 漂移示例复算：成本 $(1.0,1.2,1.5,1.9)$ 两 stage 调度，stage 2 空闲 $0.2/0.3/0.4$、总时间 7.5；对齐到 1.0 后总时间 5.0。与页面表格与图 3（85.3 px/单位）坐标全部吻合。
- gLLM 构造示例复算：$\#D=96/4=24$；$\#WP/\#T=6144/8=768$；$2048\times0.55/0.95\approx1185.7\approx1186$；$\#P=\max(\min(768,1186),32)=768$。一致。
- 线性开销变体复算（独立验证）：固定 3 片时最优确为 $5{+}4{+}3$，成本 $(25,38,39)$、目标值 $180$——与页面一致；但片段数不限时最优为 $3{+}2{+}1{+}1\times6$ 共 9 片、目标值 $130$——页面未限定三片（见重要问题 2）。
- `.dojo/scripts/validate.py wiki/pp-load-balancing/index.html` 返回 `validation ok`。
- 链接检查：model-parallelism、chunked-prefill、kv-cache、causal-mask、ppd-disaggregation 各 index.html 均存在；libs 下 katex/prism 资源均存在；overview.html 与 index.html 相互链接；站内锚点由 validate.py 覆盖通过。
- dojo:topics「并行与通信, 推理系统」在 AGENTS.md 固定大类词表内。

## 可读性与格式审查记录

- 术语首现解释：TTFT、RDMA、MFU、SP、CPP 均在首现处给出中文展开；micro-batch、chunked prefill、气泡公式由前置概念页承担且已链接。一处例外：「PP rank」未解释（见轻微问题 11）。
- 前置知识时机：气泡公式在 1.2 使用前已链接模型并行页；因果性在 2.1 使用前已链接因果掩码页。无违例。
- 公式符号：五个正文公式后均有逐项符号定义；$K,M$ 全文一致；伪代码符号与公式一致。
- 折叠块收起后正文完整性：调度表、线性变体、SGLang 调参、gLLM 伪代码、一念案例均在折叠块内，但正文结论均不依赖展开内容。通过。
- 章节衔接：1→2、2→3、3→4、4→5 均有一至两句衔接。通过。
- 学习目标覆盖：核心问题 5 条分别由第 1–5 章回答，每条解答指明论证所在章节。通过。
- 两级问题块：页面级 5 题、章节级 2/4/2/2/2 题均有「解答：」折叠块，答案独立可读、与正文一致。通过。
- 结构命名：h2 编号 1–5 连续、「来源与范围说明」不编号；h3 编号连续、「本章问题」不编号；来源章节六个固定小节命名齐全；details summary 前缀（展开/补充/代码/解答）合规；`<sup>[Cx/Fx/Nx]</sup>` 引用格式合规；正文引用其他章节使用章节标题。通过。
- 格式违例：正文与来源章节 3 处 Unicode 乘号「×」（见轻微问题 12）；SVG `<text>` 内 ASCII 近似数学写法 s1/c1（见轻微问题 9）。

## 问题

- [重要·技术] index.html 3.2 节实测收益段（「Qwen3-235B-A22B-FP8 在 PP8 下 TTFT 降 81.1%，PP4 规模保持 82.8% 的强扩展效率」）及 N4 条目：82.8% 强扩展效率归属错误。该数字在 SGLang 博客中明确属于 DeepSeek-V3.1（"Due to resource constraints, DeepSeek-V3.1 was evaluated up to PP size = 4, maintaining an efficiency of 82.8%"），Qwen3 相关表述只有泛指的 "over 80% scaling efficiency"；页面把 82.8% 放在 Qwen3 主语的分句中，读者会把它当成 Qwen3 的数字｜引文依据："DeepSeek-V3.1 was evaluated up to PP size = 4, maintaining an efficiency of 82.8%"（博客 Performance Impact 节）；"it maintains over 80% scaling efficiency for various model architectures while scaling out to PP4"（Introduction，泛指各模型）｜修复要求：3.2 正文把 82.8% 移回 DeepSeek-V3.1 条件下（例如「DeepSeek-V3.1 在 PP4 规模保持 82.8% 的强扩展效率」），N4 条目同步注明 82.8% 属于 DeepSeek-V3.1；修 overview.html 无涉及（该页无此数字）｜修复：3.2 正文改为「Qwen3-235B-A22B-FP8 在 PP8 下 TTFT 降 81.1%；DeepSeek-V3.1 在 PP4 规模保持 82.8% 的强扩展效率」；N4 条目同步注明 DeepSeek-V3.1 归属｜复验：已改，正文与 N4 均把 82.8% 归到 DeepSeek-V3.1 名下
- [重要·技术] index.html 2.3 节折叠块「补充：成本模型怎么影响最优切分」：声称「最优切分从 $7{+}3{+}2$ 变为 $5{+}4{+}3$」未限定「固定三片」。独立复算：固定 3 片时最优确为 $5{+}4{+}3$（成本 $(25,38,39)$、目标值 180），但片段数不限时最优为 $3{+}2{+}1{+}1{\times}6$ 共 9 片、目标值 130 < 180。页面同节的代码折叠块刚演示过「片段数不限的 DP」，读者按同一方式复算会得出与「$5{+}4{+}3$ 是最优」矛盾的数字｜引文依据：本地复算输出——固定 3 片最优 `(180, (5, 4, 3), [25, 38, 39])`；片段数不限最优 `(130, (3, 2, 1, 1, 1, 1, 1, 1, 1), [12, 13, 8, 9, 10, 11, 12, 13, 14])`｜修复要求：折叠块中把「最优切分从 $7{+}3{+}2$ 变为 $5{+}4{+}3$」改为明确限定三片的表述（如「固定三片的最优切分从 $7{+}3{+}2$ 变为 $5{+}4{+}3$」），避免与片段数不限情形（更细切分仍更优）混淆｜修复：折叠块改为「固定三片的最优切分从 7+3+2 变为 5+4+3」并补充片段数不限时的 9 片 130 结果与解释｜复验：已改；独立复算脚本确认固定三片 (180, (5,4,3), [25,38,39])、不限片段 (130, (3,2,1×7))，与修复后表述一致
- [轻微·技术] index.html 3.2 节：「先离线 profile 一批不同长度的请求」中的「离线」无来源支持，博客原文只说 "By profiling a series of requests with different ITLs"，未说明 profile 发生在离线还是在线阶段｜引文依据："By profiling a series of requests with different ITLs, we model the cumulative runtime as a quadratic function of sequence length"｜修复要求：删除「离线」二字，或改为「profile 一批不同输入长度的请求」｜修复：删除「离线」｜复验：已删，改为「先 profile 一批不同输入长度的请求」
- [轻微·来源] index.html C3 条目：定位写「The Pipeline Bubble 与 The Memory Wall 两节」，实际两句引文出自同一小节 "The Challenge: The 'Bubble' and The 'Wall'" 的两个编号要点，博客中不存在名为 "The Pipeline Bubble" 或 "The Memory Wall" 的独立小节｜引文依据：博客结构——"The Challenge: The 'Bubble' and The 'Wall'" 一节内并列 "1. The Pipeline Bubble: …" 与 "2. The Memory Wall: …" 两个要点｜修复要求：C3 定位改为「The Challenge: The 'Bubble' and The 'Wall' 一节的两个要点」｜修复：C3 定位改为 The Challenge: The "Bubble" and The "Wall" 一节的两个要点｜复验：已改
- [轻微·来源] index.html N2 条目：定位写「TeraPipe §4.2」，其中 1.12×/1.04× 确在 §4.2，但切分方案 $120\times4+112\times6+104\times8+64$ 出自 Table 2（Appendix C）与 Table 3，§4.2 正文没有该方案数字｜引文依据：Table 2 原始条目 "DP [(1, [120] * 4 + [112] * 6 + [104] * 8 + [64])] * 2 1.481±0.002 7.6225"｜修复要求：N2 定位改为「TeraPipe §4.2 与 Table 2/3」｜修复：N2 定位改为「TeraPipe §4.2 与 Table 2/3」｜复验：已改
- [轻微·来源] index.html F1 条目：「GPipe（Huang et al., 2019）」年份错误。GPipe 论文为 arXiv:1811.06965（2018 年 11 月），被转引的模型并行页也记「Huang et al., 2018」｜引文依据：模型并行页 blockquote.meta："GPipe（Huang et al., 2018, arXiv:1811.06965，流水线并行与气泡）"｜修复要求：F1 中「2019」改为「2018」｜修复：F1 年份 2019 改为 2018｜复验：已改，与模型并行页及 arXiv:1811.06965 一致
- [轻微·图示] index.html 图 2（2.3 节 SVG）：stage 1 行存在一个坐标错误的虚线空闲框（x=90 width=252，对应时段 [0,56]，被三个实线计算块 [0,78] 完全覆盖、渲染时不可见）；而 stage 1 真实的排空空闲 [78,134]（x=441 至 693）没有画出。figcaption 说「虚线框为空闲（填充与排空段）」，stage 1 的排空空闲缺失与调度表（stage 1 于 78 完工、134 总完工）不一致｜引文依据：SVG 第 932 行 `<rect class="dg-line" x="90" y="50" width="252" …/>` 与调度表 stage 1 行「[0,28]/[28,55]/[55,78]、完工 78」｜修复要求：删除该不可见虚线框，或把它的坐标改为 x="441" width="252"（对应 [78,134]）使 stage 1 的排空空闲与 caption、调度表一致｜修复：图 2 stage 1 的坐标错误虚线框改为 x=441 width=252（对应排空空闲 [78,134]）｜复验：已改，stage 1 排空空闲与 figcaption、调度表一致
- [轻微·图示] index.html 图 2、图 3：`<text>` 内用 ASCII 写法「s1/s2/s3」「c1–c4」表示正文与表格中以 LaTeX 书写的 $s_1$、$c_1$（带下标的片段/chunk 标识，具有数学含义），违反格式规范「SVG 的 `<text>` 只用于不含数学含义的纯文字，不得用 ASCII 近似写法代替公式」｜引文依据：图 2 `<text …>s1 · 28</text>` 等 9 处；图 3 `<text …>c1 · 1.0</text>` 等 10 处；正文调度表用「片段 1」、3.1 表格用 $c_1$｜修复要求：统一改为不含下标数学含义的纯文字（如「片段1 · 28」「块1 · 1.0」，与正文调度表「片段 1」写法一致），或改用 `<foreignObject>` 承载 $s_1$/$c_1$｜修复：图 2/图 3 的 SVG text 标签改为不含下标数学含义的纯文字（片段 1 · 28 / chunk 1 · 1.0 等）｜复验：已改 22 处；重跑渲染探针 katex=274、overlaps=0
- [轻微·技术] index.html 5.2 节「短上下文：CPP 无显著开销也无收益<sup>[C7]</sup>」及本章问题解答 2、overview.html「短上下文下无收益也无显著开销」：「无显著开销」有 Mooncake 原文支持，「也无收益」是页面的机制推断（短 prompt 切不出多个 chunk），原文未作此陈述，不宜直接挂在 C7 下｜引文依据：Mooncake §5.1 原文仅 "bringing no significant overhead for short context prefill"，无「无收益」表述｜修复要求：改为「短上下文：CPP 无显著开销（Mooncake 自述）；短序列切不出多个可并行的 chunk，自然也谈不上收益」或删去「也无收益」，两页同步修改｜修复：「无收益」改为推断分离表述（无显著额外开销有 C7 支持；收益场景是长上下文），index 两处与 overview 同步｜复验：已改：5.2、Ch5 解答、overview 三处
- [轻微·可读性] index.html 核心问题解答 3 与 C4：「PP rank」首次出现未解释（rank 即 stage 序号，越高越靠后）。读者需从上下文自行推断「更高的 PP rank」与「更后的 stage」的对应关系｜引文依据：不适用（SGLang 博客原文用 "higher PP ranks"，页面直接沿用而未加注）｜修复要求：在核心问题解答 3 或 3.1 首次出现处加一括号注（如「更高的 PP rank（更靠后的 stage）」）｜修复：PP rank 首现处加括号注（更靠后的 stage）｜复验：已在核心问题解答 3 首现处加注
- [轻微·可读性] index.html 1.1 节「图 1 中」与 3.2 节「图 3 下半部分」：正文使用「图 1」「图 3」编号引用，但三个 figure 的 figcaption 均未标注「图 1/图 2/图 3」编号，读者需自行按顺序数图确认指向｜引文依据：不适用（figcaption 文本依次为「实线块为计算时段……」「前长后短 $7{+}3{+}2$ 的流水线调度……」「上：固定大小的 chunk……」，均无编号）｜修复要求：在三个 figcaption 开头加入「图 1：」「图 2：」「图 3：」编号｜修复：三个 figcaption 开头加「图 1：」「图 2：」「图 3：」｜复验：已加，正文「图 1」「图 3」引用可对应
- [轻微·格式] index.html 引言「TP×PP 组合」、3.2 节「PP4×TP8 的 prefill 吞吐」、N4 条目「PP4×TP8 prefill 吞吐」共 3 处：Unicode 乘号「×」直接出现在公式定界符外的正文中（并行配置记法）。格式规范要求正文中的数学运算符包在 `$...$` 中由 KaTeX 渲染｜引文依据：不适用（validate.py 当前不检测该字符，但 style-guide 第 11 节字面要求覆盖数学运算符）｜修复要求：3 处统一改为 LaTeX（如「PP4$\times$TP8」「TP$\times$PP」）或给出保留现状的明确接受理由｜修复：3 处 Unicode ×改为 LaTeX（TP$\times$PP、PP4$\times$TP8 两处）｜复验：已改 3 处；grep 复查正文无残留的定界符外 ×

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 10
- 处置：修复。两条重要问题（82.8% 数字归属、线性变体未限定三片）修复后需复验：前者重新对照 SGLang 博客原文确认归属表述，后者按修复后的表述重新执行复算脚本确认无歧义；轻微问题逐条修复并在本记录填写修复与复验结果。12 条问题已全部修复并复验（两条重要问题分别经博客原文比对与独立复算脚本确认）；validate.py 与渲染探针（katex=274、overlaps=0）重跑通过。进入第 3 轮全量审查。
