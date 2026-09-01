# 逐层混合精度量化（MIX-STQ1_0）审查记录（第 2 轮）

- 页面版本：index.html 1da229fc09e308d0447a23a0f0c7357ef4b06718（overview.html bc74fcd8497bbbfa52252bbc0cdf6feff608fc06）
- 审查时间：2026-09-01
- 审查者：独立审查者（未参与写作与第 1 轮审查修复）
- 已完整阅读章节：引言、核心问题（4 条含解答折叠块）、常见误解（5 条）、1. 同样的预算，比特落在哪一层（含本章问题 2 条）、2. 敏感度怎么量——校准数据与 imatrix（含本章问题 2 条）、3. MIX-STQ1_0 配方——Hy4 的比特分配表（含本章问题 3 条）、4. 代价与收益——体积、精度与边界（含本章问题 2 条）、来源与范围说明（论断与来源（C）／公式与来源（F）／外部数字与实验条件（N）／构造示例／简化条件及其限制）、overview.html 全文

核对来源（均打开并定位）：

- HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡：文件表、英文版第 1 节 What these are、第 3 节 STQ1_0 and the mixed-precision strategy、第 4 节 Re-quantizing from bf16
- 腾讯混元官方文章「Hy4 preview 轻量版」（2026-09-01）：快科技（mydrivers/新浪转载）、IT之家（头条/凤凰转载）、网易、搜狐转载全文
- llama.cpp tools/imatrix/README
- llama.cpp PR #22836

独立复算（本页无代码块，公式与示例复算）：

- F1：(29×1.3125+48×2.0625)/77 = (38.0625+99.0)/77 = 137.0625/77 = 1.7794…≈1.78，页面公式与结果正确
- 构造示例：(1.31+2.06)/2 = 1.685（页面写 1.69，见问题 5）；20+30=50、100+25=125、125/50=2.5，均正确
- 保留率：83.2/83.7=99.40%、81.3/82.9=98.07%、81.1/81.3=99.75%、72.5/73.5=98.64%，范围 98.1%–99.8% 与页面一致
- 体积：219.83−213.66=6.17 GiB；213.66/435.20=0.491（「约为一半」成立）

机械项检查：

- 链接 ../sherry-ternary-quant/index.html 存在 ✓；overview.html 与 index.html 相互链接 ✓
- 本地资源 ../../libs/ 下 katex.min.css、katex.min.js、auto-render.min.js、prism 系列均存在 ✓
- 数学符号均为 LaTeX（$...$/$$...$$），未发现 Unicode 数学字符裸写（h1/h2/h3/summary/正文/列表/表格/callout/dojo:summary 均扫描）；图示为 HTML 结构（dg-flow/dg-stack），无 ASCII 框线图，无 SVG 数学标签 ✓
- 问题块：页面级「核心问题」4 条（规范 3–5 条内）与每章「本章问题」均有「解答：」折叠块，答案独立可读并指明章节 ✓；summary 前缀、h2/h3 编号与固定命名、前置 section 顺序、callout 颜色（仅 1 个 yellow）均合规 ✓
- dojo:topics「推理系统」是否在 AGENTS.md 固定大类词表内本轮无法核对（审查输入限制，未读 AGENTS.md），留待 validate.py 把关
- 本页无 Python 代码块，无可执行代码核对项

## 问题

- [重要·技术] index.html「来源与范围说明」F1 段：「与官方图注的 1.78 互验」——四处允许来源中均定位不到官方给出的 1.78 bpw 或相应图注｜引文依据：模型卡第 3 节原文仅有「STQ1_0 (29 layers) / IQ2_XXS (48 layers)」「1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48」，无 1.78；文件表 bpw 列为 4.86/2.44/2.38；快科技等转载与 PR #22836 亦无 1.78｜修复要求：删除「与官方图注的 1.78 互验」表述，F1 改为「由模型卡的层数（29/48）与格式 bpw（1.3125/2.0625）直接计算」｜修复：F1 段改为「由模型卡的层数（29/48）与两种格式的 bpw（1.3125/2.0625，分别见 PR #22836 与模型卡）直接计算」，删除互验表述；evidence.md F1 同步。｜复验：已复验。
- [重要·技术] index.html 第 2 章正文「统计每个权重位置在真实数据分布下的行为」「作为逐权重的重要性分数」、图示节点「imatrix：逐权重重要性分数」、第 2 章本章问题 1/2 答案与核心问题 2 答案中的「对每个权重位置统计」「逐权重」：imatrix 的统计粒度被描述为逐权重，来源不支持｜引文依据：imatrix README 原文「Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.」与统计小节「Per tensor — Σ(Act²): sum of all squared activations (the importance scores)」，即按张量收集激活平方和统计，无任何 per-weight 表述；llama.cpp imatrix 的粒度是激活通道（对应权重矩阵行）而非逐个权重｜修复要求：将「逐权重」「每个权重位置」改为 README 支持的表述（如「按张量收集校准数据前向时激活平方和等统计，作为重要性分数供量化加权误差使用」）；如需保留更细粒度表述，须降级为明确标注的推断并说明依据｜修复：第 2 章正文、图示节点、核心问题 2 答案与本章问题 1/2 答案中的「逐权重」「每个权重位置」全部改为 README 支持的表述（按张量收集激活平方和等统计作为重要性分数）；并在答案中补一句「重要性分数在 STQ1_0 编码器的公式里按权重下标使用（见 Sherry 页）」以衔接模型卡的 w[j] 记号。｜复验：已复验，全文无「逐权重」表述。
- [重要·技术] index.html 第 3 章正文「官方称路由专家部分比 UD-IQ1_M 少占 5 个多 GiB[C10]」、第 4 章「官方称这部分比 UD-IQ1_M 少占 5 个多 GiB」、核心问题 4 答案、第 4 章本章问题 1 答案、overview.html「关键结论与边界」：官方原文口径为整体方案对比且单位为 GB，页面将其归因到「路由专家部分」并改写单位为 GiB，且来源未说明两产物差异全部来自路由专家张量｜引文依据：快科技转载官方文章原文「在不提升平均比特开销前提下降低量化误差,相比UD-IQ1_M方案还可以节省5GB以上存储空间」——主语为方案整体（整模 219.83−213.66=6.17 GiB），非「路由专家部分」；模型卡亦无按张量族拆分的体积对比｜修复要求：改为官方原口径（如「官方称 MIX-STQ1_0 方案整体比 UD-IQ1_M 节省 5GB 以上存储空间，整模差 219.83−213.66=6.17 GiB」）；如需将差异归因到路由专家档位，须明确标注为页面推断而非「官方称」｜修复：五处（第 3 章正文、第 4 章正文、核心问题 4 答案、本章问题 1 答案、overview）统一改为官方原口径「官方称该方案比 UD-IQ1_M 节省 5 GB 以上存储空间」，第 4 章处补「与整模文件差 219.83−213.66=6.17 GiB 自洽」，来源章节 N 段同步；evidence.md C10 同步重写。｜复验：已复验。
- [重要·技术] index.html「来源与范围说明」C13 段（「C2、C10、C11、C13：腾讯混元官方技术文章……正文（快科技等转载一致）」）与 N3 段：四项评测数字与保留率被定位到「官方文章正文（快科技等转载一致）」，但允许核对的各篇转载正文中定位不到 MRCR、IFBench 与保留率｜引文依据：快科技/IT之家/凤凰/新浪转载正文仅含「MCP Atlas得分从83.7微降至83.2,SWE-Bench multi从82.9降至81.3」及定性结论（长文理解持平、数学小幅回落、优于 UD-IQ1_M）；MRCR 81.3→81.1、IFBench 73.5→72.5、保留率 98.1%–99.8% 与「单次运行、无误差线、无第三方复现」出自官方评测对比表的第三方英文转述（traictory 转述原文：「MCP Atlas (agentic tool use): 83.7 → 83.2 SWE-Bench Multilingual (coding): 82.9 → 81.3 MRCR (multi-turn chat): 81.3 → 81.1 IFBench (instruction following): 73.5 → 72.5. Retention runs from 98.1% to 99.8%」「These are unaudited vendor measurements: single runs, no error bars, no third-party reproduction」），数字与页面一致，但与页面标注的定位位置不符｜修复要求：更正 C13/N3 的定位描述：MCPAtlas 与 SWE-Bench 两项及定性结论出自官方文章中文转载；MRCR、IFBench、保留率与「单次运行、无误差线」等性质标注出自官方评测表的第三方转述，逐项写明｜修复：C13 段改为「四项评测与保留率数字经第三方英文转述（traictory，2026-08-31）核对，其中 MCPAtlas 与 SWE-Bench 两项亦见于中文转载；评测性质标注（单次运行、无误差线、无第三方复现）出自该转述的说明」；N3 段同步。｜复验：已复验。
- [轻微·技术] index.html 第 1 章正文「预算为平均 1.69 bpw，恰好等于两档的平均值」：两档平均值为 (1.31+2.06)/2=1.685，不等于 1.69，「恰好等于」不成立｜引文依据：不适用（复算）｜修复要求：改为「预算为平均 1.685 bpw（约 1.69）」或直接使用 1.685，并同步核对本章问题 1 答案与核心问题 1 答案中「1.69」处表述是否仍自洽｜修复：改为「预算为平均 1.69 bpw（两档的平均值为 1.685，取整后即预算）」；核心问题 1 与本章问题 1 答案中 1.69 表述与之自洽（均以 1.69 为预算值）。｜复验：已复验。
- [轻微·技术] index.html 第 2 章本章问题 2 答案：公式 $d=\sum w_i x_i q_i/\sum w_i q_i^2$ 中 $w_i$、$x_i$、$q_i$ 未定义（style-guide 第 11 节要求公式后逐项定义符号），且模型卡原文记号为 sel（「d = sum(w*sel*x) / sum(w*sel^2)」），改写差异未说明｜引文依据：模型卡第 3 节原文「Weighted least-squares scale, d = sum(w*sel*x) / sum(w*sel^2) instead of d = amax」｜修复要求：在公式后逐项定义符号并说明 sel 与 $q_i$ 的对应关系，或删去公式仅保留文字结论并指向 Sherry 页「量化决策」一章｜修复：删去该答案中的公式，改为文字结论并指向 Sherry 页「量化决策」一章（公式与符号在那里定义）。｜复验：已复验。
- [轻微·技术] index.html 第 3 章配方表「MLA（Hy4 所用注意力）」、dg-stack 图示与核心问题 3 答案「iHC（Hy4 架构组件）」：模型卡未说明 MLA 是 Hy4 所用注意力机制、iHC 是何种组件，仅以张量族名称列出｜引文依据：模型卡第 3 节表格原文仅「MLA q_b/k_b/v_b/kv_a_mqa — Q8_0 — HY4's split names miss llama.cpp's substring match」「iHC *_fn, router, norms, sink — F32 — mirrors the reference's _keep_in_fp32_modules」，无 MLA/iHC 的架构定位说明｜修复要求：改为中性表述（「MLA 分量 q_b/k_b/v_b/kv_a_mqa」「iHC 相关张量（*_fn、router、norms、sink）」）或明确标注为推断｜修复：配方表与 dg-stack、核心问题 3 答案改为中性表述「MLA 分量 q_b/k_b/v_b/kv_a_mqa」「iHC 的 *_fn、router、norms、sink」，并在配方表前注明「表中张量族名为模型卡原文命名」。｜复验：已复验。
- [轻微·技术] index.html 第 1 章表格「重要性混合基线（UD-IQ1_M）」、第 4 章表格与 overview.html「社区重要性混合基线」「社区基线 UD-IQ1_M」：「社区」定性无允许来源支持｜引文依据：模型卡原文「Using the UD-IQ1_M quantization strategy」、官方文章「相比UD-IQ1_M方案」，均未称其为社区方案｜修复要求：删除「社区」二字或明确标注为推断｜修复：删除「社区」二字（第 1 章表格「重要性混合基线」、第 4 章表格、overview「对照方案 UD-IQ1_M」）。｜复验：已复验。
- [轻微·技术] index.html 第 3 章正文「它们合计不到 1% 的参数」与本章问题 3 答案「这些张量合计不足 1% 参数」：该合计占比无来源数值依据｜引文依据：模型卡仅给出 DSA 索引器「105 tensors, 0.21 GiB total」，未给出 DSA+iHC+router+norms+sink+output 的合计参数占比｜修复要求：改为定性表述（「体量微小」），或给出估算依据并明确标注为页面估算（估算口径须可复算）｜修复：两处改为可核对的表述——正文「DSA 索引器（105 个张量共 0.21 GiB）、router、归一化层……用极小的体积代价买稳定性」；答案「其中 DSA 索引器 105 个张量共 0.21 GiB，其余均为小张量，高精度的体积代价可以忽略」。｜复验：已复验。
- [轻微·技术] index.html「来源与范围说明」多处来源定位不精确：C12 中「STQ1_0 强制需要 imatrix」出自模型卡第 4 节「Re-quantizing from bf16」而非第 3 节；C14（三个量化产物）出自模型卡开头与第 1 节「What these are」及文件表而非第 3 节；N 节「IQ1_M 的 1.75 bpw 取自官方文章」实际出自模型卡英文版第 1 节，各中文转载无 1.75｜引文依据：模型卡第 4 节原文「An imatrix is mandatory for STQ1_0 — its encoder uses it for the scale solve and zero placement」；第 1 节原文「The routed-expert gate/up projections run at 1.75 bpw (IQ1_M) and 2.0625 bpw (IQ2_XXS)」；开头「Three GGUF builds of Hy4-Preview」｜修复要求：逐条更正来源章节的节数描述，与实际出处一致｜修复：C 段补「C12 的『imatrix 强制』见模型卡『4. Building a runtime』节」；N 段改为「IQ2_XXS 的 2.0625 bpw 与 IQ1_M 的 1.75 bpw 均取自模型卡」。｜复验：已复验。
- [轻微·格式] index.html 第 1 章「构造示例。 用一个两层模型……」：标记词「构造示例」不在 style-guide 第 4 节规定的三种标记（「计算示例」「代码示例」「构造数据」）之内｜引文依据：不适用｜修复要求：改用「构造数据。」作为标记词（「构造示例」作为来源章节固定 h3 小节名保留不变）｜修复：标记词改为「构造数据。」。｜复验：已复验。
- [轻微·格式] index.html 正文与来源章节：论断编号从 C2 开始（无 C1），数字编号跳过 N2（N1、N3、N4），跳号未说明｜引文依据：不适用｜修复要求：重新连续编号（C、N 各自从实际首号连续），或在来源章节说明跳号原因｜修复：来源章节（C）小节开头补说明「编号沿用内部证据清单：C1 为术语消歧条目（不对应正文引用）、N2 已并入 C10、N4 与 N5 未在正文使用，故编号不连续」。｜复验：已复验。
- [轻微·可读性] index.html 常见误解第 5 条「两份 GGUF 都不能跑在原版 llama.cpp 上」：GGUF 首次使用未给最小含义（check.md 2.1 术语首次使用需解释）｜引文依据：不适用｜修复要求：首次出现处补最小含义，如「GGUF（llama.cpp 生态的模型文件格式）」｜修复：常见误解第 5 条改为「GGUF（llama.cpp 的模型权重文件格式）产物依赖运行时支持：……」。｜复验：已复验。

## 已核对无问题的关键论断（抽样记录引文依据）

- 体积与 bpw 三组数字：模型卡文件表「435.20 GiB 4.86」「219.83 GiB 2.44」「213.66 GiB 2.38」✓
- 97.7%：模型卡第 3 节「The three routed-expert families are 97.7% of all parameters」✓
- 29/48 分层与「选层由 imatrix 推导」：「1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48」「layer choice is imatrix-derived」✓
- down 高两档理由：「writes straight into the residual stream, so its error is not attenuated by a later gate — deliberately 2 levels higher」✓
- Q5_K 提档条件：「llama.cpp only auto-bumps these when n_expert == 8; HY4 has 256」✓
- MLA 子串匹配：「HY4's split names miss llama.cpp's substring match, so they get no automatic bump」✓
- DSA 索引器：「105 tensors, 0.21 GiB total, gates which 2048 tokens each query sees」✓
- keep-in-fp32 与输出层：「mirrors the reference's _keep_in_fp32_modules」「output (lm_head) F32 via --leave-output-tensor」✓
- 原版 llama.cpp 不可运行：「Neither file runs on stock llama.cpp. The hyv4 architecture is not upstream.」补丁分工「0001 both GGUFs need this / 0002 STQ1_0 only」✓
- 1.3125 bpw 来源：PR #22836「yielding 1.3125 bits per weight … 42 B / 256 = 1.3125 bpw」✓
- imatrix 机制（张量级）：README「Compute an importance matrix for a model and given text dataset」「Σ(Act²): sum of all squared activations (the importance scores)」✓
- STQ1_0 强制 imatrix：模型卡第 4 节「An imatrix is mandatory for STQ1_0」✓
- 同预算更低误差（C11）：官方文章「在不提升平均比特开销前提下降低量化误差」✓
- 1.5 TB 与 214GB：官方文章「原版BF16权重体积高达1.5TB……压缩至约214GB」✓
- 领先 UD-IQ1_M、长文理解持平、数学小幅回落：「整体表现优于UD-IQ1_M量化版本」「长文理解……基本持平,数学能力仅有小幅回落」✓
- 评测性质与「未覆盖知识套件/无误差线」「本地部署场景」：第三方转述「single runs, no error bars, no third-party reproduction」「No knowledge or math suites (MMLU, GSM8K)」「exactly the workloads a 200GB local build would be bought for」✓（但定位描述须按问题 4 更正）

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 9
- 处置：修复。4 条重要问题（来源定位与口径、imatrix 粒度描述、归因与单位）须全部关闭后进入第 3 轮审查；轻微问题逐条修复，确有接受理由的须在修复栏写明。核心结论（预算—分配—误差框架、29/48 配方、体积与精度数字、策略边界）经核对与复算均成立，无阻断项。
