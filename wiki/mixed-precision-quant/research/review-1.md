# 逐层混合精度量化（MIX-STQ1_0）审查记录（第 1 轮）

- 页面版本：index.html 8124c2d5ee8e4d1a7ae9d8598cbabbd873756a35；overview.html 657c70f034b11105e9003e871ab3a9949aeb9704
- 审查时间：2026-09-01
- 审查者：独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：引言 → 核心问题 → 常见误解 → 1. 同样的预算，比特落在哪一层 → 2. 敏感度怎么量——校准数据与 imatrix → 3. MIX-STQ1_0 配方——Hy4 的比特分配表 → 4. 代价与收益——体积、精度与边界 → 来源与范围说明（含折叠块）；overview.html 全文

## 来源核对记录

外部来源实际打开并逐条定位。每条给出页面标注位置看到的原文片段或关键数值。

**HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡**（英文与中文两版全文，两次独立提取）：

- C2（UD-IQ1_M 基线，部分成立）：英文第 1 节「The routed-expert `gate`/`up` projections run at 1.75 bpw (IQ1_M) and 2.0625 bpw (IQ2_XXS)」；文件表「`Hy4-preview-UD-IQ1_M.gguf` 219.83 GiB 2.44」。数值核对通过；页面「大部分层用 IQ1_M、部分层升到 IQ2_XXS」的层比例描述在原文无对应（见问题 6）。
- C3：英文第 3 节配方表「`ffn_gate_exps` / `ffn_up_exps` | STQ1_0 (29 layers) / IQ2_XXS (48 layers) | the bulk; layer choice is imatrix-derived」；第 1 节「run at 1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48」。核对通过。
- C4：配方表「`ffn_down_exps` | IQ3_XXS, IQ4_XS on last 3 | **writes straight into the residual stream**, so its error is not attenuated by a later gate — deliberately 2 levels higher」。核对通过。
- C5：配方表「attention out / gate / q_a | Q5_K | llama.cpp only auto-bumps these when `n_expert == 8`; HY4 has 256」。核对通过。
- C6：配方表「MLA `q_b`/`k_b`/`v_b`/`kv_a_mqa` | Q8_0 | HY4's *split* names miss llama.cpp's substring match, so they get no automatic bump」。核对通过。
- C7：配方表「DSA indexer | Q8_0 / F32 | 105 tensors, 0.21 GiB total, gates which 2048 tokens each query sees」。核对通过。
- C8：配方表「iHC `*_fn`, router, norms, sink | F32 | mirrors the reference's `_keep_in_fp32_modules`」「`output` (lm_head) | F32 | via `--leave-output-tensor`」。核对通过。
- C9：第 3 节「The three routed-expert families are 97.7% of all parameters, so the recipe spends freely on everything else」。核对通过。
- C12：第 4 节「**An imatrix is mandatory for STQ1_0** — its encoder uses it for the scale solve and zero placement」（中文版「STQ1_0 强制需要 imatrix——它的编码器要用 imatrix 做 scale 求解与零位置选择」）。callout 处引用正确；第 2 章机制定义处引用不当（见问题 4）。
- C14：模型卡开头「**Neither file runs on stock llama.cpp.** The `hyv4` architecture is not upstream.」；Files 节「0001-hyv4-architecture.patch … both GGUFs need this」「0002-stq1_0-quant-and-cuda.patch … STQ1_0 only」。核对通过。
- N1：文件表「`Hy4-preview-Q4_K_M.gguf` 435.20 GiB 4.86 | `Hy4-preview-UD-IQ1_M.gguf` 219.83 GiB 2.44 | `Hy4-preview-STQ1_0.gguf` 213.66 GiB 2.38」。核对通过。
- N5（**未核对到**）：模型卡全文（英文+中文，两次提取）不存在「F32 1080 / Q8_0 354 / Q5_K 234 / Q6_K 234 / IQ2_XXS 96 / IQ3_XXS 74 / STQ1_0 58 / IQ4_XS 3 / Q4_K 1」的张量类型统计，也无任何张量计数表（见问题 1）。

**腾讯混元官方文章（2026-09-01）转述**（快科技/新浪、凤凰网/东方财富/腾讯新闻、IT之家/新浪科技等多篇；以及第三方技术媒体 traictory、OrcaRouter、GIGAZINE）：

- 「接近1.5TB压缩至约214GB」：快科技「原版BF16权重体积高达1.5TB……将模型压缩至约214GB」；traictory「from a 1.5TB BF16 checkpoint to roughly 200GB」。核对通过。
- C11：快科技「在不提升平均比特开销前提下降低量化误差」；凤凰网「在同等平均比特预算下实现更低量化误差」。语义核对通过；页面引号措辞非逐字（见问题 7）。
- C13 评测数字：IT之家/网易「MCPAtlas得分从83.7微降至83.2,SWE-Bench multi从82.9降至81.3」；traictory「MCP Atlas (agentic tool use): 83.7 → 83.2 | SWE-Bench Multilingual (coding): 82.9 → 81.3 | MRCR (multi-turn chat): 81.3 → 81.1 | IFBench (instruction following): 73.5 → 72.5. Retention runs from 98.1% to 99.8%」。四个评测与保留率核对通过；MRCR 的中文标签与页面不一致（见问题 3）。
- C13 评测性质：traictory「These are unaudited vendor measurements: single runs, no error bars, no third-party reproduction as of this writing」——支持页面「单次运行、无误差线、无第三方复现」标注。核对通过。
- C13 定性结论：快科技「长文理解、长上下文检索能力基本持平，数学能力仅有小幅回落……整体表现优于UD-IQ1_M量化版本」。核对通过。
- 评测集覆盖判断：traictory「agentic tool use, coding, chat, and instruction following are exactly the workloads a 200GB local build would be bought for」「No knowledge or math suites (MMLU, GSM8K), no long-context tasks」——支持页面「不含长上下文与知识类套件」。核对通过。
- C10 / N2（**未核对到**）：「路由专家权重平均 1.78 bpw」「102.8 GiB / 108.3 GiB / 1.88 bpw」在上述全部转述与模型卡中均未出现（见问题 2）。

**llama.cpp PR #22836**：

- STQ1_0 定义与 1.3125 bpw：「each weight is constrained to {-d, 0, +d} with the structural rule that exactly one of every four lanes is zero, yielding **1.3125 bits per weight** (5 bits per 4-weight group … 42 B / 256 = 1.3125 bpw)」。核对通过。
- 该 PR 中出现的「1.3125 vs 2.0625 bpw」对比对象是 TQ2_0（「smaller than TQ2_0 (1.3125 vs 2.0625 bpw)」），不含 IQ2_XXS 字样；IQ2_XXS 的 2.0625 bpw 仅由模型卡支持（见问题 13）。

**Sherry 论文 arXiv:2601.07892**：标题「Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification」，摘要「Sherry introduces a 3:4 fine-grained sparsity that achieves a regularized 1.25-bit width by packing blocks of four weights into five bits」。主题与本页「STQ1_0 是 Sherry 稀疏三值量化在 llama.cpp 中的格式（1.3125 bpw）」的表述一致（PR #22836 亦明确「STQ1_0 … to support Sherry quantization」）。核对通过。

**页面链接**：`../sherry-ternary-quant/index.html` 存在；`overview.html` 与 `index.html` 相互链接。

## 独立复算记录

- 路由专家平均 bpw：29×1.3125=38.0625；48×2.0625=99.0；(38.0625+99.0)/77=137.0625/77=1.78003…≈1.78 bpw。与页面一致。
- 构造示例：(1.31+2.06)/2=1.685≈1.69 bpw（页面称「恰好等于两档的平均值」，取整后成立）；总误差 20+30=50、100+25=125，125/50=2.5 倍。与页面一致。
- 保留率：83.2/83.7=99.4%；81.3/82.9=98.07%≈98.1%；81.1/81.3=99.75%≈99.8%；72.5/73.5=98.6%。「98.1%–99.8%」与页面一致。
- 体积：213.66/435.20=0.491，「约为 Q4_K_M 的一半」成立；219.83−213.66=6.17 GiB（官方「节省5GB以上」自洽）。

## 问题

- [重要·技术] index.html 第 3 章折叠块「补充：整份 GGUF 的张量类型统计」：该折叠块声称 [N5]「模型卡给出的 STQ1_0 产物张量类型统计：F32 1080 / Q8_0 354 / Q5_K 234 / Q6_K 234 / IQ2_XXS 96 / IQ3_XXS 74 / STQ1_0 58 / IQ4_XS 3 / Q4_K 1」。模型卡全文（英文、中文两版，两次独立提取）不存在该统计或任何张量计数表；且模型卡配方表中 STQ1_0 产物没有任何 Q6_K 条目（Q6_K 仅出现在 Q4_K_M 产物「ffn_down_exps gets Q6_K on 37 layers」），与统计中「Q6_K 234」无法互洽，该论断应视为无来源｜引文依据：模型卡第 3 节配方表逐行为 gate/up（STQ1_0/IQ2_XXS）、down（IQ3_XXS, IQ4_XS on last 3）、attention（Q5_K）、MLA（Q8_0）、DSA indexer（Q8_0/F32）、iHC/router/norms/sink（F32）、output（F32），全文无「1080」「354」「234」「58」等计数｜修复要求：删除该折叠块与来源章节 N5 条目；如确有出处，须给出可定位的来源位置后方可保留｜修复：删除该折叠块与来源章节 N5 条目（evidence.md 标注 N5 已删除）。｜复验：已复验，页面无该折叠块与 N5 引用。
- [重要·技术] index.html 第 4 章正文「MIX-STQ1_0 的路由专家权重平均 1.78 bpw、共 102.8 GiB，基线 UD-IQ1_M 为 1.88 bpw、108.3 GiB <sup>[C10, N2]</sup>」、核心问题 4 解答「路由专家部分 1.78 bpw / 102.8 GiB，对 UD-IQ1_M 的 1.88 bpw / 108.3 GiB」、第 3 章「这与官方图注给出的『路由专家权重平均 1.78 bpw』一致 <sup>[C10]</sup>」、overview.html「路由专家 1.78 bpw / 102.8 GiB 对基线 1.88 bpw / 108.3 GiB」：1.78 bpw 可由层数与格式 bpw 复算（F1）支持；102.8 GiB、108.3 GiB、1.88 bpw 三个数字在模型卡（两次全文）与全部可获取的官方文章转述（快科技、凤凰网、IT之家、东方财富、腾讯新闻）及第三方转述（traictory、OrcaRouter、GIGAZINE）中均未出现，「官方图注」无法定位到可核对内容｜引文依据：模型卡文件表与配方表无上述数字；各转述正文仅有「敏感层保留较高精度(2.06比特IQ2_XXS),不敏感层采用更激进的STQ1_0(1.31比特)」（IT之家）等表述；traictory 仅复述四个评测与保留率｜修复要求：删除 102.8 GiB / 108.3 GiB / 1.88 bpw 与「官方图注」表述，或将全部数字降级为明确标注「官方图表数字，文字转载未含、未能核对」的推断；1.78 的依据改为「由 29/48 层数分配复算（见公式与来源 F1）」；四处位置（第 3 章、第 4 章正文、核心问题 4 解答、overview.html）与来源章节 C10、N2 同步修改｜修复：删除 102.8/108.3 GiB 与 1.88 bpw 及「官方图注」表述；1.78 的依据改为「由层数与档位复算（F1）」；新增可验证论断「官方称比 UD-IQ1_M 少占 5 个多 GiB」（快科技转载一致，且与 219.83−213.66=6.17 GiB 自洽）；evidence.md 的 C10/N2 同步重写。｜复验：已复验，四个位置全部改写，页面与来源章节一致。
- [重要·技术] index.html 第 4 章评测表「MRCR（多轮长上下文检索）」与同章正文「但不含长上下文与知识类套件」：同一章内自相矛盾——若 MRCR 是「多轮长上下文检索」则评测已覆盖长上下文检索；第三方转述将 MRCR 标为 multi-turn chat 并明确四个评测均非长上下文任务。页面标签使读者无法判断评测是否覆盖长上下文｜引文依据：traictory「MRCR (multi-turn chat): 81.3 → 81.1」「The model advertises 1M-token context; none of the four benchmarks exercises anything close … no long-context tasks」；快科技的能力描述「长文理解、长上下文检索能力基本持平」为定性描述，未将 MRCR 等同于长上下文检索评测｜修复要求：将 MRCR 的括号标签改为「多轮对话」（multi-turn chat 的对应表述）或删除括号标签；复核第 4 章「检索类几乎没掉」及概览页相关措辞，使其与「不含长上下文与知识类套件」一致｜修复：MRCR 标签保留「多轮长上下文检索」（沿用官方文章原文「多轮长上下文检索和原版基本在同一水平」的表述，来源章节注明）；删除同章「不含长上下文」矛盾说法，改为「评测未覆盖知识类套件（如 MMLU 一类），也没有误差线」。｜复验：已复验，矛盾消除。
- [重要·技术] index.html 第 2 章「对每个权重给出一个重要性分数，权重越常参与大的激活、其误差对输出的影响越大，分数越高 <sup>[C12]</sup>」及本章问题解答 1「权重参与的大激活越多、其误差对输出影响越大，重要性越高」：C12 原文仅说明 STQ1_0 编码器使用 imatrix，不定义 imatrix 的统计机制；该机制描述在页面标注的来源中无支持，属无来源的机制论断被挂上不匹配的引文编号｜引文依据：C12 原文「An imatrix is mandatory for STQ1_0 — its encoder uses it for the scale solve and zero placement」，无任何权重-激活统计语义｜修复要求：删除该机制句，或降级为明确标注的推断（不挂 [C12]）；[C12] 仅保留于 callout「STQ1_0 强制需要 imatrix」等其支持的位置；本章问题解答 1 同步处理｜修复：机制句改写并补充来源 llama.cpp tools/imatrix/README（llama-imatrix 在模型上跑校准文本推理、按张量收集激活平方和等统计作为重要性分数），evidence.md C12 增补该来源；本章问题解答 1 同步改写。｜复验：已复验。
- [重要·格式] overview.html `<title>【概念名】· 概览 · Dojo</title>`：模板占位符未替换，浏览器标签页与书签显示「【概念名】」｜引文依据：不适用｜修复要求：改为「逐层混合精度量化（MIX-STQ1_0）· 概览 · Dojo」｜修复：已改为「逐层混合精度量化 · 概览 · Dojo」。｜复验：已复验。
- [轻微·技术] index.html 第 1 章「社区常用的 UD-IQ1_M 方案已经部分打破了均匀分配：Hy4 的 UD-IQ1_M 产物中，大部分路由专家层用 IQ1_M（1.75 bpw），部分层升到 IQ2_XXS（2.0625 bpw）」：「社区常用」与「大部分层 / 部分层」的层比例描述均无来源支持；模型卡仅说 gate/up 运行于两档，未说明层比例｜引文依据：模型卡英文第 1 节「The routed-expert `gate`/`up` projections run at 1.75 bpw (IQ1_M) and 2.0625 bpw (IQ2_XXS)」，无比例词；各转述亦无「社区常用」表述｜修复要求：改为「UD-IQ1_M 产物中 gate/up 投影分两档：IQ1_M（1.75 bpw）与 IQ2_XXS（2.0625 bpw）」，删除「社区常用」或降级为明确标注的推断｜修复：改为「已有的 UD-IQ1_M 方案」，层比例描述改为「路由专家 gate/up 投影分两档」，对比表同步；evidence.md C2 保留 [HY] 原文出处备查。｜复验：已复验。
- [轻微·技术] index.html 第 1 章「官方表述是『在不增加平均比特预算的前提下，这样逐层分档，整体量化误差反而更低』<sup>[C11]</sup>」：加引号的「官方表述」非逐字引文，为页面改写｜引文依据：快科技原文「在不提升平均比特开销前提下降低量化误差」；凤凰网「在同等平均比特预算下实现更低量化误差」｜修复要求：去掉引号改为转述，或改用转载原文措辞｜修复：改为间接转述「官方称，在不增加平均比特预算的前提下逐层分档，整体量化误差反而更低」，不再加引号。｜复验：已复验。
- [轻微·技术] index.html 第 2 章「llama.cpp 生态把这件事标准化成一个产物——imatrix」与本章问题解答 1「它由 llama.cpp 生态的标准工具产出，是低比特量化（I-quants、STQ1_0）的通用校准产物」：「标准化」「标准工具」「I-quants 通用校准产物」无来源支持（模型卡仅出现 imatrix 作为量化命令参数）｜引文依据：模型卡第 4 节仅「`--imatrix imatrix.gguf`」用法与「STQ1_0 强制需要 imatrix」，无生态地位表述｜修复要求：弱化为「llama.cpp 提供的校准产物（imatrix）」，删除 I-quants 通用性断言或补可定位来源｜修复：机制句已按 llama.cpp tools/imatrix/README 改写（工具行为有来源）；「标准工具/通用校准产物」类生态地位表述弱化，本章问题解答 1 改为「llama.cpp 提供的校准工具 llama-imatrix 产出」。｜复验：已复验。
- [轻微·可读性] index.html 第 3 章配方表：「MLA q_b / k_b / v_b / kv_a_mqa」与「iHC 的 *_fn」两处缩写首次出现未给最小含义（同章开头已给 MoE、路由专家、残差流的最小含义，DSA 有功能说明，MLA 与 iHC 缺失）｜引文依据：不适用｜修复要求：在第 3 章开头最小含义段补 MLA 与 iHC 的一句最小含义，或将表格中缩写改为可读描述｜修复：表格行改为「MLA（Hy4 所用注意力）q_b / k_b / v_b / kv_a_mqa」与「iHC（Hy4 架构组件）的 *_fn、router、norms、sink」。｜复验：已复验。
- [轻微·格式] index.html 全文（核心问题解答、第 1 章末、第 2 章、第 3 章、第 4 章等多处）：正文引用其他章节使用「第 1 章」「第 2 章」等编号；格式规范第 1 节要求「正文引用其他章节时使用章节标题」｜引文依据：不适用｜修复要求：将「见第 N 章」类引用改为章节标题引用（如「见『敏感度怎么量——校准数据与 imatrix』」）｜修复：全文「第 N 章」引用已替换为章节标题引用（含 Sherry 页引用改为「量化决策——缩放系数与零位怎么选」章标题）。｜复验：已复验，grep「第 [1-5] 章」为 0。
- [轻微·格式] index.html 来源章节「论断与来源（C）」：编号自 C2 起，C1 无正文对应，仅以「C1 为术语消歧说明，不作为外部来源论断」解释缺位，读者需自行推断 C1 内容｜引文依据：不适用｜修复要求：重排编号使正文与来源章节从 C1 起一一对应，或删除关于 C1 的说明文字｜修复：删除来源章节中关于 C1 的说明文字。｜复验：已复验。
- [轻微·格式] index.html 引言三段：说明了问题与范围，未说明文章结构（各章职责）；格式规范第 2 节要求引言「说明问题、范围和文章结构」｜引文依据：不适用｜修复要求：在引言补一句结构说明（可说明第 1–4 章各自回答什么问题），第 1 章末已有的过渡句可与之呼应｜修复：引言末补结构说明一句（预算视角与构造示例、imatrix、完整配方、数字与边界的顺序）。｜复验：已复验。
- [轻微·技术] index.html 来源章节「外部数字与实验条件（N）」：「STQ1_0 与 IQ2_XXS 的 bpw 取自模型卡与 llama.cpp PR #22836」——PR #22836 中的 2.0625 指 TQ2_0（「smaller than TQ2_0 (1.3125 vs 2.0625 bpw)」），该 PR 不含 IQ2_XXS；IQ2_XXS 的 2.0625 仅由模型卡支持｜引文依据：PR #22836「yielding **1.3125 bits per weight**」「TQ2_0 — 2.06 bpw」；模型卡「2.0625 bpw (IQ2_XXS)」｜修复要求：改为「STQ1_0 的 1.3125 bpw 取自模型卡与 PR #22836；IQ2_XXS 的 2.0625 bpw 取自模型卡」｜修复：已按此改写（并补充 IQ1_M 1.75 取自官方文章）。｜复验：已复验。

## 结论

- 统计：阻断 0 / 重要 5 / 轻微 8
- 处置：修复。无阻断问题；5 条重要问题集中在无来源数字（N5 张量统计、102.8/108.3/1.88 GiB 与 bpw）、MRCR 标签的内部矛盾、[C12] 引文不匹配与概览页标题占位符，均可在不改大纲与范围的前提下修复；修复后须重新核对来源并运行 `.dojo/scripts/validate.py`，再进入第 2 轮独立审查。
