# 逐层混合精度量化（MIX-STQ1_0）审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 `608a3f1bcf496375f8075a3adc20b8254e44f3dd`（overview.html `04c204b56c0c6f0a77ec983ab32e0a0b35838f23`）
- 审查时间：2026-09-01 17:43 CST
- 审查者：独立审查者（未参与写作，未参与第 1、2 轮审查与修复）
- 已完整阅读章节（按顺序）：head/meta 与 dojo 元信息、blockquote.meta、引言（3 段）、核心问题（4 条含解答折叠块）、常见误解（5 条）、1. 同样的预算，比特落在哪一层（含构造数据表、三方案对比表、本章问题 2 条）、2. 敏感度怎么量——校准数据与 imatrix（含流程图、yellow callout、本章问题 2 条）、3. MIX-STQ1_0 配方——Hy4 的比特分配表（含配方表、公式 F1 及符号表、参数占比堆叠图、本章问题 3 条）、4. 代价与收益——体积、精度与边界（含体积表、精度表、边界清单、本章问题 2 条）、来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、简化条件及其限制）；overview.html 全文（问题背景、核心机制、关键结论与边界）。

## 来源核对记录（逐条引文依据）

按 check.md §2.2 要求，每条来源论断打开来源定位核对。核对所用来源：HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（英文节「3. STQ1_0 and the mixed-precision strategy」「4. Building a runtime」与文件表）、腾讯混元官方文章 2026-09-01（快科技/新浪/IT之家/网易等多家转载全文）、llama.cpp `tools/imatrix/README.md`、llama.cpp PR #22836，以及页面 C13 自行标注的第三方英文转述（traictory，2026-08-31，在检索转载全文时定位到原文）。

| 页面论断 | 定位来源与原文片段/关键数值 | 结论 |
|---|---|---|
| C2 UD-IQ1_M 档位分布与体积 | 模型卡 §1："The routed-expert `gate`/`up` projections run at 1.75 bpw (IQ1_M) and 2.0625 bpw (IQ2_XXS)"；文件表 "`Hy4-preview-UD-IQ1_M.gguf` 219.83 GiB 2.44" | 一致 |
| C3 gate/up 分 29/48 层、选层由 imatrix 推导 | 模型卡 §3 配方表："STQ1_0 (29 layers) / IQ2_XXS (48 layers) — the bulk; layer choice is imatrix-derived" | 一致 |
| C4 down 投影 IQ3_XXS、末 3 层 IQ4_XS、直写残差流高两档 | 模型卡 §3："`ffn_down_exps` IQ3_XXS, IQ4_XS on last 3 — **writes straight into the residual stream**, so its error is not attenuated by a later gate — deliberately 2 levels higher" | 一致（"2 levels higher" 未指明参照档，见问题 4） |
| C5 注意力 Q5_K、8 专家自动提档 | 模型卡 §3："llama.cpp only auto-bumps these when `n_expert == 8`; HY4 has 256" | 一致 |
| C6 MLA 分量 Q8_0、拆分命名错过子串匹配 | 模型卡 §3："HY4's *split* names miss llama.cpp's substring match, so they get no automatic bump" | 一致 |
| C7 DSA 索引器 Q8_0/F32、105 张量 0.21 GiB、2048 token | 模型卡 §3："105 tensors, 0.21 GiB total, gates which 2048 tokens each query sees" | 一致 |
| C8 iHC/router/norms/sink/output 保持 F32 | 模型卡 §3："mirrors the reference's `_keep_in_fp32_modules`"、"`output` (lm_head) F32 via `--leave-output-tensor`" | 一致 |
| C9 路由专家三族占参数 97.7% | 模型卡 §3："The three routed-expert families are 97.7% of all parameters" | 一致 |
| C10 比 UD-IQ1_M 节省 5 GB 以上 | 快科技转载："相比UD-IQ1_M方案还可以节省5GB以上存储空间"；与文件表 219.83−213.66=6.17 GiB（=6.62 GB）自洽 | 一致 |
| C11 同预算下逐层分档误差更低 | 快科技转载："在不提升平均比特开销前提下降低量化误差" | 一致 |
| C12 STQ1_0 强制需要 imatrix | 模型卡 §4（Re-quantizing from bf16）："**An imatrix is mandatory for STQ1_0** — its encoder uses it for the scale solve and zero placement" | 一致 |
| C13 四项评测、保留率、领先 UD-IQ1_M、长文/数学定性 | traictory："MCP Atlas (agentic tool use): 83.7 → 83.2, SWE-Bench Multilingual (coding): 82.9 → 81.3, MRCR (multi-turn chat): 81.3 → 81.1, IFBench (instruction following): 73.5 → 72.5. Retention runs from 98.1% to 99.8%"、"single runs, no error bars, no third-party reproduction as of this writing"；快科技："整体表现优于UD-IQ1_M量化版本"、"长文理解……基本持平,数学能力仅有小幅回落"；IT之家/网易："MCP Atlas 得分从 83.7 微降至 83.2,SWE-Bench multi 从 82.9 降至 81.3" | 一致（保留率复算 98.07%–99.75%，与 98.1%–99.8% 相符） |
| C14/N1 三产物体积与 bpw | 模型卡文件表：Q4_K_M 435.20 GiB / 4.86；UD-IQ1_M 219.83 GiB / 2.44；STQ1_0 213.66 GiB / 2.38 | 一致 |
| F1 路由专家平均 bpw | 复算：(29×1.3125+48×2.0625)/77 = (38.0625+99.0)/77 = 137.0625/77 = 1.7800…≈ 1.78。1.3125 见 PR #22836："yielding **1.3125 bits per weight** (5 bits per 4-weight group … 42 B / 256 = 1.3125 bpw)"；2.0625 见模型卡 | 可复算，一致 |
| imatrix 产出机制（C12 相关正文） | imatrix README："Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models."；`--show-statistics` 统计："Σ(Act²): sum of all squared activations (the importance scores)" | 一致（按张量收集、激活平方和、供量化加权使用均有依据） |
| 「两份 GGUF 都不能跑在原版 llama.cpp 上」 | 模型卡："**Neither file runs on stock llama.cpp.** The `hyv4` architecture is not upstream."；补丁表："0001-hyv4-architecture.patch … both GGUFs need this / 0002-stq1_0-quant-and-cuda.patch … STQ1_0 only"；构建命令注释 "# skip if only using Q4_K_M" 仅豁免 0002 | 字面有据但指代不明，见问题 1 |
| 「约 1.5 TB」 | 快科技/IT之家转载："将模型权重从接近1.5TB压缩至约214GB" | 数字有据，但页面来源节未映射，见问题 9 |
| MRCR 中文标签 | 官方文章转载："多轮长上下文检索基本在同一水平"；traictory 标注为 "MRCR (multi-turn chat)" | 标签借用已声明，见问题 2（与简化条件冲突） |
| 「评测未覆盖知识类套件、无误差线、单次运行」 | traictory："No knowledge or math suites (MMLU, GSM8K)…no error bars"、"These are unaudited vendor measurements: single runs, no error bars, no third-party reproduction" | 一致 |
| 「正是本地部署会买账的场景」 | traictory："agentic tool use, coding, chat, and instruction following are exactly the workloads a 200GB local build would be bought for" | 一致 |

机械验证：`.dojo/scripts/validate.py wiki/mixed-precision-quant/index.html` 返回 `validation ok`；`../sherry-ternary-quant/index.html` 存在；本地资源 `../../libs/`（katex/auto-render/prism）存在；overview.html 与 index.html 相互链接；正则扫描无 Unicode 数学字符（≈×±→∈⊙∂√≥≤）出现在公式定界符之外；本页无代码块，「代码」审查项按不适用处理。

数值复算：F1=1.78 ✓；构造示例正向 20+30=50、反向 100+25=125、倍数 2.5 ✓；(1.31+2.06)/2=1.685 与预算 1.69（取整）自洽 ✓；219.83−213.66=6.17 GiB=6.62 GB>5 GB ✓；保留率复算 MCP 99.4%、SWE 98.1%、MRCR 99.8%、IF 98.6%，区间 98.1%–99.8% ✓；213.66≈435.20/2（"约为一半"）✓。

## 问题

- [重要·技术] index.html L800（常见误解第 5 条）、L1018（第 4 章边界清单）、L1033（本章问题第 2 条解答）：「两份 GGUF 都不能跑在原版 llama.cpp 上」指代不明，且遗漏 Q4_K_M 同样不能在原版运行的事实——页面共讨论三份产物（第 4 章体积表三行），「两份」未指明是哪两份；按模型卡构建说明，hyv4 架构补丁 0001 为所有产物所需（注释「skip if only using Q4_K_M」仅豁免 0002），Q4_K_M 亦不能跑在原版 llama.cpp 上，当前表述可能使读者推断 Q4_K_M 可在原版运行｜引文依据：模型卡 "**Neither file runs on stock llama.cpp.** The `hyv4` architecture is not upstream."；补丁表 "0001-hyv4-architecture.patch … both GGUFs need this"、"0002-stq1_0-quant-and-cuda.patch … STQ1_0 only"；构建命令 "git apply 0002-stq1_0-quant-and-cuda.patch # skip if only using Q4_K_M"｜修复要求：三处统一改为明确表述，例如「三份 GGUF 都不能跑在原版 llama.cpp 上：hyv4 架构补丁为所有产物所需，STQ1_0 产物另需量化与 CUDA 补丁」；若保留「两份」措辞，须写明指代对象并补充 Q4_K_M 亦需架构补丁的事实｜修复：三处（常见误解 5、第 4 章边界段、本章问题 2 答案）统一改为「hyv4 架构补丁对所有 GGUF 产物必需（含 Q4_K_M），STQ1_0 产物还需额外的量化与 CUDA 内核补丁，原版 llama.cpp 无法运行其中任何一个」。｜复验：已复验，全文无「两份 GGUF」表述。
- [重要·技术] index.html L1054（简化条件及其限制第 2 条）：「评测集有限（不含长上下文与知识套件）」与页面自身及来源矛盾——第 4 章评测表（L1010）将 MRCR 标注为「多轮长上下文检索」并写「检索类几乎没掉」，官方文章称「多轮长上下文检索基本在同一水平」；「不含长上下文」无法成立（第三方转述的原始表述是「四个评测均未涉及接近 1M 上下文长度的任务」，而非「不含长上下文任务」）｜引文依据：快科技转载「多轮长上下文检索基本在同一水平」；traictory "The model advertises 1M-token context; none of the four benchmarks exercises anything close. No knowledge or math suites (MMLU, GSM8K), no long-context tasks"；页面 L1010「MRCR（多轮长上下文检索）」｜修复要求：删除「不含长上下文」或改为可支持的表述，如「评测数字为厂商自报且评测集有限（四个评测未覆盖知识类套件，也未涉及接近 1M 上下文长度的任务），不能推广为『压缩对所有任务无损』」，并确认与第 4 章表格中 MRCR 的标签不再冲突｜修复：改为「未覆盖知识类套件，也没有接近模型宣传上下文长度的任务（MRCR 为多轮长上下文检索，但转述方指出四项评测均未接近模型宣传的上下文长度）」。｜复验：已复验，与正文 MRCR 标签不再矛盾。
- [轻微·可读性] index.html L1016（第 4 章精度段）：同一段落内「长文理解几乎（与原版）持平、数学小幅回落」重复出现两次（段中「官方同时报告该方案领先 UD-IQ1_M 产物，长文理解几乎与原版持平、数学小幅回落」与段末「官方同时说明长文理解几乎持平、数学小幅回落」）｜引文依据：不适用｜修复要求：删除段末第二次出现的「官方同时说明长文理解几乎持平、数学小幅回落。」一句，保留段中一次｜修复：删除性质段中重复的「官方同时说明长文理解几乎持平、数学小幅回落」（保留前一段的唯一一次表述）。｜复验：已复验。
- [轻微·技术] index.html L973（第 3 章本章问题第 2 条解答）：「比 gate/up 的档位刻意高两级」指代含糊——gate/up 有 STQ1_0 与 IQ2_XXS 两档，IQ3_XXS 相对 STQ1_0 高两级、相对 IQ2_XXS 仅高一级，模型卡原文 "deliberately 2 levels higher" 亦未指明参照档｜引文依据：模型卡 §3 "IQ3_XXS, IQ4_XS on last 3 — … deliberately 2 levels higher"（无参照物）｜修复要求：改为「比 gate/up 低档的 STQ1_0 高两级（STQ1_0→IQ2_XXS→IQ3_XXS）」或沿用「刻意高两档（模型卡原文：deliberately 2 levels higher）」，消除与 IQ2_XXS 比较时的歧义｜修复：改为「相对 gate/up 的低档 STQ1_0 高两级、相对高档 IQ2_XXS 高一级（模型卡原文称 deliberately 2 levels higher，以低档为基准）」。｜复验：已复验。
- [轻微·可读性] index.html L923–L925（第 3 章配方表）：MLA 与 iHC 首次出现未给最小含义，不符合本页对 MoE、残差流、GGUF、PTQ 等术语首次出现即给最小含义的处理方式｜引文依据：不适用（可读性；模型卡仅列名 "MLA `q_b`/`k_b`/`v_b`/`kv_a_mqa`"、"iHC `*_fn`, router, norms, sink"，未展开）｜修复要求：在第 3 章首次出现处（「进入配方前补三个最小含义」段或配方表理由列）为 MLA 与 iHC 各补一句最小含义；若无来源支撑其展开，至少标注为「模型卡原文名称，本页不展开」｜修复：配方表前置段补一句「配方表中还会出现 MLA、iHC 等名称，它们是 Hy4 架构中的张量族命名（模型卡原文如此列出，本页不展开其机制）」。｜复验：已复验。
- [轻微·格式] index.html L931–L933（第 3 章公式 F1 处）：平均 bpw 复算属于计算示例，未按 style-guide §4 标记「计算示例」（第 1 章构造示例已按规范标记「构造数据」）｜引文依据：不适用｜修复要求：在「这行配方的平均 bpw 可以直接复算」处以「计算示例」标记用途｜修复：公式段前补「计算示例：」标记。｜复验：已复验。
- [轻微·可读性] index.html L985（第 4 章首段）：第 3 章给出完整配方后，第 4 章直接以「先看体积账」开始，缺一句说明本章用体积与精度数字核对第 3 章配方实际效果的衔接（第 1→2 章已有明确衔接句）｜引文依据：不适用｜修复要求：在第 4 章首段前添加一句衔接，说明本章核对第 3 章配方的实际产出｜修复：dg-stack 图后补过渡段（总结 97.7%/2.3% 分账并指向下一章核对内容）。｜复验：已复验。
- [轻微·技术] index.html L994（第 4 章体积表 UD-IQ1_M 行）：定位写作「社区重要性混合基线」，「社区」定性无来源支撑，且与第 1 章对比表中「重要性混合基线（UD-IQ1_M）」标签不一致｜引文依据：模型卡对 UD-IQ1_M 仅称 "Using the UD-IQ1_M quantization strategy, half the size"；官方文章转载仅称「UD-IQ1_M 方案」，均无「社区」定性｜修复要求：删去「社区」二字，统一为「重要性混合基线（UD-IQ1_M）」｜修复：第 4 章产物表定位列改为「重要性混合基线」。｜复验：已复验。
- [轻微·技术] index.html L753（引言首段）：「约 1.5 TB」未在来源章节映射——该句引用 [N1]，但 N1 仅覆盖模型卡文件表（213.66 GiB / 2.38 bpw），文件表不含 1.5 TB；「接近 1.5TB」出自官方文章转载｜引文依据：快科技/IT之家转载「将模型权重从接近1.5TB压缩至约214GB」；模型卡文件表无 1.5 TB 字样｜修复要求：在「外部数字与实验条件（N）」小节补充「约 1.5 TB」的来源映射（在 N3 或新增条目中注明出自官方文章转载），使该数字可定位｜修复：N 段补「引言中『约 1.5 TB』出自官方文章（快科技等转载）」。｜复验：已复验。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 7
- 处置：修复。两条重要问题（「两份 GGUF」指代与遗漏、简化条件「不含长上下文」自相矛盾）修复并复验、轻微问题逐条处置（修复或记录接受理由）、重跑 `.dojo/scripts/validate.py` 通过后，本页可发布。
- 发布判定：**暂不可发布**。发布条件逐项判定（check.md §5）：
  1. 三轮独立审查 — 本轮（第 3 轮）为独立审查已完成；第 1、2 轮的独立性与完成情况超出本轮允许材料，需编排者确认。**条件性满足**。
  2. 每条来源论断有引文依据记录 — 本轮已逐条核对并记录（见来源核对记录），全部定位成功，无需要删除或降级的论断。**满足**。
  3. 阻断与重要问题全部关闭 — 当前存在 2 条重要问题未关闭。**不满足**。
  4. 遗留轻微问题有明确接受理由 — 待修复阶段决定。**未决**。
  5. 全部学习目标由正文章节完整回答 — 核心问题 1–4 分别由第 1–4 章完整回答。**满足**。
  6. 两级问题均有解答折叠块 — 页面级 4 条、章节级 9 条均有「解答：」折叠块，答案独立可读，核心问题答案指明完整论证章节。**满足**。
  7. 数学符号全部 LaTeX、结构图为 HTML — F1 用 `$$...$$` 且附符号表；两处图示均为 HTML 结构（dg-flow/dg-stack）；正则扫描与 validate.py 均无 Unicode 数学字符。**满足**。
  8. validate.py 返回成功 — `validation ok`。**满足**（修复后需重跑）。
  9. 可运行代码的结果与页面描述一致 — 本页无代码块。**不适用**。
  10. 关键论断和数字已重新核对来源 — 本轮完成。**满足**。
  11. head 元信息齐全 — description 纯文本、dojo:summary 可渲染（LaTeX 书写）、dojo:type=concept、dojo:topics=推理系统（validate.py 词表校验通过）、dojo:tag 齐全。**满足**。
  12. overview.html 与 index.html 相互链接 — nav 双向链接存在。**满足**。
  13. 概念链接有效或明确占位 — `../sherry-ternary-quant/index.html` 存在（目标章节标题未核，超出本轮允许材料）。**满足**。
  14. 递归生成的前置概念页（Sherry 稀疏三值量化）已完成各自质检 — 超出本轮允许材料，需编排者确认。**条件性满足**。


## 发布记录（编排者，2026-09-01）

- 第 3 轮 2 重要 / 7 轻微已全部修复并复验（见各条目「修复/复验」栏）；修复后 `validate.py` 返回 validation ok；无头 Chrome 渲染探针 10 个 .katex 节点、正文无残留 $ 定界符。
- 第 1、2、3 轮均由未参与写作与修复的独立子代理执行；前置概念页 sherry-ternary-quant 已完成自身三轮质检并发布（见该页 research/review-3.md 发布记录）。
- 发布决定：三轮完成、阻断与重要问题全部关闭、机械验证通过，本页发布。
