<!-- review-meta
round: 4
page: wiki/mixed-precision-quant/index.html
reviewed_content_sha256: 8500cda789ae808d
-->
# 逐层混合精度量化（MIX-STQ1_0）审查记录（第 4 轮）

- 页面版本：ce451e8bd9ab55323d9aae6f8df69cac19fa2e28（工作树 wiki/mixed-precision-quant/index.html）
- 审查时间：2026-09-13 19:47
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何规划、修复或前序审查文件）
- 已完整阅读章节：核心问题、常见误解、1. 同样的预算，比特落在哪一层、2. 敏感度怎么量——校准数据与 imatrix、3. MIX-STQ1_0 配方——Hy4 的比特分配表、4. 代价与收益——体积、精度与边界、来源与范围说明（含全部折叠块、图注）

## 来源核对（本轮逐条回源）
- HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（https://huggingface.co/AngelSlim/Hy4-preview-GGUF/raw/main/README.md）：文件表 `435.20 GiB / 4.86`、`219.83 GiB / 2.44`、`213.66 GiB / 2.38` ✓；`The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48` ✓；`The three routed-expert families are 97.7% of all parameters` ✓；`An imatrix is mandatory for STQ1_0 — its encoder uses it for the scale solve and zero placement` ✓；`layer choice is imatrix-derived` ✓；`writes straight into the residual stream ... deliberately 2 levels higher` ✓；`llama.cpp only auto-bumps these when n_expert == 8; HY4 has 256` ✓；`HY4's split names miss llama.cpp's substring match` ✓；`DSA indexer | Q8_0 / F32 | 105 tensors, 0.21 GiB total, gates which 2048 tokens each query sees` ✓；`iHC *_fn, router, norms, sink | F32 | mirrors the reference's _keep_in_fp32_modules` ✓；`output (lm_head) | F32 | via --leave-output-tensor` ✓；补丁 `0001 ... both GGUFs need this`、`0002 ... STQ1_0 only` ✓。
- 腾讯混元官方文章（x-techcon 181825 / ifeng 8w4NPbQRMoR 原文一致）：`MIX-STQ1_0的路由专家权重……还比UD-IQ1_M少占了超5GiB`（对象为**路由专家权重**）；`敏感的层留IQ2_XXS（2.06bpw），不敏感的层用压缩更强的STQ1_0（1.31bpw）`；`在同等平均比特预算下实现更低量化误差`；`长上下文评测集MRCR、真实工具调用Agent评测MCP-Atlas等任务中分差不到1分`。与页面 [C10][C11][C13] 一致。
- 第三方英文转述 traictory「Tencent Hy4-preview Compression」（2026-08-31，https://traictory.com/news/2026-08-31-tencent-hy4-compression）：四项评测 `83.7 → 83.2`、`82.9 → 81.3`、`81.3 → 81.1`、`73.5 → 72.5` ✓；`These are unaudited vendor measurements: single runs, no error bars, no third-party reproduction as of this writing.` ✓；`The model advertises 1M-token context; none of the four benchmarks exercises anything close.` ✓。页面 N3 与「简化条件」引用属实。
- llama.cpp `tools/imatrix/README.md`：含 `Σ(Act²): sum of all squared activations (the importance scores)`，页面 [C12] 的 imatrix 机制描述属实。
- llama.cpp PR #22836：含 `STQ1_0` 与 `1.3125`。✓
- 复算：F1 `(29×1.3125+48×2.0625)/77 = 137.0625/77 = 1.7799 ≈ 1.78` ✓；构造示例 `20+30=50`、`100+25=125`、`125/50=2.5` ✓；保留率 `83.2/83.7=99.4%`、`81.3/82.9=98.1%`、`81.1/81.3=99.8%`、`72.5/73.5=98.6%`，区间 98.1%–99.8% ✓；`219.83−213.66=6.17 GiB` ✓；`213.66/435.20≈0.49`「约一半」✓；`100−97.7=2.3` ✓。
- 机械项：前置概念页 `../sherry-ternary-quant/`、`../hy4-preview-lite/`、`../hy4-preview-dataflow/` 均真实存在；无「（待生成）」占位；无 research/ 路径引用；无 Unicode 数学字符；`python3 .dojo/scripts/validate.py wiki/mixed-precision-quant/index.html` 返回 `validation ok`。

## 问题
- [轻微·表述] 引言末段与正文各章首：行程式章节预告与「先看／再看」式引导语——引言第 3 段「本页先建立预算视角并算一个构造示例，再看敏感度的来源 imatrix，然后逐行拆解 Hy4 的完整配方，最后核对体积与精度数字并划定策略边界」，第 3 章首「进入配方前补三个最小含义。」「先看体量格局：」，第 4 章首「先看体积账。」「再看精度账。」，图注节点「层档位推导（本页）」。｜引文依据：不适用｜修复要求：概念规范 style-guide §12 允许以「本页」自称、§2 要求引言说明文章结构，故本条针对的是行程式预告与「先看／再看」引导句式，而非「本页」一词本身：把引言的结构说明改为不以「本页」作主语、不预告动作顺序的直接陈述，正文各章首删去「先看／再看／进入…前」改为直接给出内容（如第 4 章首直接列出三个产物的体积表）。｜修复：｜复验：
- [轻微·表述] 第 4 章「需要如实标注的性质」段：把无来源支持的临场评价写成陈述——「评测集覆盖工具使用、编码、检索、指令遵循——正是本地部署会买账的场景」。｜引文依据：不适用｜修复要求：改为中性表述，只陈述评测集覆盖的四类任务，删去「会买账」这一无来源的判断（可保留「未覆盖知识类套件、无误差线」这类可核对的限定）。｜修复：｜复验：
- [轻微·技术] 第 4 章「路由专家部分的对比……」段：把只覆盖 gate/up 的 1.78 bpw 标注为整个路由专家族的平均值——「MIX-STQ1_0 的路由专家权重平均 1.78 bpw（由档位与层数直接复算，见「MIX-STQ1_0 配方——Hy4 的比特分配表」）」。｜引文依据：模型卡配方表 `ffn_gate_exps / ffn_up_exps | STQ1_0 (29 layers) / IQ2_XXS (48 layers)`，而 `ffn_down_exps | IQ3_XXS, IQ4_XS on last 3`（bpw 更高）；页面 F1 的复算 `(29×1.3125+48×2.0625)/77` 只含 gate/up，不含 down。｜修复要求：将标签限定为复算范围，改为「路由专家 gate/up 的权重平均 1.78 bpw」或「gate/up 一行平均 1.78 bpw」（第 4 章正文、核心问题解答与本章问题解答中同一表述一并处理）。｜修复：｜复验：

## 结论
- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。三项均为轻微表述/标注问题，不影响核心结论与主线理解（1.78 的数值本身由页面 F1 复算得出且无误，只是标签范围偏大；两处表述问题不改变任何事实）。发布的实质条件（三条来源论断均已回源核对、公式可复算、无 research/ 路径、前置概念页均存在、validate.py 通过、无「（待生成）」占位、问题块两级均作答且核心问题答案指明所在章节）均满足。建议按上列修复要求就地把三项轻微问题改净后再发布；若按 §5 保留，接受理由为「措辞与标签范围问题，不影响来源一致性与结论正确性」。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
