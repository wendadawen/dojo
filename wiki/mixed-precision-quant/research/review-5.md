<!-- review-meta
round: 5
page: wiki/mixed-precision-quant/index.html
reviewed_content_sha256: e828870ca2c03891
-->
# 逐层混合精度量化（MIX-STQ1_0）审查记录（第 5 轮）

- 页面版本：64e997c39c646a266dee9965615f69f1a3a617dd
- 审查时间：2026-09-14 17:03 CST
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题（4 条）、常见误解、1. 同样的预算，比特落在哪一层、2. 敏感度怎么量——校准数据与 imatrix、3. MIX-STQ1_0 配方——Hy4 的比特分配表、4. 代价与收益——体积、精度与边界、来源与范围说明；同时完整读取 overview.html，并抓取模型卡原文（huggingface.co/AngelSlim/Hy4-preview-GGUF/raw/main/README.md，249 行全文）、腾讯混元官方文章转载版与第三方评测转述。

## 来源核对纪要（本轮逐条回源，均通过）

- 文件体积/平均 bpw：模型卡文件表原文 `Hy4-preview-Q4_K_M.gguf | 435.20 GiB | 4.86`、`Hy4-preview-UD-IQ1_M.gguf | 219.83 GiB | 2.44`、`Hy4-preview-STQ1_0.gguf | 213.66 GiB | 2.38`，与正文、overview、图注三处完全一致；`435.20 / 213.66 = 2.04`，支持「约为 Q4_K_M 的一半」。
- 逐层配方表：模型卡 §3 表共 7 行，与页面表逐行对应（`ffn_gate_exps`/`ffn_up_exps` = STQ1_0 29 层 / IQ2_XXS 48 层；`ffn_down_exps` = IQ3_XXS、last 3 层 IQ4_XS；attention out/gate/q_a = Q5_K「only auto-bumps these when `n_expert == 8`; HY4 has 256」；MLA `q_b`/`k_b`/`v_b`/`kv_a_mqa` = Q8_0「split names miss llama.cpp's substring match」；DSA indexer = Q8_0/F32「105 tensors, 0.21 GiB total, gates which 2048 tokens each query sees」；iHC `*_fn`, router, norms, sink = F32「mirrors the reference's `_keep_in_fp32_modules`」；`output` (lm_head) = F32「via `--leave-output-tensor`」）。页面未漏行，亦未把无关行（如 Q4_K_M 构建中 `ffn_down_exps` 由 llama.cpp 自身逻辑提到 Q6_K 的 37 层）混入本表。
- UD-IQ1_M 分档：模型卡原文「The routed-expert `gate`/`up` projections run at 1.75 bpw (IQ1_M) and 2.0625 bpw (IQ2_XXS)」，支持正文 [C2]。
- imatrix 强制：模型卡 §4「**An imatrix is mandatory for STQ1_0** — its encoder uses it for the scale solve and zero placement」，支持 [C12] 的强制论断。
- 97.7%：模型卡「The three routed-expert families are 97.7% of all parameters」，支持 [C9]；2.3% = 100 − 97.7。
- F1 公式可复算：`(29×1.3125 + 48×2.0625)/77 = (38.0625 + 99.0)/77 = 137.0625/77 = 1.7800…≈1.78`，与 §3 正文、§4、三处问题解答所写 1.78 bpw 一致；`29+48=77` 与模型卡「首层 Dense，其余 77 层 MoE」一致。
- 构造示例自洽：`(1.31+2.06)/2 = 1.685 ≈ 1.69`；`20+30=50`、`100+25=125`、`125/50=2.5`，与核心问题 1、本章问题、常见误解三处 2.5 倍一致。
- 四项评测与保留率：官方/转述数字 MCPAtlas 83.7→83.2、SWE-Bench multi 82.9→81.3、MRCR 81.3→81.1、IFBench 73.5→72.5；复算保留率 83.2/83.7=99.4%、81.3/82.9=98.1%、81.1/81.3=99.75%、72.5/73.5=98.6%，落在页面所写 98.1%–99.8% 区间内；正文、核心问题 4 答案、本章问题 1 答案、overview 四处数字一致。
- 存储节省：官方文章「还比 UD-IQ1_M 少占了 5 个多 GiB」支持「5 GB 以上」；文件表 219.83 − 213.66 = 6.17 GiB。
- 运行时前提：模型卡「**Neither file runs on stock llama.cpp**」及 `0002-stq1_0-quant-and-cuda.patch   # skip if only using Q4_K_M`，支持「hyv4 补丁对所有产物必需、STQ1_0 产物还需量化与 CUDA 补丁」。
- 结构/功能：`dojo:type=concept`、`dojo:topics=推理系统`、`dojo:tag=量化`；公式全部由 KaTeX 定界符包裹（图内无 `<text>` ASCII 近似，图为 HTML div，无 `$…$` 出现在 alt）；`python3 .dojo/scripts/validate.py wiki/mixed-precision-quant/index.html` 返回 `validation ok`；三个前置概念页（sherry-ternary-quant、hy4-preview-lite、hy4-preview-dataflow）均真实存在且标题与被链文本一致；overview.html 与 index.html 互链。

## 问题

- [轻微·表述] §3 正文（`down 投影高两档的理由值得单独说。`）：以「值得单独说」这类关于本页自身组织方式的评论开场，属元话语/临场评价｜引文依据：不适用｜修复要求：改为直接陈述理由（如「down 投影高两档的理由与校准统计无关：gate/up 的输出……」），删去对本页取舍的评价性表述｜修复：｜复验：
- [轻微·来源] §2 正文（`llama-imatrix 工具在模型上跑校准文本的推理，按张量收集激活平方和等统计作为重要性分数……<sup>[C12]</sup>`）：该句描述的是 imatrix 的产出机制，但来源小节把 C12 定位为模型卡「4. Building a runtime → Re-quantizing from bf16」的「imatrix 强制」条目，机制另归 `llama.cpp tools/imatrix/README` 且未编号；按 [C12] 去查模型卡该节只有强制论断、不含机制描述，编号与文献表条目不对位｜引文依据：模型卡 §4 原文「An imatrix is mandatory for STQ1_0 — its encoder uses it for the scale solve and zero placement」（无产出机制）；本页来源小节「imatrix 的产出机制见 llama.cpp tools/imatrix/README」｜修复要求：把该句的上标改为指向机制来源（或在来源小节为 llama.cpp README 补一个编号），使上标与所引位置内容对位｜修复：｜复验：
- [轻微·来源] §4 正文（`官方称该方案比 UD-IQ1_M 节省 5 GB 以上存储空间（与整模文件差 6.17 GiB 自洽）`）：官方「5 个多 GiB」的口径是路由专家权重，括号里的 6.17 GiB 是整模文件差，两个口径不同却并置作「自洽」核对，未标明差异｜引文依据：官方文章「还比 UD-IQ1_M 少占了 5 个多 GiB」（路由专家权重）；模型卡文件表 219.83 − 213.66 = 6.17 GiB（整模）｜修复要求：标明两个数字各自的口径（路由专家权重节省 5 个多 GiB；整模文件小 6.17 GiB），或删去跨口径的「自洽」核对｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。全部事实性论断（体积、bpw、配方表 7 行、29/48 层、1.78 bpw 公式、97.7%、四项评测与保留率、运行时前提）已逐条回源核对通过；公式可复算、符号单义、summary 公式可渲染；构造示例已明确标注非实测；无元话语式「本页将…」、无会话指代、无调试叙事；页面结构、折叠块、问题块、来源编号、链接与校验脚本均合格。上述 3 条轻微问题不影响正确性与主线理解，接受为遗留轻微项。