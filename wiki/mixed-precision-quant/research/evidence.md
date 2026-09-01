# 逐层混合精度量化：核心论断与证据

来源简写：
- [HF] HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡（英文与中文节，2026-08/09 读取）
- [HY] 腾讯混元官方技术文章「Hy4 preview 轻量版」（2026-09-01）
- [TRA] traictory.com 对 Hy4 压缩的第三方分析（2026-08-31，含厂商数字转述与模型事实）
- [SHERRY] Sherry 论文 arXiv:2601.07892 与 llama.cpp PR #22836（经 Sherry 概念页核对）
- [LLAMACPP] llama.cpp 仓库 tools/imatrix/README（llama-imatrix 工具的用途与统计口径）

## C 论断

- C1：「混合精度」在训练语境指 FP16/BF16 与 FP32 主权重混用，在推理量化语境指不同层/张量使用不同比特宽度；本页指后者。来源：术语常识+两类语境对照；不作为外部来源论断，页面开头作消歧说明。已确认（消歧性质）。
- C2：UD-IQ1_M 是社区已有的重要性混合方案（官方文章称「社区常用」）：Hy4 的 UD-IQ1_M 产物中大部分路由专家层用 IQ1_M（1.75 bpw）、部分层用 IQ2_XXS（2.0625 bpw），整模 219.83 GiB / 2.44 bpw。来源：[HF] 文件表与 "What these are" 节（档位与体积）；[HY]「社区常用的 UD-IQ1_M 方案,大部分路由专家层都用 IQ1_M(1.75 bpw)」。已确认。
- C3：MIX-STQ1_0 中路由专家 gate/up 投影在 29 层用 STQ1_0（1.3125 bpw）、另 48 层用 IQ2_XXS（2.0625 bpw），层的选择由 imatrix 推导。来源：[HF] "The routed-expert gate/up projections run at 1.3125 bpw (STQ1_0) on 29 layers and 2.0625 bpw (IQ2_XXS) on the other 48 ... layer choice is imatrix-derived"。已确认。选层具体准则未公开，正文按推断边界标注。
- C4：ffn_down_exps 用 IQ3_XXS、最后 3 层 IQ4_XS：down 投影直接写回残差流，其误差不会被后续 gate 衰减，故刻意比 gate/up 高两档。来源：[HF] 配方表 "writes straight into the residual stream, so its error is not attenuated by a later gate — deliberately 2 levels higher"。已确认。
- C5：attention out/gate/q_a 用 Q5_K：llama.cpp 只在 n_expert==8 时对这些张量自动提档，Hy4 有 256 个专家，自动逻辑不生效，需配方显式指定。来源：[HF] 配方表。已确认。
- C6：MLA 分量（q_b/k_b/v_b/kv_a_mqa）用 Q8_0：Hy4 的拆分命名匹配不上 llama.cpp 的子串提档规则，完全拿不到自动提档。来源：[HF] 配方表。已确认。
- C7：DSA indexer 用 Q8_0/F32：105 个张量共 0.21 GiB，决定每个 query 能看到哪 2048 个 token。来源：[HF] 配方表。已确认。
- C8：iHC 的 *_fn、router、norms、sink 保持 F32：对齐参考实现的 _keep_in_fp32_modules；输出层（lm_head）F32 通过 --leave-output-tensor。来源：[HF] 配方表。已确认。
- C9：三个路由专家族占 Hy4 全部参数的 97.7%，因此配方把比特预算的大头花在专家上、其余张量「舍得花」。来源：[HF] "The three routed-expert families are 97.7% of all parameters"。已确认。
- C10：MIX-STQ1_0 的路由专家权重平均 1.78 bpw，且比 UD-IQ1_M 少占 5 个多 GiB。来源：平均 1.78 bpw 可由 (29×1.3125+48×2.0625)/77=1.7799 独立复算（层数与档位来自 [HF]）；「少占 5 个多 GiB」出自 [HY] 正文「路由专家权重不光评测表现更好,还比 UD-IQ1_M 少占了 5 个多 GiB」（快科技转载一致）。已确认。注：初稿曾引用 [HY] 配图中的 102.8/108.3 GiB 与 1.88 bpw，因图注文字无法被审查者独立复核，按规则删除。
- C11：在不增加平均比特预算的前提下逐层分档，整体量化误差低于统一分档。来源：[HY]「在不增加平均比特预算的前提下,这样逐层分档,整体量化误差反而更低」；机制由构造示例演示。已确认（机制性质，构造示例可复算）。
- C12：imatrix 是 llama.cpp 生态的校准产物（llama-imatrix 工具在模型上跑校准文本推理、按张量收集激活平方和等统计作为重要性分数），STQ1_0 编码器强制需要 imatrix（scale 求解与零位选择都用它）。来源：[HF] "An imatrix is mandatory for STQ1_0 — its encoder uses it for the scale solve and zero placement"；机制见 [LLAMACPP] tools/imatrix/README（"Compute an importance matrix for a model and given text dataset"、统计含 "Σ(Act²): sum of all squared activations (the importance scores)"）。已确认。
- C13：压缩后的模型在主流任务上与 BF16 差距一两分以内、检索类基本没掉、领先 UD-IQ1_M；长文理解几乎持平、数学小幅回落。来源：[HY] 正文与图。已确认（厂商自报，定性引用）。
- C14：整模文件：MIX-STQ1_0 213.66 GiB / 2.38 bpw；UD-IQ1_M 219.83 GiB / 2.44 bpw；Q4_K_M 435.20 GiB / 4.86 bpw（安全默认档）。来源：[HF] 文件表。已确认。

## F 公式

- F1：路由专家 gate/up 平均 bpw = (29×1.3125 + 48×2.0625) / 77 = 1.7799 ≈ 1.78。由 [HF] 层数与格式 bpw 直接计算，与 [HY] 的 1.78 互验。已确认。

## N 数字

- N1：文件表：Q4_K_M 435.20 GiB / 4.86 bpw；UD-IQ1_M 219.83 GiB / 2.44 bpw；STQ1_0（MIX） 213.66 GiB / 2.38 bpw。来源：[HF]。
- N2（并入 C10）：路由专家平均 1.78 bpw（F1 复算）；比 UD-IQ1_M 少占 5 个多 GiB（[HY] 正文）。
- N3：评测（厂商自报、单次运行、无第三方复现）：MCPAtlas 83.7→83.2；SWE-Bench 多语言 82.9→81.3；MRCR 81.3→81.1；IFBench 73.5→72.5；保留率 98.1%–99.8%。来源：[TRA] 转述厂商发布（数字与 [HY] 图一致方向）。条件：BF16 对 MIX-STQ1_0。
- N4：8×H20 实测（英文节）：pp512 204.56±1.42 t/s、tg128 20.47±0.02 t/s；中文节 tg128 为 19.52±0.01，两处不一致。来源：[HF]。本页如引用须注明不一致；默认不引用 decode 数字。
- N5（删除）：初稿曾引用 STQ1_0 产物的张量类型统计（type histogram），因模型卡现文无法定位该内容，按规则整段删除，页面不再使用。
- N6：STQ1_0 与 IQ2_XXS 档位 bpw：1.3125（[PR #22836] 与 [HF]）与 2.0625（[HF]「2.0625 bpw (IQ2_XXS)」）；IQ1_M 1.75（[HY]）；IQ3_XXS 等其余档位仅当名称使用，不写数值。

## 构造示例登记

- 两层两档构造示例（第 1 章）：敏感层误差低档 100/高档 20，不敏感层低档 30/高档 25，预算平均 1.69 bpw；用于演示同预算下分配方向的决定性，数字全部人为构造。
