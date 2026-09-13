<!-- review-meta
round: 8
page: wiki/kimi-k3/index.html
reviewed_content_sha256: 228bf625d0557b4f
-->
# Kimi K3 审查记录（第 8 轮）

- 页面版本：d70b331016f5a2c6ec63b6f334d3f7503cbfde99（index.html 工作树）
- 论文版本：arXiv:2607.24653v2（2026-08-07）
- 审查时间：2026-09-13 22:42
- 审查者：独立子代理（未参与写作与前序审查；未读取本页 research/）
- 已完整阅读章节（按顺序）：正文引言与元信息、核心问题（5 题）、1. 三维度信息流——K3 架构总览、2. 序列维度——KDA + 混合注意力、3. 深度维度——Block AttnRes、4. 宽度维度——Stable LatentMoE、5. 原生视觉——MoonViT-V2、6. 训练——数据、scaling law、Muon、长上下文扩展、7. 后训练——SFT→RL→MOPD 三阶段、QAT、EAGLE、8. 基础设施——3T 训练 + 1M RL + 推理、9. 性能与评价、10. 独立评价——系统性设计、开源里程碑与边界、来源与范围说明（含全部折叠块与 5 张表格）
- 机械核对：`.dojo/scripts/validate.py` → validation ok；`check_inline_js.py` → ok；`dojo:type=paper`、`dojo:topics=模型结构`（词表内）、`dojo:tag=模型架构`（词表内）均合规；引用概念页 kda/mla/nope/block-attnres/stable-latent-moe/situ-glu/quantile-balancing/moonvit-v2/per-head-muon/mopd/mxfp4-qat/eagle-speculative/flash-kda/moonep/gpu-execution-model 全部存在；index.html 与 overview.html 双向链接；无 `<img>` 携带数学 alt。
- 来源核对方式：arXiv:2607.24653v2 正文（WebFetch 可及 §1–§4.2）、HuggingFace config.json 原文与 model card。可及范围内逐条核对通过：Table 1 单元格（1.04T/2.78T、32.6B/104.2B、384/896、8/16、128K/1M、MLA→Hybrid KDA–MLA、SwiGLU→SiTU-GLU）、§2.1 3:1 混合与"the final layer always performs global attention"、§2.1.2 NoPE 与"extrapolates directly to 1M-token contexts without any positional-encoding modification"、§2.2 "8 blocks with 12-layer size … partial final block … 9 total blocks when counting the embedding layer" 与 "memory and communication overhead drop from O(Ld) to O(Nd)" 及 "N≈8 recovers most of the benefit across model scales"、§2.3.3 "Maintaining balanced loads becomes more challenging as LatentMoE increases the routed expert pool to 896 per layer."、§2.4 MoonViT-V2 "27-layer … roughly 0.4B … adopts RMSNorm … removes all bias terms" 与梯度范数结论、§2.5 Per-Head Muon "partition their momentum matrices along the head dimension and orthogonalize each head's block separately"、§3.1 四域 + 视觉语料、§3.2 "optimal peak learning rates and batch sizes differ substantially" / "cosine decay consistently achieves a lower final loss than WSD" / "Evaluated on held-out OOD validation data" / "collectively deliver an approximately 2.5× gain in overall scaling efficiency over Kimi K2"、§3.3 "cosine learning rate schedule with a 1% linear warmup" + "Weight decay is set to 0.1"、§3.4 "four-stage curriculum … 8K to 64K … 256K to 1M"、§4.1.4 QAT "we quantize the MoE expert weights … activations computed in MXFP8" 与高精度清单 "(attention projections, latent MoE projections, shared experts, and MoE routers)"、EAGLE "unrolled for seven steps" / "outputs of the 1st, 4th, and final AttnRes blocks" / LK loss "the negative logarithm of the acceptance rate itself"、Table 2 GDPval-AA v2 Elo（Fable 1747 / Sol 1736 / K3 1686 / Opus 4.8 1593 / GLM 1510 / GPT-5.5 1491）与 "All Fable 5 results are with potential fallbacks. All GPT-5.6 Sol results include potential cyberguards."；config.json 逐项核对 num_hidden_layers=93、num_experts=896、num_experts_per_token=16、num_shared_experts=2、attn_res_block_size=12、first_k_dense_replace=1、activation_situ_beta=4.0、activation_situ_linear_beta=25.0、mla_use_nope=true、latent_moe_use_norm=true、vt_num_hidden_layers=27、max_position_embeddings=1048576、full_attn_layers（24 项）/kda_layers（69 项）。§6 及之后（评测表、§6.1.3 harness 注、§6.3 Table 5、§6.4 成本）在 arXiv HTML 与 ar5iv 两种取用下均在 §4.2.7/§5.3.1 处被截断，无法取原文，相关条目仅能核页内一致性。

## 问题

- [阻断·技术] §9 开头（line 512）及小结（line 642）、核心问题 5（line 130）："持续优于 Claude Opus 4.8、GPT-5.5" 与本页 Table 2 自身的读数直接矛盾——HLE-Full 行 K3 43.5/56.0、Opus 4.8 49.8/57.9（Opus 在无/有工具两栏均高于 K3）；CritPt 行 K3 23.4、GPT-5.5 27.1（GPT-5.5 高于 K3）；GPQA Diamond 行 K3 93.5、GPT-5.5 93.5（并列，非"优于"）。论文原文对应表述本身带有范围（Abstract：“the model still trails the strongest proprietary systems but leads other models in their test suite”），页面把有条件的整体定位写成无条件的"持续优于"。｜引文依据：本页 Table 2 HLE-Full 行 `43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | —`；CritPt 行 `23.4 | 28.6 | 32.3 | 20.9 | 27.1 | 20.9`；GPQA Diamond 行 `93.5 | 92.6 | 94.1 | 91.0 | 93.5 | 91.2`；论文 Abstract 原文 "still trails the strongest proprietary systems but leads other models in their test suite"。｜修复要求：把 line 512、line 130、line 642 三处的"持续优于 Claude Opus 4.8、GPT-5.5"改为与 Table 2 一致的有条件表述（例如"总体优于……，研究级推理（CritPt、HLE-Full）为例外"），或列举例外，使正文与同页表格不再冲突。｜修复：｜复验：

- [重要·技术] §9.4 折叠块（line 579）："GPT-5.5 的努力级别低于其余模型，横向比较时需注意"与论文结果表注不符。｜引文依据：抓到的原文表注为 "All maxed out on thinking effort: max or xhigh."——即所有对照模型均取各自的最高思考努力级别，max / xhigh 只是不同厂商的命名，不构成"低于"的证据；页面未给出任何说明 xhigh 低于 max 的来源片段。｜修复要求：回源核对 §6.1.4 表注；若原文只说明各模型均为最高努力（max 或 xhigh），删除"GPT-5.5 的努力级别低于其余模型，横向比较时需注意"，或改写为"GPT-5.5 的标签为 xhigh，其余为 max"并注明二者均为各自最高档。｜修复：｜复验：

- [轻微·格式] 公式编号 F3 在正文与来源说明之间不一致：正文 line 273 标注 "（§2.2 Eq.8-9）"，来源说明 line 690 写作 "F3（Block AttnRes Eq.8-10）"。｜引文依据：不适用｜修复要求：两处统一为同一公式范围。｜修复：｜复验：

- [轻微·技术] §7 QAT（line 444）：括号把 "MoE router 保持高精度" 一并归因于 "config.json quantization_config.ignore 确认"，但该 ignore 只有 6 条 regex——`re:.*self_attn.*`、`re:.*shared_experts.*`、`re:.*mlp\.(gate|up|gate_up|down)_proj.*`、`re:.*lm_head.*`、`re:.*vision_tower.*`、`re:.*mm_projector.*`——没有任何 router 模式，config 无法"确认" router 高精度。｜引文依据：config.json `quantization_config.ignore` 六条原文（见上）；论文 §4.1.4 高精度清单原文 "(attention projections, latent MoE projections, shared experts, and MoE routers)"。｜修复要求：把 "MoE router" 一项的依据改标为论文 §4.1.4，或把该括号拆成"论文 §4.1.4 列出的高精度项（含 MoE router）"与"config.json ignore 另列出 lm_head/视觉塔/多模态投影器"两句，使归因与来源对应。｜修复：｜复验：

- [轻微·表述] §9.4（line 569）："Intelligence Index v4.1 = 57.1，#4/580（第三如果 GPT-5.6 Sol 努力变体算一个）"括号内表述语义不自洽（"第三如果……算一个"无法判定指的是哪种计数方式）。｜引文依据：不适用｜修复要求：改写为可判定的表述（例如"#4/580；若把 GPT-5.6 Sol 的各努力级别变体合并为一个条目，则排第 3"），或直接删除无法核实的括号注。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 3
- 处置：修复（阻断与重要须关闭后复验；轻微项一并修正）
- 说明：本轮未发现数字与官方材料不符、算式与结论不符、summary/overview 与正文数字冲突、图注读数与刻度不符、公式不可 KaTeX 渲染、Unicode 数学字符裸写、结构图违规、无解答折叠块、链接失效等问题；§6 之后的评测条目受取源截断限制，仅按页内一致性核对。