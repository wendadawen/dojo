<!-- review-meta
round: 4
page: wiki/fp8-block-quant/index.html
reviewed_content_sha256: 5cf41946c7778ff9
-->
# FP8 块量化审查记录（第 4 轮）

- 页面版本：4096f3c8c205a8421c94daf81d7c785bff113287
- 审查时间：2026-09-14 16:58
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 常见误解 / 1. E4M3 的位布局 / 2. 两种格式与特殊值的取舍 / 3. 缩放：把张量对准可表示区间 / 4. 分块：128×128 与 1×128 / 5. 配套策略与实证 / 来源与范围说明（含全部 <details> 折叠块、两幅 SVG 图与图注）
- 本轮核对手段：arXiv 原文（2209.05433v2 §2/§3/§3.1/§4.3、2412.19437v2 §3.3.2/§B.2，PDF 全文抽取 + HTML 渲染）、transformers v4.57.6 `quantizer_finegrained_fp8.py` 原文、GLM-5.3-Flash 官方 `config.json` 与全部 62 个分片的 safetensors 张量头（76,108 个张量）、headless Chrome 实渲染、本机 torch 2.8.0 实跑两段代码。

## 问题

- [阻断·技术] 第 5 章「GLM-5.3-Flash 的实证」列表第 2 条（「34 个 KDA 层的投影零量化（整个 KDA 路径不走 FP8，其中卷积核、$A$、$b_{dt}$ 留在 FP32）」）：把卷积核写成了 FP32，官方 checkpoint 中卷积核是 BF16，只有 $A$、$b_{dt}$ 是 FP32｜引文依据：读全部 62 个分片张量头——34 个 KDA 层的 `self_attn.{q,k,v}_conv1d.weight` 全部为 **BF16**（例：`model.language_model.layers.0.self_attn.k_conv1d.weight BF16 [8192,1,4]`，`q_conv1d`/`v_conv1d` 同），而 `self_attn.A_log F32`、`self_attn.dt_bias F32`；`config.json` 的 `quantization_config.modules_to_not_convert` 亦含 `self_attn.{q,k,v}_conv1d`。兄弟页 `wiki/glm-5-3-flash-dataflow/index.html` 明确写「其中卷积核未量化，为 BF16；$A$ 与 $b_{dt}$ 留在 FP32」，与本页矛盾｜修复要求：把「卷积核、$A$、$b_{dt}$ 留在 FP32」改为「卷积核未量化、为 BF16；$A$ 与 $b_{dt}$ 为 FP32」（$b_{dt}$ 即 `dt_bias`，$A$ 即 `A_log`）｜修复：第 5 章列表第 2 条改为「卷积核未量化、为 BF16；$A$ 与 $b_{dt}$ 为 FP32」，与 checkpoint 张量头（卷积权重 BF16、`A_log`/`dt_bias` 为 F32）及兄弟页一致｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·格式] 来源与范围说明 [F3]、[N1]，对应第 4 章 roundtrip 代码块与第 2 章 E5M2 表行：两条来源条目在来源章节定义，但正文始终未以 `<sup>[...]</sup>` 引用（正文该处引的是 [F2][F4]，E5M2 数值无上标），双向对应不成立｜引文依据：不适用（page 内 sup 全集为 C1–C15、F1/F2/F4/F5、N2/N3/N4，无 F3、N1）｜修复要求：在正文对应位置补引 [F3]（4×4 roundtrip 实测）与 [N1]（E5M2 范围/57,344/$2^{-14}$/32 binade），或删除这两个来源条目｜修复：第 4 章 roundtrip 代码块「验证的机制」补引 [F3]（改为 `<sup>[F2][F3][F4]</sup>`）、第 2 章 E5M2 表行 57,344 补引 [N1]（改为 `<sup>[N1]</sup>`），[F3]/[N1] 两条来源均有正文引用｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·图示] 图 1「E4M3 的 8 位布局」（第 1 章）：E 组四格只有 $E_3$ 着强调色（`dg-accent`），$E_2$/$E_1$/$E_0$ 为普通描边；M 组三格则全部着强调色。强调色的含义全页未定义，且应用不一致，实渲染下会读成「$E_3$ 位特殊」｜引文依据：不适用（实渲染截图：S 灰、$E_3$ 蓝、$E_2/E_1/E_0$ 灰、$M_2/M_1/M_0$ 蓝）｜修复要求：E 组四格统一着强调色（与 M 组一致），或全部改为普通描边，并在图注/goals 中不改动数值含义｜修复：图 1 的 $E_2$/$E_1$/$E_0$ 三格由 dg-box 改为 dg-accent，与 $E_3$ 及 M 组三格一致，E 组四格统一强调色；数值含义与图注未改动｜复验：已复跑 validate.py 通过并核对修改位置｜
- [轻微·技术] 第 5 章实证列表第 2 条「被量化的只有大矩阵乘权重（专家、MLA 投影、稠密 MLP）」：DSA 层的 `kv_b_proj` 属 MLA 投影，却在官方 checkpoint 中为 BF16、无 `weight_scale_inv`，且列于 `modules_to_not_convert`；按此措辞读者会认为 MLA 投影全部走 FP8｜引文依据：`model.language_model.layers.3.self_attn.kv_b_proj.weight BF16 [32768,512]`（无对应 scale 张量）；`config.json` `modules_to_not_convert` 含 `self_attn.kv_b_proj`；兄弟页把「DSA 的 kv_b_proj（展开投影）」列入未量化清单｜修复要求：把「MLA 投影」限定为实际量化的部分（如「MLA 的 $q_a$/$q_b$、$kv\_a$、$o$ 投影」），或将 kv_b_proj 显式排除｜修复：「MLA 投影」限定为「MLA 的 $q_a$/$q_b$/$kv_a$/$o$ 投影」，kv_b_proj 属未量化（同章未量化清单与兄弟页均已列出）｜复验：已复跑 validate.py 通过并核对修改位置｜

## 结论

- 已核对并确认无误的条目（供复验）：FP8 论文 Table 1 全部格式参数（E4M3 1/4/3 bias 7、E5M2 1/5/2 bias 15、448/57,344、$2^{-6}$/$2^{-14}$、18/32 binade、NaN 1/3 个）与 §3.1「240→448」「多出 7 个幅值（256–448）」「回收费 ±0/NaN 对称性（否则可到 480）」「E5M2 不回收因其已有 32 binade」；§2 饱和与不跳过更新、摊销还原；§4.3 困惑度 10.19 / 12.59 / 10.29 / 10.44 与「[7,10] 偏置匹配基线」；DSV3 §3.3.2 累加约 14 位、$K=4096$ 近 2%、$N_C=128$（=4 WGMMA）、128×128 权重 / 1×128 激活、全张量 E4M3、在线量化、微缩放一致性；§B.2 ~16B/~300B token 发散与 token 相关离群值假说；transformers v4.57.6 `quantizer_finegrained_fp8.py` L124-142 的 `scale=fp8_max/max_abs`、`clamp`、`reciprocal()` 存为 `weight_scale_inv`；GLM-5.3-Flash `config.json`（fmt=e4m3、activation_scheme=dynamic、weight_block_size=[128,128]、34 个 KDA 层）+ 全量 62 分片张量头：**scale 张量 37,338 个、形状全部 = ⌈N/128⌉×⌈K/128⌉、全部 F32**；参数量 321.323 B、张量字节 328,326,771,576（305.78 GiB）、1.0218 字节/参数、高估 2.2%；embedding/lm_head/视觉塔/indexer 均高精度。第 1、4 章两段代码在本机 torch 2.8.0 实跑，输出与「预期输出」逐字一致（448.0 / 1.562e-02 / 74.6667 / 0.142857 / 0.029085 / 4.632% / (16,32) / (128,12)）。`validate.py` 返回 `ok`；KaTeX 公式、折叠块、目录锚点、明暗主题与窄屏渲染正常；`<img>` 无 alt 数学；`<text>` 未使用；全文无元话语、「本页/本文」仅见于范围说明（style-guide §12 允许）、无第一/第二人称、无调试叙事。
- 统计：阻断 1 / 重要 0 / 轻微 3
- 处置：修复（阻断项须改后复验；三项轻微随本轮一并处理）