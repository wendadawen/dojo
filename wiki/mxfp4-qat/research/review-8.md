<!-- review-meta
round: 8
page: wiki/mxfp4-qat/index.html
reviewed_content_sha256: 793c9841a44926c6
-->
# MXFP4 量化感知训练审查记录（第 8 轮）

- 页面版本：c98b7ec629139a84f08f3b78b06b6e7c1e1be1e5（wiki/mxfp4-qat/index.html 工作树哈希）
- 审查时间：2026-09-14 17:41
- 审查者：编排者派发的独立审查者（未参与写作，未读取本页 research/ 下任何文件，含前序审查记录）
- 已完整阅读章节（按页面顺序）：《核心问题》（5 题解答折叠块）→《常见误解》→「1. MoE 专家权重——896 个专家压到 4-bit 能省多少显存」（含"每个专家参数量与 BF16/MXFP4 显存"折叠块、本章问题 3 题）→「2. MXFP4 编码——一个权重值怎么用 4-bit 表示」（含块结构图、两个 4 元素块折叠块、本章问题 3 题）→「3. QAT 机制——前向和反向做了什么，与 PTQ 差在哪」（含单权重 QAT 一步折叠块、可运行代码折叠块、STE 类比 callout、本章问题 3 题）→「4. RL 一致性——QAT 怎么贯穿 SFT 和 RL 且不产生 mismatch」（含 RL 循环图、本章问题 3 题）→「5. 选择性量化——K3 量化了哪些组件、不量化哪些」（含组件表、config.json 节选折叠块、本章问题 3 题）→「来源与范围说明」（C1–C9 / F1–F3 / N1–N7 / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）。另完整阅读 wiki/mxfp4-qat/overview.html。

## 来源核对说明（本轮实际执行）

- K3 报告基准版本：**arXiv:2607.24653v2 的 PDF**（下载自 https://arxiv.org/pdf/2607.24653v2，pdftotext -layout 抽文）。选定依据：页内 C1 标注"原文约行 928–932"，该行号在 v2 PDF 中恰为 §4.1.4 首段（"4.1.4 Deployment-Aware Post-Training / MXFP4 Quantization-Aware Post-Training To reduce memory footprint…"），说明页面以 PDF 版为准。同时用 arXiv HTML v2（/html/2607.24653v2）与 v1（/html/2607.24653v1）对照，确认 HTML 与 PDF 的文献编号不同（HTML v2 中 Jacob 为 [49]、Rouhani 为 [103]；PDF v2 中 Jacob 为 **[50]**、Rouhani 为 [104]；v1 分别为 [48]/[102]）。页内 C6 标注"K3 引用 [50]"、C5 引 arXiv:2310.10537，与 **PDF v2 的文献表**逐条相符——已回文献表确认，非编号错误。
- HuggingFace `moonshotai/Kimi-K3` 的 `config.json`（https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json，2026-09-14 取回，逐字段解析）。
- ONNX 数据类型文档"Float stored in 4 bits"（onnx.ai/onnx/technical/float4.html）与"Float stored in 8 bits"（…/float8.html）。
- Rouhani et al. 2023《Microscaling Data Formats for Deep Learning》（arXiv:2310.10537，Table 1 与 Algorithm 1 全文）。
- `ml_dtypes` README（float4_e2m1fn / float8_e8m0fnu 条目）。
- 未取回项：OCP Microscaling Formats (MX) Specification v1.0 的 PDF（opencompute.org 直连与 WebFetch 均返回 Cloudflare 403）。页面 C5/C7 对该规范的引用改为经 Rouhani et al. 2023（明确声明其 Algorithm 1"follows the semantics outlined in Section 6.3 of the OCP Microscaling Specification"）与 ONNX 文档交叉核对；§5.4.1 这一具体小节号本轮无法独立打开核对，但其所承载的格式事实（E2M1 元素、块大小 32、E8M0 共享 scale、超界截断保留符号）已由上述两个可及来源逐字确认，不构成待删条目。

## 逐项核对结果（要点）

1. 论断与来源：C1/C2/C3/C4 四段引文与 v2 PDF §4.1.4 逐字一致（含 C2 的破折号句 "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch."、C3 的 "all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision."）。C6 的 Jacob et al. 2018（CVPR，pp. 2704–2713）= PDF v2 文献表 [50]，一致。C5/C9 的 E2M1 值集 {0, 0.5, 1, 1.5, 2, 3, 4, 6}、指数 bias 1、`S.00.1₂ = 2⁻¹`、最大幅度 6、超界 `x>6→6`／`x<-6→-6` 与 ONNX float4 页逐条相符；C7 的 E8M0（无符号、bias 127、`11111111₂` 为 NaN、Min `00000000₂=2⁻¹²⁷`、Max `11111110₂=2¹²⁷`）与 ONNX float8 页、ml_dtypes（"Exponent range from -127 to 127 / Single NaN value (0xFF)"）相符。N1/N2/N4 的 config 字段（num_experts=896、num_experts_per_token=16、num_shared_experts=2、hidden_size=7168、routed_expert_hidden_size=3584、moe_intermediate_size=3072、num_hidden_layers=93、first_k_dense_replace=1、group_size=32）与 raw config.json 完全一致；N7 的 Abstract "a 2.8T parameter Mixture-of-Experts model with 104 billion activated parameters" 与 §3.2 表 1 "Total Parameters 2.78T" 与 PDF 相符；N6 的 MXFP8=FP8(E4M3/E5M2)、块 32、E8M0 scale 与 Rouhani Table 1 相符。
2. 公式与算术：3×3584×3072=33,030,144；896×92=82,432；82,432×33.03M≈2.72T；2.72T×2≈5.44 TB；2.72T×0.5≈1.36 TB，(2.72T/32)×1≈85.1 GB；1.36+0.085≈1.445 TB；5.44/1.445≈3.76；有效位宽 4+8/32=4.25 bit，16/4.25≈3.76×；块结构 1+32×4/8=17 字节/32 权重——全部复算一致。两个 4 元素块与单权重 QAT 一步的相对误差（17%/25%/11%/25%）复算一致。符号 s_b、e_b、q_i、x_i、x̂_i、ŵ、w 全文单义。
3. 代码：页面折叠块中的 Python 代码在本机 Python 3 实跑，输出与页面"预期输出"逐行一致（含 `w=0.8, 量化后 w_hat=0.75, 前向偏差=0.0500`）。块 2 的最近邻舍入（1.2→1.0、0.4→0.5、1.8→2.0、0.8→1.0）均非平局点，与 ONNX 的 RNE 规则不冲突。
4. 公式渲染：用本地 libs/katex.min.js 对页面全部 **203** 个 `$…$` / `$$…$$` 表达式以 `throwOnError:true, strict:true` 逐个渲染，0 失败；`dojo:summary` 无公式，可正常渲染。
5. 结构与功能：核心问题 5 题 + 五章本章问题各 3 题全部有解答折叠块，核心问题答案均指明完整论证所在章节；两处结构图为 HTML（非等宽框线图），无 img、无 `$…$` 出现在 alt；内链 ../moe-serving/index.html、../quantization-basics/index.html、overview.html、../../index.html 与 ../../libs/* 全部存在；`.dojo/scripts/validate.py` 返回 "validation ok"；`dojo:type=concept`、`dojo:topics=训练与优化`、`dojo:tag=量化` 均在词表内。
6. 表述：全文检索"我们/我/你/需要注意/下面/接下来/本页将/综上"等，0 命中；无调试叙事、无临场评价、无"（待生成）"占位。推断内容均以"本页推断"或在表格专列、来源说明"辅助解释与类比边界"中标明失效边界，未见把推断写成来源结论、也未见把构造示例写成来源事实。

## 问题

- [轻微·一致性] `index.html` 第 7 行 `<head>` 的 `dojo:summary`：summary 写"rollout 和训练共享量化方案，**以减少** train-inference mismatch"，而同页 `<meta name="description">`（第 6 行）、第 142/465/467/478/491/495/498/595/602/639 行正文与图注、以及 overview.html 第 41 行一律写"**消除**"，来源原文亦为 "eliminating"。同一事实在 summary 与页面其余位置取词不一致，且 summary 弱化了来源的论断强度。｜引文依据：K3 报告 v2 PDF 第 933 行 "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch."｜修复要求：把 `dojo:summary` 中"以减少 train-inference mismatch"改为"消除 train-inference mismatch"，与 description、正文、图注、overview 及来源一致。｜修复：｜复验：
- [轻微·格式] `index.html` 第 66–75 行 `<style>` 内的 `.diagram` 规则（`font-family: var(--font-mono); font-size: 0.86em; background: var(--bg-code); padding: 1rem; overflow-x: auto; border-radius: 6px; line-height: 1.5; margin: 1.5rem 0;`）全页无任何元素使用：`.dojo/scripts/check_unused_css.py wiki/mxfp4-qat/index.html` 报 "found 1 dead rules"。｜引文依据：不适用｜修复要求：删除该 `.diagram { … }` 规则块（页内结构图只用 `.flow-diagram`）。｜修复：｜复验：
- [轻微·一致性] `overview.html` 第 50 行："K3 的选择性量化策略规定 attention、shared expert、router、lm_head、vision tower 等组件保持更高精度（config.json 的 `ignore` 列表印证）"——括号里的"config.json 的 ignore 列表印证"套在一串含 router 的组件之后，而 index.html 第 531 行明确写"config.json 的 `ignore` 未排除 router，router 仍在 `targets: ["Linear"]` 的量化目标内，此处以报告'非专家组件保持更高精度'的表述为准"。两页就同一依据的覆盖范围说法不一致，会让读者以为 router 也由 config 印证。｜引文依据：config.json `text_config.quantization_config.ignore` = ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*", "re:.*vision_tower.*", "re:.*mm_projector.*"]——不含 router，也不含 latent MoE 投影。｜修复要求：把 overview 该括号改为仅指向 ignore 列表真正覆盖的项（attention 投影、共享专家、lm_head、vision tower / mm_projector），router 与 latent MoE 投影标明依据是 K3 报告 §4.1.4，与 index.html 第 531、566 行表述一致。｜修复：｜复验：
- [轻微·技术] `index.html` 第 647 行「简化条件及其限制」把 scale 选取算法描述为"按块内最大幅度选 power-of-two **使归一化值落入 E2M1 表示范围**"。来源的转换算法并不保证归一化值落入可表示范围：`shared_exp = ⌊log2(max_i(|V_i|))⌋ − emax_elem`（E2M1 的 emax_elem = 2），归一化最大值落在 [4, 8)，可以超过 E2M1 的最大可表示值 6，超出者按截断保留符号处理——这正是同页第 271、648 行描述的分支，与该句"落入表示范围"互相拉扯（第 256、331 行两处只写"按 OCP 规范的 power-of-two 规则选定"，无此问题）。｜引文依据：Rouhani et al. 2023 Algorithm 1 第 1 行 "shared_exp ← ⌊log2(max_i(|V_i|))⌋ − emax_elem"、第 4 行 "P_i = quantize_to_element_format(V_i/X), clamping normal numbers"；其正文 "normal numbers that exceed the representable range of the element format are clamped to the maximum representable value, preserving the sign. This is in accordance with the OCP MX specification."｜修复要求：把该句改为"按块内最大幅度选 power-of-two scale，使归一化最大值落入元素格式的最大 binade（可能超过最大可表示值 6，超出者按截断保留符号）"，与第 271、648 行的截断分支说明一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。四条轻微问题均为用词/一致性/死 CSS/简化描述，不改变任何核心结论、公式与数字；建议随手修复后归档，不阻塞发布。

统计：阻断 0 / 重要 0 / 轻微 4