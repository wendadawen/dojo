<!-- review-meta
round: 6
page: wiki/stable-latent-moe/index.html
reviewed_content_sha256: 6b9ff7974a492ab0
-->
# Stable LatentMoE 审查记录（第 6 轮）

- 页面版本：ed558daa105c8cf24ca47b40d43ca94c5cbef5bc
- 审查时间：2026-09-14 17:16
- 审查者：编排者派发的独立审查者
- 已完整阅读章节：核心问题；1. 扩大专家池的代价——为什么需要 LatentMoE（含本章问题）；2. Stable LatentMoE 的层结构——共享分支、路由分支与 RMSNorm 的位置（含图 1、本章问题）；3. 极端稀疏下的两个失败模式（3.1 失败模式一、3.2 失败模式二、图 2、本章问题）；4. 三件稳定化（4.1–4.4、本章问题）；5. 为什么 RMSNorm 必须插在路由聚合之后（5.1–5.4，含构造示例折叠块、本章问题）；6. 适用边界与不能推出的结论（含本章问题）；来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 来源核对依据（本轮实际打开的外部材料）

- K3 报告 §2.3/2.3.1/2.3.2/2.3.3 与 Eq. 11/12/13/14（arxiv.org/html/2607.24653v2）：Eq. 11 `u = Σ_{i∈T_k(x)} p_i E_i^routed(W↓x)`、`y = Σ_{j=1}^{Ns} E_j^shared(x) + W↑·RMSNorm(u)`；Eq. 12 `[β₁·tanh(W_g x/β₁) ⊙ Sigmoid(W_g x)] ⊙ [β₂·tanh(W_u x/β₂)]`，β₁=4（gate）、β₂=25（up），上界 `β₁β₂ = 100`；Eq. 14 `b̂_j^(t+1) ← −quantile_{1−k/n}(s_{:,j} − α^(t))`、`b_j^(t+1) ← b̂_j^(t+1) − mean(b̂^(t+1))·1`；固定步长规则 `b_j^(t+1) = b_j^(t) + γ·sign(ℓ̄ − ℓ_j^(t))`，γ「trades off slow adaptation against load oscillation」；「a chain of nearly four consecutive matrix multiplications」；「Imbalanced routing slows expert-parallel training and may leave some experts poorly trained」；「removes a common offset that leaves Top-k selection unchanged」；「balancing the load of nearly 10^3 experts exceeds the regime」；Table 1：Total Parameters 2.78T、Hidden Dimension 7,168、Latent MoE Dimension 3584 (0.5×)、MoE Hidden Dimension per Expert 3,072、Routed Experts 896、Experts Active per Token 16、Shared Experts 2；「Kimi K3 fixes the number of full-width shared experts to Ns=2 in every layer」。
- Hugging Face `moonshotai/Kimi-K3` config.json：`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`hidden_size=7168`、`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`、`dtype="bfloat16"`；`text_config.quantization_config` = `{quant_method:"compressed-tensors", format:"mxfp4-pack-quantized", num_bits:4, group_size:32}`——**权重为 MXFP4 4-bit，非 bfloat16**。
- DeepSeekMoE 论文（arXiv:2401.06066）：§3.2 标题即「Shared Expert Isolation」，「each token will be deterministically assigned to these shared experts」。
- 图内数值/几何：headless Chrome 实测页面 DOM 与截图。

## 问题

- [重要·图示] 位置：图 1「Stable LatentMoE 层数据流（K3 Eq. 11）」（第 233 行，共 17 个 `.dg-label`）｜来源：页面渲染（不适用）｜问题：`.dg-label` 为 `display:flex; flex-direction:column`（第 37–48 行 CSS），凡「文字 + 行内公式」混合的标签都会被拆成 4–6 个 flex 子项、每项独占一行，纵向排布超出 40/52 单位的盒子后被 foreignObject 裁掉，标签读不全；另「共享支（全宽，不走隐空间）」注记被下行连线穿过（压线）。｜引文依据：headless Chrome 探针测得 foreignObject[0]（输入标签）内容含 5 个 flex 子项，纵向 top 3709→3831（约 122px），而盒子只有 40 单位高；截图显示输入框只剩「x ∈ ℝ」「（」「d = 7168」三行且上下被裁，$E_1/E_2^{shared}$、Top-$k$ 路由专家、加权聚合、RMSNorm 四个标签同样被裁；测得「共享支」注记盒 x 80–320、中心 x=200，与连线 `M440 80 V100 H200 V126` 在 x=200、y 106–126 重叠。｜修复要求：让每个标签在盒内完整可见（把标签内容包进单个元素，或改为不因 flex-column 分行），并移动「共享支」注记使连线不穿过文字；修复后在 1200px 与 760px 两种宽度下重截图确认无裁剪、无压线。｜修复：｜复验：
- [重要·技术] 位置：2 章符号表「共享专家 FFN 中间维度 6144」（第 221 行）｜来源：config.json / K3 报告 §2.3、Table 1｜问题：报告与 config.json 均未给出共享专家 FFN 的中间维度；6144 由 `moe_intermediate_size × num_shared_experts`（3072×2）推得，属按 DeepSeek 共享专家约定所作的推断，却填在「config 字段」列，被当作 K3 的既有配置值。｜引文依据：K3 报告 Table 1 仅列「Shared Experts 1 → 2」，无共享专家中间维度；config.json 无 shared_expert_intermediate_size 之类字段，只有 `num_shared_experts: 2` 与 `moe_intermediate_size: 3072`。｜修复要求：把该行明确标注为按 DeepSeek 共享专家约定推得的推断（并在「简化条件及其限制」中说明），或删除该行。｜修复：｜复验：
- [重要·技术] 位置：核心问题第 5 条答案（第 135 行）、6 章正文（第 556 行）、来源章节 N 段（第 619 行）——三处「权重 dtype 为 bfloat16」｜来源：config.json｜问题：config.json 的量化配置表明 K3 的专家权重是 MXFP4 4-bit；「权重 dtype 为 bfloat16」把 config 的 `dtype`（计算 dtype）当成了权重存储 dtype，与官方材料不符。｜引文依据：config.json 顶层 `"dtype": "bfloat16"`，同时 `text_config.quantization_config` = `{"quant_method":"compressed-tensors","format":"mxfp4-pack-quantized",…,"num_bits":4}`（即权重 4-bit）。｜修复要求：改为「计算 dtype 为 bfloat16，专家权重为 MXFP4 4-bit 量化（config.json）」，或删去关于权重 dtype 的表述。｜修复：｜复验：
- [轻微·技术] 位置：3.2 节首段（第 382 行）与 3 章本章问题（第 411 行）「auxiliary-loss-free，DeepSeek-V3 引入」｜来源：K3 报告 §2.3.3｜问题：来源列表 C12 把「无辅助损失路由由 DeepSeek-V3 引入」归给 K3 §2.3.3，但该节只以引文编号标注该路由，未点名 DeepSeek-V3，归因得不到所引位置支持。｜引文依据：K3 §2.3.3 原文「Unlike auxiliary-loss-based routing [32], Kimi K3 adopts auxiliary-loss-free routing [27]」——未出现 DeepSeek-V3；报告提及 DeepSeek 仅用于 DeepSeekMoE（shared/routed）与 DeepSeek-V2（MLA）。｜修复要求：把引入来源直接引到 DeepSeek-V3（arXiv:2412.19437），或删去「DeepSeek-V3 引入」的归因。｜修复：｜复验：
- [轻微·表述] 位置：各章末过渡段（1→2 第 163 行、2→3 第 303 行、3→4 第 396 行、4→5 第 455 行、5→6 第 527 行）｜来源：不适用｜问题：五处过渡采用同一句式「本章……但……尚未（摆清/展开/论证/界定），下一章……」，属 style-guide 第 8 节禁止的固定句式；4 章过渡（第 455 行）与 5 章内引（第 285 行）的「会单独说清为什么必须在这里」另有多余重复措辞。｜引文依据：不适用｜修复要求：改写其中至少部分过渡以免固定句式，并删除重复措辞。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 2
- 处置：修复

说明：本轮未发现核心结论性错误——Eq. 11/12/14、稀疏度 56（$N/k=896/16$）、$\beta_1\beta_2=100$、$d/\ell$ 与各 config 字段、DeepSeekMoE §3.2 归因、构造示例 RMS 手算（$u_A$ RMS=1、$u_B$ RMS=10、归一化后同为 $(1,1)$）均与来源一致；`validate.py` 返回成功，overview.html 与 index.html 互相链接，五个前置概念页均真实存在，正文引文编号 [C1–C13、F1/F3/F4、N1–N8] 与来源列表双向对应，无 Unicode 数学字符、无「（待生成）」占位。阻断判定为 0 依据：所发现的均为局部来源标注/渲染缺陷，不推翻页面主结论。
