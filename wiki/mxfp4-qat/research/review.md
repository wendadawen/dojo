# MXFP4 量化感知训练独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 强推理模型对照来源）
- 页面版本：index.html 工作树哈希 `1f852b65f14c05ffa96e2c4a9e6ddd2826fba21b`；overview.html 工作树哈希 `bb9dbff3472e77c133c2ab38a5a853dcfe40c578`
- 时间：2026-08-09 15:39 CST

审查依据：`guides/concept/check.md`。段 A 按页面顺序盲读记录卡点；段 B 逐条对照 K3 报告 `/tmp/kimi-k3-research/k3-report.txt` §4.1.4（第 922–954 行）与官方 `config.json`（WebFetch 于 2026-08-09 获取）。代码已实际执行，输出与页面预期一致。validate.py 两页退出码均为 0。

## 问题

- [阻断·技术] index.html 开头 callout（第 654 行）"把一个 670 亿参数级的 MoE 大模型从 BF16 换成 4-bit 部署"：K3 报告第 99–100 行明确写 "2.8 trillion total parameters, 104 billion activated parameters"。670 亿 = 67B 与 2.8T 相差约 42×，来源无任何 67B/670 亿 的依据。该数字作为全文 hook 的"模型规模"前提，误导读者对 K3 体量的基本认知。修法：将"670 亿参数级"改为"2.8 万亿（2.8T）参数级"，或在 callout 中直接引用报告原话"2.8 trillion total parameters, 104 billion activated"。 ｜ 修复：callout "670 亿参数级"改为"2.8 万亿（2.8T）参数级"，"约 118 GB"改为"约 5.4 TB"，"约 31 GB"改为"约 1.44 TB" ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [阻断·技术] index.html S1（第 690 行）及 details"完整手算"（第 696–705 行）：专家总参数量 896 × 66.06M = 59.19B 错误。config.json 的 `num_experts: 896` 是**每层**专家数（`moe_layer_freq: 1`、`num_hidden_layers: 93`、`first_k_dense_replace: 1` → 92 个 MoE 层），且 config 另有 `routed_expert_hidden_size: 3584`（页面从未提及，误用 `hidden_size: 7168` 计算单专家参数量）。正确估算：896 × 92 × (3 × 3584 × 3072) ≈ 2.71T，与报告 2.8T 总参一致。页面 59B 比实际小约 47×。该数字同时与报告第 923–924 行"MoE expert weights — which dominate the model's parameter memory"矛盾：若专家仅 59B/2.8T ≈ 2%，不构成"dominate"。修法：(a) 明确 896 为每层专家数、MoE 层数为 92；(b) 单专家参数量改用 `routed_expert_hidden_size: 3584`，即 3 × 3584 × 3072 ≈ 33.03M；(c) 总专家参数量重算为 ≈ 2.71T，并在正文交代 `routed_expert_hidden_size` 字段的来源。 ｜ 修复：S1 正文改用 routed_expert_hidden_size=3584（非 hidden_size=7168）；明确 896 为每层专家数、92 个 MoE 层（first_k_dense_replace=1）；单专家 3×3584×3072≈33.03M，总专家 896×92×33.03M≈2.71T；details 手算同步重算；ignore 列表正则表述同步修正（见重要2） ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [阻断·技术] index.html S1（第 692、700–704、707 行）与 overview.html（第 52–53 行）：基于上条错误专家数推导的显存数字全部错误。BF16 "约 118.38 GB" 应为 ≈ 5.4 TB；MXFP4 "约 31.44 GB" 应为 ≈ 1.44 TB；"省下来的显存是几十 GB 的量级"应为"约 4 TB 量级"。overview.html "约 59B 参数、BF16 下约 118 GB / 约 31 GB" 同样错误。压缩比 3.76× 本身正确（与绝对规模无关），可保留。学习目标 1"量化到 MXFP4 能省多少显存"由本节数字回答，数字错误使该目标闭环失败。修法：按 B2 修正后的 2.71T 重算 BF16/MXFP4 显存，更新 index.html S1 正文与 details 手算、overview.html 第 52–53 行与第 50 节"为什么需要它"对应条目。 ｜ 修复：BF16 118.38 GB→5.4 TB；MXFP4 31.44 GB→1.44 TB；"几十 GB"→"约 4 TB"；details 手算全部重算（元素 1.355 TB + scale 84.7 GB）；overview.html 第 52-53 行同步修正（59B→2.71T、118 GB→5.4 TB、31 GB→1.44 TB、几十 GB→约 4 TB）；学习目标自测题同步更新 ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [阻断·技术] index.html S1 第 707 行"非专家组件（attention 投影、共享专家、router、lm_head）参数量远小于 59.19B，而且每个 token 都要计算，既不是显存瓶颈，也不是量化首选"：比较基础错误。按正确专家参数量 ≈ 2.71T，非专家约 50B，则非专家确实远小于专家——但对比对象应是"专家参数（2.71T）"而非页面错误值"59.19B"。结论"非专家不是显存瓶颈"本身正确，但论证链基于错误数字。修法：将"远小于 59.19B"改为"远小于专家参数总量（约 2.71T）"，或直接引用报告"expert weights dominate the model's parameter memory"作为依据。 ｜ 修复："远小于 59.19B"改为"远小于专家参数总量（约 2.71T）" ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [重要·盲读] index.html S5 选择表（第 955 行）"latent MoE 投影"未解释。该术语在 K3 报告第 925 行（"latent MoE projections"）与 config（`latent_moe_use_norm: true`）出现，页面正文从未说明什么是 latent MoE、它与路由专家的关系，读者无法理解该组件"不量化"的工程理由。修法：在 S5 表格前或表后加一句说明 latent MoE 指什么（如"MoE 层中除路由专家外的公共投影部分"），或引用报告原文定位；若本页范围内无法简要解释，标注"见 moe-serving 页"并加占位提示。 ｜ 修复：S5 表前段落加一句说明"latent MoE 投影指 MoE 层中除路由专家 per-expert 权重外的公共投影部分（如 MoE 层输入/输出的公共线性变换），与路由专家的 per-expert 权重相对" ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [重要·盲读] index.html S1 第 690 行"每个路由专家本身是一个 3 投影 FFN（gate_proj、up_proj、down_proj，与 config.json 量化忽略列表里的 `mlp.(gate|up|gate_up|down)_proj` 一致）"措辞误导。ignore 列表用正则匹配完整路径，路由专家路径形如 `model.layers.N.mlp.experts.M.gate_proj`，含 `.experts.M.` 中间段，**不匹配** `.*mlp\.(gate|up|gate_up|down)_proj.*`。页面后文（第 999 行）正确指出专家不在 ignore 列表，但此处"一致"易让读者误以为专家也被忽略，与正文结论自相矛盾。修法：将"与 ignore 列表里的 ... 一致"改为"投影名与 ignore 列表中的 dense FFN 投影同名，但完整路径含 `.experts.` 段故不匹配该正则，因此专家是被量化的对象"。 ｜ 修复：S1 第 690 行"与 config.json 量化忽略列表里的 mlp.(gate|up|gate_up|down)_proj 一致"改为"投影名与 ignore 列表中的 dense FFN 投影同名，但完整路径含 .experts. 段故不匹配该正则，因此专家是被量化的对象"（随阻断2一起修） ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [重要·技术] index.html S5 第 999 行"`mlp.(gate|up|gate_up|down)_proj`（顶层 dense FFN 投影）"表述不精确。config `first_k_dense_replace: 1` 表示前 1 层为 dense FFN（非 MoE），其投影路径 `model.layers.0.mlp.gate_proj` 等匹配该正则。"顶层"在中文语境多指"最上层/全局"，易误解为模型顶层结构。修法：将"顶层 dense FFN 投影"改为"前 1 层 dense FFN 的投影（`first_k_dense_replace: 1`）"。 ｜ 修复：S5 第 999 行"顶层 dense FFN 投影"改为"前 1 层 dense FFN 的投影（first_k_dense_replace = 1）" ｜ 复验：validate.py 退出码 0（index.html + overview.html 均通过）
- [轻微·盲读] index.html S2 第 722 行"正数正常可表示值只有 8 个"中"正常"易与浮点"normal vs subnormal"技术含义混淆。E2M1 列出的 0.5、0.75 实际是 subnormal（指数位 00），1.0–6.0 才是 normal。修法：将"正数正常可表示值"改为"正数可表示值"。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html 可运行代码第 815 行 `E2M1_LEVELS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]` 含 9 个值（含 0.0），与正文第 722 行"正数可表示值只有 8 个"在数量上不直接对应。不影响运行（代码输出与预期一致），但读者对照代码与正文计数时可能困惑。修法：在代码注释中说明"含 0 共 9 个，正数 8 个"，或保持代码不变、正文补一句"另加零值共 9 个可表示点"。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S4 第 933 行"K3 用 EAGLE-3 风格的 draft model 做投机解码加速"首次出现"投机解码""draft model"未作任何解释即进入"draft model 沿用同一 QAT 配置"的论证。虽标注"draft 的具体结构不是本页主题，不展开"，但读者连"什么是投机解码"都不知，难以理解为何 draft 与 target 量化行为需要对齐。修法：加一句最小说明（如"投机解码：用一个小的 draft 模型先猜若干 token，target 模型并行验证，命中即接受以加速推理"），或链接到相关概念页并标注占位。 ｜ 修复： ｜ 复验：
- [轻微·盲读] index.html S2 第 745 行"设块共享 scale 已确定为 s_b = 0.25"未在正文说明 0.25 的选取依据；details（第 745 行后）内有"真实实现中由该块 32 个权重的最大幅度按 OCP 规范的 power-of-two 规则选定"，但正文读者未展开 details 时不清楚 0.25 从何而来。修法：在正文该句后补半句"（教学给定；真实由块内最大幅度按 OCP power-of-two 规则选，见下 details）"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 4 / 重要 3 / 轻微 4
- 处置：进入修复

阻断问题集中在 S1 的量化数字链（模型总参 670 亿 → 专家 59.19B → 显存 118/31 GB → "省几十 GB"）与来源 K3 报告"2.8T 总参 + 专家 dominate"直接冲突，根因是把 `num_experts: 896` 误当作全模型专家总数（实为每层数）并漏用 `routed_expert_hidden_size: 3584`。S2/S3/S4 的机制、公式、代码、QAT/STE/RL mismatch 论证经核对均与来源一致且代码实跑通过，无需改动；S5 的 config.json 字段引用准确，仅"latent MoE 投影"未解释与 ignore 正则表述需修正。修复后建议重新对照 config.json 复核全部数字，并重跑 validate.py。

---

审查范围声明：仅读取 `index.html`、`overview.html`、K3 报告 §4.1.4、官方 `config.json`，未读取 `research/` 目录与其他页面，未修改两份文档。
