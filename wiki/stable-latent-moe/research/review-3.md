<!-- review-meta
round: 3
page: wiki/stable-latent-moe/index.html
reviewed_content_sha256: a2916fb7001cf135
-->
# Stable LatentMoE 审查记录（第 3 轮）

- 页面版本：3e1339d404427e557f2cec92079c496aea731068
- 审查时间：2026-09-13 19:11
- 审查者：独立子代理（未参与写作与前序轮次）
- 适用规范：guides/concept/check.md（dojo:type=concept）
- 已完整阅读章节：核心问题 / 1. 扩大专家池的代价——为什么需要 LatentMoE / 2. Stable LatentMoE 的层结构——共享分支、路由分支与 RMSNorm 的位置 / 3. 极端稀疏下的两个失败模式——激活爆炸与负载失衡（3.1、3.2）/ 4. 三件稳定化——Normalized LatentMoE、SiTU-GLU、Quantile Balancing（4.1–4.4）/ 5. 为什么 RMSNorm 必须插在路由聚合之后——Normalized LatentMoE 的位置必要性（5.1–5.4）/ 6. 适用边界与不能推出的结论 / 来源与范围说明
- 来源核对方式：K3 技术报告 arXiv:2607.24653v2 §2.3–2.3.3 原文；官方 `config.json`、`modeling_kimi_linear.py`、官方权重 safetensors 头；DeepSeekMoE arXiv:2401.06066 §3.2（ar5iv 全文）。`overview.html` 与 `index.html` 相互链接。
- 机械验证：`.dojo/scripts/validate.py wiki/stable-latent-moe/index.html` → validation ok。

## 问题

- [阻断·技术] §2 符号表（第 221 行）：把 `intermediate_size=33792` 标注为「共享专家 FFN 中间维度」，与官方材料不符。K3 官方建模代码用 `moe_intermediate_size × num_shared_experts` 构造共享专家，官方权重中共享专家 FFN 中间维度为 3072×2=6144；33792 是 dense 层（`first_k_dense_replace=1` 的第 0 层）FFN 的中间维度。｜引文依据：`modeling_kimi_linear.py:797-800`「if config.num_shared_experts is not None: intermediate_size = config.moe_intermediate_size * config.num_shared_experts; self.shared_experts = KimiMLP(config=config, intermediate_size=intermediate_size,)」；官方 safetensors 头 `language_model.model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight [6144, 7168]`、`...shared_experts.down_proj.weight [7168, 6144]`，而 `language_model.model.layers.0.mlp.gate_proj.weight [33792, 7168]`；`config.json`：`intermediate_size=33792`、`moe_intermediate_size=3072`、`num_shared_experts=2`。报告 §2.3–2.3.3 全文未出现 33792。｜修复要求：把该行 K3 取值改为 6144，`config 字段` 一栏改为「`moe_intermediate_size` × `num_shared_experts`」（或注明来自官方权重 `shared_experts.*_proj` 形状），或直接删除该行；若保留 `intermediate_size=33792`，必须改写其含义为「dense 层（第 0 层）FFN 中间维度」，不得再作为共享专家维度。｜修复：｜复验：

- [重要·技术] §5.4「换位置会怎样」（第 491–498 行）：三条「位置不可换」的机制论断在「来源与范围说明」中没有任何对应来源，也未标注为本文推断；其中「放在 $W_\downarrow$ 之前」的论据只在「在层输入 $x$ 上做归一化」这一读法下成立——若只在路由支内、$W_\downarrow$ 之前归一化，共享支输入不受影响，该论据不适用；而真正的理由（尺度漂移发生在专家聚合之后，归一化路由输入无法修复 $u$ 的尺度）未给出。｜引文依据：K3 报告 §2.3.1 只说明「The original LatentMoE directly applies $\mathbf{W}^{\uparrow}$ to the aggregated routed representation $\bm{u}$ … Kimi K3 instead inserts RMSNorm between expert aggregation and the up-projection.」——报告全文无换位置分析；`modeling_kimi_linear.py:831-838` 也只在聚合之后、上投影之前插入 `routed_expert_norm`。｜修复要求：在「来源与范围说明」下新增条目，明确 §5.4 三条论据为本文基于 Eq. 11 与 RMSNorm 定义的推导（非 K3 报告结论）；并补全「放在 $W_\downarrow$ 之前」的讨论，说明为何「仅在路由支内归一化 $x$」也不能消除 $u$ 聚合后的尺度漂移（漂移来自 $T_k(x)$ 与 $p_i$，发生在专家聚合之后）。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」（第 600、603 行）：来源编号 C4、C5、C6、F2 被定义但正文从未以 `<sup>[...]</sup>` 引用，与 style-guide §6 要求的「正文引用与来源章节双向对应」不符。｜引文依据：全文正文出现的编号为 C1、C2、C3、C7、C8、C9、C10、C11、C12、C13、F1、F3、F4、N1–N8；C4/C5/C6/F2 仅出现在来源说明中。｜修复要求：删除正文未使用的 C4、C5、C6、F2（把 C3 的定义单独改写、F2 并入 F1 的说明），或把它们对应的具体论断补进正文并加引用。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明」（第 600–606 行）：C/F/N 条目给出的「K3 技术报告 第 423-428 行 / 429-433 行 / 461-485 行 / 467-474 行 / 499-502 行 / 541 行 / 547-596 行 / 586-589 行」为纯行号定位，页面未提供对应的可下载来源（无 research/official/ 或 measured.md 登记），无法定位核对。｜引文依据：K3 报告在线版（arXiv:2607.24653v2）按 §/Eq. 定位，正文内容与 §2.3、Eq. 11、Eq. 12、Eq. 14 逐条一致，但行号无法对应。｜修复要求：把行号定位改为可核对的 §/Eq./Fig. 定位（如「§2.3 Eq. 11」「§2.3.2 Eq. 12 与 Fig. 4」「§2.3.3 Eq. 13-14」），或保留行号的同时补上其对应的具体文本片段出处说明。｜修复：｜复验：

- [轻微·表述] 第 98、285、453 行：存在元话语与面向读者的指令式表述。｜引文依据：「本文说明这套结构如何构造、为何如此设计。」（第 98 行）；「注意几个容易看错的地方。第一，…」（第 285 行）；「注意三件稳定化是各管一段、不可互相替代：…」（第 453 行）。｜修复要求：删去第 98 行「本文说明…」这句纯宣示（其内容已由引言上一句覆盖）；把两处「注意…」改写为不面向读者的陈述句（如「容易看错的有三点：…」「三件稳定化各管一段、不可互相替代：…」）。｜修复：｜复验：

- [轻微·格式] 第 6 行 `description`、第 360 行 SVG `dg-label`、第 277 行 SVG `<text>`：出现未包在 `$...$` 中的 Unicode/ASCII 数学符号。｜引文依据：第 6 行「稀疏度 56×」的 `×`；第 360 行「路由 FFN：门支 × 值支」的 `×`；第 277 行 `<text … class="dg-step">+</text>` 的 `+`（与 style-guide §11「任何位置出现的数学运算符必须包在 `$...$`」「`<text>` 只用于不含数学含义的纯文字」不符）。｜修复要求：第 6 行改为「稀疏度 56 倍」或「$56\times$」；第 360 行 `dg-label` 改为「路由 FFN：门支与值支」；第 277 行求和节点的 `+` 改写进 `<foreignObject>` 并用 `$+$` 渲染，或换用非数学的图形表示。｜修复：｜复验：

- [轻微·技术] §4.3（第 434 行）：引入 $n$（「$n$ 是专家数」）但未说明它与 §2 符号表中 $N$（路由专家总数 896）的关系；同一页两个近形符号都指专家数，易混。｜引文依据：第 222 行「$N$｜路由专家总数｜896」；第 434 行「$q = mk/n$（$m$ 是 batch 内 token 数、$n$ 是专家数）」；K3 报告 §2.3.3「routed to $n$ experts with Top-$k$ selection」。｜修复要求：在第 434 行 $n$ 的首次出现处补一句「$n$ 即本页 §2 的路由专家总数 $N$（K3 取 896）」，使两符号关系明确。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复
- 说明：Eq. 11（$u=\sum_{i\in T_k(x)}p_iE_i^{\mathrm{routed}}(W_\downarrow x)$、$y=\sum_j E_j^{\mathrm{shared}}(x)+W^\uparrow\mathrm{RMSNorm}(u)$）、Eq. 12（SiTU-GLU 及 $\beta_1\beta_2=4\times25=100$）、Eq. 14（QB 更新第二行为减均值）三处公式与报告逐字一致；config.json 的 `hidden_size=7168`、`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`latent_moe_use_norm=true`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0` 全部核对通过；稀疏度 $896/16=56$、$\ell=2$ 构造示例的 RMS 与归一化结果（1、10、$(1,1)$）复算无误；「无辅助损失路由 = DeepSeek-V3 引入」对应报告文献 [27]（DeepSeek-V3, 2412.19437），「shared+routed 组织」对应 [22]（DeepSeekMoE, 2401.06066 §3.2「Shared Expert Isolation」）；概念链接 moe-serving / situ-glu / quantile-balancing / deepseek-moe / latent-moe 均存在，无 `research/` 失效路径，无「（待生成）」占位。阻断与重要问题关闭前不得发布。
