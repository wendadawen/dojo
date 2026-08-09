# MoonViT-V2 独立审查

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：index.html 工作树哈希 822b5ceff81e6182697143c568f5bfd3fa01927a（目录 wiki/moonvit-v2/ 尚未提交，overview.html blob b5642d21d6571771781aecfff07995220b273c3c）
- 时间：2026-08-09

## 问题

- [重要·技术] index.html「来源与教学说明 > 外部数字与实验条件」N1（及 §1 对 MoonViT-3D 的描述）：N1 写“同一架构 MoonViT-3D 设计下 SigLIP 初始化 vs 从零训练对照”，把“同一架构”当作实验条件陈述。报告 §2.4 仅说 MoonViT-3D 用 SigLIP 初始化、MoonViT-V2 从零训练并在 Fig.6 消融中对照，未明确二者为同一架构；二者命名不同（3D vs V2），“同架构”是页面推断而非来源结论，且该推断会影响 Fig.6 作为稳定性证据的有效性判断（若架构不同，梯度差异可能来自架构而非初始化）。：删除“同一架构”断言，N1 改为只陈述报告明确给出的条件——MoonViT-3D 为 SigLIP 初始化的对照视觉编码器、MoonViT-V2 从零训练，Fig.6 在预训练消融中对照两者 vision-tower 梯度范数；若保留“同架构”说法需标注为推断并给出可定位依据。 ｜ 修复：已删除 N1（line 948）"同一架构 MoonViT-3D 设计下"断言，改为"SigLIP 初始化的对照视觉编码器 MoonViT-3D vs 从零训练的 MoonViT-V2 对照"，并补注"报告未明确二者为同一架构，'同架构'属页面推断，本页不采用"。§1（line 677）原本只描述 MoonViT-3D 为"报告里用作对照的视觉编码器变体"，无"同一架构"断言，无需修改。 ｜ 复验：validate.py 通过
- [轻微·盲读] index.html §5 边界二“主干是约 2.78T 参数的 MoE”及 overview.html「关键结论与边界」同句：“MoE”术语首次出现未展开、未给最小定义，也未标注“概念页待生成”。小白读者不知道 MoE 指什么，而该句的边界论证本只依赖规模数字。：给 MoE 一句话最小定义（如“混合专家（Mixture-of-Experts）”），或改为不带术语的“约 2.78T 参数的主干”，使边界只依赖规模数字本身。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html §5 边界二“约 2.78T 参数的 MoE”：2.78T 数字本身正确（报告参数表 Total Parameters 列为 2.78T、摘要“2.8-trillion-parameter Mixture-of-Experts”），但该数字不在 §2.4 内，而页面「主要依据」标注为 §2.4 + config.json，且「外部数字与实验条件」N 列表（现仅 N1、N2）未列入，缺少可定位来源。：在「外部数字与实验条件」补一条 N3，指明 2.78T 来源为报告参数表（Total Parameters 列）与摘要“2.8-trillion-parameter Mixture-of-Experts”。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：进入修复

### 段 B 对照来源核验汇总（均通过，未单列问题行）

- config.json vision_config 全字段逐一核对：vt_num_hidden_layers=27、vt_hidden_size=1024、qkv_hidden_size=1536、vt_intermediate_size=4096、vt_num_attention_heads=12、patch_size=14、norm_type=rmsnorm、attn_bias/linear_bias/patch_embed_proj_bias=false、merge_kernel_size=[2,2]、merge_type=sd2_tpool、pos_emb_type=divided_fixed、mm_projector_type=patchmergerv2，与 §3 配置表及正文引用完全一致。
- 报告 §2.4 引文 C1–C8 逐条核对，与报告原文一致，无扩大结论。
- F1 参数量手算：4×(1024×1536) + 2×(1024×4096) + 2×1024 = 14,682,112/层，×27 = 396,417,024 ≈ 0.40B，算术正确，与报告“roughly 0.4B”一致。
- F2 token 手算：3584/14=256 → 256²=65536 → ÷4=16384，算术正确，与报告“reduces by a factor of four…within the 1M-token context”一致；config.json max_position_embeddings=1048576 支持“1M 上下文”。
- 可运行代码块（§3 details 内）已提取并在 /tmp/moonvit_v2_review_run.py 实际执行，输出与页面「预期输出」逐行一致（attn/layer=6,291,456；mlp/layer=8,388,608；total 27L=396,417,024 ~0.40B；65536 tokens → 16384 tokens）。
- pixel-shuffle“无损重排”数学正确且在「教学解释与类比边界」明确标注降维步骤有损但不属于 pixel-shuffle 本身，类比边界清楚。
- validate.py 对 wiki/moonvit-v2 退出码为 0。
- index.html ↔ overview.html 互相链接有效；缺失前置概念（SigLIP、next-token prediction、ViT、RMSNorm、自注意力）均以“（概念页待生成）”占位。

### 审查范围说明

- 段 A 学习目标核对：因本次审查禁止读取 research/ 目录，无法取 scope.md 的原始学习目标逐题核对；仅核对了页面「读完你能回答」列出的 4 条目标，均由正文 §1–§5 章节完整回答。scope.md 与页面目标是否一致，需由编排者另补核对。
- 段 A 盲读未发现阻断级卡点：术语首现多带最小定义或“概念页待生成”占位；公式均有问题语境与逐步推导；章节切换有过渡句；折叠块（§3 参数量核对代码）收起后正文主线仍成立。
