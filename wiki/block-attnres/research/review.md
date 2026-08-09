# Block AttnRes 独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html `048e784e`、overview.html `3a260e5a`
- 时间：2026-08-09

## 段 A 盲读小结

按页面顺序（overview → index）阅读，主线理解畅通。S1→S2→S3→S4→S5 章节递进合理：瓶颈动机 → Full 公式 → Block 分块 → K3 实例化 → RMSNorm 设计选择。贯穿小例子（N=3、S=2、d=2）在 S2/S3/S5 三次复用，数字逐位可手算，是理解公式的有效抓手。折叠块（手算细节、补充推导、伪代码）收起后正文主线仍成立，不构成依赖。

学习目标逐题核对：5 个学习目标（标准残差瓶颈/AttnRes 思路、Full AttnRes 公式与 pseudo-query/keys/values/softmax kernel、Block 分块与内存 O(Ld)→O(Nd)、K3 具体配置、RMSNorm 作用与边界）均由正文章节完整回答，无遗漏。

## 段 B 对照来源小结

来源核对项：
- K3 报告 §2.2（377-419 行）：C1-C7、F1-F4 引用行号逐条核对准确，核心论断（RNN-over-depth 瓶颈、Eq.8-10 公式、RMSNorm 防大值主导、O(Ld)→O(Nd) 内存、N≈8 经验结论、末尾聚合）与报告原文一致，未发现扩大来源结论。
- K3 报告 §2（196-198、209-210 行）：已核对，209 行原文 "enable each module to selectively retrieve representations" 确认 each module 有检索能力。
- HuggingFace config.json：attn_res_block_size=12、num_hidden_layers=93、hidden_size=7168、num_attention_heads=96、full_attn_layers 24 个、kda_layers 69 个，全部与页面 S4 表格一致。

核心公式复算（Python 验证）：
- S2 Full AttnRes 6 候选（不加 RMSNorm）：内积 [0.5,0.5,1.0,0.5,0.75,0.75]，权重 [0.1386,0.1386,0.2285,0.1386,0.1779,0.1779]，h6=[0.7032,0.7032] ——与页面一致。
- S3 Block AttnRes 4 候选（不加 RMSNorm）：内积 [0.5,1.5,1.25,0.75]，权重 [0.1405,0.3818,0.2974,0.1804]，h6=[1.0585,1.2414] ——与页面一致。
- S5 加 RMSNorm 4 候选：RMSNorm 后各向量均方根均为 1.0（验证通过），内积 [0.7071,0.9487,0.9806,0.9487]，权重 [0.2057,0.2619,0.2704,0.2619]，h6=[1.0042,1.0562] ——与页面一致（末位四舍五入差异可接受）。

8 块×12 层=93 算术核对：7×12+9=93 ✓。最后一个 block 为 9 层（partial），页面标注为"由 93=7×12+9 推算"，与 K3 报告 "partial final block" 一致。

加权三次核对：K3 报告 §2 209 行确认 "each module" 有检索能力 + §2.2 414 行确认末尾聚合；具体"attention 前/MLP 前/final norm 前"三次位置与参数字段名标注为 dataflow note 源码核对的间接证据。证据层级标注诚实。

softmax kernel 核对：φ(q,k)=exp(q^T RMSNorm(k))，与报告 §2.2 390-391 行一致 ✓。

9 个候选来源核对：对最后一个 block 的 i≥2 层，候选=[b_0, b_1, ..., b_7, b_8^{i-1}] = 8 块快照（含 embedding）+ 当前 partial sum = 9 个，与页面 S4 公式一致 ✓。

## 问题

- [重要·技术] index.html S4 第 1014 行"K3 实现中预分配 9 个槽位以适配最大候选数，未填满的槽位在 attention 中按零处理或屏蔽"：此实现细节无来源标注，K3 报告 §2.2 未提及"预分配槽位"或"零处理/屏蔽"，C8/C9/N1 来源说明也未覆盖。当前表述让读者误以为这是来源确认的事实。若来自 dataflow note 源码核对，应标注为间接证据（与 C8 同级标注）；若是基于"最大候选数为 9"的合理推断，应标注为教学构造/推断。：在"预分配 9 个槽位"句末加来源标注——若为源码核对写"（来自 dataflow note 对源码的核对，间接证据）"，若为推断写"（基于最大候选数为 9 的实现推断，未核对源码）"。 ｜ 修复：已在该句末尾加注"（基于最大候选数为 9 的实现推断，未核对源码）"，将其标注为教学推断而非来源确认的事实。 ｜ 复验：
- [轻微·盲读] overview.html 第 62 行"候选 = 8 个块快照 + 当前 partial sum = 9 个"："8 个块快照"未说明包含 embedding（b_0）。K3 有 8 个 block，读者可能困惑：若 8 个 block 都有快照则加当前流应为 9，但最后一个 block 是 partial 未完成、尚无完整快照。实际"8 个块快照"= b_0（embedding）+ b_1..b_7（7 个完整 block），index.html S4 有详细解释但 overview 未澄清。：overview 该句改为"候选 = embedding + 7 个已完成块快照 + 当前 partial sum = 9 个"，或在"8 个块快照"后加括注"（含 embedding）"。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html S4 第 999 行"加上 embedding 共 9 个 block 级表征"与第 1007 行"9 个候选来源"用了同一个数字 9，但两者是不同集合：前者（9 个 block 级表征 = b_0..b_8）含 block 8 完整求和，是模型末尾视角；后者（9 个候选 = b_0..b_7 + b_8^{i-1}）含 partial sum，是最后一个 block 内 i≥2 层视角。页面有公式区分，但文字表述中两个"9"未明确区分，可能让读者混淆。：在第 999 行"9 个 block 级表征"后加一句"注意：这是模型末尾所有 block 完成后的视角；最后一个 block 内 i≥2 层的候选用 partial sum b_8^{i-1} 替代完整 b_8，也是 9 个但集合不同"。 ｜ 修复： ｜ 复验：
- [轻微·技术] index.html C8（第 1192 行）"K3 报告原文确认：每个 module（attention 模块与 MLP 模块）各加权一次（§2 第 209 行）"：报告 209 行原文为"enable each module to selectively retrieve representations"，确认了 each module 有检索能力，但未明确说"各加权一次"也未点名"attention 模块与 MLP 模块"。页面把"enable each module to retrieve"解读为"各加权一次"是合理推断但略有扩大。具体三次位置已标注为间接证据，影响可控。：C8 该句改为"K3 报告原文确认 each module 有 AttnRes 检索能力（§2 第 209 行）；具体'attention 模块与 MLP 模块各加权一次'的解读来自架构描述（§2 第 211 行 each attention layer is followed by a Stable LatentMoE layer），三次位置来自源码核对（间接证据）"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：进入修复
