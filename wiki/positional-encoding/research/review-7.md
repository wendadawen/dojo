<!-- review-meta
round: 7
page: wiki/positional-encoding/index.html
reviewed_content_sha256: 9c8d966eac2db3fc
-->
# 位置编码基础审查记录（第 7 轮）

- 页面版本：b4e4e09f52d8d36b9e427f3c897750643416307b
- 审查时间：2026-09-13 21:54
- 审查者：独立子代理
- 已完整阅读章节：核心问题、常见误解、引言（含 meta 依据块）、1. 为什么 Transformer 需要位置编码——注意力的排列等变性、2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算、3. 可学习绝对位置编码——把固定向量换成可学习参数、4. 相对位置编码——T5 的 bias 机制与"加在分数上"的本质区别、5. 四类方案对比与 NoPE 选择——为什么 K3 选 NoPE（含 5.1 K3 为什么选 NoPE）、来源与范围说明；全部折叠块（d=4 手算展开、和角公式推导、Table 3 row (e) 数字、T5 分桶方向性、类比边界、简化条件）与图注均已逐段阅读。

## 本轮核对来源（原文片段／关键数值）

- Vaswani et al. 2017, arXiv:1706.03762v7（PDF 全文，pdftotext 提取）：§3.5 原文 "P E(pos,2i) = sin(pos/100002i/dmodel )"、"P E(pos,2i+1) = cos(pos/100002i/dmodel )"、"we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks"、"The wavelengths form a geometric progression from 2π to 10000 · 2π"、"for any fixed offset k, P Epos+k can be represented as a linear function of P Epos"、"the two versions produced nearly identical results (see Table 3 row (E))"、"We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training"；Table 3 base 行 dev BLEU 25.8、row (E) "positional embedding instead of sinusoids" dev BLEU 25.7（dev newstest2013）；Table 2 big EN-DE 28.4（test newstest2014）；d_model=512。全文右对齐编号仅 (1)(2)(3)，其中 "(3) lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5 )" 属 §5.3；grep "(4)" 全文 0 命中。
- Raffel et al. 2019, arXiv:1910.10683v4 §2.1：原文 "we also share the position embedding parameters across all layers in our model"、"we use 32 embeddings for all of our models with ranges that increase in size logarithmically"、"beyond which we assign all relative positions to the same embedding"、"a given layer is insensitive to relative position beyond 128 tokens"。
- T5 源码 `_relative_position_bucket`（transformers v4.44.0 modeling_t5.py L381）：签名 `def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128)`；`max_exact = num_buckets // 2`（bidirectional 时 num_buckets 折半为 16，故 max_exact=8，`is_small = relative_position < max_exact`），远距离按 `log` 分桶后 `torch.min(..., num_buckets - 1)` 截断。
- Kimi K3 技术报告 arXiv:2607.24653v2 §2.1.2/§3.4（官方 config.json）：§2.1.2 "applies No Position Encoding (NoPE) to all MLA layers"、"The intervening KDA layers provide position-sensitive and recency-aware sequence mixing"、"the MLA layers provide unrestricted global content interaction"、"avoids modifying positional-encoding parameters when extending the context length"（"such as retuning a RoPE frequency base or applying YaRN"）；§3.4 "encodes positional information implicitly through the recurrent gating and decay mechanism of KDA"、"extrapolates directly to 1M-token contexts"；config.json（moonshotai/Kimi-K3）`"mla_use_nope": true`、`"qk_rope_head_dim": 64`、`"max_position_embeddings": 1048576`。
- RoFormer arXiv:2104.09864 §3.1/§3.2.2：Eq.(11) ⟨f_q(x_m,m), f_k(x_n,n)⟩ = g(x_m,x_n,m−n)；Eq.(16) q_mᵀ k_n = xᵀ W_q R^d_{Θ,n−m} W_k x_n。
- ALiBi arXiv:2108.12409 §3：softmax(q_i Kᵀ + m·[−(i−1),…,−2,−1,0])，"scalar m is a head-specific slope fixed before training"。
- 机械核对：KaTeX 实渲染 L175/L181/L240/L241/L288/L359 等公式与 `\text{... § ...}`、`\text{（可学习）}` 全部 OK（node 载入 libs/katex.min.js 实测，仅 § 触发 strict-warn 仍可渲染）；`.dojo/scripts/validate.py` 返回 validation ok；`dojo:topics=注意力机制` 在词表内、`dojo:tag=位置编码` 在 ALLOWED_TAGS 内；libs/（katex、prism、dojo-concept.css）与 /wiki/{rope,nope,kimi-k3,mla,standard-attention}/index.html 均存在，overview.html 与 index.html 互链；无 img alt 含 `$...$`；表格中 "桶数×头数" 的 `×` 属 validate.py 明确排除的排版字符（validate.py L53）。
- 手算复算：ω_1=1/10000^{0.5}=0.01；PE_1≈(0.8415,0.5403,0.0100,0.99995)、PE_2≈(0.9093,−0.4161,0.0200,0.99980)；2 sin1 cos1 = sin2 ≈ 0.9093；λ_0=2π、λ_255≈2π·9650（论文口径 10000·2π）；512×512=262144≈26 万。

## 问题

- [阻断·技术] head 主要依据块（L62）、§2 公式块（L181）、来源章节 C2（L498）、F1（L511）：把正弦位置编码公式的来源标注为 Vaswani 2017 "Eq.(3)(4)"（F1/C2 写作 "Eq.(3) 与 Eq.(4)"），但该论文 §3.5 的两个正弦公式并未编号；论文全文只有编号公式 (1)(2)(3)，其中的 (3) 是 §5.3 的学习率调度公式，全文不存在 Eq.(4)。引文编号与实际内容错位：按标注去查 Eq.(3)(4) 会落到与位置编码无关的公式上。｜引文依据：arXiv:1706.03762v7 全文右对齐编号仅 "Attention(Q, K, V ) = softmax( √ )V (1)"、"FFN(x) = max(0, xW1 + b1 )W2 + b2 (2)"、"lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5 ) (3)"；§3.5 的 "P E(pos,2i) = sin(...)" / "P E(pos,2i+1) = cos(...)" 两式无编号；全文 grep "(4)" 命中 0 次。｜修复要求：四处（L62、L181、C2、F1）删除 "Eq.(3)(4)"／"Eq.(3) 与 Eq.(4)"，改标为 "§3.5 正弦公式"（或 "§3.5 第 2 段两式"），不得保留不存在的公式编号。｜修复：｜复验：

- [轻微·技术] §5 对比表"与推理优化兼容性"单元格（L447）与核心问题第 5 题答案（L102）："RoPE ... 与 MLA 矩阵吸收冲突"是机制论断，页面对该论断无来源标注——C9 只覆盖"绝对构造、相对效果"（Su et al. 2021 §3.1），此处仅是指向 /wiki/mla/ 的跨页引用，页面内没有可定位的外部来源。｜引文依据：Su et al. 2021 §3.2.2 只给出 q_mᵀ k_n = xᵀ W_q R^d_{Θ,n−m} W_k x_n（相对性），未涉 MLA 吸收；K3 报告 §2.1.2 只提 "avoids modifying positional-encoding parameters... such as retuning a RoPE frequency base or applying YaRN"，未提矩阵吸收冲突。｜修复要求：为该单元格补一条可定位来源（MLA 原始论文/MLA 页所引原文，如 DeepSeek-V2 关于 RoPE 阻断 KV 吸收的章节），或把该判断降级为明确标注的推断（如"RoPE 的 per-token 旋转使 MLA 的位置无关吸收矩阵无法合并"并标注为推断）。｜修复：｜复验：

- [轻微·技术] 来源章节 C10（L506）：C10 以"§2.1.2 原文"引出转述，其中把原文 "unrestricted global content interaction" 写作"不受位置约束的全局内容交互"，"不受位置约束"是原文所无的限定性改写，却标注为原文。｜引文依据：K3 报告 §2.1.2 "the MLA layers provide unrestricted global content interaction"（对应前句 "The intervening KDA layers provide position-sensitive and recency-aware sequence mixing" 的中译无偏差）。｜修复要求：把 C10 的"§2.1.2 原文"改为"§2.1.2 原文（中译）"，或按原文直译为"无限制的全局内容交互"。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复（阻断项 Eq.(3)(4) 编号错位须先删除错误编号后再发布；两项轻微问题建议同轮一并处理）