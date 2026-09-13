<!-- review-meta
round: 5
page: wiki/positional-encoding/index.html
reviewed_content_sha256: e424b867459ddf1a
-->
# 位置编码基础审查记录（第 5 轮）

- 页面版本：index.html `ed59172d8b161d52c6170baf47a83a25cc6e6108`；overview.html `8b6e69d2522fbc55b9db17c541a963d1e86e9b9b`
- 审查时间：2026-09-13 20:22
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、常见误解、引言（含链接导语与 d=4 引例）、1. 为什么 Transformer 需要位置编码——注意力的排列等变性、2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算（含两处折叠块）、3. 可学习绝对位置编码——把固定向量换成可学习参数（含折叠块）、4. 相对位置编码——T5 的 bias 机制与"加在分数上"的本质区别（含折叠块与结构图）、5. 四类方案对比与 NoPE 选择——为什么 K3 选 NoPE（含 5.1）、来源与范围说明（含 C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）；并完整阅读 overview.html
- 外部核对：ar5iv 1706.03762（§3.1/§3.2.2/§3.5、Table 2、Table 3 row (E)、参考文献 [9]）、ar5iv 1910.10683（T5 §2.1）、HF T5 `_relative_position_bucket` 源码、ar5iv 2108.12409（ALiBi）、arxiv 2607.24653v2（K3 报告 §2.1.2/§3.4）、huggingface.co/moonshotai/Kimi-K3/config.json

## 问题

- [重要·技术] 来源与范围说明 §论断与来源（C）C1（index.html 第 497 行；正文第 117、125 行）：C1 把"排列等变论证"的来源标为「标准 Transformer 注意力 §复杂度与边界」，但该处没有排列等变论证，"§复杂度与边界"也不是该页的小节名。｜引文依据：wiki/standard-attention/index.html 第 498 行为「5. 复杂度、瓶颈与边界——标准注意力不能做什么」；该页全文 grep「排列 / 等变 / 置换」0 命中，第 95 行只把位置信息列为三个未解决问题之一（"位置信息（对输入行重排不变，需 RoPE 等外接编码）"），属主题相近而非该论证；排列等变论证实际在 wiki/nope/index.html 第 171 行「这种现象叫排列等变」，本页正文第 117、125 行也正指向 NoPE。｜修复要求：把 C1 的第二个指针由「标准 Transformer 注意力 §复杂度与边界」改为「NoPE」（或改为标准注意力第 5 章正确标题，并确认该章确有该论证）；正文不必改动。｜修复：｜复验：
- [重要·技术] overview.html 第 3 节「关键结论与边界」第 4 条（第 42 行）：称「K3 的 MLA 层选 NoPE 是因 RoPE 与矩阵吸收冲突」，把无来源支持的归因写成结论，且与本页 5.1 及 kimi-k3 页给出的原因不一致。｜引文依据：K3 报告 §2.1.2 原文「…applies No Position Encoding (NoPE) to all MLA layers. Consequently, no explicit positional encoding is applied to their queries or keys. The intervening KDA layers provide position-sensitive and recency-aware sequence mixing, while the MLA layers provide unrestricted global content interaction. This separation also avoids modifying positional-encoding parameters when extending the context length, such as retuning a RoPE frequency base or applying YaRN.」（报告全文无 absorbed/absorption 字样）；本页 5.1（第 457–461 行）给出的两条原因是「KDA 层承担位置敏感、近因感知的序列混合」与「扩展上下文长度时无需调整位置编码参数」。｜修复要求：把 overview 该条的原因改为与 K3 报告一致的两条，或删除「因 RoPE 与矩阵吸收冲突」这一归因。｜修复：｜复验：
- [轻微·技术] 来源与范围说明 §外部数字与实验条件（N）N3（第 519 行）：N3 把 d_model=512 的来源标为「§3.2.2 / Table 3 base 模型」，其中 §3.2.2 与该数值无关。｜引文依据：ar5iv 1706.03762 §3.2.2 标题为 "Multi-Head Attention"；d_model=512 见 §3.1「all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model=512」，Table 3 base 行亦列出。｜修复要求：把「§3.2.2」改为正确位置（§3.1 Encoder and Decoder Stacks，或仅保留 Table 3 base 行）。｜修复：｜复验：
- [轻微·技术] 页首 blockquote.meta 与 `<meta name="description">`（第 6、62 行）：「主要依据」只列 Vaswani 2017 与 Raffel 2019，未列本页 5.1 与来源 C10/C11 实际依赖的 K3 报告（§2.1.2/§3.4、config.json）与 ALiBi。｜引文依据：第 457–461 行与 C10（第 506 行）依据 K3 报告 §2.1.2/§3.4，config.json 中 `mla_use_nope=true`、`qk_rope_head_dim=64`；C11（第 507 行）依据 Press et al. 2021, arXiv:2108.12409。｜修复要求：在 blockquote.meta 的「主要依据」中补入 K3 报告与 ALiBi（Press et al. 2021）。｜修复：｜复验：
- [轻微·表述] 第 220 行段首「看两个尺度怎么同时工作。」：以祈使式元话语起句（等同于"下面来看…"）。｜引文依据：不适用｜修复要求：删除该句或改为陈述式（如直接给出"高频维与低频维在同一位置的变化速率不同"），不改动后文数据。｜修复：｜复验：
- [轻微·表述] 第 254 行段首「注意：」：元话语提示词（与"需要注意的是"同类）。｜引文依据：不适用｜修复要求：删去「注意：」，直接以"这个'旋转'性质只在一对维度内成立…"起句。｜修复：｜复验：
- [轻微·格式] 第 119 行「第 2 章用手算给出答案」：以章节编号指代章节，违反 style-guide §1「正文引用其他章节时使用章节标题」（本页其余引用均用标题，如"…'绝对正弦位置编码'一章"）。｜引文依据：不适用｜修复要求：改为引用章节标题，如"'绝对正弦位置编码'一章"。｜修复：｜复验：
- [轻微·格式] 第 78 行与第 119 行变量裸写：列表/正文出现「d=4、pos=1、2」「pos=1 和 pos=2」，与全页其余处「$d_{model}$」「$pos$」的写法不一致（style-guide §11；check 第 9 条「同一变量全页写法一致」）。｜引文依据：不适用｜修复要求：把「d=4」改为「$d_{model}=4$」，「pos=1、2」「pos=1 和 pos=2」改为「$pos=1$、$pos=2$」（或 $pos=1,2$），与全页统一。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 6
- 处置：修复。核心内容核对通过：Vaswani §3.5 Eq.(3)(4) 公式、注入方式、"$PE_{pos+\Delta}$ 是 $PE_{pos}$ 线性函数"、Table 3 row (E) sinusoidal 25.8 / learned 25.7（dev newstest2013，big 无 learned 消融）、Table 2 big 28.4（test newstest2014）、参考文献 [9]=Gehring 2017 均与原文一致；d=4 手算（ω₀=1、ω₁=0.01，PE₁≈(0.8415,0.5403,0.0100,1.0000)、PE₂≈(0.9093,−0.4161,0.0200,0.9998)）与和角公式复算一致；T5（32 桶、max_distance 128、每头独立各层共享、clamp、源码 bidirectional/max_exact=8）与 §2.1 及 HF 源码一致；ALiBi 线性负偏置 m（不学习）与原文一致；K3 §2.1.2/§3.4 两条 NoPE 原因与报告一致，config.json 的 `mla_use_nope=true`、`qk_rope_head_dim=64` 支持"拆出 rot 分量"。无同页数字互相矛盾、无算式错误、无失效文件路径（research/ 未引用）、无"（待生成）"占位，validate.py 返回成功（`validation ok`）。关闭上述 2 项重要问题后即可发布。
