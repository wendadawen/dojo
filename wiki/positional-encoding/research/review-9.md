<!-- review-meta
round: 9
page: wiki/positional-encoding/index.html
reviewed_content_sha256: f16e8440a187b3ef
-->
# 位置编码基础审查记录（第 9 轮）

- 页面版本：479b78328d6cf638b61e8fc723f3de8d81468434（index.html 工作树 git hash-object；工作树干净）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：引言与「核心问题」（5 条）、「常见误解」（3 条）、1. 为什么 Transformer 需要位置编码——注意力的排列等变性、2. 绝对正弦位置编码——Vaswani 2017 的 sin/cos 公式与手算、3. 可学习绝对位置编码——把固定向量换成可学习参数、4. 相对位置编码——T5 的 bias 机制与「加在分数上」的本质区别、5. 四类方案对比与 NoPE 选择——为什么 K3 选 NoPE（含 5.1）、来源与范围说明；逐段读完 5 个正文章节的「本章问题」与全部 details 折叠块（$d_{model}=4$ 手算展开、和角公式推导、Table 3 对比、T5 分桶方向性）、图注与两处对比表，并通读 overview.html 全文。
- 关键核对结果（§2.2 第 3 项要求）：Vaswani 2017 §3.5 原文 "PE(pos,2i)=sin(pos/10000^(2i/d_model))"、"（see Table 3 row (E)）"、"We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."，Table 3 base 25.8 / row (E) 25.7（dev newstest2013），Table 2 big 28.4，参考文献 [9]=Gehring et al. 2017, arXiv:1705.03122——页面对应表述一致；d_model=4 手算全部可复算（PE_1≈(0.8415,0.5403,0.0100,1.0000)、PE_2≈(0.9093,−0.4161,0.0200,0.9998)，线性性质用和角公式验证 sin2=2sin1cos1≈0.9093，旋转角 −Δωᵢ 与矩阵 [[cos,sin],[−sin,cos]] 一致）；T5 §2.1 "we use 32 embeddings … up to an offset of 128"、"beyond which we assign all relative positions to the same embedding"、逐头独立/各层共享（源码 max_exact=8 对应 |i−j|<8）；K3 报告 v2 §2.1.2 "applies No Position Encoding (NoPE) to all MLA layers … provide position-sensitive and recency-aware sequence mixing … unrestricted global content interaction … retuning a RoPE frequency base or applying YaRN"、§3.4 "extrapolates directly to 1M-token contexts"，官方 config.json 实测 mla_use_nope=true（另 qk_rope_head_dim=64、qk_nope_head_dim=128）；Press 2021 §3 "head-specific slope fixed before training"；DeepSeek-V2 §2.1.3 "RoPE is incompatible with low-rank KV compression"；Kazemnejad 2023 摘要 "NoPE outperforms other explicit positional encoding methods"；validate.py 返回 validation ok，5 个被引前置概念页（rope/nope/kimi-k3/mla/standard-attention）均存在。
- 未发现阻断级问题：核心论断、公式、数字与来源一致，无同页数字冲突，代码片段无（本页无可运行代码）。

## 问题

- [重要·技术] overview.html 第 41 行（「3. 关键结论与边界」第 3 条）：无来源支持的归因被写成结论。该条写"相对 bias …… 原版 FlashAttention 不直接支持 …… 这是后来 RoPE 被广泛采用的原因之一"。index.html 全文不含 FlashAttention（grep 命中 0 次），也未把相对 bias 的 n×n 开销与 RoPE 的采用史关联；overview.html 无任何引文标注，定位不到支持该归因的来源。｜引文依据：index.html 第 4 章表格仅有"额外推理开销……需 $n\times n$ 的逐对偏置矩阵（随序列长度平方增长）"，无 FlashAttention 或 RoPE 采用史内容；overview.html 原文"这是后来 RoPE 被广泛采用的原因之一"无来源。｜修复要求：删除该归因句（保留有依据的"相对 bias 需 n×n 逐对矩阵"部分），或补可定位来源（FlashAttention 文档/论文）。｜修复：｜复验：
- [轻微·技术] index.html 第 296 行（details「展开：Vaswani 2017 Table 3 row (e) 的完整对比数字」）：标注为"论文 §3.5 原文"的引文与原文不符——漏 "instead"，且把 "row (E)" 写成 "row (e)"。｜引文依据：arXiv:1706.03762 §3.5 原文 "We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E))."；该论文 Table 3 行标签为大写 (E)（"positional embedding instead of sinusoids"，BLEU 25.7）。｜修复要求：引文按原文补 "instead"、改为 "(E)"；全文 10 处 "row (e)"（第 62/88/112/295/296/306/317/339/501/519 行）统一为 "row (E)"。｜修复：｜复验：
- [轻微·技术] index.html 第 363 行（第 4 章「分桶」条目）：同一句先说"相对距离 $i-j$ 是整数（从 $-(n-1)$ 到 $n-1$）"，紧接着说"T5 用一个分桶函数把连续距离映射到有限个桶"——"连续"与"整数"自相矛盾。｜引文依据：不适用（同页自相矛盾）。｜修复要求：删去"连续"，改为"把 $-(n-1)$ 到 $n-1$ 的整数距离映射到有限个桶"或等价表述。｜修复：｜复验：
- [轻微·技术] index.html 第 316、394、407、429 行：同一量（嵌入维度）同页两种写法。第 316 行同一表格单元格内先写 "$L\cdot d_{model}$"、再写 "$L=512, d=512$"；第 394 行写 "$L\cdot d$"；第 407、429 行写 "$d$ 维旋转向量"。全文 $d_{model}$ 出现 46 次，裸 $d$ 全页无定义（第 187 行只定义 $d_{model}$），违反符号单义。｜引文依据：不适用（符号一致性）。｜修复要求：4 处裸 $d$ 统一改为 $d_{model}$。｜修复：｜复验：
- [轻微·技术] index.html 第 505 行 C9：引 "Su et al. 2021, arXiv:2104.09864 §3.1" 支持"RoPE 绝对构造、相对效果"。所引 §3.1（Formulation）只给出"内积只依赖相对位置"的要求（Eq 11）；按绝对位置旋转 query/key（Eq 12）在 §3.2.1，使绝对位置相消、得到相对形式的等式（Eq 16）在 §3.2.2/§3.4——所引位置只覆盖论断的"相对效果"一半，"绝对构造"一半不在该节。｜引文依据：§3.1 "we require the inner product of query $q_m$ and key $k_n$ to be formulated by a function $g$, which takes only the word embeddings $x_m$, $x_n$, and their relative position $m-n$ as input variables."；Eq 12（§3.2.1）$f_q(x_m,m)=(W_q x_m)e^{im\theta}$。｜修复要求：改引 §3.1/§3.2/§3.4（或 §3.2 与 §3.4），与"绝对构造 + 相对效果"两半对应。｜修复：｜复验：
- [轻微·功能] index.html 第 144、257、326、409、467 行：5 个 `<h3>本章问题</h3>` 均无 id。页面脚本（第 581-588 行）对无 id 的标题按 textContent 生成 id，5 处都得到 "本章问题"，运行时产生 5 个重复 id；目录生成（第 577-604 行）据此产出的 5 条"本章问题"锚点全部指向同一元素（`getElementById` 取首个，即第 1 章的），点击第 3/4/5 章的"本章问题"会跳到第 1 章，且 5 条同时高亮（第 613-615 行按 `dataset.target` 匹配）。同仓库 67 个页面为该 h3 显式指定了唯一 id（如 attention-sink 的 "questions-rule-and-motivation"、causal-mask 的 "phenomenon-questions"）。｜引文依据：不适用（页面功能）。｜修复要求：给 5 个 h3 各加唯一 id（如 questions-why-pos / questions-sinusoidal / questions-learned / questions-relative / questions-comparison）。｜修复：｜复验：

## 结论

- 处置：修复（1 条重要问题须在发布前关闭；5 条轻微问题按 §4 逐条修复并复验）
- 统计：阻断 0 / 重要 1 / 轻微 5
