<!-- review-meta
round: 4
page: wiki/standard-attention/index.html
reviewed_content_sha256: 9ac5c5816615fa13
-->
# 标准 Transformer 注意力审查记录（第 4 轮）

- 页面版本：e52ec8980202f3aa1c71f407c00fd497a7f9ee20（wiki/standard-attention/index.html 工作树哈希）
- 审查时间：2026-09-13 19:52
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. 注意力要解决什么问题 → 2. 缩放点积公式 → 3. 为什么除以 √d_k → 4. 多头注意力 → 5. 复杂度、瓶颈与边界 → 来源与范围说明（含全部折叠块、图注与两个结构图）。

## 核对依据（来源原文）

- Vaswani et al. 2017 §3.2.1 正文（arXiv:1706.03762 PDF）："While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4."
- 同上 §3.2.1 脚注（PDF 标记为脚注 4；arXiv HTML 渲染标记为 1）："To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q·k = Σ qi ki, has mean 0 and variance dk."
- 同上 §3.2.2："Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."；"headi = Attention(QWiQ, KWiK, VWiV)"；"h=8 … dk=dv=512/8=64"。
- 同上 §4 Table 1：Recurrent O(n·d²)/O(n)/O(n)；Convolutional O(k·n·d²)/O(1)/O(logk n)；Self-Attention O(n²·d)/O(1)/O(1)。§4 正文："self-attention layers are faster than recurrent layers when n < d"。
- Dao et al. 2022 (arXiv:2205.14135) §4.2.2："We generally see 2-4× speedup, and we see more speedup when using dropout and masking due to kernel fusion."（页面"快 2-4 倍"有出处）
- Katharopoulos et al. 2020 (arXiv:2006.16236) 式(6)：V′i = φ(Qi)ᵀ Σj φ(Kj)Vjᵀ ⁄ φ(Qi)ᵀ Σj φ(Kj)，并注明 "the feature map φ(·) is applied rowwise to the matrices Q and K"。
- 复算：2×2 例 softmax([0.7071,0])=[0.670,0.330]、AV=[[1.66,2.66],[2.34,3.34]]、softmax([1,0])=[0.731,0.269]；3×3 遮罩各行 [1,0,0]/[0.401,0.599,0]/[0.258,0.316,0.426]；e^16=8.89e6；8√(2ln8)=16.3、8√(2ln256)=26.6；2048²=4.19e6、32768²=1.07e9；ln256=5.545——全部与页面一致。
- 复算表格：q,k~N(0,1)、n=256 key、2000 次试验（np 复现）得 d_k=4 → 未缩放 0.173/3.94、缩放 0.044/5.06；d_k=64 → 0.736/0.77、0.042/5.06；d_k=1024 → 0.940/0.15、0.043/5.05——与页面表内数值（0.170/3.97、0.749/0.75、0.936/0.16；缩放全为 0.043/5.05）一致。

## 问题

- [重要·技术] 第 5 章 callout（第 523 行）：Linear Attention 的"重排"公式写错，把 φ 施加到了 V 且漏掉作用于 Q 的 φ，与其上一句"把 exp(q·k) 替换为核分解 φ(q)φ(k)"及来源不符。｜引文依据：Katharopoulos et al. 2020 式(6) 为 V′i = φ(Qi)ᵀ Σj φ(Kj)Vjᵀ ⁄ φ(Qi)ᵀ Σj φ(Kj)，论文明确"φ(·) is applied rowwise to the matrices Q and K"，V 不加 φ。｜修复要求：把 "$Q(\phi(K)^\top\phi(V))$" 改为 "$\phi(Q)(\phi(K)^\top V)$"（如需保留分母可写 $\phi(Q)(\phi(K)^\top V)/\phi(Q)(\phi(K)^\top \mathbf{1})$）。｜修复：｜复验：
- [轻微·格式] 来源与范围说明·论断与来源（C）（第 582 行）：C2 在来源清单中定义，但正文任何位置都没有出现 <sup>[C2]</sup>，违反"与来源章节双向对应"，且 C2 与 F1（第 595 行）内容重复。｜引文依据：不适用｜修复要求：删除 C2，或把 §2 公式处的 <sup>[F1]</sup> 改为 <sup>[C2, F1]</sup>。｜修复：｜复验：
- [轻微·格式] 第 2 章 callout（第 443 行"常见误解"）与第 1 章 callout（第 136 行）：两处都用了 callout-blue，违反 style-guide §3"blue…每篇最多 1 个"；第 443 行为"易误解澄清"，语义上应属 yellow。｜引文依据：不适用｜修复要求：把第 443 行的 class 由 callout-blue 改为 callout-yellow。｜修复：｜复验：
- [轻微·技术] 第 3 章及来源清单（第 302、311、366、583、584、597 行）：把方差推导反复标注为"§3.2.1 脚注 1"。arXiv PDF 中该脚注编号为 4（正文标记 "…extremely small gradients 4 ."，脚注正文页脚标 4），仅 arXiv HTML 渲染把它标为 1；同时 C4（第 584 行）称脚注"描述 'extremely small gradients'"，该短语实际在 §3.2.1 正文而非脚注。｜引文依据：PDF 片段 "extremely small gradients 4 ."；arXiv HTML 该脚注 <sup class="ltx_note_mark">1</sup>。｜修复要求：脚注引用去掉编号（写"§3.2.1 脚注"）或与 PDF 一致改为"脚注 4"；C4 中"extremely small gradients"的来源改为"§3.2.1 正文"。｜修复：｜复验：
- [轻微·表述] 第 3 章（第 302 行）：译文引文"（"我们怀疑大 $d_k$ 时点积变大、把 softmax 推到梯度极小的区域"）"使用第一人称复数"我们"，违反 style-guide §12 与 check.md 第 12 项（会话指代）。｜引文依据：不适用｜修复要求：改为第三人称，如"（论文怀疑大 $d_k$ 时点积变大、把 softmax 推向梯度极小的区域）"。｜修复：｜复验：
- [轻微·表述] 全页（第 138、351、447、500、520、532 行；第 65、575 行）：以"本页"为主语的自我指代与范围叙述出现 8 处（"本页在公式与机制判断上以正式定义为准""本页回答…""本页只引瓶颈""本页不展开加性注意力的机制"等），另有"本文回答：…""回顾本文开篇的问题："，属元话语式范围叙述，读感偏拼接。｜引文依据：不适用｜修复要求：正文内的"本页只引…不展开…"句改写成直接陈述（如"下面只讨论瓶颈，机制由对应变体章节承担"），把"本页回答…"改为"这一章讨论…"；范围限定集中保留在"来源与范围说明"一节。｜修复：｜复验：
- [轻微·技术] 来源与范围说明·论断与来源（C）（第 590 行）：C10 的英文引文非逐字——页面写 "Additive attention outperforms dot product attention without scaling for larger values of $d_k$, while the two mechanisms perform similarly for small values."，原文语序为 "While for small values of dₖ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dₖ."（词典故与含义一致，但引号内语序被调换、首字母被大写）。｜引文依据：见上"核对依据"第 1 条。｜修复要求：按原文语序恢复，或去掉引号改为转述。｜修复：｜复验：
- [轻微·格式] 第 5 章（第 515 行）："$n=32768$ 时 $n^2\approx 1.1\times 10^9$， Attention 矩阵本身就占数 GB 显存"中"矩阵"前有多余半角空格，且句中英文 Attention 首字母大写与全页用法不一致。｜引文依据：不适用｜修复要求：删除逗号后的多余空格，统一改为小写 attention 或中文"注意力矩阵"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 7
- 处置：修复（关闭第 1 条重要问题与轻微问题后即可发布；核心公式、方差推导、2×2/3×3 示例、饱和对照表与复杂度/路径长度数字均已回源核对或复算通过，validate.py 返回成功，overview.html 与 index.html 互链，rope/mla 前置页存在）
