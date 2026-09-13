<!-- review-meta
round: 4
page: wiki/swiglu/index.html
reviewed_content_sha256: d6d542609b796ba3
-->
# SwiGLU 审查记录（第 4 轮）

- 页面版本：e7fa4d8524fdb8ef0e7d9589c6ff7ca9ed05e973（wiki/swiglu/index.html 工作树对象哈希）
- 审查时间：2026-09-13 19:52
- 审查者：独立子代理（未参与写作与前三轮审查）
- 已完整阅读章节：头部 meta 与「主要依据」→ 核心问题（5 题，含解答折叠块）→ 最容易误解（5 条）→ 1. 从 GLU 到 SwiGLU（含结构图 SVG 与本章问题 2 题）→ 2. SwiGLU 的公式、Swish 定义与手算（含两个展开折叠块与本章问题 3 题）→ 3. 把 SwiGLU 塞进 Transformer FFN（含参数量推导折叠块与本章问题 3 题）→ 4. 经验结论与边界（4.1–4.5、本章问题 3 题、全文总结）→ 来源与范围说明（C/F/N、构造示例、类比边界、简化条件）。另读了 overview.html 全文。
- 对照来源：Shazeer 2020 arXiv:2002.05202（ar5iv HTML 版，逐节核对 §1 Eq.(3)、§2 Eq.(4)–(6) 与末段、§3.1/§3.2、Table 1、§4 末段）；Dauphin 2017 arXiv:1612.08083（ar5iv，§2 Eq.(1) 与 §5.3）；LLaMA arXiv:2302.13971 §2.2；Gemma arXiv:2403.08295 §2；Kimi K3 arXiv:2607.24653 §2.3.2（ar5iv）；各模型官方 config.json（LLaMA-7B、Mistral-7B-v0.1、Qwen2-7B、Qwen1.5-7B）。

## 问题

- [阻断·技术] 第 3 章「LLaMA 风格 8/3 d 的推导」段（index.html 第 377 行）：把 Mistral 列为「FFN 内部维度取 8/3 d」的模型，与官方 config 矛盾。｜引文依据：Mistral-7B-v0.1 官方 config.json 为 {"hidden_size":4096,"intermediate_size":14336}，即 14336/4096=3.5d，而 8/3·4096≈10922.67，两者相差约 31%；Qwen2-7B config.json 为 {"hidden_size":3584,"intermediate_size":18944}（≈5.29d），同样不是 8/3 d（仅 Qwen1.5-7B 的 {"hidden_size":4096,"intermediate_size":11008} 才接近 8/3 d；LLaMA-7B 11008 一致）。｜修复要求：删除「Mistral、Qwen」，或限定为「LLaMA 及沿用 LLaMA 的 2/3·4d 约定的模型（如 Qwen1.5-7B）」；不得再声称 Mistral 的 FFN 内部维度取 8/3 d；C11（第 543 行）仅以 LLaMA 作据，需与该句同步，不得用「多篇工程综述确认」为 Mistral/Qwen 背书。｜修复：｜复验：

- [阻断·技术] 全文把 Shazeer 实验的任务名写成 "segment-filling"：第 128、430、448、458、495 行与来源清单 N1（第 558 行），overview.html 第 46 行。｜引文依据：Shazeer 2020 §3.2 原文 "Identically to [Raffel et al. 2019], we pre-train for 524,288 steps on the span-filling objective on the C4 dataset."（ar5iv 2002.05202）；该文使用的任务名是 "span-filling objective"，全文无 "segment-filling" 一词（文档中只出现泛述 "predicting missing text segments"）。｜修复要求：把上述 6 处 index.html 与 1 处 overview.html 的 "segment-filling" 一律改为 "span-filling"，并在 C7/N1 的任务名描述处同步；不得保留 "segment-filling" 作为该实验的任务名。｜修复：｜复验：

- [轻微·表述] 第 1 章开头段（第 147、149 行）出现以「本文」为主语的自我指代与元话语："这里只回顾一句：…"、"…都在 GLU 页讲过，本文不重复。"、"本文关心的是：GLU 的门 … 这个形状本身有什么限制？"；另第 171、247、347、422 行反复使用「下一章看…」式预告（第 260 行「见下一章」同）。｜引文依据：不适用｜修复要求：删去「这里只回顾一句」「本文不重复」「本文关心的是」等以页面自身为主语的框架句，直接陈述结论；把「下一章用手算验证…」「下一章看它如何…」改为以内容为衔接的过渡（如直接给出下一节的结论句），全页不再出现「本文/本页」作主语的句子与「下一章看…」预告。｜修复：｜复验：

- [轻微·技术] 章节号指认不精确：第 448 行「Shazeer §2/§3 把 GEGLU 与 SwiGLU 同列为…」，该句实际位于 §3.2；第 535 行称「§3.1 实验设置」，而论文 §3.1 标题为 "Model Architecture"。｜引文依据：ar5iv 2002.05202 中 "The GEGLU and SwiGLU variants produce the best perplexities." 出自 §3.2 "Pre-Training and Perplexity Results"；论文 §3.1 标题为 "Model Architecture"，§3.2 标题为 "Pre-Training and Perplexity Results"。｜修复要求：第 448 行改为「§3.2」，第 535 行改为「§3.1 Model Architecture」或「§3.1」并去掉「实验设置」这一自拟标题。｜修复：｜复验：

- [轻微·技术] 4.5 节「Swish 门正侧无界可能引发激活爆炸」条（第 498 行）把 K3 的动机外推为「深层堆叠 FFN 或路由分支（如 MoE）把多个无界因子相乘，输出可能爆炸」，其中「深层堆叠 FFN」在来源中无对应陈述。｜引文依据：Kimi K3 报告 §2.3.2（ar5iv 2607.24653）动机原文为 "However, both multiplicative factors in SwiGLU are unbounded, so coincident large coordinates can produce activation outliers and increase overflow risk"，语境是 Stable LatentMoE 的路由分支；报告未讨论「深层堆叠 FFN」的相乘爆炸。｜修复要求：把该条限定为 K3 所述的 MoE 路由分支（LatentMoE 结构）情形，删除「深层堆叠 FFN」或将其标注为「本页外推」。｜修复：｜复验：

## 已核对且成立的关键条目（本轮确认无问题，供复验参照）

- Table 1 八行数值全部与来源一致：ReLU 1.677 / GELU 1.679 / Swish 1.683 / GLU 1.663 / Bilinear 1.648 / ReGLU 1.645 / SwiGLU 1.636 / GEGLU 1.633；65,536 步一组 FFN_SwiGLU 1.944 (0.010)、FFN_GEGLU 1.942 (0.004)，524,288 步一组无标准差——与页面第 436–443、448、558 行一致。
- 公式回源：SwiGLU 定义 = Shazeer §2 Eq.(5) "Swish_β(xW+b)⊗(xV+c)"；FFN_SwiGLU = §2 Eq.(6) "(Swish_1(xW)⊗xV)W_2"；FFN_Swish = §1 Eq.(3) "Swish_1(xW_1)W_2"——均一致，且 §2 明确省略偏置，与页面 §3 表述相符。
- 2/3 缩放原文一致："To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units dff by a factor of 2/3"。
- 数字复算全部成立：σ(1)≈0.7311、σ(−1)≈0.2689、Swish 边界值 0 / 0.7311 / −0.2689、手算输出 [0.7311, −0.1345]、GLU 对照 [0.7311, 0.1345]；d=4096 时 2d·4d=8d²=134,217,728、3d·4d=12d²=201,326,592、3·(8/3)=8=2·4；3072×2/3=2048；8/3·4096≈10923；LLaMA-7B 11008=43×256 取整。
- 记法核对：Dauphin 2017 Eq.(1) 为 (X∗W+b)⊗σ(X∗V+c)（σ 在 V 分支），Shazeer §2 写作 σ(xW+b)⊗(xV+c)（激活在 W 分支），页面「Dauphin 主记法 σ 在 V、Shazeer 记法激活在 W、相差 W↔V」成立；Bilinear 归因「Dauphin §5.3 / Mnih & Hinton 2007」成立（§5.3 "Non-linear Modeling" 引 [Mnih & Hinton 2007]）。
- Gemma §2 引文 "The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function."、LLaMA §2.2 "We replace the ReLU non-linearity by the SwiGLU activation function" 与 §4 "divine benevolence" 句均一致。
- head 元数据齐备且合法：description 纯文本、dojo:summary 可渲染、dojo:type=concept、dojo:topics=模型结构（在 ALLOWED_TOPICS 内）、dojo:tag=网络结构（在 ALLOWED_TAGS 内）；`python3 .dojo/scripts/validate.py wiki/swiglu/index.html` 返回 "validation ok"。数学符号全部由 KaTeX/`$...$` 书写，正文无 Unicode 数学字符；结构图为内联 SVG + foreignObject 渲染。指向前置页 swiglu/glu、situ-glu 的链接均存在，页面无 research/ 路径引用，无「（待生成）」占位。

## 结论

- 统计：阻断 2 / 重要 0 / 轻微 3
- 处置：修复。两处阻断均为来源一致性问题（Mistral 的 8/3 d 断言与官方 config 不符；实验任务名 segment-filling 与来源的 span-filling 不符），须修复后复验；三条轻微项随同一轮修复。修复范围限于所列位置及其直接引用处（C11、C7、N1、overview.html 第 46 行）。