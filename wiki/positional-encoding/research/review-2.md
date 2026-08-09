# 位置编码基础 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html ac5b744、overview.html ac5b744
- 时间：2026-08-09
- 来源：Vaswani et al. 2017, "Attention Is All You Need", arXiv:1706.03762, §3.5 与 Table 2/Table 3；Raffel et al. 2019, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", §3.2（T5 相对位置偏置），经 ar5iv 与多源交叉核对。

## 问题

- [重要·技术] "可学习绝对位置编码" 章 Table 3 row (e) 对比表与来源说明块：页面写 sinusoidal base BLEU=27.3、learned base BLEU=26.74，并标注来源"Vaswani et al. 2017 Table 3 row (e)"。但经 ar5iv 原文核对，Table 3 row (e) 的实际数字为：sinusoidal base BLEU (dev)=25.8、learned BLEU (dev)=25.7（newstest2013 开发集）。27.3 是 Table 2 中 base 模型在 newstest2014 **测试集** 上的 BLEU，不属于 Table 3。而 26.74 在论文全文中**不存在**（WebFetch 在论文中搜索 26.74 未找到任何匹配）。页面在 N1 来源说明中又写"经 The Annotated Transformer 核实"，暗示 26.74 可能来自该二手源的复现，但页面正文与表格都把 27.3/26.74 标注为"Table 3 row (e)"的原始数字，属来源误标。修法：将表格数字改为 Table 3 row (e) 实际值（sinusoidal 25.8、learned 25.7，标注为 dev BLEU），或改引 Table 2 的 base 测试 BLEU 27.3 并注明 learned 在 Table 3 dev 上为 25.7（而非编造 26.74）；同时在 N1 来源说明中纠正"Table 3 row (e) 的 BLEU 是 dev 集（newstest2013），不是 Table 2 的 test 集（newstest2014）27.3"。 ｜ 修复：将折叠块表格数字改为 Table 3 row (e) 实际值——sinusoidal base 25.8、learned base 25.7，表头改为"EN-DE BLEU (base, dev newstest2013)"标注 dev 集；L847 说明段"两者差距小于 1 BLEU、正弦略优"仍成立（25.8 vs 25.7）保留不变；N1 来源说明改为"sinusoidal base 25.8、learned base 25.7（dev 集 newstest2013；big 28.4 出自 Table 2 最终结果 test 集 newstest2014）"，删除"经 The Annotated Transformer 核实"（26.74 不在论文中、二手源复现不可靠）。validate.py 通过。 ｜ 复验：
- [轻微·盲读] "相对位置编码" 章对比表中"与 FlashAttention 兼容"行：T5 bias 标注"不兼容（需物化 $n\times n$ 偏置矩阵，Flash 的分块 IO 优化失效）"。"不兼容"表述过强——现代 FlashAttention-2 及之后版本支持注意力偏置（含相对位置偏置），只是效率下降（偏置矩阵需在分块中计算或预物化），并非完全无法运行。括号内解释"需物化 $n\times n$"是简化说法，实际上偏置可按桶函数即时计算、不一定全量物化。修法：将"不兼容"改为"效率下降"或"降低 FlashAttention 加速效果"，或加一句"现代 FlashAttention 变体可支持加法偏置但效率打折"。 ｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：进入修复（重要问题涉及来源数字误标，需修复后复验；轻微不阻断）

## 段 A 盲读小结

扮演完全小白读者按页面顺序阅读。主线理解顺畅：从"我打你"vs"你打我"钩子引出排列等变问题；RNN/CNN/自注意力三者的位置感知来源对比清晰；绝对正弦 PE 公式逐符号拆解 + d=4 手算例子把多频率概念落到可复算数字；F3 线性性质（和角公式推导）与 RoPE 伏笔连接自然；可学习绝对方案的"外推失败"权衡清楚；T5 相对 bias 的"加在分数上"与绝对方案的"加在嵌入上"对比鲜明；四类方案总对比表维度齐全；K3 选 NoPE 的两条理由（RoPE 与矩阵吸收冲突 + KDA 提供位置信息）逻辑闭合。学习目标五条均由正文章节完整回答。卡点仅上述轻微项，不阻断主线。

## 段 B 对照来源小结

逐条核对 Vaswani 2017 §3.5 与 Table 2/Table 3、Raffel 2019 §3.2（经多源交叉）：

1. 定义与机制：正弦 PE 公式 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}})$、$PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}})$（§3.5 Eq.(3)(4)）一致；"we hypothesized it would allow the model to easily learn to attend by relative positions"原文引用逐字一致；"$PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$"一致；"may allow the model to extrapolate to sequence lengths longer than the ones encountered during training"一致；引用 [9]=Gehring et al. 2017 arXiv:1705.03122（ConvS2S）一致。
2. 公式与推导：d=4 手算例子逐值复算一致（$PE_1\approx(0.8415,0.5403,0.0100,1.0000)$、$PE_2\approx(0.9093,-0.4161,0.0200,0.9998)$、$PE_3\approx(0.1411,-0.9900,0.0300,0.9996)$）；$\omega_0=1$、$\omega_1=1/100=0.01$ 复算一致；F3 和角公式推导 $\sin((pos+k)\omega_i)=\sin(pos\omega_i)\cos(k\omega_i)+\cos(pos\omega_i)\sin(k\omega_i)$ 复算一致；d=4 验证 $2\sin 1\cos 1=\sin 2\approx 0.9093$ 一致；波长范围 $2\pi$ 到 $10000\cdot 2\pi$ 一致。
3. 可运行代码：页面无可运行代码块，不适用。
4. 事实与推断：T5 相对 bias 机制（32 桶、对数分桶、每头独立、各层共享、超出 128 截断到最大桶）经多源交叉一致（Raffel 2019 §2.1 原文"32 embeddings"、"logarithmically up to an offset of 128"、"share across all layers"、"each attention head uses a different learned position embedding"）；BERT/GPT-2 采用可学习 PE 一致；big 模型 EN-DE BLEU=28.4（Table 2 test 集）一致。**Table 3 row (e) BLEU 数字发现来源不一致**（见重要问题）：页面写 27.3/26.74，实际 Table 3 为 25.8/25.7（dev 集），26.74 在论文中不存在。
5. 前置知识引用：标准注意力概念页链接 `../../wiki/standard-attention/index.html` 有效；NoPE 概念页链接 `../../wiki/nope/index.html` 有效；RoPE 概念页链接 `../../wiki/rope/index.html` 有效；Kimi K3 概念页链接 `../../wiki/kimi-k3/index.html` 有效；MLA 概念页链接 `../../wiki/mla/index.html` 有效。
6. 教学简化：d=4 手算、T5 分桶不展开完整公式、ALiBi 只列方向、RoPE/NoPE 只引用——均标注简化理由与可/不可推出边界，未发现简化导致核心结论失真。二进制计数器类比标注失效边界（离散 vs 连续、固定位数 vs $d_{model}$ 决定、唯一性证明差异）。
7. 页面功能：KaTeX 公式渲染正常；details 折叠交互正常；侧边目录锚点（why-positional-encoding / sinusoidal-pe / learned-pe / relative-pe / comparison-and-nope / sources-and-teaching-notes）有效。

发现来源不一致 1 项（Table 3 row (e) BLEU 数字误标），已列重要问题。
