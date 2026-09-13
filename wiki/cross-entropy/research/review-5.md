<!-- review-meta
round: 5
page: wiki/cross-entropy/index.html
reviewed_content_sha256: 324a997ea2d6e84f
-->
# 交叉熵审查记录（第 5 轮）

- 页面版本：c210244591e00394cfa8b085da8b18116205c6ef
- 审查时间：2026-09-13 20:09
- 审查者：独立子代理（编排者派发，未参与写作与前序审查；本轮只读 index.html、overview.html、外部来源与 guides/concept/check.md）
- 已完整阅读章节：核心问题；1. 为什么预测概率需要转换成损失（含本章问题）；2. 交叉熵的信息论定义（含本章问题）；3. 训练视角——从最大似然到 one-hot 手算（3.1 二分类手算、3.2 多分类与 softmax、本章问题）；4. 语言模型的下一 token 交叉熵（含本章问题）；来源与范围说明

## 来源与数字核对（逐条留证）

- C1 自信息：deeplearningbook.org/contents/prob.html §3.13 原文 "we define the self-information of an event x = x to be I(x) = − log P(x). (3.48)"，并注明 "In this book, we always use log to mean the natural logarithm, with base e. Our definition of I(x) is therefore written in units of nats." —— 与页面第 2 章及 C1 一致。
- C2 香农熵：同章 "H(x) = E_{x∼P}[I(x)] = −E_{x∼P}[log P(x)], (3.49) … Distributions that are nearly deterministic … have low entropy; distributions that are closer to uniform have high entropy." —— 与页面一致。
- C3 KL 散度：同章 "D_KL(P‖Q) = E_{x∼P}[log P(x) − log Q(x)]. (3.50) … most notably being non-negative. The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables … It is not a true distance measure because it is not symmetric: D_KL(P‖Q) ≠ D_KL(Q‖P) for some P and Q." —— 与页面"非负""离散情形 P=Q 时为 0""不对称"一致。
- C4 交叉熵与 KL 的关系：同章 "H(P, Q) = −E_{x∼P} log Q(x). (3.51) Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term." —— 引文逐字对合，页面对 C4 的引用与这句原文完全一致。
- C6 0 log 0 惯例：同章 "By convention, in the context of information theory, we treat these expressions as lim_{x→0} x log x = 0." —— 与页面"补充：为什么允许 0 log 0"一致。
- C5 最大似然 = NLL = 交叉熵：ml.html §5.5 原文 "Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy." 与 "Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model." —— 两句均逐字命中，页面对 C5 的引用一致。meta 引用的"式 5.59–5.61"确在该节（(5.59) 为经验分布期望形式、(5.60) 为 KL、(5.61) 为 −E[log p_model]），编号无误。
- C7 语言建模报告量：GPT-2 论文 §3.1 原文 "Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word." —— 与页面"以每个预测单元的平均负对数概率（或其缩放/指数化形式，如困惑度）报告（预测单元通常是字符、字节或词）"一致，未扩大范围（页面用"通常"与原文 commonly 对应）。
- 数值复算（Python）：−ln 0.8 = 0.223144、−ln 0.2 = 1.609438、exp(2.0,1.0,0.5) = (7.38906, 2.71828, 1.64872) 和 = 11.75606、softmax = (0.628532, 0.231224, 0.140244)、−ln 0.6285 = 0.46442、−ln 0.1402 = 1.96469、−ln 0.85 = 0.162519、−ln 0.70 = 0.356675、(0.1625+0.3567)/2 = 0.259597。页面出现的 0.2231 / 1.6094 / 7.389 / 2.718 / 1.649 / 11.756 / 0.6285 / 0.2312 / 0.1402 / 0.4644 / 1.9647 / 0.1625 / 0.3567 / 0.2596 全部与复算一致；"更精确概率 0.14024…、按它计算损失 1.9644"亦与真值 0.140244 / 1.964369 一致。
- 分布求和：两处五 token 条件分布（0.85+0.05+0.05+0.03+0.02=1.00；0.70+0.15+0.13+0.01+0.01=1.00）均为合法概率分布；softmax 三概率求和为 1。
- 图（SVG）核对：曲线端点 (70,45.1) 对应 x 轴 q≈0.020、y 轴损失≈3.91（与 −ln 0.02 = 3.912 相符），(640,270) 对应 q=1、损失 0；抽样点 (326.5,225.5) 对应 q≈0.461、损失≈0.774，落点与 −ln q 一致。x 轴刻度 0.1/0.5/1、y 轴刻度 1/2/3 的像素位置与线性映射自洽。图注"横轴只画到 q=0.02，此处损失约 3.9"成立。
- 链接与机械项：wiki/pretraining/index.html、wiki/sft/index.html 均存在，链接有效；overview.html 与 index.html 互链；无"（待生成）"占位；正文与来源说明未指向 research/ 下任何文件。dojo:type=concept，dojo:topics=「数学基础,训练与优化」（均在 AGENTS.md 固定大类内），dojo:tag=「数学与数值」（在 catalog_builder 词表内）。python3 .dojo/scripts/validate.py wiki/cross-entropy/index.html 返回 "validation ok"。公式定界符之外无 Unicode 数学字符（仅导航 UI 符号 ↑ ⌂ ◐ ☀）。
- 未发现问题项：无来源不支持的论断被写成结论（第 1 章的梯度论证为显式推理并在"辅助解释与类比边界"标界，编码代价直觉也已标界不用于推导）；构造示例均登记在"构造示例"节并声明非实测；无页内两处矛盾；无会话指代（我/我们/你）；无调试与复现叙事；无"需要注意的是""综上"一类公文连接词。

## 问题

- [轻微·可读性] 4. 语言模型的下一 token 交叉熵（正文第 2 段"（如困惑度）"、本章问题第 2 题答案、来源 C7）：术语"困惑度（perplexity）"在页面首次出现时未作任何解释，读者无法从页内得知它指什么。｜引文依据：不适用（可读性）｜修复要求：在首次出现处补一句最小解释（例如"困惑度是对该平均负对数概率取指数，可读作模型每一步的平均候选数"），或改为只写"其指数形式"并在此处引出困惑度名称。｜修复：｜复验：
- [轻微·表述] 1. 为什么预测概率需要转换成损失（第 1 段）：同一段内先说"训练对数值目标有一项要求"，紧接着说"这要求两件事"，后文又以"第一条""第二条"呼应两件事，前后数量表述不一致。｜引文依据：不适用（可读性）｜修复要求：把首句改为"训练对数值目标有两项要求"，或将"这要求两件事"改写为与"一项要求"一致的表述。｜修复：｜复验：
- [轻微·表述] 4. 语言模型的下一 token 交叉熵（正文第 2 段末句）："这对逐位置诊断模型非常方便"是临场评价（"非常方便"），而非对机制的陈述，句上也没有来源支持。｜引文依据：不适用（表述）｜修复要求：改写为事实陈述（例如"因此可以逐位置计算、逐位置诊断：某个位置的损失高，就说明模型在该前文条件下预测得差"，与本章问题第 2 题答案保持一致）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（三条轻微问题不影响正确性与主线理解，建议修复后随下轮或直接合入）