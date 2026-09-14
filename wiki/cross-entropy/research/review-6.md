<!-- review-meta
round: 6
page: wiki/cross-entropy/index.html
reviewed_content_sha256: 74e9dd306e87db39
-->
# 交叉熵审查记录（第 6 轮）

- 页面版本：04e8bc5e11b3a1b5a1e98b8c2ef233bc398e79ff
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节（按序）：引言与核心问题；1. 为什么预测概率需要转换成损失（含负对数曲线图及图注、本章问题折叠块）；2. 交叉熵的信息论定义（含「补充：为什么允许 0 log 0」折叠块、本章问题）；3. 训练视角——从最大似然到 one-hot 手算（3.1 二分类手算、3.2 多分类与 softmax、「展开：三分类例子的完整对数计算」折叠块、本章问题）；4. 语言模型的下一 token 交叉熵（本章问题）；来源与范围说明（论断与来源（C）、公式与来源（F）、构造示例、辅助解释与类比边界、简化条件及其限制）。并核对了 overview.html 与两个出链概念页。
- 核对的外部来源：Goodfellow, Bengio & Courville, *Deep Learning* §3.13 式 (3.48)–(3.51)、§5.5 式 (5.58)–(5.61)（deeplearningbook.org/contents/prob.html、ml.html，逐字取原文）；Radford et al. 2019, *Language Models are Unsupervised Multitask Learners* §3.1（cdn.openai.com PDF，逐字取原文）。

## 来源核对（本轮实证）

- C1 §3.13 原文：「To satisfy all three of these properties, we deﬁne the self-information of an event x = x to be I(x) = −log P (x). (3.48)」；「In this book, we always use log to mean the natural logarithm, with base e. Our deﬁnition of I(x) is therefore written in units of nats.」——页面自信息式与「自然对数/nats」表述一致；「取底 2 为 bits、只差常数倍」亦见同节「information measured in bits is just a rescaling of information measured in nats」。
- C2 §3.13 原文：「H(x) = Ex∼P[I(x)] = −Ex∼P[log P (x)], (3.49) also denoted H(P)」——与页面香农熵式一致。
- C3 原文：「DKL(P‖Q) = Ex∼P[log P (x) − log Q(x)] . (3.50)」；「The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables, or equal “almost everywhere” in the case of continuous variables.」；「It is not a true distance measure because it is not symmetric: DKL(P‖Q)≠DKL(Q‖P)」——页面非负/离散同分布为 0/不对称三条一致。
- C4 原文：「H(P, Q) = −Ex∼P log Q(x). (3.51)」「Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term.」——页面 C4 引文与式 (3.51) 逐字相符。
- C6 原文：「By convention, in the context of information theory, we treat these expressions as limx→0 x log x = 0.」——与页面 0log0 折叠块一致。
- C5 原文（§5.5）：「Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy.」与「Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution deﬁned by the training set and the probability distribution deﬁned by model.」——页面 C5 两条引文逐字相符；经验分布、式 5.59–5.61 均在 §5.5 内。
- C7 原文（GPT-2 §3.1）：「Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word.」——与页面「每个预测单元的平均负对数概率或其缩放/指数化形式（预测单元通常是字符、字节或词）」一致。
- 数字复算（Python）：−ln0.8=0.22314→0.2231；−ln0.2=1.60944→1.6094；softmax(2.0,1.0,0.5)：e^z=(7.389,2.718,1.649)，和 11.756，q=(0.6285,0.2312,0.1402)；−ln0.6285=0.4644，−ln0.1402=1.9647；精确 q₃=0.14024…→−ln=1.9644；−ln0.85=0.1625，−ln0.70=0.3567，(0.1625+0.3567)/2=0.2596。全部与正文、核心问题解答、构造示例清单（0.2231、1.6094、0.4644、1.9647、0.1625、0.3567、0.2596）逐一相符，正文/summary/overview/图注间无差异。
- 图内像素测量：x 轴 0.1/0.5/1 分别落在 x=116.5/349.2/640（比例一致 581.7px/单位），y 轴 1/2/3 落在 y=212.5/155/97.5（57.5px/单位）。按此比例重算曲线路径首点 (70,45.1) 对应 q≈0.02、−ln0.02=3.91；中点 (355,231.3) 对应 q=0.5101、损失 0.673；点 (469,250.0) 对应 q=0.706、损失 0.348——与 SVG path 坐标逐点吻合，图注「横轴只画到 q=0.02，此处损失约 3.9」读数正确。
- 机械项：validate.py 返回 validation ok；KaTeX 渲染 dojo:summary 三段公式（含 \mathbb{E}、\mathrm{KL}、\|）均可解析；全页 $ 定界符之外无 Unicode 数学字符（仅破折号与 UI 符号）；图内公式全部在 `<foreignObject>` 内，`<text>` 只承载中文；出链 ../../wiki/pretraining/index.html、../../wiki/sft/index.html 与 overview.html 均存在且互链；aria-label 内无 `$...$`。

## 问题

- [轻微·可读性] 核心问题「交叉熵的定义是什么，它与熵、KL 散度、负对数似然分别是什么关系？」的解答段（第 81 行）：「训练视角下（经验分布 one-hot），最小化负对数似然就是最小化交叉熵，两者是同一对象。」｜引文依据：不适用（页面内部一致性；one-hot 的说明在第 239 行「训练数据几乎总是以 one-hot 形式给出「正确答案」：一个分类样本只有真实类别」才出现）｜修复要求：在核心问题解答中 one-hot 首次出现处补最小含义（如括注「one-hot：概率全部集中在真实类别上」），或在解答内改写为不依赖该术语的表述；使该解答不依赖回看第 3 章即可读懂（style-guide 第 9 节：核心问题答案独立成段）｜修复：｜复验：
- 说明：本页 C1–C7、F1–F5 编号在正文/来源章节双向可查，C 引用与来源章节一一对应；F 条目仅作公式出处备案、未被正文以 `<sup>[F#]</sup>` 引用，经比对全站约四十个概念页同为「F 仅列公式出处」的写法，属项目既有记法，故不作为本页问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：修复（仅 1 条轻微，可直接修正；无阻断项与重要项）
