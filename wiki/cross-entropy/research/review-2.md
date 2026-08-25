# 交叉熵审查记录（第 2 轮）

- 页面版本：799674ba9eb54afb2ab11e71457979415a19a6ef
- 审查时间：2026-08-25 16:08
- 审查者：独立子代理（本会话即派发的独立审查者，未参与写作与第 1 轮）
- 已完整阅读章节（index.html）：导言、核心问题（4 条 + 解答）、§1 为什么预测概率需要转换成损失（含 §1.1 负对数曲线 SVG 图、本章问题 2 条）、§2 交叉熵的信息论定义（含「补充：为什么允许 0 log 0」折叠、本章问题 2 条）、§3 训练视角（§3.1 二分类手算、§3.2 多分类与 softmax、含「展开：三分类例子的完整对数计算」折叠、本章问题 2 条）、§4 语言模型的下一 token 交叉熵（含构造示例、本章问题 2 条）、「来源与范围说明」全部小节。
- 已完整阅读章节（overview.html）：「它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界 / 底部 footer 与 nav」。

来源核对：

- C1–C6 已逐字核对 Goodfellow et al., *Deep Learning* §3.13 (式 3.48–3.51) 与 §5.5 (式 5.59–5.61)：自信息 (3.48)、香农熵 (3.49)、KL 散度 (3.50) 与「non-negative」「not symmetric」、交叉熵 (3.51) 与「Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term」、0 log 0 惯例「lim x→0 x log x = 0」、§5.5「Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy」与「Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model」全部与页面 C1–C6 引文逐字一致。
- C7 已逐字核对 Radford et al., *Language Models are Unsupervised Multitask Learners* (OpenAI 2019) §3.1 「Language Modeling」：「Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word」与页面 C7 引文完全一致。
- 数值复算：−ln 0.8 = 0.2231、−ln 0.2 = 1.6094、softmax(2.0,1.0,0.5) = (0.6285, 0.2312, 0.1402) 指数 (7.389, 2.718, 1.649) 总和 11.756、−ln 0.6285 = 0.4644、−ln 0.1402 = 1.9644、−ln 0.85 = 0.1625、−ln 0.70 = 0.3567、(0.1625+0.3567)/2 = 0.2596、−ln 0.02 = 3.912（页面图注「约 3.9」）均与页面逐字一致；SVG 曲线 21 个数据点相对 q=0.02..1 与 y=−ln q·57.5 的最大像素偏差 0.046 px，坐标轴刻度（0.1/0.5/1 与 1/2/3）全部对位准确。
- 页面功能：浏览器实际打开 file://…/cross-entropy/index.html，KaTeX 渲染 242 个实例、0 错误、12 个 display 公式；14 个 details 折叠块（含 12 个问题解答、1 个 0 log 0 补充、1 个 softmax 展开）交互正常；16 个目录链接锚点全部存在；明暗主题下 SVG 图与 foreignObject 公式均正常可读；`.dojo/scripts/validate.py wiki/cross-entropy/index.html` 返回 `validation ok`；wiki/pretraining/index.html、wiki/sft/index.html 与 libs/* 资源全部存在，overview/index 通过 `概览` / `完整说明 →` 互链。

## 问题

- [重要·技术] index.html「来源与范围说明 → 简化条件及其限制」第 1 条（line 1065）+ overview.html「关键结论与边界」第 3 条（line 64）：当前文本分别为「连续分布的微分熵与交叉熵可以为负，本页『非负』的表述不适用于连续情形」与「连续分布的交叉熵可以为负，非负性表述不适用」；页面正文实际只在 §2 KL 散度说明中明确写了「非负」（line 884），KL 散度在连续情形仍非负（Goodfellow 原文 §3.13「KL divergence has many useful properties, most notably being non-negative」无离散/连续区分）；页面正文对 H(P) 与 H(P,Q) 都没有写「非负」。当前「非负的表述」指代不明确，读者会按最近匹配理解为 KL 的非负，与「不适用于连续情形」冲突，形成错误印象。｜引文依据：Goodfellow §3.13 原文 "The KL divergence has many useful properties, most notably being non-negative. The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables, or equal 'almost everywhere' in the case of continuous variables."（连续情形下非负性仍然成立）。｜修复要求：将 index.html line 1065 与 overview.html line 64 两处简化为对连续情形 H(P) 与 H(P,Q) 可负的具体说明，并明确 KL 散度的非负性在连续情形仍成立，使边界条件与正文 KL 非负声明一致。两处可统一表述为「连续情形下微分熵 H(P) 与交叉熵 H(P,Q) 可为负，但 KL 散度 D_KL(P‖Q) 的非负性不依赖于离散假设，本文对 KL 非负的结论在连续情形仍成立」。｜已修复：简化条件改为明确「$-\log q(y)\ge 0$ 的非负性结论依赖离散 one-hot 情形（$q(y)\le 1$）」并注明「KL 散度非负在连续情形仍成立」；overview 边界同步改写。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·可读性] overview.html「关键结论与边界」第 1 条（line 63）：「此结论在 $Q$ 是合法概率分布时成立」为冗余限定——结论「同一组预测输出，真实结果翻转，损失大小随之翻转」在 $Q$ 偏离概率分布时甚至不能定义交叉熵，限定本身不携带新信息，且读起来像在暗示 $Q$ 不是分布时该结论可能不成立。｜引文依据：不适用。｜修复要求：删除「此结论在 $Q$ 是合法概率分布时成立」整句，或改写为「此结论在交叉熵的定义域（$Q$ 为概率分布）内平凡成立（$Q$ 不为分布时交叉熵未定义）」以明确默认前提。｜已修复：overview 删除冗余限定「此结论在 $Q$ 是合法概率分布时成立」。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 1
- 处置：修复（重要问题修复并复验后即可发布；轻微问题可作为遗留接受项）
