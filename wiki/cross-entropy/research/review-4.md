<!-- review-meta
round: 4
page: wiki/cross-entropy/index.html
reviewed_content_sha256: fd9fa9b2cd36c8e3
-->
# 交叉熵审查记录（第 4 轮）

- 页面版本：f57b0ba41e2136bea378d52976f6d9caabb87e85（index.html 工作树哈希）
- 审查时间：2026-09-13
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：引言与核心问题、1. 为什么预测概率需要转换成损失、2. 交叉熵的信息论定义、3. 训练视角——从最大似然到 one-hot 手算（含 3.1 二分类手算、3.2 多分类与 softmax）、4. 语言模型的下一 token 交叉熵、来源与范围说明（含折叠块、图注与 SVG 图内标签）

## 来源核对依据（逐条定位）

- C1 Deep Learning §3.13 式 (3.48) 原文：`I(x) = − log P(x). (3.48) In this book, we always use log to mean the natural logarithm, with base e.`——与页面一致。
- C2 式 (3.49) 原文：`H(x) = E_{x∼P}[I(x)] = −E_{x∼P}[log P(x)], (3.49) also denoted H(P).`——一致。
- C3 式 (3.50) 原文：`D_KL(P‖Q) = E_{x∼P}[log P(x) − log Q(x)]. (3.50) ... most notably being non-negative. The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables`；同段 `It is not a true distance measure because it is not symmetric`——一致。
- C4 式 (3.51) 原文：`H(P,Q) = H(P) + D_KL(P‖Q) ... H(P,Q) = −E_{x∼P} log Q(x). (3.51) Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term.`——一致。
- C6 §3.13 原文：`By convention, in the context of information theory, we treat these expressions as lim_{x→0} x log x = 0.`——一致。
- C5 §5.5 原文：`Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy.` 与 `Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model.`——一致（§5.5 式 5.59 ML、5.60 KL、5.61 −E log p_model 均定位到）。
- C7 GPT-2 §3.1 原文：`Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word.`——来源条目 C7 与之一致，正文表述见下。
- 全部数值用 Python 复算并逐项吻合：−ln0.8=0.22314（页写 0.2231）、−ln0.2=1.60944（1.6094）、e^2+e^1+e^0.5=11.756、softmax=(0.62853,0.23122,0.14024)（页写 0.6285/0.2312/0.1402）、−ln0.6285=0.46442（0.4644）、−ln0.1402=1.96469（1.9647）、−ln0.14024=1.96440（折叠块写 1.9644）、−ln0.85=0.16252（0.1625）、−ln0.70=0.35667（0.3567）、均值 0.25960（0.2596）。图内 −ln q 曲线点位与坐标轴刻度按 −ln q 换算全部吻合（q=0.02 处 3.912，图注写「约 3.9」）。

## 问题

- [重要·技术] 第 2 章末段（`id="info-theory"` 第 208 行附近）与第 3 章 one-hot 化简式（第 241 行）及全书其余各处：同一个量（模型分布赋予真实结果 $y$ 的概率）在页内出现两种写法——第 2 章写作 $-\log Q(y)$（大写 $Q$），第 3 章、页面级核心问题解答、meta description 与正文其余位置写作 $-\log q(y)$（小写 $q$），且 $Q$ 在第 2 章被定义为「分布」、$q$ 在第 3 章被定义为「概率」，两者关系全文未交代。违反 check.md §2.2-2「符号全文单义」与 §2.2-9「同一变量全页写法一致」、style-guide.md §11「同一变量在页面中保持同一种写法」｜引文依据：第 208 行「$H(P,Q)=-\log Q(y)$」对第 241 行「$H(P, Q) = -\log q(y)$」，同一式两处大小写不同｜修复要求：把 one-hot 化简式全文统一为同一符号（建议统一为 $q(y)$，与第 3 章符号定义及核心问题解答一致），并在第 3 章首次引入 $q(y)$ 处说明它与第 2 章 $Q$ 的关系（$q(y)=Q(y)$），或第 2 章即改用 $q(y)$；改后核对 C4/F3 的引用位置｜修复：｜复验：
- [轻微·技术] 第 4 章正文（第 310 行）与「本章问题」解答（第 337 行）：「语言建模的结果通常正是以『每个预测单元的平均负对数概率』报告的」把来源的限定语「其缩放或指数化形式」在正文中省略，读起来像是结果直接以原始平均负对数概率报告；来源明确说报告量是该量的 scaled or exponentiated version（如 PPL/BPB）。来源章节 C7 条目本身保留了限定语，正文与之一致性不足｜引文依据：C7 原文 `a scaled or exponentiated version of the average negative log probability per canonical prediction unit`｜修复要求：把正文第 310 行与第 337 行的表述改为与 C7 条目一致（补回「或其缩放/指数化形式（如困惑度）」），不改动「逐 token 平均」这一结论｜修复：｜复验：
- [轻微·表述] 第 4 章正文（第 330 行）与「公式与来源（F）」F5（第 361 行）：「不同场景的使用」「语言模型场景的组合」把「场景」当术语使用。check.md §2.2-12 将「把『场景』当术语」列为需排除的拼接腔｜引文依据：不适用｜修复要求：把两处「场景」改为具体名词（如「不同任务」「语言模型上的组合」），全页不留以「场景」充当术语的用法｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（无阻断项；核心定义、公式、来源引文与全部手算数值经复算与回源核对无误，validate.py 返回 `validation ok`，无 research/ 残留路径与占位符，pretraining/sft 内链有效，图内公式均由 foreignObject+KaTeX 渲染且暗色主题下 `<text>` 有 `fill: var(--text)`。上列 1 项重要问题需关闭后方达发布条件）