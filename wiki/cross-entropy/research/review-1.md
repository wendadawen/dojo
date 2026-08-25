# 交叉熵审查记录（第 1 轮）

- 页面版本：index.html 0ce5ad1fba755d113d99484fa1828b460211c597（overview.html e0c3511286567c7b3077450391d3cc3b77addb8e）
- 审查时间：2026-08-25 15:56
- 审查者：编排者派发的独立审查者（未参与写作与修复）
- 已完整阅读章节（按顺序）：
  - index.html：标题与导语、核心问题（4 题及解答折叠块）、1. 为什么预测概率需要转换成损失（含负对数曲线图与本章问题）、2. 交叉熵的信息论定义（含「为什么允许 $0\log 0$」折叠块与本章问题）、3. 训练视角——从最大似然到 one-hot 手算（3.1 二分类手算、3.2 多分类与 softmax、含「完整对数计算」折叠块与本章问题）、4. 语言模型的下一 token 交叉熵（含本章问题）、来源与范围说明（论断与来源 C1–C7、公式与来源 F1–F5、构造示例、类比边界、简化条件）
  - overview.html：全文（它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界）

## 来源核对明细（每条论断的引文依据）

| 论断 | 标注位置 | 核对结果与原文摘录 |
|---|---|---|
| C1 自信息 | DL §3.13 式 (3.48) | 已核对。"T o satisfy all three of these prop erties, we deﬁne the self-information of an ev en t x = x to b e I ( x ) = − log P ( x ) . (3.48) In this b o ok, we alwa ys use log to mean the natural logarithm, with base e . Our deﬁnition of I ( x ) is therefore written in units of nats ."（页面注明自然对数/nats 与原文一致） |
| C2 香农熵 | DL §3.13 式 (3.49) | 已核对。"the Shannon entrop y , H ( x ) = E x ∼ P [ I ( x )] = − E x ∼ P [log P ( x )] , (3.49), also denoted H ( P ) … Distributions that are nearly deterministic (where the outcome is nearly certain) hav e low entrop y; distributions that are closer to uniform hav e high entrop y ."（页面「越确定越小、越接近均匀越大」与原文一致） |
| C3 KL 散度 | DL §3.13 式 (3.50) 及其后性质段落 | 已核对。"D KL ( P ∥ Q ) = E x ∼ P log P ( x ) / Q ( x ) = E x ∼ P [log P ( x ) − log Q ( x )] . (3.50) … it is the extra amoun t of information … needed to send a message containing sym b ols dra wn from probabilit y distribution P , when w e use a co de that w as designed to minimize the length of messages dra wn from probabilit y distribution Q . … most notably b eing non-negativ e. The KL div ergence is 0 if and only if P and Q are the same distribution in the case of discrete v ariables … It is not a true distance measure b ecause it is not symmetric" |
| C4 交叉熵与等价最小化 | DL §3.13 式 (3.51) 及其后一句 | 已核对。"A quantit y that is closely related to the KL divergence is the cross-en trop y H ( P , Q ) = H ( P ) + D KL ( P ∥ Q ) , whic h is similar to the KL div ergence but lacking the term on the left: H ( P , Q ) = − E x ∼ P log Q ( x ) . (3.51) Minimizing the cross-entrop y with resp ect to Q is equiv alent to minimizing the KL div ergence, b ecause Q do es not participate in the omitted term."（页面引文与原文逐字一致） |
| C5 最大似然 = 最小化 NLL = 最小化交叉熵 | DL §5.5 | 已核对。"Minimizing this KL divergence corresp onds exactly to minimizing the cross-en trop y b etw een the distributions. … Any loss consisting of a negative log-lik eliho o d is a cross-en trop y b etw een the empirical distribution deﬁned b y the training set and the probabilit y distribution deﬁned b y mo del." 与 "In softw are, w e often phrase b oth as minimizing a cost function. Maxim um likelihoo d th us b ecomes minimization of the negativ e log-likelihoo d (NLL), or equiv alen tly , minimization of the cross-en trop y ."（页面两句引文与原文逐字一致；式 5.59–5.61 存在于 §5.5，与页面「主要依据」标注相符） |
| C6 $0\log 0$ 惯例 | DL §3.13 | 已核对。"When computing man y of these quan tities, it is common to encoun ter expressions of the form 0 log 0 . By con v en tion, in the con text of information theory , w e treat these expressions as lim x → 0 x log x = 0 ." |
| C7 语言建模结果按平均负对数概率报告 | Radford et al. 2019 §3.1 | 已核对（本地 /tmp/gpt2.txt，行 333–345，位于 "3.1. Language Modeling" 节）。"Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word. We evaluate the same quantity by computing the log-probability of a dataset according to a WebText LM and dividing by the number of canonical units."（位置正确；但正文括号中的预测单元列举与原文不符，见问题 2） |

来源获取说明：Goodfellow 两章用 curl 抓取 deeplearningbook.org 完整页面后提取原文（WebFetch 首次抓取被截断，未采用其结果）；GPT-2 论文使用本地已提取文本 /tmp/gpt2.txt。三轮来源均成功访问，无「未能核对」项。

## 数值复算记录（Python）

- $-\ln 0.8=0.2231$、$-\ln 0.2=1.6094$、$-\ln 0.85=0.1625$、$-\ln 0.70=0.3567$、$-\ln 0.02=3.9120$（页面写「约 3.9」）全部一致。
- softmax：$e^{(2.0,1.0,0.5)}=(7.389,2.718,1.649)$，和 $11.756$，概率 $(0.6285,0.2312,0.1402)$；$-\ln 0.6285=0.4644$、$-\ln 0.1402=1.9644$；$e^{-0.4644}=0.6285$、$e^{-1.9644}=0.1402$ 全部一致。
- 两位置平均 $(0.1625+0.3567)/2=0.2596$ 一致。
- SVG 曲线坐标抽验（线性映射 $x=70+\frac{q-0.02}{0.98}\cdot 570$，$y=270-57.5\cdot(-\ln q)$）：$q=0.02\to(70,45.1)$、$q=0.069\to(98.5,116.2)$、$q=0.265\to(212.5,193.6)$、$q=0.5\to(349.2,230.1)$，与 path 数据一致；刻度 $0.1\to x=116.5$、$0.5\to x=349.2$、y 轴刻度 $1/2/3\to 212.5/155/97.5$ 一致。

## 机械验证

- `python3 .dojo/scripts/validate.py wiki/cross-entropy/index.html` → `validation ok`（exit 0）。
- 本地资源 ../../libs/（katex、prism 等）存在；../../index.html 存在；overview.html 与 index.html 相互链接。
- 全文（去 LaTeX 后）无 Unicode 数学字符（检出的「·」均为标题/页脚/JS 字符串中的中文间隔号，非数学符号）。
- dojo:topics「数学基础,训练与优化」经 validate.py 通过（词表校验内含）。
- 注：规范 2.2.12 要求对照 `guides/concept/style-guide.md`，按本轮派发指令该文件不在允许输入内，此项未逐条核对；已按 check.md 其余各款完成检查。

## 问题

- [重要·图示] index.html §1 负对数曲线图（`<svg viewBox="0 0 680 320">`）：x 轴名称标签 `<foreignObject x="380" y="306" width="260" height="22">` 底边到达 y=328，超出 viewBox 高度 320 共 8px；SVG 默认 `overflow:hidden`，轴名称「$q$：模型赋予真实结果的概率」文字下缘会被裁剪，导致图示关键标签显示不全（同图 x 轴刻度标签底边 304、其余元素均在 320 以内，仅此标签越界）｜引文依据：不适用（本地渲染结构问题）｜修复要求：将 viewBox 高度由 320 增至 340（或将该 foreignObject 上移使其整体位于 320 以内），并保持其余元素位置不变；复验条件：轴名称 foreignObject 的 y+height ≤ viewBox 高度且文字完整可见｜已修复：SVG viewBox 高度由 320 改为 340，x 轴名称标签（y=306, h=22）底边 328 落在 viewBox 内。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html §4「语言建模的结果通常正是以「每个预测单元的平均负对数概率」报告的（预测单元可以是 token、字符或字节）[C7]」：括号内列举与原文不符。原文为 "usually a character, a byte, or a word"（字符、字节或词），不含 token；列举紧跟 C7 标注，读者会误以为出自原文｜引文依据："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word."（GPT-2 论文 §3.1）｜修复要求：将括号改为与原文一致的「（预测单元通常是字符、字节或词）」；若要保留 token 的说明，移出该括号并明确标注为页面自身补充｜已修复：括号改为「（预测单元通常是字符、字节或词）」，与 GPT-2 原文 "usually a character, a byte, or a word" 一致。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·可读性] index.html §4 末段「SFT 在「指令-回答」样本的回答部分上最小化同一个式子」：缩写 SFT 首次出现未给出全称或中文释义，违反「术语首次使用时解释」｜引文依据：不适用｜修复要求：首次出现处写为「SFT（Supervised Fine-Tuning，监督微调）」或改用中文全称｜已修复：第 4 章末段首次出现处改为「SFT（Supervised Fine-Tuning，监督微调）」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·链接] index.html §4 末段「这个形式是后续两页的地基」：「后续两页」以纯文字提及预训练与 SFT 两页，但未给出链接；wiki/pretraining/ 与 wiki/sft/ 页面均已存在，可链接｜引文依据：不适用｜修复要求：在该句为两页添加相对链接（../pretraining/index.html、../sft/index.html），或写明两页标题；复验条件：链接目标存在且可解析｜已修复：「后续两页」改为「后续两个概念页」，语言模型预训练与 SFT 监督微调两处均加 <a> 链接（目标文件已存在）。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（修复后进入第 2 轮独立审查；无返回规划事项）
