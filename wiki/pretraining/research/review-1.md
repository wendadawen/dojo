# 语言模型预训练审查记录（第 1 轮）

- 页面版本：index.html ea79a0610ffb1aa0d51717864b6e0e648745c7e3（overview.html c5681bd5828fe13c94a0851d432ba054292f9dfe）
- 审查时间：2026-08-25 15:47
- 审查者：编排者派发的独立审查者（未参与写作与修复）
- 已完整阅读章节（按顺序）：
  - index.html：导语；核心问题（页面级，4 问，含全部解答折叠块）；「1. 语言模型是什么」（含本章问题 2 问）；「2. 一条文本的概率如何逐 token 分解」（含「展开：为什么联合概率能写成条件概率连乘」折叠块与本章问题 2 问）；「3. 预训练的目标——在大语料上最小化下一 token 交叉熵」（含本章问题 2 问）；「4. 预训练产出基座模型，行为对齐交给后训练」（含本章问题 2 问）；「来源与范围说明」（论断与来源 C1–C6、公式与来源 F1–F2、构造示例、辅助解释与类比边界、简化条件及其限制）
  - overview.html：全文（它是什么／为什么需要它／核心机制／关键结论与边界）

## 问题

- [重要·技术] index.html「主要依据」blockquote、C2 条目、F1 条目：三处均将 GPT-2 式 (1) 标注为「§2.1 式 (1)」，但 GPT-2 论文式 (1) 位于 §2「Approach」开头（pdftotext 文本行 102「2. Approach」，行 110–117 为式 (1)），§2.1 实为「Training Dataset」（行 190），不含式 (1)｜引文依据：/tmp/gpt2.txt 行 102「2. Approach」；行 110–117「p(x) = … p(sn |s1 , …, sn−1 ) (1)」；行 190「2.1. Training Dataset」｜修复要求：将「主要依据」blockquote、C2 条目、F1 条目中的「§2.1 式 (1)」改为「§2 式 (1)」｜已修复：正文、来源 C2、F1 三处均改为「GPT-2 §2 式 (1)」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html 第 3 章列表第三项：「语言建模的结果正是以『每个预测单元（token/字符/字节）的平均负对数概率』报告的[C3]」与 GPT-2 §3.1 原文不符：(a) 原文称结果通常（commonly）报告为该量的「缩放或指数化形式」（scaled or exponentiated version，如困惑度），页面写成「正是以平均负对数概率报告」；(b) 原文单位列表为 character/byte/word，不含 token，页面写成「token/字符/字节」｜引文依据：/tmp/gpt2.txt 行 340–343「Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word.」｜修复要求：改为「语言建模结果通常以每个预测单元（字符、字节或词）的平均负对数概率——或其缩放、指数化形式（如困惑度）——报告」｜已修复：正文改为「通常以『每个预测单元的平均负对数概率』的缩放或指数化形式报告（预测单元通常是字符、字节或词）」，与原文 "a scaled or exponentiated version... usually a character, a byte, or a word" 一致。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html 导语「并在近乎全网规模的文本上练成这个能力」与第 3 章「语料的规模因此可以做到近乎全网文本」「整个语料几万亿 token」：均为无来源标注的规模陈述，且与可核对来源不一致——GPT-2 的 WebText 为约 40GB 精选抓取，LLaMA §2.1 语料为 1.4T token 的公开数据混合（CommonCrawl 67%、C4 15% 等），均非「近乎全网」；「几万亿 token」也与 LLaMA 的 1.4T（即 1.4 万亿）不严格一致｜引文依据：LLaMA §2.1「Overall, our entire training dataset contains roughly 1.4T tokens after tokenization.」；摘要「We train our models on trillions of tokens…using publicly available datasets exclusively」｜修复要求：导语与第 3 章的「近乎全网（规模的）文本」改为「互联网规模的文本」或「海量互联网文本」；「几万亿 token」改为「万亿量级 token」；或在「来源与范围说明」中明确标注这些规模数字为数量级估计而非来源论断｜已修复：删除「近乎全网」「几万亿」等无来源量级表述——开头改「大规模语料」，第 3 章改「远超任何人工标注数据集」「语料有多少 token 就有多少个这样的位置」，解答折叠块同步；overview 两处同步弱化。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·格式] index.html 第 2 章折叠块「展开：为什么联合概率能写成条件概率连乘」末句正文含 Unicode 数学字符「×」：「前文的联合概率 × 给定前文的下一 token 条件概率」，违反「正文无 Unicode 数学字符直接出现」｜引文依据：不适用｜修复要求：将该处「×」改为文字「乘以」，或改写为 LaTeX（如 $p(s_1,\dots,s_{i-1})\times p(s_i\mid s_1,\dots,s_{i-1})$）｜已修复：折叠块内「×」改为 $\times$（LaTeX 渲染）。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·可读性] index.html 第 4 章正文：「架桥的工作属于后训练，从 SFT 开始」——缩写 SFT 在页面正文首次出现处未展开全称，全页任何位置均未给出「监督微调（Supervised Fine-Tuning）」全称，仅有链接锚文本「SFT 监督微调」隐含对应关系｜引文依据：不适用｜修复要求：第 4 章正文首次出现处展开为「从 SFT（监督微调，Supervised Fine-Tuning）开始」｜已修复：第 4 章首次出现处改为「SFT（Supervised Fine-Tuning，监督微调）」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。

## 来源核对记录（引文依据）

按规范 2.2 逐条打开来源定位并摘录：

- C1（LLaMA §7，语言模型定义与 next token prediction）核对一致：第 7 节 Related work「Language models」小节首段：「Language models are probability distributions over sequences of words, tokens or characters (Shannon 1948; Shannon 1951). This task, often framed as next token prediction, has long been considered a core problem in natural language processing.」页面 C1 条目引文与原文一致；正文「token 序列上的概率分布」为原文三种序列单元之一的表述，且第 1 章已定义 token 涵盖词/子词/字符，未扩大适用范围。
- C2（GPT-2 式 (1) 与 tractable sampling/estimation）内容核对一致、章节号标注错误（见问题 1）：/tmp/gpt2.txt 行 110–121：「p(x) = ∏ p(sn |s1, …, sn−1 ) (1) … This approach allows for tractable sampling from and estimation of p(x) as well as any conditionals of the form p(sn−k, …, sn |s1, …, sn−k−1).」页面引用的「使 p(x) 的采样与估计可行」与原文一致。
- C3（GPT-2 §3.1，按预测单元平均报告）位置正确（§3.1 Language Modeling，行 333）、表述部分不符（见问题 2）。「按预测单元取平均」的核心含义（average negative log probability per canonical prediction unit）被原文支持。
- C4（InstructGPT §3.1 Step 1）核对一致：Step 1「Collect demonstration data, and train a supervised policy」：「Our labelers provide demonstrations of the desired behavior on the input prompt distribution」；「We then fine-tune a pretrained GPT-3 model on this data using supervised learning.」页面第 4 章引述「用监督学习在这个数据上微调预训练 GPT-3 模型」与原文一致。
- C5（指令微调综述 §1）核对一致：「One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely' (Radford et al. 2019; …).」页面第 4 章「训练目标与用户目标错位」表述与原文一致。
- C6（Goodfellow §5.5，MLE/NLL/交叉熵等价）核对一致：「Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy.」及「Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model.」页面第 3 章「最大似然在实现上即最小化负对数似然，而任何负对数似然损失都是交叉熵」与原文一致。
- F1（链式分解公式）：同 C2，公式本身与 GPT-2 式 (1) 一致；章节号标注错误见问题 1。
- F2（预训练目标公式）：页面自声明为「F1 + C3 + C6 的组合」，非直接来源论断，组合推导复核成立（连乘取负对数＝逐 token 损失之和，按位置平均由 C3 支撑，交叉熵等价由 C6 支撑）。
- 背景陈述核对（无需修改）：第 1 章「实际系统的词表规模从数万到数十万不等」与 GPT-2 §2.2 原文一致（「the 32,000 to 64,000 token vocabularies often used with BPE」「a base vocabulary of over 130,000」，/tmp/gpt2.txt 行 268–270）；「The vocabulary is expanded to 50,257」（行 320）。
- 未能核对项：规范 2.2 第 12 条「格式一致性：符合 guides/concept/style-guide.md」——本轮派发指令禁止读取 guides/ 下除 check.md 之外的文件，故该项无法核对，留待具备权限的轮次或修复者执行 validate 后确认。其余检查项均已完成。

## 数值与机械检查记录

- Python 复算（全部通过）：逐 token 损失 $-\ln 0.25=1.3863$、$-\ln 0.80=0.2231$、$-\ln 0.60=0.5108$；连乘 $0.25\times0.80\times0.60=0.12$；损失和 $2.1203$；平均 $2.1203/3=0.7068$；$-\ln 0.12=2.1203$；第 1 章分布表合计 $0.80+0.10+0.05+0.04+0.01=1.00$。与页面数值逐一相符。
- 链接与资源（全部存在）：wiki/cross-entropy/index.html、wiki/sft/index.html、../../index.html、libs/ 下 katex.min.css、katex.min.js、auto-render.min.js、prism 系列文件。overview.html 与 index.html 相互链接。页面链接的 wiki 目标仅验证存在性，未审查其内容（按派发指令）。
- 代码：页面无声称可运行的示例代码，无此项。
- 图示：页面无结构图，无此项。
- `python3 .dojo/scripts/validate.py wiki/pretraining/index.html` 返回「validation ok」。
- head 元数据：description 为纯文本、dojo:summary 含可渲染 KaTeX、dojo:type=concept、dojo:topics、dojo:tag 均在，validate.py 通过（含主题词表校验）。
- 问题块：页面级「核心问题」4 问与 4 个章节的「本章问题」（每章 2 问）均有解答折叠块，答案独立可读、与正文结论一致；核心问题答案均指明完整论证所在章节。
- 数学符号 LaTeX 化：全页扫描仅发现 1 处正文 Unicode 数学字符「×」（见问题 4）；其余数值与运算均为 KaTeX LaTeX。em-dash「——」与间隔号「·」为标点，不计。
- 折叠内容收起后正文结论完整（折叠块均为补充推导或解答，非必要前置）；构造示例已在「来源与范围说明」标注为构造数据；简化条件（自回归限定、架构黑盒、跳过参数→分布）已声明。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复
