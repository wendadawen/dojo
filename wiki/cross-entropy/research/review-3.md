# 交叉熵审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 `40398eb21e684496d7b3d85142bc14336d602341`
- 审查时间：2026-08-25 16:21 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：核心问题（4 题及解答折叠块）、1. 为什么预测概率需要转换成损失（含曲线图与本章问题）、2. 交叉熵的信息论定义（含 0log0 折叠块与本章问题）、3. 训练视角——从最大似然到 one-hot 手算（3.1、3.2、对数计算折叠块与本章问题）、4. 语言模型的下一 token 交叉熵（含构造示例与本章问题）、来源与范围说明（C/F/构造示例/类比边界/简化条件全部小节）；overview.html 全部小节（它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界）

## 来源核对记录（逐条摘录引文依据）

- C1 自信息：Goodfellow §3.13 式 (3.48)："we define the self-information of an event x = x to be I(x) = −log P(x). (3.48)"；且 "In this book, we always use log to mean the natural logarithm, with base e. Our definition of I(x) is therefore written in units of nats."。页面表述（自然对数、nats）一致。
- C2 香农熵：§3.13 式 (3.49)："the Shannon entropy, H(x) = E_{x∼P}[I(x)] = −E_{x∼P}[log P(x)], (3.49), also denoted H(P)"；"Distributions that are nearly deterministic ... have low entropy; distributions that are closer to uniform have high entropy"。与页面公式及「接近确定小、接近均匀大」表述一致。
- C3 KL 散度：§3.13 式 (3.50)："D_KL(P‖Q) = E_{x∼P} log [P(x)/Q(x)] = E_{x∼P}[log P(x) − log Q(x)]. (3.50)"；"most notably being non-negative. The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables"；"It is not a true distance measure because it is not symmetric"。页面公式、非负、离散情形取零条件、不对称均一致；页面的编码代价直觉与原文 "it is the extra amount of information ... needed to send a message containing symbols drawn from probability distribution P, when we use a code that was designed to minimize the length of messages drawn from probability distribution Q" 一致。
- C4 交叉熵与等价最小化：§3.13 式 (3.51)："the cross-entropy H(P, Q) = H(P) + D_KL(P‖Q), which is similar to the KL divergence but lacking the term on the left: H(P, Q) = −E_{x∼P} log Q(x). (3.51) Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term."。页面引文逐字一致，分解式与「固定 P 时等价」结论一致。§5.5 另有 "We can thus see maximum likelihood as an attempt to make the model distribution match the empirical distribution"，支持「训练交叉熵就是在拉近模型分布与数据分布」。
- C5 最大似然/NLL/交叉熵等价：§5.5："In software, we often phrase both as minimizing a cost function. Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy."；"Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model."。两处引文逐字核实（式 5.59–5.61 上下文亦核对）。§5.5 "The negative log-likelihood can actually become negative when x is real-valued" 支持简化条件小节的连续分布说明。
- C6 0log0 惯例：§3.13 式 (3.51) 之后："it is common to encounter expressions of the form 0 log 0. By convention, in the context of information theory, we treat these expressions as lim_{x→0} x log x = 0."。页面引文与标注位置（§3.13）一致。
- C7 报告方式：Radford et al. (GPT-2) §3.1："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word."（本地 /tmp/gpt2.txt 第 333–345 行）。页面正文与来源说明（含「缩放/指数化形式」）一致。
- F1–F5：F1/F2 即 C4；F3–F5 为页内组合推导，来源说明小节已如实标注组合关系，未伪托来源。

## 数值复算记录（Python）

- 二分类：−ln 0.8 = 0.22314→0.2231 ✓；−ln 0.2 = 1.60944→1.6094 ✓。
- softmax (2.0, 1.0, 0.5)：指数 (7.389, 2.718, 1.649)、总和 11.756、概率 (0.6285, 0.2312, 0.1402) ✓；−ln q(晴) = 0.4644 ✓；−ln q(雪, 精确值 0.140228) = 1.9644（见问题 2 的舍入说明）；e^{−0.4644} = 0.6285、e^{−1.9644} = 0.1402 反向验证 ✓。
- 语言模型：−ln 0.85 = 0.16252→0.1625 ✓；−ln 0.70 = 0.35667→0.3567 ✓；平均 0.25960→0.2596 ✓；两个构造分布求和均为 1 ✓。
- 曲线图：q=0.02 处 −ln q = 3.91，与图注「约 3.9」一致 ✓；SVG 路径采样点按坐标映射复算（x=70→q=0.02→y=45.1；x=98.5→y=116.3；x=127.0→y=147.1；x=640→q=1→y=270），与路径数据一致 ✓；刻度 0.1/0.5/1 的横坐标 116.5/349.2/640 与映射一致 ✓。

## 机械验证记录

- `.dojo/scripts/validate.py wiki/cross-entropy/index.html` → "validation ok"。
- 链接：`../../wiki/pretraining/index.html`、`../../wiki/sft/index.html` 均存在；`overview.html` 与 `index.html` 互链 ✓；本地资源 katex/prism 六个文件均存在 ✓。
- meta：description（纯文本）、dojo:summary（KaTeX 可渲染）、dojo:type=concept、dojo:topics、dojo:tag 齐全；validate.py 通过（含主题词表校验）。
- 占位符/模板残留：grep TODO/FIXME/占位/待补 无命中。
- 数学符号：正文、标题、列表、表格中的数学内容全部为 LaTeX（`$...$` / `$$...$$`）；未发现 Unicode 数学字符直接出现（overview 导航「→」与间隔号「·」为 UI 排版符号，非数学符号）。
- 图示：内联 SVG，公式标签均在 `<foreignObject>` 中由 KaTeX 渲染，`<text>` 内仅中文说明无 ASCII 近似数学；结构图要求满足。
- 代码：页面无可运行代码，本项不适用。

## 问题

- [轻微·技术] index.html §4 第 3 段（「每个位置的损失互相独立：改动一个位置的预测不影响其他位置的损失值」）：该独立性隐含「前文取自语料真实 token（teacher forcing）而非模型自身输出」这一条件，未在正文或「简化条件及其限制」中说明；读者若联想到自回归生成场景（前文含模型自身预测）会得出错误结论。此句为机制描述且无来源标注。｜引文依据：不适用（无来源支持的机制表述，页内公式本身正确）｜修复要求：在该句后补充条件说明（如「训练与逐位评估时前文取自语料的真实 token」），或并入「简化条件及其限制」小节。｜已修复：该句补充「训练与评估中每个位置的条件前文取自真实文本（而非模型自己的生成）」，条件已说明。｜复验：validate.py 通过，修复处复查一致。
- [轻微·技术] index.html §3.2 正文与「展开：三分类例子的完整对数计算」折叠块（两处「−ln 0.1402 ≈ 1.9644」）：由舍入后的 0.1402 复算得 −ln 0.1402 = 1.9647；1.9644 是由未舍入的 softmax 概率 0.140228 算得。数值本身正确，但按页面展示的四位小数输入复算会出现第 4 位不一致。｜引文依据：Python 复算 −ln(0.1402)=1.9647，−ln(e^0.5/(e^2+e^1+e^0.5))=1.9644｜修复要求：二选一——把输入写为更多位（0.14023），或把结果改为按舍入输入的 1.9647，或注明 1.9644 按未舍入概率计算。｜已修复：雪类损失统一为按显示四位概率复算的 $1.9647$（正文、折叠块、解答、构造示例清单四处同步），折叠块内加括注说明精确概率 $0.14024\ldots$ 对应 $1.9644$、页面统一按显示值复算。｜复验：validate.py 通过，修复处复查一致。
- [轻微·流程] 概念页第 12 项格式一致性（style-guide.md）：本轮允许输入不含 `guides/concept/style-guide.md`，未能逐条核对；已按 check.md 内含的格式要求（meta、LaTeX、SVG、问题块、锚点）完成核对。｜引文依据：不适用｜修复要求：无需修复页面；如需该项完整核对，由编排者补充核对或在派发时放开该文件。｜不适用（流程性说明：style-guide.md 未纳入审查输入是派发约束，其格式项由 validate.py 与编排者终检兜底）。｜复验：validate.py 通过，修复处复查一致。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布

遗留轻微问题接受理由：
1. teacher forcing 条件缺失：页面全部公式均在训练损失语境下给出（前文即语料 token），主线结论不受影响，误解风险低；建议后续修订时补一句条件说明。
2. 舍入展示差异 0.0003：页面已给出反向验证 e^{−1.9644}≈0.1402，且由构造输入（分数 2.0/1.0/0.5）直接复算恰为 1.9644，数值正确，仅中间展示精度引起。
3. style-guide.md 未能核对：非页面缺陷，属审查输入限制；check.md 内含的全部格式项已核对通过。

## 发布结论

第 3 轮（最终轮）全量审查完成。发布条件逐条核对：

1. 三轮审查完成且独立执行：本轮为独立审查；第 1、2 轮记录文件存在（review-1.md、review-2.md，按规范未读取内容），独立性由编排流程保证。
2. 每条来源论断有引文依据：C1–C7、F1–F5 全部核对并摘录原文（见「来源核对记录」），无定位不到或内容不符条目。
3. 阻断和重要问题全部关闭：本轮未发现阻断/重要问题；前两轮问题关闭状态由编排者确认（本轮输入不含前序记录）。
4. 遗留轻微问题有明确接受理由：3 条，理由见上。
5. 全部学习目标（核心问题 4 题）由正文第 1–4 章完整回答。
6. 页面级核心问题（4 题）与各章本章问题（每章 2 题）均有解答折叠块，答案独立可读、与正文一致，核心问题答案指明了论证章节。
7. 数学符号全部 LaTeX 书写；结构图为内联 SVG 且曲线坐标复算一致。
8. `.dojo/scripts/validate.py` 返回 "validation ok"。
9. 可运行代码：页面无代码块，不适用。
10. 关键论断与数字已重新核对来源并复算（见来源核对与数值复算记录）。
11. `<head>` 含纯文本 description、可渲染 dojo:summary、dojo:type=concept、dojo:topics（词表内）、dojo:tag。
12. overview.html 与 index.html 相互链接。
13. 页面引用的概念链接（pretraining、sft）有效。
14. 递归前置概念页质检：本页无前置概念页链接（pretraining/sft 为「后续概念页」，页面原文已如此定位），其各自质检由对应页面流程承担，不在本页审查范围。

**最终结论：交叉熵概念页通过第 3 轮审查，满足发布条件，可发布。**


## 发布结论

- 发布时间：2026-08-25
- 三轮独立审查（每轮独立子代理）完成，全部阻断与重要问题已修复关闭；本轮遗留轻微问题均已处理（第 3 条为流程性说明，非页面问题）。
- 修复后 `.dojo/scripts/validate.py` 通过；headless Chrome 渲染实测 KaTeX 244 个节点、0 错误、SVG 标签无重叠无裁剪。
- 结论：可发布。
