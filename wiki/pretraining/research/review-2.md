# 语言模型预训练审查记录（第 2 轮）

- 页面版本：index.html `71aefc95f9b81edb62f72542a6941f94bc1197b2`（overview.html `a34e08816f6a72b242379b754ad8ea366f04780a`）
- 审查时间：2026-08-25 16:12 CST
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：标题与主要依据引言 → 引言 → 核心问题（4 题含解答折叠块）→ 1. 语言模型是什么（含构造示例表、本章问题）→ 2. 一条文本的概率如何逐 token 分解（含手算示例、乘法公式展开折叠块、本章问题）→ 3. 预训练的目标（含公式符号表、本章问题）→ 4. 预训练产出基座模型，行为对齐交给后训练（含本章问题）→ 来源与范围说明（论断与来源 C1–C6、公式与来源 F1–F2、构造示例、辅助解释与类比边界、简化条件）；overview.html 全文。

## 来源核对记录（C/F 条目，均已打开来源定位）

- C1（LLaMA §7）：核对通过。原文（ar5iv 2302.13971，§7 Related work「Language models」段）："Language models are probability distributions over sequences of words, tokens or characters… This task, often framed as next token prediction, has long been considered a core problem in natural language processing."
- C2 / F1（GPT-2 §2 式 (1)）：核对通过。原文（/tmp/gpt2.txt 行 102–121）："2. Approach … p(x) = ∏ p(sn|s1,…,sn−1) (1) … This approach allows for tractable sampling from and estimation of p(x)"。注意：式 (1) 位于 §2 开头、§2.1 之前。
- C3（GPT-2 §3.1）：核对通过。原文（行 333–343，"3.1. Language Modeling"）："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word."
- C4（InstructGPT §3.1 Step 1）：核对通过。原文（ar5iv 2203.02155，§3.1）："Our labelers provide demonstrations of the desired behavior on the input prompt distribution … We then fine-tune a pretrained GPT-3 model on this data using supervised learning."
- C5（综述 §1）：核对通过。原文（arxiv 2308.10792v10，§1 第一段）："One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'."
- C6（Goodfellow et al. §5.5）：核对通过。原文（deeplearningbook.org/contents/ml.html，5.5 节）："Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model."
- F2：组合论断（F1+C3+C6），组合方式已在页面声明，各组成部分均核对通过。

## 数值复算记录（Python）

- 分布表求和：0.80+0.10+0.05+0.04+0.01 = 1.00 ✓
- 序列概率：0.25×0.80×0.60 = 0.12 ✓
- 逐 token 损失：−ln0.25=1.386294、−ln0.80=0.223144、−ln0.60=0.510826，页面显示 1.3863/0.2231/0.5108 ✓
- 精确总和 2.120264 → 2.1203 ✓；但页面显示值直接相加 1.3863+0.2231+0.5108 = 2.1202（见问题 1）
- 平均 2.120264/3 = 0.706755 → 0.7068 ✓；−ln0.12 = 2.120264 ✓

## 机械检查记录

- 链接目标：wiki/cross-entropy/index.html、wiki/sft/index.html、../../index.html、overview.html 均存在 ✓；libs/ 下 7 个本地资源均存在 ✓
- overview.html ↔ index.html 相互链接 ✓
- meta：description（纯文本）、dojo:summary（含可渲染 LaTeX）、dojo:type=concept、dojo:topics、dojo:tag 均在 ✓
- `.dojo/scripts/validate.py wiki/pretraining/index.html` 返回 "validation ok" ✓
- 页面无可运行代码块，数值示例静态复算通过；无 ASCII 框线图，无内联 SVG 图示
- Unicode 数学字符扫描（去除 LaTeX 后）：index.html 仅「输出分布 → 取一个 token → 拼进前文」一句两个 →（见问题 5）；overview.html 仅导航 UI 的 ←/→ 与标题间隔号 ·（非正文数学符号，不记问题）

## 问题

- [轻微·技术] index.html 第 2 章「手算示例」段：损失总和算式「1.3863+0.2231+0.5108=2.1203」按显示值相加为 2.1202，等式在末位不成立（精确和 2.120264 舍入为 2.1203；按显示值算 2.1202/3=0.7067，与页面平均 0.7068 也不一致）｜引文依据：Python 复算 −ln0.25=1.386294、−ln0.80=0.223144、−ln0.60=0.510826、总和 2.120264｜修复要求：将「=」改为「≈」或在句中注明总和与平均由未舍入值计算后再舍入，使算式链自洽｜已修复：算式改为 $1.3863+0.2231+0.5108\approx 2.1203$。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·来源] index.html「主要依据」引言：GPT-2 定位写作「§2.1 式 (1)」，实际式 (1) 在 §2（Approach）开头、§2.1（Dataset）之前，且与来源清单 C2 条目「§2 式 (1)」自相矛盾｜引文依据：/tmp/gpt2.txt 行 102「2. Approach」→ 行 110–117 式 (1)｜修复要求：主要依据栏统一改为「§2 式 (1)」｜已修复：「主要依据」blockquote 改为「§2 式 (1)」，与 C2 一致。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·技术] index.html 第 1 章：「实际系统的词表规模从数万到数十万不等」——允许来源仅支持「数万」端，「数十万」端无来源依据且未标注推断｜引文依据：GPT-2 §2.2 "the 32,000 to 64,000 token vocabularies often used with BPE"、§2.3 "The vocabulary is expanded to 50,257"（五个允许来源中无「数十万」词表的实际系统数值）｜修复要求：收窄为来源可支持的表述（如「如 GPT-2 的 50,257、BPE 常用的 32,000–64,000」），或将「数十万」端降级为明确标注的推断｜已修复：改为「实际系统的词表规模以万计（GPT-2 的词表为 $50{,}257$ 个 token）」并新增来源条目 C7（GPT-2 §2.3 "The vocabulary is expanded to 50,257."）；本章问题解答同步。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·技术] index.html 第 4 章首段：「它很可能接着生成另一条类似的请求……在它的训练分布里，这句请求后面跟着的通常是更多网页内容」——对基座模型具体行为的预测及其机制归因无来源标注，也未登记在「构造示例/辅助解释」清单中｜引文依据：InstructGPT §1 仅支持目标错位（"predicting the next token on a webpage from the internet … the language modeling objective is misaligned"），未描述该具体续写行为；InstructGPT §1、§2 中均无逐字支持｜修复要求：将该句在「来源与范围说明」登记为说明性构造/推断，或改写为直接对应 C5 的表述后删除行为预测细节｜已修复：第 4 章该句改为「按语料分布续写意味着模型倾向生成语料风格的后续文本，而这通常不是指令所要求的回答（此为错位论断 C5 的行为推演）」，并在「辅助解释与类比边界」登记该场景为解释性推演、非实验观察。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·格式] index.html 第 1 章本章问题解答 1：「输出分布 → 取一个 token → 拼进前文」句中两个 Unicode 箭头 →（U+2192）直接出现在正文，违反规范 2.2.9「正文无 Unicode 数学字符直接出现」（此处为流程指示用法）｜引文依据：不适用｜修复要求：改为顿号/文字连接（如「输出分布、取一个 token、拼进前文」），或改用文字描述步骤顺序｜已修复：解答折叠块箭头改为「输出分布、取一个 token、拼进前文」（顿号连接）。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：修复（全部为轻微问题，逐条修复并复验后即可进入第 3 轮；无需返回规划）
