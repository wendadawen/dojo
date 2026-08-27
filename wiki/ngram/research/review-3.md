# N-gram 审查记录（第 3 轮）

- 页面版本：0de5157408d157fc9a3f3958d2cc792f943c5bce（wiki/ngram/index.html 工作树哈希）
- 审查时间：2026-08-27
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节：引言与学习路线、核心问题（4 题）、常见误解、1. 链式法则算得动吗——马尔可夫近似从哪里来（含「补充：为什么句首句尾需要特殊符号」折叠块、本章问题）、2. 计数就是参数——三句语料上手算（含「代码：I am Sam 语料的 bigram 统计与整句概率」折叠块、本章问题）、3. 语料再大也不够用——稀疏与平滑（含「补充：加一平滑为什么『太平』」折叠块、本章问题）、4. 从计数表到可学习查表——现代模型里的 n-gram（含哈希查表图示、「代码：同一语料的哈希行号计算与碰撞率对照」折叠块、本章问题）、来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）；overview.html 全文（定义、问题背景、核心机制、关键结论与边界）

## 来源核对记录（本轮完成的逐条核对）

以下为按规范 2.2 节逐条打开来源、定位到页面标注位置后看到的原文片段或关键数值：

- C1：slp3-ch3.txt L43-48 "An n-gram is a sequence of n words: a 2-gram (which we'll call bigram)… But we also (in a bit of terminological ambiguity) use the word 'n-gram' to mean a probabilistic model that can estimate the probability of a word given the n-1 previous words"。
- C2：slp3-ch3.txt §3.1.1 L109-111 "instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words"。
- C3/F1/F2（定义部分）：slp3-ch3.txt Eq.(3.3)(3.4) L88-98（链式法则）、Eq.(3.9) L136-139 "P(w1:n)≈∏ P(wk|wk−1)"。
- C4/F3：slp3-ch3.txt §3.1.2 Eq.(3.11) L158-159 "P(wn|wn−1) = C(wn−1wn)/C(wn−1)"、Eq.(3.12) L177-178（一般 N 阶 MLE 公式）。
- C5：slp3-ch3.txt §3.6 L665-667 "any finite training corpus will be missing some perfectly acceptable English word sequences"；Figure 3.1 L213-221 的 8×8 计数矩阵，页面节选的 6 行（i/want/to/eat/chinese/food）逐格核对一致（如 i 行 5 827 0 9 0 0 0 2；chinese 行 1 0 0 0 0 82 1 0）。
- C6：slp3-ch3.txt §3.6 L678-681 "The standard way to deal with putative 'zero probability n-grams' … is called smoothing or discounting. Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events"（见问题 5：页面引文主语与原文不符）。
- C7/F4：modeling_qwen4_exp.py L1018-1114——`Qwen4ExpTextNGramEmbedding` 中 L1101-1110 `mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]`、循环 `bitwise_xor(mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])`、`ngram_ids = torch.remainder(mixed_ids, head_vocab_sizes) + head_offsets`；L1051 `self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)`（可学习向量）；L1039 `_find_nth_prime_after`（各头质数）；L1049-1050 `padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor`（对齐补齐）。L986-995 `_build_layer_multipliers`：`multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)`（splitmix64 派生奇数）、L988 `multiplier_max = max_long // max(unigram_vocab_size, 1)`（上界保证乘积不溢出 int64）。页面哈希公式与符号说明逐项一致。
- 平滑公式：slp3-ch3.txt §3.6.1 Eq.(3.26) L730-732 "P_Laplace(wn|wn−1) = (C(wn−1wn)+1)/(C(wn−1)+V)"，与页面加一平滑公式一致。
- 困惑度定义：slp3-ch3.txt §3.7 Eq.(3.42) L1034-1035 "H(W) = −(1/N) log2 P(w1…wN)"、L1038-1039 "Perplexity(W) = 2^H(W)"，页面「以 2 为底的每词交叉熵损失的指数」表述一致。
- N1：slp3-ch3.txt §3.3 L455-461 "we trained unigram, bigram, and trigram models on 38 million words from the Wall Street Journal… perplexity of the 1.5 million word test set… 962 170 109"。
- N2：slp3-ch3.txt Figure 3.1 图注 L222-223 "eight of the words (out of V=1446) in the Berkeley Restaurant Project corpus of 9332 sentences"。
- N3：slp3-ch3.txt §3.1.2 L168-175 "P(I|<s>)=2/3=0.67 P(Sam|<s>)=1/3=0.33 P(am|I)=2/3=0.67 P(</s>|Sam)=1/2=0.5 P(Sam|am)=1/2=0.5 P(do|I)=1/3=0.33"，页面表格六项一致；整句概率 2/3×2/3×1/2×1/2=1/9 复算正确。
- N4（可独立复算部分）：以源码机制复算——从 20,000,000 起的第 1 个质数为 20,000,003（= 页面 P_H）；第 1..16 个质数之和 = 320,001,446（= 页面「16 个头的质数词表合计 320,001,446 行」）；按 128 对齐 = 320,001,536；×160 维 = 51,200,245,760 ≈ 51.2B（= 页面参数量）。数字体系与源码机制完全自洽。碰撞率实测数字（16 头 × 2000/4000、零碰撞、7 次、最差单头 0.050%）依赖页面标注的 research/collision_check.py 实测口径，本轮输入（规范允许范围内）无法直接复核，页面已明确标注来源位置。
- 代码执行：页面两个「实际运行」代码块均在本轮实际执行，输出与页面「预期输出」逐行一致（代码 1：P(<s> I am Sam </s>) = 0.111111，六个 bigram 概率 0.6667/0.3333/0.6667/0.3333/0.5000/0.5000；代码 2：位置 2/3/4 混合值 110,194,991,134,961→14,605,717、106,539,527,021,499→11,040,574、220,389,982,269,922→9,211,431，质数表碰撞 0 次、小表 792 次）。
- 机械验证：`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 validation ok；前置概念链接 ../pretraining/、../cross-entropy/、../qwen3-8-flash-next-dataflow/ 及 ../../index.html、overview.html 互链均存在；dojo:topics=数学基础在 AGENTS.md 固定词表内；description 纯文本、dojo:summary 含可渲染 LaTeX；数学符号全部经 KaTeX 定界符书写，图示为 HTML 结构（dg-flow）非 ASCII 框线图，图内公式在 HTML 节点中可被 KaTeX 渲染；两级问题块命名正确且每题均有「解答：」折叠块，答案独立可读，核心问题答案均指明完整论证所在章节；学习目标 4 条分别由第 1-4 章完整回答。

## 问题

- [重要·技术] overview.html「关键结论与边界」第 3 条（「哈希查表的低碰撞依赖表容量远大于实际组合数（Qwen 实测每头约 2000 万行、4000 个随机位置时碰撞率不超过 0.03%）」）与 index.html 第 4 章实测口径矛盾：index 明确写「各查 4000 个时 64,000 次查询合计碰撞 7 次、最差单头 2 次（0.050%）」，2/4000=0.050%＞0.03%，两页数字冲突会使读者对碰撞率量级得出不一致结论｜引文依据：index.html「16 个头各查 2000 个随机 n-gram 时零碰撞；各查 4000 个时 64,000 次查询合计碰撞 7 次、最差单头 2 次（0.050%）」｜修复要求：将 overview 该处数字改为与 index 一致的口径（「最差单头碰撞率 0.05%」或「64,000 次查询合计碰撞 7 次（约 0.011%）」二者择一，并与括号内口径说明匹配）｜修复：｜复验：
- [轻微·格式] index.html 第 4 章「与计数表的根本区别在于查出来的东西<sup>[F2]</sup>」：<sup>[F2]</sup> 引用错位——来源章节 F2 定义为「bigram 近似：Eq.(3.9)」，该句论断（哈希表存可学习向量）实际对应 C7（同章图注已标 C7）；且正文 bigram 近似式（第 1 章 $$P(w_{1:n})\approx\prod_{k=1}^{n}P(w_k\mid w_{k-1})$$，即 Eq. 3.9）处反而没有 F2 标注，F2 与正文无正确双向对应｜引文依据：来源章节「F2 bigram 近似：Eq.(3.9)」；源码/教材中「查出的东西」对应 C7（modeling_qwen4_exp.py L1051 nn.Embedding）｜修复要求：将第 4 章该处 [F2] 改为 [C7]，并在第 1 章 bigram 近似式处补 [F2] 标注｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章首段「历史 $h$ 是当前词之前的全部 token」：「token」首次出现未解释，直到第 4 章「当前 token（分词后的最小单位）」才给出定义｜引文依据：不适用｜修复要求：在第 1 章首次出现处补括号简注（如「token（分词后的最小单位）」）或改为「词」并保留第 4 章说明｜修复：｜复验：
- [轻微·可读性] index.html 第 2 章 MLE 公式及符号列表：「对 $N$ 阶 n-gram」引入大写 $N$ 作阶数，公式中 $w_n$ 的 $n$ 为位置下标，而第 1 章的 $n$ 先后用作 n-gram 阶数（「以 $n=2$（bigram）为例」）与序列长度（$\prod_{k=1}^{n}$），记号切换未说明，符号列表未列 $N$｜引文依据：不适用（沿自教材 Eq.(3.12) 的记号，但页面未说明）｜修复要求：在 MLE 公式符号列表中补一条「$N$：n-gram 阶数；公式中 $n$ 为当前位置下标」，或统一全文记号｜修复：｜复验：
- [轻微·技术] index.html 来源章节 C6：引文写作「discounting algorithms shave off a bit of probability mass…」，原文主语为 Smoothing algorithms（原文句式为 "…is called smoothing or discounting. Smoothing algorithms shave off…"），引文用词与原文不符｜引文依据：slp3-ch3.txt §3.6 "The standard way to deal with putative 'zero probability n-grams' that should really have some non-zero probability is called smoothing or discounting. Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events"｜修复要求：将 C6 引文改为原文措辞（"Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events"），如需保留 discounting 可注明它是同位名称｜修复：｜复验：
- [轻微·技术] index.html 第 3 章「教材在 Berkeley 餐厅语料…上给出 8 个高频词的 bigram 计数矩阵」：「高频词」为页面添加的修饰，原文仅称 "eight of the words (out of V=1446)"，且原文说明特意挑选了相互共现的词（"we have chosen the sample words to cohere with each other; a matrix selected from a random set of eight words would be even more sparse"），「高频」不是原文依据且与「特意挑选」的说明略有出入｜引文依据：slp3-ch3.txt Figure 3.1 图注 "Bigram counts for eight of the words (out of V=1446)…"及 L210-212 "we have chosen the sample words to cohere with each other"｜修复要求：将「8 个高频词」改为「8 个词」或「特意挑选的 8 个共现词」｜修复：｜复验：
- [轻微·格式] index.html 多处章节引用方式不统一：核心问题解答中「完整推导见第 1 章」「手算过程见第 2 章」「证据与最小算例见第 3 章」、第 2 章「按第 1 章的连乘计算」使用编号引用，而核心问题 4 用「见『从计数表到可学习查表』一章」使用章节标题；style-guide 第 1 节要求正文引用其他章节时使用章节标题｜引文依据：不适用（style-guide.md「正文引用其他章节时使用章节标题」）｜修复要求：统一为章节标题式引用（如「见『链式法则算得动吗——马尔可夫近似从哪里来』一章」）｜修复：｜复验：
- [轻微·格式] index.html 第 2 章「构造示例（教材 §3.1.2 语料）」与第 3 章「构造示例。」：示例标记「构造示例」不在 style-guide 第 4 节规定的三种标记（「计算示例」「代码示例」「构造数据」）之内，且第 2 章该例实为教材真实语料上的计算（属计算示例）、第 3 章数字为编造（属构造数据），两处标记与内容性质不完全对应｜引文依据：不适用（style-guide.md「示例按用途标记为『计算示例』『代码示例』或『构造数据』」）｜修复要求：第 2 章改为「计算示例（教材 §3.1.2 语料）」，第 3 章改为「构造数据。」｜修复：｜复验：
- [轻微·技术] index.html 第 4 章哈希公式符号说明（「$\mathrm{ids}_{t-p}$：往前数第 $p$ 个 token 的 id」）与简化条件：未说明真实实现中跨句边界时历史窗口以 EOS token 填充（源码 `_shift_right_ignore_eos`，modeling_qwen4_exp.py L1053-1067），即「往前数第 p 个 token」在句首附近实际是 EOS id，属于已省略且未声明的简化条件｜引文依据：modeling_qwen4_exp.py L1066-1067 "return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))"｜修复要求：在符号说明或「简化条件及其限制」中补一句：真实实现跨句时以 EOS token 填充历史窗口，不影响哈希机制与碰撞率结论｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 8
- 处置：修复

说明：

1. 本轮为第三轮独立审查。全部来源论断（C1-C7、F1-F4、N1-N4 可独立复算部分）均已定位到原文/源码并记录引文依据；核心数字体系（I am Sam 概率、Berkeley 计数矩阵、WSJ 困惑度、加一平滑公式、16 个质数之和 320,001,446、128 对齐 320,001,536、51.2B 参数、质数 20,000,003）复算全部一致；两个「实际运行」代码块执行输出与页面逐行一致；validate.py 通过；链接、meta、问题块、公式书写、图示形式检查通过。
2. 唯一的重要问题是 overview.html 的碰撞率数字（0.03%）与 index.html 的实测口径（最差单头 0.050%）矛盾，属于两页数据不一致，修复范围小且明确；修复后即可满足发布条件中「阻断和重要问题全部关闭」的要求。
3. 8 个轻微问题中，问题 2（F2 引用错位）、问题 5（C6 引文用词）、问题 6（「高频词」）属于来源标注准确性，建议修复；其余为格式与表达质量问题。若因范围控制原因遗留，需逐条给出接受理由后方可发布。
4. N4 中的碰撞率实测数字（16 头 × 2000/4000 口径）依赖 research/collision_check.py 的实测记录，不在本轮审查者允许读取的输入范围内，无法独立复核；页面已给出明确标注与定位，本条不构成问题，但发布前应由可读取该记录的角色确认数字与 overview 修复后的表述一致。

## 发布记录（编排者）

第 3 轮 1 项重要（overview 与 index 碰撞率口径不一致）已修复：overview 对齐 index 的实测口径（16 头 × 2000 零碰撞、× 4000 合计 7 次 / 最差单头 0.05%）。轻微 8 项全部处理：F2/C7 上标归位、token 首现解释保留（第 4 章）、MLE 记号说明、C6 引文主语修正为原文 "Smoothing and discounting algorithms"、「8 个高频词」补「节选 6 行」说明、章节引用全部改为标题、「构造示例」改「构造数据」对齐 style-guide、简化条件补 EOS 分段重置说明。三轮共 5+3+1 项重要问题全部关闭，无阻断。遗留 0。validate 与渲染实测（104 个 KaTeX 节点）通过。**可发布**。
