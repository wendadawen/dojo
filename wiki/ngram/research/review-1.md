# N-gram 审查记录（第 1 轮）

- 页面版本：3857b78efbad38b566c3aaf8de23536d3d30c0b5（index.html 工作树哈希）
- 审查时间：2026-08-27 14:21
- 审查者：编排者派发的独立审查者（独立上下文，未参与写作与规划）
- 已完整阅读章节：引言、核心问题、常见误解、1. 链式法则算得动吗——马尔可夫近似从哪里来、2. 计数就是参数——三句语料上手算、3. 语料再大也不够用——稀疏与平滑、4. 从计数表到可学习查表——现代模型里的 n-gram、来源与范围说明（含全部折叠块与两个代码块）、overview.html 全文

## 机械验证结果

- `.dojo/scripts/validate.py wiki/ngram`：通过（validation ok）。
- 两个代码块均实际运行，输出与页面「预期输出」逐行一致（0.111111；行号 14,605,717 / 11,040,574 / 9,211,431；碰撞 0 次 / 792 次）。
- 前置概念链接 ../pretraining/、../cross-entropy/、../qwen3-8-flash-next-dataflow/、../../index.html 均存在；overview.html 与 index.html 互链正常。
- 本地资源 ../../libs/ 下 katex / auto-render / prism 各文件存在。
- 全文无 Unicode 数学字符出现在公式定界符之外（validate.py 通过）。

## 已核对来源论断（引文依据）

- C1：slp3-ch3.txt「An n-gram is a sequence of n words…」「in a bit of terminological ambiguity」句——一致。
- C2：§3.1.1「The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words.」——一致。
- C3：Eq.(3.3)(3.4) 链式法则、Eq.(3.9) bigram 近似均在 §3.1——一致。
- C4：§3.1.2 Eq.(3.11)(3.12) MLE 计数估计——一致。
- C5：§3.6「any finite training corpus will be missing some perfectly acceptable English word sequences」；Figure 3.1 矩阵 6 行数值逐项一致（i: 5 827 0 9 0 0 0 2；want: 2 0 608 1 6 6 5 1；to: 2 0 4 686 2 0 6 211；eat: 0 0 2 0 16 2 42 0；chinese: 1 0 0 0 0 82 1 0；food: 15 0 15 0 1 4 0 0）。
- C6：§3.6 原文为「Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events.」——论断成立，但页面所引文字与原文不符（见问题 6）。
- C7/F4：modeling_qwen4_exp.py L972-1114——行号公式（乘子相乘、异或、质数取模、加偏移）、splitmix64 派生奇数乘子（L994 `2*(...)+1`）、逐头质数词表与偏移（L1037-1042）、按 divisor 补齐（L1049-1051）、乘子上界防溢出（L987-989）均与页面一致；哈希输入含当前 token（L1095-1106，见问题 2）。
- F1/F2/F3：Eq.(3.4)、Eq.(3.9)、§3.1.2——一致。
- N1：38 million 词训练、1.5 million 词 WSJ 测试集、Perplexity 962/170/109——数值一致，但实际位于 §3.3 而非页面所标 §3.5（见问题 5）。
- N2：Figure 3.1 图注「out of V=1446 … corpus of 9332 sentences」——一致。
- N3：§3.1.2 六个概率（P(I|<s>)=2/3、P(Sam|<s>)=1/3、P(am|I)=2/3、P(do|I)=1/3、P(Sam|am)=1/2、P(</s>|Sam)=1/2）逐项一致；整句 1/9≈0.1111 复算一致。
- N4：见问题 4（本轮允许输入无法核对 checkpoint 实测数值；结构性算术已复算：16 个连续质数自 20,000,003 起合计 320,001,446 行，按 128 对齐补齐为 320,001,536 行，×160 维 = 51.2B）。

## 问题

- [重要·技术] index.html 第 3 章「0 的含义是灾难性的」句：未见组合示例「want chinese」与页面自己展示的矩阵矛盾——表中 want 行 chinese 列为 6（非零）；同句「MLE 会把 $P(\mathrm{to}\mid\mathrm{want})$ 之外的组合估成 0」也不成立（want 行有 7 个非零后继）｜引文依据：slp3-ch3.txt Figure 3.1「want2 0 608 1 6 6 5 1」（chinese 列 = 6）｜修复要求：改用矩阵中真实为零的组合作示例（如 $C(i,\mathrm{to})=0$，i 行 to 列为 0；或 $C(\mathrm{eat},i)=0$），并将「$P(\mathrm{to}\mid\mathrm{want})$ 之外的组合估成 0」改为「矩阵中为零的格子对应的组合被 MLE 估为 0」｜修复：｜复验：
- [重要·技术] index.html 核心问题 Q4 答案与第 4 章图示「前置 token」节点：哈希输入写成「当前 token 的前 $n-1$ 个 token」「当前位置之前 $n-1$ 个 token 的 id」，均不含当前 token；而第 4 章公式（$p=0,\ldots,n-1$ 含当前）、页面代码（sh[0]=当前位置）与源码均含当前 token｜引文依据：modeling_qwen4_exp.py L1095-1106「shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]; mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]」（shift=0 即当前 token）｜修复要求：两处改为「当前 token 及其前 $n-1$ 个 token（共 $n$ 个）」｜修复：｜复验：
- [重要·技术] index.html 第 4 章图示「行号」节点：「模值加该头的偏移，落在 $[0, P_h)$ 内」错误——加偏移后行号落在 $[\mathrm{off}_h, \mathrm{off}_h+P_h)$，且与正文「16 个头的质数词表拼接」表述自相矛盾｜引文依据：modeling_qwen4_exp.py L1109-1110「ngram_ids = torch.remainder(mixed_ids..., head_vocab_sizes); blocks.append(ngram_ids + head_offsets)」｜修复要求：改为「模值落在 $[0,P_h)$ 内，加该头偏移后落入该头独占的行区间」｜修复：｜复验：
- [重要·技术] 来源章节 N4：所列数值（乘子 23703573157769/20109073645365、第 2 层注入、bigram 与 trigram 各 8 头、4000 随机位置碰撞率 0.025%/0.000%、词表 248320）标注「checkpoint 实测（见数据流页 research/）」，本轮允许输入（SLP3 文本与 modeling 源码）无法定位该证据，按规范视为未核对｜引文依据：本轮无（结构部分已核对：行号公式、奇数乘子、质数表、偏移、补齐逻辑见 L972-1114；首头质数 20000003、16 头质数合计 320,001,446、128 对齐后 320,001,536、×160 维 ≈51.2B 已算术复算；两代码块运行输出与页面一致）｜修复要求：由编排者以数据流页 research/ 的实测记录逐项核对 N4 数值并在本记录补填引文依据；无法核对者降级为明确标注的推断或删除｜修复：｜复验：
- [轻微·技术] 来源章节 N1 章节定位错误：WSJ 困惑度表位于 §3.3，页面标注 §3.5｜引文依据：slp3-ch3.txt L387「3.3 Evaluating Language Models: Perplexity」、L461「Perplexity 962 170 109」、L524「3.4 Sampling sentences…」（该表在 §3.3 与 §3.4 之间）｜修复要求：N1 的「§3.5」改为「§3.3」｜修复：｜复验：
- [轻微·技术] 来源章节 C6 引文文字与原文不符：页面引「The goal of smoothing is to shave a little bit of probability mass…」，原文为「Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events.」（论断本身有据）｜引文依据：上述原文（§3.6 首段）｜修复要求：将 C6 引文替换为原文片段｜修复：｜复验：
- [轻微·技术] 第 3 章 Figure 3.1 表格只展示 6 行（缺 lunch、spend），正文称「8 个高频词的 bigram 计数矩阵」｜引文依据：源图 8 行，lunch 行「2 0 0 0 0 1 0 0」、spend 行「1 0 1 0 0 0 0 0」；已展示 6 行数值逐项一致｜修复要求：补全 lunch、spend 两行，或在表格附近注明截取了其中 6 行｜修复：｜复验：
- [轻微·格式] C2、C3、C4、C7 与 F1–F4 在正文无对应 `<sup>[..]</sup>` 上标（链式法则与 bigram 近似公式、马尔可夫假设引入句、MLE 公式、第 4 章机制与 51.2B 部署句均无引用标记），不满足「与来源章节双向对应」｜引文依据：不适用（正文检索无 [C2][C3][C4][C7][F1]–[F4] 标记）｜修复要求：在对应论断处补上标引用，或将无正文对应条目并入相关条目｜修复：｜复验：
- [轻微·格式] 核心问题四条答案以「第 1 章」…「第 4 章」编号引用章节；style-guide 要求正文引用其他章节使用章节标题｜引文依据：不适用｜修复要求：改为章节标题引用（如「见『链式法则算得动吗——马尔可夫近似从哪里来』一章」）｜修复：｜复验：
- [轻微·技术] 第 4 章正文「16 个头的质数词表拼成 320,001,536 行」：按源码算法 16 个质数表合计 320,001,446 行，320,001,536 是按 make_ngram_vocab_size_divisible_by 对齐补齐后的嵌入表行数｜引文依据：modeling_qwen4_exp.py L1049-1051「padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor」；复算 sum=320,001,446，128 对齐后=320,001,536｜修复要求：改为「质数表合计 320,001,446 行、按 128 对齐补为 320,001,536 行」或「嵌入表共 320,001,536 行」｜修复：｜复验：
- [轻微·技术] 第 4 章本章问题「为什么乘子必须是奇数、表大小用质数？」答案中「用质数作各头的模数，让不同头的哈希碰撞模式尽量互不相同」为设计动机推断，源码只体现取质数的事实、未说明动机｜引文依据：源码 L1009-1015 仅有 `_find_nth_prime_after` 取质数逻辑，无动机注释｜修复要求：改为效果性表述（「不同头使用逐个递增的质数模数，行区间与碰撞模式互不相同」）或标注为推断｜修复：｜复验：
- [轻微·可读性] overview.html「碰撞率在千分之几以下」：实测 bigram 头 0.025%（万分之 2.5），该上界比实测宽松约一个数量级且含糊｜引文依据：不适用（实测值属 N4，本轮未核对，见问题 4）｜修复要求：待 N4 核对后改为具体数值（如「bigram 头 0.025%、trigram 头 0.000%」）或「不超过千分之一」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 8
- 处置：修复（无阻断问题；4 项重要问题中，前 3 项为局部事实修正，第 4 项需编排者补核 N4 实测数值或降级标注；修复后进入第 2 轮独立审查）


## 编排者补填（第 1 轮问题 4：N4 数值核对）

- 乘子真实值 23703573157769 / 20109073645365 / 8052911324071：qwen3-8-flash-next-dataflow/research/measured-output.txt probe7 B 部分（checkpoint 按 data_offsets 下载核对），与页面一致。
- 碰撞率 0.025% / 0.000%：measured-output.txt L655-656（4000 个随机位置），与页面一致。
- 第 2 层注入、bigram/trigram 各 8 头：measured-output.txt verify_structure（ple_layer_ids=[2]、权重仅第 1 层含 ple.*）与 count_params（N-gram 表 51,200,245,760），与页面一致。
- 320,001,536 = 320,001,446 按 128 对齐补齐：measured-output.txt probe4 A 部分，与页面一致（审查者本轮已复算）。
- 处理结果：N4 各数值均有实测记录支撑，页面来源说明已补具体脚本名，无需降级。
