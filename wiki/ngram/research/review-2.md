# N-gram 审查记录（第 2 轮）

- 页面版本：index.html `bf129ade3b3c1b192f509d331e9c091796c862f6`；overview.html `75dbbec27ff0cf839305b12e9f05af99fb6bf16a`
- 审查时间：2026-08-27
- 审查者：编排者派发的独立审查者（未参与写作与前序轮次）
- 已完整阅读章节（index.html 按序）：引言、核心问题（含全部解答折叠块）、常见误解、1. 链式法则算得动吗——马尔可夫近似从哪里来（含补充折叠块与本章问题）、2. 计数就是参数——三句语料上手算（含代码折叠块与本章问题）、3. 语料再大也不够用——稀疏与平滑（含补充折叠块与本章问题）、4. 从计数表到可学习查表——现代模型里的 n-gram（含代码折叠块与本章问题）、来源与范围说明（全部小节）；overview.html 全文

本轮实际使用的输入（均为规范允许范围：两个待审页面、页面引用的外部来源、本规范及规范第 2.2 条要求对照的 style-guide.md）：

- /tmp/concept-evidence/slp3-ch3.txt（SLP3 2026-08-19 草稿第 3 章全文，全文通读）
- /tmp/qwen38fn/tf/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py（重点 L972-1114、L1117-1220）
- /tmp/qwen38fn/tf/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py（同源码树配置默认值：ngram_size=3、heads_per_ngram=8、ngram_vocab_size_base=20_000_000、make_ngram_vocab_size_divisible_by=128、seed=1234、vocab_size=248320）
- /tmp/qwen38fn/config.json（checkpoint 配置，N4 引用来源：ple_layer_ids=[2]、ple_embed_dim=2560、heads_per_ngram=8、ngram_size=3）
- /tmp/qwen38fn/headers.json（checkpoint 分片索引：ngram_embedding 共 128 分片、行数合计 320,001,536、每片 160 维）
- wiki/qwen3-8-flash-next-dataflow/research/probe7_ngram_buffers.py（N4 点名引用的脚本；与 /tmp 副本 diff 逐字节一致，未读取其中以外的任何 research 文件）
- 未读取 wiki/ngram/research/ 下任何文件；未读取任何前序审查、修复或规划记录

## 核对记录（来源论断与引文依据）

以下条目按规范第 2.2 节逐条定位核对，原文片段/关键数值作为依据：

- C1（n-gram 定义与术语二义性，§3.1）：slp3-ch3.txt L43-48 "An n-gram is a sequence of n words… But we also (in a bit of terminological ambiguity) use the word 'n-gram' to mean a probabilistic model that can estimate the probability of a word given the n-1 previous words"。一致。
- C2（马尔可夫近似直觉，§3.1.1）：L109-111 "instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words"。页面引文逐字一致。
- C3 / F1 / F2（链式法则与 bigram 近似，Eq. 3.3/3.4/3.9）：L88-98 给出 Eq.(3.3)(3.4)，L136-139 给出 Eq.(3.9) P(w1:n)≈∏P(wk|wk−1)。页面两个公式与教材一致。
- C4 / F3（MLE，§3.1.2，Eq. 3.10-3.12）：L150-159 与 L176-178，Eq.(3.12) P(wn|wn−N+1:n−1)=C(wn−N+1:n−1 wn)/C(wn−N+1:n−1)。页面公式一致；$N$ 表阶、$n$ 表当前位置的用法与教材 L129-131 一致。
- C5（有限语料漏掉合法序列与零矩阵，§3.6 与 Figure 3.1）：L664-669 "any finite training corpus will be missing some perfectly acceptable English word sequences"；L213-225 为 8 词 bigram 计数矩阵。页面节选的 6 行（i/want/to/eat/chinese/food）48 个数值逐一与原文一致（如 i 行 5,827,0,9,0,0,0,2；to 行 eat 列 686、spend 列 211；chinese 行 food 列 82）。
- C6（平滑目标，§3.6）：见问题 1——引文与 2026-08-19 草稿实际文本不符（论断本身有支持）。
- C7（哈希查表存可学习向量、16 头质数词表，modeling L1018-1114）：L1051 `self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)`（可学习参数）；L1037-1042 逐头取 `_find_nth_prime_after` 的不同质数并累计偏移。机制一致；但该编号未在正文引用（见问题 4）。
- F4（哈希行号公式，源码 L1098-1110）：`mixed_ids = shifted_tokens[0]*layer_multipliers[0]`，循环 `bitwise_xor(shifted_tokens[position]*layer_multipliers[position])`，`remainder(mixed, head_vocab_sizes)` 后 `+ head_offsets`。与页面公式 $\mathrm{id}_h(t)=((\oplus_p \mathrm{ids}_{t-p}\cdot c_p)\bmod P_h)+\mathrm{off}_h$ 逐项对应（shifted_tokens[p] 即位置 t 往前数第 p 个 token）。
- 乘子派生（splitmix64 奇数乘子）：modeling L979-995 `_build_layer_multipliers`，`2*(splitmix64(value)%half_bound)+1` 恒为奇数；L987-989 `multiplier_max = max_long // vocab_size` 保证乘积不溢出 int64。用该算法 + config（vocab 248320、seed 1234、ple_layer_index=0）独立复算得乘子 [23703573157769, 20109073645365, 8052911324071]，与页面代码前两个值逐位一致；max(m×V)=5.886e18 < 2^63−1。
- N1（WSJ 困惑度 962/170/109，§3.3）：L455-461 "trained unigram, bigram, and trigram models on 38 million words from the Wall Street Journal… perplexity of the 1.5 million word test set… Unigram 962 Bigram 170 Trigram 109"。一致；overview「同域测试条件下」成立（训练测试均为 WSJ）。962/109≈8.8，页面「降到约 1/9」成立。
- N2（Berkeley 语料 9332 句、V=1446，Figure 3.1 图注）：L204 "(a sample of 9332 sentences is on the website)"；L222-223 "out of V=1446… corpus of 9332 sentences"。一致。
- N3（I am Sam 概率，§3.1.2）：L168-175 给出 P(I|<s>)=2/3、P(Sam|<s>)=1/3、P(am|I)=2/3、P(do|I)=1/3、P(Sam|am)=1/2、P(</s>|Sam)=1/2，与页面表格六项全部一致；页面代码逐字实际运行输出 `P(<s> I am Sam </s>) = 0.111111`，与「预期输出」及手算 2/3×2/3×1/2×1/2=1/9 一致。
- N4（Qwen 表尺寸与碰撞率）：
  - 第 2 层注入、仅此一层：config.json `ple_layer_ids: [2]`（modeling L1202 按 one-indexed 匹配），一致。
  - bigram/trigram 各 8 头共 16 头：config `ngram_size: 3`、`heads_per_ngram: 8`，modeling L1025 `ngram_heads=(ngram_size-1)*heads_per_ngram`。一致。
  - 16 个质数合计 320,001,446 行、128 对齐补齐到 320,001,536：按源码算法独立复算 20,000,000 后连续 16 个质数为 [20000003, 20000023, …, 20000171]，和恰为 320,001,446；ceil(320,001,446/128)×128=320,001,536。headers.json 实测 ngram_embedding 128 分片行数合计 320,001,536、每片 160 维，与 config `ple_embed_dim: 2560`÷16=160 一致。
  - 51.2B 参数：320,001,536×160=51,200,245,760≈51.2B，与 headers.json 实际权重形状一致。
  - P_H=20000003：第一个头质数复算一致。
  - 碰撞率 0.025%/0.000%：见问题 2——量级由独立复算支持，精确数字与被引用脚本不一致。
- 第 4 章代码块逐字运行：三个位置的混合值/行号（110,194,991,134,961→14,605,717 等）、质数表 0 次碰撞、211 行小表 792 次（79.2%）均与页面「预期输出」逐字符一致。
- 第 3 章构造算例复算：5/6≈0.833、(5+1)/(6+11)=6/17≈0.353、1/17≈0.059，与页面一致；1446²=2,090,916≈209 万、1446³≈3.02e9≈30 亿，与页面一致。
- 加一平滑公式：Eq.(3.26)（L730-732）一致。
- 页面功能：validate.py 通过（`validation ok: wiki/ngram/index.html`）；本地资源 ../../libs/（katex、prism 系列）齐全；概念链接 ../pretraining/、../cross-entropy/、../qwen3-8-flash-next-dataflow/ 均存在；overview.html 与 index.html 互相链接；dojo:topics「数学基础」在 AGENTS.md 固定大类列表内；正文（KaTeX 与代码块之外）无 Unicode 数学字符；图示为 HTML 流程链（dg-flow），公式均在 HTML 节点内由 KaTeX 渲染，无 ASCII 字符画、无 SVG text 数学近似写法。
- 格式（style-guide.md）：h1/h2/h3 编号与命名、前置 section 顺序（reading-time→meta→引言→learning-goals→misconceptions→正文）、details summary 三种前缀（补充/代码/解答）、示例「构造示例」标记、来源章节固定小节命名、「本页/本文」用词均合规。问题见下。

## 问题

- [重要·技术] 来源说明 C6（index.html「来源与范围说明 > 论断与来源（C）」）：C6 把平滑目标标注为 §3.6「原文」引文 "The goal of smoothing is to shave a little bit of probability mass from some more frequent events and give it to the events we've never seen"，但 2026-08-19 草稿实际文本为 "Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events"（slp3-ch3.txt L678-681，同段前句为 "The standard way to deal with putative 'zero probability n-grams'… is called smoothing or discounting"）；页面引文是旧版草稿措辞。论断本身有来源支持，仅「原文」标记不实｜引文依据：slp3-ch3.txt L678-681 "Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events"｜修复要求：将 C6 引文替换为 2026-08-19 草稿的实际句子，保持「原文」标记 truthful｜修复：｜复验：
- [重要·技术] index.html 第 4 章正文「实测数字[N4]」句与 N4 条目、overview.html「关键结论与边界」：碰撞率数字与被引用脚本不一致且口径未说明。页面称「4000 个随机位置下 bigram 头碰撞率 0.025%、trigram 头 0.000%」并标注「同脚本」（N4 点名 probe7_ngram_buffers.py）；该脚本留存版本（数据流页 research 与 /tmp 副本逐字节一致）实际为 T=2000、torch.manual_seed(0)、仅统计 head 0 与 head 8 各一个单头、口径为 (T−unique)/T，且 measured-output.txt 中无碰撞率记录。独立复算（8 头 × 4000 均匀随机位置）显示 trigram 头同样会碰撞（合计 6 次），单头单种子观测 0.000% 不代表该类头｜引文依据：probe7_ngram_buffers.py L218-219 "torch.manual_seed(0); T = 2000"、L251-254 仅遍历 hidx=(0,8) 并按 (T-u)/T 打印碰撞率；独立复算 bigram 8 头合计 2 次、trigram 8 头合计 6 次（4000 位置）｜修复要求：正文与 N4 写明口径（单头、单随机种子、(T−unique)/T、均匀随机 token 输入），并把位置数改为与脚本一致的 2000，或更新脚本归档后改用新数字；「trigram 头 0.000%」限定为该次实测的单个头或改为量级表述；overview 的「不超过 0.03%」随之核对更新。结论「碰撞率可忽略」本身由独立复算支持，无需改动｜修复：｜复验：
- [重要·技术] index.html「常见误解」第 2 条与第 3 章「本章问题」第 1 问解答：「语料够大计数就可靠」的反驳句「万亿词语料对 trigram 仍然稀疏」及「两者的缺口不会因语料变大而关闭」缺少词表规模条件。按页面自用的 V=1446（Berkeley 语料），|V|³≈30 亿，万亿词（10^12）语料反而覆盖该空间三百余倍，句子对本页运行示例可证伪；论断仅对实际规模词表（数万词型以上，如教材 Shakespeare 例 V=29,066，|V|³≈2.45×10^13 > 10^12）成立｜引文依据：slp3-ch3.txt L588-590 "There are V^2=844,000,000 possible bigrams alone… V^4=7×10^17"（Shakespeare V=29,066）；L586 "his oeuvre is not very large as corpora go (N=884,647, V=29,066)"｜修复要求：在两处补上词表规模条件（如「对数万词型以上的实际词表」），或将「缺口不会关闭」改为「缺口是否关闭取决于语料词数与 $|V|^n$ 的比较」｜修复：｜复验：
- [轻微·格式] index.html 第 1 章 bigram 近似式与第 4 章机制段/哈希公式：来源编号 C7、F2、F4 在来源章节声明但正文无任何 <sup> 引用（脚本核对：正文引用编号为 C1-C6、F1、F3、N1-N4），违反 style-guide §6「与来源章节双向对应」；第 4 章核心机制（可学习向量、16 头质数表）与哈希行号公式在正文中无引用标记，来源不可从正文追溯｜引文依据：正文 sup 标记集合与来源章节声明集合的差集为 {C7, F2, F4}｜修复要求：在第 1 章近似式处补 <sup>[F2]</sup>，在第 4 章机制段补 <sup>[C7]</sup>、哈希公式处补 <sup>[F4]</sup>（或合并为 <sup>[C7, F4]</sup>）｜修复：｜复验：
- [轻微·可读性] index.html 第 4 章「碰撞率是这套设计的命门」段：「1000 个随机 bigram 在 211 行的小表上碰撞 79.2%，在同一量级的质数表（2000 万行）上为 0.0%」——211 与 2000 万相差五个数量级，「同一量级」所指对象不明（推测意为「与真实模型每头表同量级」）；211 行小表的选取缘由（211 为质数、作小容量对照）也未说明｜引文依据：不适用｜修复要求：改写为明确表述（如「在与真实模型每头同量级的 2000 万行质数表上为 0.0%」），并在正文或代码注释中说明 211 行小表的选取缘由｜修复：｜复验：
- [轻微·格式] index.html 多处自引章节使用「第 1/2/3 章」（核心问题答案 3 处、第 2 章「整句概率按第 1 章的连乘计算」、第 4 章「第 3 章的结论」与对照表「计数表（第 2–3 章）」），style-guide §1 要求「正文引用其他章节时使用章节标题」；且主要依据中教材章节亦称「第 3 章」，「第 N 章」存在指代混淆可能｜引文依据：不适用｜修复要求：统一改为章节标题引用（如「见『链式法则算得动吗』一章」）｜修复：｜复验：
- [轻微·格式] overview.html header lead 段：「用最近 n-1 个词近似整段历史的经典语言模型」中「n-1」为纯文本，同页「定义」小节均用 $n-1$；style-guide §11 要求任何位置数学符号一律 LaTeX（index.html 的 meta description 为纯文本属正常，不在此列）｜引文依据：不适用｜修复要求：lead 段改为「用最近 $n-1$ 个词」｜修复：｜复验：
- [轻微·格式] index.html「来源与范围说明 > 辅助解释与类比边界」：描述「『预测下一个词像接话』的开篇直觉」，但正文并无「接话」表述——类比边界描述的对象在正文中不存在（实际开篇为 Walden Pond 例）｜引文依据：不适用｜修复要求：将该条改写为与实际开篇一致的描述（如「Walden Pond 押词直觉仅用于引入问题」），或删除该条｜修复：｜复验：
- [轻微·可读性] index.html 第 3 章困惑度定义句：「每词交叉熵损失的指数」未说明底数；教材 §3.7 定义为 Perplexity(W)=2^H(W)（H 以 log2 计）｜引文依据：slp3-ch3.txt L1036-1038 "The perplexity of a model P on a sequence of words W is now formally defined as 2 raised to the power of this cross-entropy"｜修复要求：补底数或改为「交叉熵（以 2 为底对数）的 2 的幂」一类与教材一致且与交叉熵页衔接的表述｜修复：｜复验：
- [轻微·可读性] index.html 第 1 章首段：「历史 $h$ 是当前词之前的全部 token」为 token 首次出现，词与 token 的关系延后到「简化条件及其限制」才说明｜引文依据：不适用｜修复要求：首次出现处加一句括注（如「token（模型的离散处理单位，本页与『词』混用时见文末简化条件」）或将该句改用「词」表述｜修复：｜复验：
- [轻微·技术] index.html 第 4 章「本章问题」第 2 问解答：「用质数作各头的模数，让不同头的哈希碰撞模式尽量互不相同」为无来源支持的动机归因——源码（L1037-1042）只体现各头取不同质数这一事实，未说明设计动机｜引文依据：modeling_qwen4_exp.py L1039 "size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)"（无动机注释）｜修复要求：降级为推断表述（如「各头取不同质数，一个直接后果是不同头的碰撞位置互不相同——设计动机源码未说明」）或删去动机从句｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 8
- 处置：修复。无阻断问题；三条重要问题（C6 引文不符、碰撞率数字与被引用脚本参数不一致且口径未说明、稀疏论断缺词表规模条件）修复后需复验；轻微问题逐条修复或在下一轮记录接受理由。页面主体（链式法则与马尔可夫近似、MLE 手算、稀疏与平滑、WSJ 数字、I am Sam 矩阵、Qwen 表结构与尺寸、两段可运行代码）已逐条对来源核实一致，范围与大纲无需返回规划。
