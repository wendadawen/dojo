<!-- review-meta
round: 6
page: wiki/ngram/index.html
reviewed_content_sha256: 30eecbb3e5472435
-->
# N-gram 概念页审查记录（第 6 轮）

- 页面版本：`f3c4a6997c565a70668f33b76c6e9086c930d293`
- 审查时间：2026-09-14 17:10
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 适用规范：`guides/concept/check.md`（`dojo:type=concept`），格式对照 `guides/concept/style-guide.md`
- 已完整阅读章节：引言 / 核心问题 / 常见误解 / 1.链式法则算得动吗（含「补充：为什么句首句尾需要特殊符号」「本章问题」）/ 2.计数就是参数（含代码折叠块、表格、本章问题）/ 3.语料再大也不够用（含加一平滑算例、「补充：加一平滑为什么太平」、本章问题）/ 4.从计数表到可学习查表（含结构图、对比表、代码折叠块、本章问题）/ 来源与范围说明（C/F/N/构造示例/辅助解释与类比边界/简化条件及其限制）；并核对了页内标题、图注、summary 与 overview.html。

## 核对情况

教材（Speech and Language Processing, 3rd ed. draft, Draft of August 19, 2026，实取 web.stanford.edu/~jurafsky/slp3/3.pdf）：

- C1 引文逐字命中：正文原文 “…An n-gram is a sequence of n words…” 与 “(in a bit of terminological ambiguity) use the word ‘n-gram’ to mean a probabilistic model…”。
- F1/C3：Eq.(3.3)(3.4) 即链式法则 `P(w1:n)=∏ P(wk|w1:k−1)`；F2 即 Eq.(3.9) `P(w1:n)≈∏ P(wk|wk−1)`，与页面两式逐字符一致。
- C2 引文逐字命中：§3.1.1 “instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words”。
- F3/§3.1.2：Eq.(3.12) 分子分母与页面 MLE 式等价；示例语料三句即 `<s> I am Sam </s> / <s> Sam I am </s> / <s> I do not like green eggs and ham </s>`（与页面一致）。教材给出 P(I|<s>)=2/3、P(Sam|<s>)=1/3、P(am|I)=2/3、P(do|I)=1/3、P(Sam|am)=1/2、P(</s>|Sam)=1/2，与页面表格六个数值逐项一致。
- N2/Figure 3.1 图注原文：“Bigram counts for eight of the words (out of V = 1446) in the Berkeley Restaurant Project corpus of 9332 sentences.” 页面节选的 6×8 计数矩阵（i/want/to/eat/chinese/food 六行）与图中数值逐格一致；C(chinese,to)=0 成立。
- N1/§3.3 原文：“trained unigram, bigram, and trigram models on 38 million words from the Wall Street Journal newspaper… the 1.5 million word test set… Perplexity 962 170 109.” 与页面 N1（3800 万训练、150 万同域测试、962/170/109）一致。
- C6/§3.6 引文逐字命中：“Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events.”；C5 的“任何有限语料漏掉合法序列”见 §3.6 首段。
- 复算：2/3×2/3×1/2×1/2=1/9≈0.1111（页内整句概率）；109/962≈0.113≈1/9；1446²=2,090,916≈209 万、1446³≈30 亿；加一算例 5/6→6/17≈0.353、0→1/17≈0.059，均正确。2^{每词交叉熵} 即 §3.7 的 `Perplexity(W)=2^{H(W)}`。

官方材料（huggingface.co/Qwen/Qwen3.8-Flash-Next 的 config.json、README 模型卡；raw.githubusercontent.com…/transformers/36deb0b5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py）：

- config.json：ngram_size=3、heads_per_ngram=8、ngram_vocab_size_base=20000000、make_ngram_vocab_size_divisible_by=128、ple_embed_dim=2560、ple_layer_ids=[2]、vocab_size=248320。
- 源码 L1202 `ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None` → ple_layer_ids 按 1 基，实际注入层 layer_idx=1，即 1 基第 2 层；模型卡“N-gram Embedding: 20,000,000 (bigrams/trigrams at layer 2)”与技术报告“We place it at Layer 2”同证。页面“在第 2 层注入”成立。
- 头数：ngram_heads=(3−1)×8=16；前 8 头 bigram、后 8 头 trigram（forward 中 `start_idx=(ngram−2)*heads_per_ngram`），页面“bigram 与 trigram 各 8 个头”成立。head_dim=2560//16=160，页面“每行 160 维”成立。
- 用源码 `_build_layer_multipliers`（seed 默认 1234、ple_layer_index=0）复算得乘子 [23703573157769, 20109073645365, 8052911324071]，页面代码取前两个，与 N4 所述一致。
- 用源码 `_find_nth_prime_after(19999999, ·)` 复算 16 个质数，和=320,001,446，按 128 对齐=320,001,536（补 90 行），×160=51,200,245,760=51.20B；页面三个数字全部一致。
- F4 引文范围：L1098-1110 即 `for ngram in range(2, ngram_size+1)` 的混合/取模/加偏移循环；C7 的 L1018-1114 即 `class Qwen4ExpTextNGramEmbedding` 全类。页面公式 `id_h(t)=((⊕_{p=0}^{n-1} ids_{t−p}·c_p) mod P_h)+off_h` 与该循环逐项一致（注意运算符优先级：源码为 `(a*M0) ^ (b*M1)`，与页面 ⊕ 的写法一致）。
- 页面「简化条件」称源码在句尾符号处做分段重置：`_shift_right_ignore_eos` 以 EOS 位置切段（`segment_start = previous_eos + 1`、`valid = position_in_segment >= shift`），成立。
- 技术报告（tech_report.pdf，QwenLM/Qwen3.8-Flash-Next）通篇未讨论哈希/质数/碰撞，报告给 n-gram 表的动机是「稀疏访问 + 确定性寻址 → 可卸载到主机内存、可预取」；页面把碰撞控制作为该表可用性的机制解释，属页面自身论证，非来自报告的论断。

代码执行：逐字运行页面两个 `language-python` 折叠块。代码 1 输出 `P(<s> I am Sam </s>) = 0.111111`，且六个条件概率逐项等于表格值；代码 2 输出三行「混合值/行号」（位置 2/3/4 → 14,605,717 / 11,040,574 / 9,211,431）、「质数表 0 次」「小表对照 792 次」，均与页面的「预期输出」逐字符一致。

机械项：`python3 .dojo/scripts/validate.py wiki/ngram/index.html` 返回 `validation ok`；`research/` 外无占位符；无 `<img>` 正文图（仅 lightbox 空 alt）、无等宽框线图；结构图为 HTML 节点，窄屏/明暗主题下可读；引用页 `../pretraining/`、`../cross-entropy/`、`../qwen3-8-flash-next-dataflow/` 均存在，index↔overview 互链；引文编号 C1–C7、F1–F4、N1–N4 在正文均有对应上标且双向可解析。

## 问题

- [轻微·表述] 引言第 2 段、「计数就是参数」章第 2 段、「语料再大也不够用」章第 1 段、「链式法则算得动吗」章第 4 段：残留元话语与自我指代的指路语——「全篇依次讨论：…。文中 Qwen 相关部分来自…」「以下为教材 §3.1.2 自带的示例语料（非本页构造）」「下面节选其中 6 行」「这是贯穿全文的权衡」。｜引文依据：不适用｜修复要求：改为直接陈述，删除对篇章自身的指路词——「全篇依次讨论：…。文中…」改为「本文讨论：…」或直接进入问题；「以下为教材 §3.1.2 自带的示例语料（非本页构造）」改为「语料取自教材 §3.1.2，三句构成全部语料」；「下面节选其中 6 行」改为「节选其中 6 行」；「这是贯穿全文的权衡」改为「这是 n 阶数与参数量的权衡」。｜修复：｜复验：
- [轻微·格式] 「外部数字与实验条件（N）」小节 N2：写作 `V=1446`，未纳入 `$...$`；正文第 3 章同一量写作 `$V=1446$`。｜引文依据：不适用｜修复要求：按 `style-guide.md` §11（数学变量须由 KaTeX 渲染、同一变量全页同一种写法）把 N2 的 `V=1446` 改为 `$V=1446$`。｜修复：｜复验：
- [轻微·表述] 第 4 章第 5 段与「核心问题」第 4 条答案：「…表容量远大于实际组合数使碰撞率可忽略」「控制在可忽略水平的办法是把表做得远大于实际组合数」——「实际组合数」未定义，且与本页把「组合数」用作「可能的组合数」（第 3 章 |V|²、|V|³）的用法相冲突，易被读成「表容量大于 |V|^n」，与本节结论无关甚至相反。｜引文依据：不适用｜修复要求：把「实际组合数」限定为「实际被哈希查询到的不同 n-gram 数」或改述为「把表做得远大于实际访问到的 n-gram 数，使碰撞率可忽略」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。全部事实性论断、公式与数字均回源核对通过：SLP3（2026-08-19 草稿）的引文、公式编号（Eq.3.3/3.4/3.9）、I am Sam 六个条件概率与整句概率、WSJ 962/170/109、Berkeley 9332 句/V=1446 与节选矩阵逐格一致；Qwen 的层号、头数、质数表行数 320,001,446→320,001,536、每行 160 维、51.2B、乘子与 16 个质数均以官方 config.json、transformers@36deb0b5 源码与模型卡复算一致；两段「实际运行」代码逐字符复现预期输出；文件内无同页矛盾、无 summary/overview/图注数字不一致、无引文编号失配、无指向不存在文件的路径、无 `$...$` 进入 alt、无「（待生成）」占位。3 项轻微问题不影响正确性与主线理解，建议修后发布。
- 未作论断：本次未读本页 `research/` 下任何文件；未把数据流页当成本页结论的唯一来源，其记录的碰撞口径与本页口径一致、数值差异可由统计口径差异解释，故不列为问题。
