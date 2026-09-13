<!-- review-meta
round: 5
page: wiki/ngram/index.html
reviewed_content_sha256: 2489e520dcc38f00
-->
# N-gram 审查记录（第 5 轮）

- 页面版本：5cad294a18c95786528911ba6f788dfffbf4b731
- 审查时间：2026-09-13 21:25
- 审查者：独立子代理（未参与写作，未参与前序轮次审查；本轮只读 index.html、overview.html、外部来源与本规范）
- 已完整阅读章节：核心问题 / 常见误解 / 1. 链式法则算得动吗——马尔可夫近似从哪里来 / 2. 计数就是参数——三句语料上手算 / 3. 语料再大也不够用——稀疏与平滑 / 4. 从计数表到可学习查表——现代模型里的 n-gram / 来源与范围说明（含全部折叠块、图注与代码块）

## 核对依据（本轮实际打开的来源与关键数值）

- SLP3 第 3 版草稿 2026-08-19 第 3 章（外部 PDF 原文）：§3.1 "An n-gram is a sequence of n words…" 与 "in a bit of terminological ambiguity"；§3.1.1 "instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words"；Eq.(3.3)(3.4) 链式法则、Eq.(3.9) bigram 近似、Eq.(3.12) MLE；§3.1.2 三句 I am Sam 语料与 P(I|<s>)=2/3、P(Sam|<s>)=1/3、P(am|I)=2/3、P(do|I)=1/3、P(Sam|am)=1/2、P(</s>|Sam)=1/2；Figure 3.1 Berkeley 计数矩阵逐格（本页 6×8=48 格全部与原文一致，"out of V = 1446 … corpus of 9332 sentences"）；§3.3 原文 "trained … on 38 million words from the Wall Street Journal … perplexity of the 1.5 million word test set" 与表 "Perplexity 962 170 109"；§3.6 原句与 "any finite training corpus will be missing some perfectly acceptable English word sequences"。以上全部与本页一致。
- 本页两处代码块实机运行：`P(<s> I am Sam </s>) = 0.111111`；行号三行（110,194,991,134,961→14,605,717 / 106,539,527,021,499→11,040,574 / 220,389,982,269,922→9,211,431）；质数表 0 次、小表 792 次。逐字符与"预期输出"一致。
- transformers@36deb0b5 `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`（原文下载核对）：`Qwen4ExpTextNGramEmbedding` 起 L1017、行号计算 L1098-1110（与 F4 标注一致）；`mixed_ids` 为 `shifted_tokens[p]*layer_multipliers[p]` 逐位异或后 `torch.remainder(…, head_vocab_sizes)` 再加 `head_offsets`，与 F4 公式逐项一致；`multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)` 确为奇数，`multiplier_max = max_long // vocab` 且 `half_bound = multiplier_max // 2` 确保证 product 不溢出 int64。
- 用源码公式 + config 复算：ple_layer_index=0 三个乘子为 23703573157769 / 20109073645365 / 8052911324071（页面 MULTS 取前两个）；16 个质数 20000003…20000171 之和 320001446，按 128 对齐得 320001536，×160=51200245760=51.2B。全部与正文一致。
- Qwen/Qwen3.8-Flash-Next@f5d08274 `config.json`（原文下载核对）：heads_per_ngram=8、ngram_size=3、ngram_vocab_size_base=20000000、make_ngram_vocab_size_divisible_by=128、vocab_size=248320、ple_layer_ids=[2]；源码 L1202 `config.ple_layer_ids.index(layer_idx + 1)` 说明注入层 0 起索引为 1，即页面所说"第 2 层"，与数据流页"第 1 层（0 起）"表述等价。
- 算式复算：1446²=2,090,916（约 209 万）、1446³=3,023,464,536（约 30 亿）；2/3×2/3×1/2×1/2=1/9；6/(6+11)=6/17≈0.353、1/17≈0.059；109/962≈1/8.8≈1/9；16×4000=64,000、2/4000=0.050%（与 overview 一致）。均无误。
- validate.py 返回 "validation ok: wiki/ngram/index.html"；正文引用的前置页 ../pretraining/、../cross-entropy/、../qwen3-8-flash-next-dataflow/ 均真实存在；overview.html 与 index.html 互相链接；alt 为空，无 `$...$`；图为 HTML 结构流程图，无等宽字符框线。

## 问题

- [轻微·来源] 来源与范围说明（C6）：C6 引文比教材原句多出 "and discounting" 两词，被当作 `原文` 引用｜引文依据：SLP3 §3.6 原句为 "Smoothing algorithms shave off a bit of probability mass from some more frequent events and give it to unseen events."（其紧邻上一句才是 "…is called smoothing or discounting."）｜修复要求：删去 "and discounting"，按原句回引，或改为不带引号的转述｜修复：｜复验：
- [轻微·可读性] 第 3 章正文"一个 0 会直接归零零概率句"：动词"归零"与宾语"零概率句"紧邻，形成"归零零概率句"，需回读才能断句｜引文依据：不适用｜修复要求：改为可直接读通的表述，如"一个 0 会让整句概率直接归零"｜修复：｜复验：
- [轻微·表述] 第 4 章代码折叠块"观察重点"："在「计数就是参数」一章产出计数概率、在这里产出行号"——"在这里"以当前页位置自指，属元话语式指代，且与前一分句"在…一章"句式不对称｜引文依据：不适用｜修复要求：改为"在「从计数表到可学习查表」一章产出行号"，与前一章名并列｜修复：｜复验：
- [轻微·来源] 第 3 章"教材在 Berkeley 餐厅语料（9332 句、词表 V=1446）上给出 8 个高频词的 bigram 计数矩阵"：教材图注只写 "Bigram counts for eight of the words (out of V = 1446)"，并说明这 8 个词是按彼此搭配挑选，而非按词频挑选｜引文依据：SLP3 Figure 3.1 图注、§3.1.2 "…we have chosen the sample words to cohere with each other; a matrix selected from a random set of eight words would be even more sparse."｜修复要求：删去"高频"，改为"教材挑选的 8 个词"或"8 个相互搭配的词"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布（0 阻断 0 重要；遗留 4 条轻微，建议顺手修复后发布）
