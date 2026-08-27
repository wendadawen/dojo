# evidence：n-gram

来源固定为：Jurafsky & Martin, Speech and Language Processing, 第 3 版草稿（Draft of August 19, 2026），第 3 章 "N-gram Language Models"，https://web.stanford.edu/~jurafsky/slp3/3.pdf（本机已下载并提取全文，文本存 /tmp/concept-evidence/slp3-ch3.txt，26 页）。Qwen 侧为 transformers@36deb0b5 源码与数据流页实测。

| 编号 | 论断 | 来源定位 | 适用条件 | 置信 |
|---|---|---|---|---|
| C1 | n-gram 是 n 个词的序列；bigram 是二元序列、trigram 是三元序列；"n-gram"一词也指估计词概率的概率模型（术语二义性） | SLP3 §3.1"An n-gram is a sequence of n words…"及"in a bit of terminological ambiguity"句 | — | 已确认 |
| C2 | n-gram 模型的直觉是不用整段历史，只用最后几个词近似历史 | SLP3 §3.1.1"The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words." | — | 已确认 |
| C3 | 句子概率可用链式法则分解；马尔可夫假设把它近似为逐词条件概率的连乘 | SLP3 Eq.(3.3)(3.4) 链式法则、Eq.(3.7)(3.9) bigram 近似 | — | 已确认 |
| C4 | MLE 用语料计数归一化估计：$P(w_n\mid w_{n-N+1:n-1})=C(w_{n-N+1:n})/C(w_{n-N+1:n-1})$ | SLP3 §3.1.2"maximum likelihood estimation or MLE…getting counts from a corpus, and normalizing the counts" | 训练与测试同分布 | 已确认 |
| C5 | 有限语料必然漏掉合法词序列；Berkeley 餐厅语料（9332 句、V=1446）8 个词的 bigram 计数矩阵含大量零 | SLP3 §3.6"There is a problem with using maximum likelihood estimates…any finite training corpus will be missing some perfectly acceptable English word sequences"；Figure 3.1 图注与矩阵 | — | 已确认 |
| C6 | 平滑的目标是从高频事件匀出一点概率质量给从未见过的事件 | SLP3 §3.6"The goal of smoothing is to shave a little bit of probability mass from some more frequent events and give it to the events we've never seen" | — | 已确认 |
| C7 | 哈希 n-gram 特征表查的是可学习向量而非频率；bigram/trigram 各 8 个头、每头一张质数大小的词表 | transformers@36deb0b5 `modeling_qwen4_exp.py` L1018-1114（NGramEmbedding/PLELayer）；数据流页 probe4/probe7 实测 | Qwen3.8-Flash-Next | 已确认 |
| F1 | 链式法则 $P(w_{1:n})=\prod_{k=1}^{n}P(w_k\mid w_{1:k-1})$ | SLP3 Eq.(3.4) | — | 已确认 |
| F2 | bigram 近似 $P(w_{1:n})\approx\prod_{k=1}^{n}P(w_k\mid w_{k-1})$（首词以 `<s>` 为条件） | SLP3 Eq.(3.9) | — | 已确认 |
| F3 | MLE 公式（同 C4） | SLP3 §3.1.2 | — | 已确认 |
| F4 | 哈希 id：$\mathrm{id}_h=(\bigoplus_p \mathrm{ids}_{t-p}\cdot c_p \bmod P_h)+\mathrm{off}_h$ | 源码 L1098-1110；数据流页实测复算 | — | 已确认 |
| N1 | WSJ 实验：3800 万词训练、150 万词测试，unigram/bigram/trigram 困惑度 962/170/109 | SLP3 §3.5"we trained unigram, bigram, and trigram models on 38 million words from the Wall Street Journal…Unigram Bigram Trigram Perplexity 962 170 109" | 同域测试集 | 已确认 |
| N2 | Berkeley 餐厅语料 9332 句、词表 V=1446 | SLP3 Figure 3.1 图注"Berkeley Restaurant Project corpus of 9332 sentences" | — | 已确认 |
| N3 | I am Sam 语料的 bigram 概率：$P(I\mid\langle s\rangle)=2/3$、$P(\mathrm{am}\mid I)=2/3$、$P(\mathrm{Sam}\mid \mathrm{am})=1/2$ 等 | SLP3 §3.1.2 语料与计算列表 | — | 已确认 |
| N4 | Qwen 表：16 头质数词表之和 320,001,446、对齐后 320,001,536 行 × 160 维 = 51.20B；实测碰撞率 bigram 0.025%/4000 位置 | 数据流页 research/count_params.py、probe7_ngram_buffers.py 实测 | Qwen3.8-Flash-Next@f5d08274 | 已确认 |

冲突与不足：教材 2026-08-19 草稿的例句已从旧版"please turn your homework in"换为 Walden Pond 系列，引用一律按本版原文；Google 万亿词语料数字未在本版提取文本中找到，不引用。
