# 大纲审查 第 2 轮

## 读者

顺序错乱｜「### 适用条件与边界」｜「隐式正则」不懂｜此前无节点讲正则化｜加一句通用释义，或作为前置概念链接。

短页标题｜短页「### 分块相加，块间注意」｜把「怎么做」和「收益」写在一起，末尾「分块把内存降到 $O(Nd)$」属「为什么需要」｜第 3 节应是单节｜删去内存收益句。

遗漏关键内容｜无。
多余｜无。

## 对照资料

资料里没有｜「### 固定权重残差随深度稀释每一层」｜「早期信息被埋」位置错｜在所列依据内无对应，最接近的是 Limitations「irreversible loss: information lost through aggregation cannot be selectively recovered in deeper layers」；原文在 1-intro.tex Introduction「Early-layer information is buried and cannot be selectively retrieved」｜改挂 1-intro.tex Introduction，或按 irreversible loss 措辞改写。

资料里没有｜「### 适用条件与边界」｜「代价是块表示要存储与传递」位置错｜在所列依据内无对应；原文在 4-infra.tex Infrastructure Design「block representations must be propagated across pipeline stages」｜改挂 4-infra.tex Infrastructure Design。

资料里没有｜短页「### 分块相加，块间注意」｜「每层对前面块表示与当前部分和做 softmax 注意」漏条件｜原文「the first layer receives the previous block representations and the token embeddings, and the subsequent layers additionally attend to the partial sum」｜改为「块内首层只注意前面块表示，其后各层再加当前部分和」。

遗漏关键内容｜「### 分块：块内相加，块间注意」｜「$L$ 层分 $N$ 块」没交代层的定义｜原文 Notation「we treat each self-attention or MLP as an individual layer」；伪代码「block_size counts ATTN + MLP」｜先交代层按注意力与前馈各算一层。

抬高次要点｜无。

## 篇幅与冗余

短页不合格｜短页「### 分块相加，块间注意」｜一节讲两件事：分块机制与内存收益｜依据原文「分块把内存降到 $O(Nd)$」｜删去内存收益句，本节只留分块机制。

重复｜无。无关｜无。预算不合理｜无。

## 处理

- 读者「隐式正则」采纳：「适用条件与边界」里「块级压缩像隐式正则」改为带一句通用释义的写法，不展开机制。
- 读者与篇幅两条「短页第三节」采纳：删去「分块把内存降到 $O(Nd)$」，该节只留分块机制；内存收益在长页「省下的是内存和通信」讲，短页不再另设。
- 对照资料「早期信息被埋」采纳：改挂 1-intro.tex Introduction（依据列加该位置），沿用「不可逆」措辞。
- 对照资料「块表示要存储与传递」采纳：改挂 4-infra.tex Infrastructure Design。
- 对照资料「块内首层」采纳：短页第三节改为「块内首层只注意前面块表示，其后各层再加当前部分和」，依据改引 Inter-Block Attention。
- 对照资料「层的定义」采纳：「分块」章先交代层按注意力与前馈各算一层，依据加 2-preliminary.tex Notation。
