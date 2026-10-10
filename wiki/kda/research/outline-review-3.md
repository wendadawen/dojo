# 大纲审查（第 3 轮）

## 读者

- 顺序错乱｜作为可学习位置编码｜「体现相对位置……一种乘性位置编码」里的「位置编码」「相对位置」未先讲｜此前「状态」「修正」「门控」「分块并行」都没讲，本节只称其为「乘性位置编码」而未定义位置编码本身｜改为链接 wiki/positional-encoding。

遗漏关键内容、多余、短页标题：无。

## 对照资料

- 资料里没有（位置错）｜门控：逐通道的遗忘｜「Gated DeltaNet 的按头标量门」的「按头」不在所标 Gated DeltaNet as Weight Decay 段内（该段只说 scalar forget gate）｜「head-wise」在 1-intro.tex Introduction L9「coarse head-wise forget gate」｜依据补标 paper/src-v2/1-intro.tex Introduction。
- 遗漏关键内容｜分块并行｜「写什么」未含 KDA 与 DPLR 的关系，而依据列了 6-discuss Relation to DPLR｜论文把「KDA 是 DPLR 的受限变体、比一般 DPLR 少算」当作卖点（main.tex abstract；3-kda.tex Efficiency Analysis）｜在分块并行内一句说明它是 DPLR 的受限变体、少算在哪。

抬高次要点：无。

## 篇幅与冗余

- 短页不合格｜短页 衰减、修正、分块｜一节塞了三件事（逐通道衰减、delta 修正、块状并行），块状并行事属实现、不是更新规则本身｜「怎么做：……最后配一套块状并行算法；状态大小不随长度变」｜删去块状并行部分，标题改为「先衰减，再用键值纠正」。

重复、无关、预算不合理：无。

## 处理

- 位置编码前置链接：成立，改为链接 wiki/positional-encoding（该页尚在生成，须存在后才能通过机械检查）。
- 门控「按头」依据：成立，依据补标 paper/src-v2/1-intro.tex Introduction。
- DPLR 关系：成立，分块并行节点「写什么」加「说明它是 DPLR 的受限变体」。
- 短页第 3 节：成立，改为「先衰减，再用键值纠正」，只讲更新规则的两步，删块状并行；预算 400→300，依据改为 3-kda.tex Kimi Delta Attention；2-preliminary.tex Gated DeltaNet as Weight Decay。

## 最后一批改动核对

本轮改动：门控依据补 1-intro.tex Introduction；分块并行加「DPLR 的受限变体」一句；位置编码加链接 wiki/positional-encoding；短页第三节改为「先衰减，再用键值纠正」（400→300，依据改 3-kda.tex Kimi Delta Attention；2-preliminary.tex Gated DeltaNet as Weight Decay）；分块并行「写什么」压缩到 80 字内。交仲裁核对。
