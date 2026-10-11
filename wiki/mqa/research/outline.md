# 大纲

## 长页

- 开头｜MQA 是什么（所有查询头共用一套键值），为什么需要一句概览（生成受显存带宽限制），前置 标准注意力、KV cache 链接成页；不展开机制｜260｜paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L259
- ### 生成慢在反复读取键值｜生成逐步进行无法并行，每步重载全部键值张量；访存与计算之比接近 1 时显存带宽成瓶颈；推导折叠｜360｜paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L207；paper/src-mqa/main.tex L209；paper/src-mqa/main.tex L249；paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255
- ### 多头各有一套键值，MQA 合成一套｜多头每头各有一套键值投影，MQA 去掉头维度只留一套；查询仍多头；按这个结构从头训练得到模型；自设两个查询头共用一套键值的例子全过程折叠、形状与比值折叠｜420｜paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L277-L279；paper/src-mqa/main.tex L328-L330；paper/src-mqa/main.tex L337
- ### 质量与训练稳定性｜跟多头相比质量略降、解码更快；先说明微调、损失尖峰、发散三个词；从头训练的模型可能损失尖峰、在长输入任务上微调时立即发散；数字折叠｜340｜paper/src-mqa/main.tex L349；paper/src-mqa/main.tex L351；paper/src-mqa/main.tex L357；paper/src-mqa/main.tex L359；paper/src-mqa/main.tex L362-L388；paper/src-gqa/main.tex L228
- ### 适用条件与边界｜只改键值头数，替换全部注意力层；键值容量随之变小；头数越多可压掉的越多；GQA 取组数折中（链 gqa）｜300｜paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L339；paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L136

## 短页

- ### 所有查询头共用一套键值｜是什么：多头每头一套键值，MQA 全部查询头共用一套、查询仍多头；前置 标准注意力、KV cache 链接成页｜280｜paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L259
- ### 生成慢在反复读取键值｜为什么需要：生成每步重读全部键值张量，数据量随头数增长、显存带宽成瓶颈；共用一套把这份数据量按键值头数缩小｜280｜paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255
- ### 把键值头数设成一套｜怎么做：键头、值头各留一套，按这个结构从头训练得到模型；末尾点名 GQA 并链接｜250｜paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L337；paper/src-gqa/main.tex L132-L134
