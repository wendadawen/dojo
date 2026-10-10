# MQA 与 GQA

- 描述：MQA 让所有查询头共用一套键和值，GQA 把查询头分成若干组、每组共用一套，两者都缩小生成时反复读取的键值数据量。
- 摘要：生成每个新 token 都要重读全部键、值张量，这一步受显存带宽限制。MQA 让所有查询头共用一套键值，把这份数据量除以头数；GQA 让每组查询头共用一套，组数在 $1$（等于 MQA）与头数（等于多头）之间取值，换来质量与速度的折中。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 长页

- 开头｜MQA 让所有查询头共用一套键值，GQA 把查询头分成组、每组一套，组数在 MQA 与多头之间取值；标准注意力、KV cache 链接成页｜280｜paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L132-L134；paper/src-mqa/main.tex L259
- ### 解码慢在反复搬键值｜生成逐步进行、无法并行，每步重载全部键值张量；访存与计算之比接近 1 时显存带宽成瓶颈；推导折叠｜380｜paper/src-mqa/main.tex L207；paper/src-mqa/main.tex L209；paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255
- ### 多头各有一套键值，MQA 合成一套｜多头每个头各有一套键值投影；MQA 让所有头共用同一套键和值，查询仍多头；形状从带头的维度变成不带，每步读的数据量除以前的头数；形状与访存比值折叠｜400｜paper/src-mqa/main.tex L133；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L277-L279；paper/src-mqa/main.tex L293-L299；paper/src-mqa/main.tex L328-L330
- ### GQA 把查询头分组｜把查询头分成 G 组，每组共用一个键头和一个值头；组数为 1 是 MQA、等于头数是多头；自设 4 头分 2 组的例子走一遍，例子折叠｜400｜paper/src-gqa/main.tex L117；paper/src-gqa/main.tex L132-L134；paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L188-L190
- ### 从现有多头检查点改造｜两条路：从头训练，或把已有检查点改造；改造分两步：组内键头、值头取平均并成一个投影，再用一小部分预训练步数继续训练；检查点、预训练两词先说明，实验条件与曲线折叠｜380｜paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L124-L130；paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L150-L153；paper/src-gqa/main.tex L178-L179；paper/src-gqa/main.tex L183-L184；paper/src-gqa/main.tex L228
- ### 质量与速度的取舍｜同一套规模下比较多头、MQA、GQA-8 的平均分与推理时间；GQA-8 质量接近多头、速度接近 MQA；实验条件、表与曲线折叠｜380｜paper/src-gqa/tables/main_results.tex L10-L13；paper/src-gqa/figures/results.tex L18-L31；paper/src-gqa/figures/time_vs_groups.tex L32-L44；paper/src-gqa/figures/uptraining_steps.tex L30-L40
- ### 适用条件与边界｜MQA 掉一点质量、训练可能不稳；GQA 折中质量与速度；只改键值头数；GQA 不施加在编码器自注意力，MQA 替换全部注意力层；论文预期越大模型收益越大｜350｜paper/src-gqa/main.tex L113；paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L138；paper/src-gqa/main.tex L140；paper/src-gqa/main.tex L228；paper/src-mqa/main.tex L255；paper/src-mqa/main.tex L259；paper/src-mqa/main.tex L339；paper/src-mqa/main.tex L349；paper/src-mqa/main.tex L353

## 短页

- ### 每个查询头不再各有一套键值｜是什么：多头每头一套键值，MQA 全部查询头共用一套，GQA 每组一套、组数 1 是 MQA、头数是多头；前置 标准注意力、KV cache 链接成页｜300｜paper/src-mqa/main.tex L259；paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L132-L134
- ### 生成慢在反复搬键值｜为什么需要：生成逐步进行，每步重读全部键、值张量，数据量随头数增长、显存带宽成瓶颈；共用键值把这份数据量按查询头组数缩小｜300｜paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255；paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L136
- ### 按组共用一套键值｜怎么做：把键头、值头的数量设为查询头组数 G，从头训练或把已有的多头检查点改造得到｜250｜paper/src-gqa/main.tex L104；paper/src-gqa/main.tex L126；paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L228
