# 大纲

## 长页

- 开头｜GQA 是什么（把查询头分组、每组共用一套键值）、为什么需要（带宽背景与折中），前置 标准注意力、KV cache 链接成页；只一句概览，不展开分组机制｜260｜paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L117；paper/src-gqa/main.tex L132-L134
- ### 生成慢在反复读取键值｜生成逐步进行无法并行，每步重载全部键值张量；访存与计算之比接近 1 时显存带宽成瓶颈；推导折叠｜360｜paper/src-mqa/main.tex L205；paper/src-mqa/main.tex L207；paper/src-mqa/main.tex L209；paper/src-mqa/main.tex L249；paper/src-mqa/main.tex L251；paper/src-mqa/main.tex L253；paper/src-mqa/main.tex L255
- ### 把查询头分组，每组一套键值｜查询头分 G 组、每组一份键头和值头；1 组是 MQA（链 mqa）、头数是多头；自设 4 头分 2 组的例子全过程折叠｜420｜paper/src-gqa/main.tex L132-L134；paper/src-gqa/main.tex L136；paper/src-gqa/main.tex L188-L190
- ### 从现有多头检查点改造｜先说明检查点、续训两个词；把组内各头的键投影、值投影分别取平均成一组，再续训一小段；续训到原预训练步数不同比例的分数折叠｜320｜paper/src-gqa/main.tex L126；paper/src-gqa/main.tex L130；paper/src-gqa/main.tex L134；paper/src-gqa/main.tex L183-L184
- ### 质量与速度的取舍｜GQA-8 平均分接近多头、推理时间接近 MQA；组数增大耗时先缓后陡；实验条件、表与曲线折叠｜360｜paper/src-gqa/main.tex L168；paper/src-gqa/main.tex L190；paper/src-gqa/tables/main_results.tex L10-L13
- ### 适用条件与边界｜三种注意力各处理什么一句带过；GQA 不用在编码器自注意力（各位置并行，带宽一般不是主要瓶颈），用在解码器与交叉注意力、预期模型越大收益越大｜280｜paper/src-gqa/main.tex L138；paper/src-gqa/main.tex L140；paper/src-gqa/main.tex L148

## 短页

- ### 把查询头分成组，每组一套键值｜是什么：查询头分 G 组、每组一套键值，1 组是 MQA、头数是多头；前置 标准注意力、KV cache 链接成页｜280｜paper/src-gqa/main.tex L132-L134
- ### 生成慢在反复读取键值｜为什么需要：生成每步重读全部键值张量，按键值头组数缩小；组数在质量与速度之间取位置｜290｜paper/src-gqa/main.tex L111；paper/src-gqa/main.tex L136；paper/src-mqa/main.tex L101；paper/src-mqa/main.tex L207-L209
- ### 按组共用一套键值｜怎么做：把已有的多头检查点组内各头取平均改造成这个结构，再续训一小段；先说明检查点、续训两个词｜250｜paper/src-gqa/main.tex L126；paper/src-gqa/main.tex L130；paper/src-gqa/main.tex L134
