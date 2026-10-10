# MLA

- 描述：MLA 是一种注意力机制，把每个位置的键和值联合压成一个低维潜在向量缓存起来，用到时再用上投影重建键和值，从而在保留全局多头注意力的同时大幅减小 KV cache。
- 摘要：MLA 把每 token 每层的键值压成一个低维潜在向量 $\mathbf{c}_t^{KV}$ 缓存，用上投影矩阵在注意力里重建键值；推理时这些矩阵可被吸收进查询与输出投影，连重建也省去，KV cache 只相当于 2.25 组 GQA。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 长页

- 开头｜生成要把键值缓存下来、缓存随序列线性增长成为瓶颈；已有办法掉分；MLA 是什么、为什么需要、怎么做；前置概念：标准多头注意力（MHA）、GQA、KV cache｜350｜paper/src-v5/main.tex Introduction；paper/src-v5/main.tex Multi-Head Latent Attention；paper/src-v5/main.tex L1166-L1194；paper/src-v5/main.tex Low-Rank Key-Value Joint Compression
- ### 把键和值压成一个潜在向量｜下投影压成低维潜在向量并只缓存它，上投影重建键和值；查询为何也低秩压缩；维度例子折叠｜300｜paper/src-v5/main.tex Low-Rank Key-Value Joint Compression
- ### 推理时不必算出键和值｜上投影并入查询、输出投影并入输出，省掉逐步重建键值；合并推导、完整式子、走一遍都折叠｜300｜paper/src-v5/main.tex L1159-L1162；paper/src-v5/main.tex Low-Rank Key-Value Joint Compression
- ### 解耦旋转位置编码｜旋转位置编码（前置概念）加在压缩键上为何使上投影无法在推理时被吸收（吸收见上一节）；改用一对解耦查询和键承载位置，缓存多出什么；推导折叠｜300｜paper/src-v5/main.tex Decoupled Rotary Position Embedding
- ### KV cache 有多小｜每 token 缓存元素数与 MHA 的公式对比，代入 DeepSeek-V2 配置得约 2.25 组 GQA；表与代入推导折叠｜250｜paper/src-v5/main.tex L322-L348；paper/src-v5/main.tex Comparison of Key-Value Cache
- ### 效果与效率｜MLA 与 MHA 在同一架构下对比，MLA 多数基准分数更高，代价是缓存（见上节）；数字折叠｜200｜paper/src-v5/main.tex L1196-L1230
- ### 适用条件与边界｜全局 softmax 求和不变；缓存仍随层数线性增长；误区「以为 MLA 取消了 KV cache」——仍要缓存解耦键，只是被压小｜150｜paper/src-v5/main.tex L311-L318

## 短页

- ### 把缓存压成一个向量｜是什么：把每个位置的键和值联合压成低维潜在向量缓存，用到时用上投影重建键值，注意力仍对全部前序位置求和；键和值、注意力见前置概念标准注意力｜300｜paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex L311；paper/src-v5/main.tex Preliminaries: Standard Multi-Head Attention
- ### 缓存是生成的瓶颈｜为什么需要：生成时逐 token 缓存键和值，缓存随序列线性增长；已有办法会掉分，MLA 要缓存更小、分数更高｜300｜paper/src-v5/main.tex Multi-Head Latent Attention；paper/src-v5/main.tex L1166-L1194
- ### 压缩、解耦、吸收｜怎么做：下投影压成潜在向量并只缓存；一对解耦查询和键承载旋转位置编码（一句话解释它是什么）；推理时上投影并入查询、输出投影并入输出｜250｜paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex Decoupled Rotary Position Embedding；paper/src-v5/main.tex L1159-L1162
