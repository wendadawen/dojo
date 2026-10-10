# 因果掩码

- 描述：因果掩码是解码器自注意力里的下三角掩码，位置 $i$ 只能关注位置 $j \le i$，它之后的位置被遮住，使模型预测位置 $i$ 时看不到未来的 token。
- 摘要：把解码器自注意力分数矩阵的上三角遮住，位置 $i$ 的输出就只由位置 $\le i$ 的输入决定。有了它，自回归模型训练时能一次算出整条序列的所有位置，解码时每步也只依赖已经生成的 token。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 长页

- 开头｜先一句话讲清自注意力、解码器、自回归；再定义因果掩码；点明它解决什么问题、大概怎么算；不写公式细节和数字｜340｜paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention
- ### 掩码把哪些位置遮住｜这一章讲掩码的允许集合与形状，内容全在一个小节｜0｜paper/src/model_architecture.tex Applications of Attention in our Model
- #### 一个位置能看哪些位置｜位置 $i$ 允许关注 $j \le i$、之后的位置被遮；由此得到保留下三角、遮住上三角的形状，形状图与小例子折叠｜380｜paper/src/model_architecture.tex Applications of Attention in our Model
- ### 掩码怎么算进去｜这一章讲掩码加在哪一步、被遮位置的分数与权重，内容全在两个小节｜0｜paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention
- #### 把被遮位置的分数设成负无穷｜先说清 softmax 把一行分数变成和为 1 的权重；再讲把非法连接设为 $-\infty$，被遮位置权重为 0；位置 2 的手算折叠｜520｜paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Scaled Dot-Product Attention
- #### 负无穷与实际用的有限负数｜只讲实现差异：换成一个很大的负数时，被遮位置的权重趋近 0 但不为 0；推导折叠｜160｜自行构造
- ### 掩码换来什么｜这一章讲掩码的两个结果，内容全在两个小节｜0｜paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex L12-L14
- #### 训练时一次算出所有位置｜先交代训练时解码器一次收到整条目标序列（相对输入右移一位）；再讲掩码与这个偏移一起保证位置 $i$ 只依赖它之前的已知输出，各位置因此无先后依赖、可并行算出｜400｜paper/src/model_architecture.tex Decoder:
- #### 解码时逐位置追加｜解码一次生成一个位置（一个词）；已经生成的位置不被遮，尚未生成的位置被遮，新位置只用到已生成的部分｜360｜paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex L12-L14
- ### 与相邻概念的区别｜只和最容易混淆的 padding mask 比一两句：一个遮未来位置、一个遮补齐的填充位｜200｜自行构造
- ### 适用条件与边界｜掩码只用在解码器自注意力；另两处注意力都不遮；训练时目标序列右移一位；分数矩阵仍是 $n \times n$、不省这次计算；两条误解各一两句｜400｜paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex Scaled Dot-Product Attention；paper/src/why_self_attention.tex L26

## 短页

- ### 只留它自己和它之前的位置｜是什么：一句话解释自注意力（每个位置看各位置的查询、键加权），再讲因果掩码把每个位置能看的位置限为它自己和它之前的位置，未来位置被遮住｜340｜paper/src/model_architecture.tex Applications of Attention in our Model；paper/src/model_architecture.tex Decoder:
- ### 不让模型提前看到答案｜为什么需要：解码一次生成一个位置，预测第 $i$ 个只能靠已经生成的前面几个，掩码保证不偷看未来｜360｜paper/src/model_architecture.tex Decoder:；paper/src/model_architecture.tex L12-L14
- ### 在被遮位置上把分数压到负无穷｜怎么做：每个位置与所有位置的查询、键算出分数，把未来位置设 $-\infty$，再按行归一成权重，被遮位置权重为 0｜300｜paper/src/model_architecture.tex Scaled Dot-Product Attention；paper/src/model_architecture.tex Applications of Attention in our Model
