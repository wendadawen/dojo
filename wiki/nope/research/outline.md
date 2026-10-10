# NoPE

- 描述：NoPE 是去掉位置编码的解码器 Transformer：它靠因果掩码和训练隐式地表示位置，长度泛化上超过显式位置编码，且不增加注意力里的计算。
- 摘要：去掉位置编码的解码器 Transformer。因果掩码让每个位置能看到的 token 数不同，位置信息由此隐式进入网络；训练后它的注意力接近相对位置编码，近处远处都看，长度外推优于显式位置编码，且不增加注意力的点积计算。
- 主题：注意力机制, 模型结构
- 标签：位置编码

## 长页

- 开头｜先交代解码器、自注意力、隐藏状态、因果掩码、位置编码、相对编码；NoPE 是什么、为什么需要、大致怎么做｜260｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Background: Positional Encoding in Transformers；paper/src-v2/main.tex NoPE；paper/src-v2/main.tex T5's Relative PE
- ### 因果掩码让输出仍与次序有关｜若注意力能看到所有位置又不加位置编码，输出就与次序无关；解码器有因果掩码，每个位置只能看前面，输出仍与次序有关｜240｜paper/src-v2/main.tex Background: Positional Encoding in Transformers
- ### 位置信息从结构里自己形成｜本章回答位置怎样不靠显式编码进入网络，只有标题，内容在两个小节｜0｜paper/src-v2/main.tex How Does NoPE Represent Positions?
- #### 第一层能写出绝对位置｜用序列开头的锚点 token（固定表示序列起点的特殊 token）和因果掩码，让每个位置的注意力权重都是 1/t，t 就是位置；构造与小例子折叠｜280｜paper/src-v2/main.tex Absolute Positional Encoding in NoPE
- #### 后续层能实现相对编码｜绝对位置进了隐藏状态后（且不被后续层覆盖），构造查询和键，使点积多出一项只与位置差有关；推导折叠｜280｜paper/src-v2/main.tex Relative Positional Encoding in NoPE
- ### 训练后的注意力像相对编码｜用注意力分布的 JS 距离（衡量两个概率分布差异的量）量出 NoPE 与相对编码最接近；与其他编码的对比只留这一句，表折叠｜280｜paper/src-v2/main.tex NoPE learns to use relative PE in practice
- ### 注意力近处远处都看｜归一化注意力距离的分布呈双峰：近处看当前步、远处看输入，这是加法任务加完整思维链（让模型把中间步骤写进输出的解法）格式上的测量；数字折叠｜200｜paper/src-v2/main.tex Which part of the sequence is attended to?
- ### 不增加注意力的计算｜注意力的点积只由查询和键算出，不加任何与位置有关的额外项；运行时间和显存不随位置增加｜220｜paper/src-v2/main.tex What Is The Effect of Positional Encoding?；paper/src-v2/main.tex Model Evaluation
- ### 适用条件与边界｜成立条件；代价（1.3B 初步结果，数字折叠）；两个容易搞反的点：不加位置编码不等于没有位置信息，思维链不能替代位置编码｜220｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Pretraining at 1.3B Scale；paper/src-v2/main.tex Generalization Evaluation；paper/src-v2/main.tex Absolute Positional Encoding in NoPE；paper/src-v2/main.tex Does Scratchpad Render The Choice of Positional Encoding Irrelevant?

## 短页

- ### 去掉位置编码的解码器｜是什么：先用一两句讲清解码器 Transformer 与注意力，再讲 NoPE 就是这样的解码器不加任何位置编码｜250｜paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Background: Positional Encoding in Transformers；paper/src-v2/main.tex NoPE
- ### 不加计算反而更能外推｜为什么需要：长度外推指训练短序列、测试长序列时仍能算对；常用位置编码在外推上表现不好，NoPE 不增加注意力的计算，外推反而更好｜300｜paper/src-v2/main.tex What Is The Effect of Positional Encoding?；paper/src-v2/main.tex Introduction
- ### 位置从可见范围里读出来｜怎么做：注意力只让每个位置看它自己和前面的位置（这种掩码叫因果掩码），可见的 token 数随位置递增，网络从这个长度读出位置｜300｜paper/src-v2/main.tex Absolute Positional Encoding in NoPE；paper/src-v2/main.tex Background: Positional Encoding in Transformers