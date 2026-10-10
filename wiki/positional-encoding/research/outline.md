# 位置编码

- 描述：位置编码是一段只随位置变化的向量，逐元素加到每个 token（文本切分后的基本单位）的输入嵌入（按 ID 查表得到的向量）上，告诉模型它在序列里的位置；它用固定频率的正弦和余弦生成。
- 摘要：输入嵌入按内容查表得到，本身不带先后，模型里也没有别的东西带来次序，所以要把一段只随位置变的编码加进输入。位置编码用不同频率的正弦和余弦写成，每一对维度一个频率：$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})$，位置 $pos$ 是序列里的下标。
- 主题：模型结构
- 标签：位置编码

## 长页

- 开头｜先一句话讲 token（文本切分后的基本单位）与输入嵌入，把自注意力链接成前置页；位置编码是什么、为什么需要、大概怎么做｜340｜paper/src-v7/model_architecture.tex Positional Encoding；paper/src-v7/model_architecture.tex Attention
- ### 注意力看不出 token 的先后｜自注意力对各位置的值加权求和、结果与次序无关（链接前置页）；模型没有循环或卷积（两者都顺着序列处理），次序只能另想办法带进来｜200｜paper/src-v7/model_architecture.tex Attention；paper/src-v7/model_architecture.tex Positional Encoding
- ### 加在输入嵌入上｜位置编码加到模型最底层的输入嵌入上；维度与嵌入相同才能逐元素相加；编码只由位置和维度下标定，与 token 内容无关｜260｜paper/src-v7/model_architecture.tex Positional Encoding
- ### 正弦式位置编码｜公式给出偶数维正弦、奇数维余弦，波长成几何级数；固定偏移下可由和角公式线性表示（推导与数值例子折叠）｜420｜paper/src-v7/model_architecture.tex Positional Encoding
- ### 适用条件与边界｜论文把注入的信息分相对位置（两位置的距离）与绝对位置（下标本身）两类，正弦式给绝对向量；它与学习式嵌入结果几乎一样，选正弦式是因为可能外推到更长序列（只是可能）｜280｜paper/src-v7/model_architecture.tex Positional Encoding；paper/src-v7/model_architecture.tex L155；paper/src-v7/results.tex L107

## 短页

- ### 一段随位置变的向量｜位置编码是什么：一段只随位置变化的向量，逐元素加到每个 token 的输入嵌入（token 是文本基本单位，嵌入按它的 ID 查表得到）上，让模型知道它排第几｜280｜paper/src-v7/model_architecture.tex Positional Encoding
- ### 注意力看不出先后｜为什么需要：注意力（链接 wiki/standard-attention）对各位置的值加权求和，权重只看查询与键、看不到次序；模型里没有别的东西带进次序｜320｜paper/src-v7/model_architecture.tex Attention；paper/src-v7/model_architecture.tex Positional Encoding
- ### 一对维度一个频率的正弦｜怎么做：位置 pos 的偶数维取 $\sin(pos/10000^{2i/d_{\text{model}}})$、奇数维取对应余弦，得到一个与嵌入同维的向量｜320｜paper/src-v7/model_architecture.tex Positional Encoding
