# Standard Attention

- 描述：标准注意力（缩放点积注意力）让每个位置用查询向量与所有位置的键向量点积得到相关度分数，经 SoftMax 归一成权重后对值向量加权求和，得到该位置的输出表示。
- 摘要：每个位置用查询去匹配所有位置的键，按匹配程度对值向量加权求和，一次算出整个序列的表示；除以 $\sqrt{d_k}$ 让分数幅度不随维度增大，多头在多个子空间并行做同一运算再拼接。
- 主题：注意力机制
- 标签：注意力

## 长页

- 开头｜先交代序列、token、嵌入、自回归；标准注意力是什么、为什么需要（循环串行、远距离路径长，注意力两两直接相连、可并行）、大概怎么做｜350｜paper/src-v7/model_architecture.tex L12；paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L139-L140；paper/src-v7/model_architecture.tex L143-L144；paper/src-v7/introduction.tex L3；paper/src-v7/introduction.tex L14；paper/src-v7/why_self_attention.tex L3；paper/src-v7/why_self_attention.tex L8；paper/src-v7/why_self_attention.tex L84
- ### 查询、键、值：把序列变成可匹配的向量｜每个位置的向量经三个投影得到查询、键、值；在单个序列内部做的叫自注意力，三者同源；投影矩阵形状｜350｜paper/src-v7/model_architecture.tex L82；paper/src-v7/model_architecture.tex L93；paper/src-v7/model_architecture.tex L107；paper/src-v7/background.tex L3
- ### 相关度：查询与键的点积｜用查询与所有位置的键点积得分数，矩阵一次算完 $S=QK^\top$；为什么用点积｜350｜paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L38；paper/src-v7/model_architecture.tex L40-L42；paper/src-v7/model_architecture.tex L44
- ### SoftMax 与加权求和｜分数经 SoftMax 成和为 $1$ 的权重，输出是值向量加权平均；SoftMax 定义折叠；自回归遮罩置 $-\infty$｜350｜paper/src-v7/model_architecture.tex L24-L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L41；paper/src-v7/model_architecture.tex L109
- ### 缩放：为什么除以 $\sqrt{d_k}$｜点积方差随维度增大，作者推测会把 SoftMax 推入梯度极小的区域，据此缩放；方差推导折叠｜350｜paper/src-v7/model_architecture.tex L53
- ### 多头：多个子空间并行｜$d_{model}$ 拆成 $h$ 个子空间并行做注意力，拼接再投影；为什么多头；多头公式与参数量折叠｜350｜paper/src-v7/model_architecture.tex L82；paper/src-v7/model_architecture.tex L85；paper/src-v7/model_architecture.tex L87-L93；paper/src-v7/model_architecture.tex L98-L99；paper/src-v7/background.tex L1
- ### 一个可以算完的例子｜自设 3 个位置、$d_k=2$ 的小矩阵，手算点积、缩放、SoftMax、加权求和；全过程折叠｜100｜自行构造
- ### 适用条件与边界｜缩放的假设、注意力本身不区分位置需另加位置编码、代价 $O(n^2)$、多头不增计算量一条误解｜300｜paper/src-v7/model_architecture.tex L53；paper/src-v7/model_architecture.tex L99；paper/src-v7/model_architecture.tex L143-L144；paper/src-v7/why_self_attention.tex L26

## 短页

- ### 按相关度从其他位置取值｜是什么：每个位置用查询去匹配各位置的键，按匹配程度从各自的值取信息，组成该位置的新表示；解释 token、嵌入、查询、键、值，键向量的维度记作 $d_k$｜300｜paper/src-v7/model_architecture.tex L25；paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L139-L140
- ### 为什么不用循环｜为什么需要：循环模型逐位置串行、远距离信息走长路径；注意力让任意两位置直接交互且可整序列并行｜300｜paper/src-v7/introduction.tex L3；paper/src-v7/why_self_attention.tex L3；paper/src-v7/why_self_attention.tex L8；paper/src-v7/why_self_attention.tex L84
- ### 一次算完整个序列｜怎么做：位置向量各乘可学矩阵得 $Q,K,V$，点积、除以 $\sqrt{d_k}$、SoftMax 归一、对 $V$ 加权求和，一次算出全部输出｜300｜paper/src-v7/model_architecture.tex L36；paper/src-v7/model_architecture.tex L40-L42；paper/src-v7/model_architecture.tex L82
