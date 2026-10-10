# RoPE

- 描述：RoPE（旋转位置编码）把查询和键向量按各自的位置旋转一个角度，使两者点积里的位置信息只以相对位置的形式进入。
- 摘要：位置 $p$ 的向量按一组预设角频率 $\theta_i$ 旋转，每对分量转 $p\theta_i$。旋转是正交的，两个向量点积后角度差只剩相对位置；角频率取 $10000^{-2(i-1)/d}$ 时，点积随相对距离增大而衰减。
- 主题：模型结构, 注意力机制
- 标签：位置编码

## 长页

- 开头｜前置：自注意力（链接 wiki/standard-attention）；位置编码两类：绝对式把位置向量加到表示里，相对式；RoPE 属相对类｜300｜paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L110；paper/src-v5/roformer_arxiv.tex L127-L136；paper/src-v5/roformer_arxiv.tex L146-L160
- ### 点积里为什么放得下位置｜自注意力的分数只由查询与键的点积定、对次序不敏感，所以要把位置编码进这个点积；约束：点积中的位置信息只由两位置的差 $m-n$ 进入，点积仍依赖内容｜320｜paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L218-L222
- ### 二维：位置就是一个旋转角｜$d=2$ 时把向量看成平面上的点、按位置 $m$ 转一个角 $m\theta$，解约束得到位置就是旋转角；$\theta$ 预设非零；推导与例子折叠｜390｜paper/src-v5/roformer_arxiv.tex L227-L260；paper/src-v5/roformer_arxiv.tex L328-L432
- ### 高维：每两个分量一组旋转｜把 $d$ 维分成 $d/2$ 组二维子空间、每组配一个预设角频率 $\theta_i$，得到旋转矩阵；点积只剩相对角度 $m-n$｜390｜paper/src-v5/roformer_arxiv.tex L262-L288；paper/src-v5/roformer_arxiv.tex L289
- ### 角频率与长程衰减｜角频率随 $i$ 几何递减、各对分量转向快慢不同；旋转是正交的、范数不变；频率取值与衰减推导折叠｜340｜paper/src-v5/roformer_arxiv.tex L283；paper/src-v5/roformer_arxiv.tex L289
- #### 远距离的相关度会衰减｜把点积按各对分量写成含 $\cos((m-n)\theta_i)$ 的和，说明它的上界随相对距离增大而衰减，符合远距离依赖更弱的直觉；上界推导折叠｜220｜paper/src-v5/roformer_arxiv.tex L304-L305；paper/src-v5/roformer_arxiv.tex L483-L504
- ### 用逐元素乘加算｜把旋转改写成取相邻分量、乘余弦正弦再相加，避开矩阵乘；$d=4$、$m=1$ 的自设例子与代码都折叠｜340｜paper/src-v5/roformer_arxiv.tex L289；paper/src-v5/roformer_arxiv.tex L435-L478；repo/bert4keras/bert4keras/backend.py L336-L377
- ### 加在自注意力的哪一步｜加在查询、键的投影之后、算分数之前，值不旋转；不增参数｜300｜paper/src-v5/roformer_arxiv.tex L206；paper/src-v5/roformer_arxiv.tex L286-L289；repo/bert4keras/bert4keras/layers.py L519-L520
- ### 适用条件与边界｜成立的几条：$d$ 取偶数、按位置旋转得到相对式、序列长度可外推、远距衰减；代价是逐元素运算；可外推到线性算的注意力；一句点明它相对加性做法是乘法而非加法｜300｜paper/src-v5/roformer_arxiv.tex L93；paper/src-v5/roformer_arxiv.tex L263；paper/src-v5/roformer_arxiv.tex L267-L289；paper/src-v5/roformer_arxiv.tex L292；paper/src-v5/roformer_arxiv.tex L305；paper/src-v5/roformer_arxiv.tex L307-L327

## 短页

- ### 旋转出来的位置｜是什么：把查询和键的每对分量按位置旋转一个角度，点积里的位置信息只以 $m-n$ 进入；自注意力作为前置概念链接 wiki/standard-attention｜300｜paper/src-v5/roformer_arxiv.tex L110；paper/src-v5/roformer_arxiv.tex L262-L288
- ### 分数要看出先后｜为什么需要：位置编码要让分数只含相对位置、并随距离减弱，加性做法把位置加到表示里、相对项只能近似取（一两句）；RoPE 用旋转做到这两点｜300｜paper/src-v5/roformer_arxiv.tex L104；paper/src-v5/roformer_arxiv.tex L199；paper/src-v5/roformer_arxiv.tex L292；paper/src-v5/roformer_arxiv.tex L304-L305
- ### 按位置转，逐元素算｜怎么做：维度两两配对，每对按 $p\theta_i$ 旋转，$\theta_i=10000^{-2(i-1)/d}$，只用在查询和键上｜320｜paper/src-v5/roformer_arxiv.tex L262-L288；paper/src-v5/roformer_arxiv.tex L435-L478
