# 线性注意力

- 描述：线性注意力把注意力里查询与键的相似度写成核的特征映射内积（相似度因此非负），用矩阵乘法的结合律交换求和顺序，把计算量从随序列长度平方增长降到线性增长。
- 摘要：把注意力分数写成 $\phi(q)^\top \phi(k)$ 的内积，先算所有键值的外积之和再乘查询，复杂度从 $O(N^2)$ 降到 $O(N)$；因果形式下这个和就是固定大小的状态，逐步更新，推理等价于循环网络。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 长页

- 开头｜标准注意力（前置页 standard-attention）的时间与内存随序列长度平方增长；线性注意力保留按相似度加权取值，另换计算顺序把两者降到线性｜350｜paper/src-v3/arxiv.tex Introduction；paper/src-v3/arxiv.tex Linearized Attention
- ### 把相似度换成特征内积｜相似度只要非负即可作注意力；解释核与特征映射，核有特征映射 φ 时相似度写成 $\phi(Q_i)^\top\phi(K_j)$；推导折叠｜450｜paper/src-v3/arxiv.tex Linearized Attention
- ### 交换求和顺序｜结合律交换两个矩阵乘积的顺序，键值外积之和只算一次；降复杂度与两种复杂度对比折叠｜400｜paper/src-v3/arxiv.tex Linearized Attention；paper/src-v3/arxiv.tex Feature Maps and Computational Cost
- #### 特征映射的选取｜φ 使相似度为正，取 $\phi(x)=\mathrm{elu}(x)+1$（解释 elu 与 relu）；多项式核（解释它）有有限维特征映射，计算量折叠｜300｜paper/src-v3/arxiv.tex Feature Maps and Computational Cost
- ### 因果掩码下的状态形式｜因果掩码（前置页 causal-mask）把求和限到 $j\le i$；引入状态 $S_i$、$Z_i$，输出由状态算出，状态逐位置常数时间更新，仍 O(N)｜450｜paper/src-v3/arxiv.tex Causal Masking
- #### 小例子：四个位置走一遍｜自己设的小向量与小特征映射，跟着算两种顺序结果一致、状态如何逐步累积，全过程折叠｜100｜自行构造
- #### 训练并行、推理循环｜训练可并行（整条序列已知），推理不能并行、用上一节状态逐步推进，一层因此等价于循环网络；梯度公式与两种复杂度折叠｜400｜paper/src-v3/arxiv.tex Gradient Computation；paper/src-v3/arxiv.tex Training and Inference；paper/src-v3/arxiv.tex Transformers are RNNs
- ### 适用条件与边界｜收益随序列变长而增大、特征维度决定计算量；两条误解：一般不等于 softmax 注意力（后者相似度是无限维核，精确的无法线性化）、几千倍加速只限自回归生成｜350｜paper/src-v3/arxiv.tex Image Generation；paper/src-v3/arxiv.tex Feature Maps and Computational Cost；paper/src-v3/arxiv.tex Transformers are RNNs

## 短页

- ### 把注意力分数换成内积｜线性注意力是什么：在标准注意力（前置页 standard-attention）基础上，相似度由指数打分换成能写成两个特征向量内积的函数（核与 φ 先讲清）｜250｜paper/src-v3/arxiv.tex Linearized Attention
- ### 长序列下的平方代价｜为什么需要：标准注意力要对每一对位置算分数并留下 $N\times N$ 矩阵，时间与内存随序列长度平方增长；线性注意力把它降为线性｜300｜paper/src-v3/arxiv.tex Introduction；paper/src-v3/arxiv.tex Linearized Attention
- ### 先求和再乘查询｜怎么做：先把所有键值外积求和再乘查询，得到一个与特征维度同大的矩阵，不与序列长度同增｜350｜paper/src-v3/arxiv.tex Linearized Attention
