# RMSNorm

- 描述：RMSNorm 是 LayerNorm 的简化：去掉减均值这一步，只把一层的加权和除以它的均方根，再乘增益；训练质量与 LayerNorm 相当而计算更省。
- 摘要：归一化本来要做减均值和除标准差两件事，RMSNorm 提出假设前者可以省掉，只保留 $\bar{a}_i = a_i / \mathrm{RMS}(\mathbf{a})\, g_i$。
- 主题：模型结构
- 标签：网络结构

## 长页

- 开头｜讲清一层怎么把输入算成加权和、归一化作用在它上面、增益与偏置这对可学参数；RMSNorm 只保留缩放的主张｜280｜paper/src-v1/rmsnorm-v1.tex Background；paper/src-v1/rmsnorm-v1.tex RMSNorm
- ### 只保留缩放｜本章讲 RMSNorm 与 LayerNorm 的差别，内容在两个小节，只有标题｜0｜paper/src-v1/rmsnorm-v1.tex RMSNorm
- #### 从 LayerNorm 去掉减均值｜LayerNorm 的减均值与除标准差两步、RMSNorm 公式、均方根的定义、乘增益保留在哪、输出为何不再零均值；减均值即中心化｜480｜paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background
- #### 手算对照｜取一个小向量分别算 LayerNorm 和 RMSNorm 的输出，看差在哪；全过程折叠｜300｜自行构造
- ### 缩放不变性与丢失的中心化不变性｜本章讲这个概念的不变性，内容在两个小节，只有标题｜0｜paper/src-v1/rmsnorm-v1.tex Invariance Analysis
- #### 整体缩放不变｜输入或权重整体乘一个正的常数，输出不变；原因是均方根的齐次性；推导折叠｜400｜paper/src-v1/rmsnorm-v1.tex Invariance Analysis
- #### 逐行缩放与平移会怎样｜只缩放个别权重行、或把输入平移，输出会改变，中心化不变性因此丢失；实践中去掉中心化质量仍相当、均值反而比基线更稳；实验数字与条件折叠｜400｜paper/src-v1/rmsnorm-v1.tex Invariance Analysis；paper/src-v1/rmsnorm-v1.tex Experiments；paper/src-v1/rmsnorm-v1.tex L353-L354
- ### 梯度上的表现｜解释学习率是什么；对增益和偏置的梯度对缩放不变，对权重的梯度带一个负相关项，起隐式学习率调节的作用；推导折叠｜350｜paper/src-v1/rmsnorm-v1.tex Gradient Analysis
- ### 省下的计算｜逐步比较两式的运算步数，列出 LayerNorm 多做的求均值与减均值；速度提升随实现变化；速度数字表折叠｜400｜paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Experiments；paper/src-v1/rmsnorm-v1.tex Conclusion and Future Work
- ### 适用条件与边界｜零均值输入时退化为 LayerNorm；论文把「只保留缩放」作为假设提出，靠实验支持而非证明｜300｜paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex L75；paper/src-v1/rmsnorm-v1.tex L133

## 短页

- ### 把加权和除以均方根｜是什么：交代层与加权和、增益与偏置、均方根；LayerNorm 减均值即中心化、除标准差即缩放，RMSNorm 去掉中心化只除均方根｜340｜paper/src-v1/rmsnorm-v1.tex RMSNorm；paper/src-v1/rmsnorm-v1.tex Background
- ### 中心化那一步可以省掉｜为什么需要：LayerNorm 的减均值要多算一步，深层网络里这笔开销可观；论文的假设是关键性质是缩放而不是中心化，去掉它质量不掉、速度更快｜330｜paper/src-v1/rmsnorm-v1.tex Introduction；paper/src-v1/rmsnorm-v1.tex RMSNorm
- ### 除均方根、乘增益｜怎么做：先算这一层所有分量的均方根，每个分量除上它，再乘一个可学习的增益向量；不再居中所以默认不要偏置｜300｜paper/src-v1/rmsnorm-v1.tex RMSNorm；repo/rmsnorm/rmsnorm_torch.py L13-L20
