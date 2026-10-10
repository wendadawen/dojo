# Delta 规则与 DeltaNet

- 描述：DeltaNet 是采用 delta 规则更新记忆的线性 Transformer：写入新键值对时，先按当前键把记忆里已有的值取出来，只按预测误差改写这条记忆，而不是直接叠加一个新键值对。
- 摘要：线性 Transformer 的记忆用加法更新，固定容量下键一多就互相碰撞。delta 规则把写入改成纠错——取回键上已有的值，朝目标值改一步；DeltaNet 的训练靠把每步更新写成可累加的形式，分块并行到序列长度上。
- 主题：注意力机制, 模型结构
- 标签：注意力

## 长页

- 开头｜线性注意力是什么（一两句）、加法记忆、delta 规则与 DeltaNet 是什么、为什么要把训练并行化；碰撞机制留到下一节｜350｜paper/src-v6/sections/intro.tex Introduction；paper/src-v6/sections/background.tex Linear Transformer: Transformers with Linear Attention；paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule
- ### 一条能改写记忆的规则｜本章回答记忆怎么改写，只有标题，内容在两个小节｜0｜paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule
- #### 加法更新为什么会失效｜线性注意力的记忆是固定大小的矩阵状态，先解释召回；直接叠加新键值对难以释放旧关联，键数超过状态维数时互相碰撞，固定容量下记不住太多｜300｜paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule
- #### 用预测误差改写记忆｜delta 更新式三项各是什么、$\beta_t$ 的作用；损失与梯度各一两句讲清，推导和取回旧值再插值的解读都折叠｜450｜paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule
- ### 训练：从逐步更新到分块并行｜本章回答分块并行怎么做到，只有标题，内容在四个小节｜100｜paper/src-v6/sections/method.tex Chunkwise Parallel Form for DeltaNet
- #### 改写成累加形式｜用伪值向量 $u_t$ 把记忆写成纯累加形式，$u_t$ 的递推只用向量、不 materialize 矩阵状态的推导折叠｜400｜paper/src-v6/sections/method.tex A Memory-efficient Reparameterization
- #### 每一步更新是一个 Householder 变换｜一两句讲清 Householder 变换；delta 更新等价于乘一个广义 Householder 矩阵，$\beta_t=1$ 且键为单位向量时退化成投影矩阵｜320｜paper/src-v6/sections/method.tex A Memory-efficient Reparameterization；paper/src-v6/sections/method.tex Feature map and normalization
- #### 分块并行算法｜把序列切成块，块间状态递推、块内用注意力一次算完；块内的 $W,U$ 是有下标的矩阵各由一个前缀递推算出，公式与伪代码折叠｜620｜paper/src-v6/sections/method.tex Chunkwise Parallel Form for DeltaNet
- #### 变换的矩阵形式与实现｜一两句讲清 UT 变换；用它把 $W,U$ 的逐步递推换成一次矩阵求逆再算；复杂度与顺序步数（引线性注意力结论），对比图折叠｜420｜paper/src-v6/sections/method.tex Practical considerations；paper/src-v6/sections/background.tex Chunkwise parallel form
- ### 例子：两次写入怎么改同一条记忆｜用 $d=2$、键取单位向量依次写入两条键值对，跟着走完每一步记忆的数值；全过程折叠｜280｜自行构造
- ### DeltaNet 这一层｜这一层有哪些向量和标量输入、各怎么算，怎么替换自注意力拼成一个 Transformer 层（自注意力作前置概念链接）；公式折叠｜300｜paper/src-v6/sections/method.tex DeltaNet Transformer
- ### 召回密集任务上的提升｜delta 规则在召回密集任务上相比加法更新带来的提升；实验条件与数字折叠｜250｜paper/src-v6/sections/experiment.tex Synthetic Benchmarks；paper/src-v6/sections/experiment.tex Results；paper/src-v6/tables/mad.tex L13-L18；paper/src-v6/tables/lm_harness.tex L13-L31
- ### 适用条件与边界｜状态规模、训练速度和长度外推上的限制与代价｜300｜paper/src-v6/sections/experiment.tex Limitations and Future Work

## 短页

- ### 记忆里改写而不是只往里加｜DeltaNet 是什么：采用 delta 规则的线性 Transformer（简短解释），写入新键值对时先取回旧值再按误差改写；解释状态当记忆｜270｜paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule
- ### 加法更新记不住太多｜为什么需要：纯粹叠加难以释放旧关联，键数超过状态维数时碰撞（讲清碰撞），固定容量下召回差；纠正写回能让同样的记忆装更多｜330｜paper/src-v6/sections/background.tex DeltaNet: Linear Transformers with the Delta Update Rule；paper/src-v6/sections/intro.tex Introduction
- ### 分块并行地训练｜怎么做：把每步更新写成可逐项相加的形式，再把序列切成块，块内一次矩阵运算算完、块间只传一个记忆状态｜320｜paper/src-v6/sections/method.tex A Memory-efficient Reparameterization；paper/src-v6/sections/method.tex Chunkwise Parallel Form for DeltaNet
