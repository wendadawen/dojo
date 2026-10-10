# Gated DeltaNet

- 描述：Gated DeltaNet 是只用一条递归写记忆规则的线性注意力模型：状态是一张键值关联的矩阵，每个位置先按门控抹掉旧关联，再按 delta 规则精确改写其中一条。
- 摘要：状态递推 $S_t = S_{t-1}\big(\alpha_t(I-\beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top$，门控 $\alpha_t$ 抹除旧记忆，delta 规则按写入强度 $\beta_t$ 精确改写一条关联。
- 主题：模型结构, 训练与优化
- 标签：注意力

## 长页

- 开头｜概览：这个概念是一条把键值对存进固定大小矩阵状态的递推；为什么需要：状态能装的关联有限，既要能丢掉不相关的关联，又要能精确改写一条｜400｜paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule
- ### 记忆：一条矩阵递推｜规则的基础：状态是一张矩阵，这条规则建立在这条递推上；线性注意力作为前置概念链接到它的页面，不展开写入与读出｜100｜paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule
- #### 外积写入与读出｜外积怎样把第 $t$ 个键值对写进状态、读出怎样按查询取回，状态大小为什么与序列长度无关；矩阵展开折叠｜250｜paper/src-v3/main.tex Mamba2: Linear Attention with decay
- #### 衰减：整体按比例缩小｜标量衰减门 $\alpha_t$ 把整个状态等比缩小、累积成乘积，说明这是不分对象的整体遗忘｜150｜paper/src-v3/main.tex Mamba2: Linear Attention with decay
- #### delta 规则：擦掉一条旧的再写新的｜delta 规则先读回键上的旧值、按写入强度 $\beta_t$ 擦掉它、再写新值，说明它只改一条关联、与整体衰减不同｜150｜paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule；paper/src-v3/main.tex Introduction
- #### 合起来：门控 delta 规则｜把衰减与 delta 合成一条递推，说明两个参数各管一件事；从在线回归的一步梯度下降看 $\beta_t$ 与 $\alpha_t$ 各是什么，梯度下降推导折叠｜450｜paper/src-v3/main.tex Formulation: Gated Delta Rule
- ### 一个可以手算的例子｜自行构造的小维度单头例子，走几步看三种操作各起什么作用；完整数字过程与每步结果全部折叠｜100｜自行构造
- ### 分块并行训练｜为什么逐位置递推不能并行、分块怎样把序列切块并用块首状态连起来，块内增量算成矩阵乘；块内递推与 WY 表示折叠｜750｜paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Algorithm: Hardware-efficient Chunkwise training
- #### 扩展 WY 表示｜把门控并进 WY 表示的递推式，说明它为什么等价于原来的递推；数学归纳证明折叠｜150｜paper/src-v3/main.tex Extended WY representation for gated delta rule
- ### 网络里的 Gated DeltaNet｜这条规则怎么组成一个层：$\vq,\vk,\vv$ 的产生、$\alpha,\beta$ 怎么算、短卷积与输出门做什么、状态多大；结构图与各组件消融折叠｜600｜paper/src-v3/main.tex Token mixer block.；paper/src-v3/figures/model.tex L409；paper/src-v3/main.tex Ablation Study；repo/GatedDeltaNet/lit_gpt/gated_delta_net.py L157-L165
- ### 适用条件与边界｜状态大小决定记忆容量、线性复杂度的代价；检索任务上 delta 与门控各起什么作用、训练长度外推到长序列；与相近概念一句区别；S-NIAH 数字折叠｜300｜paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Gated Delta Networks and Hybrid Models；paper/src-v3/main.tex Case study: Single Needle in a Haystack (S-NIAH)；paper/src-v3/main.tex Length extrapolation on long sequences

## 短页

- ### 一张会写也会忘的矩阵状态｜是什么：状态是一张键值关联的矩阵，给出 gated delta rule 递推式并说明每个符号是什么｜280｜paper/src-v3/main.tex Formulation: Gated Delta Rule；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule
- ### 门控与 delta 各补哪块短板｜为什么需要：状态大小不随序列长度增长、能装的关联有限、更新代价与序列长度无关；门控让不相关的关联被抹掉，delta 让其中一条被精确改写，两者互补｜340｜paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule
- ### 一步更新与代价｜怎么做：状态先按门控等比缩小、再按 delta 改写一条关联，其余方向不动；更新代价与序列长度无关｜280｜paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Formulation: Gated Delta Rule
