# 超连接

- 描述：超连接是一种替代残差连接的连接方式：把网络的初始输入复制成 $n$ 条并排的流，每层的输入由这 $n$ 条流加权求和得到，层的输出再按权重分发回各条流；两组权重可训练，也可以由输入算出。
- 摘要：把网络的初始输入扩成 $n$ 条并排的流，用一组可训练的权重决定每条流对当前层输入、以及层输出对每条流的贡献；残差连接是 $n=1$ 时的不可训练特例。
- 主题：模型结构
- 标签：模型架构

## 长页

- 开头｜残差连接一句并链前置页；超连接是它的替代，把网络的初始输入复制成 $n$ 条并排的流、让连接的强度可学｜300｜paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Method
- ### 残差连接的取舍与固定的连接强度｜为什么需要：Pre-Norm 在层前归一化、Post-Norm 在层后归一化，二者各自失效方式像跷跷板；连接强度不由网络学习｜220｜paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/src-v3/app.tex Post-Norm Residual Connection
- ### 把网络的初始输入扩成 n 条：记号与连接矩阵｜静态超连接的记号与形状、初始复制与末尾按行求和、两组连接权重与更新式；隐藏状态一句带过｜440｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/app.tex Transformer with Hyper-Connections
- #### 一层里走一遍｜一层的计算顺序与残差的对应；$n=2$ 的一步小矩阵手算折叠（自行构造）｜300｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/app.tex Transformer with Hyper-Connections
- #### 连成网络以后：层怎么摆｜把更新式展开成层间连接矩阵（过程折叠）；顺序与并行两种排列各对应一个矩阵；Pre-Norm、Post-Norm 是 $n=1$ 的不可训练超连接｜300｜paper/src-v3/iclr2025_conference.tex Sequential-Parallel Duality；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Unfolding hyper-connections
- ### 动态超连接：权重由输入算出｜静态与动态的区别；权重经归一化、线性变换、tanh 与小因子放缩算出，再叠加静态偏置；各符号含义与公式折叠｜300｜paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections
- ### 初始化：从残差连接出发｜把静态矩阵初始化成 Pre-Norm 残差连接，动态参数初始化为零，使训练开始时网络等价于 Pre-Norm；输出权重按 $\sqrt{n}$ 放缩｜240｜paper/src-v3/iclr2025_conference.tex Initialization；paper/src-v3/iclr2025_conference.tex L429-L431
- ### 效果：n 取多大｜主文一句设计取舍：$n=1$ 不行、$n>1$ 才有效、再大收益趋平；实验设置、消融与结果表折叠｜180｜paper/src-v3/iclr2025_conference.tex Ablation Study；paper/src-v3/iclr2025_conference.tex 7B Models；paper/src-v3/iclr2025_conference.tex MoE Models
- ### 学到了什么样的连接｜在连接矩阵里观察到：$\Lambda$ 形的长期衰减、早期层被反复访问、浅层注意力长期连接少而 FFN 输出被长期保留；限定在原文的单一模型检查点上｜240｜paper/src-v3/iclr2025_conference.tex Visualization Analysis；paper/src-v3/sec_visualization.tex L13-L18；paper/src-v3/app.tex Visualization for hyper hidden
- ### 适用条件与边界｜代价：参数与算力增量、训练时激活显存增加及其回收办法（激活显存一句解释）；误解「加宽 $n$ 倍代价就大」；实测的硬件条件｜240｜paper/src-v3/app.tex Parameters, Computation and Memory Footprint Analysis；paper/src-v3/app.tex Memory Footprint；paper/src-v3/iclr2025_conference.tex Introduction

## 短页

- ### n 条并排的流｜是什么：残差连接只有一条通路；超连接把网络的初始输入复制成 $n$ 条并排的流（残差连接链前置页）｜220｜paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections
- ### 连接强度由谁定｜为什么需要：两种常见做法（在层前或层后归一化）各自失效方式不同、连接强度训练前定死；超连接把一层里两组权重的取值交给网络自己学｜220｜paper/src-v3/iclr2025_conference.tex Introduction
- ### 汇入、过层、分发回各流｜怎么做：各流加权求和成当前层输入、过层、输出按权重分发回各流；这些权重可以固定，也可以由网络学｜320｜paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections
