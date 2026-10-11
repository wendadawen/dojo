# MoE

- 描述：MoE（混合专家）是一种把一部分层的前馈网络换成一排专家的模型结构，路由器为每个 token 只选其中少数几个参与计算，总参数随专家数增长而每 token 计算量不随之成比例增长。
- 摘要：把一部分层的前馈网络换成一排并行的专家，每个 token 由路由器打分、只让分数最高的 $k$ 个专家参与计算，这一层的输出是这些专家输出的加权和。总参数随专家数一起变大，每个 token 的计算量却不随之成比例增长。
- 主题：模型结构, 推理系统
- 标签：MoE

## 长页

- 开头｜概览：MoE 是什么、为什么需要、大致怎么工作；先交代 token、参数、层与前馈网络｜360｜paper/src-shazeer/main.tex Introduction and Related Work；paper/src-dsv3/main.tex Introduction
- ### 一排专家加一个路由器｜MoE 层怎么算：路由器打分、留 top-k、加权求和；可选共享专家（对所有 token 都算、不经路由）；输出公式与 8 选 2 例子各折叠｜700｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-shazeer/main.tex Gating Network；paper/src-dsv3/main.tex Architecture
- ### 参数多、每 token 计算少｜稀疏激活把容量和每 token 计算量分开：DeepSeek-V3 的 256 路由、top-8、671B/37B｜360｜paper/src-dsv3/main.tex Introduction；paper/src-dsv3/main.tex Hyper-Parameters；paper/src-shazeer/main.tex Introduction and Related Work
- ### 路由集中在少数专家上｜路由不均：路由器收敛到总选同几个专家并自我强化；训练侧用重要性/负载损失或无辅助损失的偏置来均衡｜420｜paper/src-shazeer/main.tex Balancing Expert Utilization；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing
- ### 适用条件与边界｜k 远小于专家数才省计算；一两句带出它落地到专家并行与服务两页｜280｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Hyper-Parameters

## 短页

- ### 一排专家，每个 token 只让几个参与计算｜是什么：先交代 token 与前馈网络，再讲 MoE 把前馈网络换成一排专家加一个路由器，每个 token 只选其中几个｜340｜paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer；paper/src-dsv3/main.tex Architecture
- ### 参数多、每 token 计算少｜为什么需要：稠密模型（每个 token 都过全部参数的普通模型）参数越大每 token 计算量越大；MoE 让总参数随专家数增长而每 token 只算选中的几个｜300｜paper/src-dsv3/main.tex Introduction；paper/src-shazeer/main.tex Introduction and Related Work
- ### 怎么选出这几个专家｜怎么做：路由器给每个专家一个分数，留下分数最高的 $k$ 个，按分数加权求和；这就是路由选择｜300｜paper/src-dsv3/main.tex Architecture；paper/src-shazeer/main.tex Gating Network
