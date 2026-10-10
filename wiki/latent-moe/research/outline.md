# LatentMoE

- 描述：LatentMoE 是一种混合专家（MoE）架构：先把每个 token 压到一个低维隐空间，再交给路由专家，用省下的通信量和权重加载量把专家总数与每个 token 的激活数放大同样的倍数。
- 摘要：把路由专家放进低维隐空间计算，按压缩倍率 $\alpha=d/\ell$ 同倍放大专家总数与 top-$k$，专家组合空间指数扩大，通信量和权重加载量与标准 MoE 持平，同等推理成本下精度更高。
- 主题：模型结构, 训练与优化
- 标签：MoE

## 长页

- 开头｜先交代 MoE 层、隐藏维度与中间维两个量的含义、专家权重按它们摆放，再讲 LatentMoE 是什么、为什么需要、怎么投回成本｜500｜paper/src-v1/sections/introduction.tex Introduction；paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/latentmoe.tex LatentMoE Architecture
- ### 能省的只有隐藏维｜两条瓶颈（加载量随隐藏维与中间维、通信量随激活专家数与隐藏维）与两条约束（中间维乘激活数、隐藏维不低于任务特征秩）推出只有隐藏维能动；时间比推导折叠｜720｜paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/design_choices.tex Model Quality
- ### 把激活压进隐空间｜本章回答怎么把压缩做成一层能训练的结构，只有标题，内容在下面这一节｜0｜paper/src-v1/sections/latentmoe.tex LatentMoE Architecture
- #### 同倍放大专家数与 top-k｜降维矩阵、隐空间里的专家权重、回投矩阵三个公式与一层的完整公式都折叠；两种配置（只放大专家数、连激活数一起放大）各自的通信量、权重加载量与精度｜780｜paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/latentmoe.tex L26-L53
- ### 压缩比能压到多小｜消融扫压缩比测得无损区间并取定值，再测必须同时放大专家数否则质量下降；验证损失曲线折叠｜560｜paper/src-v1/sections/results.tex LatentMoE Ablations；paper/src-v1/sections/results.tex L58-L66
- ### 换来的精度与速度｜同等参数下精度更高、同等精度下计算量更小的结论，投影本身的开销占比；逐任务数字与万亿规模投影折叠｜300｜paper/src-v1/sections/results.tex LatentMoE Scaling Studies；paper/src-v1/sections/results.tex L82-L98；paper/src-v1/sections/results.tex Projected Serving Impact at Trillion-Parameter Scale
- ### 适用条件与边界｜压缩只覆盖路由专家这条通路、压缩倍率由实验而定、和 MoLAE 的区别一两句｜300｜paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/results.tex LatentMoE Ablations；paper/src-v1/sections/discussion.tex Related Work

## 短页

- ### 专家改在隐空间里算｜先一句讲 MoE（专家与路由）与隐藏维度，再讲 LatentMoE 是什么：把每个 token 先压成低维向量再交给路由专家，按压缩倍率放大专家总数与激活数｜340｜paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/introduction.tex Introduction
- ### 显存与通信两处瓶颈｜为什么需要：低延迟时卡在权重加载、吞吐时卡在把 token 送到各设备的通信，两者都随隐藏维度涨，而调专家数或激活数都换不来｜340｜paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/design_choices.tex Model Quality
- ### 降维，再按倍率放大专家｜怎么做：降维矩阵、隐空间专家、回投矩阵三步，专家总数与 top-$k$ 同倍放大，通信与权重加载量不变｜300｜paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/introduction.tex Introduction
