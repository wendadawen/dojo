# Block Diffusion

- 描述：块扩散语言模型把 token 序列切成块，块间按自回归从左到右建模，块内用离散掩码扩散并行去噪。
- 摘要：块间自回归、块内并行去噪，块大小 $L'$ 在自回归（$L'=1$）与扩散（$L'=L$）之间插值，块间可复用 KV cache，能生成任意长度的序列。
- 主题：模型结构, 训练与优化
- 标签：模型架构

## 长页

- 开头｜先讲自回归模型与离散去噪扩散各是什么，块扩散是什么、为什么需要、大概怎么工作；交叉熵、注意力、KV cache 首次出现处链接｜400｜paper/src/iclr2025_conference.tex Introduction；paper/src/iclr2025_conference.tex L292-L301；paper/src/iclr2025_conference.tex L354-L377；paper/src/iclr2025_conference.tex L379-L380
- ### 块扩散的似然分解｜序列切块后似然按块分解，每块的条件概率用离散扩散建模，两端各自退化成什么；NELBO 推导与单 token 等价折叠｜450｜paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures；paper/src/iclr2025_conference.tex L470-L473；paper/src/iclr2025_conference.tex Block Diffusion NELBO；paper/src/iclr2025_conference.tex Recovering the NLL from the NELBO for Single Token Generation
- ### 块间因果、块内双向的注意力｜一个网络同时参数化所有块，块内 token 互相看，块 b 只看它之前的块；掩码三部分和一个小掩码例子折叠｜400｜paper/src/iclr2025_conference.tex Model Architecture；paper/src/iclr2025_conference.tex Specialized Attention Masks
- ### 每 token 过两次模型的训练算法｜为什么不能一次前向，两次前向各做什么，噪声与干净序列拼接的向量化掩码；加速数字与训练伪代码折叠｜450｜paper/src/iclr2025_conference.tex L433；paper/src/iclr2025_conference.tex Training；paper/src/iclr2025_conference.tex Vectorized Training；paper/src/iclr2025_conference.tex Efficiency of Training Algorithm；paper/src/algs/short_algs.tex L49-L67
- ### 逐块生成、块内并行的采样｜一次生成一个块并缓存它的 KV，块内并行去噪，任意长度怎么做到、生成步数上界；采样伪代码与长度统计折叠｜400｜paper/src/iclr2025_conference.tex Sampling；paper/src/iclr2025_conference.tex Variable-Length Sequence Generation；paper/src/iclr2025_conference.tex Sample Quality and Variable-Length Sequence Generation；paper/src/algs/short_algs.tex L69-L89
- ### 梯度方差与低方差噪声调度｜均匀采样的掩码率为何让梯度方差大、极端掩码率为何没用、单 token 的方差与差距；截断调度与按块大小定范围；单 token 方差数字、方差公式、调度对比表折叠｜400｜paper/src/iclr2025_conference.tex Diffusion Gap from High Variance Training；paper/src/iclr2025_conference.tex Case Study: Single Token Generation；paper/src/iclr2025_conference.tex Intuition: Avoid Extreme Mask Rates；paper/src/iclr2025_conference.tex Clipped Schedules for Low-Variance Gradients；paper/src/iclr2025_conference.tex Data-Driven Clipped Schedules Across Block Sizes
- ### 一个四 token 的例子｜取 4 个 token 分两块，指定各块掩码位置与时间，手算一次块扩散损失；手算过程折叠，折叠外只留设定与结论｜90｜自行构造
- ### 适用条件与边界｜似然下界的松紧随块大小变化，训练比扩散贵，块顺序生成的代价，最优块大小依任务而定；与最容易混淆的扩散 LM 一句取舍｜300｜paper/src/iclr2025_conference.tex Tightness of the NELBO；paper/src/iclr2025_conference.tex Limitations；paper/src/iclr2025_conference.tex Discussion and Prior Work

## 短页

- ### 分块去噪的语言模型｜是什么：序列切块，块间自回归、块内并行去噪；本节内一句话讲清自回归与掩码扩散｜330｜paper/src/iclr2025_conference.tex L226-L230；paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures
- ### 扩散模型缺的东西它补上｜为什么需要：扩散 LM 只能定长、似然落后，同时算所有位置所以不能复用缓存；本节内一句话讲清「把已生成部分的键和值留下来复用」；分块后可变长、可复用、似然提升｜330｜paper/src/iclr2025_conference.tex Introduction；paper/src/iclr2025_conference.tex Likelihood Evaluation
- ### 切块、块间自回归、块内并行｜怎么做：切块、块内并行去噪、逐块采样并复用缓存这条主线，每步一句，不写两次前向和噪声调度｜330｜paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures；paper/src/iclr2025_conference.tex Sampling
