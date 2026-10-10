# DFlash

- 描述：DFlash 是一种投机解码方法：用一个块扩散草稿器一次前向并行猜出整块 draft token，并靠目标模型的隐藏特征条件化，再由目标模型一次验证。
- 摘要：把目标模型若干层的隐藏态拼成上下文特征，注入轻量块扩散草稿器的每一层 KV，一次前向并行猜出整块；验证仍由目标模型完成，输出因此无损。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 长页

- 开头｜讲清投机解码与草稿、目标两个模型，为什么需要 DFlash，它大致怎么做，并交代各章顺序｜340｜paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Inference
- ### 串行起草限制了加速｜推导平均延迟与加速比（折叠），对比自回归起草串行受限与扩散起草并行出整块，说明权衡；块扩散释义一两句｜360｜paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/preliminaries.tex Diffusion Drafting；paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/exp.tex Number of Draft Layers
- ### 用目标模型的隐藏特征条件化草稿｜无条件扩散草稿器为何只有 2–3×（表折叠），目标深层隐藏态隐含编码未来 token 信息（转引外部观察），prefill 时取若干层拼成条件特征｜500｜paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/appendix.tex Diffusion Drafter without Target Feature
- ### 把特征注入每一层的 KV 缓存｜目标特征只作 K/V、绕过 [Q 投影、输出投影](wiki/standard-attention) 与前馈（逐位置各算），公式折叠；融合差别一两句；消融表折叠｜500｜paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/appendix.tex KV Injection Mechanism and Memory Overhead；paper/src-v2/sections/exp.tex KV Injection vs. Input Fusion
- ### 训练：随机锚点、分块掩码与位置加权｜说明草稿器训练只更新草稿层、以目标特征作条件，下面两节展开｜100｜paper/src-v2/sections/method.tex Training
- #### 随机锚点构块｜从响应随机取干净 token 作块首、遮其后位置并行预测，块内双向块间禁看，一次前后向训多个块；构块示意与消融表折叠｜500｜paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Random Sampling of Masked Blocks
- #### 位置加权与共享嵌入｜早期位置错则整块作废、按位置指数衰减加权；与目标共用词嵌入表（ID 到向量）和输出头（隐藏态到词表分数）且冻结；损失公式折叠｜400｜paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Training Implementation
- ### 一次前向猜一整块｜推理时怎么把输入拼成偏移序列、一次前向取全块 token、目标模型一次验证、按最长接受前缀推进，与并行验证的共同点｜400｜paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；repo/dflash/dflash/model.py L234-L249；repo/dflash/dflash/model.py L282-L294
- #### 跟着走一遍｜设一个块大小与位置，走一遍取槽位、一次前向、并行取词、验证、推进全过程，并用代码核对；过程折叠｜100｜自行构造
- ### 适用条件与边界｜无损由验证保证；起草成本不随块大小变的成立范围；损失衰减超参与草稿长度同符号不同物；训练块比推理块大能向下泛化（非对称）；条件特征代表的时长｜340｜paper/src-v2/sections/preliminaries.tex Diffusion Drafting；paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Training Implementation；paper/src-v2/sections/exp.tex Training-Inference Time Block Size；paper/src-v2/sections/exp.tex Long Context Adaptation

## 短页

- ### 条件化的并行草稿器｜讲清 DFlash 是什么：以目标模型隐藏特征为条件的块扩散草稿器，一次前向并行猜出整块 draft token，再交目标模型验证；解释草稿模型、目标模型｜200｜paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Inference
- ### 起草环节限制了加速｜讲清为什么需要：投机解码把生成分成起草与验证，自回归起草一次前向只出一个 token，草稿一长就慢，浅草稿器又猜不准｜120｜paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/intro.tex Introduction
- ### 目标特征加块扩散｜讲清怎么做：取目标模型若干层隐藏态拼成特征、注入草稿器每层 KV，一次前向并行填补整块，目标模型一次验证｜150｜paper/src-v2/sections/method.tex Inference
