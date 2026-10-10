# EAGLE-3 投机解码 draft 模型

- 描述：EAGLE-3 是投机解码里的一种草稿模型：只有一层 decoder，输入是 target 模型前向已经算出的低、中、高三层隐藏状态融合成的向量，直接输出下一个 token 的分布，并在训练时把自己多步的输出喂回去对齐推理。
- 摘要：草稿角色不再由另训的独立小模型承担，改由一层 decoder 直接读 target 前向算出的低、中、高三层隐藏状态预测 token，让多步草稿的接受率几乎不衰减。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 长页

- 开头｜概览：EAGLE-3 是什么、为什么需要、大概怎么工作；链接 [投机解码]，解释草稿模型、隐藏状态、token、decoder、全连接层｜430｜paper/src-v3/paper.tex L73-L76；paper/src-v3/paper.tex L98-L121
- ### 草稿模型为什么要读 target 的隐藏状态｜为什么草稿复用 target 的隐藏状态就能更准；另训小模型容量有限（一句带过）；前作 EAGLE 起用此法，EAGLE-3 进一步融三层｜450｜paper/src-v3/paper.tex L121；paper/src-v3/paper.tex EAGLE and EAGLE-2；paper/src-v3/paper.tex Speculative Sampling
- ### 低、中、高三层特征融成一个向量｜取 target 低中高三层各 k 维隐藏状态拼接，过全连接层压回 k 维得融合特征 g；只用最后一层不够｜520｜paper/src-v3/paper.tex Inference Pipeline；paper/src-v3/paper.tex L157
- ### 一步草稿：一层 decoder 加 target 的 LM head｜融合特征与上一 token 的嵌入拼接后过一层 decoder，再由 LM head 得到草稿 token；解释嵌入、LM head｜500｜paper/src-v3/paper.tex Inference Pipeline
- ### 多步草稿：用自己上一步的输出补位｜新位置还没有 target 特征，草稿用自己上一步的输出 a 代替 g，再与采样 token 的嵌入拼接；这样输入会偏离训练分布｜490｜paper/src-v3/paper.tex Inference Pipeline；paper/src-v3/paper.tex L124-L125
- ### 训练：把自己多步的输出喂回去｜去掉前作的特征预测损失、直接预测下一 token 后，训练时把草稿自己的输出喂回去，用因果掩码让它只见前文、模拟多步｜480｜paper/src-v3/paper.tex Draft Model Training；paper/src-v3/paper.tex L124-L125
- ### 草稿步数更深也不掉接受率｜n-α 是输入含 n 个自产向量、且此前草稿 token 都被接受时的接受率；EAGLE-3 随 n 增大几乎不变｜360｜paper/src-v3/paper.tex L306-L307；paper/src-v3/paper.tex L320
- ### 适用条件与边界｜target 中间层要对草稿可见；草稿只多一层 decoder 和融合全连接层；接受率随任务变；不改 target 与接受规则，分布不变｜400｜paper/src-v3/paper.tex L301；paper/src-v3/paper.tex Inference Pipeline；paper/src-v3/paper.tex L162-L164

## 短页

- ### 只读 target 特征的草稿模型｜一层 decoder 的草稿模型，读 target 隐藏状态融合向量预测 token；链接 [投机解码]，解释草稿模型、隐藏状态、token、decoder｜300｜paper/src-v3/paper.tex L73-L76；paper/src-v3/paper.tex Inference Pipeline
- ### 另训小模型当草稿的难处｜独立于 target 的小模型容量有限、难以逼近大模型；改读 target 已算出的隐藏状态，草稿先看到 target 的表示｜280｜paper/src-v3/paper.tex L121；paper/src-v3/paper.tex EAGLE and EAGLE-2
- ### 融合三层特征，一层 decoder 出草稿｜低中高三层隐藏状态拼接过全连接层得 g，与上一 token 的嵌入拼接，过一层 decoder 再由 LM head 得分布；解释全连接层、嵌入、LM head｜320｜paper/src-v3/paper.tex Inference Pipeline；paper/src-v3/paper.tex Draft Model Training
