# MoonViT-V2 术语表

| 术语 / 缩写 | 首次出现 | 定义或含义 |
|---|---|---|
| MoonViT-V2 | 页面开头 | Kimi K3 的视觉编码器，27 层 ViT，从零训练（next-token prediction） |
| MoonViT-3D | S1 | 报告中用作对照的视觉编码器变体，SigLIP 对比预训练初始化；与 MoonViT-V2 共享架构设计，区别在初始化与训练目标 |
| SigLIP / 对比预训练 | S1 | 把图像与文本编码进同一嵌入空间、用对比损失训练（配对靠近、不配对远离）的方法；偏向全局语义。概念页待生成 |
| 对比损失 | S1 | SigLIP 等用的训练目标，让配对图文相似度高、不配对相似度低 |
| 梯度范数 | S1 | 所有参数梯度组成向量的长度（标量）；持续高且尖峰频繁是训练不稳定的典型征兆 |
| next-token prediction（NTP） | S2 | 语言建模目标：给定已观察 token 预测下一个 token。概念页待生成 |
| ViT（Vision Transformer） | S3 | 把图像切成 patch、线性投影成向量、送入 transformer 层堆叠的架构。概念页待生成 |
| patch | S3 | 图像切分的小方块（MoonViT-V2 为 14×14 像素），每个 patch 投影成一个 token |
| 隐藏维 / vt_hidden_size | S3 | transformer 层内部表示维度，MoonViT-V2 为 1024（来自 config.json） |
| RMSNorm | S3 | 按均方根归一化、仅缩放增益、无 bias 的归一化。概念页待生成 |
| bias（偏置项） | S3 | 线性投影的可学习加性偏置；MoonViT-V2 去除所有 linear 与 attention 投影的 bias |
| 自注意力（self-attention） | S4 | token 间相互加权聚合的机制。概念页待生成 |
| 分解注意力 | S4 | 把视频注意力拆成帧内空间（一帧内 patch 间）+ 帧间时间（跨帧对应位置间）两趟 |
| 时间池化 | S4 | 沿时间维压缩 token 的操作 |
| pixel-shuffle / pixel unshuffle | S4 | 把 2×2 空间邻域搬进通道维的重排（此处为下采样方向）；无损重排，压缩序列长度不压信息量 |
| token | S4 | LLM 输入的单位；视觉 token 指视觉编码器输出、映射进 LLM 的向量 |
| MLP 投影器 | S4（提及） | 视觉编码器输出与 LLM 嵌入空间之间的轻量映射层；本文不展开其结构 |
| 1M-token 上下文 | S4 | K3 支持的最大上下文长度 |
| baseline | S5 | 对照基准，此处指 SigLIP 初始化的 MoonViT-3D |

符号约定：
- 维度记号用 config.json 字段名（vt_hidden_size 等）与口语名（隐藏维）并列首次出现，后续统一用口语名
- 数字来源统一标注：config.json = 官方配置；K3 报告 §2.4 = 原始论文
