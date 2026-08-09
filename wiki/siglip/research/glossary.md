# SigLIP 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 概念与术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| SigLIP | 页面开头 | Sigmoid Loss for Language-Image Pre-training；本文核心概念，把 CLIP softmax 损失换成 sigmoid 二分类损失 |
| CLIP | S1 | Contrastive Language-Image Pre-training（Radford et al. 2021）；用 softmax 对比损失做视觉-语言预训练，本文的对照基线 |
| InfoNCE | S1 | CLIP softmax 损失的一般形式（Oord et al. 2018）；本文只使用 CLIP 的具体形式 |
| 对比学习 | S1 | 通过"拉近正对、推远负对"学习表示的范式；CLIP 与 SigLIP 都属此范式 |
| 双塔架构 | S1 | 图像塔 + 文本塔 + 共享嵌入空间的模型结构；CLIP 与 SigLIP 共享此架构 |
| 图像塔 / 视觉编码器 | S1 | 把图像映射到嵌入的编码器；常用 ViT（已有概念页） |
| 文本塔 / 文本编码器 | S1 | 把文本映射到嵌入的编码器；常用 Transformer |
| softmax 损失 | S1 | CLIP 使用的对比损失，含分母 $\sum_j$ 的归一化项 |
| sigmoid 损失 | S2 | SigLIP 使用的损失，逐对二分类交叉熵 |
| 二分类交叉熵 | S2 | $-\log\sigma(z\cdot a)$ 形式的损失；正类 $z=+1$，负类 $z=-1$ |
| batch / 批 | S1 | 一次训练迭代用的样本集合；大小记 $|B|$ |
| 正对 / 正样本对 | S2 | batch 内配对的 (image, text) 对，$z_{ij}=+1$ |
| 负对 / 负样本对 | S2 | batch 内不配对的 (image, text) 对，$z_{ij}=-1$ |
| hardest negative | S1 | batch 内与查询相似度最高的负样本；softmax 梯度主要由它决定 |
| all-gather | S1 | 分布式训练中的通信原语；把所有设备的嵌入聚合到每个设备；CLIP softmax 需要它 |
| chunked 实现 | S4 | 论文 §3.3 提出的 SigLIP 分布式实现；不需 all-gather，分块计算 |
| Locked-image Tuning (LiT) | S4 | 论文 §4.4 的训练设置——冻结图像塔、仅训文本塔；SigLiT = SigLIP + LiT |
| SigLiT | S4 | 论文 §4.4 的具体配置——用 SigLIP 损失 + LiT 设置 |
| mSigLIP | S4 | 多语言 SigLIP；论文 §4.3 用 100 种语言 + bottleneck token embedding |
| SigLIP-2 | 文末 | 2024+ 后续工作；加入自蒸馏、掩码预测；不在本页范围 |
| WebLI | S4 | Google 内部图像-文本数据集；论文训练用 |
| ImageNet zero-shot | S4 | 不微调直接用预训练模型做 ImageNet 分类；CLIP/SigLIP 标准评估协议 |
| TPUv4 / TPUv3 | S4 | Google 训练加速器；论文用 TPUv4-core-days 计量算力 |
| MoonViT-V2 | S4 | K3 的视觉编码器；已有概念页；本页只引用其选择"不从 SigLIP 初始化"的判断 |
| K3 | S4 | Kimi K3 模型；MoonViT-V2 是其视觉编码器 |
| 互信息估计 | 不出现 | InfoNCE 的理论背景；本文不展开 |
| 温度参数 | S1 | 控制 softmax/sigmoid 中相似度被放大的程度；本文记 $t=\exp(t')$ |
| logit | S2 | sigmoid 输入前的原始分数；SigLIP 中为 $z_{ij}(t\,x_i\cdot y_j-b)$ |
| 嵌入空间 | S1 | 图像与文本嵌入所在的向量空间；CLIP/SigLIP 的共享空间 |
| L2 归一化 | S2 | 把向量除以其 L2 范数；$x_i=f(I_i)/\|f(I_i)\|_2$ |
| 分母耦合 | S1 | CLIP softmax 中每个对的损失通过分母 $\sum_j$ 依赖整个 batch；本文用语 |
| 逐对独立 | S2 | SigLIP 中每个对的损失与梯度不依赖其他对；本文用语 |
| 解耦 / batch size 与损失解耦 | S1 | 论文 §3.2 主张；损失定义不再依赖 batch 内 hard negative 池 |
| 数值稳定 / log-sum-exp trick | S1 折叠块 | 用 $\max$ 减法稳定 softmax 计算；CLIP 已用，本页不作为 SigLIP 主要动机 |
| weight decay | 不展开（论文 §4.5 提及） | 训练正则项；MoonViT-V2 训练细节，不在本页范围 |
| zero-shot / 0-shot | S4 | 不微调直接评估；同 ImageNet zero-shot |
| 摘要 / abstract | 文末来源 | 论文摘要 |
| ICCV | 文末来源 | 论文发表会议（2023 International Conference on Computer Vision） |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| SigLIP | Sigmoid Loss for Language-Image Pre-training | 页面开头 |
| SigLiT | Sigmoid + Locked-image Tuning | S4 |
| mSigLIP | multilingual SigLIP | S4 |
| LiT | Locked-image Tuning | S4 |
| CLIP | Contrastive Language-Image Pre-training | S1 |
| ViT | Vision Transformer | S1 |
| BS | Batch Size | S4 实验表 |
| TPU | Tensor Processing Unit | S4 实验表 |
| INet-0 / INet-0-shot | ImageNet zero-shot top-1 accuracy | S4 实验表 |
| XM3600 | CrossModal-3600 multilingual benchmark | S4 多语言表 |
| K3 | Kimi K3 | S4 |

## 符号

| 符号 | 首次出现 | 定义或含义 | 全文一致性 |
|---|---|---|---|
| $f$ | S1 | 图像编码器（图像塔）；$f(I_i)$ 输出图像嵌入 | 全文一致 |
| $g$ | S1 | 文本编码器（文本塔）；$g(T_j)$ 输出文本嵌入 | 全文一致 |
| $I_i$ | S1 | batch 内第 $i$ 张图像 | 全文一致 |
| $T_j$ | S1 | batch 内第 $j$ 段文本 | 全文一致 |
| $x_i$ | S1 | 归一化图像嵌入 $x_i=f(I_i)/\|f(I_i)\|_2$ | 与 CLIP 论文一致；F4 |
| $y_j$ | S1 | 归一化文本嵌入 $y_j=g(T_j)/\|g(T_j)\|_2$ | 与 CLIP 论文一致；F4 |
| $\|B\|$ | S1 | batch size（batch 内样本对数） | 全文一致 |
| $i, j$ | S1 | batch 内 image 与 text 的索引；$i, j\in\{1,\ldots,\|B\|\}$ | 全文一致 |
| $t$ | S1 | 温度参数；可学习；$t=\exp(t')$；F5 | 全文一致 |
| $t'$ | S3 | 温度的可学习 log 参数；初始化 $t'=\log 10$ | F5、F6 |
| $b$ | S2 | SigLIP 损失的可学习 bias；初始化 $b=-10$；F6 | 仅 SigLIP 用；CLIP 无此项 |
| $z_{ij}$ | S2 | (image $i$, text $j$) 的配对标签；配对 $z_{ij}=+1$，不配对 $z_{ij}=-1$ | 全文一致 |
| $\sigma$ | S2 | sigmoid 函数 $\sigma(a)=1/(1+e^{-a})$；F3 | 全文一致 |
| $\mathcal{L}_{\text{CLIP}}$ | S1 | CLIP softmax 损失；F2 | 全文一致 |
| $\mathcal{L}_{\text{SigLIP}}$ | S2 | SigLIP sigmoid 损失；F1 | 全文一致 |
| $\mathcal{L}_{ij}$ | S2 折叠块 | 单个对的损失项 $\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$ | 仅数字例子折叠块 |
| $\exp$ | S3 | 指数函数；用于 $t=\exp(t')$ | 全文一致 |
| $\log$ | S1 | 自然对数 | 全文一致 |
| $\sum_j$ | S1 | 对 batch 内所有 $j$ 求和；CLIP softmax 分母 | 全文一致 |
| $\|\cdot\|_2$ | S2 | L2 范数；用于嵌入归一化 | 全文一致 |
| $\cdot$ | S1 | 向量内积；$x_i\cdot y_j$ | 全文一致 |
| $|B|^2$ | S2 | batch 内所有 (image, text) 对数 | 全文一致 |

## 术语漂移检查

- "sigmoid 损失" 与 "SigLIP 损失" 在本文中可互换使用，均指 F1 公式；首次出现时明确："SigLIP 损失（sigmoid 损失）"。
- "softmax 损失" 与 "CLIP 损失" 在本文中可互换使用，均指 F2 公式；首次出现时明确："CLIP softmax 损失"。
- "bias $b$" 一律带符号 $b$；不写 "偏置项" 或 "偏差"。
- "温度 $t$" 一律带符号 $t$；$t'$ 是其 log 参数；不混用 $\tau$ 或 $T$。
- "正对 / 负对" 不写 "正样本 / 负样本"——后者在对比学习中容易与"样本本身"混淆。
- "图像塔 / 文本塔" 与 "图像编码器 / 文本编码器" 可互换；首次出现时明确。
- "batch size" 写作 $\|B\|$；不写 $N$（避免与 ViT 概念页的 patch 数 $N$ 混淆）。
- "可学习" 与 "learnable" 可互换；首次出现时用 "可学习"。
- "from scratch" 与 "从零训练" 可互换；首次出现时用 "from scratch"。
