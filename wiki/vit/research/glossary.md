# Vision Transformer（ViT）：术语表

登记全文首次出现的术语、缩写和符号。后续阶段写作和审查以此为准。

## 术语

| 术语 | 首现位置 | 定义或含义 |
|---|---|---|
| Vision Transformer（ViT） | 标题 | Dosovitskiy 2021 提出的图像分类架构，把图像切成 patch 序列后送入标准 Transformer 编码 |
| CNN（Convolutional Neural Network） | S1 | 用局部卷积核扫描图像的网络家族；本文作对照 |
| Locality（局部性） | S1 | CNN 归纳偏置之一：卷积核只看局部邻域像素，假设"相邻像素关系强于远距离像素" |
| Translation Invariance（平移不变性） | S1 | CNN 归纳偏置之二：卷积核在图像上共享权重，假设"目标平移后特征不变" |
| Inductive Bias（归纳偏置） | S1 | 模型架构对数据做的先验假设；CNN 有空间归纳偏置，标准 Transformer 无 |
| 归纳偏置 vs 大数据训练 | S1 | Dosovitskiy 2021 §4.4 的核心论断：大数据训练可以胜过归纳偏置 |
| Patch（图像块） | S2 | 把图像切成的不重叠 $P\times P$ 方块；是图像预处理产物 |
| Token（令牌） | S2 | Transformer 输入序列的单位；ViT 中是 patch 投影后的 $D$ 维向量 |
| Tokenization（令牌化） | S2 | 把图像 patch 经线性投影变成 Transformer 可处理 token 的过程 |
| Patch Embedding | S2 | patch 展平后经线性投影 $E$ 得到的 $D$ 维向量 $x_p^i E$ |
| Class Token（[CLS]） | S2 | prepend 在 patch 序列最前的可学习 $D$ 维向量，承载全局信息用于分类；借鉴 BERT |
| BERT [CLS] token | S2 折叠块 | NLP 中用 [CLS] token 做句子分类的设计；ViT 借鉴此设计 |
| 位置编码（Positional Encoding） | S2 | 加在 patch 嵌入上的可学习向量，弥补注意力位置无关性 |
| 可学习 1D 绝对位置编码 | S2 | ViT 选择的位置编码形式；与 sin/cos 与相对位置编码并列（已在 positional-encoding 页给出家族对比） |
| 线性投影（Linear Projection） | S2 | 用矩阵 $E\in\mathbb{R}^{P^2 C\times D}$ 把 patch 展平向量投影到 $D$ 维 |
| 序列长度（Sequence Length） | S2 | Transformer 输入 token 数；ViT 中是 $N+1$（含 class token） |
| 矩阵乘法（Matrix Multiplication） | S2 | $AB$ 的维度规则；本页用占位提示 |
| Transformer 编码器（Encoder） | S3 | Vaswani 2017 提出的标准结构；ViT 只用 encoder 不用 decoder |
| 编码块（Encoder Block） | S3 | MSA 子层 + MLP 子层 + 残差 + LayerNorm 的组合，重复 $L$ 次 |
| 多头自注意力（Multi-Head Self-Attention, MSA） | S3 | 标准缩放点积注意力的多头版本；已有概念页 standard-attention |
| 前馈网络（Multi-Layer Perceptron, MLP） | S3 | 两层 + GELU 的非线性变换；ViT 编码块的第二个子层 |
| LayerNorm（层归一化，LN） | S3 | 把每行特征归一化到均值 0、方差 1；ViT 用 pre-LN |
| Pre-LN | S3 | LN 在子层前应用；ViT 选择；与 Vaswani 2017 的 post-LN 对照 |
| Post-LN | S3 折叠块 | LN 在子层后应用；Vaswani 2017 原始 Transformer 的选择 |
| 残差连接（Residual Connection） | S3 | $y=F(x)+x$；已有概念页 residual-connection |
| GELU（Gaussian Error Linear Unit） | S3 | 平滑的 ReLU 变体；MLP 内部的非线性激活 |
| 分类头（Classification Head） | S3 | 接在 $z_L^0$ 后的网络；预训练时是 MLP，微调时是单线性层 |
| 图像表示（Image Representation, $y$） | S3 | $z_L^0$ 经 LN 后的 $D$ 维向量，用作分类头输入 |
| 配置（Configuration） | S3 | 模型层数 $L$、隐藏维 $D$、头数 $h$、MLP size、参数量的一组取值 |
| ViT-Base / Large / Huge | S3 | 论文 Table 1 的三档配置（12/24/32 层、86M/307M/632M 参数） |
| 命名约定（ViT-L/16 等） | S3 | "档位/patch size" 命名，如 ViT-L/16 = Large + patch 16 |
| ImageNet / ImageNet-1k | S4 | 1.3M 图、1k 类的图像分类数据集；标准微调基准 |
| ImageNet-21k | S4 | 14M 图、21k 类的 ImageNet 超集；中等规模预训练数据 |
| JFT-300M | S4 | Google 私有数据集，300M 图、18k 类；论文大数据预训练来源 |
| TPUv3-core-days | S4 | 预训练计算量度量单位；1 TPUv3-core-day = 1 个 TPUv3 核训练 1 天 |
| BiT-L（ResNet152x4） | S4 | CNN 对照模型；用 ResNet-152 加宽 4 倍 + JFT-300M 预训练 |
| Noisy Student（EfficientNet-L2） | S4 | CNN + 自训练对照模型；ImageNet + JFT 训练 |
| GAP（Global Average Pooling） | S2 折叠块 | 用所有 patch token 的平均做分类头输入；与 class token 对照的替代方案 |
| DeiT | S4 变体列表 | ViT 变体；蒸馏 + 数据增强解决小数据训练问题 |
| Swin Transformer | S4 变体列表 | ViT 变体；层级窗口注意力引入 CNN 风格的归纳偏置 |
| BEiT / MAE | S4 变体列表 | ViT 变体；掩码图像建模自监督预训练 |
| DINOv2 | S4 变体列表 | ViT 变体；自监督训练 + 大数据 |
| MoonViT-V2 | S4 变体列表 | K3 的 ViT 变体；27 层、RMSNorm、去 bias、从零训练、视频分解注意力；已有概念页 |

## 符号

| 符号 | 首现位置 | 含义 |
|---|---|---|
| $x$ | S2 公式 F1 | 输入图像，$\mathbb{R}^{H\times W\times C}$ |
| $H$ | S2 | 图像高度（像素） |
| $W$ | S2 | 图像宽度（像素） |
| $C$ | S2 | 图像通道数（RGB=3） |
| $P$ | S2 | Patch 大小（像素），论文主线 $P=16$ |
| $N$ | S2 公式 F1 | Patch 数 = 序列长度，$N=HW/P^2$ |
| $x_p^i$ | S2 公式 F1 | 第 $i$ 个 patch 展平后的 $P^2 C$ 维向量 |
| $E$ | S2 公式 F1 | Patch 线性投影矩阵，$\mathbb{R}^{P^2 C\times D}$ |
| $D$ | S2 公式 F1 | 模型隐藏维度（ViT-B $D=768$、ViT-L $D=1024$、ViT-H $D=1280$） |
| $x_{\text{class}}$ | S2 公式 F1 | 可学习 class token，$\mathbb{R}^{D}$ |
| $E_{\text{pos}}$ | S2 公式 F1 | 可学习 1D 位置编码，$\mathbb{R}^{(N+1)\times D}$ |
| $z_0$ | S2 公式 F1 | Transformer 输入，$\mathbb{R}^{(N+1)\times D}$ |
| $[;\,]$ | S2 公式 F1 | 沿序列维拼接 |
| $L$ | S3 公式 F2 | Transformer 编码块层数 |
| $\ell$ | S3 公式 F2 | 层索引，$\ell=1\ldots L$ |
| $z_\ell$ | S3 公式 F3 | 第 $\ell$ 层输出，$\mathbb{R}^{(N+1)\times D}$ |
| $z_\ell'$ | S3 公式 F2 | 第 $\ell$ 层 MSA 子层输出（MLP 子层输入） |
| $\text{LN}$ | S3 公式 F2 | LayerNorm |
| $\text{MSA}$ | S3 公式 F2 | 多头自注意力（内部公式见 [标准注意力](../../wiki/standard-attention/index.html)） |
| $\text{MLP}$ | S3 公式 F3 | 两层 + GELU 的前馈网络 |
| $h$ | S3 | 多头注意力的头数（ViT-B $h=12$、ViT-L/H $h=16$） |
| $d_k$ | S3 | 每头维度，$d_k=D/h$（ViT-B $d_k=64$） |
| $y$ | S3 公式 F4 | 图像表示，$y=\text{LN}(z_L^0)\in\mathbb{R}^{D}$ |
| $z_L^0$ | S3 公式 F4 | 第 $L$ 层输出 $z_L$ 的第 0 行（class token 对应位置） |

## 缩写

| 缩写 | 全称 | 含义 |
|---|---|---|
| ViT | Vision Transformer | 本文概念，Dosovitskiy 2021 提出 |
| CNN | Convolutional Neural Network | 卷积神经网络，本文对照 |
| MSA | Multi-Head Self-Attention | 多头自注意力，已有概念页 |
| MLP | Multi-Layer Perceptron | 前馈网络，ViT 编码块的第二个子层 |
| LN | Layer Normalization | 层归一化 |
| GAP | Global Average Pooling | 全局平均池化，class token 的替代方案 |
| BERT | Bidirectional Encoder Representations from Transformers | NLP 模型，ViT 借鉴其 [CLS] token |
| BiT-L | Big Transfer (Large) | CNN 对照模型，ResNet152x4 + JFT-300M |
| GELU | Gaussian Error Linear Unit | 平滑的 ReLU 变体 |
| TPU | Tensor Processing Unit | Google 计算单元；TPUv3-core-days 是计算量度量 |
| RGB | Red Green Blue | 三通道彩色图像 |
| NLP | Natural Language Processing | 自然语言处理 |

全文符号含义保持一致：$x$ 始终指输入图像，$P$ 始终指 patch 大小，$D$ 始终指隐藏维度，$L$ 始终指层数，$h$ 始终指头数（与 [标准注意力](../../wiki/standard-attention/index.html) 概念页一致）。224×224 patch 16 手算例子复用同一组符号。
