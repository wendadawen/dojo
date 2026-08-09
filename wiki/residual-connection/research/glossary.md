# 残差连接：术语表

登记全文首次出现的术语、缩写和符号。后续阶段写作和审查以此为准。

## 术语

| 术语 | 首现位置 | 定义或含义 |
|---|---|---|
| 残差连接（Residual Connection） | 标题 | 把一层或若干层的输入直接加到这些层的输出上的机制；同义：Skip Connection、Shortcut Connection |
| 跳跃连接（Skip Connection） | S1 首段 | 残差连接的同义；强调"绕过中间层" |
| 捷径连接（Shortcut Connection） | S2 | 残差连接中那条跨层直连的路径；本页中特指恒等加法 $+x$ 或线性投影 $W_s x$ |
| 恒等捷径（Identity Shortcut） | S2 | 捷径上不做任何变换，直接把 $x$ 加到输出；无参数 |
| 投影捷径（Projection Shortcut） | S4 | 捷径上做线性投影 $W_s x$ 用于维度匹配；有参数 |
| 残差函数 $F$ | S2 | 残差块内若干非线性层构成的函数；待学习 |
| 底层映射 $H$（desired underlying mapping） | S2 | 残差块整体希望拟合的目标函数；$H(x)=F(x)+x$ |
| 退化问题（Degradation Problem） | S1 | plain 网络加深后训练误差先饱和后上升的现象；非过拟合、非梯度消失 |
| 普通网络（Plain Network） | S1 | 无残差连接的层堆叠网络 |
| 梯度消失（Vanishing Gradient） | S1 | 反向传播中梯度因连乘小数而趋零；BN 与良好初始化已基本解决 |
| 集成解释（Ensemble Interpretation） | S3 | Veit et al. 2016 提出的观察性解释：ResNet 可展开为 $2^n$ 条路径的集合 |
| 路径（Path） | S3 | 集成解释中，从输入到输出的一条具体走法；每个残差块处可"进入 $F$"或"跳过" |
| 残差块（Residual Block） | S2 | 由 $F$ 与一条捷径构成的最小单元；输出 $y=F(x)+x$ |
| 恒等映射（Identity Mapping） | S2 | $H(x)=x$；"什么都不做" |
| 批归一化（Batch Normalization, BN） | S1 | 独立的数值稳定技术；本页只作为 ResNet 实验条件提及，不展开机制 |
| 层归一化（Layer Normalization, LN） | S4 | Transformer 中使用的归一化；本页只引用公式形式，不展开机制 |
| Post-LN | S4 | 原版 Transformer 的归一化位置：$\text{LayerNorm}(x+\text{Sublayer}(x))$ |
| Pre-LN | S4 | 现代 LLM 的归一化位置：$x+\text{Sublayer}(\text{LayerNorm}(x))$ |
| 子层（Sublayer） | S4 | Transformer 中一个注意力或前馈模块；本页作为黑箱变换使用 |
| 1×1 卷积 | S4 | 投影捷径 $W_s$ 的工程实现；本页只提实现方式，不展开卷积机制 |
| ResNet | S1 | He et al. 2016 提出的残差网络；残差连接的提出载体 |
| Transformer | S4 | Vaswani et al. 2017 提出的架构；每个子层外都包残差 |
| ILSVRC 2015 / ImageNet | S1 | ResNet 评估的数据集与竞赛 |
| CIFAR-10 | S1 | ResNet 退化现象与超深网络实验的数据集 |

## 符号

| 符号 | 首现位置 | 含义 |
|---|---|---|
| $x$ | S2 公式 F1 | 残差块的输入向量（或张量） |
| $y$ | S2 公式 F1 | 残差块的输出向量（或张量） |
| $F(x,\{W_i\})$ 或 $F(x)$ | S2 公式 F1 | 残差块内若干非线性层构成的函数；$\{W_i\}$ 为其参数集合 |
| $\{W_i\}$ | S2 公式 F1 | $F$ 内部各层的待学习参数集合 |
| $H(x)$ | S2 | 残差块整体希望拟合的底层映射；$H(x)=F(x)+x$ |
| $h_l$ | S3 公式 F4 | 第 $l$ 个残差块的输出（也是第 $l+1$ 个块的输入） |
| $h_L$ | S3 公式 F4 | 第 $L$ 个（最后一个）残差块的输出 |
| $L$ | S3 公式 F4 | 残差块总数 |
| $l$ | S3 公式 F4 | 残差块索引 |
| $F_i(h_i)$ | S3 公式 F4 | 第 $i$ 个残差块内的残差函数 |
| $W_s$ | S4 公式 F2 | 投影捷径的线性投影矩阵；只在维度不匹配时使用 |
| $\partial y/\partial x$ | S3 公式 F3 | $y$ 对 $x$ 的雅可比；本页用作标量导数以简化讲解 |
| $\partial F/\partial x$ | S3 公式 F3 | $F$ 对 $x$ 的导数（雅可比） |
| $\partial h_L/\partial h_l$ | S3 公式 F4 | 从第 $l$ 块到第 $L$ 块的梯度 |
| $n$ | S3 集成解释 | 残差块数量；展开后路径数为 $2^n$ |
| $\sigma$ | 不使用 | 本页不引入激活函数符号；1 神经元例子无激活 |
| $\text{Sublayer}(x)$ | S4 | Transformer 子层对 $x$ 的变换；作为黑箱 |

## 缩写

| 缩写 | 全称 | 含义 |
|---|---|---|
| BN | Batch Normalization | 批归一化 |
| LN | Layer Normalization | 层归一化 |
| plain | Plain Network | 无残差连接的普通网络 |
| ResNet | Residual Network | 残差网络 |

全文符号含义保持一致：$F$ 始终指残差函数，$x$ 始终指残差块输入，$H$ 始终指底层映射。1 神经元手算例子复用同一组符号（$x$ 为输入、$w$ 为 $F$ 内的单一参数）。
