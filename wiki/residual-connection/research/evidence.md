# 残差连接：核心论断与证据

来源优先级：原始论文 > 权威教材/同行评审综述 > 官方文档 > 固定版本源码。本页核心论断全部来自两篇原始论文与 Transformer 原论文。

## C 论断（核心机制）

### C1 退化问题

- **论断内容**：在 plain 网络（无捷径连接）中，随着深度增加，训练误差先饱和后上升；该现象不是过拟合（训练误差同时升高），也不是梯度消失（已用 BN 和良好初始化解决）。
- **来源定位**：He et al. 2016, "Deep Residual Learning for Image Recognition", CVPR 2016, arXiv:1512.03385, §1 Abstract 与 §1 第 2 段，Figure 1（CIFAR-10 上 56 层 plain 训练误差高于 20 层 plain）。
- **适用条件**：使用 BN 与良好初始化的 plain 网络在 ImageNet / CIFAR-10 上的实测现象。
- **置信状态**：已确认。

### C2 恒等映射重构

- **论断内容**：把待拟合的底层映射记为 $H(x)$，不让层堆叠直接拟合 $H(x)$，而让它拟合残差 $F(x):=H(x)-x$，则 $H(x)=F(x)+x$；当最优映射接近恒等时，把 $F$ 推向 0 比让非线性层拟合 $H(x)=x$ 更容易。
- **来源定位**：He et al. 2016, §3.1 第 1–3 段。
- **适用条件**：捷径为恒等加法或仅线性投影。
- **置信状态**：已确认。

### C3 梯度直通项

- **论断内容**：由 $y=F(x)+x$ 得 $\partial y/\partial x = \partial F/\partial x + 1$；梯度中存在不经过 $F$ 的常数项 1，使深层叠加时梯度乘积中始终有一条全 1 的直通路径。
- **来源定位**：He et al. 2016, §3.1（公式隐含）；He et al. 2016, "Identity Mappings in Deep Residual Networks", ECCV 2016, arXiv:1603.05027, §2 显式分析。
- **适用条件**：捷径为恒等映射（$+x$），且后续激活/归一化不破坏恒等性（pre-activation 形式）。
- **置信状态**：已确认。

### C4 集成解释

- **论断内容**：含 $n$ 个残差块的 ResNet 可展开为 $2^n$ 条从输入到输出的路径集合；路径长度不同；lesion study 表明删除单个模块不会使网络崩溃（与 VGG 删层即崩溃形成对照）；在 110 层 ResNet 中，多数梯度来自长度 10–34 的路径，长路径几乎不贡献梯度。
- **来源定位**：Veit et al. 2016, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", NIPS 2016, §2–§4。
- **适用条件**：恒等捷径；适用于解释训练后的行为，不用于设计目标。
- **置信状态**：已确认。该论断是观察性解释，不是 ResNet 的原始设计依据；本文按此定位使用。

### C5 Transformer 子层残差

- **论断内容**：Transformer 每个子层（注意力、前馈）外都包一层残差连接与归一化；原论文使用 Post-LN：$\text{output}=\text{LayerNorm}(x+\text{Sublayer}(x))$。
- **来源定位**：Vaswani et al. 2017, "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762, §3.2.1 与 Figure 1。
- **适用条件**：原版 Transformer；后续变体（GPT-2 之后）改用 Pre-LN，不影响残差连接本身的存在。
- **置信状态**：已确认。

## F 公式

### F1 残差块（恒等捷径）

- **公式**：$y = F(x,\{W_i\}) + x$
- **来源定位**：He et al. 2016, §3.1, Eq.(1)。
- **适用条件**：$F$ 与 $x$ 维度相同；加法为逐元素。
- **置信状态**：已确认。

### F2 残差块（投影捷径）

- **公式**：$y = F(x,\{W_i\}) + W_s x$
- **来源定位**：He et al. 2016, §3.1, Eq.(2)。
- **适用条件**：$F$ 与 $x$ 维度不同（通道数或空间尺寸变化）；$W_s$ 为线性投影（实验中用 1×1 卷积实现）。
- **置信状态**：已确认。论文同时指出：维度相同时用投影不会更好，故只在维度不同时使用。

### F3 梯度结构

- **公式**：$\dfrac{\partial y}{\partial x} = 1 + \dfrac{\partial F}{\partial x}$
- **来源定位**：由 F1 直接对 $x$ 求导；He et al. 2016 (Identity Mappings) §2 显式给出。
- **适用条件**：恒等捷径；$F$ 可微。
- **置信状态**：已确认。

### F4 多层梯度乘积

- **公式**：$\dfrac{\partial h_L}{\partial h_l} = \prod_{i=l}^{L-1}\left(1+\dfrac{\partial F_i}{\partial h_i}\right)$
- **来源定位**：对 $h_{i+1}=h_i+F_i(h_i)$ 逐层应用链式法则；He et al. 2016 (Identity Mappings) §2。
- **适用条件**：每段均为恒等捷径；展开后的乘积项中包含一项"全 1"的直通路径（每一因式都取 1）。
- **置信状态**：已确认。

## N 数字

### N1 退化现象的实证数字

- **数字**：ImageNet 上 34 层 plain 网络的验证误差高于 18 层 plain 网络（约 28.41% vs 27.94% top-1 验证误差，论文 Figure 4 训练曲线同向）；CIFAR-10 上 56 层 plain 训练误差高于 20 层 plain（Figure 1）。
- **来源定位**：He et al. 2016, §1 Figure 1 与 §4.2 Figure 4。
- **适用条件**：使用 BN 的 plain 网络对比实验。
- **置信状态**：已确认。本页正文使用该数字时只引趋势与图号，不引精确小数（避免不同复现的细微差异）。

### N2 残差网络的对比数字

- **数字**：34 层 ResNet 的验证误差低于 18 层 ResNet（约 25.03% vs 27.94% top-1 验证误差）；ResNet-152 集成在 ImageNet 测试集上 top-5 错误率 3.57%，赢得 ILSVRC 2015 冠军。
- **来源定位**：He et al. 2016, §4.1 Table 2 / §4.2 Figure 4 / Abstract。
- **适用条件**：ImageNet 2012 分类任务。
- **置信状态**：已确认。

### N3 集成解释的有效路径长度

- **数字**：在 110 层 ResNet（54 个残差块）上，贡献主要梯度的路径长度为 10–34 层。
- **来源定位**：Veit et al. 2016, §3 Figure 5 与摘要。
- **适用条件**：CIFAR-10 上训练的 110 层 ResNet。
- **置信状态**：已确认。

## 来源清单

| 编号 | 引用 | 用途 |
|---|---|---|
| [He2016] | He, K., Zhang, X., Ren, S., Sun, J. "Deep Residual Learning for Image Recognition." CVPR 2016. arXiv:1512.03385 | C1, C2, C3, F1, F2, F3, F4, N1, N2 |
| [He2016b] | He, K., Zhang, X., Ren, S., Sun, J. "Identity Mappings in Deep Residual Networks." ECCV 2016. arXiv:1603.05027 | C3, F3, F4 的显式分析 |
| [Veit2016] | Veit, A., Wilber, M. J., Belongie, S. "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." NIPS 2016 | C4, N3 |
| [Vaswani2017] | Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762 | C5 |

无冲突论断。所有核心论断置信状态均为"已确认"。
