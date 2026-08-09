# SigLIP 核心论断与证据

来源：Zhai, Mustafa, Kolesnikov, Beyer. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023 (Oral). arXiv:2303.15343v4 (27 Sep 2023). https://arxiv.org/abs/2303.15343

## 核心论断（C）

### C1 SigLIP 用 sigmoid 逐对二分类损失替代 CLIP 的 softmax 对比损失

- **论断内容**：SigLIP 损失对 batch 内所有 $|B|\times|B|$ 对 (image, text) 独立计算二分类交叉熵，正对 $z_{ij}=+1$、负对 $z_{ij}=-1$；不需要 batch 内全局归一化。
- **来源定位**：Zhai et al. 2023 §3.2 第 1–2 段，公式（紧接 §3.1 之后）。
- **适用条件**：图像塔与文本塔均为可微编码器，输出可 L2 归一化的嵌入。
- **置信状态**：已确认（论文 §3.2 直接给出公式与符号定义）。

### C2 CLIP softmax 损失分母需对整个 batch 求和

- **论断内容**：CLIP 损失 $-\frac{1}{2|B|}\sum_i(\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_i\cdot y_j}}+\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_j\cdot y_i}})$ 中每个对的损失通过 softmax 分母 $\sum_j e^{t x_i\cdot y_j}$ 与整个 batch 耦合；对称地沿 image→text 与 text→image 两个方向各做一次 softmax。
- **来源定位**：Zhai et al. 2023 §3.1 第 2 段与公式（CLIP/InfoNCE 形式）。
- **适用条件**：标准 CLIP 训练目标，温度参数 $t$ 参数化为 $\exp(t')$。
- **置信状态**：已确认。

### C3 SigLIP 每个对的梯度独立于 batch 内其他对

- **论断内容**：由于 sigmoid 损失 $\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$ 中每个对单独构成一项，对某对 $(i,j)$ 的梯度只依赖该对的 $x_i, y_j, z_{ij}, t, b$，不依赖 batch 内其他对（CLIP softmax 中分母 $\sum_j$ 项导致每个对的梯度依赖所有 $y_j$）。
- **来源定位**：Zhai et al. 2023 §3.2 第 1 段（"operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization"），§3.3 chunked 实现（图 1 阐明可分块计算）。
- **适用条件**：标准反向传播；嵌入已被 L2 归一化。
- **置信状态**：已确认（由公式形式直接得出）。

### C4 可学习 bias $b$ 抵消初始正负样本不平衡；初始化 $b=-10$, $t'=\log 10$

- **论断内容**：batch 内正对仅 $|B|$ 个、负对 $|B|^2-|B|$ 个；正负比例 $1:(|B|-1)$。在 $|B|=32\text{k}$ 时比例为 $1:32767$。若没有 $b$，初始 logits 偏向"大多数负对"使训练初期产生大量错误方向梯度。论文 §3.2 明确引入可学习 $b$ 并初始化 $b=-10$，使初始 $\sigma(-b)=\sigma(10)\approx 0.99995$（等价于"初始时假设任何对都偏向正类"），抵消该不平衡。$t'$ 初始化为 $\log 10$，即 $t=10$。
- **来源定位**：Zhai et al. 2023 §3.2 第 3 段（"At initialization, the heavy imbalance coming from the many negatives dominates the loss... we introduce an additional learnable bias term $b$... We initialize $t'$ and $b$ to $\log 10$ and $-10$ respectively."）。
- **适用条件**：标准 SigLIP 训练初始化；arXiv v3 专门澄清了 $t$ 与 $t'$ 的初始化。
- **置信状态**：已确认（论文原文直接给出初始化值与动机）。

### C5 SigLIP 把 batch size 与损失函数解耦——小 batch 下显著优于 softmax

- **论断内容**：CLIP softmax 损失依赖 batch 内 hard negative 池（"对最难负样本的相对优势"）；batch 小时 hardest negative 往往太弱、梯度噪声大。SigLIP 每对的梯度由 $z_{ij}(t\,x_i\cdot y_j-b)$ 单独决定，与 batch 内其他对无关，因此小 batch 下仍能产生固定幅度的梯度。论文 §4.2 报告：sigmoid 损失在 batch < 16k 时显著优于 softmax；batch 增大时差距缩小。
- **来源定位**：Zhai et al. 2023 §3.2 第 1 段（"conceptually decouples the batch size from the definition of the task"）、§4.2 第 1 段（"the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16 k. As the train batch size grows, the gap closes"）。
- **适用条件**：相同模型架构与训练数据下对比损失函数。
- **置信状态**：已确认。

### C6 SigLiT 4 TPUv4 / 2 天训出 ImageNet zero-shot 84.5%

- **论断内容**：用 ViT-g/14 作冻结图像塔 + Large 文本塔（12 层），batch 20k，训 107k 步 / 2 天 / 4 TPUv4，达到 ImageNet zero-shot 84.5%。
- **来源定位**：Zhai et al. 2023 Table 1（SigLiT g/14 L 行）+ §4.4 第 2 段（"with a ViT-g/14 model as the vision tower and a Large text tower, we can train at 20 k batch size on four chips for 107 k steps in under two days. This further pushes the 0-shot ImageNet classification accuracy up to 84.5%"）。
- **适用条件**：冻结预训练 ViT-g/14 + 仅训文本塔；用 LION 优化器 + decoupled weight decay $10^{-7}$ + 6.5k 步 warm-up 至 peak $10^{-4}$ + cosine decay。
- **置信状态**：已确认。

### C7 SigLIP B/16 from scratch 32 TPUv4 / 2 天训出 ImageNet zero-shot 72.1%；同等水平 CLIP 用约 2500 TPUv3-days

- **论断内容**：从零训练 ViT-B/16 + Base 文本塔，batch 32k，32 TPUv4，2 天 → ImageNet zero-shot 72.1%（5 天 → 73.4%）。论文引用 [30] 报告 CLIP 同等水平 72.6% 用约 2500 TPUv3-days。
- **来源定位**：Zhai et al. 2023 Table 1（SigLIP B/16 B 行，BS=32k, 32 TPUv4, 2 days, 72.1；5 days, 73.4）+ §4.5 末段（"SigLIP achieves 72.1% 0-shot accuracy. This presents a significant training cost reduction e.g. compared to CLIP (approx. 2500 TPUv3-days for 72.6%) reported in [30]"）。
- **适用条件**：from scratch（不加载预训练视觉权重）；WebLI 数据集；32k batch。
- **置信状态**：已确认。

### C8 batch size 32k 接近最优；超过 32k 后 ImageNet zero-shot 几乎不再提升，多语言 retrieval 反而下降；1M 边际收益快速消失

- **论断内容**：论文 §4.2 + §4.3 + 摘要给出三组证据：(a) SigLIP 在 batch=32k 达最佳（73.2% INet-0），softmax 在 batch=98k 才达最佳但仍不如 sigmoid；(b) 多语言 mSigLIP Table 2 显示 batch 从 32k → 240k，ImageNet zero-shot 几乎不变（73.2 → 73.1），但 XM3600 多语言 retrieval 平均从 34.9 降到 32.7；(c) 摘要："we push the batch size to the extreme, up to one million, and find that the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient"。
- **来源定位**：Zhai et al. 2023 摘要 + §4.2 第 1–2 段 + §4.3 第 1 段 + Table 2。
- **适用条件**：SigLIP / mSigLIP / softmax 三组对照；30B examples seen；WebLI 数据集。
- **置信状态**：已确认。

## 核心公式（F）

### F1 SigLIP 损失

$$\mathcal{L}_{\text{SigLIP}}=-\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\frac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j+b)}}=-\frac{1}{|B|}\sum_{i,j}\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$$

- **来源**：Zhai et al. 2023 §3.2 公式。论文原文写作 $\log\frac{1}{1+e^{z_{ij}(-t x_i\cdot y_j+b)}}$；由 $\sigma(a)=1/(1+e^{-a})$ 等价改写为 $\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$。
- **符号**：$x_i=f(I_i)/\|f(I_i)\|_2$ 归一化图像嵌入；$y_j=g(T_j)/\|g(T_j)\|_2$ 归一化文本嵌入；$z_{ij}\in\{+1,-1\}$；$t=\exp(t')$ 可学习温度；$b$ 可学习 bias。

### F2 CLIP softmax 损失（对照）

$$\mathcal{L}_{\text{CLIP}}=-\frac{1}{2|B|}\sum_{i=1}^{|B|}\left(\log\frac{e^{t x_i\cdot y_i}}{\sum_{j=1}^{|B|}e^{t x_i\cdot y_j}}+\log\frac{e^{t x_i\cdot y_i}}{\sum_{j=1}^{|B|}e^{t x_j\cdot y_i}}\right)$$

- **来源**：Zhai et al. 2023 §3.1 公式（论文直接以 CLIP/InfoNCE 形式给出，含对称的 image→text 与 text→image 两项）。
- **符号**：与 F1 共享 $x_i, y_j, t$；CLIP 中无 $b$。

### F3 Sigmoid 函数

$$\sigma(a)=\frac{1}{1+e^{-a}}$$

- **来源**：标准定义，论文 §3.2 隐式使用。

### F4 嵌入归一化

$$x_i=\frac{f(I_i)}{\|f(I_i)\|_2},\qquad y_j=\frac{g(T_j)}{\|g(T_j)\|_2}$$

- **来源**：Zhai et al. 2023 §3.1 末段（与 CLIP 共享）。

### F5 温度参数化

$$t=\exp(t'),\qquad t'\text{ 全局可学习参数}$$

- **来源**：Zhai et al. 2023 §3.1 末段。

### F6 初始化值

$$t'=\log 10\ (\text{即}\ t=10),\qquad b=-10$$

- **来源**：Zhai et al. 2023 §3.2 第 3 段。

## 外部数字与实验条件（N）

### N1 SigLiT 关键数字（Table 1 + §4.4）

| 配置 | 图像塔 | 文本塔 | BS | #TPUv4 | Days | INet-0 |
|---|---|---|---|---|---|---|
| SigLiT B/8 L | ViT-AugReg-B/8（frozen） | L（12 层） | 32k | 4 | 1 | 79.8% |
| SigLiT g/14 L | ViT-g/14（frozen） | L | 20k | 4 | 2 | **84.5%** |

- **来源**：Zhai et al. 2023 Table 1 + §4.4 第 2 段。
- **实验条件**：LION 优化器 + decoupled weight decay $10^{-7}$；6.5k 步 warm-up 至 $10^{-4}$ + cosine decay；65k 步 / 107k 步；WebLI 数据集。
- **置信状态**：已确认。

### N2 SigLIP from scratch 关键数字（Table 1 + §4.5）

| 配置 | BS | #TPUv4 | Days | INet-0 |
|---|---|---|---|---|
| SigLIP B/16 B | 16k | 16 | 3 | 71.0% |
| SigLIP B/16 B | 32k | 32 | 2 | **72.1%** |
| SigLIP B/16 B | 32k | 32 | 5 | 73.4% |

- **对照**：CLIP 72.6% 用约 2500 TPUv3-days（来自 [30]）。
- **来源**：Zhai et al. 2023 Table 1 + §4.5 末段。
- **实验条件**：from scratch；WebLI 数据集；温度与 bias 按 F6 初始化。
- **置信状态**：已确认。

### N3 多语言 mSigLIP batch size 扫描（Table 2）

| BS | INet-0 | XM avg | XM de | XM en | XM zh |
|---|---|---|---|---|---|
| 16k | 71.6 | 34.8 | 54.7 | 46.5 | 30.7 |
| 32k | **73.2** | **34.9** | 54.8 | 46.2 | 32.5 |
| 64k | 73.2 | 34.4 | 55.4 | 46.5 | 32.0 |
| 128k | 73.2 | 33.6 | 54.3 | 46.6 | 30.6 |
| 240k | 73.1 | 32.7 | 54.7 | 46.6 | 23.7 |

- **来源**：Zhai et al. 2023 Table 2（§4.3）。
- **实验条件**：Base 模型；30B examples seen；WebLI 100 语言。
- **置信状态**：已确认。

### N4 极限 batch size 1M

- **论断**：摘要 + §4.2 报告将 batch size 推至 1M，发现"the benefits of growing batch size quickly diminish"。
- **来源**：Zhai et al. 2023 摘要 + §4.2。
- **实验条件**：论文未给出 1M 的具体精度数字，只给出"快速递减"的结论。
- **置信状态**：已确认（结论性），具体数字论文未报告。

## 来源选择优先级

- 主要来源：原始论文 arXiv:2303.15343v4（ICCV 2023 Oral）。
- 辅助来源（仅用于辅助定位，不作为核心论断唯一依据）：
  - WebSearch 综述（emergentmind, abhik.ai, zeroentropy）——用于交叉验证公式形式；
  - mixpeek 2025 综述——用于"softmax 在 false-negative 场景下可能更优"的边界（标注为综述观点）。
- 不使用：博客 csdn、fesianxu（中文综述）作为核心论断唯一依据；只用于辅助理解。

## 置信状态总览

所有 C1–C8 与 F1–F6 均为 **已确认** 状态，直接由论文 §3.1 §3.2 §4.2 §4.3 §4.4 §4.5 + Table 1 + Table 2 + 摘要支持。无存在冲突或证据不足的核心论断。

## 进入生产阶段的检查

- 全部核心论断已编号并完成来源定位与置信状态。
- 公式 F1–F6 来源明确，与论文一致。
- 外部数字 N1–N4 来源明确，实验条件记录。
- 可进入 outline.md 的教学大纲设计。
