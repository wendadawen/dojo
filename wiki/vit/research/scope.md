# Vision Transformer（ViT）：内容范围

## 1.1 概念含义

- **概念名称**：Vision Transformer（ViT）
- **英文名称**：Vision Transformer；常见缩写：ViT
- **一句话定义**：ViT 把一张图像切成固定大小的 patch，每个 patch 当作一个 token 经线性投影嵌入，再 prepend 一个可学习的 class token、加上可学习的 1D 位置编码，最后送入标准 Transformer 编码器，用 class token 的最终隐状态做分类。
- **正式定义**（与权威来源一致）：Dosovitskiy et al. 2021 §3.1 给出三条公式：

  $$z_0=[x_{\text{class}};\,x_p^1 E;\,\ldots;\,x_p^N E]+E_{\text{pos}},\quad E\in\mathbb{R}^{P^2 C\times D},\;E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$$

  $$z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1},\qquad z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell',\qquad \ell=1\ldots L$$

  $$y=\text{LN}(z_L^0)$$

  其中 $N=HW/P^2$ 是 patch 数（即序列长度），$D$ 是隐藏维度，$L$ 是 Transformer 层数。本定义与 Dosovitskiy et al. 2021（ICLR 2021, arXiv:2010.11929v2）§3.1 Eq.(1)–Eq.(4) 一致。

- **本文采用的语境**：图像分类的纯 Transformer 编码器（Dosovitskiy 2021 原始形式）。覆盖 patch 切分、线性投影、class token、可学习 1D 位置编码、pre-LN 残差编码块、class token 分类头。不展开检测、分割、视频等下游任务变体。
- **历史定位**：本文是 Dosovitskiy et al. 2021 提出的原始 ViT，是后续 DeiT、Swin、BEiT、MAE、DINOv2、MoonViT-V2 等 ViT 家族变体的基础前置概念。

### 包括什么

| 纳入项 | 纳入理由 |
|---|---|
| CNN 的两个归纳偏置（locality + translation invariance）及其在大数据下的局限 | 解释"为什么要试 ViT"，是概念存在的动机 |
| Patch 切分 + 线性投影公式 $x_p^i E$ | 概念核心机制之一，缺则无法回答"图像如何变成 token" |
| Class token 的 prepend 与作用 | 概念核心机制之一，决定分类头如何接 |
| 可学习 1D 位置编码 $E_{\text{pos}}$ 的加法 | 概念核心机制之一，弥补注意力位置无关性 |
| 序列长度 $N=HW/P^2$ 的手算 | 把抽象公式落到可复算数字 |
| Transformer 编码块公式 $z_\ell', z_\ell$（pre-LN + MSA + 残差 + MLP） | 解释如何复用标准 Transformer，回答"是什么在编码" |
| 分类头 $y=\text{LN}(z_L^0)$ 与 class token 的对应关系 | 解释为什么 prepend 一个 class token 而不是用 GAP |
| 模型配置表 ViT-B/L/H 的层数、隐藏维、头数、参数量 | 给具体可核对数字，避免抽象 |
| 数据规模边界（小数据下 ViT 不如 CNN，大数据下 ViT 超越 CNN） | 概念的适用边界，是论文核心结论 |
| 与标准 Transformer 的复用关系（前置概念页 standard-attention） | 解释"为什么 ViT 能直接套 Transformer" |

### 不包括什么

| 排除项 | 排除理由 |
|---|---|
| DeiT 的蒸馏策略、Swin 的层级窗口注意力 | 独立概念，本文只在变体列表中提一句 |
| BEiT/MAE 的掩码图像建模预训练 | 独立概念（自监督），本文只在变体列表中提一句 |
| DINOv2 的自监督训练 | 独立概念，本文只作变体列表 |
| 检测/分割下游任务（DETR、Mask2Former 用 ViT backbone） | 应用层，本文只在"作为 backbone"处一句话提及 |
| 视频时序注意力分解（MoonViT-V2 的帧内/帧间分解） | 独立概念（MoonViT-V2 是本文的变体），本文只在变体列表与前置说明处提及 |
| 训练超参数（学习率、warmup、batch size）的最优选择 | 工程细节，本文只使用论文给出的配置作示例 |
| 数据增强（mixup、cutmix、randaug）的具体策略 | 工程细节，本文只在"训练 recipe"处一句话提及 |
| 知识蒸馏与教师-学生 | 独立概念，不在论文原始范围内 |
| 多模态对齐（CLIP、SigLIP） | 独立概念，本文只说明 ViT 是其视觉 backbone 的常见选择 |

### 相邻概念

| 相邻概念 | 关键区别 | 是否纳入 |
|---|---|---|
| 标准 Transformer（Vaswani 2017） | ViT 几乎完全复用其编码块；区别在输入 token 的来源——NLP 是词向量，ViT 是 patch 投影 | 纳入，作为前置概念页链接 |
| CNN（ResNet、EfficientNet） | 用卷积归纳偏置（locality + translation invariance）；ViT 主动去除这些偏置 | 纳入，作为对照动机 |
| SigLIP / CLIP | 用对比损失对齐图文嵌入，ViT 仅做视觉编码；MoonViT-V2 用 SigLIP 初始化作对照 | 不纳入，只在 MoonViT-V2 对照处提及 |
| Swin Transformer | 层级窗口注意力，引入 CNN 风格的归纳偏置；原始 ViT 是单尺度全局注意力 | 不纳入，独立概念 |
| MoonViT-V2 | K3 的 ViT 变体（27 层、从零训练、RMSNorm、去 bias、视频分解注意力） | 不纳入，已有概念页 [MoonViT-V2](../../wiki/moonvit-v2/index.html)，是本文的变体 |
| 位置编码（positional encoding） | ViT 用可学习 1D 绝对位置编码；位置编码本身是个独立概念家族（sin/cos、相对、RoPE、NoPE） | 纳入前置，已有概念页 [位置编码](../../wiki/positional-encoding/index.html) |

## 1.2 学习目标

### Q1：ViT 要解决什么问题——为什么要把图像切成 patch 当 token 喂给标准 Transformer，而不是继续用 CNN？

- **完成答案**：读者应能说明：CNN 的 locality 与 translation invariance 是先验归纳偏置，在小数据上有效但限制了模型在大数据下学到的关系种类；Dosovitskiy 2021 §1 假设大数据训练可以"trump inductive bias"，故把图像切成 patch 序列、直接喂给不假设空间结构的标准 Transformer，让模型从数据中自己学出空间关系。读者应能指出：ViT 不是"用 attention 改进 CNN"，而是"用纯 Transformer 替代 CNN"。
- **为什么是核心目标**：不理解"为什么试纯 Transformer"这一动机，就无法理解后续 patch 设计、class token、位置编码这些组件的存在意义，也无法理解 MoonViT-V2 为何能从零训练 ViT。
- **依赖内容**：标准注意力的最小机制（已有概念页）、CNN 局部卷积的最小概念（正文一句话给）。

### Q2：图像如何变成 token 序列——patch embedding、class token、位置编码各自做什么？

- **完成答案**：读者应能写出 $z_0=[x_{\text{class}};\,x_p^1 E;\,\ldots;\,x_p^N E]+E_{\text{pos}}$ 并逐项解释：$x\in\mathbb{R}^{H\times W\times C}$ 切成 $N=HW/P^2$ 个 $P\times P\times C$ patch、每个展平为 $P^2 C$ 维、用矩阵 $E\in\mathbb{R}^{P^2 C\times D}$ 线性投影到 $D$ 维；prepend 一个可学习 class token $x_{\text{class}}\in\mathbb{R}^D$；加上可学习 1D 位置编码 $E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$；得到 $N+1$ 个 $D$ 维 token 作为 Transformer 输入。读者应能手算：$224\times 224$ 图像、$P=16$ 时 $N=14\times 14=196$，加 class token 得 $197$ 个 token；ViT-B/16 配置下 $D=768$，输入张量形状为 $197\times 768$。
- **为什么是核心目标**：这是 ViT 的核心机制，缺则无法回答"图像如何变成 Transformer 输入"；后续变体（DeiT、Swin、MoonViT-V2）都从 patch embedding 出发做改动。
- **依赖内容**：矩阵乘法（占位提示 + 一句话衔接）；标准注意力（已有概念页）。

### Q3：Transformer 编码块如何处理 patch 序列——每一步在做什么、与标准 Transformer 有何不同？

- **完成答案**：读者应能写出编码块公式 $z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$、$z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$、分类头 $y=\text{LN}(z_L^0)$ 并逐项解释：$\text{LN}$ 是 LayerNorm，$\text{MSA}$ 是多头自注意力（每个 patch token 与所有 patch token 做标准缩放点积注意力，已在 standard-attention 概念页给出），$\text{MLP}$ 是两层 + GELU 的前馈网络，残差连接在每块后加上；$L$ 层堆叠后，class token 的输出 $z_L^0$ 经 LN 给最终表示 $y$。读者应能指出：与 Vaswani 2017 原始 Transformer 的区别——(a) ViT 用 pre-LN（LN 在子层前）而 Vaswani 用 post-LN；(b) ViT 只用 encoder 不用 decoder；(c) ViT 的 MSA 内部公式与 standard-attention 完全一致，只是 $Q,K,V$ 的来源从 token 嵌入变成 patch 嵌入。读者应能算出 ViT-B/12/16 配置：12 层、$D=768$、12 头、每头 $d_k=64$。
- **为什么是核心目标**：理解 ViT 不是新架构，而是标准 Transformer 的"输入端改造"。这是 MoonViT-V2 用 RMSNorm 替换 LN、去 bias 的前提。
- **依赖内容**：Q2 的 patch 序列；standard-attention（已有概念页）；residual-connection（已有概念页）。

### Q4：ViT 的适用边界——什么时候 ViT 比 CNN 强、什么时候反而更弱？

- **完成答案**：读者应能说明：Dosovitskiy 2021 §3.2–§4.2 给出边界——在 ImageNet-1k（1.3M 图）上从零训练 ViT 不如 ResNet；预训练在 ImageNet-21k（14M 图）后匹配 BiT-L（ResNet152x4）；预训练在 JFT-300M（300M 图）后 ViT-L/16 在 ImageNet 上 87.76%、ViT-H/14 达 88.55%，超过 BiT-L 的 87.54% 且用更少 TPUv3-core-days。读者应能指出：边界条件是预训练数据规模——大规模数据下"无归纳偏置"反而成为优势（模型可以学到 CNN 学不到的全局长程关系）；小规模数据下 ViT 缺乏 CNN 的局部先验、欠拟合。读者应能说明该结论成立的条件（大数据预训练 + 微调）和不成立的场景（小数据从零训练）。
- **为什么是核心目标**：明确概念的能力边界，是后续 MoonViT-V2"从零训练视觉编码器"挑战 SigLIP 预训练这一对照设定的前提。
- **依赖内容**：Q1–Q3 的机制；CNN 的对照设定。

## 1.3 内容分级

### 核心内容

| 核心内容 | 对应目标 | 必须讲清的结论 |
|---|---|---|
| CNN 两个归纳偏置（locality + translation invariance） | Q1 | 局部卷积 + 平移不变先验；小数据下有效但限制可学关系 |
| ViT 把图像切成 patch 序列当 token | Q1, Q2 | 主动去除空间归纳偏置，让模型从数据学 |
| Patch 切分 + 线性投影 $x_p^i E$ | Q2 | $N=HW/P^2$；每个 patch 展平 $P^2 C$ 维后投影到 $D$ 维 |
| Class token prepend | Q2 | 在序列最前加一个可学习 token，承载全局信息用于分类 |
| 可学习 1D 位置编码 $E_{\text{pos}}$ | Q2 | 弥补注意力位置无关性；ViT 选 1D 学习而非 sin/cos |
| $z_0$ 完整公式 | Q2 | 上述三步合成 $z_0=[x_{\text{class}};\,x_p^1 E;\ldots;\,x_p^N E]+E_{\text{pos}}$ |
| 224×224 / patch 16 / 196 token / 197 序列长度手算 | Q2 | 把 $N=HW/P^2$ 落到论文数字 |
| Transformer 编码块 $z_\ell', z_\ell$ | Q3 | pre-LN + MSA + 残差 + MLP；$L$ 层堆叠 |
| 分类头 $y=\text{LN}(z_L^0)$ | Q3 | class token 的输出作最终表示 |
| 与标准 Transformer 的差异（pre-LN、encoder-only、Q/K/V 来自 patch 嵌入） | Q3 | ViT 是输入端改造而非新架构 |
| 模型配置表 ViT-B/L/H | Q3 | 层数、$D$、头数、参数量（86M/307M/632M） |
| 数据规模边界（ImageNet-1k 不如 ResNet；JFT-300M 超越 BiT-L） | Q4 | 大数据训练胜过归纳偏置；小数据下反之 |
| 关键结果数字 87.76% / 88.55% / 87.54% 与 TPUv3-core-days | Q4 | 量化 ViT 在大数据下相对 CNN 的优势与计算效率 |

### 辅助内容

| 辅助内容 | 服务的核心内容/误解 |
|---|---|
| BERT [CLS] token 的来源类比 | 支撑 Q2：class token 不是 ViT 发明而是借用 NLP |
| 1D vs 2D 位置编码的对比（论文消融） | 支撑 Q2：ViT 选 1D 不比 2D 差，体现"无空间先验"的设计取向 |
| Patch 大小 $P$ 的选择（$P=14$ vs $P=16$ vs $P=32$） | 支撑 Q2：$P$ 越小 token 越多、计算越大、精度上限越高 |
| $D$ 与 patch 投影维度的对应（$D=P^2 C$ 的等价输入维度） | 支撑 Q2：投影矩阵的形状含义 |
| ResNet-152 / BiT-L 的对照数字 | 支撑 Q4：把"小数据下 ViT 不如 CNN"量化 |
| ViT-H/14 命名约定（H=Huge, 14=patch size） | 支撑 Q3：解读配置表命名 |

### 扩展内容

| 扩展内容 | 是否纳入 |
|---|---|
| DeiT 的蒸馏策略 | 不纳入（独立概念，仅列在变体列表） |
| Swin 的层级窗口注意力 | 不纳入（独立概念，仅列在变体列表） |
| BEiT/MAE 的掩码图像建模 | 不纳入（独立概念，仅列在变体列表） |
| DINOv2 自监督 | 不纳入（独立概念，仅列在变体列表） |
| 检测/分割下游（DETR、Mask2Former） | 不纳入（应用，仅一句提及 ViT 作 backbone） |
| MoonViT-V2 的视频分解注意力 | 不纳入（独立概念，已有页面，仅在前置说明处链接） |
| 训练超参数最优选择 | 不纳入（工程，本文用论文配置作示例） |
| 数据增强细节 | 不纳入（工程，本文只一句话提"训练 recipe"） |
| CLIP/SigLIP 多模态对齐 | 不纳入（独立概念，仅在 backbone 用途处提及） |
| 知识蒸馏 | 不纳入（独立概念，不在原始 ViT 论文范围） |

## 1.4 前置知识映射

| 前置知识 | 被哪些目标依赖 | 概念页状态 |
|---|---|---|
| 标准缩放点积注意力（$\text{softmax}(QK^\top/\sqrt{d_k})V$ + 多头） | Q3 | 已有概念页：[`../standard-attention/index.html`](../../wiki/standard-attention/index.html)；正文引用结论，不重复推导 |
| 残差连接（$y=F(x)+x$） | Q3 | 已有概念页：[`../residual-connection/index.html`](../../wiki/residual-connection/index.html)；正文引用结论 |
| 位置编码（绝对/相对/可学习家族） | Q2 | 已有概念页：[`../positional-encoding/index.html`](../../wiki/positional-encoding/index.html)；正文引用"可学习 1D 绝对位置编码"作为 ViT 的选择，不展开家族对比 |
| 矩阵乘法（$AB$ 的维度规则） | Q2 | 缺失，登记为 `matrix-multiplication`，正文用占位提示 + 一句话衔接（"$xE$ 是矩阵乘法，输出形状由维度规则决定"） |
| CNN 与卷积（局部 + 共享权重） | Q1 | 缺失，登记为 `cnn`，正文用占位提示 + 一句话衔接（"CNN 用局部卷积核扫描图像"） |
| LayerNorm（沿特征维归一化） | Q3 | 缺失，登记为 `layer-norm`，正文用占位提示 + 一句话衔接（"LN 把每行特征归一化到均值 0、方差 1"） |
| GELU 激活函数 | Q3 | 缺失，登记为 `gelu`，正文用占位提示 + 一句话衔接（"GELU 是平滑的 ReLU 变体"） |

登记的四个缺失前置概念（`matrix-multiplication`、`cnn`、`layer-norm`、`gelu`）均为深度学习入门基础。按 `guides/concept.md` 第 6 条"第 3 层起只登记不生成"的精神处理：登记、不递归生成、正文保留阅读所需的最小衔接（一两句话 + 公式定义），不内联大段背景讲解。MoonViT-V2 处于递归深度 1（其概念页将 ViT 列为前置），MoonViT-V2 的其余前置（SigLIP、next-token prediction）已在它自身的概念页登记。

## 1.5 明确不展开的内容

| 不展开项 | 与概念的关系 | 不展开原因 |
|---|---|---|
| DeiT/Swin/BEiT/MAE/DINOv2 各自的机制 | ViT 家族变体 | 独立概念，本文只在变体列表提一句名字 |
| MoonViT-V2 的视频分解注意力与从零训练细节 | ViT 家族变体（已有页面） | 独立概念，本文只链接到 MoonViT-V2 概念页 |
| 检测/分割下游任务（DETR、Mask2Former） | 应用 | 独立概念，本文只一句提及"ViT 作 backbone" |
| CLIP/SigLIP 的对比预训练 | 多模态对齐 | 独立概念，本文只在 backbone 用途处一句话提及 |
| 训练超参数（学习率、warmup、batch size）的最优选择 | 训练工程 | 不影响概念理解；本文用论文 Table 1 + Table 2 的配置作示例 |
| 数据增强细节（mixup、cutmix、randaug） | 训练工程 | 不影响概念理解；本文只在"训练 recipe"处一句话提"标准数据增强" |
| 知识蒸馏与教师-学生 | 训练技术 | 独立概念，不在原始 ViT 论文范围 |
| 知识蒸馏与教师-学生 | 训练技术 | 独立概念，不在原始 ViT 论文范围 |
| 多模态对齐（CLIP、SigLIP）训练目标 | 独立概念 | 本文只说明 ViT 是其视觉 backbone 的常见选择，不展开对齐损失 |

## 1.6 常见误解和适用边界

### 常见误解

1. **误解**：ViT 用 attention 替代 CNN 的卷积，所以 attention 与卷积是两种并列的视觉处理方式。
   **正确**：ViT 不是"用 attention 替换卷积"，而是"把图像变成 token 序列后整体丢给标准 Transformer"。Attention 是 Transformer 内部的一个子操作（标准缩放点积注意力），ViT 的真正创新在 tokenization（patch embedding + class token + position embedding）。卷积与 attention 的并列对比只在"是否假设空间结构"这一层面有意义。
   **形成原因**：把 ViT 简单概括为"用 attention 看图"。
   **影响目标**：Q1, Q3。

2. **误解**：Patch 是 ViT 特有的"图像 token"概念。
   **正确**：Patch 只是把图像切成方块、展平成向量的预处理步骤；token 化的真正机制是后面的线性投影 $E$。任何序列输入（文本词向量、音频片段）都可以视为 token；ViT 的特殊之处是把"图像 patch + 线性投影"作为 token 化方式，让 Transformer 不需要任何架构改动。
   **形成原因**：把 "patch" 与 "token" 混用，误以为 patch 本身是 Transformer 的输入单位。
   **影响目标**：Q2。

3. **误解**：Class token 是"分类才需要的辅助向量"，去掉它做 GAP 也能一样好。
   **正确**：Dosovitskiy 2021 §3.1 与附录 D 给出对照——原始 ViT 用 class token；附录消融显示在 JFT-300M 预训练下 GAP 与 class token 性能相近，但论文主线使用 class token，因为它继承自 BERT 且能让 Transformer 内部把"全局信息"显式聚合到一个位置。GAP 是替代方案而非默认方案。
   **形成原因**：看到后续一些模型用 GAP 就以为 class token 没必要。
   **影响目标**：Q2。

4. **误解**：ViT 在 ImageNet 上 88.55% 超过 ResNet，所以 ViT 普遍比 CNN 强。
   **正确**：88.55% 是 ViT-H/14 在 JFT-300M 预训练后的微调结果。Dosovitskiy 2021 §3.2 明确报告：在 ImageNet-1k（1.3M 图）上从零训练 ViT 不如 ResNet；ViT 的优势只在"大数据预训练 + 微调"条件下成立。脱离数据规模谈架构优劣是误读。
   **形成原因**：只引用最终 SOTA 数字，省略预训练数据规模这一前提。
   **影响目标**：Q4。

5. **误解**：ViT 是新架构，与标准 Transformer 是两种东西。
   **正确**：ViT 的编码块与 Vaswani 2017 的 Transformer encoder 几乎完全相同——MSA、MLP、残差、LayerNorm 都是 Transformer 的标准件。ViT 的"新"只在输入端（patch embedding + class token + 1D 位置编码）与输出端（用 class token 而非 pooled 输出做分类）。Q/K/V 的内部计算与 standard-attention 概念页完全一致。
   **形成原因**：把"Vision Transformer"中的 "Vision" 误读为"另一种 Transformer"。
   **影响目标**：Q3。

### 适用边界

- **解决的问题**：在大数据预训练 + 微调条件下，把图像分类从 CNN 范式迁移到标准 Transformer 范式，让模型从数据中学习空间关系而非依赖局部先验；与 CNN 相比在足够数据下达到更高精度且训练计算更少。
- **不解决的问题**：(a) 小数据从零训练——缺乏归纳偏置导致欠拟合，Dosovitskiy 2021 §3.2 已实验确认；(b) 检测/分割下游任务——需要加 task-specific head（DETR、Mask2Former），ViT 本身只做编码；(c) 视频时序建模——原始 ViT 只处理单帧；MoonViT-V2 等变体才扩展到视频；(d) 多模态对齐——需要 CLIP/SigLIP 的对比损失目标，ViT 仅作视觉 backbone。
- **成立条件**：(1) 输入图像大小固定（论文用 $224\times 224$，可微调到 $384$、$512$ 等，需要位置编码插值）；(2) patch 大小 $P$ 整除 $H$ 与 $W$，否则需要 pad 或 crop；(3) 大数据预训练（JFT-300M 或 ImageNet-21k 量级）才能达到论文报告精度。
- **条件不满足时**：(1) 输入大小变化时位置编码 $E_{\text{pos}}$ 形状不匹配，需要做 2D 插值（论文 §3.2 提及）；(2) $P$ 不整除时图像边缘信息丢失，工程上常用 center crop 到可整除尺寸；(3) 小数据从零训练时 ViT 比 CNN 差——这是 DeiT 等后续工作要解决的问题（用更强 augmentation + 蒸馏）。
