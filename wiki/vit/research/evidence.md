# Vision Transformer（ViT）：核心论断与证据

来源优先级：原始论文 > 权威教材/同行评审综述 > 官方文档 > 固定版本源码。本页核心论断全部来自 Dosovitskiy et al. 2021 原论文（ICLR 2021, arXiv:2010.11929v2）。WebSearch 获取的二手资料仅用于交叉确认公式与论文位置，不作为核心论断的唯一依据。

## C 论断（核心机制）

### C1 ViT 主动去除 CNN 的空间归纳偏置，把图像分类迁移到标准 Transformer 范式

- **论断内容**：CNN 的 locality（局部卷积核）与 translation invariance（平移不变先验）在小数据下有效但限制了模型在大数据下可学到的关系种类；Dosovitskiy 2021 §1 假设"大规模训练胜过归纳偏置"，把图像切成 patch 序列直接喂给不假设空间结构的标准 Transformer，让模型从数据中自己学出空间关系。ViT 不是"用 attention 改进 CNN"，而是"用纯 Transformer 替代 CNN"。
- **来源定位**：Dosovitskiy et al. 2021, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021, arXiv:2010.11929v2, §1 第 1 段（"we train a model on large image datasets, while avoiding binding priors about the input"）与 §4.4（"large scale training trumps inductive bias"）。
- **适用条件**：预训练数据足够大（ImageNet-21k 14M 或 JFT-300M 300M 图量级）。
- **置信状态**：已确认。

### C2 图像切成 patch 后线性投影成 token 是 ViT 的核心 tokenization 机制

- **论断内容**：输入 $x\in\mathbb{R}^{H\times W\times C}$ 切成 $N=HW/P^2$ 个 $P\times P\times C$ patch，每个 patch 展平为 $P^2 C$ 维向量，再用线性投影矩阵 $E\in\mathbb{R}^{P^2 C\times D}$ 投影到 $D$ 维。$N$ 是 Transformer 的有效序列长度。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(1) 与 Figure 1。原文："We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder."
- **适用条件**：$P$ 整除 $H$ 与 $W$；否则需要 pad 或 crop。
- **置信状态**：已确认。

### C3 Class token（[CLS]）prepend 在序列最前，承载全局信息用于分类

- **论断内容**：在 patch 序列最前 prepend 一个可学习 class token $x_{\text{class}}\in\mathbb{R}^{D}$，使序列长度从 $N$ 变为 $N+1$。class token 与 patch token 一起经过 $L$ 层 Transformer 编码，最终输出 $z_L^0$ 用作分类头输入。该设计借鉴 BERT 的 [CLS] token。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(1) 与 §3.1 第 3 段（"Following Viereck et al. and Devlin et al., we prepend a learnable embedding to the sequence of embedded patches (z_0^0), whose state at the output of the Transformer encoder (z_L^0) serves as the image representation y"）。
- **适用条件**：分类任务；GAP 是替代方案（附录 D 消融）。
- **置信状态**：已确认。附录 D.1 给出 GAP 与 class token 在 JFT-300M 预训练下性能相近的消融，但论文主线使用 class token。

### C4 可学习 1D 绝对位置编码 $E_{\text{pos}}$ 加在 patch 嵌入上，弥补注意力的位置无关性

- **论断内容**：位置编码 $E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$ 直接加到 patch 嵌入序列（含 class token）上；论文选择"learned 1D position embeddings"而非 sin/cos 或 2D 形式。该选择源于注意力本身位置无关（已在 [位置编码](../../wiki/positional-encoding/index.html) 概念页说明），需要外接位置信息。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(1)（$+E_{\text{pos}}$）与 §3.1 第 4 段（"Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings"）。附录 D.4 给出 1D vs 2D 位置编码的消融，1D 不比 2D 差。
- **适用条件**：输入分辨率固定；分辨率变化时需要插值。
- **置信状态**：已确认。

### C5 编码块用 pre-LN + MSA + 残差 + MLP，与标准 Transformer 几乎完全一致

- **论断内容**：每层编码块公式 $z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$、$z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$，$\ell=1\ldots L$。MSA 是多头自注意力（内部公式与 [标准注意力](../../wiki/standard-attention/index.html) 完全一致），MLP 是两层 + GELU，残差连接在每块后加上。与 Vaswani 2017 原始 Transformer 的区别：(a) ViT 用 pre-LN（LN 在子层前）而 Vaswani 用 post-LN；(b) ViT 只用 encoder 不用 decoder；(c) Q/K/V 的来源从 token 嵌入变成 patch 嵌入。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(2)–Eq.(3)。原文："$z_\ell' = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$, $\ell=1\ldots L$"；"$z_\ell = \text{MLP}(\text{LN}(z_\ell')) + z_\ell'$, $\ell=1\ldots L$"。
- **适用条件**：标准配置；后续变体（MoonViT-V2）用 RMSNorm 替换 LN、去 bias，但整体结构不变。
- **置信状态**：已确认。pre-LN 选择是 ViT 与 Vaswani 2017 的可核对差异（Xiong et al. 2020 等后续工作讨论了 pre-LN 的训练稳定性优势，本页只在差异处一句话提及，不展开）。

### C6 分类头 $y=\text{LN}(z_L^0)$——class token 的输出作最终表示

- **论断内容**：Transformer 编码器输出 $z_L\in\mathbb{R}^{(N+1)\times D}$，取第 0 行（class token 对应位置）$z_L^0$，经 LayerNorm 得到图像表示 $y\in\mathbb{R}^D$。预训练时分类头是一个 MLP（含一个隐藏层 + GELU），微调时简化为单线性层。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(4)（"$y = \text{LN}(z_L^0)$"）与 §3.1 第 3 段（"Both during pre-training and fine-tuning, a classification head is attached to $z_L^0$. The classification head is implemented by an MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time."）。
- **适用条件**：分类任务；下游检测/分割等任务用 task-specific head 替代。
- **置信状态**：已确认。

### C7 数据规模是 ViT 相对 CNN 优势的边界条件

- **论断内容**：Dosovitskiy 2021 §3.2 报告在 ImageNet-1k（1.3M 图）上从零训练 ViT 不如 ResNet；§4.2 报告预训练在 ImageNet-21k（14M 图）后 ViT-L/16 在 ImageNet 上达 85.30%；预训练在 JFT-300M（300M 图）后 ViT-L/16 达 87.76%、ViT-H/14 达 88.55%，超过 BiT-L（ResNet152x4）的 87.54%。论文 §4.4 总结"large scale training trumps inductive bias"。
- **来源定位**：Dosovitskiy et al. 2021, §3.2（小数据结果）、§4.2 Table 2（大数据结果）、§4.4（结论）。
- **适用条件**：大数据预训练 + 微调。
- **置信状态**：已确认。论文 Table 2 同时给出 TPUv3-core-days（计算量），ViT-H/14 用 2.5k、BiT-L 用 9.9k、Noisy Student 用 12.3k——ViT 用更少计算达到更高精度。

### C8 ViT 模型配置 ViT-B/L/H 与命名约定

- **论断内容**：论文 Table 1 给出三档配置：ViT-Base（12 层、$D=768$、12 头、MLP size 3072、86M 参数）；ViT-Large（24 层、$D=1024$、16 头、MLP size 4096、307M 参数）；ViT-Huge（32 层、$D=1280$、16 头、MLP size 5120、632M 参数）。命名"ViT-L/16"表示 Large 配置 + patch size 16。
- **来源定位**：Dosovitskiy et al. 2021, Table 1（§3.1 末尾）。
- **适用条件**：Base/Large 配置直接沿用 BERT；Huge 是本文新增。
- **置信状态**：已确认。

## F 公式

### F1 Patch embedding + class token + 位置编码（合成的 $z_0$）

- **公式**：$z_0=[x_{\text{class}};\,x_p^1 E;\,\ldots;\,x_p^N E]+E_{\text{pos}}$，其中 $E\in\mathbb{R}^{P^2 C\times D}$、$E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$、$N=HW/P^2$。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(1)。
- **适用条件**：$P$ 整除 $H$ 与 $W$；输入分辨率固定。
- **置信状态**：已确认。

### F2 编码块（MSA 子层）

- **公式**：$z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$，$\ell=1\ldots L$。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(2)。
- **适用条件**：pre-LN 设置；MSA 内部公式见 [标准注意力](../../wiki/standard-attention/index.html)。
- **置信状态**：已确认。

### F3 编码块（MLP 子层）

- **公式**：$z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$，$\ell=1\ldots L$。MLP 是两层 + GELU。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(3)。
- **适用条件**：pre-LN 设置；GELU 是平滑的 ReLU 变体（占位提示 + 一句话衔接）。
- **置信状态**：已确认。

### F4 分类头

- **公式**：$y=\text{LN}(z_L^0)$。
- **来源定位**：Dosovitskiy et al. 2021, §3.1, Eq.(4)。
- **适用条件**：分类任务。
- **置信状态**：已确认。

## N 数字

### N1 模型配置（ViT-B/L/H）

- **数字**：ViT-Base: 12 层、$D=768$、12 头、MLP size 3072、86M 参数；ViT-Large: 24 层、$D=1024$、16 头、MLP size 4096、307M 参数；ViT-Huge: 32 层、$D=1280$、16 头、MLP size 5120、632M 参数。论文主线 patch size 16，Huge 也用 14（ViT-H/14）。
- **来源定位**：Dosovitskiy et al. 2021, Table 1。
- **适用条件**：原始配置；Base 与 Large 沿用 BERT。
- **置信状态**：已确认。

### N2 224×224 输入 + patch 16 → 196 token 序列长度

- **数字**：$H=W=224$、$P=16$、$C=3$（RGB）。$N=HW/P^2=224\times 224/16^2=50176/256=196$。加 class token 得 $N+1=197$ 个 token。ViT-B/16 配置下 $D=768$，输入张量 $z_0\in\mathbb{R}^{197\times 768}$。
- **来源定位**：Dosovitskiy et al. 2021, §3.1（patch 切分公式）与 Table 1（ViT-B 配置）。224×224 是 ImageNet 标准输入尺寸。
- **适用条件**：224×224 输入 + patch 16 + RGB 图像。
- **置信状态**：已确认。

### N3 关键实验结果与计算量对比

- **数字**：ImageNet top-1 微调精度——ViT-L/16 (JFT-300M) 87.76%、ViT-H/14 (JFT-300M) 88.55%、ViT-L/16 (ImageNet-21k) 85.30%、BiT-L (ResNet152x4, JFT-300M) 87.54%、Noisy Student (EfficientNet-L2) 88.4–88.5%。预训练计算量（TPUv3-core-days）——ViT-H/14 2.5k、ViT-L/16 (JFT) 0.68k、ViT-L/16 (I21k) 0.23k、BiT-L 9.9k、Noisy Student 12.3k。
- **来源定位**：Dosovitskiy et al. 2021, Table 2（§4.2）。
- **适用条件**：对应预训练数据集 + ImageNet 微调。
- **置信状态**：已确认。论文同时报告 ImageNet ReaL、CIFAR-10/100、Oxford-IIIT Pets、Oxford Flowers-102、VTAB 等多数据集结果，本页只引 ImageNet top-1 与计算量对比，其他数字不在核心论断内。

## 来源清单

| 编号 | 引用 | 用途 |
|---|---|---|
| [Dosovitskiy2021] | Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021. arXiv:2010.11929v2 | C1–C8, F1–F4, N1–N3 |
| [Vaswani2017] | Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762 | C5 差异对照（pre-LN vs post-LN、encoder-only） |
| [Devlin2019] | Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019. arXiv:1810.04805 | C3 [CLS] token 来源（仅一句提及借鉴 NLP） |
| [Xiong2020] | Xiong, R. et al. "On Layer Normalization in the Transformer Architecture." ICML 2020. arXiv:2002.04745 | C5 pre-LN 训练稳定性的对照（不展开） |
| [K3-Report] | Kimi K3 技术报告 §2.4（MoonViT-V2 视觉编码器章节） | 变体对照（仅链接，不在核心论断内） |

无冲突论断。所有核心论断置信状态均为"已确认"，主要依据为 Dosovitskiy et al. 2021 原论文与 Table 1 / Table 2 / 附录 D 的可核对数字。
