# Vision Transformer（ViT）：教学大纲

## 1. 页面开头

**钩子**：把一张 $224\times 224$ 的猫图给 ResNet，它会用 $3\times 3$ 卷积核逐层扫描局部像素、堆几十层才把"猫"判断出来；把同一张图给 ViT，它先把图切成 $14\times 14=196$ 个 $16\times 16$ 的方块，每个方块当一个 token 喂给标准 Transformer，让 12 层自注意力自己决定"猫耳朵的 patch 该看哪些 patch"——结果是后者在大数据下精度更高、训练计算更少。

**一句话解释**：Vision Transformer（ViT）把图像切成 patch 序列、用线性投影变成 token、prepend 一个 class token、加上可学习 1D 位置编码后送入标准 Transformer 编码器，用 class token 的最终隐状态做分类。

**要解决的具体问题**：CNN 的局部卷积与平移不变先验在小数据下有效但限制模型在大数据下可学到的关系种类；ViT 主动去除这些空间归纳偏置，把图像分类迁移到 NLP 已证明可扩展的标准 Transformer 范式。

**学习承诺（与 scope.md Q1–Q4 一致）**：
- 说明 ViT 要解决什么问题——为什么把图像切成 patch 当 token 喂给标准 Transformer，而不是继续用 CNN；
- 写出 patch embedding + class token + 位置编码的合成公式 $z_0$，逐项解释符号，手算 $224\times 224$ patch 16 → 196 token；
- 写出编码块 $z_\ell', z_\ell$ 与分类头 $y$ 公式，说明与标准 Transformer 的差异（pre-LN、encoder-only、Q/K/V 来源）；
- 指出 ViT 的数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱。

**首个具体场景**：224×224 ImageNet 图像、patch 16、ViT-Base 配置（$D=768$、12 层、12 头），用具体数字算一遍 $z_0$ 的形状与序列长度。

**过渡到第一章**：先看清 CNN 的两个归纳偏置到底是什么、在大数据下为什么成为限制，再理解 ViT 为何主动去除它们。

## 2. 章节设计

### S1 ViT 要解决什么问题——CNN 的归纳偏置与大数据下的局限

- **主要教学问题**：CNN 的 locality 与 translation invariance 是什么？为什么在大数据下它们反而成为限制？ViT 的设计取向是什么？
- **对应范围**：Q1；C1。
- **正文要点**：
  1. CNN 的两个归纳偏置（占位链接到 `cnn` + 一句话衔接）：
     - Locality（局部性）：卷积核只看局部邻域像素，假设"相邻像素关系强于远距离像素"。
     - Translation invariance（平移不变）：卷积核在图像上共享权重，假设"目标平移后特征不变"。
  2. 这两个偏置在小数据下是优势：参数共享、平移鲁棒、不需要从头学空间结构；但限制了模型在大数据下可学到的关系种类——比如"猫坐在沙发上"这种长程关系，CNN 要堆很多层才能让远距离像素交互。
  3. NLP 已经验证标准 Transformer 的可扩展性：数据越多、模型越大，性能持续上升（BERT → GPT-3 趋势）。Dosovitskiy 2021 §1 假设这种可扩展性来自"无空间归纳偏置"——让模型从数据中自己学空间关系。
  4. ViT 的设计取向：把图像变成与文本 token 同构的 patch 序列，直接喂给标准 Transformer，不做任何 vision-specific 的架构改动。Attention 在 Transformer 内部按"内容相关加权"自动学出该看哪些 patch，不预设局部性。
  5. **常见误解**（callout）：ViT 不是"用 attention 替换卷积"，而是"用纯 Transformer 替代 CNN"。Attention 只是 Transformer 内部的一个子操作；ViT 的真正创新在 tokenization（下一章）。
  6. 与 NLP Transformer 的对应：文本 token = 词向量；图像 token = patch 投影。Transformer 内部完全相同。
- **讲解材料及职责**：
  - CNN 归纳偏置 callout（标为教学解释）：把 locality 与 translation invariance 落到直觉，列失效边界。
  - CNN vs ViT 对照表：参数共享方式、空间先验、远距离交互路径、可学关系。
- **前置知识安排**：`cnn` 占位链接 + 一句话衔接；不内联展开。
- **完成检查**：
  - 用一句话说出 CNN 的两个归纳偏置（locality + translation invariance）；
  - 用一句话说出这些偏置在小数据下是优势、在大数据下为什么成为限制；
  - 说出 ViT 的设计取向是"无空间先验、让模型从数据学"。
- **过渡**：现在知道"为什么试纯 Transformer"。但 Transformer 的输入是 token 序列——图像怎么变成 token？下一章看 patch embedding + class token + 位置编码三步。

### S2 图像如何变成 token 序列——patch embedding、class token、位置编码

- **主要教学问题**：$z_0=[x_{\text{class}};\,x_p^1 E;\,\ldots;\,x_p^N E]+E_{\text{pos}}$ 的每项做什么？每个符号代表什么？224×224 / patch 16 → 多少 token？
- **对应范围**：Q2；C2, C3, C4, F1, N1, N2。
- **正文要点**：
  1. **第一步 Patch 切分**：输入 $x\in\mathbb{R}^{H\times W\times C}$（$H$ 高、$W$ 宽、$C$ 通道数；RGB 图像 $C=3$）。切成 $N=HW/P^2$ 个 $P\times P\times C$ 的不重叠 patch。$P$ 是 patch 大小，论文主线用 $P=16$。
  2. **第二步 展平 + 线性投影**：每个 patch 展平为 $P^2 C$ 维向量 $x_p^i\in\mathbb{R}^{P^2 C}$；用矩阵 $E\in\mathbb{R}^{P^2 C\times D}$ 线性投影到 $D$ 维。占位链接到 `matrix-multiplication` + 一句话衔接（"$xE$ 是矩阵乘法，输出 $D$ 维"）。投影后的 $x_p^i E\in\mathbb{R}^{D}$ 是一个 patch token。
  3. **第三步 Class token prepend**：在序列最前 prepend 一个可学习向量 $x_{\text{class}}\in\mathbb{R}^{D}$，借鉴 BERT 的 [CLS] token（一句提及，不展开 BERT）。序列长度从 $N$ 变为 $N+1$。
  4. **第四步 位置编码**：注意力本身位置无关（已在 [位置编码](../../wiki/positional-encoding/index.html) 概念页说明），需要外接位置信息。ViT 选可学习 1D 绝对位置编码 $E_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}$，直接加到序列上。1D vs 2D 在论文附录 D 消融中相近，本文不展开对比。
  5. **合成公式 F1**：$z_0=[x_{\text{class}};\,x_p^1 E;\,\ldots;\,x_p^N E]+E_{\text{pos}}$。$z_0\in\mathbb{R}^{(N+1)\times D}$ 是 Transformer 的输入。$[;]$ 表示沿序列维拼接。
  6. **手算例子（教学示例，正文）**：$H=W=224$、$P=16$、$C=3$。$N=224\times 224/16^2=50176/256=196$。加 class token 得 $197$ 个 token。ViT-B/16 配置下 $D=768$，$z_0\in\mathbb{R}^{197\times 768}$。
  7. **投影矩阵的形状含义**：$E\in\mathbb{R}^{P^2 C\times D}=\mathbb{R}^{768\times 768}$（$P^2 C=16\times 16\times 3=768$，恰好等于 $D=768$——这是 Base 配置的巧合，不是必须）。$E_{\text{pos}}\in\mathbb{R}^{197\times 768}$。
  8. **Patch 大小 $P$ 的影响**：$P$ 越小 token 越多、计算越大、精度上限越高；$P=32$ 时 $N=49$，$P=14$ 时 $N=256$。论文 Table 1 同时给出 ViT-H/14 配置（patch 14）。
  9. **常见误解**（callout）：Patch 不是 token；patch 是图像预处理产物，token 化的真正机制是后面的线性投影 $E$。把 "patch" 与 "token" 混用是误读。
  10. **Class token vs GAP**（一句话）：附录 D 消融显示在 JFT-300M 预训练下 GAP 与 class token 性能相近；论文主线用 class token 是因为继承自 BERT 且能让 Transformer 内部把"全局信息"显式聚合到一个位置。GAP 是替代方案而非默认。
- **讲解材料及职责**：
  - $z_0$ 合成公式 F1 + 完整符号表：建立正式定义。
  - 224×224 手算例子（正文）：把 $N=HW/P^2$ 落到论文数字。
  - Patch 切分 ASCII 图示：把图像 → 196 个 patch → 196 个 token 的过程可视。
  - 形状对照表：$x$ / patch / $x_p^i$ / $E$ / $x_p^i E$ / $E_{\text{pos}}$ / $z_0$ 各自的维度。
- **前置知识安排**：`matrix-multiplication` 占位链接 + 最小衔接；已有概念页 `positional-encoding` 链接。
- **完成检查**：
  - 写出 $z_0$ 公式并说出每个符号；
  - 算出 224×224 patch 16 的 token 数（$N=196$、含 class token 序列长度 $197$）；
  - 说出 class token 与位置编码各自的作用；
  - 说出"patch 与 token 的区别"。
- **过渡**：$z_0$ 全有了，下一步该 Transformer 编码了。下一章看编码块 $z_\ell', z_\ell$ 与分类头 $y$。

### S3 Transformer 编码块——与标准 Transformer 几乎完全一致

- **主要教学问题**：$z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$、$z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$、$y=\text{LN}(z_L^0)$ 各做什么？与 Vaswani 2017 原始 Transformer 有何差异？ViT 是新架构吗？
- **对应范围**：Q3；C5, C6, C8, F2, F3, F4, N1。
- **正文要点**：
  1. **整体框架**：$L$ 层编码块堆叠，每层接收 $z_{\ell-1}\in\mathbb{R}^{(N+1)\times D}$ 输出 $z_\ell\in\mathbb{R}^{(N+1)\times D}$。每层内部含两个子层——多头自注意力（MSA）+ 前馈网络（MLP），每个子层前用 LayerNorm（pre-LN）、后用残差连接（已在 [残差连接](../../wiki/residual-connection/index.html) 概念页给出）。
  2. **公式 F2（MSA 子层）**：$z_\ell'=\text{MSA}(\text{LN}(z_{\ell-1}))+z_{\ell-1}$。LN 把每行特征归一化到均值 0、方差 1（占位链接到 `layer-norm` + 一句话衔接）；MSA 是多头自注意力，内部公式与 [标准注意力](../../wiki/standard-attention/index.html) 完全一致 $\text{softmax}(QK^\top/\sqrt{d_k})V$，只是 $Q,K,V$ 来自 patch 嵌入的线性投影，而非 NLP 的词嵌入。
  3. **公式 F3（MLP 子层）**：$z_\ell=\text{MLP}(\text{LN}(z_\ell'))+z_\ell'$。MLP 是两层 + GELU（占位链接到 `gelu` + 一句话衔接——GELU 是平滑的 ReLU 变体）。残差连接在两个子层后都加上。
  4. **公式 F4（分类头）**：$y=\text{LN}(z_L^0)$。$z_L\in\mathbb{R}^{(N+1)\times D}$ 是 $L$ 层后的输出，取第 0 行 $z_L^0$（class token 对应位置），经 LN 得到图像表示 $y\in\mathbb{R}^{D}$。预训练时分类头是一个 MLP（含一个隐藏层 + GELU），微调时简化为单线性层。
  5. **与标准 Transformer 的差异**（callout）：(a) ViT 用 pre-LN（LN 在子层前）而 Vaswani 2017 用 post-LN——pre-LN 训练更稳定（Xiong 2020 讨论过，本页只一句提及）；(b) ViT 只用 encoder 不用 decoder（无交叉注意力子层）；(c) Q/K/V 的来源从 token 嵌入变成 patch 嵌入。除此之外，MSA、MLP、残差、LayerNorm 都是 Transformer 的标准件——ViT 不是新架构，而是输入端改造。
  6. **手算 ViT-B/12/16 配置**（正文）：$L=12$、$D=768$、12 头、每头 $d_k=D/h=768/12=64$。每层 MSA 内部计算量与 [标准注意力](../../wiki/standard-attention/index.html) 概念页 §5 给出的 $O((N+1)^2 d_k)$ 一致；这里 $N+1=197$，注意力矩阵 $197\times 197$。
  7. **模型配置表 N1**（正文表格）：ViT-Base（12 层、$D=768$、12 头、MLP 3072、86M）；ViT-Large（24 层、$D=1024$、16 头、MLP 4096、307M）；ViT-Huge（32 层、$D=1280$、16 头、MLP 5120、632M）。命名"ViT-L/16" = Large 配置 + patch 16。
  8. **ViT-H/14 命名约定**（一句）：H = Huge，14 = patch size；Huge 是本文新增，Base 与 Large 沿用 BERT。
  9. **常见误解**（callout）：ViT 不是"新 Transformer 架构"——它复用标准 Transformer 的全部标准件；MoonViT-V2 等变体用 RMSNorm 替换 LN、去 bias，是在 ViT 基础上的修改，不改变 Transformer 的整体结构。
- **讲解材料及职责**：
  - 公式 F2 + F3 + F4 + 完整符号表：建立编码块与分类头定义。
  - 模型配置表 N1（正文表格）：给具体可核对数字。
  - ViT-B/12/16 手算（正文）：算每头维度 $d_k=64$、注意力矩阵 $197\times 197$。
  - 与标准 Transformer 差异对照表：pre-LN vs post-LN、encoder-only、Q/K/V 来源。
- **前置知识安排**：已有概念页 `standard-attention`、`residual-connection`；`layer-norm`、`gelu` 占位链接 + 一句话衔接。
- **完成检查**：
  - 写出 $z_\ell', z_\ell, y$ 三条公式并解释每个符号；
  - 说出 pre-LN 与 post-LN 的差异，以及 ViT 选择 pre-LN；
  - 算出 ViT-B/12/16 的每头维度（$d_k=64$）与注意力矩阵大小（$197\times 197$）；
  - 说出 ViT 与标准 Transformer 的三处差异（pre-LN、encoder-only、Q/K/V 来源）。
- **过渡**：机制全清楚了。但论文报告的 88.55% 是在什么条件下达到的？ViT 一定比 CNN 强吗？下一章看数据规模边界。

### S4 数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱

- **主要教学问题**：ViT 在 ImageNet 上的 88.55% 是怎么来的？小数据从零训练 ViT 会怎样？大数据下 ViT 相对 CNN 的优势是什么？
- **对应范围**：Q4；C7, N3。
- **正文要点**：
  1. **三个预训练数据集**（正文表格）：ImageNet-1k（1.3M 图、1k 类）；ImageNet-21k（14M 图、21k 类）；JFT-300M（300M 图、18k 类，私有）。Dosovitskiy 2021 §4.1。
  2. **小数据从零训练的边界**（§3.2）：在 ImageNet-1k 上从零训练 ViT 不如 ResNet——缺乏 CNN 的局部先验、欠拟合。这是后续 DeiT 等工作（更强 augmentation + 蒸馏）要解决的问题，本页只一句提及。
  3. **大数据预训练 + 微调的优势**（§4.2 Table 2，正文表格）：
     - ImageNet top-1 微调精度：ViT-L/16 (JFT-300M) 87.76%、ViT-H/14 (JFT-300M) 88.55%、ViT-L/16 (ImageNet-21k) 85.30%；
     - 对照：BiT-L（ResNet152x4, JFT-300M）87.54%、Noisy Student（EfficientNet-L2, ImageNet+JFT）88.4–88.5%。
  4. **计算量对比**（正文表格，TPUv3-core-days）：ViT-H/14 用 2.5k、ViT-L/16 (JFT) 0.68k、ViT-L/16 (I21k) 0.23k、BiT-L 9.9k、Noisy Student 12.3k。ViT 用更少计算达到更高精度——大数据下"无归纳偏置"反而成为优势。
  5. **核心结论**（§4.4，callout）："large scale training trumps inductive bias"——大规模训练胜过归纳偏置。这是 Dosovitskiy 2021 的核心论断，也是后续 ViT 家族（DeiT、Swin、BEiT、MAE、DINOv2、MoonViT-V2）的共同前提。
  6. **适用边界总结**：成立条件——大数据预训练 + 微调；不成立场景——小数据从零训练、检测/分割下游（需 task-specific head）、视频时序（原始 ViT 只处理单帧，MoonViT-V2 才扩展到视频）。
  7. **常见误解**（callout）：ViT 在 ImageNet 上 88.55% 超过 ResNet 不等于"ViT 普遍比 CNN 强"——88.55% 是 ViT-H/14 在 JFT-300M 预训练后的微调结果，脱离预训练数据规模谈架构优劣是误读。
  8. **变体家族一句话提及**：DeiT（蒸馏 + 增强）、Swin（层级窗口）、BEiT/MAE（掩码预训练）、DINOv2（自监督）、MoonViT-V2（K3 视觉编码器，已有概念页）。各变体在 ViT 基础上改动训练目标、注意力模式或归一化方式，但 patch 化 + Transformer 编码的整体框架不变。
- **讲解材料及职责**：
  - 三个数据集表格：把 ImageNet-1k/21k/JFT-300M 的规模与类别数对比。
  - ImageNet top-1 精度 + TPUv3-core-days 对照表：把 ViT vs CNN 的精度与计算量对照。
  - 数据规模边界对照表：小数据 / 大数据下 ViT 与 CNN 的相对优势。
- **前置知识安排**：无新前置；只用 Q1–Q3 已引入的术语。
- **完成检查**：
  - 说出 ViT 在 ImageNet 上 88.55% 对应的预训练数据集（JFT-300M）与模型（ViT-H/14）；
  - 说出小数据从零训练 ViT 的相对表现（不如 ResNet）；
  - 说出大数据下 ViT 相对 CNN 的两个优势（精度更高、计算更少）；
  - 说出该结论的成立条件与至少一个不成立场景。
- **过渡**：（文末收束）ViT 把图像分类从 CNN 范式迁移到标准 Transformer 范式；理解了它，就理解了所有后续变体（DeiT、Swin、MoonViT-V2 等）要保住什么、改什么。

## 3. 讲解顺序

S1 → S2 → S3 → S4。先讲为什么需要它（CNN 归纳偏置与大数据下的限制），再讲是什么（patch embedding + class token + 位置编码 + 编码块），最后讲数据规模边界。每章只引入一组新记号：S1 引入 CNN 两个归纳偏置；S2 引入 $x, P, N, E, x_{\text{class}}, E_{\text{pos}}, z_0, D$；S3 引入 $L, \ell, \text{LN}, \text{MSA}, \text{MLP}, y$；S4 引入 ImageNet-1k/21k/JFT-300M 与 TPUv3-core-days。前置概念（cnn、matrix-multiplication、layer-norm、gelu）在首次依赖时给占位链接 + 一句话衔接；已有概念页（standard-attention、residual-connection、positional-encoding）正文引用结论。

## 4. 贯穿例子

**贯穿例子**：224×224 ImageNet 图像 + patch 16 + ViT-Base 配置（$D=768$、12 层、12 头）。
- S2 首次出现：算出 $N=196$、$N+1=197$、$z_0\in\mathbb{R}^{197\times 768}$、$E\in\mathbb{R}^{768\times 768}$、$E_{\text{pos}}\in\mathbb{R}^{197\times 768}$。
- S3 复用：在 ViT-B/12/16 配置下算每头 $d_k=768/12=64$、注意力矩阵 $197\times 197$。
- S4 复用：224×224 + patch 16 对应 ViT-B/16 与 ViT-L/16；H/14 用 224×224 + patch 14 → $N=256$，序列长度 257。

单一 224×224 + patch 16 例子无法覆盖 ViT-H/14 的 patch 14 变体；S4 用 H/14 作对照数字。

## 5. 讲解材料职责

| 材料 | 服务教学问题 | 位置 |
|---|---|---|
| 公式 F1 $z_0$ 合成 + 符号表 | S2：建立 tokenization 定义 | S2 正文 |
| 公式 F2 + F3 + F4 编码块与分类头 | S3：建立编码机制 | S3 正文 |
| CNN 归纳偏置 callout | S1：动机 | S1 正文 |
| CNN vs ViT 对照表 | S1：动机 | S1 正文 |
| Patch 切分 ASCII 图示 | S2：把图像 → token 流程可视 | S2 正文 |
| 形状对照表（$x$ / patch / $E$ / $z_0$） | S2：维度变化 | S2 正文 |
| 224×224 patch 16 手算例子 | S2：把 $N=HW/P^2$ 落到数字 | S2 正文 |
| 模型配置表 N1 | S3：可核对数字 | S3 正文 |
| ViT-B/12/16 每头维度手算 | S3：把 $d_k=D/h$ 落到数字 | S3 正文 |
| 与标准 Transformer 差异对照表 | S3：差异定位 | S3 正文 |
| 三个数据集表格 | S4：数据规模 | S4 正文 |
| ImageNet top-1 + TPUv3-core-days 对照表 | S4：精度与计算量对比 | S4 正文 |
| 数据规模边界对照表 | S4：能解决 vs 不能解决 | S4 正文 |
| "大数据胜过归纳偏置" callout | S4：核心结论 | S4 正文 |

无"为配代码而配代码"的材料。本章不安排可运行代码：核心机制（patch 切分 + 线性投影 + 编码块）用 224×224 手算与 ViT-B 配置手算即可验证，无需程序；编码块与 [标准注意力](../../wiki/standard-attention/index.html) 完全一致，已有页面给出验证。

## 6. 正文与折叠块分工

**必须放正文**：
- CNN 两个归纳偏置（locality + translation invariance）与大数据下成为限制（S1）
- ViT 的设计取向：无空间先验、让模型从数据学（S1）
- $z_0$ 合成公式 F1 + 全部符号解释（$x, P, N, E, x_{\text{class}}, E_{\text{pos}}, D, z_0$）（S2）
- 224×224 + patch 16 → 196 token / 197 序列长度手算（S2）
- Class token 与位置编码各自的作用（S2）
- 编码块公式 F2 + F3 + 分类头 F4 + 全部符号（$L, \ell, \text{LN}, \text{MSA}, \text{MLP}, y$）（S3）
- 与标准 Transformer 的三处差异（pre-LN、encoder-only、Q/K/V 来源）（S3）
- 模型配置表 N1（ViT-B/L/H 层数、隐藏维、头数、参数量）（S3）
- ViT-B/12/16 每头维度 $d_k=64$ 与注意力矩阵 $197\times 197$ 手算（S3）
- 三个预训练数据集（ImageNet-1k/21k/JFT-300M）（S4）
- ImageNet top-1 精度与 TPUv3-core-days 对照数字（S4）
- "大数据胜过归纳偏置"核心结论（S4）
- 数据规模边界：小数据下 ViT 不如 ResNet、大数据下 ViT 超越 CNN（S4）
- 全部前置概念占位链接与已有概念页链接

**可放折叠块**：
- BERT [CLS] token 来源的简要说明（S2 折叠块，NLP 借鉴的最小衔接）
- 1D vs 2D 位置编码消融的简要说明（S2 折叠块，附录 D 消融数字）
- Pre-LN 与 post-LN 的训练稳定性对照（S3 折叠块，Xiong 2020 一句话提及）
- ViT-H/14 patch 14 → 256 token 的手算对照（S4 折叠块，配置命名约定的扩展）
- GAP vs class token 消融的简要说明（S2 折叠块，附录 D 消融）

折叠块全部收起时，正文仍能回答 Q1–Q4：每个学习目标的结论与至少一个可复算数字（224×224 patch 16 → 196 token）都在正文。

## 7. 范围与证据约束

本章全部内容来自 scope.md 纳入范围与 evidence.md 已确认论断。无新增学习目标、无新增核心论断、无范围外内容。若写作中发现缺口，返回规划阶段。
