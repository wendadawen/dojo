<!-- review-meta
round: 9
page: wiki/vit/index.html
reviewed_content_sha256: 63769011dc052136
-->
# ViT 审查记录（第 9 轮）

- 页面版本：wiki/vit/index.html 工作树哈希 b43e1f7bcb41a3cd55aa6e98c23c6a0f06aec1d2（sha256 前缀 d54eb8469fe64c44）
- 审查时间：2026-09-14 17:54
- 审查者：独立子代理
- 类型规范：dojo:type=concept → guides/concept/check.md（并参照 guides/concept/style-guide.md）
- 已核对文献版本：Dosovitskiy et al. 2021, arXiv:2010.11929v2（2021-06-03 ICLR camera-ready；PDF 从 arxiv.org 取全文核对）；Xiong et al. 2020 arXiv:2002.04745；Carion et al. 2020 arXiv:2005.12872；Cheng et al. 2022 arXiv:2112.01527；Radford et al. 2021 arXiv:2103.00020；Zhai et al. 2023 arXiv:2303.15343。
- 已完整阅读章节：核心问题 / 最容易误解 / 1. ViT 要解决什么问题 / 2. 图像如何变成 token 序列 / 3. Transformer 编码块 / 4. 数据规模边界 / 来源与范围说明（含全部折叠块、callout、表格、chapter-summary 与 JS）。

## 核对依据（关键原文与数值）

- §1 原文（PDF 第 73 行）："We find that large scale training trumps inductive bias."；同节 "train models of unprecedented size, with over 100B parameters"。→ 支持 [C1] 与引言 100B 陈述。
- Table 1 原文：ViT-Base 12 / 768 / 3072 / 12 / 86M；ViT-Large 24 / 1024 / 4096 / 16 / 307M；ViT-Huge 32 / 1280 / 5120 / 16 / 632M；正文 "The 'Base' and 'Large' models are directly adopted from BERT and we add the larger 'Huge' model."。→ 页面 §3 配置表与命名约定逐格一致。
- Table 2 原文行（PDF 第 339–348 行）：ImageNet ViT-H/14 88.55 ± 0.04；ViT-L/16 (JFT) 87.76 ± 0.03；ViT-L/16 (I21k) 85.30 ± 0.02；BiT-L (ResNet152x4) 87.54 ± 0.02；Noisy Student 88.4/88.5∗。TPUv3-core-days 2.5k / 0.68k / 0.23k / 9.9k / 12.3k。表注："mean and standard deviation of the accuracies, averaged over three fine-tuning runs"；脚注 "∗ Slightly improved 88.5% result reported in Touvron et al. (2020)."。→ 页面 §4 表、正文读数与 [N3] 一致；2.5k/9.9k≈1/4、2.5k/12.3k≈1/5 复核成立。
- §4.1 原文（PDF 第 259–261 行）："ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images"；"ImageNet-21k with 21k classes and 14M images"；"JFT ... with 18k classes and 303M high-resolution images"。→ 页面数据集表一致。
- §3.1 Eq.(1)–(4) 与页面 [F1]–[F4] 逐字一致：z_0=[x_class; x_p^1 E; …; x_p^N E]+E_pos，E∈R^{(P²C)×D}，E_pos∈R^{(N+1)×D}；z'_ℓ=MSA(LN(z_{ℓ−1}))+z_{ℓ−1}；z_ℓ=MLP(LN(z'_ℓ))+z'_ℓ；y=LN(z_L^0)。正文 "Layernorm (LN) is applied before every block, and residual connections after every block"（pre-LN）、"The MLP contains two layers with a GELU non-linearity"、分类头 "MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time"。→ 页面 §3 全部一致。
- §3.1 位置编码："We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4)"。附录 D.3："a small multi-layer perceptron (MLP) with tanh as non-linearity in the single hidden layer"、"In order to stay as close as possible to the original Transformer model, we made use of an additional [class] token"；Figure 9 说明 "Both work similarly well, but require different learning-rates."；D.4 "there is little to no difference between different ways of encoding positional information"。→ 页面两个补充折叠块与 class token vs GAP 表述一致。
- §4.3（PDF 第 388–435 行）：Figure 3 说明 "While large ViT models perform worse than BiT ResNets (shaded area) when pre-trained on small datasets…"；正文 "Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets."；"The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes."。→ 支持 [C7] 小数据边界。
- 外部编号核对：[N4] 四个 arXiv 编号均命中正确论文——2005.12872 "End-to-End Object Detection with Transformers"（Carion et al.）、2112.01527 "Masked-attention Mask Transformer for Universal Image Segmentation"（Cheng et al. 2022）、2103.00020 "Learning Transferable Visual Models From Natural Language Supervision"（Radford et al.）、2303.15343 "Sigmoid Loss for Language Image Pre-Training"（Zhai et al.）。[C5] 引用的 2002.04745 确为 Xiong et al. 2020 "On Layer Normalization in the Transformer Architecture"，其结论正是 pre-LN 相比 post-LN 梯度更良性、可省 warm-up。
- 手算复核：224×224/16²=196；N+1=197；P²C=16·16·3=768；D/h=768/12=64；P=32→49、P=14→256；197²×64 与 O((N+1)²d_k) 一致。均与页面一致。
- 站内链接核对：standard-attention（§1 注意力要解决什么问题、§4 多头注意力、§5 复杂度与边界、来源 [F1] 均真实存在）、residual-connection、positional-encoding、moonvit-v2、clip、siglip 全部存在；无"（待生成）"占位。MoonViT-V2 侧"27 层 / 从零训练 / RMSNorm / 去 bias / 视频分解注意力"与其页面一致。
- 功能与格式：unlabeled 无 Unicode 数学字符（唯一一处 224×224 出现在 meta description，属纯文本栏，不被 §11 覆盖）；标题/summary/正文/表格中数学符号全部为 LaTeX；无图片、无 SVG 图、无 <pre> 代码块（故代码执行项不适用）；alt 无 $...$；无交互视图；公式定界符规范。.dojo/scripts/validate.py 返回 "validation ok: wiki/vit/index.html"。dojo:topics=多模态、dojo:tag=视觉与多模态 均在白名单内。overview.html 与 index.html 相互链接，数值（196 patch / 197 token / 88.55% / 87.54% / 2.5k vs 9.9k）与正文一致。

## 问题

- [轻微·技术] 来源：Dosovitskiy 2021 arXiv:2010.11929v2 §3.1｜位置：「1. ViT 要解决什么问题」正文第 2 段（"模型也无法学到与平移等变先验相冲突的关系（如'目标在图像左上角'这一类位置敏感任务）"）及本章问题第 2 题解答（"学不出来"）｜问题：把"CNN 学不到位置敏感关系"写成无条件机制结论，来源不支持该论断｜引文依据：§3.1 仅陈述 "In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model."，全文没有"无法学到"这类断言；§4.3 把 CNN 的小数据优势归因于归纳偏置与过拟合差异，也未称此类关系"学不出来"｜修复要求：降级为可核对表述（如"这类关系与平移等变先验冲突，CNN 要学到需要更多数据或更深堆叠"），或明确标注为推断｜修复：｜复验：
- [轻微·技术] 来源：Dosovitskiy 2021 arXiv:2010.11929v2 §3.1（patch 切分 N=HW/P²）｜位置：「1. ViT 要解决什么问题」表格下方 chapter-summary 首句 "$n$ 为把图像视作一维序列时的位置数（这里取边长像素数 224）"｜问题：$n$ 的文字定义与其取值自相矛盾，读 $O(\log_k n)$ 时会误判量级｜引文依据：224×224 图像展平为一维序列的位置数是 50176（像素）或 196（patch），224 是边长方向的像素数；同句却称"一维序列时的位置数"｜修复要求：把 $n$ 定义改为"图像边长方向的像素数，这里取 224"（或等效表述），使定义与取值一致｜修复：｜复验：
- [轻微·技术] 来源：Dosovitskiy 2021 arXiv:2010.11929v2 §4.2 Table 2｜位置：「4. 数据规模边界」Table 2 表内 Noisy Student 行与正文读数（"88.4–88.5"两处）｜问题：把来源的两个独立数值写成区间，丢失"两个来源各一个数"的信息，易读成精度区间｜引文依据：Table 2 原文为 "88.4/88.5∗"，脚注 "∗ Slightly improved 88.5% result reported in Touvron et al. (2020)."（斜杠分隔本文 88.4 与 Touvron 2020 的 88.5，非区间）｜修复要求：改写为 "88.4 / 88.5" 并在表注或 chapter-summary 补一句 88.5 来自 Touvron et al. 2020｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布（阻断与重要项均为 0；3 条轻微项为表述精度问题，建议顺带修复，不阻塞发布）
