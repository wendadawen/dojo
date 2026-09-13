<!-- review-meta
round: 3
page: wiki/vit/index.html
reviewed_content_sha256: e26c09ae68d3b955
-->
# Vision Transformer（ViT）审查记录（第 3 轮）

- 页面版本：6f9b6d4330366c6f4e454045b5dbbd1481c3f78c
- 审查时间：2026-09-13 19:13
- 审查者：独立子代理（未参与写作与前序审查）
- 规范：guides/concept/check.md（dojo:type = concept）
- 主要来源：Dosovitskiy et al. 2021, arXiv:2010.11929v2（正文与附录以 ar5iv HTML 原文逐段核对）
- 机械验证：`python3 .dojo/scripts/validate.py wiki/vit/index.html` → validation ok
- 已完整阅读章节：核心问题、最容易误解、1. ViT 要解决什么问题（含本章问题）、2. 图像如何变成 token 序列（含本章问题）、3. Transformer 编码块（含本章问题）、4. 数据规模边界（含本章问题）、来源与范围说明；对照核对了 head、overview.html 与 wiki/standard-attention、wiki/positional-encoding、wiki/residual-connection、wiki/moonvit-v2 引用页

## 核对记录（关键来源位置，供复验）

- §3.1 Eq.(1)：`z_0 = [x_class; x_p^1 E; …; x_p^N E] + E_pos,  E∈R^{(P²·C)×D}, E_pos∈R^{(N+1)×D}` —— 与页面 [F1] 一致。
- Eq.(2)(3)：`z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}`、`z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ`；正文 "Layernorm (LN) is applied before every block, and residual connections after every block"（pre-LN 成立）—— 与 [F2][F3][C5] 一致。
- Eq.(4)：`y = LN(z_L^0)` —— 与 [F4] 一致。
- Table 1：ViT-Base 12/768/3072/12/86M；ViT-Large 24/1024/4096/16/307M；ViT-Huge 32/1280/5120/16/632M —— 与 [N1] 及正文表完全一致。
- "Model Variants. We base ViT configurations on those used for BERT … The 'Base' and 'Large' models are directly adopted from BERT and we add the larger 'Huge' model." —— 与 [C8]、line 348 一致。
- §3.1："The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time." —— 与 line 309 前半一致（但激活见问题 2）。
- Table 2：ViT-L/16 ImageNet-21k 85.30±0.02 / 0.23k；ViT-L/16 JFT-300M 87.76±0.03 / 0.68k；ViT-H/14 JFT-300M 88.55±0.04 / 2.5k；BiT-L 87.54±0.02 / 9.9k；Noisy Student 88.4–88.5 / 12.3k —— 与 [N3] 及正文表一致。
- §1："When trained on mid-sized datasets such as ImageNet … modest accuracies of a few percentage points below ResNets"；"We find that large scale training trumps inductive bias."（位于 §1 Introduction，非摘要）—— [C1] 引 §1 正确。
- §4.3："The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes."（支持"小数据不如 CNN"）。同段另见："Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets."（见问题 1）。
- 附录 D.3：class token vs GAP；"Both work similarly well, but require different learning-rates."（见问题 4）。
- 附录 D.4：1-D 0.64206 / 2-D 0.64001（差异不显著，与页面一致）。
- 内部引用：standard-attention §5 "复杂度、瓶颈与边界" 存在且给出 `O(n² d_k)`；O(1)/位置无关性引用成立。4 个前置概念页（standard-attention / residual-connection / positional-encoding / moonvit-v2）均真实存在，无占位。

## 问题

- [重要·技术] 「4. 数据规模边界」第 2 段（line 387），及 line 149、line 420：小数据下 ViT 弱于 ResNet 的机制被写成"欠拟合"，与所引来源 §4.3 的"过拟合"结论相反｜引文依据：§4.3 "Vision Transformers **overfit** more than ResNets with comparable computational cost on smaller datasets."；§1 仅说 "lack some of the inductive biases inherent to CNNs … and therefore do not generalize well when trained on insufficient amounts of data."（全文无 underfit/欠拟合 表述）｜修复要求：把三处"欠拟合"改为与来源一致的表述（如"过拟合更严重 / 泛化更差"），并确保所标 §4.3 能直接支持新表述｜修复：｜复验：
- [重要·技术] line 309（$z_L^0$ 条目）：预训练分类头隐藏层的激活写成 GELU｜引文依据：附录 D.3 "The output of this token is then transformed into a class prediction via a small multi-layer perceptron (MLP) with **tanh** as non-linearity in the single hidden layer. This design is inherited from the Transformer model for text, and we use it throughout the main paper."（编码块内 MLP 才是 GELU）｜修复要求：分类头激活改为 tanh（编码块 MLP 的 GELU 保留），或删除该激活描述｜修复：｜复验：
- [重要·技术] 「主要依据」行（line 95）与 [C7]（line 473）：把 §4.4 标注为"结论"｜引文依据：ar5iv 4.4 标题为 "4.4 Scaling Study"（内容为 pre-training compute 对比），论文结论为 "5 Conclusion"；§4.4 不含结论性段落｜修复要求：把 §4.4 的"结论"标注改为 §4.4 Scaling Study，或改引 §5 Conclusion；核实修改后该处论断（大数据下 ViT 精度更高、计算更少）确有对应来源位置｜修复：｜复验：
- [重要·技术] line 131 与 line 220：把"论文主线使用 class token"的理由 (2) 写成"让 Transformer 内部把'全局信息'显式聚合到一个位置"｜引文依据：附录 D.3 自述理由为 "In order to stay as close as possible to the original Transformer model, we made use of an additional [class] token … This design is inherited from the Transformer model for text"；Figure 9 caption "Both work similarly well, but require different learning-rates."（论文未给出"显式聚合全局信息"这一理由）｜修复要求：删除理由 (2)，或降级为明确标注的推断；理由 (1) 按来源改为"沿用文本 Transformer / BERT 的 [class] token 设计"｜修复：｜复验：
- [轻微·技术] line 215「为什么选可学习 1D 而非 sin/cos 或 2D 位置编码」：把选 1D 的原因写成"最简单、不引入额外空间先验"｜引文依据：§3.1 "We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4)."｜修复要求：改为来源给出的理由，或标注为推断｜修复：｜复验：
- [轻微·技术] line 263：把 Table 1 出现 ViT-H/14 的原因写成"Huge 模型用更细粒度 patch 配合更大模型容量"，并把"$P$ 越小…精度上限越高"写成无条件论断｜引文依据：Table 1 仅列出 ViT-Huge（patch 14），论文未说明为何 Huge 选用 patch 14，也未论证"patch 越小精度上限越高"｜修复要求：删除该因果断言或标注为推断｜修复：｜复验：
- [轻微·技术] line 381：数据集表把 JFT 图像数写作 300M｜引文依据：§4.1 "JFT (Sun et al. 2017) with 18k classes and **303M** high-resolution images."｜修复要求：改为 303M，或注明"约 300M（JFT-300M 为常用名）"｜修复：｜复验：
- [轻微·技术] line 152："Dosovitskiy 2021 §1 假设这种可扩展性来自'无空间归纳偏置'"把页面自身的概括写成来源假设｜引文依据：§1 原文为 "We find that large scale training trumps inductive bias."（未提出"可扩展性来自无空间归纳偏置"的假设句）｜修复要求：改写为来源原意，或标注为推断｜修复：｜复验：
- [轻微·格式] line 484：正文出现 Unicode"×"（"224×224 是 ImageNet 标准输入尺寸"），与全文其余位置 `$224\times 224$` 写法不一致；同表数据列"±"（85.30 ± 0.02 等）亦为 Unicode 数学符号｜引文依据：不适用｜修复要求：该处改为 KaTeX 渲染；"±" 一并按 check.md §2.2-9 处理或说明例外｜修复：｜复验：
- [轻微·格式] line 489：构造示例条目写"ViT-B/12/16 每头维度 $d_k=64$ 手算"，命名与全文约定不符｜引文依据：不适用（页面自设约定为"ViT-<规格>/<patch>"，如 ViT-B/16）｜修复要求：改为"ViT-B/16"｜修复：｜复验：
- [轻微·表述] line 190、288、369、462、502：口语化过渡与临场评价——"现在知道'为什么试纯 Transformer'。""$z_0$ 全有了，下一步该 Transformer 编码了。""机制全清楚了。""本文从 CNN 归纳偏置出发，解释了…回到核心问题：…均已作答"｜引文依据：不适用｜修复要求：改为客观陈述或直接以内容衔接（保留"下一章看"式的导航句即可，去掉评价性/口语化措辞与"本文…"自述）｜修复：｜复验：
- [轻微·表述] line 414、418、455、458：把"场景"当术语（表格列头"场景"、"成立条件与不成立场景"）｜引文依据：不适用（check.md §2.2-12 将"把'场景'当术语"列为 AI 拼接腔）｜修复要求：改为"情形""条件"等具体词｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 4 / 轻微 8
- 处置：修复。事实性数字（Table 1、Table 2、196/197、Eq.(1)–(4)、pre-LN、命名约定）全部回到 arXiv:2010.11929v2 原文核对无误，无阻断问题；但问题 1（欠拟合/过拟合方向相反）与问题 2（分类头激活 GELU/tanh）与所引来源直接冲突，问题 3、4 属来源归因错误，须逐条修复并复验后再判可发布。
