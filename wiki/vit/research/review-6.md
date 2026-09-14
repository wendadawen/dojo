<!-- review-meta
round: 6
page: wiki/vit/index.html
reviewed_content_sha256: b9a3dfa5caf833f7
-->
# ViT（Vision Transformer）审查记录（第 6 轮）

- 页面版本：b646dc03057d285b4534e0f54af0acf894fabba7
- 审查时间：2026-09-14 17:16
- 审查者：独立子代理
- 已完整阅读章节：核心问题、最容易误解、1. ViT 要解决什么问题——CNN 的归纳偏置与大数据下的局限（含本章问题）、2. 图像如何变成 token 序列——patch embedding、class token、位置编码（含三个折叠补充块与本章问题）、3. Transformer 编码块——与标准 Transformer 几乎完全一致（含折叠补充块与本章问题）、4. 数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱（含本章问题）、来源与范围说明（含论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）

## 来源核对依据（本轮查证）

以 arXiv:2010.11929v2（ar5iv HTML 版）逐条核对，均一致：Eq.(1) z₀=[x_class; x_p¹E; …; x_p^N E]+E_pos、Eq.(2) z′_ℓ=MSA(LN(z_{ℓ−1}))+z_{ℓ−1}、Eq.(3) z_ℓ=MLP(LN(z′_ℓ))+z′_ℓ、Eq.(4) y=LN(z_L⁰)；Table 1（ViT-B 12/768/3072/12/86M、ViT-L 24/1024/4096/16/307M、ViT-H 32/1280/5120/16/632M）；Table 2（ViT-L/16 I21k 85.30/0.23k、ViT-L/16 JFT 87.76/0.68k、ViT-H/14 JFT 88.55/2.5k、BiT-L 87.54/9.9k、Noisy Student 88.4–88.5/12.3k）；§4.1（ImageNet-1k 1.3M/1k、ImageNet-21k 14M/21k、JFT 303M/18k）；§3.1 “The MLP contains two layers with a GELU non-linearity”“Similar to BERT's [class] token”“We follow the original Transformer (Vaswani et al. 2017) as closely as possible”；§4.3 “Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets”；附录 D.3（GAP 与 class token “Both work similarly well, but require different learning-rates”，及 tanh 出处）；附录 D.4（1-D≈0.642 vs 2-D≈0.640）。手算均可复算：224×224/16²=196、N+1=197、P²C=768、d_k=768/12=64、P=32→49、P=14→256、2.5/9.9≈1/4、2.5/12.3≈1/5。文献编号 arXiv:2010.11929v2 / 1810.04805 / 2002.04745 / 2005.12872 / 2112.01527 / 2103.00020 / 2303.15343 均对应所引论文。overview.html 与 index.html 数字一致且互链；`python3 .dojo/scripts/validate.py wiki/vit/index.html` 返回 validation ok；页面无 Unicode 数学字符、无 `<img>`（无 alt 公式问题）、核心问题与四个本章问题均有解答折叠块。

## 问题

- [重要·技术] 第 4 章「核心结论」段：§1 原文引文 “large scale training trumps inductive bias” 标注引用编号 `<sup>[C7]</sup>`，但 [C7] 登记位置为 §4.3/§4.2 Table 2/§4.4（来源与范围说明 [C7] 条目），不含 §1；同一句引文在第 1 章正文（line 142）标注的是 [C1]，而 [C1] 定义恰为 “§1（"We find that large scale training trumps inductive bias"）”。同一句引文在正文两处使用不同引用编号，且 [C7] 所指章节定位不到该引文。｜引文依据：论文 §1 “We find that large scale training trumps inductive bias.”；页内 [C1] “Dosovitskiy et al. 2021, §1（"We find that large scale training trumps inductive bias"）”、[C7] “§4.3（小数据结果）、§4.2 Table 2（大数据结果）、§4.4 Scaling Study”。｜修复要求：把该处引文编号改为 [C1]（与第 1 章一致）；若同时要覆盖该段 §4 实证，写为 `<sup>[C1, C7]</sup>`。｜修复：｜复验：

- [轻微·技术] 第 3 章逐项解释「$z_L^0$」条目：“预训练时分类头接在 $z_L^0$ 后是一个 MLP（含一个隐藏层 + tanh 非线性），微调时简化为单线性层。”该内容归属上标 [C6] 所登记的 “§3.1, Eq.(4) 与 §3.1 第 2 段”，但 §3.1 只说 “The classification head is implemented by a MLP with one hidden layer at pre-training time … by a single linear layer at fine-tuning time”，未提 tanh；tanh 的出处是附录 D.3。按 [C6] 所指 §3.1 定位不到 tanh 依据（事实本身正确，仅来源位置不精确）。｜引文依据：§3.1 “The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.”；附录 D.3 “a small multi-layer perceptron (MLP) … with tanh as non-linearity in the single hidden layer”。｜修复要求：在 [C6] 定义中补入附录 D.3，或将该处 tanh 依据直接指向附录 D.3。｜修复：｜复验：

- [轻微·表述] 「最容易误解」第 1 条：“ViT 是"把图像变成 token 序列后整体丢给标准 Transformer"”。“丢给”与全页统一用语“喂给”（标题、description 摘要及各章正文共 11 处）不一致，属口语化措辞。｜引文依据：不适用｜修复要求：改为“喂给”（或“送入”），与全页用词保持一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复