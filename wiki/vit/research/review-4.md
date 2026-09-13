<!-- review-meta
round: 4
page: wiki/vit/index.html
reviewed_content_sha256: a2bd09098b3e2fb6
-->
# Vision Transformer（ViT）审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 119ae0114924b9f4fc0f874eac5b8387422491cd（`git hash-object wiki/vit/index.html`）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何规划、修复或前序审查记录）
- 已完整阅读章节：head 与引言、核心问题、最容易误解、1. ViT 要解决什么问题——CNN 的归纳偏置与大数据下的局限、2. 图像如何变成 token 序列——patch embedding、class token、位置编码、3. Transformer 编码块——与标准 Transformer 几乎完全一致、4. 数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱、来源与范围说明；含全部 20 个折叠块、流程块图注与全部表格
- 已核对来源：Dosovitskiy et al. 2021, arXiv:2010.11929v2 全文（摘要、§1、§3.1 逐段、§4.1–§4.4、附录 D.3/D.4、Table 1、Table 2 及封面脚注）；Xiong et al. 2020, arXiv:2002.04745 题录；站内 wiki/standard-attention、wiki/residual-connection、wiki/positional-encoding、wiki/moonvit-v2
- 机械验证：`python3 .dojo/scripts/validate.py wiki/vit/index.html` 返回 validation ok；无 `research/` 路径引用；无「（待生成）」占位；正文 15 个来源上标 [C1–C8][F1–F4][N1–N3] 与来源章节双向对应；折叠块 summary 前缀仅「解答：」「补充：」；公式定界符外无 Unicode 数学字符（唯一 `×` 在 `<meta description>` 纯文本，规范允许）；索引/公式数字经回源核对一致

## 问题

- [重要·技术] 4. 数据规模边界「适用边界」表第 3–5 行（检测/分割下游任务、视频时序建模、多模态对齐）：三行给出 DETR/Mask2Former「需加专用头」、MoonViT-V2 等变体才支持视频、CLIP/SigLIP 用对比损失对齐等机制归因，页面在这些行未加任何来源上标，且「来源与范围说明」的「辅助解释与类比边界」「简化条件及其限制」两节均未登记它们。整表置于论文来源章节之下，读者会误以为这些结论出自 Dosovitskiy 2021。｜引文依据：Dosovitskiy 2021 全文检索 "DETR" 0 次、"Mask2Former" 0 次、"SigLIP" 0 次，"CLIP" 2 次均为 "gradient clipping"；§4.3 Pre-training Data Requirements 与 §4.4 Scaling Study 只讨论预训练数据规模、迁移与预训练计算量，全文无检测/分割、视频时序、多模态对齐内容。｜修复要求：为这三行补可核对来源（各方法原始论文，或站内对应概念/论文页）并在行内标上标，或删除这三行并在「辅助解释与类比边界」中改标为明确推断；overview.html「关键结论与边界」第 3 条同段需同步修改。｜修复：｜复验：

- [轻微·技术] 1. ViT 要解决什么问题：CNN 归纳偏置条目写作 "Translation invariance（平移不变性）"，机制说明为「同一个卷积核在图像上共享权重」，实为平移等变性；第 1 章表格「空间先验」列的 "locality + translation invariance" 与本章问题解答同。｜引文依据：Dosovitskiy 2021 §3.1 Inductive bias 段原文 "In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model"；§1 亦写 "such as translation equivariance and locality"。｜修复要求：术语改为 translation equivariance（平移等变性），并在条目内说明「不变性需池化等操作才出现」；overview.html 首条同段同步。｜修复：｜复验：

- [轻微·技术] 引言、第 1 章辅助解释与常见误解、第 1 章表格把 ViT 的空间先验写成「无」（表格「空间先验＝无（patch + 标准 Transformer）」、callout「标准 Transformer 不预设任何空间规则」、常见误解「它'无空间先验'的核心设计取向」），比来源措辞更强。｜引文依据：§3.1 Inductive bias 段 "Vision Transformer has much less image-specific inductive bias than CNNs"、"In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global"、"The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches"。｜修复要求：把「无空间先验」限定为「几乎没有 vision-specific 归纳偏置」，并点明 patch 切分仍保留少量 2D 邻域结构；或在「简化条件及其限制」中登记该简化及其不可推出的结论。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明」[C8] 标注「Dosovitskiy 2021, Table 1（§3.1 末尾）」；Table 1 实际位于 §4.1 Setup。｜引文依据：ar5iv 全文层级中 Table 1 "Details of Vision Transformer model variants" 紧接 §4.1 Setup 的 "Training & Fine-tuning." 段，位置在 §4.1 内，不在 §3.1。｜修复要求：把 [C8] 的位置由「§3.1 末尾」改为「§4.1」。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明」[C3]「§3.1 第 3 段」、[C4]「§3.1 第 4 段」、[C6]「§3.1 第 3 段」的段落序号与原文相差一段。｜引文依据：原文 §3.1 共 4 段——第 1 段模型概述与 patch 切分（Eq.1）、第 2 段 class token 与分类头（"Similar to BERT's [class] token, we prepend a learnable embedding…"、"The classification head is implemented by a MLP with one hidden layer…"）、第 3 段位置编码（"Position embeddings are added to the patch embeddings…"）、第 4 段编码器（Eq.2–3）。｜修复要求：class token 与分类头改为第 2 段、位置编码改为第 3 段；或删去段号只保留「§3.1」与 Eq. 编号，避免段序歧义。｜修复：｜复验：

- [轻微·技术] 第 2 章折叠块「补充：class token vs GAP」中「后续一些模型（如 DeiT 部分变体）改用 GAP」未给来源，且 DeiT 主线的分类设计是 class token + distillation token，并非 GAP。｜引文依据：页面未标注该举例出处，无法核对（原文 §3.1 与附录 D.3 未提及 DeiT）。｜修复要求：删除该举例，或替换为确有出处、采用 GAP 的模型并补来源上标。｜修复：｜复验：

- [轻微·表述] 章节过渡反复使用固定句式「下一章看…」共 4 处（第 1 章末、第 2 章末、第 3 章末、第 4 章末），另第 2 章有元话语引导句「把流程画出来：」。｜引文依据：不适用。｜修复要求：按 style-guide §8 改写为说明前一节结论与下一节问题关系的句子（不使用固定句式，也不为形式完整而加过渡），删除「把流程画出来：」这类引导语。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 6
- 处置：修复

（本轮未发现阻断问题：Table 1 三个模型配置 12/768/3072/12/86M、24/1024/4096/16/307M、32/1280/5120/16/632M 与 Table 1 一致；Table 2 各行 85.30±0.02/0.23k、87.76±0.03/0.68k、88.55±0.04/2.5k、87.54±0.02/9.9k、88.4–88.5/12.3k 与原文一致（Noisy Student 预训练数据 ImageNet + JFT-300M 亦有 §4.2 原文支持）；§4.1 三个数据集 1.3M/1k、14M/21k、303M/18k 一致；Eq.(1)–(4)、§3.1「learned 1D position embeddings」「MLP with one hidden layer + tanh」「two layers with a GELU」、D.3「Both work similarly well, but require different learning-rates」、D.4「little to no difference」均与原文一致；页面内手算 224×224/P=16→196、+class token→197、P²C=768、d_k=768/12=64、2.5k 约为 9.9k 的 1/4 与 12.3k 的 1/5 均可复算无误；核心问题与各章「本章问题」均有独立可读解答并指向对应章节；站内前置概念页 standard-attention / residual-connection / positional-encoding / moonvit-v2 均真实存在且内链有效。）