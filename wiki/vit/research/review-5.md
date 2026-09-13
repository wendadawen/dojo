<!-- review-meta
round: 5
page: wiki/vit/index.html
reviewed_content_sha256: 248f5d5621069c37
-->
# Vision Transformer（ViT）审查记录（第 5 轮）

- 页面版本：b57bf7abcb9ee61c5d530eebb2bfbb32d96ede98
- 审查时间：2026-09-13 20:27
- 审查者：独立子代理（未参与写作，亦未参与前序轮次审查与修复）
- 已完整阅读章节（按顺序）：标题与引言、核心问题、最容易误解、1. ViT 要解决什么问题——CNN 的归纳偏置与大数据下的局限（含本章问题与折叠块）、2. 图像如何变成 token 序列——patch embedding、class token、位置编码（含本章问题与折叠块）、3. Transformer 编码块——与标准 Transformer 几乎完全一致（含本章问题与折叠块）、4. 数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱（含本章问题与折叠块）、来源与范围说明（论断与来源（C）／公式与来源（F）／外部数字与实验条件（N）／构造示例／辅助解释与类比边界／简化条件及其限制）
- 来源核对：Dosovitskiy et al. 2021, ICLR 2021, arXiv:2010.11929v2（§1、§3.1、§3.2、§4.1–§4.4、Table 1、Table 2、附录 D.3／D.4）逐条回源；Xiong et al. 2020（arXiv:2002.04745）；BERT（Devlin et al. 2019, arXiv:1810.04805）。
- 机械验证：`.dojo/scripts/validate.py wiki/vit/index.html` → `validation ok`；`dojo:type=concept`、`dojo:topics=多模态`、`dojo:tag=视觉与多模态` 均在 `.dojo/scripts/catalog_builder.py` 词表内；`index.html` 与 `overview.html` 双向链接；正文引用的 6 个概念页（standard-attention／residual-connection／positional-encoding／moonvit-v2／clip／siglip）均真实存在；无 `research/` 文件路径引用、无"（待生成）"占位；无 `<pre>/<code>` 代码块，故"可运行代码"一项不适用（N/A）。

## 回源核对（通过项，作为引文依据）

- §1 原文："Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality"与"However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias."——支持正文"CNN 两个归纳偏置（locality + translation equivariance）"与"large scale training trumps inductive bias"两处论断。
- §3.1 原文："In model design we follow the original Transformer (Vaswani et al. 2017) as closely as possible."；附录 D.3："In order to stay as close as possible to the original Transformer model, we made use of an additional [class] token, which is taken as image representation."与"The output of this token is then transformed into a class prediction via a small multi-layer perceptron (MLP) with tanh as non-linearity in the single hidden layer."——支持 class token 借鉴 BERT、预训练头含 tanh 非线性、微调阶段单线性层。
- 附录 D.3："the difference in performance is fully explained by the requirement for a different learning-rate."——支持"GAP 与 class token 整体性能相当（需不同学习率）"。附录 D.4："little to no difference between different ways of encoding positional information."——支持"1D vs 2D 差异不显著"。§3.1："we have not observed significant performance gains from using more advanced 2D-aware position embeddings."——支持位置编码选型理由。
- Table 1：ViT-Base 12/768/3072/12/86M、ViT-Large 24/1024/4096/16/307M、ViT-Huge 32/1280/5120/16/632M——与正文配置表逐格一致。§3.1 的命名约定 ViT-L/16、ViT-H/14（patch 16／14）一致；论文对 Huge 选 patch 14 未给理由，正文"论文未说明 Huge 选用 patch 14 的理由"成立。
- §4.1 数据集：ImageNet 1.3M/1k、ImageNet-21k 14M/21k、JFT-300M 303M/18k，且原文称 JFT 为 "high-resolution images"——与正文表及 chapter-summary 一致。
- Table 2（§4.2，caption："We report mean and standard deviation of the accuracies, averaged over three fine-tuning runs."）：ViT-L/16 (ImageNet-21k) 85.30±0.02／0.23k、ViT-L/16 (JFT-300M) 87.76±0.03／0.68k、ViT-H/14 (JFT-300M) 88.55±0.04／2.5k、BiT-L (ResNet152x4) 87.54±0.02／9.9k、Noisy Student (EfficientNet-L2) 88.4/88.5／12.3k——正文表内全部数字（含 ± 标准差）与计算量逐项一致。
- 算式复算：224×224=50176，50176/256=196，N+1=197；14×14=196；16×16×3=768；768/12=64；224/14=16→16²=256、224/32=7→7²=49；9.9/2.5≈3.96≈1/4、12.3/2.5=4.92≈1/5——全部与正文标注相符，符号 $N,D,P,L,d_k,E,E_{\text{pos}},z_L^0$ 全文单义。
- 延伸引用回源：DETR（arXiv:2005.12872）、Mask2Former（arXiv:2112.01527）、CLIP（arXiv:2103.00020）、SigLIP（arXiv:2303.15343）编号无误；站内 moonvit-v2 页确为"K3 从零训练的视觉编码器（27 层、RMSNorm、去 bias、分解注意力）"，与正文第 441 行一致；standard-attention 页 §5 给出 $O(n^2 d_k)$ 且 $d_k=64$，与正文 §3 构造示例引用一致。

## 问题

- [轻微·格式] head 内联 `<style>` 的 `.diagram { ... }`（第 51–60 行）：页面无任何元素使用 `.diagram`（正文流程图用 `.flow-diagram`），属死 CSS｜引文依据：不适用｜修复要求：删除 `.diagram` 规则｜修复：｜复验：
- [轻微·表述] 「来源与范围说明」前过渡段（第 463 行）："上述论断、公式与数字分别出自论文的具体章节，下面按类别登记对应位置与适用边界。"以"下面…登记"引出下文，属元话语｜引文依据：不适用｜修复要求：删除"下面按类别登记对应位置与适用边界"，改为直接陈述后续小节登记来源与边界，不用"下面"引导句｜修复：｜复验：
- [轻微·表述] 结尾段（第 504 行）："核心问题中的三项——tokenization 三步、$z_0$ 与编码块公式、数据规模边界——均已作答。"为对页面自身作答覆盖情况的自指陈述（元话语）｜引文依据：不适用｜修复要求：删除"均已作答"这一覆盖声明，或不以页面自身为主语，改为对内容的直接总结｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（无阻断与重要问题；关闭上述 3 处轻微后即可发布。核心数字、公式与来源论断本轮全部回源核对通过，未发现事实错误、算错或同页自相矛盾。）