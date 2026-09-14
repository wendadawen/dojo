<!-- review-meta
round: 4
page: wiki/clip/index.html
reviewed_content_sha256: ea3b398e0b2dbde7
-->
# CLIP 审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 `2dc02e81e0ee650fa4e2b0fab305337d3045688b`（overview.html `f7c46e1a87b7bc1012c769892c9bea18335cc859`）
- 审查时间：2026-09-14 16:50
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复；未读取 research/ 下任何文件）
- 适用规范：`guides/concept/check.md`（head `dojo:type=concept`）
- 已完整阅读章节：核心问题（页面级 4 题）→ 最容易误解 → 1. 为什么用自然语言监督训练视觉模型——标注瓶颈与 zero-shot 需求 → 2. 双塔架构与共享嵌入空间——两个编码器怎么被拉到一起 → 3. 对称 softmax 对比损失——batch 就是分类题的选项 → 4. zero-shot 分类——文本编码器是生成分类器的网络 → 5. 训练配置与边界——softmax 损失的 batch 耦合及其他 → 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；并完整阅读 overview.html

## 来源核对依据（逐条回源，本轮核对到的原文片段）

- **C1/N1（400M 图文对、500k 查询、每查询至多 20k）**：Radford 2021 摘要「we constructed a new dataset of 400 million (image, text) pairs」；§2.2「a set of 500,000 queries」「up to 20,000 (image, text) pairs per query」「WIT for WebImageText」。核对一致。
- **C2/C9（对称交叉熵与损失谱系）**：§2.3「To our knowledge this batch construction technique and objective was first introduced in the area of deep metric learning」「the multi-class N-pair loss」「popularized for contrastive representation learning by Oord et al. 2018 as the InfoNCE loss」「recently adapted for contrastive (text, image) representation learning in the domain of medical imaging by Zhang et al. 2020」。页面所引「To our knowledge」原句存在，谱系表述一致。
- **C3/C4（zero-shot 合成分类器与 hypernetwork）**：§3.1.2「the text encoder is a hypernetwork」「generates the weights of a linear classifier based on the text」「We first compute the feature embedding of the image and the feature embedding of the set of possible texts」「we cache the zero-shot classifier once it has been computed by the text encoder」「reuse it for all subsequent predictions」。页面「文本嵌入可离线预计算并缓存」有原文支持。
- **C5/N5（prompt 提升）**：§3.1.4「just using this prompt improves accuracy on ImageNet by 1.3%」「we ensemble 80 different context prompts and this improves performance by an additional 3.5%」。1.3+3.5=4.8，「近 5 个百分点」成立。
- **C6/C7/N7（架构）**：§2.3「We instead use only a linear projection to map from each encoder's representation to the multi-modal embedding space.」「We do not use the non-linear projection between the representation and the contrastive embedding space」；§2.4「the ResNet-D improvements」「the antialiased rect-2 blur pooling」「We also replace the global average pooling layer with an attention pooling mechanism.」「adding an additional layer normalization to the combined patch and position embeddings」「a slightly different initialization scheme」「a 63M-parameter 12-layer 512-wide model with 8 attention heads」「the activations of the highest layer of the transformer at the [EOS] token」「49,152 vocab size」。逐项一致。
- **C8/F2/N2/N6（训练配置）**：§2.5「We train a series of 5 ResNets and 3 Vision Transformers.」「The learnable temperature parameter τ was initialized to the equivalent of 0.07 ... and clipped to prevent scaling the logits by more than 100」「We use a very large minibatch size of 32,768.」「The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs.」「For the ViT-L/14 we also pre-train at a higher 336 pixel resolution for one additional epoch」「Unless otherwise specified, all results reported in this paper as "CLIP" use this model」。一致（N3 的「默认模型依据 §2.5 末段」定位准确）。
- **N3（76.2% / 95%）**：§3.1.3（"Initial Comparison to Visual N-Grams"）「In Table 1 we compare Visual N-Grams to CLIP. The best CLIP model improves accuracy on ImageNet from a proof of concept 11.5% to 76.2% and matches the performance of the original ResNet-50 despite using none of the 1.28 million crowd-labeled training examples available for this dataset.」「this model has a 95% top-5 accuracy, matching Inception-V4」。章节号与 Table 1 定位均正确；页面把「含 80 模板 ensemble 收益」明确标为本页推断，处理得当。
- **N4（16/27 与细粒度、16-shot）**：§3.1.5「zero-shot CLIP outperforms this baseline slightly more often than not and wins on 16 of the 27 datasets.」「satellite image classification (EuroSAT and RESISC45)」「lymph node tumor detection (PatchCamelyon)」「counting objects in synthetic scenes (CLEVRCounts)」「German traffic sign recognition (GTSRB)」「zero-shot CLIP outperforms logistic regression on ResNet-50 features by over 20%」（Stanford Cars、Food101）「on two others, Flowers102 and FGVCAircraft, zero-shot CLIP underperforms by over 10%」「On ImageNet, zero-shot CLIP matches the performance of a 16-shot linear classifier trained on the same feature space.」。一致。
- **C10（SigLIP §3.3 / §4.1）**：SigLIP（arXiv:2303.15343v1）§3.3「Computing the loss when data is split across D devices necessitates gathering all embeddings with expensive all-gathers and, more importantly, the materialization of a memory-intensive |B|×|B| matrix of pairwise similarities.」；§4.1「When the batch size is smaller than 16 k, sigmoid loss outperforms softmax loss by a large margin.」章节号与引文均正确。
- **F1（损失公式三方核对）**：SigLIP §3.1 的 softmax 损失公式与页面 $\mathcal{L}_{\text{CLIP}}$ 完全一致（`-\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\left(\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_i\cdot y_j}}+\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_j\cdot y_i}}\right)`）。页面正文、第 3 章解答、核心问题第 2 条解答三处公式逐字一致。

## 数字复算（脚本核对）

- 四个负对数概率：$-\log\frac{e^8}{e^8+e^2}=0.002475685$、$-\log\frac{e^9}{e^1+e^9}=0.000335406$、$-\log\frac{e^8}{e^8+e^1}=0.000911466$、$-\log\frac{e^9}{e^2+e^9}=0.000911466$；均值 $0.001158506$，按 6 位小数四舍五入即页面所写 $0.001159$。页面给出的中间量 $e^8=2980.958$、$e^8+e^2=2988.347$、$0.997527$、$0.999665$、$0.999089$ 全部吻合。
- 负对改 $0.75$ 的对照：$\frac{e^8}{e^8+e^{7.5}}=0.622459$，$-\log\approx 0.474$，与页面一致。
- $\tau=0.07$ 对应 $t=1/0.07=14.286\approx 14.3$，页面「取整为 10」的构造说明成立。
- 页内一致性：`76.2%`、`top-5 95%`、`32768`、`0.07`、`截断 100`、`1.3%`、`3.5%`、`近 5 个百分点`、`4 亿`、`128 万`、`80 模板`、`16/27`、`16k` 在正文／summary／核心问题解答／本章问题解答／overview.html 之间取值一致，无互相矛盾。

## 机械核对

- `.dojo/scripts/validate.py wiki/clip/index.html` → `validation ok`。
- 前置概念链接 `wiki/vit/index.html`、`wiki/standard-attention/index.html`、`wiki/cross-entropy/index.html`、`wiki/siglip/index.html`、`wiki/moonvit-v2/index.html` 均实际存在；无「（待生成）」占位；index.html 与 overview.html 双向互链。
- `dojo:type=concept`、`dojo:topics=多模态`（在 AGENTS.md 固定大类内）、`dojo:tag=视觉与多模态`（在 catalog_builder.py `ALLOWED_TAGS` 内）、`description` 为纯文本、`dojo:summary` 中 `$4$`／`$N$`／`$1$` 的 `$` 成对且可 KaTeX 渲染。
- 全文（含标题、summary、列表、表格、图注）未出现 Unicode 数学符号（`×`、`≈`、`∈`、`·` 等一律写在 `$...$` 内；出现的 `—`、`→` 为正文破折号与「image→text」方向标签，非数学符号）；同一变量写法全页一致。
- 图为 HTML/CSS（`.dg-flow`）结构，无 SVG `<text>`、无等宽字符框线图；图内公式（$I$、$f$、$x_i$、$g$、$y_j$）均在 HTML 文本节点中由 KaTeX 渲染，图注已定义 $x_i$／$y_j$ 来源。
- 无 `<pre><code>` 可运行代码块，故第 3 项（代码执行核对）不适用；无 `<img>` 带 `$...$` 的 alt。
- 页面级「核心问题」4 题与 5 个章节的「本章问题」共 13 题均有解答折叠块，答案独立可读，且每条都指明完整论证所在章节。

## 问题

- [轻微·可读性] 第 3 章首段（`index.html` 第 213 行）：句末缺中心词，句子不完整。原文为「训练目标：最大化正对的余弦相似度、最小化负对的<sup>[C2]</sup>。」——「最小化负对的」后直接接引文编号与句号，中心词「相似度」缺失；同页另两处（核心问题第 1 条解答「最小化 $N^2-N$ 个负对的相似度」、最容易误解第 2 条「最小化 $N^2-N$ 个负对的相似度」）均完整。｜引文依据：不适用（表述问题；对照同页第 77、108 行两处平行表述）｜修复要求：把「最小化负对的」补为「最小化负对的相似度」，其余不变。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：修复（唯一轻微项补一词即可；核心结论、公式、数字与来源全部核对通过，无阻断或重要问题）
