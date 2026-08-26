# evidence.md：CLIP 核心论断与证据

来源固定：Radford, Jong, Wu, Kim. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021, arXiv:2103.00020。核对途径：ar5iv HTML 全文（https://ar5iv.labs.arxiv.org/html/2103.00020），2026-08-26 抓取核对。SigLIP 侧交叉引用：Zhai et al. 2023, arXiv:2303.15343v4。

## C 论断

- C1（CLIP 用"预测哪段文本配哪张图"的对比任务做预训练，4 亿网络图文对）：摘要 "predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet"。置信：已确认
- C2（联合训练图像编码器与文本编码器，最大化 N 个正对余弦相似度、最小化 N²−N 个负对相似度，优化对称交叉熵）：§2.3 "CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N²−N incorrect pairings. We optimize a symmetric cross entropy loss over these similarity scores."。置信：已确认
- C3（zero-shot 机制：文本编码器按类别名/描述合成线性分类器，与图像嵌入比余弦相似度后 softmax 取最大）：Figure 1 说明 "At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset's classes." + §3.1.2 "predict the most probable (image, text) pair according to CLIP... normalized into a probability distribution via a softmax."。置信：已确认
- C4（文本编码器是 hypernetwork——根据文本生成线性分类器权重）：§3.1.2 "the text encoder is a hypernetwork... which generates the weights of a linear classifier based on the text specifying the visual concepts that the classes represent."。置信：已确认
- C5（prompt 模板消歧：类别名单字缺上下文，训练文本多为完整句子；"A photo of a {label}." 为默认模板，单此模板在 ImageNet 提升 1.3%；80 prompt ensemble 再提升 3.5%，合计近 5%）：§3.1.4 "A common issue is polysemy..."、 "we found that using the prompt template 'A photo of a {label}.' to be a good default... just using this prompt improves accuracy on ImageNet by 1.3%."、 "On ImageNet, we ensemble 80 different context prompts and this improves performance by an additional 3.5% over the single default prompt... prompt engineering and ensembling improve ImageNet accuracy by almost 5%."。置信：已确认
- C6（图像编码器两个系列：修改版 ResNet（注意力池化等改动）与 ViT；投影到共享空间只用线性投影、无非线性投影头）：§2.4 "We consider two different architectures for the image encoder."、ResNet-D 改动与 attention pooling 原文、"We do not use the non-linear projection between the representation and the contrastive embedding space... We instead use only a linear projection"。置信：已确认
- C7（文本编码器为 63M 参数 12 层 512 宽 8 头 Transformer，取 [EOS] 位置激活经 LayerNorm 后线性投影进共享空间）：§2.4 "As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads... the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text which is layer normalized and then linearly projected into the multi-modal embedding space."。置信：已确认
- C8（温度 τ 以对数参数化乘性标量直接优化；初始化等效 0.07，截断防止 logits 放大超过 100 以防训练不稳定）：§2.3 "the temperature parameter... is directly optimized during training as a log-parameterized multiplicative scalar" + §2.5 "The learnable temperature parameter τ was initialized to the equivalent of 0.07 from (Wu et al. 2018) and clipped to prevent scaling the logits by more than 100 which we found necessary to prevent training instability."。置信：已确认
- C9（CLIP 损失与 InfoNCE 的关系：该 batch 构造与目标先以 multi-class N-pair loss 提出、经 InfoNCE 推广、由 Zhang et al. 2020 适配到图文对比）：§2.3 "this batch construction technique and objective was first introduced... as the multi-class N-pair loss [Sohn 2016], was popularized... by [Oord et al. 2018] as the InfoNCE loss, and was recently adapted for contrastive (text, image) representation learning... by [Zhang et al. 2020]."。置信：已确认
- C10（softmax 分母依赖整个 batch 是 SigLIP 更换损失的动机；CLIP 训练在多设备上需聚合全部嵌入）：SigLIP 论文 §3.1/§3.3（CLIP softmax 分母需对 batch 求和、chunked 实现避免 all-gather）与 SigLIP 页面 C2/C3 一致。置信：已确认（交叉来源一致）

## F 公式

- F1（对称 softmax 对比损失）：$\mathcal{L}_{\text{CLIP}}=-\frac{1}{2N}\sum_{i=1}^{N}\left(\log\frac{e^{t\,x_i\cdot y_i}}{\sum_{j=1}^{N}e^{t\,x_i\cdot y_j}}+\log\frac{e^{t\,x_i\cdot y_i}}{\sum_{j=1}^{N}e^{t\,x_j\cdot y_i}}\right)$。来源：§2.3 对称交叉熵文字描述 + Figure 3 伪代码（logits = 相似度矩阵 × exp(t)；loss = 两个方向交叉熵的平均）。ar5iv 版本未渲染 Eq.(1) 原式，依据文字描述 + 伪代码 + SigLIP 论文 §3.1 给出的 CLIP 损失形式三方一致还原。置信：已确认（三方一致）
- F2（温度缩放：$t=1/\tau$，τ 可学习对数参数化）：§2.3 + §2.5。置信：已确认
- F3（余弦相似度 $x_i\cdot y_j\in[-1,1]$，嵌入 L2 归一化）：§2.3 maximize cosine similarity；SigLIP 页 F4 同一定义。置信：已确认

## N 数字

- N1：WIT 数据集 400M (image, text) 对；查询 500,000 个、每查询至多 20,000 对做类平衡。来源：摘要 + §2 数据集构建。置信：已确认
- N2：batch size 32,768。来源：§2.5 训练配置 "We use a very large minibatch size of 32,768."。置信：已确认
- N3：ViT-L/14@336px zero-shot ImageNet top-1 76.2%、top-5 95%（匹配 Inception-V4），不用 ImageNet 128 万训练样本、与原始 ResNet-50 持平。来源：§3.1.3 + Table 1；默认报告模型为 ViT-L/14@336px 的依据为 §2.5 末段 "Unless otherwise specified, all results reported in this paper as 'CLIP' use this model"。置信：已确认
- N4：27 个数据集中 zero-shot CLIP 在 16 个上胜过 ResNet-50 特征上的全监督线性分类器（含 ImageNet）；ImageNet 上 zero-shot CLIP 匹配 16-shot 线性分类器。来源：§3.1.5 + Figure 5/7。置信：已确认
- N5：prompt 单模板 +1.3%、80 prompt ensemble +3.5%、合计近 5%。来源：§3.1.4。置信：已确认
- N6：训练算力：RN50x64 在 592 块 V100 上 18 天，最大 ViT 在 256 块 V100 上 12 天；共训练 5 个 ResNet 与 3 个 ViT、32 epochs。来源：§2.5。置信：已确认
- N7：文本编码器 base 规模 63M 参数 / 12 层 / 512 宽 / 8 头 / 49,152 词表 / 76 最大序列长度。来源：§2.4。置信：已确认

## 原图候选

不使用原图。双塔结构与相似度矩阵用 HTML 结构与表格表达即可，无需论文 Figure 1/3 的截图。
