<!-- review-meta
round: 6
page: wiki/siglip/index.html
reviewed_content_sha256: c937004ae65570bf
-->
# SigLIP 概念页审查记录（第 6 轮）

- 页面版本：4d75c819eeab6092777ded8d13298b4111babb84（index.html 工作树）
- 审查时间：2026-09-13 21:21
- 审查者：独立子代理（未参与写作与前序轮次）
- 规范：guides/concept/check.md（dojo:type=concept）
- 来源获取：arXiv:2303.15343v4 原文 PDF（下载并逐页提取正文/公式/图），另对照 overview.html 与本页链接指向的概念页
- 已完整阅读章节：核心问题（页面级问答块）、最容易误解、1. CLIP softmax 损失的"全局耦合"问题——为什么需要换掉它（含本章问题）、2. SigLIP 损失公式——逐对独立的二分类（含本章问题）、3. 可学习温度 t 与可学习 bias b——为什么 b 是关键（含本章问题）、4. batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界（含本章问题）、来源与范围说明（C/F/N/构造示例/类比边界/简化条件）

## 核对结论摘要（无问题的部分）

- 公式 [F1]：[论文 §3.2] `-\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\frac{1}{1+e^{z_{ij}(-t x_i·y_j-b)}}` 与 Algorithm 1（`logits=dot(zimg,ztxt.T)*t+b`、`labels=2*eye(n)-ones(n)`、`l=-sum(log_sigmoid(labels*logits))/n`）逐项一致；[F2]：与 §3.1 公式 `-\frac{1}{2|B|}\sum_{i=1}^{|B|}(\log\frac{e^{t x_i·y_i}}{\sum_j e^{t x_i·y_j}}+\log\frac{e^{t x_i·y_i}}{\sum_j e^{t x_j·y_i}})` 一致。
- 张量表：Table 1（SigLiT B/8 L 32k/4/1/79.8、SigLiT g/14 L 20k/4/2/84.5、SigLIP B/16 B 16k/16/3/71.0、32k/32/2/72.1、32k/32/5/73.4）与 Table 2（16k/32k/64k/128k/240k → INet-0 71.6/73.2/73.2/73.2/73.1，XM avg 34.8/34.9/34.4/33.6/32.7，XM zh 30.7/32.5/32.0/30.6/23.7）逐格一致。
- 引文：摘要 1M 句、§1 "performs significantly better … smaller than 16 k"、§4.1 "outperforms softmax loss by a large margin"、§4.2 "the softmax loss required 98 k for optimal"、§4.4 "20 k batch size on four chips for 107 k steps in under two days … up to 84.5%"、§4.5 "72.1% … approx. 2500 TPUv3-days for 72.6% reported in [30]"、§3.1 末段 t=exp(t')、§3.2 初始 t'=log 10 / b=-10，均逐字核对无误。
- 手算：σ(−10)≈4.54e-5、logσ(−10)≈−10.0000454、4×(…)=−40.0002、12×(−4.54e-5)=−0.0005、L=+10.0002；b=0 时 16×(−0.6931)=−11.09、L=+2.77；梯度 ∂logσ(a)/∂a=σ(−a) 及正负比例 1:32767 / 1:999999、负对数 1.07e9 / 1e12 均复算通过。
- 机械项：无 `<img>` 正文图（lightbox alt 为空），无 `$…$` alt；无 我们/你/本页/元话语；validate.py 返回 `validation ok`；dojo:topics=多模态 在允许词表内；链接 clip/vit/standard-attention/moonvit-v2 目录均真实存在；overview.html 与 index.html 互链。

## 问题

- [重要·技术] 第 3 章"可学习温度 t 与可学习 bias b"第四段（"若没有 b 会怎样"）与第六段（梯度段）、该章末 callout、以及"最容易误解"第 3 条：页面把"没有 b 时初始梯度把包括正对在内的所有对都推向更负、与训练目标相反"当作论文结论陈述，并写明"论文 §3.2 明确指出…使训练初期产生大量错误方向梯度"｜引文依据：论文 §3.2 原话为 "At initialization, the heavy imbalance coming from the many negatives dominates the loss, leading to large initial optimization steps attempting to correct this bias. … This makes sure the training starts roughly close to the prior and does not require massive over-correction."；§4.9 为 "the bias term ensures that the training starts close to the prior, preventing dramatic over-correction in early optimization"。论文讲的是"失衡使损失被占多数的负对主导 → 初始更新步幅过大（过校正）"，并未说初始梯度方向与目标相反、把正对也压低。页面自身在梯度段先写"正对 z=+1 与负对 z=−1 的梯度绝对值相同但方向相反"（即正对梯度朝提高正对 logit），紧接着又写"模型整体被推向'所有对都更负'，包括正对"，两句自相矛盾；对各对独立 logit 而言也不存在一个把正对也压低的总方向（对 ℓ=x·y≈0、b=0，逐对梯度 ∂L/∂ℓ=−z·σ(−zℓ)，正对梯度朝正）。以随机初始化嵌入做一次梯度步实测（|B|=4/16/64、t=10、b=0）正对平均余弦相似度上升而非下降，负对下降｜修复要求：把"没有 b → 初始梯度方向错误 / 把所有对（包括正对）都推向更负 / 与训练目标相反"整段机制改为与 §3.2、§4.9 一致的表述（失衡使损失被负对主导 → 初始优化步幅过大、需要过校正；b=−10 使训练"starts close to the prior"，即初始 σ(b)≈1/(|B|−1) 与正负样本比例匹配，从而不需过校正），删除"包括正对"这一无来源支持的断言；把"论文 §3.2 明确指出…大量错误方向梯度"改为与 §3.2 原话一致的措辞。若保留"方向错误"的解读，须显式标注为本页推断，不得与 §3.2 原话混同｜修复：｜复验：
- [轻微·技术] 第 3 章末段（"b 是可学习的"）与"来源与范围说明·辅助解释与类比边界"：写 b 最终值"范围约 −15 到 −2"｜引文依据：论文图 6"Learned bias"面板纵轴刻度为 0/−5/−10/−15，四条曲线（Random / Hard / Hard, matched pairs / Easy）数据点落在约 −13（1:16k 处最低）至约 −3（1:1.6 处最高），图 6 正文仅定性说 "as fewer negatives are present, the bias and logits become more positive overall"，未给数值范围；故"−15 到 −2"两端均与图不符｜修复要求：把"范围约 −15 到 −2"改为与图 6 读数一致的"约 −13 到 −3"，或删除具体数值范围、只保留"b 最终值随负采样比例（1:16k → 1:1.6）单调变化"的定性结论｜修复：｜复验：
- [轻微·技术] `<head>` 的 `<meta name="description">`：写"支持更大 batch 和更好迁移"，而 `dojo:summary` 写"并保持良好的迁移表现"，正文第 4 章只支持"batch < 16 k 时 sigmoid 显著优于 softmax、batch 增大时优势缩小"｜引文依据：图 2 题注 "Sigmoid loss outperforms the softmax loss significantly with small batch sizes, and performs similarly at larger batch sizes."（大 batch 下两者相近，论文未主张 sigmoid 迁移更好）；本页 dojo:summary "该形式更容易扩展 batch，并保持良好的迁移表现"｜修复要求：把 description 的"和更好迁移"改为与 dojo:summary 一致的"并保持良好迁移表现"，或删除该分句，避免 description 比正文/summary 更强的无来源论断｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复