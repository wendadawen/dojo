<!-- review-meta
round: 4
page: wiki/siglip/index.html
reviewed_content_sha256: 0f965fc65d6b22d5
-->
# SigLIP 审查记录（第 4 轮）

- 页面版本：458663f926408a67bf87efcbc210346ae3fdecc4
- 审查时间：2026-09-13 19:50
- 审查者：独立子代理
- 已完整阅读章节：核心问题、最容易误解、1. CLIP softmax 损失的"全局耦合"问题——为什么需要换掉它、2. SigLIP 损失公式——逐对独立的二分类、3. 可学习温度 $t$ 与可学习 bias $b$——为什么 $b$ 是关键、4. batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界、来源与范围说明

## 核对范围与结果

来源：Zhai, Mustafa, Kolesnikov, Beyer, "Sigmoid Loss for Language Image Pre-Training", ICCV 2023 Oral, arXiv:2303.15343v4（正文 PDF 已下载逐页核对，非只看编号）。

已回源确认一致的条目（引文依据均为论文原文）：
- 摘要原文："the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization"；"we push the batch size to the extreme, up to one million, and find that the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient"。
- §1："We find that the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16k. As the train batch size grows, the gap closes."；"it conceptually decouples the batch size from the definition of the task"；"it is usually stabilized by subtracting the maximum input value before applying the softmax"。
- §3.1 公式与符号：$-\frac{1}{2|\mathcal{B}|}\sum_{i}\left(\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_i\cdot y_j}}+\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_j\cdot y_i}}\right)$；$x_i=f(I_i)/\|f(I_i)\|_2$；"The scalar $t$ is parametrized as $\exp(t')$"。页面 F2/F4/F5 与之一致。
- §3.2 公式：$-\frac{1}{|\mathcal{B}|}\sum_{i}\sum_{j}\log\frac{1}{1+e^{z_{ij}(-t x_i\cdot y_j-b)}}$；"At initialization, the heavy imbalance coming from the many negatives dominates the loss… we introduce an additional learnable bias term $b$… We initialize $t'$ and $b$ to $\log 10$ and $-10$ respectively."；Algorithm 1 用 `labels = 2*eye(n)-ones(n)`、`l = -sum(log_sigmoid(labels*logits))/n`。页面 F1/F6 与算法描述一致。
- §3.3 与图 1："the memory cost at any given moment is reduced from $|\mathcal{B}|^2$ to $b^2$"、device 间 swap 文本嵌入 + cross-device sum。页面 chunked 折叠块一致。
- §4.2："SigLIP performs best at batch size 32k, whereas the softmax loss required 98k for optimal performance and still didn't outperform the sigmoid based variant."
- §4.4："we can train at 20k batch size on four chips for 107k steps in under two days. This further pushes the 0-shot ImageNet classification accuracy up to 84.5%"；文本塔"a depth of only 12 layers (instead of 24)"。
- §4.5："SigLIP achieves 72.1% 0-shot accuracy. This presents a significant training cost reduction e.g. compared to CLIP (approx. 2500 TPUv3-days for 72.6%) reported in [30]"；[30] = Li et al., "Scaling language-image pre-training via masking"（FLIP），与页面标注一致。
- Table 1：SigLiT B/8 L 32k/4/1 → 79.8；SigLiT g/14 L 20k/4/2 → 84.5；SigLIP B/16 B 16k/16/3 → 71.0、32k/32/2 → 72.1、32k/32/5 → 73.4。页面两张表逐格一致。
- Table 2：INet-0 71.6/73.2/73.2/73.2/73.1；XM avg 34.8/34.9/34.4/33.6/32.7；XM de 54.7/54.8/55.4/54.3/54.7；XM en 46.5/46.2/46.5/46.6/46.6；XM zh 30.7/32.5/32.0/30.6/23.7。页面 Table 2 逐格一致。
- 图 6（§4.8 引用）：报告"the final value of the learned bias"，Learned-bias 面板纵轴约 −15～+5，页面"范围约 −15 到 +5"与之相符。
- 构造示例手算复算：$\sigma(-10)=4.5398\times10^{-5}$、$\log\sigma(-10)\approx-10.0000454$，4 正对 ≈ −40.0002；12 负对 ≈ −0.0005；合计 ≈ −40.0007，除以 $|B|=4$ 得 ≈ −10.0002；$b=0$ 时 16×(−0.6931)/4 = −2.77。全部与页面数字一致。
- 正负比例：$|B|=32\text{k}$ → 1:32767、负对 ≈1.07×10⁹；$|B|=1\text{M}$ → 1:999999、负对 ≈10¹²。与 $|B|^2-|B|$ 复算一致。
- 机械项：`python3 .dojo/scripts/validate.py wiki/siglip/index.html` 返回 `validation ok`（exit 0）；无 research/ 路径引用、无"（待生成）"；前置页 clip / vit / standard-attention / moonvit-v2 均存在；overview.html 与 index.html 相互链接。

## 问题

- [重要·技术] 第 1 章「本章问题」第 3 题解答（index.html:190）：把"小 batch 下 softmax 的 hardest negative 池太小"这一未获论文归因的推断，写成"论文 §1 引言显示 sigmoid 在 batch < 16k 时显著优于 softmax 正源于此"，与本页第 162 行自述"（此为合理推断，论文未作此归因）"自相矛盾——同一页内两处对同一机制给出相反的来源归属。｜引文依据：§1 原文仅给出观察 "We find that the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16k. As the train batch size grows, the gap closes."，全段无 hardest-negative 归因；§4.2 也只有 "When the batch size is smaller than 16 k, sigmoid loss outperforms softmax loss by a large margin"，未解释原因。｜修复要求：删去该句中的来源归属，改写为明确标注的推断（如"一种可能的解释是……，论文未作此归因"），与第 162 行标注一致。｜修复：｜复验：
- [轻微·技术] 「论断与来源（C）」[C3]（index.html:447）：引文 "operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization" 被标注为"§3.2 第 1 段"，但该句在全文仅出现于摘要。｜引文依据：该短语全文仅 1 处（摘要："the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization"）；§3.2 第 1 段原文为 "The sigmoid-based loss processes every image-text pair independently"。｜修复要求：将引文出处改为"摘要"；该论断的实质仍由 §3.2 第 1 段 "processes every image-text pair independently" 与 §3.3 支持，无需改动论断本身。｜修复：｜复验：
- [轻微·表述] 第 4 章 Table 2 图注（index.html:382）："注意 ImageNet zero-shot 在 32k 后几乎不再提升……"用"注意"对读者下指令，属元话语。｜引文依据：不适用。｜修复要求：删去"注意"，直接陈述事实（"ImageNet zero-shot 在 32k 后几乎不再提升（73.2 → 73.1），但多语言 retrieval 反而下降……"）。｜修复：｜复验：
- [轻微·技术] 第 4 章「适用边界」表（index.html:405）："SigLIP 不需 all-gather｜单设备或 chunked 实现｜严格最优 retrieval 性能仍可能依赖大 batch + 多设备"一行无来源支持，且与论文 §4.3 方向相反。｜引文依据：§4.3 原文 "we didn't observe clear improvements with a batch size larger than 32k. A batch size of 32k is sufficient…" 与 "going beyond 32k batch size leads to worse results on average while on ImageNet zero-shot transfer it stays flat"——论文并未支持"最优 retrieval 依赖更大 batch"。｜修复要求：删除该行，或改为论文支持的边界（32k batch 已达 state-of-the-art retrieval，更大 batch 反而下降）。｜修复：｜复验：
- [轻微·技术] 第 3 章 callout（index.html:312）："博客介绍时常省略 $b$，只写 $-\log\sigma(z_{ij}\cdot t\,x_i\cdot y_j)$……"是关于外部内容的一般性断言，无来源支持却写成事实。｜引文依据：不适用。｜修复要求：改为条件式表述（如"只写 $-\log\sigma(z_{ij}\cdot t\,x_i\cdot y_j)$ 的写法会让人误以为直接替换 sigmoid 即可"），删去对"博客"的概括断语。｜修复：｜复验：
- [轻微·格式] head 内联 `<style>`（index.html:39-48）定义了 `.diagram` 类，但全页正文无任何元素使用该类（`grep 'class="diagram"' index.html` 无命中），为死 CSS。｜引文依据：不适用。｜修复要求：删除未使用的 `.diagram` 规则（`.pair-matrix` 规则仍在用，保留）。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复。无阻断问题；数字、公式、实验条件均已逐条回源核对并通过复算；重要 1 条（来源归属与页内自述矛盾）与轻微 5 条修复后复验。