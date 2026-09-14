<!-- review-meta
round: 8
page: wiki/siglip/index.html
reviewed_content_sha256: 4b522675442e7e13
-->
# SigLIP 审查记录（第 8 轮）

- 页面版本：62f331660a6b886e1e958c262015a19d86de2121（wiki/siglip/index.html 工作树哈希）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次审查）
- 已完整阅读章节（按顺序，含全部折叠块与图注）：核心问题、最容易误解、1. CLIP softmax 损失的"全局耦合"问题——为什么需要换掉它、2. SigLIP 损失公式——逐对独立的二分类、3. 可学习温度 $t$ 与可学习 bias $b$——为什么 $b$ 是关键、4. batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界、来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制）；并对照 overview.html 逐数字核对。
- 来源获取方式：arXiv:2303.15343v4 全文 PDF（pdftotext + PyMuPDF 逐页定位、图内数值像素测量）、页面自身与 overview.html。

## 问题

- [轻微·技术] 来源与范围说明 →「公式与来源（F）」[F1]（对应正文第 2 章"SigLIP 损失公式"）：页面把分数形式写作 $\log\frac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j-b)}}$（指数内 $-b$），而论文 §3.2 印刷式指数内是 $+b$，[F1] 却把"§3.2 公式"列为该形式的来源。核对确认：论文印刷式与其自身 Algorithm 1 不自洽（Algorithm 1 第 9–11 行为 `logits = dot(zimg, ztxt.T) * t + b`、`labels = 2 * eye(n) - ones(n)`、`l = -sum(log_sigmoid(labels * logits)) / n`，配合 §4.9 的 $b=-10$ 只有页面的 $-b$ 形式成立），页面的写法是对的，问题只在来源标注未说明这一差异｜引文依据：v4 PDF p.3 §3.2 式原样为 $\log\frac{1}{1+e^{z_{ij}(-t\mathbf{x}_i\cdot\mathbf{y}_j+b)}}$；Algorithm 1 第 9–11 行同上；§4.9"We set the bias and temperature initialization to b = −10 and t0 = log 10"｜修复要求：在 [F1] 与该式下方"等价形式"折叠块加一句注明——论文 §3.2 印刷式指数内为 $+b$、与论文 Algorithm 1 不一致，本页采用与 Algorithm 1 一致的 $-b$｜修复：｜复验：

- [轻微·技术] 来源与范围说明 →「外部数字与实验条件（N）」[N1] 及第 4 章 SigLiT 表注：页面把 §4.4 列为 SigLiT B/8 L 一行 79.8% 的来源，但 §4.4 正文（与论文摘要）写的是 79.7%，79.8% 仅见于 Table 1 行｜引文依据：§4.4 第 2 段"achieving 79.7% 0-shot ImageNet classification accuracy"；Table 1 行"SigLiT B/8 L\* 32 k 4 1 79.8"｜修复要求：把该数字的来源限定为 Table 1，或注明 Table 1（79.8%）与 §4.4/摘要（79.7%）在论文内部即不一致，页面取 Table 1 值｜修复：｜复验：

- [轻微·技术] 来源与范围说明 →「简化条件及其限制」简化三：声称"只引用 SigLiT 84.5% 与 SigLIP 72.1% 两个代表数字""不可推出 SigLiT B/8 L 79.8%、SigLIP 16k batch 71.0% 等中间配置"，与页面实际内容不符——第 4 章两张表分别列出 79.8%、84.5%、71.0%、72.1%、73.4% 与 mSigLIP 五档 batch（16k/32k/64k/128k/240k）全部数据，[N1][N2][N3] 也逐一列出｜引文依据：页面第 4 章"SigLiT 关键数字"表、"SigLIP from scratch 关键数字"表、"batch size 扫描"表；[N1][N2][N3]｜修复要求：改写简化三使其与页面实际引用一致（如"正文结论只依赖 84.5% 与 72.1% 两个代表数字；中间配置与 mSigLIP 全表数据见第 4 章表格，本页不作进一步推导"）｜修复：｜复验：

- [轻微·技术]「最容易误解」第 2 条："论文 §1 引言显示 sigmoid 显著优于 softmax 仅在 batch < 16k；batch 增大时优势缩小；batch > 32k 后多语言 retrieval 反而下降"——前两个分句出自 §1 引言，第三个分句（多语言 retrieval 下降）的出处是 §4.3 与 Table 2；§1 引言只说"a reasonable batch size, i.e. 32 k, is sufficient … This conclusion also holds for multilingual SigLIP training"，并未给出 retrieval 下降的结论｜引文依据：§1"the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16 k. As the train batch size grows, the gap closes."；§4.3"going beyond 32 k batch size leads to worse results on average"；Table 2 XM avg 34.9（32k）→ 32.7（240k）｜修复要求：把第三个分句的来源改为 §4.3 / Table 2，与第 4 章"三点结论"处的标注保持一致｜修复：｜复验：

## 结论

- 逐条回源核对结果（本轮已核对项）：论文 Table 1 五行配置（SigLiT B/8 L\* 32k/4/1=79.8、SigLiT g/14 L 20k/4/2=84.5、SigLIP B/16 B 16k/16/3=71.0、32k/32/2=72.1、32k/32/5=73.4）与页面两张表逐一相同（像素级裁图复核）；Table 2 五档 batch 的 INet-0（71.6/73.2/73.2/73.2/73.1）、XM avg（34.8/34.9/34.4/33.6/32.7）、XM de、XM en、XM zh（30.7/32.5/32.0/30.6/23.7）与页面表格逐一相同；§4.5 末段 72.1% 对照 CLIP（约 2500 TPUv3-days 达 72.6%）与引用 [30]（Yanghao Li et al., FLIP，= 论文文献表 [30]）无误；§3.2 第 3 段 $t'=\log 10$、$b=-10$、§1 引言"conceptually decouples the batch size from the definition of the task"、§1"usually stabilized by subtracting the maximum input value"、§3.2"the heavy imbalance coming from the many negatives dominates the loss"、§4.9"the bias term ensures that the training starts close to the prior, preventing dramatic over-correction in early optimization"、§4.2 末段"the softmax loss required 98 k for optimal performance and still didn't outperform the sigmoid based variant"、§4.3"going beyond 32 k"，均与页面引文一致；Figure 6"Learned bias"面板按像素测量（0 线 y≈180px，1 单位≈64.95px）读数区间约 $-12.8$~$-3.4$，与页面"范围约 $-13$ 到 $-3$"相符；$|B|=4$ 手算可复算（$4\times(-10.0000454)+12\times(-0.0000454)\approx-40.0007$，$\div4=+10.0002$；$b=0$ 时 $16\times(-0.6931)=-11.09$，$\div4=+2.77$）；正负比例表（1:3 / 1:32767 / 1:999999）与边界表条件均与来源一致；数学符号全部由 KaTeX 渲染、无 Unicode 数学字符直出、$ 定界符平衡；overview.html 与 index.html 双向链接、四个前置概念页（clip / vit / standard-attention / moonvit-v2）真实存在、K3 与 MoonViT-V2 的衔接表述与 MoonViT-V2 页不矛盾；`python3 .dojo/scripts/validate.py wiki/siglip/index.html` 返回 validation ok；通读全文未发现元话语（无"本页将/下面来看/需要注意的是"）、会话指代（无我/我们/你）、调试叙事、临场评价或公文腔。
- 处置：可发布（0 阻断 / 0 重要；上述 4 条轻微建议随下一轮修复，或注明接受理由）
- 统计：阻断 0 / 重要 0 / 轻微 4