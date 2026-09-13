<!-- review-meta
round: 5
page: wiki/siglip/index.html
reviewed_content_sha256: fbeceb1803e1e740
-->
# SigLIP审查记录（第 5 轮）

- 页面版本：d17a24c1b91a06363f6c8608ce34f18fda247502（工作树，index.html 未提交改动）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. CLIP softmax 损失的"全局耦合"问题——为什么需要换掉它 / 2. SigLIP 损失公式——逐对独立的二分类 / 3. 可学习温度 $t$ 与可学习 bias $b$——为什么 $b$ 是关键 / 4. batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界 / 来源与范围说明（含全部 details 折叠块与图注）
- 来源核对：Zhai, Mustafa, Kolesnikov, Beyer, "Sigmoid Loss for Language Image Pre-Training", ICCV 2023 Oral, arXiv:2303.15343v4（2023-09-27，PDF 全文与图 1/2/3/6、Table 1/2/5、§1、§3.1、§3.2、§3.3、§4.1–§4.5、§4.8、§4.9）；页码以 v4 PDF 为准。
- 已核对通过项（不再重复列出）：摘要/§1 引言/§4.1 的 sigmoid 与 softmax 在 batch < 16k 的对比表述逐字一致；Algorithm 1（logits = dot*t + b，labels = 2*eye−ones，l = −sum(log_sigmoid(labels*logits))/n）与页面 F1 右侧形式一致；Table 1 全部行（79.8/84.5/71.0/72.1/73.4）、Table 2 全部数值（INet-0 71.6/73.2/73.2/73.2/73.1，XM avg 34.8/34.9/34.4/33.6/32.7，XM zh 30.7/32.5/32.0/30.6/23.7）逐格吻合；§4.2 末段"softmax 需 98k 才达最优且仍未超过 sigmoid"定位正确（v4 PDF 第 5 页右栏，§4.2 末）；§4.4 引文"20 k batch size on four chips for 107 k steps in under two days"与 84.5% 一致；§4.5 引文"72.1% … approx. 2500 TPUv3-days for 72.6% reported in [30]"与 [30]=Li et al.(FLIP) 一致；§3.1 末段 $t=\exp(t')$、$x_i=f(I_i)/\|f(I_i)\|_2$ 与 [F4][F5] 一致；§3.2 第 3 段 bias 引文与初始化 $t'=\log10$、$b=-10$ 一致；正负比例表 32767、999999 及 1.07e9、1e12 数值自洽；`.dojo/scripts/validate.py wiki/siglip/index.html` 返回 validation ok；页面引用的 clip / vit / standard-attention / moonvit-v2 四个前置概念页均真实存在，无"（待生成）"占位、无 research/ 路径引用。

## 问题

- [阻断·技术] 第 292 行（§3 折叠块"展开：$|B|=4$、$t=10$、$b=-10$、初始 $x_i\cdot y_j=0$ 的完整损失手算"）：手算给出的 $\mathcal{L}_{\text{SigLIP}}\approx -10.0002$ 与同页 F1 公式相差一个符号。F1（第 190 行）为 $\mathcal{L}_{\text{SigLIP}}=-\frac{1}{|B|}\sum_{i,j}\log\sigma(z_{ij}(t\,x_i\cdot y_j+b))$，带前导负号；折叠块各对"损失"取的是 $\log\sigma(\cdot)\le 0$（正对 $\log\sigma(-10)\approx-10.0000454$、负对 $\log\sigma(10)\approx-0.0000454$），未归一化总和 $-40.0007$，故 $\mathcal{L}=-\frac{1}{4}\times(-40.0007)=+10.0002$，而页面写作 $-10.0002$（负的损失不可能）。数字复核：$-\frac{1}{4}(4\log\sigma(-10)+12\log\sigma(10))=+10.000181$。同块"对照：若 $b=0$"同样写反：$16\times\log 0.5=-11.09$，按公式应得 $+2.77$，页面写 $-2.77$。｜引文依据：F1（第 190 行）"$-\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\sigma(z_{ij}(t\,x_i\cdot y_j+b))$"；折叠块第 292 行"未归一化总损失 $=-40.0002+(-0.0005)\approx -40.0007$。按公式除以 $|B|=4$ 得 $\mathcal{L}_{\text{SigLIP}}\approx -10.0002$"（与公式前导负号矛盾）。｜修复要求：把两处损失值改为正号（$\mathcal{L}_{\text{SigLIP}}\approx +10.0002$、$b=0$ 时 $\approx +2.77$），或明确区分"逐对项 $\log\sigma(\cdot)$（$\le 0$）"与"总损失 $-\frac{1}{|B|}\sum$（$\ge 0$）"，使代入公式后的数值与标注的 $\mathcal{L}_{\text{SigLIP}}$ 符号一致。｜修复：｜复验：

- [重要·技术] 第 297 行（§3 末段）与第 464 行（"辅助解释与类比边界"）：称论文 §4.8 图 6 中可学习 bias $b$ 的最终值"范围约 $-15$ 到 $+5$"。图 6 的"Learned bias"子图纵轴刻度为 0/−5/−10/−15，四条曲线（Random/Hard/Hard, matched pairs/Easy）最高仅约 $-2$；"$-15$ 到 $+5$"的上界 $+5$ 实际取自同图第三个子图"Average logit of pos and neg"（纵轴 5/0/−5/−10/−15/−20），与 $b$ 不是同一量。｜引文依据：v4 PDF 第 7 页图 6，"Learned bias"子图纵轴 0, −5, −10, −15；"Average logit of pos and neg"子图纵轴 5, 0, −5, −10, −15, −20；图注"the final value of the learned bias, and the average logits of positive and negative pairs"。｜修复要求：把 $b$ 最终值范围改为图 6 的"Learned bias"子图读数（约 $-15$ 到 $-2$，或"约 $-15$ 至 0 附近"），不得用 average-logit 的 $+5$ 作 $b$ 的上界；两处同步。｜修复：｜复验：

- [重要·技术] 第 134 行（§1）与第 175 行（§1 本章问题解答）："CLIP 论文与对比学习综述（Oord et al. 2018 InfoNCE）都把这一性质视为对比学习的核心"（第 175 行表述为"CLIP 论文与 InfoNCE 综述都把这视为对比学习的核心性质"）。该归因在页面与来源章节均无对应的原文片段与定位；§2.2 所列来源论断的核对要求"无法给出片段的条目视为未核对"，此条属未核对的来源归因（对两个被点名来源的立场判断）。｜引文依据：未定位到（页面正文、[C2]（仅引 Zhai 2023 §3.1 与 Radford et al. 2021 原文，用于"分母需对 batch 求和"）及"简化条件及其限制/简化一"均未给出支持"CLIP 论文与 InfoNCE 都把 hardest-negative 相对优势视为核心"的片段）。｜修复要求：删除该归因句，或降级为明确标注的推断（如"一种常见解释是……，页面未给出该两篇原文的直接表述"），或补上可定位的引文片段（来源、章节/页、原句）。｜修复：｜复验：

- [轻微·表述] 第 283 行（§3 第 4 段段首）："更准确地说：sigmoid 损失对 logit $a=z_{ij}(t\,x_i\cdot y_j+b)$ 的梯度是……"。该引导语属元话语式过渡——上一段（第 281 行）已给出"初始被负对主导、方向错误"的结论，本段只是补上梯度公式，删去"更准确地说："不影响衔接，反而更直接。｜引文依据：不适用。｜修复要求：去掉"更准确地说："，直接以"对 logit $a=z_{ij}(t\,x_i\cdot y_j+b)$，sigmoid 损失的梯度是 $\partial\log\sigma(a)/\partial a=\sigma(-a)$……"起句。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 1
- 处置：修复（阻断项与两个重要项需在本轮修复并复验；修复后重跑 `.dojo/scripts/validate.py`。第 1 项修改数值后须按 F1 重新核对；第 2、3 项涉及来源定位，修复后需重新回源。）