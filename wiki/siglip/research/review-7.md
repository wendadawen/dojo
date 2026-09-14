<!-- review-meta
round: 7
page: wiki/siglip/index.html
reviewed_content_sha256: 50ec6b843d606faa
-->
# SigLIP审查记录（第 7 轮）

- 页面版本：b4ba671e40c6552b914e01b6a19029fb6ae1ad0d
- 审查时间：2026-09-13 21:53
- 审查者：独立子代理（第 7 轮，未参与写作与前序审查）
- 已完整阅读章节：核心问题、最容易误解、1. CLIP softmax 损失的「全局耦合」问题——为什么需要换掉它、2. SigLIP 损失公式——逐对独立的二分类、3. 可学习温度 $t$ 与可学习 bias $b$——为什么 $b$ 是关键、4. batch size 解耦与实验边界——SigLIP 在小算力下的优势与边界、来源与范围说明（含全部折叠块与图注）

## 回源核对依据

来源：Zhai, Mustafa, Kolesnikov, Beyer. "Sigmoid Loss for Language Image Pre-Training." arXiv:2303.15343v4（ICCV'23 Oral，2023-09-27）。逐项对照结果：

- F2 CLIP 损失：§3.1 公式 $-\frac{1}{2|B|}\sum_{i}\left(\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_i\cdot y_j}}+\log\frac{e^{t x_i\cdot y_i}}{\sum_j e^{t x_j\cdot y_i}}\right)$ 与页面一致；$x_i=f(I_i)/\|f(I_i)\|_2$、$y_j=g(T_j)/\|g(T_j)\|_2$（§3.1 末段）、$t=\exp(t_0)$（同段）核对一致。
- F1 SigLIP 损失：§3.2 公式与 Algorithm 1（第 9 行 logits = dot(zimg, ztxt.T) * t + b；第 10 行 labels = 2*eye(n)-ones(n)；第 11 行 l = -sum(log_sigmoid(labels*logits))/n）一致；页面分数形式 $1/(1+e^{z_{ij}(-t x_i\cdot y_j-b)})$ 是 Algorithm 1 形式的正确等价改写（论文印刷版 Eq.2 指数项印作 $-t x_i\cdot y_j+b$，与自身 Algorithm 1 相悖，页面未沿用该笔误，取正确符号），系数 $1/|B|$ 与 §3.2 一致。
- F6 init：$t'=\log 10$（即 $t=10$）、$b=-10$（§3.2 第 3 段 + §4.9）。[C4]「arXiv v3 专门澄清 $t$ 与 $t'$ 的初始化」与 arXiv 页评论行「v3: clarify t vs t' init」一致（核对通过，非问题）。
- Table 1：SigLiT B/8 L 32k/4/1/79.8、g/14 L 20k/4/2/84.5；SigLIP B/16 B 16k/16/3/71.0、32k/32/2/72.1、32k/32/5/73.4。§4.5 末段「SigLIP achieves 72.1% 0-shot accuracy… compared to CLIP (approx. 2500 TPUv3-days for 72.6%) reported in [30]」，[30]=Li et al.(FLIP)。全部一致。
- Table 2（§4.3，30B seen）：16k/32k/64k/128k/240k 的 INet-0 与 XM avg / XM de / XM en / XM zh 五列数字逐格核对一致（如 32k：73.2 / 34.9 / 54.8 / 46.2 / 32.5；240k XM zh 23.7）。
- §1/§4.1：「the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16 k. As the train batch size grows, the gap closes.」「conceptually decouples the batch size from the definition of the task」原文一致。
- §4.4 SigLiT：4 chips / 2 days / 20k batch / 107k steps / 84.5%；LION + decoupled wd $10^{-7}$ + 6.5k 步 warm-up 至 $10^{-4}$ + cosine decay；文本塔为 12 层 L 变体；LiT dataset [59]。一致（79.7 与 79.8 论文自身并存，页面取 Table 1 的 79.8）。
- §4.2：softmax 需 98k 达最优、仍不及 sigmoid；4096 vs 2048 显存对照。一致。
- §4.8 图 6：页面「$b$ 最终值随 batch 组成变化，范围约 $-13$ 到 $-3$」经 300 dpi 渲染读图核对，曲线端点约 $-13$（Hard, matched pairs @1:16k）到约 $-3$（Random @1:1.6），一致。
- §3.3 chunked：无 all-gather、任一时刻只物化 $b\times b$（Figure 1 为 4×4）小矩阵、swap 文本嵌入 + cross-device sum，一致。
- §4.9 引文「the bias term ensures that the training starts close to the prior, preventing dramatic over-correction in early optimization」逐字一致。
- 手算（§3 折叠块）：$\log\sigma(-10)=-10.0000454$、$\log\sigma(10)=-0.0000454$、16 项和 $-40.0007$、$L\approx+10.0002$；$b=0$ 时 16 项和 $-11.09$、$L\approx+2.77$；$\sigma(-10)\approx0.0000454$；正负比例 1:32767（32k）、1:999999（1M）。全部可复算且正确（32k 按 32768 处理时负对数 $1.07\times10^9$ 亦正确）。梯度 $\partial\log\sigma(a)/\partial a=\sigma(-a)$、$b\ne0$ 时正负对梯度量级相抵的说法经计算成立。
- 机械项：`dojo:type=concept`、`dojo:topics=多模态`（词表内）、`dojo:tag=视觉与多模态`（ALLOWED_TAGS 内）；`.dojo/scripts/validate.py` 返回 validation ok；链接概念页 wiki/{clip,vit,standard-attention,moonvit-v2}/index.html 均存在；本地 libs 资源齐全；无 `$...$` 出现在 alt；无 inline SVG/等宽框图；无「（待生成）」；overview.html 与 index.html 互链。

## 问题

- [轻微·表述] §3「可学习温度 $t$ 与可学习 bias $b$」：「logit」一词全页指两个不同的量——§2/[F1] 按 Algorithm 1 把 logit 定义为 $t\,x_i\cdot y_j+b$，§3 正文与手算却把 $a=z_{ij}(t\,x_i\cdot y_j+b)$ 也称作「logit」（「对 logit $a=z_{ij}(t\,x_i\cdot y_j+b)$」「初始 logit $a=z_{ij}(t\cdot 0+b)$」），符号含义未全文单义，读者可能误以为 $\sigma$ 直接作用在 logit 上。｜引文依据：论文 Algorithm 1 第 9、11 行「logits = dot(zimg, ztxt.T) * t + b」「l = -sum(log_sigmoid(labels * logits)) / n」——$\sigma$ 的输入是 labels*logits，不是 logits。｜修复要求：§3 把 $a$ 改称「sigmoid 的输入 $a=z_{ij}(t\,x_i\cdot y_j+b)$」（或等价说法），不再把 $a$ 叫 logit。｜修复：｜复验：
- [轻微·表述] §2 本章问题第 3 题括注把形式差异写成「无分母、单向、二分类、有 bias」，其中「单向」与论文「sigmoid loss is symmetric」以及同章表格「一次性对 $|B|^2$ 对求和（CLIP 对称两次）」不一致，易被读成 sigmoid 只处理一个方向。｜引文依据：论文 §1「Importantly, the sigmoid loss is symmetric, requires just a single pass」（同章表格亦写「一次性对 $|B|^2$ 对求和」）。｜修复要求：把「单向」改为与表格一致的「一次性对 $|B|^2$ 对求和」或「单个 pass、无需对称两次」。｜修复：｜复验：
- [轻微·表述] §1「小 batch 下梯度噪声大」：「batch 小时（如 $|B|=256$），hardest negative 往往是"运气不好没采到难样本"」措辞口语化且指代含混（主语与因果都不明）。｜引文依据：不适用。｜修复要求：改为可判定的表述，如「batch 小的时候，批内相似度最高的那个负对往往只是随机采到的普通负样本——真正的难样本没被采进来」。｜修复：｜复验：
- [轻微·技术] §4「适用边界」表第三行「batch 极小（如 1:1）时 $b$ 作用减弱」是无来源支持的判断，且未按本页其他位置的做法（如「此为合理推断，论文未作此归因」）标注为推断；「1:1」对应退化 batch（$|B|=2$），与页面别处从 $|B|=4$ 起示意不一致。｜引文依据：论文 §4.8 只给出各正负比下 $b$ 的收敛值与「the imbalance does not seem to be a major reason for concern」，未给出该边界判断。｜修复要求：把该结论标为推断，或删去该单元格。｜修复：｜复验：
- [轻微·表述] §1「分布式训练需要 expensive all-gather」：句中把英文形容词 expensive 直接嵌入中文（其余英文均为 batch size / all-gather 这类技术名词），表达不统一。｜引文依据：不适用。｜修复要求：改为「需要昂贵的 all-gather」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布（遗留 5 条轻微问题均为表述与术语层面，不影响核心结论、来源一致性与主线理解）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
