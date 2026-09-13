<!-- review-meta
round: 3
page: wiki/siglip/index.html
reviewed_content_sha256: ba56db2dc12548ed
-->
# SigLIP 审查记录（第 3 轮）

- 页面版本：index.html 工作树哈希 ce353badeaf88ba4a997ea6d582ea39f1d0f9b87；overview.html 哈希 0a9be409aa0ce01197b8f9c77d5147c93fc30c80
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 核对来源：arXiv:2303.15343v4（2023-09-27，ICCV 2023 Oral）PDF 全文（含 §3.1/§3.2/§3.3/§4.1–§4.9、Table 1、Table 2、Table 4、Figure 1/2/6、Algorithm 1、参考文献）；官方实现 google-research/big_vision（big_vision/utils.py 的 sigmoid_xent、big_vision/models/proj/image_text/two_towers.py 的 bias/temperature 参数、Algorithm 1 伪代码）
- 已完整阅读章节：核心问题（4 题）、最容易误解、1. CLIP softmax 损失的"全局耦合"问题、2. SigLIP 损失公式、3. 可学习温度 $t$ 与可学习 bias $b$、4. batch size 解耦与实验边界、来源与范围说明；含全部「本章问题」解答折叠块、公式等价推导折叠块、$|B|=4$ 手算折叠块、两个 pair-matrix 图注与全部表格图注
- 机械验证：`.dojo/scripts/validate.py wiki/siglip/index.html` → `validation ok`；内部链接 clip / vit / standard-attention / moonvit-v2 均存在；overview.html 与 index.html 互链；无 research/ 残留路径、无「（待生成）」占位
- 复算：$\sigma(-10)\approx 4.54\times10^{-5}$、$\log\sigma(-10)\approx-10.0000454$、$\sigma(10)\approx0.9999546$、$\log\sigma(10)\approx-0.0000454$、$\log 0.5\approx-0.6931$；手算 $4\times(-10.0000454)+12\times(-0.0000454)=-40.0007$，$/4=-10.0002$；$b=0$ 时 $16\times(-0.6931)/4=-2.77$ —— 均与页面一致；Table 1 / Table 2 全部数字与论文一致

## 问题

- [阻断·技术] 来源：Zhai 2023 §3.2 公式与 Algorithm 1｜位置：第 2 章正文第 200、202 行、折叠块第 248 行、[F1] 第 455 行｜问题：页面把论文 §3.2 的损失公式原文写成 $\log\frac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j-b)}}$（$-b$）并明确称"论文原文写作"该形式；但论文 §3.2 印出的公式是 $-\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|}\log\frac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j+b)}}$（指数里 b 前是**加号**）。页面右侧的等价形式 $\log\sigma(z_{ij}(t\,x_i\cdot y_j+b))$ 与论文 Algorithm 1（`logits = dot(zimg, ztxt.T) * t + b`、`l = -sum(log_sigmoid(labels * logits)) / n`，labels $=2\cdot\text{eye}(n)-\text{ones}(n)$）一致，故被教的公式本身正确，错在把与论文印出不符的形式标注为"论文原文"｜引文依据：arXiv:2303.15343v4 §3.2 排版公式分母指数逐字为 $z_{ij}(-t\,x_i\cdot y_j+b)$（400 dpi 渲染截图核对，b 前为「+」）；Algorithm 1 第 9–11 行 `logits = dot(zimg, ztxt.T) * t + b` / `labels = 2 * eye(n) - ones(n) # -1 with diagonal 1` / `l = -sum(log_sigmoid(labels * logits)) / n`｜修复要求：不得把与论文印出不符的形式写成"论文原文"。或（a）改以 Algorithm 1 作为 $\log\sigma(z_{ij}(t\,x_i\cdot y_j+b))$ 的来源；或（b）如实引用 §3.2 印出的 $+b$ 形式并说明它与 Algorithm 1 的差异。修改后须重新核对来源｜修复：｜复验：

- [阻断·技术] 来源：Zhai 2023 Table 1 图注 + §4.1、§4.4｜位置：第 4 章表 1 图注第 351 行、「本章问题」解答第 426 行、[N1] 第 463 行｜问题：页面三处把 SigLiT 的训练数据集写成"WebLI 数据集"；论文中 SigLiT 用的是 LiT image-text 数据集，WebLI 是 SigLIP from-scratch（§4.2）用的数据集｜引文依据：Table 1 图注 "SigLiT model with a frozen public B/8 checkpoint [42], trained on the **LiT image-text dataset [59]**"；§4.1 "…we use the same precomputed embeddings…and train a base size text tower from scratch…using the LiT image-text dataset [59]"；§4.4 "We follow the same setup as in section 4.1."｜修复要求：三处改为 "LiT image-text dataset（[59]）"，或删去数据集一句；[N2]（SigLIP from scratch）的 WebLI 标注正确、保留｜修复：｜复验：

- [阻断·技术] 来源：Zhai 2023 Table 1 及其脚注、§4.4｜位置：「本章问题」解答第 426 行（与表 1 第 347 行自相矛盾）｜问题：页面称 SigLiT 84.5% 配置为"Large 文本塔（12 层）"；论文中 12 层变体只用于 B/8 行（文本塔记作 L*，脚注 "We use a variant of the L model with 12 layers."），g/14 行是标准 L（24 层）。页面表 1 第 347 行 g/14 L 行文本塔写的是 "L"，第 426 行却写"（12 层）"，同页两处互相矛盾，且与官方 Table 1 不符｜引文依据：Table 1 两行为 "SigLiT B/8 **L\*** 32 k 4 1 79.8" 与 "SigLiT g/14 **L** 20 k 4 2 84.5"，脚注 "\* We use a variant of the L model with 12 layers."；§4.4 对 g/14 仅称 "with a ViT-g/14 model as the vision tower and a Large text tower"｜修复要求：第 426 行删去"（12 层）"，与表 1 及论文一致｜修复：｜复验：

- [重要·技术] 来源：Zhai 2023 §1 引言、§4.1、§4.2｜位置：第 99、118、162、190、338、402、411、436 行、[C5] 第 449 行｜问题：页面 9 处把"batch < 16k 时 sigmoid 显著优于 softmax"这一结论标注为"论文 §4.2"。该结论的实际出处是 §1 引言与 §4.1（SigLiT 实验）；§4.2 正文只写 "< 32 k batch size 时 SigLIP 优于 CLIP(WebLI) 基线"，并无 16k 一句。页面第 120 行又把同一动机句标为 §1，前后标注不一致｜引文依据：§1（p.2 左栏）"We find that the sigmoid loss performs significantly better than the softmax loss when the batch size is smaller than 16 k. As the train batch size grows, the gap closes."；§4.1 "When the batch size is smaller than 16 k, sigmoid loss outperforms softmax loss by a large margin."；§4.2 "Figure 2 middle plot shows SigLIP results, With less than 32 k batch size, SigLIP outperforms CLIP (WebLI) baselines."｜修复要求：把该结论及第 338 行所引原句的出处统一改为 §1（引言）/§4.1；凡以 §4.2 标注处或改来源、或改为 §4.2 正文的实际表述（"< 32k 优于 CLIP"）｜修复：｜复验：

- [重要·技术] 来源：Zhai 2023 §1 引言｜位置：第 168 行、[C5] 第 449 行；overview.html 第 32 行｜问题：页面称 §3.2 第 1 段含 "conceptually decouples the batch size from the definition of the task"；该句实际在 §1 引言。§3.2 第 1 段无此句。且页面第 120 行把同一句标为 §1，同页两处标注互相矛盾｜引文依据：§1（p.1）"Additionally, it conceptually decouples the batch size from the definition of the task."；§3.2 第 1 段为 "Instead of the softmax-based contrastive loss, we propose a simpler alternative that does not require computing global normalization factors."｜修复要求：三处统一标注为 §1（引言）｜修复：｜复验：

- [重要·技术] 来源：Zhai 2023 §4.8｜位置：第 307 行、第 475 行｜问题：页面两处称"论文 §4.5 图 6 显示 $b$ 最终值与负采样策略有关"。图 6（Learned bias / batch composition）位于 §4.8 "Negative ratio in sigmoid loss"，不在 §4.5；§4.5 是 "SigLIP with a small amount of TPU-v4 chips"｜引文依据：§4.8 正文与图 6 图注 "Figure 6: The effect of batch composition. We simulate various batch compositions by masking out negatives…"；§4.5 标题 "4.5. SigLIP with a small amount of TPU-v4 chips"｜修复要求：两处改为 §4.8（图 6 编号本身正确，保留）｜修复：｜复验：

- [重要·技术] 来源：Zhai 2023 参考文献 [30]｜位置：第 366、431、451、464 行｜问题：页面称 CLIP 72.6% 的 2500 TPUv3-days 出自"论文引用 [30]（Pham et al.）"。论文 [30] 的作者是 Yanghao Li et al.（FLIP），非 Pham｜引文依据：参考文献 "[30] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. CoRR, abs/2212.00794, 2022."｜修复要求：把 "[30] Pham et al." 改为 "[30] Li et al.（FLIP）"，或只保留"论文引用 [30]"｜修复：｜复验：

- [重要·技术] 来源：Zhai 2023 §4.4、§4.5｜位置：开篇第 86 行｜问题：开篇"用 4 个 TPUv4、2 天训练，SigLIP 达到 ImageNet zero-shot 84.5%；同等水平的 CLIP 需要约 2500 TPUv3-days"把两个不同口径并置：84.5% 是 SigLiT（冻结 ViT-g/14 图像塔），而 CLIP 的 ~2500 TPUv3-days 对应的是 72.6%（§4.5），并非"同等水平"。读者会得到"CLIP 达到 84.5% 需 2500 TPUv3-days"的错误印象｜引文依据：§4.4 "With a ViT-g/14 model as the vision tower and a Large text tower, we can train at 20 k batch size on four chips for 107 k steps in under two days. This further pushes the 0-shot ImageNet classification accuracy up to 84.5%."；§4.5 "…compared to CLIP (approx. 2500 TPUv3-days for 72.6%) reported in [30]."｜修复要求：拆开表述——84.5% 属 SigLiT（冻结图像塔），2500 TPUv3-days 对应 CLIP 72.6%；删去"同等水平"或将 2500 TPUv3-days 与 72.6% 绑定｜修复：｜复验：

- [轻微·技术] 来源：Zhai 2023 §4.5｜位置：「本章问题」解答第 431 行｜问题：页面写"同精度下 SigLIP 的训练算力降低约一个数量级以上"，论文只说 "significant training cost reduction"，未给出"一个数量级以上"的量化结论；该量化（2500/64≈39，且 TPUv3 与 TPUv4 单芯吞吐不同）属页面推断｜引文依据：§4.5 "This presents a significant training cost reduction e.g. compared to CLIP (approx. 2500 TPUv3-days for 72.6%)"｜修复要求：删除"约一个数量级以上"，改为"训练算力大幅降低（论文原话 significant training cost reduction）"，或标注为推断｜修复：｜复验：

- [轻微·格式] 来源：不适用｜位置：第 366、431 行｜问题：正文直接出现 Unicode 数学字符："32 TPUv4 × 2 天 = 64 TPUv4-days" 中的 `×` 与 `=`，未由 KaTeX 渲染｜引文依据：不适用（规范 §2.2 第 9 条：标题、summary、正文、列表和表格中无 Unicode 数学字符直接出现）｜修复要求：改为 KaTeX，如 `$32$ TPUv4 $\times$ 2 天 $=64$ TPUv4-days`，或改写为文字「32 个 TPUv4 连续 2 天，合计 64 TPUv4-天」｜修复：｜复验：

- [轻微·格式] 来源：不适用｜位置：第 418 行｜问题：符号 $b$ 全文用于「可学习 bias」，此处又用于 "per-device batch size"（"$b\times b$（per-device batch size 平方）"），同一变量两种含义，违反符号全文一致｜引文依据：不适用（规范 §2.2 第 9 条：同一变量全页写法一致）｜修复要求：per-device batch size 换用其它符号（如 $b_{\text{dev}}$），并保留括号说明｜修复：｜复验：

- [轻微·技术] 来源：Zhai 2023 §3.2／Algorithm 1｜位置：第 2 章符号解释第 210 行｜问题："对 $|B|^2$ 个对求和后除以 $|B|$（…这意味着正对与负对的权重不同…）"表述不准确：同一求和除以 $|B|$ 时每个对的权重相同（均为 $1/|B|$），正对与负对的单对权重并无不同；失衡来自两类对的**数量**（$|B|$ vs $|B|^2-|B|$），不是单对权重｜引文依据：§3.2 公式系数为 $-\frac{1}{|B|}$（对所有 $|B|^2$ 个 $L_{ij}$ 同一个系数）｜修复要求：改为"正负对数量差异使负类在总损失中占主导"，或删去该分句｜修复：｜复验：

- [轻微·表述] 来源：不适用｜位置：第 86、162、194、208、210、215、270、304、334、338、384、418 行｜问题：存在元话语与会话式过渡："本文回答：…""本文的改造对象"（86）、"本页推断"（162）、"下一章看…"（194、334）、"下一章专门讲…/下一章解释原因/下一章详细手算/下一章讲…"（208、210、215、270）、"本章把这一机制主张与论文实验数字对齐"（338）、"注意：…"（304）、"读出三点："（384）、"本页不展开完整伪代码…只在正文一句话提及"（418）｜引文依据：不适用（规范 §2.2 第 12 条：排除元话语"本页将…""下面来看…""需要注意的是"等）｜修复要求：删除"本文/本页/下一章…/读出三点/注意："等元话语，改为直接陈述结论与机制；章节衔接用中性小标题或直接进入内容｜修复：｜复验：

- [轻微·表述] 来源：不适用｜位置：overview.html 第 49 行｜问题：overview 写"batch = 1 M 边际收益快速消失（论文摘要 + Table 2）"，但 Table 2 的 batch 最大只到 240k，1M 结论来自摘要（与 §4.1/§4.2），不含 Table 2｜引文依据：Table 2 列为 16k/32k/64k/128k/240k；摘要 "…up to one million, and find that the benefits of growing batch size quickly diminish"｜修复要求：删去"Table 2"，改为"（论文摘要）"｜修复：｜复验：

## 结论

- 统计：阻断 3 / 重要 5 / 轻微 6
- 处置：修复
- 说明：页面核心数字（SigLiT 84.5%／4 TPUv4／2 天／batch 20k、SigLIP from scratch 72.1%／32 TPUv4／2 天、CLIP 72.6%／~2500 TPUv3-days、Table 2 的 32k 最优与 XM 下降、1M 边际收益消失）与 Table 1/Table 2 逐格核对一致，手算与渐近值全部复算通过；问题集中在公式/条目的**来源标注与配置细节**（§3.2 公式符号、"16k"与"conceptually decouples"的章节出处、图 6 归属、[30] 作者、SigLiT 数据集、g/14 文本塔层数）以及表述层的元话语。三条阻断（F1 公式原文标注、SigLiT 数据集、g/14 层数且同页矛盾）须先关闭，全部重要问题须关闭后方可发布。