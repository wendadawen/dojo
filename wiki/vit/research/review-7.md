<!-- review-meta
round: 7
page: wiki/vit/index.html
reviewed_content_sha256: 1f3553be1cdf1467
-->
# Vision Transformer（ViT）审查记录（第 7 轮）

- 页面版本：02b3c5a449e5c9c2893afa8f13db86e8a561466e（`git hash-object wiki/vit/index.html`）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题、最容易误解、1. ViT 要解决什么问题——CNN 的归纳偏置与大数据下的局限（含本章问题折叠块）、2. 图像如何变成 token 序列——patch embedding、class token、位置编码（含本章问题与 3 个补充折叠块）、3. Transformer 编码块——与标准 Transformer 几乎完全一致（含本章问题与 pre-LN 补充折叠块）、4. 数据规模边界——什么时候 ViT 比 CNN 强、什么时候反而更弱（含本章问题）、来源与范围说明（C/F/N、构造示例、辅助解释、简化条件）
- 来源核对版本：Dosovitskiy et al. 2021（ICLR 2021 camera-ready = **arXiv:2010.11929v2**，经 ar5iv 渲染全文逐条核对）。跨页引用核对 standard-attention 工作树版本。

## 结论先行：本轮未发现阻断问题

页面主线（tokenization 三步、$z_0$ 与编码块公式、数据规模边界）与来源一致，逐条核对结果如下（均给出定位到的原文片段或数值）：

- **Eq.(1)** 逐字一致：`z_0 = [x_class; x_p^1 E; x_p^2 E; ⋯; x_p^N E] + E_pos`，`E ∈ ℝ^{(P²·C)×D}, E_pos ∈ ℝ^{(N+1)×D}`（v2 §3.1）。与正文 line 194、[F1]、核心问题 2 答案完全一致。
- **Eq.(2)/(3)/(4)** 逐字一致：`z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}`、`z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ`、`y = LN(z_L^0)`（v2 §3.1）。
- **"large scale training trumps inductive bias"** 确在 **§1 Introduction**："We find that large scale training trumps inductive bias."——页面 line 142/402/458 归到 §1，正确。
- **Table 1 配置**：ViT-Base 12/768/3072/12/86M、ViT-Large 24/1024/4096/16/307M、ViT-Huge 32/1280/5120/16/632M——与 v2 Table 1 逐格一致（页面表格 line 332–334、[N1]）。
- **Table 2 数值**：ViT-L/16(I21k) 85.30±0.02 / 0.23k；ViT-L/16(JFT) 87.76±0.03 / 0.68k；ViT-H/14(JFT) 88.55±0.04 / 2.5k；BiT-L 87.54±0.02 / 9.9k；Noisy Student 88.4/88.5* / 12.3k——与 v2 Table 2 一致（页面表格 line 385–389、[N3]）。1/4、1/5 复算：9.9/2.5≈3.96、12.3/2.5≈4.92，成立。
- **§4.1 数据集**：ImageNet 1k 类 1.3M 图 / ImageNet-21k 21k 类 14M / JFT 18k 类 303M——一致。
- **§4.3 小数据结论**：原文 "Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets"、ViT-B/32 "performs much worse on the 9M subset"——支持页面 line 377「同算力下比 ResNet 过拟合更严重、泛化更差」。
- **D.3**：分类头 "tanh as non-linearity in the single hidden layer"；"In order to stay as close as possible to the original Transformer model"、GAP 差异 "fully explained by the requirement for a different learning-rate"——支持 line 121/200/210/299/463。
- **D.4 / §3.1**：1D vs 2D "little to no difference"、§3.1 "we have not observed significant performance gains from using more advanced 2D-aware position embeddings"——支持 line 205。
- **"仅 patch 切分保留少量 2D 邻域结构"**：§3.1 "the 2D neighborhood structure is used very sparingly"、§5 "we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step"——支持。
- **配置沿用 BERT**：§4.1 "We base ViT configurations on those used for BERT"、"'Base' and 'Large' models are directly adopted from BERT"——支持 line 338。
- **手算**：224×224/16²=196、+1=197；768/12=64；P=32→49、P=14→256；$16\times16\times3=768$——全部复算通过。
- **cross-ref 章节号**：standard-attention §1（路径 $O(1)$）、§4（多头）、§5（$O(n^2 d_k)$）、F1、$d_k=64$（"论文 $d_k=64$"）均逐一在该页核对到。
- **链接与功能**：standard-attention / residual-connection / positional-encoding / moonvit-v2 / clip / siglip 六个页面均存在；overview.html 与 index.html 互链；`validate.py` 返回 "validation ok: wiki/vit/index.html"；正文无 `<pre>`/`<code>`（无可运行代码）；正文/标题/summary/表格无越界 Unicode 数学字符（唯一 `×` 在 description 纯文本，允许）；无 img 故无 alt 内 `$...$`。

## 问题

- [重要·技术] 第 1 章表格「远距离交互路径」行与章末定义句（index.html line 148、line 153）：CN

N 的长程交互路径写成 $O(n/k)$，既无来源，也与本页作为该对照权威所链接的前置页 standard-attention 给出的 $O(\log_k n)$ 直接矛盾｜引文依据：standard-attention 工作树 line 74「远距离交互需堆叠多层（路径 $O(\log_k n)$）」、line 127 表行「CNN（kernel $k$）… $O(\log_k n)$」、line 604「CNN $O(knd^2)$ / $O(1)$ / $O(\log_k n)$，Vaswani 2017 §4 Table 1」；而被核对的 v2 原文 §3.1 对 CNN 只写 "locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer"，未给 CNN 长程路径的复杂度，故 $O(n/k)$ 无来源可依｜修复要求：把第 1 章表格该行与章末定义句改为与 standard-attention 一致的 $O(\log_k n)$（并注明等同于 Vaswani 2017 §4 Table 1），或删除该行数值改为定性描述「需堆叠多层、远多于 ViT 的一层」；不得保留无来源且与前置页冲突的 $O(n/k)$｜修复：｜复验：
- [轻微·可读性] line 153：$n$ 的定义「$n$ 为图像的线性位置数（序列化后的一维位置数，例如边长像素数）」中两个说法互相冲突——把 224×224 图像序列化后的一维位置数是 $HW=50176$，而「边长像素数」是 224｜引文依据：不适用｜修复要求：把 $n$ 改成单一定义，例如「$n$ 为把图像视作一维序列时的位置数（本文取边长像素数 224）」，删去与之冲突的「序列化后的一维位置数」｜修复：｜复验：
- [轻微·技术] line 253：「论文 Table 1 给出 ViT-H/14 配置（patch 14）」把 patch 大小归到 Table 1｜引文依据：v2 Table 1 表头为 Model / Layers / Hidden size $D$ / MLP size / Heads / Params，ViT-Huge 行 32/1280/5120/16/632M，**无 patch 列**；「ViT-H/14」命名与 patch 14 出现在 §4.2 Table 2 及正文，不在 Table 1｜修复要求：把出处改为 §4.1/§4.2 正文（或「Table 2 的 ViT-H/14 命名」），Table 1 只用于层数/维度/头数/参数量｜修复：｜复验：
- [轻微·技术] line 142：「NLP 已经验证标准 Transformer 的可扩展性：BERT → GPT-3 趋势显示数据越多、模型越大、性能持续上升」为无引文编号的经验性论断｜引文依据：不适用（本句无 [C]/[N] 编号）；被核对的 v2 §1 仅称 Transformer 已成 NLP 事实标准，未给出 BERT→GPT-3 的规模-性能趋势｜修复要求：补一条来源（如 Kaplan et al. 2020 缩放律）或删去 GPT-3 具体趋势、改为有来源的定性表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复

统计：阻断 0 / 重要 1 / 轻微 3