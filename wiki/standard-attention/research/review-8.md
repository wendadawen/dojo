<!-- review-meta
round: 8
page: wiki/standard-attention/index.html
reviewed_content_sha256: f7f48d7c8bcf7815
-->
# 标准 Transformer 注意力审查记录（第 8 轮）

- 页面版本：fbce37a1a9e32d64a13b3db08938b59050d03d03（index.html）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节（按顺序）：核心问题（4 题含解答）→ 最容易误解 → 1. 注意力要解决什么问题——从 RNN 的"逐步传递"到"直接查询"（含本章问题）→ 2. 缩放点积公式——每个符号与每一步（含 2×2 构造示例、4 步流程图与表、softmax 数值稳定折叠块、本章问题）→ 3. 为什么除以 √d_k——缩放因子的方差推导与不缩放的后果（含方差推导折叠块、雅可比折叠块、d_k 对照表折叠块、本章问题）→ 4. 多头注意力——拆分子空间、拼接、参数量等价（含多头流程图、3×3 因果遮罩折叠块、本章问题）→ 5. 复杂度、瓶颈与边界（含复杂度表、Flash/Linear 区分 callout、边界总结表、本章问题）→ 来源与范围说明 → overview.html 全文交叉核对。

## 逐条来源核对（§2.2 要求，均给出关键片段/数值）

- **C2/F1 公式与 C5/F2 多头公式**：arXiv:1706.03762v7 §3.2.1 Eq.(1) 为 $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$；§3.2.2 Eq.(2) 原文 "MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O where head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)"。与页面一致。
- **C3/F3 方差推导**：§3.2.1 脚注原文为分量独立、均值 0、方差 1 时 "their dot product ... has mean 0 and variance d_k"。页面 $\text{Var}(q\cdot k)=d_k$、$\text{std}=\sqrt{d_k}$ 与之一致；推导三步（单项方差 1、独立性使方差可加、求和得 d_k）可复算。
- **C4 不缩放后果**：§3.2.1 正文 "pushing the softmax function into regions where it has extremely small gradients"，并称按 $1/\sqrt{d_k}$ 缩放。页面链条（饱和→雅可比近零→梯度近零）相符；雅可比 $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ 由商法则复算成立。
- **C10 加性对比**：§3.2.1 原文 "While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k."。页面 [C10] 引文与该句逐字一致。
- **C7/F5 因果遮罩**：§3.2.3 原文 "masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"。页面 $M_{ij}=-\infty\ (j>i)$ 写法一致。
- **N1 超参**：§3.2.2 与 Table 3 原文 $d_{model}=512$、$h=8$、"d_k=d_v=d_model/h=64"；$W^O\in\mathbb{R}^{hd_v\times d_{model}}=\mathbb{R}^{512\times512}$ 成立。
- **N2/C8 Table 1 复杂度**：Self-Attention $O(n^2\cdot d)$/O(1)/O(1)；Recurrent $O(n\cdot d^2)$/O(n)/O(n)；Convolutional $O(k\cdot n\cdot d^2)$/O(1)/$O(\log_k n)$。页面表格三行逐格一致；"n<d 时自注意力比 RNN 计算量更小"由 $n^2d<nd^2\Leftrightarrow n<d$ 复算成立。
- **C9 位置无关**：§3.5 原文 "we must inject some information about the relative or absolute position of the tokens in the sequence"，支持"注意力本身不引入位置信息"。
- **Flash Attention 数字**：arXiv:2205.14135v2 §4 "Benchmarking Attention" 原文 "FlashAttention is up to 3× faster than the standard attention implementation" "across common sequence lengths from 128 to 2K"，摘要 "15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record"。页面"（128–2K）上最多约 3 倍快于标准注意力实现（BERT-large、序列长度 512，相对 MLPerf 1.1 训练记录，端到端训练提速 15%）"逐点一致。
- **Linear Attention**：Katharopoulos et al. 2020 以 $\phi(q)\phi(k)$ 替代 $\exp(q\cdot k)$、重排为 $\phi(Q)(\phi(K)^\top V)$、复杂度 $O(n d^2)$，页面表述一致。
- **Michel et al. 2019 多头冗余**：页面标注为"只引论文设计目标、冗余结果不在核心论断内"，归因与措辞未越界。
- **数值复算（本机）**：2×2 例 $\text{softmax}([0.7071,0])=[0.670,0.330]$、$AV$ 得 $[1.66,2.66;2.34,3.34]$、形状 $2\times2$；3×3 遮罩例每行归一 $[1,0,0]$、$[0.401,0.599,0]$、$[0.258,0.316,0.426]$；$\ln 256=5.545$、$1/256=0.0039$；$2048^2\approx4.2\times10^6$、$32768^2\approx1.1\times10^9$；$8\sqrt{2\ln 8}\approx16.3$（复算最大值 $\approx11.4$）、$8\sqrt{2\ln 256}\approx26.6$（复算 $\approx22.6$），与页面"约 11/约 23 且渐近式对有限 n 高估"一致。
- **d_k 对照表复算（Python，n=256、2000 次试验）**：d_k=4 → 0.171/3.97/0.044/5.07；d_k=64 → 0.754/0.737/0.044/5.05；d_k=1024 → 0.943/0.143/0.043/5.05。与页面表格 0.170/3.97/0.043/5.05、0.749/0.75/0.043/5.05、0.936/0.16/0.043/5.05 在复算误差内一致；正文"约 0.75、约 0.75"与"0.94、0.16"同表一致。
- **页面功能**：`.dojo/scripts/validate.py wiki/standard-attention/index.html` → `validation ok`，退出码 0；dojo:topics=注意力机制、dojo:tag=注意力 均在 AGENTS.md/catalog_builder.py 词表内；dojo:summary 含 `$...$` 可渲染；overview.html 与 index.html 双向互链；../rope/index.html 与 ../mla/index.html 均真实存在（75KB/53KB），无"（待生成）"占位；无 Unicode 数学字符出现在标题/summary/正文/列表/表格（√、² 仅出现在 head description）；alt 无 `$...$`；两张结构图均为 HTML 结构（非等宽字符框线图）。

## 问题

- [轻微·可读性] overview.html L53：句末"本页只讲标准形式"以「本页」为主语的自我指代，属可改写为客观表述的元话语｜引文依据：不适用｜修复要求：改写为不以「本页」为主语的表述（如"标准形式之外、带相对位置偏置的变体不在讨论范围"），保留原有范围说明信息｜修复：｜复验：
- [轻微·格式] index.html L610：来源与范围说明的「构造示例」列出「不缩放 vs 缩放 2×2 对照」（$\text{softmax}([1,0])=[0.731,0.269]$ 未缩放），但该未缩放数值与对照在正文中未出现——`0.731` 全页仅出现于 L610 一处，正文只展示缩放后的 $[0.670,0.330]$；正文的"不缩放 vs 缩放"对照用的是另一构造（L334–347 的 d_k=64、n=256 模拟表）｜引文依据：`grep -c "0.731" index.html` = 1（仅 L610）；正文 L248 仅出现 $\text{softmax}([0.7071,0])=[0.670,0.330]$｜修复要求：二者取一——要么在 §2 的 2×2 构造示例处补出未缩放对照 $\text{softmax}([1,0])=[0.731,0.269]$，要么将该条从「构造示例」列表中删去，使列表仅描述正文实际展示的样例｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微均为表达/文档一致性问题，不影响核心结论、公式与数字的来源一致性；建议按修复要求一并处理。全部学习目标由正文章节回答；页面级「核心问题」4 题与 5 个章节的「本章问题」均有解答折叠块且与正文结论一致；关键论断、公式与数字已逐条回源核对并复算。）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
