<!-- review-meta
round: 6
page: wiki/linear-attention/index.html
reviewed_content_sha256: a4bb46903df1af64
-->
# 线性注意力审查记录（第 6 轮）

- 页面版本：983170de96002d389025b144cbb2c265ecd49afb（index.html 工作树哈希；overview.html 9d1edec0）
- 审查时间：2026-09-14 17:01
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（4 条）→ 最容易误解 → 1. softmax 注意力——O(N²) 瓶颈在哪一步（含本章问题 3 条）→ 2. 核函数与结合律——把相似度分解并重排（含本章问题 3 条）→ 3. 因果掩码——变成固定大小的递归状态（含本章问题 3 条）→ 4. 表达力代价——代价来自公式的哪一部分（4.1 / 4.2 / 4.3，含本章问题 3 条）→ 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）；并完整阅读 overview.html。
- 来源获取方式：Katharopoulos et al. 2020 取自 arXiv:2006.16236v3 全文 PDF（pdftotext 全文核对）；Choromanski et al. 2020 取自 arXiv:2009.14794（ar5iv HTML 正文 §1–§2.3）。KDA/K3 归属另经外部材料核对。
- 已核对并确认无误的项（供复验）：Katharopoulos Eq.(2)(3)(4)(5)(6)(7)(9)(10)(11)(12)(16)(17)(18)(19)(20) 逐式比对一致；引文四句逐字比对一致（"scales with O(N²)"、"the only constraint … is to be non-negative"、"…once and reuse them for every query"、"We prefer elu(·) over relu(·)…"）；Table 1 MNIST 0.621/0.644、Table 3 WSJ PER 5.12/8.08、Table 2 CIFAR-10 4,462× 与摘要 "up to 4000x" 全部一致；Choromanski §2.1 Eq.(1) 的 A=exp(QKᵀ/√d)、时间 O(L²d)、空间 O(L²+Ld)，§2.3 Eq.(5) 与 Lemma 1 的正随机特征构造（φ(x)=exp(−‖x‖²/2)/√m·(exp(ω₁ᵀx),…,exp(ω_mᵀx))，E[φ(q)ᵀφ(k)]=exp(qᵀk)）逐条一致；全部手算（QK^T=[[1,0,1],[0,1,1],[1,1,2]]；φ(K)ᵀV=[[4,3],[3,4]]；φ(K)ᵀ1=(5,5)；V′₁=(11,10)/15≈(0.733,0.667)；s₁…s₃、z₁…z₃；V′₃=(14,14)/20=(0.7,0.7)；V′₁=(5,0)/5=(1,0)）复算无误；权重对照表 0.212/0.576→2.72、0.300/0.400→1.33、6/6/8、20 全部复算无误；9×10⁶/9×10⁸/9×10¹⁰、3→300000 约 5 个数量级、9→9×10¹⁰ 约 10 个数量级均正确；引用的四个概念页（standard-attention、kda、delta-rule、gated-deltanet）与全部 libs 资源、index.html 均存在；dojo:topics=注意力机制、dojo:tag=注意力 均在 ALLOWED_TOPICS / ALLOWED_TAGS 词表内；dojo:summary 的 KaTeX 定界符完整；两处图注与图上内容一致；`python3 .dojo/scripts/validate.py wiki/linear-attention/index.html` 返回 `validation ok`。

## 问题

- [重要·一致性] 4.3 节（index.html 行 439 与行 441）：小节标题写「两种核的权重分布对比」，紧接的正文却写「对比它在三种核下对 $K_1, K_2, K_3$ 的注意力权重」，而同节表格（行 443–452）只有「softmax 权重」与「$\mathrm{elu}{+}1$ 相似度 / 权重」两组核，不存在第三种核；4.1/4.2 也只给出 elu+1 与随机特征两种，且 4.2 的随机特征并未进入该表。同一小节内标题与正文对「几种核」自相矛盾。｜引文依据：不适用（页面内部矛盾；表头为「$j$ ｜ $Q_3^T K_j$ ｜ softmax 权重 ｜ $\mathrm{elu}{+}1$ 相似度 ｜ $\mathrm{elu}{+}1$ 权重」，共 2 种核）｜修复要求：把行 441 的「三种核」改为「两种核」，使 h3 标题、正文、表头三处对核的数量同义；若确需第三种核，须在表中补出对应列并在标题中同步，不得只改一处。｜修复：｜复验：

- [轻微·一致性] overview.html 行 31 与 index.html 行 71 / 行 146：同一个 $9\times10^{10}$ 在两文件中指不同的量。overview 写「$N = 30$ 万时这一步就要 $9 \times 10^{10}$ 次运算」，把元素数说成运算次数；index 行 71 写「约 $9 \times 10^{10}$ 个 query-key 相似度」，行 146 明确「需要 $O(N^2 d)$ 次乘加…所以总共 $N^2 \times d$ 次」。$300000^2 = 9\times10^{10}$ 是相似度（矩阵元素）个数，乘加次数为 $9\times10^{10}\times d$，两处口径相差一个 $d$ 因子。｜引文依据：index 行 146「计算这一步需要 $O(N^2 d)$ 次乘加（$N \times N$ 矩阵每个元素是 $d$ 维向量的内积，需要 $d$ 次乘加，所以总共 $N^2 \times d$ 次）」｜修复要求：overview 该句改为「这一步要构造 $9 \times 10^{10}$ 个 query-key 相似度」或补出乘加量级，使 overview 与 index 对同一数字的口径一致。｜修复：｜复验：

- [轻微·技术] index.html 行 201：「softmax 注意力中 $Q K^T$ 的每个元素是 $\exp(Q_i^T K_j / \sqrt{d})$」表述不准确——$QK^T$ 的元素是内积 $Q_i^T K_j$，指数与 $1/\sqrt{d}$ 缩放作用在 $QK^T$ 之上（指数化后的矩阵才是相似度矩阵）。同页行 121–123、行 127 都正确区分了「先算 $QK^T$」与「softmax/exp」，此处与它们不一致。｜引文依据：Katharopoulos 2020 §3.1 Eq.(2) 原文「$Al (x) = V' = \mathrm{softmax}(QK^T/\sqrt{D})V$」——先算 $QK^T$，$\exp$ 与 $1/\sqrt{D}$ 在 softmax 内作用于其上；Choromanski Eq.(1)「$A = \exp(QK^T/\sqrt{d})$」同理｜修复要求：改为「$Q K^T$ 的元素 $Q_i^T K_j$ 经 $\exp(\cdot/\sqrt{d})$ 后即相似度 $\exp(Q_i^T K_j/\sqrt{d})$」，使行 201 与行 121–127 对 $QK^T$ 的用法一致。｜修复：｜复验：

- [轻微·技术] index.html 行 456：「Katharopoulos 2020 的实验反映了这一代价：MNIST…（线性略差），WSJ…（线性明显差）」——该句把这两个实测差距直接归因为「线性核没有指数聚焦能力」这一机制，来源未做此归因；且来源对同一特征映射的自评是表现相当，页面未提，读者会得到「来源结论是 linear 全面更差」的印象。引用的两个数字本身与 Table 1 / Table 3 一致，故不构成阻断。｜引文依据：Katharopoulos 2020 §3.2.1「we show that the feature map of equation 7 performs on par to the full transformer, while significantly reducing the computational and memory requirements」；摘要「Our linear transformers achieve similar performance to vanilla transformers」；Table 1 注「Our linear transformers achieve almost the same bits/dim as the full softmax attention」｜修复要求：把该句改为页面推断（如「据此可推断，MNIST/WSJ 上的差距与核的聚焦能力下降方向一致」），并补一句来源自评「来源结论：该特征映射在多数任务上与 full transformer 表现相当，仅 WSJ PER 有明显差距」，使来源结论与页面推断分离。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复