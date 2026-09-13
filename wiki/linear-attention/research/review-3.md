<!-- review-meta
round: 3
page: wiki/linear-attention/index.html
reviewed_content_sha256: c0d68655f2dd38eb
-->
# 线性注意力审查记录（第 3 轮）

- 页面版本：29609a473ab784fc3ce64ef2cf195cdb3672e2a9
- 审查时间：2026-09-13 19:06
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. softmax 注意力——`O(N^2)` 瓶颈在哪一步（含本章问题）/ 2. 核函数与结合律——把相似度分解并重排（含本章问题）/ 3. 因果掩码——变成固定大小的递归状态（含本章问题）/ 4. 表达力代价——从公式的哪里来（含 4.1、4.2、4.3、本章问题）/ 来源与范围说明（全部 h3）/ overview.html
- 机械验证：`python3 .dojo/scripts/validate.py wiki/linear-attention/index.html` → `validation ok`（exit 0）
- 来源核对方式：Katharopoulos 2020（ar5iv `2006.16236`）、Choromanski 2020（ar5iv `2009.14794`）逐条抓原文；`<head>` topics/tag 对照 `.dojo/scripts/catalog_builder.py` 词表

## 逐项复算（全部通过）

- `QK^T = [[1,0,1],[0,1,1],[1,1,2]]` ✔；`phi(K)=K+1=[[2,1],[1,2],[2,2]]` ✔
- `phi(K)^T V = [[4,3],[3,4]]` ✔；`phi(K)^T 1 = (5,5)` ✔
- `V'_1 = (11,10)/15 ≈ (0.733,0.667)`，直接求和 `(5·(1,0)+4·(0,1)+6·(1,1))/15 = (11,10)/15` ✔ 两路一致
- 递归：`s_1=[[2,0],[1,0]]`、`z_1=(2,1)`；`s_2=[[2,1],[1,2]]`、`z_2=(3,3)`；`s_3=[[4,3],[3,4]]=phi(K)^T V`、`z_3=(5,5)` ✔
- `V'_3 = (14,14)/20 = (0.7,0.7)` ✔；`V'_1` 用 `s_1,z_1` 得 `(5,0)/5=(1,0)=V_1` ✔
- 权重表：`Q_3·K = (1,1,2)`；softmax `0.212/0.212/0.576` ✔；`elu+1` 相似度 `6/6/8`、权重 `0.300/0.300/0.400` ✔；比值 `0.576/0.212≈2.72=e^{2-1}` ✔、`0.400/0.300≈1.33` ✔
- Unicode 数学字符扫描：正文/标题/折叠块无未渲染数学字符（`4000×` 的 `×` 按 validate.py 第 53 行属"中文技术散文普通排版字符"，不违规，validate 通过）

## 问题

- [重要·技术] 来源与范围说明 → 外部数字与实验条件（行 515）：N2 把 WSJ 语音识别 PER 数据标为"Katharopoulos Table 1"，但 Table 1 是 MNIST 图像生成，WSJ 数据在 **Table 3**｜引文依据：ar5iv `2006.16236` 中 Table 1 = MNIST（softmax 0.621 / linear 0.644 bits/dim），Table 2 = CIFAR-10，Table 3 = Speech Recognition（WSJ，softmax 5.12 / linear 8.08 PER）｜修复要求：把 WSJ 的引注由"Table 1"改为"Table 3"（MNIST 保留 Table 1，可写"Table 1、Table 3"），并同步核对同一行"论文实验模型规模"表述｜修复：｜复验：

- [重要·技术] 行 381 与行 487：把"这正是 K3 的 KDA 选择线性注意力家族的根本原因——它支持常数时间推理和固定大小 KV 状态"写成因果结论，但页面 blockquote.meta 只列 Katharopoulos 2020 与 Choromanski 2020 两篇来源，"来源与范围说明"中 C1–C5/F1–F5/N1–N2 均无 K3/KDA 条目，该归属无来源支持｜引文依据：不适用（页面本身未给出来源）｜修复要求：为"K3 的 KDA 属线性注意力家族、因此获得常数时间推理"这一归属补可定位来源（K3/KDA 技术报告或官方源码），或降级为显式标注的推断（"据 KDA 的递归更新形式，可归入线性注意力家族"）并链接到 wiki 的 KDA/DeltaNet 页；不得以无条件因果句保留｜修复：｜复验：

- [重要·可读性] 行 163 图注："自上而下序列长度递增：$QK^T$ 的元素数按 $N^2$ 增长，$N$ 翻 2 个数量级、元素数翻 4 个数量级。"该图三层为 $N=3$、$N=6$、$N=300000$，图内不存在相差 2 个数量级的一对（3→6 仅差 0.3 个数量级；整体 3→300000 差约 5 个数量级、元素数 9→$9\times10^{10}$ 差约 10 个数量级），"2 个数量级/4 个数量级"与图内容不符（该数字对应正文行 146 的 3000→300000 区间）｜引文依据：不适用｜修复要求：按图内三层重写图注（例如"$N$ 从 3 增到 300000 涨约 5 个数量级，元素数涨约 10 个数量级"），或调整图内层标签使数量级关系自洽；不得保留与图内数据不符的数量级表述｜修复：｜复验：

- [轻微·技术] 来源说明 C1（行 494）：写作"Katharopoulos et al. 2020, §3.2.1 原文 'the computational cost of softmax attention scales with O(N²)'"，该句实际位于 **§3.2 "Linearized Attention"**，不在 §3.2.1｜引文依据：ar5iv `2006.16236` 中该句紧跟 §3.2 标题，"§3.2.1 Feature Maps and Computational Cost"另起一段｜修复要求：把 C1 中该引文的定位由"§3.2.1"改为"§3.2"｜修复：｜复验：

- [轻微·技术] 来源说明 C1（行 494）：写作"Choromanski et al. 2020, §1, Eq.(1)"，Performer 的 Eq.(1)（$\mathbf{A}=\exp(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})$，时间 $O(L^2d)$、空间 $O(L^2+Ld)$）在 **§2.1 "Preliminaries - regular attention mechanism"**｜引文依据：ar5iv `2009.14794` 中 Eq.(1) 位于 §2.1；Eq.(5) 位于 §2.3｜修复要求：把定位由"§1, Eq.(1)"改为"§2.1, Eq.(1)"（C5 的"§2, Eq.(5)"可保留为 §2.3）｜修复：｜复验：

- [轻微·格式] 全页公式自行编号：出现 11 处 `\tag{}`（`\tag{F1}`…`\tag{F5}`、`\tag{F2 之前}`、`\tag{F2 重排}`、`\tag{F3 之前}`、`\tag{结合律}`），与 style-guide 第 11 节"公式不自行编号，引用论文 Eq. 编号"不符；全站仅本页使用 `\tag{`（`grep -rl '\\tag{' wiki/*/index.html` 只命中本页）｜引文依据：不适用｜修复要求：去除 `\tag{}`，出处改用正文文字或已在用的 `<sup>[Fx]</sup>` 上标标注；如确需保留编号体系，先确认与 style-guide 第 11 节一致（尤其"F2 之前""F3 之前"这类非规范的编号文本）｜修复：｜复验：

- [轻微·格式] 行 514 N1："CIFAR-10 … linear 比 softmax 快 4000×（Katharopoulos Table 2，论文实验设置）"，但 Table 2 的数值是 4,462×，"4000×"是论文摘要的取整口径，与所引表号（Table 2）数值不一致｜引文依据：ar5iv `2006.16236` Table 2 CIFAR-10 Linear 17.85 images/sec = 4,462×；摘要写"up to 4000x faster"｜修复要求：改为"（Katharopoulos 摘要：up to 4000×；Table 2 为 4,462×）"或直接写"约 4,000×（Table 2 为 4,462×）"，使数字与所引位置一致｜修复：｜复验：

- [轻微·格式] N1 在正文中无引用：来源章节列出 N1（行 514），但正文全文无 `<sup>[N1]</sup>`（正文仅出现 `[N2]`，行 456），违反 style-guide 第 6 节"与来源章节双向对应"｜引文依据：不适用｜修复要求：在正文相应处（如第 1 章或第 4 章谈代价/速度）补 `<sup>[N1]</sup>` 引用，或删除 N1 条目｜修复：｜复验：

- [轻微·格式] 行 511 h3 命名为"外部数字与实验条件"，style-guide 第 1 节规定该固定小节名为"外部数字与实验条件（N）"（同目录 standard-attention 使用带"（N）"写法）｜引文依据：不适用｜修复要求：补全为"外部数字与实验条件（N）"｜修复：｜复验：

- [轻微·格式] h1（行 64）"线性注意力：用核函数把注意力从 $O(N^2)$ 降到 $O(N)$"缺英文缩写，style-guide 第 1 节要求 h1 格式为 `概念名（英文缩写）：核心作用或结论`（对照 kv-cache/causal-mask/delta-rule 均含括号缩写）｜引文依据：不适用｜修复要求：补英文缩写，改为"线性注意力（Linear Attention）：…"，或在页面说明无通行缩写并统一到站内规则｜修复：｜复验：

- [轻微·可读性] 行 205："关键假设来了：假如 $\mathrm{sim}$ 可以分解为…"，"…来了"属会话腔元话语，不在 style-guide 第 12 节允许的推导引导语（关键观察/同理/因此）之列｜引文依据：不适用｜修复要求：改为"关键假设是："或"核心前提是："等陈述式引语｜修复：｜复验：

- [轻微·格式] 行 129/235/343/441 用"构造示例。"作固定引入句（共 4 处，标签后另有多余空格），style-guide 第 4 节规定示例标记为"计算示例""代码示例""构造数据"且"不使用固定引入句"（"构造示例"仅在来源章节第 1 节作为固定 h3 名）｜引文依据：不适用｜修复要求：正文内改用"计算示例""构造数据"等规范标记并删除标签后多余空格；来源章节 h3"构造示例"保持不变｜修复：｜复验：

## 已核对通过的来源论断（摘要）

- C1：Katharopoulos §3.2 原句 "the computational cost of softmax attention scales with 𝒪(N²)" 存在 ✔；Choromanski "Time and space complexity … are O(L²d) and O(L²+Ld)" 存在 ✔（仅定位小节号有误，见上）
- C2：Katharopoulos §3.2 原句 "the only constraint we need to impose to sim(·), in order for equation 3 to define an attention function, is to be non-negative" 存在 ✔
- C3：Katharopoulos §3.2 原句 "we can compute ∑ φ(Kⱼ)Vⱼᵀ and ∑ φ(Kⱼ) once and reuse them for every query" 存在 ✔
- C4：因果式与 RNN 式（Eq. 9–12、16–20）存在 ✔
- C5：Katharopoulos §3.2.1 "We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative" 存在 ✔、Eq.(7) `φ(x)=elu(x)+1` ✔；Choromanski Lemma 1 正随机特征无偏近似存在 ✔（F5 公式与 Lemma 1 的 $h(x)=\exp(-\|x\|^2/2)$、$D=\mathcal{N}(0,I_d)$ 一致）
- N2 数值：MNIST 0.644 vs 0.621、WSJ 8.08 vs 5.12 与原文一致 ✔（表号引用有误，见上）
- `dojo:topics=注意力机制` 在 ALLOWED_TOPICS 内 ✔，`dojo:tag=注意力` 在 ALLOWED_TAGS 内 ✔；description 为纯文本、summary 可渲染 ✔
- 前置链接 `../standard-attention/index.html` 存在 ✔；overview.html 与 index.html 相互链接 ✔；页面无指向已移除 `research/` 路径、无"（待生成）"占位 ✔

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 9
- 处置：修复（核心公式与数值、主要来源引文均已复算/核对通过，无阻断；3 项重要问题须在发布前关闭，9 项轻微问题按需修复）