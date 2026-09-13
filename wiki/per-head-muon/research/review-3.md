<!-- review-meta
round: 3
page: wiki/per-head-muon/index.html
reviewed_content_sha256: af9b7f6913032bb8
-->
# Per-Head Muon 审查记录（第 3 轮）

- 页面版本：f29640899b5e77a0a37f648f3a3712177e3f5fba
- 审查时间：2026-09-13 19:10
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：页面开头（主要依据 + 引言）→ 核心问题 → 1. 整块正交化——为什么对多头不够好 → 2. 按头正交化——切分动量矩阵并逐头正交化 → 3. 效果与开销——均衡更新尺度与开销变化 → 4. 分布式实现——P2P 参数取回 → 来源与范围说明（含全部 details 折叠块、两幅内联 SVG 图与页脚脚本）

## 来源核对（通过项）

不构成问题，记录以备复验：

- K3 技术报告 §2.5 "Per-Head Muon" 与 §5.2.2 "P2P-based Muon Orthogonalization" 经 ar5iv/arXiv HTML（arXiv:2607.24653）逐句核对，C1–C5 引文全部逐字命中："full-matrix orthogonalization treats all heads as a single coupled block …", "we partition their momentum matrices along the head dimension and orthogonalize each head's block separately …", "while smaller-scale heads receive insufficiently normalized updates; per-head orthogonalization equalizes the update scale across heads.", "this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on … per-head blocks are cheaper …", 以及 §5.2.2 的 "each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication … eliminating the full-parameter buffer …"、"… pipelined at the granularity of model-chunk buffers."。
- F2 经 Keller Jordan "Muon" 博客核对：`Ortho(G) = arg min_O { ||O - G||_F : … }`，且等价于取 `UV^T`；博客发布日 "December 8, 2024"，页面标注 "2024-12" 正确。
- 构造示例算术复算一致：$M=\begin{bmatrix}3&4\\0.3&0.4\end{bmatrix}$ 秩 1，$\sigma_1=\sqrt{25.25}\approx5.025$，$MM^\top$ 特征值 $25.25,0$，$u_1\approx[0.995,0.0995]$，$v_1=[0.6,0.8]$，$\mathrm{Ortho}(M)$ 两行范数 $0.995$ 与 $0.0995$；本章问题第 2 题示例 $[4,3]/[0.4,0.3]$ 同得 $0.995/0.0995$；示例 $[1,0]/[0,2]$ 得 $[1.0,1.0]$。Gram 开销比 $\approx1/H$ 的推导自洽。
- `python3 .dojo/scripts/validate.py wiki/per-head-muon/index.html` → validation ok；`dojo:topics=训练与优化` 在 AGENTS.md 允许列表内；前置概念页 wiki/muon-optimizer/index.html、wiki/newton-schulz/index.html 均存在；overview.html 与 index.html 双向链接存在；页内无 Unicode 数学字符（"~2×" 的 "×" 按 validate.py 注释属中文散文排版字符，不判违规）。

## 问题

- [重要·技术] §1 第 4 段"这个例子是极端情形…"、§1 本章问题第 1 题解答、§3 第 1 段：把"全矩阵正交化压低小尺度头行块范数"写成"行空间有重叠 / 各头不完全平行"时普遍出现的现象，并称只有"完全正交"时才不出现。该成立条件错误，且页内两处对"行空间有重叠"的用法互相矛盾｜引文依据：按页面自己的定义复算——$M=\begin{bmatrix}3&4\\0.3&0.5\end{bmatrix}$（两头尺度 $5$ 与 $0.583$、两行不平行、行空间相交）为满秩（$MM^\top$ 特征值 $25.34$ 与 $0.0031$ 均非零），$\mathrm{Ortho}(M)$ 为正交阵，实测两头行块范数为 $[1.0,\ 1.0]$，并无 10:1 压低；$M=\begin{bmatrix}1&0\\1&1\end{bmatrix}$（两行既不平行也不正交）同样得 $[1.0,\ 1.0]$，可见"完全正交才不压低"不成立。反向对照：页面示例 $M=\begin{bmatrix}3&4\\0.3&0.4\end{bmatrix}$ 两行平行 → 秩 1 → 实测 $[0.995,\ 0.0995]$。即"$\mathrm{Ortho}(M)$ 各行范数为 1"的充要条件是 $M$ 行满秩（$\mathrm{rank}(M)=H d_h$），压低只出现在 $\mathrm{rank}(M)<H d_h$ 的秩亏情形，与各行是否正交、是否平行无关；本页示例恰是秩 1 退化情形，正文据此断言非平行情形"同样存在"缺乏依据。此外 §1 本章问题第 1 题解答把"行空间有重叠"直接等同于"平行同向"，与正文"行空间有重叠的非平行情形同样存在"的措辞自相矛盾｜修复要求：把现象成立条件由"行空间有重叠 / 非平行"改为"$\mathrm{rank}(M)<H d_h$（各头行块线性相关、整块动量为秩亏矩阵）"；在示例处明确标注其为秩亏极情形；删去或改写"在行空间有重叠的非平行情形下同样存在"与"完全正交时各头不互相干扰、行块范数均为 1"（后者应改为"行满秩时各头行块范数均为 1"）；使 §1 正文、§1 本章问题解答、§3 首段对"行空间有重叠"的用法一致，消除页内矛盾｜修复：｜复验：

- [轻微·格式] 来源与范围说明下 h3 标题写作"外部数字与实验条件"，缺规范固定后缀"（N）"｜引文依据：guides/concept/style-guide.md §11 规定来源章节固定命名为"外部数字与实验条件（N）"；站内 wiki/attention-sink/index.html:349、wiki/deepseek-v4-1/index.html:756 等 58 个页面均写作"外部数字与实验条件（N）"｜修复要求：标题改名为"外部数字与实验条件（N）"，与规范和站内多数页面一致｜修复：｜复验：

- [轻微·表述] §1→§2、§2→§3、§3→§4 三处章末过渡句（"本章讲清了…但…——下一章讲…"）使用同一固定句式｜引文依据：不适用｜修复要求：至少两处改写为不同措辞，保留"先总述本章已得结论、再点出结论尚未解决的下一步问题"的信息，不改成报章节名的空过渡｜修复：｜复验：

- [轻微·来源] §2"机制澄清"段与"简化三"："NS 多项式系数、迭代步数、归一化方式都与原版 Muon 一致"以及系数 $(3.4445,-4.7750,2.0315)$、5 步迭代、bfloat16 归一化等，是页面推断，K3 §2.5 未陈述，页面未标为推断｜引文依据：K3 §2.5 原文只有 "we partition their momentum matrices along the head dimension and orthogonalize each head's block separately"，未提任何 NS 系数、步数或归一化方式｜修复要求：把"与原版 Muon 一致"一句标为推断（如"由 C2 可推断正交化过程本身未改，仅作用对象变化"），或删去 K3 未陈述的系数、步数与归一化细节｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复。来源引文、构造示例算术、结构（核心问题 / 本章问题 / 来源与范围说明 / 简化条件）、链接与 validate.py 均通过；唯一重要问题为 §1 起"压低现象"的成立条件写错并引发页内矛盾，需按上述修复要求改正后再复验，3 个轻微问题一并处理。
