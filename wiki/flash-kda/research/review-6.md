<!-- review-meta
round: 6
page: wiki/flash-kda/index.html
reviewed_content_sha256: f0d0bbc538dd1842
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 6 轮）

- 页面版本：index.html 工作树 sha256 = 33b7288ae5234745e3aba87b09a3991d57b67aa31f4ce135c9aff69e51292133
- 审查时间：2026-09-13 21:11
- 审查者：编排者派发的独立子代理（第 6 轮审查者，未参与写作，也未参与前序轮次的审查与修复）
- 来源获取：Kimi K3 Technical Report（arXiv:2607.24653）正文 HTML，定位 §5.1.1「KDA Kernels across Regimes」（FlashKDA、设备内 CP）、§5.1.2「KDA Context Parallelism」Eq.17、§5.4.2「High-Performance Kernels · KDA」、§2.1.1 Eq.1 与符号定义、scaled sigmoid 的 α 范围、§2.1.2「Gated MLA」、参考文献 [13]/[72]/[142]；官方 config.json（https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json，逐字段解析）；NVIDIA H100 SXM5 的 SM 数以公开规格核对。仅使用 index.html、overview.html、上述外部来源与本规范；未读取本页 research/ 下任何文件。
- 已完整阅读章节：核心问题、最容易误解、1. 串行状态 vs GPU 并行、2. FlashKDA、3. 设备内 context parallelism、4. KCP（4.1–4.5，含行 476 的「展开」折叠块）、5. KDA 解码（5.1–5.3）、来源与范围说明（论断与来源 C／公式与来源 F／外部数字与实验条件 N／构造示例／辅助解释与类比边界／简化条件及其限制）、overview.html。

## 核对通过项（对照来源记录，含关键数值/原文）

- 数字：config.json 顶层 `"dtype": "bfloat16"`、`linear_attn_config.head_dim=128`、`num_heads=96`、`kda_layers`(69)、`full_attn_layers`(24)、`gate_lower_bound=-5.0`，与正文「69 层 KDA…其余 24 层为 Gated MLA 全局注意力」、[N1] dk=dv=128、[N2] 96 head 一致。128×128×2B=32768B≈32KB，两片段合计≈64KB 与行 125/429 一致。
- H100：[N3]「H100 SXM5 有 132 块 SM」与公开规格一致。
- α 范围：[N4] 称 α 由 scaled sigmoid 产出、范围 (e^{-5},1)，报告 §2.1.1 原文 `\bm{\alpha}_{t}^{h} = \exp(\bm{g}_{t}^{h}) \in (e^{g_{\min}},1)^{d_k}`、`With g_{\min}=-5 … \alpha_{t,j}^{h} > e^{-5}`，一致。
- 公式：正文 Eq.1 `S_t = M_t S_{t-1} + β_t k_t v_t^⊤, M_t:=(I-β_t k_t k_t^⊤)Diag(α_t)` 与报告 §2.1.1 Eq.1 逐符号一致；boxed Eq.17 `S_t^{[i+1]} = S̃_t^{[i+1]} + M_{t←1}^{[i+1]}S_T^{[i]}` 与报告 `\mathbf{S}_{[i+1]}^{t}=\widetilde{\mathbf{S}}_{[i+1]}^{t}+\mathbf{M}_{[i+1]}^{t\leftarrow1}\mathbf{S}_{[i]}^{T_i}` 一致；行 393 的求和形式把报告 Eq.17 外层因子 `\mathbf{M}_{[i+1]}^{T_{i+1}\leftarrow1}` 折入连乘 `∏_{l=j+1}^{i}M^{[l]}`，与报告第三行 `\sum_{j=1}^{i}(\prod_{l\leftarrow j+1}^{i}\mathbf{M}_{[l]}^{T_l\leftarrow1})\widetilde{\mathbf{S}}_{[j]}^{T_j}` 等价（连乘按文档顺序自右向左，即 M^{[i]}M^{[i-1]}…M^{[j+1]}）。
- 手算示例（行 444–487）：逐步复算 4 步 ground truth，S1=[[1,2],[0,0]]、S2=[[1,2],[3,4]]、S3=[[5.5,7],[3,4]]、S4=[[5.5,7],[8.5,10]]；两段 M_{T←1}=M2M1=M4M3=0.5I；S̃_T^{[1]}=[[1,2],[3,4]]、S̃_T^{[2]}=[[5,6],[7,8]]；prefix scan 重组 S_T^{[2]}=S4；误用直接求和得 [[6,8],[10,12]]。全部与页面所写矩阵相符。
- 引文：[C1] 与报告 §5.1.1 `The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime.` 相符；[C2] 与 `We therefore develop FlashKDA, a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation…` 相符；[C3] 与 `An automatic SM-level context-parallel (CP) planner… partitions the sequence across the SMs of a single rank… this parallelism is entirely intra-device and incurs no cross-device communication.` 相符；[C4] 与 `This direct summation, however, is insufficient for KDA…`、`KCP requires only a fixed-size all-gather…` 相符；[C5] 与 §5.4.2 `the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state…`、`a design independently proposed in the concurrent work ReplaySSM.`、`Because the projection caches never leave the decode stage…` 相符。softmax CP 引用 [72] 经参考文献表核对为 `Ring attention with blockwise transformers for near-infinite context`。
- 机械项：`.dojo/scripts/validate.py wiki/flash-kda/index.html` 返回 `validation ok`；无「（待生成）」占位；正文引用的 kda / linear-attention / gpu-execution-model 三页均存在；overview.html 与 index.html 互链；唯一 `<img>` 为 lightbox，`alt=""` 无 `$...$`；结构图为 HTML（dg-flow/dg-stack），类名在 ../../libs/dojo-concept.css 中均有定义。

## 问题

- [重要·技术] 行 391：求和形式的引出句把时间下标与 rank 下标写错——`$t=T_{i+1}$ 时离开 rank $i$ 的状态`，但页面自身把 `$S_T^{[i]}$` 定义为「离开 rank $i$、进入 rank $i+1$ 的状态」（行 426 图注、行 383–385），离开 rank $i$ 的状态出现在 $t=T_i$（rank $i$ 段末），而非 $t=T_{i+1}$；紧随其后的等式 `S_T^{[i]}=\tilde S_T^{[i]}+\sum_{j=1}^{i-1}(\prod_{l=j+1}^{i}M_{T\leftarrow1}^{[l]})\tilde S_T^{[j]}` 求和上限为 $i-1$、连乘到 $[i]$，本身确实描述 rank $i$，与报告 Eq.17 中以 $t=T_{i+1}$、LHS 为 $\mathbf{S}_{[i+1]}^{T_{i+1}}$ 的形式相差一个 rank 下标。读者按引出句回查会与等式、图注三方对不上，正是本章（Eq.17 的下标记账）的核心。｜引文依据：报告 §5.1.2 `At $t=T_{i+1}$, both quantities $\mathbf{M}_{[i+1]}^{T_{i+1}\leftarrow 1}$ and $\widetilde{\mathbf{S}}_{[i+1]}^{T_{i+1}}$ can be computed using only the local tokens`（$t=T_{i+1}$ 对应 rank $i+1$ 整段）；页面行 426 图注 `$S_T^{[i]}$ 表示离开 rank $i$、进入 rank $i+1$ 的状态`。｜修复要求：把行 391 的 `$t=T_{i+1}$` 改为 `$t=T_i$`（保持「离开 rank $i$ 的状态」与 LHS `$S_T^{[i]}$` 不变），或改为 `$t=T_{i+1}$ 时离开 rank $i+1$ 的状态` 并同步把等式 LHS 与求和上下限改成 rank $i+1$ 的形式；二选一，不得同时保留 $T_{i+1}$ 与 rank $i$。｜修复：｜复验：

- [轻微·表述] 行 444：构造示例的理由「（略去通道衰减，使手算落在整数上）」与同页结果不符——$\beta=0.5$ 使 $M$ 含 0.5、且后续状态出现 0.5I、$S_3$ 的 5.5、$S_4$ 的 8.5/5.5 等半整数，并非整数；行 628「$M_{T\leftarrow1}=0.5I$ 干净」也不支持「整数」的说法。｜引文依据：不适用（页面内部：行 451 `$M_1=M_3=\begin{pmatrix}0.5&0\\0&1\end{pmatrix}$`、行 465 `$S_4=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$`）。｜修复要求：把「使手算落在整数上」改为与事实相符的说法，如「使衰减因子保持为 1、手算数值简洁」。｜修复：｜复验：

- [轻微·表述] 行 544（图注）：`$t_2$ 被拒绝后需要回滚到 $t_1$ 之后（$S_a$ 之后的位置）再继续——但状态没有“回到 $S_a$”的快照` 中括号与后文矛盾——$t_1$ 被接受后状态即 $S_a$，回滚目标就是 $S_a$，写成「$S_a$ 之后的位置」与同句「回到 $S_a$」冲突。｜引文依据：不适用（页面内部：行 526 `状态原地更新到 $S_a$；验证 ✓ 接受`）。｜修复要求：删去括号或改为「回滚到 $t_1$ 之后，即状态 $S_a$」。｜修复：｜复验：

- [轻微·格式] 行 625：[N4] 在正文中没有任何 `<sup>[N4]</sup>` 上标引用（全页该标记仅出现在来源条目自身），而 α 由 scaled sigmoid 产出、范围 $(e^{-5},1)$ 这一事实在行 628、637 出现时未带上标，来源章节与正文的双向对应不完整。｜引文依据：不适用（机械核对：`grep` 得正文 0 处 `<sup>[N4]</sup>`，行 628/637 提及该范围时无上标）。｜修复要求：在行 444 或行 628/637 提及真实 α 范围处补 `<sup>[N4]</sup>`，或删除 [N4] 条目。｜修复：｜复验：

- [轻微·格式] 行 479（「展开」折叠块）：局部推导用 `$\tilde S_1^{(1)}$`、`$\tilde S_1^{(2)}$`、`$\tilde S_2^{(1)}$`、`$\tilde S_2^{(2)}$` 表示零起点状态，与全页及本块自身的写法 `$\tilde S_T^{[i]}$`（下标为本地 token 数 T、方括号为 rank）不一致——此处换成下标为 rank、上标括号为步数，同一符号出现两套索引约定。｜引文依据：不适用（页面内部：行 479 同段既写 `$\tilde S_T^{[1]}$` 又写 `$\tilde S_1^{(1)}$`）。｜修复要求：折叠块内改用与正文一致的 `$\tilde S_t^{[1]}$`（t 为步数）或直接写 `$\tilde S_t^{[1]}\big|_{t=1}$` 之类只保留一套索引的写法。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复（1 项重要须关闭；4 项轻微可在同轮一并订正）
