<!-- review-meta
round: 7
page: wiki/moonep/index.html
reviewed_content_sha256: bd0e14457b5acac7
-->
# MoonEP 完美均衡专家并行审查记录（第 7 轮）

- 页面版本：140cf5ebefb773793f095f9da9b64e64f8a2e20a
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题（4 题）；引言与开篇构造示例；1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路（含图 1、本章问题）；2. 冗余专家的界——$E/R$ 上界与基本紧性（含 Theorem 1/2、两处折叠块、本章问题）；3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程（含 3.1–3.4、图 2、伪代码折叠块、本章问题）；4. MoonEP 的边界——解决与不解决，以及与 ECHO/UltraEP/DeepEP 的区别（含 4.1–4.4、对比表、本章问题）；来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）。
- 来源获取：Kimi K3 Technical Report（arXiv:2607.24653v1，2026-07-27，全文 HTML 落盘后逐段提取）§5.2.1、§E、§5.2、§2.3、§2.3.3、§5.2.2；overview.html 交叉比对。页面无位图/SVG 标尺图，图内数值为流程节点文字，与正文逐字一致，无需像素测量。

## 问题

- [轻微·格式] 「构造示例」独立公式（$\sum_{r=0}^{R-1} n_r = S\times K\times R$）与「2. 冗余专家的界」独立公式（$M(I)=\min_P\max_r\{m_r(P)\}$）之后直接接标题/正文，未按 style-guide §11「公式后紧跟 `<ul>` 逐项定义每个符号」给出符号表；同仓库其它概念页（如 wiki/kda）该式后均带 `<ul>`｜引文依据：不适用｜修复要求：在两式后各补一个 `<ul>`，逐项列出 $n_r,S,K,R$ 与 $M(I),P,m_r,I$ 的含义（符号本身已在紧邻正文出现，补表只为格式一致）｜修复：｜复验：
- [轻微·格式] 全页自称在「本文」与「本页」间混用：引言写「本文依据 K3 技术报告…」（第 70 行），meta 与 4.2、来源章写「本页」（第 67、423、498 行）｜引文依据：不适用｜修复要求：按 style-guide §12 统一为「本页」或「本文」其一｜修复：｜复验：
- [轻微·功能] 四章末的 `<h3>本章问题</h3>` 均无 id，端脚本按 `textContent.replace(/[\s#?？：]/g,'-')` 生成同一 id="本章问题"，运行时产生 4 个重复 id，侧边目录四条「本章问题」锚点全部指向第 1 章，滚动高亮也会同时点亮四条｜引文依据：不适用（页面功能类）｜修复要求：参照仓库既有写法（wiki/linear-attention/index.html 的 `id="chapter-N-questions"`）为每章 `<h3>本章问题</h3>` 加唯一 id｜修复：｜复验：

## 来源核对记录（已核对且无问题，含引文依据）

- §5.2.1 原文「MoonEP requires every rank to receive exactly $S\times K$ tokens」「a balanced plan always exists with at most $E/R$ redundant experts per rank and that this bound is essentially tight」「Reserving $E/R$ redundant-expert slots per rank therefore guarantees … training is never interrupted」「Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size $S\times K\times R$, whereas MoonEP requires only a fixed $S\times K$ buffer owing to the perfect balance」「This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead」「ECHO and UltraEP presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists … the cap itself requires manual tuning while still leaving residual imbalance」——与页面 C2/C3/C7/C8/C10 及 3.1/3.2/4.3 表述逐条一致。
- §5.2.1「Computing the exact optimum at every training step is prohibitively expensive … near-optimal, incurs negligible overhead, and always respects the $E/R$ upper bound」「a fused permute/unpermute operator in which the planning kernel precomputes the destination of every token … views of the communication buffer are returned directly to the computation, eliminating intermediate copies」——与 3.3 一致。
- §E 原文「$M(I)=\min_{P}\max_{r}\{m_r(P)\}$」「$M(I)\leq E/R$ (Theorem 1)」「$M=\lceil E(R-1)/R^2\rceil\approx E/R$ (Theorem 2)」「each expert receives $\frac{SKR^2}{E(R-1)}$ tokens」「at most $R-1$ fills; meanwhile, each rank is filled at most once, so its remote tokens come from a single rank」「$M(I)=\min_P\max_r\{m_r(P)\}\leq\max_r\{m_r(P^*)\}\leq\frac{E}{R}$ (28)」——与页面 F1/F2/F3、Theorem 1 补充折叠块、最坏构造段、公式 (28) 引用一致。
- 教学构造 $E=4,R=2,S=4,K=1$ 复算：总量 $S\times K\times R=8$；情形 A 迁 4 pair、rank 1 需专家 0 副本即 1 个冗余 ≤ $E/R=2$；情形 B 每被用专家收 $SKR^2/(E(R-1))=4$、下界 $\lceil SK/(SKR^2/(E(R-1)))\rceil=\lceil 4/4\rceil=1=M(I^*)<E/R=2$——与页面一致。
- §5.2「activations, gradients, and optimizer states exceed the memory budget」+ §5.2.2「all GPU memory is allocated on the main compute stream and managed within a single memory pool, avoiding multi-stream fragmentation」「Unified activation manager」——与 4.2 对 §5.2.2 的定性一致。
- §2.3「896 routed experts with 16 active experts per token」、§2.3「2.8-trillion-parameter scale」、§1「3T-class parameters」、§5.2 标题「Infra for 3T-class Pre-Training」、§2.3.3「Kimi K3 adopts auxiliary-loss-free routing … we introduce Quantile Balancing (QB)」、全文无 K3 训练用 EP size $R$ / 每 rank 序列长度 $S$——与来源章 N 逐项一致。
- 4.2「报告 §5.2.1 最后一段」为 §5.2.1 正文末段「Expert-GEMM scheduling and overlap」（其后即 §5.2.2），核验属实；per-expert 偏斜 / fixed-order workload-oblivious / 解析硬件代价模型 + 离线 autotuning / shared expert 独立 stream 均与该段原文一致。
- overview.html 与 index.html 的数字与结论一致（$S\times K$、$S\times K\times R$、$E/R$、$\lceil E(R-1)/R^2\rceil\approx E/R$、2.8T/896/16）；两页互链；引用的 wiki/moe-serving/index.html、wiki/gpu-execution-model/index.html 均真实存在。
- 机械项：`.dojo/scripts/validate.py` 返回成功；`$` 定界符配平（474，偶数）；C1–C10、F1–F4 全部在正文与来源章双向对应且编号自洽（含 `<sup>[C4, C6]</sup>` 组合式）；数学字符均走 KaTeX，`×` 属 validate.py 允许的纯排版字符且仅出现在 `<title>`/description 等纯文本位与 `<pre><code>` 内；无 `（待生成）` 占位、无 `$...$` 进入 alt、伪代码块明确标为「language-text」且未声称可运行。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复 3 条轻微后可发布（0 阻断 / 0 重要；核心结论、公式、数字、引文编号与 K3 报告 §5.2.1 / §E 逐条对齐，教学构造可复算，无需返回规划）