<!-- review-meta
round: 4
page: wiki/gated-deltanet/index.html
reviewed_content_sha256: bf633dfe65c5d9b7
-->
# Gated DeltaNet 审查记录（第 4 轮）

- 页面版本：1ea7a0256e58b9553b7705e7999ba6f5cf2e203d
- 审查时间：2026-09-13 19:39
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. DeltaNet 与 Mamba2 各自缺什么 / 2. Gated DeltaNet 的公式与符号（含 2.1、2.2） / 3. 手算 Gated DeltaNet 一步更新（含 3.1 与展开折叠块） / 4. 退化关系与并行训练算法（含 4.1、4.2 与代码折叠块） / 5. 与 KDA 的关系及实验效果（含 5.1–5.3） / 来源与范围说明；另读 overview.html 全文
- 来源获取：arXiv:2412.06464v1 PDF（下载原文并转文本核对编号与引文）、arXiv:2406.06484（DeltaNet 公式）、NVlabs/GatedDeltaNet 官方源码、K3 报告（arXiv:2607.24653 §2.1.1）、Qwen3-Next 与 Qwen3.5-397B-A17B 官方模型卡/config
- 机械核对：`python3 .dojo/scripts/validate.py wiki/gated-deltanet/index.html` → validation ok；代码块用 numpy 2.0.2 实际执行，输出与页面「预期输出」逐行一致（含两条退化验证 True）；Table 2/Table 3 全部数字与 v1 原文一致；F1=Eq.8、F3=§2.2、F4=arXiv:2406.06484 §2.2、chunk C 为 16 的倍数且取 64、§3.2 蓝色门控逐元素乘法的引文均核对通过；`dojo:topics=注意力机制` 在 AGENTS.md 允许词表内；无 research/ 失效路径。

## 问题

- [重要·技术] 正文「1. DeltaNet 与 Mamba2 各自缺什么」第 132 行、来源条目 C1（第 669 行）与 C2（第 671 行）、正文第 138 行：两处直接引文标注的章节号与 v1 原文不符。｜引文依据：v1 §1 Introduction（p.2–3）原文 "Mamba2 addresses this limitation by introducing a simple gated update rule, St = αt St−1 + vt kt⊺, which uniformly decays all key-value associations at each time step by a dynamic ratio, αt. However, this approach does not account for the varying importance of different key-value associations," 与 "since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased."；对整份 v1 文本检索 "varying importance"、"rapidly clear" 均只命中 §1，§2.2（Mamba2）与 §2.3（DeltaNet）正文不含这两句。页面把前句标为 §2.2（C2 及正文第 138 行）、把后句标为 §2.3（正文第 132 行，C1 条已含 §1）。｜修复要求：把 C2 与正文第 138 行、第 132 行的章节号改标为 §1（公式 F3 的 §2.2 归属正确，勿改）。｜修复：｜复验：
- [重要·技术] 「4.2 并行训练算法」补充折叠块（第 425 行）：给出的扩展 WY 表示与 v1 §3.2 Eq.9 不符，漏掉求和项内每项的 γ 比值。｜引文依据：v1 Eq.9 为 "P^r_[t] = γ^r_[t](I − Σ_{i=1}^r (γ^i_[t]/γ^r_[t]) w^i_[t] k^i⊤_[t])"，即 P^r = γ^r I − Σ_i γ^i w^i k^i⊤；页面写作 $\mathbf{P}_{[t]}^r = \gamma_{[t]}^r(I - \sum_i w_i k_i^\top)$ = γ^r I − Σ γ^r w k^⊤，逐项系数由 γ^i 被写成 γ^r，恰好抹掉了该折叠块自称要展示的"纳入衰减项"效果。｜修复要求：按 Eq.9 实际形式改写（保留 γ^i/γ^r 系数），或删去该式、只保留"扩展 WY 表示纳入 α_t"的文字说明。｜修复：｜复验：
- [重要·技术] overview.html 第 31 行：引号内的"论文引文"并非原文，且章节号错误。｜引文依据：overview 写 论文 §2.3 说它 "lacks the ability to rapidly clear outdated information, particularly during context switches"；v1 §1 原文为 "the model lacks the ability to rapidly clear outdated **or irrelevant** information, **especially** during context switches **where previous data needs to be erased**"（"especially"→"particularly"、漏 "or irrelevant"、漏末句，且章节应为 §1 而非 §2.3）。｜修复要求：overview 该处改为逐字原文，并把章节号改为 §1。｜修复：｜复验：
- [轻微·表述] 全文多处以"本页"为主语的自我指代（元话语）。｜引文依据：不适用。位置：第 68、138、191、425、590、633、711 行，如"本页只引用此公式作对比""本页默认 $d_v = d_k = d$""本页不展开 UT 变换…""本页构造了一组教学数字"。｜修复要求：改为不含"本页"的表述（如"以下只引用此公式作对比""本章默认…""此处不展开…"），或直接删除。｜修复：｜复验：
- [轻微·技术] 「5.1」第 586 行：把 KDA 的 $\alpha_t$ 取值范围写成 $\alpha_t \in (0,1)^{d_k}$，与同页表格（第 580 行）及同段所述 lower-bounded（$g \in (g_{\min},0)$、$\alpha_t$ 下界 $e^{-5}$，第 588 行）自相矛盾。｜引文依据：K3 报告 §2.1.1 为 $\alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k}$，$g_{\min}=-5$。｜修复要求：把 (0,1)^{d_k} 改为 $(e^{g_{\min}}, 1)^{d_k}$（或 $(e^{-5},1)^{d_k}$）。｜修复：｜复验：
- [轻微·技术] 「5.3」第 632 行：K3 层数数字"93 层 backbone 中 69 层用 KDA"无行内来源标记。｜引文依据：K3 报告/官方 config 为 93 层、69 层 KDA + 24 层 Gated MLA，数字本身正确；页面该句无 <sup>[Nx]</sup>。｜修复要求：为该数字补来源标记（在来源章节新增一条指向 K3 报告），或删除该数字。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 3
- 处置：修复（F1 公式、三模型手算与对比表、代码及其输出、Table 2/3 全部数字、退化关系、chunkwise 与 C/16 倍数条件均已回源核对通过，无阻断问题；上列重要问题需修复后再复验）