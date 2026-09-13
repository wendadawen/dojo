<!-- review-meta
round: 4
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 9338996dce3b1960
-->
# Quantile Balancing 审查记录（第 4 轮）

- 页面版本：aa3dff2569a2bb93153949e14b82f814f1ddd285
- 审查时间：2026-09-13 19:47
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. auxiliary-loss-free 路由——bias 做什么、不做什么 / 本章问题 / 2. DeepSeek-V3 的固定步长更新——为什么 896 专家时更难维持均衡 / 本章问题 / 3. QB 的核心机制——从一次前向推导下一个 bias（3.1–3.9，含「代码：手算例子的完整 QB 计算」折叠块）/ 本章问题 / 4. QB 为什么好——从平衡分配到对偶的视角（含「补充：从对偶目标到 coordinate minimizer」折叠块）/ 本章问题 / 5. 训练时的直方图估计与推理冻结（含「代码：直方图估计的分位数恢复」）/ 本章问题 / 来源与范围说明
- 来源获取方式：K3 技术报告 arXiv:2607.24653v2（并回退核对 v1）、DeepSeek-V3 技术报告 arXiv:2412.19437 §2.1.2、HuggingFace moonshotai/Kimi-K3/config.json（均用 WebFetch/curl 抓原文逐段核对）

## 已完成的机械核对（全部通过）

- 执行页面「代码：手算例子的完整 QB 计算」折叠块中的 Python（纯标准库，python3 直接运行），实际输出与页面「预期输出」逐行一致：初始 loads [4,3,1,0] → QB 后 [2,2,2,2]，改变的 token 为 T4(E1→E3), T5(E1→E4), T8(E2→E4)。
- 复算全部表格：3.5 分数矩阵、3.6 margins 表（8×4，逐格 $s_{i,j}-\alpha_i$）、3.7 分位数表（各专家降序第 3 大与 $\tilde{b}$）、3.8 mean-centering（$-0.2-0.1+0.0+0.1)/4=-0.05$，$b=(-0.15,-0.05,0.05,0.15)$）、3.9 新路由表（$s+b$ 与 Top-1），全部与正文数字一致。
- 对照 config.json 原文：`num_experts: 896`、`num_experts_per_token: 16`、`moe_router_activation_func: "sigmoid"`、`topk_method: "noaux_tc"`、`num_shared_experts: 2`，与页面 C17/N1/§2 一致。
- 对照 K3 报告 Table 1 原文：「Routed Experts 384 896 ↑133%」「Experts Active per Token 8 16 ↑100%」，与页面「K2 的 384 → K3 的 896」一致。
- 对照 K3 报告逐条定位：Eq.13（$\mathcal{T}_i=\operatorname{argtop}_k(\bm s_i+\bm b)$、$p_{i,j}=s_{i,j}/\sum_{r\in\mathcal T_i}s_{i,r}$）、Eq.14（$\widehat b_j\leftarrow-\operatorname{quantile}_{1-k/n}(\bm s_{:,j}-\bm\alpha)$ 与 mean-centering）、Eq.20（平衡分配）、Eq.23（对偶目标）、Eq.25–26（两向 $(1-k/n)$ 分位数 coordinate minimizer）、Eq.27（$\partial\mathcal L/\partial\beta_j=mk/n-\sum_i\chi(\cdot)$）、Appendix D（$[b_{\min}-1,b_{\max}+1]$、$B=1000$、$w=(b_{\max}-b_{\min}+2)/B$、$\lceil q\rceil$、pooled-batch 分位数、EMA），页面表述与公式与原文一致；「nearly $10^3$ experts 几步内收敛」「$B=1000$ 误差 a few $10^{-3}$」「低于 raw margins 交换成本 1%」均有原文对应句。
- 页面内部链接 `../moe-serving/index.html`、`../aux-loss-free-routing/index.html` 均存在；页面无指向 research/ 的路径，无「（待生成）」占位。
- `.dojo/scripts/validate.py wiki/quantile-balancing/index.html` 返回 `validation ok`。

## 问题

- [重要·技术] 来源与范围说明「论断与来源（C）」C2 与「公式与来源（F）」F2（index.html 第 742、749 行），及正文 §2 的 `<sup>[C2, F2]</sup>` 上游：「K3 报告 §2.3.3 引用 [30]」的引用编号错误，K3 报告该处引用的是 [27] 而非 [30]。｜引文依据：K3 报告 §2.3.3 原文「The original method updates $\bm b$ with the fixed-step rule $b_j^{(t+1)}=b_j^{(t)}+\gamma\operatorname{sign}(\bar{\ell}-\ell_j^{(t)})$ [27]」，同节「Kimi K3 adopts auxiliary-loss-free routing [27]」；报告参考文献 [27] =「DeepSeek-AI, A. Liu, B. Feng, ...」（即 DeepSeek-V3 报告），而 [30] =「(2026) DeepSWE benchmark (Website) Datacurve」，与 sign 更新无关。v1、v2 两版该处均引用 [27]。｜修复要求：把 C2、F2 两处的「引用 [30]」改为「引用 [27]」。论断本身（DeepSeek-V3 §2.1.2 的固定步长 sign 更新，overload 减 γ、underload 加 γ，等价于 $b_j+\gamma\operatorname{sign}(\bar\ell-\ell_j)$）已由 DeepSeek-V3 报告原文「we will decrease the bias term by $\gamma$ if its corresponding expert is overloaded, and increase it by $\gamma$ if its corresponding expert is underloaded」证实，无需改动；仅修正编号。｜修复：｜复验：

- [轻微·格式] 第 150 行（「最容易误解」列表项）、第 281 行（§3.3 正文段落）、第 604 行（§4.3 表格单元格）：数学符号未用 LaTeX 渲染，且同一符号全页写法不一致。第 150 行写「近 10³ 专家」，而第 401、607、782 行及 overview.html 同一量均写「近 $10^3$ 专家」；第 281 行正文写「取负就是 b̃：」，紧邻公式却用 $\tilde{b}$；第 604 行表格写「梯度方向 × 固定步长」，而第 766 行同一乘号写 $8 \times 4$。｜引文依据：不适用（规范依据 guides/concept/style-guide.md §11「数学符号一律写为 LaTeX……该要求覆盖……列表项、表格单元格……不因位置而放宽」「同一变量在页面中保持同一种写法」；validate.py 的 BARE_MATH_CHARS 未收录 U+00B3/U+00D7/U+0303，故机械校验放行）。｜修复要求：第 150 行「10³」改为 $10^3$；第 281 行「b̃」改为 $\tilde{b}$；第 604 行表格「×」改为 $\times$。｜修复：｜复验：

- [轻微·表述] 第 275 行「注意这里用的是……」、第 401 行「……但请注意：QB 是交替求解器的一轮更新」：元话语，直接指令读者「注意」，属于规范点名的「需要注意的是」类句式。｜引文依据：不适用（guides/concept/check.md 第 2.2 节第 12 项）。｜修复要求：改为不含指令读者的陈述句——第 275 行可直接陈述「这里用的是 raw score $s_{i,j}$ 减 cutoff $\alpha_i$……」；第 401 行删去「但请注意：」，改写为「这个例子一步就达到完美均衡，但这是本例的特殊性：QB 是交替求解器的一轮更新……」。｜修复：｜复验：

- [轻微·表述] 第 219 行「另一种情况中……」写作「另一个场景中负载 $\ell_j = 51$」：把「场景」当口语填充词使用，属规范点名的 AI 拼接腔用词。｜引文依据：不适用（guides/concept/check.md 第 2.2 节第 12 项「把『场景』当术语」）。｜修复要求：改为「另一种情况」或「若负载 $\ell_j = 51$（只多了 1 个）」。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 核对结论：核心机制、公式（Eq.13/14/20/23/25/26/27）、全部数字（896/top-16/B=1000/<1%/近 10³ 专家几步内收敛）、手算示例的每一步与代码输出均已回源核对无误；页面无内部矛盾、无构造示例被写成来源结论、无失效链接。唯一来源性缺陷是 C2/F2 的引用编号 [30] 应为 [27]。
- 处置：按上表修复 1 项重要与 3 项轻微后即可发布（修复后需重跑 validate.py 并复核第 150/281/604 行渲染效果）。
