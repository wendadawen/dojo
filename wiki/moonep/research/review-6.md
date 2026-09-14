<!-- review-meta
round: 6
page: wiki/moonep/index.html
reviewed_content_sha256: f5fd223124c41b4d
-->
# MoonEP 完美均衡专家并行审查记录（第 6 轮）

- 页面版本：9d6e28005a9e34b4c66fcf4661c6cb0cfc946239（index.html 工作树哈希）
- 审查时间：2026-09-13 21:14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：引言与「核心问题」块 → 1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路 → 2. 冗余专家的界——$E/R$ 上界与基本紧性 → 3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程 → 4. MoonEP 的边界——解决与不解决，以及与 ECHO/UltraEP/DeepEP 的区别 → 来源与范围说明；含全部 details 折叠块（Theorem 1 证明复述、$E=4,R=2$ 手算展开、forward 伪代码）与两处图注
- 来源获取：Kimi K3 Technical Report（arXiv:2607.24653，全文 HTML 下载后逐段定位 §5.2.1、§5.2.2、§E、§2.3、§2.3.3）；MoonEP 官方仓库 https://github.com/MoonshotAI/MoonEP（WebFetch 核对存在性与 README 描述）；同页 overview.html 与两处前置概念链接

## 问题

- [轻微·格式] h1 标题（第 63 行）与第 1 章图注节点（第 132 行）：数学变量与运算符直接以 Unicode 字符书写，未包在 `$...$` 中——标题作「恰好收到 S×K token」，图注作「过载 2×」；而全页其余位置（含 h3「3.1 通信 buffer 从 $S\times K\times R$ 降到 $S\times K$」、正文、表格、summary）一律写作 `$S\times K$`。违反 check.md 2.2 第 9 条「标题、summary、正文、列表和表格中无 Unicode 数学字符直接出现；同一变量全页写法一致」与 style-guide 第 107 行。｜引文依据：`<h1 class="title">MoonEP 完美均衡专家并行：让每个 EP rank 恰好收到 S×K token</h1>`；`<div class="dg-node-note">收 8 个 pair——过载 2×</div>`｜修复要求：标题与图注内的 `S`、`K`、倍率改为 `$S\times K$`、`$2\times$`（或 `2 倍`）由 KaTeX 渲染；`<head>` 的纯文本 description 不在本项范围内，保持原样即可。｜修复：｜复验：
- [轻微·技术] 「来源与范围说明 › 外部数字与实验条件（N）」（第 498 行）：括号内把 `"3T-class parameters"` 标注为 §5.2 的原文，但该字符串在报告中只出现一次，位于 §1 引言（"scaling the pre-trained foundation to unprecedented 3T-class parameters"）；§5.2 只以 `3T-class` 指代规模（小节标题 "Infra for 3T-class Pre-Training"，正文 "Natively multimodal pre-training at the 3T-class poses three critical problems"），并不含 "3T-class parameters"。标注位置与实际出处不符。｜引文依据：报告全文 6 处 `3T-class` 中，`3T-class parameters` 仅见于引言「scaling the pre-trained foundation to unprecedented 3T-class parameters」；§5.2 正文为「Natively multimodal pre-training at the 3T-class poses three critical problems」。｜修复要求：把该括注拆开——`§1 引言以 "3T-class parameters" 指代其模型规模；§5.2 小节标题亦用 "3T-class"`，或删去引号原文只保留「§5.2 以 "3T-class" 指代其模型规模」。｜修复：｜复验：
- [轻微·表述] 第 445 行（§4.4 比较表之后、本章问题之前）：`<p>公式与教学简化的来源定位见下文。</p>` 是纯指向句，未点明目标小节名，属为形式完整添加的过渡（style-guide §8「不使用固定句式，也不为形式完整而添加过渡」），且其所指章节实际在「本章问题」之后。｜引文依据：不适用（对照同仓 quantile-balancing 页同类过渡写作「……见下文「来源与范围说明」」，点明了目标小节）｜修复要求：改为点明目标小节（如「论断、公式与教学构造的来源定位及边界见『来源与范围说明』」）或直接删除该句。｜修复：｜复验：

## 核对记录（逐条来源论断）

- C1（§5.2.1 开头）：原文 "In conventional EP schemes, token loads are imbalanced across ranks. The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation." 页面 §1、§4.1 的「计算不均衡降低训练吞吐」「routed-expert 激活形状动态变化导致显存碎片」与之逐条对应，未扩大适用范围。页面另加「完美均衡消解前一条；碎片这条报告列为不均衡的后果但未声明被消除，rank 内 per-expert token 数仍随层偏斜」的限制，与 §5.2.1 "Expert-GEMM scheduling and overlap" 段 "the per-expert token counts within each rank remain skewed" 一致，属正确的保守表述。
- C2：原文 "MoonEP preserves the overall computation flow of conventional schemes such as DeepEP and additionally introduces online planning and migration of redundant experts." 及 forward "plan the redundant experts from the router outputs of the current micro-batch and layer and prefetch them before the routed-expert computation" — 页面引言、§1、§4.4 一致。
- C3/F2（"Perfect balance with bounded redundant experts" 段）：原文 "MoonEP requires every rank to receive exactly S×K tokens... We prove that a balanced plan always exists with at most E/R redundant experts per rank and that this bound is essentially tight"; "Reserving E/R redundant-expert slots per rank therefore guarantees that planning always admits a feasible solution, so training is never interrupted." — 页面 §1 末、§2、§2 折叠块一致。
- C4（"Online planning" 段）：原文 "Computing the exact optimum at every training step is prohibitively expensive. We therefore compute exact solutions offline with integer linear programming (ILP) for representative cases as references and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the E/R upper bound." — 页面 §3.3 第 1 步一致，「prohibitively expensive」直引无误。
- C5：原文 "In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks." — 页面 §3.4 两步与「只归约到 home rank、非全量 all-reduce」的限定一致。
- C6/C7（"Zero-copy communication" 段）：原文 "tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies. Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size S×K×R, whereas MoonEP requires only a fixed S×K buffer owing to the perfect balance." — 页面 §3.1、§3.3 保留了 "Under worst-case imbalance" 限定，并在 §3.1 与来源说明中明确「报告未说明 DeepEP 非最坏情况下的实际 buffer 分配策略」，未把条件观察写成无条件论断；$S\times K\times R \to S\times K$ 的 $R$ 倍节省算术成立。
- C8（"Sync-free execution with static shapes" 段）：原文 "the host must synchronize with the device at every layer to obtain the actual computation shapes before launching the expert computation, stalling the pipeline between layers... This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead." — 页面 §3.2 一致。
- C9（"Expert-GEMM scheduling and overlap" 段）：原文 "a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers... a workload-aware scheduler that adapts its parameters to the current token distribution before launch and keeps them fixed during execution. A lightweight heuristic selects these parameters using an analytical cost model of hardware metrics, with key coefficients calibrated through offline autotuning. For the shared experts, we dispatch their GEMMs to a separate stream" — 页面 §4.2 一致；该子节确为 §5.2.1 最后一个子节，页面「报告 §5.2.1 最后一段」表述成立。
- C10（同上段）：原文 "prior work such as ECHO and UltraEP presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists within the cap, and the cap itself requires manual tuning while still leaving residual imbalance." — 页面 §4.3 一致。
- F1/F2/F3（§E）：原文 "Let m_r(P) denote the number of redundant experts placed on rank r under plan P... M(I)=min_P max_r{m_r(P)}. We prove that M(I)≤E/R always holds (Theorem 1) and that this bound is essentially tight: there exist router outputs for which M=⌈E(R−1)/R²⌉≈E/R (Theorem 2)." — 页面 §2 的 $M(I)=\min_P\max_r\{m_r(P)\}$、Theorem 1、Theorem 2 三条陈述与公式一致；Theorem 2 构造中的 $SKR^2/(E(R-1))$ 与 $\lceil SK/(SKR^2/(E(R-1)))\rceil=\lceil E(R-1)/R^2\rceil$ 复算无误。
- §E Theorem 1 证明：原文 "Each fill makes one underloaded rank balanced and it never changes afterwards, so the process terminates after at most R−1 fills; meanwhile, each rank is filled at most once, so its remote tokens come from a single rank... these tokens belong to at most E/R local experts on rank s, hence m_r(P*)≤E/R" — 页面 §2 正文与折叠块复述一致；页面补充的「专家均匀分片」为 EP 常规前提，未改变结论。
- F4：$\sum_{r}n_r=S\times K\times R$ 由 C3 与 pair 总量守恒直接推出，页面已自标「无独立报告来源」，处理正确。
- N 段数字：§2.3 "896 routed experts with 16 active experts per token" 与 "2.8-trillion-parameter scale" 均在 §2.3 Stable LatentMoE 正文内（已按 HTML 偏移确认 §2.3 起止），页面 $E=896$、$K=16$ 与「2.8 万亿」一致；§2.3.3 "Kimi K3 adopts auxiliary-loss-free routing" 与 §5.2.2 "managed within a single memory pool, avoiding multi-stream fragmentation" 分别支持页面 §4.2 的两处引用。唯一出处偏差见上第 2 条。
- 构造示例复算（教学构造，页面已自标不来自报告）：$E=4,R=2,S=4,K=1$ → $E/R=2$、每 rank 发 $S\times K=4$ 个 pair、全网 $S\times K\times R=8$、平衡值 4；情形 A 迁 4 个同源 pair → 1 个冗余专家 $≤2$ ✓；情形 B 中 $E(R-1)/R=2$ 个专家均分 8 个 pair（每专家 4 个），rank 0 取同专家 4 个 pair → $M(I^*)=1=\lceil 4\times1/4\rceil<E/R=2$ ✓。全页无 a×b 与标注积不符、分项之和≠合计、正文与 summary/overview 数字冲突的情况。
- 代码：折叠块为 `language-text` 伪代码，未声称可运行，按静态审查核对；其 planning→预取→expert-grouped dispatch→返回 buffer view 的步骤与 §5.2.1 文字一致，无与正文矛盾的输出描述。
- 页面功能与结构：validate.py 返回 `validation ok`；两处前置概念链接（wiki/moe-serving、wiki/gpu-execution-model）均真实存在，无「待生成」占位；锚点 id 唯一；结构图为 HTML 结构（.dg-flow）非字符框线图，图内公式走 KaTeX，箭头含义与「rank 0/rank 1 并行目的地、非先后关系」已在图注定义；核心问题 4 条与四章「本章问题」均有解答折叠块，核心问题答案均指明完整论证所在章节；无 alt 含 `$...$`。
- 表述：通读全文（含折叠块与图注）未发现「本页将…/下面来看…/需要注意的是」式元话语、我/我们/你等会话指代、调试与复现踩坑叙事、临场评价，也未发现抽象名词堆叠或把「场景」当术语的 AI 拼接腔；「只讲训练场景」中的「场景」为限定语而非术语。第 445 行的空指向过渡见上第 3 条。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。本轮未发现阻断或重要问题：核心结论（每 rank 恰好收 $S\times K$、$E/R$ 上界与基本紧性、buffer/host 同步/forward-backward 三项收益、与 ECHO/UltraEP/DeepEP 的边界）逐条回源核对一致，公式与构造示例均可复算，来源定位（C1–C10、F1–F4）与标注章节相符。3 条轻微问题均不影响正确性与主线理解，接受理由：第 1 条为标题/图注写法一致性问题（KaTeX 渲染不受影响、语义无歧义）；第 2 条为来源说明内部一处引文出处括注偏差，不改变任何事实性论断；第 3 条为一处空指向过渡句，读者可按目录直达目标章节。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
