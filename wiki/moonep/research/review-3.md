<!-- review-meta
round: 3
page: wiki/moonep/index.html
reviewed_content_sha256: 0c5a7f35e1719fe6
-->
# MoonEP 完美均衡专家并行审查记录（第 3 轮）

- 页面版本：53377fb50b5981c2433219e70086caa2ec99d32f
- 审查时间：2026-09-13 19:30
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节（顺序）：引言与符号说明 → 核心问题（4 条，含解答折叠块）→ 1. 传统 EP 的不均衡——根源与 MoonEP 的核心思路（含图 1、表 1、本章问题）→ 2. 冗余专家的界——$E/R$ 上界与基本紧性（含 F1、Theorem 1/2、两个折叠块、本章问题）→ 3. 完美均衡的工程收益——buffer、host 同步与 forward/backward 流程（3.1–3.4、图 2、伪代码折叠块、表 2、本章问题）→ 4. MoonEP 的边界——解决与不解决，以及与 ECHO/UltraEP/DeepEP 的区别（4.1–4.4、表 3、本章问题）→ 来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制 / 全文总结）。overview.html 全文亦已阅读。

## 来源核对方式与逐条定位（原文片段）

来源材料：Kimi K3 Technical Report（`https://github.com/MoonshotAI/Kimi-K3` 的 `k3_tech_report.pdf`，逐段比对 §2.2、§2.3、§2.3.3、§5.2、§5.2.1、附录 §E）；MoonEP 官方仓库 README（`https://github.com/MoonshotAI/MoonEP`）；页面内链目标页 `wiki/moe-serving/index.html`、`wiki/gpu-execution-model/index.html` 均实际存在，链接文字与标题一致。

- C1（不均衡 + 显存碎片，§5.2.1 开头）：报告 "The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation." ✔ 页面表述一致。
- C2（保留 DeepEP 计算流 + 在线规划迁移，§5.2.1 首段）："MoonEP preserves the overall computation flow of conventional schemes such as DeepEP [149] and additionally introduces online planning and migration of redundant experts." ✔
- C3（每 rank 收 $S\times K$，§5.2.1 "Perfect balance with bounded redundant experts" 段）："MoonEP requires every rank to receive exactly S × K tokens, where S is the sequence length and K is the number of experts selected per token" ✔
- C4（在线规划/ILP/近最优 kernel，§5.2.1 "Online planning" 段）："Computing the exact optimum at every training step is prohibitively expensive. We therefore compute exact solutions offline with integer linear programming (ILP) … and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the E/R upper bound." ✔
- C5（backward 暂存 + reduce 回 home rank，§5.2.1 首段）："In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks." ✔
- C6（fused permute/unpermute + buffer view 免拷贝，§5.2.1 "Zero-copy communication" 段）："…tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies." ✔
- C7（DeepEP 最坏 $S\times K\times R$，同段）："Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size S × K × R, whereas MoonEP requires only a fixed S × K buffer owing to the perfect balance." ✔ 页面已标"最坏不均衡"限定。
- C8（静态形状消除每层 host 同步，§5.2.1 "Sync-free execution with static shapes" 段）："This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead." ✔
- C9（per-expert 偏斜 + Expert-GEMM scheduler + shared expert 独立 stream，§5.2.1 "Expert-GEMM scheduling and overlap" 段）："Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed… we dispatch their GEMMs to a separate stream so that they overlap with other kernels." ✔（§5.2.1 末段定位正确）
- C10（ECHO/UltraEP cap，§5.2.1 "Perfect balance…" 段）："prior work such as ECHO [139] and UltraEP [134] presets the number of redundant experts or imposes a per-rank token cap. Training is then forced to stop whenever no feasible plan exists within the cap…" ✔
- F1（min-max 目标，§E 开头）："M (I) = minP maxr {mr (P )}" ✔
- F2（Theorem 1，§E）："We prove that M (I) ≤ E/R always holds (Theorem 1)…" 关键引理"the remote tokens of each rank come from only one other EP rank"、"This is repeated until all ranks are perfectly balanced"、"the process terminates after at most R − 1 fills" ✔ 页面复述一致，公式 (28) 编号一致。
- F3（Theorem 2，§E）："there exist router outputs for which M = ⌈E(R − 1)/R2 ⌉ ≈ E/R"；"each expert receives SKR²/(E(R−1)) tokens"；"rank 0 requires at least ⌈E(R−1)/R²⌉ redundant experts" ✔
- F4（总量守恒）：页面自标"由 C3 与 token-expert pair 总量守恒直接推出，无独立报告来源" ✔ 诚实标注。
- N（2.8T / 3T-class / §2.3.3）：§2.3 "This enables Kimi K3 to scale channel mixing to 896 routed experts with 16 active experts per token"；§2.3 含 "2.8-trillion-parameter scale"；§2.3.3 "Unlike auxiliary-loss-based routing, Kimi K3 adopts auxiliary-loss-free routing." ✔（唯一问题见下 N 条目）

数值复算（全部正确）：$E=4,R=2,S=4,K=1$ 下 $E/R=2$、$S\times K=4$、$S\times K\times R=8$ 均一致；情形 A 迁移 4 个 pair 需 1 个冗余专家（$1\le 2$）；情形 B $\lceil E(R-1)/R^2\rceil=\lceil 4\times1/4\rceil=1<E/R=2$；每个被使用专家 $SKR^2/(E(R-1))=16/4=4$，$\lceil SK/4\rceil=1$。算式与结论相符，符号 $S,K,E,R,M(I),m_r(P)$ 全文写法一致。伪代码为 `<pre><code class="language-text">` 且正文标为"伪代码"，非可运行代码，无需执行核对。KaTeX 定界符合规；`validate.py wiki/moonep/index.html` 返回 "validation ok"。

## 问题

- [重要·技术] index.html §4.2「MoonEP 不解决的三个问题」第 3 项（第 421 行）与 §4.1（第 415 行）、overview.html 第 48 行｜来源：K3 报告 §5.2.1；MoonEP 官方仓库 README｜位置：index.html 第 415、421 行；overview.html 第 48 行｜问题：把"内存碎片"列为 MoonEP 不解决的问题并断言"与完美均衡无关、由 §5.2.2 独立处理"。报告把"激活形状动态变化导致的显存碎片"作为 token 负载不均衡的直接后果，并由 MoonEP 的静态形状消除；§5.2.2 处理的是显存预算（激活/梯度/优化器状态），不是该碎片。页面自己的 §1（第 112 行）据 [C1] 把"routed-expert 激活形状动态变化导致显存碎片"列为报告的两个直接后果之一，与 §4.2 结论互相矛盾｜引文依据：报告 §5.2.1 "The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation."；"With perfect balance, every rank receives exactly S × K tokens and the computation shapes of all layers are statically known."；仓库 README "fully static memory shapes mean no fragmentation, and training never OOMs."｜修复要求：把"内存碎片"移出"不解决"清单，改写为"MoonEP 通过静态形状消除 routed-expert 激活形状动态变化造成的显存碎片；§5.2.2 处理的是另一类问题（激活/梯度/优化器状态的显存预算，含单内存池避免 multi-stream 碎片）"，并同步修正 overview.html 第 48 行与 index §4.1 措辞，使 §1/§4.1/§4.2 三处归类一致｜修复：｜复验：

- [重要·技术] index.html §外部数字与实验条件（N）（第 496 行）｜来源：K3 报告 §2.2、§2.3｜位置：index.html 第 496 行｜问题：断言"报告未公开 K3 训练 MoE 的具体 $E, R, K, S$ 取值"，但报告公开了模型层 $E$ 与 $K$：routed expert 总数 896、每 token 激活 16 个；未公开的只是训练侧 EP size $R$、每 rank 序列长度 $S$ 等配置。把已公开的 $E$、$K$ 写成未公开，属与官方材料不符｜引文依据：报告 §2.3 "This enables Kimi K3 to scale channel mixing to 896 routed experts with 16 active experts per token, corresponding to a sparsity of 56."；§2.2 "effectively activating 16 of 896 routed experts for each token"｜修复要求：改为"报告公开了模型层的 $E=896$、$K=16$（§2.3），但未公开 K3 训练时的 EP size $R$、每 rank 序列长度 $S$ 等具体配置；本页所用 $E=4,R=2,S=4,K=1$ 均为教学构造，不代表真实配置"｜修复：｜复验：

- [重要·技术] overview.html 第 47 行｜来源：MoonEP 官方仓库 README｜位置：overview.html 第 47 行｜问题：断言"MoonEP 是训练方案，不适用于推理场景"，与官方仓库矛盾——仓库明确给出推理用法（允许 $B<E/R$，推荐 3–4 个 prefetch 槽位）。把"本页范围（只讲训练）"写成"MoonEP 不适用推理"｜引文依据：仓库 README "Inference (prefetch only, no gradients): `B < E/R` is allowed, **`B = 3–4` is recommended**."｜修复要求：删除"不适用于推理场景"的断言，改为范围声明（如"本页只讲训练场景；推理场景见 MoE 推理与服务基础"），不替 MoonEP 划定适用范围｜修复：｜复验：

- [轻微·表述] index.html 第 79 行、第 112 行｜来源：不适用｜位置：index.html 第 79、112 行｜问题：以第一人称"我持有的 $E/R$ 个专家…"作会话指代，违反 check.md 表述维度（会话指代：我/我们/你）与 style-guide §12｜引文依据：不适用｜修复要求：改为"该 rank 持有的 $E/R$ 个专家…"｜修复：｜复验：

- [轻微·表述] index.html 第 106 行、第 261 行｜来源：不适用｜位置：index.html 第 106、261 行｜问题：元话语——"下面先用一个最小例子把"不均衡"具体化""接下来看完美均衡带来的三个工程收益怎样落地"｜引文依据：不适用｜修复要求：改为直接陈述，如"以 $E=4,R=2,S=4,K=1$ 为例，把不均衡具体化""完美均衡带来三个工程收益"，去掉"下面先用/接下来看"｜修复：｜复验：

- [轻微·表述] index.html 第 161、232、384、443 行｜来源：不适用｜位置：index.html 第 161、232、384、443 行｜问题：四章末过渡使用同一固定句式"本章说明了……。但……——下一章讲……"，违反 style-guide §8"不使用固定句式，也不为形式完整而添加过渡"｜引文依据：不适用｜修复要求：按各章实际结论改写为互不相同的过渡句，去掉统一的"本章说明了 X。但 Y——下一章讲 Z"模板｜修复：｜复验：

- [轻微·技术] index.html §论断与来源（C）（第 475–483 行）｜来源：K3 报告｜位置：index.html 第 477–484 行｜问题：来源清单要求"具体定位"，但 C3、C4、C5、C6、C8、C9、C10 条目只复述论断、未给出报告内位置（章节或段落名），无法按 check.md 2.2 定位核对（仅 C1/C2 给了"§5.2.1 开头/§5.2.1"）｜引文依据：报告内对应位置：§5.2.1 "Perfect balance with bounded redundant experts" 段（C3）、"Online planning" 段（C4）、首段 backward 句（C5）、"Zero-copy communication" 段（C6、C7）、"Sync-free execution with static shapes" 段（C8）、"Expert-GEMM scheduling and overlap" 段（C9、C10）｜修复要求：为上述每条 C 补报告内具体段落名（或章节号）｜修复：｜复验：

- [轻微·可读性] index.html 第 421 行｜来源：不适用｜位置：index.html 第 421 行｜问题："（这是常见误解，见下方边界）"位于「4.2 MoonEP 不解决的三个问题」（即边界章内），"见下方边界"自指、指向不明；页面又未按 style-guide §2 设 misconceptions 块交代该常见误解的内容，读者无法据此定位误解｜引文依据：不适用｜修复要求：去掉自指或改为明确指向具体小节标题，并在此处用一句说明该误解为"router 均衡 ≠ EP 后 rank 均衡"｜修复：｜复验：

- [轻微·格式] index.html 第 6 行（`description` 元信息）｜来源：K3 报告 §5.2.1｜位置：index.html 第 6 行｜问题：description 把"workload-aware GEMM 调度"与 MoonEP 的三项收益并列为其特性，但正文 §4.2 明确该调度处理的是 MoonEP 不解决的 rank 内 per-expert 偏斜，元信息列表与正文边界陈述不一致｜引文依据：报告 §5.2.1 "Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed, and a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers."｜修复要求：从 description 收益列表中移除"workload-aware GEMM 调度"，或改写为"并说明其不解决的 per-expert 偏斜由 workload-aware GEMM 调度处理"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 6
- 处置：修复
- 说明：核心结论（完美均衡使每 rank 恰收 $S\times K$、$E/R$ 上界与基本紧性、buffer 从 $S\times K\times R$ 降到 $S\times K$、免每层 host 同步、forward/backward 流程）与报告 §5.2.1/§E 原文逐句一致，全部数值示例可复算且正确，两级问题块均有解答且指向所在章节，来源与范围说明六节齐备，内链目标页真实存在，`validate.py` 通过——故无阻断项。3 项重要问题均为来源一致性/边界陈述问题（主要涉及 MoonEP 的作用范围），需按修复要求逐条改正并重新对照报告与仓库复核。