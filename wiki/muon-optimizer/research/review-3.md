<!-- review-meta
round: 3
page: wiki/muon-optimizer/index.html
reviewed_content_sha256: 985a85b2a63ecaa5
-->
# Muon 优化器审查记录（第 3 轮）

- 页面版本：55667d94f588d970b70ec1683c60a0419c85302e
- 审查时间：2026-09-13 19:09
- 审查者：独立子代理
- 规范：guides/concept/check.md（dojo:type=concept），另对照 guides/concept/style-guide.md
- 已完整阅读章节（按顺序）：blockquote.meta → 引言 → 核心问题（4 题）→ 最容易误解 → 1. 动量更新矩阵为什么需要正交化——条件数高与稀有方向被忽略（含本章问题）→ 2. Newton-Schulz 迭代如何近似正交化——只用矩阵乘法把奇异值推向 1（含 2.1 手算例子、三个折叠块、本章问题）→ 3. Muon 的完整更新流程与几何含义——动量、正交化与参数更新（含 3.1/3.2/3.3、代码折叠块、本章问题）→ 4. Muon 的适用边界——哪些参数用 Muon，哪些仍用 AdamW（含 4.1/4.2、本章问题）→ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）→ head meta 与脚本
- 核对来源：Keller Jordan 等《Muon: An optimizer for hidden layers in neural networks》（2024，https://kellerjordan.github.io/posts/muon/，已抓原文逐句核对）；Moonshot AI《Muon is Scalable for LLM Training》（arXiv:2502.16982，已取 PDF 正文核对）；PyTorch `torch.optim.Muon` 官方文档（2.9 内置，已核对默认值与 release notes）。
- 机械验证：`.dojo/scripts/validate.py wiki/muon-optimizer/index.html` → validation ok；2.1 节 numpy 代码实际执行，输出与页面"预期输出"逐行一致；核心公式全部复算通过；内链 ../newton-schulz/index.html、../svd/index.html 均真实存在；index.html 与 overview.html 相互链接。

## 问题

- [重要·技术] 「核心问题」第 1 题解答（正文开篇 learning-goals 第 1 条）：把被正交化的对象写成 $M_t$，与第 1、3 章的符号定义直接冲突，同一符号 $M_t$ 在页内指代两个不同对象｜引文依据：核心问题解答"Muon 先按 Nesterov momentum 累积动量得到 $M_t$，再对 $M_t$ 做 Newton-Schulz 正交化"；第 1 章"$M_t = \beta\, M_{t-1} + G_t$，$M_0 = 0$"；第 3 章"先构造前瞻动量 $\tilde{M}_t = G_t + \beta\, M_{t-1}$，再更新 $M_t = \beta\, M_{t-1} + G_t$"、"前瞻动量 $\tilde{M}_t$ 是被正交化的对象，最终的更新方向是 $O_t \approx UV^\top$ 而非原始动量 $M_t$"；博客原文"Muon moves the momentum to before the orthogonalization."｜修复要求：把核心问题第 1 题解答中的"累积动量得到 $M_t$，再对 $M_t$ 做 Newton-Schulz 正交化"改为"得到前瞻动量 $\tilde{M}_t$，再对 $\tilde{M}_t$ 做正交化"，与第 3 章符号一致；改后全页 $M_t$ 仅指动量缓冲。｜修复：｜复验：

- [轻微·技术] 2.1 手算例子正文与「展开：完整 5 步手算过程」表格第 1 步大奇异值：$\phi(0.9939)$ 复算为 1.0000178，按 4 位小数应为 1.0000，页面写作 1.0001｜引文依据：正文"$\phi(0.9939) = 2(0.9939) - 1.5(0.9939)^3 + 0.5(0.9939)^5 \approx 1.0001$"，表格"第 1 步 大奇异值 1.0001"；复算 $2(0.9939)-1.5(0.9939)^3+0.5(0.9939)^5 = 1.0000178 \to 1.0000$｜修复要求：将该处正文与表格第 1 步大奇异值改为 1.0000；并统一口径——2.1 正文以舍入后的 0.9939/0.1104 为输入，而表中 0.2189/0.4222/0.7383 由全精度初值 $0.9/\sqrt{0.82}$、$0.1/\sqrt{0.82}$ 算出，需注明所用初值或统一取整方式。｜修复：｜复验：

- [轻微·技术] 第 2 章"调优原则有两条"一条、同章 callout 与「最容易误解」第 2 条：把 $[0.7,\ 1.3]$ 写成"对所有 $x \in [0,1]$"成立的普遍性质，实际对调优系数多项式不成立｜引文依据：页面"以及对所有 $x \in [0,1]$，允许 $\lim_{N\to\infty}\phi^N(x) \in [0.7,\ 1.3]$"；博客"our goal will be to maximize $a$ subject to $\lim_{N\to\infty}\varphi^N(x) \in [0.7,1.3]$"（同一目标，且前文为"For every $x\in[0,1]$, we want $\varphi^N(x)$ to converge to a value in $[1-\varepsilon,1+\varepsilon]$"）；复算：调优系数 $\phi(x)=3.4445x-4.7750x^3+2.0315x^5$ 在 $x\in[0,1]$ 上的极限集为 4-循环 $\{0.6819, 1.1343, 0.7530, 1.0467\}$，最小 0.6819 $< 0.7$，且 $x\to 0$ 时趋于 0。｜修复要求：把该条改写为"设计目标/期待区间"，或注明它在 $x$ 接近 0 与个别区间不严格成立；不得表述为对所有 $x \in [0,1]$ 严格的数学性质。｜修复：｜复验：

- [轻微·技术] 3.2「与 Shampoo 的关系」与「辅助解释与类比边界」：Shampoo 等价陈述漏掉源文的"关闭动量"限定；F6 归属只记"Keller Jordan 2024 博客"，未记博客所给出的原始出处｜引文依据：博客"It is therefore possible to interpret Muon with momentum turned off as a kind of 'instantaneous' or 'accumulation-free' Shampoo"、"If preconditioner accumulation is removed, then Bernstein & Newhouse (2024) observed that the update becomes the following (also see Anil (2024a))"；页面"Muon 可视为'无累积 Shampoo'"、"F6（Shampoo 无累积 = $UV^\top$）：均来自 Keller Jordan 2024 博客"。公式本身核对无误：博客 $(G_tG_t^\top)^{-1/4}G_t(G_t^\top G_t)^{-1/4}=UV^\top$ 与页面一致，页面折叠块推导逐步正确。｜修复要求：等价陈述改为"（关闭动量时）Muon 的更新方向与无累积 Shampoo 等价"；在 3.2 与 F6 补记 Bernstein & Newhouse (2024) 的归属。｜修复：｜复验：

- [轻微·格式/来源] head 的 `<meta name="description">` 与 `<meta name="dojo:summary">`：写入"K3 的 Per-Head Muon"，但本页正文既未介绍也从未链接该主题，且"K3"归属在页内无来源｜引文依据：description"……K3 的 Per-Head Muon 是其按头切分改进。"；dojo:summary"K3 的 Per-Head Muon 将正交化从整块投影矩阵改为逐头执行。"；全页检索 "Per-Head"/"K3"/"按头" 仅命中上述两行 meta（正文 0 处），相关链接只出现在 overview.html。｜修复要求：description 与 dojo:summary 只描述本页内容，删去 Per-Head Muon 一句；若确需保留，改为在正文给出指向 per-head-muon 页的链接并补来源，或移入 overview。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」→ 外部数字与实验条件（N）：N12 在来源清单中定义但正文无任何 [N12] 引用，不符合来源编号双向对应｜引文依据：正文检索无 [N12]；仅来源小节出现"N12（贡献者 Jordan、Jin、Boza、You、Cesista、Newhouse、Bernstein）：来自博客 Citation。"（核对该 7 人与博客 Citation 名单一致）｜修复要求：删除 N12，或在正文（如 blockquote.meta 或引言）加入对应 [N12] 引用。｜修复：｜复验：

## 核对通过的关键条目（供复验参照）

- C1 定义/MomentUm 展开、C2 仅隐藏层 2D 且 embedding/输出层用 AdamW（"That such dynamics are also different for the output layer does not seem to follow from the theory, and is instead driven by empirics."）、C3 动量在正交化前且 Nesterov 默认、C4 条件数高近似低秩（"based on manual inspection … typically have very high condition number"）、C5 稀有方向为 speculate、C6/C7/C8（QKV 分开出自 Vlado Boza；bfloat16 稳定）：均与博客一致。
- F1–F5 与博客逐字一致（$G'=aG+b(GG^\top)G+c(GG^\top)^2G$、$U\phi(S)V^\top$、$U\phi^N(S)V^\top$、$\mathrm{Ortho}(G)=\arg\min\|O-G\|_F$、$X=G/(\|G\|_F+\varepsilon)$）；F7 与 Moonshot 论文一致（Lemma 1：满秩 $[A,B]$ 矩阵的理论 Muon update RMS 为 $1/\sqrt{\max(A,B)}$；AdamW 的实际 update RMS 约 0.2–0.4；缩放式 $W_t=W_{t-1}-\eta_t(0.2\cdot O_t\sqrt{\max(A,B)}+\lambda W_{t-1})$）。
- 外部数字均复核：FLOP 开销 NanoGPT $5\times768/524288=0.7\%$、Llama 405B $5\times16384/16000000=0.5\%$、每步 $2(2nm^2+m^3)$、上界 $Tm/B$；10/15/24 提速 35%、"12 of the new NanoGPT speedrunning records … 7 different researchers"；1.5B 10 vs 13.3 8xH100-hours；CIFAR-10 3.3→2.6 A100-seconds。PyTorch 默认 lr=1e-3、weight_decay=0.1、momentum=0.95、nesterov=True、ns_coefficients=(3.4445,-4.775,2.0315)、ns_steps=5、eps=1e-7、adjust_lr_fn 默认 'original'（另一取值 'match_rms_adamw' 即 Moonshot RMS 对齐）——与官方文档一致，Muon 于 PyTorch 2.9 引入。
- 代码：嵌入的 numpy 代码实际运行，输出与页面"预期输出"完全一致（`||G||_F = 0.905539`、`[0.9939 0.1104]`、基线 `[1.0001 1.]`、调优 `[0.7529 0.7034]`、`O^T O ≈ I`）。
- 表述维度：未发现元话语、会话指代（我/我们/你）、调试叙事、临场评价或 AI 拼接腔；两级问题块命名正确（核心问题 / 本章问题），每题均有 `解答：` 折叠块且答案独立可读（除上述第 1 条符号冲突外）。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 5
- 处置：修复（重要问题 1 条需关闭；轻微问题可一并修）
