<!-- review-meta
round: 7
page: wiki/muon-optimizer/index.html
reviewed_content_sha256: 595e9eb955ef4bdc
-->
# Muon 优化器审查记录（第 7 轮）

- 页面版本：e79170586d1ac9cbc47f2141feb4276563271a7e（index.html 工作树哈希）
- 审查时间：2026-09-14 17:07
- 审查者：编排者派发的独立审查者（未参与写作与历次修复）
- 页面类型：concept（依据 `<meta name="dojo:type" content="concept">`），适用规范 `guides/concept/check.md`，格式规范 `guides/concept/style-guide.md`
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 动量更新矩阵为什么需要正交化——条件数高与稀有方向被忽略 / 2. Newton-Schulz 迭代如何近似正交化——只用矩阵乘法把奇异值推向 1（含 2.1 手算例子及全部折叠块与图注） / 3. Muon 的完整更新流程与几何含义——动量、正交化与参数更新（含 3.1 与 SGD-momentum、AdamW 的几何区别 / 3.2 与 Shampoo 的关系 / 3.3 工程开销与外部证据） / 4. Muon 的适用边界——哪些参数用 Muon，哪些仍用 AdamW（含 4.1 QKV 应分开应用 / 4.2 RMS 对齐缩放因子） / 来源与范围说明。全文（含折叠块）逐段通读，未用关键词检索代替阅读。

## 问题

- [轻微·格式] 第 200 行与第 208、430 行（及第 3 章更新流程、来源说明 N5）：NS 迭代步数在页面中先后使用两个记号——「应用 $N$ 步 NS 迭代的输出为 $U\,\phi^N(S)\,V^\top$，其中 $\phi^N$……共 $N$ 次（$N$ 即 NS 迭代步数，默认取 5；后文记号 $T$ 表示同一量）」（200 行），随后「默认迭代步数 $T = 5$」（208 行）、「$T \cdot m / B$（$T=5$ 为迭代步数）」（430 行）。同一变量在页面中用 $N$ 与 $T$ 两种写法，与 check.md 2.2 第 9 条「同一变量全页写法一致」不符；页面虽有一句衔接说明，读者面仍是两个记号指代同一量。｜引文依据：不适用（格式类）｜修复要求：将迭代步数统一为一个记号（建议全程用 $T$，把 $\phi^N(S)$ 改为 $\phi^T(S)$ 并在首次出现处给一次定义），删除「后文记号 $T$ 表示同一量」的桥接句；改动后核对第 200、208、430 行及来源说明 N5 的写法一致｜修复：｜复验：

## 核对依据摘录（支撑本轮「无阻断/无重要」判定）

- 主源博客 Keller Jordan 等《Muon: An optimizer for hidden layers in neural networks》(2024, https://kellerjordan.github.io/posts/muon/)：
  - 单步公式与 SVD 等价：「G' = aG + b(GGᵀ)G + c(GGᵀ)²G = U(aS + bS³ + cS⁵)Vᵀ」，与页面 181/187 行一致。
  - 系数：基线 (2, −1.5, 0.5)；调优 (3.4445, −4.7750, 2.0315)，「found by an ad-hoc gradient-based search」，代码默认 steps=5, eps=1e-7；调优原则「make a as large as possible (since φ'(0) = a …)」与「settling in [0.7, 1.3] — i.e. ε up to about 0.3 is tolerable」，与页面 208、211、261 行一致。
  - 归一化：「we simply replace G by G/‖G‖_F before starting the NS iteration」＋「This rescaling is benign because Ortho(cG) = Ortho(G)」；矩阵高于宽时转置（undo at the end），与页面 202–206、218、354、372 行一致。
  - 条件数：「SGD-momentum and Adam updates on 2D transformer parameters have 'very high condition number' — nearly low-rank, with a few directions dominating all neurons」；「Orthogonalization is speculated to lift the scale of small-but-important 'rare directions.'」——页面 136、142、158 行按「经验观察 + 作者推测」如实标注，未升格为结论。
  - bfloat16 / coupled Newton：「can be stably run in bfloat16」「must run in at least float32 precision to avoid numerical instability」「Full SVD … far too slow」，与页面 214 行一致。
  - FLOP：「each NS step … costs 2(2nm² + m³) matmul FLOPs (≤ 6nm² if square)」；Bernstein/You/Cesista「improving the iteration's efficiency from 6nm² to 4nm² + 2m³」；「per-training-step overhead is at most Tm/B」；NanoGPT m=768, B=524288 → 0.7%，Llama 405B m=16384, B=16,000,000 → 0.5%，与页面 430 行一致。
  - 外部数字：「record on 10/15/24 with a 35% training-speed gain … remained the optimizer across all 12 subsequent records, set by 7 different researchers」；「GPT-2 XL-level HellaSwag performance in 10 8×H100-hours, versus 13.3 hours for AdamW」；「CIFAR-10 … from 3.3 to 2.6 A100-seconds」，与页面 432 行一致。
  - Shampoo：「If preconditioner accumulation is removed … Bernstein & Newhouse (2024) observed that the update becomes … Which is the orthogonalized gradient」；「(GGᵀ)^{−1/4} G (GᵀG)^{−1/4} = UVᵀ」，与页面 412–426 行一致，归因（Bernstein & Newhouse 2024，另见 Anil 2024a）也一致。
  - 参数分组：「AdamW should be used for the embedding and final classifier head layers」「follows from the modular norm theory (Large et al. 2024)」「does not seem to follow from the theory … driven by empirics」；QKV「performs better when Q, K and V are optimized separately」（Vlado Boza），与页面 478、482 行一致。
  - Citation 作者：Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein，与页面 N12 一致。
- Moonshot AI《Muon is Scalable for LLM Training》(arXiv:2502.16982) 第 2.2 节「Matching update RMS of AdamW」：Eq.4「W_t = W_{t-1} − η_t(0.2·O_t·√max(A,B) + λW_{t-1})」，Lemma 1 给出满秩 [A,B] 矩阵「√(1/max(A,B))」；AdamW 典型 update RMS「usually around 0.2 to 0.4」；「Muon can directly reuse the learning rate and weight decay tuned for AdamW」。与页面 486–505 行一致（页面省略权重衰减项，已在「简化条件及其限制」说明）。
- modular norm theory 文献：arXiv:2405.14813 = Tim Large 等《Scalable Optimization in the Modular Norm》(2024)，与页面 552 行一致。
- PyTorch 官方文档 `torch.optim.Muon`（docs.pytorch.org，2.9 起内置）：构造默认 `lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None`，且「If not specified, we will default to use "original"」，`'match_rms_adamw'` 为 Moonshot RMS 对齐选项。与页面 507–520 行默认超参表、505 行、N11 说明一致。

## 公式与代码复算

- 页面 numpy 代码实跑（本机 python3 + numpy，float64）：输出与页面「预期输出」逐行一致——`||G||_F = 0.905539`、`谱归一化后奇异值 = [0.9939 0.1104]`、`基线系数 5 步后奇异值 = [1.0001 1. ]`、`调优系数 5 步后奇异值 = [0.7529 0.7034]`、`基线 O^T O = [[1. 0.] [0. 1.0003]]`。
- 手算表（232–246 行）按全精度初值 0.99388/0.11043 逐位迭代复算：0.9939/0.1104 → 1.0000/0.2189 → 1.0000/0.4222 → 1.0000/0.7383 → 1.0000/0.9826 → 1.0000/1.0001，与表格四位小数逐格一致；调优系数第一步 0.3740 亦复算一致。
- 折叠块推导复算：$GG^\top = US^2U^\top$、$(GG^\top)^2 = US^4U^\top$ → $U(aS+bS^3+cS^5)V^\top$（191–198 行）成立；Shampoo 等价 $(US^{-1/2}U^\top)(USV^\top)(VS^{-1/2}V^\top) = UV^\top$（416–424 行）成立；$\|UV^\top\|_F^2 = r$ 的双重求和展开（494–503 行）成立；$\sqrt{r/(nm)} = \sqrt{1/\max(m,n)}$ 成立。$2 \times 2$ 例：$\gamma = 0.2\sqrt2 \approx 0.283$，$\operatorname{RMS}(I_2) = 0.7071$，缩放后 $0.2$，均与 505 行一致。
- `python3 .dojo/scripts/validate.py wiki/muon-optimizer`：`validation ok`。`dojo:topics=训练与优化` 在 AGENTS.md 固定大类词表内；`dojo:type=concept`、`description`、`dojo:summary`、`dojo:tag` 齐备，`dojo:summary` 不含公式无渲染风险。

## 其他检查

- 引文编号：正文出现的 C1–C9、F1–F7、N4/N6/N7/N8/N9/N10/N11/N12 均在「来源与范围说明」有对应定义，编号与所引来源（C/F/N 分类、博客与 Moonshot 归属）一致；无悬空编号、无同号两义。
- 页面链接：`../newton-schulz/index.html`、`../svd/index.html` 均真实存在（已确认文件在库）；`overview.html` 与 `index.html` 相互链接；无「（待生成）」占位。
- 数学符号：公式定界符外仅 `index.html:661` 的 `·`（JS 阅读时长分隔符）与 364–384 行伪代码块内的 `β ε ᵀ ≤ ‖ ·` 出现；后者属 `<pre><code>` 代码块，style-guide §11 明确「代码块内的字符按代码原样保留，不受此约束」，不构成违例。标题/summary/正文/列表/表格/callout 内无 Unicode 数学字符。
- alt 属性：全页仅 lightbox 占位 `alt=""`，无 `$...$`。
- 表述：无第一人称复数、无直接称呼读者、无「下面来看/需要注意的是」式元话语、无调试叙事与临场评价；「本文」「本章」的使用符合 style-guide §12（自称用「本页」或「本文」）与问题块固定命名要求，未按缺陷处理。示例标记「构造示例」与来源章 h3 固定命名及站内多数页一致，非缺陷。
- 图：本页无结构图/内联 SVG（`.diagram` 样式类定义存在但未使用），不涉及图注读数或 `<text>` ASCII 近似问题。
- 交互：目录、折叠、复制、lightbox 均由脚本增强，无脚本时正文仍可完整阅读；无「交互视图无脚本时不可读」问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布。核心结论、公式、代码输出与外部数字均回源核对通过；唯一遗留为上述 1 处轻微记号统一问题（不影响正确性与主线理解，可在本轮修复中一并处理，或接受为不影响发布的轻微项）。

统计：阻断 0 / 重要 0 / 轻微 1