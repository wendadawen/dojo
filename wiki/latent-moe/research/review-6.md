<!-- review-meta
round: 6
page: wiki/latent-moe/index.html
reviewed_content_sha256: bcd9379dcccc36c2
-->
# LatentMoE 审查记录（第 6 轮）

- 页面版本：12c0449e3d82850ccd763df4254c9857319e2ed9
- 审查时间：2026-09-13 21:13
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查）
- 已完整阅读章节：核心问题、最容易误解、1. 扩大专家池的代价、本章问题、2. LatentMoE 的层结构、本章问题、3. 压缩比与 Reinvestment、本章问题、4. 压缩下限与投影成本、本章问题、5. 与 Stable LatentMoE 的关系、本章问题、来源与范围说明（含全部折叠块与两张数据流图）

## 问题

- [轻微·技术] dojo:summary 与标题：summary 写「扩大专家池时，通信量与专家权重带宽不再随专家数线性增长」、标题写「扩大专家池不让通信与显存带宽线性增长」，把增长驱动量记为「专家数」。正文 §1 与 §4 均明确驱动量是被激活专家数 $k$（每 token dispatch $\propto k\cdot d$，与专家总数 $N$ 无关），overview 也写作「扩大 $N$ 与 $k$」。摘要的「专家数」写法会让人误以为标准 MoE 的每 token 路由开销随 $N$ 增长，与正文自述不符。｜引文依据：[L] Figure 1 caption "reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"（是常数因子缩小，非「随 $N$ 的增长被消除」）；正文 §1「扩大路由倍数（$N$ 与 $k$ 同时放大）时，dispatch 通信量 $\propto k \cdot d$」。｜修复要求：summary 与标题改为以「激活专家数 / 路由倍数」为驱动量（如「扩大路由倍数时通信量不再按 $k\cdot d$ 线性增长」），与 §1、§4 及 overview 用词一致。｜修复：｜复验：
- [轻微·格式] §3（正文段与对比表）：同一变量 top-$k$ 全页写法不一致——正文段「$\ell$-MoE_acc 进一步同时放大专家数与 top-$k$，即 $N' = \alpha N$、$K' = \alpha K$」用大写 $K$，同段紧邻处又写「top-$k$」「$\binom{N}{k}$」「$\binom{\alpha N}{\alpha k}$」；紧随其后的构造示例写「$k' = \alpha k = 8$」、对比表写「$k \to \alpha k$」、§3 同段又写「保持 $K$ 与中间维度 $m$ 不变，$U_\text{eff}=K\cdot m$」与「放大 $k$ 时 $m$ 也不动」。$K$ 未出现在 §2 符号表中，读者无法判断 $K$ 是否即 $k$。违反 check.md 2.1.10「同一变量全页写法一致」。｜引文依据：不适用｜修复要求：全页统一为小写 $k$ / $k'$ / $N'=\alpha N,\ k'=\alpha k$；若保留 $K$ 以贴合论文记法（$U_\text{eff}\propto K\cdot m$），须在 $K$ 首次出现处注明 $K \equiv k$。｜修复：｜复验：
- [轻微·格式] 第 2 章 MLA 折叠块（`<details>` 正文首句）：正文以「辅助解释。 」开头，全角句号后紧跟半角空格，与页面其余折叠块（直接进入正文，如设计原则块、Nemotron-3 块）体例不一致，疑为遗留标签。｜引文依据：不适用｜修复要求：删除「辅助解释。 」前缀及其后半角空格，正文直接从「LatentMoE 的『down-projection…』」起句。｜修复：｜复验：

## 来源核对（关键论断与引文依据，本轮逐条回源）

- [L] Elango et al., "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts", arXiv:2601.18089（2026-01-26）：存在，标题、作者（Velmugil Elango 等，NVIDIA）一致。
- C1/C7（结构定义、router 在 $d$ 维）：§3 原文 "The routing weights p′=Softmax(W′_r·x) are computed from the original token x∈ℝ^d"；"all operations outside the routed experts—including the MoE routing mechanism and shared experts—continue to operate in the original hidden dimension d"。一致。
- C3/F3（开销按 $d/\ell$ 缩小）：Figure 1 caption 原文 "reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ"。一致。
- C4/F4（reinvestment）：§3 给出 ℓ−MoE_eff（放大 $N$、$K$ 不变）与 ℓ−MoE_acc（$N,K$ 同按 $\alpha=d/\ell$ 放大，"recommended"）定义式；"superior model accuracy at iso-inference cost, thereby pushing the Pareto frontier"。与页面「推荐 ℓ-MoE_acc 作为 Pareto 最优变体」及 $\binom{N}{k}\to\binom{\alpha N}{\alpha k}$ 的表述一致。
- C4（$U_\text{eff}$）：§2 原文 "this effective nonlinear budget per token is proportional to the total width of the selected experts: Ueff ∝ K·m"，设计原则 III "preserving the effective nonlinear budget, K·m"。页面写 $U_\text{eff}=K\cdot m$（论文用 ∝，但原则句以 $K\cdot m$ 指代该量），可接受。
- C2（设计原则 I、II）：§2 五条原则逐条核对，I 显存带宽主导低延迟、II 高吞吐受通信限制（最小化 routed hidden dim 与激活专家数）、III/IV/V 与页面折叠块列表一致。
- N4（16B 消融）：§4.1 用 16BT-2BA 模型、ℓ−MoE_eff 配置、Figure 3/Figure 4 标题原文；"model quality is preserved for compression ratios α≤4"；Figure 4 "Comparison of validation loss … when hidden dimension compressed by 4×"，补偿专家缩放 $N'=\alpha N$ 有效。与页面「16B 总参/2B 激活、压缩比到 4 质量保持、只放大 $N$ 可补偿」一致；另核对论文 §4.1 消融基线为 hidden 2048→512（$\alpha=4$），与页面一致。
- N5（9%）：§4.3.1 原文 "Relative to native Kimi-K2-1T, Kimi-K2-1T-LatentMoE introduces additional computation due to latent projection operators. In our projections, native Kimi-K2-1T remains close, within up to ∼9% of Kimi-K2-1T-LatentMoE, indicating that projection overhead is small…"。页面「其投影分析给出这部分额外计算至多约 9%」与之一致。
- N6（仅登记未引用）：§4.3.1 "1.24×–3.46× projected slowdown across the frontier" 与约 350B 额外参数（≈1.35T iso-accuracy 基线）。与来源说明登记一致，正文未直接引用。
- N1（K3 配置）：[K3] §2.3 与官方 config.json 双向核对——hidden_size=7168、routed_expert_hidden_size=3584、num_experts=896、num_experts_per_token=16、num_shared_experts=2；$896/16=56$，$7168/3584=2$。全部一致。
- C6（三件稳定化）：[K3] §2.3 原文——Normalized LatentMoE "inserts RMSNorm between expert aggregation and up-projection"；Eq. 11 "$y=\sum E_j^{shared}(x)+W_\uparrow\,\mathrm{RMSNorm}(u)$"；SiTU-GLU "softcap(x,β)=β tanh(x/β)"（β₁=4、β₂=25）；Quantile Balancing 由单次前向从 margin 的 (1−k/n) 分位点直接设定 bias。§2.3 亦给出 "This extreme sparsity amplifies two failure modes … exploding internal activations in the routed branch" 与负载均衡超出既有方法适用区间，及 2.8T 总参。页面第五章表述全部一致。
- N2/N3（Nemotron-3）：[R] Sebastian Raschka, "Nemotron 3 Ultra and Latent MoE Scaling"（sebastianraschka.com/blog/2026/nemotron-3-ultra-latent-moe.html）——Super 4096→1024→4096、120B 总参/12B 激活；Ultra 8192→2048→8192、550B 总参/55B 激活、每层 512 路由专家、top-22、1 共享专家、路由专家中间维度 5120、共享专家中间维度 10240；两者同为 4× 压缩。全部一致。
- 代码：折叠块 Python 代码在本环境实际执行，输出与页面「预期输出」逐行一致（alpha=4.0；32768/8192；65536/16384；C(8,2)=28；C(32,8)=10518300；375653.57；2*4096*1024=8388608）。正文组合数 10,518,300、「约 1052 万」「约 37 万倍」自洽，$2\cdot d\cdot\ell$ 投影参数量与 N4 简化说明自洽。
- 机械项：`../moe-serving/index.html`、`../deepseek-moe/index.html`、`../stable-latent-moe/index.html` 三页均存在，无「（待生成）」占位；overview.html 与 index.html 相互链接；`python3 .dojo/scripts/validate.py wiki/latent-moe` 返回 `validation ok`；全文无 Unicode 数学字符直出，无 `×/≤/∝` 等字符出现在标题、summary、正文、列表、表格中；唯一 `<img>` 为 lightbox 占位（`alt=""`），无 alt 内 `$...$`；§2 数据流图为 HTML 结构（`.flow-diagram`），非等宽字符框线图；公式全部 KaTeX 渲染。
- 表述：逐段通读（含折叠块与图注）未发现「本页将…/下面来看…/需要注意的是」类元话语，无「我/我们/你」会话指代，无调试与复现踩坑叙事，无临场评价；「场景」仅作普通名词（低延迟场景/高吞吐场景）使用，未当术语。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：修复（上述 3 条轻微项关闭后即可发布；本轮未发现阻断与重要问题，事实性论断、数字、公式与代码均回源核对通过）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
