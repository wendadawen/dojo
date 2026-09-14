<!-- review-meta
round: 7
page: wiki/linear-attention/index.html
reviewed_content_sha256: d33320adb8805764
-->
# 线性注意力审查记录（第 7 轮）

- 页面版本：cc0020ee1615cb0ad8e336b3037217e60bf4bf79（git hash-object wiki/linear-attention/index.html）
- 审查时间：2026-09-14 17:36
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. softmax 注意力——$O(N^2)$ 瓶颈在哪一步 → 2. 核函数与结合律——把相似度分解并重排 → 3. 因果掩码——变成固定大小的递归状态 → 4. 表达力代价——代价来自公式的哪一部分 → 来源与范围说明（含全部 details 折叠块、图注、callout 与 overview.html）
- 核对来源版本：Katharopoulos et al. 2020 "Transformers are RNNs"，arXiv:2006.16236v3（ICML 2020）；Choromanski et al. 2020 "Rethinking Attention with Performers"，arXiv:2009.14794（ICLR 2021）。均经 ar5iv HTML 渲染版逐条定位。机械验证：`python3 .dojo/scripts/validate.py wiki/linear-attention/index.html` → validation ok。

## 问题

- [轻微·表述] 正文三处章节过渡（第 1 章末「本章定位了…但不构造…能否…——下一章用核函数与结合律回答。」；第 2 章末「本章用核分解与结合律把训练前向降到 $O(N)$。但生成场景下还要每步重算聚合吗——下一章用因果掩码挖出递归形式。」；第 3 章末「本章把因果掩码下的线性注意力写成固定大小递归状态。但 softmax 为什么不能也这样重排——下一章回答代价从哪里来。」）：三句共用同一骨架「本章<小结>。但<问题>——下一章<动作>」，属 `guides/concept/style-guide.md` §8 明令不使用的固定过渡句式｜引文依据：不适用｜修复要求：至少改写其中两处的句型（如把小结与提问合并为一句、或换用陈述式衔接），使三处过渡不再共享同一骨架｜修复：｜复验：
- [轻微·格式] `index.html` 第 11 行 `<title>` 写作 `线性注意力：用核函数把注意力从 O(N²) 降到 O(N)`，其中 `²`（U+00B2）是 Unicode 上标数字；同一页面第 64 行 h1 同一表达式写作 `$O(N^2)$`，标题与 h1 对同一公式用了两种写法。`guides/concept/style-guide.md` §11 要求「数学变量、希腊字母、上下标…都必须包在 `$...$` 中…不因位置而放宽」，`check.md` §9 亦要求「标题…中无 Unicode 数学字符直接出现」。全站 98 个页面中仅本页 `<title>` 出现 Unicode 上标（`grep -l '<title>[^<]*[²³⁰¹⁴⁵⁶⁷⁸⁹]' */index.html` 只命中本页）。注：`.dojo/scripts/validate.py` 把 `title` 列入 `UI_CONTEXT_TAGS` 跳过检查，且 U+00B2 不在其 `BARE_MATH_CHARS`（上标区为 U+2070–U+209F）内，故 validate 不报错，但写法仍与规范及其余 97 页不一致｜引文依据：不适用｜修复要求：把 `<title>` 内的 `O(N²)`/`O(N)` 改为纯文本 `O(N^2)`/`O(N)`（或与全站一致改为不含公式的措辞），与 h1 写法统一｜修复：｜复验：

## 核对依据（本轮逐条回源，全部通过）

- C1/F1（softmax 复杂度）：ar5iv 2006.16236 §3.2 原文 "the computational cost of softmax attention scales with 𝒪(N²)"；softmax attention 定义在 §3.1 Eq.(2)。ar5iv 2009.14794 §2.1 Eq.(1) `A = exp(QK^T/√d)`、`D = diag(A 1_L)`，复杂度原文 "are O(L²d) and O(L² + Ld) respectively"。页面 §1 与 C1/F1 一致。
- C2（sim 非负）：§3.2 原文 "the only constraint we need to impose to sim(·), in order for equation 3 to define an attention function"；广义注意力为 Eq.(3)。页面第 205 行表述一致。
- C3/F2（结合律重排）：§3.2 原文 "compute ∑ϕ(Kj)VjT and ∑ϕ(Kj) once and reuse them for every query"；公式为 Eq.(4)(5)(6)。页面第 219 行 `(φ(Q)φ(K)^T)V = φ(Q)(φ(K)^TV)` 与之一致。
- C4/F3（因果递归与 RNN 形式）：§3.3 Eqs.(8)–(12)，§3.4 Eqs.(16)–(20)。逐条核对 RNN 形式：(16) `s₀ = 0`、(17) `z₀ = 0`、(18) `sᵢ = sᵢ₋₁ + φ(xᵢW_K)(xᵢW_V)ᵀ`、(19) `zᵢ = zᵢ₋₁ + φ(xᵢW_K)`、(20) `yᵢ = f_l( φ(xᵢW_Q)ᵀsᵢ / (φ(xᵢW_Q)ᵀzᵢ) + xᵢ )`，与页面第 385 行 callout 内公式逐项相同（页面按简化记号写 K_i、V_i）。
- C5/F4（elu+1）：§3.2.1 Eq.(7)，原文 "We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative."；页面第 429 行引语与之一致。
- C5/F5（Performer 随机特征）：ar5iv 2009.14794 §2.3 Eq.(5) `φ(x) = h(x)/√m (f_1(ω_1^T x),…,f_l(ω_m^T x))`，Lemma 1（PRFs for Softmax）给出 `SM(x,y) = E_{ω~N(0,I_d)}[exp(ω^T x − ‖x‖²/2) exp(ω^T y − ‖y‖²/2)]`，并指出取 `h(x)=exp(−‖x‖²/2), l=1, f_1=exp, D=N(0,I_d)`。故页面第 435 行 φ 形式与第 437 行 `E[φ(q)^T φ(k)] = exp(q^T k)` 成立；SM(x,y)=exp(x^T y) 为 §2.3 Eq.(6)。
- softmax 核无穷维：`exp(q^Tk) = Σ_{n≥0} (q^Tk)^n/n!` 的多项式特征展开确实无限维，页面第 419 行论断成立，且第 498 行 C5 已注明该论证为页面自身的泰勒展开论证而非论文原文。
- N1（速度）：arXiv:2006.16236v3 摘要原文 "up to 4000x faster on autoregressive prediction of very long sequences"；Table 2（CIFAR-10 自回归）Linear 17.85 images/sec vs Softmax 0.004，倍率 **4,462×**。页面 N1 同时给出「摘要 up to 4000x」与「Table 2 实测 4,462×」，正文第 456 行取「约 4000×」，与摘要口径一致。
- N2（质量数字）：Table 1（MNIST bits/dim）Softmax 0.621、Linear(ours) 0.644，表注 "achieve almost the same bits/dim as the full softmax attention"；Table 3（WSJ PER）Softmax 5.12、Linear 8.08。页面 N2 与第 456 行数字完全一致。§3.2.1 原文 "performs on par to the full transformer"，摘要 "Our linear transformers achieve similar performance to vanilla transformers"，与页面第 515 行所引一致。
- 算式复算（全部通过）：$QK^T$=[[1,0,1],[0,1,1],[1,1,2]]；φ(K)=K+1=[[2,1],[1,2],[2,2]]；φ(K)ᵀV=[[4,3],[3,4]]；φ(K)ᵀ1=(5,5)；V'_1=(11,10)/15≈(0.733,0.667)；递推 s₁=[[2,0],[1,0]]、z₁=(2,1)、s₂=[[2,1],[1,2]]、z₂=(3,3)、s₃=[[4,3],[3,4]]、z₃=(5,5)；V'_3=(14,14)/20=(0.7,0.7)；因果 V'_1=(5,0)/5=(1,0)=V₁。s₃/z₃ 与非递归形式 φ(K)ᵀV、φ(K)ᵀ1 相等，页面「两种算法结果应当相同」成立。
- 4.3 对比表复算：Q₃ᵀK_j = 1,1,2；softmax 权重 e/(2e+e²)=0.2119≈0.212（×2）、e²/(2e+e²)=0.5761≈0.576，和=1；权重比 0.576/0.212=2.717≈e=2.718；elu+1 相似度 6,6,8（和 20），权重 0.300/0.300/0.400，比值 1.333≈1.33。页面第 454 行全部数字成立。
- 图注读数：第 163 行图注「$N$ 从 $3$ 增到 $300000$ 涨约 5 个数量级，元素数从 $9$ 增到约 $9\times 10^{10}$ 涨约 10 个数量级」——$10^5$ 与 $10^{10}$，与图上三档刻度（$N=3$→$9$；$N=6$→$36$；$N=300000$→$9\times10^{10}$）及 $N^2$ 关系一致。第 232、340 行图注与图内节点文字一致。
- 页面链接：`../standard-attention/index.html`、`../kda/index.html`、`../delta-rule/index.html`、`../gated-deltanet/index.html` 四个概念页在 `wiki/` 下均真实存在，无「（待生成）」占位；overview.html 与 index.html 双向链接；validate.py 未报断链。
- 结构项：h2 编号 1–4 连续、「来源与范围说明」不编号；来源章下六个 h3 均为固定命名；两处「本章问题」与页面级「核心问题」共 4 题均有 `解答：` 折叠块，四个核心问题答案末尾分别指向「1.」「2.」「3.」「4.」全标题章节，指向的 h2 标题逐字一致；结构图为 HTML div（`dg-stack`/`dg-flow`），无等宽字符框线图；`<text>` 中无 ASCII 近似数学写法（本页无内联 SVG）；`dojo:summary` 为行内 `$...$` 且可渲染；`<head>` 含纯文本 description、`dojo:type=concept`、topics「注意力机制」、tag「注意力」。
- 表述扫描：未发现元话语（「本页将…」「下面来看…」「需要注意的是」）、第一/第二人称、调试叙事或临场评价；无来源支持的判断（第 113、456、459 行）均已就地标注为「经验判断，未逐篇核对原文」或「页面推断，来源未就此归因」；KDA 相关陈述（第 381、487 行）以「据 KDA 的递归更新形式」限定，并标注「研究脉络推断」。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（两条轻微问题建议在下一轮顺带清理，均不影响正确性与主线理解）
