<!-- review-meta
round: 5
page: wiki/linear-attention/index.html
reviewed_content_sha256: 6350f0e0af7800f6
-->
# 线性注意力审查记录（第 5 轮）

- 页面版本：b6563dff37318e2dfdac8552d1af9b416784edfd（`git hash-object wiki/linear-attention/index.html`）
- 审查时间：2026-09-13 20:16
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：引言与元信息 → 核心问题 → 最容易误解 → 1. softmax 注意力——$O(N^2)$ 瓶颈在哪一步（含两张结构图、本章问题与折叠答案）→ 2. 核函数与结合律——把相似度分解并重排（含"展开：用直接求和验证重排结果一致"折叠块）→ 3. 因果掩码——变成固定大小的递归状态（含状态流转图、"展开：用 $s_1,z_1$ 算 $V'_1$"折叠块）→ 4. 表达力代价——从公式的哪里来（4.1 / 4.2 / 4.3 与本章问题）→ 来源与范围说明（论断与来源 C、公式与来源 F、外部数字与实验条件 N、构造示例、辅助解释与类比边界、简化条件及其限制）；另读完 overview.html 全文。

## 来源逐条核对（引文依据）

- C1 / F1：Katharopoulos et al. 2020 §3 Eq.(2) 为 $A_l(x)=V'=\mathrm{softmax}\!\left(QK^T/\sqrt D\right)V$；§3.2 原文 "From equation 2, it is evident that the computational cost of softmax attention scales with O(N^2)... The same is true for the memory requirements because the full attention matrix must be stored"。Choromanski et al. 2020 §2.1：$\mathbf A=\exp(\mathbf Q\mathbf K^\top/\sqrt d)$，时间 $O(L^2d)$、空间 $O(L^2+Ld)$。与页面一致。
- C2：§3.2 原文 "Note that the only constraint we need to impose to sim (·), in order for equation 3 to define an attention function, is to be non-negative."（Eq.(3) 即页面第 2 章的广义注意力式）。一致。
- C3 / F2：Eq.(4)(5) 为把 $\phi(Q_i)^\top$ 提到求和号外的两式，Eq.(6) 为 $\left(\phi(Q)\phi(K)^\top\right)V=\phi(Q)\left(\phi(K)^\top V\right)$；"once and reuse them for every query" 一句位于 §3.2 段末。重排与 Eq.(6) 的对应关系成立；仅出处小节标注有误（见问题 3）。
- C4 / F3：Eq.(8) 因果形式、Eq.(9)(10)(11)(12) 定义 $S_i,Z_i$ 与 $V'_i=\phi(Q_i)^\top S_i/\phi(Q_i)^\top Z_i$、Eq.(16)–(20) 完整 RNN 形式（含 $W_Q,W_K,W_V$、残差 $+x_i$、层激活 $f_l$）。页面第 3 章 callout 中的公式与 Eq.(18)(19)(20) 逐符号对应。
- C5 / F4 / F5：Eq.(7) $\phi(x)=\mathrm{elu}(x)+1$；§3.2.1 原文 "We prefer elu(·) over relu(·) to avoid setting the gradients to 0 when x is negative"；§3.2.1 原文 "the feature function that corresponds to the exponential kernel is infinite dimensional, which makes the linearization of exact softmax attention infeasible"（页面用泰勒展开给出的理由属数学事实，与来源结论同向）。Choromanski §2.3 Eq.(5)+Lemma 1：$h(x)=\exp(-\|x\|^2/2)$、$\omega\sim\mathcal N(0,I_d)$、$f=\exp$，无偏；页面 4.2 的 $\phi(x)$ 式与之等价（$\exp(-\|x\|^2/2)/\sqrt m$ 在外、各分量为 $\exp(\omega_i^Tx)$）。
- N1：摘要 "they are up to 4000x faster on autoregressive prediction of very long sequences"；Table 2 行 "Linear (ours) 3.40 17.85 (4,462×)"。页面的 4,000× 与 4,462× 均可定位。
- N2：Table 1 "Softmax 0.621 / Linear (ours) 0.644"（MNIST bits/dim）；Table 3 "Softmax 5.12 / Linear (ours) 8.08"（WSJ Validation PER）。与页面一致。
- 全部算式复算通过：$QK^T=\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$；$\phi(K)=\begin{pmatrix}2&1\\1&2\\2&2\end{pmatrix}$；$\phi(K)^\top V=\begin{pmatrix}4&3\\3&4\end{pmatrix}$；$\phi(K)^\top\mathbf 1=(5,5)$；$V'_1=(11,10)/15\approx(0.733,0.667)$，直接求和验证 $5,4,6$（分母 15、分子 $(11,10)$）一致；$s_1=\begin{pmatrix}2&0\\1&0\end{pmatrix},s_2=\begin{pmatrix}2&1\\1&2\end{pmatrix},s_3=\begin{pmatrix}4&3\\3&4\end{pmatrix}$，$z_1=(2,1),z_2=(3,3),z_3=(5,5)$，$V'_3=(14,14)/20=(0.7,0.7)$，$V'_1=(5,0)/5=(1,0)=V_1$；4.3 表 softmax 权重 $0.212/0.212/0.576$（和 $2e+e^2$ 化简后为 $1/(2+e)$ 与 $e/(2+e)$）、$\mathrm{elu}{+}1$ 权重 $6/20,6/20,8/20$，比值 $2.72$ 与 $1.33$ 均复算通过。
- 机械项：`.dojo/scripts/validate.py wiki/linear-attention/index.html` 返回 `validation ok`；无模板占位符、无重复 id、无坏本地引用；[C1–C5]/[F1–F5]/[N1–N2] 上标与来源章节双向对应且无缺漏；overview.html 与 index.html 互链；standard-attention、kda、delta-rule、gated-deltanet 四个被引概念页真实存在；无运行代码块（代码项不适用）。

## 问题

- [轻微·表述] 行 71（引言第一句）："它在生成第 30 万个字时还要回头计算这个字与前 30 万个字的两两相似度——光是这一步注意力计算就要处理约 $9\times10^{10}$ 个 query-key 相似度"——前半句说的是单个 query 对 30 万个 key（计数为 $N=3\times10^{5}$），后半句给的 $9\times10^{10}$ 是 $N^2$，两句的数量级所指对象不同，读者会把"这一个 token 与前 30 万个 token 的比较"直接读成 $N^2$。｜引文依据：Katharopoulos §3.3.2 "the cost per timestep for transformers is not constant; instead, it scales with the square of the current sequence length because attention must be computed for all previous timesteps"（$N^2$ 量级来自逐步重算全部历史时间步）；§3.2 "the computational cost of softmax attention scales with O(N^2)" 指的是完整注意力矩阵。｜修复要求：把"这一步"改为指向整段序列构造 $QK^T$ 的规模，或补一句说明 $9\times10^{10}$ 来自对全部历史时间步重算，使前后两个数量级指向同一对象。｜修复：｜复验：

- [轻微·表述] 行 415 h2 标题"4. 表达力代价——从公式的哪里来"，行 100 核心问题"这个代价从公式哪里来"：中文不成立，"从公式的哪里来"应为"来自公式的哪一部分"。同一标题还被行 103、行 235 两处括注原样复用。｜引文依据：不适用｜修复要求：改为通顺表述（如"4. 表达力代价——代价来自公式的哪一部分"），并同步改写行 100、行 103、行 235 三处引用文字。｜修复：｜复验：

- [轻微·技术] 行 496 C3 把引文出处标为"§3.2.1 原文"，但该句在 §3.2 正文内，位于 "3.2.1. Feature maps and computational cost" 标题之前（C1、C5 的出处标注方式正确，不受影响）。｜引文依据：Katharopoulos §3.2 段末 "In contrast, our proposed linear transformer from equation 5 has time and memory complexity O (N ) because we can compute $\sum_j\phi(K_j)V_j^\top$ and $\sum_j\phi(K_j)$ once and reuse them for every query."；其后才是 "3.2.1. FEATURE MAPS AND COMPUTATIONAL COST"。｜修复要求：把 C3 的出处小节号改为 §3.2。｜修复：｜复验：

- [轻微·格式] 同一相似度在页面上有三种写法：行 73、行 265 写 $\exp(q\cdot k)$，行 417、行 419 写 $\exp(q^T k)$，行 205 定义 $\mathrm{sim}(q,k)=\exp(q^Tk/\sqrt d)$；核内积也混用 $\phi(q)\cdot\phi(k)$（行 73）与 $\phi(q)^T\phi(k)$（行 209 等），行 255 一句内同时出现 $\phi(Q_1)^T\phi(K_1)=(2,1)\cdot(2,1)=5$。｜引文依据：不适用｜修复要求：统一为 $q^Tk$ 与 $\phi(q)^T\phi(k)$ 写法；行 417、行 419 用无缩放的 $\exp(q^Tk)$ 是为对齐 Performer 的 $\mathrm{SM}(x,y)$ 定义，应在该处说明它与第 2 章带 $\sqrt d$ 写法的关系。｜修复：｜复验：

- [轻微·技术] 行 113 与行 459："经验上 $N$ 较小时 softmax 仍更快（常数项更小、IO 更友好）"未给出任何来源，却分别出现在"最容易误解"条目与"适用边界"callout 的结论位置；同页的机制论断与实验数字都逐条标注了出处。｜引文依据：不适用（Katharopoulos 2020 §3.2.1 只给出针对二次多项式核的判据 "This makes the computational complexity favorable when N > D^2"，没有给出小 $N$ 下 softmax 更快、常数项更小、IO 更友好的结论）｜修复要求：补出该经验判断的出处，或按页面已有的推断标注方式处理（与行 487"研究脉络推断，未逐篇核对原文"一致）。｜修复：｜复验：

- [轻微·表述] overview.html"关键结论与边界"第二条："同一组输入下 softmax 的最高/最低权重比可达 $e\approx2.72$，而 $\mathrm{elu}+1$ 核只有约 $1.33$"——该比值取自 index.html §4.3 那张"softmax 列暂略 $\sqrt d$ 缩放"的对比表（若带上 $\sqrt d$，比值应为 $e^{1/\sqrt2}\approx2.03$），overview 未带这一条件；且"可达"易被读成上界，而该比值随分数差增大会远超 $e$。｜引文依据：index.html 行 441 "为简化对比，下表 softmax 列暂略 $\sqrt{d}$ 缩放（完整应为 $\exp(Q_3\cdot K_j/\sqrt{2})$）"；行 454 的 $0.576/0.212\approx2.72$ 与 $0.400/0.300\approx1.33$ 均为本页 $N=3,d=2$ 构造示例的取值。｜修复要求：在 overview 该条注明这是本页构造示例在略去 $\sqrt d$ 缩放下的取值，或改写为"例中 softmax 权重比 $2.72$，对应 $\mathrm{elu}+1$ 的 $1.33$"。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：修复（6 条轻微项按上表逐条处理后即可发布；无阻断与重要问题，来源论断、公式、实验数字与构造示例均已回源核对并复算通过）