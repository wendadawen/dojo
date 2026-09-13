<!-- review-meta
round: 7
page: wiki/rope/index.html
reviewed_content_sha256: 4511b0304b490d83
-->
# RoPE 审查记录（第 7 轮）

- 页面版本：70981ef17c3854599562cd5d01526007d0e29494（`git hash-object wiki/rope/index.html`）
- 审查时间：2026-09-13 21:52
- 审查者：独立子代理（未参与写作，未参与前序任何轮次；本轮未读取 `wiki/rope/research/` 下任何文件）
- 已完整阅读章节：核心问题（5 条）；1. 为什么 Transformer 需要位置编码——RoPE 与四类方案的差异；2. 2 维 RoPE 是怎么旋转的——最小可手算机制；3. 内积为何只依赖 $m-n$——旋转矩阵群的性质；4. d 维推广——分块对角旋转矩阵与 $\theta_i$ 几何级数；5. 远程衰减——相位抵消与 QK-only 机制；6. 适用边界——长度外推、K3 MLA 的 NoPE 选择；来源与范围说明（含全部 `<details>` 折叠块、表格与 SVG 图注）
- 来源获取：Su et al. 2021 arXiv:2104.09864v5 全文（本地 PDF 抽取 + ar5iv HTML）；Kazemnejad et al. 2023 arXiv:2305.19466 摘要；Raffel et al. 2020（T5）JMLR 全文；Kimi K3 报告 arXiv:2607.24653 条目与仓库内 `wiki/mla/index.html`、`wiki/kimi-k3/index.html` 引用

## 问题

- [重要·技术] 第 3 章（index.html L328 黄色 callout）：「注意 $R_{n-m}$ 与 $R_{m-n}$ 的区别：……由于 $R_{-t} = R_t^T$，两者方向相反但内积结果相同（内积是标量，转置不改变）。」这句断言对同一对 $q$、$k$ 不成立，且与同页 F2 公式自相矛盾｜引文依据：同页 F2（L202）$\langle R_m q, R_n k \rangle = (q_0 k_0 + q_1 k_1)\cos((n-m)\theta) - (q_0 k_1 - q_1 k_0)\sin((n-m)\theta)$；反例 $q=(1,0)$、$k=(0,1)$、$\theta=\pi/4$、$m=1$、$n=2$：用 $R_{n-m}$（相对角 $(n-m)\theta=\pi/4$）得 $0\cdot(\sqrt{2}/2)-1\cdot(\sqrt{2}/2)=-0.7071$，用 $R_{m-n}$（相对角 $(m-n)\theta=-\pi/4$）得 $0\cdot(\sqrt{2}/2)-1\cdot(-\sqrt{2}/2)=+0.7071$，二者反号；成立的是 $q^T R_{n-m} k=(q^T R_{n-m} k)^T=k^T R_{m-n} q$，即必须同时交换 $q$、$k$ 的角色才保持同一标量（这正是「转置不改变」真正给出的恒等式），而 callout 未提交换、读起来等价于 $q^T R_{n-m} k=q^T R_{m-n} k$。页面在 2 维一章、核心问题与「常见误解」callout 反复强调 $\sin$ 项不可省、内积依赖带符号的 $(n-m)$，与该句冲突｜修复要求：改写为准确表述，例如「两个矩阵互为转置（角度差差一个反号）；把 $R_{n-m}$ 换成 $R_{m-n}$ 相当于同时交换 $q$、$k$ 的角色：$q^T R_{n-m} k = k^T R_{m-n} q$，对同一对 $q$、$k$ 两者一般不同（差一个 $\sin$ 项的反号）」；或删除该句｜修复：｜复验：
- [轻微·技术] 第 2 章末（index.html L204）：「论文给出的完整公式是上面这个形式。」把 F2 的实部 $\cos/\sin$ 展开式说成论文原样给出的公式，与论文写法及本页 F2 条自述不一致｜引文依据：论文 §3.2.1 Eq.(12) 原文为复数形式 `g(xm , xn , m − n) = Re[(W q xm )(W k xn )∗ ei(m−n)θ ]`（PDF 抽取文本），§3.4.1 Eq.(20)–(33) 推出的终式 Eq.(33) `fq (xm , m) = (W q xm )eimθ` 同样是复数形式，论文未给出实部 $\cos/\sin$ 展开式的编号公式；本页 F2 条（L623）自己标注「§3.2.1 Eq.(13) 应用；§3.4.1 给出严格推导（Eq.(20)–(33)），无独立编号」｜修复要求：改为准确表述（如「论文 Eq.(12) 取实部展开后即上面这个形式」），或直接去掉对论文写法的断言｜修复：｜复验：
- [轻微·格式] 页面多处数学变量/关系符未用 `$...$` 包裹，且与同页其他位置的写法不一致：L309 `<summary>补充：R_m^T R_n = R_{n-m} 的三角恒等式推导</summary>`；L407 `<summary>展开：d=4 多频率手算例子</summary>`；L251「真实模型 d=64/128 且 base=10000」；L408「真实模型 base=10000」；L614「$\theta_i$ 与 base=10000 为固定常数」；L633「N2（d=128 时 …）：F6 代入 d=128 计算」；L640「d=128 衰减表…（Python 实现，d=128、base=10000）」；L649「简化 2：…只画 d=8 的结构示意…d=128 的具体频率分布与 d=8 相同」｜引文依据：不适用（格式类；判据为 `guides/concept/style-guide.md` §5「summary、h2、h3 标题内的数学符号使用 `$...$` 行内公式」与 §11「页面任何位置出现的数学变量……都必须包在 `$...$` 或 `$$...$$` 中」「同一变量在页面中保持同一种写法」；对照同页 L382 写 `$d=8$`、L88 写 `$R_m^T R_n=R_{n-m}$`、L336 写 `$R_m^T R_n = R_{n-m}$`，写法不统一）｜修复要求：把上述位置改为 LaTeX 行内公式（`补充：$R_m^T R_n = R_{n-m}$ 的三角恒等式推导`、`展开：$d=4$ 多频率手算例子`、`$d=64/128$`、`$\mathrm{base}=10000$`、`$d=128$`、`$d=8$`），全页统一｜修复：｜复验：

## 本轮回源复核（未发现问题，逐条留证）

- 公式编号：页面 meta 的编号表与实际论文一致——Eq.(1)–(2) §2.1、Eq.(3)–(4) §2.2、Eq.(5)–(10) §2.3、Eq.(11) §3.1、Eq.(12)–(13) §3.2.1、Eq.(14)–(16) §3.2.2、Eq.(17)–(19) §3.3、Eq.(20)–(33) §3.4.1、Eq.(34) §3.4.2、Eq.(35)–(37) §3.4.3。
- C3/C4/N1：论文原文「the rotary matrix with pre-defined parameters Θ = {θi = 10000−2(i−1)/d , i ∈ [1, 2, ..., d/2]}」（§3.2.2）与「Following Vaswani et al. [2017], we set θi = 10000−2i/d .」（§3.3）与页面引文逐字一致；页面 `i` 从 0 起与论文从 1 起数学等价，页面已注明。
- C1/F1/F4/F5：Eq.(13) 与 Eq.(14) 的投影下标为 `{q, k}`、不含 `v`；Eq.(16)「RdΘ,n−m = (RdΘ,m )⊺ RdΘ,n」，与页面 C1、F3 表述一致。
- C5/F8：论文原文「Note that the value of 1/d/2 Σ_{i=1}^{d/2} |Si | decay with the relative distance m − n increases by setting θi = 10000−2i/d , as shown in Figure (2).」+ 图 2 caption「Figure 2: Long-term decay of RoPE.」（图内无维度标注），与页面 C5 引文与其「论文未标注图中维度取值」一致。
- F2 与 d=128 表格：Python 复算 θ0=1、θ1=0.86596、θ16=0.1、θ63=1.15478e-4，「转一圈位置数」6/7/63/54410 与页面一致；归一化内积表 |m−n|=0,1,5,10,50,100,500,1000,5000,10000 → 1.0000、0.9702、0.7373、0.6691、0.5462、0.4772、0.2985、0.1590、−0.0070、−0.0279，逐值与页面表格一致。
- 手算例：2 维例（$q=k=(1,0)$、$\theta=\pi/4$、$m=1,n=2$）内积 $\sqrt{2}/2$；L283 例（$q=(1,1)$、$k=(0,1)$、$\theta=\pi/6$、$m=0,n=1$）$(\sqrt{3}-1)/2\approx0.3660$（用直接旋转复核一致）；d=4、base=4 例 $\cos1+\cos0.5\approx1.4179$、$n\theta_i$ 相位 2 rad 与 1 rad；d=4、base=10000 时 $\theta_1=0.01$ —— 全部复算相符。
- N4：T5 原文「we use 32 embeddings for all of our models with ranges that increase in size logarithmically up to an offset of 128…」+「within a given layer each attention head uses a different learned position embedding」，支持页面「每头约 32 个 bucket」。
- N3：arXiv:2305.19466 摘要「ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks」，与页面 §1 表格与 N3 的表述一致；页面已把该列标为定性推断。
- C8：arXiv:2607.24653 条目存在（标题 "Kimi K3: Open Frontier Intelligence"，2.8T/104B/1M-token 上下文）；`wiki/mla/index.html` §5.1 引 K3 §2.1.2 原文「Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to all MLA layers.」对应 `mla_use_nope=true`，与页面 §6.2 的机制与因果表述一致。
- 链接与锚点：`standard-attention`、`positional-encoding`、`causal-mask`、`nope`、`mla`、`kimi-k3`、`overview.html`、`../../index.html` 与本地 css/js 全部存在；正文引用的 MLA「为什么 RoPE 要解耦」「K3 的 Gated MLA」与 Kimi K3「长上下文扩展」章节均存在。
- 机械项：`python3 .dojo/scripts/validate.py wiki/rope/index.html` 返回 `validation ok`；无 `（待生成）` 占位；`alt`/`aria-label` 内无 `$...$`，SVG 公式均在 `<foreignObject>` 中；两级问题块（核心问题 5 条、各章本章问题 3 条）均有解答折叠块。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复