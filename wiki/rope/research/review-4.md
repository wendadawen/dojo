<!-- review-meta
round: 4
page: wiki/rope/index.html
reviewed_content_sha256: e8b4c45194825203
-->
# RoPE 旋转位置编码审查记录（第 4 轮）

- 页面版本：79f95a15cb90066e786b5939b1124d38ed64b9de
- 审查时间：2026-09-13 19:48
- 审查者：独立子代理
- 已完整阅读章节：引言与核心问题（5 条）；1. 为什么 Transformer 需要位置编码——RoPE 与四类方案的差异（含本章问题）；2. 2 维 RoPE 是怎么旋转的——最小可手算机制（含本章问题与 Figure 图注）；3. 内积为何只依赖 $m-n$——旋转矩阵群的性质（含本章问题）；4. d 维推广——分块对角旋转矩阵与 $\theta_i$ 几何级数（含本章问题与两个折叠块）；5. 远程衰减——相位抵消与 QK-only 机制（含本章问题与钟表类比 callout）；6. 适用边界——长度外推、K3 MLA 的 NoPE 选择（含本章问题）；来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件）。
- 说明：overview.html 已一并阅读；页面无可运行代码块，未执行代码核对。

## 问题

- [阻断·技术] 第 110 行（核心问题 Q4 解答）：同一批数字在页内被描述为两种互相矛盾的来源——此处称"实测"，第 480 行与第 638 行则明确称"非实测数据""构造示例"。｜引文依据：第 110 行"以 $q=k$ 且每对分量恒定的构造情形（$d=128$、$\mathrm{base}=10000$）实测：$|m-n|=1$ 时归一化和 $0.9702$…"；对照第 638 行"d=128 衰减表：按归一化内积公式 $\frac{1}{64}\sum_{i=0}^{63}\cos((n-m)\theta_i)$ 直接计算（Python 实现，d=128、base=10000），非实测数据"（第 480 行同）。数值本身经复算正确（0.9702/0.6691/0.1590/−0.0070 与公式计算一致）。｜修复要求：把第 110 行的"实测"改为与第 480、638 行一致的表述（如"按归一化公式直接计算"），全页对这批数字的来源只用一种说法。｜修复：｜复验：

- [重要·可读性] 第 417 行（d=4 手算折叠块末段）：用绝对旋转角的正弦/余弦描述"某对旋转到多快"，所用的 cos 值并非该对的内积贡献，未加区分，易让读者把内积贡献误解为 $\cos(n\theta_i)$ 而非 $\cos((n-m)\theta_i)$。｜引文依据：同折叠块上文刚算出"总内积 = $0.5403 + 0.8776 = 1.4179$"（两对贡献为 $\cos((n-m)\theta_0)=\cos1\approx0.5403$、$\cos((n-m)\theta_1)=\cos0.5\approx0.8776$），而末段写"$\theta_0=1$ 让第 0 对在位置 $n=2$ 已旋转到 $\cos(2)$ 约为 $-0.416$…$\theta_1=0.5$ 让第 1 对在位置 $n=2$ 只旋转到 $\cos(1)$ 约为 $0.540$"，其中 $-0.416$、$0.540$ 均非两对的内积贡献。｜修复要求：改写为用相对角 $(n-m)\theta_i$ 说明多尺度差异，或明确点出这两个数是 key 在 $n=2$ 处的绝对旋转状态（角度 $n\theta_i$）而非内积贡献。｜修复：｜复验：

- [轻微·技术] 第 632 行（N1）：把 base=10000 的来源标为"§3.2, Eq.(12)"，但 Eq.(12) 是 §3.2.1 的 2 维复数解，其正文不含 10000；10000 首次出现在 Eq.(15) 之后的参数定义（§3.2.2）与 §3.3。｜引文依据：RoFormer 摘要与正文核对，Eq.(12)"a solution to our formulation [Eq.11] is: fq(xm,m)=(Wq xm)e^{imθ}"，不含 10000；10000 首次出现在 Eq.(15) 之后的"Θ={θᵢ=10000^(−2(i−1)/d), i∈[1,2,…,d/2]}"。｜修复要求：把 N1 的定位改为 §3.2.2（Eq.(15) 后）与 §3.3。｜修复：｜复验：

- [轻微·技术] 第 143–146 行（§1 表格）："长度外推"列给出的定性判断（learned 差／sinusoidal 中等／ALiBi 较好／T5 中等）无任何来源标注，属无来源支持的判断写成结论；同章解答中的"T5 有约 32 个 bucket 每头"亦无来源；"ALiBi 较好"还与仓库内 <a>NoPE</a> 页所引 NoPE 论文结论（Kazemnejad 2023：ALiBi 不适合长度外推）方向相反。｜引文依据：表格单元格均无 `<sup>` 引用；wiki/nope/index.html 第 340 行"ALiBi｜不适合长度外推"（来源：NoPE 论文摘要）。｜修复要求：为该列判断补来源或降级为明确标注的推断，并使其与 NoPE 论文结论在措辞上一致（如需保留"ALiBi 外推优于 learned APE"这一较弱说法，需说明其与长度泛化结论的差别）。｜修复：｜复验：

- [轻微·表述] 元话语多处：第 93 行"需要注意 $\sin$ 项不总为零"、第 111 行"两点需要注意"、第 306 行"下面用三角恒等式直接验证"、第 453 行"先看内积的衰减"、第 543 行"先看适用条件"、第 420 行"实现层面有个等价性值得注意"，均属"下面来看…/需要注意的是"式引导语与临场提示。｜引文依据：不适用。｜修复要求：改为直接陈述（如删除"需要注意/先看/下面…验证/值得注意"等引导，直接给出事实或结论）。｜修复：｜复验：

- [轻微·技术] 第 154 行：以"论文摘要原文"引出的英文片段无省略号地截断，与摘要原文不符。｜引文依据：摘要原文为"…the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation."，页面只引到"…relative position dependency"为止，省略了"in self-attention formulation"。｜修复要求：在被省略处加省略号（如"…relative position dependency …"），或补全整句。｜修复：｜复验：

## 核对但未发现问题的项（供复验参考）

- 公式与数值复算：核心问题 Q2 的 $\sqrt2/2\approx0.7071$、本章问题 Q2 的 $(\sqrt3-1)/2\approx0.3660$、d=4 例子（$\theta_0=1,\theta_1=0.5$，总内积 1.4179）、$d=128$ 频率表（$\theta_0=1$、$\theta_1\approx0.8660$、$\theta_{16}=0.1$、$\theta_{63}\approx1.155\times10^{-4}$、$2\pi/\theta_i$ 的 6/7/63/54410）与衰减表全部数值，均用 Python 独立复算一致。
- 来源核对：论文摘要原文（含"encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency…"）；Eq.(11) 相对位置目标、Eq.(12)–(13) 2 维解与旋转矩阵、Eq.(14)–(15) 维度形式与块对角、Eq.(16) 自注意力内积、Eq.(20)–(33) §3.4.1、Eq.(34) §3.4.2、Eq.(35)–(37) §3.4.3 编号均与页面 blockquote/F 条一致；§3.4.3"1/(d/2)Σ|S_i| decay with the relative distance…as shown in Figure 2"及"论文未标注图中维度取值"属实；"Following Vaswani et al. 2017, we set θi=10000^{−2i/d}"属实。
- C6 采用列表：逐模型核对 PaLM（arXiv:2204.02311 §2 "We use RoPE embeddings …"）确用 RoPE；LLaMA/Mistral/Qwen/Falcon/Gemma/GPT-NeoX 为公认事实。
- K3 相关（C8）：与仓库内 MLA 页、Kimi K3 页一致——`mla_use_nope=true`、K3 对所有 MLA 层用 NoPE、KDA 递归门控/衰减隐式提供位置、1M 上下文无需 RoPE 重缩放或 YaRN；config.json 保留 `qk_rope_head_dim=64` 支撑"结构上保留 rot 分量接口"的说法。
- 机制与边界：PI 公式 $m'=m\cdot L_{\text{train}}/L_{\text{target}}$、YaRN 按频率分组缩放、LongRoPE 按维度非均匀搜索与百万级上下文，均与所引 arXiv 一致。
- 机械项：validate.py 对 wiki/rope/index.html 返回成功；无 research/ 路径引用、无"（待生成）"占位；六个前置概念页（standard-attention、positional-encoding、causal-mask、nope、mla、kimi-k3）均存在；overview.html 与 index.html 互链；dojo:topics=注意力机制（在固定大类内）、dojo:tag=位置编码（在封闭词表内）、dojo:type=concept；无 Unicode 数学字符泄漏进正文/标题/summary（description 为纯文本，允许）；结构图为内联 SVG，标签用 foreignObject+KaTeX，无 ASCII 近似数学写法；无代码块故无代码执行项。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复（阻断项与重要项须先关闭；轻微项给出接受理由或一并修正）