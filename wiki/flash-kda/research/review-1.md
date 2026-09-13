<!-- review-meta
round: 1
page: wiki/flash-kda/index.html
reviewed_content_sha256: b502faf762ce23a7
-->
# FlashKDA 与 KDA Context Parallelism 审查记录（第 3 轮）

- 页面版本：`bd48f832660a0c1af8ca41bf6e6468684e757717`（wiki/flash-kda/index.html 工作树哈希）
- 审查时间：2026-09-13 18:47
- 审查者：独立子代理（未参与写作，未参与前序轮次）
- 已完整阅读章节：核心问题（5 条，含解答折叠块）、最容易误解、1. 串行状态 vs GPU 并行、2. FlashKDA、3. 设备内 context parallelism、4. KCP（4.1–4.5，含 4.5 展开折叠块）、5. KDA 解码（5.1–5.3）、来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制，全部读完）、全部图注
- 核对来源：Kimi K3 Technical Report（arXiv:2607.24653 HTML 与官方 PDF，两份逐句对照）§2.1.1 Eq.1/Eq.5、§5.1.1、§5.1.2 Eq.17、§5.4.2；官方 config.json（HuggingFace `moonshotai/Kimi-K3`）；guides/concept/check.md、guides/concept/style-guide.md
- 机械验证：`python3 .dojo/scripts/validate.py wiki/flash-kda/index.html` → `validation ok`；概念链接 wiki/kda、wiki/gpu-execution-model、wiki/linear-attention 三个前置页均真实存在；overview.html 与 index.html 双向链接；无"（待生成）"占位；正文无 Unicode 数学字符（validate 通过）
- 说明：本页 research/ 下不存在 official/ 目录，官方来源经外部获取；PDF 中 `\left( \right)` 一类大括号会被 pdftotext 丢弃，括号判定以 arXiv HTML 的 LaTeX 为准。

## 问题

- [阻断·技术] 来源：K3 报告 §2.1.1 Eq.1、§5.1.2｜位置：行 127（§1 的 Eq.1 展示）、行 371（§4.2 再次展示）、行 615（[C4] 引文）、行 630（构造示例）｜问题：$M_t$ 的定义漏掉括号，$\mathrm{Diag}(\alpha_t)$ 由"作用于 $S_{t-1}$"被改写成"并到 $k_t k_t^\top$ 上"，两式数学上不等价；[C4] 又把这一改动后的式子当作"原文"引用；§4.5 手算示例建立在这个错误 $M_t$ 上——页面写"$\alpha=0.5$ 时 $M_t = I - 0.5\,k_t k_t^\top$"，据此得 $M_1=\mathrm{diag}(0.5,1)$、$M累积=0.5I$ 与 ground truth $S_4=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$；而按来源公式 $M_t=(I-k_t k_t^\top)\mathrm{Diag}(\alpha)$ 应得 $M_1=\mathrm{diag}(0,0.5)$、$M累积_1=M_2M_1=0$，示例的全体矩阵与 ground truth 都要重算｜引文依据：报告 §2.1.1 Eq.1 LaTeX「\mathbf{S}_{t}=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})\mathbf{S}_{t-1}+\beta_{t}\bm{k}_{t}\bm{v}_{t}^{\top}」，§5.1.2「\mathbf{M}_{t}:=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})」；报告 §2.1.1 正文「KDA applies channel-wise decay before the delta-rule update」亦要求 $\mathrm{Diag}(\alpha_t)$ 作用在 $S_{t-1}$ 上；姊妹页 wiki/kda/index.html 行 239 写的是 $S_t=(I-\beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_t k_t v_t^\top$，即正确形式｜修复要求：行 127、371、615、630 的 $M_t$ 一律改为 $M_t:=(I-\beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)$，并在首次出现处说明 $\mathrm{Diag}(\alpha_t)$ 先作用于 $S_{t-1}$ 再擦写；§4.5 手算必须在该式下重算 $M_t$、$M累积_i$、$\tilde S_i$、ground truth $S_4$ 及"误用直接求和"的错误值，或把该示例明确改写为"构造的线性递归（非 KDA 真实 $M_t$）"并给出与真实公式的差异说明；改动后重新核对 §4.3/§4.4 中所有引用 $M$ 的推导｜修复：｜复验：

- [重要·技术] 来源：K3 报告 §5.1.1｜位置：行 298、行 93（核心问题第 3 条解答折叠块）｜问题：页面称设备内 CP "SM 之间通过 shared memory / L2 协作，不走网络"，但报告只说明该并行完全发生在单设备内、不产生跨设备通信，未给出"SM 间通过 shared memory / L2 协作"这一机制描述，属无来源支持的机制描述｜引文依据：报告 §5.1.1「this parallelism is entirely intra-device and incurs no cross-device communication」（全节无 shared memory / L2 相关表述）｜修复要求：删除 shared memory / L2 的机制描述，或降级为明确标注的推断（如"SM 间同步由设备内机制完成，报告未展开"），不得作为事实陈述保留｜修复：｜复验：

- [重要·技术] 来源：K3 报告 §5.1.2｜位置：行 423（§4.4 图内文字"$S_T^{[2]}=\tilde S_2+M累积_2\cdot S_T^{[1]}$（即 rank 2 的入状态）"）｜问题：把 $S_T^{[2]}$ 标注为"rank 2 的入状态"与页面自用的定义矛盾。按 §4.3 采用、且与报告一致的定义，$S_T^{[i]}$ 表示"离开 rank $i$、进入 rank $i+1$ 的状态"，故 rank 2 的入状态是 $S_T^{[1]}$；$S_T^{[2]}$ 是 rank 2 处理后的状态（§4.5 中等于 $S_4$）。图内 rank 1/rank 2 标签与上标 0/1/2 也差一位，读者无法判断每个状态属于哪个 rank｜引文依据：报告 §5.1.2「\mathbf{S}_{[i]}^{T_{i}} denotes the state leaving rank i and entering rank i+1」；页面行 466「$S_T^{[2]}=\tilde S_2+M累积_2\cdot S_T^{[1]}=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$，与顺序计算的 $S_4$ 完全一致」｜修复要求：把行 423 括注改为"即 rank 2 处理后的状态（序列末状态）"；并统一图示的 rank 编号与上标（例如改标为 rank 0/rank 1，或在图内直接用"进入 rank 2 的入状态 $=S_T^{[1]}$"）｜修复：｜复验：

- [重要·表述] 来源：不适用｜位置：行 176、246、304、487、582｜问题：每章收尾的过渡句使用同一固定模板"本章讲了 X。但 Y——下一章讲 Z。"，四处几乎逐字重复，属规范禁止的固定句式与元话语式收尾；行 174 与行 176 又对同一件事（"框架已建立，后续四章各解一个 regime"）连续复述｜引文依据：guides/concept/style-guide.md 第 8 节「用一至两句说明前一节结论与下一节问题的关系。不使用固定句式，也不为形式完整而添加过渡」；guides/concept/check.md 2.2 第 12 项把"本页将…""下面来看…"一类元话语列为不合格表述｜修复要求：五处收尾改为各不相同的衔接句，直接由内容承接下一章问题（例如从"空隙""纯 TP 填不满""直接求和失效"这些具体缺口切入），删除"本章讲了…"的复盘句式；行 174 / 176 的重复表述二选一｜修复：｜复验：

- [轻微·技术] 来源：K3 报告 §5.1.2 Eq.17｜位置：行 403、行 429、行 509｜问题：把交换的 $M$ 说成与 $S$ 同形状（"$S\in\mathbb{R}^{d_k\times d_v}$ 及同形状的 $M$"）；报告定义累积转移为 $d_k\times d_k$，与页面行 385 自身的 $M_{t\leftarrow1}^{[i+1]}\in\mathbb{R}^{d_k\times d_k}$ 不一致（K3 中 $d_k=d_v=128$ 故尺寸恰好相同，但形状定义不同）｜引文依据：报告 Eq.17「\mathbf{M}_{[i+1]}^{t\leftarrow 1}:=\prod_{r\leftarrow 1}^{t}\mathbf{M}_{r}\in\mathbb{R}^{d_{k}\times d_{k}}」｜修复要求：改为"状态 $S\in\mathbb{R}^{d_k\times d_v}$ 与累积转移 $M\in\mathbb{R}^{d_k\times d_k}$"，使行 403/429/509 与行 385 一致｜修复：｜复验：

- [轻微·技术] 来源：K3 报告 §5.4.2｜位置：行 553、行 557、行 597｜问题：把报告的"the projected inputs of the draft tokens"具体化为"经过 $W_{q/k/v/\beta/\alpha}$ 投影后的 $q,k,v,\beta,\alpha$"，并称其为"几个 $d_k$ 维向量"；报告未列举这些量，且 $v$ 是 $d_v$ 维、$\beta$ 是标量，都不是 $d_k$ 维向量，属未标注的推断＋量级描述不精确｜引文依据：报告 §5.4.2「The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself」（未出现 q/k/v/β/α 的列举）｜修复要求：把列举改为明确标注的解释（"即 KDA 前向的投影输入 $q,k,v,\beta,\alpha$"），并删去"几个 $d_k$ 维向量"的维数断言或改为准确表述｜修复：｜复验：

- [轻微·表述] 来源：不适用｜位置：行 453（"下面用 KCP 分解重组，看能否得到同样的值"）、行 244（"工程定位："）｜问题：残留临场引导语与元话语标签｜引文依据：guides/concept/check.md 2.2 第 12 项列举"下面来看…"等元话语为不合格表述｜修复要求：行 453 改为直陈（如"用 KCP 分解重组得到："）；行 244 去掉"工程定位："标签，直接叙述｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 3 / 轻微 3
- 处置：修复。阻断项（Eq.1/$M_t$ 漏括号，连带 [C4] 引文与 §4.5 手算）必须先改回带括号形式并按来源公式重算示例；重要项分别删除无来源的 shared memory/L2 机制描述、修正 §4.4 图注下标、消除五处固定过渡句式。公式与手算改动后须重新核对 §2.1.1/§5.1.2 来源并重跑 validate.py。
- 已核对无误的项（留档）：N1 $d_k=d_v=128$ 与 N2 96 head 与官方 config.json 一致（`linear_attn_config.head_dim=128`、`v_head_dim=128`、`num_heads=96`；`dtype=bfloat16` 支持 32KB 换算）；N4 $\alpha\in(e^{-5},1)$ 与报告 §2.1.1 Eq.5「\alpha^{h}_{t}=\exp(g^{h}_{t})\in(e^{g_{\min}},1)^{d_{k}}」（$g_{\min}=-5$）一致；N3 H100 132 SM；[C1]/[C2]/[C3]/[C4]/[C5] 引文主体与报告 §5.1.1、§5.1.2、§5.4.2 原文逐句吻合（仅 [C4] 的 $M_t$ 括号见上）；引文编号 [14]/[141]/[25] 与官方 PDF 一致（arXiv HTML 另行编号，非误）；§4.4 prefix scan 递推式与报告 Eq.17 重索引后等价，$S\leftarrow M^{T_j\leftarrow1}_{[j]}S+\tilde S^{T_j}_{[j]}$ 与报告一致。