<!-- review-meta
round: 7
page: wiki/mxfp4-qat/index.html
reviewed_content_sha256: 56b15e98711fc2c5
-->
# MXFP4 量化感知训练审查记录（第 7 轮）

- 页面版本：eabea491ba5ed5258091306aa08725d7a741d136
- 审查时间：2026-09-14 17:08
- 审查者：独立子代理
- 已完整阅读章节：引言 → 核心问题 → 常见误解 → 1. MoE 专家权重——896 个专家压到 4-bit 能省多少显存 → 2. MXFP4 编码——一个权重值怎么用 4-bit 表示 → 3. QAT 机制——前向和反向做了什么，与 PTQ 差在哪 → 4. RL 一致性——QAT 怎么贯穿 SFT 和 RL 且不产生 mismatch → 5. 选择性量化——K3 量化了哪些组件、不量化哪些 → 来源与范围说明（含全部 details 折叠块、代码块、flow-diagram 图注与 overview.html）

## 问题

- [重要·技术] 来源与范围说明 · C6：C6 写"Jacob et al. 2018 …（CVPR 2018, pp. 2704–2713，K3 引用 [49]）"，其中引文编号与所引版本文献表不符。｜引文依据：官方 k3_tech_report.pdf §4.1.4 原文为 "We perform quantization-aware training (QAT) [50] throughout the entire post-training stage"；同文件文献表条目 "[50] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko (2018) Quantization and training of neural networks for efficient integer-arithmetic-only inference. In CVPR, pp. 2704–2713"，而 "[49] Yanping Huang et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism" 与该论断无关。arXiv HTML 渲染版（arxiv.org/html/2607.24653、ar5iv）中该条为 [48]，同样不是 [49]。CVPR 2018、pp. 2704–2713、年份 2018 三项均核对无误，仅编号错误。｜修复要求：将 C6 的"K3 引用 [49]"改为"K3 引用 [50]"，或改为不带编号的"K3 引用 Jacob et al. 2018"。｜修复：｜复验：

- [轻微·技术] 来源与范围说明 · C9：括号内"$S.00.1_2 = 2^{-1}$ 为最小正规值"的术语有误。｜引文依据：E2M1 指数 bias 为 1，$S.00.1_2$ 落在指数域为 0 的分支（ONNX "Float stored in 4 bits" 原文公式："exponent = 0: $(-1)^S b_0 2^{-1}$"），是次正规值，不是正规值；E2M1 的最小正规值是 $1.0$（指数域 = 1）。ONNX 文档该处原表只标 "Min $S.00.1_2 = 2^{-1}$"，未称其为"正规值"。本页他处（第 2 章正文、简化条件）以"最大正规值"指 $6$（$6$ 确为正规值），据此最小正规值应为 $1.0$，与"0.5 是最小正规值"自相矛盾。｜修复要求：改为"$S.00.1_2 = 2^{-1}$ 为最小非零值（即最小次正规值）"。｜修复：｜复验：

- [轻微·技术] 第 2 章 · 反量化公式与其符号表：同一量 $\hat{x}_i$ 存在两种写法，$q_i$ 一符两义。｜引文依据：公式写作 "$\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$"（正文与 flow-diagram 图注同）；紧随其后的符号表却写 "$\hat{x}_i = s_b \cdot q_i$"，并把 $q_i$ 定义为"第 $i$ 个权重量化到的 E2M1 码点，幅度取自 $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ 并带符号"。前者把 $q_i$ 当作需要 $\mathrm{FP4}(\cdot)$ 解码的码点，后者把 $q_i$ 当作已解码的数值直接乘 scale；按符号表对 $q_i$ 的定义，$s_b \cdot q_i$ 不成立。代码折叠块的"验证的机制"进一步写"nearest_e2m1 对应 [F1] 中的 $\mathrm{FP4}(q_i)$ 量化"，而 nearest_e2m1 是量化（取值舍入），F1 是反量化，映射方向也不一致。｜修复要求：统一 $q_i$ 的单一含义——若 $q_i$ 表示 E2M1 数值，则符号表与公式一并去掉 $\mathrm{FP4}(\cdot)$（统一写作 $\hat{x}_i = s_b \cdot q_i$）；若 $q_i$ 表示 4-bit 码，则符号表补写 $\mathrm{FP4}(q_i)$ 为"码到数值"的解码映射，并改正"幅度取自 …"一句。｜修复：｜复验：

- [轻微·来源] 开篇 blockquote.meta · 主要依据：把 OCP MX 规范与其配套论文合并为同一署名。｜引文依据：meta 写"OCP Microscaling Formats v1.0（Rouhani et al. 2023，arXiv:2310.10537）"，使 arXiv 编号看起来属于该规范；本页 C5 却将二者分列："OCP Microscaling Formats (MX) Specification v1.0 (September 2023) §5.4.1、§6.3；Rouhani et al. 2023 "Microscaling Data Formats for Deep Learning"（arXiv:2310.10537，Algorithm 1）"。arXiv:2310.10537 是 Rouhani 等的论文编号，OCP MX 规范是另一份独立文档。｜修复要求：meta 改为把规范与论文并列，如"OCP Microscaling Formats (MX) Specification v1.0；Rouhani et al. 2023（arXiv:2310.10537）"。｜修复：｜复验：

- [轻微·表述] 各章末尾过渡句（第 1→2、2→3、3→4、4→5 章）：四句共用同一条模板句式。｜引文依据：不适用。四句分别为"本章算清了专家权重的显存量级。但…——下一章讲 MXFP4 的编码格式。"/"本章说明了 MXFP4 的编码与量化误差。但…——下一章讲 QAT 的前向伪量化与反向 STE。"/"本章讲清了 QAT 的机制与 PTQ 的差别。但…——下一章讲 RL 中 rollout 与训练的量化一致性。"/"本章解决了 RL 中的 train-inference mismatch。但…——下一章讲选择性量化的组件划分。"，同一句式"本章……了……。但……——下一章讲……。"重复四次；guides/concept/style-guide.md §8 要求章节衔接"不使用固定句式"。｜修复要求：至少改写其中两句，使四章过渡不同构同一句式。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 4
- 处置：修复

### 本轮已核对无误、不报的项（供复验参考）

- 事实与数字：896 路由专家 / 16 per token / 2 共享专家 / hidden_size=7168 / routed_expert_hidden_size=3584 / moe_intermediate_size=3072 / num_hidden_layers=93 / first_k_dense_replace=1 / group_size=32 / num_bits=4 / symmetric / mxfp4-pack-quantized / ignore 六条正则，均与 HuggingFace moonshotai/Kimi-K3 config.json 逐字一致。
- 报告引文：C2/C3/C4 的英文引文与 N7 的"2.8T / 104 billion activated"、Table 1（§3.2）"Total Parameters 2.78T / Activated Parameters 104.2B"、4.1.4 小节标题 "Deployment-Aware Post-Training" 均与官方 PDF 一致。
- 算术：33.03M 单专家、896×92=82,432、2.72T、BF16 5.44 TB、MXFP4 1.445 TB、5.44/1.445≈3.76×、4+8/32=4.25 bit、16/4.25≈3.76×、17 字节/32 权重、块 1/块 2 误差与相对误差(17%/25%/11%/25%)、w=0.80→ŵ=0.75 手算，全部复算通过。
- 代码：页面 Python 代码实跑，输出与"预期输出"逐行一致。
- 格式：dojo:topics=训练与优化、dojo:tag=量化 均在词表内；validate.py 通过；两处前置概念页（../moe-serving/index.html、../quantization-basics/index.html）真实存在；无 Unicode 数学字符、无"（待生成）"、无第二人称/第一人称复数（style-guide §12 允许"本页/本文"自称）；details summary 前缀（展开/代码/补充）与问题块命名（核心问题/本章问题/解答：）符合 style-guide。