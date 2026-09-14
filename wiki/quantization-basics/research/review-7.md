<!-- review-meta
round: 7
page: wiki/quantization-basics/index.html
reviewed_content_sha256: cd92fd3bea6a096c
-->
# 量化基础审查记录（第 7 轮）

- 页面版本：f26040465da13d87fa3f5ba5e229363d5963084d（wiki/quantization-basics/index.html 工作树）
- 审查时间：2026-09-14
- 审查者：独立子代理（编排者派发的独立审查者，未参与写作，未参与前序轮次）
- 已完整阅读章节：引言、核心问题、1. 量化要解决什么问题——压缩比与精度损失的权衡、2. 均匀量化怎么把浮点变成整数——仿射变换与对称特例（含 2.1 对称量化特例、2.2 手算 [1.2,3.4,5.6]）、3. 量化误差从哪来——离散级粗、舍入与裁剪、4. 对称/非对称 与 量化粒度（含 4.1、4.2）、5. PTQ vs QAT 与 浮点 block-wise 量化（含 5.1、5.2、5.3）、来源与范围说明；含全部 details 折叠块、表格、两段代码块、概况页 overview.html 与页内脚本

## 来源核对（本轮实际核对到的版本）

- Gholami, Kim, Dong, Yao, Mahoney, Keutzer《A Survey of Quantization Methods for Efficient Neural Network Inference》arXiv:2103.13630（全文 HTML 版）：
  - §III-B 原文「Q(r)=Int(r/S)−Z」「r̃=S(Q(r)+Z)」——页面 C1 反号说明「本文 z=−Z，两式等价」代回复算成立（x_q=Int(x/S)−Z、x̂=S(x_q+Z)）。
  - §III-B 原文「the recovered real values r̃ will not exactly match r due to the rounding operation」——对应 C5/F1。
  - §III-C 原文「[α,β] denotes the clipping range」「S=(β−α)/(2^b−1)」、restricted range 记「max(|r|)/(2^{n−1}−1)」——对应 C2/C3（页面 b 即来源 n）。
  - §III-C 原文「this approach is susceptible to outlier data in the activations」「could unnecessarily increase the range and … reduce the resolution」——对应 C6，逐字命中。
  - §III-E 小节 Layerwise / Groupwise / Channelwise Quantization——对应 C4。
  - §III-G 小节 QAT / Post-Training Quantization，STE 原文「ignores the rounding operation and approximates it with an identity function」——对应 C7/C8。
- Jacob et al. 2018 CVPR 原文 PDF 返回 HTTP 403，改由综述 §III-G 与 NVIDIA 博客（fake-quantization + STE）交叉核对，结论一致。
- OCP Microscaling Formats v1.0（经 emergentmind.com/topics/mxfp4 与 zeroentropy.dev/concepts/mxfp4 二次来源）：块 32 + 8-bit E8M0 power-of-two scale + 4-bit E2M1 元素；「8 bits scale + 32 × 4 bits = 136 bits per block」「4.25 bits/element」；E2M1 码本「{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}」，「maximum absolute representable element is 6」——对应 C9/N2/N4。
- NVIDIA TensorRT-LLM 仓库 examples/models/core/gpt_oss/README.md 原文「GPT-OSS is a reasoning model with MoE weights quantized with mxfp4. All the other weights are in bf16.」——对应 C10，逐字命中。
- NVIDIA Developer Blog：AbsMax、affine (s,z)、对称 z=0、Llama2 7B FP16「~14 GB」——对应 C2/N1。
- karam-nus.github.io/language-modelling/20_quantization_fundamentals：PTQ「Uses calibration dataset (~128–512 samples)」——对应 C7/N3。

## 数值复算（均可复算，与页面一致）

- 7B×2 B=14 GB、÷2=7 GB、÷4=3.5 GB；INT4 对称 14 非零等级+0=15。
- [1.2,3.4,5.6]：s=5.6/7=0.8、x_q=[2,4,7]、x̂=[1.6,3.2,5.6]、误差 [+0.4,−0.2,0]，上界 s/2=0.4。
- [1.2,3.4,5.6,50.0]：s=50/7≈7.143、x_q=[0,0,1,7]、x̂≈[0,0,7.143,50]、误差 ≈[−1.2,−3.4,+1.543,0]。
- [2.0,4.0,6.0]：s=6/7≈0.857、x_q=[2,5,7]、x̂≈[1.71,4.29,6.0]、误差 ≈[−0.29,+0.29,0]，s/2≈0.43。
- 有效位宽：块 32 (32×4+8)/32=4.25 bit；块 128 (128×4+8)/128=4.0625 bit。
- E2M1：16 码字→15 个不同数值（±0 同值），最大绝对值 6。

## 机械与功能核对

- 代码：页面「代码：对称均匀量化与离群值放大（可独立运行）」脚本按原样 python3 执行，输出（E1/E2/E3）与页面「预期输出」逐行一致。
- validate.py：`validation ok: wiki/quantization-basics/index.html`。
- 引用编号：正文/图注/summary 侧 C1–C10、F1–F4、N1–N4 与来源章文献表逐条对应，无跳号、无未引用项、无编号冲突（v1/v2 差异项已按本轮核对的全文版处理）。
- 页面链接：`../mxfp4-qat/index.html` 存在；overview.html 与 index.html 互链；无「（待生成）」占位。
- 两级问题块：核心问题 5 条、各章本章问题共 15 条，每条均有「解答：」折叠块，答案独立可读且与正文结论一致；核心问题答案均指明完整论证章节。
- 公式：数学符号均由 KaTeX 承载，`dojo:summary` 无公式需渲染；章节编号（1–5、2.1/2.2、4.1/4.2、5.1/5.2/5.3）与 `来源与范围说明` 固定小节命名合规；details summary 前缀（补充/展开/代码）合规。

## 问题

- [轻微·可读性] §2.1（L179）：句内「（牺牲 $-128$ 这一端换算术对称）」中「换算术对称」不成词，疑为「换取（网格关于 0）对称」误植，读者须回看 L222 本章问题答案才能确认含义｜引文依据：不适用（页面内文字）｜修复要求：改为可独立读通的表述，如「（牺牲 $-128$ 这一端，换取网格关于 0 对称）」｜修复：｜复验：
- [轻微·格式] §1（L121 与 L135）：同一量词两处写法不一致——L121 用原文字符「INT8 每参数 1 字节（2× 压缩），INT4 每参数 0.5 字节（4× 压缩）」，L135 同页写作「压缩 $2\times$」「压缩 $4\times$」｜引文依据：不适用（guide/concept/style-guide.md §11「同一变量在页面中保持同一种写法」，数学运算符须用 LaTeX）｜修复要求：两处统一为 LaTeX 形式（如 $2\times$、$4\times$）｜修复：｜复验：
- [轻微·表述] 各正文章末过渡句（L153、L240、L294、L376）：四章末尾由同一固定句式「A 已经……，但 B 还需……——C」构成，属 style-guide §8 明令不用的固定句式｜引文依据：不适用（guide/concept/style-guide.md §8「不使用固定句式，也不为形式完整而添加过渡」）｜修复要求：打破该模板，逐章用一两句直接陈述前一节结论与下一节问题的实质关系｜修复：｜复验：
- [轻微·表述] 来源与范围说明·构造示例（L616）：「本章伪代码与可运行代码」中「本章」在本章（来源与范围说明）内无所指——该伪代码与可运行代码在第 5 章 5.2 节；同列表其它条目均以章名指代（如「均匀量化章」「量化误差章」「粒度章」）｜引文依据：不适用（页面内指代）｜修复要求：改为明确章节指代，如「第 5 章 5.2 节的伪代码与可运行代码」｜修复：｜复验：

## 未构成问题的核对说明（避免重复申报）

- 「等级数 $2^b$」（L142/L276/L289）与「对称 INT4 15 个等级 / 剩 $2^b-1$ 个」（L123/L246）：前者指 b-bit 整数网格容量、后者指对称受限范围，L246 已显式给出 $2^b-1$ 的换算，非同页矛盾，不报。
- §5.3「MXFP4 相对定点 INT4 的两点优势」：由 C9/N2 已核对的格式事实（E2M1 自带指数位、块共享 scale、power-of-two scale）直接推得，作为解释而非来源结论陈述，边界在正文与来源章均已交代，不报。
- 全页无「本页」为主语的元话语、无第一/第二人称会话指代、无调试叙事与临场评价；「本文」自称合规（style-guide §12）。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（仅 4 条轻微项；阻断与重要均为 0，无返回规划项）