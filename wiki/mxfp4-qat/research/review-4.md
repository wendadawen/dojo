<!-- review-meta
round: 4
page: wiki/mxfp4-qat/index.html
reviewed_content_sha256: 12a63a9b48634fe7
-->
# MXFP4 量化感知训练审查记录（第 4 轮）

- 页面版本：880d2b3134c605bb133e091910d2663f7852491d（git hash-object wiki/mxfp4-qat/index.html；工作树干净）
- 审查时间：2026-09-13 20:20 CST
- 审查者：编排者派发的独立审查者（未参与写作与前三轮审查）
- 已完整阅读章节：核心问题 / 常见误解 / 前置概念 / 1. MoE 专家权重 / 2. MXFP4 编码 / 3. QAT 机制 / 4. RL 一致性 / 5. 选择性量化 / 来源与范围说明（含全部折叠块、结构图与图注）
- 外部来源核对方式：Kimi K3 技术报告 PDF（github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf，pdftotext 全文）、HuggingFace moonshotai/Kimi-K3 config.json（curl 原文）、Rouhani et al. 2023 原文（arXiv:2310.10537v3 PDF）、本地执行页面代码与技术算术复算。

## 问题

- [阻断·技术] §2 MXFP4 编码（第 223、230、245、248–249 行）与来源说明 C7（第 606 行）：块共享 scale 写作 $s_b = 2^{e_b}$，同一段又定义 $e_b$ 为「E8M0 的 8-bit 无符号指数，取值 $0$ 到 $254$」、并标注「指数范围从 $2^{-127}$ 到 $2^{127}$（指数 bias 为 127）」。按页面给出的定义代入，$s_b = 2^{e_b}$ 在 $e_b\in\{0,\dots,254\}$ 上取值 $\{1,\dots,2^{254}\}$，与同段标注的范围 $2^{-127}\sim 2^{127}$ 互相矛盾。E8M0 的正确关系是 $s_b = 2^{e_b-127}$（8 位全 1 即 255 保留为 NaN）｜引文依据：第 223 行「取值 $s_b = 2^{e_b}$，指数范围从 $2^{-127}$ 到 $2^{127}$（指数 bias 为 127，$255$ 保留为 NaN）」与第 249 行「$e_b$：E8M0 的 8-bit 无符号指数，取值 $0$ 到 $254$」两处自相矛盾；E8M0 语义为 value $=2^{E-127}$，$E\in[0,254]$｜修复要求：把 $s_b=2^{e_b}$ 统一改为 $s_b=2^{e_b-127}$（或把 $e_b$ 重新定义为去偏指数、取值域写为 $-127\sim 127$），同步第 230 行结构图、第 245 行反量化公式、第 248–249 行符号表与 C7；改动后回源核对 E8M0 定义｜修复：｜复验：
- [重要·技术] 来源说明 C6（第 608 行）：C6 称 QAT（Jacob et al. 2018）为「K3 引用 [49]」，与报告不符——报告 §4.1.4 引用的是 [50]，[49] 是 Gpipe｜引文依据：报告原文 "We perform quantization-aware training (QAT) [50] throughout the entire post-training stage"；报告参考文献 [49] = "Yanping Huang et al. 'Gpipe: Efficient training of giant neural networks using pipeline parallelism'"，[50] = "Benoit Jacob et al. 'Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only...'"｜修复要求：C6 中「K3 引用 [49]」改为「K3 引用 [50]」｜修复：｜复验：
- [重要·技术] §5 选择性量化（第 513、531、566 行）与核心问题 5 解答（第 150 行）：页面称「这张表的依据不只是 K3 报告的文字描述，还能在官方 config.json 里直接核对」，但 config.json 的 `ignore` 只列 6 条正则，不含 router、也不含 latent MoE 投影；表中「latent MoE 投影」「MoE router」两行仅由报告 C3 支持，config.json 并未把 router 排除（router 仍是 `targets: ["Linear"]` 的量化目标），无法「直接核对」，且与报告「routers remain in higher precision」存在张力｜引文依据：config.json `ignore: ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*", "re:.*vision_tower.*", "re:.*mm_projector.*"]`；报告 C3 "all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision"｜修复要求：把「这张表…可直接在 config.json 核对」限定为 config.json 覆盖的行（attention、共享专家、dense FFN 投影、lm_head、vision tower、mm_projector），并注明「latent MoE 投影」「MoE router」两行仅由报告 C3 支持；第 150 行「官方 config.json 的 ignore 列表可直接核对」同步收窄｜修复：｜复验：
- [轻微·技术] §1（第 174 行）、核心问题 1 解答（第 122 行）、展开块（第 183、199 行）、N3（第 630 行）：$896\times 92\times 33.03\mathrm{M}$ 复算为 2.7227T，应约 2.72T，页面统一写作 2.71T；由它派生的 BF16 约 5.42 TB、MXFP4 约 1.44 TB 也相应偏小（压缩比 3.76× 不变）｜引文依据：82,432 × 33,030,144 = 2,722,740,830,208 = 2.7227T（本地复算；另 2.7227T×2 = 5.445 TB，MXFP4 = 1.3614+0.0851 = 1.446 TB）｜修复要求：把 2.71T 改为 2.72T（或写 2.7T），并同步由它派生的 5.4 TB / 1.44 TB 与展开块内的分项数字｜修复：｜复验：
- [轻微·表述] 第 113、166、174、191、221、256、327、345、421、481、483、513、569、595 行：元话语与自我指代——以「本文」为主语的「本文以 K3 技术报告 §4.1.4 与官方 config.json 为依据，回答：…」「本文依赖两个前置概念…本文只使用其结论…本文仅引用其结论，不重复推导」；元话语「先看为什么…不够用」「下面用一个 4 元素的教学块…手算」「下面手算单权重的一步」「下面的可运行代码实现了…」「这里也带出 draft model 的一致性问题」「把前面几章的结论汇总成一张选择表」「这里要破的两个常见误解」「回顾全文：…」「这为后文『选择性量化』一章埋下伏笔」「注意 config 给的 hidden_size = 7168 是…」。（「本页推断」是规范要求的推断标注，不计入本项）｜引文依据：不适用｜修复要求：删去「先看/下面/这里/回顾全文/汇总成/埋下伏笔/注意」等起头，改为直接陈述；「本文…」改为直接陈述或以名词作主语｜修复：｜复验：

## 已核对通过（本轮无问题）

- 代码：折叠块内 Python 原样执行，输出与页面「预期输出」逐行一致（block1 误差全零；block2 量化 [1.0,0.5,2.0,1.0]、反量化 [0.25,0.125,0.5,0.25]、误差 [0.05,-0.025,-0.05,-0.05]；$w=0.8\to\hat w=0.75$、前向偏差 0.0500；STE 传回 1.0）。
- 手算：块 1/块 2/本章问题 1 的归一化、最近邻取舍、反量化与误差全部复算一致；相对误差 17%/25%/11%/25% 复算一致；每权重有效位宽 4+8/32=4.25 bit、16/4.25≈3.76× 复算一致。
- 数字回源：config.json 的 num_experts=896、num_experts_per_token=16、num_shared_experts=2、hidden_size=7168、routed_expert_hidden_size=3584、moe_intermediate_size=3072、num_hidden_layers=93、first_k_dense_replace=1、group_size=32、num_bits=4、type=float、symmetric=true、strategy=group、format=mxfp4-pack-quantized、quant_method=compressed-tensors、ignore 六项均与页面一致；报告 Abstract「2.8T … 104 billion activated parameters」、Table 1「Total Parameters 2.78T / #Layers 93 / Latent MoE Dimension 3584 / MoE Hidden Dimension per Expert 3072 / Routed Experts 896 / Experts Active per Token 16 / Shared Experts 2 / 1 Dense Layer」均与页面一致。
- 引文：C2「During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch.」、C3「all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision.」、C4「Draft fine-tuning follows the post-training QAT configuration (§ 4.1.4), with MoE expert weights in MXFP4 and their input activations in MXFP8…」均与报告原文逐字一致；C1「QAT throughout the entire post-training stage, covering both SFT and RL」一致；N6 的 MXFP8 元素候选（E4M3/E5M2）与 arXiv:2310.10537 Table 1 一致，且页面已声明报告未指明具体元素。
- 截断规则：arXiv:2310.10537 Algorithm 1 第 4 行「Pi = quantize_to_element_format(Vi/X), clamping normal numbers」与页面「超出最大正规值时截断、保留符号」一致，页面已声明示例未触发该分支。
- E2M1：非负可表示值 {0,0.5,1,1.5,2,3,4,6}、非零正值 7 个、最大幅度 6，复算与源一致。
- 结构：两级问题（核心问题 5 题 / 各章本章问题各 3 题）均有解答折叠块，核心问题答案均指明完整论证所在章节；moe-serving、quantization-basics 两个前置概念页均存在；index.html 与 overview.html 互链；`python3 .dojo/scripts/validate.py wiki/mxfp4-qat/index.html` 返回 `validation ok`。

## 结论

- 统计：阻断 1 / 重要 2 / 轻微 2
- 处置：修复。阻断项（E8M0 scale 公式与指数范围矛盾）与两项重要项（C6 引文编号 [49]→[50]、§5「整表可在 config.json 核对」表述过宽）必须在本轮修复后回源重核并复验；两项轻微项随本轮一并修复。
