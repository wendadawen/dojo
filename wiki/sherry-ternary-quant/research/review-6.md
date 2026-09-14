<!-- review-meta
round: 6
page: wiki/sherry-ternary-quant/index.html
reviewed_content_sha256: 22c571a4b52326dd
-->
# Sherry 稀疏三值量化审查记录（第 6 轮）

- 页面版本：28313196a33afba9b8586c2523e107e00e226699
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，未参与前序轮次审查）
- 已完整阅读章节：引言 / 核心问题（5 条）/ 常见误解 / 1. 三值量化的打包困境（含本章问题）/ 2. 3:4 稀疏三值——4 个权重恰好 5 bit（2.1、2.2、2.3、本章问题）/ 3. STQ1_0 字节布局——1.3125 bpw 的账（3.1、3.2、本章问题）/ 4. 量化决策——缩放系数与零位怎么选（4.1、4.2、4.3、本章问题）/ 5. 为什么快——SIMD 解码与实测（5.1、5.2、本章问题）/ 6. 训练侧的 weight trapping 与 Arenas（含本章问题）/ 来源与范围说明
- 核对来源：arXiv:2601.07892v1 全文 HTML（正文 + 附录 C/D/F/G）、llama.cpp PR #22836 正文与 Performance 表（gh pr view 原文）、HuggingFace AngelSlim/Hy4-preview-GGUF 模型卡、腾讯混元「Hy4 preview 轻量版」发布稿（2026-09-01）；页面 Python 代码以 python3 实跑。

## 复核结果摘要（本轮未记为问题的项）

论文标题、作者、v1 日期、被 ACL 2026 接收（出自 PR 述及）与页面一致；论文 §2.3 三值下限 1.58（log2 3）、§3.1 与附录 A/C 的 5-bit 结构、附录 C.1「index 4 位、B−1≤4、vpshufb 16 项」、附录 C.2 的 16 模式 / 12 状态与 50% 阈值、附录 D 的 Sparse-AbsMean 与 L2 最优性、附录 F 的 ER<750 与总维度 4096、Arenas 式 Y=XTα+λ_t·XW 均已逐条对上。Table 1 与 Table 4 的每个数值（含 732.69±20.00 / 147.47±1.36 / 728.69±19.88 / 138.87±0.96 / 689.25±16.61 / 175.06±1.30 / 768.47±14.75 / 109.62±16.93、34.01/1360.0、132.13/256.56、116.83/233.44、148.27/205.50、7.55/6190.0、41.87/873.65、38.80/846.01、45.55/712.40）、18% 与由 Table 4 复算的约 27%、精度 0.519/0.519 与 0.567/0.576、ARC-Challenge 两规模反超、11%/20%/6%/35% 体积与吞吐比、"averaged over three independent runs"、Q1_0 在 PR 表中确无 bpw 列（页面「—」正确）均一致。模型卡的 42 B/256、1.3125 bpw、「-89.7% weighted SSD」与「a further -4.1% of the remainder」、770B/1.5 TB/214 GB 概数均一致。页面代码实跑输出与页内「预期输出」逐行一致（117.5101 / 19.6010 / 17.7533、SSD 0.0300 与 0.0600、42 B、1.3125）。validate.py 返回成功；图内公式用 KaTeX（非 SVG <text>）；正文无定界符外的 Unicode 数学字符；无「（待生成）」占位；两处概念链接（hy4-preview-lite、mixed-precision-quant）与 overview.html 均有效；dojo:type=concept、dojo:topics=推理系统、dojo:tag=量化 均在词表内。

## 问题

- [阻断·技术] §1「三值量化的打包困境」末尾对照表「2:4 稀疏三值」行的「bit/权重」列：写作「约 1.0（4 bit 索引 12/16 未用满）」。该数字在来源中定位不到，且与页面自身口径矛盾——页面 §2.2/§2.3 已定义「一组 4 个权重 = 4 bit 索引 + 1 bit 整体符号 = 5 bit」；2:4 的镜像折算模式数为 C(4,2)·2^(2−1)=12，同样落在 4 bit 索引内，仍需那 1 bit 符号位，故为 5 bit/组 = 5/4 = 1.25 bit/权重，与 Sherry 相同。表格此格只计索引、丢掉符号位，得 4/4 = 1.0，使 2:4 反而显得比 Sherry 更小，与本章「3:4 才是最优打包」的论证方向相反。｜引文依据：论文附录 C.2「a 2:4 scheme only utilizes C(4,2)·2^(2−1)=12 states, resulting in bit-waste.」；附录 C.1「In a hardware-aligned implementation, these B bits are partitioned into 1 sign bit and B−1 index bits…This constrains the index to 4 bits (2^4=16 entries), implying B−1≤4」；全文未给 2:4 的 bit 宽度，其缺陷被表述为 LUT 只用 12/16 与 50% 稀疏阈值，而非位宽更低。｜修复要求：把该格改为「1.25（4 bit 索引 12/16 未用满）」并与 3:4 行口径一致，或删去位宽数值、只保留「码本用不满（12/16）」并注明该行位宽与 3:4 相同。｜修复：｜复验：
- [轻微·技术] §1 第三段「…论文指出这类单元并不与超低比特量化配合，对三值权重不可用」：前半句与来源一致，「对三值权重不可用」是页面加上的推断，来源只说到该类单元面向 16/32 位浮点、与超低比特量化未结合、该交叉点基本未被探索。｜引文依据：论文附录 B.3「most efforts are not coordinated with ultra-low bit quantization, as they are largely designed for Sparse Tensor Cores on GPUs, which currently prioritize 16-bit or 32-bit floating-point arithmetic…the intersection of N:M sparsity and ternary quantization remains largely unexplored.」｜修复要求：把「对三值权重不可用」改为与来源等强度的表述（如「并不与超低比特量化结合」），或将其明确标为推断。｜修复：｜复验：
- [轻微·格式] §5.2「实测数字」表后第一段开头「逐对读这张表：」：以读者为对象的指令式引导语，属规范要排除的元话语（与「下面来看…」同类）。｜引文依据：不适用｜修复要求：删去该引导句，直接从「相对同为三值的 TQ1_0…」起句陈述比较结论。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 2
- 处置：修复（修完后重跑 .dojo/scripts/validate.py，第 7 轮从修复后的完整页面重启审查）
