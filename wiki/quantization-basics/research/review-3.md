<!-- review-meta
round: 3
page: wiki/quantization-basics/index.html
reviewed_content_sha256: 497ea0143755a2ca
-->
# 量化基础审查记录（第 3 轮）

- 页面版本：0afaec558f012419905af696c5525ad5648a098c
- 审查时间：2026-09-13 19:10
- 审查者：独立子代理
- 已完整阅读章节：核心问题 / 1. 量化要解决什么问题——压缩比与精度损失的权衡 / 2. 均匀量化怎么把浮点变成整数——仿射变换与对称特例（含 2.1、2.2 与全部折叠块）/ 3. 量化误差从哪来——离散级粗、舍入与裁剪（含折叠块）/ 4. 对称/非对称 与 量化粒度——分布适配与离群值隔离（含 4.1、4.2、折叠块、表格）/ 5. PTQ vs QAT 与 浮点 block-wise 量化——训练时是否见过量化误差（含 5.1–5.3、伪代码与可运行代码、对照表、callout）/ 来源与范围说明；并完整阅读 overview.html

## 机械验证结果（本轮）

- 运行 `.dojo/scripts/validate.py wiki/quantization-basics/index.html` → `validation ok`。
- 可运行代码已实际执行（Python 3，`/tmp/q.py`，原样复制页面 5.2 的代码块）：输出与页面「预期输出」逐行一致——E1 `scale=0.8, x_q=[2,4,7], error=[0.4,-0.2,0.0]`；E2 `scale=7.1429, x_q=[0,0,1,7], error=[-1.2,-3.4,1.5429,0.0]`；E3 块 A `scale=0.8, ...error=[0.4,-0.2,0.0]`、块 B `scale=7.1429, x_hat=[50.0]`。
- 手算复算全部通过：$[1.2,3.4,5.6]$ INT4 对称 → $s=0.8$、$x_q=[2,4,7]$、$\hat{x}=[1.6,3.2,5.6]$；本章问题 $[2.0,4.0,6.0]$ → $s=6/7\approx0.857$、$x_q=[2,5,7]$、$\hat{x}\approx[1.714,4.286,6.0]$、误差 $\approx\mp0.286$ 均 $\le s/2\approx0.4286$；$[1.2,3.4,5.6,50.0]$ per-tensor → $s=50/7\approx7.1429$、$x_q=[0,0,1,7]$；per-block 有效位宽 $(32\times4+8)/32=4.25$、$(128\times4+8)/128=4.0625$。
- 结构项：`dojo:type=concept`、`dojo:topics=训练与优化`（词表内）、`dojo:tag=量化`（词表内）；`overview.html` 与 `index.html` 互链；被引用页 `wiki/mxfp4-qat/index.html` 真实存在；无「（待生成）」占位、无指向 `research/` 的路径；5 个正文 h2 均有「本章问题」且每题有 `解答：` 折叠块，页面级「核心问题」5 条均指向完整论证所在章节。

## 问题

- [阻断·技术] 「来源与范围说明」C1–C7、N1、N3 的来源标注：所有 Gholami 论文的章节号（§1、§2.1、§2.2、§2.3、§2.4、§2.5）在 `arXiv:2103.13630` 中均不存在，按标注章节号无法定位核对。该论文用大写罗马数字编号：`I. INTRODUCTION`、`II. GENERAL HISTORY OF QUANTIZATION`、`III. BASIC CONCEPTS OF QUANTIZATION`，其下为 `A.`、`B.`……；仿射量化公式在 `III-B Uniform Quantization`（Eq. 2），对称/非对称在 `III-C`，粒度在 `III-E`，PTQ 相关在 `III-G Fine-tuning Methods` 与第 IV 章。页面全部 8 处 Gholami 定位（C1 §2.1、C2 §2.2、C3 §2.2、C4 §2.3、C5 §2.4、C7 §2.5、N1 §1、N3 §2.5）都指向不存在的编号，读者按图索骥必然落空。｜引文依据：PDF（arxiv.org/pdf/2103.13630，pdftotext）第 66 行 `I. I NTRODUCTION`、第 361 行 `III. BASIC C ONCEPTS OF Q UANTIZATION`、第 330 行 `A. Problem Setup and Notations`、第 367 行 `B. Uniform Quantization`，第 111 行正文 `Q(r) = Int(r/S) − Z, (2)`。注：其中 C1/C2/C3/C6/C7 与 N1 的**实质内容**在该论文中可定位（见下条各"引文依据"），问题在于标注的章节号错误，须改成实际编号（如 §III-B、§III-C、§III-D、§III-E、§III-G、§I）。｜修复要求：把 C1/C2/C3/C4/C5/C6/C7、N1 的 Gholami 章节号全部改为论文实际编号（III-B、III-C、III-D/III-E、III-G、I 等）；无法在该论文中定位到对应内容的条目（见下条 C5）按该条要求处理。改后重新打开论文逐条核对。

- [重要·技术] 5.3 延伸与 N4：「E2M1 可表示的 16 个有限值为 $\{0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6\}$」——标注的数量"16"与所列集合的基数不符：该集合共 15 个不同数值（$0$ 与 $\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6$ 共 $1+7\times2=15$）。同一句内数量与集合矛盾，读者数一遍即得 15。｜引文依据：HuggingFace blog《Exact E2M1 on Hopper》——`There are 16 four-bit encodings because the sign bit also permits positive and negative zero, although those two encodings represent the same numerical value.` 该文并明确 `16 four-bit encodings but only 15 distinct numerical values`；AMD ROCm blog《High-Accuracy MXFP4, MXFP6, and Mixed-Precision Models》记 `MXFP4 (E2M1, range [-6, 6])`，最大幅度 6，与所列集合一致。｜修复要求：把"16 个有限值"改为"15 个不同数值（16 个 4-bit 码字，其中 $\pm0$ 编码同一数值）"，或直接写"元素可表示的数值为 $\{0, \pm0.5, \pm1, \pm1.5, \pm2, \pm3, \pm4, \pm6\}$（共 15 个）"，使数量与集合一致；N4 同步修改。

- [重要·技术] 3. 量化误差从哪来，开篇与 C5：「误差不是某个 bug，而是均匀量化机制固有的三个来源的叠加。$^{[C5]}$」，并把三来源固定为"离散级粗、舍入、裁剪"。被引来源不支持这一三分法：Gholami 论文全文没有 "grid coarseness"/"离散级粗" 概念，也没有把误差枚举为三类来源的表述（仅有 `rounding operation` 与截断范围 calibration 的分散讨论）；NVIDIA 博客亦未列出该三分法。这是把本页自己的归纳包装成来源结论。｜引文依据：Gholami PDF 全文检索 `coarseness` / `three source` / `sources of error` 均无命中；相关处仅 `Note that the recovered real values r̃ will not exactly match r due to the rounding operation.` 与 `reduce the resolution of quantization`（离群值一节），未构成三来源分类。｜修复要求：删除该句的 `[C5]` 标注，改写为明确标注的本页归纳（如"本页把误差拆成三个来源"），并只对"舍入误差上界 $s/2$"与"裁剪误差"保留可核对的来源标注；C5 条目相应改为只登记可被来源支持的子论断。

- [重要·技术] overview.html「2. 为什么需要它」：「位宽低则网格等级少（INT4 对称只有 15 个非零等级）」——数值错误且与主页面矛盾。INT4 对称共 15 个等级（含 0），其中非零等级为 14 个（$\pm1$ 到 $\pm7$）。主页面 1 章与 3 章均写「15 个等级（含 0，$\pm 1$ 到 $\pm 7$ 共 14 个非零等级）」，与本页自相矛盾。｜引文依据：`index.html` 第 163 行 `INT4 对称有 15 个等级（含 0，±1 到 ±7 共 14 个非零等级）`；`overview.html` 第 36 行 `INT4 对称只有 15 个非零等级`。｜修复要求：把 overview.html 的「15 个非零等级」改为「15 个等级（含 0，非零 14 个）」或「14 个非零等级」，与 index.html 一致。

- [轻微·技术] 2 章 C1/F1：页面把 zero-point $z$ 定义为 $x_q=\mathrm{round}(x/s)+z$、$\hat{x}=s(x_q-z)$，并称 Gholami「同一公式的现代统一写法」。但 Gholami 的式子符号相反：$Q(r)=\mathrm{Int}(r/S)-Z$、$\tilde r=S(Q(r)+Z)$（即本页 $z=-Z$，页面对实数 0 映射到的整数取 $z$，论文取 $-Z$）。两者可互相换算、页面公式自身正确，但称两者"同一公式"不准确，读者对照原文会遇到符号不一致。｜引文依据：Gholami PDF 第 111 行 `Q(r) = Int(r/S) − Z, (2)`；第 449 行 `r̃ = S(Q(r) + Z)`。｜修复要求：在 C1/F1 处注明 zero-point 的符号约定差异（本页 $z$ 与论文 $Z$ 反号），或把该公式的来源改为仅标注采用现代统一写法的来源。

- [轻微·技术] 5.1 与 C7/N3：「用少量校准数据（典型 128–512 样本）」被同时归于 `Gholami et al. 2021 §2.5` 与 `karam-nus.github.io/...`。该 128–512 数值在 Gholami 论文中不存在，仅由 karam-nus 页面支持（该页原文 `Uses calibration dataset (~128–512 samples)`），属来源标注扩大。｜引文依据：Gholami PDF 检索 `128`/`512` 仅命中参考文献编号与无关页码；ar5iv 版亦确认该综述未给出校准样本数。｜修复要求：N3/C7 去掉 Gholami 归属，只保留 karam-nus 来源，并保留"常见工程值、非硬性规则"的限定。

- [轻微·维护] head 内自定义样式（第 38–66 行）定义了 `.flow-diagram`、`.flow-box`、`.flow-row`、`.flow-arrow`、`.flow-note` 等规则，但正文自始至终没有使用任何 `flow-diagram` 结构（全页无 `<svg>`、无 `foreignObject`、无该类节点）。属死 CSS，与仓库「清死 CSS」的既有做法不一致。｜引文依据：不适用（机械检索：`grep -nE "<svg|flow-diagram|foreignObject"` 仅命中第 38–66 行的 CSS 定义，正文无引用）。｜修复要求：删除这组未被使用的 `.flow-diagram*` 规则，或补上对应的结构图。

## 结论

- 统计：阻断 1 / 重要 3 / 轻微 3
- 处置：修复。阻断项（Gholami 章节号）与两项来源类重要问题（C5 三来源无来源支持、E2M1 数量与集合不符）修复后须重新核对来源；overview.html 的数值错误一并修正。修改完成后应重跑 `.dojo/scripts/validate.py`，并从修复后的完整页面重开一轮审查。