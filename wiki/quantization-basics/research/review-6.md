<!-- review-meta
round: 6
page: wiki/quantization-basics/index.html
reviewed_content_sha256: ca4d1bb921995319
-->
# 量化基础审查记录（第 6 轮）

- 页面版本：7529440685f51591dca6ad861a283e6dd9502b19（工作树 wiki/quantization-basics/index.html 的 git blob 哈希）
- 审查时间：2026-09-14 17:11
- 审查者：独立子代理
- 已完整阅读章节：引言与「核心问题」块；1. 量化要解决什么问题——压缩比与精度损失的权衡（含本章问题）；2. 均匀量化怎么把浮点变成整数——仿射变换与对称特例（含 2.1 对称量化特例、2.2 手算、本章问题）；3. 量化误差从哪来——离散级粗、舍入与裁剪（含本章问题）；4. 对称/非对称 与 量化粒度——分布适配与离群值隔离（含 4.1、4.2、本章问题）；5. PTQ vs QAT 与 浮点 block-wise 量化——训练时是否见过量化误差（含 5.1、5.2、5.3、本章问题）；来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件及其限制）。以上含全部折叠块、callout、伪代码块与可运行代码块；另完整读了 overview.html 并逐条比对一致性。

## 问题

- [重要·技术] head「主要依据」块与「来源与范围说明」C8：所引 CVPR 2018 论文《Quantization and training of neural networks for efficient integer-arithmetic-only inference》的作者列表写作「Jacob, Kligys, Chen, Zhu, Corrado, Le」，与所引版本（arXiv:1712.05877 / CVPR 2018 Open Access）的文献表不符——Corrado 与 Le 都不是该文作者，实际 8 位作者为 Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko。该条目是页面自报的主要依据之一，作者名可在编目页直接证伪，属来源信息与所引版本不符（不影响任何技术论断，故未计为阻断）。｜引文依据：arXiv:1712.05877 作者栏 "Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko"；CVPR 2018 Open Access（openaccess.thecvf.com，DOI 10.1109/CVPR.2018.00286）同；全文检索无 Corrado、无 Le。｜修复要求：把「主要依据」块与 C8 两处的作者列表改为与文献表一致的 8 位作者，或统一写作「Jacob et al.」。｜修复：｜复验：

- [重要·技术] 「来源与范围说明」C6：以引文片段「outliers no longer poison their channel」标注来源 emergentmind.com/topics/mxfp-formats，但该 URL 页面正文（含引用列表）不含此句，也不含 poison 与 channel 两词，外部检索同样定位不到该句，引文依据无法核对。C6 的论断本身（per-tensor 下离群值毒化全张量）由同条目所引 Gholami §III-C 支持，故问题只在被引片段是否真实存在。｜引文依据：（支持论断）ar5iv 2103.13630 §III-C："this approach is susceptible to outlier data in the activations. These could unnecessarily increase the range and, as a result, reduce the resolution of quantization."；（不支持该引文片段）对 emergentmind.com/topics/mxfp-formats 的两次定向抓取均返回「poison、channel 均不存在」。｜修复要求：删除该引文片段，或换成可在标注页面定位到的原句及其出处。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明」C7：把 Figure 4 定位在 §IV（原文「§III-G 与 §IV（Figure 4 的 PTQ 分支）」），但 Figure 4 实际在 §III-G「Fine-tuning Methods」之下，§IV 的标题是「Advanced Concepts: Quantization Below 8 bits」，不含该图。按页面标注的章节去查会扑空。｜引文依据：ar5iv 2103.13630 中 Figure 4 题注 "Comparison between Quantization-Aware Training (QAT, Left) and Post-Training Quantization (PTQ, Right)"，位于 §III-G 标题之后、§III-G1 之前；同页 §IV 标题为 "Advanced Concepts: Quantization Below 8 bits"。｜修复要求：把 Figure 4 的定位改为 §III-G；若 §IV 对 PTQ 流程确无所指，一并删除该章节号。｜修复：｜复验：

- [轻微·可读性] 「来源与范围说明」→「辅助解释与类比边界」第 1 条：列出类比「量化网格类比刻度尺」，但正文通篇（含折叠块与 callout）没有出现刻度尺类比；同节另外两条类比（STE 透明窗户见 5.2 callout、per-tensor 单点故障见第 3 章 callout）都在正文有对应位置。读者按该清单回正文核对会找不到出处。｜引文依据：不适用｜修复要求：或在正文相应位置补上该类比并保持边界说明，或把该条从清单中删除。｜修复：｜复验：

- [轻微·可读性] 5.2 伪代码折叠块末段：「这就是 STE『透明窗户』的含义」在术语引入之前使用该术语——「透明窗户」这一类比直到本章 callout（"可以把 STE 暂时理解为『透明的窗户』"）才首次给出，位置在该折叠块之后约百行。｜引文依据：不适用｜修复要求：把该句改为不含未定义术语的表述（如「反向时梯度原样穿过量化算子」），或把「透明窗户」的引入前移到该折叠块之前。｜修复：｜复验：

- [轻微·技术] 第 3 章离群值对比折叠块末段：「离群值让正常值的误差从不到 0.4 放大到 1 以上」与同页数字不符——无离群值时的误差是 $[+0.4,-0.2,0]$，最大幅度恰为 $0.4$（$=s/2$），不是「不到 0.4」；同页代码观察段写作「误差幅度从 $0.4$ 以内放大到 $3.4$」，两处表述不一致。｜引文依据：不适用（页面自身数字：第 2 章表与手算给出 $s=0.8$、$s/2=0.4$、误差 $[+0.4,-0.2,0]$，运行代码输出同为 0.4）｜修复要求：把该句改为「不超过 $0.4$」或「$0.4$ 以内」，与代码观察段统一。｜修复：｜复验：

## 无问题核对（外部来源片段与机械项）

- C1/F1：ar5iv 2103.13630 §III-B 记 $Q(r)=\mathrm{Int}(r/S)-Z$、$\tilde r=S(Q(r)+Z)$；页面 $z=-Z$ 的换算与两式等价，核对通过。
- C2/F2：同源 §III-C "S is chosen as max(|r|)/(2^{n−1}−1), which only uses the range of [−127,127]" 支持页面的对称受限范围与 scale 公式。
- C3/F3：同源 §III-C "α=r_min, and β=r_max" 的 non-symmetric 方案支持 $s=(\beta-\alpha)/(q_{\max}-q_{\min})$、$z=\mathrm{round}(q_{\min}-\alpha/s)$；折叠块推导可复算。
- C4：同源 §III-E「Quantization Granularity」给出 layerwise / channelwise / groupwise 的分级，支持 per-tensor → per-channel → per-block 的由粗到细陈述。
- C5：同源 §III-B "the recovered real values r̃ will not exactly match r due to the rounding operation" 支持舍入误差来源；三来源三分法页面已在正文与 C5 注中明示为教学归纳，非来源结论。
- C6 论断部分：§III-C "These could unnecessarily increase the range and, as a result, reduce the resolution of quantization" 支持离群值毒化。
- C7 校准数据：karam-nus.github.io/language-modelling/20_quantization_fundamentals 原文 "Uses calibration dataset (~128–512 samples)"，页面注明为常见工程值。
- C9/N2/N4：zeroentropy.dev/concepts/mxfp4 与 emergentmind.com/topics/mxfp4 均给出块 32 + E8M0 + E2M1、「136 bits per block」「4.25 bits/element」、E2M1 数值集 $\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$ 与最大幅值 6，与页面一致。
- C10：NVIDIA TensorRT-LLM 的 gpt-oss 说明原文 "MoE weights quantized with mxfp4. All the other weights are in bf16" 与页面引文一致。
- 数字复算：$s=5.6/7=0.8$、$x_q=[2,4,7]$、$\hat x=[1.6,3.2,5.6]$、误差 $[+0.4,-0.2,0]$；$s=50/7\approx7.143$、$x_q=[0,0,1,7]$、$\hat x\approx[0,0,7.143,50]$、误差 $\approx[-1.2,-3.4,1.543,0]$；$6/7\approx0.857$ 组 $x_q=[2,5,7]$；$(32\times4+8)/32=4.25$、$(128\times4+8)/128=4.0625$、$0.25/4=6.25\%$；7B×2B=14 GB、INT8 7 GB、INT4 3.5 GB。全部与页面一致，且与 overview.html 一致。
- 代码：抽出 `<pre><code class="language-python">` 块用 python3 实跑，三例输出与页面「预期输出」块逐行一致（含 7.1429 / 1.5429 / 块 A 与 E1 误差相同）。
- 页面功能与机械项：validate.py 返回 "validation ok"；KaTeX/Prism/CSS 本地资源均存在；h2/h3 锚点齐备（末 6 个 h3 由页面脚本生成唯一 id）；页面无图与 SVG，故无图注/像素读数项；无 `$...$` 出现在 alt；无「（待生成）」；`dojo:topics=训练与优化`、`dojo:tag=量化` 均在词表内；index.html 与 overview.html 互链；`../mxfp4-qat/index.html` 存在且链接文字与目标页标题相符；`×` 属 validate.py 明示排除的中文排版字符（代码块内 `∂ℓ/∂ŵ` 亦属排除范围）。
- 表述通读：无第一人称复数、无第二人称、无「本页将/下面来看/需要注意的是」式元话语（引言泛指句与章末过渡句按 style-guide 第 7、8 节属结构说明），「本文」自称符合 style-guide 第 12 节。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复（本轮无阻断项；两条重要项均为引文与所引来源不符，按第 4 节逐条修复后进入下一轮复验）
