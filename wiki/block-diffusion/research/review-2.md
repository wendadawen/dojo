# 块扩散（block-diffusion）审查记录（第 2 轮）

- 页面版本：index.html 691fede074cae1de78e43649afb32f9817d9131c；overview.html ca010d1715214c14700db50379c96f2707726f7d
- 审查时间：2026-08-19 18:50
- 审查者：独立子代理（第 2 轮，未参与写作与第 1 轮审查修复）
- 已完整阅读章节（index.html 按序）：head 元信息、blockquote.meta、开篇 callout、引言、核心问题（5 题含解答）、1. 块间自回归、块内并行、2. block-causal 注意力掩码（含 SVG 与补充块）、3. KV cache 与灵活长度、4. 块内去噪（含伪代码、状态表、first-hitting 补充块）、5. 当草稿器用（含流程图与贯穿示例）、来源与范围说明全部六小节、各章本章问题及解答；overview.html 全文。
- 来源核对：Arriola et al. TeX 源码（/tmp/dflash-research/bd-tex/）逐条核对了 F1–F4 的公式编号、C1–C8/C11 的机制论断与 N2 的全部数字（§6.2 Table 3 与附录 D）；DFlash TeX 源码（/tmp/dflash-research/tex/）核对了 C9/C10/N1（§3.2、§4.1、§4.2、§5 Implementation）。核对中确认 `algs/inference_alg.tex` 未被主 TeX 任何 `\input` 引用（见 E7）。
- 机械验证：`.dojo/scripts/validate.py wiki/block-diffusion/index.html` 返回 validation ok；裸 Unicode 数学字符检查（含 overview）仅在代码块内命中（豁免）；两页互链与五个概念页链接（causal-mask、standard-attention、speculative-decoding、dflash、dflash2）均存在；本地 libs 资源齐全。

## 问题

### 重要

- [重要·技术] index.html:6-7（meta description 与 dojo:summary）、:753（开篇 callout）、overview.html:43、:51：三处页面级摘要把「单轮去噪」特例写成块扩散的一般机制——「串行前向从 token 数降到块数」「块内 $L'$ 个位置一次并行去噪」「串行前向次数从 $L$ 降到 $B=L/L'$」「（callout）块内 4 个位置一次并行写出」。这与页面自己的表格（index.html:834：「$B\times$（每块去噪轮数）」）矛盾，也与第 4 章的逐轮去噪机制矛盾。｜引文依据：Arriola §6.2「the number of generation steps (NFEs) is upper-bounded by $L$ since tokens are never remasked」，Table 3 中 BD3LM 在 $L=1024$ 时 NFE 为 1K（$\approx L$，非 $B=L/L'$）；论文全文无「串行次数降到 $B$」的表述。第 1 章本章问题 2 解答（index.html:865）已正确写为「（每步内部一次或多轮并行去噪）」。｜修复要求：dojo:summary 与 description 改为「块间从左到右自回归、块内一轮或多轮并行去噪；串行前向次数为 $B\times$ 每块去噪轮数（上界 $L$，单轮时为 $B=L/L'$）」或同等准确表述；callout「块内 4 个位置一次并行写出」补一句「（这里取一轮写完；块内也可逐轮去噪，见第 4 章）」；overview.html:43、:51 同步修改。修改后逐一核对不再出现「降到块数 $B=L/L'$」的无条件表述。

- [重要·技术] index.html:788（核心问题 4 解答）、overview.html:58：「按置信度揭开一部分」为第 1 轮「置信度→first-hitting」修复的残留，与第 4 章正文（index.html:1068「与预测置信度无关」）和补充块（index.html:1110 first-hitting 描述）直接矛盾。｜引文依据：Arriola 附录 D（Inference → Improved Categorical Sampling）「the transition probability is the same for all masked tokens for a given $t$. Thus, the first timestep where a token is unmasked can be analytically sampled」——揭开哪些位置由解析采样的随机时间决定，与预测置信度无关。｜修复要求：两处「按置信度揭开」改为「按噪声调度（随机时间）揭开」的准确表述；改后在两页内 grep「置信度」，除「与预测置信度无关」类否定句外不得再有肯定式「按置信度揭开」。

- [重要·技术] index.html:1118-1119（第 4 章本章问题 1 的 summary 与解答）：summary 仍写「按置信度揭开一部分」；解答第 ③ 步写「按置信度揭开一部分」，随后又写「每轮揭开哪些由预测置信度决定——按 noise schedule 决定揭开顺序……（与按预测置信度高低排序无关）」——同一句先断言由置信度决定、再断言由 noise schedule 决定，自相矛盾且残留旧机制。｜引文依据：同上（Arriola 附录 D first-hitting 采样器）；正文 index.html:1068 已修正为「与预测置信度无关」。｜修复要求：summary 与解答第 ③ 步统一为「按噪声调度/first-hitting 的随机时间揭开一部分（已揭开不遮回）」，删除「由预测置信度决定」分句；改后该解答内不再同时出现两种归因。

- [重要·技术] index.html:1090（伪代码折叠块「简化条件」）：两处残留——(a)「真实实现里每轮揭开多少位置由噪声调度决定（置信度阈值随轮数变化）」中的「置信度阈值随轮数变化」描述的是置信度阈值揭开机制，与上方伪代码（1081 行「S ← noise_schedule(t)」）和第 4 章正文矛盾；(b)「伪代码只保留『按置信度揭开、不遮回、逐块推进』的骨架」——伪代码实际保留的是「按 noise schedule 揭开」，不是「按置信度揭开」。｜引文依据：Arriola 附录 D first-hitting 采样器（同上）；index.html:1081 伪代码行。｜修复要求：删除或改写「（置信度阈值随轮数变化）」；「按置信度揭开」改为「按 noise schedule 揭开」；改后该折叠块内无「置信度」字样（或仅出现在否定句中）。

- [重要·技术] index.html:1126（第 4 章本章问题 2 解答）：「BD3LM 的对比实验里每块 $T=25$ 步即达到与 SSD-LM 上千步可比的质量[N2]」错置主语——$T=25$ 是 SSD-LM 的设置（NFE 压到可比时 SSD-LM 退化到 gen PPL 281.3），不是 BD3LM 的每块步数（BD3LM 的 NFE 上界是 $L$，$L=1024$ 时报告 1K）。第 1 轮已修正主文（index.html:1106 正确写为「SSD-LM 用 $T=25$ 步退化到」），此处漏改。｜引文依据：Arriola 附录 D「Thus to fairly compare with SSD-LM, we also report generative perplexity for $T=25$ diffusion steps so that the number of generation steps does not exceed the sequence length (second row in Table 3)」；Table 3：SSD-LM 两行为 37.2/40K NFE 与 281.3/1K NFE，BD3LM $L‘=4$ 为 25.7/1K NFE。｜修复要求：改为「BD3LM 以不超过 $L$ 的 NFE 达 gen PPL 25.7，SSD-LM 需 $T=1$K 步/块（$\geq40$K NFE）达 37.2、压到可比 NFE（$T=25$）退化到 281.3」或同等准确表述，与 index.html:1106 一致；不得出现「BD3LM 每块 $T=25$ 步」。

- [重要·来源] index.html:1194（论断与来源（C），DFlash 定位）：C9/C10/N1 标注为「§1、§2.2、§3.1、§5」。按论文实际编号（§1 Intro、§2 Related、§3 Preliminaries、§4 Method、§5 Experiments）：§2.2 是 Related Work「Diffusion Language Models」，不含单步起草与延迟论断；§3.1 是「Speculative Decoding Speedup」，不含 C9 的单步并行起草；C9/C10 的实际支撑位置 §3.2、§4.1、§4.2 未列入。｜引文依据：DFlash §3.2「Diffusion drafters generate all $\gamma$ tokens in parallel within a single forward pass」「For moderate block sizes, $T_{\text{draft}}$ is therefore largely insensitive to $\gamma$」；§4.1「All masked positions within a block are decoded in parallel in a single forward pass」；§4.2「We randomly sample \emph{anchor tokens} from the response, use each anchor as the first position of a block」。｜修复要求：定位改为「§3.2、§4.1、§4.2、§5」；改后按新编号逐条能定位到上述原文。

- [重要·来源] index.html:1194（论断与来源（C），Arriola 定位）：定位清单含「附录 SAR Inference 算法」，但该算法在论文中不存在——`algs/inference_alg.tex`（SAR Inference）未被 `iclr2025_conference.tex` 任何 `\input` 引用，只是 arXiv 源码包里的未使用文件。C7（逐块采样循环）的实际支撑在正文 §3.2 的 Alg. 2「Block Diffusion Sampling」；C8（揭开不遮回、NFE 上界 $L$）的实际支撑在 §6.2 与附录 D。｜引文依据：`grep -n 'inference_alg' iclr2025_conference.tex` 无命中；§6.2「\algo{} adopts an efficient sampler from masked diffusion, where the number of generation steps (NFEs) is upper-bounded by $L$ since tokens are never remasked」；§3.2 Alg. 2 逐块 Sample 循环与 KV 并入。｜修复要求：「附录 SAR Inference 算法」改为「§3.2 Alg. 2、§6.2、附录 D（Experimental Details → Inference）」；first-hitting 相关论断（index.html:1110）的定位也落在附录 D 而非「SAR Inference 算法」。

- [重要·可读性] index.html:999（BD3LM 首次出现）、:1106（SSD-LM 首次出现）：两个专名首次使用均未解释。全文以「块扩散」泛称行文，读者在 999 行突然遇到「BD3LM 的做法是……」、在 1106 行遇到「BD3LM 与 SSD-LM 的对比中……」，无法得知二者与 Arriola 论文、与本页泛称的关系（overview.html:48 有「代表工作是 Arriola 等人的 BD3LM」，index.html 无对应交代）。SSD-LM 未说明它是另一种块扩散实现，N2 句子因此难以独立读懂。｜引文依据：不适用（overview.html:48 对照）；Arriola §6.2「We also compare to SSD-LM, an alternative block diffusion formulation. Unlike our discrete diffusion framework, SSD-LM uses Gaussian diffusion」。｜修复要求：BD3LM 首次出现处加最小解释（如「BD3LM（Arriola 等论文中的块扩散语言模型，即本页范式的代表实现）」）；SSD-LM 首次出现处加最小解释（如「SSD-LM（另一种块扩散实现：块间自回归、块内用高斯扩散）」）。

### 轻微

- [轻微·格式] index.html:1068：引用编号「[C8 修正]」不是规范编号（style-guide §6 只允许 [Cx] 及组合），「修正」为第 1 轮修复过程的残留标记，来源章节中不存在「C8 修正」条目。｜引文依据：不适用。｜修复要求：改为 <sup>[C8]</sup>；全文 grep 确认无其他非规范编号。

- [轻微·格式] index.html:1099-1101（第 4 章状态演进表）：表格单元格内 7 处裸「[MASK]」与全文的 $[\mathrm{MASK}]$ 写法不一致（style-guide §11 明确覆盖表格单元格，且要求同一对象写法一致）。｜引文依据：index.html:815、:1066、:1119 均用 $[\mathrm{MASK}]$。｜修复要求：表格内 [MASK] 统一改为 $[\mathrm{MASK}]$。

- [轻微·格式] index.html:842、:850、:1022、:1093、:1135：正文以「第 4、5 章」「第 2 章」「第 1 章」「第 4 章」等编号引用其他章节，style-guide §1 要求「正文引用其他章节时使用章节标题」；页面其余位置（核心问题解答、index.html:1126）均已用「『N. 标题』」格式，写法不统一。｜引文依据：不适用。｜修复要求：上述 5 处改为章节标题引用（如「见『4. 块内去噪——从全 mask 到整块 token』」）；1106 行「（下一章）」可保留或一并改为标题。

- [轻微·图示] index.html:884（SVG 列组标签）：「块 1」标签 x=139 未对准块 1 列区域中心——块 1 列跨度为 [82, 234]（加粗边界框 x=82、宽 152），中心应为 x≈158；同图「块 2」标签 x=310 恰为其区域 [234, 386] 的中心。两标签定位规则不一致，「块 1」视觉上偏向列 2 上方（行标签 y=144、y=296 均已正确居中）。｜引文依据：不适用（坐标计算：列号 1–4 中心 101/139/177/215，块 1 区域中心 (82+234)/2=158；块 2 区域中心 (234+386)/2=310）。｜修复要求：x="139" 改为 x="158"，与「块 2」的居中规则一致。

- [轻微·来源] index.html:1200（N2 条目）：「N2：BD3LM 与 SSD-LM 对比中每块 $T=25$ 步达到可比 NFE」主语缺失且读作 BD3LM 的设置（$T=25$ 实为 SSD-LM 的每块步数）；条目未收录正文实际引用的数字（25.7 / 37.2 / 281.3 / NFE 1K vs ≥40K），「（LM1B/OWT 实验设置）」中 LM1B 与该对比无关（gen PPL 表为 OWT 训练模型）。｜引文依据：Arriola §6.2 Table 3 与附录 D「SSD-LM … undergoes $BT$ generation steps」。｜修复要求：改写为完整可核对条目，如「N2（gen PPL 对比，Arriola §6.2 Table 3，OWT 训练）：BD3LM $L'=4$ 在 $L=1024$ 达 gen PPL 25.7（NFE 1K，上界 $L$）；SSD-LM $T=1$K/块（$\geq$40K NFE）37.2，$T=25$（NFE 可比）退化到 281.3」。

- [轻微·可读性] index.html:1203（构造示例小节）：「表中置信度高低亦为设定值」引用了表中不存在的量——状态表演示已改为「按 noise schedule 揭开位置 1（示意，非按置信度）」（index.html:1100），表中不再出现置信度，此句是修复残留。｜引文依据：不适用。｜修复要求：删除该句。

- [轻微·技术] index.html:999（第 2 章补充块）：「BD3LM 的做法是把所有块拼成一条序列、用块间禁止注意的稀疏掩码」表述不完整、按字面读与第 2 章「后面的块看前面的块」规则矛盾。向量化训练的拼接是 $\x_{\text{noisy}}\oplus\x$（带噪块+干净全序列），掩码为「带噪块之间互不可见（块对角），但每个带噪块可见之前块的干净副本」。｜引文依据：Arriola 附录 C「noisy tokens attend to other noisy tokens in their block and to all clean tokens in preceding blocks」。｜修复要求：补全为「带噪块之间互不可见、但每个带噪块可见之前块的干净副本的稀疏掩码」或同等准确表述。

- [轻微·技术] index.html:1008（第 2 章本章问题 1 解答）：「掩码呈现三个区域：两个 $4\times4$ 的块内全可见区、左下 $4\times8$ 的跨块可见区、右上 $4\times4$ 的全遮挡区」——随后列出的是 4 个区域（2+1+1），「三个区域」计数错误。｜引文依据：不适用（figcaption index.html:992 列出同样四个区域）。｜修复要求：改为「四个区域」或「三类、共四个区域」。

- [轻微·技术] index.html:1186（第 5 章本章问题 2 解答）：「块大小取 8–16[N1]」与 N1 条目及正文不一致——DFlash 实际部署块大小为 16（Qwen3）/ 10（LLaMA-3.1），8 仅出现在消融（index.html:1135「消融中另含 8」）。｜引文依据：DFlash §5 Implementation「use a block size of 16 (10 for LLaMA 3.1)」；消融 §5.5 使用 8 与 16。｜修复要求：改为「块大小 16（Qwen3）/10（LLaMA-3.1），消融含 8」。

- [轻微·可读性] index.html:834（表 1 全并行掩码扩散列）：「去噪轮数（与 $L$ 无关的固定长度）」中「固定长度」指向序列长度而非轮数，与本行「串行前向次数」语义不接。｜引文依据：不适用。｜修复要求：改为「去噪轮数（固定值，与 $L$ 无关）」或同等清晰表述。

- [轻微·可读性] index.html:767（核心问题 1 解答）：「串行步从 $L$ 次降到 $B=L/L'$ 次，每步内部块内 $L'$ 个位置并行处理」缺「一轮或多轮」限定，与第 1 章本章问题 2 解答（index.html:865「（每步内部一次或多轮并行去噪）」）不一致，单独读会得出「总串行前向 $=B$」的结论。｜引文依据：index.html:865 对照；Arriola §6.2 NFE 上界 $L$。｜修复要求：补「（每步内部一轮或多轮）」限定语，与 865 行一致。

## 结论

- 统计：阻断 0 / 重要 8 / 轻微 11
- 处置：修复（无需返回规划）。

补充说明：

1. 第 1 轮修复总体到位的部分已验证通过：F1–F4 公式编号（Eq.(1)/(4)/(6)/(7)）、$\mathbf{K}/\mathbf{V}$ 粗体一致性、SVG 8×8 网格结构与行标签、块大小 16/10 及「消融中另含 8」、N2 主文（index.html:1106）、构造示例三处标注、来源说明六小节齐全、两级问题块每题均有解答折叠块、validate.py 通过、两页互链与概念链接有效。
2. 本轮 8 条重要问题中，E2/E3/E4/E5 是第 1 轮修复（置信度→first-hitting、N2 主语）未同步到的次要位置（核心问题解答、本章问题解答、伪代码简化条件、构造示例小节、overview），E1 是摘要层（dojo:summary/description/overview）沿用单轮理想化表述，E6/E7/E8 为第 1 轮未发现的来源定位与术语首现问题。修复面均局限于问题位置及直接受影响引用，不涉及范围或大纲调整。
3. 修复完成后需重跑 `.dojo/scripts/validate.py`，并在两页内复查「置信度」「B=L/L'」「第 N 章」三个关键词的残留；随后进行第 3 轮独立审查。
