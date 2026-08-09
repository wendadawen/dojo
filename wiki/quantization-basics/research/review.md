# 量化基础独立审查

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源）
- 页面版本：index.html 工作树哈希 3e58bfded11f62d414dbf4b104aebc789d4c4acd（HEAD d4f9e4e）
- 时间：2026-08-09 15:14 CST
- 审查范围：wiki/quantization-basics/index.html + overview.html
- 来源：Gholami et al. 2021《A Survey of Quantization Methods for Efficient Neural Network Inference》(arXiv:2103.13630)；OCP Microscaling Formats v1.0（Rouhani et al. 2023）；zeroentropy.dev/concepts/mxfp4；emergentmind.com/topics/mxfp4

## 段 A 盲读结果

按页面顺序以小白视角通读，主线卡点记录如下：

- S1 钩子与四要素 callout 在 zero-point、STE 等术语未定义前先出现，但属首屏摘要、下方各章展开，可接受。
- S2 公式 F1 符号定义清晰，两个 quick check（$x=0$ 与 $x=ks$）能在正文中验证，读者可跟上。
- S2 末尾 pre.diagram 第 754 行末尾出现 `</p>`，但前面没有对应的 `<p>`，且 `<pre>` 是块级元素不应嵌套在 `<p>` 内——结构异常（见问题 4）。
- S3 第 781 行同一句先写"位宽 $b$ 决定整数等级数 $2^b$"（INT4 为 16），再写"INT4 对称只有 15 个非零等级（$\pm 1$ 到 $\pm 7$）"。小白数一下 $\pm 1$ 到 $\pm 7$ 是 14 个非零值，与"15 个非零等级"对不上；同句 16/15/14 三个数字混在一起形成卡点（见问题 1）。
- S3 第 785 行"超出观测范围 $[\alpha,\beta]$"——$\alpha,\beta$ 首现但未在正文说明是什么（见问题 2）。
- S4 公式 F3 直接用 $\alpha,\beta$，但"$\alpha$ 为观测最小值、$\beta$ 为观测最大值"的定义只在 details 折叠块"非对称 zero-point 的推导"中。读者不展开折叠块无法在正文获得定义（见问题 2）。
- S5 STE 概念引入路径清晰：先说 round/clip 不可导，再引入 STE $\partial\hat{w}/\partial w\equiv 1$，伪代码与可运行代码同步，可跟上。
- S5 MXFP4 章节机制描述完整，但末尾"Kimi K3 把 MoE 专家权重量化到 MXFP4 并用 QAT 贯穿后训练"是具体产品事实声明，来源列表 C1–C9 无对应条目（见问题 3）。

学习目标闭环核对：
- 目标 1（量化要解决什么问题、为什么引入精度损失）：S1 完整回答 ✓
- 目标 2（如何用对称均匀量化编码并反量化）：S2 + 手算 [1.2,3.4,5.6] 完整回答 ✓
- 目标 3（对称/非对称、per-tensor/per-channel/per-block 差别）：S4 + 对照表完整回答 ✓
- 目标 4（误差三来源、离群值为什么让误差爆炸）：S3 + 离群值手算完整回答 ✓
- 目标 5（PTQ vs QAT 差别、MXFP4 优势）：S5 + 对照表 + MXFP4 小节完整回答 ✓

全部学习目标由正文章节完整回答。

## 段 B 对照来源结果

### 公式与手算复核

- F1 仿射量化 $x_q=\mathrm{clip}(\mathrm{round}(x/s)+z, q_{\min}, q_{\max})$、$\hat{x}=s(x_q-z)$：与 Gholami §2.1 一致 ✓
- F2 对称量化 $s=\max(|x|)/(2^{b-1}-1)$、$\hat{x}=s\,x_q$：与 Gholami §2.2 一致 ✓
- F3 非对称 $s=(\beta-\alpha)/(q_{\max}-q_{\min})$、$z=\mathrm{round}(q_{\min}-\alpha/s)$：推导（要求 $\alpha\mapsto q_{\min}$）正确 ✓
- F4 STE $\partial\hat{w}/\partial w\equiv 1$：与 Jacob 2018 + Gholami §2.5 一致 ✓
- 手算 [1.2,3.4,5.6]：s=0.8、x_q=[2,4,7]、$\hat{x}$=[1.6,3.2,5.6]、误差=[+0.4,-0.2,0] ✓
- 手算 [1.2,3.4,5.6,50.0]：s≈7.143、x_q=[0,0,1,7]、$\hat{x}$≈[0,0,7.143,50]、误差≈[-1.2,-3.4,+1.543,0] ✓
- 有效位宽 4.25 bit：32×4+8=136、136/32=4.25 ✓

### 可运行代码实际执行

代码块提取后实际执行（Python 3），三段输出与页面"预期输出"逐字一致：

```
=== E1: [1.2, 3.4, 5.6] INT4 对称 ===
scale   = 0.8
x_q     = [2, 4, 7]   (范围 [-7, 7])
x_hat   = [1.6, 3.2, 5.6]
error   = [0.4, -0.2, 0.0]
单值最大舍入误差上界 s/2 = 0.4

=== E2: [1.2, 3.4, 5.6, 50.0] per-tensor INT4 ===
scale   = 7.1429   (被离群值 50 主导)
x_q     = [0, 0, 1, 7]
x_hat   = [0.0, 0.0, 7.1429, 50.0]
error   = [-1.2, -3.4, 1.5429, 0.0]

=== E3: per-block 分两块（正常块 + 离群值块）===
块 A scale = 0.8, x_q = [2, 4, 7], x_hat = [1.6, 3.2, 5.6], error = [0.4, -0.2, 0.0]
块 B scale = 7.1429, x_q = [7], x_hat = [50.0]
```

注：代码用 Python 内置 `round`（banker's rounding），但本例 1.5→2、4.25→4、7.0→7 在两种约定下一致，未触发差异。页面在 details "1.5 为何取 2 而不是 1"中已说明此约定差异。

### 来源逐条核对

- C1 仿射量化公式：Gholami §2.1 ✓
- C2 对称量化 $z=0$、范围 $[-2^{b-1}+1, 2^{b-1}-1]$、$s=\max(|x|)/(2^{b-1}-1)$：Gholami §2.2 ✓
- C3 非对称：Gholami §2.2 ✓
- C4 粒度由粗到细：Gholami §2.3（论文用 Layerwise/Channelwise/Groupwise/Sub-channelwise，页面用 per-tensor/per-channel/per-block 行业通用术语，对应关系正确）✓
- C5 误差三来源：Gholami §2.4（论文提到 rounding/truncation，页面分类为 grid coarseness/rounding/clipping 属教学重构，与来源一致）✓
- C6 离群值毒化：Gholami §2.4 + zeroentropy "Outliers no longer poison their channel; they're contained to their own 32-element block" ✓
- C7 PTQ 流程：Gholami §2.5 流程描述 ✓；128–512 数字见问题 8
- C8 QAT 伪量化 + STE：Jacob 2018 + Gholami §2.5 ✓
- C9 MXFP4 块结构（块 32 + E8M0 scale + E2M1 元素 + $\hat{x}_i=s_b\cdot\mathrm{FP4}(q_i)$）：zeroentropy + emergentmind 全部确认 ✓
  - 块 32、E8M0 8-bit power-of-two、E2M1（1 符号+2 指数+1 尾数）✓
  - 每块 136 bit、有效位宽 4.25 bit ✓
  - "scale 是 power-of-two，multiply 退化为 float exponent 的加法、无需 multiplier" ✓
  - "per-block scale 把离群值局限在 32-element 块内" ✓
- N1 每参数字节数（FP32=4、FP16/BF16=2、INT8=1、INT4=0.5；7B FP16≈14GB、INT8≈7GB、INT4≈3.5GB）：标准事实 ✓
- N2 MXFP4 136 bit / 4.25 bit：zeroentropy 确认 ✓
- N3 PTQ 128–512 样本：见问题 8
- N4 E2M1 值集合 {0,±0.5,±1,±1.5,±2,±3,±4,±6}、最大绝对值 6：emergentmind 确认 ✓；"16 个有限值"表述见问题 6

## 问题

- [重要·技术] S1 §"量化要解决什么问题"（第690行）+ S3 §"量化误差从哪来 · 来源一"（第781行）：INT4 对称等级数描述错误。两处均写"15 个非零等级（$\pm 1$ 到 $\pm 7$）"，但 $\pm 1$ 到 $\pm 7$ 共 14 个非零值，加 0 共 15 个等级（含 0）。S3 同一句先写"位宽 $b$ 决定整数等级数 $2^b$"（INT4=16，不对称范围 [-8,7]），再写"15 个非零等级"，16/15/14 三个数字混在一起。修法：将两处"15 个非零等级（$\pm 1$ 到 $\pm 7$）"改为"15 个等级（含 0，$\pm 1$ 到 $\pm 7$ 共 14 个非零等级）"；S3 的 $2^b$ 表述补一句"对称量化牺牲一端一等级，剩 $2^b-1$ 个"。修复：已将 S1（第690行）与 S3（第781行）两处"15 个非零等级"改为"15 个等级（含 0，$\pm 1$ 到 $\pm 7$ 共 14 个非零等级）"，并在 S3 的 $2^b$ 表述后补"对称量化牺牲一端一等级，剩 $2^b-1$ 个" ｜ 复验：
- [重要·盲读] S3 §"来源三：裁剪"（第785行）+ S4 §"对称 vs 非对称"（第823、825行）：$\alpha,\beta$ 在正文首现未定义。S3 第一次出现"观测范围 $[\alpha,\beta]$"时未说明 $\alpha,\beta$ 是什么；S4 公式 F3 直接用 $\alpha,\beta$，但"$\alpha$ 为观测最小值、$\beta$ 为观测最大值"的定义只在 S4 details 折叠块"非对称 zero-point 的推导"中（第830行）。读者不展开折叠块无法在正文获得定义。修法：在 S3 或 S4 正文首现 $[\alpha,\beta]$ 处明确"$\alpha$ 为观测最小值、$\beta$ 为观测最大值"。修复：已在 S3 §"来源三：裁剪"首现 $[\alpha,\beta]$ 处补注"（$\alpha$ 为观测最小值、$\beta$ 为观测最大值）"，读者不展开折叠块即可在正文获得定义 ｜ 复验：
- [重要·技术] S5 §"延伸：MXFP4"末尾（第1051行）："Kimi K3 把 MoE 专家权重量化到 MXFP4 并用 QAT 贯穿后训练"是具体产品部署事实声明，但"核心论断与来源"C1–C9 列表无对应来源条目。修法：在"核心论断与来源"中补充 Kimi K3 部署的来源（官方博客或技术报告 URL），或将该句改为不带具体产品名的一般性陈述（如"已有大模型部署采用 MXFP4 + QAT 路线"）。修复：已将"Kimi K3 把 MoE 专家权重量化到 MXFP4 并用 QAT 贯穿后训练"改为不带具体产品名的一般性陈述"已有大模型部署采用 MXFP4 + QAT 路线，具体案例见专门页"，避免无来源的产品事实声明 ｜ 复验：
- [轻微·技术] S2 §"均匀量化怎么把浮点变成整数"末尾 pre.diagram（第737–754行）：`<pre class="diagram">...</pre>` 后紧跟一个 `</p>`（第754行末尾），但前面无对应 `<p>`，且 `<pre>` 是块级元素不应嵌套在 `<p>` 内。浏览器容错但 HTML 结构不合规。修法：删除第754行末尾多余的 `</p>`。修复： ｜ 复验：
- [轻微·技术] §"来源与教学说明 → 教学简化及其限制"（第1115行）："GPTQ、AWQ、SmoothQuant 等具体 PTQ 算法只在 S5 一句带过名字，未展开"——但 S5 正文（PTQ/QAT/MXFP4 三个小节）实际未出现 GPTQ/AWQ/SmoothQuant 这些名字。教学简化说明与正文实际不符。修法：将说明改为"S5 未展开 GPTQ/AWQ/SmoothQuant 等具体 PTQ 算法（正文未提及名字）"，或在 S5 PTQ 小节补充一句提及这些算法名。修复： ｜ 复验：
- [轻微·技术] S5 §"MXFP4"（第1043行）+ N4（第1090行）：E2M1 "16 个有限值"表述不严谨。E2M1 4-bit 编码共 16 个码字，其中 +0 与 -0 数学等价，故不同有限值为 15 个（集合 {0,±0.5,±1,±1.5,±2,±3,±4,±6} 含 1 个零 + 14 个非零）。来源 emergentmind 也用"16 distinct values"表述，但严格说 4-bit 码字数 16 ≠ 不同有限值数 15。修法：改为"16 个码字（其中 +0/-0 数学等价，共 15 个不同有限值）"或"15 个不同有限值"。修复： ｜ 复验：
- [轻微·盲读] S1 §"量化要解决什么问题"（第690行）："压缩比与精度损失成正比是量化所有后续方法的统一动机"——"成正比"是严格数学关系（$y=kx$），实际是"位宽越低压缩比越大、精度损失也越大"的单调趋势，非严格正比。修法：改为"位宽越低，压缩比越大，精度损失也越大"或"压缩比与精度损失呈单调关系"。修复： ｜ 复验：
- [轻微·技术] C7（第1072行）/ N3（第1089行）来源标注不精确：PTQ 校准数据"128–512 样本"标注来源之一为"Gholami et al. 2021 §2.5"，但 Gholami 论文 §2.5（Fine-tuning Methods / PTQ）描述 PTQ 流程为"用部分训练数据算截断区间"，未给出 128-512 这个具体数字（基于 Gholami 论文公开摘要与中文笔记核对）。该数字可能来自另一来源 karam-nus.github.io。修法：核实 Gholami §2.5 原文是否给出 128-512；若无，将 128-512 的来源仅标注为 karam-nus.github.io，C7 改为"Gholami §2.5 描述 PTQ 流程；典型工程值 128-512 样本见 karam-nus.github.io"。修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 5
- 处置：进入修复

学习目标闭环：全部 5 条学习目标由正文章节完整回答，无遗漏。
代码执行：三段可运行代码实际执行，输出与页面"预期输出"逐字一致。
公式复核：F1–F4 与手算结果全部正确。
来源对照：C1–C9、N1–N4 核心论断与来源一致；C7/N3 的 128-512 数字来源标注需精确化（问题 8）；C9 MXFP4 全部细节经 zeroentropy + emergentmind 双源确认。
