<!-- review-meta
round: 6
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: f87e988ffad1a9a2
-->
# 知识蒸馏（Knowledge Distillation）审查记录（第 6 轮）

- 页面版本：0f71b557477181de561805769418a899c7c38fd4（wiki/knowledge-distillation/index.html 工作树哈希）
- 审查时间：2026-09-13 21:11
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 1. 硬标签丢掉了什么——非目标类概率携带的暗知识（含本章问题）/ 2. 温度 softmax——把分布软化（含本章问题）/ 3. 手算温度 softmax——3 类 logits 在 $T=1$ 与 $T=5$ 下的分布（含代码折叠块与本章问题）/ 4. KD 总损失——软项 + 硬项 + $T^2$ 缩放及 4.1 为什么必须乘 $T^2$（含梯度推导折叠块与本章问题）/ 5. 边界——KD 解决什么、不解决什么、与 MOPD 的关系及 5.1、5.2（含本章问题）/ 来源与范围说明

## 来源核对说明

- 官方材料：arXiv:1503.02531 全文（HVD15）逐段核对；引用概念页 wiki/mopd/index.html 核对 MOPD 描述。
- 代码执行：第 297–317 行 Python 脚本本地实际运行，输出与页面「预期输出」（第 321–331 行）逐行一致（含 $T=0.5/2/10/100$ 与 $1/3$）。
- 核对通过的来源数字（片段）：§3「This net achieved 67 test errors ... 800 rectified linear hidden units and no regularization achieved 146 errors ... matching the soft targets ... at a temperature of 20, it achieved 74 test errors」；§3「the distilled model only makes 206 test errors of which 133 are on the 1010 threes ... If this bias is increased by 3.5 ... makes 109 errors of which 14 are on 3s ... gets 98.6% of the test 3s correct」；§4 Table 1「Baseline 58.9% / 10.9%; 10xEnsemble 61.1% / 10.7%; Distilled Single model 60.8% / 10.7%」；§2「Since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ ...」；§2 Eq.2「$\partial C/\partial z_i=(1/T)(q_i-p_i)$」、Eq.4「$\partial C/\partial z_i\approx(1/NT^2)(z_i-v_i)$」；§4.1「used a relative weight of 0.5 on the cross-entropy for the hard targets」。以上均与页面表述一致。

## 问题

- [轻微·格式] 第 7 行 `<meta name="dojo:summary">`：摘要里的损失式漏掉 $T^2$ 因子，与正文 §4/§4.1 及 overview 的公式不一致｜引文依据：summary 写「典型损失为 $\alpha\,\mathrm{KL}_{soft}+(1-\alpha)\,\mathrm{CE}_{hard}$」，而正文第 369 行与 overview 第 39 行均为 $\alpha T^2\mathrm{KL}+(1-\alpha)\mathrm{CE}$｜修复要求：把 summary 损失式补为 $\alpha T^2\mathrm{KL}_{soft}+(1-\alpha)\mathrm{CE}_{hard}$，或删去具体公式只保留文字描述｜修复：｜复验：
- [轻微·表述] 第 6 行 `<meta name="description">`：「训练小学生模型」连读为「小学生」，与标题「训练更小的学生」不一致，语义易被误读｜引文依据：不适用｜修复要求：改为「训练更小的学生模型」｜修复：｜复验：
- [轻微·格式] 第 216 行表格单元格：「最大 logit 类别概率 → 1」中的箭头以 Unicode 字符直接出现，未包在 `$...$` 中，与同一行「$T \to 0^+$」的写法不一致｜引文依据：不适用（依据 style-guide §11：数学符号一律由 KaTeX 渲染，同一符号保持同一写法）｜修复要求：改写为「$\to 1$」｜修复：｜复验：
- [轻微·表述] 第 440 行（4.1 推导折叠块开头）：把总量级被压低 $1/T^2$ 说成「再做一次链式法则」，与第 457 行「两重压低效应（链式法则的 $1/T$ + 分布差异被压缩）」及正文推导不符；第二个 $1/T$ 来自高温下 $q_i-p_i$ 收缩，不是第二次链式法则｜引文依据：HVD15 §2 Eq.2→Eq.3→Eq.4：Eq.2 只含一次链式因子 $1/T$，第二个 $1/T$ 来自 $q_i-p_i\approx(z_i-v_i)/(NT)$｜修复要求：将「再做一次链式法则后总量级被压低 $1/T^2$」改为「再由高温下分布差异被压缩，总量级被压低 $1/T^2$」｜修复：｜复验：
- [轻微·来源] 第 520 行：「ensemble 的增益几乎完全被吸收进一个可部署的单模型里」强于来源表述｜引文依据：HVD15 §4.1 原文「More than 80% of the improvement in frame classification accuracy achieved by using an ensemble of 10 models is transferred to the distilled model」（帧准确率 58.9%→61.1%，蒸馏到 60.8%，约 86%）｜修复要求：改为「ensemble 超过 80% 的帧准确率增益被吸收」或直接给出「>80%」的量化表述｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布（无阻断与重要问题；上述 5 项轻微问题建议本轮一并修复。核心论断、公式、数字、代码输出均与 HVD15 及 MOPD 页一致，问题块两级完备，公式全由 KaTeX 渲染，结构图为内联 SVG，validate.py 通过）