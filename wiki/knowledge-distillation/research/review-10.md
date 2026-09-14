<!-- review-meta
round: 10
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: ea31dace87cb555f
-->
# 知识蒸馏审查记录（第 10 轮）

- 页面版本：81db7f7d4e7e1855fc03072b6b0d789e9e3826c4（工作树 git hash-object）
- 审查时间：2026-09-14
- 审查者：独立子代理（编排者派发的独立审查者）
- 依据规范：guides/concept/check.md；guides/concept/style-guide.md
- 来源版本：Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531**v1** (2015-03-09)。本轮所有 C/F/N 论断均回到该版本的 PDF 正文逐条核对，下文引文即取自该版本。
- 已完整阅读章节：头部 meta 与 style；主要依据 blockquote；引言；核心问题（5 问）；1. 硬标签丢掉了什么——非目标类概率携带的暗知识；本章问题；2. 温度 softmax——把分布软化；本章问题；3. 手算温度 softmax——3 类 logits 在 T=1 与 T=5 下的分布；本章问题；4. KD 总损失——软项 + 硬项 + T² 缩放；4.1 为什么必须乘 T²（含折叠推导）；本章问题；5. 边界——KD 解决什么、不解决什么、与 MOPD 的关系；5.1 适用条件；5.2 与 K3 MOPD 的关系；本章问题；来源与范围说明（C/F/N/构造示例/类比边界/简化条件）；全部折叠块与 SVG 图注。

## 核对记录（无问题项，列出关键数值以证来源一致）

- C6 引文：论文 §2 原文 "Since the magnitudes of the gradients produced by the soft targets scale as 1/T 2 it is important to multiply them by T 2" —— 与页内引文一致。
- N5 引文：论文 §2 原文 "the best results were generally obtained by using a condiderably lower weight on the second objective function"（保留原论文拼写错误 "condiderably"）—— 一致；§4.1 "used a relative weight of 0.5 on the cross-entropy for the hard targets" 与页内"硬项权重为软项的一半"一致。
- N1/C7：§3 "This net achieved 67 test errors whereas a smaller net with two hidden layers of 800 rectified linear hidden units and no regularization achieved 146 errors ... at a temperature of 20, it achieved 74 test errors" —— 67/146/74、T=20 一致。
- N2/C8：§3 "only makes 206 test errors of which 133 are on the 1010 threes ... If this bias is increased by 3.5 ... makes 109 errors of which 14 are on 3s ... 98.6% of the test 3s correct" —— 1010−133=877（86.8%）、1010−14=996 一致。
- N3/C9：§4 Table 1 Baseline 58.9%/10.9%、10xEnsemble 61.1%/10.7%、Distilled Single model 60.8%/10.7%；"8 hidden layers each containing 2560 ... about 85M ... about 2000 hours"；"More than 80% of the improvement in frame classification accuracy ... is transferred" —— 全部一致（(60.8−58.9)/(61.1−58.9)=86.4%>80%）。
- N4：§3 "when this was radically reduced to 30 units per layer, temperatures in the range 2.5 to 4 worked significantly better" —— 一致。
- F1/F2：论文 Eq.(1) q_i = exp(z_i/T)/Σ_j exp(z_j/T) 与页内公式一致；F4/Eq.(4) ∂C/∂z_i ≈ (1/(NT²))(z_i−v_i) 与折叠块一致，T² 后 1/N(z_i−v_i) 可复算。
- 手算：T=1 → (0.665241,0.244728,0.090031)、T=5 → (0.40176,0.328933,0.269307)，与页内 3 位小数表及变化量 0.263/0.084/0.179 一致。
- 代码：实际执行页内 Python（kdcheck.py），输出逐行与"预期输出"完全一致（含 T=0.5/2/10/100 与 1/3=0.333333）。
- 暗知识：grep 论文全文无 "dark"，页内"HVD15 全文无该词"成立（注：页内"构造示例"已把示例数字标注为说明用）。
- 链接：../../index.html、overview.html、../../wiki/mopd/index.html、../opd/index.html 均存在；overview 与 index 互链；libs 资源存在；validate.py 返回 ok；SVG 无 `<text>` 数学近似，公式均在 foreignObject 内。

## 问题

- [轻微·表述] 第 1→2、2→3、3→4、4→5 章交界（index.html 行 196、256、365、491）：四段章节过渡套用同一模板「本章[回顾]。但[缺口]——下一章[预告]。」。行 196"本章已经说清……。但……——下一章引入温度 softmax"；行 256"本章给出……。但……——下一章手算一个 3 类的例子"；行 365"本章手算并代码验证了……。但……——……下一章组装完整损失"；行 491"本章给出……。下一章把损失放回真实实验"。check 规范 2.2.12 与 style-guide §8 要求"不使用固定句式"，content-examples B 节的衔接也正是逐段不同写法。｜引文依据：不适用｜修复要求：改写这四段衔接，去掉重复的「本章……但……——下一章……」套式，直接写前一节结论与下一节问题的实质关系，句式互不雷同。｜修复：｜复验：
- [轻微·技术] §1 构造示例表（行 154）与 §1 末尾过渡段（行 196）：同一个"一张 7 的图"场景，表给出教师非目标类概率 0.048（9）、0.018（8）、0.010（2），过渡段写"其他类的概率都在 0.001 量级"，两处量级相差近两个数量级，读者无法判断教师原始 softmax 到底多尖、也就难以判断温度为何必要。｜引文依据：HVD15 §1 原文 "a probability of 10−6 of being a 3 and 10−9 of being a 7"（论文示例量级为 1e-6/1e-9，而表内为 0.048）；页内行 196 自述"其他类的概率都在 0.001 量级"。｜修复要求：让表与行 196 对同一场景给出一致量级，或在行 196 明确说明 0.992/0.001 是比 §1 表更尖的另一情形、并说清二者关系。｜修复：｜复验：
- [轻微·格式] 行 628 用"本页"，行 98/387/549/576/586/599 用"本文"：同页自称混用"本页"与"本文"，违反 style-guide §12（自称使用"本页"或"本文"其一）。｜引文依据：不适用｜修复要求：全文自称统一为"本页"或"本文"中的一种。｜修复：｜复验：
- [轻微·格式] §1/§4.1（行 154、442—447）与 §4（行 371—388）：学生在温度 T 下的输出分布在 §1、§4.1 折叠块写作 $q$（教师写作 $p^t$），在 §4 的损失公式与符号表中写作 $p_T^{\text{student}}$（教师 $p_T^{\text{teacher}}$），同一量页内两种写法，违反 check 规范 2.2.9"同一变量全页写法一致"。｜引文依据：不适用｜修复要求：统一同一量的符号，或在 §4 符号表注明 $p_T^{\text{student}}$/$p_T^{\text{teacher}}$ 与 §1/$p^t$、$q$ 的对应关系。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：修复（4 项轻微）；无阻断、无重要问题，核心论断、公式、数字、引文编号与引文均已回到 arXiv:1503.02531v1 逐条核对一致，可发布。
