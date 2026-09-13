<!-- review-meta
round: 3
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: 0b13fcba79cb5768
-->
# 知识蒸馏（Knowledge Distillation）审查记录（第 3 轮）

- 页面版本：3162e32753fa41e0f2d1d462d27efafc1c057c2d（index.html 工作树哈希）
- 审查时间：2026-09-13 19:05
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 输入：仅使用 index.html、overview.html、外部来源（arXiv:1503.02531 ar5iv 全文）、guides/concept/check.md、guides/concept/style-guide.md
- 已完整阅读章节（按顺序）：核心问题 → 1. 硬标签丢掉了什么 → 2. 温度 softmax → 3. 手算温度 softmax → 4. KD 总损失（含 4.1 为什么必须乘 $T^2$，含折叠推导与代码块）→ 5. 边界（含 5.1 适用条件、5.2 与 K3 MOPD 的关系）→ 来源与范围说明（含全部折叠块与图注）

## 已核对来源（引文依据）

- MNIST 数字：ar5iv §3 原文「a larger model with two hidden layers of 1200 units ... 67 test errors」「800 units ... 146 errors」「74 errors ... softening at T=20」——与页面 67 / 146 / 74 / $T=20$ 一致（第 477–479、582 行）。
- 小容量学生温度：ar5iv §3「when this was radically reduced to 30 units per layer, temperatures in the range 2.5 to 4 worked significantly better」——与页面「每层仅 30 单元 ... $T \approx 2.5$–$4$」一致（第 224、509、585 行）。
- 「无数字 3」实验：ar5iv §3「the distilled model only makes 206 test errors of which 133 are on the 1010 threes in the test set ... If this bias is increased by 3.5 ... the distilled model makes 109 errors of which 14 are on 3s」——与页面 206/133、bias +3.5、1010 个 3、996/1010 正确（1010−14）一致（第 163、565、583 行）。
- $T^2$ 缩放：ar5iv §2「Since the magnitudes of the gradients produced by the soft targets scale as 1/T² it is important to multiply them by T² when using both hard and soft targets.」——与页面一致（第 563 行）。
- 高温零均值梯度：ar5iv §2 Eq.4「∂C/∂z_i ≈ (1/NT²)(z_i − v_i)」；等价形式「distillation is equivalent to minimizing 1/2(z_i−v_i)², provided the logits are zero-meaned separately for each transfer case」——与页面折叠推导、$N=K$、logit matching 的成立条件一致（第 432、434–435 行）。$N$ 为类别数（零均值下 $\sum_j\exp(z_j/T)\to N$），页面标注正确。
- $\alpha$ 取值倾向：ar5iv §2「the best results were generally obtained by using a considerably lower weight on the second objective function」——与页面「硬项权重远小于软项，$\alpha$ 接近 1」一致（第 373、586 行）。
- 语音实验：ar5iv §4「58.9% / 10.9%（baseline）」「61.1% / 10.7%（10xEnsemble）」「60.8% / 10.7%（distilled）」「8 hidden layers each containing 2560 rectified linear units ... about 85M parameters」「~2000 hours」——与页面表格与 N3 一致（第 489–498、584 行）。
- 代码：页面 Python 手算脚本实测输出逐行与页面「预期输出」一致（$T=1$ → 0.665241/0.244728/0.090031；$T=5$ → 0.40176/0.328933/0.269307；$T=0.5$ → 0.866813/0.11731/0.015876；$T=2$ → 0.50648/0.307196/0.186324；$T=10$ → 0.367165/0.332225/0.30061；$T=100$ → 0.336672/0.333322/0.330006）。
- 手算与表内差值：7.389/2.718/1.000 与 1.492/1.221/1.000 归一化结果、下降 0.263 / 上升 0.084 / 上升 0.179，全部复算一致（第 264–290 行）。
- 链接：../../wiki/mopd/index.html、../opd/index.html 均真实存在；页面内无 research/ 路径引用、无「（待生成）」占位；`.dojo/scripts/validate.py` 返回 success。
- 公式渲染/结构图形式：数学均为 LaTeX 由 KaTeX 渲染，结构图为内联 SVG + `<foreignObject>`，`<text>` 内无 ASCII 近似（validate 通过）；未发现元话语禁用词（"需要注意的是""下面来看"等）与第一人称「我们」。

## 问题

- [阻断·技术] 来源章节 C1（第 558 行）及正文第 105、157、160 行：把 "dark knowledge" 一词定位到 HVD15 §1，但论文全文不含该词；页面三处表述"Hinton 称之为暗知识（dark knowledge）""'dark knowledge' 一词出自论文 §1"属把非论文用语包装成来源原文。｜引文依据：对 arXiv:1503.02531 的 ar5iv 全文两次抓取均返回「the phrase "dark knowledge" does not appear anywhere on this page」；§1、§2 讨论软标签携带的信息但未使用该措辞；WebSearch 显示该词源于 Hinton 的报告/演讲，社区将其归于 Hinton，并非论文正文用词。｜修复要求：删除"C1：'dark knowledge' 一词出自论文 §1"这一具体定位；C1 改为"概念见 HVD15 §1–§2；'暗知识'为业界/Hinton 报告中的称谓，非论文原文用词"，并同步修改第 105、157、160 行，使"暗知识"不再被表述为论文用语。｜修复：｜复验：
- [重要·技术] §4 训练流程图（第 398、409、415 行）：图中硬损失节点写作 $\mathrm{CE}(y,\,p_1^{\text{student}})$、图注写"硬损失用学生的 $T=1$ 分布与真实标签比较"，但第 409 行连边把 $p_T^{\text{student}}$（温度 $T$ 下的软分布）直接连到硬损失节点，图中不存在 $T=1$ 学生分布节点——图示与损失定义、图注互相矛盾。｜引文依据：第 398 行节点文字「硬损失 $\mathrm{CE}(y,\,p_1^{\text{student}})$ $T=1$ 标准 softmax，$y$ 为真实标签」；第 409 行 `<polyline points="500,232 500,296">` 起点 500,232 为 $p_T^{\text{student}}$ 节点（410,176,高 56）下沿、终点 500,296 为硬损失节点上沿；第 415 行图注「硬损失用学生的 $T=1$ 分布与真实标签比较」。｜修复要求：新增 $p_1^{\text{student}}$ 节点（学生标准 softmax）并连到硬损失，或重画使硬损失输入显式标注 $T=1$，删除 $p_T^{\text{student}}$ 直连硬损失的连边，使图示与 $\mathcal{L}$ 定义一致。｜修复：｜复验：
- [轻微·格式] 第 159、439 行：使用两个 `callout-blue`，违反 blue 每篇最多 1 个的上限，且两处语义均非"开篇引入/范围说明"（159 行是概念说明、439 行是注意事项）。｜引文依据：guides/concept/style-guide.md §3「blue：开篇引入/范围说明（每篇最多 1 个）」；「yellow：注意事项/边界条件/易误解澄清」。｜修复要求：两处改 `callout-yellow`，blue 全篇保留不超过 1 处。｜修复：｜复验：
- [轻微·格式] 第 500 行：使用 `<div class="callout callout-red">`，违反 red 不作 callout 颜色的规定。｜引文依据：guides/concept/style-guide.md §3「red/green/gray 不作为 callout 颜色（red 留给 misconceptions，green 留给 learning-goals）」。｜修复要求：改为 `callout-yellow`，或将该段改为 `misconceptions` 块。｜修复：｜复验：
- [轻微·表述] 第 94 行：引言"部署一个 70B 参数的教师模型，推理一次要数百毫秒、显存上百 GB"给出无来源的具体数字，作为无条件陈述。｜引文依据：guides/concept/style-guide.md §7「只有来源可靠且有助于定位问题时才使用具体数字」；页面 F/N/C 来源列表未登记这三个数字。｜修复要求：删除或以"例如大型教师模型"等不含量值的表述替换，或明确标注为示意而非实测。｜修复：｜复验：
- [轻微·表述] 第 165 行："这里就回答了第一个问题——硬标签是 one-hot……"以问题编号形成元话语式指代，读起来像写作旁白。｜引文依据：guides/concept/check.md §2.2 第 12 项（元话语）。｜修复要求：改为直接陈述结论（如"硬标签是 one-hot，非目标类全零，把'错得离哪个类近'的信息全删了……"），删除"这里就回答了第一个问题"一句。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 4
- 处置：修复。数字核对全部通过（MNIST 67/146/74、无数字 3 的 206/133/996/1010、语音 58.9/61.1/60.8 与 10.9/10.7、$T=20$ 与 2.5–4、$T^2$ 与 $1/NT^2$ 梯度、$\alpha$ 倾向），手算与代码输出逐项一致，无 Unicode 未渲染公式、无失效链接。待关闭项为 1 条阻断（"dark knowledge" 来源定位）与 1 条重要（流程图硬损失连边），关闭后方可发布。