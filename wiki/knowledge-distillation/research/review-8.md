<!-- review-meta
round: 8
page: wiki/knowledge-distillation/index.html
reviewed_content_sha256: 46292ea0465d2d9f
-->
# 知识蒸馏（Knowledge Distillation）审查记录（第 8 轮）

- 页面版本：7af4eb0e39baac519104eb6af761f71279b6807e（index.html 工作树哈希）
- 审查时间：2026-09-14 17:39
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：核心问题；1. 硬标签丢掉了什么——非目标类概率携带的暗知识（含本章问题）；2. 温度 softmax——把分布软化（含本章问题）；3. 手算温度 softmax——3 类 logits 在 $T=1$ 与 $T=5$ 下的分布（含 Python 验证代码折叠块与本章问题）；4. KD 总损失——软项 + 硬项 + $T^2$ 缩放（含内联 SVG 训练流程图、4.1 为什么必须乘 $T^2$ 折叠块、本章问题）；5. 边界——KD 解决什么、不解决什么、与 MOPD 的关系（含 5.1 适用条件、5.2 与 K3 MOPD 的关系、本章问题）；来源与范围说明

## 核对来源与版本

- HVD15：arXiv:1503.02531。经 arXiv abs 页确认该文**只有 v1**（2015-03-09 提交，无 v2），本次核对所用即 v1 PDF（`pdftotext` 提取全文）。以下所有引文依据均取自该 v1 版，**不存在 v1/v2 编号差异问题**。
- K3 报告：arXiv:2607.24653 §4.1.3「Multi-Teacher On-Policy Distillation」（WebFetch 核对，确认九专家、per-token 对数概率比稠密奖励、接入 RL 框架）。
- 被引概念页 wiki/mopd/index.html、wiki/opd/index.html 均真实存在（`../opd/`、`../../wiki/mopd/` 两条链接均可达）；页面无「（待生成）」占位，无指向不存在文件的路径。
- 其余已逐条回源核对且**一致**的关键项（不列入问题）：MNIST 教师 67 / 硬标签学生 146 / 蒸馏学生 74、$T=20$、30 单元学生 $T\approx2.5$–$4$（v1 §3）；「无数字 3」1010/996、未校正 206 错/133 个是 3、bias +3.5（v1 §3）；语音 58.9/61.1/60.8、WER 10.9/10.7/10.7、8 层×2560 单元、约 85M 参数、约 2000 小时（v1 §4, Table 1）；「More than 80% of the improvement…」（v1 §4.1）；$T^2$ 引文与 Eq.1/2/3/4（v1 §2、§2.1）；「dark knowledge」在 v1 全文出现 0 次（页面声明「HVD15 全文无该词」成立）；手动算式 $(2,1,0)$ 在 $T=1$/$T=5$ 的 0.665/0.245/0.090 与 0.402/0.329/0.269 复算无误；Python 代码实际执行，输出与页面「预期输出」逐行一致；`.dojo/scripts/validate.py` 返回 `validation ok`；KaTeX/Prism/折叠块/目录锚点/本地资源在无头 Chrome 渲染下均正常；无 元话语/会话指代（无「我们」「你」），`本文` 自称符合 style-guide §12。

## 问题

- [重要·技术] §4「KD 总损失」权重 $\alpha$ 条目：把 HVD15 §4.1 的「relative weight of 0.5 on the cross-entropy for the hard targets」直接等价为「按本文 $\alpha$ 为软损失权重的约定即 $\alpha = 0.5$」，该映射在页面自己的约定下不成立，且与同段紧邻论断自相矛盾｜引文依据：HVD15 v1 §2「We found that the best results were generally obtained by using a condiderably lower weight on the second objective function.」；§4.1「For the distillation we tried temperatures of [1, 2, 5, 10] and used a relative weight of 0.5 on the cross-entropy for the hard targets」。原文给出的是硬项相对软项的**比值**（soft:hard = 1:0.5），不是归一化后各占一半的绝对权重；而页面 L = $\alpha\cdot T^2\mathrm{KL}+(1-\alpha)\mathrm{CE}$ 中 $\alpha$ 为软项权重，取 $\alpha = 0.5$ 即软、硬权重相等，恰好与紧邻的「硬标签交叉熵应取明显更低的权重」冲突。读者据此复现会得到 soft:hard = 1:1 而非来源的 1:0.5｜修复要求：删除「（按本文 $\alpha$ 为软损失权重的约定即 $\alpha = 0.5$）」这一映射，或改写为与来源一致的比值表述（§4.1 对硬项取相对权重 0.5，即硬项权重为软项的一半），使该处数值与同段的「明显更低」不再矛盾；修改后重新核对 v1 §2 与 §4.1 原文｜修复：｜复验：

- [轻微·技术] §4 权重条目 与 §来源与范围说明 N5 中的「原文」引文与来源逐字不符｜引文依据：HVD15 v1 §2 原句为「…obtained by using a condiderably lower weight on the second objective function.」（源文含拼写 `condiderably`），页面两处均写作「a considerably lower weight on the second objective function」并冠以「原文：」。属静默改正源文拼写｜修复要求：引文与来源逐字一致——保留 `condiderably` 原拼写，或写作 `considerably` 并加 `[sic]` 说明；两处（§4 与 N5）同步处理｜修复：｜复验：

- [轻微·格式] §4 训练流程图（内联 SVG）中 4 个节点的两行标签未按设计换行｜引文依据：不适用（像素实测：以无头 Chrome 渲染 760px 宽页面，截取 #kd-loss 区域实测节点几何）。页面自定义 CSS `.flow-svg foreignObject div { display: flex; align-items: center; justify-content: center; … }` 中的 `display:flex` 使节点内 `<br>` 不产生换行，`$p_T^{\text{teacher}}$<br>温度 $T$ 下的软分布`、`$p_T^{\text{student}}$<br>温度 $T$ 下的软分布`、`$p_1^{\text{student}}$<br>$T=1$ 标准 softmax`、`硬损失 $\mathrm{CE}(y,\,p_1^{\text{student}})$<br>$y$ 为真实标签` 四者均渲染为单行；其中硬损失节点实测显示为「硬损失CE(y, p_1^{student})y为真实标签」，公式与「y 为真实标签」之间无任何分隔，易被误读为公式的一部分｜修复要求：使 foreignObject 内层容器的内容按块级排布（改用 `display:block`，或 `flex-direction: column`，或把每个 `<br>` 替换为独立块元素），让两行标签按设计换行；改后重新渲染确认标签仍居于节点内、不压线｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复
