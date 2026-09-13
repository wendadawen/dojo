<!-- review-meta
round: 5
page: wiki/quantile-balancing/index.html
reviewed_content_sha256: 96affa46e6d6bcc3
-->
# Quantile Balancing 审查记录（第 5 轮）

- 页面版本：0a0d71a14313b02db60216bd7149aedbdcf6dbd3
- 审查时间：2026-09-13 20:23
- 审查者：编排者派发的独立审查者（独立子代理，未参与写作，也未参与前序轮次）
- 已完整阅读章节：核心问题（页面级问题块）、最容易误解、1. auxiliary-loss-free 路由——bias 做什么、不做什么（含本章问题）、2. DeepSeek-V3 的固定步长更新——为什么 896 专家时更难维持均衡（含本章问题）、3. QB 的核心机制——从一次前向推导下一个 bias（3.1–3.9、四步流水线图、代码折叠块、本章问题）、4. QB 为什么好——从平衡分配到对偶的视角（4.1–4.3、两个折叠块、本章问题）、5. 训练时的直方图估计与推理冻结（5.1–5.3、伪代码折叠块、本章问题）、来源与范围说明、结论段
- 来源核对：Kimi K3 Technical Report（k3_tech_report.pdf：§2.3.3、Appendix C、Appendix D、Table 1、Fig. 5 图注）、DeepSeek-V3 Technical Report §2.1.2（arXiv:2412.19437）、HuggingFace moonshotai/Kimi-K3 config.json。
- 机械核对：按页面代码块原样执行 Python 示例，输出与页面「预期输出」逐行一致（初始 loads [4,3,1,0] → [2,2,2,2]；改变 token T4(E1→E3)、T5(E1→E4)、T8(E2→E4)）；`python3 .dojo/scripts/validate.py wiki/quantile-balancing/index.html` 返回 "validation ok"。
- 表述维度：全文（含折叠块、图注、问题块答案）通读，未发现元话语式的「本页将…/下面来看…」、会话指代（我/我们/你）、调试与复现踩坑叙事或临场评价；「本文」「见下文」属全站既有写法（style-guide §12 允许自称使用「本页/本文」，全站另有 4 页用「见下文」做交叉引用），不计为问题。

## 问题

- [重要·技术] 来源与范围说明「论断与来源（C）」C2 与「公式与来源（F）」F2：把 K3 报告 §2.3.3 对 DeepSeek-V3 固定步长 sign 更新的引用编号写成 [27]。K3 报告该处引用的是 [30]；[27] 是另一篇论文（Griffin），按该编号回源会定位到完全无关的来源｜引文依据：K3 报告 §2.3.3 原文 "the original method updates b with the fixed-step rule $b_j^{(t+1)} = b_j^{(t)} + \gamma \text{sign}(\bar{\ell} - \ell_j)$ [30], for which $\gamma$ trades off slow adaptation against load oscillation"；参考文献 [30] = "DeepSeek-AI et al. DeepSeek-V3 Technical Report. 2024. arXiv: 2412.19437"；[27] = "Soham De et al. Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"｜修复要求：把 C2、F2 两条来源说明中的「K3 报告 §2.3.3 引用 [27]」改为「引用 [30]」；正文 `<sup>[C2, F2]</sup>` 的 C/F 编号本身无需改动｜修复：｜复验：

- [轻微·技术] 3.4 mean-centering 后的「因果性」callout（"一个 batch 绝不用自己推导的 bias 路由——否则会造成信息泄漏"）：来源只给出更新必须在下一步生效的因果性要求，「会造成信息泄漏」是页面自行添加的解释，未标注为推断，且来源未使用「信息泄漏」这一说法｜引文依据：K3 报告 §2.3.3 "For causality, the update takes effect only in the next step [30], i.e., a batch is never routed with a bias derived from itself."；Appendix C 亦仅重申 "preserves train–inference consistency"，无泄漏表述｜修复要求：删除「否则会造成信息泄漏」，或改写为与来源一致的因果性表述（如「该 batch 的路由只能由更早的 bias 决定」），不得保留未标注的来源外推断｜修复：｜复验：

- [轻微·格式] 第 3 章四步流水线块（index.html 第 249–261 行）与页内 `<style>`（第 29–62 行）：顺序流程使用页面自定义的 `.flow-steps / .flow-step / .flow-arrow` 三条 CSS 加页内样式表，未使用组件库既有的 `dg-flow / dg-node / dg-node-title / dg-node-note / dg-arrow`（定义在 libs/dojo-concept.css，全站 32 页在用）；guides/concept/content-examples.md A5 要求「顺序流程与层级堆叠使用组件库结构」，仓库既有共享组件已具备该能力（窄屏自动转纵向）｜引文依据：不适用｜修复要求：用 dg-flow/dg-node/dg-arrow 结构重写该流程块，并删除页内 `.flow-steps`、`.flow-step`、`.flow-arrow` 三条规则（其余页内样式不动）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复
- 备注：本页核心结论与全部关键数字已逐条回源核对——Eq.13/14、Appendix C Eq.20–27 与 Algorithm 1、Appendix D 分箱范围 `[b_min−1, b_max+1]`、`w=(b_max−b_min+2)/B` 与恢复公式、Table 1（K2 384 → K3 896）、`B=1000` 误差「几个 10⁻³」、「低于 raw margins 交换成本的 1%」、「近 10³ 专家几步内收敛」均与报告原文一致；构造示例 m=8/n=4/k=1 的分数矩阵、margins、第 3 大分位数、mean-centering 与新旧路由（(4,3,1,0)→(2,2,2,2)）经复算与实跑代码全部正确，构造数据已在「构造示例」「简化条件及其限制」中标注边界。除上列 3 条外未发现其它来源不符、页内矛盾或表述问题。