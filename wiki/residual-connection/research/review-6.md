<!-- review-meta
round: 6
page: wiki/residual-connection/index.html
reviewed_content_sha256: 7a74c4e977f2b56a
-->
# 残差连接审查记录（第 6 轮）

- 页面版本：87b47e3294afab289ea991be00291e3b8e67fe41（index.html 工作树哈希）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、1. 深层网络为什么退化——加更多层反而更差、2. 残差块公式——把"学恒等"变成"学零"、3. 梯度如何流动——为什么深网络不再"消失"、4. 维度不匹配与适用边界（4.1 投影捷径 / 4.2 残差连接不解决什么 / 4.3 在 Transformer 中的应用）、来源与范围说明；含全部 details 折叠块、图注与内联 SVG 结构图；overview.html 一并阅读。

## 来源核对（关键条目与原文片段/数值）

- He et al. 2016 (arXiv:1512.03385) §1/Figure 1：CIFAR-10 上 56 层 plain 训练误差高于 20 层；§4.1/Figure 4：34 层 plain 训练误差高于 18 层 plain —— 与页面第 1 章及 C1 一致。
- 同文 Table 2（10-crop top-1）：plain-18 27.94、plain-34 28.54、ResNet-18 27.88、ResNet-34 25.03 —— 页面表格"约 27.9 / 28.5 / 27.9 / 25.0"一致。
- 同文 Table 3：ResNet-34 A/B/C 25.03 / 24.52 / 24.19（top-1），论文称 A/B/C 差异小、投影捷径对解决退化并非必要 —— 与 C6 一致。
- 同文 §3.2：Eq.(1) y=F(x,{W_i})+x、Eq.(2) y=F(x,{W_i})+W_s x —— 与 F1/F2 逐字一致。
- 同文 §4.2：1202 层 ResNet 在 CIFAR-10 上 test error 7.93%（劣于 110 层 6.43%），作者归因于过拟合 —— 与页面"轻微过拟合"一致。
- 同文摘要："An ensemble of these residual nets achieves 3.57% error on the ImageNet test set."、"won the 1st place on the ILSVRC 2015 classification task" —— 与 N2 一致。
- He et al. 2016 Identity Mappings (arXiv:1603.05027) §2：Eq.(3) x_{l+1}=x_l+F(x_l,W_l)、Eq.(5) ∂E/∂x_l=∂E/∂x_L(1+∂/∂x_l ΣF) —— 支撑 F3/F4/C3（页面以 h 记法改写，等价）。
- Veit et al. 2016 (arXiv:1605.06431) 摘要："most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep" —— 与 N3 及正文"10–34 层"一致；正文 lesion study："deleting any layer in VGG reduces performance to chance levels"、删除单个残差模块"no other block removal lead to a noticeable change" —— 与正文第 3 章一致。
- Vaswani et al. 2017 (arXiv:1706.03762) §3.1：残差连接包住各子层、输出 "LayerNorm(x+Sublayer(x))" —— 与 C5 及正文 Post-LN 公式一致。
- Xiong et al. 2020 (arXiv:2002.04745) 摘要：把归一化放进残差块内部（Pre-LN）使梯度"well-behaved at initialization" —— 与 C8 一致。
- Kimi K3 报告 (arXiv:2607.24653v2) §2.2 Attention Residuals：Eq.(8)(9)(10) 存在，标准残差被描述为"a bottleneck reminiscent of RNNs over time" —— 与 C7 一致；链接 ../block-attnres/index.html 文件真实存在。
- 手算复算：1 神经元例 y=x+w·x，w=0→2、w=0.1→2.2；plain w=1→2、w=0.1→0.2。3 层例：残差输出 1.1×2=2.2 → 2.42 → 2.662、梯度 1.1^3=1.331；plain 输出 0.2 → 0.02 → 0.002、梯度 0.1^3=0.001。全部与页面一致。
- 8 路径表（3 块）：2^3=8 行，长度 0/1/1/2/1/2/2/3 —— 枚举完整、计数正确；乘积展开 2^{L-l} 项、L-l=3 时 8 项亦正确。
- 图形几何核验：节点 x(20,97)-F(122,42)-+(340,115,r18)-y(504,97)，两条路径箭头端点止于 + 圆上/左侧（340,95 与 x=318）而不压圆，两处 12px 标注位于对应连线下方，无压线重叠；图内公式全部置于 foreignObject 并由 KaTeX 渲染，aria-label 无 $...$。
- 结构项：validate.py 返回 "validation ok"；dojo:topics=数学基础、dojo:tag=网络结构 均在词表内；overview.html 与 index.html 互链；引用编号 C1–C8 / F1–F4 / N1–N3 全部定义且均被正文引用，无孤儿编号；无"（待生成）"占位；两级问题块每题均有解答折叠块。

## 问题

- [轻微·表述] 第 1/2/3 章末过渡（第 184、279、390 行）：三处过渡句使用同一骨架"本章说明了 X。但 Y——下一章讲 Z。"，起始措辞与句式完全同构，属规范 §8 所指"固定句式"｜引文依据：不适用｜修复要求：保留每处"本章结论→下一章待解问题"的逻辑衔接，但改写其中一两处的起始措辞（不再以"本章说明了"重复开头，改用该章具体结论作主语），使三处不再同构｜修复：｜复验：

## 结论

- 处置：可发布（唯一轻微项为过渡句表述打磨，不影响正确性与主线理解，可接受；如修复仅改动三处过渡句，不触及任何事实、公式与数字）
- 统计：阻断 0 / 重要 0 / 轻微 1