<!-- review-meta
round: 4
page: wiki/residual-connection/index.html
reviewed_content_sha256: 645ad8c8460542cd
-->
# 残差连接审查记录（第 4 轮）

- 页面版本：2abddae073975d22c8d2c5a7be2b2f6f17d23018
- 审查时间：2026-09-13 21:18
- 审查者：独立子代理（未参与写作，未参与前 1–3 轮审查与修复）
- 已完整阅读章节：引言、核心问题（4 条含解答）、1. 深层网络为什么退化——加更多层反而更差、2. 残差块公式——把"学恒等"变成"学零"、3. 梯度如何流动——为什么深网络不再"消失"、4. 维度不匹配与适用边界（4.1 投影捷径 / 4.2 残差连接不解决什么 / 4.3 在 Transformer 中的应用）、来源与范围说明；另读 overview.html、两处「展开」折叠块、两处 callout、SVG 结构图与图注。

## 来源核对（本轮逐条回源，摘录核对位置原文/数值）

- C1/N1（退化问题）：ar5iv 原文 Figure 1 标注 "Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer 'plain' networks."，Figure 4 标注 "Training on ImageNet. …… Left: plain networks of 18 and 34 layers."，与页面「CIFAR-10 20 vs 56（Figure 1）」「ImageNet 34 vs 18（Figure 4）」的标注一致，未错标图号。
- 第 1 章表格数字：He et al. 2016 Table 2 为 plain-18 27.94 / plain-34 28.54 / ResNet-18 27.88 / ResNet-34 25.03（top-1），与页面「约 27.9% / 28.5% / 27.9% / 25.0%」逐项吻合。
- C6/F2：Table 3 为 A/B/C = 25.03 / 24.52 / 24.19，与 C6 记录一字不差；原文 "C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts." 与 "the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem." 支持「维度相同时额外投影仅微小提升、可归因参数增加、投影并非必要」。
- C3/F3/F4：arXiv:1603.05027 §2 Eq.(5) "∂ℰ/∂x_l = ∂ℰ/∂x_L(1 + ∂/∂x_l Σ F)"，并称该项 "propagates information directly without concerning any weight layers"，支持「常数 1 直通项」论断与乘积展开式。
- C4/N3：arXiv:1605.06431 摘要 "most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep."；§4.1 "deleting any layer in VGG reduces performance to chance levels"，支持 N3 与「删 VGG 一层即崩溃」的对照。
- C5：arXiv:1706.03762 §3.1 "the output of each sub-layer is LayerNorm(x + Sublayer(x))"，支持「原版 Post-LN」表述。
- C7：与站内 wiki/block-attnres、wiki/kimi-k3 对「Kimi K3 Technical Report §2.2 Eq.(8)(9)(10)、AttnRes 对残差来源做注意力加权」的核对一致，链接 ../block-attnres/index.html 真实存在。
- 边界表：ar5iv §4.2 "Its test error is still fairly good (7.93%)" 与 "We argue that this is because of overfitting."，支持「1202 层在 CIFAR-10 上轻微过拟合」；Figure 7 标注 "Standard deviations (std) of layer responses on CIFAR-10."，支持辅助解释边界一节。
- 全页公式/示例复算：1 神经元表 4 行（2+0=2、2+0.2=2.2、1×2=2、0.1×2=0.2）、3 层梯度表（(1+0.1)^3=1.331、0.1^3=0.001）、展开折叠块（2.2→2.42→2.662）、8 路径表 8 行长度 0–3，全部自洽；链式展开 (1+f_l)(1+f_{l+1})(1+f_{l+2}) 的 1+3+3+1=8 项与「2^{L-l} 项」一致。
- 机械项：.dojo/scripts/validate.py 返回 "validation ok"；无 Unicode 数学字符出现在公式定界符之外；无「（待生成）」占位；无第一/第二人称（我/我们/你）；无 img alt 含 $...$；SVG 内公式均在 foreignObject，`<text>` 只含纯文字；dojo:type=concept、dojo:topics=数学基础（在固定大类内）、description/summary/tag 齐全；index.html 与 overview.html 双向链接。

## 问题

- [轻微·技术] 4.3 在 Transformer 中的应用，第 2 段「后续工作广泛采用 Pre-LN 形式[C8]（本页不展开其采纳史）」：[C8] 只能支持 Pre-LN 这一公式形式及其「初始化时梯度平稳」，不支持「后续工作广泛采用」这一采纳广度论断；该论断既非 [C8] 所载，也未另附一手材料。｜引文依据：Xiong et al. 2020 摘要原文为 "if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization"，仅称 Pre-LN 为「recently proposed」，未述及采纳范围。｜修复要求：删去「广泛采用」这一无来源论断（保留「后续工作提出/采用 Pre-LN 形式」的弱表述亦可），或改用能证明采纳范围的一手材料；[C8] 仅保留用于它支持的「Pre-LN 形式 / 初始化梯度平稳」。｜修复：｜复验：
- [轻微·技术] 来源与范围说明·外部数字与实验条件（N）N2「ResNet-152 集成在 ImageNet 测试集上 top-5 错误率 3.57%，赢得 ILSVRC 2015」：3.57%、top-5、ILSVRC 2015 冠军均与来源一致，但集成对象被窄化为「ResNet-152」；原文集成的是论文给出的若干 ResNet，而非专指 152 层。｜引文依据：arXiv:1512.03385 摘要 "An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task."（"these residual nets"，非 "152-layer ensemble"）。｜修复要求：把「ResNet-152 集成」改为「ResNet 集成」或「由若干 ResNet 组成的集成」；数值与冠军表述保留。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：可发布（2 项轻微已列明可判定的修复要求，建议按此修复或在下一轮记录接受理由；无阻断、无重要遗留）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
