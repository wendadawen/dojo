<!-- review-meta
round: 5
page: wiki/residual-connection/index.html
reviewed_content_sha256: d1efe1fb2b2e42e2
-->
# 残差连接审查记录（第 5 轮）

- 页面版本：d1458c8a837d97d102781e6b21b5da66f7c0e564（wiki/residual-connection/index.html，工作树）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查与修复；未读取 research/ 下任何文件）
- 已完整阅读章节：核心问题；1. 深层网络为什么退化——加更多层反而更差；2. 残差块公式——把"学恒等"变成"学零"；3. 梯度如何流动——为什么深网络不再"消失"；4. 维度不匹配与适用边界（4.1 维度不匹配时：投影捷径 / 4.2 残差连接不解决什么 / 4.3 在 Transformer 中的应用）；来源与范围说明。含全部折叠块、图注与 overview.html。

## 问题

- [重要·技术] 4.1 节「维度不匹配时：投影捷径」第二段：括号把"补零"捷径对应的情形写成"仅空间尺寸变化时"，与所引论文的分工不符，形成错误规则。｜引文依据：He et al. 2016 §3.3 原文 "When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2."；§4.1 "In Table 3 we compare three options: (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free"。即：补零是"通道数增加"这一维的参数免费做法，空间尺寸跨两个特征图时捷径按 stride 2 取样——两者不是"通道数变化 → 1×1 卷积 / 仅空间尺寸变化 → 补零"的分工；按现写法读者会记下"补零用于只变空间尺寸"这一与来源相反的规则（且按该写法在仅空间变化时补零并不能对齐尺寸）。｜修复要求：改写该括号使其与来源一致，例如"工程上通常用 $1\times1$ 卷积投影匹配通道数变化；空间尺寸跨两个特征图时，捷径按步长 2 取样（补零只用于仅通道数增加、空间不变的参数免费情形）"，或删去括号内的情况标注。｜修复：｜复验：

- [轻微·技术] 4.3 节「在 Transformer 中的应用」第二段："后续工作提出 Pre-LN 形式[C8]"——引文位置使读者把 Pre-LN 的提出归给 [C8]（Xiong et al. 2020），而该来源摘要原文把 Pre-LN 称为"recently proposed"，正文亦把提出者列为他人，故 [C8] 不支持"提出"这一归因。｜引文依据：arXiv:2002.04745 摘要 "if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization"；正文 "the Transformer with Pre-Layer Normalization (Pre-LN) ( Baevski & Auli 2018 ; Child et al. 2019 ; Wang et al. 2019 )"。｜修复要求：把"提出"与 [C8] 解耦，改为"后续工作提出 Pre-LN 形式；Xiong et al. 2020 分析了它的初始化梯度行为[C8]"，或写明提出者（Baevski & Auli 2018 等）。｜修复：｜复验：

## 已核对来源（引文依据）

- C1/N1（He 2016 ResNet，ar5iv 全文）：§1 "Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error"；Fig. 1 图注 "Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer 'plain' networks."；Fig. 4 图注 "Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers." —— 与第 1 章表述一致。
- §1 "this problem ... has been largely addressed by normalized initialization ... and intermediate normalization layers [BN]"；§4.1 "These plain networks are trained with BN ... the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish." —— 支持"梯度消失已被 BN 与良好初始化基本解决""He 的实验中梯度没有消失而退化仍出现"。
- 第 1 章表格数字：Table 2 "plain 18 layers 27.94 / 34 layers 28.54；ResNet 18 layers 27.88 / 34 layers 25.03"（top-1，10-crop validation）—— 页面"约 27.9 / 28.5 / 27.9 / 25.0"一致；N2 的"约 25.0% vs 27.9%"一致。
- C2/F1/F2：§3.1 "we explicitly let these layers approximate a residual function F(x):=H(x)−x. The original function thus becomes F(x)+x."；§3.2 Eqn.(1) y=F(x,{W_i})+x、Eqn.(2) y=F(x,{W_i})+W_s x；§3.2 "Experiments in this paper involve a function F that has two or three layers"。
- C6：§4.1 Table 3 行 "ResNet-34 A 25.03 / B 24.52 / C 24.19"（top-1）；"C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts"；"the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem." —— 引文编号与 Table 3 定位正确。
- 1202 层（§4.2）："this 10^3-layer network is able to achieve training error < 0.1%（Fig. 6, right）... Its test error is still fairly good (7.93%, Table 6) ... We argue that this is because of overfitting. The 1202-layer network may be unnecessarily large (19.4M)"（Table 6：ResNet-110 6.43 / ResNet-1202 7.93）。页面"轻微过拟合"比来源弱，但来源确把原因归为过拟合并称结果"still fairly good"，未改变结论，本轮不另立条目。
- 来源章节"辅助解释"引用的 Figure 7：Fig. 7 图注 "Standard deviations (std) of layer responses on CIFAR-10 ... Top: the layers are shown in their original order."；§4.2 "Fig. 7 shows that ResNets have generally smaller responses than their plain counterparts." —— 支持"多数层响应偏小是观察结果"。
- C3/F3/F4（He 2016 Identity Mappings，§2 Analysis）：Eqn.(5) "∂E/∂x_l = ∂E/∂x_L (1 + ∂/∂x_l Σ_{i=l}^{L−1} F(x_i,W_i))"，"can be decomposed into two additive terms"，"the gradient of a layer does not vanish even when the weights are arbitrarily small." —— 页面 F4 的 ∏(1+∂F_i/∂h_i) 与该式等价（逐层链式法则），标注 §2 正确。
- C4/N3（Veit 2016，1605.06431）：摘要 "most of the gradient in a residual network with 110 layers comes from paths that are only 10-34 layers deep"；正文 "deleting individual modules from residual networks has a minimal impact on performance"、"deleting any layer in VGG reduces performance to chance levels"、"residual networks have 2^n paths connecting input to output"、"in these networks, input always flows from the first layer straight through to the last in a single path"。—— 第 3 章 lesion study 与 2^n 路径表述一致。
- C5（Vaswani 2017 §3.1）："a residual connection around each of the two sub-layers, followed by layer normalization"，"LayerNorm(x+Sublayer(x)"；Figure 1 "The Transformer - model architecture." —— 一致。
- C7（Kimi K3 技术报告 arXiv:2607.24653v2 §2.2 Attention Residuals）：Eq.(8) k_i=v_i={h_1 (i=0); f_i(h_i) (1≤i≤l−1)}；Eq.(9) α_{i→l}=φ(q_l,k_i)/Σ_j φ(q_l,k_j)、h_l=Σ_i α_{i→l}·v_i；摘要 "Attention Residuals (AttnRes) ... allows each layer to selectively attend to representations from all preceding layers"。—— "对残差项做注意力加权而非等权相加"得到支持。
- C8（Xiong 2020 摘要）：见[轻微]条目引文；"把归一化放进残差块内部即 Pre-LN，使初始化时梯度更平稳"与摘要一致。
- 公式可复算：3 块梯度 ∏(1+0.1)=1.331、plain 0.1^3=0.001（差 3 个数量级）；1 神经元例 x=2、F=w·x：residual w=0→2、w=0.1→2.2，plain w=1→2、w=0.1→0.2；折叠块 h1=2.2、h2=2.42、h3=2.662，∂h3/∂x=1.331 与 0.001，均与正文逐项一致；$(1+f_l)(1+f_{l+1})(1+f_{l+2})$ 的 8 项展开逐项吻合；8 条路径表长度 0–3 与 $2^3$ 一致。符号全文单义（$F/F_i$、$h_i$、$W_s$），summary 内 $\partial y/\partial x=1+\partial F/\partial x$、$h_{l+1}=h_l+F(h_l)$ 为可渲染 KaTeX。
- 机械项：`.dojo/scripts/validate.py wiki/residual-connection/index.html` 返回 "validation ok"；公式分界符外无 Unicode 数学字符；SVG 图内公式全部在 foreignObject，`<text>` 仅有"2–3 个卷积/线性层""恒等捷径（无参数）"纯文字；图内元素按 viewBox 0 0 560 230 测量无压线/重叠（最右元素至 x=548），节点与箭头含义由图注定义；`$...$` 未出现在 alt/aria-label 中；链接 ../../index.html、overview.html、../block-attnres/index.html 均真实存在，overview 与 index 互链，无"（待生成）"占位；C1–C8/F1–F4/N1–N3 在正文与来源章节双向对应，无编号漂移；页面级「核心问题」4 条与各章「本章问题」均有 `解答：` 折叠块，答案与正文结论一致并指向所在章节。
- 表述：全文（含折叠块与图注）无第一人称复数、无第二人称、无调试叙事与临场评价；「本页/本文」的自指用法符合 style-guide §12，未出现"本页将…""下面来看…"类元话语；页内无交互视图，去脚本后正文与折叠块仍可读。

## 结论

- 处置：修复（1 条重要 + 1 条轻微；无阻断）
- 统计：阻断 0 / 重要 1 / 轻微 1