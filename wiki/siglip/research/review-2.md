# SigLIP 独立审查（第二次）

- 审查者：独立上下文（AI 模拟）
- 页面版本：8961173c4cc70404e145320bbf8314df6765066e（index.html 工作树哈希）
- 时间：2026-08-09

## 审查范围

- 输入：`wiki/siglip/index.html`、`wiki/siglip/overview.html`、`guides/concept/check.md`、`guides/concept/content-examples.md`
- 外部来源：WebSearch "SigLIP Zhai 2023 arxiv 2303.15343"（损失公式 §3.2、bias 初始化 §3.2 第 3 段、SigLiT 84.5% Table 1 + §4.4、SigLIP 72.1% Table 1 + §4.5、batch size 扫描 Table 2、1M 边际收益消失摘要）+ WebFetch arxiv.org/abs/2303.15343 与论文翻译核对公式符号
- 未读取 `research/` 目录、未读取仓库中其他概念页（仅验证链接路径存在）

## 段 A 盲读小结

按页面顺序阅读，四个学习目标在正文中均有正文回答，但学习目标第 3 条（$t$ 与 $b$ 的职责、为什么 $b=-10$）的回答存在核心公式与机制错误（见阻断问题）：

1. SigLIP 损失函数与 CLIP 差异 → "CLIP softmax 损失的全局耦合问题"+"SigLIP 损失公式"两章回答
2. 逐对独立在大 batch 下保持小 batch 优势 → "batch size 解耦与实验边界"章回答
3. $t$ 与 $b$ 的职责、为什么 $b=-10$ → "可学习温度 $t$ 与可学习 bias $b$"章回答，**但 $b$ 的作用机制描述与原论文相反**（见阻断问题）
4. 小算力场景优势与边界 → "batch size 解耦与实验边界"章回答

术语首现解释充分（CLIP 双塔、L2 归一化、softmax 分母耦合、sigmoid 逐对独立、温度、bias、InfoNCE 均有定义或最小概念说明）。三个折叠块（softmax 数值稳定 trick、等价形式推导、$|B|=4$ 手算）前后均能接回主线。前置链接 ../../wiki/vit/index.html、../../wiki/standard-attention/index.html、../../wiki/moonvit-v2/index.html 路径均存在。

## 段 B 对照来源小结

1. 定义与机制：[C1] SigLIP 用 sigmoid 逐对二分类替代 CLIP softmax 与 §3.2 一致；[C2] CLIP softmax 分母依赖整个 batch 与 §3.1 一致；[C3] 每个对梯度独立与 §3.2 + §3.3 chunked 实现一致；[C5] batch size 与损失解耦、小 batch 下显著优于 softmax 与 §3.2 + §4.2 一致；[C6] SigLiT 84.5% / 4 TPUv4 / 2 天与 Table 1 + §4.4 一致；[C7] SigLIP 72.1% / 32 TPUv4 / 2 天、CLIP ~2500 TPUv3-days 与 Table 1 + §4.5 末段一致；[C8] 32k 接近最优、>32k INet-0 几乎不再提升、多语言 retrieval 反而下降、1M 边际收益消失与摘要 + §4.2 + §4.3 + Table 2 一致。
2. 公式与推导：[F2] CLIP softmax 损失与 §3.1 一致；[F3] sigmoid 定义标准；[F4] 嵌入归一化与 §3.1 一致；[F5] 温度参数化 $t=\exp(t')$ 与 §3.1 一致；[F6] 初始化 $t'=\log 10$（$t=10$）、$b=-10$ 与 §3.2 第 3 段一致。**[F1] SigLIP 损失公式符号错误**（见阻断问题）。
3. 可运行代码：页面无可运行代码块，只有 ASCII 围道图与公式，不适用。
4. 事实与推断：[N1] SigLiT B/8 L 79.8%、SigLiT g/14 L 84.5% 与 Table 1 一致；[N2] SigLIP 16k/16 TPUv4/3 天 71.0%、32k/32 TPUv4/2 天 72.1%、32k/32 TPUv4/5 天 73.4% 与 Table 1 一致；[N3] batch size 扫描 16k→240k 的 INet-0 与 XM avg 与 Table 2 一致；[N4] 1M 边际收益消失与摘要一致。
5. 前置知识引用：vit、standard-attention、moonvit-v2 三个目录均存在，链接层级正确。
6. 教学简化：五项简化（CLIP softmax 公式直接引用、$|B|=4$ 手算用简化假设、只引用两个代表数字、不展开 chunked 伪代码、MoonViT-V2 一句话提及）均有说明。**但简化二（$|B|=4$ 手算）基于错误公式**（见阻断问题）。
7. 页面功能：KaTeX 渲染、details 折叠、自动生成目录锚点（h2 均有显式 id: clip-softmax-global-coupling、siglip-loss-formula、temperature-and-bias、batch-size-decoupling-and-boundary、sources-and-teaching-notes）结构正确。

数字复算（除 [F1] 公式相关外）：SigLiT g/14 L 84.5%、SigLIP 72.1%、CLIP ~2500 TPUv3-days 与论文一致 ✓；batch size 扫描 32k 最优、>32k INet-0 几乎不再提升、XM avg 34.9→32.7、XM zh 32.5→23.7 与 Table 2 一致 ✓；正负比例 |B|=32k → 1:32767、|B|=1M → 1:999999 计算正确 ✓。

## 问题

- [阻断·技术] §2 [F1] 公式 + §3 "b 的作用" + §3 折叠块"$|B|=4$ 手算" + §3 常见误解 callout + §3 检查问题（多处联动）：[F1] 公式符号错误，导致 $b$ 作用机制描述与原论文完全相反。页面 [F1] 写 $\log\sigma(z_{ij}(t\,x_i\cdot y_j-b))$（$-b$），并声称原论文形式为 $\log\tfrac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j+b)}}$（指数 $+b$）。但原论文 §3.2 Algorithm 1 伪代码为 `logits = t·x_i·y_j + b`（$+b$），等价形式为 $\log\sigma(z_{ij}(t\,x_i\cdot y_j+b))$ = $\log\tfrac{1}{1+e^{z_{ij}(-t\,x_i\cdot y_j-b)}}$（指数 $-b$）。符号反了。由此导致：(a) §3 "b 的作用" 段写"初始 $b=-10$ 使初始 logit $a=z_{ij}(t\cdot 0-(-10))=z_{ij}\cdot 10$，正对 $\sigma(10)\approx 0.99995$、负对 $\sigma(-10)\approx 0.0000454$，$b$ 让初始模型假设任何对都偏向正类（$\sigma(-b)=\sigma(10)\approx 0.99995$）"——原论文实际是 $b=-10$ 让 logit $=t\cdot 0+(-10)=-10$，$\sigma(-10)\approx 0.0000454$，模型初始预测偏向**负**类（与数据先验"负对远多于正对"一致），正对 $z=+1$ 时 $\log\sigma(-10)\approx -10$（损失大）、负对 $z=-1$ 时 $\log\sigma(10)\approx -0.00005$（损失小），正对产生大梯度抵消负对的数量优势（论文 §3.2 第 3 段 "the heavy imbalance coming from the many negatives dominates the loss... we introduce $b$" + §4.9 "the bias term ensures that the training starts close to the prior"）。(b) 折叠块手算：页面写"4 个正对总损失 $\approx -0.000182$、12 个负对总损失 $\approx -120.0005$、总损失 $\approx -120.0007$、除以 4 得 $\approx -30.0002$"——基于错误公式；正确应为"4 个正对总损失 $\approx -40$、12 个负对总损失 $\approx -0.0006$、总损失 $\approx -40.0006$、除以 4 得 $\approx -10.0002$"；页面"对照 $b=0$"段中"$b=-10$ 通过让初始正对 loss 极小、负对 loss 极大"也反了，应为"正对 loss 极大、负对 loss 极小"。(c) 常见误解 callout "$\sigma(-b)\approx 0.99995$ 抵消该不平衡" 应为 "$\sigma(b)=\sigma(-10)\approx 0.0000454$，让模型初始预测偏向负类（与数据先验一致）"。(d) 检查问题"算出 $b=-10$ 时 $\sigma(-b)=\sigma(10)\approx 0.99995$" 应为 "$\sigma(b)=\sigma(-10)\approx 0.0000454$"。(e) §3 "更准确地说" 段中 logit 定义 $a=z_{ij}(t\,x_i\cdot y_j-b)$ 应改为 $a=z_{ij}(t\,x_i\cdot y_j+b)$（$b=0$ 时该段梯度分析仍然成立，但符号需同步修正）。这是学习目标第 3 条（"$t$ 与 $b$ 各自承担什么职责、为什么 $b=-10$"）的核心回答，公式与机制错误使完全小白读者形成"$b$ 让模型偏向正类"的错误理解，属于核心公式与核心机制错误。修法：(1) 将 [F1] 两种等价形式中的 $-b$ 改为 $+b$、指数 $+b$ 改为 $-b$；(2) 同步修正 §2 "关键机制"段、§2 折叠块"等价形式推导"、§3 "b 的作用"段、§3 折叠块"$|B|=4$ 手算"（含"对照 $b=0$"段）、§3 "更准确地说"段、§3 常见误解 callout、§3 检查问题中所有 $-b$ / $\sigma(-b)$ 的写法为 $+b$ / $\sigma(b)$ 对应形式，并把"$b$ 让初始模型假设任何对都偏向正类"改为"$b$ 让初始模型预测偏向负类（与数据先验一致），正对产生大梯度抵消负对的数量优势"；(3) 修正后重新对照原论文 §3.2 第 3 段与 §4.9 确认机制描述一致。 ｜ 修复：已将 [F1] 两种等价形式中的 $-b$ 改为 $+b$、指数 $+b$ 改为 $-b$（index.html §2 F1 公式、等价形式说明、关键机制段、等价推导折叠块、§3 更准确地说段、§4 章首、[F1] 来源说明；overview.html 是什么段与无分母耦合段）。同步修正机制描述：§3 "b 的作用"段改为"初始 logit $=z_{ij}\cdot(-10)$，正对损失大($\approx -10$)、负对损失小($\approx -0.00005$)，$b$ 让模型预测偏向负类(与数据先验一致)，少数正对大梯度抵消多数负对数量优势"；§3 |B|=4 手算改为正对总损失 $\approx -40.0002$、负对总损失 $\approx -0.0005$、总损失 $\approx -40.0007$、除以 4 得 $\approx -10.0002$；§3 对照 b=0 段改为"正对 loss 极大、负对 loss 极小、预测偏向负类"；§3 常见误解 callout 改为 $\sigma(b)=\sigma(-10)\approx 0.0000454$；§3 检查问题改为 $\sigma(b)\approx 0.0000454$；教学示例改为正对 $\log\sigma(-10)$、负对 $\log\sigma(10)$；σ 计算与类比边界标题同步改为"预测偏向负类"；ASCII 围道图 t·x_i·y_j-b 改为 +b；overview.html bias 描述同步。validate.py 通过。 ｜ 复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 0
- 处置：进入修复
