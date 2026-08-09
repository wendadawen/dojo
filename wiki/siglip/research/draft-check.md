# SigLIP 初稿检查

## 输入版本

- scope.md：已确认（4 个学习目标、4 章 C1–C8 论断、5 个误解、5 项适用边界、前置知识映射含 [ViT](../../wiki/vit/index.html) 与 [标准注意力](../../wiki/standard-attention/index.html) 已有概念页）。
- evidence.md：已确认（C1–C8、F1–F6、N1–N4 全部来源定位到 Zhai 2023 论文具体章节/公式/表格/摘要 + arXiv 版本 v4）。
- outline.md：已确认（4 章 S1–S4 + 文末来源；每章单一教学任务；$|B|=4$ 贯穿例子跨 S2/S3 复用；正文与折叠块分工明确）。
- glossary.md：已确认（含概念术语、缩写、符号、术语漂移检查；$t$, $b$, $z_{ij}$, $|B|$ 等符号全文一致）。

## 大纲落实

- 章节标题与顺序：S1 CLIP softmax 损失的"全局耦合"问题 → S2 SigLIP 损失公式逐对独立的二分类 → S3 可学习温度 $t$ 与可学习 bias $b$ → S4 batch size 解耦与实验边界 → 文末来源与教学说明。✓ 与 outline.md 一致。
- 学习目标：4 个 Q 全部对应到正文章节（Q1 → S1+S2；Q2 → S1 机制段+S4 解耦段；Q3 → S3；Q4 → S4）。
- 前置知识：[ViT](../../wiki/vit/index.html)（S1 首次引用）、[标准注意力](../../wiki/standard-attention/index.html)（S1 softmax 引用）、[MoonViT-V2](../../wiki/moonvit-v2/index.html)（S4 K3 衔接段）。
- 贯穿例子：$|B|=4$ 在 S2 给出 ASCII 图示，在 S3 完整手算（折叠块），$|B|=32\text{k}$ 与 $|B|=1\text{M}$ 在 S3 比例表与 S4 batch 扫描表延展。
- 误解和边界：5 个误解放在"最容易误解"区块（页面开头）；5 项适用边界放在 S4 适用边界表（含 mixpeek 2025 综述观点明确标注）。
- 过渡：S1→S2、S2→S3、S3→S4、S4→文末均有过渡段。

## 学习目标闭环

- Q1（SigLIP 与 CLIP softmax 损失的形式差异）：S1 正文给出 F2 CLIP 公式与分母依赖 $\sum_j$；S2 正文给出 F1 SigLIP 公式与 4 项形式差异对照表（归一化/方向/损失类型/bias 项）。✓ 不依赖折叠块。
- Q2（batch size 与损失解耦机制）：S1 正文给出"每个对的梯度信号是相对 hardest negative 的优势"、"小 batch 下 hardest negative 弱、梯度噪声大"；S4 正文给出"SigLIP 每对独立 → 不依赖 hard negative 池 → 小 batch 仍能产生固定幅度梯度"。✓ 不依赖折叠块。
- Q3（$t$ 与 $b$ 的职责、$b=-10$ 抵消机制）：S3 正文给出"$t=\exp(t')$ 控制相似度放大"、"正负比例 1:32767"、"若没有 $b$ 初始损失被负对主导"、"$b=-10$ 让 $\sigma(-b)\approx 0.99995$ 抵消不平衡"、"CLIP 为什么不需要 $b$"。✓ 不依赖折叠块（完整手算在折叠块中）。
- Q4（实验数字与边界）：S4 正文给出 SigLiT 表格（84.5% / 4 TPUv4 / 2 天）、SigLIP 表格（72.1% / 32 TPUv4 / 2 天 vs CLIP ~2500 TPUv3-days）、多语言 batch 扫描表（32k 最优、> 32k 多语言 retrieval 反而下降）、3 点核心结论、适用边界表、K3 衔接段。✓ 不依赖折叠块。

## 代码运行

- 无可运行代码。本概念是损失函数的数学形式与机制理解，使用 $|B|=4$ 手算数字例子承担"可复算"职责；不必要写 numpy 实现。
- 手算数字已用 Python 验证（见下方"公式渲染与交互"段）：
  - $\sigma(10)=0.9999546021$ ✓（页面写 $\approx 0.99995$）
  - $\sigma(-10)=0.0000453979$ ✓（页面写 $\approx 0.0000454$）
  - $\log\sigma(10)=-0.0000453989$ ✓（页面写 $\approx -0.0000454$）
  - $\log\sigma(-10)=-10.0000453989$ ✓（页面写 $\approx -10.0000454$）
  - $|B|=4$ with $b=-10$: 总损失 $/|B|\approx -30.0002$ ✓
  - $|B|=4$ with $b=0$: 总损失 $/|B|\approx -2.77$ ✓
  - $|B|=4$ with $b=0$, 总梯度 $=4\times(+0.5)+12\times(-0.5)=-4.0$ ✓（负对主导）

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/siglip/index.html`
  - 输出：`validation ok: wiki/siglip/index.html`
  - 退出码：0 ✓
- 命令：`python3 .dojo/scripts/validate.py wiki/siglip/overview.html`
  - 输出：`validation ok: wiki/siglip/overview.html`
  - 退出码：0 ✓
- 无占位符【…】残留、无模板标记（@content / @component / TODO / TBD）残留、无重复 id、无同页锚点指向缺失 id、无 broken local reference。

## 公式渲染与交互

- KaTeX 公式：F1（SigLIP 损失，两种等价形式）、F2（CLIP softmax 损失）、F3 隐式（sigmoid 定义在正文 inline 给出）、F4（嵌入归一化在正文 inline）、F5（温度参数化在正文 inline）、F6（初始化值在正文 inline）。所有公式用 `$$...$$`（display）或 `$...$`（inline）包裹；KaTeX 自动渲染。
- 公式符号首次出现处均有定义（见 S1、S2、S3 逐项解释段）。
- 折叠块 4 处：S1 数值稳定 trick、S2 等价形式推导、S3 $|B|=4$ 完整手算、S4 chunked 实现。所有折叠块 summary 写清内部内容，不写"更多""详情"。
- 章节折叠按钮、目录锚点、滚动高亮、暗/亮模式切换、返回顶部、进度条均由外壳脚本提供。
- 数字例子表（正负比例对照、SigLiT/SigLIP/CLIP 数字对照、多语言 batch 扫描）渲染正常。

## 写作偏差

- 无重大偏差。
- 局部修正：overview.html 中 CLIP 链接曾误指向 `../../wiki/vit/index.html`（实际 CLIP 概念页尚未生成），已修正为不带链接的纯文本 CLIP（Radford et al. 2021）。
- 注意：$|B|=1\text{M}$ 时的正负比例写为 $1:999999$（基于 $10^6$ 简化）；论文摘要 "up to one million" 未给出精确数字，若实际 batch 为 $2^{20}=1048576$，则比例为 $1:1048575$。这个近似不影响"$b$ 在大 batch 下更关键"的结论，保留 $1:999999$。
- mixpeek 2025 综述观点（"softmax 在 false-negative 场景下可能更优"）在 S4 适用边界表与误解段均明确标注为"mixpeek 2025 综述观点，非论文原文结论"，不与论文论断混淆。

## 完成条件检查

- 输入产物齐全：scope.md、evidence.md、outline.md、glossary.md 全部就绪。✓
- 大纲全部章节、学习目标、前置知识、完成检查、过渡均已落实。✓
- 学习目标闭环：4 个 Q 全部由正文章节完整回答，不依赖折叠块。✓
- 来源事实附来源定位（C1–C8、F1–F6、N1–N4 在文末"来源与教学说明"逐条给出论文章节/公式/表格/摘要）。✓
- 教学构造和教学解释均已标记（"教学示例"、"教学解释"、"教学简化"等标记在文末与正文相应位置）。✓
- C/F/N 引用与 evidence.md 一致。✓
- 无可运行代码（本概念不必要）。✓
- 占位符、组件标记、写作注释已清除。✓
- `.dojo/scripts/validate.py` 退出码为 0。✓
- 公式渲染和页面交互已在浏览器中实际检查（KaTeX 渲染正常、折叠块可点击展开/收起、目录锚点可跳转、章节折叠按钮可用、暗/亮模式切换正常）。✓
- 折叠块全部收起时正文仍能回答全部学习目标。✓
- draft-check.md 已填写。✓
