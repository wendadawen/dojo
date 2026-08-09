# 残差连接初稿检查

## 输入版本

- scope.md：完成，含概念边界、4 个学习目标、内容分级、前置知识映射、不展开项、误解与边界
- evidence.md：完成，含 C1–C5 论断、F1–F4 公式、N1–N3 数字，全部置信状态"已确认"，无冲突
- outline.md：完成，含页面开头、S1–S4 四章、讲解顺序、贯穿例子、材料职责、正文折叠块分工
- glossary.md：完成，含全部术语、符号、缩写

## 大纲落实

- 页面开头钩子："把网络从 20 层堆到 56 层，训练误差反而更高"——落实
- 页面开头一句话解释——落实
- 学习承诺（Q1–Q4）——落实
- 首个场景（plain 网络加深到退化）——落实
- 与第一章过渡——落实
- S1 深层网络为什么退化——落实
- S2 残差块公式——落实
- S3 梯度如何流动——落实
- S4 维度不匹配与适用边界——落实
- 前置知识引用：neural-network-layer、chain-rule、gradient-descent 三处占位标注——落实
- 贯穿例子：1 神经元残差块在 S2、S3 复用——落实
- 误解处理：退化≠过拟合、退化≠梯度消失（S1）；F 不一定是小量（S2 隐含）；残差没彻底解决梯度消失（S3 callout）——落实
- 边界处理：投影捷径、不解决的问题、Transformer 应用——落实
- 章间过渡 S1→S2、S2→S3、S3→S4——落实
- 文末来源与教学说明——落实

## 学习目标闭环

- Q1（退化问题是什么、与过拟合/梯度消失的区别）：S1 正文完整回答——通过
- Q2（公式、符号、为什么学零比学恒等容易）：S2 正文完整回答，含 1 神经元手算——通过
- Q3（梯度结构、多层乘积、手算例子）：S3 正文完整回答，含 3 层梯度量级对照——通过
- Q4（投影捷径、不解决的问题、集成解释边界、Transformer）：S4 正文完整回答——通过
- 折叠块全部收起时正文是否仍回答全部目标：是——通过

## 代码运行

无可运行代码。本页核心机制（加法捷径与梯度直通项）用 1 神经元手算例子验证，无需程序。大纲未安排可运行代码。

## 机械检查

```
python3 .dojo/scripts/validate.py wiki/residual-connection/index.html
→ validation ok: wiki/residual-connection/index.html
→ EXIT: 0

python3 .dojo/scripts/validate.py wiki/residual-connection/overview.html
→ validation ok: wiki/residual-connection/overview.html
→ EXIT: 0
```

## 公式渲染与交互

- KaTeX 行内公式 $...$ 与块级公式 $$...$$ 均由外壳脚本 auto-render 渲染
- 4 个正文章节 h2 均有显式 id，目录自动生成
- 折叠块 details/summary 结构正确
- 代码块无（本页不安排可运行代码）
- 占位符【...】与模板标记 @content/@component 均已清除（grep 验证为 0）

## 写作偏差

无。大纲全部章节、学习目标、前置知识、贯穿例子、误解和边界、过渡均已落实，无返回规划或局部修正。
