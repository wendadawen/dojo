# Quantile Balancing 初稿检查

## 输入版本

- scope.md：完成，概念歧义已裁定（无歧义），5 个学习目标（Q1–Q5），内容分级完整，前置知识映射完成（auxiliary-loss-free routing 占位、MoE 引用已有页面、分位数内联定义）
- evidence.md：完成，17 条核心论断（C1–C17）、7 条公式（F1–F7）、3 条数字（N1–N3），全部来源定位到 K3 报告 §2.3.3/§C/§D 或 config.json，置信状态均为已确认
- outline.md：完成，7 章（S1–S7），每章单一教学任务，贯穿例子为 m=8/n=4/k=1 手算，材料职责表完整
- glossary.md：完成，40+ 术语/符号，首次出现位置和定义齐全

## 大纲落实

- 章节落实：S1 aux-loss-free 路由 ✓ / S2 DeepSeek-V3 sign 更新 ✓ / S3 QB 核心机制 ✓ / S4 手算例子 ✓ / S5 对偶理论 ✓ / S6 直方图估计 ✓ / S7 来源说明 ✓
- 学习目标落实：Q1→S1+S2 / Q2→S3 / Q3→S4 / Q4→S5 / Q5→S6，每个目标有正文章节完整回答
- 前置知识落实：MoE 基本路由引用 wiki/moe-serving（已有页面，链接正常）；auxiliary-loss-free routing 占位（页面待生成，em 标签提示）；分位数内联最小定义
- 贯穿例子落实：S3 首次引入 8×4 分数矩阵，S4 完整手算（分数→cutoff→margins→分位数→b̃→centering→验证），S5 用对偶视角解释"一步到 coordinate minimizer"
- 误解和边界落实：3 条误解在页面开头 misconceptions 组件 + 正文对应位置处理；适用边界在 S4 末尾和 S6 末尾
- 过渡落实：每章末尾有完成检查 + 过渡到下一章的具体问题

## 学习目标闭环

- Q1（QB 解决什么问题 / sign 失效）：S1 解释 bias 机制，S2 解释 sign 更新公式和 γ 两难，sign 丢失幅度信息，896 专家时失效。正文完整回答。✓
- Q2（QB 公式每一步）：S3 分四步拆解 Eq.14——Top-(k+1)→cutoff α_i、margins=s−α、b̃=−quantile、mean-centering，每步有符号解释和直觉含义。正文完整回答。✓
- Q3（手算 m=8,n=4,k=1）：S4 完整手算——分数矩阵、cutoff、margins 矩阵、每专家分位数排序表、b̃ 计算、mean-centering、验证新路由 (2,2,2,2)。代码折叠块验证。正文完整回答。✓
- Q4（QB vs sign 本质区别）：S5 从平衡分配→对偶目标→coordinate minimizer 解释 sign 是 SignSGD、QB 是精确解，对照表对比。正文完整回答。✓
- Q5（直方图估计 / 推理冻结）：S6 解释分箱范围、scatter-add+all-reduce、分位数恢复公式、误差界、推理冻结。伪代码折叠块展示流程。正文完整回答。✓

折叠块全部收起时正文仍能回答 Q1–Q5。✓

## 代码运行

- 代码块：wiki/quantile-balancing/index.html 中的可运行代码（纯 Python，无第三方库）
- 运行命令：`python3 /tmp/qb_page_code.py`
- 退出码：0
- 输出与页面"预期输出"一致：
  - 初始 loads = [4, 3, 1, 0]，target q=2 ✓
  - E1 3rd largest = +0.2, b_tilde = -0.2 ✓
  - E2 3rd largest = +0.1, b_tilde = -0.1 ✓
  - E3 3rd largest = +0.0, b_tilde = -0.0 ✓
  - E4 3rd largest = -0.1, b_tilde = +0.1 ✓
  - b_new = [-0.15, -0.05, +0.05, +0.15] ✓
  - QB 后 loads = [2, 2, 2, 2] ✓
  - 改变的 token: T4(E1->E3), T5(E1->E4), T8(E2->E4) ✓

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/quantile-balancing/index.html`
- 结果：`validation ok: wiki/quantile-balancing/index.html`
- 检查项：无模板占位符 【…】、无 @component/@copy 标记、无重复 id、无 broken local reference（修复 aux-loss-free-routing 链接为占位后通过）

## 公式渲染与交互

- KaTeX 公式（$...$ 和 $$...$$）使用模板提供的本地 KaTeX 渲染
- Prism 代码高亮（language-python 和 language-text）使用模板提供的本地 Prism
- 折叠块（details）、表格、callout、章节折叠按钮、侧边目录、进度条、返回顶部均由模板脚本提供
- 公式 Eq.13、Eq.14、Eq.20、Eq.23、Eq.25-26、Eq.27、直方图恢复公式均正确使用 KaTeX 语法

## 写作偏差

- 无写作偏差。大纲的全部章节、学习目标、前置知识、完成检查和过渡均已落实，未增删核心章节、未增加新学习目标、未更换贯穿例子、未改变前置知识映射。
- 修复记录：aux-loss-free-routing 链接从 a href 改为 em 占位提示（页面未生成，避免 broken reference）。
