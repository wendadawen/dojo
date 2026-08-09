# 辅助损失无关路由 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 已写入 `research/`，规划完成条件全部满足（5 个学习目标互不重复、前置知识映射到 `moe-serving/` 与 `quantile-balancing/`、核心论断已对照原始论文与 DeepSeek-V3 报告、章节单一任务与贯穿例子齐备）。

## 大纲落实

- 页面开头：钩子场景（4 专家 top-2 的坍塌画面）+ 一句话定义 + 学习承诺 5 项 + 4 条常见误解 + 主要依据 meta。
- S1（负载坍塌与辅助损失两难）：路由坍塌、专家并行吞吐动机、辅助损失方案 $L_{bal}=\alpha\sum f_i P_i$、$\alpha$ 两难、干扰梯度定义、Table 2 对照。
- S2（bias 加在路由分数上）：F1 路由分数、F2 选择规则、F3 归一化、F5 MoE 层输出、两个边界检查（全体 bias 平移不变、bias 不进 $g$）。
- S3（固定步长 sign 更新）：F4 更新规则、负载 $c_i$ 与均值 $\bar c_i$ 定义、偏差 $e_i$、规则式而非梯度式、sign vs 幅度变体、因果约束、初始化。
- S4（手算 4 专家 top-2）：4 token × 4 专家打分表、首轮选择、负载统计、bias 更新逐步代入、第二轮选择验证单步影响小、$\gamma=0.05$ 对照展示 sign 步长特性。
- S5（DeepSeek-V3 配置与边界）：context-box 列 N1-N3、$\gamma$ 调度、sequence-wise loss 互补、推理冻结、K3 改进点。
- 文末：来源与教学说明 6 节齐全（C1-C6 论断来源、F1-F6 公式来源、N1-N5 数字来源、教学示例、教学解释边界、教学简化）。
- 折叠块：4 处（变体实验对照、伪代码、S4 手算逐步代入、$\gamma=0.05$ 对照）。
- 贯穿例子：4 专家 top-2 在 S2/S3/S4 推进；S5 切换到 DeepSeek-V3 真实数字。
- 误解与边界：4 条常见误解在页面开头集中给出，并在 S2/S3/S5 各自对应章节强化；适用边界集中在 S5 末尾列表。
- 过渡：每章末尾有"过渡段"指向下一章要解决的问题。

## 学习目标闭环

- Q1（为什么需要负载均衡，辅助损失缺陷）：S1 完整回答。路由坍塌 + 专家并行吞吐动机 + $\alpha$ 两难 + 干扰梯度。
- Q2（bias 做什么不做什么）：S2 完整回答。F2 选择规则、F3 归一化、F5 MoE 输出、两个边界检查。S3 末段再次强调"规则式不进梯度"。
- Q3（sign 更新规则与 sign 选择理由）：S3 完整回答。F4 + 负载定义 + 规则式 vs 梯度式 + 因果约束 + sign vs 幅度对照。
- Q4（手算 4 专家 top-2 训练步）：S4 完整回答。打分表 → 首轮选择 → 负载统计 → bias 更新 → 第二轮验证。所有数字逐步代入，可复算。
- Q5（DeepSeek-V3 配置与边界）：S5 完整回答。N1-N3 配置、$\gamma$ 调度、序列内互补、推理冻结、K3 改进点。

折叠块全部收起时正文仍能回答 5 个学习目标：S4 的逐步手算在折叠块内，但首轮选择、负载统计、更新规则、第二轮验证结果都在正文；变体实验对照在折叠块内，但"sign vs 幅度"的结论性判断在正文；$\gamma=0.05$ 对照在折叠块内，正文只用 $\gamma=0.001$ 主线已说明 sign 步长特性。

## 代码运行

无可运行代码（Python 块）。本页只有 1 处伪代码（language-text），按 write.md A6 标记为伪代码、不是 Python；按 write.md §6 第 3 项"无代码时写明无可运行代码"。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/aux-loss-free-routing/index.html`
- 结果：`validation ok: wiki/aux-loss-free-routing/index.html`，退出码 0。
- 命令：`python3 .dojo/scripts/validate.py wiki/aux-loss-free-routing/overview.html`
- 结果：`validation ok: wiki/aux-loss-free-routing/overview.html`，退出码 0。

## 公式渲染与交互

- KaTeX：display 用 `$$…$$`，inline 用 `$…$`，与外壳 `auto-render.min.js` 配置一致；F1-F6 六个公式全部用 display 模式；行内符号如 $b_i$、$s_{i,t}$、$\gamma$、$\bar c_i$ 用 inline。
- 公式编号：F1-F6 在公式末尾用 `\text{(F1)}` 等标注，与 evidence.md 一致。
- Prism：1 处伪代码标 `language-text`，无 `language-python` 代码块。
- 折叠交互：4 处 `<details>` + `<summary>`；外壳脚本已绑定章节折叠按钮（h2 上的 ▼）。
- 目录锚点：6 个 h2 章节 ID（s1-why-balance、s2-bias-in-forward、s3-bias-update、s4-hand-example、s5-deepseek-v3-config、sources-and-teaching-notes）全部唯一稳定。
- 浏览器检查：本环境无法实际打开浏览器；validate.py 已确认无占位符、无残留模板标记、无重复 ID、无指向缺失 ID 的锚点、无失效本地引用。审查阶段需在浏览器中实际打开核对 KaTeX 渲染、折叠交互、目录滚动高亮。

## 手算例子复算

S4 的 4 专家 top-2 例子的所有数字逐步复算（保留两位小数）：

- 首轮选择（$b=0$）：
  - $t_1$ 排序 0.60($E_3$) > 0.30($E_0$) > 0.10($E_1$) > 0.05($E_2$)，选 $\{E_3,E_0\}$。
  - $t_2$ 排序 0.50($E_3$) > 0.40($E_0$) > 0.20($E_1$) > 0.10($E_2$)，选 $\{E_3,E_0\}$。
  - $t_3$ 排序 0.55($E_3$) > 0.35($E_0$) > 0.20($E_2$) > 0.15($E_1$)，选 $\{E_3,E_0\}$。
  - $t_4$ 排序 0.45($E_3$) > 0.30($E_1$) > 0.25($E_0$) > 0.10($E_2$)，选 $\{E_3,E_1\}$。
- 负载统计：$c_0=3$、$c_1=1$、$c_2=0$、$c_3=4$，总 $\sum c_j=8$（与 4 token × top-2 = 8 一致），$\bar c_i=2$。
- 偏差：$e_0=-1$（过载）、$e_1=+1$（欠载）、$e_2=+2$（更欠载）、$e_3=-2$（更过载）。
- 更新（$\gamma=0.001$）：$b_0=-0.001$、$b_1=+0.001$、$b_2=+0.001$、$b_3=-0.001$。
  - 注意 $E_1$ 偏差 $+1$ 与 $E_2$ 偏差 $+2$ 走相同步长 $+0.001$，直接验证 sign 丢失幅度信息。
- 第二轮（新 bias）：所有 token 的 top-2 不变。结论：单步 $\gamma=0.001$ 不足以改写选择。
- 对照（$\gamma=0.05$）：更新后 $b=(-0.05,+0.05,+0.05,-0.05)$，逐 token 检查 top-2 仍全部不变——展示要改写路由需 bias 累积到能跨越分数间距。

## 事实核对

- C1（路由坍塌）：arXiv:2408.15664 §1，引用 Shazeer 2017。
- C2（辅助损失两难）：arXiv:2408.15664 §1 与 Table 2（baseline $\alpha=0.001$）。
- C3（bias 只做选择不进权重）：DeepSeek-V3 报告 Eq.16 与原始论文 Eq.1-2。
- C4（sign 规则式更新）：原始论文 Algorithm 1、§3.2；DeepSeek-V3 报告 §2.2 文字描述。
- C5（sign vs 幅度变体对照）：原始论文 Table 3。补折叠块表数据已重新对照原文：
  - sign, u=0.001：PPL 9.50, MaxVio 0.044
  - magnitude, u=0.01：PPL 9.53, MaxVio 0.028
  - magnitude, u=0.001：PPL 9.51, MaxVio 0.036
  - magnitude, u=0.0001：PPL 9.51, MaxVio 0.040
- C6（互补 sequence-wise loss）：DeepSeek-V3 报告 Eq.17-19、§2.2 文字描述。
- N1（DeepSeek-V3 规模）：与 `moe-serving/` 已有数字一致。
- N2（$\gamma$ 调度）：DeepSeek-V3 报告 §2.2 明确：前 14.3T 用 0.001，最后 500B 设 0，总 14.8T。
- N3（$\alpha=0.0001$）：DeepSeek-V3 报告 §2.2。
- N4（原始论文 1B/3B、100B/200B token、u=0.001）：arXiv:2408.15664 §4.1、Table 2。
- N5（变体对照）：arXiv:2408.15664 Table 3（已重新核对，修正了乘法 bias 行——该数据在 Table 4，不在 Table 3）。

## 写作偏差

- 偏差：初稿变体实验对照表曾错误列入"乘法 bias"行（实际 Table 4 单独呈现），且 PPL/MaxVio 数字臆造。已修正：折叠块只列 Table 3 的加法 sign 与加法幅度变体，并注明乘法变体在 Table 4 不展开。修正后重新对照 WebFetch 提取的论文 Table 3 原文，数据一致。
- 偏差：初稿 S4 文本中曾出现 "tdtd" HTML 拼写错误，已修正。
- 偏差：初稿 S4 $\gamma=0.05$ 对照段落叙述混乱（"等等，重排"等口语），已重写为逐 token 列出 top-2 结果的清晰文本。
- 偏差：初稿 overview.html 与 index.html 末尾 `<!-- @content -->` 注释残留触发 validate.py 失败，已删除注释（保留一行普通说明）。
- 其余无偏差：章节、学习目标、前置知识、贯穿例子、误解和边界、过渡均按 outline.md 落实。
