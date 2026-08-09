# EAGLE-3 投机解码 draft 模型初稿检查

## 输入版本

- `research/scope.md`：完成。包含概念含义、5 个学习目标、内容分级、前置知识映射、不展开内容、6 条常见误解与适用边界。
- `research/evidence.md`：完成。10 条 C 论断（C1-C10）、5 条 F 公式（F1-F5）、5 条 N 数字（N1-N5），每条均带来源定位与置信状态。
- `research/outline.md`：完成。6 个正文章节 + 文末来源说明；每章教学问题、对应范围、正文要点、讲解材料职责、前置知识安排、完成检查、过渡齐备。
- `research/glossary.md`：完成。登记全文术语、缩写、符号 35 项。

## 大纲落实

- S1 章节标题「为什么独立小模型 draft 有两难——EAGLE 的思路转向」（id=`why-new-draft`）：✓ 落实。包含传统独立小模型两难、EAGLE 思路转向、传统 vs EAGLE 对照表、EAGLE 不改变投机解码框架的边界。
- S2 章节标题「EAGLE-1 的核心机制——在 feature 空间做自回归」（id=`eagle-1-feature-ar`）：✓ 落实。包含两个关键观察、time-shifted token、EAGLE-1 ASCII 图示、EAGLE-1 性能数字。
- S3 章节标题「EAGLE-3 的两项架构改变——直接 token 预测 + 多层特征融合」（id=`eagle-3-changes`）：✓ 落实。包含两项改变各一小节、EAGLE-3 架构 ASCII 图示、EAGLE-1 vs EAGLE-3 对照表、性能数字。
- S4 章节标题「推理时的自回归 draft——单层 decoder 如何生成 γ 个 draft token」（id=`inference-pipeline`）：✓ 落实。包含 prefill、Step 1、Step 2+ 三小节、推理 pipeline ASCII 图示、贯穿手算例子（"How can I" + γ=3）、完整手算折叠块、伪代码折叠块。
- S5 章节标题「训练 draft 模型——training-time test 与接受率损失」（id=`training-ttt-loss`）：✓ 落实。包含 TTT 核心思想、TTT 因果 mask ASCII 图示、标准训练 vs TTT 对照表、L_E3 与 L_LK 公式、LK vs KL 折叠推导。
- S6 章节标题「工程实例——K3 的 EAGLE-3 部署与边界」（id=`k3-engineering-boundaries`）：✓ 落实。包含 MTP 初始化、三层 feature 来源与 W_E3、QAT 配置、K3 配置对照表、边界与不展开列表。
- 文末「来源与教学说明」（id=`sources-and-teaching-notes`）：✓ 落实。包含核心论断与来源（C1-C10）、核心公式与来源（F1-F5）、外部数字与实验条件（N1-N5）、教学示例、教学解释与类比边界、教学简化及其限制。

### 学习目标闭环

- Q1（EAGLE 系列如何复用 target 隐藏状态）：✓ 由 S1 正文完整回答（传统两难 + EAGLE 思路 + 对照表 + lossless 边界）。
- Q2（EAGLE-3 两项架构改变）：✓ 由 S3 正文完整回答（改变 1 直接 token 预测 + 改变 2 多层融合 + 对照表）。
- Q3（推理 pipeline + 自替代）：✓ 由 S4 正文完整回答（三步时序 + ASCII 图示 + 手算例子 + 伪代码）。
- Q4（TTT + LK loss vs KL）：✓ 由 S5 正文完整回答（TTT 思想 + 因果 mask + 标准训练 vs TTT 对照 + 两版损失 + LK vs KL 推导）。
- Q5（K3 MTP 初始化）：✓ 由 S6 正文完整回答（MTP 初始化 + W_E3 + 三层 feature + QAT + 配置表）。

### 前置知识引用

- speculative-decoding：✓ 在 S1 正文首次引用时给出链接 `../../wiki/speculative-decoding/index.html`，S5 再次引用接受率 α 公式来源。
- mxfp4-qat：✓ 在 S6 QAT 配置小节给出链接 `../../wiki/mxfp4-qat/index.html`。

### 贯穿例子

- "How can I" + γ=3 + k=4 + 词表 5 token：✓ 在 S4 展开。包含设定、Step 1/2/3 主干计算、完整三步手算折叠块、偏差对照（‖g_I - a_I‖_2 ≈ 0.380）。

### 误解和边界

- scope.md 列出 6 条常见误解，分别在 S1（EAGLE 不改框架、draft 不是独立小模型）、S2（EAGLE-1 输出是 feature 不是 token）、S3（直接 token 预测与 EAGLE-1 有区别）、S4（自替代引入噪声）、S5（TTT 不是多步训练、LK loss 不是因为 KL 算不出来）处理。✓
- 适用边界：S6 末尾的边界与不展开列表完整覆盖（解决/不解决/不改变/不展开 4 项）。✓

### 过渡

- S1→S2：「具体怎么复用？EAGLE-1 给出第一个答案：在 feature 空间做自回归。下一章讲 EAGLE-1 的具体机制。」✓
- S2→S3：「EAGLE-3 发现扩大训练数据对 EAGLE-1 提升有限——这个瓶颈源于『feature 回归任务占用了 draft 容量』。下一章讲 EAGLE-3 的两项架构改变如何突破这个瓶颈。」✓
- S3→S4：「架构讲完了——但推理时单层 decoder 怎么自回归生成 γ 个 token？为什么后续步骤要用自己的输出替代 target feature？下一章用贯穿例子手算走一遍。」✓
- S4→S5：「推理时 draft 必须自替代、噪声会累积——训练时如何让 draft 在自替代噪声下仍输出有效分布？下一章讲 training-time test 与损失函数。」✓
- S5→S6：「训练机制讲完了——最后一章把所有概念落到 K3 的工程实例上，并说明 EAGLE-3 的边界。」✓

## 代码运行

页面内代码块清单：

1. S2 ASCII 图示（`<pre class="diagram">`）：EAGLE-1 draft 模型结构图。非可运行代码，图示用途。
2. S3 ASCII 图示：EAGLE-3 draft 模型结构图。非可运行代码，图示用途。
3. S4 ASCII 图示：推理 pipeline 三步时序图。非可运行代码，图示用途。
4. S4 伪代码折叠块（`language-text`）：EAGLE-3 推理 pipeline 形式化。非可运行代码（明确标记「以下是伪代码，不是 Python」）。
5. S5 ASCII 图示：TTT 因果 mask 示意图。非可运行代码，图示用途。

页面无可运行代码块。

补充：S4 手算例子的数字（$a_I$、$q_1$、$a_{do}$、$q_2$、$a_{it}$、$q_3$ 与 ‖g_I - a_I‖_2 ≈ 0.380）在写作过程中用独立 Python 脚本验证过，输出与页面描述一致。验证脚本不在页面内，仅作为写作过程记录。脚本输出：

```
=== Step 1: input = (g_can, e_I) ===
pre-activation (W_a @ input) = [0.08, 0.45, 0.14, 0.13]
a_I = tanh(pre) = [0.0798, 0.4219, 0.1391, 0.1293]
logits = W_lm @ a_I = [0.1597, 0.8438, 0.2782, 0.2585, 0.385]
q_1 = softmax(logits) = [0.1547, 0.3066, 0.1742, 0.1708, 0.1938]
argmax = do (prob 0.3066)

=== Step 2: input = (a_I (替代 g_I), e_do) ===
a_do = [0.0921, 0.0837, 0.3175, 0.0397]
q_2 = [0.1805, 0.1775, 0.2834, 0.1626, 0.196]
argmax = it (prob 0.2834)

=== Step 3: input = (a_do (替代 g_do), e_it) ===
a_it = [0.0259, 0.1033, 0.0224, 0.4928]
q_3 = [0.1426, 0.1664, 0.1416, 0.3627, 0.1868]
argmax = now (prob 0.3627)

=== Bias check: g_I_real vs a_I ===
L2 distance = 0.3795

Draft sequence: ['do', 'it', 'now']
```

页面正文与折叠块中的数字与此输出一致（小数点 3-4 位四舍五入）。

## 机械检查

命令与结果：

```
$ python3 .dojo/scripts/validate.py wiki/eagle-speculative/index.html
validation ok: wiki/eagle-speculative/index.html
exit code: 0

$ python3 .dojo/scripts/validate.py wiki/eagle-speculative/overview.html
validation ok: wiki/eagle-speculative/overview.html
exit code: 0
```

补充静态检查（独立脚本，非 validate.py）：

- 占位符 `【...】`：0 个残留 ✓
- 模板标记 `@content`、`@component`、`TODO`、`TBD`：0 个残留 ✓
- 重复 id：0 个（32 个 id 全部唯一）✓
- 同页锚点 `#xxx`：全部指向存在 id ✓
- 本地资源引用：全部存在（`../../libs/katex.min.css`、`../../libs/katex.min.js`、`../../libs/auto-render.min.js`、`../../libs/prism-primer-light.css`、`../../libs/prism-primer-dark.css`、`../../libs/prism.min.js`、`../../libs/prism-python.min.js`、`../../index.html`、`overview.html`、`index.html`、`../../wiki/speculative-decoding/index.html`、`../../wiki/mxfp4-qat/index.html`）✓
- KaTeX 公式分隔符：11 个 `$$...$$` display 公式 + 236 个 `$...$` inline 公式，全部成对闭合 ✓（注：script 标签内 JavaScript 模板字面量 `${minutes}` 不是公式分隔符，KaTeX auto-render 默认跳过 `<script>` 内容）

## 公式渲染与交互

公式渲染：未在浏览器中实际打开页面（本环境无浏览器）。已通过静态语法检查：

- 所有 `$$...$$` 与 `$...$` 分隔符成对闭合。
- 关键公式语法检查（手工对照 KaTeX 支持的 LaTeX 命令）：
  - `\mathbb{R}`、`\mathbb{R}^{k \times 3k}`、`\sum`、`\min`、`\max`、`\log`、`\tanh`、`\mathrm{softmax}`、`\text{...}`、`\alpha`、`\gamma`、`\in`、`\times`、`\cdot`、`\mid`、`\|`、`\sqrt{...}`、`\frac{...}{...}`、`\begin{bmatrix}...\end{bmatrix}`、`\ldots`、`\langle`、`\rangle` 等均为 KaTeX 标准支持的命令。
  - 矩阵 `\begin{bmatrix}...\end{bmatrix}` 用于 S4 的 $W_a$ 与 $W_{\text{lm}}$ 显示，KaTeX 支持。

交互：页面使用模板自带脚本（进度条、暗/亮模式切换、侧边目录、章节折叠、j/k 快捷键、代码块复制按钮、返回顶部）。这些脚本已在模板中验证过、本页未修改 script 内容。

## 写作偏差

无写作偏差。

- 章节标题、顺序、教学任务均按 outline.md 落实，未自行增删或重排核心章节。
- 5 个学习目标全部由正文章节完整回答，无折叠块独占。
- 前置知识映射未改变（仅引用 speculative-decoding 与 mxfp4-qat）。
- 贯穿例子固定为 "How can I" + γ=3，未更换。
- 误解和边界按 scope.md 处理位置落实。
- 来源事实全部附 C/F/N 编号，与 evidence.md 一致。
- 教学构造（手算例子）与教学解释（TTT 类比）均已标记，失效边界已列出。
- 折叠块全部收起时，正文仍能回答全部 5 个学习目标（手算例子完整版在折叠块，但正文已展示主干计算与结论；TTT 因果 mask 细节在折叠块，但正文已说明核心思想；LK vs KL 推导在折叠块，但正文已说明核心原因）。
