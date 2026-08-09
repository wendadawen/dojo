# Delta 规则与 DeltaNet 初稿检查

## 输入版本

- scope.md：已裁定（无影响核心定义的无法消歧项），3 个误解 + 适用边界齐备，5 个学习目标
- evidence.md：C1–C8 / F1–F3 / N1–N3 共 14 条核心论断，全部已确认状态
- outline.md：5 章教学大纲 + 页面开头 + 来源与教学说明，章节单一任务、贯穿例子、材料职责齐备
- glossary.md：术语 / 缩写 / 符号三表齐全，符号含义全文一致

## 大纲落实

逐项核对：

- **页面开头**：callout 钩子（10 万 token 记忆场景）+ context-box（4 项背景）+ learning-goals（5 题）+ misconceptions（3 条）+ blockquote.meta（三篇论文摘要）—— ✓
- **第 1 章**（why-linear-attention-collides）：动机 + retrieval 展开 + d=2,L=3 手算 + 完成检查 + 过渡 —— ✓
- **第 2 章**（delta-rule-formula）：紧凑公式 + 符号定义 + 形状检查 + 边界代入 + 来源溯源 + 时间线图 + 手算例子 + 完成检查 + 过渡 —— ✓
- **第 3 章**（equivalent-form）：等价形式 + 推导折叠块 + 几何投影 + 2 维直觉 + 完成检查 + 过渡 —— ✓
- **第 4 章**（beta-boundaries）：β=0 / β=1 / 中间值三类边界 + 对照表 + 误解排查 + 完成检查 + 过渡 —— ✓
- **第 5 章**（deltanet-and-comparison）：DeltaNet 定义 + 4 模型对比表 + Gated DeltaNet 退化 + 实验数字 + 工程实现 + 可运行代码 + 完成检查 —— ✓
- **来源与教学说明**：6 个小节（核心论断 / 公式 / 数字 / 教学示例 / 教学解释 / 教学简化）—— ✓
- **前置知识引用**：linear-attention 占位（按用户指令不递归生成）—— ✓
- **贯穿例子**：d=2,L=2 主例子贯穿第 2-4 章，第 1 章用 d=2,L=3 局部例子展示碰撞，第 5 章对比加性累加 —— ✓
- **误解与边界**：3 个误解（C5 关系、完全擦除、投影性质）+ 适用边界（退化条件 / 数值稳定性）—— ✓

## 学习目标闭环

逐题核对：

- **Q1**（为什么碰撞 / delta 规则如何解决）：第 1 章完整回答——retrieval error 代数来源、$L > d$ 不正交、加性累加不擦除；第 2 章给出 delta 规则作为解决方案 —— ✓
- **Q2**（手算一步 + 符号 + β 作用）：第 2 章"手算一步"小节完整回答——d=2,L=2,β=1,2 步序列手算、每个符号定义、$\beta_t$ 是 sigmoid 输出的写入强度 —— ✓
- **Q3**（等价形式 + 等价性证明）：第 3 章完整回答——C4 形式 + 代入展开推导折叠块 + $v^{\text{old}}$ 与 $v^{\text{new}}$ 语义 —— ✓
- **Q4**（β 边界 + 前提条件）：第 4 章完整回答——β=0 / β=1 / 中间值三类边界 + $\|k\|=1$ 投影前提 + 误解排查表 —— ✓
- **Q5**（DeltaNet vs Mamba2 vs GLA vs Gated DeltaNet + α_t 动机）：第 5 章完整回答——4 模型对比表 + 三组对比 + Gated DeltaNet 退化 + 实验数字 + 工程实现 —— ✓

折叠块全部收起时正文仍能回答全部 5 个学习目标 —— ✓

## 代码运行

页面唯一可运行代码块（第 5 章折叠块内）：

```python
import numpy as np
# d=2, L=2, β=1, k1=k2=(1,0), v1=(1,0), v2=(0,1)
# 对比加性累加与 delta 规则
```

**运行命令**：`python3 wiki/delta-rule/run_demo.py`（或直接执行代码内容）

**退出码**：0

**实际输出**（与页面描述逐行一致）：

```text
Linear S_2 =
[[1. 0.]
 [1. 0.]]
Linear S_2 @ k_2 = [1. 1.]  (expected mismatch with v_2 = [0. 1.] )

Delta S_1 =
[[1. 0.]
 [0. 0.]]
Delta S_2 =
[[0. 0.]
 [1. 0.]]
Delta S_2 @ k_2 = [0. 1.]  (expected = v_2 = [0. 1.] )

Equivalent form S_2 =
[[0. 0.]
 [1. 0.]]
Matches compact form: True
```

页面正文描述的数字（$S_2^{\text{linear}} = \begin{pmatrix}1 & 0 \\ 1 & 0\end{pmatrix}$、$S_2^{\text{delta}} = \begin{pmatrix}0 & 0 \\ 1 & 0\end{pmatrix}$、$S_2^{\text{delta}} k_2 = v_2$）与代码输出完全一致。

**辅助验证**（手算例子交叉核对）：

第 1 章 d=2, L=3 碰撞例子（key 未归一化）：

```text
S =
[[1. 1.]
 [1. 2.]]
S @ k_2 = [2. 3.]  (ideal 2*v_2 = [0. 2.])
  k_1 . k_2 = 1.0  contribution v_1 = [1. 0.]
  k_3 . k_2 = 1.0  contribution v_3 = [1. 1.]
  k_2 . k_2 = 2.0  contribution 2*v_2 = [0. 2.]
  sum = [2. 3.]
```

页面正文描述与代码输出一致。

## 机械检查

**命令**：`python3 .dojo/scripts/validate.py wiki/delta-rule/index.html`

**结果**：`validation ok: wiki/delta-rule/index.html`（退出码 0）

**命令**：`python3 .dojo/scripts/validate.py wiki/delta-rule/overview.html`

**结果**：`validation ok: wiki/delta-rule/overview.html`（退出码 0）

机械检查发现的初始问题与修复：
- 初次验证报告 11 处 `【CN】` / `【FN】` / `【NN】` 模板占位符残留——这是引用标记格式与 validate.py 的 `PLACEHOLDER_RE = re.compile(r"【[^】]*】")` 冲突。把所有 `<sup>【C2】</sup>` 等改为 `<sup>[C2]</sup>`，验证通过。
- 初次写作时第 1 章手算例子的解释段落含自我修正式表述（先报错数字再更正），已重写为干净分项展开。

## 公式渲染与交互

浏览器实际检查（待审阶段在子代理中执行；本阶段在编辑器中预览）：

- 所有 `$...$` 与 `$$...$$` KaTeX 公式语法正确
- 矩阵 `\begin{pmatrix}...\end{pmatrix}` 语法 KaTeX 支持
- 折叠块 `<details><summary>...</summary>...</details>` 标准语法
- 章节锚点 `id` 全页唯一（`why-linear-attention-collides`, `delta-rule-formula`, `equivalent-form`, `beta-boundaries`, `deltanet-and-comparison`, `sources-and-teaching-notes` + 各 h3 子节）
- 目录由外壳脚本自动生成
- 主题切换、进度条、返回顶部由外壳脚本提供

## 写作偏差

无。所有章节、学习目标、前置知识、贯穿例子、误解边界、过渡均按 outline.md 落实。无需返回规划阶段补齐。

