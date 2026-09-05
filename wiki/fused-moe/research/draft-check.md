# FusedMoE 初稿检查

## 输入版本
- scope.md：内容范围与学习目标已确定
- evidence.md：C1–C16、F1–F3、N1–N4 已对应到 vLLM v0.28.0 源码行号与 arXiv:2211.15841 摘要
- outline.md：五章大纲已确定，章节问题与答案要点已同步
- glossary.md：术语与符号登记完毕

## 大纲落实
- 章节：5 章 + 来源与范围说明，全部就位
- 学习目标：5 个核心问题，每个有完整解答折叠块
- 前置知识：第 1 章引用 moe-serving、gpu-execution-model；第 4 章引用 swiglu；第 5 章引用 model-parallelism、eplb、moe-serving
- 贯穿示例：构造示例 A（4 token、4 专家、top-2、block-2）覆盖第 2/3/4 章
- 误解和边界：4 条误解块（页面开头红色边框）+ 5 章各自边界说明
- 过渡：第 1→2 章说明问题转组织方式；第 2→3 章说明数组到内核；第 3→4 章说明索引到完整前向；第 4→5 章说明机制到边界

## 目标覆盖检查
- Q1 朴素循环浪费：由第 1 章完整回答（三个来源各一段）
- Q2 三数组语义与构造：由第 2 章完整回答（手算 + SVG 图 + 折叠块补充）
- Q3 内核索引：由第 3 章完整回答（公式 + 块 1 手算 + 掩码三处）
- Q4 完整数据流：由第 4 章完整回答（流程图 + 形状 + t0 手算 + 代码验证）
- Q5 职责边界：由第 5 章完整回答（组成 + 边界表 + EP 配合 + 调优 + 极小批量路径）

## 代码运行
- 页面代码块：从 HTML 提取、unescape 后实跑，输出与页面预期输出逐字符一致（误差 < 1e-9）

## 机械检查
- `.dojo/scripts/validate.py wiki/fused-moe/index.html` → ok
- `.dojo/scripts/validate.py wiki/fused-moe/overview.html` → ok

## 公式渲染与交互
- KaTeX 渲染 382 处，0 错误（headless Chrome 实测）
- display 公式 4 处（F1、F2、C[s1]/C[s2] 行、F3）
- SVG 条带图：12 个槽位文本全部完整落在单个矩形内，0 跨界
- 22 个 details 全部有 summary，前缀全部为「解答：」「补充：」「展开：」「代码：」
- body 横向溢出 0

## 写作偏差
- 无