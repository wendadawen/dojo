# Sherry 稀疏三值量化初稿检查

- 输入版本：scope / evidence / outline / glossary 均已定稿（2026-09-01）
- 大纲落实：6 章 + 来源与范围说明（落实）；5 个核心问题各配解答折叠块（落实）；前置概念最小含义在开头给出、MIX-STQ1_0 链接到 mixed-precision-quant 页（落实）；贯穿示例 x=(0.6,-0.1,0.5,-0.7) 在第 2/3/4 章推进（落实）；常见误解 4 条在开头 misconceptions（落实）；章间过渡按依赖写（落实）
- 目标覆盖检查：Q1→第 1、2 章（落实）；Q2→第 2 章（落实）；Q3→第 3 章（落实）；Q4→第 4 章（落实）；Q5→第 5 章（落实）；均未被折叠块独占
- 代码运行：`python3 research/quant_demo.py`，退出码 0；输出与页面「预期输出」逐行一致（amax 117.5101 / WLS 19.6010 / WLS+imatrix 17.7533；贯穿示例 SSD 0.0300 vs amax 0.0600；42B、1.3125 bpw）
- 机械检查：`python3 .dojo/scripts/validate.py wiki/sherry-ternary-quant/index.html`——报 1 项：broken local reference ../mixed-precision-quant/index.html（该页面属本批次任务 2，生成后需重跑校验）；其余通过
- 公式渲染与交互：无头 Chrome 实测（同目录注入探针）——136 个 .katex 节点渲染、正文无残留 $ 定界符、17 个 details、28 条目录；页面无 SVG 图（两张图均为 HTML 结构图），无标签重叠问题
- 写作偏差：无

## 质检轮次补充（2026-09-01）

- 第 1 轮：0 阻断 / 5 重要 / 7 轻微，全部修复（见 review-1.md）；修复后 validate 通过。
- 第 2 轮：0 阻断 / 2 重要 / 4 轻微，全部修复（见 review-2.md）；修复后 validate 通过、渲染探针 159 katex/0 残留 $。
- 第 3 轮：0 阻断 / 0 重要 / 6 轻微，全部修复（见 review-3.md 含发布记录）；修复后 validate 通过。
- 最终状态：三轮独立审查完成，阻断与重要问题全部关闭，页面发布。
