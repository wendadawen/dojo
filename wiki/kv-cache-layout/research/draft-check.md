# KV cache 布局（NHD/HND）初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均为 2026-09-03 完成稿；evidence.md 补录 C21（v0.13.0 旧实现）后未再变更范围
- 大纲落实：
  - 页面开头（场景引入 + meta + 引言 + 核心问题 4 条）已落实
  - 章节：第 1 章三节、第 2 章两节、第 3 章四节、第 4 章五节 + 来源与范围说明，与 outline.md 一致
  - 学习目标：Q1–Q4 分别由第 1–4 章完整回答
  - 前置知识：kv-cache / mqa-gqa / paged-attention 链接在首次依赖处给出；stride 在 4.1 正文内最小解释（按 scope 决定不递归生成）
  - 贯穿示例：p=4、H_kv=2、d=2 的 16 元素页贯穿图 1、图 2、2.2 节落位、4.1 节代码
  - 误解与边界：误解 1（HND 普遍更好）在 3.4；误解 2（切换=复制）在 4.1；误解 3（HDN）在 1.3；误解 4（碎片两层）在 1.3
  - 过渡：每章末尾均有衔接句
- 目标覆盖检查：Q1 由 1.1–1.3 回答；Q2 由 2.1–2.2 回答；Q3 由 3.1–3.4 回答；Q4 由 4.1–4.5 回答。核心问题 4 条与章节问题 12 条均配解答折叠块（渲染探针 details=17：16 解答 + 1 代码折叠块）
- 代码运行：4.1 节 as_strided 代码于 2026-09-03 运行（python envs/default，torch 2.13.0），页面预期输出与实际 stdout 逐行一致（物理顺序 0–15；token0 两头 [[0,1],[2,3]]；头0 整页 [0..7]；stride (4,2,1)/(8,2,1)）
- 机械检查：`python3 .dojo/scripts/validate.py wiki/kv-cache-layout/index.html` → validation ok；overview.html → validation ok
- 公式渲染与交互：headless Chrome 探针（页面同目录副本，相对路径 ../../libs/ 生效）：index 的 .katex 节点 198、SVG rect 66、SVG 文本 86、两两矩形重叠 0、badSummary 0；overview 的 .katex 节点 12。首次探针 katex=0 系副本放 /tmp 导致相对路径断裂，改放同目录后正常
- 编号双向对应：正文上标引用 C1–C21、F1–F3、N1–N4 全部出现，来源章节登记无缺漏、无孤儿
- 占位符与组件标记：无残留
- 写作偏差：无（未增删章节、未换示例、未移动正文必要内容入折叠块）
