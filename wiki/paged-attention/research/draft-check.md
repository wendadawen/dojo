# PagedAttention 初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节：1 连续预留的三类浪费 / 2 分页机制 / 3 页间不连续 / 4 页大小怎么选 / 来源与范围说明 ✓
  - 核心问题：4 题（三类浪费 / 分页机制 / 传输缺点 / 页大小权衡），每题配解答折叠块 ✓
  - 前置知识：kv-cache 链接（开篇与第 3 章）、prefix-caching 与 strata 链接（第 2、4 章相应位置）✓
  - 贯穿示例：请求 A（上限 2048/实际 100）与 B（上限 512/实际 500）→ 页 16 重算 → 20,000 token 手册（与 kv-cache/strata 贯穿示例对齐）✓
  - 误解与边界：第 2 章（非新注意力算法）、第 3 章（利用率≠容量）✓
- 目标覆盖检查：Q1→第 1 章 ✓；Q2→第 2 章 ✓；Q3→第 3 章 ✓；Q4→第 4 章 ✓
- 代码运行：无可运行代码
- 机械检查：`validate.py wiki/paged-attention/index.html`——剩余 broken reference 仅 ../strata/index.html ×3（本任务后续生成）；overview 通过
- 公式渲染与交互：headless Chrome 探针——KaTeX 渲染 11 处、零残留 `$`；无 SVG（未用内联 SVG，结构图为 HTML 组件）
- 写作偏差：无
