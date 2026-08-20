# KV cache 初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节：1 注意力为什么需要缓存 / 2 prefill 与 decode / 3 缓存有多大 / 4 为什么显存成为瓶颈 / 来源与范围说明 ✓
  - 核心问题：4 题（Q1 缓存理由 / Q2 两阶段 / Q3 大小公式 / Q4 显存瓶颈），每题配解答折叠块 ✓
  - 前置知识：standard-attention 链接在引言与第 1 章首次引用 ✓；paged-attention / prefix-caching / strata 链接在第 4 章过渡处（后三者本任务内生成）
  - 贯穿示例：4 token 序列（第 1、2 章）→ Llama-3.1-8B 20,000 token 手册（第 2–4 章）✓
  - 误解与边界：第 4 章处理三个误解（固定大小/窗口≠容量/缓存不省存储）✓
  - 过渡：每章末尾衔接 ✓
- 目标覆盖检查：Q1→第 1 章 ✓；Q2→第 2 章 ✓；Q3→第 3 章 ✓；Q4→第 4 章 ✓；全部由正文完整回答，无折叠块独占
- 代码运行：无可运行代码（机制可手算，未安排代码组件）
- 机械检查：`python3 .dojo/scripts/validate.py wiki/kv-cache/index.html`——剩余 2 项 broken local reference（../paged-attention/index.html、../strata/index.html），均为本任务后续生成的页面，待其生成后复验；其余检查通过。`validate.py wiki/kv-cache/overview.html` 通过
- 公式渲染与交互：headless Chrome 探针实测——KaTeX 渲染 140 处、正文零残留 `$`、h2=6、details=15；修复过一处 `$H_{\text{kv}}<H_q$` 中裸 `<` 破坏定界符的问题（已转义为 `&lt;`）
- 写作偏差：无。中途发现 overview 模板占位符与 paper 模板不同（概念版为【概念名】等），已按 concept 模板重新替换
