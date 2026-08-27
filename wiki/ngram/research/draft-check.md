# N-gram 初稿检查

- 输入版本：scope / evidence / outline / glossary 均为 2026-08-27 定稿
- 大纲落实：四章（链式法则→计数→稀疏→哈希查表）、4 个学习目标、贯穿示例（I am Sam 语料）、三处前置链接（pretraining/cross-entropy/数据流页）、误解三条、来源说明——逐项落实
- 目标覆盖检查：Q1→第 1 章、Q2→第 2 章、Q3→第 3 章、Q4→第 4 章，核心问题与本章问题均配解答折叠块
- 代码运行：ngram_demo.py（/usr/bin/python3）退出码 0；I am Sam 概率与教材逐项一致（2/3、1/3、1/2），整句 0.111111，碰撞率 0%/79.2%——与页面代码块预期输出一致
- 机械检查：validate.py 通过（index 与 overview）
- 公式渲染与交互：headless Chrome 实测 97 个 KaTeX 节点、TOC 15 条、无占位符、站内链接 4 个有效
- 写作偏差：第 3 章加一平滑示例从教材 want 行计数改为纯构造数字（避免引入未核实的行合计），已标注构造示例
