# Chunked Prefill（chunked-prefill）初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成并按其写作（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节结构：6 章（generation stall / 切块机制与代价 / 搭车与 stall-free / token budget 权衡 / CPP / 边界与相邻工作）+ 来源与范围说明 ✓
  - 学习目标：5 题各配解答折叠块，答案指明论证章节 ✓
  - 前置知识：moe-serving、causal-mask、gpu-execution-model、model-parallelism 链接就位；beyond-buzz-disaggregation 链接指向同批生成页面 ✓
  - 贯穿示例：8000 token prompt / 64 decode 贯穿 1、3 章；4096→8/4 块重复读手算（第 2 章）；τ=512 迭代构成（第 3 章）✓
  - 误解与边界：误解 1（第 6 章）、误解 2（第 2 章数据流小节+本章问题）、误解 3（第 3 章"不是零"）✓
  - 衔接：各章末过渡句就位 ✓
- 代码运行：无可运行代码（本页为机制概念页，大纲未分配代码材料）
- 原图：本页无原图（机制图自绘）
- 机械检查：`python3 .dojo/scripts/validate.py wiki/chunked-prefill/index.html` → validation ok；overview.html → validation ok
- 公式渲染与交互：headless Chrome 探针实测——KaTeX 渲染 44 处；本页 SVG 标签全部为纯文字（<text>，无数学含义），foreignObject 0 个、无重叠
- 写作偏差：无。组装时修正一处笔误（"piggybacked co-cluded"→"piggybacked co-located"）。
- 来源冲突记录：KV 重复读计数 Sarathi 原文为 N 次、Sarathi-Serve 为 N-1 次（是否计入首次读取），采用 Sarathi-Serve 计数并在来源说明中记录。
