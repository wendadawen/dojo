# 模型并行（model-parallelism）初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成并按其写作（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节结构：4 章（TP 切分 / PP 气泡 / 通信代价与拓扑 / 取舍）+ 来源与范围说明 ✓
  - 核心问题：4 题各配解答折叠块，答案指明论证章节 ✓
  - 前置知识：standard-attention、gpu-communication、moe-serving、mla 链接就位；chunked-prefill 与 beyond-buzz-disaggregation 链接指向同批生成页面 ✓
  - 贯穿示例：4 维 FFN 两卡切分手算 + 4 stage×4 micro-batch 气泡图 + TP4×PP2 组合图 ✓
  - 误解与边界：误解 1（第 1 章术语处）、误解 2（第 2 章本章问题第 3 题）、误解 3（第 3 章本章问题第 1 题）✓
  - 评价章节：第 4 章取舍（标注分析性判断）✓
  - 过渡：各章末衔接句就位 ✓
- 代码运行：无可运行代码（本页为纯机制概念页，大纲未分配代码材料）
- 原图：本页无原图（概念页，机制图自绘）
- 机械检查：`python3 .dojo/scripts/validate.py wiki/model-parallelism/index.html` → validation ok；overview.html → validation ok
- 公式渲染与交互：headless Chrome 探针实测——KaTeX 渲染 163 处、SVG foreignObject 标签 5 个、标签重叠 0 处（第一次探针 katex=0 系探针文件放在 /tmp 导致相对路径失效，移到页面同目录后正常）
- 写作偏差：无。手算折叠块初稿用 GeLU 近似数值（触发 validate 的裸 ≈ 字符报错），改为符号化推导（记 $G(\cdot)$ 逐元素函数，等价性对任意逐元素函数成立），比原稿更严谨，属于局部修正。
