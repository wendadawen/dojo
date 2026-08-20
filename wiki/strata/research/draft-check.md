# Strata 初稿检查

- 输入版本：scope / evidence / outline / glossary 均已完成（research/ 目录，2026-08-19）
- 大纲落实：
  - 章节：1 瓶颈定位（4 小节）/ 2 GPU-assisted I/O（2 小节）/ 3 布局解耦（3 小节）/ 4 Cache-aware 调度（4 小节）/ 5 实验验证（6 小节）/ 6 方法评价（3 小节）/ 来源与范围说明 ✓
  - 核心问题：5 题（瓶颈定位 / I/O 机制 / 调度三阶段 / 收益与条件 / 适用边界），每题配解答折叠块 ✓
  - 前置知识：kv-cache / paged-attention / prefix-caching / standard-attention / gpu-execution-model / gpu-communication 链接（首次出现处）✓
  - 贯穿示例：20,000 token 文档问答 + Llama-3.1-8B（128 KB/token、2.56 GB） + 3 客服请求 A/B/C；与 LooGLE avg 输入 21,613 同量级 ✓
  - 误解与边界：方法评价章开头 callout 声明、第 6 章三小节集中处理 ✓
- 目标覆盖检查：Q1→第 1 章 ✓；Q2→第 2、3 章 ✓；Q3→第 4 章 ✓；Q4→第 5 章 ✓；Q5→第 6 章 ✓
- 代码运行：无可运行代码
- 机械检查：`validate.py wiki/strata/index.html` 与 `validate.py wiki/strata/overview.html` 均通过
- 公式渲染与交互：headless Chrome 探针——KaTeX 渲染 58 处、零残留 `$`；13 张原图 base64 全部通过 PIL 解码验证（900×variable JPEG/PNG）；SVG 自绘 HiRadixTree 状态流转图 4 个标签与节点框无重叠（修复过 2 处压框）；Figure 7 在正文以自绘 SVG 替代，来源说明完整列出 14 张原图与对应位置
- 写作偏差：无