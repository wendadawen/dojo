# PP 负载均衡初稿检查

- 输入版本：scope / evidence / outline / glossary 均为 2026-09-03 初版（wiki/pp-load-balancing/research/）
- 大纲落实：
  - 章节：5 章 + 来源与范围说明，标题与 outline.md 一致（1 两类等待与两个放大器；2 token 维度切分；3 时间对齐；4 batch 维度均衡；5 组合与边界）——已落实
  - 学习目标：Q1–Q5 各由对应章节完整回答，核心问题块 5 条每条带解答折叠块——已落实
  - 前置知识：开头引用段给出 model-parallelism / chunked-prefill / kv-cache / causal-mask 四个链接，均为已有页面——已落实
  - 贯穿示例：12 token / 3 stage，Ch1 空等（234）、Ch2 切分与最优（162→134）、Ch3 漂移（7.5→5.0）、Ch4 配平手算（#D=24、#P=768）——已落实
  - 误解与边界：M1–M5 织入正文（均分≠均衡、气泡非仅填充不足、动态 chunk 目的、throttling≠budget、DP 不能照搬）；5.2 汇总边界——已落实
  - 过渡：各章末尾有承上启下段——已落实
- 目标覆盖检查：Q1→第 1 章（两类依赖/两放大器/气泡公式）；Q2→第 2 章（因果性/前长后短/目标函数/CPP）；Q3→第 3 章（漂移累积/动态 chunk 方程与 smooth factor）；Q4→第 4 章（固定预算波动/解耦与全局反馈）；Q5→第 5 章（联合切分/边界）。每章末尾本章问题均带解答折叠块，独立成段——通过
- 代码运行：research 阶段用 /tmp/pplb_dp.py 运行（python3，退出码 0），输出四行与页面代码折叠块的预期输出逐字一致（134/(7,3,2)、102/单 token、162、102）——通过
- 机械检查：`python3 .dojo/scripts/validate.py wiki/pp-load-balancing/index.html` → validation ok；`python3 .dojo/scripts/validate.py wiki/pp-load-balancing/overview.html` → validation ok——通过
- 公式渲染与交互：无头 Chrome（headless + virtual-time-budget 8000）注入探针实测：index.html katex 节点 267、SVG 图 3 张（54 个 text）、details 23 个、SVG text 两两重叠 0 处；overview.html 无公式无图（概览规范不含推导与示例，属预期）——通过
- 写作偏差：无。层维度划分按用户指示排除，在 1.2、5.2 与范围说明中声明边界；gLLM×CPP 叠加无文献支撑，未写成论断
