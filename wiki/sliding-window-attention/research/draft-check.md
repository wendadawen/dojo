# 滑动窗口注意力初稿检查

- 输入版本：scope / evidence / outline / glossary 均已定稿（research/ 下四个文件）
- 页面：wiki/sliding-window-attention/index.html（完整说明）、wiki/sliding-window-attention/overview.html（概览）

## 大纲落实

- 页面开头：具体问题（1M 请求下每个 token 都要回头看的代价）+ 范围 callout + 4 条核心问题 + 4 条常见误解 ✓
- 章节：4 章（可见集 / 代价 / 感受野 / 环形缓冲），编号连续，职责单一 ✓
- 学习目标：4 条，每条由对应章节完整回答 ✓
- 前置知识：standard-attention、causal-mask、kv-cache、rope 均已有页面并给出链接；attention-sink 由本流程生成后补链接 ✓
- 贯穿示例：位置 4999 的 query 在四章中依次推进（可见集 / 打分次数 / 感受野 / 槽位 7）✓
- 误解与边界：4 条误解放在页面开头；适用边界写在 scope 与正文（序列短于窗口时与全注意力等价）✓
- 过渡：第 2、3 章开头各有一句衔接 ✓

## 目标覆盖检查

- Q1 可见集：第 1 章完整回答（含对照表与 $-1$ 槽位补充）✓
- Q2 代价：第 2 章完整回答（打分次数表 + 64 KiB/2.50 MiB 账）✓
- Q3 感受野：第 3 章完整回答（公式 F1 + 5120 对 1M + 汇聚点边界）✓
- Q4 环形缓冲：第 4 章完整回答（槽位映射 + 两种索引 + 位置口径）✓
- 无目标被折叠块独占：四个目标的结论均在正文陈述 ✓

## 代码运行

- 代码块 1 个（第 4.1 节环形缓冲写入与索引）：`python3 ring_demo.py`，退出码 0，输出与页面「预期输出」逐字一致（W=8、序列长 20，末步槽位 [4,5,6,7,0,1,2,3] 映射回 [12..19]）✓

## 机械检查
- `python3 .dojo/scripts/validate.py wiki/sliding-window-attention/index.html` -> validation ok
- `python3 .dojo/scripts/validate.py wiki/sliding-window-attention/overview.html` -> validation ok
- 无残留占位符（【）、无组件标记（@copy-start / @content）

## 公式渲染与交互（headless Chrome 实测，探针把结果写 document.title）
- 命令：`Google Chrome --headless --dump-dom file:///.../wiki/sliding-window-attention/_probe.html`（探针注入 `</head>` 之前）
- 结果：katex=191 details=16 summaries=16 questions=5 h2=7 svgtext=14 fo=0 overlap=0 placeholder=0
- 说明：details 与 summary 数量相等（每个问题都有解答折叠块）；questions=5 表示 1 个「核心问题」列表 + 4 个「本章问题」列表；overlap=0 表示图内标签无重叠；placeholder=0 表示页面正文无残留占位符

## 写作偏差
- 无。全部章节、学习目标、前置知识引用、材料与正文/折叠分工按 outline.md 落实；未新增或删减章节。
