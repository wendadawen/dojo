# 注意力汇聚点初稿检查

- 输入版本：scope / evidence / outline / glossary 均已定稿
- 页面：wiki/attention-sink/index.html（完整说明）、wiki/attention-sink/overview.html（概览）

## 大纲落实

- 页面开头：具体问题（删掉最早的几句话模型会崩）+ 范围 callout + 4 条核心问题 + 4 条常见误解 ✓
- 章节：4 章（现象 / 成因 / 两种做法 / 后果与边界）✓
- 学习目标：4 条，每条由对应章节完整回答 ✓
- 前置知识：standard-attention、kv-cache 已有页面并给出链接；sliding-window-attention 由本流程生成后链接 ✓
- 贯穿示例：1M 流式会话淘汰开头位置的后果在四章中依次推进 ✓
- 误解与边界：4 条误解在页面开头；能力边界（不扩展上下文窗口）写在第 4.3 节 ✓
- 过渡：第 2、3、4 章开头各有一句衔接 ✓

## 目标覆盖检查

- Q1 现象：第 1 章完整回答（含「分数高≠信息重要」对照表）✓
- Q2 成因：第 2 章完整回答（softmax 分母 + 为什么是初始位置）✓
- Q3 两种形态：第 3 章完整回答（对照表 + 解析解 + checkpoint 形状）✓
- Q4 缓存与推理含义：第 4 章完整回答（份额表 + 能力边界）✓
- 无目标被折叠块独占 ✓

## 代码运行

- 代码块 1 个（第 4.2 节份额计算）：`python3` 运行，退出码 0，输出 0.78% / 5.46% / 53.69% / 95.88%，与页面一致 ✓

## 机械检查
- `python3 .dojo/scripts/validate.py wiki/attention-sink/index.html` -> validation ok
- `python3 .dojo/scripts/validate.py wiki/attention-sink/overview.html` -> validation ok
- 无残留占位符（【）、无组件标记（@copy-start / @content）

## 公式渲染与交互（headless Chrome 实测，探针把结果写 document.title）
- 命令：`Google Chrome --headless --dump-dom file:///.../wiki/attention-sink/_probe.html`（探针注入 `</head>` 之前）
- 结果：katex=50 details=16 summaries=16 questions=5 h2=7 svgtext=0 fo=0 overlap=0 placeholder=0
- 说明：details 与 summary 数量相等（每个问题都有解答折叠块）；questions=5 表示 1 个「核心问题」列表 + 4 个「本章问题」列表；overlap=0 表示图内标签无重叠；placeholder=0 表示页面正文无残留占位符

## 写作偏差
- 无。全部章节、学习目标、前置知识引用、材料与正文/折叠分工按 outline.md 落实；未新增或删减章节。
