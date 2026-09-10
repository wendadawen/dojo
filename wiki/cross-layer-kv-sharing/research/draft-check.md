# 跨层 KV 复用初稿检查

- 输入版本：scope / evidence / outline / glossary 均已定稿
- 页面：wiki/cross-layer-kv-sharing/index.html（完整说明）、wiki/cross-layer-kv-sharing/overview.html（概览）

## 大纲落实

- 页面开头：具体问题（每层各存一份全局 KV 的代价）+ 范围 callout + 4 条核心问题 + 4 条常见误解 ✓
- 章节：4 章（复用对象 / 省下多少 / 两级复用 / 前提与边界）✓
- 学习目标：4 条，每条由对应章节完整回答 ✓
- 前置知识：standard-attention、kv-cache、mla、dsa 均已有页面并给出链接 ✓
- 贯穿示例：40 层、1M token 的份数与分组在四章中依次推进 ✓
- 误解与边界：4 条误解在页面开头；边界（短序列收益有限）写在第 2、4 章 ✓
- 过渡：第 2、3、4 章开头各有一句衔接 ✓

## 目标覆盖检查

- Q1 复用对象：第 1 章完整回答（YOCO 结构 + 公式 F1 + 结构图）✓
- Q2 省多少：第 2 章完整回答（复杂度对照 + "once" 的准确含义 + MLA 分工）✓
- Q3 两级复用：第 3 章完整回答（三模式对照表 + 结构图）✓
- Q4 前提与边界：第 4 章完整回答（分组表 + 运行期实测 + 代价）✓
- 无目标被折叠块独占 ✓

## 代码运行

- 无可运行代码（本页的实测证据是运行期 cache 归属，见 wiki/deepseek-v4-1/research/verify_csa2_modes.py，输出已写入正文与来源说明）

## 机械检查
- `python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/index.html` -> validation ok
- `python3 .dojo/scripts/validate.py wiki/cross-layer-kv-sharing/overview.html` -> validation ok
- 无残留占位符（【）、无组件标记（@copy-start / @content）

## 公式渲染与交互（headless Chrome 实测，探针把结果写 document.title）
- 命令：`Google Chrome --headless --dump-dom file:///.../wiki/cross-layer-kv-sharing/_probe.html`（探针注入 `</head>` 之前）
- 结果：katex=79 details=14 summaries=14 questions=5 h2=7 svgtext=0 fo=0 overlap=0 placeholder=0
- 说明：details 与 summary 数量相等（每个问题都有解答折叠块）；questions=5 表示 1 个「核心问题」列表 + 4 个「本章问题」列表；overlap=0 表示图内标签无重叠；placeholder=0 表示页面正文无残留占位符

## 写作偏差
- 无。全部章节、学习目标、前置知识引用、材料与正文/折叠分工按 outline.md 落实；未新增或删减章节。
