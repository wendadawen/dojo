# MXFP4 量化感知训练初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成且完成条件满足（歧义已裁定、5 个学习目标、核心论断全部已确认、大纲完整、术语表齐全）
- 大纲落实：
  - 页面开头（callout 钩子 + context-box + learning-goals + blockquote.meta + 前置引用）：已落实
  - S1 把 896 个 MoE 专家压到 4-bit 能省多少显存：已落实，含 N3 手算折叠块
  - S2 MXFP4 怎么编码一个权重：已落实，含块结构 ASCII 图、F1、E2M1 集合、有效位宽、E1 折叠块（块 1 无损 + 块 2 有损对照）
  - S3 QAT 前向/反向与 PTQ 对照：已落实，含 F2、E2 手算折叠块、可运行代码折叠块、STE 教学解释 callout、误解 2 与 4
  - S4 K3 贯穿 SFT+RL 与 mismatch：已落实，含 RL 最小框架、rollout/训练循环 ASCII 图、draft 一致性、误解 2 在 RL 语境再强调
  - S5 选择性量化与 config 证据：已落实，含对照表格、config.json 折叠块、误解 1 与 5
  - 文末来源与教学说明：已落实（核心论断/公式/外部数字/教学示例/教学解释/教学简化六小节齐全）
- 学习目标闭环：
  - Q1（专家为何首选、省多少）：S1 正文 + N3 折叠块完整回答（59.19B、118.38 GB→31.44 GB、3.76×）
  - Q2（MXFP4 编码、共享 scale 为何强）：S2 正文 + F1 + 块结构图 + E1 折叠块完整回答（E2M1 集合、4.25 bit、共享 scale 承担动态范围）
  - Q3（QAT 前向/反向、PTQ 差别）：S3 正文 + F2 + E2 折叠块 + 可运行代码完整回答（fake-quant 用 $\hat{w}$、STE 直通、PTQ 训练未模拟量化）
  - Q4（贯穿 SFT+RL、rollout/训练共享方案）：S4 正文 + 循环图完整回答（mismatch 来源、共享方案消除、draft 沿用）
  - Q5（量化/非量化组件划分）：S5 正文 + 对照表 + config 折叠块完整回答（专家 MXFP4、激活 MXFP8、非专家高精度、ignore 字段证据）
  - 全部目标由正文章节完整回答，折叠块收起时正文仍能回答（核心定义、公式用途、机制结论均在正文）
- 代码运行：
  - 折叠块"可运行代码：MXFP4 量化模拟与 QAT 一步"——运行命令 `python3`（代码内联于页面，逻辑与 /tmp/mxfp4_demo.py 一致），退出码 0
  - 实际输出与页面"预期输出"块逐行一致：block1 误差全零；block2 误差 [0.05, -0.025, -0.05, 0.0125]；w=0.8, w_hat=0.75, 前向偏差 0.0500；STE 传回梯度 1.0
  - N3 显存估算同代码验证：每专家 66.06M、共 59.19B、BF16 118.38 GB、MXFP4 31.44 GB、压缩 3.76x
- 机械检查：
  - 命令：`python3 .dojo/scripts/validate.py wiki/mxfp4-qat/index.html` → 退出码 0，输出 `validation ok`
  - 命令：`python3 .dojo/scripts/validate.py wiki/mxfp4-qat/overview.html` → 退出码 0，输出 `validation ok`
  - 无占位符【…】、无 @content/@component/TODO/TBD 标记残留、无重复 id、同页锚点均指向存在 id、本地引用全部存在（`../../index.html`、`overview.html`、`../moe-serving/index.html` 均存在）
  - KaTeX：display `$$` 共 9 对（18 个），成对；inline `$...$` 在 prose 中成对（唯一不成对的 `$` 出现在模板 `<script>` 内的 JS 正则 `/[\s#?？：]/g`，KaTeX auto-render 默认跳过 script 块，不影响渲染）
  - details 开/闭 5/5 配对；h2 共 6 个（S1–S5 + sources）
- 公式渲染与交互：
  - 本环境未实际打开浏览器做视觉检查（浏览器视觉检查属 check 阶段职责，由编排者安排）
  - 已做的静态检查：KaTeX 分隔符配对、Prism 代码块 language 标注（python/text/json）、details/code-details 结构、表格结构完整
  - STE 教学解释 callout 已标"教学解释"与失效边界
- 写作偏差：无。大纲全部章节、学习目标、前置引用、贯穿例子（896 专家主线 + E1/E2 局部例子）、误解（1/2/4/5）、边界、过渡均按 outline.md 落实；未新增范围外内容、未更换贯穿例子、未把正文必要内容移入折叠块

## 待 check 阶段处理

- 前置概念 quantization-basics：本页正文用占位提示（`<em>quantization-basics</em> 概念页——该页面尚未生成`），未生成、未递归（按编排者指示占位）
- 浏览器视觉检查、KaTeX 实际渲染效果、目录锚点跳转、折叠交互：由 check 阶段独立审查者处理
- content.json 与首页：本阶段不更新（按编排者指示，check 通过后才更新）
