# DeepSeekMoE 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成，规划完成条件已满足；依据论文 Dai et al. 2024 (arXiv:2401.06066) 经 WebSearch + WebFetch 核对，未凭记忆。

## 大纲落实

- 章节：S1 传统 MoE 的两个毛病 / S2 细粒度专家分割 / S3 共享专家隔离 / S4 相同算力下的证据 / S5 影响与继承 + 文末来源与教学说明。与 outline.md 一致，未增删核心章节。
- 学习目标：Q1–Q4 写入 learning-goals 组件，与 scope.md 一致。
- 前置知识：moe-serving 在 S1（首次引用 top-K MoE 公式）、S2（FFN 参数量）、S3（shared/routed 基本区分）、S5（DeepSeek-V3 路由差异）引用；stable-latent-moe 在 S5 引用。链接路径 ../moe-serving/index.html、../stable-latent-moe/index.html，目标页面均存在。
- 贯穿例子：4 专家 top-1、m=2 在 S2 首次出现（参数量 4→4、计算量 1→1、组合数 4→28），在 S3 推进（加 K_s=1 → 路由 7、激活路由 1、总激活 2、参数量 4、计算量 1）。单一例子覆盖 S2/S3 核心计算。
- 误解和边界：M1（分割不省计算量）在 S2 计算量守恒段讲清；M2（共享专家非路由专家）在 S3 公式与"为什么不参与路由"段讲清；M3（DSM≠DS-V3）在 S5 讲清；M4（组合数≠性能）在 S4 callout 讲清。
- 过渡：S1→S2（"根是专家太粗，第一把刀砍根"）、S2→S3（"通用知识被重复学还没解决"）、S3→S4（"听起来合理，要看实验"）、S4→S5（"看影响"）。每章末有过渡段。

## 学习目标闭环

- Q1（知识混合/冗余成因与对专门化的妨碍）：S1 正文完整回答。折叠块全收起仍可回答。
- Q2（细粒度分割守恒 + 手算 4 专家 m=2）：S2 正文 + 对照表格 + 手算 + 可运行代码完整回答。折叠块全收起仍可回答。
- Q3（共享专家分工 + 完整公式）：S3 正文 + 公式 F5/F6 + 贯穿手算推进 + ASCII 图示完整回答。折叠块全收起仍可回答。
- Q4（2B/16B/145B 性能数字 + 计算量恒定前提）：S4 正文 + 对照表格完整回答。折叠块全收起仍可回答。

## 代码运行

- 代码块（S2 折叠块，language-python）：运行命令 `python3 /tmp/page_code2.py`（从页面提取的代码原样保存），退出码 0。实际输出与页面"预期输出"块逐字一致，包括三段：教学例子（4→28，守恒 True）、论文例子（120→4426165368，守恒 True）、完整 DeepSeekMoE 加 K_s=1（路由 7、激活路由 1、总激活 2、守恒 True）。代码含全部导入（import math）与最小输入，可独立运行。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/deepseek-moe/index.html`
- 结果：`validation ok: wiki/deepseek-moe/index.html`，退出码 0。
- 同命令对 overview.html：`validation ok: wiki/deepseek-moe/overview.html`，退出码 0。
- 占位符检查：index.html 与 overview.html 中 `【` 计数均为 0，无模板占位符残留；无 @content/@component/TODO/TBD 标记残留。

## 公式渲染与交互

- KaTeX 资源：../../libs/katex.min.css、katex.min.js、auto-render.min.js 均存在于仓库 libs/ 目录；分隔符配置为 $...$（行内）与 $$...$$（块级），throwOnError:false。
- Prism 资源：../../libs/prism-primer-light.css、prism-primer-dark.css、prism.min.js、prism-python.min.js 均存在。
- 公式：F3/F4（S2）、F5/F6（S3）使用 $$...$$ 块级；行内符号如 $m$、$K_s$、$\binom{8}{2}$ 使用 $...$。符号与 glossary.md 一致。
- details 折叠块：S2 可运行代码块（code-details）、S2 组合数补充、S3 负载均衡补充——结构均为 summary + 内容，summary 写明补充哪个结论。
- 目录锚点：所有 h2 均有显式 id（s1-two-flaws、s2-fine-grained-segmentation、s3-shared-expert-isolation、s4-evidence、s5-inheritance、sources-and-teaching-notes），h3 亦有 id。
- 本次为生成环境，已做资源路径与 HTML 结构层面验证；实际浏览器渲染（KaTeX 公式呈现、折叠交互、目录高亮）待 check 阶段在浏览器中复验。

## 写作偏差

- 无。按 outline.md 落实，未增删核心章节、未更换贯穿例子、未改变前置知识映射、未把正文必要内容移入折叠块。S3 折叠块仅放负载均衡的"一句补充"，核心公式 F5/F6 与守恒推导均在正文。
