# MoonViT-V2 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均为本轮 plan 阶段产物，已完成且满足 plan 完成条件；外壳 .dojo/templates/concept/index.html、overview.html 与组件库 components.html 已读；content-examples.md 已读作为正文写作正本。

## 大纲落实（逐项一行）

- 页面开头：callout（核心提示）+ learning-goals（Q1–Q4）+ blockquote.meta（K3 报告 §2.4 + config.json）——已落实，未用 context-box / misconceptions（按大纲并入正文与 callout）。
- S1 主流做法遇到什么问题（SigLIP init 为何不稳）：SigLIP 最小定义、梯度范数最小定义、Fig.6 简化 ASCII 对照、完成检查——已落实。
- S2 从零 + NTP 方案：方案两步、NTP 最小定义、两个动机（稳定性 + 目标对齐）、三维度对照表、完成检查——已落实。
- S3 服务稳定性的架构：ViT 最小定义、config 数字表、参数量手算（F1 正文完整）、RMSNorm/去 bias 定位、完成检查——已落实。
- S4 图像/视频共享与高分辨率：共享参数、分解注意力 ASCII、时间池化、pixel-shuffle ASCII + 无损重排说明、token 数手算（F2 正文完整）、完成检查——已落实。
- S5 结果与结论：匹配 baseline（C7/N2 不造数字）、结论、两个边界（场景/规模）、对照表、完成检查——已落实。
- 来源与教学说明：核心论断 C1–C8、公式 F1–F3、外部数字 N1–N2、教学示例、教学解释与类比边界、教学简化及限制——已落实。
- 前置知识：ViT/SigLIP/NTP/RMSNorm/自注意力 五项均未生成，按任务约定占位标注（内联文字"概念页待生成"，不用 href 指向不存在文件），正文给最小定义——已落实。
- 贯穿例子：3584×3584 场景在 S3（参数规模建立规模感，未用 3584）、S4（3584 首次完整出现，token 手算 65536→16384）、S5（复用"可负担"作为工程支撑）——已落实。
- 误解与边界：scope §2.6 的四条误解分别落入 S5（对比预训练"没用"误读）、S2（"从零"字面误读，正文将两动机并列）、S3（RMSNorm/去 bias "惯例"误读，正文给报告定位）、S4（pixel-shuffle "有损"误读，正文明确无损重排）——已落实。

## 学习目标闭环（逐题核对）

- Q1（为什么从零训练）：由 S2 正文完整回答——方案两步 + 两个动机（稳定性 C2、目标对齐 C3）；S1 提供问题铺垫。正文独立作答，不依赖折叠块。✓
- Q2（架构组件 + RMSNorm/去 bias 服务稳定）：由 S3 正文完整回答——config 数字表 + 参数量手算 + RMSNorm/去 bias 的报告定位（C4）；折叠块仅放代码验证，正文已给完整手算。✓
- Q3（一套参数处理图/视频 + 1M 内 3584）：由 S4 正文完整回答——共享参数 + 分解注意力 + 时间池化 + pixel-shuffle + token 手算 65536→16384；折叠块不独占任何结论。✓
- Q4（结果 + 结论 + 边界）：由 S5 正文完整回答——匹配 baseline（C7/N2）+ 结论 + 两个边界。✓

全部学习目标由正文章节完整回答，折叠块全部收起时正文仍能作答。

## 代码运行

- 代码块 1（参数量与 token 数验证，S3 折叠块）：
  - 命令：`python3 /tmp/moonvit_v2_verify.py`（代码与页面内代码块完全一致）
  - 退出码：0
  - 实际输出与页面"预期输出"块逐行一致（Q/K/V/O=1,572,864；attn/layer=6,291,456；mlp/layer=8,388,608；norms/layer=2,048；per_layer=14,682,112；total 27L=396,417,024 ~0.40B；65536 tokens → 16384 tokens）。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/moonvit-v2/index.html`
  - 结果：validation ok（退出码 0）
- 命令：`python3 .dojo/scripts/validate.py wiki/moonvit-v2/overview.html`
  - 结果：validation ok（退出码 0）
- 占位符【…】、@content/@component/TODO/TBD 标记：两文件均已清除（grep 无匹配）。
- 同页锚点、重复 id、断裂本地引用：validate.py 全部通过。
- 未生成前置概念页处理：五项均用内联文字"概念页待生成"标注，不产生 href 断链（沿用 nope / block-attnres 既有惯例）。

## 公式渲染与交互

- KaTeX：正文使用 $…$（行内）与 $$…$$（行间）定界符，与外壳 auto-render 配置一致；公式含 \text{}、{,} 千分位、\times、\frac 等 KaTeX 支持语法。
- 参数量手算（F1）四步公式与 token 数手算（F2）两步公式均为行间公式，浏览器渲染预期正常。
- Prism 代码高亮：language-python 与 language-text 块，外壳脚本已接管。
- 侧边目录、章节折叠按钮、j/k 快捷键、暗/亮模式切换、返回顶部：均由外壳脚本提供，id 唯一性已通过 validate。
- 图片点击放大：本页无图片，不涉及。

## 写作偏差

无写作偏差。未自行增删核心章节（S1–S5 + 来源说明与大纲一致）、未新增学习目标（Q1–Q4 与 scope 一致）、未更换贯穿例子（3584×3584）、未把正文必要内容移入折叠块（F1/F2 手算在正文，折叠块仅放代码验证）、未使用证据不足论断（C1–C8 全部来自 K3 报告 §2.4 + config.json）。

写作过程中发现的一处机械兼容处理：前置概念占位链接若用 href 指向不存在文件会触发 validate.py 断链检查，按既有惯例改为内联文字"概念页待生成"标注，不改变大纲与内容范围。

## 备注

- 按任务约定：不更新 content.json，不执行 check 阶段（由编排者安排）。
- 临时验证脚本 /tmp/moonvit_v2_verify.py 为本流程创建，可保留供 check 阶段复验。
