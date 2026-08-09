# MOPD 初稿检查

## 输入版本

- `research/scope.md`：完成。概念歧义已裁定（MOPD vs OPD，采用 K3 §4.1.3 含义）；5 个学习目标 Q1–Q5 各有完成答案；核心内容 C1–C10、辅助内容 A1–A5、扩展内容 E1–E3 已分级；前置知识（知识蒸馏、RLHF/策略梯度）均未生成，按占位处理；误解 M1–M4 与适用边界已记录。
- `research/evidence.md`：完成。核心论断 C1–C7、公式 F1–F2、数字 N1–N2 全部来自 K3 技术报告 §4.1.2/§4.1.3，置信状态均为"已确认"，无冲突或证据不足项。
- `research/outline.md`：完成。5 章对应 5 个学习目标；讲解顺序 S1→S5；贯穿例子为三 token 词表 {A,B,C}；材料职责（公式、数字例子、ASCII 图示、对照表格、context-box、blockquote.meta）已分配；无伪代码与可运行代码组件。
- `research/glossary.md`：完成。缩写、符号、术语漂移防护齐全。

## 大纲落实（逐项一行）

- 页面开头：blockquote.meta（主要依据）、context-box（9 专家构成与所处流程）、learning-goals（Q1–Q5）均已落实，钩子问题在前言段落。
- S1「为什么要把九个专家合成一个统一模型」：9 专家构成、分别部署成本、简单平均不可行、条件化学生合并方案、误解 M1 澄清、对照表格（分别部署/平均/MOPD）已落实。
- S2「学生自己生成的 token，由教师逐个打分」：on-policy 含义、前置概念占位（知识蒸馏/策略梯度）、Eq.15 全符号解释、对数比值分解 F2、三 token 手算例子、ASCII 图示（采样→打分→奖励回传循环）、折叠块（完整逐 token 计算）已落实。
- S3「裁剪和停梯度让奖励信号稳定可控」：sg 作用、误解 M2、clip 作用、误解 M3、极端概率比对照、折叠块（不同 Rmax 对照表）、裁剪双面性铺垫已落实。
- S4「九个教师按领域和努力程度轮流指导」：(d,e) 路由、学生条件化于 e 不条件化于 d、RLHF 占位、对照表格（MOPD vs RLHF）、RL 框架接入（partial rollout、per-token 正则化）、ASCII 图示（9 教师路由）、折叠块（mini-batch 配对示例）已落实。
- S5「MOPD 能做什么、不能做什么」：两阶段流程、教师上限、top-k 无优势（C6）、误解 M4、裁剪双面性、对照表格（能做/不能做）、适用条件三条已落实。
- 文末「来源与教学说明」：核心论断与来源、核心公式与来源、外部数字与实验条件、教学示例、教学解释与类比边界、教学简化及其限制 六小节齐全。
- 贯穿例子：三 token 词表在 S2 首次出现（正/负奖励方向），S3 复用（极端概率比裁剪），S4 复用（mini-batch 多教师路由），S5 复用（学生向教师靠拢不超越）。
- 误解和边界：M1（S1 callout）、M2/M3（S3 callout）、M4（S5 callout）、适用边界（S5 末段）均已落实。
- 过渡：S1→S2、S2→S3、S3→S4、S4→S5 章末过渡句均已落实。

## 学习目标闭环（逐题核对）

- Q1（为什么合并）：由 S1 正文章节完整回答（9 专家构成、分别部署成本、平均不可行、条件化学生合并）。通过。
- Q2（Eq.15 符号与手算）：由 S2 正文章节完整回答（逐符号解释、对数比值分解、三 token 手算例子正/负方向）。通过。
- Q3（裁剪与停梯度）：由 S3 正文章节完整回答（sg 阻断梯度路径、clip 裁剪奖励信号非概率比、极端值对照）。通过。
- Q4（多教师路由与 RL 集成）：由 S4 正文章节完整回答（(d,e) 路由、学生条件化、稠密奖励复用 RL 框架与 partial rollout）。通过。
- Q5（边界）：由 S5 正文章节完整回答（两阶段第二阶段、教师上限、top-k 无优势、裁剪双面性、适用条件三条）。通过。

全部 5 个学习目标由正文章节完整回答，折叠块全部收起时正文仍能回答（折叠块只承载完整逐 token 计算、Rmax 对照表、mini-batch 配对示例等补充细节）。

## 代码运行

无可运行代码。大纲未分配可运行代码组件（机制为对数与裁剪，不涉及需要执行验证的算法；强行加代码会隐藏而非澄清机制）。无伪代码组件（机制用公式与数字例子已足够）。页面内 `<pre class="diagram">` 为 ASCII 图示，非代码，无需运行。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/mopd/index.html`
- 结果：`validation ok: wiki/mopd/index.html`，退出码 0。
- 命令：`python3 .dojo/scripts/validate.py wiki/mopd/overview.html`
- 结果：`validation ok: wiki/mopd/overview.html`，退出码 0。
- 两份文档均无模板占位符（【…】）、无组件标记（@content/@component/TODO/TBD）、无重复 id、无指向缺失 id 的同页锚点、无指向不存在文件的本地资源引用。

## 公式渲染与交互

- 本环境无法实际打开浏览器，改用等价验证：
  - KaTeX 库存在性确认：`libs/katex.min.js`、`libs/auto-render.min.js`、`libs/katex.min.css` 均在仓库 `libs/` 目录下，页面引用路径 `../../libs/` 相对 `wiki/mopd/` 解析正确。
  - KaTeX 渲染验证：用 node 加载 `libs/katex.min.js`，读取 `index.html` 全文，提取全部 `$$…$$` 显示公式（3 块，去除脚本区误配）与 `$…$` 行内公式（184 处），逐个调用 `katex.renderToString(..., {throwOnError:true})`：显示公式 3/3 通过，行内公式 184/184 通过，无 parse error。
  - 关键公式 Eq.15（含 `\text{}`、`\mid`、`\frac`、`\left/\right`、`\!`、`\,`、`R_{\max}`、`y_{<t}`、`\pi_\theta`、`\pi_{\text{teacher}}^{(d,e)}`）单独渲染通过。
  - 手算数值复算：用 python3 `math.log` 重新计算页面所有对数比值与裁剪结果（log(0.7/0.5)≈0.336、log(0.2/0.3)≈−0.405、log(0.1/0.2)≈−0.693、log(90)≈4.50、log(900)≈6.80、log(0.9/0.5)≈0.588、log(0.9/0.1)≈2.197、各 log 分量、6.802−5=1.802、6.802−3=3.802），全部与页面陈述一致。
- 交互结构检查：6 个 `body > h2` 均有显式 id（why-merge、per-token-reward、clip-and-stopgrad、multi-teacher-routing、boundaries、sources-and-teaching-notes），侧边目录、章节折叠按钮、j/k 快捷键脚本依赖此结构；learning-goals 的 h2 位于 `<section>` 内，不会被 TOC/折叠脚本误处理。
- 真实浏览器中的 KaTeX 渲染外观、折叠块展开收起、目录高亮、暗/亮切换、复制按钮等交互的实际观感，需在 check 阶段（独立审查）由编排者安排的浏览器打开确认；本阶段已用上述等价手段覆盖公式可解析性与机械正确性。

## 写作偏差

无返回规划阶段的偏差。一处需记录的局部决定：前置概念（知识蒸馏、RLHF/策略梯度）按任务要求未生成、占位处理，正文用 callout 形式给出最小衔接（教师把高概率分给偏好 token、策略梯度把奖励当常数乘子），未给出指向不存在文件的链接，故 validate.py 的本地引用检查通过。若后续编排者递归生成这两个前置概念页，需把占位 callout 替换为概念页链接。
