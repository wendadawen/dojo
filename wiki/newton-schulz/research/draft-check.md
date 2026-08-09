# Newton-Schulz 迭代初稿检查

- 检查时间：2026-08-09
- 检查者：生成者自检（过程记录，非独立审查）
- 输入版本：
  - scope.md：完成（含 Q1–Q5 学习目标、前置映射、误解 M1–M4、边界）
  - evidence.md：完成（C1–C7、F1–F6、N1–N2，全部已确认，无冲突/不足）
  - outline.md：完成（页面开头 + S1–S5 + 来源说明，材料职责、正文折叠分工、组件选择齐备）
  - glossary.md：完成（22 项术语，符号一致性说明）

## 大纲落实

- 章节顺序：S1 why-no-svd → S2 iteration-and-mechanism → S3 convergence-and-preprocessing → S4 zero-singular-values → S5 muon-application → sources-and-teaching-notes，与 outline 一致。
- 页面开头：blockquote.meta（主要依据）+ learning-goals（5 条，与 scope Q1–Q5 一致）+ 钩子场景（diag(0.5,0)）✓
- 学习目标：Q1→S1、Q2→S2、Q3→S3、Q4→S4、Q5→S5，一一对应 ✓
- 前置知识引用：SVD（占位链接 ../svd/index.html，附最小衔接）、正交矩阵/Frobenius 范数/sign（内联最小定义）✓
- 贯穿例子 diag(0.5,0)：S2 首现（f(0.5)=0.6875、f(0)=0）→ S4 收尾（完整 6 步表，σ=0.5→1、σ=0→0）✓；辅助例子 G=[[1,1],[0,1]]：S3 + 代码 ✓
- 误解与边界：M1（所有奇异值拉平 1）→S4 callout-red ✓；M2（直接套迭代）→S3 ✓；M3（NS 等价 SVD）→S1 callout-blue ✓；M4（单调）→S3 完成检查 ✓
- 每章完成检查：S1–S5 各有 2–3 个可检查问题，无答案括注 ✓
- 过渡：每章末总结 + 指出下一章缺口 ✓
- 来源与教学说明：6 个 h3 小节（核心论断/核心公式/外部数字/教学示例/教学解释/教学简化），无内容的分组已删（外部数字小节说明"无外部数字"）✓

## 学习目标闭环

- Q1（解决什么问题、为何不用 SVD）：S1 正文回答——求极分解正交因子 $W\approx UV^{\!\top}$、Procrustes 解、SVD 贵而 NS 只用矩阵乘 ✓
- Q2（公式符号与收敛机制）：S2 正文回答——逐符号解释、奇多项式与 SVD 可交换→标量 f→sign，折叠块给可交换性推导 ✓
- Q3（收敛条件与预处理）：S3 正文回答——$\sigma_{\max}<\sqrt3$、$f(\sqrt3)=0$ 阈值来源、Frobenius 预处理保证 $\sigma_{\max}\le1<\sqrt3$ ✓
- Q4（零奇异值保持 0）：S4 正文回答——奇多项式 $f(0)=0$ 故 $\sigma=0$ 不动点、与极因子秩保持一致、完整 6 步表为证 ✓
- Q5（Muon 用途）：S5 正文回答——正交化动量矩阵使各方向力度均衡、cursed quintic 系数和≠1 非收敛却固定步数用 ✓
- 折叠块全收起时正文仍回答 Q1–Q5：公式、机制三步、条件、预处理、零边界、Muon 用途均在正文 ✓

## 代码运行

- 代码块：1 个（language-python，纯 Python，不依赖第三方库）
- 运行命令：`python3 /tmp/ns_page_code.py`（从 index.html 提取 language-python 代码块后运行）
- 退出码：0
- 实际输出与页面"预期输出"块逐行一致（diag 6 步 sigma 序列、final diag=[0.9999...,0.0]；G 的 ||G||_F=1.7321、8 步误差序列、final Q 与 UV^T 一致）✓
- 旁证：另用 numpy 计算 N2 的极因子 $UV^{\!\top}$，与代码 final Q 一致（[0.8944,0.4472],[-0.4472,0.8944]）✓

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/newton-schulz/index.html`
- 结果：`validation ok`，退出码 0 ✓
- 命令：`python3 .dojo/scripts/validate.py wiki/newton-schulz/overview.html`
- 结果：`validation ok`，退出码 0 ✓
- 占位符 【…】：0（grep 无匹配）✓
- 模板标记 @content/@component/TODO/TBD：0（grep 无匹配）✓
- 重复 id：无 ✓
- 同页锚点：无（未用 #anchor）✓
- 本地引用：全部解析（../../index.html、overview.html、index.html、../svd/index.html、../per-head-muon/index.html、../muon-optimizer/index.html、../../libs/* 均 OK）✓
- 角度括号：KaTeX 公式内 `<`/`>` 均用 `&lt;`/`&gt;` 或 `\le`/`\ge`，无未转义 raw `<`（脚本核查 0 处真实问题，唯一误匹配为 JS 脚本内 `$`）✓

## 公式渲染与交互

- 浏览器打开 index.html（file://）实际检查：
  - KaTeX：inline `$...$` 与 display `$$...$$` 均正常渲染（σ、$\sqrt3$、矩阵 bmatrix、cases 分段函数、\lVert\rVert、\tfrac 均显示）；auto-render 配置与模板一致 ✓
  - 目录：侧边 TOC 自动生成 S1–S5 + 来源说明 ✓
  - 章节折叠按钮：每个 h2 有 ▼ 按钮 ✓
  - 代码块复制按钮：hover 显示 ✓
  - 折叠块：4 个 details（S2 推导、S2 diag 手算、S3 非对角计算、S3 代码）+ S3 代码折叠块均默认收起，收起后正文完整 ✓
  - 主题切换、返回顶部、进度条：模板脚本提供 ✓
- overview.html：KaTeX inline 渲染 ✓，与 index.html 双向链接 ✓

## 写作偏差

- 无重大偏差。
- 微调：outline 原计划 S2 数字例子折叠块给"前几步"，实际写入前 4 步（k=0..3），与 S4 完整 6 步表分工，避免重复——属局部衔接，不改大纲。
- 新增：为使 SVD 占位链接可解析，创建了 wiki/svd/index.html 最小占位页（仿原 newton-schulz 占位风格），未更新 content.json。该页登记为 depth-1 待生成前置概念页。
- 未做：未做 check 阶段（独立审查），未更新 content.json 与首页 index.html——按任务要求。
