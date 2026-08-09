# MoonEP 完美均衡专家并行初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成（2026-08-09）
- 模板：`.dojo/templates/concept/index.html` 与 `overview.html`
- 来源：K3 技术报告 §5.2.1（约 1305–1354 行）与 §E（约 2941–3022 行），文件位置 `/tmp/kimi-k3-research/k3-report.txt`

## 大纲落实（逐项一行）

- 钩子问题开头（callout-blue）：落实，30 秒内点明 router 偏好动态 → 不均衡 + 动态形状 + host 同步三个问题。
- 一句话解释：落实，"通过在线规划和迁移冗余专家让每个 EP rank 恰好收到 S×K token"。
- 学习承诺（learning-goals 4 项）：落实，与 scope.md Q1–Q4 一致。
- 前置知识引用（context-box）：落实，引用 moe-serving 与 gpu-execution-model 两个已有概念页。
- 来源摘要（blockquote.meta）：落实，指向 K3 报告 §5.2.1 与 §E + MoonEP GitHub 仓库。
- 首个具体场景：落实，$E=4,R=2,S=4,K=1$ 贯穿例子在 S1 前引入。
- S1（s1-imbalance-and-idea）：落实，讲清传统 EP 不均衡来源 + 冗余专家定义 + 完美均衡目标，含 ASCII 图示与对照表。
- S2（s2-bound-and-tightness）：落实，讲清 $M(I)$ 目标、Theorem 1 关键引理与上界、Theorem 2 最坏构造与基本紧，含补充折叠块（完整证明）与数字例子折叠块。
- S3（s3-buffer-sync-flow）：落实，讲清 buffer 降到 S×K、消除 host 同步、forward 三步、backward 两步，含 ASCII 图示、伪代码折叠块、五维对照表。
- S4（s4-boundary-and-comparison）：落实，讲清解决/不解决各三个问题、与 ECHO/UltraEP/DeepEP 区别，含四方案对照表。
- 误解与边界：落实，scope.md §7.1 的 6 条误解在正文相应位置处理（router 训练误读在 S1 与 S4、冗余专家常驻误读在 S1 与 S3、E/R 是经验值误读在 S2 与 S4、per-expert 偏斜误读在 S4、推理适用误读在 S4、DeepEP buffer 总是 S×K×R 误读在 S3）。
- 来源与教学说明：落实，含核心论断来源、核心公式来源、教学示例、教学解释与类比边界、教学简化及其限制五节。
- 章节完成检查：每个正文章节末尾均有"完成本章后用下面几个问题检查自己是否跟上"引导与 3–4 个可检查项。

## 学习目标闭环（逐题核对）

- Q1（传统 EP 不均衡来源 + MoonEP 核心思路）：由 S1 正文完整回答。S1 给出 router 偏好动态 → rank 收到 pair 数动态（C1）、冗余专家定义（C2）、完美均衡目标（C3）、总量守恒（F4）。折叠块全收起仍能回答。
- Q2（冗余专家 + E/R 上界 + 基本紧）：由 S2 正文完整回答。S2 给出 $M(I)$ 目标（F1）、Theorem 1 关键引理与上界结论（F2）、Theorem 2 最坏构造与基本紧（F3）、工程预留含义。补充折叠块只展开证明细节，数字例子折叠块只展开手算。折叠块全收起仍能回答。
- Q3（buffer 降到 S×K + 消除 host 同步 + forward/backward 流程）：由 S3 正文完整回答。S3 给出 buffer 对比（C7，含最坏限定）、host 同步消除（C8）、forward 三步（C4, C6）、backward 两步（C5）。伪代码折叠块只展开 forward 步骤细节。折叠块全收起仍能回答。
- Q4（解决/不解决 + 与 ECHO/UltraEP/DeepEP 区别）：由 S4 正文完整回答。S4 汇总三个解决、三个不解决（含 Expert-GEMM scheduler 处理 per-expert 偏斜，C9）、与 ECHO/UltraEP 区别（C10）、与 DeepEP 关系。四方案对照表在正文。

## 代码运行

- 无可运行代码（本页只有伪代码折叠块，标为 language-text，不是 Python，不声称可运行）。
- 伪代码展示 forward 的 planning + 预取 + dispatch 三步，输入、状态、核心步骤、输出齐全，未用 train()/optimize() 黑箱。
- 教学简化在伪代码后与"来源与教学说明 > 教学简化及其限制"中说明（省略 combine、router 训练、optimizer step 等通用 MoE 步骤）。

## 机械检查

- 命令：`python3 .dojo/scripts/validate.py wiki/moonep/index.html`
- 结果：`validation ok: wiki/moonep/index.html`，退出码 0。
- 命令：`python3 .dojo/scripts/validate.py wiki/moonep/overview.html`
- 结果：`validation ok: wiki/moonep/overview.html`，退出码 0。
- 占位符检查：`grep -c "@content\|@component\|【\|概念名" wiki/moonep/index.html` = 0；`grep -c "@content\|@component\|【\|概念名" wiki/moonep/overview.html` = 0。

## 公式渲染与交互

- 浏览器打开 `wiki/moonep/index.html` 检查：
  - KaTeX 渲染：$M(I)=\min_P\max_r\{m_r(P)\}$、$M(I)\le E/R$、$\lceil E(R-1)/R^2\rceil\approx E/R$、$S\times K$、$S\times K\times R$、$E/R$、$R-1$、$S\times K\times R$ 等公式正常渲染（行内与块级）。
  - 折叠块：S2 的"Theorem 1 构造性证明完整复述"与"数字例子"折叠块、S3 的"伪代码"折叠块均可正常展开/收起。
  - 目录锚点：侧边目录显示 S1–S4 + 来源与教学说明，点击平滑滚动；j/k 快捷键跳章节正常。
  - 主题切换：亮/暗模式切换正常，代码块 Prism 主题同步切换。
  - 章节折叠按钮：h2 右侧 ▼ 按钮可折叠/展开章节内容。
- 浏览器打开 `wiki/moonep/overview.html` 检查：
  - 4 段结构正常显示（为什么需要它 / 核心直觉 / 关键结论与边界）。
  - 内联 $S\times K$、$S\times K\times R$、$E/R$、$\lceil E(R-1)/R^2\rceil$ 公式正常渲染。
  - 链接到 moe-serving 与 index.html 均有效。

## 写作偏差

- 无偏差。大纲的章节、学习目标、前置知识、贯穿例子、误解和边界、过渡均已按 outline.md 落实，未增删核心章节、未更换贯穿例子、未改变前置知识映射。
- S2 数字例子折叠块中发现一个细节：$E=4,R=2$ 时 Theorem 2 的下界公式 $\lceil E(R-1)/R^2\rceil=\lceil 4/4\rceil=1$，但实际最坏构造需要 2 个冗余专家（专家 {2,3} 都被涉及）。已在折叠块内标明这是 ceiling 与构造细节的差异、大 R 下两者趋同，不改变 Theorem 2"基本紧"的结论。该细节属局部补充，未改变大纲，未引入新论断。
- 未返回规划阶段，未改变研究范围或教学大纲。
