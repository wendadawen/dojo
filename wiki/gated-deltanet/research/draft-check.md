# Gated DeltaNet 初稿检查

- 输入版本：scope.md / evidence.md / outline.md / glossary.md 均已完成，规划完成条件满足（概念歧义已裁定、5 个学习目标、核心内容与前置知识映射齐全、核心论断来源定位、教学大纲完整）。

- 大纲落实：
  - 页面开头（钩子 + context-box + learning-goals + misconceptions + 来源摘要）：已落实
  - S1「DeltaNet 与 Mamba2 各自缺什么」（Q1, C1/C2/C7）：已落实，含三模型对照表
  - S2「Gated DeltaNet 公式与符号」（Q2, F1/C3/C4/C5）：已落实，含符号定义、形状检查、α_t/β_t 职责、先衰减后擦写等价解读、边界检查
  - S3「手算一步更新」（Q2, F1）：已落实，3 步 2 维教学示例 + 三模型结果对照表 + 折叠块逐步代入
  - S4「退化与并行训练」（Q3/Q4, F2/C6）：已落实，含退化代入、chunkwise 算法说明、可运行代码折叠块
  - S5「与 KDA 关系及实验」（Q5, C8/N1/N2/N3）：已落实，含 α_t 设计对照表、实验数字表、现实应用
  - 文末「来源与教学说明」：已落实（核心论断/公式/数字/教学示例/教学解释/教学简化六小节齐全）
  - 前置知识引用：delta-rule、linear-attention、kda 均给出概念页链接，链接有效
  - 贯穿例子：3 步 2 维序列，第 1-2 步建两个正交关联，第 3 步覆写 + 衰减，可手算
  - 误解和边界：4 条误解在 misconceptions 组件 + 正文相应章节处理
  - 过渡：每章末尾有过渡句指向下一章

- 学习目标闭环：
  - Q1（DeltaNet 缺陷 + 为什么加 α_t）：S1 正文完整回答（delta 规则只擦单方向、Mamba2 无选择性、两者互补）
  - Q2（手算 + α_t/β_t 职责）：S2 给公式与职责分工，S3 手算验证，正文完整
  - Q3（退化关系）：S4 正文完整回答（α→1 退化为 DeltaNet、β→0 退化为 Mamba2 特殊情形，从公式代入）
  - Q4（并行训练）：S4 正文完整回答（串行无法用 GPU、chunkwise 切 chunk、WY 表示、α_t 不破坏结构）
  - Q5（与 KDA 的 α_t 区别）：S5 正文完整回答（标量 vs channel-wise、无下界 vs lower-bounded）
  - 折叠块全部收起时正文仍能回答 Q1-Q5：已确认（折叠块只放完整代入、代码、补充说明，核心结论在正文）

- 代码运行：
  - 代码块：wiki/gated-deltanet/index.html §4 可运行代码折叠块
  - 运行命令：`python3 /tmp/gated_deltanet_demo.py`
  - 退出码：0
  - 实际输出与页面描述一致：
    - DeltaNet S_3 = [[0,0],[1,1]]，S_3@k_3=[0,1]=v_3，S_3@k_2=[0,1]=v_2 ✓
    - Gated DeltaNet S_3 = [[0,0],[1,0.5]]，S_3@k_3=[0,1]=v_3，S_3@k_2=[0,0.5]（衰减）✓
    - Mamba2 S_3 = [[0.5,0],[0,0.5]]，S_3@k_3=[0.5,0]（未覆写），S_3@k_2=[0,0.5] ✓
    - 退化验证 alpha=1 == DeltaNet: True ✓
    - 退化验证 beta=0 == alpha*S_2: True ✓
  - 代码变量与公式符号映射：gated_delta_step(S_prev, k, v, alpha, beta) 对应 F1 公式 $S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$

- 机械检查：
  - 命令：`python3 .dojo/scripts/validate.py wiki/gated-deltanet/index.html`
  - 结果：`validation ok: wiki/gated-deltanet/index.html`（退出码 0）
  - 命令：`python3 .dojo/scripts/validate.py wiki/gated-deltanet/overview.html`
  - 结果：`validation ok: wiki/gated-deltanet/overview.html`（退出码 0）
  - 无占位符残留、无模板标记、无重复 id、无断链

- 公式渲染与交互：
  - KaTeX 公式（$...$ 与 $$...$$）已在页面中标记，外壳脚本自动渲染
  - 折叠块（details/summary）结构正确
  - 代码块（language-python/language-text）Prism 高亮
  - 目录、章节折叠、返回顶部等交互由外壳脚本处理
  - 前置概念页链接（../delta-rule/、../linear-attention/、../kda/）均指向已存在的页面

- 写作偏差：无。大纲的全部章节、学习目标、前置知识、完成检查和过渡均已落实，未引入大纲外内容，未把正文必要内容移入折叠块。

- 概念歧义处理说明：任务描述把 arXiv:2406.06484 标为 "Gated Delta Networks"（NeurIPS 2024），实际该编号是 DeltaNet 论文 "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"；arXiv:2412.06464 才是 "Gated Delta Networks: Improving Mamba2 with Delta Rule"（ICLR 2025，即 Gated DeltaNet）。本页在 scope.md §1.1 记录此裁定，正文以 arXiv:2412.06464 为 Gated DeltaNet 正式来源，DeltaNet 公式引用 arXiv:2406.06484（由 wiki/delta-rule/ 覆盖）。
