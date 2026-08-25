# SFT 初稿检查

- 输入版本：scope / evidence / outline / glossary 均为初版（2026-08-25）；evidence 中 C7、F1 已在写作前完成核对（InstructGPT ar5iv 全文检索定位原文），C8 为写作时补充的已核对论断（综述 §1 三收益）。
- 大纲落实：
  - 页面开头（翻译指令续写场景 → SFT 定义 → 学习目标）✓
  - 第 1 章错位（综述表述 + 对照表 + few-shot 对比 + 三收益）✓
  - 第 2 章数据与目标（数据形态 + 公式 F1 + teacher forcing + 拼接/特殊 token + loss masking 原文 + 贯穿示例进场）✓
  - 第 3 章手算（五位置表 + mask/no-mask 两种平均 + 完整计算折叠块）✓
  - 第 4 章流程定位（dg-flow 三步图 + RM 从 SFT 模型出发 + 损失表达能力论证 + 贯穿示例推进）✓
  - 第 5 章边界（四个小节 + 训练事实表）✓
  - 学习目标 5 条与核心问题块一致 ✓
  - 前置知识：pretraining 页（第 1 章）、cross-entropy 页（第 2 章）链接就位 ✓
  - 误解 4 条与 scope 一致（5.1–5.4）✓
  - 过渡：每章末尾指向下一章 ✓
- 目标覆盖检查：Q1（第 1 章）、Q2（第 2 章）、Q3（第 3 章）、Q4（第 4 章）、Q5（第 5 章）均有正文章节完整回答；核心问题 5 条、章节问题每题均有解答折叠块 ✓
- 代码运行：无可运行代码（大纲未分配；数值由 Python 实算后写入：4.6052、2.9957、3.9120、1.2040、0.1054、1.3093、0.6547、12.8223、2.5645、11.5129、89.8%）✓
- 机械检查：`python3 .dojo/scripts/validate.py wiki/sft/index.html` → validation ok；overview.html → validation ok ✓
- 公式渲染与交互：headless Chrome 实测：`.katex` 节点 133 个、`.katex-error` 0 个、details 折叠块 17 个；概念页链接 4 处，目标文件（cross-entropy、pretraining）均存在 ✓
- 写作偏差：编号修正一处（页面误用 C9，改为 C8 保持连续）；evidence.md 同步补 C8 条目。未增删章节、未更换示例。
