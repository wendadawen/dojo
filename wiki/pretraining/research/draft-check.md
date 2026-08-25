# 语言模型预训练初稿检查

- 输入版本：scope / evidence / outline / glossary 均为初版（2026-08-25），来源均已实际打开核对（GPT-2 PDF 提取文本、LLaMA ar5iv、InstructGPT ar5iv、综述 ar5iv、Deep Learning 官方网页版）。
- 大纲落实：
  - 页面开头（输入法联想场景）✓
  - 第 1 章语言模型是什么（分布表 + 「整句由逐 token 合成」澄清）✓
  - 第 2 章链式分解（GPT-2 式 (1) + 手算表 + 乘法公式折叠块）✓
  - 第 3 章预训练目标（公式 + 两种说法等价 + 自监督 + 语料视角）✓
  - 第 4 章基座模型与后训练（错位 + InstructGPT 起点 + 链接 SFT 页）✓
  - 学习目标 4 条与核心问题块一致 ✓
  - 前置知识：交叉熵页链接 3 处（第 2、3 章 + 来源 F2）；条件概率/乘法公式页内最小展开 ✓
  - 贯穿示例（五 token 词表 + 「天气真」）覆盖第 1–4 章 ✓
  - 术语对照（decoder-only vs 掩码 LM）在来源章节简化条件说明 ✓
  - 过渡：每章末尾指向下一章 ✓
- 目标覆盖检查：Q1（第 1 章）、Q2（第 2 章）、Q3（第 3 章）、Q4（第 4 章）均有正文章节完整回答；核心问题 4 条、章节问题每题均有解答折叠块 ✓
- 代码运行：无可运行代码（大纲未分配；数值由 Python 实算后写入：0.12、1.3863、0.2231、0.5108、2.1203、0.7068）✓
- 机械检查：`python3 .dojo/scripts/validate.py wiki/pretraining/index.html` → validation ok；overview.html → validation ok ✓
- 公式渲染与交互：headless Chrome 实测：`.katex` 节点 68 个、`.katex-error` 0 个；概念页链接 4 处（指向 cross-entropy 与 sft，其中 sft 页面为同批任务、写作顺序在本页之后——交付时已存在）✓
- 写作偏差：无（未增删章节、未更换示例）。初稿曾误用 Markdown 链接语法，已改为 `<a>` 标签。
