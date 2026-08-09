# Delta 规则与 DeltaNet 独立审查

- 审查者：独立上下文（AI 模拟 / 真实目标读者）
- 页面版本：index.html db7e7ad893c95ddf84b136cf0e2898e2a3ad3e0a / overview.html e3164393af90fcc838684a15433f7145c16c088f
- 时间：2026-08-09 15:06 CST

## 问题

- [重要·技术] index.html §5「实际效果」MAD benchmark 表格：Average 列（DeltaNet 71.8 / Mamba 69.3）由 6 个任务计算得出（含 Compress），但表格仅展示 5 列（In-Context Recall / Fuzzy Recall / Noisy Recall / Selective Copy / Memorize），未展示 Compress 列。读者用可见 5 列手算 DeltaNet 均值得 (100+35.7+100+100+52.8)/5=77.7≠71.8，Mamba 为 (90.4+6.7+90.1+86.3+89.5)/5=72.6≠69.3，均对不上。页面自身 N2 参考文字明确列出了 Compress=42.2（DeltaNet）/ 52.7（Mamba），与表格形成内部矛盾。Transformer 行（Average 74.5）与 GLA 行（Average 60.0）同理不可验证。｜ 修复：在表格下方加注"Average 列由 6 个任务（含未展示的 Compress 任务，DeltaNet 42.2 / Mamba 52.7）取均值，故不能由上表可见 5 列直接算出；完整数字见 N2"，消除表格与 N2 的内部矛盾。 ｜ 复验：

- [重要·技术] index.html §5「实际效果」表格后正文：「DeltaNet 在 4 类 recall / copy 任务上几乎满分」表述不准确。4 类 recall/copy 任务（In-Context Recall、Fuzzy Recall、Noisy Recall、Selective Copy）中仅 3 类得 100（In-Context Recall、Noisy Recall、Selective Copy），Fuzzy Recall 为 35.7，远非「几乎满分」。若计入被表格省略的 Compress（42.2），则 5 类 recall/copy/compress 任务中仅 3 类满分，差距更大。该表述会令读者对实验结论产生明显误解。｜ 修复：已将"4 类 recall / copy 任务上几乎满分"改为"3 类 recall / copy 任务上满分（In-Context Recall、Noisy Recall、Selective Copy 得 100），仅 Fuzzy Recall 较弱（35.7）"。 ｜ 复验：

- [轻微·技术] index.html §2「来源与命名」及 C5 参考：引文「akin to the famous error-correcting delta-rule」标注出自 Schlag 2021 §4.2，但该精确引文实际出自 §1（Introduction）。§4.2 使用不同措辞「essentially implements the famous error-correcting delta rule (Widrow & Hoff, 1960)」。引文本身真实存在、归属正确，仅章节号有误。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 1
- 处置：进入修复
