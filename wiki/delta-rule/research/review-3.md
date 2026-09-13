<!-- review-meta
round: 3
page: wiki/delta-rule/index.html
reviewed_content_sha256: e97e9fb385581a4e
-->
# Delta 规则与 DeltaNet 审查记录（第 3 轮）

- 页面版本：6bc9b9ed38aeccf66ad1c88e0839793c89a519f1
- 审查时间：2026-09-13 19:05
- 审查者：独立子代理（未参与写作，也未参与前序轮次审查）
- 已完整阅读章节：核心问题、最容易误解、1 为什么线性注意力会"记不清"（1.1 retrieval 时到底发生了什么、1.2 手算例子、本章问题）、2 Delta 规则的紧凑公式与手算（2.1 来源与命名、2.2 手算一步、本章问题）、3 等价改写（3.1 回到手算例子、3.2 几何直觉、本章问题）、4 边界情况与 β_t 的退化（4.1–4.4、本章问题）、5 DeltaNet 与相邻模型对比（5.1–5.5，含代码折叠块、本章问题）、全文总结、来源与范围说明

已核对来源：arXiv:2406.06484（v2 全文，含 §2.2/§3.1/§3.3/§4.1/§4.2 与 Table 2/Table 3）、arXiv:2412.06464（v1/v2/v3 全文与摘要页）、arXiv:2102.11174（ar5iv 全文，§4.1/§4.2）。代码块在本地实机执行，输出与「预期输出」逐行一致；`.dojo/scripts/validate.py wiki/delta-rule/index.html` 返回 `validation ok`。全部表格数字（Table 2 MAD 6 列 ×4 行、Table 3 PPL 4 行）已逐格回原文核对，均一致；C3/C4/等价推导/边界/投影的算式全部可复算且符号一致。

## 问题

- [重要·技术] 5.4 实际效果（第 608 行）：把 DeltaNet 的最强项 Fuzzy Recall 说成弱项。原文把 Fuzzy Recall 列为 DeltaNet 相对其他模型的优势项，页面写成"3 类 recall / copy 任务上满分……仅 Fuzzy Recall 较弱（35.7）"，读者会以为 DeltaNet 的 fuzzy recall 差；实际 35.7 是本表所有模型中的最高值（Transformer 29.8、Mamba 6.7、GLA 6.9）。｜引文依据：arXiv:2406.06484 §4.1 原文 "DeltaNet is better at recalling tasks, especially on Fuzzy Recall, although it struggles on the 'Memorize' task"；同节 Table 2 Fuzzy Recall 列 Mamba 6.7 / GLA 6.9 / Transformer 29.8 / DeltaNet 35.7。｜修复要求：改写为与来源一致的表述——Fuzzy Recall 是 DeltaNet 优于其他模型的项（35.7 为表内最高），不得写成"较弱"；若要强调"相对其满分项偏低"，必须同时点明它是表内最高。｜修复：｜复验：
- [重要·技术] 5.4 实际效果（第 608 行）：为 Memorize 落后编造来源未给出的机制归因。页面写"Mamba 的标量衰减更擅长'渐进式记住全部历史'，DeltaNet 的方向擦除对'全部保留'反而不利"，来源只报告数字，未给任何机制解释。｜引文依据：arXiv:2406.06484 §4.1 仅有数据与一句概括，无机制说明；核对后来源对 Memorize（52.8 vs Mamba 89.5）"provide no explanation"。｜修复要求：删除该机制归因，或改写为明确标注的推断（"一种可能的解释是……"）且不得以来源结论口吻叙述；overview.html 第 42 行同一表述需同步处理。｜修复：｜复验：
- [重要·技术] 3.2 几何直觉（第 394 行）：英语引文标注为「原文称」但非原文，属伪引文。页面写 'Yang et al. 2024 NeurIPS §3.3 原文称 "erasing information in one subspace while preserving the other $d-1$ subspaces"'，而来源用词不同。｜引文依据：arXiv:2406.06484 §3.3 原文为 "only erases information in one subspace while keeping the other d−1 subspace intact"（keep…intact / subspace 单数），页面写成 "preserving … subspaces"。｜修复要求：改为逐字引文（"only erases information in one subspace while keeping the other d−1 subspace intact"），或去掉引号改为不带"原文称"的中文转述。｜修复：｜复验：
- [重要·技术] 5.3 Gated DeltaNet 的退化（第 588 行）：引文位置标错。页面把 "gating enables rapid memory erasure while the delta rule facilitates targeted updates" 归到「Yang, Kautz & Hatamizadeh ICLR 2025 §3.1 设计动机的代数体现」。｜引文依据：该句位于 arXiv:2412.06464 的 Abstract（v3 摘要原文 "We observe that these mechanisms are complementary: gating enables rapid memory erasure while the delta rule facilitates targeted updates."），"各管一摊"的动因说明在 Introduction，v1/v2 正文 §3.1 无此句。｜修复要求：把出处改为「摘要」或「引言」（若确指引言原句则写 Introduction），不得标 §3.1。｜修复：｜复验：
- [重要·技术] 引言（第 71 行）与 5.5 工程实现与现实应用（第 614、616 行）：无来源支撑的事实被写成结论。页面断言"现实应用如 Kimi K3 的 KDA（Kimi Delta Attention）"、"Kimi K3 模型的 KDA 即基于 delta 规则递归"、"开源实现在 flash-linear-attention 库（fla-org/flash-linear-attention，Triton kernel）"，而「来源与范围说明」只登记 C1–C8/F1–F3/N1–N2（Schlag 2021、Yang 2024、Yang 2025 三篇论文），Kimi K3 与 fla 库均无任何条目。｜引文依据：不适用（页面未给来源；三篇论文来源中不含 Kimi K3 / fla 库的内容）。｜修复要求：为这两处补可定位来源（模型技术报告或官方仓库），或删除该论断、降级为不带具体模型名的范围外提示；引言与 5.5 两处需一致处理。｜修复：｜复验：
- [重要·可读性] 2 Delta 规则的紧凑公式与手算（第 218–222 行）：正文自述与渲染顺序矛盾，且违反内容示例 A2「公式出现前说明它要解决的问题」。页面先给出 boxed 公式 C3（第 220 行），紧接着才写"公式出现前先说清它要解决的问题：上一章看到……Delta 规则就是这种规则"——该句声称在公式之前交代背景，实际排在公式之后，且内容与第 1、2 章开头重复。｜引文依据：不适用。｜修复要求：把"公式要解决的问题"整段移到第 220 行公式之前，删去"公式出现前先说清"这句自指元话语，或删除该段（问题已在第 1 章与本章开头交代）。｜修复：｜复验：
- [重要·表述] 1.1（第 125 行）、1.1 结尾（第 135 行）：元话语。第 125 行"但压缩有代价——下面要看这个代价是怎么产生的"，第 135 行"下面直接用这两个公式推导'记不清'的来源"——"下面要看…/下面直接用…"是规范点名的元话语句式，删除后信息不损失。｜引文依据：不适用。｜修复要求：改为直接陈述，例如第 125 行删去"下面要看这个代价是怎么产生的"（后半句"但压缩有代价"已足够），第 135 行改为"由这两个公式可直接看出'记不清'的来源"。｜修复：｜复验：
- [重要·表述] 3.1 回到手算例子（第 383 行）：第一人称复数会话指代。"这次我们看清楚了'擦的是什么'"中的"我们"，违反 style-guide §12「不使用第一人称复数」。｜引文依据：不适用。｜修复要求：改为无人称表述，例如"这样就看清楚了'擦的是什么'"或"该形式显式写出了被擦除的旧值"。｜修复：｜复验：

- [轻微·可读性] 2.1 来源与命名（第 268、271 行）：figcaption 与正文重复。figcaption 末句"本页以 Yang 2024 的公式为正式定义，Schlag 2021 为溯源"与紧随其后的正文"本页以 Yang 2024 NeurIPS 的公式为正式定义，Schlag 2021 作为 delta rule 的提出者溯源，Gated DeltaNet 在第 5 章对比"内容重叠。｜引文依据：不适用。｜修复要求：删除其中一处，或让正文补充 figcaption 没有的信息（如 Gated DeltaNet 在第 5 章对比）。｜修复：｜复验：
- [轻微·表述] 1.2（第 187 行）、4 章开头（第 467 行）：临场评价式元话语。"这段数字是想说清一件事"、"这是验证理解的最简方式，也是误解多发点"属于对自身叙事的现场评点。｜引文依据：不适用。｜修复要求：删去这类评点句，直接给出结论或过渡。｜修复：｜复验：
- [轻微·表述] 2.2（第 293 行）、5.4（第 608 行）：把"场景"当术语使用。第 293 行"……'模型改变主意'的场景"，第 608 行"擅长的场景""针对'key-value 检索'场景的设计选择"，三处用"场景"充当类别名词。｜引文依据：不适用。｜修复要求：改为具体说法，如"这一类需要精确覆写的任务""（第 293 行）模拟同一 key 改绑新值的情形"。｜修复：｜复验：

（另附机械核对结论，未单列问题：公式批注"（DeltaNet 形式，C3）""（……形式，C4）"以纯文本 C3/C4 出现，未按 style-guide §6 用 `<sup>[Cx]</sup>`；「来源与范围说明」中 C8、F1、F2 定义后正文从未引用，而 C4 仅以行内文本出现——双向对应关系不完整，属轻微格式问题，建议一并修。）

## 结论

- 统计：阻断 0 / 重要 8 / 轻微 3
- 处置：修复。核心结论、全部公式与推导、两个表格数字、代码块输出均与来源一致（已逐项核对：Table 2 MAD 六列四行、Table 3 PPL 四行、C3/C4 等价推导、β_t=0/1 边界、I−kkᵀ 投影、Gated DeltaNet 退化三例、numpy 代码实机运行输出与「预期输出」逐行相同），无阻断问题；但来源标注保真度（Fuzzy Recall 取向、Memorize 归因、§3.3 伪引文、摘要句误标 §3.1、Kimi K3 无来源）与表述维度（元话语、会话指代"我们"、场景当术语）须逐条修复后复验。
