<!-- review-meta
round: 3
page: wiki/gated-deltanet/index.html
reviewed_content_sha256: dfbe28d75e6ddbbf
-->
# Gated DeltaNet 审查记录（第 3 轮）

- 页面版本：c32aa1fd55c65d815bbe1cddcc80e0d67d9894e0（git hash-object wiki/gated-deltanet/index.html）
- 审查时间：2026-09-13 19:05
- 审查者：独立子代理（未参与写作与前序审查）
- 适用规范：guides/concept/check.md、guides/concept/style-guide.md（页面 dojo:type=concept）
- 已完整阅读章节：核心问题 → 最容易误解 → 1. DeltaNet 与 Mamba2 各自缺什么 → 2. Gated DeltaNet 的公式与符号 → 2.1 α_t 与 β_t 的职责分工 → 2.2「先衰减后擦写」的等价解读 → 3. 手算 Gated DeltaNet 一步更新 → 3.1 三模型在同一序列上的结果对比 → 4. 退化关系与并行训练算法 → 4.1 Gated DeltaNet 的退化 → 4.2 并行训练算法 → 5. 与 KDA 的关系及实验效果 → 5.1 Gated DeltaNet 与 KDA 的 α_t 区别 → 5.2 实验效果 → 5.3 现实应用 → 来源与范围说明（论断与来源（C）/公式与来源（F）/外部数字与实验条件/构造示例/辅助解释与类比边界/简化条件及其限制）→ 全文脚本与折叠块
- 外部来源核对方式：arXiv:2412.06464 最新版（v3，ICLR 2025 版，本地 PDF 全文）与 arXiv:2412.06464v1（/html 原始 HTML 全文，逐句 grep）；Qwen3-Next 与 Kimi K3 报告经 WebSearch 核对

## 机械验证结果（复核用）

- `python3 .dojo/scripts/validate.py wiki/gated-deltanet/index.html` → `validation ok`（exit 0）。
- 页面代码块提取后原样执行：输出与页面「预期输出」逐行一致；退化验证两行均为 `True`（`Gated DeltaNet == DeltaNet`、`Gated DeltaNet == alpha * S_2`）。
- 手算全部复算通过：S_1=[[1,0],[0,0]]，S_2=[[1,0],[0,1]]，S_3=[[0,0],[1,0.5]]；DeltaNet S_3=[[0,0],[1,1]]；Mamba2(β=0) S_3=[[0.5,0],[0,0.5]]；本章问题 S_t=[[0.5,1],[0,0]]，查询 k_t→(1,0)⊤、查询 (1,0)⊤→(0.5,0)⊤，全部与正文/表一致。
- 数字核对：N1/N3 与 v3 Table 3（Gated DeltaNet Wiki PPL 16.42、LMB 12.17；Mamba2 16.56/12.56；DeltaNet 17.71/16.88；Transformer++ 18.53/18.32；GDN-H1 16.07/12.12；H2 15.91/12.55；常识推理 avg 55.32/54.89/52.14/52.25/56.40/56.18）一致；N2 与 v3 Table 2（S-NIAH-2 4K：18.6/56.2/92.2；S-NIAH-3 4K：22.4/4.6/27.6）一致；e^{-5}≈0.0067 正确。
- 前置链接：delta-rule、linear-attention、kda、qwen3-5-dataflow 四页均存在；overview.html 与 index.html 互链。
- 表述扫描：全文（含折叠块、图注）无「我们/你/下面来看/需要注意的是/场景（当术语）」等；无 Unicode 数学字符；符号写法一致。

## 问题

- [重要·技术] 页首 blockquote.meta（第 68 行）与来源节 C1–C7/F1/N1–N3：来源所标版本与页面所用章节号/公式号/表号不符。页面标为「ICLR 2025（arXiv:2412.06464）」，却使用 §2.3（DeltaNet 缺全局遗忘）、§2.2（Mamba2）、§3.1 Eq.8（gated delta rule）、§3.2（chunkwise 并行算法）、§4 Table 2（语言建模/常识）、§4 Table 3（S-NIAH）；这组编号只对应 arXiv **v1**（2024-12-09）。页面所标 ICLR 2025 版即 arXiv **v3**（2025-03-06）中，§2 仅有 2.1（Mamba2）/2.2（DeltaNet）、无 §2.3；gated delta rule 是 **Eq.10**；chunkwise 算法在 **§3.3**；**§3.2 是 S-NIAH 案例研究**；语言建模表是 **Table 3**、S-NIAH 表是 **Table 2**（且位于 §3.2，不在 §4）。读者按「ICLR 2025」取最新 PDF 将逐条定位失败。｜引文依据：v3 §2 目录「2.1 Mamba2: Linear Attention with Decay / 2.2 DeltaNet: Linear Attention with Delta Rule」；v3 §3.1「$\mathbf{S}_t=\mathbf{S}_{t-1}(\alpha_t(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\intercal))+\beta_t\mathbf{v}_t\mathbf{k}_t^\intercal$ (10)」；v3 §3.3「Algorithm: Hardware-efficient chunkwise training」；v3 §3.2「Case Study: Single Needle in a Haystack (S-NIAH)」；v3「Table 2: Zero-shot performance comparison on S-NIAH benchmark suite」「Table 3: Performance comparison on language modeling and zero-shot common-sense reasoning」。v1 则完全吻合页面编号：§2.3「Delta Networks: Linear Attention with Delta Rule」、§3.2「Algorithm: Hardware-efficient Chunkwise training」、Eq.(8) 为 gated rule、Table 2=语言建模、Table 3=S-NIAH。｜修复要求：二选一并贯彻全页——（a）在页首与每个 C/F/N 明确标注所依据版本为 arXiv:2412.06464v1；或（b）把全部章节号、公式号、表号改为 ICLR 2025（v3）编号（§2.1 Mamba2、§2.2 DeltaNet、§3.1 Eq.10、§3.3 chunkwise、Table 3 语言建模、Table 2 S-NIAH）。不得既不标版本又沿用 v1 编号。｜修复：｜复验：

- [重要·技术] 第 132 行/C1（第 669 行）、第 671 行 C2、第 421 行/C6（第 679 行）：三处以「原文」+引号给出的英文引文均非逐字，属把改写当引文。｜引文依据：（C1）页面作 "it lacks the ability to rapidly clear outdated information, particularly during context switches"；v1 §2.3 实为 "the model lacks the ability to rapidly clear outdated **or irrelevant** information, **especially** during context switches where previous data needs to be erased."。（C2）页面作 "Mamba2 decays all key-value associations uniformly by a dynamic ratio α_t"；v1 §2.2 实为 "which uniformly decays all key-value associations **at each time step** by a dynamic ratio, α_t"。（C6）页面作 "the gating term only performs element-wise multiplication and does not affect the matrix multiplication structure, thus compatible with tensor core GPU optimization"；v1 §3.2 实为 "the gating term (colored in blue) only performs **elementwise** multiplication **with (intermediate) variables without affecting matrix multiply structures, enabling** tensor core GPU optimization."。｜修复要求：三处均改为与来源逐字一致的原句（含彩色/括号等原样成分），或去掉「原文」字样与引号、改为明确标注的转述。｜修复：｜复验：

- [重要·技术] 第 191、196 行与第 586、590 行：同一符号 $S$ 在本页出现两种形状且未注明。§2 定义 $S_t \in \mathbb{R}^{d_v \times d_k}$（行=value 维）并给出 $o_t = S_t q_t$；但第 590 行的 KDA 公式 $S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$ 只在 $S \in \mathbb{R}^{d_k \times d_v}$ 时维度自洽（$\mathrm{Diag}(\alpha_t)$ 为 $d_k \times d_k$ 左乘，写入项为 $k_t v_t^\top$）。第 586 行「$S_{t-1}$ 的不同行可以按不同速率衰减」同样要求行=key 通道，与 §2 的「行=value 维」矛盾。读者以 §2 约定复算 KDA 公式会得到维度不匹配。｜引文依据：被引的 KDA 页明确记录该约定差——「$S_t \in \mathbb{R}^{d_k \times d_v}$……两者是 $S_{\text{K3}} = S_{\text{DeltaNet}}^\top$ 的转置」；K3 报告中 KDA 公式为 $S_t=(I-\beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_t k_t v_t^\top$，$S\in\mathbb{R}^{d_k\times d_v}$。｜修复要求：在 §5.1 增补一句转置约定说明（KDA 采用 $S \in \mathbb{R}^{d_k \times d_v}$，与本页 §2 的 $S \in \mathbb{R}^{d_v \times d_k}$ 互为转置，机制等价），或把 KDA 公式改写为与本页 §2 同一约定。｜修复：｜复验：

- [轻微·技术] 第 151 行与 C7（第 681 行）：引文归属位置有误。页面把 "we observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates" 标为「§1 原文」；该句实为论文**摘要**。v1 §1 的对应表述是 "Recognizing the complementary advantages of the gated update rule and the delta rule in memory management"，措辞不同。｜引文依据：v1 摘要含 "We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates."；v1 §1 正文为 "Recognizing the complementary advantages of the gated update rule and the delta rule in memory management"。｜修复要求：改标为「摘要」，或改用 §1 实际语句并逐字引用。｜修复：｜复验：

- [轻微·技术] 第 624 行：S-NIAH-2/3 的描述与来源不符且版本混用。「（number/uuid 针，非真实文本）」中：(a)「uuid 针」取自 v3（v1 的 S-NIAH-3 为 word-based needle），而本页其余编号来自 v1；(b)「非真实文本」与来源相反。｜引文依据：v1「This becomes evident in NIAH-2 and NIAH-3 where needles are grounded in real-world text data: DeltaNet's performance degrades significantly」；v3 §3.2「In the S-NIAH-2/3 with real-world-essay context」。｜修复要求：删除「非真实文本」，按来源改为 needle 落在真实文本/essay 中；并把 uuid/word 表述与最终确定的版本对齐。｜修复：｜复验：

- [轻微·格式] 第 695 行：来源节 h3 作「外部数字与实验条件」，style-guide §1 规定的固定名称为「外部数字与实验条件（N）」。｜引文依据：不适用。｜修复要求：补为「外部数字与实验条件（N）」。｜修复：｜复验：

- [轻微·技术] 第 631 行：「Qwen3.5-397B-A17B 沿用同样的 3:1 混合比例——60 层中 45 层 Gated DeltaNet、15 层带输出门的全注意力」为具体层数与比例，但该 <li> 无 [N] 上标，来源节 N4 只覆盖 Qwen3-Next；数字型论断缺可核对来源。｜引文依据：来源节 N4 段落仅述 Qwen3-Next 的 3:1 与 80B-A3B。｜修复要求：为该条补 [N] 来源（如对应模型 config.json），或把层数细节登记入来源节并加 [N] 引用。｜修复：｜复验：

- [轻微·格式] 第 293 行：示例标记与规范不符并有多余空格。style-guide §4 规定示例按用途标记为「计算示例」「代码示例」或「构造数据」，页面用「构造示例。 取 $d = 2$ 维……」，且「。」后有游离空格。｜引文依据：不适用。｜修复要求：按规范改为「计算示例」等三种标记之一，并删除多余空格。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 5
- 处置：修复。核心结论、全部公式、手算与可运行代码均已复算通过，数字与来源数值一致，无阻断项；但来源所标版本与全部章节号/公式号/表号错配（重要）、三处「原文」引文非逐字（重要）、KDA 公式的 $S$ 形状约定与本页 §2 冲突（重要）须修复。修复范围限于上述位置及直接受影响的引用位置；定义、公式与数字修改后须按 v1 或 v3 之一重新核对来源，代码勿改动（已核对通过）。
