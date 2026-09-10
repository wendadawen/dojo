# 因果掩码审查记录（第 2 轮）

- 页面版本：`c93d0bc71a826d2ec3035eb7adb0a7093b9ccd25`（git hash-object wiki/causal-mask/index.html）
- 审查时间：2026-09-10 20:21
- 审查者：独立子代理（未参与写作，未参与第 1 轮审查与修复；本轮未读取 `research/` 下任何文件）
- 已完整阅读章节：标题与 meta/引言 → 核心问题（5 条）→ 常见误解 → 1. 因果掩码是什么规则，解决什么问题（含本章问题）→ 2. 因果掩码如何机械地实现（含本章问题）→ 3. 手算 3-token 例子（含 `展开：`折叠块、`代码：`折叠块及其预期输出、本章问题）→ 4. 训练时并行、推理时 KV-cache 隐含（含本章问题）→ 5. 边界与 NoPE 的结构前提（含本章问题）→ 来源与范围说明（论断与来源 C1–C7、公式与来源 F1–F2、构造示例、辅助解释与类比边界、简化条件及其限制）。`overview.html` 已全文阅读。

## 机械验证结果

命令：`/usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py wiki/causal-mask/index.html`
→ `validation ok`，exit 0。

脚本另核对：

| 项目 | 方法 | 结果 |
|---|---|---|
| 引用编号双向闭合 | 正则抽取正文 `<sup>[..]</sup>` 与来源章节定义 | 引用集 = 定义集 = {C1,C2,C3,C4,C5,C6,C7,F1,F2}；两差集均为空 |
| 相邻双上标 | 匹配 `</sup>\s*<sup>` | 0 处 |
| Unicode 数学字符 | 去 `$$…$$`/`$…$`/`<pre>`/`<code>` 后按 Unicode 类别扫描 Symbol | 仅 UI 图标 `⌂ ◐ ☀ ↑ ✓ ▼`，无数学字符 |
| TAB 字符 | 全文计数 | 0 |
| 残留占位符 | 匹配 待生成/TODO/TBD/占位/placeholder/FIXME 等 | 0 |
| 可运行代码块 | 抽取唯一 `language-python` 块写入 `/tmp/cm_check.py` 实跑，`/usr/bin/python3` returncode 0、stderr 空；与页面 `language-text` 预期输出做字符串比较 | **逐字符一致**（含 `pos 1: [0, -inf, -inf]`、`weights = ['1.0000', '0.0000', '0.0000']  (sum=1.0000)  output = 1.0000`、对照行 `output = 2.0000`） |
| 页面互链 | `index.html` nav → `overview.html`；`overview.html` nav → `index.html` | 双向有效 |
| 前置概念链接 | `wiki/standard-attention/index.html`、`wiki/nope/index.html` | 两文件均存在 |
| 折叠块前缀 | 统计 `<summary>…：` | `解答：`17 个（5 页面级 + 12 章节级）、`展开：`1、`代码：`1，符合规范三种前缀 |
| 用词 | 检索第二人称、第一人称复数、S1/S2 章节代号 | 均为 0 处 |

来源逐条核对（PDF 版 arXiv:1706.03762v7 与 HTML 版、PyTorch 源码、NoPE 论文）：

- **C1**｜§3.1 原文（PDF p.4）："…prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i." —— 标注位置包含引文，✔ 逐字一致。§3.2.3："Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position." ✔
- **C2**｜§3.2.3："We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2." ✔（该句在正文，非脚注）。Figure 2 左图：将 PDF 第 4 页渲染为位图后确认图中存在红色 `Mask (opt.)` 方框 ✔（PDF 文本层无此串，需按图核）。
- **C3**｜PyTorch `torch/nn/modules/transformer.py`：docstring 逐字含 "The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)." ✔；`torch/nn/modules/activation.py` `attn_mask` docstring 逐字含 "For a float mask, the mask values will be added to the attention weight."，`merge_masks` docstring 逐字含 "combined with logical ``or``" ✔；实现为 `merged_mask = attn_mask_expanded + key_padding_mask_expanded`，布尔张量相加即按位或，故页面"浮点掩码相加、布尔掩码按位或"成立 ✔。
- **C4**｜§3："At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next." ✔；Table 1：Self-Attention Sequential Operations = O(1)，Recurrent = O(n) ✔。
- **C5**｜页面自标"属本文的工程推断，非论文原文" ✔，符合规范对推断的标注要求。
- **C6**｜NoPE 论文（Kazemnejad et al., arXiv:2305.19466）摘要末句逐字为 "Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences."，页面引文是该句逐字子串 ✔；但**摘要不含"因果掩码是结构前提"的机制陈述**，该机制出现在正文 §2/§8/附录 C.1（见问题 1）。
- **C7**｜§3.2.3："Each position in the encoder can attend to all positions in the previous layer of the encoder." ✔
- **F1**｜§3.2.1 Eq.(1)："Attention(Q, K, V) = softmax(QKᵀ/√d_k)V" ✔，与 §3.2.3 掩码合并的写法与页面一致。
- **F2**｜由 F1 与可见规则展开，手算与代码复算一致 ✔。
- 任务说明中"页面标注 NoPE 论文 arXiv:2405.18719"与页面不符：页面唯一给出的 NoPE 编号为 **2305.19466**（正确）。经查 2405.18719 是 *Contextual Position Encoding (CoPE)*，与本页主题无关；页面未出现该编号，无问题。

## 问题

- [重要·技术] 来源与范围说明 C6（及正文第 5 章 `<sup>[C6]</sup>` 处）：把"NoPE 隐式编码位置的结构前提"这一机制挂到 NoPE 论文**摘要**上，但摘要只支持"NoPE 无需显式位置编码也可用/可泛化"，不含因果掩码打破置换不变性的机制陈述，读者按标注位置核对找不到该支持。｜引文依据：摘要末句 "Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences."（✔ 存在）；而机制句在正文 §2 "However, decoder-only Transformers with causal attention mask are not permutation invariant and can model sequences even without explicit position information (Tsai et al., 2019)."、§8 "Decoder-only Transformers, due to their causal attention mask, are not order-agnostic and can operate without explicit positional information."、附录 C.1 "relies on the causal attention mask in the decoder-only Transformer and the softmax function to recover absolute positions."｜修复要求：保留摘要句作为"NoPE 有效"的支持，另把 C6 的机制引用定位改为上述正文 §2/§8（或附录 C.1），写明小节号；论断本身成立，不需要删除。｜修复：重写来源与范围说明 C6 段：摘要末句仅保留为「NoPE 无需显式位置编码也能有效并泛化到更长序列」的支持；机制陈述改挂 NoPE 论文（arXiv:2305.19466, v2）正文并写明小节号——§2 Background「decoder-only Transformers with causal attention mask are not permutation invariant and can model sequences even without explicit position information (Tsai et al., 2019)」、§8 Related Work「Decoder-only Transformers, due to their causal attention mask, are not order-agnostic and can operate without explicit positional information」、附录 C.1「relies on the causal attention mask in the decoder-only Transformer and the softmax function to recover absolute positions」。已用 WebFetch 打开 https://arxiv.org/html/2305.19466v2 逐句复核三处原文与所属小节（§2 / §8 / 附录 C.1 Absolute Positional Encoding in NoPE），逐字一致。正文第 5 章 `<sup>[C6]</sup>` 标注位置未动（其指向 C6 定义段，定义段已改到真正含机制陈述的正文）。｜复验：`/usr/bin/python3 .dojo/scripts/validate.py wiki/causal-mask/index.html` → `validation ok`，exit 0；引用双向闭合脚本结果 cited=defined={C1,C2,C3,C4,C5,C6,C7,F1,F2}，`cited - defined = []`、`defined - cited = []`，两差集均为空。
- [轻微·可读性] 第 1 章（"训练时位置 $t$ 的预测本应只依赖 $1..t$"）与 C1 引文：正文与公式用"位置 $t$ 只能 attend 到位置 $1..t$"（含 $t$ 自身），C1 引文说预测只能依赖 "positions less than $i$"，且引文含 "the output embeddings are offset by one position"，但全文未解释这个 offset，读者无法把"$\le t$"与"$< i$"对应，也无法理解第 4 章"一次喂入整条目标序列"为何不含移位输入。｜引文依据："This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."｜修复要求：在第 1 章该引文处补一至两句，说明 target 右移一位后 decoder 输入位置 $i$ 对应输出 token $i-1$，因此"可见位置 $\le i$"与"预测依赖位置 $<i$"相互一致；引用偏移的句子必须给出该解释，不得只引不释。｜修复：在第 1 章该 §3.1 引文段之后新增一段解释 offset 半句（引用文字未改）。先用 WebFetch 复核 https://arxiv.org/html/1706.03762v7 §3.1，确认「the output embeddings are offset by one position」逐字在原文该段内；新增段说明：decoder 输入不是原目标序列，而是右移一位（开头补起始符号）后的序列，输入位置 $i$ 承载目标序列第 $i-1$ 个 token，故「输入位置 $i$ 可 attend 到 $1..i$」与「预测位置 $i$ 只依赖小于 $i$ 的已知输出」是同一件事——可见的输入位置 $1..i$ 承载目标序列第 $0..i-1$ 个 token（第 $0$ 个是起始符号），真正的已生成输出是第 $1..i-1$ 个，正好是「小于 $i$」的那些；并点明第 4 章「一次喂入整条目标序列」喂入的正是这条右移一位的输入序列，掩码保证每个输入位置只看到自己的前缀。｜复验：validate → `validation ok`，exit 0；复验脚本扫公式块之外全文，无 Unicode 数学字符（新增公式 `$i-1$`、`$0..i-1$`、`$1..i$` 全部写在 `$...$` 内）；`</sup>\s*<sup>` 相邻双上标 0 处；TAB 0；残留占位符 0。
- [轻微·技术] 第 2 章末"实现上，精确 $-\infty$ 是数学极限…可用足够大的负数（如 $-10^5$）…被掩位置的 $e^{s_j}$ 也恰好下溢为 $0$"：$-10^5$ 是无来源的具体数值示例，"下溢为 $0$"未给成立条件（当未掩分数与 $-10^5$ 同量级时相加结果不构成大负数，掩码失效），该段虽自称"辅助说明"却未收入来源章节的"辅助解释与类比边界"。｜引文依据：不适用（工程惯例陈述，非论文论断）。｜修复要求：把该段标注为工程惯例示例并补一句成立条件（未掩分数在 $\mathcal{O}(1)$ 量级、远小于该负数的绝对值），或将整段移入"来源与范围说明"的"辅助解释与类比边界"小节。｜修复：采用前半方案。第 2 章末段改写为明确标注「<strong>工程惯例示例</strong>（非论文原文）」，并补成立条件：「未掩分数在 $\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值；否则把 $-10^5$ 加到一个同量级的未掩分数上并不构成足够大的负数，掩码会失效」。同一章问题答案段同步补上该成立条件，避免两处口径不一致。另按要求把该工程惯例示例收入「来源与范围说明 → 辅助解释与类比边界」小节，写明其为工程惯例示例（非论文原文）及成立条件。原引文/机制结论未删改。｜复验：validate → `validation ok`，exit 0；复验脚本抽取唯一 `language-python` 块写入临时文件用 `/usr/bin/python3` 实跑，returncode 0、stderr 空，输出与页面 `language-text` 预期输出逐字符一致（True）；引用双向闭合两差集为空；无 Unicode 数学字符（新增 `$\mathcal{O}(1)$`、`$-10^5$` 均在 `$...$` 内）、无相邻双上标、TAB 0、无占位符。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（关闭问题 1 后方可进入第 3 轮；问题 2、3 为轻微，可在同轮一并修复，若选择接受须在记录中写明理由）

补充说明（不影响结论）：本轮 `validate.py` 通过，引用双向闭合、代码实跑与预期输出逐字符一致、无 Unicode 数学字符与占位符；C1–C5、C7、F1、F2 的来源标注位置均逐字核对通过，仅 C6 的来源定位需修正。
