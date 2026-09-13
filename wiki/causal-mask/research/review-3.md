<!-- review-meta
round: 3
page: wiki/causal-mask/index.html
reviewed_content_sha256: db78ab32a96a5298
-->
# 因果掩码审查记录（第 3 轮）

- 页面版本：`4c3cd373b415240973dfa94a29e2152d18d1022c`（`git hash-object wiki/causal-mask/index.html`）；overview：`9b45526464903daf9f4f3dd287e7bd2f4e178c10`
- 审查时间：2026-09-10 20:47 CST
- 审查者：编排者派发的独立审查者（未参与写作，未读取 `wiki/causal-mask/research/` 下任何文件）
- 已完整阅读章节：核心问题、常见误解、1. 因果掩码是什么规则，解决什么问题、2. 因果掩码如何机械地实现、3. 手算 3-token 例子、4. 训练时并行、推理时 KV-cache 隐含、5. 边界与 NoPE 的结构前提、来源与范围说明（含全部 19 个 `<details>` 折叠块，其中 17 个「解答：」、1 个「展开：」、1 个「代码：」）；并完整阅读 `overview.html`

## 机械验证结果

命令：`/usr/bin/python3 /Users/wendadawen/code/dojo/.dojo/scripts/validate.py wiki/causal-mask/index.html`
结果：`validation ok: ...`，退出码 0。

| 项 | 结果 |
|---|---|
| validate.py | 成功（exit=0） |
| 引用双向闭合 | 正文引用 `[C1,C2,C3,C4,C5,C6,C7,F1,F2]`，来源章节定义集合完全相同，无单侧标签；`<sup>` 开/闭 31/31 |
| 相邻双上标 | `</sup>\s*<sup>` 出现 0 次 |
| Unicode 数学字符 | index 0 个；overview 仅箭头 `←`/`→`（导航符号，非数学） |
| TAB | index、overview 均 0 |
| 占位符 | 无「待生成/TODO/TBD/占位/FIXME/XXX/Lorem」 |
| 可运行代码块 | 见下「代码实跑」 |
| `<head>` 五项 | `description`（纯文本）、`dojo:summary`（含 `$...$`）、`dojo:type=concept`、`dojo:topics=注意力机制, 模型结构`、`dojo:tag=注意力机制` 均存在 |
| overview↔index 互链 | index nav → `overview.html`；overview nav → `index.html` |
| 前置概念链接 | `../../wiki/standard-attention/index.html`、`../../wiki/nope/index.html` 均存在 |
| 结构 | h2 编号 1–5 连续 + 未编号「核心问题/常见误解/来源与范围说明」；5 个正文 h2 各含未编号 h3「本章问题」；来源章节 h3 用固定命名，无 N 内容故正确省略「外部数字与实验条件（N）」 |

**代码实跑**：抽取 index.html 中 `language-python` 代码块实跑，stdout 与页面「预期输出」逐字符相同（含 `pos 1..3` 权重 `1.0000/0.5000/0.0000`、`0.3333`，`sum=1.0000`，输出 `1.0000/1.5000/2.0000`，双向对照三行 `2.0000`）。判定一致。

### 来源核对（check.md §2.2 四步）

| 标注 | 页面标注位置 | 定位结果与原文片段/关键数值 |
|---|---|---|
| [C1] | §3.1 | 命中：「We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.」（页面引文含原文的 "combined with fact" 写法，逐字一致） |
| [C1] | §3.2.3 | 命中：「self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.」 |
| [C2] | §3.2.3 | 命中：「We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2.」 |
| [C2] | Figure 2 左图 | 下载 `1706.03762v7/Figures/ModalNet-19.png` 并目视：方框含 `MatMul / SoftMax / Mask (opt.) / Scale / MatMul`，输入 `Q K V`；`Mask (opt.)` 位于 `Scale` 与 `SoftMax` 之间，支持「施加在 softmax 输入」 |
| [C3] | `torch/nn/modules/transformer.py` | 命中：`_generate_square_subsequent_mask` docstring「The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).」；实现 `torch.triu(torch.full((sz,sz), float("-inf")), diagonal=1)` 亦与上三角形状一致 |
| [C3] | `torch/nn/modules/activation.py` | 命中：「If a FloatTensor is provided, it will be added to the attention weight.」（页面引文 "For a float mask, the mask values will be added to the attention weight" 见 attn_mask 参数 docstring，逐字一致）；`merge_masks` docstring「combined with logical ``or``」；实现中两掩码合成为 `attn_mask_expanded + key_padding_mask_expanded`（布尔加法饱和，等价按位或） |
| [C4] | §3 | 命中：「At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.」 |
| [C4] | §4 / Table 1 | 命中 Table 1：Self-Attention「Sequential Operations O(1)」、Recurrent「O(n)」；正文「a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations.」 |
| [C5] | §3 推理侧推论 | 页面已明确标注「属本文的工程推断，非论文原文」，符合 §2.2 第 7 条处理结果 |
| [C6] | NoPE arXiv:2305.19466 摘要 | 命中末句：「explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.」 |
| [C6] | 同上 §2 Background | 命中：「decoder-only Transformers with causal attention mask are not permutation invariant and can model sequences even without explicit position information (Tsai et al., 2019).」 |
| [C6] | 同上 §8 Related Work | 命中：「Decoder-only Transformers, due to their causal attention mask, are not order-agnostic and can operate without explicit positional information.」 |
| [C6] | 同上 Appendix C.1 | 命中：「...relies on the causal attention mask in the decoder-only Transformer and the softmax function to recover absolute positions.」 |
| [C7] | §3.2.3 | 命中：「Each position in the encoder can attend to all positions in the previous layer of the encoder.」 |
| [F1] | §3.2.1 Eq.(1) + §3.2.3 | Eq.(1) 原文为 `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V`，不含 `+M`；页面 F1 明言为「Eq.(1) + 掩码（C2/C3）的合并」，定位与表述一致 |
| [F2] | F1 的展开 | `o_t=Σ_{j=1}^{t} α_{t,j} v_j`，求和上界 `t` 由 C1 可见规则给出；页面已标注为 F1 的展开，非独立来源公式 |
| 手算数值 | — | 位置 1 权重 `[1,0,0]` 输出 `1`；位置 2 `[1/2,1/2,0]` 输出 `1.5`；位置 3 `[1/3,1/3,1/3]` 输出 `2`；双向对照三位置均 `2`。复算与代码实跑均一致 |

`overview.html` 与 index 结论一致（只用于 decoder、与填充掩码正交、NoPE 结构前提、训练并行 `O(1)`、推理隐含），无与 index 冲突的表述。

## 问题

- [轻微·可读性] 第 5 章正文「深层作用：NoPE 的结构前提」段（`id="boundaries-and-nope"` 章，约 index.html:1031）及该章「本章问题」第 2 问解答（约 index.html:1048）：两处把 3-token 手算例子写成「本章开头的 $3$-token 例子」「本章的 $3$-token 例子」，但该例实际在第 3 章「手算 $3$-token 例子」，第 5 章开头并无此例。｜引文依据：不适用（页面内部交叉引用错误）｜修复要求：把这两处的「本章（开头）的 $3$-token 例子」改为按 style-guide §1 的章节标题引用，如「第 3 章『手算 $3$-token 例子』中的 $3$-token 例子」｜修复：两处均改为标题引用——index.html:1031「本章开头的 $3$-token 例子」与 index.html:1048「本章的 $3$-token 例子」，按 style-guide §1 改为「"手算 $3$-token 例子"一章中的 $3$-token 例子」（未沿用「第 3 章」编号，与同页 index.html:728 既有写法一致）；脚本 `research/fix3.py` 对每个 old 串断言命中 1 次。｜复验：全文 `本章开头`/`本章的 $3$-token 例子` 命中 0；第 5 章两处现均指向第 3 章标题；`validate.py` exit=0。
- [轻微·格式] 第 1 章「引文里还有半句需要解释」段（约 index.html:738）：用「第 4 章」编号引用章节，未按 style-guide §1「正文引用其他章节时使用章节标题」的固定写法（同页其余 7 处交叉引用均使用章节标题）。｜引文依据：不适用｜修复要求：改为「『训练时并行、推理时 KV-cache 隐含』一章」｜修复：index.html:738「第 4 章"一次喂入整条目标序列"」改为「"训练时并行、推理时 KV-cache 隐含"一章的"一次喂入整条目标序列"」；核对时另发现来源章节 index.html:1079 亦有一处同类编号引用「第 2 章『工程中可用足够大的负数…』」，一并改为「"因果掩码如何机械地实现"一章的『工程中可用足够大的负数…』」。｜复验：全文正则 `第 ?[0-9]+ ?章` 命中 0，交叉引用现全部使用章节标题；相邻双上标 0 处。
- [轻微·技术] 第 2 章符号说明第 1 项（约 index.html:785）：写 `$Q,K,V\in\mathbb{R}^{n\times d}$` 并把 `$d$` 定义为模型维度，与同一列表第 2 项 `$d_k$`（key 的维度）冲突——标准注意力中 `$Q,K\in\mathbb{R}^{n\times d_k}$`、`$V\in\mathbb{R}^{n\times d_v}$`，`$d_k$` 才是缩放所用维度。｜引文依据：论文 Eq.(1) 分母为 `√dₖ`（§3.2.1），且 §3.2.2 给出多头的 `$d_k=d_v=d_{model}/h$`｜修复要求：改为 `$Q,K\in\mathbb{R}^{n\times d_k}$`、`$V\in\mathbb{R}^{n\times d_v}$`，或显式写出 `$d_k=d_v=d$` 的简化假设后再沿用 `$n\times d$`｜修复：index.html:785 按论文原文改为 `$Q,K\in\mathbb{R}^{n\times d_k}$`、`$V\in\mathbb{R}^{n\times d_v}$`，并把「$n$ 为序列长度、$d$ 为模型维度」改为「$n$ 为序列长度，$d_k$、$d_v$ 分别为 key、value 的维度」。依据：arXiv:1706.03762v7 §3.2.1「The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$」；§3.2.2「$d_k=d_v=d_{model}/h$」。｜复验：与同列表第 2 项 `$d_k$` 的「key 的维度」定义一致、不再冲突；全页无 Unicode 数学字符；`validate.py` exit=0。
- [轻微·技术] 第 1 章同一段（约 index.html:738）：对「the output embeddings are offset by one position」的解释「右移一位（开头补一个起始符号）」未给出出处、也未在「辅助解释与类比边界」标注为解释性推断（该节仅覆盖「抄答案」「排列等变」「$-10^5$」三项）。｜引文依据：Figure 1（`Figures/ModalNet-21.png`）decoder 输入标注为「Outputs (shifted right)」，支持「右移一位」；「预先补起始符号」为常规工程实现细节，论文正文未出现该措辞｜修复要求：把 Figure 1 的「Outputs (shifted right)」补为该段出处，或在「辅助解释与类比边界」中标注「起始符号」为工程实现惯例｜修复：两处都做——index.html:738 为「the output embeddings are offset by one position」补 `<sup>[C1]</sup>` 出处标注，并在正文补出 Figure 1 图内标注「Outputs (shifted right)」、同时括注「开头补一个起始符号」为工程实现惯例；来源章节 [C1] 条目（index.html:1063）补 Figure 1 出处；「辅助解释与类比边界」（index.html:1079）补入该工程惯例说明。｜复验：下载 arXiv:1706.03762v7 的 `Figures/ModalNet-21.png` 目视确认 decoder 输入标注为「Outputs (shifted right)」（与图内 Masked Multi-Head Attention、Positional Encoding 标注一致），引文落位准确；引用双向闭合 C1–C7/F1–F2 定义集合一致，`<sup>` 开/闭 32/32。
- [轻微·格式] §5 发布条件「递归生成的前置概念页已完成各自质检」：`wiki/standard-attention/research/` 与 `wiki/nope/research/` 目录内仅有 `review.md`、`review-2.md`，无第三轮记录；对照 `wiki/cross-entropy/research/`、`wiki/rope/research/` 均有 `review-1/2/3.md`。｜引文依据：不适用（仅目录清单观察，未读取前置页 review 内容）｜修复要求：由编排者确认 `standard-attention`、`nope` 的第三轮审查是否已完成；若未完成，按 §5 该项在发布前不满足，需先补齐｜修复：本条为发布流程确认项，页面无对应文字可改，未改动 index.html/overview.html；核对 `wiki/standard-attention/research/`、`wiki/nope/research/` 目录清单，二者均仅有 `review.md`、`review-2.md`，无第三轮记录（对照 `wiki/cross-entropy/research/`、`wiki/rope/research/` 有 `review-1/2/3.md`）。｜复验：两前置页第三轮质检记录仍缺失，§5「递归生成的前置概念页已完成各自质检」暂不满足，须由编排者安排前置页补齐第三轮独立审查后方可发布；属本页范围外的待办项，非本页可修复。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：可发布（附 1 项编排者确认）

本轮未发现阻断或重要问题。全部来源论断均按 §2.2 四步定位到原文片段并记录（C1–C7、F1、F2 及 Table 1、Figure 1/2、PyTorch 源码均命中），无扩大来源的表述；[C5] 与「$-10^5$」已按规范降级为明确标注的工程推断，`overview.html` 与 index 结论一致。

对 check.md §5 发布条件的逐条判定：

| §5 条件 | 判定 |
|---|---|
| 三轮审查均完成且每轮为独立审查者 | 本轮为第 3 轮独立审查；前两轮按编排记录存在（未读取以保持独立性）→ 满足 |
| 每条来源论断有引文依据，无法核对者已删除或降级 | 满足（见「来源核对」表） |
| 所有阻断与重要问题均已关闭 | 本轮 0 阻断 0 重要 → 满足 |
| 遗留轻微问题有明确接受理由 | 本轮 5 条轻微，修复要求均可复验；若不修复，接受理由为「不影响机制结论与数值复算」→ 满足（须在发布记录中写明） |
| 全部学习目标由正文章节完整回答 | 满足（核心问题 5 条分别由第 1–5 章回答） |
| 两级问题均有解答折叠块 | 满足（核心问题 5 个 + 章节问题 12 个，`<summary>` 前缀均为「解答：」） |
| 数学符号全部 LaTeX，结构图为 HTML/内联 SVG | 满足（无 Unicode 数学字符，无等宽框线图，validate.py 通过） |
| validate.py 返回成功 | 满足（exit=0） |
| 可运行代码结果与页面描述一致 | 满足（实跑输出逐字符相同） |
| 关键论断与数字已重新核对来源 | 满足 |
| `<head>` 五项元数据有效 | 满足 |
| overview 与 index 相互链接 | 满足 |
| 引用的概念链接有效或明确占位 | 满足（standard-attention、nope 页面均存在） |
| 递归生成的前置概念页已完成各自质检 | **待确认**：前置页目录内缺第三轮审查记录，需编排者核实 |

据此：本页内容层面满足 §5 各项发布条件，可发布；唯一未在本次独立核对范围内闭合的是「前置概念页 third 轮质检记录」一项，需编排者确认后落发布记录。若需完全齐备 §5，建议同时修复上述 5 条轻微问题（均为可直接定位的一处或两处文字改动，不影响本页结论）。
