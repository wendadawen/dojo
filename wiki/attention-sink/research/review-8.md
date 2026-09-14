<!-- review-meta
round: 8
page: wiki/attention-sink/index.html
reviewed_content_sha256: 7cb866cb7341eb77
-->
# 注意力汇聚点审查记录（第 8 轮）

- 页面版本：5654345879d3（工作树与 HEAD 一致，git status 干净）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题、常见误解、1. 现象：开头的位置拿到远超其重要性的注意力、2. 成因：softmax 必须把注意力分配完、3. 两种做法：留下位置，或留一个参数（含 3.1 保留初始位置的 KV、3.2 每个头一个可学习标量、展开：参数化汇聚点的解析解（构造示例）、代码：份额随汇聚点标量的变化）、4. 对缓存与推理意味着什么（含 4.1/4.2/4.3）、来源与范围说明、overview.html 全文。折叠块内容已逐条阅读。

## 核对依据（本轮实测/回源）

- 份额公式数值：本地执行页面代码，输出 `0.78% / 5.46% / 53.69% / 95.88%`，与正文、summary、overview、核心问题解答完全一致（`e^2` 分母 9.389056，与 `≈9.39` 一致）。
- StreamingLLM（arXiv:2309.17453，ar5iv 全文）逐句核对：§3.1 / Table 1「0+1024=5158.07、4+1020=5.40、4"\n"+1020=5.60」、Table 2 标题「Introducing four initial tokens generally suffices; further additions have diminishing returns.」、Table 3「Learnable Sink 1+1023=18.01，Vanilla 2+1022=18.05」、§4.2「two language models, each with 160 million parameters ... Pythia-160M codebase」、§1「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」、§3.2「focuses on positions within the cache rather than those in the original text」、附录 A「does not extend the models' context window or enhance their long-term memory capabilities」——均与页面表述一致。
- DeepSeek 官方源码（HF 仓库 `inference/model.py`、`inference/kernel.py`）：`self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))` 与 [C4] 逐字一致；`sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])` 证实「循环结束后把 sink 项加到分母、与可见槽位共享同一 max」；sink 从未进入 `reduce_max` 的累积，证实「不参与行最大值」；kernel 注释的 finite lower bound 对应「整行无有效索引」边界。DSparkAttention 继承 Attention，证实 MTP 层同带该参数（40+3=43）。
- 官方 `config.json`：`num_attention_heads=64`、`head_dim=512`、`sliding_window=128`、`index_topk=512`——与「形状 [64]」「窗口大小 128」一致；HF 仓库分片 `model-00001-of-00048` 证实 [N4]「48 个分片头」。
- `validate.py wiki/attention-sink/index.html` 返回 `validation ok`；正文无 Unicode 数学字符、无第一/第二人称、无占位符、无 `<img>/<svg>`（无图，图示项不适用）。

## 问题

- [轻微·表述] 核心问题 1 解答摘要（第 76 行）与「常见误解」首条（第 106 行）：两处用「起作用的是绝对位置」概括汇聚点机制，而 3.1 节第 200 行写明该方法「决定模型看到的是『缓存内的相对距离』，而不是原文中的绝对距离」。同一页对「绝对/相对位置」给出相反口径，读者可能误以为汇聚点必须落在原序列的绝对下标上。｜引文依据：论文 §3.2「StreamingLLM focuses on positions within the cache rather than those in the original text.」（ar5iv 全文核对）；论文与摘要均未出现「absolute position」表述。｜修复要求：把这两处的「绝对位置」改为不与「绝对距离」冲突的表述，如「序列开头的位置（对所有后续 query 可见）」，并与第 1 章正文「这些位置在序列最前面、对所有后续位置可见」保持一致。｜修复：｜复验：
- [轻微·来源] 来源说明 [C1]（第 338 行）：定位标注为「§3.1 与 Figure 2」，但该条的第二句引文「a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task」实际位于论文第 1 节 Introduction（下一句为「We term these tokens "attention sinks".」），§3.1 内不含该句。照标注到 §3.1 定位会找不到该引文。｜引文依据：ar5iv 全文检索确认 §3.1 不含「surprisingly large amount of attention」；该句在 §1 Introduction「To understand the failure of window attention...」之后。｜修复要求：为 [C1] 补上第 1 节定位（例如「§1 引言与 §3.1；Figure 2」），或对该条引文单独标注所在章节，使每句引文都能在标注位置找到。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 2
- 处置：修复（仅两处轻微项，均为表述/定位层面）。

本轮说明：页面全部事实性论断、数字、引文与代码输出均回源核对通过——StreamingLLM 的引文（含 Table 1/2/3 数值与附录 A）逐句与论文原文一致；DeepSeek-V4.1-Flash 的 `attn_sink` 定义、分母构造、行最大值语义与参数形状均与官方 `model.py`/`kernel.py`/`config.json` 逐字一致；份额公式可复算且正文/summary/overview/图注（本页无图）四处数值统一。无可判定的元话语、会话指代、调试叙事或 AI 拼接腔。故不报告阻断与重要问题，仅保留上述两条轻微项。
