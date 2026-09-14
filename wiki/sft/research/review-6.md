<!-- review-meta
round: 6
page: wiki/sft/index.html
reviewed_content_sha256: bea6668691570ecc
-->
# SFT 审查记录（第 6 轮）

- 页面版本：f0be929561ad837ace9bba3894ec80abf78d7d43（wiki/sft/index.html 工作树内容哈希）
- 审查时间：2026-09-14 17:42
- 审查者：独立子代理（未参与写作与前序审查）
- 已完整阅读章节：核心问题、1. 预训练目标与用户目标的错位、2. SFT 的数据与训练目标、3. 手算一条 SFT 样本的损失、4. SFT 在后训练流程中的位置、5. 实践中的边界与误解（5.1–5.4）、来源与范围说明（论断与来源 C / 公式与来源 F / 外部数字与实验条件 N / 构造示例 / 辅助解释与类比边界 / 简化条件及其限制），含全部本章问题折叠块与「展开：本例五个位置的完整对数计算」折叠块。
- 核对所用来源版本：arXiv:2308.10792（v1–v6 的 PDF 文本与 arXiv HTML v6、v1）；arXiv:2203.02155（ar5iv HTML 全文）；arXiv:2307.09288（ar5iv HTML 全文）；arXiv:1904.10509（PDF）。
- 机械验证：`.dojo/scripts/validate.py wiki/sft/index.html` 返回 `validation ok`；无 Unicode 数学字符直接出现；无「待生成」占位；`../../wiki/pretraining/index.html`、`../../wiki/cross-entropy/index.html`、`../../index.html`、`wiki/sft/overview.html` 均存在且互链；无带 `$...$` 的 img alt。图（后训练三步流程图）为 HTML 结构，无 math，明暗主题下可读。

## 问题

- [阻断·技术] 来源与范围说明「论断与来源（C）」C1，及正文第 65 行「综述将该术语与 instruction tuning「指令微调」互换使用[C1]」：C1 在引号中给出的脚注引文「In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably」在被引来源 arXiv:2308.10792 的任何版本中都定位不到；且该综述通篇使用「IT」，"SFT" 仅出现一次（InstructGPT 小节内的 "supervised fine-tuning (SFT)"），并未将二者互换使用，故此项为「来源不支持」的来源论断。｜引文依据：对 2308.10792 v1–v6 的 PDF（pdftotext 提取）检索 "interchangeab"/"unless" 命中 0；arXiv HTML v6 中脚注块仅有两处，内容为作者单位/邮箱/项目页（"♠Zhejiang University, ♣Shannon.AI, ▲Nanyang Technological University, ⧫Amazon；Email: sy_zhang@zju.edu.cn；Project page…"），无术语互换脚注。摘要原文只有「Instruction tuning refers to the process of further training LLMs on a dataset consisting of (INSTRUCTION, OUTPUT) pairs in a supervised fashion」。｜修复要求：删除 C1 中引号内的脚注引文及第 65 行「综述将该术语与 instruction tuning「指令微调」互换使用」的表述（或改为不援引该来源、明确标注为通行用法）；不得保留定位不到的引文。｜修复：｜复验：

- [重要·技术] 来源与范围说明「论断与来源（C）」C8：引号内文本把来源原文的「IT」改成了「SFT」，构成非逐字引用。｜引文依据：2308.10792 §1 原文（PDF v6）："The benefits of IT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) IT allows for a more controllable and predictable model behavior compared to standard LLMs…；and (3) IT is computationally efficient…"。页面 C8 写作 "The benefits of SFT are threefold: … (2) SFT allows for a more controllable and predictable model behavior…"，两处 IT→SFT 系替换。｜修复要求：引文还原为源文「IT」，或用方括号/省略号标出改动，或在引文外以正文说明「该综述以 IT 指称、本页按 SFT 表述」。｜修复：｜复验：

- [轻微·技术] 第 2 章 F1 公式后的符号表（第 152–156 行）：公式 $\mathcal{L}(\theta) = -\frac{1}{|D|}\sum_{(x,y)\in D}\frac{1}{|y|}\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x, y_{<t})$ 用到 $|D|$、$|y|$，但符号表只定义了 $D$、$y_t$、$y_{<t}$、$p_\theta$；而 $|y|$（回答 token 数）正是第 3 章 mask 分母讨论的关键量。｜引文依据：不适用。｜修复要求：在符号表中补一行定义 $|y|$ 为回答的 token 数（$|D|$ 为样本数）。｜修复：｜复验：

- [轻微·表述] 第 2 章末（第 162 行）「为什么 mask？回到公式：…」与第 4 章（第 284 行）「为什么必须有分工？回到损失的表达能力。」为同一「为什么…？回到…」过渡句式复用，属章节过渡固定句式。｜引文依据：不适用。｜修复要求：改写其中一处过渡，避免同一句式重复。｜修复：｜复验：

## 已核对且与来源一致（无问题项，供复验参考）

- 手算全部可复算：表格五个 $-\ln p$ 分别为 4.6052 / 2.9957 / 3.9120 / 1.2040 / 0.1054；mask 平均 0.6547、不 mask 平均 2.5645、指令部分合计 11.5129、占比 89.8%，与正文、核心问题答案、本章问题答案、折叠块「展开：本例五个位置的完整对数计算」四处一致（折叠块中「mask 版合计 1.3093（按精确值求和后四舍五入）」与表内两位相加得 1.3094 的差异已在文中说明，复算成立）。
- C2：2308.10792 §1「the mismatch between the training objective and users' objective… users want the model to "follow their instructions helpfully and safely"」，与正文第 110 行一致。
- C3：2203.02155 §3.1「Step 1: Collect demonstration data, and train a supervised policy…Step 3: Optimize a policy against the reward model using PPO…Steps 2 and 3 can be iterated continuously」；Fig. 2 图注「the three steps of our method: (1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)」；§3.5「Starting from the SFT model with the final unembedding layer removed」，与正文与图注一致。
- C4：2307.09288 §3.1「we concatenate all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments. We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens. Finally, we fine-tune the model for 2 epochs.」；§3.3（System Message for Multi-Turn Consistency）「we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages」，页面对 GAtt 语境的限定与标注正确。
- C5：2307.09288 §3.1「SFT annotations in the order of tens of thousands was enough… stopped annotating SFT after collecting a total of 27,540 annotations」「By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved」。
- C6 / N2：2203.02155 §3.5「We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2… our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings」。
- C7：2203.02155 §1「outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3」。
- N1：2203.02155 §3.2「The SFT dataset contains about 13k training prompts」。
- N3：2307.09288 §3.1「cosine learning rate schedule with an initial learning rate of 2×10^-5, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens」，与表格一致。
- F1：链式分解引用 GPT-2 §2 式 (1)，1904.10509 原文「p(x) = ∏ p(xi | x1, …, xi−1; θ) (1)」成立。
- 结构/功能：核心问题 5 条、每章「本章问题」均有解答折叠块且指向完整论证章节；后训练三步图为 HTML 结构非字符框线图；overview.html 与 index.html 互链且数字（27,540 / 1 epoch / 16 epochs / 1.3B / 175B）一致。

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 2
- 处置：修复
