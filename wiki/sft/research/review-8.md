<!-- review-meta
round: 8
page: wiki/sft/index.html
reviewed_content_sha256: 3a502d1e97489485
-->
# SFT（监督微调）审查记录（第 8 轮）

- 页面版本：index.html 工作树哈希 e853d0a346f23f71fce072ec870d5fd4fd50c990
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节，按顺序：引言与「核心问题」；1. 预训练目标与用户目标的错位；2. SFT 的数据与训练目标；3. 手算一条 SFT 样本的损失；4. SFT 在后训练流程中的位置；5. 实践中的边界与误解；来源与范围说明（含全部折叠块与 details）

## 来源核对（本轮逐条回源，写明版本）

- Zhang et al., "Instruction Tuning for Large Language Models: A Survey", arXiv:2308.10792。**按 v10（2025-10-06）核对**：摘要含 "can also be referred to as supervised fine-tuning (SFT)"；摘要脚注含 "unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably"（对应 C1）；摘要含定义 "further training LLMs on a dataset consisting of (instruction, output) pairs ... in a supervised fashion"。§1 含 "the mismatch between the training objective and users' objective"、"LLMs are typically trained on minimizing the contextual word prediction error on large corpora"、用户目标 "follow their instructions helpfully and safely"（对应 C2）；§1 三重收益 "bridges the gap between the next-word prediction objective of LLMs and the users' objective"、"SFT allows for a more controllable and predictable model behavior ..."、"SFT is computationally efficient and can help LLMs rapidly adapt to a specific domain"（对应 C8）。**版本注**：另核 v6（2024-10-16）摘要无该脚注、无 SFT 等价表述；v8（2024-12-01）摘要已含 SFT 等价表述与脚注——与 C1/C8 的版本说明一致。
- Ouyang et al., "InstructGPT", arXiv:2203.02155。§1 "Our main findings"：'outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3'、'despite having over 100x fewer parameters'、'These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data.'、'This result holds true even when we add a few-shot prompt to GPT-3.'（对应 C7）。§3.1 Step 1–3 与 Fig. 2 图注（SFT→RM→PPO；'We use the output of the RM as a scalar reward'）；'Steps 2 and 3 can be iterated continuously'（对应 C3 与图注「循环迭代」）。§3.2 'The SFT dataset contains about 13k training prompts'（对应 N1）。§3.5 'Starting from the SFT model with the final unembedding layer removed'（对应 C3）；'our SFT models overfit on validation loss after 1 epoch'、'training for more epochs helps both the RM score and human preference ratings'、'We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2'（对应 C6/N2）。
- Touvron et al., "Llama 2", arXiv:2307.09288。§3.1 'each sample consists of a prompt and an answer'、'we concatenate all the prompts and answers from the training set'、'A special token is utilized to separate the prompt and answer segments'、'We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt'、'we backpropagate only on answer tokens'；'We stopped annotating SFT after collecting a total of 27,540 annotations.'、'SFT annotations in the order of tens of thousands was enough to achieve a high-quality result.'、'setting aside millions of examples from third-party datasets'、'our results notably improved'；'an initial learning rate of 2×10−5, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens'、'for 2 epochs'、'a cosine learning rate schedule'（对应 C4/C5/N3）。§3.3 GAtt 'set the loss to 0 for all the tokens from the previous turns, including assistant messages'（对应 C4 多轮情形）。
- Radford et al., GPT-2 (2019) §2 式 (1)：p(x)=∏ p(s_n|s_1,…,s_{n−1})，确为链式分解（对应 F1）。
- 数值复算：−ln0.01=4.60517→4.6052；−ln0.05=2.99573→2.9957；−ln0.02=3.91202→3.9120；−ln0.30=1.20397→1.2040；−ln0.90=0.10536→0.1054；mask 版 (1.2040+0.1054)/2=0.6547；不 mask 版 (4.6052+2.9957+3.9120+1.2040+0.1054)/5=12.8223/5=2.5645；指令部分 4.6052+2.9957+3.9120=11.5129，占比 11.5129/12.8223=89.8%。全部与页面一致。
- 链接核对：../../wiki/pretraining/index.html、../../wiki/cross-entropy/index.html 均真实存在；overview.html 与 index.html 相互链接。`.dojo/scripts/validate.py wiki/sft/index.html` 返回 `validation ok`。页面无 `<pre><code>` 可运行代码，代码项不适用。

## 问题

- [轻微·格式] 来源与范围说明「外部数字与实验条件（N）」/ 第 333 行表格：N1（InstructGPT SFT 数据集约 13k 训练 prompts，§3.2）在全页正文与表格中均无 `<sup>[N1]</sup>` 引用（全页 `[N1]` 出现 0 次），N1 成为单向条目，违反 style-guide §6「与来源章节双向对应」｜引文依据：InstructGPT §3.2 "The SFT dataset contains about 13k training prompts"；正文表格「约 $13$k 训练 prompts」处无上标｜修复要求：在第 333 行表格该事实或正文对应处补 `<sup>[N1]</sup>`，使 N1 与正文双向对应｜修复：｜复验：
- [轻微·一致性] dojo:summary（第 7 行）与 overview.html（第 34 行）的损失公式写成 `$\mathcal{L}=-\sum\log p_\theta(y_t\mid x, y_{&lt;t})$`，与正文第 150 行的定义式 `$\mathcal{L}(\theta)=-\frac{1}{|D|}\sum_{(x,y)\in D}\frac{1}{|y|}\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x, y_{&lt;t})$` 同名 $\mathcal{L}$ 却相差归一化因子（$1/|D|$、$1/|y|$），写法不一致｜引文依据：不适用（页面内部一致性，跨正文/summary/overview）｜修复要求：将 summary/overview 的公式改为与正文一致（补 $-\frac{1}{|D|}\sum\frac{1}{|y|}$），或明确标注其为未归一化的单样本形式，使 $\mathcal{L}$ 全页单义｜修复：｜复验：
- [轻微·表述] 引言第 65 行「SFT（…）补上这一课」为口语化/比喻措辞，与概念页中性书面语要求不符｜引文依据：不适用｜修复要求：改为中性表述（如「SFT 针对的正是这一点」），不加未标注的比喻｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。核心定义（C1）、目标错位（C2）、三步流程与 RM 起点（C3）、拼接与损失掩码（C4）、数据质量与数量（C5）、验证损失过拟合与 16 epochs（C6）、1.3B 偏好于 175B（C7）、SFT 收益（C8）、损失公式链式分解（F1）均逐条回源核对、原文片段与关键数值一致；构造示例各概率与损失、89.8% 占比可复算；页面无自相矛盾数字，无未闭合问题块，无 Unicode 数学字符，validate.py 通过。遗留 3 条轻微问题均不影响正确性与主线理解，接受理由：N1 缺上标不影响读者定位来源（N2/N3 已引），公式约简在摘要位置属常见缩写，口语化措辞仅为表达润色。
