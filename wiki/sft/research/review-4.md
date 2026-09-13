<!-- review-meta
round: 4
page: wiki/sft/index.html
reviewed_content_sha256: bd4896f685712d08
-->
# SFT 审查记录（第 4 轮）

- 页面版本：05faf9a9ab6348f2c676544eb62d061304b093e8
- 审查时间：2026-09-13 19:48
- 审查者：独立子代理（未参与写作，未读取本页 research/ 记录）
- 已完整阅读章节：页面级「核心问题」→ 1. 预训练目标与用户目标的错位（含本章问题）→ 2. SFT 的数据与训练目标（含本章问题）→ 3. 手算一条 SFT 样本的损失（含本章问题）→ 4. SFT 在后训练流程中的位置（含本章问题）→ 5. 实践中的边界与误解（5.1–5.4，含本章问题）→ 来源与范围说明

## 来源核对（已打开来源、定位到标注位置并摘录原文）

- Zhang et al., arXiv:2308.10792 摘要 / §1：脚注 1 原文 "In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably."；§1 原文 "the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'"；§1 原文 "The benefits of IT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) IT allows for a more controllable and predictable model behavior … and (3) IT is computationally efficient and can help LLMs rapidly adapt to a specific domain without extensive retraining or architectural changes."。页面 C1、C2、C8 及正文 §1 表述与之一致。
- Ouyang et al., arXiv:2203.02155：§1 "Our main findings" 原文 "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 …"（注：摘要写 "100x fewer"，§1 写 "over 100x fewer"，页面 C7 引文与 §1 逐字一致）；§3.2 原文 "The SFT dataset contains about 13k training prompts"；§3.5 原文 "We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2." / "we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings" / "Starting from the SFT model with the final unembedding layer removed"；§3.1 原文 "Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best policy …"。页面 C3、C6、C7、N1、N2 与之一致。
- Touvron et al., arXiv:2307.09288：§3.1 原文 "We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens." / "we fine-tune the model for 2 epochs" / "a cosine learning rate schedule with an initial learning rate of 2×10−5, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens" / "SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations." / "By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved."；§3.3（System Message for Multi-Turn Consistency / GAtt）原文 "we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages"。页面 C4、C5、N3 与之一致。
- 手算复算（自然对数）：−ln0.01=4.6052、−ln0.05=2.9957、−ln0.02=3.9120、−ln0.30=1.2040、−ln0.90=0.1054；mask 版 (1.2040+0.1054)/2=0.6547；不 mask 版 12.8223/5=2.5645；指令合计 4.6052+2.9957+3.9120=11.5129，占比 11.5129/12.8223≈89.8%。全部与页面一致。
- 机械项：`python3 .dojo/scripts/validate.py wiki/sft/index.html` 返回 "validation ok"；正文无 Unicode 数学字符、无「（待生成）」占位、无 research/ 残留路径；前置概念链接 ../../wiki/pretraining/index.html 与 ../../wiki/cross-entropy/index.html 均存在；overview.html 与 index.html 互链；结构图为 HTML 结构（无等宽字符框线图）。

## 问题

- [轻微·表述] 正文「2. SFT 的数据与训练目标」首段：元话语「有了动机，现在看 SFT 到底用什么、怎么训。」以引导语替代内容陈述。｜引文依据：不适用｜修复要求：删除「有了动机，现在看」这类导向语，改为直接陈述该章内容（如「SFT 的数据形态由综述给出：…」）。｜修复：｜复验：
- [轻微·表述] 正文「2. SFT 的数据与训练目标」第 4 段首句：「现在把贯穿示例定下来。」为元话语。｜引文依据：不适用｜修复要求：改为直接引入（如「贯穿后续计算的构造样本：…」），去掉「现在…定下来」的编排语。｜修复：｜复验：
- [轻微·表述] 开篇引言段末句：「本文说明 SFT 为什么必要、数据与训练目标的具体形态、一条样本的损失如何手算、它在后训练流程中的位置，以及实践中的边界与常见误解。」以「本文」为主语的路线图式元话语。｜引文依据：不适用｜修复要求：删除该句，或改写为不含「本文」指代的内容陈述。｜修复：｜复验：
- [轻微·表述] 正文「2. SFT 的数据与训练目标」第 3 段：「紧接着是本页最关键的一句原文」——自我指代（「本页」）＋临场评价（「最关键」）。｜引文依据：不适用｜修复要求：改为「Llama 2 §3.1 原文：…」，去掉「本页」「最关键」两处。｜修复：｜复验：
- [轻微·表述] 正文「2. …」末句「模型对每个位置的下一 token 分布在下一章给出并手算。」与「5. 实践中的边界与误解」首句「最后处理四个使用 SFT 时最容易做错决策的点。前三个有论文原文支撑，第四个来自 InstructGPT 的评测结论。」——章际导航与编排式元话语。｜引文依据：不适用｜修复要求：删除或改写为内容陈述（如直接说「本节讨论 SFT 使用的四个容易误判的点」）。｜修复：｜复验：
- [轻微·技术] 「5. 实践中的边界与误解」首句称「前三个有论文原文支撑，第四个来自 InstructGPT 的评测结论」，但 5.1「SFT 不限定参数更新方式」全节无任何 [C]/[N] 引用（对照：5.2 标注 [C5][N3]，5.3 标注 [C6][N2]，5.4 标注 [C7]），页内自相矛盾。｜引文依据：5.1 正文 "SFT 刻画的是数据（标注的指令-回答对）与目标（模仿示范的监督损失），不规定参数怎么更新" 处无引用标注；5.2 处标注 [C5][N3]、5.3 处标注 [C6][N2]（页内自证）。｜修复要求：或为 5.1 补可定位的来源引用（综述定义不含参数更新方式），或改写首句使其与各节实际引用情况一致。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：可发布。技术维度（定义与机制、公式与推导、数字与来源、构造示例与来源事实的区分、简化条件、不确定信息处理、页面链接、公式书写、图示、问题块）经逐条回源核对未发现阻断或重要问题：核心数值（13k prompts、16 epochs、residual dropout 0.2、27,540 条、2 epochs、2×10⁻⁵、weight decay 0.1、batch size 64、序列长度 4096、1.3B/175B）与三篇来源逐字一致，手算全部可复算且自洽，构造示例已在「来源与范围说明」中明确标注。遗留 6 条轻微问题均为表述/页内自洽层面，不影响正确性与主线理解，建议随本轮修复一并关闭。