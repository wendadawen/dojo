# SFT 监督微调审查记录（第 3 轮）

- 页面版本：baa6551262d02763b380c644457b18348eeeb6d4（wiki/sft/index.html 工作树哈希）
- 审查时间：2026-08-25 16:21
- 审查者：编排者派发的独立审查者（未参与写作与前两轮审查）
- 已完整阅读章节（index.html，按顺序）：head/meta 与开头场景 → 核心问题（5 题）→ 1. 预训练目标与用户目标的错位（含本章问题）→ 2. SFT 的数据与训练目标（含公式、样本表、本章问题）→ 3. 手算一条 SFT 样本的损失（含展开折叠块、本章问题）→ 4. SFT 在后训练流程中的位置（含流程图、本章问题）→ 5. 实践中的边界与误解（5.1–5.4、训练事实表、本章问题）→ 来源与范围说明（C/F/N/构造示例/类比边界/简化条件）。overview.html 全文（它是什么/为什么需要它/核心机制/关键结论与边界）。

## 来源论断核对（每条含引文依据）

- C1（SFT 定义与术语互换）✅：Zhang et al. 2308.10792v10 摘要 "Instruction tuning refers to the process of further training LLMs on a dataset consisting of (instruction, output) pairs in a supervised fashion"；脚注 1 "In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably."。页面表述一致。
- C2（目标错位）✅：同文 §1 "One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'"。页面表述一致。
- C3（三步流程与 RM 起点）✅：InstructGPT §3.1 "We then apply the following three steps (Figure 2)" 及 Step 1/2/3 各段；"Steps 2 and 3 can be iterated continuously"；§3.5 "Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward."。图 caption 的循环迭代与「初始策略」表述均有依据。
- C4（拼接与损失掩码）✅：Llama 2 §3.1 Fine-Tuning Details "To ensure the model sequence length is properly filled, we concatenate all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments. We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens."；§3.3 多轮见问题 1（引文准确但正文归因有扩大）。
- C5（数据质量重于数量）✅：Llama 2 §3.1 Quality Is All You Need "We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations."；"By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved."。
- C6（过拟合与继续训练）✅：InstructGPT §3.5 "we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting."。
- C7（1.3B 偏好于 175B）✅：InstructGPT §1 "On our test set, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions."。页面「小一百多倍/少一百多倍」与 175/1.3≈134.6 相符。
- C8（SFT 收益）⚠️ 见问题 2：原文 §1 "The benefits of SFT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) SFT allows for a more controllable and predictable model behavior...; and (3) SFT is computationally efficient and can help LLMs rapidly adapt to a specific domain..."。页面所引两条均准确，但未标明是三条中的两条。
- F1（损失公式链式分解）✅：GPT-2（Radford et al., 2019）§2 式 (1) "p(x) = ∏ p(s_n | s_1, ..., s_{n−1})"（"it is common to factorize the joint probabilities over symbols as the product of conditional probabilities"）。页面公式形式由 C1+C4 组合得出且已如实标注。
- N1 ✅：InstructGPT §3.2 "The SFT dataset contains about 13k training prompts (from the API and labeler-written)"。
- N2 ✅：InstructGPT §3.5 "We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2."。
- N3 ✅：Llama 2 §3.1 "we use a cosine learning rate schedule with an initial learning rate of 2×10⁻⁵, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens"；"we fine-tune the model for 2 epochs."。

## 数值复算（Python，全部通过）

- 逐位置损失：−ln(0.01)=4.6052，−ln(0.05)=2.9957，−ln(0.02)=3.9120，−ln(0.30)=1.2040，−ln(0.90)=0.1054 ✅
- 指令合计 11.5129、回答合计 1.3093、总合计 12.8223；mask 平均 1.3093/2=0.6547；不 mask 平均 12.8223/5=2.5645；指令占比 11.5129/12.8223=89.8% ✅（页面各处数字一致）
- e^(−4.6052)≈0.01、e^(−1.2040)≈0.30 ✅；175/1.3≈134.6（「一百多倍」）✅

## 机械检查

- `python3 .dojo/scripts/validate.py wiki/sft/index.html` → `validation ok`，退出码 0 ✅
- 概念链接目标存在：wiki/pretraining/index.html、wiki/cross-entropy/index.html、../../index.html、overview.html ✅
- index.html ↔ overview.html 相互链接 ✅
- index.html head：纯文本 description ✅、dojo:summary（KaTeX 可渲染源码，`&lt;` 已转义）✅、dojo:type=concept ✅、dojo:topics=训练与优化（validate.py 词表校验通过）✅、dojo:tag=后训练 ✅。overview.html 为附属概览页，不承载 dojo: 元数据，与 validate.py 校验范围一致。
- 正文（含标题、summary、列表、表格）无 Unicode 数学字符直接出现；overview 的 ←/→ 为导航箭头非数学符号 ✅；结构图为 HTML（dg-flow），无等宽字符框线图 ✅
- 17 组 details/summary 配对：核心问题 5 + 各章本章问题 11 + 计算展开 1，全部有解答且答案独立可读、与正文一致；核心问题答案均指明完整论证章节 ✅
- 无 `<pre>` 代码块，无可运行代码声明，第 2.2 节第 3 条按不适用处理 ✅
- 无占位符、无未处理条件分支 ✅

## 问题

- [重要·技术] index.html §2 第 1 段（"多轮对话样本是它的扩展……"）：句子内部矛盾且扩大来源语境——"每个『助手回复』仍是要学的输出"与"Llama 2 对多轮数据的处理正是把此前轮次（含此前的助手消息）的 token 损失置零"不能同时成立（若含此前助手消息置零，则并非每个助手回复都是要学的输出）；且 Llama 2 原文该句出自 §3.3 多轮一致性/GAtt（system message）训练语境，被正文表述为"Llama 2 对多轮数据的处理"这一一般陈述｜引文依据：Llama 2 §3.3 "To fix this issue, which could hurt the training, we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages."（前文为 "Instead of augmenting all context-dialogue turns with the instruction, we can drop it in all but the first turn, but this would lead to a mismatch at training time..."，属 GAtt 小节）｜修复要求：改写该句消除矛盾——一般多轮样本的形态（此前轮次并入条件）与 Llama 2 §3.3 的做法（该语境下此前轮次含助手消息全部置零、只对最后一轮回复反向传播）分开表述，并标明该做法出自 §3.3 的多轮一致性/GAtt 语境；§2 本章问题 1 解答中的同一并置同步修改｜已修复：多轮句改为分开表述——一般形态（哪些回复计入损失因实现而异）与具体做法（Llama 2 GAtt 多轮一致性训练中把此前轮次含助手消息的损失置零、只训最后一轮回复）分别陈述；C4 条目标注「GAtt 多轮一致性训练的语境，非 SFT 数据的一般规则」；本章问题 1 解答同步。｜复验：validate.py 通过，修复处复查一致。
- [轻微·技术] index.html §1 第 3 段与 C8 条目：综述收益原文为 "The benefits of SFT are threefold"，页面正文只引其中两条（弥合目标差距、行为更可控可预测），未标明是三条中的两条，读者会误以为收益仅此两条；C8 的引文同样在 (2) 后截断｜引文依据："The benefits of SFT are threefold: (1) ... bridges the gap ...; (2) SFT allows for a more controllable and predictable model behavior ...; and (3) SFT is computationally efficient and can help LLMs rapidly adapt to a specific domain ..."｜修复要求：正文补出第三条（计算高效、可快速适配领域）或将 C8 引文补全至 (3)，并使正文与引文一致｜已修复：正文补全三重收益（含第三条计算高效），C8 条目补引原文 "(3) SFT is computationally efficient and can help LLMs rapidly adapt to a specific domain without extensive retraining or architectural changes."。｜复验：validate.py 通过，修复处复查一致。
- [轻微·可读性] index.html §5.3 段落与训练事实表："residual dropout 0.2"、"余弦学习率衰减"首次出现未作任何解释｜引文依据：不适用（术语解释问题；数值本身见 N2 核对）｜修复要求：在首次出现处加括号一句话说明（如 residual dropout：残差分支上施加的 dropout 正则），或在"简化条件"一节声明这些超参数仅作引用性事实、不展开｜已修复：5.3 首次出现处加括注「余弦学习率衰减：学习率随训练按余弦曲线逐步下降；residual dropout $0.2$：一种随机置零部分连接的正则化手段」。｜复验：validate.py 通过，修复处复查一致。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复

## 发布条件逐条核对

1. 三轮审查均完成且由独立审查者执行——本轮（第 3 轮）由未参与写作与前两轮的独立审查者完成；第 1、2 轮的执行情况不在本轮允许输入范围内，由编排者依据 review-1.md、review-2.md 确认。
2. 每条来源论断都有引文依据记录——C1–C8、F1、N1–N3 共 12 条全部核对并摘录引文（见上）；无「未能核对」项。
3. 所有阻断和重要问题均已关闭——❌ 本轮新发现 1 条重要问题（§2 多轮对话句）未关闭；2 条轻微未修复。
4. 遗留轻微问题具有明确接受理由——❌ 尚无（待修复或给出接受理由）。
5. 全部学习目标由正文章节完整回答——✅ 页面以「核心问题」承担学习目标，5 题分别由 §1–§5 完整回答。
6. 页面级核心问题与每章本章问题均有解答折叠块——✅ 共 17 组，无只列问题未作答的情况。
7. 数学符号全部 LaTeX、结构图为 HTML/内联 SVG——✅。
8. `.dojo/scripts/validate.py` 返回成功——✅（validation ok，退出码 0）。
9. 可运行代码的结果与页面描述一致——✅ 不适用（页面无代码块）；全部数值示例经 Python 复算一致。
10. 关键论断和数字已重新核对来源——✅（本轮全部 12 条重核）。
11. head 元数据（description、dojo:summary、dojo:type、dojo:topics、dojo:tag）——✅ index.html 全部具备且合法。
12. overview.html 与 index.html 相互链接——✅。
13. 页面引用的概念链接有效或有明确占位——✅（pretraining、cross-entropy 均存在）。
14. 递归生成的前置概念页已完成各自质检——不在本轮允许输入范围内（禁止读取其 research/ 目录），由编排者确认。

## 总处置

修复。第 1 条重要问题修复并复验通过、轻微问题关闭（修复或给出接受理由）、且第 1/2/14 项由编排者确认后，页面可发布。本轮不写发布结论。


## 发布结论

- 发布时间：2026-08-25
- 三轮独立审查（每轮独立子代理）完成，全部阻断与重要问题已修复关闭；本轮 1 重要 2 轻微均已修复（多轮表述语境、三收益补全、术语解释）。
- 编排者确认：前两轮审查均由独立子代理执行且记录完整（review-1.md、review-2.md）；前置概念页（交叉熵、语言模型预训练）已完成各自三轮独立审查与修复。
- 修复后 `.dojo/scripts/validate.py` 通过；headless Chrome 渲染实测 KaTeX 节点正常、0 错误。
- 结论：可发布。
