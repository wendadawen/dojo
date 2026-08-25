# SFT（监督微调）审查记录（第 1 轮）

- 页面版本：index.html 工作树哈希 `771e3ea633e82135eaef8e8f6c06aaa369f508db`（overview.html 同刻哈希 `b0b161173a1c49892b41720286ba611d07f15a8d`）
- 审查时间：2026-08-25 15:52 CST
- 审查者：编排者派发的独立审查者（未参与写作与规划）
- 已完整阅读章节（index.html，按顺序）：引言、核心问题（5 题及解答折叠块）、1. 预训练目标与用户目标的错位（含本章问题 2 题）、2. SFT 的数据与训练目标（含本章问题 3 题）、3. 手算一条 SFT 样本的损失（含「展开：完整对数计算」折叠块与本章问题 2 题）、4. SFT 在后训练流程中的位置（含三步流程图示与本章问题 2 题）、5. 实践中的边界与误解（5.1–5.4、训练事实汇总表、本章问题 2 题）、来源与范围说明（论断与来源 C1–C8、公式与来源 F1、外部数字 N1–N3、构造示例、辅助解释与类比边界、简化条件）；overview.html 全文（它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界 / 页脚）。

## 问题

- [轻微·技术] index.html「来源与范围说明」C7 及顶部「主要依据」blockquote：<>InstructGPT 的 "Our main findings" 实际位于 §1 Introduction，页面标注为「摘要与 §4」；§4 标题为 Results，不含该引文｜引文依据：ar5iv 版 InstructGPT 章节列表为「1 Introduction；2 Related work；3 Methods and experimental details（3.1–3.6）；4 Results（4.1–4.3）」；"Our main findings" 与 "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data." 均在 §1｜修复要求：C7 的来源定位由「摘要与 §4 'Our main findings'」改为「摘要与 §1 'Our main findings'」，顶部主要依据行中的「§4」同步改为「§1」｜已修复：C7 来源定位改为「摘要与 §1 'Our main findings'」，顶部主要依据行改为「§1、§3.1、§3.2、§3.5」，5.4 首句改为「InstructGPT 的主要发现（§1）」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html F1（公式与来源）：GPT-2 论文不存在 §2.1 子节，链式分解式 (1) 位于 §2 "Approach"｜引文依据：/tmp/gpt2.txt 第 102 行为「2. Approach」（其后直接是第 110–117 行的 `p(x) = ∏ p(sn|s1,…,sn−1) (1)`），全文无「2.1」层级标题｜修复要求：F1 中「GPT-2（Radford et al., 2019）§2.1 式 (1)」改为「§2 式 (1)」｜已修复：F1 改为「GPT-2（Radford et al., 2019）§2 式 (1)」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html 引言首段与「来源与范围说明」C1、overview.html「它是什么」：SFT「与 instruction tuning『指令微调』同义」把单篇综述的行文约定扩大为普适同义，且 C1 引文省略了脚注限定语｜引文依据：综述脚注 1 全文为 "In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably."（页面引文从 "supervised fine-tuning" 起，丢弃了前半句限定语）｜修复要求：正文两处改为「在该综述中与 instruction tuning（指令微调）混用」之类的限定表述（或等价的、保留限定语的写法），C1 引文补全「In this paper, unless specified otherwise,」｜已修复：引言改为「综述将该术语与 instruction tuning『指令微调』互换使用」；C1 引文补全限定语 "In this paper, unless specified otherwise," 并注明「同义以该综述的使用惯例为据」；overview 同步。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html §2 第 1 段末句与 overview.html「核心机制」第 1 条：「多轮对话样本是它的扩展：把此前的轮次并入条件部分，每个『助手回复』仍是要学的输出」为未标注来源的数据格式论断｜引文依据（可定位的支持来源）：Llama 2 §3.3 "Assume we have access to a multi-turn dialogue dataset between two persons (e.g., a user and an assistant), with a list of messages [u1,a1,…,un,an], where un and an correspond to the user and assistant messages for turn n, respectively." 及 "These steps produce an SFT dataset, on which we can fine-tune Llama 2-Chat."；Llama 2 §3.1 对 SFT 数据仅描述单轮 prompt+answer｜修复要求：为该句补来源标注（新增一条 C 引用 Llama 2 §3.3，或在现有引用中扩展），或在「来源与范围说明」新增一条「说明性扩展」登记（明确写出此句为页面自行扩展、注明支持来源）｜已修复：正文中该句后补充「Llama 2 对多轮数据的处理正是把此前轮次（含此前的助手消息）的 token 损失置零」并标注 [C4]；C4 条目增补 §3.3 引文 "we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages"。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html §1 第 3 段与第 1 章本章问题 2 解答：「few-shot prompting 在输入上下文里放几个示例，能临时诱导模型模仿示例的格式」为未标注来源的机制/效果表述，且「模仿示例的格式」与来源措辞不一致｜引文依据（可定位的支持来源）：InstructGPT §1 "This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions."；§4.1 "one can obtain significant step-size improvements by using a well-crafted few-shot prompt (GPT-3 (prompted))"｜修复要求：两处补来源标注（InstructGPT §1 / §4.1），并将「模仿示例的格式」改为与来源一致的「更好地遵循指令」，或在「来源与范围说明」登记为解释性表述｜已修复：§1 与本章问题 2 解答均改写——few-shot 表述改为「在输入中给出任务的若干示例，可以引导模型在当前上下文内完成任务，但参数没有更新」，并以 C7（含 "This result holds true even when we add a few-shot prompt to GPT-3"）支撑「不充分」；删除「模仿示例的格式」措辞。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·技术] index.html §5.4 第 1 段末句：「参数量少一百倍」与原文 "over 100x fewer parameters" 不一致（175B / 1.3B ≈ 134.6，为「一百多倍」）｜引文依据：InstructGPT §1 "…are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters."｜修复要求：改为「参数量少一百多倍」或「参数量不到 GPT-3 的百分之一」｜已修复：§5.4 与 §1 相关句均改为「参数量少一百多倍」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。
- [轻微·可读性] index.html 核心问题 5 解答（「训满 16 epochs」首次出现）至 §5.3：术语 epoch 全文未在首次使用处解释｜引文依据：不适用｜修复要求：在首次出现处（核心问题 5 解答或 §5.3，以先出现者为准）加括注「epoch＝把训练集完整过一遍」，或将该处及后续 epochs 表述改为「完整训练轮数」｜已修复：§5.3 首次出现处加括注「epoch：训练数据完整过一遍」。｜复验：validate.py 通过，修复处与引文/原文一致（本轮修复后人工比对）。

## 来源核对记录（引文依据）

以下论断均已打开来源、定位到页面标注位置并摘录原文（WebFetch ar5iv 版本 + 本地 /tmp/gpt2.txt）：

- C1（SFT/指令微调定义，综述摘要、§1、脚注 1）：✅ 核对一致。摘要 "Instruction tuning refers to the process of further training LLMs on a dataset consisting of (instruction, output) pairs in a supervised fashion…"；§1 "It involves further training LLMs using (instruction, output) pairs, where instruction denotes the human instruction for the model, and output denotes the desired output that follows the instruction."；脚注 1 见问题 3（限定语被省略，已列轻微问题）。
- C2（目标错位，综述 §1）：✅ 核对一致。"One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'." 页面译文「在大语料上最小化上下文词预测误差」「遵循指令、有帮助且安全」与原文对应。
- C3（三步流程与 RM 初始化，InstructGPT §3.1、§3.5）：✅ 核对一致。§3.1 "Step 1: Collect demonstration data, and train a supervised policy… We then fine-tune a pretrained GPT-3 model on this data using supervised learning."；"Step 2: Collect comparison data, and train a reward model…"；"Step 3: Optimize a policy against the reward model using PPO… We use the output of the RM as a scalar reward."；"Steps 2 and 3 can be iterated continuously"（支持图注「循环迭代」）；§3.5 "Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward."
- C4（拼接与损失掩码，Llama 2 §3.1）：✅ 核对一致。"For the fine-tuning process, each sample consists of a prompt and an answer. To ensure the model sequence length is properly filled, we concatenate all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments. We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens. Finally, we fine-tune the model for 2 epochs."
- C5（数据质量与数量，Llama 2 §3.1 Quality Is All You Need）：✅ 核对一致。"By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved."；"We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations."
- C6（验证损失过拟合与继续训练收益，InstructGPT §3.5）：✅ 核对一致。"We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2… we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting."
- C7（1.3B 被偏好于 175B，InstructGPT 摘要与 "Our main findings"）：✅ 内容核对一致，但章节定位错误（"Our main findings" 在 §1 而非 §4，见问题 1）。§1 "…outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions."
- C8（SFT 收益，综述 §1）：✅ 核对一致。"The benefits of SFT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) SFT allows for a more controllable and predictable model behavior compared to standard LLMs…"
- F1（链式分解，GPT-2 2019 式 (1)）：✅ 内容核对一致，但子节号错误（§2.1 不存在，见问题 2）。/tmp/gpt2.txt §2 "Approach"："Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities… p(x) = ∏ p(sn | s1, …, sn−1) (1)"。
- N1（约 13k 训练 prompts，InstructGPT §3.2）：✅ 核对一致。"The SFT dataset contains about 13k training prompts (from the API and labeler-written)…"
- N2（16 epochs、cosine 衰减、residual dropout 0.2，InstructGPT §3.5）：✅ 核对一致（引文见 C6）。
- N3（Llama 2 超参，§3.1）：✅ 核对一致。"For supervised fine-tuning, we use a cosine learning rate schedule with an initial learning rate of 2×10⁻⁵, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens." + "we fine-tune the model for 2 epochs"。

## 数值复算记录（Python）

- 逐 token 损失：−ln 0.01 = 4.6052；−ln 0.05 = 2.9957；−ln 0.02 = 3.9120；−ln 0.30 = 1.2040；−ln 0.90 = 0.1054 ✅ 与页面一致。
- mask 版平均 (1.2040+0.1054)/2 = 0.6547 ✅；不 mask 版平均 12.8223/5 = 2.5645 ✅；指令部分合计 11.5129 ✅；总合计 12.8223 ✅；指令占比 11.5129/12.8223 = 89.8% ✅。
- 反向验证 e^−4.6052 ≈ 0.01、e^−1.2040 ≈ 0.30 ✅。
- 175B/1.3B ≈ 134.6（用于问题 6 判断「over 100x」）。

## 机械检查记录

- 页面链接：`../../wiki/pretraining/index.html`、`../../wiki/cross-entropy/index.html`、`overview.html`、`../../index.html` 及 overview 的反向链接 `index.html`、`../../index.html` 目标文件均存在 ✅；index 与 overview 相互链接 ✅（仅核对了链接目标存在，未审查目标页内容）。
- 本地资源：libs/ 下 katex.min.css/js、auto-render.min.js、prism 全套均存在 ✅。
- 数学符号扫描（剥离 KaTeX 与标签后）：正文无 Unicode 数学字符；出现的「→」均为流程/数据流记号（如「SFT → 奖励模型 → 偏好优化」，与图示箭头语义一致），「①–④」为枚举序号，均不判为数学符号 ✅。同一变量（$p_\theta$、$y_t$、$y_{<t}$、$\mathcal{L}$）全页写法一致 ✅。
- 图示：三步流程图为 HTML flex 结构（.dg-flow/.dg-node/.dg-arrow），非等宽字符框线图；图内无公式；节点含义与箭头方向在正文与图注中定义；窄屏下 flex 换行为纵向堆叠，配色走 CSS 变量（明暗主题自适应）✅。
- 问题块：页面级「核心问题」5 题与 5 个章节的「本章问题」（2/3/2/2/2 题）全部有解答折叠块，答案独立可读、与正文一致；核心问题解答均指明完整论证所在章节 ✅。
- 可运行代码：页面未包含声称可运行的代码，无需执行验证 ✅。
- 页面 head（index.html）：纯文本 description ✅、dojo:summary（含可渲染 LaTeX）✅、dojo:type=concept ✅、dojo:tag ✅、dojo:topics=「训练与优化」——**未能核对**：AGENTS.md 的主题词表不在本轮允许输入内，该条留待修复阶段由 `.dojo/scripts/validate.py` 强制校验。
- 格式一致性（规范 2.2 第 12 条）：style-guide.md 不在本轮允许输入内，**未能核对**；已按 check.md 可判定的机械项（LaTeX 化、图示、问题块、链接）执行检查。
- 综述 §2.1 提到指令数据实例含 instruction、可选 input、output 三要素，页面采用摘要/§1 的 (instruction, output) 二元表述——属定义措辞选择，不构成不一致，未列问题。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 7
- 处置：修复（7 条轻微问题均为来源定位/标注与术语解释层面的局部修改，不影响核心结论；逐条修复并复验后进入第 2 轮独立审查）
