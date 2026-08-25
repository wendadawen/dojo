# SFT 监督微调审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 `1506be518aaa6f58ca7a67c0c780364d58d417de`；overview.html 工作树哈希 `83f32fe4389f64f525f899af5976ca5780655f21`
- 审查时间：2026-08-25 16:11
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节（按顺序）：页头元信息（title / description / dojo:summary / dojo:type / dojo:topics / dojo:tag）→ 引言（翻译续写场景）→ 核心问题（5 问及解答折叠块）→ 1. 预训练目标与用户目标的错位（含对照表、本章问题）→ 2. SFT 的数据与训练目标（含损失公式与符号表、拼接位置表、本章问题）→ 3. 手算一条 SFT 样本的损失（含「展开：本例五个位置的完整对数计算」折叠块、本章问题）→ 4. SFT 在后训练流程中的位置（含三步流程图、本章问题）→ 5. 实践中的边界与误解（5.1–5.4、训练事实汇总表、本章问题）→ 来源与范围说明（C/F/N、构造示例、辅助解释与类比边界、简化条件）→ overview.html 全文（它是什么 / 为什么需要它 / 核心机制 / 关键结论与边界）

## 来源核对记录（每条论断的引文依据）

核对方式：ar5iv HTML 全文下载后剥离标签定位原文；GPT-2 用本地提取文本 /tmp/gpt2.txt。所有引文为原文摘录。

- **C1（SFT/指令微调定义与互换使用）核对通过**。综述摘要："This paper surveys research works in the quickly advancing field of instruction tuning (IT), which can also be referred to as supervised fine-tuning (SFT)"；"Instruction tuning refers to the process of further training LLMs on a dataset consisting of (instruction, output) pairs in a supervised fashion"。脚注 1："In this paper, unless specified otherwise, supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably."。§1："It involves further training LLMs using (instruction, output) pairs, where instruction denotes the human instruction for the model, and output denotes the desired output that follows the instruction."。页面正文、本章问题 2.1 解答及 overview 的表述与之一致。
- **C2（目标错位）核对通过**。综述 §1："One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'"。页面 §1 的转述（「最小化上下文词预测误差」「遵循指令、有帮助且安全」）与原文逐点对应。
- **C3（三步流程与 RM 起点）核对通过**。InstructGPT §3.1："Step 1: Collect demonstration data, and train a supervised policy... Step 2: Collect comparison data, and train a reward model... Step 3: Optimize a policy against the reward model using PPO"；"Steps 2 and 3 can be iterated continuously"。Fig. 2 图注："A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)"。§3.5："Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward."。页面 §4 流程图、图注（循环迭代）及「RM 从 SFT 模型出发、去掉最后的输出层」均与原文一致。
- **C4（拼接与损失掩码）核对通过**。Llama 2 §3.1 Fine-Tuning Details："For the fine-tuning process, each sample consists of a prompt and an answer. To ensure the model sequence length is properly filled, we concatenate all the prompts and answers from the training set. A special token is utilized to separate the prompt and answer segments. We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens."。多轮情形 §3.3（System Message for Multi-Turn Consistency，已确认小节编号）："To fix this issue, which could hurt the training, we simply set the loss to 0 for all the tokens from the previous turns, including assistant messages."。页面 §2 的两处引用及本章问题 2.2 解答中的英文引文逐字一致。
- **C5（数据质量重于数量）核对通过**。Llama 2 §3.1 Quality Is All You Need："We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations."；"By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved."。页面 §5.2 的「数万条量级」「27,540 条后停止标注」「放弃数百万条第三方样本、结果显著变好」逐点对应。
- **C6（验证损失过拟合与继续训练收益）核对通过**。InstructGPT §3.5："we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting."。页面 §5.3 表述一致。
- **C7（1.3B 偏好于 175B，few-shot 后仍成立）核对通过，但 §1 一处使用方向颠倒，见问题 1**。InstructGPT §1 Our main findings："On our test set, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions."；摘要亦有 "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters"。页面 §5.4、核心问题 5 解答、本章问题 1.2 解答方向正确；§1 第 3 段一处颠倒（问题 1）。
- **C8（SFT 收益）核对通过**。综述 §1："The benefits of SFT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) SFT allows for a more controllable and predictable model behavior compared to standard LLMs."。页面 §1 只引用了 (1)(2)，未扩大。
- **F1（链式分解）核对通过**。GPT-2（Radford et al., 2019）§2 Approach 式 (1)："Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities (Jelinek & Mercer, 1980) (Bengio et al., 2003): p(x) = ∏ p(s_n | s_1, ..., s_{n-1}) (1)"（/tmp/gpt2.txt 102–119 行）。页面 F1 将公式标注为「C1 + C4 组合得出、链式分解见 GPT-2 §2 式 (1)」，来源归属诚实。
- **N1（约 13k 训练 prompts）核对通过**。InstructGPT §3.2："The SFT dataset contains about 13k training prompts (from the API and labeler-written)"。
- **N2（16 epochs、余弦衰减、residual dropout 0.2）核对通过**。InstructGPT §3.5："We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2."
- **N3（Llama 2 超参）核对通过**。§3.1："we use a cosine learning rate schedule with an initial learning rate of 2 × 10^{-5}, a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens"；"Finally, we fine-tune the model for 2 epochs."；27,540 见 C5。§5 汇总表全部数字与原文一致。
- **开头场景的机制归因**（「预训练语料里这句文本后面通常是更多网页文本」）：属解释性叙述，其一般机制由 C2 及 InstructGPT §1（"the language modeling objective used for many recent large LMs—predicting the next token on a webpage from the internet—is different from the objective 'follow the user's instructions helpfully and safely'... the language modeling objective is misaligned"）支撑，§1 正文已引用 C2，可接受。

## 数值复算记录（Python）

- 逐位置负对数：-ln(0.01)=4.6052，-ln(0.05)=2.9957，-ln(0.02)=3.9120，-ln(0.30)=1.2040，-ln(0.90)=0.1054 —— 与页面表格一致。
- mask 版平均：(1.2040+0.1054)/2 = 0.6547 —— 一致。
- 不 mask 版平均：(4.6052+2.9957+3.9120+1.2040+0.1054)/5 = 12.8223/5 = 2.5645 —— 一致。
- 指令部分合计 4.6052+2.9957+3.9120 = 11.5129，占总损失 11.5129/12.8223 = 89.8% —— 一致。
- 复算 e^{-4.6052}≈0.01、e^{-1.2040}≈0.30 —— 与「展开」折叠块的可复算说明一致。
- 页面无声称可运行的代码块，第 3 项（代码执行）不适用。

## 机械检查记录

- `python3 .dojo/scripts/validate.py wiki/sft/index.html` → validation ok；`wiki/sft/overview.html` → validation ok（含 meta 完整性、dojo:topics 词表、锚点、链接）。
- 链接目标存在：wiki/pretraining/index.html ✓、wiki/cross-entropy/index.html ✓、../../index.html ✓、overview.html ↔ index.html 互链 ✓。
- 本地资源存在：katex.min.css / katex.min.js / auto-render.min.js / prism-primer-light.css / prism-primer-dark.css / prism.min.js / prism-python.min.js 均 ✓。
- 公式全部以 `$...$`/`$$...$$` KaTeX 书写；目录锚点由带 id 的 h2/h3 生成；details 折叠块结构完整；图示为 HTML 结构（dg-flow），无等宽字符框线图，图内无公式。
- Unicode 数学符号扫描（剥离 script/style/标签后）：index.html 正文/summary/列表含 U+2192「→」×7（其中 1 处 U+2191 为返回顶部按钮，界面元素）；overview.html 含 U+2192×4、U+2190×1（←/→ 各 1 处为导航链接界面元素）。判定见问题 2。
- 两级问题块：核心问题 5 问、各章本章问题（2/3/2/2/2 问）均有独立可读的解答折叠块；核心问题答案均指明完整论证所在章节；学习目标（核心问题）由第 1–5 章完整回答。

## 问题

- [重要·技术] index.html §1 第 3 段（「不改参数的替代方案存在但不充分……」）：few-shot 对比句主客方向颠倒。「给 GPT-3 加上 few-shot 提示后，其输出仍被人类偏好于小一百多倍参数的完整后训练模型」按「A 被偏好于 B = A 优于 B」的构造（同页核心问题 5 解答「1.3B 模型输出被人类偏好于 GPT-3 175B」即此用法），字面含义是 few-shot GPT-3 优于 1.3B 完整后训练模型，与所引 C7 的结论正好相反，且与本章问题 1.2 解答（「给 GPT-3 加 few-shot 提示也无法追上完整后训练的小模型」）和 §5.4（「这一结论在给 GPT-3 加上 few-shot 提示后仍然成立」）自相矛盾｜引文依据：InstructGPT §1 "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. ... This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions."｜修复要求：将该句改为正确方向（例如以完整后训练模型为主语：「给 GPT-3 加上 few-shot 提示后，小一百多倍参数的完整后训练模型输出仍被人类偏好于它」，或等价改写），使其与 C7、本章问题 1.2 解答及 §5.4 一致；修改后重新核对 C7 原文｜已修复：句子改写为「给 GPT-3 加上 few-shot 提示后，小一百多倍参数的完整后训练模型，其输出仍被人类偏好于 GPT-3」，与 C7 原文方向一致；与本章问题解答、5.4 一致。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·格式] index.html dojo:summary（第 7 行）、核心问题 4 解答（第 783 行）、§1 第 3 段（第 812 行）、来源说明 C3（第 1050 行）；overview.html「为什么需要它」（第 51 行）、「核心机制」（第 58 行）：流程箭头 U+2192「→」为 Unicode 数学类符号（General Category Sm），直接出现在 summary、正文与列表中，规范 2.2 第 9 条字面要求这些位置无 Unicode 数学字符直接出现（导航链接与返回顶部按钮中的 ←/→/↑ 为界面元素，不在违规之列）｜引文依据：不适用（扫描结果：两页正文/summary/列表共 9 处 U+2192）｜修复要求：将上述 9 处流程箭头改写为 KaTeX（如 `$\to$`），或按 style-guide 的相应条款记录明确的接受理由后保留，并保持两页处理一致｜已修复：两页全部正文字面箭头改为 $\to$（KaTeX 渲染）——dojo:summary 1 处、正文 3 处、overview 2 处；导航「完整说明 →」为模板界面元素，保留。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。
- [轻微·技术] index.html §5.4（第 1012 行）及 overview.html「关键结论与边界」末条（第 66 行）：「说明后训练（SFT 为其起点）带来的是行为方式的转变，而不是把更多知识装进参数」——「行为方式的转变」可由 C7 直接得出，但「而不是把更多知识装进参数」的全称否定是页面的延伸推断，以「说明」引出写成了来源结论；且 §5.4 标题用「记忆」、正文用「知识」，两页措辞不一致｜引文依据：InstructGPT §1 "These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data."（来源仅支持「同架构、仅微调之别、偏好反超」，未做「行为方式 vs 知识」的二分论断）｜修复要求：将该句降级为明确标注的推断（例如「这更支持把收益理解为行为方式的转变，而非知识容量的增加」），或删去全称否定部分；统一标题与正文的「记忆/知识」措辞，overview 同步｜已修复：5.4 标题改为「后训练改变的是行为方式」；正文改为「这个对比支持『后训练改变的是行为方式』的判断——参数内部具体发生了什么变化，不在本页范围」；overview 同步删除「不只是装知识」的全称否定。｜复验：validate.py 通过；修复处逐句复查与原文方向一致。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复（重要问题 1 须关闭后进入下一轮；两条轻微问题修复或记录接受理由）
