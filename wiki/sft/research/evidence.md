# SFT 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。所有来源均已实际打开核对并摘录原文片段。

## C 论断

- C1（已确认）：SFT（监督微调）指在 (instruction, output) 对上以监督方式继续训练 LLM，弥合下一词预测目标与用户「遵循指令」目标之间的差距。
  - 来源定位：arXiv:2308.10792（指令微调综述）摘要与 §1。原文："Instruction tuning refers to the process of further training LLMs on a dataset consisting of (instruction, output) pairs in a supervised fashion, which bridges the gap between the next-word prediction objective of LLMs and the users' objective of having LLMs adhere to human instructions."
  - 适用条件：LLM 后训练语境。
  - 同一来源脚注 1："supervised fine-tuning (SFT) and instruction tuning (IT) are used interchangeably"（术语同义）。

- C2（已确认）：SFT 的动机是训练目标与用户目标的错位。
  - 来源定位：综述 §1。原文："One of the major issues with LLMs is the mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'."
  - 适用条件：预训练模型未经后训练时。

- C3（已确认）：标准后训练流程为 SFT → 奖励模型（RM）→ PPO 偏好优化三步。
  - 来源定位：InstructGPT（arXiv:2203.02155）§3.1。原文："Step 1: Collect demonstration data, and train a supervised policy... Step 2: Collect comparison data, and train a reward model... Step 3: Optimize a policy against the reward model using PPO." 及 Fig. 2 图注 "(1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)"。
  - 适用条件：经典 RLHF 流程；DPO 等直接偏好算法可替代第 2、3 步（正文以「偏好优化」统称，不展开）。

- C4（已确认）：SFT 的常见实现把损失限制在回答 token 上，对用户指令部分置零。
  - 来源定位：Llama 2（arXiv:2307.09288）§3.1 Fine-Tuning Details。原文："We utilize an autoregressive objective and zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens."
  - 适用条件：Llama 2 的实现；作为「常见做法」呈现，不表述为定义的必然部分。
  - 补充（同节）：样本由 prompt 与 answer 拼接，用特殊 token 分隔，序列拼接填充至 4096。

- C5（已确认）：SFT 数据量与质量——数万条量级的高质量标注即可达到高质量结果。
  - 来源定位：Llama 2 §3.1 Quality Is All You Need。原文："We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations." 及 "By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved."
  - 适用条件：Llama 2 的模型规模与数据分布；作为实践规律呈现而非普适定律。

- C6（已确认）：SFT 验证损失过拟合后继续训练仍能提升下游偏好指标。
  - 来源定位：InstructGPT §3.5 Supervised fine-tuning (SFT)。原文："We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2... we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting."
  - 适用条件：InstructGPT 的实验设置（GPT-3 系列、13k 数据）。

- C7（已确认）：后训练后的 1.3B InstructGPT（含 SFT 与后续偏好优化）输出被人类偏好于 GPT-3 175B；两模型架构相同，唯一区别是人类数据微调。
  - 来源定位：InstructGPT 摘要与 §1 "Our main findings" 段。原文："outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data."
  - 适用条件：InstructGPT 指完整三阶段后训练产物（SFT+RM+PPO），非纯 SFT 模型；正文表述须保持这一精确性。同段还有 "This result holds true even when we add a few-shot prompt to GPT-3"。
  - 另（§3.5 RM 段）："Starting from the SFT model with the final unembedding layer removed"——支持「SFT 产物是后续步骤的初始策略」。

- C8（已确认）：SFT 的收益：弥合下一词预测目标与指令遵循之间的差距；让模型行为更可控、更可预测。
  - 来源定位：综述 §1。原文："The benefits of SFT are threefold: (1) Finetuning an LLM on the instruction dataset bridges the gap between the next-word prediction objective of LLMs and the users' objective of instruction following; (2) SFT allows for a more controllable and predictable model behavior compared to standard LLMs."
  - 适用条件：综述对 SFT 的总体判断。

## F 公式

- F1（已确认，定义组合）：因果语言模型的自回归训练目标（SFT 沿用）。
  - 内容：$L(\theta) = -\sum_{(x,y)\in D}\sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$（SFT 版本；无 mask 版本对整个序列求和）。
  - 来源定位：链式分解 $p(x)=\prod_i p(s_i\mid s_{<i})$ 见 GPT-2（Radford et al., 2019）§2 式 (1)（PDF 提取核对）；SFT 数据形态下的形式由 C1+C4 组合得出，属定义的组合，不单独标注外部来源。前置概念页「语言模型预训练」「交叉熵」承载公式细节。
  - 适用条件：token 级求和、teacher forcing（训练时回答的每个前缀已知）。

## N 数字

- N1（已确认）：InstructGPT SFT 数据集约 13k 训练 prompts。来源：§3.2 "The SFT dataset contains about 13k training prompts (from the API and labeler-written)"。
- N2（已确认）：InstructGPT SFT 训练 16 epochs，cosine 学习率衰减，residual dropout 0.2。来源：§3.5（见 C6 引文）。
- N3（已确认）：Llama 2 SFT 共 27,540 条标注，训练 2 epochs，初始学习率 2e-5（cosine），weight decay 0.1，batch size 64，序列长度 4096。来源：§3.1（见 C4、C5 引文及 Fine-Tuning Details 段）。

## 来源清单

- InstructGPT：Ouyang et al., "Training language models to follow instructions with human feedback", arXiv:2203.02155（§3.1、§3.2、§3.5、§4，已逐节核对引文）
- 指令微调综述：Zhang et al., "Instruction Tuning for Large Language Models: A Survey", arXiv:2308.10792（摘要、脚注 1、§1，已核对引文）
- Llama 2：Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv:2307.09288（§3.1，已核对引文）
- GPT-3：Brown et al., "Language Models are Few-Shot Learners", arXiv:2005.14165（§2.1 自回归目标，写作时核对式号）

## 冲突与不足

- 无未决项。loss masking「存在不 mask 的实现」：暂无固定版本官方来源支持，正文只以 Llama 2 为据呈现 masking 做法，并标注其为「常见做法」（scope 已裁定措辞）。C7 已核对并注明「InstructGPT 指完整后训练产物」的精确性约束。
