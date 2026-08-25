# 语言模型预训练 核心论断与证据

## C 论断

- C1（已确认）：语言模型是词、token 或字符序列上的概率分布；该任务常被表述为 next token prediction，是 NLP 的核心问题。
  - 来源定位：LLaMA 论文（arXiv:2302.13971）§7 Related work "Language models" 段。原文："Language models are probability distributions over sequences of words, tokens or characters (Shannon 1948; Shannon 1951). This task, often framed as next token prediction, has long been considered a core problem in natural language processing (Bahl et al. 1983; Brown et al. 1990)."
- C2（已确认）：token 序列的联合概率可分解为条件概率连乘 $p(x)=\prod_{i=1}^{n} p(s_i \mid s_1,\dots,s_{i-1})$，使采样与估计可行。
  - 来源定位：GPT-2 论文（Radford et al., 2019）§2，式 (1)。原文（PDF 提取 /tmp/gpt2.txt 行 107–125）："symbols as the product of conditional probabilities (Jelinek & Mercer, 1980) (Bengio et al., 2003): p(x) = ∏ p(s_n | s_1, ..., s_{n−1}) (1) ... This approach allows for tractable sampling from and estimation of p(x) as well as any conditionals of the form p(s_{n−k}, ..., s_n | s_1, ..., s_{n−k−1})."
  - 注：式 (1) 的乘积指标在 PDF 提取中排版受损，写作时按原文形式 $p(x) = \prod_{i} p(s_i \mid s_{<i})$ 呈现，来源仍标 GPT-2 §2 式 (1)。
- C3（已确认）：语言建模结果通常以每个预测单元的平均负对数概率（或其缩放/指数化）报告。
  - 来源定位：GPT-2 论文 §3.1。原文："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word."
- C4（已确认）：预训练之后的对齐流程从 SFT 开始：在标注者示范数据上微调预训练模型。
  - 来源定位：InstructGPT（arXiv:2203.02155）§3.1 Step 1。原文："Our labelers provide demonstrations of the desired behavior on the input prompt distribution... We then fine-tune a pretrained GPT-3 model on this data using supervised learning."
- C5（已确认）：预训练目标与用户目标的错位。
  - 来源定位：指令微调综述（arXiv:2308.10792）§1。原文："LLMs are typically trained on minimizing the contextual word prediction error on large corpora; while users want the model to 'follow their instructions helpfully and safely'."
- C6（已确认）：最小化负对数似然等价于最小化交叉熵（预训练「最大化似然」与「最小化交叉熵」两种说法的等价依据）。
  - 来源定位：Deep Learning §5.5。原文："Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy."

## F 公式

- F1（已确认）：链式分解 $p(s_1,\dots,s_n)=\prod_{i=1}^{n} p(s_i\mid s_1,\dots,s_{i-1})$。来源：GPT-2 §2 式 (1)。
- F2（已确认）：预训练目标（语料平均下一 token 交叉熵）$\mathcal{L}(\theta) = -\frac{1}{|D|}\sum_{x\in D} \frac{1}{|x|}\sum_{i} \log p_\theta(s_i \mid s_{<i})$。依据：F1 + C3（按预测单元平均）+ C6（交叉熵等价）的定义组合，写作时标注组合来源而非单一出处。

## N 数字

- 本页无来源数字要求；手算示例为构造数据，Python 实算。

## 来源清单

- LLaMA：Touvron et al., arXiv:2302.13971，§7（已摘录原文）
- GPT-2：Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI 2019，§2 式 (1)、§3.1（PDF 已下载提取至 /tmp/gpt2.txt，行 107–125、335–348，已摘录）
- InstructGPT：Ouyang et al., arXiv:2203.02155，§3.1（已摘录）
- 指令微调综述：Zhang et al., arXiv:2308.10792，§1（已摘录）
- Goodfellow et al., Deep Learning，§5.5（已摘录）

## 冲突与不足

- 无冲突项。GPT-2 式 (1) 的排版受损已在 C2 注明处理方式。
