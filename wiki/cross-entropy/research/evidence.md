# 交叉熵 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。来源均已实际打开并摘录原文。

## C 论断

- C1（已确认）：自信息定义为 $I(x) = -\log P(x)$，本书用自然对数（nats）。
  - 来源定位：Goodfellow et al., Deep Learning, §3.13, 式 (3.48)。原文："we define the self-information of an event x = x to be I(x) = − log P(x). (3.48) In this book, we always use log to mean the natural logarithm, with base e."
- C2（已确认）：香农熵 $H(P) = -\mathbb{E}_{x\sim P} \log P(x)$ 是分布不确定性的度量。
  - 来源定位：同上，式 (3.49)。原文："We can quantify the amount of uncertainty in an entire probability distribution using the Shannon entropy, H(x) = E_{x∼P}[I(x)] = −E_{x∼P} [log P(x)], (3.49) also denoted H(P)."
- C3（已确认）：KL 散度 $D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}[\log P(x) - \log Q(x)]$ 非负、不对称。
  - 来源定位：同上，式 (3.50)。原文："we can measure how different these two distributions are using the Kullback-Leibler (KL) divergence: D_KL(P∥Q) = E_{x∼P} log P(x)/Q(x) = E_{x∼P}[log P(x) − log Q(x)]. (3.50)"；"most notably being non-negative. The KL divergence is 0 if and only if P and Q are the same distribution"。
- C4（已确认）：交叉熵 $H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$，即 $H(P,Q) = -\mathbb{E}_{x\sim P} \log Q(x)$；对固定的 $P$，最小化交叉熵等价于最小化 KL。
  - 来源定位：同上，式 (3.51)。原文："A quantity that is closely related to the KL divergence is the cross-entropy H(P, Q) = H(P) + D_KL(P∥Q), which is similar to the KL divergence but lacking the term on the left: H(P, Q) = −E_{x∼P} log Q(x). (3.51) Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term."
- C5（已确认）：最大似然等价于最小化负对数似然（NLL），也就等价于最小化交叉熵；任何由 NLL 组成的损失都是经验分布与模型分布之间的交叉熵。
  - 来源定位：Deep Learning §5.5（式 5.59–5.61 之后）。原文："Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy."；"Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model."
- C6（已确认）：$0\log 0$ 按惯例取 0。
  - 来源定位：Deep Learning §3.13（式 3.51 之后）。原文："it is common to encounter expressions of the form 0 log 0. By convention, in the context of information theory, we treat these expressions as lim x→0 x log x = 0."
- C7（已确认）：语言建模的结果通常以「每个预测单元的平均负对数概率」的缩放或指数化形式报告。
  - 来源定位：GPT-2 论文（Radford et al., 2019）§3.1。原文："Results on language modeling datasets are commonly reported in a quantity which is a scaled or exponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word."

## F 公式

- F1（已确认）：信息论定义 $H(P,Q) = -\mathbb{E}_{x\sim P}\log Q(x)$。来源：Deep Learning 式 (3.51)。
- F2（已确认）：分解式 $H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$。来源：同上。
- F3（由 C5 推出的化简，属定义组合）：one-hot 经验分布下 $H(P,Q) = -\log Q(y)$，其中 $y$ 为真实结果。依据：C5（经验分布下 NLL 即交叉熵）+ one-hot 定义；不单独标注外部来源。
- F4（由 F3 与 softmax 定义组合）：多分类交叉熵 $-\log \dfrac{e^{z_y}}{\sum_j e^{z_j}}$。softmax 公式本身为数学常识（页内给定义）；组合形式不单独标注外部来源。
- F5（由 F3 在语言模型场景的组合）：token 级损失 $-\log p_\theta(y_t \mid \text{前文})$。组合形式，标注为定义组合。

## N 数字

- 本页无来源数字要求；手算示例全部为构造数据（$-\ln 0.8 \approx 0.223$ 等，写作时用 Python 计算并直接写入，禁止手抄心算）。

## 来源清单

- Goodfellow, Bengio, Courville, Deep Learning, MIT Press 2016。官方网页版 §3.13（式 3.48–3.51）、§5.5（式 5.59–5.61 段落），https://www.deeplearningbook.org/contents/prob.html 与 /contents/ml.html，已逐条摘录原文。
- Radford et al., Language Models are Unsupervised Multitask Learners, OpenAI 2019, §3.1（canonical prediction unit 表述，已摘录原文；PDF 已下载提取至 /tmp/gpt2.txt 行 335–348）。

## 冲突与不足

- 无冲突项。softmax 的引入不依赖外部来源（数学定义）。
