# Cross-Entropy

- 描述：交叉熵是度量「用一个预测分布去描述真实结果」的平均代价，训练时最小化的就是它，标签只标出真实类别时化简为真实类预测概率的负对数。
- 摘要：把「模型认为真实结果有多可能」变成可求导的代价：$H(P,Q) = -\mathbb{E}_{x\sim P}\log Q(x)$；标签只标出真实类别时化简为 $-\log Q(y)$，最小化它等价于最大似然。
- 主题：训练与优化, 概率与统计
- 标签：训练

## 资料

- Ian Goodfellow, Yoshua Bengio, Aaron Courville. Deep Learning. MIT Press, 2016. ISBN 978-0262035613. [https://www.deeplearningbook.org](https://www.deeplearningbook.org)
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. Language Models are Unsupervised Multitask Learners. OpenAI, 2019. [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 短页

### 用预测概率度量代价

交叉熵按模型给真实结果的概率来算代价。设真实结果有概率分布 $P$，模型给出预测分布 $Q$，交叉熵是
$$H(P,Q) = -\mathbb{E}_{x\sim P}\log Q(x),$$
$\mathbb{E}_{x\sim P}$ 表示按 $P$ 加权平均。$P$ 在结果上取概率 $P(x)$，模型给同一个结果 $Q(x)$，把 $-\log Q(x)$ 按 $P$ 平均就是交叉熵。模型给真实结果的概率越高，交叉熵越小；概率越低，交叉熵越大。它与熵和 KL 散度相关：熵 $H(P) = -\mathbb{E}_{x\sim P}\log P(x)$ 是按 $P$ 自己取平均，KL 散度 $D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}[\log P(x)-\log Q(x)]$ 衡量两个分布的差距。交叉熵等于熵加 KL 散度，KL 散度非负，所以交叉熵不会小于熵，最小值在预测分布与真实分布一致时取到。
依据：blog/deeplearningbook-prob.txt L1683-L1697；blog/deeplearningbook-prob.txt L1728-L1746；blog/deeplearningbook-prob.txt L1751-L1872

### 最小化交叉熵就是最大似然

训练要让已见的数据在新参数下出现得最可能，这就是最大似然。训练不直接最大化数据出现的概率，而是最小化交叉熵，因为数据出现得越可能，交叉熵越小，两者是同一件事。取负号把最大化改成最小化，参数每一步更新都朝交叉熵下降的方向走。
依据：自行构造

### 只算真实类那一项

分类训练把真实类别写成一个向量，只在真实类别那一位置 $1$、其余位置为 $0$，这种表示叫 one-hot。交叉熵按真实分布加权平均，one-hot 只在真实类别那一项带权重，其余项权重为零，平均退化成一项：损失是模型给真实类别概率 $Q(y)$ 的负对数 $-\log Q(y)$。分类任务里 $Q(y)$ 就是模型在真实类别上输出的那一个概率。
依据：blog/deeplearningbook-ml.txt L2658-L2752；blog/deeplearningbook-ml.txt L3883-L3895；blog/deeplearningbook-ml.txt L4444

## 长页

一个分类模型做二分类，结果的类别是「晴」和「非晴」两类。它给「晴」的概率是 $0.8$，给「非晴」的概率是 $1-0.8=0.2$，实际结果正是晴。训练要把这个模型的参数往「给真实结果更高概率」的方向调，得先把这一步的表现算成一个数。看它有没有判对是一种办法：概率 $0.8$ 大于 $0.2$，判对，记 $0$ 分。可是对错只有两种取值，把 $0.8$ 调成 $0.81$，分数仍是 $0$，参数找不到调整的方向。
依据：自行构造

另一种办法看模型给真实结果的概率：概率低，代价就大。给「晴」$0.8$ 得到一个较小的值，给「晴」$0.2$ 得到一个较大的值。交叉熵就是这种办法，它把模型给真实结果的概率取负对数当作代价。
依据：blog/deeplearningbook-prob.txt L1854-L1872；blog/deeplearningbook-ml.txt L2746-L2752

在训练集上把每个样本的代价平均起来，就是这组参数的总代价；让这个总代价最小，等价于让已见数据在这组参数下出现的概率最大。
依据：blog/deeplearningbook-ml.txt L2697-L2752

### 信息量为什么取负对数

一个结果携带多少信息，用自信息（self-information）度量。信息量要满足三点：确定会发生的事件信息量为零；越罕见的事件信息量越大；两个独立事件同时发生，信息量相加。同时满足这三点的取法是 $I(x) = -\log P(x)$。$P(x)$ 是结果 $x$ 发生的概率：确定事件 $P=1$，取负对数得 $0$；概率越小，负对数越大；两事件概率相乘对应取对数后相加。对数底取 $e$ 时单位叫 nats，取 $2$ 时单位叫 bit。交叉熵里的每一项 $-\log Q(x)$，就是模型把结果 $x$ 当成概率 $Q(x)$ 时携带的信息量。
依据：blog/deeplearningbook-prob.txt L1683-L1721

::: 三条要求怎样定到 $-\log P(x)$ 上
1. 把信息量写成概率的某个函数 $f(P)$。
2. 确定事件信息量为零：$f(1)=0$。
3. 独立事件相加：$f(P_1 P_2) = f(P_1) + f(P_2)$。
4. 只有对数函数把乘法变成加法，所以 $f$ 取对数，写成 $f(P) = c\log P$。
5. 越罕见的事件信息量越大，要求 $f$ 随 $P$ 变小而变大，即 $f$ 单调递减；$\log P$ 随 $P$ 变大而变大，所以系数 $c<0$，记 $a=-c>0$，写成 $f(P) = -a\log P$。
6. 取 $a=1$（自然对数）就得到 $I(x) = -\log P(x)$。
依据：自行构造
:::

### 熵与 KL 散度

熵把自信息按分布平均起来。香农熵定义为 $H(P) = \mathbb{E}_{x\sim P}[-\log P(x)]$，即自信息的期望。分布越接近确定，熵越小；越接近均匀，熵越大。离散变量下，它给出把结果写成比特串时平均每个结果所需位数的下界；二值变量的熵在两种结果概率相等时最大。
依据：blog/deeplearningbook-prob.txt L1728-L1746；blog/deeplearningbook-prob.txt L1829-L1853

KL 散度衡量用一个分布替代另一个分布多付的代价。同一结果空间上的两个分布 $P$、$Q$，定义为 $D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{x\sim P}[\log P(x) - \log Q(x)]$。离散变量下，本来用为 $P$ 设计的编码，平均每个结果要用 $-\mathbb{E}_{x\sim P}\log P(x)$ 位；改用为 $Q$ 设计的编码，变成 $-\mathbb{E}_{x\sim P}\log Q(x)$ 位，多出来的就是 KL 散度。它非负，当且仅当两个分布在离散情形下相同（连续情形下几乎处处相等）时为零；它也不对称，$D_{\mathrm{KL}}(P\|Q)$ 一般不等于 $D_{\mathrm{KL}}(Q\|P)$，所以不是距离。
依据：blog/deeplearningbook-prob.txt L1751-L1815

::: 把期望展开成求和
$$D_{\mathrm{KL}}(P\|Q) = \sum_x P(x)\left(\log P(x) - \log Q(x)\right),$$
对每个结果算出 $\log P(x) - \log Q(x)$，再按 $P(x)$ 加权求和。取两个结果的分布 $P = (0.5, 0.5)$、$Q = (0.25, 0.75)$ 算一遍：第一个结果贡献 $0.5 \times \log(0.5/0.25)$，第二个贡献 $0.5 \times \log(0.5/0.75)$。自然对数下 $\log 2 \approx 0.693$、$\log(2/3) \approx -0.405$，两项相加约为 $0.5 \times (0.693 - 0.405) \approx 0.144$。
依据：自行构造
:::

### 交叉熵

把真实分布 $P$ 的结果改用模型分布 $Q$ 来编码，平均每个结果的代价就是交叉熵，定义为
$$H(P,Q) = -\mathbb{E}_{x\sim P}\log Q(x),$$
$\log$ 取以 $2$ 为底的对数时，它的值就是平均所需位数；下面用自然对数，单位是 nats。$P$ 是结果的真实分布，$Q$ 是模型给出的预测分布。它和熵只差一个分布：把 $-\mathbb{E}_{x\sim P}\log Q(x)$ 里的 $Q$ 换成 $P$，得到熵 $H(P)$；两者相减得到 KL 散度 $D_{\mathrm{KL}}(P\|Q)$，于是
$$H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$
$H(P)$ 只由真实分布决定，与 $Q$ 无关，所以对 $Q$ 求最小的时候，交叉熵与 KL 散度取到同一个最小值点：最小化交叉熵等价于最小化 KL 散度，交叉熵能降到的最小值就是 $H(P)$。计算中出现 $0\log 0$ 时，按信息论的惯例取 $\lim_{x\to 0} x\log x = 0$。
依据：blog/deeplearningbook-prob.txt L1729-L1883

### 训练里的交叉熵

训练集是一组样本，每个样本带一个真实标签。经验分布把训练集看成等可能的结果：大小为 $N$ 的样本集，第 $i$ 个样本的概率是 $1/N$。最大似然估计选参数 $\theta$，让这组样本在模型下出现的概率最大：
$$\theta_{\mathrm{ML}} = \arg\max_\theta \prod_{i=1}^{N} Q(x^{(i)};\theta).$$
$Q$ 是模型给出的预测分布，$x^{(i)}$ 是第 $i$ 个样本，$\theta$ 是模型参数。连乘容易数值下溢，取对数把它变成求和，且不改变取到最大值的那个 $\theta$；再除以样本数 $N$，写成对经验分布求期望：
$$\theta_{\mathrm{ML}} = \arg\max_\theta \mathbb{E}_{x\sim\hat{p}_{\mathrm{data}}}\log Q(x;\theta).$$
把最大化改成最小化（取负号），按交叉熵的定义写，就是最小化经验分布与模型分布之间的交叉熵。分类训练把真实类别写成 one-hot：一个长度等于类别数的向量，只有真实类别那一位置 $1$，其余位置为 $0$。按 one-hot 求平均，只有真实类别那一项带权重 $1$，其余项权重是 $0$，交叉熵只剩 $-\log Q(y)$，其中 $y$ 是真实类别，$Q(y)$ 是模型给真实类别的概率。
依据：blog/deeplearningbook-prob.txt L1048-L1067；blog/deeplearningbook-ml.txt L2658-L2752；blog/deeplearningbook-ml.txt L3883-L3895；blog/deeplearningbook-ml.txt L4444

::: 二分类和多分类的手算
1. 接前面二分类的例子，真实类别是「晴」：模型给「晴」的概率 $0.8$ 时，交叉熵 $-\log 0.8 \approx 0.223$；概率 $0.2$ 时，$-\log 0.2 \approx 1.609$。
2. 多分类，三个类别，模型给真实类别的概率 $0.7$：交叉熵 $-\log 0.7 \approx 0.357$。
3. 同一个多分类样本，模型给真实类别的概率 $0.1$：交叉熵 $-\log 0.1 \approx 2.303$。
依据：自行构造
:::

### 下一 token 的交叉熵

一段文本先被切成 token（词、子词或字符），全部可用的 token 组成词表。语言模型给这段文本的每个位置算出下一个位置上各个 token 的概率分布：softmax 把模型在这一位置的一组打分变成词表上的概率分布（每个打分取指数再归一）。把真实的下一个 token 记为 $y_t$，它前面的 token 记为 $y_{<t}$；模型在这几个 token 的条件下给出预测分布 $Q$，这一位置的交叉熵是 $-\log Q(y_t)$，$Q(y_t)$ 是模型给真实 token $y_t$ 的概率。把这段文本各位置的负对数概率按位置平均，得到这段文本的损失。
依据：paper/gpt2.txt L64-L75；blog/deeplearningbook-ml.txt L2775-L2815

语言建模的结果常以平均负对数概率报告：把整段文本的负对数概率按字符、字节或词这类预测单元取平均。
依据：paper/gpt2.txt L226-L235

::: 小词表上逐位置算一遍
1. 取词表 $\{\text{A}, \text{B}, \text{C}\}$（$3$ 个 token）。文本有 $3$ 个 token，从第 $1$ 个位置起每步预测下一个 token，两个预测位置的真实下一个 token 分别是 B、C。
2. 位置 $1$：模型给 B 的概率 $0.6$，损失 $-\log 0.6 \approx 0.511$。
3. 位置 $2$：模型给 C 的概率 $0.5$，损失 $-\log 0.5 \approx 0.693$。
4. 两个位置平均，$(0.511 + 0.693)/2 \approx 0.602$，这就是这段文本的损失。
依据：自行构造
:::

### 适用条件与边界

交叉熵要成立，结果空间得先定下来，$P$ 是结果真实（或训练集经验）分布，$Q$ 是模型分布。任何由负对数似然（把模型给真实结果的概率取负对数，再按样本平均）组成的损失，都是这个训练集的经验分布与模型分布之间的交叉熵；常见的误解是以为只有 softmax 或伯努利输出才有交叉熵。均方误差也是交叉熵，它是经验分布与一个高斯模型之间的交叉熵（把高斯分布的对数似然展开，指数被对数消掉，只剩误差的平方项）。
依据：blog/deeplearningbook-prob.txt L1854-L1883；blog/deeplearningbook-ml.txt L2746-L2752；blog/deeplearningbook-ml.txt L2886-L2951
