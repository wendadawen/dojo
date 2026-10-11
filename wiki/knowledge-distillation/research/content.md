# Knowledge Distillation

- 描述：知识蒸馏是把一个已训练大型模型的知识迁移到小型模型的方法，做法是让小模型去拟合大模型给出的输出概率分布，而不是只拟合真实标签。
- 摘要：在 softmax 里加热度 $T$，大模型输出的软目标里带着类与类之间的相似结构；小模型拟合这些软目标，常能与大模型的泛化能力接近，而规模小得多、便于部署。
- 主题：训练与优化, 概率与统计
- 标签：训练

## 资料

- Geoffrey Hinton, Oriol Vinyals, Jeff Dean. Distilling the Knowledge in a Neural Network. NIPS 2014 Deep Learning Workshop; arXiv:1503.02531v1, 2015. [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

## 短页

### 学生模仿老师的输出分布

知识蒸馏用一个已经训练好的大模型（老师）去训练一个小模型（学生），学生的目标不是只学真实标签，而是模仿老师对每个输入给出的整个概率分布。模型最后一层的 softmax 把每个类别的原始打分变成一组和为 1 的概率，老师在这个分布里不只给出哪个答案对，还给出其它每个答案各自的概率。学生拟合整个分布，也就学到了老师「认为哪些类别彼此相像」。
依据：paper/src-v1/distillation11.tex Distillation；paper/src-v1/introduction11.tex Introduction

### 部署要省，真实标签信息不足

训练和部署的要求不同：训练可以从大量数据里慢慢提取结构，不必实时，可以用很多算力；部署到大量用户时要受延迟和算力的限制，模型因此要小。真实标签每个样本只指出一个正确类别，丢掉了错误类别之间的相对大小；而老师概率分布里错误类别的相对大小，大体反映了它把新数据分对的方式。老师本身在新数据上做得好时，学生模仿这个分布，在新数据上也往往比只学真实标签做得更好。
依据：paper/src-v1/introduction11.tex Introduction

### 温度与软目标

给老师的 softmax 加一个温度 $T$：$T=1$ 时是原来的分布，$T$ 越大分布越平，原本接近 0 的概率被放大，这样的输出叫软目标。用同一个温度让学生的 softmax 去拟合软目标，训练完学生推理（用训练好的模型对新输入算出预测）时把温度调回 $T=1$。如果手里有真实标签，再让学生同时拟合真实标签，两项目标（各是一个[交叉熵](wiki/cross-entropy)）按权重相加，真实标签那一项权重取得低；因为软目标产生的梯度随 $T^2$ 缩小，这一项要乘 $T^2$，换温度时两项的相对影响才大体不变。
依据：paper/src-v1/distillation11.tex Distillation

## 长页

一个分类模型的最后一层通常是一个 softmax：它把模型为每个类别算出的原始分数 $z_i$ 变成一组概率 $q_i$，把所有类别的分数一起比较，
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$
其中分母对所有类别 $j$ 求和，$T$ 是温度，通常取 1。分数 $z_i$ 也叫 logit，是 softmax 的输入。
依据：paper/src-v1/distillation11.tex Distillation

模型在训练时要面对又大又冗余的数据，从中提取结构，这个过程不必实时，可以用很多算力；部署到大量用户时却对延迟和算力有更严的要求。同一个模型很难同时满足这两件事，所以训练阶段和部署阶段可以用不同的模型：训练时用一个又大又笨重的模型，它可以是若干个分别训练的模型组成的整体，也可以是一个很大、加了强正则化的模型（正则化指训练时给要最小化的那个函数加额外的约束项）；部署时再用一个小模型。
依据：paper/src-v1/introduction11.tex Introduction；paper/src-v1/abstract11.tex L3-L20

知识蒸馏就是用一个训练过程连接这两个阶段的一种训练方式：先有一个已经训练好的大模型，再训练一个小模型去模仿它。它拿大模型输出的概率分布当训练目标，所以大模型叫老师，小模型叫学生；学生要模仿的正是老师输出的整个概率分布，而不是只学真实标签。
依据：paper/src-v1/introduction11.tex Introduction

### 软目标：为什么概率分布比标签信息多

老师对每个输入都给所有类别打了分，即使它几乎总能给出正确答案，错误类别的概率之间也分出了大小。以数字识别为例，一张 2 的图可能被老师给 3 一个很小的概率、给 7 更小的概率，换一张潦草的 2 则可能反过来。这个「哪个错误答案更像它」的相对大小，是老师对数据相似结构的认识；真实标签只说这张图是 2，把这一层丢掉了。
依据：paper/src-v1/introduction11.tex Introduction

拿这样的概率分布当训练目标，叫软目标；只用真实标签当目标，叫硬目标。分布较平（各类概率差距较小）的软目标比硬目标携带的信息更多：每个样本传递的类别间关系信息更多，样本之间的梯度方差也更小。学生因此常常能用比老师少得多的数据来训练，并可以用更大的学习率（学习率是每步参数更新的幅度）。
依据：paper/src-v1/introduction11.tex Introduction

软目标本身有个障碍：老师很自信时，错误类别的概率接近 0，比如一个是 $10^{-6}$、一个是 $10^{-9}$，两者相差很大，却对[交叉熵](wiki/cross-entropy)代价函数（训练时要最小化的那个函数）几乎没有影响，因为都太接近 0。要让这部分信息起作用，就得把分布变软，办法是给 softmax 加温度。
依据：paper/src-v1/introduction11.tex Introduction

### 温度与两个目标函数

最简单的蒸馏形式是：给老师的 softmax 加一个较高的温度 $T$，得到一组较软的软目标；再用同样的高温训练学生，让学生的 softmax 去逼近这些软目标；学生训练好之后，推理（用训练好的模型对新输入算出预测）时用回 $T=1$。温度只作用于每个样本的分数，不改变各类的先后顺序，只是把整体的分布压平。
依据：paper/src-v1/distillation11.tex Distillation

训练学生用的数据集合叫迁移集，可以用老师训练时用的那一份，也可以用一份没有标签的数据。只知道软目标还不够时，把真实标签也用上会更好：学生同时拟合两者。一种做法是让两项代价加权相加，第一项是学生与老师软目标之间的交叉熵，学生这一侧用与老师相同的高温；第二项是学生与真实标签之间的交叉熵，学生这一侧用 $T=1$，两项用的是同一组 logit；第二项的权重通常比第一项低很多。
依据：paper/src-v1/distillation11.tex Distillation；paper/src-v1/introduction11.tex Introduction

两项一起用时要把软目标那一项乘 $T^2$：软目标产生的梯度大小按 $1/T^2$ 缩小，乘上 $T^2$ 后，换温度做实验时两项的相对贡献大体不变。
依据：paper/src-v1/distillation11.tex Distillation

::: 温度把分布压平：一个能手算的例子
取三个类别的分数 $z_1=2$、$z_2=1$、$z_3=0$，放进 softmax 公式算概率。
依据：自行构造

1. 用 $T=1$：三个指数是 $e^2$、$e^1$、$e^0$，约 $7.389$、$2.718$、$1$，和为 $11.107$，概率约 $0.665$、$0.245$、$0.090$。
2. 用 $T=4$：分数先各除以 4，得 $0.5$、$0.25$、$0$，三个指数约 $1.649$、$1.284$、$1$，和为 $3.933$，概率约 $0.419$、$0.326$、$0.254$。
3. 温度从 1 升到 4，最小的第三个类别从 $0.090$ 升到 $0.254$，分布被压平，最小的概率被放大得最明显。
依据：自行构造

```python
import math

z = [2.0, 1.0, 0.0]
for T in (1, 4):
    e = [math.exp(x / T) for x in z]
    s = sum(e)
    print("T =", T, [round(x / s, 3) for x in e])
```

```text
T = 1 [0.665, 0.245, 0.09]
T = 4 [0.419, 0.326, 0.254]
```
:::

#### 匹配 logits 是蒸馏的特例

学生每个样本都对每个 logit 产生一个交叉熵梯度。把学生的 logit 记为 $z_i$、老师的 logit 记为 $v_i$，老师软目标的概率记为 $p_i$、学生当前的记为 $q_i$，代价函数记为 $C$，在温度 $T$ 下这个梯度是
$$\frac{\partial C}{\partial z_i} = \frac{1}{T}\left(q_i - p_i\right)$$
温度相对 logit 的量级很高时，可以把指数做一阶近似；再假设每个样本的 logit 各自减去自己的均值（去均值），梯度化简为 $\frac{1}{NT^2}(z_i-v_i)$，$N$ 是类别数。也就是说，在高温极限下蒸馏等价于最小化 $\frac{1}{2}(z_i-v_i)^2$，即让学生的 logit 去匹配老师的 logit。温度较低时，蒸馏对远低于均值的 logit 关注得少：这些 logit 在训练老师时几乎不受约束、可能带有较大噪声，但也可能带着有用信息，哪一种效应更强要靠实验判断。
依据：paper/src-v1/distillation11.tex Matching logits is a special case of distillation

::: 从交叉熵梯度到匹配 logits 的推导
依据：paper/src-v1/distillation11.tex Matching logits is a special case of distillation

1. 写出梯度。学生对第 $i$ 个 logit 的交叉熵梯度等于学生概率与老师概率之差再乘 $1/T$，即 $\frac{\partial C}{\partial z_i}=\frac{1}{T}(q_i-p_i)$。
2. 高温下把指数一阶近似。$e^{z_i/T}$ 在 $T$ 远大于 logit 量级时约等于 $1+z_i/T$，于是 $\frac{\partial C}{\partial z_i}\approx\frac{1}{T}\left(\frac{1+z_i/T}{N+\sum_j z_j/T}-\frac{1+v_i/T}{N+\sum_j v_j/T}\right)$，$N$ 是类别数。
3. 去均值化简。若每个样本的 logit 各自减去均值，则 $\sum_j z_j=\sum_j v_j=0$，上式化为 $\frac{\partial C}{\partial z_i}\approx\frac{1}{NT^2}(z_i-v_i)$。
4. 读出结论。$\frac{1}{2}(z_i-v_i)^2$ 对 $z_i$ 的梯度是 $z_i-v_i$，与上式只差一个正常数因子 $1/(NT^2)$，两者的极小点相同，所以在高温极限下蒸馏等价于匹配 logit。
依据：自行构造
:::

### 软目标传了什么：几个实验

在手写数字数据集 MNIST 上，一个小模型直接训练有 146 个错误；改在原来的训练之外再加一项拟合老师软目标的任务、并把它当作唯一的正则手段（老师是一个很大的模型，测试错误 67 个），这个学生降到 74 个错误。软目标把「怎么适应新数据」也传了过去。
依据：paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST

迁移集里完全没有数字 3 时，学生也几乎能认对它：错误大多来自 3 这一类学到的偏置（每个类别的可调常数）太低，把偏置调高后，学生认对了 98.6% 的 3。数据很少时软目标仍然有效：只用 3% 的语音训练数据，硬目标训练会记住训练数据、在新数据上变差，软目标训练却接近用全量数据的结果。
依据：paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST；paper/src-v1/regularizer11.tex Soft Targets as Regularizers

在语音识别上，把 10 个模型的平均结果蒸馏到单个学生模型：学生与 10 个模型平均的表现接近，规模和单个模型一样。
依据：paper/src-v1/speech11.tex Experiments on speech recognition；paper/src-v1/speech11.tex Results

::: MNIST 与语音实验的条件和数字
MNIST：老师用全部 60000 个训练样本，两层各 1200 个整流线性单元（对输入取正部分的单元），训练时随机丢弃一部分单元（dropout）、并限制权重的大小，输入图像最多平移抖动 2 个像素；小模型两层各 800 个单元、不做正则。迁移集省略数字 3 时，学生共 206 个错误，其中 133 个落在测试集的 1010 个 3 上；把 3 的偏置加 3.5 后，学生 109 个错误，只有 14 个落在 3 上。迁移集只留 7 和 8 时，学生测试错误率 47.3%；把 7、8 的偏置减 7.6 后降到 13.2%。
依据：paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST

语音：只用 3%（约 20M 个样本）训练那个约 85M 个参数的模型，硬目标在训练数据上分对 67.3%、在新数据上 44.5%（准确率开始下降时就要早点停）；软目标在训练数据上 65.4%、在新数据上 57.0%；用全量数据的硬目标基线是 63.4%、58.9%。这里每个数都是分类正确的比例。
依据：paper/src-v1/regularizer11.tex Soft Targets as Regularizers

语音识别的集成（把多个模型的输出合起来）与蒸馏：模型是 8 个中间层、每层 2560 个整流线性单元、输出 14000 个状态（即 14000 个输出类别），约 85M 个参数；单个基线在新数据上分对 58.9%、词错率（认错的词占全部词的比例）10.9%；10 个模型的平均是 61.1%、10.7%；蒸馏出的单模型是 60.8%、10.7%。
依据：paper/src-v1/speech11.tex Experiments on speech recognition；paper/src-v1/speech11.tex Results
:::

### 适用条件与边界

一个常见的误解是把训练好的模型里的「知识」等同于它那组参数；知识其实是输入到输出的映射，正因为如此，换一个更小的模型、用软目标去训练，它能被保留下来。
依据：paper/src-v1/introduction11.tex Introduction

温度不是越高越好。学生远小于老师、无法容纳老师的知识时，中间温度表现更好：把学生压到每层 30 个单元时，2.5 到 4 的温度比更高或更低的都好；学生足够大（每层 300 个单元以上）时，8 以上的温度结果接近。老师本身在新数据上做得好时，学生也往往在新数据上更好。
依据：paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST；paper/src-v1/distillation11.tex Matching logits is a special case of distillation；paper/src-v1/introduction11.tex Introduction
