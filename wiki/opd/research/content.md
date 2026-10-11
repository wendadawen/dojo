# OPD

- 描述：OPD 在训练时让学生模型（被训练的小模型）自己生成序列，教师模型（提供监督的强模型）对这条序列里学生实际写下的每个 token（文本最小单位）逐个打分，分数作训练信号更新学生。
- 摘要：训练序列由学生自己采样，监督是教师逐 token 给出的对数概率比 $r=\log\pi_{\text{teacher}}(y_t)-\log\pi_S(y_t)$，它充当策略梯度里的 advantage；逐 token 的稠密反馈把每个 episode 的信号从与 token 数无关的 $O(1)$ 提高到 $O(N)$。
- 主题：训练与优化, 模型结构
- 标签：训练

## 资料

- Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. [https://arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649)
- Kevin Lu, Thinking Machines Lab. On-Policy Distillation. Thinking Machines Lab: Connectionism, 2025. [https://thinkingmachines.ai/blog/on-policy-distillation/](https://thinkingmachines.ai/blog/on-policy-distillation/)

## 短页

### 写自己的句子，让教师逐字批改

OPD（On-Policy Distillation，在线策略蒸馏）是一种教师-学生式的训练方法。训练时先由学生模型自己生成一条序列，再让教师模型对这条序列里学生实际写下的每个 token 逐个打分：这个 token 在教师分布下比在学生分布下更可能出现，分数就为正，反之分数为负。这些分数就是训练信号，用来更新学生。这里 token 指语言模型（按前文预测下一个 token 的模型）处理的文本最小单位，学生是被训练的模型，教师是提供监督的、能力较强的模型；策略指模型在当前参数下给下一个 token 的分布。和常见的[蒸馏](wiki/knowledge-distillation)（让学生模型去匹配教师模型的输出分布）一样，学生学的是教师的输出分布；不同的是训练用的序列由学生自己写，而不是由教师或数据集提供。
依据：blog/tml-on-policy-distillation.txt L91；blog/tml-on-policy-distillation.txt L111；blog/tml-on-policy-distillation.txt L65；blog/tml-on-policy-distillation.txt L75

### 训练时见不到自己写的前缀

自回归模型每一步用前面已生成的 token 预测下一个，一次只生成一个。如果在教师写的序列或固定数据集上训练，学生学到的都是在那些前缀之后该怎么写；可它推理时要接的前缀是自己写的，两者分布不一致。推理指模型运行、按输入生成输出。一旦学生早期写下一个教师不会写的 token，它就进入训练时从未出现过的状态，后面的预测建立在没学过的前缀上，错误逐步放大。序列越长，越可能出现这种偏离，后果也越重。要让学生学会从自己的错误里恢复过来，训练时的前缀就得是它自己会写出来的。OPD 的做法是把训练序列换成学生自己的生成结果，让教师的反馈落在学生实际到达的状态上。
依据：blog/tml-on-policy-distillation.txt L79；paper/gkd-v3.txt L184

### 采样、逐 token 打分、进策略梯度

学生从输入出发生成一条序列，采样过程中它已经给出自己每一步的对数概率（logprobs，即模型对词表——它能输出的全部 token 的集合——里每个 token 给出的对数概率）。接着把同一条序列交给教师再算一遍，得到教师在同样那些 token 上的 logprobs，这一步只取标量分数，不需要教师的梯度。最后逐 token 算出学生与教师对数概率之差，取它的相反数，这个值记作该 token 的 advantage，即策略梯度里与生成概率的梯度相乘的标量；策略梯度是直接用生成每个 token 的概率的梯度更新模型参数的一类做法。按策略梯度的规则，正的 advantage 抬高该 token 的对数概率，负的压低它。参数只更新学生，教师不动。
依据：blog/tml-on-policy-distillation.txt L111；blog/tml-on-policy-distillation.txt L135；blog/tml-on-policy-distillation.txt L137；blog/tml-on-policy-distillation.txt L139；blog/tml-on-policy-distillation.txt L141

## 长页

OPD（On-Policy Distillation，在线策略蒸馏）是一种教师-学生式的训练方法：训练时由学生模型自己生成序列，教师模型对学生实际写下的每个 token 逐个打分，分数作为训练信号更新学生。它的监督方式是[蒸馏](wiki/knowledge-distillation)——让学生匹配教师的输出分布；这里 token 是语言模型（按前文预测下一个 token 的模型）处理的文本最小单位，学生是被训练的模型，教师是提供监督的强模型，推理指模型按输入生成输出。训练不需要另训一个给输出打分的模型。常见做法是在人写的答案或教师生成的序列上训练学生，训练时学生见到的是别人写的前缀，推理时却要接自己写的前缀，两者分布不一致，误差会随序列变长放大。另一种做法是让学生自己写、再按一个标量奖励做强化学习（按奖励调整参数的做法），奖励只在整条序列结束时给一次，反馈很稀疏。OPD 把两者合起来：训练分布是学生自己的，监督是教师在每个 token 上的逐位置反馈。
依据：blog/tml-on-policy-distillation.txt L63-L65；blog/tml-on-policy-distillation.txt L71；blog/tml-on-policy-distillation.txt L75；blog/tml-on-policy-distillation.txt L79；blog/tml-on-policy-distillation.txt L91；blog/tml-on-policy-distillation.txt L121；paper/gkd-v3.txt L184

### 在别人写的序列上训练，推理要接自己的序列

自回归模型一次生成一个 token，每个预测都以之前的 token 为条件。常见做法是用固定的输出序列训练学生，来源可以是人写的标准答案，也可以是教师生成的序列。学生的参数更新，基于的都是这些序列里的前缀。问题出在推理：这时学生要接的前缀是它自己一步步写出来的。训练时见到的部分序列与推理时生成的部分序列分布不一致，这就是训练与推理之间的分布失配。

失配的后果是误差复合。学生一旦在某一步写下教师不会写的 token，此后它面对的整个前缀就偏离了训练分布；在这个没学过的前缀上，它的预测更容易继续出错，错误一步步累积，生成的文本质量随之下降。序列越长，出现一个早期错误、并让后面所有预测都建立在错误前缀上的机会越多。
依据：paper/gkd-v3.txt L184；paper/gkd-v3.txt L108；blog/tml-on-policy-distillation.txt L79

### 把训练分布换成学生自己的分布

失配的根源是训练序列不是学生自己会生成的。这里策略指模型在当前参数下给下一个 token 的分布。按训练序列由谁生成，训练分两类：用固定的、外部给定的序列来训练，学生学的是别人写的输出，这类叫 off-policy；用学生当前策略自己采样出的序列来训练，训练分布就是学生自己的分布，这类叫 on-policy。教师-学生式的训练按这两类分，就得到 off-policy 蒸馏与 on-policy 蒸馏。

这两类不是互斥的两种方法，而是同一个框架的两个端点。把固定数据集项和学生自采样项按一个比例混合，就得到更一般的框架（Generalized Knowledge Distillation，GKD）；学生自采样占的比例记作 $\lambda$，取值在 0 到 1 之间。$\lambda$ 取 0（只用固定数据集）就是监督蒸馏，$\lambda$ 取 1（只用学生自采样）就是 on-policy 蒸馏。无论取哪个端点，比较师生分布用的都是同一套散度：散度衡量两个概率分布差多远，这里在序列的每个位置比较一次师生的下一个 token 分布。
依据：paper/gkd-v3.txt L241-L248；paper/gkd-v3.txt L186；blog/tml-on-policy-distillation.txt L65；paper/gkd-v3.txt L143-L146；paper/gkd-v3.txt L163

::: 逐位置的散度
选哪一种散度、朝哪个方向比，是框架里的另一个自由选择。最常用的是 KL 散度，$\mathrm{KL}(P\|Q)=\sum_c P(c)\log\frac{P(c)}{Q(c)}$，其中 $P$、$Q$ 是任意两个分布，$c$ 遍历词表（模型能输出的全部 token 的集合）；它不对称，$\mathrm{KL}(P\|Q)$ 与 $\mathrm{KL}(Q\|P)$ 是两个不同的量，分别叫 forward KL 与 reverse KL。
依据：paper/gkd-v3.txt L242；paper/gkd-v3.txt L143-L146

对输出序列 $y$ 和输入 $x$，逐位置的散度在每个位置比较师生的下一个 token 分布，再对序列长度 $L_y$ 取平均（GKD 原文把教师、学生的分布记作 $p_T$、$p_S^\theta$，$\theta$ 是学生参数，$y_{<n}$ 是第 $n$ 个 token 之前的前缀）：
$$\mathcal{D}\big(p_T\|p_S^\theta\big)(y|x) := \frac{1}{L_y} \sum_{n=1}^{L_y} \mathcal{D}\big(p_T(\cdot|y_{<n},x) \,\|\, p_S^\theta(\cdot|y_{<n},x)\big)$$
依据：paper/gkd-v3.txt L163
:::

训练不对学生的采样分布求梯度，也就是不让梯度反传（沿计算图逐层传回参数）到采样这一步；这让训练更稳定、计算也更省。
依据：paper/gkd-v3.txt L248；paper/gkd-v3.txt L193

### 教师在每个 token 上打分

第一步，学生从输入出发，按自己的当前参数生成一条序列；生成的时候它已经在每个位置算过自己的对数概率（logprobs，模型对词表每个 token 给的对数概率），把这些值留下即可。

第二步，把这条由学生写出的序列交给教师，做一次前向（模型按输入算一遍输出），得到教师在同样那些 token 上的对数概率。教师只输出这些标量，不输出梯度。

第三步，逐 token 比较两边的对数概率。学生采到的第 $t$ 个 token 是 $y_t$。用学生在 $y_t$ 上的对数概率减教师在同一 token 上的对数概率，得到学生分布对教师分布的逐 token reverse KL 的一次采样值；reverse KL 是把学生分布当第一个分布、教师分布当第二个分布的那种比较方向。取它的相反数，即 $\log\pi_{\text{teacher}}(y_t)-\log\pi_S(y_t)$，记作该 token 的分数 $r$。$r$ 为正表示教师给学生写下的这个 token 的概率高于学生给的；教师给该 token 的概率比学生高得越多，分数越高。

这个分数充当该 token 的 advantage，也就是策略梯度里与生成概率的梯度相乘的标量；策略梯度按生成概率的梯度更新参数，$\pi_S(y_t)$ 是学生给 $y_t$ 的下一 token 概率。$r$ 为正就抬高该 token 的对数概率，为负就压低。每步只更新学生的参数，教师始终不动。

这样做的信号逐 token 且稠密：一个 episode（一次生成整条序列）里每个 token 各带一份反馈，反馈的信息量随 token 数增长。
依据：blog/tml-on-policy-distillation.txt L91；blog/tml-on-policy-distillation.txt L111-L113；blog/tml-on-policy-distillation.txt L119；blog/tml-on-policy-distillation.txt L135；blog/tml-on-policy-distillation.txt L137；blog/tml-on-policy-distillation.txt L139；blog/tml-on-policy-distillation.txt L141；blog/tml-on-policy-distillation.txt L285

::: 分数的期望与 reverse KL
把分数 $r$ 在学生采到的 token 上取期望（$y_t$ 是学生实际采到的第 $t$ 个 token），得到的就是负的逐 token reverse KL：
$$\mathbb{E}_{y_t\sim\pi_S}\big[r\big] = -\mathrm{KL}\big(\pi_S \,\|\, \pi_{\text{teacher}}\big)$$
依据：blog/tml-on-policy-distillation.txt L111
:::

### 一个手算的逐 token 打分

取一个三 token 的词表，学生在某个前缀下的下一个 token 分布与教师在同一前缀下的分布，学生实际写下其中一个 token，按上文给出的 $r$ 计算它的分数。
依据：自行构造

::: 三个 token 的手算全过程
词表记作 $\{A,B,C\}$。学生在当前前缀下的下一个 token 分布是 $\pi_S=(0.3,\,0.4,\,0.3)$，教师在同一前缀下的分布是 $\pi_{\text{teacher}}=(0.6,\,0.3,\,0.1)$。分数按上文给出的 $r$ 计算，对数取自然对数。

1. 学生采到 $A$：$r=\log(0.6/0.3)=\log 2\approx 0.693$，为正，教师给 $A$ 的概率高于学生，更新会抬高 $A$ 的对数概率。
2. 学生采到 $B$：$r=\log(0.3/0.4)=\log 0.75\approx -0.288$，为负，教师给 $B$ 的概率低于学生，更新会压低 $B$ 的对数概率。
3. 学生采到 $C$：$r=\log(0.1/0.3)=\log(1/3)\approx -1.099$，负得更多，教师给 $C$ 的概率最低，$C$ 被压得更多。
4. 把 $r$ 在学生的采样上取期望，核对它等于负的 reverse KL：
$$\mathbb{E}[r]=0.3\times 0.693 + 0.4\times(-0.288) + 0.3\times(-1.099)\approx -0.237$$
$$\mathrm{KL}\big(\pi_S\|\pi_{\text{teacher}}\big)=0.3\log\frac{0.3}{0.6}+0.4\log\frac{0.4}{0.3}+0.3\log\frac{0.3}{0.1}\approx 0.237$$
两者相差一个负号，符合 $\mathbb{E}[r]=-\mathrm{KL}(\pi_S\|\pi_{\text{teacher}})$。
依据：自行构造
:::

分数只用到师生两个分布和实际采到的那个 token；$r$ 的正负与大小取决于学生采到哪个 token。
依据：自行构造

### 适用条件与边界

要在学生的任意状态上查教师的打分，训练框架要能对教师做前向、取出它在指定 token 上的 logprobs。学生初始化要覆盖教师的支持集，也就是教师会写出、而学生也能以非零概率写出的那部分 token；覆盖不足时，教师会写出的一部分 token 学生几乎采不到，采不到就拿不到这些 token 上的分数，需要显著更大的批量（一次并行处理的样本数，batch）来提高采到它们的机会。若这类 token 很多，通常先用监督式的训练扩大学生的支持集，再进入 OPD。

OPD 的信号方向指向教师分布。有的教师由强化学习训练而来，过程中会经历一系列中间策略、最后收敛到一个；OPD 不建模这些中间策略，只学最终收敛到的那个策略。

在数据使用上，OPD 学的是教师的完整分布而不是单个答案，因此可以反复用同一个 prompt（一次训练的输入）多训几轮；靠标量奖励的强化学习对同一个 prompt 反复训练多个 epoch（把同一批数据过几遍）容易退化成只记住最终答案。
依据：blog/tml-on-policy-distillation.txt L115；blog/tml-on-policy-distillation.txt L119；blog/tml-on-policy-distillation.txt L161；blog/tml-on-policy-distillation.txt L277；blog/tml-on-policy-distillation.txt L303；blog/tml-on-policy-distillation.txt L325；blog/tml-on-policy-distillation.txt L309
