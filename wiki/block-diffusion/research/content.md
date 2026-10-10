# Block Diffusion

- 描述：块扩散语言模型把 token 序列切成块，块间按自回归从左到右建模，块内用离散掩码扩散并行去噪。
- 摘要：块间自回归、块内并行去噪，块大小 $L'$ 在自回归（$L'=1$）与扩散（$L'=L$）之间插值，块间可复用 [KV cache](wiki/kv-cache)，能生成任意长度的序列。
- 主题：模型结构, 训练与优化
- 标签：模型架构

## 资料

- Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, Volodymyr Kuleshov. Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models. ICLR 2025 (Oral), arXiv:2503.09573, 2025. [https://arxiv.org/abs/2503.09573](https://arxiv.org/abs/2503.09573)
- Kuleshov Group. BD3-LMs: official implementation. 2025. [https://github.com/kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)
- Marianne Arriola 等. Block Diffusion: project page. 2025. [https://m-arriola.com/bd3lms/](https://m-arriola.com/bd3lms/)

## 短页

### 分块去噪的语言模型

token 是模型读写文本的最小单位，可以是一个词或词的一部分。自回归生成指每次只预测下一个 token，用前面所有 token 作条件。块扩散语言模型把一条 token 序列切成固定长度的块，块与块之间按自回归从左到右生成，块内部并行还原。块内用的是掩码扩散：把块里的一部分 token 换成掩码符号，模型一次给出所有被掩位置的原 token 预测，掩码符号是一个专用来占位的符号。
依据：paper/src/iclr2025_conference.tex L226-L230；paper/src/iclr2025_conference.tex Masked \algos{}；paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures

### 扩散模型缺的东西它补上

离散扩散语言模型一次为整条序列去噪，生成时每一步都要处理整条序列，已经生成的部分不能在下一步复用。模型处理序列时靠一种叫注意力的计算，让每个位置和别的位置互相比较、交换信息，这种计算为每个位置算出几个向量，其中两个叫键和值，供别的位置取用。块扩散把序列切成块后，块间是自回归的，已经生成完的块不再改动，它的键和值可以留下来复用，这就是 [KV cache](wiki/kv-cache) 省下的那部分重算。同时，块内可以并行给出多个 token，不必像自回归那样一次一个。块大小在两端之间调：块等于 1 就是逐 token 的自回归模型，块等于整条序列就退化成一次为全序列去噪的扩散模型。块扩散给出的似然（模型给整段文本的概率）不再受固定长度限制，可以连续生成任意多个块，得到比训练时的序列长度更长的序列。
依据：paper/src/iclr2025_conference.tex Introduction；paper/src/iclr2025_conference.tex Likelihood Evaluation

### 切块、块间自回归、块内并行

1. 把序列切成 $B$ 个长度为 $L'$ 的块，似然按块相乘，块 $b$ 的条件概率只依赖它之前的块。
2. 每个块内部用离散掩码扩散去噪：按一个掩码率把块里的 token 掩掉一部分，模型预测被掩位置的原 token，逐步降低掩码率。
3. 生成时一次出一个块，块内并行去噪，生成完这个块后把它的原 token 与它算出的键和值追加缓存，再生成下一个块。
依据：paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures；paper/src/iclr2025_conference.tex Masked \algos{}；paper/src/iclr2025_conference.tex Forward Process；paper/src/iclr2025_conference.tex Reverse Process；paper/src/iclr2025_conference.tex Sampling

## 长页

块扩散语言模型（block diffusion）在两个已有的语言模型之间插值。自回归模型把一条序列 $\mathbf{x} = (\mathbf{x}^1, \dots, \mathbf{x}^L)$ 的似然拆成每个位置在前文条件下的概率之积，$\log p_\theta(\mathbf{x}) = \sum_{\ell=1}^{L} \log p_\theta(\mathbf{x}^\ell \mid \mathbf{x}^{<\ell})$，其中 $L$ 是 token 数，$\mathbf{x}^{<\ell}$ 是位置 $\ell$ 之前的所有 token，$p_\theta$ 是模型给出的概率；生成时每个 token 依赖前一个，$L$ 个 token 要 $L$ 步。离散去噪扩散模型改从噪声还原：先定义一个把干净序列逐步加噪的前向过程，再训练一个网络执行逆向过程，它的似然是真实似然的一个下界，对整条序列可以并行处理。
依据：paper/src/iclr2025_conference.tex L292-L301；paper/src/iclr2025_conference.tex L354-L377

块扩散取两者的折中。它把序列切成 $B$ 个长度为 $L'$ 的块，块间按自回归建模，块内用离散扩散去噪，块大小 $L'$ 在两端之间插值。要这么做，是为了同时具备自回归的似然质量和连续生成任意多个块的能力，以及扩散的块内并行。生成时先出一个块，块内多个 token 并行定下来，再把这个块的结果作为下一个块的条件。
依据：paper/src/iclr2025_conference.tex L379-L380；paper/src/iclr2025_conference.tex Introduction

### 块扩散的似然分解

把序列按长度 $L'$ 切成 $B = L/L'$ 个块后，似然按块分解成
$$\log p_\theta(\mathbf{x}) = \sum_{b=1}^{B} \log p_\theta(\mathbf{x}^b \mid \mathbf{x}^{<b}),$$
其中 $\mathbf{x}^b$ 是第 $b$ 个块，$\mathbf{x}^{<b}$ 是它之前的所有块，$p_\theta$ 是模型给出的概率。每一项 $p_\theta(\mathbf{x}^b \mid \mathbf{x}^{<b})$ 都用离散扩散来建模，条件里的 $\mathbf{x}^{<b}$ 当作已知的前文。块大小取 $L'=1$ 时每个块只有一个 token，退化成自回归模型；取 $L'=L$ 时只有一个块，退化成一次为全序列去噪的扩散模型。
依据：paper/src/iclr2025_conference.tex Block Diffusion Distributions and Model Architectures

把每一块的 NELBO 加起来，就得到块扩散的训练目标。NELBO 是负的 ELBO，也就是似然下界的相反数，优化它就是最大化似然下界。这里用的是掩码扩散：把一部分 token 换成掩码符号，模型学会把它们还原。对掩码扩散，这个目标化简成被掩位置上交叉熵的加权和，[交叉熵](wiki/cross-entropy)度量模型给真实 token 的概率有多低。扩散时间 $t$ 从 0 到 1，$\alpha_t$ 是到时刻 $t$ 还没被掩的概率，随 $t$ 单调下降，$\alpha_0=1$、$\alpha_1=0$，于是掩码率是 $1-\alpha_t$；$\mathbf{x}_t^b$ 是块 $b$ 在时刻 $t$ 被掩掉一部分 token 后的结果，$\mathbb{E}_q$ 表示对被掩位置的随机选择求平均。目标写成
$$\mathcal{L}_\text{BD}(\mathbf{x}; \theta) := \sum_{b=1}^{B} \mathbb{E}_{t \sim [0,1]} \mathbb{E}_{q} \left[ \frac{\alpha_t'}{1-\alpha_t} \log p_\theta(\mathbf{x}^b \mid \mathbf{x}_t^b, \mathbf{x}^{<b}) \right],$$
$\alpha_t'$ 是 $\alpha_t$ 对 $t$ 的导数，权重 $\alpha_t'/(1-\alpha_t)$ 随掩码率变化，使低掩码率（被掩 token 少、信号弱）的位置获得更大的权重。$L'=1$ 时这个目标在期望上等于自回归的负对数似然。
依据：paper/src/iclr2025_conference.tex Masked \algos{}；paper/src/iclr2025_conference.tex Block Diffusion NELBO；paper/src/iclr2025_conference.tex L470-L473；paper/src/iclr2025_conference.tex Recovering the NLL from the NELBO for Single Token Generation

::: 从块扩散 NELBO 到单 token 等价
取线性调度 $\alpha_t = 1-t$，则 $\alpha_t' = -1$，权重 $\alpha_t'/(1-\alpha_t) = -1/t$。块大小 $L'=1$ 时每个块只有一个 token，记 $\mathbf{m}$ 是掩码符号，$q(\mathbf{x}_t^b = \mathbf{m} \mid \mathbf{x}^b) = t$ 是该 token 被掩的概率，于是

1. 把期望展开成被掩与未掩两种情形：$\mathbb{E}_q[\log p_\theta] = t \log p_\theta(\mathbf{x}^b \mid \mathbf{x}_t^b=\mathbf{m}) + (1-t)\log p_\theta(\mathbf{x}^b \mid \mathbf{x}_t^b=\mathbf{x}^b)$。未掩时模型直接输出输入本身，这一项为 0。
依据：paper/src/iclr2025_conference.tex Recovering the NLL from the NELBO for Single Token Generation
2. 代入权重 $-1/t$：$- \frac{1}{t} \cdot t \log p_\theta(\mathbf{x}^b \mid \mathbf{x}_t^b=\mathbf{m}) = -\log p_\theta(\mathbf{x}^b \mid \mathbf{m}, \mathbf{x}^{<b})$。
依据：paper/src/iclr2025_conference.tex Recovering the NLL from the NELBO for Single Token Generation
3. 对所有 $B=L$ 个块求和，得到 $-\sum_{b=1}^{L} \log p_\theta(\mathbf{x}^b \mid \mathbf{m}, \mathbf{x}^{<b})$，正是自回归的负对数似然。
依据：paper/src/iclr2025_conference.tex Recovering the NLL from the NELBO for Single Token Generation
:::

### 块间因果、块内双向的注意力

块扩散的所有块由同一个网络参数化。这个网络是一个 Transformer（一种用注意力处理整条序列的网络结构），输入是 $L$ 个 token，它在位置 $\ell$ 处先算出三个向量：查询、键、值，用查询和各个位置的键做比较，按比较的结果加权取回对应的值，这套计算叫[标准注意力](wiki/standard-attention)。块扩散给注意力加一条掩码规则：同一个块内的 token 互相能看到，块 $b$ 里的 token 还能看到它之前所有块的 token，看不到它之后块里的 token。这条规则使块 $b$ 的输出只用到 $\mathbf{x}^{<b}$ 和块 $b$ 自身，块内保留双向信息。
依据：paper/src/iclr2025_conference.tex Model Architecture

这个网络同时输出两种东西：每个块被掩位置的干净 token 预测，以及这个块的键和值，后者供训练和生成复用。写全它的签名是 $\mathbf{x}_\text{logits}^b, \mathbf{K}^b, \mathbf{V}^b \gets \mathbf{x}_\theta^b(\mathbf{x}_t^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$，其中 $\mathbf{x}_\text{logits}^b$ 是块 $b$ 干净 token 的预测，$\mathbf{K}^b, \mathbf{V}^b$ 是这个块的键和值，$\mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1}$ 是前面各块的键和值。
依据：paper/src/iclr2025_conference.tex Model Architecture
::: 掩码的三部分与一个小例子
掩码 $\mathbf{M}_\text{full}$ 作用在拼接起来的噪声序列与干净序列 $\mathbf{x}_t \oplus \mathbf{x}$ 上，分成四块，其中左下角一块（干净序列看噪声序列）全为 0，其余三块分别是三种子掩码

1. $\mathbf{M}_\text{BD}$（块内自注意力）：噪声块 $\mathbf{x}_t^b$ 的 token 只看同一个块内的 token，位置 $i,j$ 同块时取 1。
依据：paper/src/iclr2025_conference.tex Specialized Attention Masks
2. $\mathbf{M}_\text{OBC}$（偏移块因果）：噪声块 $\mathbf{x}_t^b$ 的 token 看前面各块的干净 token，$j$ 所在块在 $i$ 之前时取 1。
依据：paper/src/iclr2025_conference.tex Specialized Attention Masks
3. $\mathbf{M}_\text{BC}$（块因果）：干净块 $\mathbf{x}^b$ 的 token 看同块及之前块的 token，$j$ 与 $i$ 同块或在之前时取 1。
依据：paper/src/iclr2025_conference.tex Specialized Attention Masks

取序列长度 $L=4$、块大小 $L'=2$，两个块 $b=1,2$，块 1 占位置 1、2，块 2 占位置 3、4。

1. 块 1 的噪声 token 位置 1、2 互相可见（$\mathbf{M}_\text{BD}$），看不到别的。
依据：自行构造
2. 块 2 的噪声 token 位置 3、4 互相可见；按 $\mathbf{M}_\text{OBC}$ 还能看到块 1 的干净 token 位置 1、2；看不到块 2 自己的干净 token。
依据：自行构造
3. 干净序列里块 1 的 token 位置 1、2 只看彼此，块 2 的干净 token 位置 3、4 看位置 1 到 4（$\mathbf{M}_\text{BC}$）。
依据：自行构造
:::

### 每 token 过两次模型的训练算法

块扩散的损失用通常的单序列输入一次前向算不完。原因是每个块的损失要在被掩的 $\mathbf{x}_t^b$ 上算，而作为条件的前面各块用的是干净 token $\mathbf{x}^{<b}$，同一段序列在两种形态下各要经过一遍网络，所以每个 token 至少过两次模型。做法是分两次前向：第一次把干净的整条序列 $\mathbf{x}$ 送进网络，只为得到各块的键和值，$(\emptyset, \mathbf{K}^{1:B}, \mathbf{V}^{1:B}) \gets \mathbf{x}_\theta(\mathbf{x})$，其中 $\emptyset$ 表示这一路用不到的输出；第二次对每个块 $b$ 用被掩的 $\mathbf{x}_t^b$ 和前面各块缓存的键和值，算出该块的预测，$\mathbf{x}_\text{logits}^b, \emptyset, \emptyset \gets \mathbf{x}_\theta^b(\mathbf{x}_t^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$。
依据：paper/src/iclr2025_conference.tex L433；paper/src/iclr2025_conference.tex Training

若按块循环，每个块各调一次模型算它的预测，加上第一次取键和值，共 $B+1$ 遍前向。向量化的写法把被掩序列与干净序列拼成一段 $\mathbf{x}_t \oplus \mathbf{x}$ 送进网络一次算出所有块的预测，这段拼接序列的注意力掩码使被掩 token 只看同块的其他被掩 token 和前面各块的干净 token。向量化的写法把两次前向合成一次，相对分两次前向的写法训练速度提升 20-25%。
依据：paper/src/iclr2025_conference.tex Vectorized Training；paper/src/iclr2025_conference.tex Efficiency of Training Algorithm

::: 训练伪代码
每个训练步里，先采样各块的时间 $t_b$ 和被掩块，再走两次前向算损失。
依据：paper/src/algs/short_algs.tex L49-L67

1. 采样 $t_1, \dots, t_B \sim \mathcal{U}[0,1]$，对每个块按 $t_b$ 掩出 $\mathbf{x}_{t_b}^b$。
依据：paper/src/algs/short_algs.tex L49-L67
2. 第一次前向取键和值：$\emptyset, \mathbf{K}^{1:B}, \mathbf{V}^{1:B} \gets \mathbf{x}_\theta(\mathbf{x})$。
依据：paper/src/algs/short_algs.tex L49-L67
3. 第二次前向算各块预测：$\mathbf{x}_\text{logits}^b, \emptyset, \emptyset \gets \mathbf{x}_\theta^b(\mathbf{x}_{t_b}^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$，拼成 $\mathbf{x}_\text{logits}$。
依据：paper/src/algs/short_algs.tex L49-L67
4. 用 $\mathbf{x}_\text{logits}$ 算块扩散损失，取一次梯度步。
依据：paper/src/algs/short_algs.tex L49-L67
:::

### 逐块生成、块内并行的采样

生成时一次出一个块。对块 $b$，用一个扩散采样过程 $\textsc{Sample}(\mathbf{x}_\theta^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$ 从条件分布 $p_\theta(\mathbf{x}_s^b \mid \mathbf{x}_t^b, \mathbf{x}^{<b})$ 里采样出这个块的 token，$s$ 是扩散过程里 $t$ 之前的一个时刻，条件用前面各块预先算好的键和值给出。块生成完之后，把它的键和值追加进缓存，$\mathbf{K} \gets \mathbf{K}^{1:b-1} \oplus \mathbf{K}^b$，下一个块直接用，不必重算前面各块，这和自回归生成里的缓存是同一件事。
依据：paper/src/iclr2025_conference.tex Sampling

生成时块内所有位置的 token 在同一批扩散步骤里并行定下来，不必一次一个，因此生成一条 $L$ 个 token 的序列花的模型调用次数可以小于 $L$；同时块与块之间是自回归的，序列有多长就可以不断生成新的块，不受训练时上下文的长度限制。掩码扩散的采样还有一个上界：一旦某个位置的 token 从掩码还原成原 token，就不再被重新掩上，所以每个块生成步数最多等于块长。
依据：paper/src/iclr2025_conference.tex Sampling；paper/src/iclr2025_conference.tex L814

::: 采样伪代码
对每个块 $b$ 调用一次扩散采样过程，得到一个块的 token，再把这个块过一遍网络取它的键和值、追加进缓存。
依据：paper/src/algs/short_algs.tex L69-L89

1. 从当前缓存 $\mathbf{K}, \mathbf{V}$ 取前面各块的键和值，采样出块 $b$：$\mathbf{x}^b \gets \textsc{Sample}(\mathbf{x}_\theta^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$。
依据：paper/src/algs/short_algs.tex L69-L89
2. 把块 $b$ 过一遍网络，取它自己的键和值：$\emptyset, \mathbf{K}^b, \mathbf{V}^b \gets \mathbf{x}_\theta^b(\mathbf{x}^b, \mathbf{K}^{1:b-1}, \mathbf{V}^{1:b-1})$。
依据：paper/src/algs/short_algs.tex L69-L89
3. 把块 $b$ 接到输出序列末尾，把它的键和值追加进缓存，供下一个块用。
依据：paper/src/algs/short_algs.tex L69-L89
:::

::: 生成长度的统计
生成长度统计：训练数据本身中位 717 个 token、最长 131K；一次为全序列去噪的扩散模型生成受限于训练上下文 1024，最长只到 1024；块扩散 $L'=16$ 能出到最长 9982 个 token，中位数 798。
依据：paper/src/iclr2025_conference.tex Sample Quality and Variable-Length Sequence Generation
:::

### 梯度方差与低方差噪声调度

把块大小取 1、本应与自回归等价，实测却差约两个困惑度点。困惑度（PPL）是模型对文本不确定程度的度量，越低越好，这里指测试集上的困惑度。差距的原因在训练时梯度的方差更大：训练时每个 token 的时间 $t$ 是随机抽的，抽到小掩码率时大部分 token 没被掩、重建过于容易，抽到大掩码率时最优解是数据分布的边缘分布（每个位置出现各 token 的总体频率）、也过于容易学，这两种极端掩码率给出的学习信号都弱，却让梯度估计的方差变大。更直接的原因是均匀采样下平均只有一半 token 处在被掩状态，梯度估计用到的 token 数只有自回归的一半，噪声相应更大。
依据：paper/src/iclr2025_conference.tex Case Study: Single Token Generation；paper/src/iclr2025_conference.tex Intuition: Avoid Extreme Mask Rates

NELBO 本身与噪声调度的选择无关，但训练时用蒙特卡洛估计（随机抽样若干位置来估计这个目标）的方式算它，估计的方差随调度变。为此把训练时的掩码率限制在一个区间里，$1-\alpha_t \sim \mathcal{U}[\beta, \omega]$，两端 $\beta, \omega$ 在 $[0,1]$ 内。这个区间随块大小变：块越小，最优区间越偏重掩码，所以在训练中每轮验证时按损失方差搜出 $\beta, \omega$。
依据：paper/src/iclr2025_conference.tex Diffusion Gap from High Variance Training；paper/src/iclr2025_conference.tex Clipped Schedules for Low-Variance Gradients；paper/src/iclr2025_conference.tex Data-Driven Clipped Schedules Across Block Sizes；paper/src/iclr2025_conference.tex Experiments

::: 单 token 与调度对比的数字
块大小取 1 时每个块只有一个 token，可以直接用全程掩码（每个 token 都以概率 1 被掩、不再随机抽时间）来训练。在 328M token 上训练后测损失方差：按 NELBO 训练是 1.52，全程掩码训练降到 0.11；困惑度从 $\le 25.56$ 降到与自回归一致的 22.88。
依据：paper/src/iclr2025_conference.tex Case Study: Single Token Generation
:::

::: 不同块大小下的调度对比
不同掩码率区间的困惑度与损失方差（PPL 与 Var. NELBO，都越低越好）：

| 块大小 | 调度 | PPL | Var. NELBO |
|---|---|---|---|
| 4 | $\mathcal{U}[.5,1]$ | 29.16 | 8.28 |
| 4 | $\mathcal{U}[0,1]$ | 29.23 | 23.65 |
| 16 | $\mathcal{U}[.3,.8]$ | 31.19 | 3.62 |
| 16 | $\mathcal{U}[0,1]$ | 31.33 | 7.39 |
| 128 | $\mathcal{U}[0,.5]$ | 31.72 | 1.03 |
| 128 | $\mathcal{U}[0,1]$ | 31.78 | 3.80 |
依据：paper/src/iclr2025_conference.tex Data-Driven Clipped Schedules Across Block Sizes
:::

### 一个四 token 的例子

取 token 编号 1、2、3、4，切成两块，块 1 是 (1, 2)，块 2 是 (3, 4)。线性调度指 $\alpha_t = 1-t$、掩码率等于 $t$，一个被掩位置的代价是 $-\frac{1}{t}\log p$，$t$ 越小越重。块 1 取 $t_1=0.5$、掩掉第一个位置，块 2 取 $t_2=1$、两个位置都被掩。
依据：自行构造

::: 手算这次块扩散损失
设块 1 被掩位置的预测概率 $p_1=0.5$，块 2 的 $p_3=0.25$、$p_4=0.8$。

1. 权重：块 1 是 $1/0.5=2$，块 2 是 $1/1=1$。
依据：自行构造
2. 各块贡献：块 1 为 $2 \times (-\log 0.5)=1.386$，块 2 为 $(-\log 0.25)+(-\log 0.8)=1.386+0.223=1.609$。
依据：自行构造
3. 两块相加，损失约为 $2.996$。
依据：自行构造
:::

块 1 的权重比块 2 大，因为它掩码率低、被掩 token 少，权重放大用来补偿信号弱。
依据：自行构造

### 适用条件与边界

块扩散的似然是真实似然的下界，这个下界随块大小 $L'$ 变化：$L'=1$ 时最紧，等于自回归的负对数似然，$L'$ 变大时越来越松。
依据：paper/src/iclr2025_conference.tex Tightness of the NELBO

代价方面，训练块扩散比训练普通扩散模型开销大，向量化的写法把这一差距限制在两倍以内，实践中还会先用一次标准的扩散损失预训练来缩小差距；生成时块之间仍是顺序产生，块小的时候速度和可控性会和自回归一样受块顺序的约束，最优块大小依任务而定，要更多可控性就取更大的块。
依据：paper/src/iclr2025_conference.tex Limitations

和最容易混淆的扩散语言模型比：块扩散使用的是离散噪声，能给出可计算的似然估计；而另一类做法在连续嵌入（把 token 映成实数的向量）上做扩散，需要连续数据那套算法，似然更差。
依据：paper/src/iclr2025_conference.tex Discussion and Prior Work
