# DeepSeekMoE

- 描述：DeepSeekMoE 是一种混合专家（MoE）架构，把每个专家切成更小的多个、每个 token 激活更多个，再隔离出始终激活的共享专家，让路由专家更专精。
- 摘要：把专家切细，扩大激活专家可组合的方式；再隔离出共享专家承载公共知识，让路由专家更少冗余、更专精。
- 主题：模型结构
- 标签：MoE

## 资料

- Damai Dai, Chengqi Deng, Chenggang Zhao 等. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv:2401.06066v1, 2024. [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)
- DeepSeek-AI. DeepSeek-MoE: official implementation. 2024. [https://github.com/deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)

## 短页

### 每个 token 激活更多的小专家

大语言模型的主干是一叠结构相同的层，每层把序列里每个位置（token，文本切分后的基本单位）的向量重算一遍。混合专家（MoE）层把层里对每个位置各算一次的那部分（前馈层）换成多个结构相同的小网络，每个叫一个专家；每个专家为这个位置打一个分数，路由按分数只选其中几个来算。从专家里再隔出一部分，让它们每个 token 都参与，这样的专家叫共享专家，其余的叫路由专家，路由只从路由专家里挑。DeepSeekMoE 把每个专家的中间维切小、专家总数变多、每个 token 激活更多个。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Shared Expert Isolation；paper/src-v1/main.tex Preliminaries: Mixture-of-Experts for Transformers

### 专家该更专精

常规 MoE 的专家数少，一个专家接到的 token 可能覆盖多类知识，把各类知识都装进自己的参数、难以同时用上，这叫知识混杂；不同专家又可能都需要同一份公共知识，于是各自重复学一遍，参数冗余。两者都让专家不够专精。
依据：paper/src-v1/main.tex Introduction

### 切细与共享专家

切细：把每个专家的中间维降到原来的 $1/m$（$m$ 是切分倍数），专家总数从 $N$ 变成 $mN$；为维持每个 token 的计算量，激活的专家数从 $K$ 增到 $mK$。共享：从专家里隔离 $K_s$ 个作共享专家，它们不放路由，每个 token 都固定分给它们；为维持计算量，路由激活的专家数减 $K_s$ 个。
依据：paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Shared Expert Isolation

## 长页

大语言模型的主干是一叠结构相同的 Transformer 层（一种把序列里每个位置的向量逐层重算的网络结构），层数常到几十。每层里，[注意力](wiki/standard-attention)先在位置之间传信息，算完得到一组向量，记作隐藏状态 $\mathbf{u}_{1:T}^{l}$：$T$ 是序列长度，$l$ 是层号，每个位置一个向量，位置以 token 为单位，token 是文本切分后的基本单位。接着前馈层（FFN）逐个位置各算各的，内部先把每个位置的向量升到一个更宽的中间维、算完再降回原维；层算出的输出以[残差连接](wiki/residual-connection)加回输入，即把这一层的输入直接加到输出上。一些 MoE 语言模型按固定间隔把一部分前馈层换成混合专家（MoE）层：前馈层变成多个结构相同的小网络，每个叫一个专家，每个 token 由一步路由按一个分数只选其中几个来算，分数是各专家给这个位置打的分，这样参数总量可以很大，而每个 token 实际用到的计算量维持较低。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Preliminaries: Mixture-of-Experts for Transformers；paper/src-v1/main.tex Hyper-Parameters

专家数少时，一个专家接到的 token 可能覆盖多类知识，它要把这些知识装进同一组参数、难以同时用上，这叫知识混杂；不同专家又可能都需要同一份公共知识，于是各自重复学一遍，参数冗余。DeepSeekMoE 针对这两点各做一处改动：把每个专家的中间维切小、让专家总数变多，每个 token 激活更多个，知识就能分到更多专家中分别学习；再把公共知识收进共享专家，其余专家不必重复。
依据：paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex Shared Expert Isolation

### 把专家切细

#### 切分与激活公式

第 $l$ 层里，注意力算完之后进入前馈层（这里换成 MoE 层）、位置 $t$ 的向量记作 $\mathbf{u}_t^l$，第 $i$ 个专家记作 $\mathrm{FFN}_i$，$N$ 是专家总数。每个专家给这个 token 算一个分数：token 与专家的一个中心向量 $\mathbf{e}_i^l$ 做内积，经 SoftMax 归一（把一组数变成总和为 1 的正数），记作 $s_{i,t}$；门值 $g_{i,t}$ 只保留这些分数里最高的 $K$ 个，其余置零。层的输出是各专家输出的加权和，再加回输入。
依据：paper/src-v1/main.tex Preliminaries: Mixture-of-Experts for Transformers

$$ \mathbf{h}_t^l = \sum_{i=1}^{N} g_{i,t}\, \mathrm{FFN}_i\!\left(\mathbf{u}_t^l\right) + \mathbf{u}_t^l $$
$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}\big(\{s_{j,t} \mid 1 \le j \le N\},\, K\big) \\ 0, & \text{否则} \end{cases} $$
$$ s_{i,t} = \mathrm{Softmax}_i\!\left( {\mathbf{u}_t^l}^\top \mathbf{e}_i^l \right) $$

切细把每个专家的中间维降到原来的 $1/m$，专家总数从 $N$ 变成 $mN$；专家变小，为维持每个 token 的计算量，激活数从 $K$ 增到 $mK$。$m$ 是切分倍数，模型里实际取值 2 或 4。切细后层的输出改成对所有 $mN$ 个专家求和，门值只在其中取 top-$mK$：
依据：paper/src-v1/main.tex Fine-Grained Expert Segmentation；paper/src-v1/main.tex L595-L599

$$ \mathbf{h}_t^l = \sum_{i=1}^{mN} g_{i,t}\, \mathrm{FFN}_i\!\left(\mathbf{u}_t^l\right) + \mathbf{u}_t^l $$
$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}\big(\{s_{j,t} \mid 1 \le j \le mN\},\, mK\big) \\ 0, & \text{否则} \end{cases} $$

专家参数总量不变：专家数乘 $m$，每个专家的中间维除以 $m$，两边相抵。
依据：paper/src-v1/main.tex Fine-Grained Expert Segmentation

#### 组合灵活性与粒度收益

切细还改变了激活专家能组合出的种类数。门值只保留 top-$mK$，可选的专家从 $N$ 个变成 $mN$ 个，激活数也从 $K$ 变多，组合的空间随之放大。
依据：paper/src-v1/main.tex Fine-Grained Expert Segmentation

::: 组合数与粒度
1. 取 $N=16$、每个 token 激活 $2$ 个，组合数是 $\binom{16}{2}=120$。
2. 把每个专家切成 $4$ 份：专家共 $64$ 个、每个 token 激活 $8$ 个，组合数是 $\binom{64}{8}=4,426,165,368$。
依据：paper/src-v1/main.tex Fine-Grained Expert Segmentation

再取一组小数字：$N=4$、每个 token 激活 $1$ 个，组合数是 $\binom{4}{1}=4$；把每个专家切成 $3$ 份，专家共 $12$ 个、每个 token 激活 $3$ 个，组合数是 $\binom{12}{3}=220$。
依据：自行构造

粒度消融（消融指固定其他变量、只改一个变量作对比）：在同样总参数与激活参数下，把每个专家切成 $2$ 份或 $4$ 份（$32$ 个专家即 $1$ 个共享加 $31$ 个路由，或 $64$ 个即 $1$ 加 $63$），评测成绩（把各基准上的表现折算成一个数，越大越好）随切得越细持续提升。
依据：paper/src-v1/main.tex L585-L599
:::

### 隔离共享专家

切细之外，再隔出一部分专家作共享专家。设共享专家数为 $K_s$，它们不参与路由，每个 token 都固定分给全部 $K_s$ 个。为维持计算量，路由部分的激活数从 $mK$ 减到 $mK-K_s$。完整的一层写成共享部分与路由部分两段：
依据：paper/src-v1/main.tex Shared Expert Isolation

$$ \mathbf{h}_t^l = \sum_{i=1}^{K_s} \mathrm{FFN}_i\!\left(\mathbf{u}_t^l\right) + \sum_{i=K_s+1}^{mN} g_{i,t}\, \mathrm{FFN}_i\!\left(\mathbf{u}_t^l\right) + \mathbf{u}_t^l $$
$$ g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{Topk}\big(\{s_{j,t} \mid K_s+1 \le j \le mN\},\, mK-K_s\big) \\ 0, & \text{否则} \end{cases} $$

共享部分没有门值，恒为 1；路由部分的门值只在下标 $K_s+1$ 到 $mN$ 的专家里取 top-$(mK-K_s)$。这样一个 MoE 层共有 $K_s$ 个共享专家、$mN-K_s$ 个路由专家。
依据：paper/src-v1/main.tex Shared Expert Isolation

::: 各专家的个数
取 $N=8$、$m=2$、$K=2$、$K_s=2$：
1. 专家总数 $mN=16$，其中共享 $K_s=2$ 个、路由 $mN-K_s=14$ 个。
2. 若不设共享专家，激活数应为 $mK=4$；设了 2 个共享后，路由部分激活 $mK-K_s=2$ 个。
3. 每个 token 实际用到 2 个共享加 2 个路由，共 4 个，与不设共享时相同。
依据：自行构造
:::

### 负载均衡

训练时会把一个要压低的标量加到总目标里，靠它把模型的行为调向期望方向，这样的标量叫损失。均衡损失就是这一类用来约束路由的损失。自动学出来的路由有两个隐患：一是路由坍缩的风险，模型总选少数几个专家，其余得不到充分训练；二是专家分到多台设备时，负载不均会加剧计算瓶颈。为缓解，训练时加两个损失：专家级均衡损失让各专家被选中的次数与平均分数都接近均匀，设备级均衡损失只要求各设备上的总计算量接近，不强制专家级严格均匀。
依据：paper/src-v1/main.tex Load Balance Consideration

::: 两个均衡损失的公式
$$ \mathcal{L}_{\mathrm{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i, \qquad f_i = \frac{N'}{K'T}\sum_{t=1}^{T} \mathbb{1}(\text{token } t \text{ 选中专家 } i), \qquad P_i = \frac{1}{T}\sum_{t=1}^{T} s_{i,t} $$
$$ \mathcal{L}_{\mathrm{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i', \qquad f_i' = \frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} f_j, \qquad P_i' = \sum_{j\in\mathcal{E}_i} P_j $$

其中 $f_i$ 是专家 $i$ 被选中的比例、$P_i$ 是它的平均分数（亲和度），$\mathbb{1}(\cdot)$ 是指示函数，$N'=mN-K_s$、$K'=mK-K_s$，$D$ 是设备数，路由专家按设备分成 $D$ 组、第 $i$ 组记作 $\mathcal{E}_i$，$\alpha_1$ 是专家级损失的系数、$\alpha_2$ 是设备级损失的系数。专家级系数取得小：2B 那一档的验证实验取 0.01，16B 那一档（$B$ 是十亿参数的记法）取 0.001；设备级系数取得比专家级大。专家级取小是为了防止路由坍缩又不损害模型表现，设备级取大是为了各设备计算均衡。
依据：paper/src-v1/main.tex Load Balance Consideration；paper/src-v1/main.tex Expert-Level Balance Loss.；paper/src-v1/main.tex Device-Level Balance Loss.；paper/src-v1/main.tex Training Settings.
:::

### 专精到什么程度

路由专家之间冗余低、每个更不可替代；共享专家装的则是路由专家没有的基础知识，去掉它、另激活一个路由专家，验证损失就升。激活的路由专家数减到 4 个时，验证损失仍与常规 MoE 相当。
依据：paper/src-v1/main.tex L623-L634；paper/src-v1/main.tex L656-L665

::: 冗余与激活数的数字
1. 去掉共享专家、改多激活一个路由专家：验证损失从 1.808 升到 2.414。
2. 激活路由专家数从 7 降到 4，验证损失已与常规 MoE（同样的模型、按 top-2 路由）相当。
3. 只激活 3 个路由专家、总专家参数与常规 MoE 相同、激活的专家参数只有它一半的版本，整体表现仍优于常规 MoE。
依据：paper/src-v1/main.tex L623-L634；paper/src-v1/main.tex L656-L665
:::

### 适用条件与边界

粒度不是越细越好：专家太小有拖低计算效率的风险，16B 那一档没有再切更细，把每个专家定为标准前馈层的 0.25 倍；更大规模上再切更细仍然可行。
依据：paper/src-v1/main.tex Model Settings.；paper/src-v1/main.tex L689-L690

共享专家数与激活路由专家数之比取 1:3（如 2 个共享配 6 个激活路由专家，也就是前面那档 16B 的配法）。在 64 个专家的粒度下试过隔离 1、2、4 个共享专家，验证损失分别是 1.808、1.806、1.811，差别不大，按 1:3 配时最低。
依据：paper/src-v1/main.tex L601-L605；paper/src-v1/main.tex L687-L688

代价落在注意力一侧：16B 那一档的注意力参数只有约 0.5B，多选类任务的表现偏弱。
依据：paper/src-v1/main.tex Internal Comparison with DeepSeek 7B
