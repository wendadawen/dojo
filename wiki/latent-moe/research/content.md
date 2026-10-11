# LatentMoE

- 描述：LatentMoE 是一种混合专家（MoE）架构：先把序列里每个位置的向量压到一个低维隐空间，再交给路由专家，用省下的通信量和权重加载量把专家总数与每个位置的激活数放大同样的倍数。
- 摘要：把路由专家放进低维隐空间计算，按压缩倍率 $\alpha=d/\ell$ 同倍放大专家总数与 top-$k$，专家组合空间指数扩大，通信量和权重加载量与标准 MoE 持平，同等推理成本下精度更高。
- 主题：模型结构, 训练与优化
- 标签：MoE

## 资料

- Venmugil Elango, Nidhi Bhatia, Roger Waleffe 等. LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts. arXiv:2601.18089v1, 2026. [https://arxiv.org/abs/2601.18089](https://arxiv.org/abs/2601.18089)

## 短页

### 专家改在隐空间里算

混合专家（[MoE](wiki/moe)）层把前馈层换成多个结构相同的小网络，前馈层是每个位置各算一次的子层；每个小网络叫一个专家，小网络的个数叫专家总数。每个 token（文本切分后的基本单位）由一步路由按分数选其中几个专家来算，被选中的个数叫激活数。token 在这一层里的向量长度叫隐藏维度 $d$，每个专家的权重按这个长度摆放。LatentMoE 把 token 的向量先压到更短的隐维度 $\ell$，让路由专家在这个低维空间里算，再把结果投回 $d$；省下的通信量（把 token 送到各设备所需的数据量）和权重加载量（从显存读专家权重所需的数据量，显存是 GPU 上的内存）用来放大专家总数与激活数。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/introduction.tex Introduction

### 显存与通信两处瓶颈

低延迟、小批量的推理（用训练好的模型算输出）里一次只处理很少的 token，每个被选中的专家要先等它的权重从显存读进来才能开始计算，瓶颈是权重加载量而不是算力。吞吐场景里专家分散在多张卡上，token 要把自己的向量发给被选中的专家、再把结果收回来，这段通信量同样是瓶颈。权重加载量随隐藏维度 $d$ 和专家的中间维 $m$（每个专家内部权重的宽度）一起涨，通信量随激活专家数与 $d$ 一起涨。中间维乘激活专家数决定每个 token 经过的非线性变换有多宽，不能减小；隐藏维度压到任务本身需要的信息量以下又会丢信息。剩下能动的是隐藏维度。
依据：paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/design_choices.tex Model Quality

### 降维，再按倍率放大专家

第一步，用一个可学的降维矩阵 $W_\downarrow\in\mathbb{R}^{\ell\times d}$ 把 token 从 $d$ 维压到 $\ell$ 维。第二步，把专家权重改成隐空间里的形状，让它们在低维向量上算，token 的分发也在隐空间里发生。第三步，用回投矩阵 $W_\uparrow\in\mathbb{R}^{d\times\ell}$ 把专家输出送回 $d$ 维。专家总数从 $N$ 放到 $N'=\alpha N$，每个 token 的激活数从 $K$ 放到 $K'=\alpha K$；放大抵消了压缩，通信量与权重加载量回到标准 MoE 的水平，而专家之间能组合出的种类相应扩大。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/introduction.tex Introduction

## 长页

大语言模型的主干是一叠结构相同的 Transformer 层，每一层含注意力和前馈两个子层。注意力（[标准注意力](wiki/standard-attention)）在位置之间传信息，前馈层逐个位置各算一次。位置以 token 为单位，token 是文本切分后的基本单位。一层里每个位置的向量长度叫隐藏维度，记作 $d$；前馈层先把这个 $d$ 维向量投到一个中间维 $m$，再投回来，每个 token 用到的参数都摆在这两个维度撑起的矩阵里。混合专家（[MoE](wiki/moe)）层把前馈层换成多个结构相同的小网络，每个叫一个专家，每个 token 由一步路由按分数只选其中几个来算，被选中的叫激活专家。这样参数总量可以远大于每个 token 实际用到的计算量。
依据：paper/src-v1/sections/introduction.tex Introduction；paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck

低延迟的小批量推理里，每个激活专家要等权重从显存读进来才算得动；吞吐场景里，token 要在多张卡之间来回发。两种吃紧都随 $d$ 涨，而直接减小 $d$ 到任务需要的信息量以下会丢信息。LatentMoE 把每个 token 先乘一个降维矩阵压成 $\ell$ 维的隐空间向量，专家在这个短向量上算，算完再乘一个回投矩阵回到 $d$ 维。付出的是两个矩阵的参数，换来的是发向量和读权重的量都按 $\alpha=d/\ell$ 缩小；这份节省不用来降低成本，而是按同一倍率把专家总数与每个 token 的激活专家数一起放大，专家之间的组合方式随之扩大，推理成本却几乎不变。
依据：paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/design_choices.tex Model Quality

### 能省的只有隐藏维

一个 MoE 层里，每个 token 的路由给每个专家算一个分数，只保留最高的 $K$ 个作为门值（门值是这些保留下来的分数），$K$ 是每个 token 的激活专家数，$N$ 是专家总数。把专家铺到多张卡上时，每张卡放 $N/\mathrm{EP}$ 个，$\mathrm{EP}$ 是放专家的卡数。记 $T_\mathrm{exp}$ 为一个专家在一次前向（一次完整的前向计算）里平均收到的 token 数，也就是路由之后的有效批量；记 $T_\mathrm{total}$ 为路由前所有卡上的 token 总数。$d$ 是隐藏维度，$m$ 是专家的中间维。
依据：paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/design_choices.tex LatentMoE Core Design Principles

一个专家要把它那份权重从显存读进来，一份的量是 $d\cdot m$；一个 token 激活几个专家，就要读几份，所以一张卡的权重加载量正比于 $K\cdot d\cdot m$。通信量是一张卡一层要把 token 的向量发出去的量，正比于 $(N/\mathrm{EP})\cdot T_\mathrm{exp}\cdot d$。两项都随 $d$ 涨。
依据：paper/src-v1/sections/design_choices.tex Memory Bandwidth Bottleneck；paper/src-v1/sections/design_choices.tex Communication Bottleneck

::: 通信时间与计算时间之比的推导
取一组示例配置：$N=128$，$K=8$，$d=4096$，$m=1536$，用 $\mathrm{EP}=64$ 张 GB200（一种用 NVLink 卡间互联的 GPU），每张卡放 $N/\mathrm{EP}=2$ 个专家；单卡以 FP4 精度（4 比特浮点）算，峰值算力 $F=10$ PFLOPs（1 PFLOPs 即每秒 $10^{15}$ 次浮点运算），卡间单向带宽 $\mathrm{BW}_\mathrm{NVL}=900$ GB/s。
依据：paper/src-v1/sections/design_choices.tex LatentMoE Core Design Principles
1. 一张卡一层要发的量 $M_\mathrm{comm}=5\cdot T_\mathrm{exp}\cdot d$；系数 5 里 0.5 字节是 FP4 派发、2 字节是 BF16（16 比特浮点）聚合，各乘上每张卡的 2 个专家。
依据：paper/src-v1/sections/design_choices.tex Communication Bottleneck
2. 两个专家的计算量 $C_\mathrm{comp}=4\cdot T_\mathrm{exp}\cdot d\cdot m$。
依据：paper/src-v1/sections/design_choices.tex Communication Bottleneck
3. 通信时间与计算时间之比 $t_\mathrm{comm}/t_\mathrm{comp}=(5F)/(4m\,\mathrm{BW}_\mathrm{NVL})$，把上面的数代进去约为 9，说明这一层的执行时间大量花在通信上。
依据：paper/src-v1/sections/design_choices.tex Communication Bottleneck
:::

能动的是 $m$、$K$、$d$ 三个量。$m$ 只进权重加载量、不进通信量；$K$ 与 $d$ 共同决定通信量，权重加载量里也有 $K$。两条约束把它们各自限住。一条是有效非线性预算 $U_\mathrm{eff}\propto K\cdot m$：每个 token 能表达的非线性程度随激活专家数与中间维的乘积涨，减 $K$ 或减 $m$ 都会降低它。另一条是任务自身的特征秩 $r_\mathrm{eff}$：任务要保留的信息有一个最低自由度，$d$ 压到它以下就会丢信息。$m$ 与 $K$ 都受前一条约束，一减就要靠另一个补；只有 $d$ 不受 $U_\mathrm{eff}$ 限，能一直压到 $r_\mathrm{eff}$ 为止。$d$ 一压，权重加载量与通信量同时降，是两处瓶颈里最值得压的方向。
依据：paper/src-v1/sections/design_choices.tex Model Quality

把省下的成本投回专家数与激活数：$N$ 与 $K$ 同时乘一个倍率 $\alpha$，专家组合的种类数从 $\binom{N}{K}$ 涨到 $\binom{\alpha N}{\alpha K}$，且 $\binom{\alpha N}{\alpha K}\ge\binom{N}{K}^{\alpha}$；而通信量正比于 $(N/\mathrm{EP})\cdot T_\mathrm{exp}\cdot d$，把 $T_\mathrm{exp}\cdot N=T_\mathrm{total}\cdot K$ 代入，它正比于 $(T_\mathrm{total}\cdot K\cdot d)/\mathrm{EP}$，不随 $N$ 涨；权重加载量也不随 $N$ 涨。所以 $K$ 涨 $\alpha$ 倍、$d$ 同时压 $\alpha$ 倍，两项成本相互抵消。
依据：paper/src-v1/sections/design_choices.tex Communication Bottleneck；paper/src-v1/sections/design_choices.tex Model Quality

### 把激活压进隐空间

#### 同倍放大专家数与 top-k

压缩只作用在路由专家这条路上。算路由分数的机制和一组始终激活的共享专家（每个 token 都固定分给它们、不参与路由）仍在原来的 $d$ 维空间里算，它们不是瓶颈所在；被压到隐空间、也被放大的是路由专家这一路，放大按同一个倍率把专家总数与每个 token 激活的专家数（top-$k$ 里的 $K$）一起放大。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture

::: 三个矩阵的形状与一层的完整公式
1. 降维矩阵 $W_\downarrow\in\mathbb{R}^{\ell\times d}$，把 $x\in\mathbb{R}^d$ 压成 $\mathbb{R}^{\ell}$ 的向量。
2. 每个专家的权重都在隐空间里，形状由 $\ell$ 与 $m$ 撑起：把输入从 $\ell$ 升到 $m$ 的那一层是 $m\times\ell$，把结果降回 $\ell$ 的那一层是 $\ell\times m$。按后文成本里那个量级，记作一份 $\ell\cdot m$ 的权重。
3. 回投矩阵 $W_\uparrow\in\mathbb{R}^{d\times\ell}$，把加权求和后的 $\ell$ 维向量送回 $d$ 维。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture
$$\ell\text{-MoE}(x)=W_\uparrow\Big(\sum_{i\in\mathcal{T}}p'_i\,E_i(W_\downarrow x;\ell)\Big)+\sum_{j=N'+1}^{N'+S}E_j(x;d)$$
其中 $N'=\alpha N$ 是放大后的路由专家总数；路由权重 $p'=\mathrm{Softmax}(W'_r x)\in\mathbb{R}^{N'}$（Softmax 把打分向量归一成各专家的权重）由可学矩阵 $W'_r\in\mathbb{R}^{N'\times d}$ 在原始 $d$ 维 token 上算出，$p'_i$ 是它给第 $i$ 个专家的那一维；$\mathcal{T}$ 是按 $p'$ 选出的前若干个专家的下标。上式按只放大专家数的配置写，$\mathcal{T}$ 取前 $K$ 个；默认配置把 $K$ 换成 $K'=\alpha K$。$S$ 是共享专家数，$E_j(x;d)$ 是它们。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture
:::

放大的方式有两种。一种只把专家总数放大 $\alpha$ 倍、每个 token 的激活数不变，叫 $\ell\text{-MoE}_\mathrm{eff}$：通信量与权重加载量都降到原来的 $1/\alpha$，精度与标准 MoE 持平，等于用更少的成本换同样的精度。另一种把专家总数与激活数一起放大 $\alpha$ 倍，叫 $\ell\text{-MoE}_\mathrm{acc}$：专家组合多样性和每个 token 的非线性预算一起变大，精度更高，而通信量与权重加载量回到标准 MoE 的水平。默认取后一种。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/latentmoe.tex L26-L53

::: 一个能手算的例子
设定：隐藏维度 $d=8$，隐维度 $\ell=2$，压缩倍率 $\alpha=4$，中间维 $m=4$，专家总数 $N=6$，每个 token 激活 $K=1$ 个，不放共享专家。
依据：自行构造
1. 专家参数总量：标准 MoE 每个专家 $d\cdot m=8\times4=32$，共 $6\times32=192$；LatentMoE 专家数放到 $N'=\alpha N=24$，每个专家 $\ell\cdot m=2\times4=8$，共 $24\times8=192$。两边相同。
依据：自行构造
2. 权重加载量：低延迟时只读被选中专家的权重。标准读 $1$ 个专家共 $32$；$\ell\text{-MoE}_\mathrm{eff}$ 读 $1$ 个共 $8$，是标准的四分之一；$\ell\text{-MoE}_\mathrm{acc}$ 读 $K'=\alpha K=4$ 个共 $4\times8=32$，与标准相同。
依据：自行构造
3. 通信量：每个 token 要把送进被选中专家的向量发出去。标准发 $1$ 份 $d=8$ 维共 $8$；$\ell\text{-MoE}_\mathrm{eff}$ 发 $1$ 份 $\ell=2$ 维共 $2$；$\ell\text{-MoE}_\mathrm{acc}$ 发 $4$ 份 $\ell=2$ 维共 $8$，与标准相同。
依据：自行构造
4. 专家组合数：标准是 $\binom{6}{1}=6$；$\ell\text{-MoE}_\mathrm{acc}$ 是 $\binom{24}{4}=10626$，组合空间大了三个数量级。
依据：自行构造
:::

### 压缩比能压到多小

压多少才不丢精度，靠实验定。在同一配置上只改压缩倍率（专家总数随之按 $\alpha$ 放大）、其余超参数（不靠训练更新、需人为设定的量）不动，各取几个值比较验证损失（在留出数据上算的损失）：$\alpha\le4$ 时质量保持，于是取 $\alpha=4$。
依据：paper/src-v1/sections/results.tex LatentMoE Ablations

把 $d$ 压 4 倍而不放大专家数，验证损失变差；补回 $N'=\alpha N$ 之后，压缩带来的损失大幅减轻，超参数也不必为压缩后的模型大幅调整。
依据：paper/src-v1/sections/results.tex LatentMoE Ablations；paper/src-v1/sections/results.tex L46-L54

::: 消融的条件与结果
- 消融（改掉某一部分再对比的实验）用 16B（B 即十亿）总参数、2B 激活参数的一档，它的 $d=2048$、$N=64$、$K=6$、$S=2$、$m=1408$。
- 扫压缩比时用 $\ell\text{-MoE}_\mathrm{eff}$ 配置，专家总数按 $\alpha$ 放大，其余超参数不变；结果是 $\alpha\le4$ 时质量保持。
- $\alpha=4$ 在 95B 总、8B 激活那一档（$d=4096$、$N=128$、$K=6$、$S=2$、$m=2688$）同样有效。
依据：paper/src-v1/sections/results.tex LatentMoE Ablations；paper/src-v1/sections/results.tex L7-L31
:::

### 换来的精度与速度

$\ell\text{-MoE}_\mathrm{eff}$ 用远少的算力达到与标准 MoE 相当或更好的精度，$\ell\text{-MoE}_\mathrm{acc}$ 在参数与算力相当的情况下精度更高。
依据：paper/src-v1/sections/results.tex LatentMoE Scaling Studies；paper/src-v1/sections/results.tex L82-L98

投影本身带来的计算量占比很小：加了降维与回投之后，开销最多比原生模型多约 9%。
依据：paper/src-v1/sections/results.tex Projected Serving Impact at Trillion-Parameter Scale

::: 万亿规模的服务投影
- 用一个性能模拟器，在大量运行点上估吞吐与延迟。
- 以一个万亿参数级的模型为原型配上 LatentMoE。折算成等效参数倍率 $\lambda\approx1.35\times$，即相当于标准 MoE 的 1.35T 参数，要多出约 350B 参数。
- 在投影出的前沿上，把标准 MoE 放大到同样的精度，会比 LatentMoE 版慢约 $1.24\times$ 到 $3.46\times$。
依据：paper/src-v1/sections/results.tex Projected Serving Impact at Trillion-Parameter Scale
:::

### 适用条件与边界

收益只覆盖路由专家的权重加载与它们之间的通信。另外，压缩倍率 4 是实验扫出来的取值。
依据：paper/src-v1/sections/latentmoe.tex LatentMoE Architecture；paper/src-v1/sections/results.tex LatentMoE Ablations

和最容易混的做法 MoLAE 的区别：MoLAE 是对训练完的专家权重做低秩近似、只在隐空间里近似一部分专家的权重，不为 token 分发省通信；LatentMoE 把降维与放大专家数绑在一起，分发通信一起省。
依据：paper/src-v1/sections/discussion.tex Related Work
