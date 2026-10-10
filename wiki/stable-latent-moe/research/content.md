# Stable LatentMoE

- 描述：Stable LatentMoE 是一种混合专家（MoE）层，它把路由专家的计算从模型全宽移到更窄的隐空间上进行，再用三处改动稳定训练：归一化、有界激活和分位数负载均衡。
- 摘要：把被挑中的路由专家从全宽 $d$ 移到宽度为 $\ell$ 的隐空间，由此得到 896 个路由专家、每 token 激活 16 个、稀疏度 56；三处稳定化分别抑制路由分支的激活爆炸与专家负载失衡。
- 主题：模型结构, 训练与优化
- 标签：MoE

## 资料

- Kimi Team. Kimi K3: Open Frontier Intelligence. arXiv:2607.24653v2, 2026. [https://arxiv.org/abs/2607.24653](https://arxiv.org/abs/2607.24653)

## 短页

### 隐空间里算专家，三处加稳定

前馈层是主干网络里逐个位置各算一次的那部分，通常写成两层线性投影夹一个非线性，有的写法把非线性换成两条投影逐元素相乘。混合专家（MoE）把这一层的计算换成一排并行的专家子网络：每个 token（文本切分后的基本单位）由一步路由按算出的分数，只挑其中少数几个参与计算，被挑中的叫路由专家，其余不参与计算。还有一类共享专家，不参与挑选，每个 token 都固定经过。
依据：paper/k3-src/2-model-architecture.tex L211-L215；paper/k3-src/2-model-architecture.tex L223-L224

Stable LatentMoE 是一种 MoE 层。它把路由专家投到一个比模型全宽 $d$ 更窄的隐空间 $\ell$ 里计算，隐空间指一组比全宽更低维的中间向量表示：路由分支先用一个降维矩阵把 token 表示投进 $\ell$ 维，交给选中的路由专家，再把结果聚合并投回 $d$ 维；共享专家仍走全宽。这样在扩大路由专家的规模时，不必让每个专家的输入输出都保持全宽，路由分支的流量随之下降。
依据：paper/k3-src/2-model-architecture.tex L214-L215；paper/k3-src/2-model-architecture.tex L224

### 近千专家下的两个失败模式

常规 MoE 里，每个被选中的路由专家都接收全宽的 token 表示，所以专家池和每个 token 激活的专家数一扩大，路由通信和专家权重的流量就跟着这个倍数一起增大。稀疏度是路由专家总数除以每 token 激活的专家数，衡量专家池里被激活的那一小部分有多小。
依据：paper/k3-src/2-model-architecture.tex L214-L215

专家池扩到 896 个、每 token 激活 16 个，稀疏度就是 56。一是路由分支把降维矩阵、路由专家自身的前馈网络、上投影矩阵依次相乘、接成一条链，链上连着近四个矩阵乘法；这条链的条件数差（条件数衡量一串运算对输入扰动的放大程度，越大越容易让数值失控），在 2.8 万亿参数的规模上会让路由分支内部的激活值（这一支算出的中间数值）爆炸。二是要让近一千个专家各拿到大致相等的 token，靠的是一种无辅助损失路由——它不给辅助损失（加在训练目标上、用来使各专家负载趋于均衡的一项），而是直接给每个专家调一个偏置，这个偏置加在该专家被挑选时用的分数上；专家池到这个量级，这套更新超出了它仍然表现良好的区间，负载会失衡。
依据：paper/k3-src/2-model-architecture.tex L215-L220；paper/k3-src/2-model-architecture.tex L291-L292

### 投到隐空间，三处各加稳定

1. 共享专家直接处理 $d$ 维输入，输出也是 $d$ 维。
2. 路由分支用一个降维矩阵把输入投进 $\ell$ 维隐空间，分发给选中的路由专家，把它们的输出按路由权重聚合成 $\mathbf{u}$。
3. 对 $\mathbf{u}$ 做一次[RMSNorm](wiki/rmsnorm) 归一化，再经上投影矩阵投回 $d$ 维。
4. 把共享专家的输出与上投影的结果相加，作为这一层的输出。

三处改动分别修上面两个问题的不同侧面：归一化把 $\mathbf{u}$ 的尺度变化切断在进入上投影之前；给前馈层里相乘的两个因子各加上界，抑制激活值爆炸；用路由分数的分位数定专家偏置，使各专家负载趋于均衡。
依据：paper/k3-src/2-model-architecture.tex L223-L240；paper/k3-src/2-model-architecture.tex L253-L274；paper/k3-src/2-model-architecture.tex L289-L338

## 长页

大语言模型的主干是一叠 Transformer 层（Transformer 是这类模型的主干结构，一叠同构的层，每层先让各位置之间交换信息、再逐个位置做变换）。层里有一部分叫前馈层，它对每个位置各算一次，通常写成两层线性投影夹一个非线性，有的写法把非线性换成两条投影逐元素相乘。混合专家（MoE）把前馈层换成一排并行的专家子网络：每个 token（文本切分后的基本单位）由一步路由按分数只挑其中几个参与计算，被挑中的叫路由专家。[DeepSeekMoE](wiki/deepseek-moe) 那一类组织还把一部分专家设为共享专家，它们不参与挑选，每个 token 都固定经过；这种挑选叫 top-$k$，即按分数取前 $k$ 个。稀疏度是路由专家总数除以每 token 激活的专家数。
依据：paper/k3-src/2-model-architecture.tex L211-L215；paper/k3-src/2-model-architecture.tex L223-L224；paper/k3-src/2-model-architecture.tex L292-L296

把路由专家的计算从全宽 $d$ 移到宽度为 $\ell$ 的隐空间（比全宽更低维的中间向量表示），这一设计叫 LatentMoE：共享专家保留全宽路径处理通用变换，路由专家在更窄的隐空间里计算。Stable LatentMoE 用这层结构，并补三处稳定化。
依据：paper/k3-src/2-model-architecture.tex L214-L215；paper/k3-src/2-model-architecture.tex L221

这里用的路由不给辅助损失（加在训练目标上、用来使各专家负载趋于均衡的一项），而是直接按规则调整偏置，这个偏置加在该专家被挑选时用的分数上。Stable LatentMoE 用三处改动分别处理：在路由分支里插一道 [RMSNorm](wiki/rmsnorm)；对前馈层里相乘的两个因子分别取一个平滑上界，即 [SiTU-GLU](wiki/situ-glu)；专家的偏置不再按固定的步长更新，而由路由分数的分位数定出，即 Quantile Balancing。Stable LatentMoE 用在 K3 上，K3 是采用这层结构的一个模型。
依据：paper/k3-src/2-model-architecture.tex L217-L221；paper/k3-src/2-model-architecture.tex L253-L274；paper/k3-src/2-model-architecture.tex L289-L338

::: K3 的规模与配置取值
K3 的隐空间宽度取全宽的一半。

| 量 | 含义 | K3 取值 |
|---|---|---|
| $d$ | 模型全宽 | 7168 |
| $\ell$ | 路由隐空间宽度 | 3584 |
| $N$ | 路由专家总数 | 896 |
| $k$ | 每 token 激活的路由专家数 | 16 |
| $N_s$ | 共享专家数 | 2 |
| $N/k$ | 稀疏度 | 56 |
依据：paper/k3-src/2-model-architecture.tex L215；paper/k3-src/3-pre-training.tex L67-L89
:::

### 一层怎么算

这一层从前一层拿到输入 $\mathbf{x}\in\mathbb{R}^{d}$，$d$ 是模型全宽。它分成共享分支和路由分支，两支的结果相加作为这一层的输出。

共享分支有 $N_s$ 个共享专家，每个共享专家 $E_j^{\mathrm{shared}}$ 是一个 $d$ 进 $d$ 出的前馈网络，直接处理 $\mathbf{x}$。路由分支先把 $\mathbf{x}$ 经降维矩阵 $\mathbf{W}^{\downarrow}$ 投到 $\ell$ 维隐空间，得到 $\mathbf{z}=\mathbf{W}^{\downarrow}\mathbf{x}\in\mathbb{R}^{\ell}$；路由按分数挑出前 $k$ 个路由专家，记作集合 $\mathcal{T}_k(\mathbf{x})$。每个选中的路由专家 $E_i^{\mathrm{routed}}$ 是一个 $\ell$ 进 $\ell$ 出的前馈网络，它们的输出按路由权重 $p_i$ 加权求和，得到聚合表示 $\mathbf{u}\in\mathbb{R}^{\ell}$。$\mathbf{u}$ 先过 RMSNorm 归一化，再经上投影矩阵 $\mathbf{W}^{\uparrow}$ 投回 $d$ 维。把两支合起来：

$$\mathbf{u}=\sum_{i\in\mathcal{T}_k(\mathbf{x})} p_i\,E_i^{\mathrm{routed}}(\mathbf{W}^{\downarrow}\mathbf{x}),\qquad \mathbf{y}=\sum_{j=1}^{N_s} E_j^{\mathrm{shared}}(\mathbf{x})+\mathbf{W}^{\uparrow}\operatorname{RMSNorm}(\mathbf{u}).$$

依据：paper/k3-src/2-model-architecture.tex L223-L234

::: 一个手算的小例子：一层从输入走到输出
取一层的最小配置：全宽 $d=2$、隐空间 $\ell=2$、2 个路由专家、每 token 选 1 个（$k=1$）、1 个共享专家。

1. 输入 $\mathbf{x}=[2,2]$。
2. 降维矩阵取 $\mathbf{W}^{\downarrow}=[[1,0],[0,1]]$，得 $\mathbf{z}=\mathbf{W}^{\downarrow}\mathbf{x}=[2,2]$。
3. 路由选中专家 1；设 $E_1^{\mathrm{routed}}$ 是恒等映射，则 $\mathbf{u}=p_1 E_1^{\mathrm{routed}}(\mathbf{z})=[2,2]$，只有一个获胜者时 $p_1=1$。
4. $\mathbf{u}$ 的均方根是 $\sqrt{(2^2+2^2)/2}=2$，$\operatorname{RMSNorm}(\mathbf{u})=\mathbf{u}/2=[1,1]$，增益取 1。
5. 上投影矩阵取 $\mathbf{W}^{\uparrow}=[[1,2],[3,4]]$，得 $\mathbf{W}^{\uparrow}[1,1]^{\top}=[3,7]$。
6. 共享专家取 $E^{\mathrm{shared}}(\mathbf{x})=[[1,0],[0,2]]\mathbf{x}=[2,4]$。
7. 相加得 $\mathbf{y}=[2,4]+[3,7]=[5,11]$。
依据：自行构造
:::

### 极端稀疏放大出的两个失败模式

稀疏度 56 的意思是：896 个路由专家里，每个 token 只激活 16 个。

第一个在路由分支内部。路由分支把降维矩阵 $\mathbf{W}^{\downarrow}$、路由专家自身的前馈网络、上投影矩阵 $\mathbf{W}^{\uparrow}$ 依次相乘、接成一条链，这条链上连着近四个矩阵乘法。结构本身的条件数差（条件数衡量一串运算对扰动的放大程度），在 2.8 万亿参数的规模上，它会产出路由分支内部的激活值爆炸。
依据：paper/k3-src/2-model-architecture.tex L215-L219

第二个在专家负载上。训练里要让近一千个专家各拿到大致相等的 token，靠的是无辅助损失路由：它给每个专家一个偏置，偏置加在用于挑选的分数上，按固定步长更新。专家池到近千这一量级，这个固定步长的更新超出了它仍然表现良好的区间，负载会失衡；失衡会拖慢专家并行训练（把不同专家放到不同设备上并行），也可能让部分专家训练不足。
依据：paper/k3-src/2-model-architecture.tex L220；paper/k3-src/2-model-architecture.tex L291-L306

### 聚合之后、上投影之前加归一化

归一化这里指 [RMSNorm](wiki/rmsnorm)：把向量各分量平方的平均开方当尺度，逐个分量除以它，再乘一个可学增益。

原始的隐空间路由把上投影矩阵 $\mathbf{W}^{\uparrow}$ 直接作用在聚合表示 $\mathbf{u}$ 上。$\mathbf{u}$ 是选中的路由专家输出按路由权重加权求和，它的大小会随这次选中了哪些专家、权重各是多少而变，这种变化会跟着 $\mathbf{u}$ 一起进 $\mathbf{W}^{\uparrow}$，再进与共享分支的相加。Stable LatentMoE 在专家聚合之后、上投影之前插一道 RMSNorm，先把它归一化，路由分支对 $\mathbf{u}$ 尺度变化的敏感度就降低了。除了稳定训练，这道 RMSNorm 还持续改善验证损失（验证集上的损失）。
依据：paper/k3-src/2-model-architecture.tex L236-L240

::: 插入 RMSNorm 前后
原始隐空间路由与加了归一化的做法只差一道 RMSNorm。

- 原始：$\mathbf{y}=\sum_j E_j^{\mathrm{shared}}(\mathbf{x})+\mathbf{W}^{\uparrow}\mathbf{u}$。
- 归一化后：$\mathbf{y}=\sum_j E_j^{\mathrm{shared}}(\mathbf{x})+\mathbf{W}^{\uparrow}\operatorname{RMSNorm}(\mathbf{u})$。
依据：paper/k3-src/2-model-architecture.tex L237-L238
:::

### 两个乘性因子各加上界

这里的前馈激活是 [SwiGLU](wiki/swiglu)：两条线性投影逐元素相乘，其中一条先经 Swish 作门控（门控指用一条分支逐元素调制另一条）。Swish 指 $\operatorname{Swish}(z)=z\,\sigma(z)$，$\sigma$ 是 sigmoid 函数，$\sigma(z)=1/(1+e^{-z})$。SwiGLU 的两个乘性因子都没有上界，两个因子如果在同一个坐标上同时偏大，它们的乘积恰好在那个坐标上偏得更多，可能形成一个远高于其余坐标的激活值（外点），在低精度算术（分量用较少二进制位表示的浮点运算）下抬高溢出的风险。
依据：paper/k3-src/2-model-architecture.tex L253-L257；paper/k3-src/2-model-architecture.tex L271

SiTU-GLU 对两个因子分别取一个平滑饱和的上界函数 softcap：$\operatorname{softcap}(z,\beta)=\beta\tanh(z/\beta)$，它在原点附近近似线性、在大值处饱和到 $\beta$。定义写作：

$$\operatorname{SiTU\text{-}GLU}(\mathbf{x})=\Big[\beta_1\tanh\!\Big(\frac{\mathbf{W}_g\mathbf{x}}{\beta_1}\Big)\odot\sigma(\mathbf{W}_g\mathbf{x})\Big]\odot\Big[\beta_2\tanh\!\Big(\frac{\mathbf{W}_u\mathbf{x}}{\beta_2}\Big)\Big].$$

K3 取 $\beta_1=4$ 给门那一支、$\beta_2=25$ 给上支。
依据：paper/k3-src/2-model-architecture.tex L262-L271

::: 上界 100 与近原点的响应
1. $\tanh$ 把实数压进 $(-1,1)$，乘 $\beta$ 后值域是 $(-\beta,\beta)$。两个因子各自饱和到 $\beta_1$ 与 $\beta_2$，大到正侧时乘积不超过 $\beta_1\beta_2=4\times 25=100$。
2. 原点附近 $\tanh(z/\beta)\approx z/\beta$，两个因子分别近似 $\mathbf{W}_g\mathbf{x}$ 与 $\mathbf{W}_u\mathbf{x}$，整个式子近似 SwiGLU。
依据：paper/k3-src/2-model-architecture.tex L250；paper/k3-src/2-model-architecture.tex L270-L271
:::

### 用分位数给专家定 bias

负载均衡用的偏置 $b_j$ 只加在用于 top-$k$ 挑选的分数上，不进专家输出的加权比例，所以它只改变把 token 分给哪个专家，不改变各专家输出的相对权重。无辅助损失路由里，这套更新原本按固定步长做，步长要在适应太慢与负载来回震荡之间折中。
依据：paper/k3-src/2-model-architecture.tex L291-L305

Quantile Balancing 换掉了这个更新规则。记一批 $m$ 个 token、$N$ 个专家、每个 token 选 $k$ 个，则每个专家的目标负载是 $q=mk/N$ 个 token。它把挑选从 top-$k$ 改成在有偏置的分数上取 top-$(k{+}1)$：前 $k$ 个照常路由，第 $(k{+}1)$ 个的分数记为截断值 $\alpha_i$，一个专家要排进某个 token 的前 $k$，其分数就要超过该 token 的截断值。某个专家的偏置越高，它的分数越过各 token 截断值的机会就越多，拿到的 token 也越多；要求它恰好拿到 $q$ 个，阈值就落在各 token 的边距 $s_{i,j}-\alpha_i$（$s_{i,j}$ 是 token $i$ 在专家 $j$ 上的路由分数，$i$ 指 token、$j$ 指专家）的第 $(q{+}1)$ 大处。因为 $q/m=k/N$，这个位置也就是边距的 $(1-k/N)$ 分位数；把偏置设成这个分位数的相反数，负载就落到目标值。
依据：paper/k3-src/2-model-architecture.tex L303-L327

::: 与固定步长更新之别，以及直方图估计
1. 固定步长：$b_j\leftarrow b_j+\gamma\operatorname{sign}(\bar{L}-L_j)$，$\gamma$ 是步长，$\bar{L}$ 是平均负载、$L_j$ 是该专家当前负载；符号函数只给方向，步长恒定。
2. 分位数法：偏置由分位数一次算出，不再逐步逼近；更新只在下一步生效，一个 batch 不会用自己算出的偏置路由。
3. 规模上，边距有几百万条，分散在各张 GPU 与各累积步（梯度累积，把几个小批次的梯度攒起来再更新一次，这里每一步都有一批 token），精确取分位数不可行；改用一张各专家边距的直方图，一次全局汇总各卡的分箱计数，再从合计计数取分位数，误差不超过一个分箱宽度。
依据：paper/k3-src/2-model-architecture.tex L304-L310；paper/k3-src/2-model-architecture.tex L329-L337
:::

### 适用条件与边界

这些结论成立的范围是 K3 的规模与配置：896 个路由专家、每 token 激活 16 个、隐空间宽度 3584、全宽 7168、2.8 万亿参数。三处稳定化针对的都是极端稀疏下出现的两个失败模式——固定步长的无辅助损失更新在近千专家时失衡，路由分支的连续矩阵乘法链条件数差。
依据：paper/k3-src/2-model-architecture.tex L215-L221；paper/k3-src/3-pre-training.tex L59-L89

资料没有给出别的规模是否需要这三处改动：在更小的专家池或更低的稀疏度下，它们相对普通 MoE 是否仍必要，这里推不出来。
依据：paper/k3-src/2-model-architecture.tex L217-L221

三处稳定化不属于 LatentMoE 的定义：LatentMoE 只规定路由专家在隐空间里计算，本页说的稳定效果依赖这三处额外改动，不能当成 LatentMoE 本身的性质。
依据：paper/k3-src/2-model-architecture.tex L221；paper/k3-src/2-model-architecture.tex L236-L240
