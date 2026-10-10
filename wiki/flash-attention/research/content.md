# FlashAttention

- 描述：FlashAttention 是一种精确注意力算法，用分块把计算留在 GPU 片内，减少慢速显存的读写，在序列变长时更快、更省显存。
- 摘要：把注意力按块搬进片内 SRAM，用流式 SoftMax 逐块归约、反向按需重算注意力矩阵，访问慢速显存的次数少于标准注意力，结果与标准注意力在数学上相同（精确而非近似）。
- 主题：注意力机制, 推理系统
- 标签：推理加速

## 资料

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135v2. [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- Dao-AILab. flash-attention: the official implementation. 2026. [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

## 短页

### 把注意力交给更快的存储

标准注意力让每个位置用查询向量和所有位置的键向量做点积得到分数，把分数归一成权重，再对值向量加权求和，一次算出整段输出，见[标准注意力](wiki/standard-attention)。FlashAttention 是一种精确注意力算法，算出的结果和它在数学上相同，变的是在 GPU（显卡上的并行处理器）上怎么算。
依据：paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/abstract.tex L2-L10；paper/src-v2/src/algo.tex L28-L38

GPU 的存储器分多级，运行运算的单元旁边是一块小而快的片内存储器（SRAM），再往外是容量大而慢的显存（HBM）；FlashAttention 的改动就是把计算留在片内，$N$ 是序列长度。
依据：paper/src-v2/src/abstract.tex L6；paper/src-v2/src/intro.tex L31-L32；paper/src-v2/src/background.tex Hardware Performance

### 瓶颈在搬运，不在算数

标准注意力的中间结果是两个 $N\times N$ 的矩阵：分数矩阵，和它归一化后的权重矩阵。标准实现把它们写回显存，序列变长时它们按平方增长，反复读写使耗时变长。注意力的运算量也按序列长度的平方增长，但单看运算量少，并不等于墙钟时间（实际经过的时间）更短：其中 SoftMax 这类操作是访存受限的，搬运数据的时间占主要部分。
依据：paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/intro.tex L9；paper/src-v2/src/intro.tex L12-L14

### 把中间矩阵留在片内

做法是分块：把查询、键、值切成小块，一块搬到片内算出一部分结果，算完把这一块的贡献累加进输出，再换下一块；整个 $N\times N$ 的中间矩阵始终不写回显存。SoftMax 给每一行的分数做归一化：每个分数取指数，再除以这一行指数之和；为数值稳定，实际先减去这一行的最大值再取指数。要算这个和，得先看过一行的全部分数，所以它不能只看一块就算完：每处理一块，就把这一块的行最大值和指数和并进已经记下的统计；统计变了，之前累加的输出要按新旧统计的倍数整体缩放一次，再把这一块的结果加进去。
依据：paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation

训练要用反向传播：模型算出结果和目标的差距后，从输出往回求每个中间量和参数的梯度，用来更新参数；反向传播要那个 $N\times N$ 的中间矩阵来算梯度，但这时它已经不存在，只留了输出、SoftMax 统计和 dropout（训练时随机丢弃一部分计算结果）用的随机数状态，用查询、键、值块把它重新算出来，重算比从显存读回更快。
依据：paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation；paper/src-v2/src/algo_details.tex L180-L181

## 长页

Transformer 是一类神经网络，主干由一叠结构相同的层组成，每层里有一层注意力，让序列里每个位置从其他位置取信息。标准注意力给每个位置算出查询、键、值三个向量，用查询与所有位置的键做点积得到一组分数，把分数逐行归一成权重（SoftMax：每个分数取指数，再除以这一行指数之和），再按权重对所有位置的值向量加权求和，整段算成 $\text{softmax}(QK^\top/\sqrt{d})V$，一次算出所有位置的输出，见[标准注意力](wiki/standard-attention)。设序列长度为 $N$；注意力分成若干个头（每个头用一套自己的投影矩阵把向量投到维数较小的子空间里各算一遍，再拼回原维度）并行做，每个头里查询和键的维度为 $d$，分数矩阵 $QK^\top$ 有 $N$ 行 $N$ 列。
依据：paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/intro.tex Introduction

这个 $N\times N$ 的分数矩阵是开销的来源：序列变长，它按平方增长。
依据：paper/src-v2/src/abstract.tex L2-L4；paper/src-v2/src/intro.tex L9

FlashAttention 是一种精确注意力算法：算出的结果与标准注意力在数学上相同，改的是在 GPU（显卡上的并行处理器）上怎么算。它把查询、键、值切成块，一块搬进片内高速存储器（SRAM）算完再累加，片内存储器的容量记作 $M$，它的容量比显存（HBM）小几个量级。SoftMax 逐块归约（把一块一块算出的结果并成整行的结果）：逐块记录这一行的最大值和指数和，逐块更新输出，不把 $N\times N$ 的中间结果写回显存。训练要用反向传播：模型算出结果和目标的差距（损失）后，从输出往回逐层求各个中间量和参数的梯度，用来更新参数。反向传播需要那个 $N\times N$ 矩阵时，用查询、键、值块重新算出来，而不是把它存下来。
依据：paper/src-v2/src/abstract.tex L6-L7；paper/src-v2/src/algo.tex L31-L74；paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/intro.tex L39

### 注意力的瓶颈在哪

#### 更快的存储更小

GPU 的内存分多级。一级是显存（HBM，高带宽显存），容量大、速度相对慢；另一级是每个流多处理器（GPU 上并行执行的一个单元）上的片上存储器（SRAM），容量小、速度快得多。以 A100（一款数据中心 GPU 型号）为例：显存有 40 到 80 GB，带宽 1.5 到 2.0 TB/s；每个流多处理器的片上 SRAM 有 192 KB，带宽估计约 19 TB/s。片上 SRAM 比显存快一个量级，容量却小了好几个量级。一个运算从显存把输入读进寄存器和 SRAM，算完再把输出写回显存，所以访存要经过这两级，而快的这一级只能放下很小的一块。
依据：paper/src-v2/src/background.tex Hardware Performance

::: 一次注意力要搬多少字节
取序列长度 $N=512$、头维度 $d=32$，用单精度浮点（每个数 4 字节）估算，只数这些数组的大小，不数它们被读写的遍数。
依据：自行构造

1. 查询、键、值、输出四个数组各 $Nd$ 个数：$4\times512\times32\times4=262144$ 字节，约 0.25 MiB。
依据：自行构造

2. 分数矩阵和权重矩阵各 $N^2$ 个数：$2\times512^2\times4=2097152$ 字节，约 2 MiB。
依据：自行构造

3. 后者约是前者的 8 倍，且这个倍数随 $N$ 增大而增大，所以随序列变长，开销越来越集中在 $N\times N$ 的数组上。
依据：自行构造
:::

#### 计算受限与访存受限

一个运算在 GPU 上是快还是慢，看它受什么限制。算术强度指每搬运一个字节的数据能做多少次算术运算。算术强度高、运算量大到访存时间可以忽略的，叫计算受限，典型例子是内维（相乘再求和的那些维度）较大的矩阵乘；算术强度低、时间主要由访存决定的，叫访存受限，逐元素运算（对每个元素各算各的，如取指数）和归约（把一串数并成一个，如求和、SoftMax）都属于这一类。GPU 的运算速度比访存速度涨得快，所以这些操作越来越受访存限制。
依据：paper/src-v2/src/background.tex Hardware Performance

#### 标准注意力的访存开销

标准注意力把两个 $N\times N$ 的中间数组写回显存。第一步，按块把查询、键读进来，算分数矩阵，写下；第二步，把分数矩阵读回来，算 SoftMax 得到权重，写下；第三步，把权重和值读进来，算输出，写下。分数矩阵被写一遍、读一遍，权重矩阵也一样，这些 $N\times N$ 的读写随序列长度的平方增长；其中 SoftMax 这类操作是访存受限的，读写量越大，耗时就越长。
依据：paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/theory.tex L13-L24；paper/src-v2/src/proofs.tex L85-L105

::: 三步各读写多少
按算法逐步数，每步读入和写出的元素个数；查询、键、值、输出各 $Nd$ 个，分数、权重各 $N^2$ 个。
依据：paper/src-v2/src/proofs.tex L85-L105

1. 算 $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$：读查询、键，写分数，共 $Nd+Nd+N^2$ 个。
依据：paper/src-v2/src/proofs.tex L91-L94

2. 算 $\mathbf{P}=\text{softmax}(\mathbf{S})$：读分数，写权重，共 $N^2+N^2$ 个。
依据：paper/src-v2/src/proofs.tex L96-L98

3. 算 $\mathbf{O}=\mathbf{P}\mathbf{V}$：读权重、值，写输出，共 $N^2+Nd+Nd$ 个。
依据：paper/src-v2/src/proofs.tex L100-L102

三步合计里 $N^2$ 的项占主导，整体是 $\Theta(Nd+N^2)$。
依据：paper/src-v2/src/proofs.tex L104-L105
:::

### 分块算精确注意力

#### 分块与 SoftMax 整行归一化的冲突

分块要先解决 SoftMax 的整行归一化。SoftMax 沿每一行做归一化：位置 $i$ 的权重是它那一行的分数取指数后，各自除以这一行指数之和。要算这个和，得先看过这一行的全部分数，所以不能只看一块就把某一块的权重定下来。为了数值稳定，SoftMax 实际这样算：记这一行的最大值为 $m$，把每个分数减去 $m$ 再取指数，得到一组指数，把它们的和记为 $\ell$，权重就是每个指数除以 $\ell$；减去最大值只是让指数不至于太大，结果不变。
依据：paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation；paper/src-v2/src/algo_details.tex Memory-efficient forward pass

两段能合起来算。设一段的统计是 $(m^{(1)},\ell^{(1)})$、另一段是 $(m^{(2)},\ell^{(2)})$，合并后的最大值取两者中较大的，合并后的指数和要把两段的 $\ell$ 各自按「到新最大值差多少」重新缩放再相加，即 $\ell=e^{m^{(1)}-m}\ell^{(1)}+e^{m^{(2)}-m}\ell^{(2)}$。逐块累计当前的最大值和指数和，即可合并出整行。
依据：paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation

#### 流式 SoftMax 的分块合并示例

取一个位置对四个键的分数，分成两块，逐块合并出的指数和与整行一次算出的相同。
依据：自行构造

::: 四个分数分两块合并
设某个位置对四个键的分数依次是 $5$、$1$、$3$、$2$，分成两块：$x^{(1)}=(5,1)$、$x^{(2)}=(3,2)$。SoftMax 用减最大值的方式算。
依据：自行构造

1. 第一块：最大值 $m^{(1)}=5$；指数（分数减 $m^{(1)}$ 后取 $\exp$）是 $f^{(1)}=(e^{0},e^{-4})=(1,\,0.018316)$，指数和 $\ell^{(1)}=1.018316$。
依据：自行构造

2. 第二块：最大值 $m^{(2)}=3$；指数 $f^{(2)}=(e^{0},e^{-1})=(1,\,0.367879)$，指数和 $\ell^{(2)}=1.367879$。
依据：自行构造

3. 合并最大值：$m=\max(5,3)=5$。
依据：自行构造

4. 合并指数和：$\ell=e^{5-5}\times1.018316+e^{3-5}\times1.367879=1.018316+0.185122=1.203438$。第二块的指数和先乘 $e^{-2}\approx0.135335$ 缩到以 $5$ 为基准。
依据：自行构造

5. 直接对四个分数整体算：最大值仍是 $5$，指数是 $1$、$0.018316$、$0.135335$、$0.049787$，和是 $1.203438$，与逐块合并的完全相同。
依据：自行构造

6. 这个位置的权重：每个分数减 $5$ 取指数再除以 $\ell$，得 $(0.8310,\,0.0152,\,0.1125,\,0.0414)$。
依据：自行构造
:::

#### 分块算法

分块算法把查询、键、值都切成块，块的大小按片内 SRAM 的容量 $M$ 定，要让键值块、查询块、输出块和分数块同时放进 SRAM；算法里取键值块 $B_c=\lceil M/(4d)\rceil$、查询块 $B_r=\min(\lceil M/(4d)\rceil,d)$。算法分两层循环：外层遍历键值块，每进一个块就把它搬进 SRAM；内层遍历查询块，对每一对 $(\mathbf{Q}_i,\mathbf{K}_j)$ 在片内算出分数块 $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^\top$，求出这一小块的行最大值和指数和，连同已经记下的统计一起更新，再用更新后的统计刷新输出块 $\mathbf{O}_i$。同一块 $\mathbf{K}_j,\mathbf{V}_j$ 被所有查询块复用。输出和两个统计量 $m$、$\ell$ 是逐块累加出来的，每次只写回一小块。
依据：paper/src-v2/src/algo.tex L85-L116；paper/src-v2/src/algo_details.tex Forward Pass；paper/src-v2/src/proofs.tex L116-L131

<svg viewBox="0 0 700 250" font-family="sans-serif" font-size="13">
  <defs><marker id="fa-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#475569"/></marker></defs>
  <rect x="8" y="80" width="160" height="90" rx="8" fill="#eef2ff" stroke="#475569"/>
  <foreignObject x="8" y="80" width="160" height="90">
    <div style="display:flex;flex-direction:column;justify-content:center;height:100%;text-align:center;color:#1e293b">
      <div>慢速显存 HBM</div>
      <div>查询 键 值 输出</div>
    </div>
  </foreignObject>
  <line x1="168" y1="125" x2="236" y2="125" stroke="#475569" marker-end="url(#fa-arrow)"/>
  <foreignObject x="178" y="108" width="52" height="20"><div style="text-align:center;color:#475569">取块</div></foreignObject>
  <rect x="240" y="16" width="296" height="218" rx="8" fill="#f8fafc" stroke="#475569"/>
  <foreignObject x="240" y="16" width="296" height="218">
    <div style="padding:6px 10px;color:#1e293b;line-height:1.7">
      <div style="text-align:center">片内 SRAM：分块在此算出，不保留整块</div>
      <div>1 取键值块 $\mathbf{K}_j$、$\mathbf{V}_j$</div>
      <div>2 逐个取查询块 $\mathbf{Q}_i$</div>
      <div>3 算分数块 $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^\top$</div>
      <div>4 更新 $m_i$、$\ell_i$、输出块 $\mathbf{O}_i$</div>
    </div>
  </foreignObject>
  <line x1="536" y1="125" x2="604" y2="125" stroke="#475569" marker-end="url(#fa-arrow)"/>
  <foreignObject x="546" y="108" width="52" height="20"><div style="text-align:center;color:#475569">写回</div></foreignObject>
  <rect x="606" y="80" width="86" height="90" rx="8" fill="#eef2ff" stroke="#475569"/>
  <foreignObject x="606" y="80" width="86" height="90">
    <div style="display:flex;flex-direction:column;justify-content:center;height:100%;text-align:center;color:#1e293b">
      <div>HBM</div>
      <div>输出 $\mathbf{O}$</div>
    </div>
  </foreignObject>
</svg>
依据：自行构造

::: 每一块里的公式
记第 $i$ 个查询块在此前所有键值块上累计的行最大值为 $m_i$、指数和为 $\ell_i$、输出为 $\mathbf{O}_i$。记法中 $\text{diag}(\ell_i)$ 是把向量铺成对角矩阵，$\text{rowmax}$、$\text{rowsum}$ 是逐行取最大值、逐行求和。处理键值块 $j$ 时：
依据：paper/src-v2/src/algo.tex L101-L111

1. 在片内算分数块 $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^\top\in\mathbb{R}^{B_r\times B_c}$。
依据：paper/src-v2/src/algo.tex L105

2. 这一块的行最大值 $\tilde{m}_{ij}=\text{rowmax}(\mathbf{S}_{ij})$；指数 $\tilde{\mathbf{P}}_{ij}=\exp(\mathbf{S}_{ij}-\tilde{m}_{ij})$；这一块的指数和 $\tilde{\ell}_{ij}=\text{rowsum}(\tilde{\mathbf{P}}_{ij})$。
依据：paper/src-v2/src/algo.tex L106-L107

3. 更新最大值 $m_i^{\text{new}}=\max(m_i,\tilde{m}_{ij})$，更新指数和 $\ell_i^{\text{new}}=e^{m_i-m_i^{\text{new}}}\ell_i+e^{\tilde{m}_{ij}-m_i^{\text{new}}}\tilde{\ell}_{ij}$。
依据：paper/src-v2/src/algo.tex L108

4. 刷新输出并写回：
依据：paper/src-v2/src/algo.tex L109-L111

$$\mathbf{O}_i\leftarrow\text{diag}(\ell_i^{\text{new}})^{-1}\left(\text{diag}(\ell_i)e^{m_i-m_i^{\text{new}}}\mathbf{O}_i+e^{\tilde{m}_{ij}-m_i^{\text{new}}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_j\right)$$

把 $m_i\leftarrow m_i^{\text{new}}$、$\ell_i\leftarrow\ell_i^{\text{new}}$ 写回；算法结束时返回的 $\mathbf{O}$ 就是 $\text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$，运算量是 $O(N^2d)$，额外的存储是 $O(N)$ 而不是 $O(N^2)$。
依据：paper/src-v2/src/algo.tex L111-L124
:::

### 重算替代存储

#### 省下什么、多算什么

标准注意力的反向传播要用到两个 $N\times N$ 的矩阵本身：分数矩阵 $\mathbf{S}$ 和权重矩阵 $\mathbf{P}$，它们是算查询、键、值的梯度的中间量。它们太大，留在显存里开销太大，FlashAttention 保存输出 $\mathbf{O}$、逐行统计 $m$、$\ell$，以及 dropout（训练时随机丢弃一部分计算结果）用的随机数状态；反向传播需要 $\mathbf{S}$、$\mathbf{P}$ 时，用片内已有的查询、键、值块把它们重新算出来。算梯度要用到两个中间量：概率的梯度 $\mathbf{dP}=\mathbf{dO}\,\mathbf{V}^\top$（$\mathbf{dO}$ 是输出 $\mathbf{O}$ 的梯度），以及分数梯度里的一行折减 $D_i=\mathbf{dO}_i^\top\mathbf{O}_i$；它们都只用块内的数据。
依据：paper/src-v2/src/algo.tex L64-L74；paper/src-v2/src/algo_details.tex Memory-efficient backward pass；paper/src-v2/src/algo_details.tex Backward Pass

::: 反向传播怎么按块重算
给定输出梯度 $\mathbf{dO}$，要算查询、键、值的梯度 $\mathbf{dQ}$、$\mathbf{dK}$、$\mathbf{dV}$。利用前向保存的 $m$、$\ell$，用查询、键、值块重算分数块和权重块，再逐块累加梯度。为简洁，分数块按 $\mathbf{Q}_i\mathbf{K}_j^\top$ 写，SoftMax 前的缩放常数（标准注意力在分数上乘的 $1/\sqrt{d}$）略去，它只是分数上的一个乘数，不改变分块归约的形式。$\circ$ 表示逐元素相乘。
依据：paper/src-v2/src/algo_details.tex Backward Pass；paper/src-v2/src/algo_details.tex L154

1. 用保存的统计重算权重块：$\mathbf{P}_{ij}=\text{diag}(\ell_i)^{-1}\exp(\mathbf{S}_{ij}-m_i)$，其中分数块 $\mathbf{S}_{ij}$ 由 $\mathbf{Q}_i\mathbf{K}_j^\top$ 重算。
依据：paper/src-v2/src/algo_details.tex L291-L293

2. 值梯度按块累加：$\tilde{\mathbf{dV}}_j\leftarrow\tilde{\mathbf{dV}}_j+\mathbf{P}_{ij}^\top\mathbf{dO}_i$。
依据：paper/src-v2/src/algo_details.tex L299-L300

3. 概率的梯度块 $\mathbf{dP}_{ij}=\mathbf{dO}_i\mathbf{V}_j^\top$。
依据：paper/src-v2/src/algo_details.tex L301-L302

4. 行折减 $D_i=\text{rowsum}(\mathbf{dO}_i\circ\mathbf{O}_i)$，再由它和 $\mathbf{P}_{ij}$ 得到分数梯度块 $\mathbf{dS}_{ij}=\mathbf{P}_{ij}\circ(\mathbf{dP}_{ij}-D_i)$。
依据：paper/src-v2/src/algo_details.tex L305-L306

5. 查询梯度按块累加 $\mathbf{dQ}_i\leftarrow\mathbf{dQ}_i+\mathbf{dS}_{ij}\mathbf{K}_j$，键梯度在片内累加 $\tilde{\mathbf{dK}}_j\leftarrow\tilde{\mathbf{dK}}_j+\mathbf{dS}_{ij}^\top\mathbf{Q}_i$。
依据：paper/src-v2/src/algo_details.tex L307-L309

整条反向的额外存储是 $O(N)$ 而不是 $O(N^2)$；访问显存的次数与前向同阶，是 $\Theta(N^2d^2M^{-1})$。
依据：paper/src-v2/src/algo_details.tex L317-L327
:::

#### 重算为什么反而更快

重算要多做矩阵乘，运算量因此上升，但它省下的是读写 $N\times N$ 数组的访存，而注意力的耗时主要由访存决定，省下的访存收益大于多做的运算。重算还有一个结果：注意力矩阵按块重新算出来，全程留在片内，于是显存里不必再存这个 $N\times N$ 矩阵。两件事合起来，反向传播反而更快。
依据：paper/src-v2/src/algo.tex L64-L74；paper/src-v2/src/theory.tex L52-L64

### 访问次数与下限

#### 算法要搬多少字节

顺着分块算法数访问显存的次数。键和值各被读进片内一次；查询和输出因为要被所有键值块各用一次，会重复读若干遍。设序列长度为 $N$、头维度为 $d$、片内 SRAM 容量为 $M$，键值块的大小 $B_c$ 约为 $M/d$，键值块个数 $T_c=N/B_c$ 约为 $Nd/M$（这里略去块尺寸公式中的常数因子 4），查询和输出就大致被读写 $T_c$ 遍。把这些加起来，FlashAttention 访问显存的次数是 $\Theta(N^2d^2M^{-1})$；作为对照，标准注意力是 $\Theta(Nd+N^2)$。
依据：paper/src-v2/src/theory.tex L13-L34；paper/src-v2/src/proofs.tex L109-L114

::: 访问次数怎么数出来
设序列长度为 $N$、头维度为 $d$、片内 SRAM 容量为 $M$。
依据：paper/src-v2/src/proofs.tex L107-L145

1. 键、值各读进片内一次，约 $Nd$ 次访问；查询、输出各被读写 $T_c$ 遍，约 $Nd\,T_c$ 次。合计 $\Theta(Nd+Nd\,T_c)=\Theta(Nd\,T_c)$。
依据：paper/src-v2/src/proofs.tex L109-L114

2. 三种数组都要能放进 SRAM：键值块 $B_cd=O(M)$，查询输出块 $B_rd=O(M)$，分数块 $B_rB_c=O(M)$。
依据：paper/src-v2/src/proofs.tex L116-L131

3. 由此取 $B_c=\Theta(M/d)$、$B_r=\Theta(\min(M/d,d))$，于是 $T_c=N/B_c=\Theta(Nd/M)$。
依据：paper/src-v2/src/proofs.tex L132-L140

4. 代入得访问次数 $\Theta(Nd\,T_c)=\Theta(N^2d^2/M)$。
依据：paper/src-v2/src/proofs.tex L142-L145

对常见的 $d$（64 到 128）和 $M$（约 100 KB），$d^2$ 比 $M$ 小很多，因此 FlashAttention 比标准注意力少很多。
依据：paper/src-v2/src/theory.tex L20-L24
:::

#### 没有更好的精确算法

还存在一个下界：只要要求精确（结果与标准注意力在数学上相同），就不存在一种算法，能在片内容量的各个取值上都把访问次数的量级压得比 $\Theta(N^2d^2M^{-1})$ 更低。
依据：paper/src-v2/src/theory.tex L36-L50；paper/src-v2/src/proofs.tex L149-L164

#### 块多大合适

块越大，键值块个数越少，查询被重复读写的遍数越少，访问显存越少，运行时随之下降。这个下降有个上限：键值块的行数到 256 以后，运行时不再由访存决定，转而由运算等其他因素决定，继续放大块不再有收益。
依据：paper/src-v2/src/theory.tex L52-L64

### 快了多少、能干什么

#### 运行时与访存

同一份注意力，FlashAttention 的浮点运算次数比标准实现更多，但访问显存的字节数少得多（约一个量级），耗时因此更短；它的显存占用随序列长度线性增长，长序列上比标准实现小很多。
依据：paper/src-v2/src/theory.tex L54-L57；paper/src-v2/src/theory.tex L66-L101；paper/src-v2/src/experiments.tex Benchmarking Attention；paper/src-v2/tables/memory_use.tex L10-L23

::: 一组具体数字
在同一份注意力上做前向加反向：
依据：paper/src-v2/src/theory.tex L66-L101

1. 浮点运算次数：标准实现 66.6 GFLOPs，FlashAttention 75.2 GFLOPs，重算使它更多。
依据：paper/src-v2/src/theory.tex L54-L56；paper/src-v2/src/theory.tex L75-L76

2. 访问显存的字节数：标准实现 40.3 GB，FlashAttention 4.4 GB。
依据：paper/src-v2/src/theory.tex L76-L77

3. 耗时：标准实现 41.7 ms，FlashAttention 7.3 ms。
依据：paper/src-v2/src/theory.tex L77

显存占用（单位 MB）随序列长度线性增长，FlashAttention：1024 时 209，4096 时 836，16384 时 3344，65536 时 13376。
依据：paper/src-v2/tables/memory_use.tex L23
:::

#### 能放更长的上下文

它能容纳更长的序列：序列能放多长，直接决定模型一次能看到多少上下文。
依据：paper/src-v2/src/abstract.tex L9-L10；paper/src-v2/src/intro.tex L37；paper/src-v2/src/experiments.tex Better Models with Longer Sequences；paper/src-v2/src/theory.tex L20-L24

### 适用条件与边界

分块成立依赖两点：SoftMax 要按减最大值的方式做分块归约，这是数值稳定的写法；访问次数的分析则假定片内 SRAM 容量满足 $d\le M\le Nd$。
依据：paper/src-v2/src/algo.tex L46-L52；paper/src-v2/src/theory.tex L14-L15

更快指的是墙钟时间（实际经过的时间），不是浮点运算次数：这里的浮点运算次数反而更多，快是因为访存少。
依据：paper/src-v2/src/algo.tex L64-L74；paper/src-v2/src/theory.tex L52-L57

代价是工程上的：每种注意力要单独写一份在 CUDA（英伟达 GPU 的编程平台）上运行的实现，不能像通用框架那样用同一段代码；实现也不一定能搬到别的 GPU 架构上。
依据：paper/src-v2/src/discussion.tex Limitations and Future Directions

加速倍率随序列长度变化，不是固定值：注意力耗时随序列长度的平方增长，到了较长的序列，一些按线性耗时近似的注意力方法会开始更快；这里比较的是精确实现。
依据：paper/src-v2/src/experiments.tex L244-L249
