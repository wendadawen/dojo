# Gated DeltaNet

- 描述：Gated DeltaNet 是一种线性注意力模型，它把已经处理过的序列压进一张固定大小的矩阵状态，更新时既整体缩小旧关联，又精确改写其中一条。
- 摘要：状态递推 $S_t = S_{t-1}\big(\alpha_t(I-\beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top$，门控 $\alpha_t$ 抹除旧记忆，delta 规则按写入强度 $\beta_t$ 精确改写一条关联。
- 主题：模型结构, 训练与优化
- 标签：注意力

## 资料

- Songlin Yang, Jan Kautz, Ali Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025, arXiv:2412.06464v3. [https://arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464)
- NVIDIA. GatedDeltaNet: official implementation. 2025. [https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

## 短页

### 一张会写也会忘的矩阵状态

Gated DeltaNet 属于线性注意力：这类模型不保存每个位置对所有历史位置的注意力，而是把已经处理过的序列压进一张固定大小的矩阵状态 $\rmS$（形状 $d_v \times d_k$，$d_k$ 是键的维度、$d_v$ 是值的维度），每处理一个位置就在这张状态上更新一次。更新规则是

$$\rmS_t = \rmS_{t-1}\big(\alpha_t(\rmI - \beta_t \vk_t \vk_t^\top)\big) + \beta_t \vv_t \vk_t^\top.$$

$t$ 是位置，$\vk_t$（$d_k$ 维）叫键、$\vv_t$（$d_v$ 维）叫值，由当前位置的输入算出；$\rmI$ 是单位矩阵，$\vk_t \vk_t^\top$ 是键与自己的转置相乘（列乘行）得到的矩阵，也叫外积。$\alpha_t \in (0,1)$ 是门控，控制整张状态缩小多少；$\beta_t \in (0,1)$ 是写入强度，控制键 $\vk_t$ 这个方向上的关联被改动多少，两个都是当前位置算出的标量。$\alpha_t$ 这一项叫整体衰减；$\rmI - \beta_t \vk_t \vk_t^\top$ 这一项叫 delta 规则，意思是把键 $\vk_t$ 上旧的值擦掉、再写进新值。读出时用查询 $\vq_t$（$d_k$ 维）乘状态，$\vo_t = \rmS_t \vq_t$。
依据：paper/src-v3/main.tex Formulation: Gated Delta Rule；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule

### 门控与 delta 补足的部分

状态大小只由键和值的维度决定，不随序列长度增长，因此能可靠存放的键值对数量有限，序列一旦比它长，新旧关联就互相干扰、难以区分。只有整体衰减时，每一步把所有关联按同一个比例缩小，无法只丢掉其中一条；只有 delta 精确改写时，一次只改一条关联，缺少快速清空的手段。Gated DeltaNet 把两者合成一条递推：门控 $\alpha_t$ 接近 0 时状态被清空，接近 1 时保留；写入强度 $\beta_t$ 控制那一条关联被换掉多少。
依据：paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule

### 一步更新与代价

每一步先把整个状态按 $\alpha_t$ 缩小，再在键 $\vk_t$ 的方向上按 $\beta_t$ 改写一条关联，其余方向不动。状态每步的形状不变，更新的计算量和需要保存的状态都与序列已经处理了多少个位置无关。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Formulation: Gated Delta Rule

## 长页

线性注意力把已经处理过的序列压进一张固定大小的矩阵状态，每处理一个位置就更新这张矩阵，读的时候用查询从矩阵里取回对应的值。每个位置有两个由输入算出的向量：键标记这条记录对应哪一处，值是被记下的内容。状态的大小只由这两个向量的维度决定，与序列长度无关，因此按顺序处理一遍就能得到每个位置的输出。Gated DeltaNet 是这类模型里的一种，它的内容就是更新这张状态的一条递归规则。

状态能存下的键值对数量受它的维度限制，序列变长后会不够用，需要把不再相关的关联丢掉；已经存下的关联又需要在遇到新值时被改成新的，而不是简单叠加。
依据：paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule

### 记忆：一条矩阵递推

Gated DeltaNet 只替换[线性注意力](wiki/linear-attention)那条递推的更新部分，状态仍是同一张矩阵，写入和读出沿用同样的形式。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule

#### 外积写入与读出

最基础的形式里，第 $t$ 步把当前位置的外积加到状态上：

$$\rmS_t = \rmS_{t-1} + \vv_t \vk_t^\top.$$

$\rmS_t$ 是第 $t$ 步的状态，$\rmS_{t-1}$ 是上一步的状态；$\vk_t$ 是键（$d_k$ 维），$\vv_t$ 是值（$d_v$ 维），$\vv_t \vk_t^\top$ 是值与键的转置相乘（列乘行）得到的 $d_v \times d_k$ 矩阵。读出用查询 $\vq_t$（$d_k$ 维）乘状态，$\vo_t = \rmS_t \vq_t$。查询方向与历史上某个键方向重合时，点积把那个键对应的值取回；$\vo_t$ 是这些值按重合程度的加权和。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay

::: 展开成矩阵并行形式
1. 把读出式 $\vo_t = \rmS_t \vq_t$ 按递推展开，第 $i$ 步的输出是所有不超过 $i$ 的键位置各自贡献的和：$\vo_i = \sum_{j=1}^i \vv_j(\vk_j^\top \vq_i)$。
2. 把整段序列的查询、键、值分别按行排成矩阵 $\rmQ$、$\rmK$、$\rmV$，上式可以写成 $\rmO = (\rmQ \rmK^\top \odot \rmM)\rmV$。
3. $\rmM$ 是因果掩码，它把查询位置 $i$ 不允许看到的、排在它后面的键位置（$i < j$）置 0，其余置 1；$\odot$ 是逐元素相乘。这样整段输出一次算完，不再逐位置递推。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay
:::

#### 衰减：整体按比例缩小

在这条递推上加一个数据相关的标量门控 $\alpha_t \in (0,1)$，得到

$$\rmS_t = \alpha_t \rmS_{t-1} + \vv_t \vk_t^\top.$$

$\alpha_t$ 由当前位置的输入算出，乘在整个状态前，所以每一步把所有已经存下的关联按同一个比例缩小，不分对象。跨多个位置累积起来记作 $\gamma_t = \prod_{i=1}^t \alpha_i$。想丢掉某一条旧关联时，其余的关联也一起被缩小。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay

#### delta 规则：擦掉一条旧的再写新的

delta 规则改动的不是整体比例，而是状态里与当前键对应的那一条。它先把键 $\vk_t$ 上已经存着的值 $\rmS_{t-1}\vk_t$ 读回来，按写入强度 $\beta_t \in (0,1)$ 把这部分擦掉，再沿键 $\vk_t$ 的方向写进新值 $\beta_t \vv_t + (1-\beta_t)\rmS_{t-1}\vk_t$。两个动作合起来是一步秩一更新：

$$\rmS_t = \rmS_{t-1}(\rmI - \beta_t \vk_t \vk_t^\top) + \beta_t \vv_t \vk_t^\top.$$

$\rmI - \beta_t \vk_t \vk_t^\top$ 只改变键 $\vk_t$ 所在的那个方向，其余方向保持不动；$\beta_t$ 同时是这一步的写入强度和旧值到新值之间的插值比例，$\beta_t = 1$ 时旧值被完全换掉。
依据：paper/src-v3/main.tex Delta Networks: Linear Attention with Delta Rule；paper/src-v3/main.tex Introduction

#### 合起来：门控 delta 规则

把衰减和 delta 乘在一起，就得到 Gated DeltaNet 的更新规则：

$$\rmS_t = \rmS_{t-1}\big(\alpha_t(\rmI - \beta_t \vk_t \vk_t^\top)\big) + \beta_t \vv_t \vk_t^\top.$$

括号里的 $\alpha_t$ 控制整个状态缩小多少，$\beta_t$ 控制当前键上的那一条关联改写多少，两个标量各自控制一件事。$\alpha_t$ 接近 0 时状态几乎被清空，$\alpha_t$ 接近 1 时衰减几乎不发生、只剩 delta 式的精确改写；$\beta_t = 0$ 时这一步什么都不写、只发生衰减。递推右边只用到上一步的状态、当前位置的键和值、以及两个标量，与序列长度无关。
依据：paper/src-v3/main.tex Formulation: Gated Delta Rule；paper/src-v3/main.tex Introduction

::: 从一步梯度下降推出这条规则
1. 把状态看成一张要在线拟合键值对的权重矩阵：每来一个键值对，就按当前误差调整一次状态。取当前键值对上的平方误差 $\mathcal{L}(\rmS) = \frac{1}{2}\|\rmS \vk_t - \vv_t\|^2$，它衡量状态在键 $\vk_t$ 上读出的值离目标值 $\vv_t$ 差多远。
2. 对 $\rmS$ 求梯度，$\nabla \mathcal{L}(\rmS) = (\rmS \vk_t - \vv_t)\vk_t^\top$。
3. 沿梯度下降一步、步长取 $\beta_t$，得 $\rmS_t = \rmS_{t-1} - \beta_t(\rmS_{t-1}\vk_t - \vv_t)\vk_t^\top$。
4. 把括号展开，$\rmS_t = \rmS_{t-1}(\rmI - \beta_t \vk_t \vk_t^\top) + \beta_t \vv_t \vk_t^\top$，就是 delta 规则；$\beta_t$ 是这一步的学习率。
5. 在状态上再乘一个衰减系数 $\alpha_t$，得到 $\rmS_t = \rmS_{t-1}\big(\alpha_t(\rmI - \beta_t \vk_t \vk_t^\top)\big) + \beta_t \vv_t \vk_t^\top$。$\alpha_t$ 相当于加在权重上的衰减。
依据：paper/src-v3/main.tex Formulation: Gated Delta Rule
:::

### 一个可以手算的例子

取单个头（同一层里并行算的一组独立的键值查询与状态；这里只用一张 $2 \times 2$ 的状态矩阵、键和值各 2 维），从全零状态开始算四步：前两步写入两条互不相同的关联，第三步用 $\beta=0.5$ 改其中一条，第四步用 $\alpha=0.5$ 把整个状态缩小一半。
依据：自行构造

::: 四步的完整取值
$\rmS_0$ 是 $2 \times 2$ 全零矩阵，每一步的键、值、$\alpha$、$\beta$ 如下，按 $\rmS_t = \rmS_{t-1}(\alpha_t(\rmI - \beta_t \vk_t \vk_t^\top)) + \beta_t \vv_t \vk_t^\top$ 计算。$\alpha$、$\beta$ 一般取 0 到 1 之间的数，例子里的 $\alpha=1$、$\beta=1$、$\beta=0$ 取的是区间的端点，用来让每一步结果干净。
依据：自行构造

1. 第 1 步 $\vk_1=(1,0)$、$\vv_1=(1,0)$、$\alpha_1=1$、$\beta_1=1$：$\rmS_1=\vv_1\vk_1^\top=\begin{pmatrix}1 & 0\\ 0 & 0\end{pmatrix}$，键 $(1,0)$ 上存了值 $(1,0)$。
2. 第 2 步 $\vk_2=(0,1)$、$\vv_2=(0,1)$、$\alpha_2=1$、$\beta_2=1$：$\rmS_2=\rmS_1+\vv_2\vk_2^\top=\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}$，两条关联各占一个方向。
3. 第 3 步 $\vk_3=(1,0)$、$\vv_3=(3,0)$、$\alpha_3=1$、$\beta_3=0.5$：$\rmS_3=\rmS_2(\rmI-0.5\,\vk_3\vk_3^\top)+0.5\,\vv_3\vk_3^\top=\begin{pmatrix}2 & 0\\ 0 & 1\end{pmatrix}$。键 $(1,0)$ 上读出的值从 $1$ 变成 $2$，正好是旧值 $1$ 和新值 $3$ 按 $1-\beta_3=0.5$ 与 $\beta_3=0.5$ 插值的结果；键 $(0,1)$ 那条没有变。
4. 第 4 步取 $\alpha_4=0.5$、$\beta_4=0$：$\rmS_4=0.5\,\rmS_3=\begin{pmatrix}1 & 0\\ 0 & 0.5\end{pmatrix}$，两条关联一起被缩小一半，没有一条被单独改写。
5. 用查询 $\vq=(1,0)$ 读出：第 2 步 $\rmS_2\vq=(1,0)$，第 3 步 $\rmS_3\vq=(2,0)$，取到的都是键 $(1,0)$ 在那一刻对应的值。
依据：自行构造

```python
S = [[0.0, 0.0], [0.0, 0.0]]
steps = [
    ((1.0, 0.0), (1.0, 0.0), 1.0, 1.0),
    ((0.0, 1.0), (0.0, 1.0), 1.0, 1.0),
    ((1.0, 0.0), (3.0, 0.0), 1.0, 0.5),
    ((0.0, 0.0), (0.0, 0.0), 0.5, 0.0),
]
for n, (k, v, a, b) in enumerate(steps, 1):
    S = [[a * (S[i][j] - b * sum(S[i][m] * k[m] for m in range(2)) * k[j])
          + b * v[i] * k[j] for j in range(2)] for i in range(2)]
    print("第", n, "步 S =", S)
```

```text
第 1 步 S = [[1.0, 0.0], [0.0, 0.0]]
第 2 步 S = [[1.0, 0.0], [0.0, 1.0]]
第 3 步 S = [[2.0, 0.0], [0.0, 1.0]]
第 4 步 S = [[1.0, 0.0], [0.0, 0.5]]
```
:::

### 分块并行训练

逐位置执行更新规则时，第 $t$ 步必须先算出第 $t-1$ 步的状态，一整段序列只能串行地一步步算。训练希望把这些计算变成矩阵乘、让尽量多的位置并行。做法是按块处理：把序列切成若干长度为 $C$ 的块，一块内的位置一起算，块与块之间顺序传递状态。

块首拿到上一块传下来的状态 $\rmS_{[t]}$。把更新规则在块内逐步展开，块内第 $r$ 个位置的状态可以分成两部分：块首状态经过块内前 $r$ 步的变换，加上块内前 $r$ 步自己新写进来的部分。两部分的系数都由块内的键、值、门控和写入强度决定，与块前面的内容无关，因此都能在块内一次并行算完。门控进入后，块首状态在块内每一步都被乘上一个 $\alpha$，累计成一个衰减因子 $\gamma$；块内新写进来的部分也带对应的 $\gamma$ 比值。于是每块的处理量只与块长 $C$ 有关，整段序列的计算量随序列长度线性增长。

块内状态的变换和写入部分都用 WY 表示写成矩阵形式。WY 表示把一串 $\rmI - \beta_i \vk_i \vk_i^\top$ 的连乘压成 $\rmI - \sum_i \vw_i \vk_i^\top$ 这种只含一项求和的形式，其中 $\vw_i$ 由一条递推给出；写入部分多带一个 $\gamma$ 比值，形如一串 $\tilde{\vu}_i \vk_i^\top$ 的加权和。再用 UT 变换（一种把这两条递推一次解成下三角矩阵的求法）把它们解成矩阵，块内的整段更新就归约成矩阵乘。
依据：paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Algorithm: Hardware-efficient Chunkwise training

::: 块内展开与 WY 表示
1. 块内展开 $r$ 步：$\rmS_{[t]}^r = \rmS_{[t]}\mathbf{F}_{[t]}^r + \mathbf{G}_{[t]}^r$，$\rmS_{[t]}$ 是块首状态，$\mathbf{F}_{[t]}^r$ 是块首状态经过的变换，$\mathbf{G}_{[t]}^r$ 是块内新写进来的部分。
2. $\mathbf{F}_{[t]}^r = \prod_{i=1}^r \alpha_{[t]}^i(\rmI - \beta_{[t]}^i\vk_{[t]}^i\vk_{[t]}^{i\top}) = \gamma_{[t]}^r \rmP_{[t]}^r$：块内累积衰减 $\gamma_{[t]}^r = \prod_{i=1}^r \alpha_{[t]}^i$ 乘上 $r$ 个 $\rmI - \beta\vk\vk^\top$ 的连乘 $\rmP_{[t]}^r$。
3. 用 WY 表示把连乘写成 $\rmP_{[t]}^r = \rmI - \sum_{i=1}^r \vw_{[t]}^i \vk_{[t]}^{i\top}$，其中 $\vw_{[t]}^r = \beta_{[t]}^r\big(\vk_{[t]}^r - \sum_{i=1}^{r-1}\vw_{[t]}^i(\vk_{[t]}^{i\top}\vk_{[t]}^r)\big)$。这条递推只用到写入强度和键，不含门控。
4. 写入部分同理写成 $\mathbf{G}_{[t]}^r = \sum_{i=1}^r \frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}\tilde{\vu}_{[t]}^i\vk_{[t]}^{i\top}$，$\tilde{\vu}_{[t]}^r = \beta_{[t]}^r\big(\vv_{[t]}^r - \sum_{i=1}^{r-1}\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}\tilde{\vu}_{[t]}^i(\vk_{[t]}^{i\top}\vk_{[t]}^r)\big)$，与 $\vw$ 的递推同形，只是每一项多一个 $\gamma$ 比值。
5. 写成矩阵：$\rmK_{[t]}$、$\rmV_{[t]}$ 是块内键、值按行排成的矩阵，$\operatorname{diag}(\beta_{[t]})$ 是写入强度排成的对角矩阵，$\Gamma_{[t]}$ 的第 $(i,j)$ 元取 $\gamma_{[t]}^i/\gamma_{[t]}^j$（$i\ge j$，其余为 0），$\operatorname{strictLower}(\cdot)$ 取矩阵的严格下三角（不含对角线）。不含门控的 $\rmW_{[t]}$ 由第 3 步那条递推解出：$\rmW_{[t]} = \big[\rmI + \operatorname{strictLower}\big(\operatorname{diag}(\beta_{[t]})\rmK_{[t]}\rmK_{[t]}^\top\big)\big]^{-1}\operatorname{diag}(\beta_{[t]})\rmK_{[t]}$；带门控的 $\overleftarrow{\rmW_{[t]}}$ 是它按行乘上块内累积衰减：$\overleftarrow{\rmW_{[t]}} = \operatorname{diag}(\gamma_{[t]})\rmW_{[t]}$。
6. 带 $\gamma$ 比值的写入部分把 $\Gamma_{[t]}$ 并进下三角来解：$\widetilde{\rmU_{[t]}} = \big[\rmI + \operatorname{strictLower}\big(\operatorname{diag}(\beta_{[t]})(\Gamma_{[t]}\odot\rmK_{[t]}\rmK_{[t]}^\top)\big)\big]^{-1}\operatorname{diag}(\beta_{[t]})\rmV_{[t]}$。
7. 代回得到一块的状态与输出：$\rmS_{[t+1]} = \gamma_{[t]}^C\rmS_{[t]} + (\widetilde{\rmU_{[t]}} - \overleftarrow{\rmW_{[t]}}\rmS_{[t]}^\top)^\top\overrightarrow{\rmK_{[t]}}$，$\overrightarrow{\rmK_{[t]}}$ 是把块内键按行乘上 $\gamma_{[t]}^C/\gamma_{[t]}^r$ 的结果；$\gamma_{[t]}^C$ 是块内总衰减。输出由衰减后的块首贡献加块内贡献给出，全部是矩阵乘。
依据：paper/src-v3/main.tex Algorithm: Hardware-efficient Chunkwise training
:::

#### 扩展 WY 表示

块内连乘里多了门控 $\alpha$，原来的 WY 表示要跟着改。把每一步多乘的 $\alpha$ 并进去，状态写成带累积衰减 $\gamma$ 的求和：$\rmS_t = \sum_{i=1}^t \frac{\gamma_t}{\gamma_i}\vu_i\vk_i^\top$，其中 $\vu_i = \beta_i\big(\vv_i - \sum_{j=1}^{i-1}\frac{\gamma_i}{\gamma_j}\vu_j\vk_j^\top\vk_i\big)$，比原来多出每个旧项的 $\gamma$ 比值。可以证明这个形式和 gated delta 规则逐步展开的结果完全一致。
依据：paper/src-v3/main.tex Extended WY representation for gated delta rule

::: 带门控的 WY 表示的归纳证明
1. 要证的是 $\rmS_t = \sum_{i=1}^t \frac{\gamma_t}{\gamma_i}\vu_i\vk_i^\top$，其中 $\vu_t = \beta_t\big(\vv_t - \sum_{i=1}^{t-1}\frac{\gamma_t}{\gamma_i}\vu_i\vk_i^\top\vk_t\big)$。
2. 起点 $t=1$：递推给出 $\rmS_1 = \beta_1\vv_1\vk_1^\top$，右边只剩 $i=1$ 项 $\frac{\gamma_1}{\gamma_1}\vu_1\vk_1^\top = \beta_1\vv_1\vk_1^\top$，两边相等。
3. 归纳步：把 $\rmS_{t+1} = \rmS_t(\alpha_{t+1}(\rmI - \beta_{t+1}\vk_{t+1}\vk_{t+1}^\top)) + \beta_{t+1}\vv_{t+1}\vk_{t+1}^\top$ 代入归纳假设，第一项给每个已有的 $\frac{\gamma_t}{\gamma_i}$ 再乘一个 $\alpha_{t+1}$，变成 $\frac{\gamma_{t+1}}{\gamma_i}$。
4. 把含 $\vk_{t+1}$ 的项并起来，剩下的系数正好是 $\vu_{t+1}$ 的定义，于是右边写成 $\sum_{i=1}^{t+1}\frac{\gamma_{t+1}}{\gamma_i}\vu_i\vk_i^\top$，与左边一致。
依据：paper/src-v3/main.tex Extended WY representation for gated delta rule
:::

### 网络里的 Gated DeltaNet

这条规则放进网络时，还要把每个位置的键、值、查询和两个标量算出来。Gated DeltaNet 的层沿用常规的 Transformer 解码器结构：每一层有一个在位置之间传信息的子层和一个逐位置独立计算的前馈层。它把前者（自注意力：每个位置用查询和前面所有位置的键做点积，再按点积大小对对应的值加权取回）换成这条规则的输出，其余部分不变。

隐藏状态（每一层传给下一层的向量序列，每个位置一个向量）先经线性投影得到查询、键、值，再各过一层沿序列方向的短卷积（只看当前位置和它前面少数几个位置的卷积）和 SiLU 激活（一种平滑的非线性，$\mathrm{SiLU}(x) = x\cdot\sigma(x)$，$\sigma$ 是 sigmoid 函数 $\sigma(x) = 1/(1+e^{-x})$）；查询和键额外做 L2 归一化，把它们的长度定成 1。门控 $\alpha$ 和写入强度 $\beta$ 由隐藏状态算出，是取值落在 $(0,1)$ 的标量。规则算出的输出经过归一化，再乘一个由隐藏状态算出的输出门，最后过一个输出投影回到原来的维度。每个头各有一张自己的状态矩阵，形状是值维乘键维（$d_v \times d_k$），与序列长度无关。
依据：paper/src-v3/main.tex Token mixer block.；paper/src-v3/figures/model.tex L409；repo/GatedDeltaNet/lit_gpt/gated_delta_net.py L217-L225

::: 块设计与组件消融
依据：paper/src-v3/figures/model.tex L409

<figure>
<svg viewBox="0 0 780 150" role="img" aria-label="Gated DeltaNet 的层结构">
<defs><marker id="gd-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#333"/></marker></defs>
<rect x="4" y="58" width="60" height="36" rx="5" fill="#eef1ff" stroke="#333"/>
<foreignObject x="4" y="58" width="60" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1e293b">输入</div></foreignObject>
<rect x="82" y="26" width="104" height="36" rx="5" fill="#eef1ff" stroke="#333"/>
<foreignObject x="82" y="26" width="104" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">线性投影 $\vq\vk\vv$</div></foreignObject>
<rect x="82" y="92" width="104" height="36" rx="5" fill="#eef1ff" stroke="#333"/>
<foreignObject x="82" y="92" width="104" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">线性投影 $\alpha\beta$</div></foreignObject>
<rect x="204" y="26" width="112" height="36" rx="5" fill="#e8f7ee" stroke="#333"/>
<foreignObject x="204" y="26" width="112" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">短卷积 + SiLU</div></foreignObject>
<rect x="204" y="92" width="112" height="36" rx="5" fill="#e8f7ee" stroke="#333"/>
<foreignObject x="204" y="92" width="112" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">取值在 (0,1) 的标量</div></foreignObject>
<rect x="334" y="54" width="130" height="44" rx="6" fill="#fdf3d8" stroke="#333"/>
<foreignObject x="334" y="54" width="130" height="44"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">gated delta rule</div></foreignObject>
<rect x="482" y="58" width="76" height="36" rx="5" fill="#eef1ff" stroke="#333"/>
<foreignObject x="482" y="58" width="76" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1e293b">归一化</div></foreignObject>
<rect x="576" y="58" width="92" height="36" rx="5" fill="#e8f7ee" stroke="#333"/>
<foreignObject x="576" y="58" width="92" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;text-align:center;color:#1e293b">输出门 $\otimes$</div></foreignObject>
<rect x="686" y="58" width="88" height="36" rx="5" fill="#eef1ff" stroke="#333"/>
<foreignObject x="686" y="58" width="88" height="36"><div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1e293b">输出投影</div></foreignObject>
<path d="M64 76 H82 M186 44 H204 M186 110 H204 M316 44 Q327 44 330 58 M316 110 Q327 110 330 94 M464 76 H482 M558 76 H576 M668 76 H686" stroke="#333" fill="none" marker-end="url(#gd-arrow)"/>
</svg>
<figcaption>查询、键、值经线性投影与短卷积进入 gated delta rule，$\alpha$、$\beta$ 经线性投影进入；输出经归一化、输出门与输出投影返回。</figcaption>
</figure>
依据：自行构造

Gated DeltaNet 在 400M 参数、15B token（文本被切分成的单位）上做组件消融（把层里的某个部分去掉或换掉，再看效果），平均困惑度（衡量语言模型预测下一个 token 的指标，越低越好）越低越好、平均准确率越高越好：

| 配置 | 平均困惑度 | 平均准确率 |
|---|---:|---:|
| Gated DeltaNet（头维 128） | 27.35 | 47.26 |
| 换成纯 delta 规则 | 30.87 | 45.12 |
| 去掉短卷积 | 28.95 | 46.16 |
| 去掉输出门 | 29.12 | 45.46 |
| 去掉输出归一化 | 27.55 | 47.07 |
| 头维改 64 | 28.31 | 46.35 |
| 头维改 256 | 27.13 | 47.38 |

在去掉组件、改头维的这几档里，短卷积和输出门去掉后变差最多，输出归一化的影响最小；门控加 delta 这一对优于纯 delta 规则；头维 128 处在 64 和 256 之间、接近最好。
依据：paper/src-v3/main.tex Ablation Study；paper/src-v3/main.tex L1911-L1946
:::

### 适用条件与边界

状态大小固定是这个概念的核心限制：序列很长时能可靠存下的关联数有限，需要在一段长文里精确取出大量互不相关信息的任务会不够用。换来的是每步更新的计算量与运行时（模型跑一次前向）要保存的状态都与序列长度无关。

在单针检索（在一段长文里藏进一处信息，再提问把它取出来）上，只有整体衰减时，长距离的关联被过早丢掉；只有 delta 精确改写时，需要存放的关联一多，既区分不出某一条、也删不掉它；两类机制合起来时，这两种情况都能应对。

训练长度是 4K，外推到 20K 时，它在多种长文任务上的困惑度（衡量语言模型预测下一个单位的误差，越低越好）整体较低，但各任务的趋势并不一致，不是每个任务都随长度单调变好。

最容易与它混淆的是 Mamba2（另一种只用整体衰减的线性注意力模型）：两者的状态递推只差一步，Mamba2 缺了 Gated DeltaNet 对单条关联的精确改写，也因此少一个写入强度参数。
依据：paper/src-v3/main.tex Introduction；paper/src-v3/main.tex Mamba2: Linear Attention with decay；paper/src-v3/main.tex Gated Delta Networks and Hybrid Models；paper/src-v3/main.tex Case study: Single Needle in a Haystack (S-NIAH)；paper/src-v3/main.tex Length extrapolation on long sequences；paper/src-v3/main.tex Setup

::: 单针检索上的成绩
1.3B 模型、零样本（不做任务相关的微调，直接拿预训练模型考），单针检索三类任务（S-NIAH-1 取回一段口令、S-NIAH-2 取回一个数字、S-NIAH-3 取回一个更长的串）在不同上下文长度上的准确率：

| 任务与长度 | 只有衰减 | 只有 delta | Gated DeltaNet |
|---|---:|---:|---:|
| S-NIAH-1，4K | 65.4 | 99.0 | 91.4 |
| S-NIAH-1，8K | 30.4 | 98.8 | 91.8 |
| S-NIAH-2，2K | 98.8 | 45.6 | 99.8 |
| S-NIAH-2，8K | 17.0 | 14.4 | 29.6 |
| S-NIAH-3，4K | 4.6 | 22.4 | 27.6 |

S-NIAH-1 里只有衰减的一档随长度掉得最快；S-NIAH-2 里只有 delta 的一档在长序列上也掉下来；S-NIAH-3 在 4K 上只有衰减的一档最低。三类任务合起来看 Gated DeltaNet 更稳。
依据：paper/src-v3/main.tex L410-L430
:::
