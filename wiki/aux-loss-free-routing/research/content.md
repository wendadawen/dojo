# 辅助损失无关路由

- 描述：辅助损失无关路由是混合专家（MoE）训练里的一种负载均衡做法：给每个专家加一个偏置，只加到「选哪些专家」的判定上、不加到门控权重，再按专家负载用固定小步更新这个偏置，因此不用额外写辅助损失项（鼓励均衡的额外损失项）。
- 摘要：把专家偏置 $b_i$ 加到路由分数上，只用 $s_{i,t}+b_i$ 决定 top-k 选谁，门控值仍取原始 $s_{i,t}$；每步训练后按 $b_i \leftarrow b_i + \gamma\,\mathrm{sign}(\bar c_i - c_i)$ 更新偏置，过载减、欠载增，不产生干扰梯度。
- 主题：训练与优化, 模型结构
- 标签：MoE

## 资料

- Lean Wang, Huazuo Gao, Chenggang Zhao 等. Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv:2408.15664v1, 2024. [https://arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664)
- DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437v2, 2025. [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

## 短页

### 一个只进选择的偏置

混合专家（[MoE](wiki/deepseek-moe)）层里有一排结构相同的小网络，每个叫一个专家；每个 token（文本切分后的基本单位）只选其中几个来算。每个 token 选哪几个专家，由路由分数（token 与专家之间的一个打分）决定。辅助损失无关路由给每个专家加一个偏置：选专家时，看的是每个专家的路由分数加上它自己的偏置，取最高的几个；但被选中之后，这个专家在输出里占多少权重，仍按它不含偏置的原始分数算，这个相乘用的权重叫门控权重。它在训练中随专家负载被逐步调整。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex with Auxiliary-Loss-Free Load Balancing

### 不写损失项也能均衡

MoE 训练要防负载失衡。不加约束，路由会把多数 token 分给少数专家，其余专家得不到足够训练；专家分散到多块设备上并行计算时，负载不均还会让设备忙闲不均、拖慢计算。常见对策是在损失函数里加一项辅助损失来鼓励均衡，但这一项算出的梯度会掺进主任务的梯度里，干扰语言建模的目标（让模型预测下一个 token；训练按损失算出的梯度更新参数，这一步叫反向传播）。辅助损失无关路由不走损失项这条路：它把均衡交给一个按规则更新、不进反向传播的偏置，因此不产生这种干扰梯度。
依据：paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-lfb/iclr2024_conference.tex The Dilemma Between Load Balance and Model Performance

### 加偏置选专家，按负载调偏置

选择这一步：给每个专家一个偏置 $b_i$，选专家看的是 $s_{i,t}+b_i$，其中 $s_{i,t}$ 是该 token 对第 $i$ 个专家的路由分数，取最高的 $K_r$ 个（$K_r$ 是每个 token 选几个专家）。被选中的专家，其门控权重取原始分数 $s_{i,t}$，没被选中的取 0，再对这些门控权重归一化（除以被选中专家门控权重之和），$b_i$ 不参与归一化。更新这一步：每个训练步结束后，统计每个专家在这一步分到的 token 数 $c_i$ 与平均负载 $\bar c_i$；$c_i$ 高于平均（过载）就把它的偏置减一个固定小步 $\gamma$，低于平均（欠载）就加 $\gamma$。偏置初始为 0，训练中反复执行这两步，负载被逐步拉向均衡。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.；paper/src-dsv3/main.tex Basic Architecture of 

## 长页

大语言模型（用大量文本训练、能按上文续写的模型）的主干是一叠结构相同的网络层，这类层叫 Transformer 层。每层里有一块对所有位置各算一次的前馈网络（前馈层），作用是把每个位置的向量做一次变换；其中若干前馈层常被换成混合专家（[MoE](wiki/deepseek-moe)）层。MoE 层里有多个结构相同的小网络，每个叫一个专家；对每个 token（文本切分后的基本单位），路由分数给每个专家打一个分，只选分最高的几个专家来算，其余不参与。这样一个 token 只用到一部分参数，参数总量可以远大于每个 token 实际参与计算的量。
依据：paper/src-lfb/iclr2024_conference.tex Mixture-of-Experts；paper/src-dsv3/main.tex Basic Architecture of ；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance

不加约束时，路由会持续把多数 token 分给少数专家，被选中的专家得到更多训练，其余专家得不到足够训练，这种失效叫路由坍塌（routing collapse）。辅助损失无关路由把均衡交给一个偏置：给每个专家加一个偏置，只用它调整「选哪些专家」的判定，不加到门控权重，再按专家负载逐步更新这个偏置；偏置不进反向传播（按损失算出的梯度回传并据此更新参数，这一步叫反向传播，损失是衡量模型输出与目标差距、被最小化的函数），因此不给主任务添额外梯度。
依据：paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-dsv3/main.tex with Auxiliary-Loss-Free Load Balancing

### 为什么要绕开辅助损失

负载失衡的另一重代价来自专家并行——把不同专家放到不同设备上并行计算，叫专家并行；负载不均会让一部分设备排队、另一部分空闲，拖慢整体计算。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex L477-L478

常见的对策是在损失函数里加一项辅助损失，推动 token 均匀分到各专家。对长度为 $T$ 的序列，它写成 $L_{bal}=\alpha\sum_{i=1}^{N}f_i P_i$：$N$ 是路由专家总数（辅助损失只作用在路由专家上），$f_i$ 是分给专家 $i$ 的 token 比例，$P_i$ 是专家 $i$ 的平均路由分数，$\alpha$ 是控制这项强弱的权重。$f_i$ 与 $P_i$ 都随训练变化，$\alpha$ 越大，均衡被推得越紧。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance

问题在于这是一项额外的损失项，它的梯度会掺进主任务的梯度里，使更新方向偏离语言建模目标（让模型预测下一个 token），这种多余的梯度叫干扰梯度。$\alpha$ 因此两难：取小了不足以抑制失衡，取大了会损害模型性能。
依据：paper/src-lfb/iclr2024_conference.tex Introduction；paper/src-lfb/iclr2024_conference.tex Auxiliary Loss for Load Balance；paper/src-lfb/iclr2024_conference.tex The Dilemma Between Load Balance and Model Performance

### 偏置只决定选哪些专家，不改门控权重

第 $t$ 个 token 送进 MoE 层前是一个向量 $\mathbf{u}_t$；每个专家 $i$ 有一个同样维度的可学习向量 $\mathbf{e}_i$，叫中心向量。路由分数取两者的内积再过一个 sigmoid，$s_{i,t}=\sigma(\mathbf{u}_t^{\top}\mathbf{e}_i)$，$\sigma$ 把任意实数压到 0 到 1 之间。
依据：paper/src-dsv3/main.tex Basic Architecture of ；paper/src-lfb/iclr2024_conference.tex Mixture-of-Experts

带偏置的判定是：给每个专家一个偏置 $b_i$，选专家时看的不是 $s_{i,t}$ 而是 $s_{i,t}+b_i$。若 $s_{i,t}+b_i$ 落在所有 $\{s_{j,t}+b_j\}$ 的前 $K_r$ 名（$K_r$ 是每个 token 选几个路由专家），专家 $i$ 被选中，此时它的门控值取原始分数 $s_{i,t}$；没被选中则取 0。把未归一化的门控值记作 $g'_{i,t}$，被选中时为 $s_{i,t}$，否则为 0。
依据：paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.；paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy

门控值还要归一化：$g_{i,t}=g'_{i,t}/\sum_{j=1}^{N_r}g'_{j,t}$，分母只对被选中的专家求和（其余 $g'_{j,t}$ 为 0），$N_r$ 是路由专家总数；分母只含路由分数，不含偏置 $b_i$。MoE 层的输出把共享专家（始终激活、不参与路由的专家）与路由专家两部分加权相加，再加回输入：$\mathbf{h}'_t=\mathbf{u}_t+\sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}(\mathbf{u}_t)+\sum_{i=1}^{N_r}g_{i,t}\mathrm{FFN}_i^{(r)}(\mathbf{u}_t)$，$N_s$ 是共享专家数，$\mathrm{FFN}^{(s)}$、$\mathrm{FFN}^{(r)}$ 分别是共享、路由专家的前馈网络。
依据：paper/src-dsv3/main.tex Basic Architecture of ；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.

::: 为什么只有偏置的相对差起作用
1. 选专家时比较的是每个专家的 $s_{j,t}+b_j$。
依据：自行构造
2. 把每个 $b_j$ 同时加上同一个常数 $c$，每个分数都变成 $s_{j,t}+b_j+c$。
依据：自行构造
3. 所有分数同加一个数，大小排序不变，前 $K_r$ 名仍是同一批专家，选中的专家不变。
依据：自行构造
4. 所以真正起作用的是偏置之间的相对差，全体偏置一起平移没有效果。
依据：自行构造
:::

因为 $b_i$ 不出现在任何乘进输出的项里，它不进入计算图（训练时记录每一步运算、供反向传播求梯度的结构），反向传播不会为它算梯度；它的更新在每步训练之外单独进行。这就是「辅助损失无关」的来处：不写任何辅助损失项，也就不产生干扰梯度。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-lfb/iclr2024_conference.tex Introduction

### 按负载偏差的符号更新偏置

每个训练步里统计每个专家被分到多少 token：记 $c_i$ 为专家 $i$ 在这一步的整批数据上分到的 token 数，$\bar c_i=\frac{1}{N_r}\sum_{j=1}^{N_r}c_j$ 是所有路由专家负载的平均值。两者之差 $\bar c_i-c_i$ 是负载偏差：为正表示专家 $i$ 分到的 token 少于平均（欠载），为负表示多于平均（过载）。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-lfb/iclr2024_conference.tex L239-L258；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.

偏置按一条固定规则更新：$b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(\bar c_i-c_i)$。$\gamma$ 是步长，叫 bias update speed；$\mathrm{sign}$ 取符号，正数为 $+1$、负数为 $-1$、零为 $0$。欠载的专家偏差为正，$b_i$ 增加 $\gamma$，下一次它的 $s_{i,t}+b_i$ 更容易排进前 $K_r$；过载的专家偏差为负，$b_i$ 减少 $\gamma$，更不容易被选中；负载恰好等于平均则不变。偏置在训练开始时初始化为 0；原始资料的实验取 $\gamma=0.001$。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-lfb/iclr2024_conference.tex L239-L258；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.；paper/src-lfb/iclr2024_conference.tex Update rate.

偏置不通过反向传播学习：梯度下降（按梯度更新参数的那套方法）看不到它，每步训练结束后单独按规则更新一次。用 $\mathrm{sign}$ 而不是按偏差幅度更新，是因为把更新式换成 $b_i\leftarrow b_i+\gamma(\bar c_i-c_i)$ 的连续版本后，偏差大的专家偏置变化更快、负载更均衡一点，但模型性能没有变好，所以保留了更简单的 $\mathrm{sign}$ 版本。$\mathrm{sign}$ 只保留方向、丢掉幅度：差 1 个 token 和差很多个 token，走的步长一样。
依据：paper/src-lfb/iclr2024_conference.tex Update rule.；paper/src-lfb/iclr2024_conference.tex L239-L258

更新用的是历史数据（已经算过的部分）的负载，而不是当前序列里还没算到的 token 的信息。若拿当前序列的负载去调偏置，序列里靠前的 token 就用到了靠后 token 的信息，破坏语言模型的因果约束、泄露未来的 token。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy

::: 符号更新与按幅度更新的对照
把更新式里偏差的符号 $\mathrm{sign}(\bar c_i-c_i)$ 换成偏差本身 $\bar c_i-c_i$，在 1B（10 亿）参数模型上的验证困惑度（perplexity，衡量模型预测下一个 token 的确信程度、越低越好）与全局最大偏差 MaxVio（越低越均衡）如下。连续版本的 MaxVio 更低（更均衡），但困惑度没有更低。
依据：paper/src-lfb/iclr2024_conference.tex Update rule.；paper/src-lfb/iclr2024_conference.tex Experimental Setups

| 更新式 | 验证困惑度 | MaxVio |
|---|---:|---:|
| $b_i=b_i+\gamma\,\mathrm{sign}(\bar c_i-c_i)$，$\gamma=0.001$ | 9.50 | 0.044 |
| $b_i=b_i+\gamma\,(\bar c_i-c_i)$，$\gamma=0.001$ | 9.51 | 0.036 |
| $b_i=b_i+\gamma\,(\bar c_i-c_i)$，$\gamma=0.01$ | 9.53 | 0.028 |
依据：paper/src-lfb/iclr2024_conference.tex Update rule.
:::

#### 手算一个 4 专家选 2 的训练步

设 4 个路由专家（编号 0 到 3）、每个 token 选 2 个、一批 4 个 token，初始偏置全为 0；为便于手算，把步长取为 $\gamma=0.1$（实际用 $0.001$）。
依据：自行构造

::: 两轮选择的完整过程
第一轮 4 个 token 对 4 个专家的路由分数 $s_{i,t}$ 如下（这一步偏置全为 0，$s_{i,t}+b_i=s_{i,t}$）。
依据：自行构造

| token | 专家 0 | 专家 1 | 专家 2 | 专家 3 |
|---|---:|---:|---:|---:|
| $t_1$ | 0.55 | 0.30 | 0.50 | 0.10 |
| $t_2$ | 0.52 | 0.48 | 0.20 | 0.15 |
| $t_3$ | 0.35 | 0.10 | 0.20 | 0.30 |
| $t_4$ | 0.60 | 0.45 | 0.25 | 0.20 |

第一轮按每行取最大的 2 个：
依据：自行构造

1. $t_1$ 选专家 0（0.55）和专家 2（0.50）。
2. $t_2$ 选专家 0（0.52）和专家 1（0.48）。
3. $t_3$ 选专家 0（0.35）和专家 3（0.30）。
4. $t_4$ 选专家 0（0.60）和专家 1（0.45）。
依据：自行构造

第一轮各专家被选中次数 $c=(4,2,1,1)$，平均 $\bar c_i=(4+2+1+1)/4=2$。负载偏差 $d_i=\bar c_i-c_i$ 为 $(2-4,\,2-2,\,2-1,\,2-1)=(-2,0,1,1)$，符号为 $(-1,0,1,1)$；乘以 $\gamma=0.1$ 后偏置更新为 $b=(-0.1,\,0,\,0.1,\,0.1)$，专家 0 的偏置降、专家 2 和 3 的偏置升。
依据：自行构造

第二轮用新偏置，比较 $s_{i,t}+b_i$：
依据：自行构造

1. $t_1$：$(0.45,0.30,0.60,0.20)$，选专家 2（0.60）和专家 0（0.45）。
2. $t_2$：$(0.42,0.48,0.30,0.25)$，选专家 1（0.48）和专家 0（0.42）。
3. $t_3$：$(0.25,0.10,0.30,0.40)$，选专家 3（0.40）和专家 2（0.30），专家 0 落选。
4. $t_4$：$(0.50,0.45,0.35,0.30)$，选专家 0（0.50）和专家 1（0.45）。
依据：自行构造

第二轮各专家被选中次数变成 $c=(3,2,2,1)$：专家 0 从 4 降到 3，专家 2 从 1 升到 2，其余不变，负载更均衡了。
依据：自行构造

```python
s = [[0.55, 0.30, 0.50, 0.10],
     [0.52, 0.48, 0.20, 0.15],
     [0.35, 0.10, 0.20, 0.30],
     [0.60, 0.45, 0.25, 0.20]]
n, topk, gamma = 4, 2, 0.1
b = [0.0] * n

def choose(scores):
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:topk]

def counts(bias):
    c = [0] * n
    for row in s:
        for i in choose([row[j] + bias[j] for j in range(n)]):
            c[i] += 1
    return c

c1 = counts(b)
mean = sum(c1) / n
b2 = [round(b[i] + gamma * (1 if mean > c1[i] else (-1 if mean < c1[i] else 0)), 4)
      for i in range(n)]
c2 = counts(b2)
print("第 1 轮 c:", c1)
print("平均负载:", mean)
print("第 2 轮 b:", b2)
print("第 2 轮 c:", c2)
```

```text
第 1 轮 c: [4, 2, 1, 1]
平均负载: 2.0
第 2 轮 b: [-0.1, 0.0, 0.1, 0.1]
第 2 轮 c: [3, 2, 2, 1]
```
:::

### 适用条件与边界

这套方法只看每个训练步整批数据上的负载 $c_i$：它把批一级的负载拉平，不检查单个序列内部。可能出现一个序列里的 token 都选中同一组专家，而批一级仍然均衡。要限制单个序列内的极端失衡，需要另加一项序列级的均衡损失来补。
依据：paper/src-dsv3/main.tex Complementary Sequence-Wise Auxiliary Loss.；paper/src-dsv3/main.tex Batch-Wise Load Balance VS. Sequence-Wise Load Balance

$\gamma$ 取得过小，偏置收敛慢、早期负载均衡差；取得过大，偏置在训练后期来回波动，反而破坏后期的负载均衡。$\mathrm{sign}$ 只保留方向、每步走固定的一小步，因此要靠很多步慢慢逼近均衡。
依据：paper/src-lfb/iclr2024_conference.tex Update rate.；paper/src-lfb/iclr2024_conference.tex Update rule.

一个常见的误解，是把 $b_i$ 当成会进加权的偏置，以为被选中后门控值用的是 $s_{i,t}+b_i$；实际上门控值与归一化分母都用原始 $s_{i,t}$。
依据：paper/src-lfb/iclr2024_conference.tex Auxiliary-Loss-Free Load Balancing Strategy；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing.
