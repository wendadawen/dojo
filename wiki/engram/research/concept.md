# Engram

- 描述：Engram 是插在 Transformer 少数几层前的条件记忆模块，把每个位置结尾的几个 token 按规范化文本等价折成规范 id，再对这些 id 组成的 n-gram 做哈希，取回静态嵌入表的若干行，由当前隐状态算出的标量门调制后写回残差流。
- 摘要：MoE 用条件计算稀疏地激活参数，Engram 用条件记忆稀疏地查表取回静态嵌入。它把位置 $t$ 结尾的 n-gram 里的 token 折成规范 id，用每个阶数配的多个哈希头映射到素数大小的桶，取回向量拼成 $\mathbf{e}_t$，再用当前隐状态 $\mathbf{h}_t$ 与记忆投影出的键做归一化点积、过 $\sigma$ 函数得到标量门 $\alpha_t$，调制后的值经一个短卷积写回残差流。每 token 取回的行数是常数，表变大不增加每 token 浮点运算量。
- 主题：模型结构
- 标签：模型架构

## 资料

- Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai 等（北京大学、DeepSeek-AI）. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372v2 [cs.CL]，2026-07-12. [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)
  - 摘录（§2.2 词表压缩）：To maximize semantic density, we implement a vocabulary projection layer. Specifically, we pre-compute a surjective function $\mathcal{P}:V\to V^{\prime}$ that collapses raw token IDs into canonical identifiers based on normalized textual equivalence (using NFKC (Whistler, 2025), lowercasing, etc.). In practice, this process achieves a 23% reduction in the effective vocabulary size for a 128k tokenizer (see Appendix C).
  - 摘录（§2.2 哈希）：Directly parameterizing the combinatorial space of all possible $N$ -grams is intractable. Following Tito Svenstrup et al. (2017), we adopt a hashing-based approach. To mitigate collisions, we employ $K$ distinct hash heads for each $N$ -gram order $n$ . Each head $k$ maps the compressed context to an index within an embedding table $\mathbf{E}_{n,k}$ (of prime size $M_{n,k}$ ) via a deterministic function $\varphi_{n,k}$ :
  - 摘录（§2.2 实现）：In practice, $\varphi_{n,k}$ is implemented as a lightweight multiplicative-XOR hash.
  - 摘录（§4.2 U 型）：Our experiments uncover a distinct U-shaped scaling law, revealing that even simple lookup mechanisms, when treated as a first-class modeling primitive, act as essential complements to neural computation.
  - 摘录（§4.3 配置与卷积初始化）：Engram-27B is strictly derived from the MoE-27B architecture to ensure fair comparison. We reduce the number of routed experts from 72 to 55 and reallocate the freed parameters to a 5.7B-parameter embedding module ( $\rho=74.3\%$ ), keeping the total model size constant at 26.7B.
  - 摘录（§4.3 优化）：For optimization, the embedding parameters are updated using Adam (Kingma, 2014) with a learning rate scaled by $5\times$ and no weight decay, while the convolution parameters are initialized to zero to strictly preserve the identity mapping at the start of training.
  - 摘录（§6.2 层敏感性）：Sweeping the insertion depth of a single Engram module confirms that early injection (Layer 2) is optimal, whereas efficacy degrades in deeper layers.
  - 摘录（§6.2 单层最优）：The sweep shows that Layer 2 achieves the best single-layer performance (Val Loss $=1.770$ ), outperforming Layer 1 and
  - 摘录（§6.4 推理抑制）：Factual knowledge benchmarks suffer a catastrophic collapse, retaining only 29–44% of the original performance (e.g., TriviaQA at 29%), confirming that the Engram module acts as the primary repository for parametric knowledge. Conversely, reading comprehension tasks are remarkably resilient, retaining 81–93% (e.g., C3 at 93%), suggesting that context-grounded tasks rely primarily on the backbone’s attention mechanism rather than Engram.
  - 摘录（§7 吞吐设置）：We insert a massive 100B-parameter Engram layer into the second Transformer block, with the entire embedding table resident in host DRAM.
  - 摘录（§6.4 吞吐结论）：As detailed in Table 4, offloading a 100B-parameter embedding table incurs a negligible throughput penalty, peaking at only $2.8\%$ on the 8B backbone.
  - 摘录（§6.4 通信量）：Crucially, the effective communication volume per step scales with the number of activated slots rather than the total embedding table size.

## 短页

### 是什么

Engram 是一个条件记忆模块，插在 Transformer 少数几层的前面。它给模型补上一条查表通路：把每个位置结尾的一小段上下文当作键，用哈希在一条大嵌入表里查出若干行，再把查回的内容写回主干。表只被查，不参与每个 token 的浮点运算。

### 为什么需要

命名实体、固定搭配这类知识是局部的、静态的，本来查一次表就能取回；而标准 Transformer 没有原生的查找原语，只能靠若干层注意力和前馈网络把它们重新拼出来。Engram 把这类工作接过去，于是稀疏参数分成两路：一路是 [MoE](wiki/deepseek-moe) 的专家，稀疏地激活来做动态计算；一路是 Engram 的表，稀疏地查表来取回静态记忆。

### 大概怎么工作

每个位置分两段。检索段先把原始 token id 按规范化后的文本等价折叠成更少的规范 id，再对每个 [n-gram](wiki/ngram) 阶、每个哈希头算一个行号，从素数大小的桶里取回向量，拼成一个记忆向量。融合段用当前隐状态作查询、用这份记忆投影出的键做一次归一化点积，得到一个 $0$ 到 $1$ 的标量门，门控后的值再经过一个短的因果卷积写回主干；主干之后照常做注意力与 MoE。行号只由输入 token 决定，可以在对应的层执行之前就算出来。

## 长页

语言建模里的工作有两种质地。组合推理要一层层往前算；而命名实体、固定搭配这类知识是局部的、静态的、高度套话化的，本来查一次表就能取回。标准 Transformer 没有原生的查找原语，只能靠若干层注意力和前馈网络把它们重新拼出来：一份被论文转引的分析里，「Diana, Princess of Wales」这个实体要到第 6 层才拼装完整。这等于把本该留给高层推理的深度花在重建一张静态查找表上。

Engram 接的就是这类工作。它是一个插在 Transformer 少数几层前面的模块，每个位置只用它结尾的几个 token 去查一次表：查哪几行由 token 决定，与上下文无关；查回的内容要不要用、用多少，则由上下文决定。表只被查，不参与每个 token 的浮点运算，因此它与 [MoE](wiki/deepseek-moe) 是两条并列的稀疏轴：一条稀疏地激活参数做动态计算，一条稀疏地查表取回静态记忆。

模块在每个位置上分两段工作。检索段把原始 token id 折叠成更少的规范 id，再对每个 n-gram 阶、每个哈希头算一个行号，从素数大小的桶里取回向量拼成记忆向量。融合段用当前隐状态与这记忆投影出的键做一次归一化点积，得到一个 $0$ 到 $1$ 的标量门，门控后的值经过一个短因果卷积写回残差流；主干之后照常做注意力与 MoE。

### 一个 n-gram 怎么变成表里的行号

标准子分词器的目标是可逆还原，会给语义等价的写法分配互不相干的 id：`Apple`、` apple`、`APPLE` 各占一行。这些写法在语义上是同一个键，分开索引会白白耗掉表容量。Engram 先算一个把原始 id 归到规范 id 的投影 $\mathcal{P}$，规则是 NFKC 规范化、去掉重音、转小写、把连续空白折成一个空格。对 128k 的词表，这一步把有效词表压掉 23%。折叠之后，位置 $t$ 的 [n-gram](wiki/ngram) 是它结尾的一串规范 id $g_{t,n}=(x'_{t-n+1},\dots,x'_{t})$，阶数 $n$ 从 2 取到最大阶 $N$。

把所有可能的 n-gram 组合都参数化是不可行的：组合数按词表大小的 $n$ 次方增长。Engram 改用哈希，并且给每个阶数配 $K$ 个哈希头，让同一个上下文落到若干互不相干的行上，以此分散碰撞。第 $n$ 阶第 $k$ 个头把 $g_{t,n}$ 映射到一张自己的表 $\mathbf{E}_{n,k}$ 的行号 $z_{t,n,k}$，这张表的行数取素数。把所有阶数、所有头取回的向量拼接起来，就是这个位置的记忆向量 $\mathbf{e}_t$，长度是阶数个数乘头数再乘每头维度。

论文说这个哈希在实现里是轻量的乘法异或。它的映射是确定的：行号只由以该位置结尾的那几个 token 决定，同一个片段在任何位置都落到同一行。

::: 检索段的公式与符号

$$z_{t,n,k}\triangleq\varphi_{n,k}(g_{t,n}),\qquad \mathbf{e}_{t,n,k}=\mathbf{E}_{n,k}[z_{t,n,k}]$$

- $\varphi_{n,k}$：第 $n$ 阶第 $k$ 个头的哈希函数。
- $z_{t,n,k}$：算出的行号，取值在 $0$ 到 $M_{n,k}-1$ 之间。
- $M_{n,k}$：表 $\mathbf{E}_{n,k}$ 的行数，取素数。
- $\mathbf{e}_{t,n,k}$：取回的那一行向量。

$$\mathbf{e}_{t}\triangleq\mathop{\|}_{n=2}^{N}\mathop{\|}_{k=1}^{K}\mathbf{e}_{t,n,k}$$

- $\|$：向量拼接。
- $K$：每个阶数的哈希头数。
- $\mathbf{e}_t$：位置 $t$ 的记忆向量。

:::

::: 例子的完整过程：四个 token 的行号

取压缩后的 id 序列 $(7,3,9,5)$，最大阶数 3，于是要算 2 元和 3 元两种。乘子取 $(3,5,7)$，分别对应回看 0、1、2 个位置；2 元的两个头桶大小取 11 和 13，3 元的两个头取 17 和 19。越出序列开头的位置用填充 id 2 补。

```python
PAD = 2
MULT = (3, 5, 7)                       # 回看 0、1、2 个位置各一个乘子
BUCKETS = {2: (11, 13), 3: (17, 19)}   # 2 元与 3 元各两个头的桶大小（素数）
IDS = (7, 3, 9, 5)                     # 压缩后的 token id 序列，按时间顺序排列

def mix(t, order):
    value = 0
    for k in range(order):
        j = t - k
        token = IDS[j] if j >= 0 else PAD   # 越出序列开头就用填充 id
        product = token * MULT[k]
        value = product if k == 0 else value ^ product
    return value

for t in range(len(IDS)):
    v2, v3 = mix(t, 2), mix(t, 3)
    print(f"位置 {t}：2 元混合值 {v2} 行号 {[v2 % m for m in BUCKETS[2]]}"
          f"｜3 元混合值 {v3} 行号 {[v3 % m for m in BUCKETS[3]]}")
```

```text
位置 0：2 元混合值 31 行号 [9, 5]｜3 元混合值 17 行号 [0, 17]
位置 1：2 元混合值 42 行号 [9, 3]｜3 元混合值 36 行号 [2, 17]
位置 2：2 元混合值 20 行号 [9, 7]｜3 元混合值 37 行号 [3, 18]
位置 3：2 元混合值 34 行号 [1, 8]｜3 元混合值 55 行号 [4, 17]
```

以位置 3 为例。2 元只看位置 3 和它前一个位置：$5\times3=15$，$9\times5=45$，$15\oplus45=34$；对 11 取模得 1，对 13 取模得 8。3 元再异或 $3\times7=21$：$34\oplus21=55$；对 17 取模得 4，对 19 取模得 17。位置 0 没有更早的 token：算 2 元时缺一个回看位置，算 3 元时缺两个，都用填充 id 2 补上。

:::

### 取回的静态向量怎么被当前上下文接受

取回的嵌入只由输入 token 决定，与上下文无关，是一份静态先验；它可能对不上当前的语境，哈希碰撞与一词多义也会带进噪声。融合段要回答的是：这份记忆此刻该不该用、用多少。

Engram 借用了[自注意力](wiki/standard-attention)的形式：当前隐状态 $\mathbf{h}_t$ 已经通过前面的注意力层聚合了上下文，用它作查询；取回的记忆向量 $\mathbf{e}_t$ 投影出键 $\mathbf{k}_t=\mathbf{W}_K\mathbf{e}_t$ 与值 $\mathbf{v}_t=\mathbf{W}_V\mathbf{e}_t$。门是一个标量：

$$\alpha_{t}=\sigma\!\left(\frac{\mathrm{RMSNorm}(\mathbf{h}_{t})^{\top}\mathrm{RMSNorm}(\mathbf{k}_{t})}{\sqrt{d}}\right)$$

- $\mathbf{h}_t$：位置 $t$ 的当前隐状态。
- $\mathrm{RMSNorm}$：[RMSNorm](wiki/rmsnorm)，在算点积之前分别对查询和键做一次，论文给的理由是保证梯度稳定。
- $\mathbf{k}_t$：记忆投影出的键向量。
- $d$：隐状态维度；除以 $\sqrt{d}$ 是注意力里的那步缩放。
- $\sigma$：sigmoid 函数，把结果压进 $0$ 到 $1$。
- $\alpha_t$：标量门。

门控后的值是 $\alpha_t\mathbf{v}_t$。这个门做的是一次对齐检查：取回的记忆与当前上下文矛盾时，点积变小，门趋近 $0$，这份噪声被压下去。它不是开关：点积为 $0$ 时门仍给出 $0.5$，只把这份值减半。

门控之后接一个短的[深度卷积](wiki/depthwise-conv)。论文给的理由是扩大感受野、增加非线性；卷积是因果的，每个位置只看自己和更早的位置。卷积的参数初始化为零，因此训练刚开始时这一支路是恒等映射，它的输出等于输入，而这份输入仍然写回主干。

最后通过[残差连接](wiki/residual-connection)写回主干：$\mathbf{H}^{(\ell)}\leftarrow\mathbf{H}^{(\ell)}+\mathbf{Y}$，写完才轮到这一层常规的注意力与 MoE。

#### 多分支主干上的改法

论文的主干把残差流加宽成 $M$ 条并行分支、用可学习的连接权重调制信息流，也就是[超连接](wiki/hyper-connections)。把 Engram 接到这种主干上要选一处参数共享：嵌入表与值投影矩阵在所有分支间共享，键投影矩阵每条分支各一份，每条分支用自己的门去乘同一份共享的值向量。这样一份值投影加 $M$ 份键投影可以融合成一个稠密的矩阵乘。

::: 融合段的其余公式

卷积与残差：

$$\mathbf{Y}=\mathrm{SiLU}\!\left(\mathrm{Conv1D}\!\left(\mathrm{RMSNorm}(\tilde{\mathbf{V}})\right)\right)+\tilde{\mathbf{V}}$$

- $\tilde{\mathbf{V}}$：整个序列上门控后的值，形状是「序列长度 $\times$ 维度」。
- $\mathrm{Conv1D}$：一维卷积，核宽 $w=4$，膨胀率取最大 n-gram 阶。
- $\mathrm{SiLU}$：一种平滑的激活函数。
- $\mathbf{Y}$：这一层的 Engram 输出。

多分支主干上第 $m$ 条分支的门：

$$\alpha_{t}^{(m)}=\sigma\!\left(\frac{\mathrm{RMSNorm}(\mathbf{h}_{t}^{(m)})^{\top}\mathrm{RMSNorm}(\mathbf{W}_{K}^{(m)}\mathbf{e}_{t})}{\sqrt{d}}\right)$$

- $\mathbf{h}_t^{(m)}$：第 $m$ 条分支在位置 $t$ 的隐状态。
- $\mathbf{W}_K^{(m)}$：第 $m$ 条分支自己的键投影矩阵。
- $\mathbf{v}_t$：各分支共享的值向量 $\mathbf{W}_V\mathbf{e}_t$。
- $\alpha_t^{(m)}$：这一条分支上的标量门。

:::

### 记忆插在哪一层，稀疏预算怎么分

Engram 不插在每一层。插在哪几层，与稀疏参数怎么分给记忆和专家，是两个分开定的问题。

插入层的位置由一对反向的要求定：插得早，模块能更早接手局部模式的重建，不必等主干花掉深度去拼装它们；插得晚，隐状态已经聚合了足够的全局上下文，门控判断更准，多分支的表示也才分化出可供逐分支调制的差别。逐层扫描的结果是第 2 层最好，越深越差；把同样的预算拆成两个较小的模块分别放置，比集中在一个模块上更好。论文的 27B 模型把 Engram 插在第 2、15 层。

组件消融从参考配置出发逐个去掉设计，影响最大的是三项：多分支主干上的分支独立融合、上下文感知门控、词表压缩。去掉深度卷积只有轻微退化。固定预算下把容量分给 4 元 n-gram 略差，论文的猜测是它摊薄了更常见的 2 元、3 元模式，同时不排除表更大时高阶 n-gram 会有用。

预算怎么分被写成一次分配：总参数量与每 token 浮点运算量都固定，把不激活的那部分参数按比例 $\rho$ 分给 MoE 专家，剩下的给 Engram。$\rho=1$ 就是纯 MoE。验证损失随 $\rho$ 呈 U 型，两端都更差，最优落在把约 20%–25% 的稀疏预算改分给 Engram 的位置：纯 MoE 缺少专管静态模式的记忆，几乎全给记忆则丢掉了条件计算的能力，记忆替代不了计算。另一个方向的结果是 MoE 的分配压到 $\rho\approx40\%$ 时，少掉一半以上的专家，性能仍与纯 MoE 基线相当。论文写明的适用范围是「在所考察的尺度内」。

表单独放大是另一条曲线：验证损失随槽位数在对数坐标下线性下降，即一条幂律。也就是说加大表持续有收益，而且不需要额外算力。

::: 这几组实验的条件与读数

插入层与组件消融的设置：主干 12 层 3B 的 MoE 模型（0.56B 激活参数），训练 100B token，挂 1.6B 参数的记忆，纯 MoE 基线的验证损失是 1.808。参考配置用 2 元和 3 元 n-gram、插在第 2、6 层，验证损失 1.768；单模块扫层时第 2 层最好（1.770）。论文 27B 模型的 Engram 配置是最大 3 元 n-gram、8 个头、维度 1280，嵌入参数用 Adam 更新且学习率放大 5 倍、不做权重衰减。

分配律实验的两个算力档：

| 算力 | 总参数 | 激活参数 | 纯 MoE 基线的专家数 |
|---|---:|---:|---:|
| $2\times10^{20}$ 浮点运算 | 约 5.7B | 568M | 106 |
| $6\times10^{20}$ 浮点运算 | 约 9.9B | 993M | 99 |

两档的稀疏比都固定在 $P_{\mathrm{tot}}/P_{\mathrm{act}}\approx10$。$P_{\mathrm{tot}}$ 是总可训练参数量，$P_{\mathrm{act}}$ 是每 token 激活的参数量，两者之差 $P_{\mathrm{sparse}}$ 是不激活的那部分预算。9.9B 档上验证损失从 $\rho=100\%$ 的 1.7248 降到 $\rho\approx80\%$ 附近的 1.7109，差 0.0139；最优位置在两档都落在 $\rho\approx75\%$–$80\%$。分配压到 $\rho\approx40\%$ 时剩 46 个与 43 个专家。

无限记忆档的设置：固定 3B 主干（568M 激活参数），训练 100B token，槽位数从 $2.58\times10^5$ 扫到 $1.0\times10^7$，最多加到约 13B 参数。论文用 OverEncoding（把 n-gram 嵌入与词表嵌入求平均的做法）作对照，它在同样的内存预算下扩展潜力小得多；SCONE 等以推理为目标、带额外模块与训练算力的工作与这里的等算力约束不可比，没有纳入对照。

:::

### 这条轴换来了什么

论文把 Engram 扩到 27B 规模。四个模型都训练 262B token、激活 3.8B 参数：稠密的 Dense-4B 是基线，MoE-27B 与 Engram-27B 的总参数量同为 26.7B，Engram-40B 把记忆扩得更大。Engram-27B 由 MoE-27B 改来，路由专家从 72 减到 55，腾出的参数放进 5.7B 的记忆模块，因此两者严格同等参数、同等算力。

结果是 Engram-27B 全面超过 MoE-27B：MMLU 从 57.4 升到 60.4，BBH 从 50.9 升到 55.9。涨幅不限于知识类，通用推理的 BBH、ARC-Challenge 与阅读理解类的 DROP 涨得都比 MMLU 更多。把记忆扩到 18.5B 参数的 Engram-40B 训练损失继续下降，多数基准也更好，但并非每一项都超过 Engram-27B，论文归因于训练 token 不足。

长上下文上，论文把结果解释为局部依赖交给查表之后、注意力容量可以腾出来处理全局上下文。同等算力下 Engram-27B 的 LongPPL 四项都领先，RULER 上多查询大海捞针从 84.2 升到 97.0，只有单针大海捞针略低于基线；更极端的一档里，只用 82% 预训练算力的 Engram-27B 在 LongPPL 上与训练充分的基线基本持平，RULER 上八类里有六类更高。

机制上论文用两个工具检查它是不是等效于加深了网络。LogitLens 把每层的隐状态接上最后的语言模型头，算它与最终输出分布的 KL 散度：Engram 的曲线在早期层明显更低，说明特征拼装完成得更快。CKA 用线性核比较两层表示的相似度：Engram-27B 第 5 层的表示最接近 MoE 基线的第 12 层左右，即对角线整体上移，论文据此认为 Engram 等效于增加了模型的有效深度。

论文还在推理时把记忆的输出完全抑制掉、主干不动，看出各项能力的依赖程度：事实知识类只保住 29%–44%，阅读理解类保住 81%–93%。门控可视化给出的图景一致，激活集中出现在多 token 命名实体与固定搭配上。

::: 基准与长上下文的读数

预训练基准（MoE-27B 与 Engram-27B 同为 26.7B 参数、3.8B 激活参数）：

| 基准 | MoE-27B | Engram-27B |
|---|---:|---:|
| MMLU | 57.4 | 60.4 |
| CMMLU | 57.9 | 61.9 |
| BBH | 50.9 | 55.9 |
| ARC-Challenge | 70.1 | 73.8 |
| DROP | 55.7 | 59.0 |
| HumanEval | 37.8 | 40.8 |
| MATH | 28.3 | 30.7 |

语言建模损失：Pile 测试集从 1.960 降到 1.950，验证集从 1.634 降到 1.622。长上下文扩展沿用 DeepSeek-V3 的做法，预训练之后用 YaRN（一种扩展上下文窗口的方法）做 32768 token 的上下文训练 5000 步、30B token。RULER 的八类是单针、多键、多值、多查询四种大海捞针，加上变量追踪、常见词提取、频繁词提取与问答。

| 指标（32k） | MoE-27B（50k 步） | Engram-27B（41k 步） | Engram-27B（50k 步） |
|---|---:|---:|---:|
| LongPPL 书本 | 4.38 | 4.37 | 4.14 |
| LongPPL 长思维链 | 14.16 | 14.26 | 13.41 |
| 多查询大海捞针 | 84.2 | 89.5 | 97.0 |
| 变量追踪 | 77.0 | 83.2 | 89.0 |
| 频繁词提取 | 73.0 | 99.6 | 99.3 |
| 常见词提取 | 4.5 | 3.8 | 5.9 |

:::

### 确定性寻址带来的系统做法

MoE 的路由依赖运行时的隐状态，选哪些专家要等算出来才知道；Engram 的行号只由输入 token 决定，在对应的层执行之前就已经全部算好。这个确定性带来两条工程上的后果。

训练时表按模型并行分片到各设备，前向用 All-to-All（一种在设备间全交换数据的集合通信原语）收集被激活的行，反向把梯度分发回去，总容量随加速器数量线性增长。

推理时既然行号提前已知，系统可以先把嵌入从容量充裕的主机内存异步取回，用前面若干层的计算掩盖这次传输。这给放置位置加了第二个要求：放得越深，可用的计算窗口越长，而建模效果偏好早介入，位置要同时满足两边。容量还能再往下一层分：自然语言的 n-gram 服从 Zipf 分布，少数模式占掉绝大多数访问，因此常查的嵌入可以放在显卡的高带宽显存、次常查的放在主机内存、长尾放到固态硬盘上。

论文在一个推理原型上量过这份代价：吞吐量的下降在百分之三以内，每步的通信量与被激活的槽位数成正比，而不是与表的总大小成正比。

::: 吞吐实验的设置与读数

实验把一张 100B 参数的表整个放在主机内存里，插在第二个 Transformer 块，预取与第一个块的计算重叠；硬件是 H800、512 条序列、序列长度在 100 到 1024 之间均匀分布。

| 主干 | 基线吞吐（token/s） | 加 100B Engram 后 | 变化 |
|---|---:|---:|---:|
| 4B 稠密 | 9031.62 | 8858.28 | 约 −1.9% |
| 8B 稠密 | 6315.52 | 6140.02 | 约 −2.8% |

论文说明这是一份保守下界：实验强制所有取回都从主机内存取，没有用上按频率分层的缓存。

:::

### 适用条件与边界

- 门控是唯一一道闸门。取回的记忆与当前上下文是否相符，交给那个标量门判断；而门本身由当前隐状态算出，隐状态没聚合到足够的上下文时，判断就不准。
- 位置只能折中。插得早，门控拿不到足够的上下文；插得晚，又错过了接手局部重建的时机。
- 容量换来的不是免费参数。每 token 浮点运算量确实不随表增长，但存储与带宽的账还在：表要占显存或主机内存，取回时多走一跳。这套账成立的前提是有足够的计算窗口可以掩盖这次传输。
- 哈希碰撞被压低但没有消失。多个头分散了它，撞上时取回的是另一个上下文的嵌入，仍要靠门控筛掉。
- 效果数字有其边界。分配律只在两种算力档、固定稀疏比的条件下考察；门控可视化是从若干样本里挑出最可读的那几条分支，不是全量统计。
