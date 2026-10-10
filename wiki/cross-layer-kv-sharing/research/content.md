# 跨层 KV 复用

- 描述：跨层 KV 复用让多层共用同一份全局 KV：后一半层不再各自计算，而是读取前一半层末层投影出的唯一一份，把原本每层各存一份改成后一半各层共用一份。
- 摘要：前一半层算出唯一一份全局 KV $\hat K=\mathrm{LN}(X^{L/2})W_K$、$\hat V=\mathrm{LN}(X^{L/2})W_V$，后一半层各层复用它；缓存复杂度从 $\mathcal{O}(LND)$ 降到 $\mathcal{O}((N+L)D)$，prefill 还能提前退出一半的层。
- 主题：注意力机制, 内存与缓存
- 标签：KV cache

## 资料

- Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, Furu Wei. You Only Cache Once: Decoder-Decoder Architectures for Language Models. arXiv:2405.05254v2, 2024. https://arxiv.org/abs/2405.05254v2
- YOCO 官方实现（Microsoft unilm 仓库 YOCO/ 子目录），commit 31c5b904。https://github.com/microsoft/unilm

## 短页

### 多层读同一份 KV

把 $L$（层数）层分成前后两段，前一段叫下段，后一段叫上段。下段逐层算，最后一层的输出经一次投影（乘一个矩阵）得到一份覆盖整条序列的全局 KV；上段各层不再各自算全局 KV，而是都读这唯一的一份。这里的 key、value 是[标准注意力](wiki/standard-attention)里每个位置算出的两组向量，存起来供后续位置读取就是 [KV cache](wiki/kv-cache)。
依据：paper/src-v2/main.tex L60-L61；paper/src-v2/main.tex Cross-Decoder

### KV cache 随层数线性增长

[标准注意力](wiki/standard-attention)里，每个位置算出的 key、value 只被本层使用，生成下一个 token（文本被切成的单元）时还要再用，所以每一层都要把它们缓存下来。各层各存一份，缓存总量随层数线性增长；长序列下它成为主要的内存占用，prefill（为整段输入一次算出各层表示、生成第一个 token 之前的阶段）也慢。
依据：paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Inference Advantages

### 分成两段，上段复用

下段各层用只存固定大小 cache 的高效注意力（如[滑动窗口注意力](wiki/sliding-window-attention)）逐层算，最后一层的输出经投影成一份全局 KV。上段各层改做交叉注意力：query（本位置要匹配的向量）由本层算，key、value 用共享的那一份，不再自算全局 KV。全局 KV 因此与层数无关，缓存复杂度从 $\mathcal{O}(LND)$ 降到 $\mathcal{O}((N+L)D)$（$N$ 序列长度、$D$ 维度，多出的 $L$ 来自下段各层各留一份固定大小的局部 cache）；prefill 算完下段、拿到全局 KV 即可退出，不必进上段，少算一半的层。
依据：paper/src-v2/main.tex Cross-Decoder；paper/src-v2/main.tex Inference Advantages；paper/src-v2/main.tex Self-Decoder

## 长页

标准 Transformer 指把注意力逐层堆叠、每层都用[标准注意力](wiki/standard-attention)的模型。它的每一层都要为整段上下文各存一份 key 和 value，层数越多，这部分缓存越大。跨层 KV 复用把 $L$（层数）层分成下段和上段：下段逐层计算，最后算出一份覆盖整条序列的全局 KV；上段各层不再各自生成全局 KV，而是通过交叉注意力读取这一份。交叉注意力指 query（本位置要匹配的向量）来自本层、key 和 value 来自别处（这里是下段算出的那份）的注意力。下段各层自己的局部缓存仍然保留。
依据：paper/src-v2/main.tex L60-L61；paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Inference Advantages

### 每层各存一份 KV 的代价

标准注意力里，每个位置用它的 query 和前面位置的 key 做点积得到一组匹配权重，再对这些位置的 value 加权求和，得到本层的输出；key 和 value 都由本层的隐状态（本层算出的向量表示）算出，只服务本层。生成下一个 token（文本被切成的单元）时，前面每个 token 的 key、value 还要再用，因此每一层都要把它们缓存下来。层数越多，各层各存一份，缓存总量随层数线性增长。
长序列下，存下来的 key、value（[KV cache](wiki/kv-cache)）成为主要的内存瓶颈，而模型参数占的是固定大小；prefill（为整段输入一次算出各层表示、生成第一个 token 之前的阶段）序列越长这一步越慢。
依据：paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Inference Advantages

::: 长序列下的两个量级
1. 一个 65B（B 指十亿参数）的模型，另配分组查询注意力（多个查询头共用一组 key/value 头，头指一组并行的 query/key/value 向量）与 8-bit 的 KV 量化（量化是用更少位宽存数值），512K 个 token 的 KV cache 约占 86GB，比一张 80GB 显存的 GPU 还大。
2. 一个 7B 的模型，另用两种推理加速手段（Flash-Decoding 与 kernel fusion）、四张 GPU，prefill 450K 个 token 约要 110 秒，1M 个 token 约要 380 秒。
依据：paper/src-v2/main.tex L100-L103
:::

### 复用的是哪一层的 KV

$L$ 层被分成两半：前 $L/2$ 层是下段，自己逐层算 KV，也叫自解码器；后 $L/2$ 层是上段，用交叉注意力读下段算出的 KV，也叫交叉解码器。下段逐层计算，得到的最后一层隐状态 $X^{L/2}$ 被投影（乘一个矩阵）成全局 KV。上段各层不再从自己的隐状态生成全局 KV，而是共用这一份：每层的 query 仍由本层的隐状态算，key、value 用共享的那一份。全局 KV 因此只有一份，与层数无关。
依据：paper/src-v2/main.tex You Only Cache Once (YOCO)；paper/src-v2/main.tex Cross-Decoder；repo/yoco/yoco/models/decoder/yoco.py L207-L235

::: 全局 KV 怎么算，层数怎么变少
1. 下段逐层算，最后一层（第 $L/2$ 层）的输出记为 $X^{L/2}$。
2. $X^{L/2}$ 经归一化后乘投影矩阵，得到全局 KV：$\hat K=\mathrm{LN}(X^{L/2})W_K$，$\hat V=\mathrm{LN}(X^{L/2})W_V$；$\mathrm{LN}$ 是归一化，$W_K$、$W_V$ 是两个 $D\times D$ 的可学习矩阵，$D$ 是隐藏维度。
3. 上段第 $l$ 层用自己的隐状态算 query $\hat Q^l=\mathrm{LN}(X^l)W_Q^l$（$X^l$ 是第 $l$ 层的隐状态，$W_Q^l$ 是一个 $D\times D$ 矩阵），再对共享的 $\hat K$、$\hat V$ 做注意力，结果加回本层隐状态。
依据：paper/src-v2/main.tex Cross-Decoder
4. 取 $L=4$，共 4 层，下段是层 1、2，上段是层 3、4。每层各算一份时，全局 KV 有 4 份；这里由下段末层（层 2）算出 1 份，层 3、层 4 都读它，全局 KV 只有 1 份。
依据：自行构造
:::

### 缓存与 prefill 各省下多少

缓存方面：每层各存一份时，全局 KV 的条目正比于层数，复杂度是 $\mathcal{O}(LND)$（$N$ 序列长度、$L$ 层数、$D$ 维度）；只存一份后，复杂度降到 $\mathcal{O}((N+L)D)$，相比约省 $L$ 倍。
prefill 方面：prefill 时全局 KV 由下段算出，下段算完即可退出、不必进上段，少算一半的层。资料给出的 attention 复杂度，每层都做整段注意力的 Transformer，prefill 是 $\mathcal{O}(LN^2D)$；复用后只算下段，下段用只存常数 cache 的高效注意力、每层对 $N$ 个位置的计算量随序列长度线性增长，降到 $\mathcal{O}(LND)$。
依据：paper/src-v2/main.tex Inference Advantages；paper/src-v2/main.tex Self-Decoder

::: 两个复杂度的算法
1. 每层各存：每层存 $N$ 个 token 的 key、value，共 $L$ 层，量级 $LND$，即 $\mathcal{O}(LND)$。
2. 复用后：全局 KV 只有一份，量级 $ND$；下段 $L/2$ 层各存一份大小有限的局部 cache，其大小不随序列增长（如窗口大小 $C$），量级为常数个数，计入 $\mathcal{O}(LD)$。合起来 $\mathcal{O}((N+L)D)$，随长度增长的那部分只剩 $ND$。
3. 长序列下 $N$ 远大于 $L$，所以两式相比约省 $L$ 倍。
4. prefill：Transformer 每层对 $N$ 个位置各做一次长度 $N$ 的整段注意力，每层 $O(N^2)$，共 $\mathcal{O}(LN^2D)$；复用后只算下段，下段每层计算量随序列长度线性增长，共 $\mathcal{O}(LND)$。
依据：paper/src-v2/main.tex Inference Advantages；paper/src-v2/main.tex Self-Decoder
:::

### 适用条件与边界

「只存一次」严格指全局 KV。一种常见的理解是把它当成整个模型不再随序列长度存缓存；实际上下段仍要保存一定量的局部缓存，只是被限制在固定大小（如一个窗口），长序列下它相对全局缓存可以忽略。这也是为什么复用后的缓存复杂度是 $\mathcal{O}((N+L)D)$，而不是全模型与序列长度无关。
收益成立的条件是下段用只存常数 cache 的高效注意力，例如[滑动窗口注意力](wiki/sliding-window-attention)。
代价是上段各层不再各自维护一份全局 KV，它们对上下文的访问都经由同一份 key、value。这不是说上段失去了全局注意力：交叉注意力仍覆盖整段序列，只是读的是共享的那一份。
依据：paper/src-v2/main.tex Introduction；paper/src-v2/main.tex Self-Decoder；paper/src-v2/main.tex Inference Advantages；paper/src-v2/main.tex L60-L61
