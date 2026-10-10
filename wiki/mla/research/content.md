# MLA

- 描述：MLA 是一种注意力机制，把每个位置的键和值联合压成一个低维潜在向量缓存起来，用到时再用上投影重建键和值。
- 摘要：MLA 把每 token 每层的键值压成一个低维潜在向量 $\mathbf{c}_t^{KV}$ 缓存，用上投影矩阵在注意力里重建键值；推理时这些矩阵可被吸收进查询与输出投影，连重建也省去，KV cache 只相当于 2.25 组的分组查询注意力。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 资料

- DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434v5, 2024. [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI. DeepSeek-V2: official implementation. 2024. [https://github.com/deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)

## 短页

### 把缓存压成一个向量

MLA 是一种注意力机制。多头注意力算一个位置的输出时，用这个位置的查询向量和前面各位置的键向量算相似度，再把各位置的值向量按相似度加权求和（相似度先经 softmax 归一化成一组和为 1 的权重）；查询、键、值都是输入经投影得到的（见[标准注意力](wiki/standard-attention)）。生成逐 token（逐个输出的文本单元）推进，前面位置的键和值算过就不再变，于是把它们缓存下来复用，这份缓存叫 KV cache（见 [KV cache](wiki/kv-cache)）。MLA 不缓存完整的键和值，而是先把每个位置的键和值联合压成一个低维潜在向量 $\mathbf{c}_t^{KV}$（压缩得到、不直接可读的表示向量）缓存，要用时再还原成键和值。注意力的形式不变，仍对全部前序位置加权求和。因为缓存的是低维向量而不是高维的键和值，KV cache 变小。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex L311；paper/src-v5/main.tex Preliminaries: Standard Multi-Head Attention

### 缓存是生成的瓶颈

生成时每写一个 token 都要用前面所有位置的键算注意力，这些键和值要一直留着，缓存的占用随序列长度线性增长。序列越长、同时处理的请求越多，它占的显存（GPU 上存放数据的存储）越多；显存有限，装得下的请求就少，生成越慢。已有的做法让多个注意力头共用键和值来缩小缓存，但分数低于标准多头注意力。MLA 要同时做到缓存更小且分数不低于标准多头注意力。
依据：paper/src-v5/main.tex Multi-Head Latent Attention；paper/src-v5/main.tex L1166-L1194

### 压缩、解耦、吸收

下投影压缩：用下投影矩阵把位置的输入向量压成一个低维潜在向量，只缓存它，取用时用两个上投影矩阵分别还原成键和值。解耦旋转位置编码：旋转位置编码是给查询和键按位置加上角度、让注意力看出先后次序的编码，直接对压缩后的键加它，会让下一步的矩阵合并（把上投影并进查询投影和输出投影）没法成立，所以另用一对较小的查询和键单独承载它。矩阵吸收：推理时把键的上投影矩阵并进查询投影（得到查询的那个矩阵）、值的上投影矩阵并进输出投影（注意力输出之后那个投影），省去为每个查询重建键和值。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex Decoupled Rotary Position Embedding；paper/src-v5/main.tex L1159-L1162

## 长页

多头注意力算一个位置的输出时，用这个位置的查询向量和前面每个位置的键向量做内积，按内积大小把对应位置的值向量加权求和；查询、键、值都由输入经投影矩阵得到（见[标准注意力](wiki/standard-attention)）。生成逐 token（逐个输出的文本单元）进行，第 $t$ 个 token 只能看到前 $t$ 个位置。前面位置的键和值算过就不再变，它们被缓存下来供后面各步复用，这份缓存叫 KV cache（见 [KV cache](wiki/kv-cache)）。序列越长、层数越多，它越大，逐渐成为限制一次能同时处理的请求数和序列长度的瓶颈。
依据：paper/src-v5/main.tex Introduction；paper/src-v5/main.tex Preliminaries: Standard Multi-Head Attention

已有的减缓存做法是让多个注意力头（各自独立做注意力的并行单元）共用键和值，例如分组查询注意力（GQA）。这类做法压小缓存，但对比实验里基准得分低于标准多头注意力（MHA）。
依据：paper/src-v5/main.tex Multi-Head Latent Attention；paper/src-v5/main.tex L1166-L1194

MLA 不减少键和值的份数，而是把每个位置的键和值联合压成一个低维潜在向量 $\mathbf{c}_t^{KV}$（压缩得到、不直接可读的表示向量）缓存，用到时再用上投影矩阵把它还原成键和值。注意力仍是每个头各自对全部前序位置做加权求和，KV cache 却从缓存高维的键和值变成只缓存一个低维向量。它缓存比标准多头注意力小很多，分数多数更高。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex Multi-Head Latent Attention

### 把键和值压成一个潜在向量

设一个注意力层的输入是 $d$ 维向量 $\mathbf{h}_t$，$t$ 是位置编号。MLA 用三个矩阵做低秩压缩：下投影矩阵 $W^{DKV}$ 是 $d_c \times d$，把 $\mathbf{h}_t$ 压成 $d_c$ 维的潜在向量 $\mathbf{c}_t^{KV}$；两个上投影矩阵 $W^{UK}$、$W^{UV}$ 都是 $d_h n_h \times d_c$，把 $\mathbf{c}_t^{KV}$ 分别还原成键 $\mathbf{k}_t^{C} = W^{UK}\mathbf{c}_t^{KV}$ 和值 $\mathbf{v}_t^{C} = W^{UV}\mathbf{c}_t^{KV}$，这里 $\mathbf{k}_t^{C}$ 是各头的键拼成的整体。这里 $n_h$ 是注意力头数，$d_h$ 是每个头的维度，$d_c$ 是压缩维度，取远小于 $d_h n_h$。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression；paper/src-v5/main.tex Preliminaries: Standard Multi-Head Attention

生成时缓存的是 $\mathbf{c}_t^{KV}$ 而不是 $\mathbf{k}_t^{C}$、$\mathbf{v}_t^{C}$，所以键值这部分缓存每 token 每层只有 $d_c$ 个元素，比标准多头注意力小得多；解耦键的缓存另算。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression

查询也做同样的低秩压缩：先用 $W^{DQ}$ 把 $\mathbf{h}_t$ 压到 $d_c^{\prime}$ 维，再由 $W^{UQ}$ 还原成 $\mathbf{q}_t^{C} = W^{UQ}\mathbf{c}_t^{Q}$，其中 $\mathbf{c}_t^{Q} = W^{DQ}\mathbf{h}_t$，$d_c^{\prime}$ 是查询压缩维度。这一步不减小 KV cache，压缩查询是为了减小训练时的中间激活显存（前向过程中间结果的显存占用）。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression

::: 一个位置压小再还原
取一个位置，$d=8$、$n_h=2$、$d_h=4$、$d_c=2$，输入向量 $\mathbf{h}=(1,0,-1,0,2,0,-2,0)$，下投影矩阵取 $W^{DKV}$ 的前两行：第一行 $(1,0,0,1,0,0,0,0)$，第二行 $(0,1,0,0,0,0,0,1)$。这里只看压缩与还原的维度变化，不做完整注意力。
依据：自行构造

1. 压缩：$\mathbf{c}^{KV}$ 的第一个分量是 $\mathbf{h}$ 第 1、4 个分量之和，$1+0=1$；第二个分量是第 2、8 个分量之和，$0+0=0$。得 $\mathbf{c}^{KV}=(1,0)$，只有 2 个元素。
2. 还原：$\mathbf{k}^{C} = W^{UK}\mathbf{c}^{KV}$，上投影取 $8\times 2$ 的矩阵，每行从两个分量里各取一个相加，得到一个 8 维的键向量，形状与标准多头注意力的键相同。
3. 缓存对比：这个位置在标准多头注意力下要缓存键和值各 $n_h d_h = 8$ 个元素，共 $16$ 个；MLA 只缓存 $\mathbf{c}^{KV}$ 的 $2$ 个，缩小到八分之一。这里只算键值压缩出的这份缓存，解耦键另占 $d_h^R$ 个。
依据：自行构造
:::

### 推理时不必算出键和值

推理（模型训练好后逐 token 生成的那一步）时缓存里只有 $\mathbf{c}_t^{KV}$，最直接的做法是每一步先用它算出该位置的键和值再算注意力。MLA 把这一步也省了。单个头的注意力分数是查询与键的内积，键由上投影还原，代入还原式：

$$\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i} = \mathbf{q}_{t,i}^\top W^{UK}_i \mathbf{c}_j^{KV} = (W^{UK}_i{}^\top \mathbf{q}_{t,i})^\top \mathbf{c}_j^{KV}$$

这里 $\mathbf{q}_{t,i}$ 是位置 $t$ 第 $i$ 个头的查询，$\mathbf{k}_{j,i}$ 是位置 $j$ 第 $i$ 个头的键（整体 $\mathbf{k}_j^{C}$ 里属于第 $i$ 个头的那一份），$W^{UK}_i$ 是键上投影矩阵里属于第 $i$ 个头的部分。右端把 $W^{UK}_i{}^\top$ 和查询合成一个向量，只用缓存的 $\mathbf{c}_j^{KV}$ 就能算分数，不必先算出 $\mathbf{k}_{j,i}$；这一步用的是矩阵乘法的结合律。值这一侧同理：注意力输出是各位置的值按权重相加，值只被一个标量权重相乘再相加，所以值的上投影矩阵可以直接并进输出投影（注意力输出之后那个投影矩阵）。把两处合起来，$W^{UK}$ 并进查询的上投影 $W^{UQ}$，$W^{UV}$ 并进输出投影 $W^{O}$（这种把矩阵并进另一处的做法叫吸收），推理时不再算出键和值，也就没有这一步的计算开销。
依据：paper/src-v5/main.tex L1159-L1162；paper/src-v5/main.tex Low-Rank Key-Value Joint Compression

::: 合并是怎么成立的
对一个头算一次。潜在向量 $\mathbf{c}^{KV}=(2,3)$，键上投影 $W^{UK}=(1,1)$，查询 $\mathbf{q}=4$，值上投影 $W^{UV}=(1,-1)$，输出投影 $W^{O}=(1)$。
依据：自行构造

1. 先算键再算分数：$\mathbf{k}=1\times 2+1\times 3=5$，分数 $\mathbf{q}^\top\mathbf{k}=4\times 5=20$。
2. 吸收后算分数：把 $W^{UK}$ 并进查询得 $(1,1)\times 4=(4,4)$，再与 $\mathbf{c}^{KV}$ 内积，$4\times 2+4\times 3=20$，与上一步相同，说明不必先算出键。
3. 先算值再算输出：$\mathbf{v}=1\times 2+(-1)\times 3=-1$；设该位置的注意力权重为 $1$，输出贡献为 $-1$。
4. 吸收后算输出：先按权重求和得 $(2,3)$，再乘并进了 $W^{UV}$ 的输出投影 $(1,-1)$，$1\times 2-1\times 3=-1$，与上一步相同，说明值也不必先算出。
依据：自行构造
:::

### 解耦旋转位置编码

这里用的是旋转位置编码（见[旋转位置编码](wiki/rope)）：它把查询和键按各自位置旋转一个角度，内积随相对位置变化。但把旋转位置编码直接加在压缩键上，会破坏把 $W^{UK}$ 并进查询的合并。加过之后键变成 $R_j\mathbf{k}_j^{C} = R_j W^{UK}\mathbf{c}_j^{KV}$，$R_j$ 是只和位置 $j$ 有关的旋转矩阵；合并靠的是 $W^{UK}$ 并进查询，现在查询和它之间夹着一个随位置变化的 $R_j$，矩阵乘法不满足交换律，$W^{UK}$ 就并不进查询了。于是每次生成都要重算前面所有位置的历史键，推理变慢。
依据：paper/src-v5/main.tex Decoupled Rotary Position Embedding

做法是解耦：另用一对专门承载位置的查询和键。每个头有一份解耦查询 $\mathbf{q}_{t,i}^{R}$，所有头共用一份解耦键 $\mathbf{k}_t^{R}$，二者都是 $d_h^R$ 维，旋转位置编码只加在它们上面。每个头的查询和键由两段拼接：一段是压缩查询、压缩键，另一段是解耦查询、解耦键。这样压缩的那一段不带位置，把 $W^{UK}$ 并进查询的合并照旧成立。代价是解耦键也要缓存，每 token 每层的缓存从 $d_c$ 增加到 $d_c + d_h^R$ 个。
依据：paper/src-v5/main.tex Decoupled Rotary Position Embedding

::: 解耦后的查询与键
每个头的查询是两段的拼接 $\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^{C}; \mathbf{q}_{t,i}^{R}]$，键是 $\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^{C}; \mathbf{k}_t^{R}]$，注意力分数除以 $\sqrt{d_h + d_h^R}$（按拼接后的维度开方缩放，使分数的尺度不随维度增大）。解耦查询由查询的潜在向量经 $W^{QR}$ 再旋转得到，解耦键由输入 $\mathbf{h}_t$ 经 $W^{KR}$ 再旋转得到，$W^{QR}$ 与 $W^{KR}$ 是这两处的新投影矩阵。$[\cdot;\cdot]$ 表示拼接。
依据：paper/src-v5/main.tex Decoupled Rotary Position Embedding
:::

### KV cache 有多小

标准多头注意力每 token 每层缓存 $2 n_h d_h$ 个元素，键和值各占 $n_h d_h$ 个；乘以层数 $l$，每 token 全部层共 $2 n_h d_h l$ 个。MLA 每 token 每层缓存 $d_c + d_h^R$ 个，是压缩后的潜在向量加上所有头共用的解耦键。
依据：paper/src-v5/main.tex L318；paper/src-v5/main.tex L256

把提出 MLA 的模型 DeepSeek-V2 的配置代进去：头数 $n_h=128$、头维 $d_h=128$、压缩维 $d_c=512$、解耦维 $d_h^R=64$。
依据：paper/src-v5/main.tex L465-L478

每 token 每层缓存 $d_c + d_h^R = 512 + 64 = 576$ 个元素；分组查询注意力每 token 每层缓存 $2 n_g d_h$ 个，$n_g$ 是组数，代入 $d_h=128$ 得 576 个相当于 2.25 组。
依据：自行构造

::: 两种注意力的缓存量
| 注意力机制 | 每 token 每层的缓存元素数 |
|---|---|
| 标准多头注意力（MHA） | $2 n_h d_h$ |
| MLA | $d_c + d_h^R$ |

把 DeepSeek-V2 的配置 $d_c = 4 d_h$、$d_h^R = d_h / 2$ 代入，MLA 每层缓存约 $4.5 d_h$，与头数无关。
依据：自行构造
:::

### 效果与效率

缩小缓存没有降低效果：在只差注意力机制的对比里，MLA 的基准得分多数高于标准多头注意力（MHA），少数略低。
依据：paper/src-v5/main.tex L1196-L1230

::: MLA 与 MHA 的对比
在只差注意力机制、其余相同的两个规模上比较，四个基准共 8 项分数：MLA 高于 MHA 7 项，1 项略低。
依据：paper/src-v5/main.tex L1199-L1223
:::

### 适用条件与边界

MLA 只改缓存的存法，不改注意力的形式：输出仍是对全部前序位置按 softmax 权重求和（softmax 把查询与各键的内积算成一组和为 1 的权重），不是只对邻近若干位置求和（滑窗），也不丢弃位置，全局注意力因此保留。
依据：paper/src-v5/main.tex L311

缓存仍随层数和序列长度线性增长，每个位置仍要占 $d_c + d_h^R$ 个元素；省掉的是值的缓存和逐步重建键、值的计算。
依据：paper/src-v5/main.tex L318；paper/src-v5/main.tex L1158-L1162

一条常见误解是以为 MLA 把 KV cache 去掉了：它仍缓存每个位置的潜在向量，只是这个向量比完整的键和值小。
依据：paper/src-v5/main.tex Low-Rank Key-Value Joint Compression
