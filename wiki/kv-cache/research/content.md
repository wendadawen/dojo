# KV cache

- 描述：KV cache 是自回归生成时把每层已算过的 key 和 value 留下来复用的机制，让生成每个新 token 时不必重算全部历史。
- 摘要：自回归模型逐个预测 token，每步都要用到全部历史 token 的 key 和 value。把它们按层缓存下来、每步只追加当前 token，每步注意力开销从随序列长度二次降到线性。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 资料

- HuggingFace Transformers 文档. How caching works. [https://huggingface.co/docs/transformers/en/cache_explanation](https://huggingface.co/docs/transformers/en/cache_explanation)
- HuggingFace Transformers 文档. Cache strategies. [https://huggingface.co/docs/transformers/en/kv_cache](https://huggingface.co/docs/transformers/en/kv_cache)

## 短页

### 把每层算过的 key 和 value 存下来

大语言模型写文本时一次只预测一个 token——token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分——而每个新 token 都要用到前面所有 token 的信息，这种一个接一个往下写的方式叫自回归生成。模型由若干结构相同的层堆叠而成，每层里都有一层[注意力](wiki/standard-attention)：它把每个位置的输入投影成三个向量，分别是 query、key 和 value，算某个位置的输出时，用它的 query 去和各个位置的 key 比较，按比较结果把对应的 value 加权求和。KV cache 做的事，就是在这个逐步生成的过程中，把每一层已经算出来的 key 和 value 留在显存（GPU 的内存）里，供后面的 token 直接取用，省去重算。
依据：blog/hf-cache-explanation.md How caching works；blog/hf-kv-cache.md Cache strategies；blog/hf-cache-explanation.md Attention matrices

### 不存就要每步重算全部历史

模型每预测一个新 token，都要用到全部历史 token 的 key 和 value。如果每次都用时才算，那么生成第 $t$ 个 token 时，要从头把整段序列重算一遍：不光当前这个位置要算，前面 $t-1$ 个位置也要连同它们的注意力一起重新过一遍。序列长度指已经生成的 token 个数。不缓存时，每步都要这样重跑一遍，每步开销随序列长度二次增长；缓存后每步只算当前这一个位置、只和已有的各个位置各比较一次，开销随长度线性增长。
依据：blog/hf-cache-explanation.md Attention matrices；blog/hf-cache-explanation.md L58-L63

### 每步只算新 token，追加进缓存

有了缓存之后，每步对当前这个 token 算它的 query、key 和 value：key 和 value 接在缓存里旧的部分之后，追加进缓存；query 拿去和缓存里的全部 key 比较，按比较结果把缓存里的全部 value 加权求和，得到这一步的输出。历史的 key 和 value 不再重算，每步只往缓存里加一个新 token 的量，旧的部分保持不变。
依据：blog/hf-cache-explanation.md Cache class；blog/hf-cache-explanation.md Cache storage implementation

## 长页

大语言模型写文本时一次只预测一个 token——token 是模型读写文本的最小单位——而每个新 token 都要用到前面所有 token 的信息，这种一个接一个往下写的方式叫自回归生成。模型由若干结构相同的层堆叠而成，每层里都有一层[注意力](wiki/standard-attention)，它把每个位置的输入投影成 query、key 和 value 三个向量，算某个位置的输出时，用它的 query 去和各个位置的 key 比较，按比较结果把对应的 value 加权求和。
依据：blog/hf-cache-explanation.md How caching works；blog/hf-cache-explanation.md Attention matrices

模型处理每个位置，都会在该层算出一套 key 和 value。在只看当前位置及其之前位置的因果注意力下，一个位置处理完之后，它的 key、value 不再随后面出现的 token 改变，因此可以算一次、反复使用。不把它们留下来，每生成一个 token 就要把前面所有位置的 key 和 value 重算一遍。KV cache 就是在生成过程中把每一层已经算出的 key 和 value 存进显存（GPU 的内存），每步只追加新 token 的那一份，让后面的 token 直接取用。
依据：blog/hf-cache-explanation.md How caching works；blog/hf-kv-cache.md Cache strategies；blog/hf-cache-explanation.md Attention matrices

### 每步只算当前 token，历史 key、value 复用

生成时模型一次只处理一个新位置，但注意力的算法没有变。注意力为当前位置的 query 和各个位置的 key 各算一个分数——两者的点积，点积越大表示两者越匹配。生成是从左到右一步步走的，当前位置看不到后面还没出现的 token，这由一个掩码保证：掩码让模型注意不到未来位置。
依据：blog/hf-cache-explanation.md Attention matrices

把这些分数除以一个缩放数再送进 softmax，softmax 把一组数变成一组和为 1 的权重，越大的数分到的权重越大；最后用这组权重把各个位置的 value 加权求和：
依据：blog/hf-cache-explanation.md Attention matrices

$$\text{softmax}\!\left(\frac{q_t K^\top}{\sqrt{d}} \times \text{mask}\right) V$$
依据：blog/hf-cache-explanation.md Attention matrices

其中 $q_t$ 是位置 $t$ 的 query，$K$ 和 $V$ 是各个位置的 key、value 排成的矩阵，$d$ 是每个 query、key、value 向量的维度，mask 是前面说的掩码。进入这一步计算的 query 只有当前这一个位置的那一份：$q_t$ 由位置 $t$ 的输入乘投影矩阵得到，每来一个新位置就有一个新的 query。当前这个位置同时也会算出它自己的 $k_t$ 和 $v_t$，它们和历史上各个位置的 key、value 一起进入比较和加权；历史位置的 key、value 在各自算出的那一步之后就不再改变。所以每步要新算的只有当前这一个位置的一份 query、key 和 value，用过的 query 此后不再需要。
依据：blog/hf-cache-explanation.md Attention matrices

::: 从公式看每步算了什么
位置 $t$ 这一步的输出，是它自己的 query 和所有已经出现的位置（1 到 $t$）的 key、value 一起算出来的。

1. 第 1 个 token 进来，算它的 $q_1$、$k_1$、$v_1$，注意力只用 $q_1$ 和 $k_1$，得到第一个输出。
依据：blog/hf-cache-explanation.md Attention matrices

2. 第 2 个 token 进来，算 $q_2$、$k_2$、$v_2$，注意力用 $q_2$ 和 $k_1, k_2$ 比较、取 $v_1, v_2$。
依据：blog/hf-cache-explanation.md Attention matrices

3. 到第 $t$ 个 token，注意力用 $q_t$ 和 $k_1, \dots, k_t$ 比较、取 $v_1, \dots, v_t$。
依据：blog/hf-cache-explanation.md Attention matrices

4. 每一步里，已经算过的那些位置的 $k$、$v$ 都不再变化（各自在算它们的那一步定下），需要新算的只有当前这一份。
依据：自行构造
:::

### 缓存按层各存一份，每步追加一个 token

注意力在模型的每一层里各算各的，缓存也按层进行，所以每一层都各存一份 key 和一份 value。生成往下走时，每层的缓存各自增长。
依据：blog/hf-cache-explanation.md Attention matrices；blog/hf-cache-explanation.md Cache storage implementation

每步算出的新的 $k_t$、$v_t$ 接在过去的 key、value 后面，缓存就是这样一步步变长的：新的 key 缓存是过去的 key 拼上当前 $k_t$，新的 value 缓存是过去的 value 拼上当前 $v_t$。
依据：blog/hf-cache-explanation.md Cache class

::: 拼接公式
把拼接写成公式：

$$K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \qquad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)$$
依据：blog/hf-cache-explanation.md Attention matrices

其中 $K_{\text{past}}$、$V_{\text{past}}$ 是缓存里已有的历史 key、value，$k_t$、$v_t$ 是当前这一步新算出来的，concat 表示沿序列方向接在一起。
依据：blog/hf-cache-explanation.md Attention matrices
:::

### 缓存省下多少计算

不缓存时，每生成一步都要把整段历史重新过一遍注意力：序列里每个位置都要和它之前的各个位置各比较一次，这些比较排成一张行数、列数都等于长度的分数表，工作量随长度平方增长。缓存之后，每步只需要算当前这一个位置、只和已有的各个位置各比较一次，不再回头重算。
依据：blog/hf-cache-explanation.md L58-L63；blog/hf-cache-explanation.md Attention matrices

::: 不缓存与缓存的每步开销对比
| 不缓存 | 缓存 |
|---|---|
| 每步重算全部历史的 key 和 value | 每步只算当前的 key 和 value |
| 每步注意力开销随序列长度**二次**增长 | 每步注意力开销随序列长度**线性**增长 |
依据：blog/hf-cache-explanation.md L58-L63
:::

### 缓存有多大

注意力把一次计算分成若干个头，每个头各自算一套 query、key 和 value，所以缓存里存的是每个头那一套；头数记作 $H_{\text{kv}}$，每个头里向量的维度记作 $d_{\text{head}}$。生成时还可以一次并行处理多条序列，一次处理多少条叫批大小。把这两件事算上，每一层的 key 缓存和 value 缓存的形状都是 `[batch_size, num_heads, seq_len, head_dim]`：这四项依次是批大小、KV 头数、序列长度、头维度；生成过程中，序列长度随每个新 token 增加。
依据：blog/hf-cache-explanation.md Cache storage implementation；blog/hf-cache-explanation.md Attention matrices

把形状的各项相乘就得到缓存的元素数，元素数与批大小、序列长度都成正比，所以显存随序列长度和批大小线性增长。
依据：自行构造

::: 显存公式与一组手算
key 和 value 各存一份，所以形状的各项相乘后再乘 2；每一层各存一份，所以再乘层数 $L$；每个数占的字节数记作 $b$，再乘 $b$。合起来：

$$\text{显存} = 2 \times L \times B \times T \times H_{\text{kv}} \times d_{\text{head}} \times b$$
依据：自行构造

其中 $L$ 是层数，$B$ 是批大小，$T$ 是序列长度，$H_{\text{kv}}$ 是 KV 头数，$d_{\text{head}}$ 是头维度，$b$ 是每个数占的字节数，最前面的 2 是 key 和 value 两份。
依据：自行构造

取一组小数字走一遍：层数 $L=2$，批大小 $B=1$，序列长度 $T=3$，KV 头数 $H_{\text{kv}}=2$，头维度 $d_{\text{head}}=4$，每个数 2 字节。
依据：自行构造

1. 先算一个位置在一层里、只算 key 一份的元素数：$H_{\text{kv}} \times d_{\text{head}} = 2 \times 4 = 8$。
依据：自行构造

2. key 和 value 两份，一个位置一层是 $2 \times 8 = 16$ 个元素。
依据：自行构造

3. 一个位置、两层的值是 $2 \times 16 = 32$ 个元素。
依据：自行构造

4. 三个位置、批大小为 1，是 $3 \times 1 \times 32 = 96$ 个元素。
依据：自行构造

5. 每个数 2 字节，总共 $96 \times 2 = 192$ 字节。
依据：自行构造

```python
L = 2      # 层数
B = 1      # 批大小
T = 3      # 序列长度
H = 2      # KV 头数
d = 4      # 头维度
b = 2      # 每个数的字节数
elements = 2 * L * B * T * H * d
print(elements)
print(elements * b)
```

```text
96
192
```

把序列长度从 3 加到 6，只改这一项，字节数就翻倍，说明显存随序列长度线性增长。
依据：自行构造
:::

### 适用条件与边界

缓存只在推理时用——推理指模型已经训练好、拿来做预测的阶段，训练指模型学习参数的阶段。训练时启用缓存可能引发意外错误，所以训练阶段要把它关掉。
依据：blog/hf-cache-explanation.md L29-L30

缓存会占去显存里相当一部分，在长序列（长上下文）里可能成为瓶颈：序列越长，缓存占的显存越多，越容易让显存不够用。
依据：blog/hf-kv-cache.md Cache offloading

缓存省掉的是计算——历史位置的 key 和 value 不再重算；它并不省显存。
依据：blog/hf-cache-explanation.md L58-L63
