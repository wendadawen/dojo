# Chunked Prefill

- 描述：Chunked Prefill 是把一条预填充请求切成若干计算量相近的块、分多次迭代算完，让正在解码的请求每轮与其中一块同批算。
- 摘要：把长预填充切成小块分次算：每块仍让 GPU 接近满载，正在解码的请求每轮与这一块合进同一个批，于是不必等整段预填充算完，解码不被打断，每轮的计算量也接近。
- 主题：推理系统
- 标签：推理加速

## 资料

- Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. arXiv:2308.16369v1, 2023. [https://arxiv.org/abs/2308.16369](https://arxiv.org/abs/2308.16369)
- Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. arXiv:2403.02310v2, 2024. [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

## 短页

### 把预填充切成几块算

预填充把整段提示一次送进模型、并行处理 token，并给出第一个输出 token；提示是送进模型、要它接着生成的输入文本，token 是文本经分词得到的最小单位。解码逐个生成后面的 token，每步只输入上一步生成的那个。

模型由很多结构相同的层叠成，每个位置的数据逐层往后算；[标准注意力](wiki/standard-attention)让位置之间交换信息，每个位置用它算出的查询去和前面位置的键匹配，再按匹配程度取回对应的值（查询、键、值都是位置在注意力里算出的向量）。[KV cache](wiki/kv-cache)把已处理位置的键和值存下来供后面复用。

Chunked Prefill 不把整段提示一次算完，而是按 token 把它切成大小相近的几块、分多次迭代算，每次算一块。为了让后面几块能看到前面的内容，每个已算过的位置在各层算出的键和值都存进 KV cache；同时一块一块地设置[因果掩码](wiki/causal-mask)，只让每个位置看它前面的位置。这样切块不改变结果。

依据：paper/src-v1/4-design.tex L14；paper/src-v2/4-design.tex L15；paper/src-v2/2-background.tex L20-L22；paper/src-v2/2-background.tex L9；paper/src-v1/4-design.tex L21

### 预填充算力利用率高，解码利用率低

预填充一次并行处理几百到上千个 token，就能让 GPU 接近满载。解码每一步只处理一个 token，却要把整套模型权重（模型里经训练得到、推理时反复读取的那批参数）从显存（GPU 上的内存）读一遍，瓶颈落在显存带宽（单位时间能读写的字节数）而不是算力（这一步是内存受限的），算力闲置；只有把大量请求并成一个大批，解码的算力才能被有效利用。当两者混在一条流水里，一条新请求的预填充会插进正在解码的请求之间，让它们停下等这段预填充算完，这段等待随提示长度增长。

依据：paper/src-v1/1-intro.tex L5-L7；paper/src-v1/3-motivation.tex L7-L23；paper/src-v2/2-background.tex L22；paper/src-v2/3-motivation.tex L13-L57；paper/src-v2/3-motivation.tex L69-L70

### 一块预填充带一批解码

一轮算一块预填充，正在解码的请求与这一轮同批一起算：把这一块和所有正在运行的请求放进同一个批。批里的线性运算（权重与输入的乘法）对预填充块和解码 token 合并成一次矩阵乘，模型权重只从显存读一次，解码不再受显存带宽限制，多出来的开销由预填充分摊。于是预填充分几次算完，解码既没有被整段暂停，每一轮的计算量也由这一块的大小决定。

依据：paper/src-v1/4-design.tex L21-L22；paper/src-v1/4-design.tex L37-L60；paper/src-v2/4-design.tex L24-L28

## 长页

预填充把整段提示一次送进模型、并行处理 token（提示是要模型接着生成的文本，token 是文本经分词得到的最小单位），解码逐个生成后面的 token。模型由很多结构相同的层叠成；[标准注意力](wiki/standard-attention)让位置之间交换信息，每个位置用查询与前面位置的键匹配、取回对应的值（查询、键、值都是位置算出的向量）。[KV cache](wiki/kv-cache)存下已处理位置的键和值供后面复用。
依据：paper/src-v2/2-background.tex LLM Inference Process；paper/src-v2/2-background.tex The Transformer Architecture；paper/src-v1/1-intro.tex L5-L7

解码每步只有一个 token，却要读一遍整套模型权重，瓶颈在显存带宽、算力闲置。两者混在一条流水里时，新请求的预填充插进正在解码的请求之间，打断它们。Chunked Prefill 把一条预填充请求拆成大小相近的若干块、分多次迭代算完，正在解码的请求每轮与当前这一块同批算；结果与整段一次算一样。
依据：paper/src-v1/1-intro.tex L5-L15；paper/src-v2/1-intro.tex L28；paper/src-v2/3-motivation.tex L70；paper/src-v1/3-motivation.tex L23；paper/src-v1/4-design.tex L58；paper/src-v1/4-design.tex L21；paper/src-v2/4-design.tex L15

### 切块不改结果：逐块设掩码

切块要保证结果不变：每一块都要设置[因果掩码](wiki/causal-mask)——块内的每个查询位置只能看它自己以及它在整段提示里前面的位置，不能看后面的位置。第一块的查询只能看本块内的键；随后每一块不但看本块，还要看它之前所有块留下的键和值，这些键和值已经按顺序进了缓存。
依据：paper/src-v1/4-design.tex L21-L28；paper/src-v2/2-background.tex L20-L22

#### 掩码随块移动

一块一块往下算时，掩码窗口跟着块一起往右移。
依据：自行构造

::: 提示 9 个 token、每块 3 个时的掩码
1. 提示长度取 9 个 token，每块 3 个，分成三块：位置 1 到 3、4 到 6、7 到 9。
2. 第一块的查询：位置 1 只看键 1；位置 2 看键 1 到 2；位置 3 看 1 到 3。
3. 第二块的查询：位置 4 看键 1 到 4；位置 5 看 1 到 5；位置 6 看 1 到 6。前三个键来自缓存，后三个键在本块内。
4. 第三块的查询：位置 7 看键 1 到 7；位置 8 看 1 到 8；位置 9 看 1 到 9。
依据：自行构造
:::

#### 逐块算与整段算等价

逐块算为什么与整段算得到一样的结果，关键在于每个位置只依赖它前面的位置。
依据：paper/src-v1/4-design.tex L21-L22；paper/src-v1/2-background.tex The prefill and decode phases

::: 等价性的推导
1. 记位置 $t$ 的查询向量为 $\mathbf{q}_t$，它在这一层算出的键向量和值向量分别为 $\mathbf{K}_t$、$\mathbf{V}_t$。
2. 位置 $t$ 的注意力输出只由 $\mathbf{q}_t$ 和位置 $1$ 到 $t$ 的键值对 $(\mathbf{K}_1,\mathbf{V}_1),\dots,(\mathbf{K}_t,\mathbf{V}_t)$ 决定，掩码保证它看不到 $t$ 之后的位置。
3. 整段一次算时，位置 $t$ 用到的正是 $(\mathbf{K}_1,\mathbf{V}_1),\dots,(\mathbf{K}_t,\mathbf{V}_t)$。
4. 逐块算时，轮到位置 $t$ 所在的块，前面所有块的全部键值、以及本块中 $t$ 及其之前的键值都已算过，掩码又只让 $\mathbf{q}_t$ 看到 $1$ 到 $t$，所以它用到的键值对同样是 $(\mathbf{K}_1,\mathbf{V}_1),\dots,(\mathbf{K}_t,\mathbf{V}_t)$。
5. 对每个位置都成立，逐块算的输出与整段算逐位相同。
依据：自行构造
:::

### 解码与预填充块合成一批

一轮里放一块预填充，其余位置留给正在解码的请求，两者合成同一个批。批里有一个预填充块和若干解码请求，本轮的 token 总数由这一块加上这些解码决定。这种做法建立在迭代级批处理之上：请求可以在每个迭代之后进出批，而不是等一整批全部跑完。解码请求因此不用等整段预填充，只需与当前这一块同批算一轮。
依据：paper/src-v1/4-design.tex L37-L60；paper/src-v2/2-background.tex Scheduling Policies for LLM Inference；paper/src-v2/3-motivation.tex L48-L57

#### 合并线性运算

预填充和解码在各层走的是同一套模型权重。把两者的 token 拼进同一个矩阵后，各层的权重与输入的乘法由两次变成一次矩阵乘：预填充块提供大的行数，解码 token 作为多出来的几行一起乘，权重只从显存读一次就服务了这一批里的所有 token。解码原本把大部分时间花在反复取权重上，合并后这部分开销由预填充分摊，解码由此从内存受限转为计算受限。
依据：paper/src-v1/4-design.tex L56-L60

#### 并入预填充几乎不增时间

再加入一些解码 token，这一轮的时间几乎不变，于是一块预填充能同时推进很多个解码。
依据：paper/src-v2/3-motivation.tex L49-L57

::: 同批解码的每 token 时间
1. 只算预填充的批（1024 个 token，批大小 4），整个批耗时 234.8 毫秒。
2. 只算解码的批（批大小 4），解码每个 token 要 12.49 毫秒。
3. 混合批（1021 个预填充 token 加 3 个解码请求），整个批耗时 238.4 毫秒，其中解码每个 token 只要 1.2 毫秒。
依据：paper/src-v1/tables/tbl-compute-split.tex L11-L16
:::

### 为什么它同时改善吞吐和延迟

两个极端都不好。整段预填充会让正在解码的请求长时间停下，每一轮之间的间隔被拉长到整段预填充的时长；反过来只做解码，批里的算力利用率低，吞吐无法提升。切块把预填充切成小块分散到很多轮里：每一轮的预填充 token 数被块的大小限住，所以某条请求的解码最多只需等一小块，间隔有界；同时每一轮都有一块预填充让算力得到利用，解码随之推进，吞吐提高。
依据：paper/src-v2/3-motivation.tex L66-L80；paper/src-v2/4-design.tex L24-L28

### 块大小与预算怎么定

块大小决定每轮里预填充占多少。块太大，一轮里能同批的解码个数就少；块太小，每块的矩阵乘太小，GPU 利用率不高。块大小要看应用里预填充与解码 token 数之比（预填充 token 数除以解码 token 数）：这个比越小，说明解码占的比重越高，就要用小一些的块让更多解码并入同一个批；比越大，可以用越大的块。每轮允许的 token 总数（token 预算）则按延迟目标定：预算越小每轮越短，但切块开销越大。还有一个硬件细节：GPU 算矩阵乘法时把矩阵切成固定大小的瓦片、分给线程块（GPU 上并行执行的一批线程），各线程块算同样多的运算；矩阵维度不是瓦片大小的整数倍时会有空转，所以块长加上同批的解码数最好取成瓦片大小的整数倍。
依据：paper/src-v1/4-design.tex L68-L94；paper/src-v2/4-design.tex L30-L38

### 切块的开销

每算一块，注意力都要把之前所有块的键和值重新读一遍：越靠前的块被后面每一块重读，重复读取的次数越多。块越小，重读越多，预填充越慢。
依据：paper/src-v1/4-design.tex L25-L29；paper/src-v2/4-design.tex L33-L35

::: 不同块长下的开销
1. 每块 64 个 token 时，注意力的耗时约是整段一次算的 3 倍，整个预填充的耗时约 5 倍。
2. 每块 256 与 512 时，预填充的端到端损失分别控制在 20% 与 10% 以内。
依据：paper/src-v1/5.3-eval-ablation.tex L10-L15
:::

### 适用条件与边界

这套办法成立有一个前提：提示要足够长，切出来的块还得大到能让 GPU 接近满载。生产环境的提示常在 1K 到 4K 个 token，块长取几百个 token 就能保住预填充效率。它提高的是解码侧效率；预填充本身的吞吐不会因此变快，所以整机吞吐的提升在 25% 上下。
依据：paper/src-v1/4-design.tex L14；paper/src-v2/4-design.tex L24；paper/src-v1/5.1-eval-single-gpu.tex L34
