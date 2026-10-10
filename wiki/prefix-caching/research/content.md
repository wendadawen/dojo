# 前缀缓存

- 描述：前缀缓存把已处理请求的 KV cache 留下来，新请求若与旧请求共享前缀，就直接复用这段 KV、跳过它的重算。
- 摘要：共享前缀的请求不必重算同一段 KV：把已算过的 KV cache 按前缀存起来，新请求做一次长前缀匹配，命中就取用这段缓存，只算没命中的后缀。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 资料

- vLLM Team. Automatic Prefix Caching. vLLM documentation. [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- vLLM Team. Automatic Prefix Caching（design）. vLLM documentation. [https://docs.vllm.ai/en/latest/design/prefix_caching.html](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie 等. SGLang: Efficient Execution of Structured Language Model Programs. arXiv:2312.07104v2, 2024. [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)

## 短页

### 缓存已有的 KV cache，前缀相同就复用

前缀缓存是把已经处理过的请求的 [KV cache](wiki/kv-cache) 留下来的一层缓存：KV cache 是模型逐个生成 token 时每层保留下来的、已经算过的键和值。新请求进来时，如果它的输入与某个旧请求共享一段相同的前缀，那这段前缀的 KV 就可以直接取用，不必重算。
依据：blog/vllm-automatic-prefix-caching.md Introduction；paper/src-v2/sec-radix-attention.tex L5-L7

### 相同的开头被反复重算

很多请求以同一段开头：一个应用的系统提示（请求开头固定附带的设定文本）、固定的少样本示例（提示里固定附带的少量示例输入）、同一会话累积的对话历史、被反复引用的同一段长文档。不利用这一点，每个请求都要把这整段重算一遍，相同的 KV 反复算、反复占显存（显存指 GPU 的内存）。
依据：paper/src-v2/sec-intro.tex L23-L28；blog/vllm-automatic-prefix-caching.md Example workloads；paper/src-v2/sec-radix-attention.tex L5-L6；paper/src-v2/appendix.tex L26

### 按相同开头建索引，命中就取用

把已算过的 KV 按它对应的开头存进一份索引。新请求来时，用它自己的开头去索引里找最长的一段相同开头；命中的那段 KV 直接取用，只算没命中的剩下部分。新算出的 KV 再并进索引，供后面的请求复用。
依据：paper/src-v2/sec-radix-attention.tex L12-L22

## 长页

自回归语言模型一次生成一个 token，token 是文本被分词器切出的最小单位。推理分两段：先对整段输入做一次前向前进（把整段输入送进网络，逐层算一遍），叫预填充；再逐个生成输出 token，叫解码，每一步都基于它前面所有 token。模型由多层网络堆叠，每层为每个位置算出一个隐藏状态向量；注意力是一种在每个位置之间传递信息的机制，它让每个位置由自己的隐藏状态算出键、值两个向量，再用它们和前面各位置的键、值交互，得到这一位置的输出，[因果掩码](wiki/causal-mask) 把可见范围限制在当前位置及其之前。把每层已算过的键、值张量（张量是多维数组，这里每个位置的键、值各是一个向量）留下来供后续步复用，这部分就是 [KV cache](wiki/kv-cache)。
依据：paper/src-v2/appendix.tex Background on the KV Cache；paper/src-v2/sec-radix-attention.tex L7

许多请求的输入以同一段开头：多个请求共用同一段系统提示或同一组少样本示例（提示里固定附带的少量示例输入），同一会话的多轮请求共享不断累积的历史，同一段长文档被反复提问。不利用这一点，每个请求都要把这整段前缀重算一遍，相同的 KV 被反复计算、反复占显存（显存指 GPU 的内存）。前缀缓存把已处理请求的 KV cache 留下来，新请求若与某个旧请求共享前缀，就直接取用那段 KV，只计算没被共享的剩余部分。
依据：blog/vllm-automatic-prefix-caching.md Introduction；blog/vllm-automatic-prefix-caching.md Example workloads；paper/src-v2/sec-radix-attention.tex L5-L6；paper/src-v2/sec-intro.tex L23-L28；paper/src-v2/appendix.tex L26

做法是把已算过的 KV 连同它对应的 token 序列存进一份索引；新请求来时按共享前缀从索引中取用命中的部分，其余部分照常计算，算完的 KV 再并入索引供后续请求复用。
依据：paper/src-v2/sec-radix-attention.tex L12-L22

### 前缀为什么能复用

自注意力里，位置 $t$ 的键和值是由从开头到位置 $t$ 的这段 token 算出来的，不参与和它后面任何位置的交互：位置 $t$ 的输出取位置 $t$ 及之前所有位置的键、值，而位置 $t$ 自身的键、值算出来之后，不会再随后面出现的 token 改变。
依据：paper/src-v2/appendix.tex Background on the KV Cache；blog/vllm-prefix-caching-design.md Overview

两段输入的 token 序列若以同一段 token 开头，则对这段前缀里的每一个位置，它前面（含自己）的 token 都完全相同。每个位置的 KV 只由这些 token 决定，所以这两段输入在这段前缀上的 KV 逐位相同，可以整段复用，只需要为它们在分岔点之后各自不同的后缀另算。
依据：paper/src-v2/appendix.tex Background on the KV Cache

::: 逐位相同是怎么来的
记位置 $t$ 的键、值为 $\mathrm{kv}(x_1, \dots, x_t)$，$x_i$ 是第 $i$ 个 token，$L$ 是两段输入共享前缀的长度。
依据：自行构造

1. 位置 $t$ 的 KV 只用到 $x_1, \dots, x_t$，不依赖任何 $t$ 之后的 token。
依据：自行构造
2. 第一段输入是 $x_1, \dots, x_L, a_1, a_2, \dots$，第二段是 $x_1, \dots, x_L, b_1, b_2, \dots$，它们共享前 $L$ 个 token。
依据：自行构造
3. 对 $t = 1, \dots, L$，两段的 $x_1, \dots, x_t$ 相同，代入同一个 $\mathrm{kv}$ 得到相同的键、值。
依据：自行构造
4. 于是前 $L$ 个位置的 KV 逐位相同，可整段复用；从位置 $L+1$ 起两段不同，后缀各自算。
依据：自行构造
:::

### 在已有缓存里找到可复用的前缀

已算过的 KV 要按它对应的 token 序列组织，新请求来时才能快速找出最长的一段可复用前缀。一种结构是 radix tree（压缩前缀树）：它是一种前缀树，区别是每条边可以挂一段长度不定的 token 序列、而不只是一个元素，因此更省空间。
依据：paper/src-v2/sec-radix-attention.tex L12；paper/src-v2/sec-radix-attention.tex L15-L18

树上从根到某个节点的一条路径对应一段 token 序列，这段序列的 KV cache 就随该节点保存，存成不连续的布局：这段 KV 不是连续放在一起，而是分成许多不连续的小片。
依据：paper/src-v2/sec-radix-attention.tex L15-L18

新请求到来时，把它的输入 token 序列放进树里做前缀匹配，找出能与已有路径重合的最长一段。命中的那段 KV 直接取用、不再计算；没命中的剩余部分照常前向计算，算出的新 KV 再作为一条新路径插入树中。请求处理完成后，它的 KV 留在树里，而不是丢弃。
依据：paper/src-v2/sec-radix-attention.tex L12-L22

::: 一段前缀树随三次请求生长
这个例子里的 KV 按单个 token 复用，token 用编号表示，每条边挂一段 token，一个节点保存从根到它这条路径对应的 KV。
依据：自行构造

1. 请求甲的输入是 10、20、30、40、50。树是空的，整段作为一条边插入，用到的 KV 现算。
依据：自行构造
2. 请求乙的输入是 10、20、30、60。前缀匹配到 10、20、30，这段 KV 复用；剩下的 60 作为新边接在 30 之后，只算 60 的 KV。
依据：自行构造
3. 请求丙的输入是 10、20、70。前缀只匹配到 10、20；70 与 30 在第三个 token 处分岔，从第三个 token 起另算，新增一条 70 的边。
依据：自行构造
4. 三次请求都在树下留下路径，此后任何以 10、20、30 开头的输入都能直接命中这一段。
依据：自行构造
:::

### 缓存以块为单位，只复用整块

KV cache 在显存里不是按整段序列连续存放，而是切成固定大小的块，一个块放固定数量的 token。索引也按块来建：一块的 KV 要被复用，它在序列里对应的这一整块 token 必须已经算完，前缀尾部不满一块的部分不复用。
依据：blog/vllm-prefix-caching-design.md Hashing；blog/vllm-prefix-caching-design.md Example

每块有一个哈希键，用来在缓存里标识自己。键由三部分拼成：父块的哈希值、本块的 token，以及让本块唯一的额外哈希。把父块的哈希也算进去，使一块的键间接包含它前面所有块，于是键相同就代表从序列开头到这一块为止的整段 token 都相同。
依据：blog/vllm-prefix-caching-design.md Hashing

::: 14 个 token 里只有前 8 个命中
块大小 $4$，缓存里已有请求 1 的 KV。
依据：blog/vllm-prefix-caching-design.md Example

1. 请求 2 有 $14$ 个 token，前 $10$ 个与请求 1 相同。
依据：blog/vllm-prefix-caching-design.md Example
2. 这前 $10$ 个覆盖第 1 块、第 2 块（共 $8$ 个）和第 3 块的前 $2$ 个。
依据：blog/vllm-prefix-caching-design.md Example
3. 第 3 块只匹配 $2$ 个 token、不满一块，不命中，于是命中的只有前 $2$ 块、共 $8$ 个 token。
依据：blog/vllm-prefix-caching-design.md Example
:::

### 显存有限：LRU 换出与引用计数

缓存不能无限大：被缓存的 KV 与正在运行的请求共用同一块显存池，没有为缓存单独预留固定空间。显存不够用时，系统换出缓存里最久没用到的部分，留给新的请求。
依据：paper/src-v2/sec-radix-attention.tex L18-L20

正在被请求使用的块不能换出，否则那个请求的复用就落空。每个块记一个引用计数，表示当前有多少个请求在用它；只有计数为零的块才可以换出。
依据：paper/src-v2/sec-radix-attention.tex L18-L20；blog/vllm-prefix-caching-design.md Block Allocation

有一种实现用前缀树组织缓存，换出从最久未用的叶子开始：先移除一个枝末的叶子，它的公共祖先只要还有别的叶子在用就仍然留着，直到这些祖先也变成叶子才轮到它们，这样公共前缀能尽量留在缓存里。另一种实现按块建索引，块排在一个空闲队列里，队列按最近使用时间排序，分配新块时从队首取；若取到的那块还是一块缓存块，就当场换出——从记录块哈希的缓存表里删掉块号、删掉哈希键，之后即使同样的 token 再来也不命中。
依据：paper/src-v2/sec-radix-attention.tex L18-L20；blog/vllm-prefix-caching-design.md Eviction (LRU)；blog/vllm-prefix-caching-design.md Block Allocation

::: 块大小 4、共 10 块时的释放与换出顺序
依据：blog/vllm-prefix-caching-design.md Example

1. 一个请求用完被释放，它释放的 $3$ 块按逆序加入空闲队列尾部：请求的最后一块在这 $3$ 块里先入队、最靠近队首（$3$ 块整体仍在队列尾部）。最后一块要更多 token 才能算出哈希、最不可能被其他请求复用，所以让它比同请求的另两块先被换出。
依据：blog/vllm-prefix-caching-design.md Free；blog/vllm-prefix-caching-design.md Example
2. 新请求进来时，先把它命中的缓存块从空闲队列里移出、引用计数加一，这些块这一轮不会被换出。
依据：blog/vllm-prefix-caching-design.md Block Allocation
3. 其余块从空闲队列头逐个取用，取到的若是缓存块就按前文说的方式当场换出。
依据：blog/vllm-prefix-caching-design.md Block Allocation；blog/vllm-prefix-caching-design.md Eviction (LRU)
:::

### 调度顺序也影响命中率

命中率定义成被命中的输入 token 数除以输入 token 总数，衡量一次请求的输入里有多少 token 不需要重算。
依据：paper/src-v2/sec-radix-attention.tex L50-L52

多个请求排队时，执行顺序会影响命中率。若在互不相关、前缀不同的请求之间频繁切换，刚缓存的段很快被换出，命中率被降低。按已匹配前缀长度排序、优先执行匹配更长的请求，能减少这种来回切换。
依据：paper/src-v2/sec-radix-attention.tex L50-L55

这种顺序等价于对请求构成的前缀树做深度优先访问：一个分支访问完再换下一个分支，已缓存的公共祖先在这段时间里持续被命中、不会被换出。
依据：paper/src-v2/sec-radix-attention.tex L61-L63；paper/src-v2/appendix.tex Proof of the

::: 长前缀优先为什么能逼近最优
设定：一批请求的输入构成一棵前缀树，缓存容量不小于最长请求的长度。
依据：paper/src-v2/appendix.tex Proof of the

1. 树上每条边对应的 KV 至少要算一次，所以全部请求的计算量下界是所有边的 KV 之和。
依据：paper/src-v2/appendix.tex Proof of the
2. 按深度优先顺序访问：一条边第一次被算到时，接着把它整棵子树算完；子树计算期间这条边一直被命中，不会被换出。
依据：paper/src-v2/appendix.tex Proof of the
3. 于是每条边的 KV 只算一次，计算量正好取到下界，命中率取到上界。
依据：paper/src-v2/appendix.tex Proof of the
4. 按已匹配前缀长度排序等价于深度优先顺序：起初缓存为空，先算的请求在树里对应某个节点，从根到它的这一路都被缓存。之后每选一个请求，它都是尚未访问的节点里、与已缓存路径共享前缀最长的那个——也就是挂在已缓存路径某个节点下的下一个节点，正是深度优先会走的那一步。
依据：paper/src-v2/appendix.tex Proof of the
:::

### 省下了什么，代价是什么

前缀缓存省的是预填充：命中的那段输入不必再前向计算。它不省解码，生成输出 token 的部分与有没有共享前缀无关，仍逐个 token 算。当多数时间花在生成回答上（比如输出很长）、或新请求与已有请求不共享前缀时，它不带来收益。
依据：blog/vllm-automatic-prefix-caching.md Limits

算得省：命中的段跳过计算，处理同样请求的总计算量下降。占得少：多个请求共享同一段 KV、不各存一份，同时处理的请求数（批）因此能开得更大。命中率越高，这两个方向的好处越大：批更大、吞吐更高、首 token 延迟更低。
依据：paper/src-v2/sec-eval.tex End-to-End Performance；paper/src-v2/sec-eval.tex Ablation Study

管理这份缓存本身要花一点开销，实测这段开销不到总时间的 0.3%。
依据：paper/src-v2/sec-eval.tex Ablation Study

::: 命中率与开销的数字
1. 在一批任务上，命中率从 50% 到 99% 不等。
依据：paper/src-v2/sec-eval.tex End-to-End Performance
2. 线上服务里，两个模型的命中率分别是 52.4% 和 74.1%，后者首 token 延迟平均降到 1.7 分之一。
依据：paper/src-v2/sec-eval.tex End-to-End Performance
3. 在一段完全没有可复用前缀的负载上，跑 100 个请求用 74.3 秒，其中维护缓存结构只占 0.2 秒。
依据：paper/src-v2/sec-eval.tex Ablation Study
:::

### 适用条件与边界

前缀缓存只在共享前缀多的时候有用。典型场景是同一段长文档被反复提问、同一会话的多轮对话、大量请求共用同一系统提示或同一组少样本示例。反过来，若新请求与已有请求不共享前缀，命中不到任何一段缓存，不带来收益。
依据：blog/vllm-automatic-prefix-caching.md Limits；blog/vllm-automatic-prefix-caching.md Example workloads；paper/src-v2/sec-radix-attention.tex L5-L6；paper/src-v2/appendix.tex L26

命中粒度是块：共享前缀的尾部不满一块时，这部分不复用，要重算。
依据：blog/vllm-prefix-caching-design.md Hashing

缓存要与正在运行的请求争显存，显存紧张时缓存会被换出，命中随之落空。
依据：paper/src-v2/sec-radix-attention.tex L20；blog/vllm-prefix-caching-design.md Eviction (LRU)

一条容易混淆的地方：省下的是预填充的计算与显存——共享段只算一次、只存一份；但缓存本身仍占显存，把已算过的 KV 存下来是要花存储的，并不是零代价。
依据：paper/src-v2/appendix.tex Background on the KV Cache；paper/src-v2/sec-eval.tex End-to-End Performance
