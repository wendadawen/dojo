# KV cache 布局

- 描述：KV cache 布局指每一层 key、value 缓存的各个维度在内存里的排列顺序，两种排法记作 NHD 和 HND，区别是序列位置维和头维谁排在前。
- 摘要：同一份 KV cache，NHD 把序列位置排在头之前，同一个位置的各个头相邻；HND 把头排在序列位置之前，同一个头在整段序列上的向量相邻。布局决定读一段缓存时拿到的是同一个位置的别的头，还是同一个头的相邻位置。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 资料

- FlashInfer 官方文档. KV-Cache Layout in FlashInfer. [https://docs.flashinfer.ai/tutorials/kv_layout.html](https://docs.flashinfer.ai/tutorials/kv_layout.html)
- vLLM 官方 RFC. Standardize KV-cache Layouts. [https://github.com/vllm-project/vllm/issues/42082](https://github.com/vllm-project/vllm/issues/42082)
- FlashInfer 源码. include/flashinfer/layout.cuh. [https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/layout.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/layout.cuh)
- vLLM 源码. vllm/v1/kv_cache_layout.py. [https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_layout.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_layout.py)
- vLLM 源码. vllm/config/cache.py. [https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)
- vLLM 源码. vllm/v1/attention/backends/utils.py. [https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py)

## 短页

### 缓存里各维怎么排

生成时模型依次产生一个个 token（token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分）。模型由若干结构相同的层叠成，每层里都有注意力，它把每个位置的输入投影成 query、key 和 value 三个向量（见[标准注意力](wiki/standard-attention)）。每层都把前面位置的 key 和 value 留在显存（GPU 做并行计算时用的内存）里复用，这就是 [KV cache](wiki/kv-cache)。一层的缓存分成若干个页，一页里是一段序列位置，每个位置有一组头（注意力把每个位置的向量分成若干份，每份叫一个头），每个头有一小段向量。布局指的就是这些维度的先后，两种排法各用一串字母表示分给每页的那三维怎么排：NHD 的三个字母依次对应序列位置、头、每头向量，每个位置里各个头相邻；HND 把头排到最前，同一个头在各个位置上的向量相邻。
依据：blog/flashinfer-kv-layout.rst L9-L12；blog/flashinfer-kv-layout.rst L123-L129

### 读一段缓存先读到什么

算注意力要同时用当前位置和缓存里的全部历史位置（见[标准注意力](wiki/standard-attention)）：缓存里的数是连续存放的，一次读进来的是沿某一维相邻的一段，这一段是哪些数由布局决定。NHD 下读到的是同一个位置的各个头，HND 下读到的是同一个头在相邻位置上的向量。
依据：repo/flashinfer/layout.cuh L28-L48；blog/flashinfer-kv-layout.rst L11-L12

### 选 NHD 还是 HND

两种布局排的是同样那批数，只是维度换了先后。选 NHD，是因为它和算 key、value 的那步矩阵乘法输出的排列一致，可以直接拿来做缓存，不必转置；选 HND，是因为缓存用低精度格式（每个数占的位数比常规少的浮点格式，例如 fp8）时它对 GPU（做并行计算的处理器）实现更友好。在 fp16（与 fp8 同类、每个数占的位数更多的一种浮点格式）缓存上两种布局看不出明显的性能差别，默认取 NHD。
依据：blog/flashinfer-kv-layout.rst L14-L20

## 长页

生成时模型依次产生一个个 token（token 是模型读写文本的最小单位，可以是一个词，也可以是词的一部分）。模型由若干结构相同的层叠成，每层里都有注意力，它把每个位置的输入投影成 query、key 和 value 三个向量（见[标准注意力](wiki/standard-attention)）。每层都把前面位置的 key 和 value 留在显存（GPU 做并行计算时用的内存）里给后面的 token 复用，这就是 [KV cache](wiki/kv-cache)。一层的缓存按页、序列位置、头、每头向量这几维组织，页是一段序列位置，头是注意力对每个位置向量的划分。布局就是这些维度的排列顺序，NHD 和 HND 是其中两种：NHD 的三个字母依次对应序列位置、头、每头向量，HND 把头排到最前。排列不同，读相邻数据时拿到的是哪一维就不同。
依据：blog/flashinfer-kv-layout.rst L9-L20；repo/flashinfer/layout.cuh L28-L48；repo/vllm/v1_kv_cache_layout.py L11-L28

### 排在后三维里的两个维度

一层的缓存里，缓存的维度按页、序列位置、头、每头向量排成四维，后三维是序列位置、头、每头向量。布局只在这后三维中的序列位置和头之间做选择，页维和每头向量维在两种布局里都不动。
依据：repo/vllm/v1_kv_cache_layout.py L11-L28；blog/flashinfer-kv-layout.rst L9-L12

#### 两种顺序的形状

缓存按页分配：显存切成固定大小的页，一页放固定的一段序列位置，页再分给不同的序列。把一页里的维度展开，NHD 把序列位置排在最前，写成页、序列位置、头、每头向量；HND 把头排在最前，写成页、头、序列位置、每头向量。
依据：blog/flashinfer-kv-layout.rst L123-L142

取一页最小的情形排一遍。
依据：自行构造

::: 一页里两种顺序的元素排列
设一页放 $T=3$ 个序列位置、有 $H=2$ 个头、每个头 $d=2$ 个数，一页共 $3 \times 2 \times 2 = 12$ 个数。用 $t$ 表示第几个位置，$h$ 表示第几个头，$k$ 表示这个头里第几个数，都从 1 起。
依据：自行构造

1. NHD 的顺序是序列位置、头、数：先列第 1 个位置的各个头，再列第 2 个位置……依次是 $(t,h,k) = (1,1,1), (1,1,2), (1,2,1), (1,2,2), (2,1,1), \dots$。
依据：自行构造

2. HND 的顺序是头、序列位置、数：先列第 1 个头在三个位置上的向量，再列第 2 个头……依次是 $(h,t,k) = (1,1,1), (1,1,2), (1,2,1), (1,2,2), (1,3,1), \dots$。
依据：自行构造

3. 两种顺序的第 1、2 个数相同，都是位置 1、头 1 的两个数；第 3 个数起才分开——NHD 的第 3 个是位置 1、头 2 的第 1 个数，HND 的第 3 个是头 1、位置 2 的第 1 个数。
依据：自行构造
:::

#### 相邻数据是哪一维

NHD 里，一个头向量之后是同一个位置的下一个头；把一个位置的各个头都排完，才到下一个位置。HND 里，一个头向量之后是同一个头在下一个位置上的向量；把一个头在所有位置上的向量都排完，才到下一个头。
依据：repo/flashinfer/layout.cuh L35-L48

从序列里的一个位置走到下一个位置要跨过多少个元素，从一个头走到下一个头又要跨过多少个。相邻的那一维步长小，另一个方向步长大，两种布局下小的是哪一边正好相反。
依据：repo/flashinfer/layout.cuh L45-L46

::: 两种布局的步长
用 $s_n$ 表示沿序列方向走一个位置跨过的元素数，$s_h$ 表示沿头方向走一个头跨过的元素数，$d$ 是每头向量的长度，$H$ 是头数，$T$ 是这段缓存里的位置数（一页里就是每页的位置数）。
依据：自行构造

1. NHD 下，一个头向量之后是自己的下一个数，走完一整个头向量才到下一个头，所以 $s_h = d$；走完一个位置的所有头才到下一个位置，所以 $s_n = H \times d$。
依据：repo/flashinfer/layout.cuh L45-L46

2. HND 下换成相邻的是位置：同一个头在各位置上的向量排在一起，走一个位置只跨一个头向量，$s_n = d$；走一个头要跨过整段序列，$s_h = T \times d$。
依据：repo/flashinfer/layout.cuh L45-L46

3. 取 $H=2$、$d=2$、$T=3$：NHD 的 $s_n = 4$、$s_h = 2$；HND 的 $s_n = 2$、$s_h = 6$。同一个缓存，从位置 1 走到位置 2，NHD 要跨 4 个数，HND 只跨 2 个。
依据：自行构造
:::

### 为什么会有两种

算 key、value 的那一步是输入乘投影矩阵，输出的排列本来就是序列位置、头、每头向量这个顺序。NHD 与它一致，投影出来的结果可以直接拿来当缓存，不必转置；HND 不是这个顺序，排成 HND 要多一次换序，但在缓存用低精度数值格式（每个数占的位数比常规少的浮点格式，例如 fp8）时，它对 GPU（做并行计算的处理器）实现更友好。在 fp16（与 fp8 同类、每个数占的位数更多的一种浮点格式）缓存上两种看不出明显的性能差别，默认取 NHD。
依据：blog/flashinfer-kv-layout.rst L14-L20

### 适用条件与边界

这个区分只对多头缓存成立。当注意力把 key、value 合成一个张量（多维数组）、没有头维时（如 [MLA](wiki/mla)），缓存里就没有这一维，也就没有这两种排法之分。
依据：blog/flashinfer-kv-layout.rst L149-L155
