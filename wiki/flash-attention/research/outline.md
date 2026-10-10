# FlashAttention

- 描述：FlashAttention 是一种精确注意力算法，用分块与重算把中间结果留在 GPU 片内，减少慢速显存的读写，从而在序列变长时更快、更省显存。
- 摘要：把注意力按块搬进片内 SRAM，用流式 SoftMax 逐块归约、反向按需重算注意力矩阵，访问慢速显存的次数少于标准注意力，结果与标准注意力在数学上相同（精确而非近似）。
- 主题：注意力机制, 推理系统
- 标签：推理加速

## 长页

- 开头｜它是什么、为什么需要、大概怎么做；交代注意力并链接它的页面，给出 $N$、$d$、$M$ 的含义，用一句给出 SoftMax，交代反向传播｜500｜paper/src-v2/src/abstract.tex L2-L10；paper/src-v2/src/intro.tex Introduction；paper/src-v2/src/algo.tex L28-L38；paper/src-v2/src/background.tex Hardware Performance
- ### 注意力的瓶颈在哪｜只有标题，内容在三小节｜0｜paper/src-v2/src/background.tex Hardware Performance
- #### 更快的存储更小｜两级存储的带宽与容量差（以 A100 为例），快的容量只有上百 KB、带宽高一个量级，访存要经这两级；访存字节数的估算折叠｜200｜paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/figs/banner.tex L1-L21
- #### 计算受限与访存受限｜算术强度的定义，矩阵乘（内维较大时）属计算受限、逐元素与归约（如 SoftMax）属访存受限，算力相对访存越来越快｜180｜paper/src-v2/src/background.tex Hardware Performance
- #### 标准注意力的访存开销｜标准实现把中间数组写回慢存储：算分数写一次、SoftMax 读写一次、再乘值矩阵；访问次数的量级随序列长度平方增长；各步读写量折叠｜280｜paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/theory.tex L13-L24；paper/src-v2/src/proofs.tex L85-L105
- ### 分块算精确注意力｜只有标题，内容在三小节｜0｜paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation
- #### 分块与 SoftMax 整行归一化的冲突｜SoftMax 沿一整行做，一个位置要看到这行全部分数才知道权重，所以不能只拿一块算；把一行切成两段，最大值、指数和、权重怎样合并｜340｜paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation；paper/src-v2/src/algo_details.tex Memory-efficient forward pass
- #### 流式 SoftMax 的分块合并示例｜四个分数分两块，逐块更新最大值与指数和、再合成权重并给原结果重新缩放；全程数值折叠｜160｜自行构造
- #### 分块算法｜块的尺寸怎么按片内容量定，外层扫键值块、内层扫查询块，在片内算分数块、更新统计与输出块；每步公式折叠｜360｜paper/src-v2/src/algo.tex L85-L116；paper/src-v2/src/algo_details.tex Forward Pass
- ### 重算替代存储｜只有标题，内容在两小节｜0｜paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation
- #### 省下什么、多算什么｜反向传播要用分数与概率两个 $N\times N$ 矩阵本身算查询、键、值的梯度；只留输出与 SoftMax 统计，反向时按块重算分数块；公式折叠｜380｜paper/src-v2/src/algo.tex L64-L74；paper/src-v2/src/algo_details.tex Memory-efficient backward pass；paper/src-v2/src/algo_details.tex Backward Pass
- #### 重算为什么反而更快｜多算的是矩阵乘，省下的是读写 $N\times N$ 数组的访存，后者才是耗时的部分；重算把注意力矩阵按块留在片内｜320｜paper/src-v2/src/algo.tex L64-L74；paper/src-v2/src/theory.tex L52-L64；paper/src-v2/src/algo_details.tex Comparison with
- ### 访问次数与下限｜只有标题，内容在三小节｜0｜paper/src-v2/src/theory.tex Analysis: IO Complexity of
- #### 算法要搬多少字节｜沿用上节的块尺寸，键与值各读一次、查询重复读若干遍，访问次数与 $N$、$d$、$M$ 的关系；趟数推导折叠｜260｜paper/src-v2/src/theory.tex L13-L34；paper/src-v2/src/proofs.tex L107-L145
- #### 没有更好的精确算法｜存在一个下界，任何精确注意力算法在各档片内容量上都改进不了这个量级｜160｜paper/src-v2/src/theory.tex L36-L50；paper/src-v2/src/proofs.tex L149-L164
- #### 块多大合适｜块越大趟数越少、访存越少，运行时随之下降；块大到某个值以后运行时改由计算主导｜180｜paper/src-v2/src/theory.tex L52-L64
- ### 快了多少、能干什么｜只有标题，内容在两小节｜0｜paper/src-v2/src/experiments.tex Experiments
- #### 运行时与访存｜前向加反向的耗时与访存字节对比，显存占用随序列长度线性增长；数字表折叠｜280｜paper/src-v2/src/theory.tex L66-L101；paper/src-v2/src/experiments.tex Benchmarking Attention；paper/src-v2/tables/memory_use.tex L1-L27
- #### 能放更长的上下文｜承接上一小节：显存与耗时的改善让同一硬件能容纳更长的序列；不列各系统、各任务的对比｜140｜paper/src-v2/src/abstract.tex L2-L10；paper/src-v2/src/experiments.tex Better Models with Longer Sequences；paper/src-v2/src/theory.tex L20-L24
- ### 适用条件与边界｜分块成立的假设、代价与常见误解：更快指墙钟时间而非浮点运算次数、SoftMax 分块依赖数值稳定的缩放、要为每种注意力单独写实现、加速倍率随序列长度变化｜300｜paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation；paper/src-v2/src/theory.tex L52-L64；paper/src-v2/src/experiments.tex Benchmarking Attention；paper/src-v2/src/discussion.tex Limitations and Future Directions

## 短页

- ### 把注意力交给更快的存储｜是什么：一种精确注意力算法，结果与标准注意力在数学上相同，链接标准注意力的页面；改动的是在 GPU 上怎么算，把数据分块搬进片内｜300｜paper/src-v2/src/abstract.tex L2-L10；paper/src-v2/src/algo.tex L28-L38
- ### 卡在搬运上，不在算数上｜为什么需要：两级存储带宽差一个量级，中间那个 $N\times N$ 的矩阵（$N$ 是序列长度）反复进出慢存储，浮点运算次数少不等于墙钟快｜360｜paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/intro.tex L12-L14
- ### 把中间矩阵留在片内｜怎么做：让 $N\times N$ 的中间矩阵不落回慢存储；查询、键、值分块搬进片内，用流式 SoftMax 逐块归约，反向按块重算；三个词在节内各讲一句｜320｜paper/src-v2/src/algo.tex An Efficient Attention Algorithm With Tiling and Recomputation
