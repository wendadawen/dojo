# HiSparse

- 描述：HiSparse 是稀疏注意力解码时的分层 KV 存储：完整 KV 放在主机内存，GPU 上每个请求只留一块固定大小的热缓冲，每步按 top-k 把缺的条目换入。
- 摘要：稀疏注意力每步只读 top-k 个 KV，但完整 KV 仍占满显存；HiSparse 把全量 KV 放主机内存，GPU 只留热缓冲，用换入 kernel 按需取回，解码并发随之提高。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 资料

- Zhiqiang Xie, Zhangheng Huang, Tingwei Huang. HiSparse: Turbocharging Sparse Attention with Hierarchical Memory. LMSYS Org 博客, 2026-04-10. [https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- SGLang 团队. HiSparse: Hierarchical Sparse Attention（使用文档）. sglang 121aa8dcc72ff81db69b17c80bed2db5513e7ed4. [https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx)
- SGLang 团队. sgl-project/sglang（HiSparse 实现）. 2026. [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)

## 短页

### 把全量 KV 放到主机内存

大模型生成文本时一个 token（文本切出的最小单位）一个 token 地往外写，这个阶段叫解码。
依据：blog/lmsys-sglang-hisparse.md L8-L15

每写一个 token，都要按相关度从前面所有 token 的 KV（每个 token 对应的键和值两组向量）里读取信息，这个计算叫注意力。
依据：blog/lmsys-sglang-hisparse.md L8-L15

KV cache 就是把这些 KV 存下来，不必每步重算。稀疏注意力每步只挑相关度最高的 k 个 KV 参与计算，这 k 个叫 top-k。
依据：blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L1-L10

HiSparse 是针对这种解码的分层存储：完整的 KV 放在主机内存（CPU 一侧的内存）里，GPU 显存里每个请求（一次生成任务）只留一块固定大小的小缓存，叫热缓冲，放经常被用到的那些 KV。
依据：blog/lmsys-sglang-hisparse.md L8-L23；blog/sglang-hisparse-guide.md L1-L10；blog/sglang-hisparse-guide.md L13-L16

### 稀疏省了计算，没省容量

稀疏注意力避免了计算和读写量随上下文急剧上升，但为了快速访问，完整上下文的 KV 仍要留在 GPU 显存里，尽管每步只有一小部分被用到。每个请求占用的显存随长度增长，同时解码的请求越多，显存越不够用，并发往往受显存容量限制，而不是受计算限制。
依据：blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L11-L17

### GPU 只留缓冲，缺的按需换入

每一步选出 top-k 之后，先看这些条目是否已经在热缓冲里，在的直接用；不在的从主机内存拷进来，占用缓冲里最久没被用到的位置。注意力只读缓冲里的 top-k 条。每个请求在 GPU 上只占一块固定大小的缓冲，而不是完整序列，同样的显存能容纳更多请求同时解码。
依据：blog/sglang-hisparse-guide.md L19-L32；blog/sglang-hisparse-guide.md L13-L16；blog/lmsys-sglang-hisparse.md L19-L31

## 长页

大模型生成文本分两个阶段。预填充一次处理整段输入，解码在这之后逐个 token（文本切出的最小单位）往外写。解码时每个 token 在模型里有查询向量、键、值三个向量：用当前 token 的查询向量和前面各 token 的键比相关度，再按得分读取它们的值，这个计算叫注意力；键和值合称 KV。为了不每步重算，把它们存下来，这块存储叫 KV cache，放在 GPU 的高带宽显存（HBM）里，序列越长，KV cache 越大。
依据：blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L1-L10

稀疏注意力的想法是不读全部 KV，每步按注意力分数挑出 top-k 个最相关的 token，只用它们做注意力，避免计算和读写量随上下文长度急剧上升。DeepSeek Sparse Attention（DSA）就是这一类，选哪几个 token 由模型本身完成。
依据：blog/lmsys-sglang-hisparse.md L8-L15；blog/sglang-hisparse-guide.md L1-L10；blog/sglang-hisparse-guide.md L26

稀疏注意力省了计算，却没有省容量：为了快速访问，完整上下文的 KV 必须全部留在 GPU 的 HBM 里，尽管每步只有一小部分被用到。显存先被 KV 占满，并发就受限，整体吞吐也随之受限，这往往是容量瓶颈而不是计算瓶颈。HiSparse 是 SGLang（一个大模型推理框架）里的功能，它把完整的 KV 放到主机内存（CPU 一侧的内存），GPU 上每个请求只留一块热缓冲，缓冲里放经常被用到的 KV；每步选出 top-k 后，缓冲里没有的条目再从主机拷进来，同样的显存因此可以容纳更多请求。
依据：blog/lmsys-sglang-hisparse.md L8-L21；blog/sglang-hisparse-guide.md L1-L18

### 容量瓶颈：稀疏注意力为什么仍受显存限制

完整上下文的 KV 全部留在显存里，每个请求占用的显存随序列变长而增加，这限制了可达到的并发数和整体吞吐。博客的测试观察到，没有 HiSparse 的基线吞吐很早就停止增长。
依据：blog/lmsys-sglang-hisparse.md L8-L15；blog/lmsys-sglang-hisparse.md L15；blog/sglang-hisparse-guide.md L11-L17

::: 一个小数字的例子
设一台机器除去模型权重（模型自身的参数，同样占显存）后，剩 80 GB 显存供 KV 使用，每个 token 的 KV 占 0.1 MB，请求长度是 10 万个 token。
1. 每个请求的 KV 是 $100000 \times 0.1 = 10000$ MB，也就是 10 GB。
2. 并发上限是 $80 / 10 = 8$ 个请求。
3. 若每个请求在 GPU 上只留 5000 个 token 的热缓冲，每个请求占 $5000 \times 0.1 = 500$ MB，GPU 侧可容纳 $80000 / 500 = 160$ 个请求。
4. 这 160 个请求的完整 KV 放在主机内存里，共 $160 \times 10 = 1600$ GB，约 1.6 TB。主机内存只有 1 TB 时，主机侧最多放 $1000 / 10 = 100$ 个，并发上限由主机侧决定。
依据：自行构造
:::

### 两层存储：主机放全量，GPU 放热缓冲

完整的 KV 放在主机的固定内存里，这块内存叫主机池。固定内存是被锁定在物理内存、不会被换出的主机内存，GPU 可以直接从它拷贝数据。GPU 上每个请求有一块热缓冲。文档把 GPU 一侧的池称为 device pool，主机池与它的容量之比由 `host_to_device_ratio` 决定。缓冲以页为单位分配，页是若干个连续 token 的位置。供替换的位置最多 `device_buffer_size` 个，另有末尾保留的一页，专门放新生成的 token，不参与替换。在 DeepSeek V4 之外的模型上，序列还短时只按页分配所需的部分，凑满 `device_buffer_size` 之后再加上保留页。
依据：blog/lmsys-sglang-hisparse.md L19-L23；blog/sglang-hisparse-guide.md L7；blog/sglang-hisparse-guide.md L35；blog/sglang-hisparse-guide.md L38；blog/sglang-hisparse-guide.md L109；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L226-L233；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L504-L583；repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L322-L330

### 解码一步：从 top-k 到注意力

模型由若干层依次计算，输入从第一层算到最后一层叫一次前向。若干层共用同一份 top-k 的情形叫共享索引；没有共享索引时，每层各自选 top-k、各自换入。文档把一步的流程写成：
1. 前向：生成下一个 token。
2. 选 top-k：每层按注意力分数，挑出最相关的前文 token，top-k 个条目就是这些 token 的 KV。
3. 换入：由一个 GPU kernel 把 top-k 里不在热缓冲的条目从主机拷进缓冲。kernel 指在 GPU 上并行执行的一段函数。
4. 注意力：用 top-k 条目在缓冲里的位置做注意力计算。
5. 备份：把上一个 token 的 KV 从显存异步拷回主机。异步指拷贝在后台进行，不阻塞后面的计算。备份的目的是让换入 kernel 之后还能从主机取回这个 token。
依据：blog/sglang-hisparse-guide.md L19-L32；blog/sglang-hisparse-guide.md L121-L126；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1027-L1034；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1104-L1114；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L724-L760

### 换入：命中、LRU 与未命中

#### 短序列快路径与命中检测

序列长度不超过 `device_buffer_size` 时，所有 token 都在缓冲里，且位置与 token 一一对应，top-k 直接查位置就行，不需要从主机拷贝。序列超过 `device_buffer_size` 之后，换入 kernel 先把这一步的 top-k 放进一张哈希表，再扫描缓冲里的每个位置：位置上的 token 在哈希表里，就是命中，该位置本步要用；不在，就是可淘汰的位置。如果 top-k 包含最新 token，它直接绑定到缓冲末尾那一页，按命中处理，不参与 LRU 替换。
依据：repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L384-L445；repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L484-L545

#### LRU 重排与未命中拷贝

LRU 是最近最久未使用的替换策略。每个请求在每一层各保存一个位置顺序，最前面是最久没用的位置，最后面是最近用过的。扫描完命中之后，没被选中的位置保持原来的相对顺序，作为可淘汰位置；top-k 里没有出现在缓冲里的条目是未命中。第 $i$ 个未命中占用排在最前面的第 $i$ 个可淘汰位置，由 GPU 上的线程把它从主机拷进这个位置。换完之后，位置顺序重排：剩下没被淘汰的位置留在最前，新换入的位置排在它们后面，命中的位置排在最后，即最近使用端。未命中越少，换入的拷贝就越少。
依据：repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L513-L721；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L279-L283；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1034-L1038

::: 一个手算的例子
设缓冲有 4 个位置 s0 到 s3，里面分别是 token 10、20、30、40，位置顺序是 s0、s1、s2、s3（s0 最久没用）。这一步 top-k 选中 token 20、40、50。
1. 命中：20 在 s1，40 在 s3。
2. 可淘汰位置：s0（装着 10）和 s2（装着 30），按原来的顺序，s0 在前。
3. 未命中：50。它占用最前面的可淘汰位置 s0，把 token 50 从主机拷进 s0。
4. 重排：没被淘汰的可淘汰位置 s2 排最前，新换入的 s0 其次，命中的 s1、s3 排最后。
5. 新的位置顺序是 s2、s0、s1、s3。
依据：自行构造
:::

#### 共享索引时的提前换入

共享索引时，被复用的那一层叫起点层，复用它的叫跳过层。跳过层要换入哪些条目，起点层选完 top-k 就已知道。起点层的换入 kernel 因此记下未命中计划，即哪些主机位置拷到缓冲的哪些位置；各层缓冲的位置按层对齐，每个跳过层因此只需回放这份计划，并提前在侧流上发出。侧流是 GPU 上与主计算流并行的另一条任务队列，拷贝由此与起点层到该跳过层之间各层的计算重叠，不占关键路径。该预取对符合条件的模型自动启用（不使用流水线并行、不使用投机解码），可用 `SGLANG_DISABLE_HISPARSE_PREFETCH=1` 关闭。
依据：blog/sglang-hisparse-guide.md L121-L126；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L1088-L1152；repo/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh L657-L659

### 与 PD 分离结合：KV 直达主机

PD 分离指把预填充和解码放在不同的实例上，预填充算完之后，要把 KV 传给解码实例；预填充和解码在同一批 GPU 上进行，则叫合并部署。RDMA 是网卡直接读写对端内存的传输方式，不经过对端 CPU。在 HiSparse 下，预填充实例通过 RDMA 把 KV 直接写进解码实例的主机池，绕过解码侧的 GPU。这样避免了传输过程中解码侧显存的瞬时占用高峰，也省去了先到 GPU、再由 GPU 拷回主机池的那次拷贝（staging）。解码实例随后为请求分配热缓冲，再按需换入。预填充侧不需要改动：预填充实例不用开 HiSparse，也不知道它的存在。
依据：blog/sglang-hisparse-guide.md L33-L47；blog/sglang-hisparse-guide.md L17；blog/sglang-hisparse-guide.md L198；blog/lmsys-sglang-hisparse.md L19-L23；repo/sglang/python/sglang/srt/managers/hisparse_coordinator.py L406-L449

### 参数怎么选

HiSparse 的参数写在 `--hisparse-config` 里。`top_k` 是每步选出的条目数。`host_to_device_ratio` 按主机可用内存来配：文档的例子是主机内存约 1TB 配 5，约 2TB 配 10。另有 `swap_in_block_size`，默认 960，是换入 kernel 的线程块大小（一个线程块是 GPU 上一起执行的一组线程）。
依据：blog/sglang-hisparse-guide.md L78-L119；blog/sglang-hisparse-guide.md L196-L206

### 效果与代价

在博客的测试里，用 HiSparse 之后吞吐随并发近线性增长。收益来自并发能提高，代价是额外的主机到 GPU 的拷贝。
依据：blog/lmsys-sglang-hisparse.md L8-L17；blog/lmsys-sglang-hisparse.md L15

::: 博客给出的数字
1. 吞吐随并发的变化：GLM-5.1-FP8 模型（一个 DSA 架构的大语言模型，FP8 是 8 位浮点数精度），32k 输入、8k 输出，PD 合并部署在 8 张 H200 GPU 上，并发 256 时 HiSparse 超过基线吞吐的 3 倍。
2. 不同序列配置：同一个模型，在两个 H20 节点的 PD 分离部署上扫描多种输入输出长度，长上下文场景最高达到 5 倍的吞吐提升。
依据：blog/lmsys-sglang-hisparse.md L8-L17；blog/lmsys-sglang-hisparse.md L33-L42；blog/lmsys-sglang-hisparse.md L74-L75
:::

### 适用条件与边界

HiSparse 目前支持 DSA 架构的模型（例如 DeepSeek-V3.2、GLM-5.1）以及 DeepSeek V4 和 MiniMax M3，这些模型都原生选出 top-k 个 token 做注意力，使得只把 top-k 对应的 KV 放在 GPU 上成为可能。它面向高并发场景：并发低时，稀疏加载带来的额外 IO 开销大于节省的显存。文档称这样做没有精度损失（without accuracy loss）。
依据：blog/sglang-hisparse-guide.md L1-L10；blog/lmsys-sglang-hisparse.md L8-L17；blog/lmsys-sglang-hisparse.md L74-L79
