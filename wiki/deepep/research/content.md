# DeepEP

- 描述：DeepEP 是一个 GPU 通信库，把 MoE（混合专家）模型每层里的 token（模型处理的最小单位）送到它用到的专家所在的 GPU，再把结果收回。
- 摘要：MoE 每层都要把 token 发到专家所在的卡再收回来。DeepEP 用两套内核做这次搬运：训练与 prefill 用高吞吐的一套，decode 用低延迟的一套。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 资料

- DeepSeek-AI. DeepEP: an efficient expert-parallel communication library. GitHub. [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- DeepSeek-AI. DeepEP V1 (Legacy) documentation. GitHub. [https://github.com/deepseek-ai/DeepEP/blob/b306af0/docs/legacy.md](https://github.com/deepseek-ai/DeepEP/blob/b306af0/docs/legacy.md)

## 短页

### 把 token 送到专家所在的 rank

混合专家（[MoE](wiki/moe-serving)）模型里有一组专家，每个 token 只交给其中少数几个处理，用哪几个由门控网络（为每个 token 选专家的模块）选出。专家并行把这些专家分到不同 GPU 上：一个通信组里的每颗 GPU 叫一个 rank，各 rank 放一部分专家。DeepEP 负责把 token 发到它用到的专家所在的 rank，这一步叫 dispatch；专家算完，再把各专家的输出送回 token 原来的 rank，这一步叫 combine。
依据：blog/deepep-readme-v2.md L3；blog/deepep-legacy-v1.md L3-L13

### 跨卡收发的难点

MoE 的每一层都要做一次 dispatch 和一次 combine。接收侧要知道每个本地专家各收到多少 token，这个数取决于别处门控网络的结果，事先算不出。载荷常压成 FP8（8 位浮点）这类低精度。搬运要占 GPU 的 SM（流式多处理器，GPU 的执行单元），与正在算的专家计算争同一批资源。
依据：blog/deepep-readme-v2.md L162；blog/deepep-readme-v2.md L3；blog/deepep-readme-v2.md L39

### 两套内核，分层转发或纯 RDMA

DeepEP 用两套内核分别服务推理（模型读入输入、逐步算出输出）的两个阶段；内核指在 GPU 上并行执行、占若干 SM 的一段程序。训练与 prefill（把整段输入一次算完）用高吞吐内核：跨节点那一段走 RDMA（跨节点的直接内存访问网络），到目标节点后再用节点内 NVLink（GPU 之间的直连）转发给专家。decode（逐个 token 生成）用低延迟内核：都走 RDMA，接收张量（多维数组）按可能的最大量预留好、不等真实计数，网络请求先发出、返回一个 hook（一个稍后再调用的函数），等要用数据时调它确认数据到达，这段等待不占 SM。
依据：blog/deepep-legacy-v1.md L3-L13；blog/deepep-legacy-v1.md L11；blog/deepep-legacy-v1.md L229-L290

## 长页

混合专家（[MoE](wiki/moe-serving)）模型里有一组专家，每个 token 只交给其中少数几个处理，具体用哪几个由门控网络（为每个 token 选专家的模块）选出。token 是模型处理的最小单位。这些专家被分放在不同的 GPU 上，这种把专家分到各卡的并行方式叫专家并行。一个通信组里的每颗 GPU 叫一个 rank，每个 rank 负责一部分专家。MoE 的每一层算下来要搬两次数据：把 token 发到它用到的专家所在的 rank，专家算完再把结果送回 token 原来的 rank。这两步合称 MoE 的 dispatch 与 combine，是 DeepEP 要解的通信问题。每个 token 带一个长度等于隐藏维的向量，隐藏维就是 token 向量的维度。
依据：blog/deepep-readme-v2.md L3；blog/deepep-legacy-v1.md L3-L13

### 把 token 送到专家所在的 rank

#### 从路由到发送计划

门控网络为每个 token 从所有专家里选出 K 个，K 叫 top-K，也就是每个 token 用到的专家个数；这 K 个专家各有一个权重。每个专家编号唯一对应它所在的 rank，所以拿到门控给出的专家编号与权重，就能算出本 rank 上每个 token 该发往哪些 rank。DeepEP 的 dispatch 接口收 token、专家编号、权重这三样，发送侧据此算出每个 rank 要收哪些 token 的清单。
依据：repo/DeepEP/deep_ep/buffers/ep.py L589-L600；blog/deepep-legacy-v1.md L165-L169

#### dispatch 与 combine 搬什么、量有多大

dispatch 把每个 token 复制成 K 份，每份发到对应专家所在的 rank，每份带着那个专家的权重。同一份 token 可能留在自己的 rank，也可能送到别的 rank。专家算完，combine 把各专家的输出送回 token 原来的 rank 相加，得到这个 token 在该层路由专家上的总输出。一个 token 要发 K 份、每份长度是隐藏维，所以每 token 的通信量随 K 和隐藏维线性增长；内核支持把载荷压成 FP8（8 位浮点）再发。
依据：blog/deepep-readme-v2.md L158-L282；blog/deepep-legacy-v1.md L116-L228；blog/deepep-readme-v2.md L3；blog/deepep-readme-v2.md L279

#### 接收侧要什么布局

专家计算在接收侧按本地专家分组做。送进来的 token 必须先按专家排好：同一个本地专家收到的 token 排成连续的一段，各段按对齐补到固定长度；接收侧还要拿到每个本地专家各收到多少 token，才能把每一段单独取出来做矩阵乘。DeepEP 的 dispatch 保证这个布局，并给出每个本地专家的接收计数。
依据：blog/deepep-readme-v2.md L162；repo/DeepEP/deep_ep/buffers/ep.py L43-L52

#### 与计算重叠的延迟收尾

搬运和计算可以重叠。通信调用先把请求发出去，不马上等结果，而是返回一个代表这批通信的对象；调用方接着去做与这批数据无关的计算，到了真要用这批数据时，再在这个对象上等一次、取回结果。
依据：blog/deepep-readme-v2.md L283-L289；blog/deepep-readme-v2.md L261-L277

#### DeepEP 只管搬运

决定路由（token 发给哪些专家）、执行专家的矩阵乘、以及安排搬运与计算的重叠，这三件事都由调用方做。DeepEP 只提供搬运本身，以及让调用方排布重叠用的延迟收尾；路由与专家计算都不在库里。
依据：blog/deepep-readme-v2.md L3；blog/deepep-readme-v2.md L332-L337；blog/deepep-readme-v2.md L261-L289；blog/deepep-legacy-v1.md L3-L13

### 高吞吐内核：按网络分层转发

DeepEP 的内核分两套，对应推理的两个阶段。prefill 把用户给的一整段输入一次算完，对单步延迟不敏感；decode 在其后逐个 token 生成，每生成一个 token 都要经过这层一次。训练与 prefill 用第一套，称高吞吐内核。
依据：blog/deepep-legacy-v1.md L3-L13；blog/deepep-legacy-v1.md L11

#### 两段转发

集群里的网络分两层：同一节点内的 GPU 由 NVLink 直连，带宽高；跨节点要走 RDMA（远程直接内存访问，绕过对方 CPU 直接读写内存），带宽低一些。高吞吐内核用 hybrid（分层）方式把两层都用上：一个 token 要到别的节点上的专家，先经跨节点的 RDMA 发到目标节点，到了目标节点之后，再经该节点内的 NVLink 转发到专家所在的 GPU。分层转发比全程只用一种网络能拿到更高的带宽。
依据：repo/DeepEP/deep_ep/buffers/ep.py L215-L219；repo/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh L481；blog/deepep-legacy-v1.md L3-L13；blog/deepep-legacy-v1.md L17-L26

::: 一份 token 跨节点时各段发多少
设隐藏维 $h=7168$、$K=8$、载荷用 FP8（每元素 1 字节），一个 token 复制 8 份。
1. 每份的长度是 $h=7168$ 个元素。
2. 一份在跨节点段上的字节数 $= 7168 \times 1 = 7168$ 字节。
3. 记这 8 份里有 $m$ 份要去别的节点，$m$ 在 $0$ 到 $8$ 之间；跨节点段的总字节数是 $m \times 7168$。
4. 落在本节点其他 GPU 上的份只走 NVLink，字节数相同。
依据：自行构造
:::

#### 接收数怎么定

接收侧在数据到齐之前不知道本 rank 会收到多少 token，这个数取决于别处门控的结果。要拿到精确的数，就得让 CPU 等一个 GPU 传来的接收计数信号，再据此决定接收张量的形状。把一段 GPU 工作记录好、之后原样重放（CUDA graph）时，中间夹一个 CPU 等待会破坏重放。所以 decode 那种每步都要收发、又常录成图的场景不能走这条取数路线，另一套面向单步延迟的内核改用固定槽位。
依据：blog/deepep-legacy-v1.md L171；repo/DeepEP/deep_ep/buffers/ep.py L617-L624；blog/deepep-readme-v2.md L162

#### 通信占多少 SM

搬运要占 GPU 的 SM（流式多处理器，GPU 上执行线程块的单元；线程块是一组并行执行的线程），这些 SM 在搬运期间不能用于专家计算。占多少按拓扑和通信量估：DeepEP 可以按 RDMA 带宽、NVLink 带宽、单个 SM 的显存读写带宽，直接解析算出需要的 SM 数。这个算法假设门控把 token 均衡分给各专家；若门控先把专家分成若干组、每个 token 先选组再选组内专家，该算法不适用。
依据：repo/DeepEP/deep_ep/buffers/ep.py L431-L456；blog/deepep-readme-v2.md L39

::: 高吞吐内核的带宽
在一台 H800（一款 NVIDIA 数据中心 GPU）上，每批 4096 个 token，隐藏维 7168，先按组选前 4 组、再选前 8 个专家，FP8 dispatch、BF16（16 位浮点）combine。#EP 为该组 rank 数。
| 类型 | dispatch #EP | 瓶颈带宽 | combine #EP | 瓶颈带宽 |
|---|---|---|---|---|
| 节点内 | 8 | 153 GB/s（NVLink） | 8 | 158 GB/s（NVLink） |
| 跨节点 | 16 | 43 GB/s（RDMA） | 16 | 43 GB/s（RDMA） |
| 跨节点 | 32 | 58 GB/s（RDMA） | 32 | 57 GB/s（RDMA） |
| 跨节点 | 64 | 51 GB/s（RDMA） | 64 | 50 GB/s（RDMA） |
依据：blog/deepep-legacy-v1.md L17-L26
:::

### 低延迟内核：固定槽位与接收 hook

decode 用第二套内核，称低延迟内核，它逐 token、对单步延迟敏感。
依据：blog/deepep-legacy-v1.md L11

#### 固定槽位布局

低延迟内核不等真实计数，按配置的最大 token 数先把接收空间预留好。实际收到多少 token、哪些有效，由每个专家的接收计数给出。这块空间按最大量预留，显存占用远大于高吞吐模式，所以配置的最大 token 数要按实际批（batch，一次处理的 token 数）取小。
依据：blog/deepep-legacy-v1.md L229-L290；blog/deepep-legacy-v1.md L247-L248；blog/deepep-readme-v2.md L162

::: 接收空间的容量与计数
decode 走不做 CPU 同步的取数方式，接收空间的容量按配置的最大值分配。专家计算核用每专家接收计数确定有效范围，多出的容量不算数据。
依据：blog/deepep-readme-v2.md L162
:::

#### 接收 hook

低延迟内核用同一个思路，形式是一个 hook：调用时内核只把 RDMA 请求发出去，立刻返回这个 hook；要用数据时再调用它，等数据全部到达才返回。这段等待由网络和调用方完成，不占 GPU 的 SM。
依据：blog/deepep-legacy-v1.md L229-L290

#### 代价

低延迟内核换来的收益是低单步延迟；它按最坏情况预留、显存占用大，而批越大这份预留越大，所以配置的最大 token 数要小。
依据：blog/deepep-legacy-v1.md L247-L248

::: 低延迟内核的延迟与带宽
在一台 H800（一款 NVIDIA 数据中心 GPU）上，每批 128 个 token，隐藏维 7168，top-8 专家，FP8 dispatch、BF16（16 位浮点）combine。
| dispatch #EP | 延迟 | RDMA 带宽 | combine #EP | 延迟 | RDMA 带宽 |
|---|---|---|---|---|---|
| 8 | 77 us | 98 GB/s | 8 | 114 us | 127 GB/s |
| 128 | 192 us | 39 GB/s | 128 | 369 us | 39 GB/s |
| 256 | 194 us | 39 GB/s | 256 | 360 us | 40 GB/s |
依据：blog/deepep-legacy-v1.md L28-L39
:::

### 数值例子：一个两步 EP 系统的往返

以 rank 0 的 2 个 token 为例走一遍（EP 即专家并行）：它们发到哪个 rank、节点 A 上的专家各收到多少、combine 之后回到什么数。
依据：自行构造

::: 例子的设定
4 个 rank 排成两组节点：rank 0、1 在节点 A，rank 2、3 在节点 B。4 个专家 E0–E3，专家 E$e$ 放在 rank $e$。每个 token 的向量长度（隐藏维）为 3，分量取整数；top-K 取 2，即每个 token 送到 2 个专家。每个 rank 持有 2 个 token，这里只列 rank 0 的 a、b。假设专家对输入做恒等变换，即专家输出等于它收到的 token 向量，这样只看得清搬运与求和。
依据：自行构造
:::

::: 发送计划与节点 A 上的专家收到的 token
rank 0 的发送计划：

| token | 向量 | 选中的专家（权重） | 目标 rank | 路径 |
|---|---|---|---|---|
| a | (1,0,0) | E2(0.5)、E3(0.5) | 2、3 | 跨节点 |
| b | (0,1,0) | E0(0.5)、E1(0.5) | 0、1 | 同节点 |

节点 A 上的专家收到来自 rank 0 的 token：

| 专家 | 收到的 token | 接收计数 |
|---|---|---|
| E0 | b | 1 |
| E1 | b | 1 |

接收空间按配置的最大容量预留：本例每个 rank 最多发 2 个 token，组内有 4 个 rank 都是潜在来源，故按最坏 8 个 token 预留，实际只用到少量、多出的空着。
依据：自行构造
:::

::: combine 把各专家输出相加
专家输出等于输入向量，权重在专家计算时已经乘到输出上，combine 把同一 token 的各专家输出送回原 rank 相加。
1. a：$0.5 \times (1,0,0) + 0.5 \times (1,0,0) = (1,0,0)$。
2. b：$0.5 \times (0,1,0) + 0.5 \times (0,1,0) = (0,1,0)$。
依据：自行构造
:::

::: 例子的代码复算
```python
import sys; sys.stdout.reconfigure(newline="")
ranks = {'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3}
tokens = {'a': (1, 0, 0), 'b': (0, 1, 0)}
route = {'a': [('E2', 0.5), ('E3', 0.5)],
         'b': [('E0', 0.5), ('E1', 0.5)]}
hidden, local = 3, ('E0', 'E1')

print('发送计划')
for name, vec in tokens.items():
    for e, w in route[name]:
        path = '同节点' if ranks[e] < 2 else '跨节点'
        print(f'  {name} -> {e} (rank {ranks[e]}) 权重 {w} {path}')

print('节点 A 上的专家收到（来自 rank 0）')
recv = {}
for name in tokens:
    for e, _ in route[name]:
        if e in local:
            recv.setdefault(e, []).append(name)
for e in local:
    got = recv.get(e, [])
    print(f'  {e}: {got} 计数 {len(got)}')

print('combine 结果')
for name, vec in tokens.items():
    out = [0] * hidden
    for e, w in route[name]:
        for i in range(hidden):
            out[i] += w * vec[i]
    print(f'  {name}: {tuple(out)}')
```

```text
发送计划
  a -> E2 (rank 2) 权重 0.5 跨节点
  a -> E3 (rank 3) 权重 0.5 跨节点
  b -> E0 (rank 0) 权重 0.5 同节点
  b -> E1 (rank 1) 权重 0.5 同节点
节点 A 上的专家收到（来自 rank 0）
  E0: ['b'] 计数 1
  E1: ['b'] 计数 1
combine 结果
  a: (1.0, 0.0, 0.0)
  b: (0.0, 1.0, 0.0)
```
依据：自行构造
:::

### 适用条件与边界

DeepEP 不管 token 分给谁。门控若分得不均，把重载专家的计算分散到复制出的实例上、或把 token 重路由，都由调用方负责，库不介入。
依据：blog/deepep-readme-v2.md L332-L337

硬件上有几道门槛：要用 Hopper（NVIDIA 的一代数据中心 GPU 架构）及更新的 GPU、节点内要有 NVLink、跨节点要有 RDMA 网络；网络需为 InfiniBand（一种 RDMA 网络）或兼容的 RoCE。
依据：blog/deepep-readme-v2.md L43-L57；blog/deepep-readme-v2.md L452-L473

低延迟内核的固定槽位按最坏情况预留，批大时显存不够用，所以配置的最大 token 数要小。
依据：blog/deepep-legacy-v1.md L247-L248
