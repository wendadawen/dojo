# DeepEP

- 描述：DeepEP 是 MoE 专家并行的通信库，用一个群里各颗 GPU 之间的 all-to-all 内核，把 token 按路由送到专家所在的卡，再把各专家的输出送回原卡求和。
- 摘要：MoE 每层都要把 token 发到专家所在的 rank 再收回来；DeepEP 用高吞吐与低延迟两套 CUDA 内核做这件事，训练与 prefill 走分层转发，decode 走纯 RDMA 与固定槽位。
- 主题：并行与通信, 推理系统
- 标签：并行与通信

## 长页

- 开头｜一句定义 token、MoE、专家、专家并行、rank、隐藏维、dispatch/combine；DeepEP 是什么；为何要专门做搬运｜250｜blog/deepep-readme-v2.md L3；blog/deepep-legacy-v1.md L3-L13
- ### 把 token 送到专家所在的 rank｜本章回答 dispatch 与 combine 在搬什么、要满足什么要求，内容都在小节里｜0｜blog/deepep-readme-v2.md L3
- #### 从路由到发送计划｜一句定义路由、门控、top-K；门控给每个 token 选出专家编号与权重，二者对应到 rank，由此算出每个 rank 的发往清单｜200｜blog/deepep-legacy-v1.md L165-L169；repo/DeepEP/deep_ep/buffers/ep.py L593-L600
- #### dispatch 与 combine 搬什么、量有多大｜dispatch 复制 K 份发往专家所在 rank；combine 把各专家输出按原 rank 求和送回；载荷可为 FP8，量随 K 与隐藏维增长｜280｜blog/deepep-readme-v2.md L158-L282；blog/deepep-legacy-v1.md L116-L228
- #### 接收侧要什么布局｜接收侧按本地专家分成连续段、并知道每段多少 token 才能送进计算核；内核给出每本地专家的接收计数｜250｜blog/deepep-readme-v2.md L162；repo/DeepEP/deep_ep/buffers/ep.py L43-L52
- #### 与计算重叠的延迟收尾｜通信调用先发出、把等结果推迟到用之前；一句说明延后收尾与事件，数据就绪再取结果｜200｜blog/deepep-readme-v2.md L283-L289；blog/deepep-readme-v2.md L261-L277
- #### DeepEP 只管搬运｜路由与专家计算由调用方负责；库提供延迟收尾、事件等重叠机制，怎么排布由调用方安排｜150｜blog/deepep-readme-v2.md L3；blog/deepep-readme-v2.md L332-L337；blog/deepep-legacy-v1.md L3-L13
- ### 高吞吐内核：按网络分层转发｜本章讲训练与 prefill 用的高吞吐内核怎样用两种网络；先一句说明 prefill/decode 是推理两阶段｜0｜blog/deepep-legacy-v1.md Normal kernels with NVLink and RDMA forwarding
- #### 两段转发｜一句定义 NVLink、RDMA、节点；跨节点先经 RDMA 到目标节点、再经 NVLink 转发给专家，两网重叠；逐段字节折叠｜450｜repo/DeepEP/deep_ep/buffers/ep.py L215-L219；repo/DeepEP/deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh L481；blog/deepep-legacy-v1.md L3-L13
- #### 接收数怎么定｜一句定义 CUDA graph；要精确接收数时 CPU 同步、与图重放不兼容，故 decode 另取他法｜180｜blog/deepep-readme-v2.md L162；blog/deepep-legacy-v1.md L171；repo/DeepEP/deep_ep/buffers/ep.py L617-L624
- #### 通信占多少 SM｜搬运占用 SM、与专家计算争抢；SM 数可按带宽模型解析算出（假设均衡门控、group-limited gate 不适用）；推导与带宽数字折叠｜220｜repo/DeepEP/deep_ep/buffers/ep.py L431-L456；blog/deepep-readme-v2.md L39
- ### 低延迟内核：固定槽位与接收 hook｜本章讲 decode 用的低延迟内核；一句说明 decode 与固定槽位｜0｜blog/deepep-legacy-v1.md Low-latency kernels with pure RDMA
- #### 固定槽位布局｜接收张量按配置的最大容量预留、不等真实计数；布局形状折叠｜300｜blog/deepep-legacy-v1.md L229-L290；blog/deepep-readme-v2.md L162
- #### 接收 hook｜跨节点走纯 RDMA；一句定义 hook：调用只发请求、返回后调 hook 才算就绪，等待不占 SM｜240｜blog/deepep-legacy-v1.md L229-L290
- #### 代价｜固定槽位按最坏预留、显存占用大；带宽与延迟表条件不同、逐行引用，表与条件折叠｜250｜blog/deepep-legacy-v1.md L15-L39；blog/deepep-legacy-v1.md L247-L248；blog/deepep-readme-v2.md L162
- ### 数值例子：一个两步 EP 系统的往返｜自设 6 rank 两组节点、6 专家、隐藏维 3、top-2；发送计划、字节、槽位占用、加权归约与代码全折叠｜100｜自行构造
- ### 适用条件与边界｜硬件门槛（Hopper、节点内 NVLink、跨节点 RDMA）；何时用哪套内核；负载不均的冗余与重路由由调用方做；常见误解｜300｜blog/deepep-readme-v2.md L37-L41；blog/deepep-readme-v2.md L43-L57；blog/deepep-legacy-v1.md L41-L53；blog/deepep-legacy-v1.md L9-L11；blog/deepep-readme-v2.md L332-L337；blog/deepep-readme-v2.md L450-L480

## 短页

- ### 把 token 送到专家所在的 rank｜是什么：DeepEP 是 MoE 专家并行的通信库；各卡放一部分专家，每层把 token 发到专家所在卡、算完送回｜280｜blog/deepep-readme-v2.md L3；blog/deepep-legacy-v1.md L3-L13
- ### 跨卡收发难在哪｜跨卡收发难在哪：每层都跨卡收发；内核要给出每本地专家的接收计数、支持低精度载荷、还占 SM｜280｜blog/deepep-legacy-v1.md L3-L13；blog/deepep-readme-v2.md L162
- ### 两套内核，分层转发或纯 RDMA｜怎么做：训练与 prefill 用高吞吐内核，decode 用低延迟内核（跨节点纯 RDMA、预留固定槽位、hook 等候）｜280｜blog/deepep-legacy-v1.md L3-L13；blog/deepep-legacy-v1.md Performance
