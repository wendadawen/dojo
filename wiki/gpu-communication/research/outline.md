# GPU 通信

- 描述：GPU 通信是让多块 GPU 协作时交换数据的机制：算法上由集合通信把各卡的数据合并（如 ring all-reduce），合并后的字节经机内 NVLink 或跨机网卡与 RDMA 传输。
- 摘要：数据并行训练每步要把各卡算出的梯度合并：用 all-reduce 求出各卡数组的和，再按卡数取平均，使每块卡得到同一份平均梯度。ring all-reduce 把它拆成 scatter-reduce 与 all-gather 两阶段，每卡通信量为 $\frac{2(N-1)}{N}$ 份数据、与 GPU 数无关；字节走的路径机内是 NVLink，跨机是网卡加 RDMA。
- 主题：并行与通信
- 标签：并行与通信

## 长页

- 开头｜讲清 GPU 通信是什么、数据并行训练为什么要每步把各卡梯度合并（即 all-reduce）、以及这些流量跑在哪些互联上｜350｜blog/baidu-ring-allreduce.txt L24-L35；blog/nccl-overview.txt L785-L800
- ### all-reduce 要交换的数据｜讲清 rank 与集合通信的定义、all-reduce 交出什么拿到什么、可拆成 reduce-scatter 与 all-gather 两个原语且各是什么｜550｜blog/nccl-collectives.txt L785-L826；blog/baidu-ring-allreduce.txt L44-L45
- ### 朴素方案为什么不够用｜讲清把所有梯度发给一块卡再广播回去的做法通信量随 GPU 数线性增长、成为训练瓶颈，对比数字折叠｜350｜blog/baidu-ring-allreduce.txt L24-L35
- ### ring all-reduce 的两个阶段｜讲清环拓扑、scatter-reduce 的 N-1 步收发累加、all-gather 的 N-1 步收发覆盖，用一个 3 卡小数组走一遍的过程折叠｜900｜blog/baidu-ring-allreduce.txt L36-L42；blog/baidu-ring-allreduce.txt L43-L92；blog/baidu-ring-allreduce.txt L94-L136
- ### 每块卡要搬多少数据｜讲清每卡通信量公式 $2(N-1)K/N$、有上限 $2K$ 不随卡数线性增长、以及带宽最优性，推导过程折叠｜700｜blog/baidu-ring-allreduce.txt L138-L145
- ### 机内的路径：NVLink｜讲清一块机器内 GPU 之间怎么连：NVLink 直连的带宽与全互联，以及机内多卡 IO 由它承载｜550｜blog/nvlink.txt L570-L580
- ### 跨机的路径：网卡、RDMA 与 GPUDirect｜讲清数据出网必经 PCIe 与网卡、TCP/IP 与 RDMA 的区别、RDMA 读写操作，GPUDirect 直读显存，pinning 代价折叠｜950｜blog/rdma-networks-programming-guide.txt L519-L526；blog/rdma-networks-programming-guide.txt L530-L553；blog/rdma-networks-programming-guide.txt L562-L567；blog/gpudirect-rdma.txt L79-L120；blog/gpudirect-rdma.txt L244-L247
- ### 适用条件与边界｜讲清 ring 只在延迟可忽略、数据量大时最优，受最慢链路限制，延迟不可忽略时不再最优；NVLink 可经 Switch 跨节点；GPUDirect 的前提｜500｜blog/baidu-ring-allreduce.txt L37；blog/baidu-ring-allreduce.txt L145；blog/gpudirect-rdma.txt L81；blog/nvlink.txt L580-L587

## 短页

- ### 它把各卡的数据合并｜讲清 GPU 通信与 all-reduce 是什么：数据并行训练里每块卡一份数据、各存一份模型副本，每步各卡交出数组、都拿到它们的和｜350｜blog/baidu-ring-allreduce.txt L24-L26；blog/nccl-collectives.txt L789-L796
- ### 为什么朴素做法会拖慢训练｜讲清把所有数组发给一块卡再广播回去、通信量随 GPU 数增长成为瓶颈，需要一个与卡数无关的算法｜300｜blog/baidu-ring-allreduce.txt L27-L35
- ### 怎么让各链路的流量均匀｜讲清 ring all-reduce 两阶段与每卡流量封顶 2 份｜350｜blog/baidu-ring-allreduce.txt L36-L42；blog/baidu-ring-allreduce.txt L138-L145
