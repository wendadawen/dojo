# GPU Communication

- 描述：GPU 通信是让多块 GPU 协作时交换数据的机制：算法上由集合通信把各卡的数据合并（如 ring all-reduce），合并后的字节经机内 NVLink 或跨机网卡与 RDMA 传输。
- 摘要：数据并行训练每步要把各卡算出的梯度合并：用 all-reduce 求出各卡数组的和，再按卡数取平均，使每块卡得到同一份平均梯度。ring all-reduce 把它拆成 scatter-reduce 与 all-gather 两阶段，每卡经手的数约为整个数组的两倍、不随卡数线性增长；字节走的路径机内是 NVLink，跨机是网卡加 RDMA。
- 主题：并行与通信
- 标签：并行与通信

## 资料

- Andrew Gibiansky. Bringing HPC Techniques to Deep Learning. Baidu Research technical blog, 2017. [https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
- NVIDIA. NCCL Overview. NCCL 2.32.3. [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- NVIDIA. Collective Operations. NCCL 2.32.3. [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- NVIDIA. GPUDirect RDMA. CUDA 13.4. [https://docs.nvidia.com/cuda/gpudirect-rdma/index.html](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- NVIDIA. RDMA Aware Networks Programming Guide. DOCA. [https://networking-docs.nvidia.com/doca/sdk/rdma-aware-networks-programming-guide](https://networking-docs.nvidia.com/doca/sdk/rdma-aware-networks-programming-guide)
- NVIDIA. NVLink and NVLink Switch. [https://www.nvidia.com/en-us/data-center/nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/)

## 短页

### 它把各卡的数据合并

数据并行训练里，每块 GPU 存一份完整的模型副本，各拿一小批不同的样本，各自算出这批样本的梯度（梯度是每个参数对应的一个数，指示它该往哪个方向、调多大幅度）；要让每块卡手里的梯度一致，就把它们合并。合并用的操作是 all-reduce（全归约）：每块卡交出一个同样形状的数组，最后每块卡都拿到各卡数组对应元素的和，再按卡数取平均，于是每块卡用同一份平均梯度更新参数。
依据：blog/baidu-ring-allreduce.txt L24-L26；blog/nccl-collectives.txt L789-L796

这套在卡与卡之间搬运数据、把结果分发回每块卡的机制就叫 GPU 通信。它要搬运的数组长度可达上亿。
依据：blog/baidu-ring-allreduce.txt L25-L26

### 为什么朴素做法会拖慢训练

最直接的做法是让一块卡当汇总点：其余每块卡把梯度数组发给它，它把收到的数组加起来取平均，再把平均结果发回给每块卡。汇总点这一块卡既要收下所有卡的数据，又要给所有卡发回去，它的网络端口成了瓶颈。
依据：blog/baidu-ring-allreduce.txt L27-L32

要传的数据量随卡数增长：汇总点收发的是与卡数成正比的份数，卡越多每步花在通信上的时间越长；这种线性增长很快让继续加卡变得不切实际。需要换一个通信量与卡数无关的算法。
依据：blog/baidu-ring-allreduce.txt L33-L37

### 怎么让各链路的流量均匀

把各卡排成一个逻辑环，每块卡只往右邻居发、只从左邻居收，数组先等分成 N 份（N 是卡数），然后把一次 all-reduce 拆成两阶段，各收发 N-1 步。第一阶段 scatter-reduce（分散归约）：每步把手上的一份发给右邻居、从左邻居收一份并加到自己的同一份上，N-1 步后每块卡手上有完整的一份和。第二阶段 all-gather（全收集）：同样收发 N-1 步，但收到的一份直接覆盖自己那一份，N-1 步后每块卡凑齐全部份，每份都是各卡贡献之和。
依据：blog/baidu-ring-allreduce.txt L36-L42；blog/baidu-ring-allreduce.txt L43-L96

每块卡两阶段各收发 N-1 次、每次搬 $K/N$ 个数，合计 $2(N-1)K/N$ 个数（$K$ 是数组总数）。这个量不随卡数线性增长，约等于整个数组的两倍。
依据：blog/baidu-ring-allreduce.txt L138-L145

## 长页

多块 GPU 一起训练同一个模型时，卡与卡之间要交换数据，这套交换数据的机制就是 GPU 通信。最常见的一步发生在数据并行训练里：每块卡存一份完整的模型副本，各自分到一小批不同的样本，各自算出这批样本的梯度（前向是把一批样本算成输出，反向是由输出与目标的差反推各参数的梯度）；要让每块卡手里的梯度一致，就把它们合并。合并用一个 all-reduce（全归约）操作完成：每块卡交出一个同样形状的数组，最后每块卡都拿到各卡数组对应元素的和。要得到数据并行训练用的梯度，再把各卡的和按卡数取平均；因为每块卡拿到同一份平均梯度，合并后每块卡用同一份梯度更新参数，效果等同于把所有样本放在一块卡上训练。
依据：blog/baidu-ring-allreduce.txt L24-L26；blog/nccl-collectives.txt L789-L796

数组就是一串数，长度可达上亿。数组在两块卡之间搬动，走的是机器的互联：同一台机器内的 GPU 由 NVLink 直连，跨机器通常要经过网卡，用为这种搬运设计的 RDMA（远程直接内存访问）。算法决定搬多少份数据、分几步搬，互联决定每一步实际能跑多快。
依据：blog/baidu-ring-allreduce.txt L26；blog/nvlink.txt L580；blog/gpudirect-rdma.txt L79；blog/rdma-networks-programming-guide.txt L249

### all-reduce 要交换的数据

参与一次集合通信的每个 rank（秩）对应一块 GPU。集合通信是一类通信方式：每个 rank 都要参与，调用时用相同的长度和相同的元素类型，共同完成一次操作。
依据：blog/nccl-collectives.txt L785-L788

all-reduce 是集合通信里的一种。$N$ 个 rank 参与时，每个 rank 拿出一个长度为 $K$ 的数组 $\text{in}$，通信结束时每个 rank 的接收数组 $\text{out}$ 都相同，第 $i$ 个元素是各 rank 第 $i$ 个元素的和：$\text{out}[i]=\text{in}_0[i]+\text{in}_1[i]+\cdots+\text{in}_{N-1}[i]$。除了求和，这个操作也支持取最大、取最小这类归约。
依据：blog/nccl-collectives.txt L789-L796

all-reduce 可以拆成两个更小的操作。scatter-reduce（分散归约）先把各 rank 的数组按元素归约，再把结果切成 $N$ 份，每份由一块卡持有；all-gather（全收集）反过来，把各 rank 手里的份收齐再分发给所有 rank，每个 rank 得到完整的 $N$ 份。先做一次 scatter-reduce，再做一次 all-gather，两者合起来正好是一次 all-reduce。
依据：blog/nccl-collectives.txt L815-L826

### 朴素方案为什么不够用

最直接的做法是让一块卡当汇总点：其余每块卡把自己的梯度数组发给它，它把来自所有卡的数组合起来、取平均，再把平均结果发回给每块卡。汇总点这块卡既要收下所有卡的数据，又要给所有卡发回去，它的网络端口成了整次训练的瓶颈。
依据：blog/baidu-ring-allreduce.txt L27-L32

要传的数据量随卡数增长：汇总点收发的是与卡数成正比的份数，卡越多，每步花在通信上的时间越长；这种线性增长很快让继续加卡变得不切实际。需要换一个通信量与卡数无关的算法。
依据：blog/baidu-ring-allreduce.txt L32-L37

::: 汇总点在具体模型上的代价
用一个三亿参数的语音识别模型估算，汇总点每步收发全部梯度、按每 GB 约 1 秒的链路带宽计。
依据：blog/baidu-ring-allreduce.txt L33

1. 模型有三亿个可训练参数，梯度数组与参数一一对应，总大小约 1.2 GB。
依据：blog/baidu-ring-allreduce.txt L26；blog/baidu-ring-allreduce.txt L33
2. 卡数增到两块时，汇总点每步要多花约 1.2 秒。
依据：blog/baidu-ring-allreduce.txt L33
3. 卡数增到十块时，每步要多花约 10.8 秒，通信时间随卡数一起增长。
依据：blog/baidu-ring-allreduce.txt L33
:::

### ring all-reduce 的两个阶段

ring all-reduce 把所有卡排成一个逻辑环：每块卡有一个左邻居和一个右邻居，只往右邻居发数据，只从左邻居收数据。数组先被等分成 $N$ 份（$N$ 是卡数），整个操作分两个阶段，每个阶段都收发 $N-1$ 步。
依据：blog/baidu-ring-allreduce.txt L38；blog/baidu-ring-allreduce.txt L42；blog/baidu-ring-allreduce.txt L44-L45；blog/baidu-ring-allreduce.txt L49；blog/baidu-ring-allreduce.txt L96

第一阶段叫 scatter-reduce（分散归约）。每一步里，每块卡把手上的一份发给右邻居，同时从左邻居收一份，并把收到的一份加到自己的同一份上。第 $n$ 号卡第一步发第 $n$ 份、收第 $n-1$ 份（下标按环回计，$n$ 为 0 时环回到第 $N-1$ 份），之后每一步都发上一步刚收到的那份。收发 $N-1$ 步后，每块卡手上有完整的一份和，也就是这一份在所有卡上的贡献之和。
依据：blog/baidu-ring-allreduce.txt L49；blog/baidu-ring-allreduce.txt L78-L82

第二阶段叫 all-gather（全收集）。它和 scatter-reduce 一样收发 $N-1$ 步，但收到的一份直接覆盖自己那一份，不再相加。第 $n$ 号卡第一步发第 $n+1$ 份、收第 $n$ 份（同样按环回计，$n$ 为 $N-1$ 时发第 0 份），之后每一步都发刚收到的那份。收发 $N-1$ 步后，每块卡凑齐了全部 $N$ 份，每份都是各卡贡献之和。
依据：blog/baidu-ring-allreduce.txt L96；blog/baidu-ring-allreduce.txt L125-L126

::: 三块卡的两个阶段
取三块卡（记为卡 0、卡 1、卡 2），数组长度 6，每块卡的数据如下，目标是求出每块卡同一位置的三个数之和。
依据：自行构造

1. 卡 0 的数据是 1、2、3、4、5、6；卡 1 是 10、20、30、40、50、60；卡 2 是 100、200、300、400、500、600。数组分成 3 份，每份 2 个数：卡 0 的第 0 份是 1、2，第 1 份是 3、4，第 2 份是 5、6，另两块卡同理。
依据：自行构造
2. 每块卡同一位置的三个数之和：第 0 份是 111、222，第 1 份是 333、444，第 2 份是 555、666。
依据：自行构造
3. scatter-reduce 第 1 步：卡 0 发第 0 份（1、2）给卡 1，收卡 2 的第 2 份（500、600），加到自己第 2 份上得 505、606；卡 1 发第 1 份给卡 2，收卡 0 的第 0 份加到第 0 份上得 11、22；卡 2 发第 2 份给卡 0，收卡 1 的第 1 份加到第 1 份上得 330、440。
依据：自行构造
4. scatter-reduce 第 2 步：每块卡发上一步刚收到的那份。卡 0 发第 2 份（505、606）给卡 1，收卡 2 的第 1 份（330、440）加到第 1 份上得 333、444，正是第 1 份的全局和；卡 1 收卡 0 的第 2 份加到第 2 份上得 555、666；卡 2 收卡 1 的第 0 份加到第 0 份上得 111、222。此时卡 0 持有第 1 份的全局和，卡 1 持有第 2 份，卡 2 持有第 0 份。
依据：自行构造
5. all-gather 第 1 步：每块卡发下一份、收自己编号那份，收到就覆盖。卡 0 发第 1 份（333、444）给卡 1，收卡 2 的第 0 份（111、222）盖住自己第 0 份；卡 1 发第 2 份（555、666）给卡 2，收卡 0 的第 1 份盖住自己第 1 份；卡 2 发第 0 份给卡 0，收卡 1 的第 2 份盖住自己第 2 份。此时每块卡已有两份完整数据。
依据：自行构造
6. all-gather 第 2 步：每块卡发刚收到的那份。卡 0 发第 0 份给卡 1，收卡 2 的第 2 份（555、666）补齐；卡 1 发第 1 份给卡 2，收卡 0 的第 0 份补齐；卡 2 发第 2 份给卡 0，收卡 1 的第 1 份补齐。三块卡都得到 111、222、333、444、555、666。
依据：自行构造
:::

### 每块卡要搬多少数据

每块卡在 scatter-reduce 里收发 $N-1$ 次，每次搬 $K/N$ 个数（$K$ 是数组的总元素数）；all-gather 再收发 $N-1$ 次，每次同样搬 $K/N$ 个数。两阶段加起来，每块卡经手的数是
$$2(N-1)\frac{K}{N}$$
这个量有上限：$2(N-1)/N$ 小于 2，所以每块卡经手的数不超过整个数组的两倍（$2K$），不随卡数线性增长。
依据：blog/baidu-ring-allreduce.txt L138-L145

::: 上限的来历
把系数单独看：$\frac{2(N-1)}{N}=2-\frac{2}{N}$，随 $N$ 增大单调上升并趋近 2，始终小于 2，所以每块卡经手的数不超过整个数组的两倍。
依据：自行构造
:::

### 机内的路径：NVLink

一台机器里装有多块 GPU，它们之间的数据走 NVLink。NVLink 是 GPU 与 GPU 之间的直连链路，把一台服务器内的多块 GPU 输入输出连在一起，是机内 GPU 互通的高带宽通道；最新一代给每块 GPU 提供 3 TB/s 的带宽。
依据：blog/nvlink.txt L570；blog/nvlink.txt L580

多块 GPU 的 NVLink 再由 NVLink Switch 芯片汇集，构成整个机架范围的全互联：任意两块卡之间都能按 NVLink 的完整速度通信，不必绕路。比如把 72 块 GPU 连成全互联，整机架 GPU 带宽合到 216 TB/s。
依据：blog/nvlink.txt L570；blog/nvlink.txt L580

### 跨机的路径：网卡、RDMA 与 GPUDirect

跨机时数据走网卡出网：GPU 显存里的数据先送到网卡，跨机传输，再写入对端的显存。
依据：blog/gpudirect-rdma.txt L79

传统网络走 TCP/IP：数据由操作系统在自己的缓冲区和应用的内存之间来回拷贝，每一次传输都要操作系统介入，两端各自发生拷贝。RDMA（远程直接内存访问）走另一条路：通信的两端各有一段叫队列对的收发队列（发送队列与接收队列绑在一个对象里），两边建立连接、登记好内存之后，发送方把请求投进网卡的队列，网卡直接把数据写进对端登记好的内存，或从对端内存直接读回来，全程不经过对端的 CPU。
依据：blog/rdma-networks-programming-guide.txt L353-L356；blog/rdma-networks-programming-guide.txt L377-L381；blog/rdma-networks-programming-guide.txt L389-L391；blog/rdma-networks-programming-guide.txt L519；blog/rdma-networks-programming-guide.txt L525-L526；blog/rdma-networks-programming-guide.txt L552-L553；blog/rdma-networks-programming-guide.txt L562-L567

RDMA 的写和读都不通知对端，数据落到对端内存里对端并不知情；只有带立即数的写操作才会把那个立即数告诉对端。少掉 CPU 介入和内存拷贝，降低了延迟、减轻了 CPU 负担。
依据：blog/rdma-networks-programming-guide.txt L249；blog/rdma-networks-programming-guide.txt L564-L567

跨机时数据还要从 GPU 显存送到网卡，不经过主机内存。GPUDirect RDMA 提供这条直连路径：网卡按 PCIe（连接设备与主机的总线）的地址直接读写 GPU 显存的一块窗口，数据不必先拷进主机内存再上网卡。这条路径要求两个设备挂在同一个上游 PCIe 根联合体之下（根联合体是多条 PCIe 链路汇聚到的上层设备）。
依据：blog/gpudirect-rdma.txt L79-L81；blog/gpudirect-rdma.txt L88

::: 显存登记的代价
网卡要直接读 GPU 显存，那块显存必须先登记：登记时把显存钉在物理地址上、把地址表交给网卡。
依据：blog/gpudirect-rdma.txt L106-L109

1. 登记本身开销不小，可能花到毫秒级。
依据：blog/gpudirect-rdma.txt L244
2. 如果每传一次就登记、传完就释放，光花在登记上的时间就会拖慢传输。
依据：blog/gpudirect-rdma.txt L245
3. 所以实现上把登记过的区域留着复用，只在空间将满或登记失败时才释放一部分，避免反复登记。
依据：blog/gpudirect-rdma.txt L247
:::

### 适用条件与边界

ring all-reduce 的带宽最优有一个前提：延迟相对带宽可以忽略。数据量大、传输次数少时这个前提成立，每次收发的启动延迟摊到大批数据上可以忽略；数组小、传输次数多时，$N-1$ 步的往返延迟累加起来可能超过数据本身的时间，ring 不再有优势。
依据：blog/baidu-ring-allreduce.txt L37；blog/baidu-ring-allreduce.txt L145

整次通信的速度由环上最慢的一条相邻链路决定，哪一对邻居之间的链路最慢，整圈就按它的速度走。把同一台机器内的 GPU 在环上排在一起，可以减少网络争用，避免链路带宽被争用稀释。
依据：blog/baidu-ring-allreduce.txt L145

NVLink 不只在机器内。用 NVLink Switch，NVLink 连接可以扩展到机器之间，把多台机器连成一个大的多卡集群。
依据：blog/nvlink.txt L580-L587
