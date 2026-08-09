# MoE 大模型推理与服务基础 glossary：术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| token | S1 | 文本被切分后的小块，模型处理的基本单位（一个字/词片段量级） |
| 参数 / 权重 | S1 | 模型里存储的大量数字；一次计算就是输入与这些数字做乘加 |
| 前向计算 | S1 | 输入 token 逐层流过模型产出一个新 token 的一趟计算 |
| 自回归生成 | S1 提出，S5 展开 | 逐 token 生成：每次基于已有全部 token 算出下一个 |
| Transformer 层 | S1 | 大模型的基本堆叠单元，每层含一个 attention 模块与一个 FFN 模块 |
| attention（注意力） | S1 | 让 token 之间交换信息的模块（内部机制不展开） |
| FFN（前馈网络） | S1 | 对每个 token 独立做同一套变换的模块，由两个矩阵构成 |
| d / d_ff | S1 | 模型隐层宽度 / FFN 中间层宽度（Vaswani 配置 512 / 2048） |
| MoE（Mixture-of-Experts） | S2 | 把 FFN 换成一排专家子网络 + router，每 token 只激活少数专家的结构 |
| 专家（expert） | S2 | MoE 层内的一个小 FFN 子网络，不是独立模型 |
| router（路由器/门控） | S2 | 为每个 token 给所有 routed expert 打分并选出 top-k 的小型网络 |
| routed expert | S2 | 按路由结果被选择性激活的专家 |
| shared expert | S2 | 处理所有 token 的共享专家 |
| top-k | S2 | 每个 token 只激活打分最高的 k 个 routed expert |
| 稀疏激活 | S2 | 总参数很多但每 token 只用其中一小部分的性质 |
| g_i / E_i(x) / S_k(x) | S2 | 门控权重 / 第 i 个专家的输出 / token x 选中的专家集合（F1 符号） |
| 负载均衡（MoE） | S2 提出，S3 复现 | 各专家被选中次数是否均匀；不均时慢的专家/卡拖累整体 |
| 显存 | S3 | GPU 上存放权重与中间数据的高速存储，容量有限 |
| EP（expert parallelism，专家并行） | S3 | 把专家分片到多张 GPU，每卡只存一部分专家 |
| DP / TP | S3 | 数据并行（复制整模型）/ 张量并行（按矩阵切分），仅作区分一句带过 |
| 激活（activation） | S3 | token 经过某层后的中间向量（dispatch 搬运的对象） |
| dispatch / combine | S3 | 把 token 激活发往选中专家所在卡 / 把专家输出送回的通信操作 |
| all-to-all | S3 | 每卡都可能与所有卡互换数据的集合通信模式 |
| 微批（microbatch） | S4 | 把一个大批拆成的小批，TBO 的重叠单位 |
| TBO（two-batch overlap） | S4 | 一个微批的通信与另一个微批的计算重叠 |
| SBO（single-batch overlap） | S4 | 同一微批内 shared expert 计算与 routed expert 通信/计算重叠 |
| 气泡（bubble） | S4 | 通信期间算力闲置的时间 |
| prefill | S5 | 并行处理全部输入 token、建 KV cache、产出首 token 的阶段 |
| decode | S5 | 每步基于前缀 KV cache 生成一个 token 的阶段 |
| KV cache | S5 | 各层缓存的 key/value 张量，后续生成步复用 |
| TTFT | S6 | time to first token，首 token 延迟 ≈ prefill 时长 |
| TPOT | S6 | time per output token，除首 token 外平均每 token 生成时间 |
| SLO | S6 | 服务级别目标：对 TTFT/TPOT 的上限与达成率要求 |
| 达成率（SLO attainment） | S6 | 满足 SLO 的请求占比目标（如 90%） |
| goodput | S6 | 满足 SLO 达成率前提下系统能承接的最大请求率 |
| 裸吞吐（throughput） | S6 | 不顾延迟是否达标的每秒处理量，与 goodput 对照 |
| PD 合设（colocation） | S7 | prefill 与 decode 共享同一实例与一份权重的部署方式 |
| PD 分离（PDD） | S7 | prefill 与 decode 部署在不同 GPU 实例的部署方式 |
| 部署单元（deployment unit） | S7 | 实现目标 P:D 配比的最小整副本组合 |
