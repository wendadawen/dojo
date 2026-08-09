# MoonEP 完美均衡专家并行 · 术语表

| 术语 / 缩写 / 符号 | 首次出现位置 | 定义或含义 |
|---|---|---|
| MoonEP | 页面开头 | Moonshot Expert Parallelism，K3 报告 §5.2.1 提出的 MoE 训练专家并行方案 |
| EP（expert parallelism） | S1 | 专家并行，把专家分片到多张 GPU 的并行方式；本页指训练场景 |
| rank | S1 | 一个 EP 进程对应的 GPU；本页中"rank"指 EP rank |
| $E$ | S1 | 专家总数 |
| $R$ | S1 | EP size，即 EP rank 数 |
| $S$ | S1 | 每 rank 本地序列长度（micro-batch 切到 DP rank 后每 rank 持有的 token 数） |
| $K$ | S1 | 每 token 选的专家数（top-k 中的 k） |
| token-expert pair | S1 | 一个 token 与它选中的一个专家构成的二元组；EP dispatch 的基本单位 |
| home rank | S1 | 一个专家被分片后所在的原 rank；该专家的权重常驻 home rank |
| 冗余专家（redundant expert） | S1 | 把某个 home rank 上的专家**临时复制**到另一个 rank 上、让该 rank 在本地处理原本要发出的 token；forward 预取、backward 归还 |
| 完美均衡（perfect balance） | S1 | 每 rank 恰好收到 $S\times K$ 个 token 的状态 |
| router | S1 | MoE 中给每个 token 打分、选 top-k 专家的模块 |
| shared expert | S4 | 所有 token 都经过的共享专家；与 routed expert 相对 |
| routed expert | S1 | 被 router 选择性激活的专家；本页中"专家"默认指 routed expert |
| dispatch / combine | S1 | EP 的两个 all-to all 阶段：dispatch 把 token-expert pair 发到专家所在 rank；combine 把专家输出送回原 rank |
| permute / unpermute | S3 | MoonEP 实现里 fused 算子的两个方向：permute = dispatch 方向（token 发到远端 expert-grouped 位置）；unpermute = combine 方向 |
| DeepEP | S1 | 传统 EP 方案的代表性实现；MoonEP 保留其总体计算流 |
| ECHO | S4 | 相邻方案，预设冗余专家数 |
| UltraEP | S4 | 相邻方案，施加 per-rank token cap |
| $m_r(P)$ | S2 | 规划 $P$ 下 rank $r$ 上的冗余专家数 |
| $M(I)$ | S2 | router 输出 $I$ 下的最优（最小化最大冗余数），$M(I)=\min_P\max_r\{m_r(P)\}$ |
| $E/R$ | S2 | 每 rank 至多需要的冗余专家数上界（Theorem 1）；home rank 上本地专家数 |
| $\lceil E(R-1)/R^2\rceil$ | S2 | 最坏 router 输出下需要的冗余专家数下界（Theorem 2）；大 $R$ 下近似 $E/R$ |
| planning kernel | S3 | MoonEP 在线规划冗余专家的 GPU kernel；近最优、开销可忽略、总尊重 $E/R$ 上界（报告声明） |
| ILP（integer linear programming） | S3 | MoonEP 离线求精确最优的整数线性规划；用于代表性 case 作为 GPU kernel 的参考 |
| fused permute/unpermute | S3 | MoonEP 的融合算子；planning kernel 预计算每 token 目的地，直接发到远端 expert-grouped 位置，返回 buffer view 免中间拷贝 |
| expert-grouped 位置 | S3 | 通信 buffer 里按专家分组的存储位置；同一专家的所有 token 连续存放，便于后续 GEMM |
| zero-copy 数据路径 | S3 | token 直接发到目标位置、无需中间拷贝的通信路径 |
| 通信 buffer | S3 | 接收 dispatch 来的 token 的 GPU buffer；MoonEP 固定 $S\times K$，DeepEP 最坏 $S\times K\times R$ |
| host-device 同步 | S3 | CPU（host）与 GPU（device）之间的同步点；传统 EP 每层需要一次以拿到真实计算形状 |
| 静态形状 | S3 | 计算张量形状在编译期/launch 前已知；MoonEP 因每 rank 收 $S\times K$ 而静态 |
| reduce buffer | S3 | MoonEP backward 中暂存冗余专家梯度的本地 buffer |
| reduce 回 home rank | S3 | 把暂存在本地 reduce buffer 的冗余专家梯度归约到原 home rank 的梯度 buffer |
| all-reduce | S3 | 跨 rank 的归约原语；MoonEP 的 reduce-back 与之同语义但只针对冗余专家梯度 |
| per-expert token 偏斜 | S4 | 同一 rank 内不同专家收到的 token 数不均；MoonEP 不解决，由 Expert-GEMM scheduler 处理 |
| Expert-GEMM scheduler | S4 | MoonEP 的 workload-aware 调度器；launch 前根据当前 token 分布调参、launch 后固定 |
| autotuning | S4 | 报告声明 scheduler 的关键系数通过离线 autotuning 标定；具体未公开 |
| TBO / SBO | （不出现） | 推理场景的通信掩盖思路，本页不使用（属 moe-serving 页） |
| 辅助损失（load balance loss） | S4 | 训练 MoE 时让 router 把 token 均匀分到专家的损失项；与 MoonEP 独立 |

## 一致性约束

- "rank" 在全文统一指 EP rank，不与 DP rank / PP rank / TP rank 混用（本页不出现后三者）。
- "冗余专家" 统一指临时副本，不是常驻副本；首次出现处加粗强调。
- "完美均衡" 统一指每 rank 收 $S\times K$，不与"per-expert 均衡"混。
- $S\times K$ 与 $S\times K\times R$ 的语义区分在每次出现时由上下文明确（每 rank vs 总量或最坏 buffer）。
- DeepEP 在本页统一指"传统 EP 方案的代表性实现"；不展开其内部机制。
