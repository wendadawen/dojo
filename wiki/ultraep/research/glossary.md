# UltraEP 术语表

符号沿用论文 §4.3 Table 1，全页保持一致写法。

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| $\mathcal{R}$ | 第 3 章问题形式化 | 一个 EP 组内的全部 rank，规模 $R:=\lvert\mathcal{R}\rvert$ |
| $r$ | 第 3 章 | 源 rank 下标（token 从这里发出） |
| $t$ | 第 4 章 | 目标 rank 下标（承载某物理实例的 rank） |
| $\mathcal{E}$ | 第 3 章 | 一个 EP 组内的全部逻辑专家 |
| $e$ | 第 3 章 | 逻辑专家下标 |
| $h(e)$ | 第 3 章 | 逻辑专家 $e$ 的 home rank，承载其主实例 |
| $\mathcal{H}(e)$ | 第 3 章 | 承载 $e$ 各物理实例的 rank 集合，含 $h(e)$ |
| $\mathcal{E}_r$ | 第 3 章 | rank $r$ 上的主专家集合，即 $\{e\in\mathcal{E}\mid h(e)=r\}$ |
| $N_{\mathrm{layers}}$ | 第 3 章 | 模型的 MoE 层数 |
| $N_{\mathrm{slot}}$ | 第 3 章 | 每个 rank 的冗余槽数 |
| $\Lambda=\{\lambda_{r,e}\}$ | 第 3 章 | 全局负载矩阵，$\lambda_{r,e}$ 是源 rank $r$ 上被分配给逻辑专家 $e$ 的 token 负载 |
| $\lambda_e$ | 第 4 章 | 专家 $e$ 的总负载，$\lambda_e=\sum_{r\in\mathcal{R}}\lambda_{r,e}$ |
| $\ell_r$ | 第 4 章 | rank $r$ 的初始负载，$\ell_r=\sum_{e\in\mathcal{E}_r}\lambda_e$ |
| $U=\{u_{e,r}\}$ | 第 3 章 | 解出的负载配额表；$u_{e,r}>0$ 当且仅当 rank $r$ 承载 $e$ 的一个物理实例，该实例承接 reroute 后负载 $u_{e,r}$ |
| $X=\{x_{r,s}\}$ | 第 3 章 | 冗余槽分配；$s\in[N_{\mathrm{slot}}]$，$x_{r,s}=e$ 表示 rank $r$ 的槽 $s$ 放 $e$ 的副本，否则为 $\varnothing$ |
| $Q=\{q_{r,e,t}\}$ | 第 4 章 | reroute 拆分：从源 rank $r$ 发往 $e$ 在 rank $t$ 上物理实例的 token 数 |
| $u_{\min}$ | 第 4 章 | 新建副本的最小有用配额，论文取 1024 |
| $\beta$ | 第 4 章 | 目标均衡系数，论文取 1.01 |
| $\tau$ | 第 4 章 | 候选负载阈值，二分搜索的对象 |
| $\mathrm{exc}_r(\tau)$ | 第 4 章 | 阈值 $\tau$ 下 rank $r$ 必须卸掉的超额负载，$\max(\ell_r-\tau,0)$ |
| $\mathrm{slk}_r(\tau)$ | 第 4 章 | 阈值 $\tau$ 下 rank $r$ 还能吸收的空闲，$\max(\tau-\ell_r,0)$ |
| $\mathrm{cap}_e$ | 第 4 章 | 专家 $e$ 剩余可转移的负载 |
| $t^\star$ | 第 4 章 | 当前空闲最大的合法目标 rank |
| $\delta$ | 第 4 章 | 单次接受转移的负载量 |
| $\tilde U$ | 第 4 章 | 某次阈值探测中的临时配额计划 |
| $\hat{\lambda}_{r,e}$ | 第 4 章 | 本地配额消费后 $(r,e)$ 的残余需求 |
| $\hat{u}_{e,t}$ | 第 4 章 | 本地配额消费后实例 $(e,t)$ 的残余配额 |
| $T_{\text{solve\_rep}}^{fwd}$ | 第 3 章 | 前向的规划求解延迟 |
| $T_{\text{reroute}}^{fwd}$ | 第 3 章 | 前向的 reroute 延迟 |
| $T_{w\_\text{distr}}^{fwd}$ | 第 3 章 | 前向的权重分发延迟 |
| $T_{\text{tok\_a2a}}^{fwd/bwd}$ | 第 3 章 | 前向/反向的 token all-to-all 延迟 |
| $T_{\text{moe}}^{fwd/bwd}$ | 第 3 章 | 前向/反向的 MoE 计算延迟 |
| $S$、$K$ | 方法评价章节 | 引用 MoonEP 时的每 rank token 数与 top-$k$；仅在该处出现 |

## 术语

| 术语 | 英文 / 缩写 | 首次出现 | 含义 |
|---|---|---|---|
| 专家并行 | expert parallelism, EP | 页面开头 | 把 MoE 的专家切分到多块 GPU，token 按路由结果经 all-to-all 送到承载所选专家的 GPU |
| 大规模专家并行 | large-EP | 页面开头 | 论文指 32 路或 64 路量级的 EP |
| 机架级节点 | rack-scale node, RSN | 页面开头 | 把 scale-up 域从单台 4/8 卡服务器扩到整机架、通常 64+ 卡的形态；机架内跨服务器的 GPU 仍由机架级 scale-up fabric 直连 |
| scale-up / scale-out | — | 第 2 章 | scale-up 是机内或机架内的高带宽直连域（数百 GB/s 每卡、load/store 内存语义）；scale-out 是包交换网络（通常每张网卡数十 GB/s） |
| 逻辑专家 | logical expert | 第 3 章 | 模型定义的专家身份 |
| 物理专家 | physical expert | 第 3 章 | 某个 rank 实体化出来的专家副本 |
| 主槽 / 主专家 | main slot / main expert | 第 3 章 | 承载逻辑专家原始实例的固定槽位；保留完整的权重、梯度与优化器状态 |
| 冗余槽 | redundant slot | 第 3 章 | 每 rank 预留的额外槽位，放一个副本或留空；不存优化器状态，权重与梯度 buffer 跨层复用 |
| 配额 | quota | 第 4 章 | 分配给某个专家实例的最终 token 负载 $u_{e,r}$；同时决定副本建不建与建了之后承接多少 |
| 重路由 | reroute | 第 3 章 | 把 router 输出从 token→逻辑专家改写成 token→物理专家 |
| 精确负载 | exact load | 页面开头 | gating 之后立即可得的本次真实 token 分布，区别于按历史统计做的预测 |
| 热路径 | hot path / critical path | 页面开头 | 单次前向中无法被其他计算掩盖、直接计入端到端延迟的执行段 |
| 计算拖尾 | straggler | 页面开头 | 因分到的 token 明显多于同伴而拖慢整个同步步的 rank |
| 不均衡度 | imbalance ratio | 第 1 章 | 论文有两种用法：Figure 4 的「每专家」不均衡是最大专家负载除以均值；主结果中的 rank 级不均衡是最大 rank 负载除以均值。页面每次使用时说明是哪一种 |
| notify-dispatch | — | 第 3 章 | token all-to-all 之前交换路由元数据、确定各对端收发规模与偏移的步骤 |
| tile | — | 第 5 章 | 把权重或梯度切成的固定大小搬运单元 |
| chunk | — | 第 5 章 | 若干连续 tile 组成的中继调度单元 |
| persistent kernel | — | 第 5 章 | 一次启动后常驻、反复从任务流取下一份工作的 kernel |
| 双缓冲 | double buffering | 第 5 章 | tile $i$ 在写出时同一 block 已开始载入 tile $i+1$ |
| 中继 | relay | 第 5 章 | 两级 fan-out 中承担一阶段接收、二阶段转发的 rank |
| 中继前沿 | relay frontier | 第 5 章 | 一阶段中继集合的宽度，取在 $\sqrt{\lvert\mathcal{H}(e)\rvert-1}$ 附近 |
| Wgrad / Dgrad | weight gradient / data gradient | 第 3 章 | 反向中对权重求梯度与对输入求梯度两部分计算 |
| virtual layer ID | — | 第 5 章 | 把放置与 reroute 元数据同（真实层, microbatch）在环形缓冲里哈希得到的标识，经 torch.autograd 传递，让反向取回匹配的前向均衡计划 |
| 强制均衡 ideal | force-balanced ideal | 页面开头 | 论文构造的上界基线：改 router 让 token 均匀分到各专家 |
| EPLB | Expert Parallelism Load Balancer | 页面开头 | 广泛部署的均衡器，用冗余专家策略按给定负载算布局；常见部署用近期路由历史周期性重均衡 |
| EPLB+ | — | 第 6 章 | 论文构造的强化基线：喂精确负载、用标准 EPLB 加 round-robin reroute，通信机制换成 UltraEP 的 |
| LPLB | — | 第 6 章 | 线性规划求解器，在 EPLB 之上为每个 microbatch 调整 reroute，限制每专家至多一个副本 |
| DeepEP | — | 第 3 章 | token all-to-all 通信库；UltraEP 用它做 dispatch/combine，§8.5 中也把它适配来做专家权重搬运的通信基线 |
| TTFT / TPOT | time to first token / time per output token | 页面开头 | 首 token 延迟主要由 prefill 决定；每输出 token 时间反映稳态 decode 速度 |
| RPS | requests per second | 第 6 章 | serving 实验的请求速率 |
| TFLOPS/GPU | — | 第 6 章 | 训练实验的吞吐指标：每卡实际达到的浮点算力 |

## 写法约定

- 变量与希腊字母一律 `$...$`，不用 Unicode 数学字符
- 上标 fwd / bwd 统一写作 $T_{\cdot}^{fwd}$、$T_{\cdot}^{bwd}$
- 集合基数写 $\lvert\mathcal{H}(e)\rvert$，不写 `|H(e)|`
- 倍数写「1.42$\times$」，百分比写「94.6%」
- 论文方法名统一写 UltraEP（不写 \sys、不加书名号）
- 模型名统一：GLM4.5-106B-A12B、Qwen3-235B-A22B、GLM4.7-358B-A32B、DeepSeek-V3-671B-A37B、RefMoE-288B-A16B；正文重复出现时可简称 GLM4.5-106B、Qwen3-235B、GLM4.7-358B、DeepSeek-V3
- EP 配置写 EP64、EP40、EP32；$N_{\text{slot}}$ 用公式写法
