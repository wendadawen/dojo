# Quantile Balancing 术语表

| 术语/符号 | 首次出现 | 定义/含义 |
|----------|---------|---------|
| Quantile Balancing (QB) | 页面开头 | K3 的 MoE 负载均衡方法，从路由分数分位数推导专家 bias |
| MoE | S1 | 混合专家模型，每个 token 只激活部分专家 |
| auxiliary-loss-free routing | S1 | 路由框架，bias 加到 router 分数做 Top-k 选择但不进归一化权重，不引入辅助损失项 |
| router | S1 | 线性层 W_r，对每个 token 输出每个专家的分数 |
| router score s_{i,j} | S1 | token i 对专家 j 的分数，s_i = σ(W_r x_i) ∈ (0,1) |
| sigmoid (σ) | S1 | 将实数映射到 (0,1) 的函数，K3 router 的激活函数 |
| bias b_j | S1 | 专家 j 的偏置项，加到 router 分数上参与 Top-k 选择 |
| Top-k 选择 | S1 | 选分数最高的 k 个专家 |
| Top-(k+1) 选择 | S3 | 选分数最高的 k+1 个，第 (k+1) 个作为 cutoff |
| mixture weight (p_{i,j}) | S1 | token i 对专家 j 的归一化权重，p = s/Σs，不含 bias |
| load (ℓ_j) | S2 | 专家 j 在一个 batch 中收到的 token 数 |
| mean load (ℓ̄) | S2 | 所有专家的平均负载 = mk/n |
| sign 更新 | S2 | DeepSeek-V3 的 bias 更新方式 b ← b + γ·sign(ℓ̄ − ℓ_j) |
| γ (gamma) | S2 | sign 更新的固定步长超参数 |
| expert-parallel (EP) training | S2 | 专家并行训练，不同专家分布在不同 GPU 上 |
| target load q | S3 | 每个专家的目标 token 数 = mk/n |
| cutoff α_i | S3 | token i 的 Top-(k+1) 路由中第 (k+1) 个 biased score，即"进 Top-k 必须超过的线" |
| margin | S3 | s_{i,j} − α_i，raw score 减 cutoff |
| quantile (分位数) | S3 | 一组数中指定比例以下的阈值；(1−k/n) 分位数 = 恰好 k/n 比例的数超过它 |
| (1−k/n)-quantile | S3 | margins 的分位数，其值使恰好 q 个 margin 超过它 |
| b̃ (tilde b) | S3 | mean-centering 前的 bias |
| mean-centering | S3 | b ← b̃ − mean(b̃)，去掉公共偏移 |
| causality (因果性) | S3 | 更新在下一步生效，batch 不用自己推导的 bias |
| balanced assignment | S5 | 每个专家恰好服务 q 个 token 的分配方案 |
| linear relaxation | S5 | 将整数约束松弛为连续约束，对二部图 b-matching 仍保持整性 |
| dual objective L(α,β) | S5 | 平衡分配对偶问题，α 是 token 侧乘子，β 是专家侧乘子 |
| β_j | S5 | 专家侧对偶变量，β_j = −b_j |
| coordinate minimizer | S5 | 固定其他变量时单个变量的精确最优解 |
| SignSGD | S5 | 只用梯度符号的随机梯度下降 |
| (sub)gradient | S5 | 次梯度，对不可微目标定义的广义梯度 |
| alternating solver | S5 | 交替固定一方求另一方最优的迭代求解器 |
| histogram (直方图) | S6 | 分箱统计频次的离散表示 |
| bin | S6 | 直方图的一个区间 |
| bin count H_{j,b} | S6 | 专家 j 的第 b 个 bin 中的 margin 计数 |
| all-reduce | S6 | 分布式通信原语，所有 rank 汇总数据并广播结果给所有 rank |
| scatter-add | S6 | 将值分散累加到目标数组的对应位置 |
| bin width w | S6 | 直方图一个 bin 的宽度 = (b_max − b_min + 2)/B |
| EMA (exponential moving average) | S6 | 指数移动平均，用衰减系数加权历史值 |
| frozen bias | S6 | 推理时 bias 固定不变 |
| m | S3 | batch 中的 token 数 |
| n | S3 | 专家数 |
| k | S3 | 每个 token 选择的专家数 |
| B | S6 | 直方图 bin 数 |
| 896 | S2 | K3 的 routed expert 数 |
| 16 | S2 | K3 的 num_experts_per_token（k=16） |
