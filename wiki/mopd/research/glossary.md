# MOPD 术语表（glossary.md）

登记全文首次出现的术语、缩写和符号。后续阶段写作和审查以此为准，保证全文含义一致。

## 缩写

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| MOPD | 开头 | Multi-Teacher On-Policy Distillation，多教师在线策略蒸馏。K3 §4.1.3 命名的方案。 |
| OPD | S2 | On-Policy Distillation，在线策略蒸馏。单教师的一般方法，MOPD 的基础。 |
| RL | S1 | Reinforcement Learning，强化学习。 |
| RLHF | S4 | Reinforcement Learning from Human Feedback，基于人类反馈的强化学习。用奖励模型给出（通常稀疏的）奖励训练策略。 |
| SFT | S1（C7 引用） | Supervised Fine-Tuning，监督微调。K3 中与 MOPD 并列的专家轨迹消费方式。 |
| partial rollout | S4 | 部分滚动采样。K3 RL 框架的长程任务调度机制：当一部分轨迹完成即推进优化，未完成的暂停排队下轮续作。 |

## 符号

| 符号 | 首次出现 | 定义或含义 |
|---|---|---|
| $d$ | S2（Eq.15） | 领域索引，取值 {通用任务, 通用 agent, 编码 agent}。 |
| $e$ | S2（Eq.15） | 努力程度索引，取值 {low, high, max}。 |
| $(d,e)$ | S4 | 领域与努力程度的组合，唯一确定 9 个教师中的一个。 |
| $\pi_{\text{teacher}}^{(d,e)}$ | S2（Eq.15） | 对应领域 $d$、努力程度 $e$ 的专家教师模型，已训练并冻结。 |
| $\pi_\theta$ | S2（Eq.15） | 学生模型，参数为 $\theta$，训练中更新；条件化于 $e$。 |
| $x$ | S2（Eq.15） | 输入 query。 |
| $y_{<t}$ | S2（Eq.15） | 已生成的前缀响应（第 $t$ 个 token 之前）。 |
| $y_t$ | S2（Eq.15） | 第 $t$ 个 token，由学生 $\pi_\theta$ 采样得到。 |
| $\log$ | S2（Eq.15） | 自然对数。 |
| $\text{sg}(\cdot)$ | S2（Eq.15） | 停梯度算子，把括号内整体当常数，不对其求导。 |
| $\text{clip}(\cdot,-R_{\max},R_{\max})$ | S2（Eq.15） | 把值限制在 $[-R_{\max},R_{\max}]$：超出上界取 $R_{\max}$，低于下界取 $-R_{\max}$，区间内不变。 |
| $R_{\max}$ | S2（Eq.15） | 裁剪阈值，$R_{\max}>0$，约束极端 advantage 信号。 |
| $r_{\text{opd}}^{d}(y_t\mid e,x,y_{<t})$ | S2（Eq.15） | 领域 $d$、努力程度 $e$ 下 token $y_t$ 的 per-token OPD 奖励。 |
| $\nabla\log\pi_\theta(y_t)$ | S3 | 策略梯度项，奖励作为常数乘子作用于其上。最小衔接，不展开推导。 |

## 术语漂移防护

- "教师"全文统一指 $\pi_{\text{teacher}}^{(d,e)}$，已训练冻结；不与"奖励模型"混用（RLHF 的奖励模型是另一对象）。
- "学生"全文统一指 $\pi_\theta$，条件化于 $e$。
- "奖励"在本文指 per-token 的 $r_{\text{opd}}$（稠密）；与 RLHF 的序列级奖励区分时明确加修饰词。
- "裁剪"指 $\text{clip}$ 对奖励信号的作用；不用于描述对实际概率比的硬约束。
- "停梯度"指 $\text{sg}$ 对整个对数比值的作用；不用于描述"教师冻结"本身（教师冻结是上游事实，sg 是公式中的实现）。
- "在线策略"指学生自己生成 $y_t$；不与"在线蒸馏"（teacher 和 student 同时训练）混用——本文教师是离线预训练并冻结的。
