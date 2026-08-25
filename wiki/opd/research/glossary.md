# OPD（On-Policy Distillation）— 术语表

| 术语/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| on-policy（在线策略） | S1 | 训练数据分布来自学生当前策略自身的采样 |
| off-policy | S1 | 训练数据来自固定数据集或教师生成的序列 |
| 分布失配（distribution mismatch） | S1 | 训练时见到的序列与学生推理时自己生成的序列分布不一致 |
| 复合误差（compounding error） | S1 | 学生早期错误使其进入训练未见状态、误差逐步放大的现象 |
| $\lambda$（学生数据比例） | S1 | GKD 目标中学生自采样项的混合权重，$\lambda\in[0,1]$ |
| supervised KD | S1 | GKD 的 $\lambda=0$ 端点：固定数据集上最小化师生散度 |
| on-policy KD | S1 | GKD 的 $\lambda=1$ 端点：学生自采样序列上最小化师生散度 |
| GKD | S1 | Generalized Knowledge Distillation，Agarwal et al. 2023 的统一蒸馏框架 |
| 知识蒸馏（KD） | S1 | 用教师输出分布监督学生的方法族；off-policy 基准形态见概念页 knowledge-distillation |
| 策略梯度（最小定义） | S2 | 按 $\nabla\log\pi_S(y_t)\cdot A$ 更新参数的规则；$A$ 为标量乘子 |
| advantage | S2 | 策略梯度中与 $\nabla\log\pi$ 相乘的标量信号；OPD 中即逐 token 分数 $r$ |
| rollout | S2 | 模型从 prompt 出发生成一段完整（或不完整）序列的过程 |
| partial rollout | S2 | 不采样到终局即用于训练的不完整生成 |
| KL 散度（最小定义） | S2 | $\mathrm{KL}(P\|Q)=\sum_c P(c)\log(P(c)/Q(c))$，非对称 |
| forward KL | S4 | $\mathrm{KL}(\pi_{\text{teacher}}\|\pi_S)$ 方向；mass-covering |
| reverse KL | S2 | $\mathrm{KL}(\pi_S\|\pi_{\text{teacher}})$ 方向；mode-seeking |
| mode-seeking / mass-covering | S4 | 散度优化方向的行为倾向：集中到单一模式 vs 覆盖多个模式 |
| 广义 JSD($\beta$) | S4 | 对两个分布的凸组合取混合系数 $\beta$ 的对称有界散度（GKD 公式 (1)） |
| 支持集（support） | S4 | 分布取非零概率的取值集合；教师策略落在学生支持集内是覆盖条件 |
| logprobs | S2 | 模型对序列各 token 的对数概率；教师只需一次前向即可得到 |
| 折扣因子（discount factor） | S2 | RL 中未来奖励的衰减系数；TML 的 OPD 实现取 0 |
| $\pi_S$ / $\pi_\theta$ | S2 | 学生策略（TML 记 $\pi_\theta$，本文统一 $\pi_S$；GKD 记 $p_S$） |
| $\pi_{\text{teacher}}$ / $\pi_T$ / $p_T$ | S2 | 教师策略（统一 $\pi_{\text{teacher}}$，公式引用保留原记号时说明对应） |
| $r$ | S2 | 逐 token 分数 $r=\log(\pi_{\text{teacher}}(y_t)/\pi_S(y_t))$，即负 reverse KL 采样值 |
| $y_t$、$y_{<t}$、$x$ | S2 | 第 $t$ 个 token、其前缀、输入 prompt |
| $\mathrm{sg}(\cdot)$ | S2 | 停梯度算子（K3 Eq.15 使用） |
| $R_{\max}$ | S2 | K3 Eq.15 的裁剪阈值 |
| 强到弱蒸馏（Strong-to-Weak Distillation） | S3 | Qwen3 对轻量模型的蒸馏流水线：off-policy 打底 + on-policy 对齐 |
| forking tokens | S3 | 分岔 token：把生成引向不同分支的高熵 token；教师打分集中于此 |
| bits per episode | S3 | 每个训练 episode（一条生成）传递的信息量；RL 为 $O(1)$、蒸馏为 $O(N)$ |
| RLVR / RLHF | S3 | 用可验证奖励/奖励模型做强化学习；序列级（稀疏）信号 |
| MOPD | S5 | Multi-Teacher On-Policy Distillation，多教师扩展；见概念页 mopd |
| $\alpha$ | S4 | GKD+RL 组合中蒸馏项相对 RL 目标的强度（公式 (5)） |

## 术语使用规则

- 中文行文用"on-policy 蒸馏/OPD"指本概念；"在线策略蒸馏"仅在标题与首次定义处出现。
- 学生/教师概率统一用 $\pi_S$、$\pi_{\text{teacher}}$；直接引用 GKD 公式时保留 $p_S$、$p_T$ 原记号并随文说明。
- "奖励"（reward）与"分数"（score）在本文同义使用时首次说明：逐 token 分数在 RL 栈里充当奖励/advantage。
- 不使用"在线蒸馏"（online distillation 另有所指，易混淆）称呼本概念。
