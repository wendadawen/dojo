# 辅助损失无关路由：核心论断与证据

来源优先级：原始论文（arXiv:2408.15664, 2024-08-28）> DeepSeek-V3 技术报告（arXiv:2412.19437）> 权威综述与官方实现。

## C 论断（机制）

### C1：MoE 训练存在负载坍塌风险
- 论断：MoE 路由器在无约束情况下会倾向于把多数 token 路由到少数专家，未被选中的专家梯度接近零、训不动，形成 routing collapse。
- 来源：arXiv:2408.15664 §1（引言），引用 Shazeer et al. 2017；DeepSeek-V3 报告 §2.2 同样表述。
- 适用条件：稀疏 top-k MoE 训练。
- 置信：已确认。

### C2：传统辅助损失把均衡写进梯度目标
- 论断：经典做法是引入辅助损失 $L_{bal}=\alpha\sum_i f_i P_i$ 一类项，鼓励均衡；$\alpha$ 难调——过大会引入非可忽略的干扰梯度损害主任务，过小压不住坍塌。
- 来源：arXiv:2408.15664 §1，Table 2 对比 baseline（α=0.001）；DeepSeek-V3 报告 §2.2 同样描述。
- 适用条件：所有采用辅助损失方案的 MoE 训练。
- 置信：已确认。

### C3：bias 加到路由分数上仅做 top-k 选择
- 论断：$s_{i,t}+b_i$ 仅用于判断是否进入 top-k；进入则该专家的门控值取原始 $s_{i,t}$，未进入则为 0；bias 不进 mixture weight。
- 来源：DeepSeek-V3 报告 Eq.16；原始论文 §3.1 Eq.1-2。
- 适用条件：DeepSeek-V3 与原论文 Loss-Free Balancing 方法。
- 置信：已确认。

### C4：bias 由规则式 sign 更新，不进反向传播
- 论断：每训练步结束后，按负载偏差 $e_i=\bar c_i-c_i$ 更新 $b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(e_i)$，其中 $c_i$ 是该步专家 $i$ 收到的 token 数，$\bar c_i$ 是专家平均负载；bias 不通过反向传播学习，不是模型参数。
- 来源：原始论文 Algorithm 1、§3.2；DeepSeek-V3 报告 §2.2 "if the corresponding expert is overloaded, we decrease the bias by γ; if underloaded, increase by γ"。
- 适用条件：训练阶段、批规模足够大；原始论文强调"用历史 batch 的负载信息更新，避免 future token leakage"。
- 置信：已确认。

### C5：sign 而非误差幅度，是为了稳定但代价是收敛慢
- 论断：原始论文实验了 $b_i\leftarrow b_i+\gamma\cdot e_i$ 的连续变体和乘法 bias 变体，最终采用 sign 版本；sign 只保留方向、丢弃幅度，单步稳定但收敛速度受 $\gamma$ 限制；专家数极多时此缺陷被放大。
- 来源：arXiv:2408.15664 Table 3（变体对比）；K3 报告对 896 专家时的失效分析（见 `quantile-balancing/`）。
- 适用条件：DeepSeek-V3 的 256 专家可接受；K3 的 896 专家时 sign 收敛过慢。
- 置信：已确认。

### C6：本方法只解决批级均衡，序列内均衡需 complementary sequence-wise loss
- 论断：aux-loss-free 只看批级 $c_i$；可能一个序列内所有 token 都激活同一组专家。DeepSeek-V3 另加 $\alpha=0.0001$ 的 sequence-wise balance loss $L_{Bal}=\alpha\sum f_i P_i$ 互补。
- 来源：DeepSeek-V3 报告 Eq.17-19；原始论文未涵盖此项（V3 才补）。
- 适用条件：DeepSeek-V3 训练。
- 置信：已确认。

## F 公式

### F1：路由分数
$$s_{i,t}=\sigma(u_t^\top e_i)$$
- 来源：DeepSeek-V3 报告 Eq.16（用 sigmoid）；原始论文一般写作 $G(u_t^\top e_i)$。
- 含义：$u_t$ 是 token $t$ 的输入向量，$e_i$ 是专家 $i$ 的中心向量（router 权重的第 $i$ 行），$\sigma$ 是 sigmoid。
- 适用：DeepSeek-V3 上线版本。

### F2：带 bias 的选择
$$g'_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}+b_i\in\mathrm{Topk}(\{s_{j,t}+b_j\mid 1\leq j\leq N_r\},K_r)\\ 0, & \text{otherwise}\end{cases}$$
- 来源：DeepSeek-V3 报告 Eq.16；原始论文 Eq.1-2。
- 含义：选谁由 $s+b$ 决定，但被选中后的门控值取原始 $s_{i,t}$（不含 bias）。

### F3：归一化门控权重
$$g_{i,t}=\frac{g'_{i,t}}{\sum_{j=1}^{N_r}g'_{j,t}}$$
- 来源：DeepSeek-V3 报告 §2.1 Eq.15（DeepSeekMoE 基本结构），原始论文隐式采用。
- 含义：分母只对被选中的 $K_r$ 个专家求和（其余 $g'_{j,t}=0$），且不包含 $b$。

### F4：bias 更新
$$b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(\bar c_i-c_i)$$
- 来源：原始论文 Algorithm 1，§3.2；DeepSeek-V3 报告 §2.2 描述。
- 含义：$\bar c_i=\frac{1}{N_r}\sum_j c_j$ 是平均负载；$c_i$ 是该步实际分配给专家 $i$ 的 token 数；$\gamma$ 是固定步长超参（论文中记为 $u$ 或 $\gamma$，DeepSeek-V3 报告统一记为 $\gamma$）。

### F5：MoE 层输出
$$h_t'=u_t+\sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}(u_t)+\sum_{i=1}^{N_r}g_{i,t}\mathrm{FFN}_i^{(r)}(u_t)$$
- 来源：DeepSeek-V3 报告 Eq.14（DeepSeekMoE 基本结构）。
- 含义：shared expert 全部激活（$N_s$ 个），routed expert 按 $g_{i,t}$ 加权（$N_r$ 个中只激活 $K_r$ 个）。

### F6：complementary sequence-wise balance loss
$$L_{Bal}=\alpha\sum_{i=1}^{N_r}f_i P_i,\quad f_i=\frac{N_r}{K_r T}\sum_{t=1}^T\mathbb{1}(s_{i,t}\in\mathrm{Topk}),\quad P_i=\frac{1}{T}\sum_{t=1}^T s'_{i,t}$$
- 来源：DeepSeek-V3 报告 Eq.17-19。
- 含义：本页只引此式用于说明它是 aux-loss-free 之外的互补机制，不展开推导。

## N 数字

### N1：DeepSeek-V3 规模
- 数字：671B 总参数 / 37B 每 token 激活；每 MoE 层 256 routed expert + 1 shared expert；top-8；58 个 MoE 层（前 3 层稠密）。
- 来源：DeepSeek-V3 报告摘要 §2.1；与 `moe-serving/` 已有数字一致。
- 适用：DeepSeek-V3 配置。

### N2：γ 调度
- 数字：$\gamma=0.001$ 用于前 14.3T token 训练；最后 500B token 设 $\gamma=0$（冻结 bias）；总训练 14.8T token。
- 来源：DeepSeek-V3 报告 §2.2（"bias update speed γ is set to 0.001 for the first 14.3T tokens, and 0.0 for the last 500B tokens"）。
- 适用：DeepSeek-V3 训练调度。

### N3：sequence-wise loss 系数
- 数字：$\alpha=0.0001$。
- 来源：DeepSeek-V3 报告 §2.2。
- 适用：DeepSeek-V3 训练调度。

### N4：原始论文实验配置
- 数字：1B 与 3B 参数模型；100B / 200B token 训练；bias 最佳更新率 $u=0.001$。
- 来源：arXiv:2408.15664 §4.1、Table 2、Table 3。
- 适用：原始论文实验，与 V3 上线版本一致选用 $\gamma=0.001$。

### N5：变体实验对照
- 数字：原始论文 Table 3 报告 sign 版本 PPL 9.50、MaxVio 0.044；连续幅度版本 $b_i\leftarrow b_i+\gamma e_i$ PPL 9.51、MaxVio 0.036；二者差异在实验误差内，故采用更简单的 sign。
- 来源：arXiv:2408.15664 Table 3。
- 适用：方法选择依据。
