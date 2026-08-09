# 辅助损失无关路由：术语表

| 名称 | 首次出现 | 定义/含义 |
|---|---|---|
| MoE（Mixture-of-Experts，专家混合） | S1 | Transformer 层把 FFN 换成一排专家 + router，每个 token 只激活 top-k 个；前置概念页 `moe-serving/` |
| expert（专家） | S1 | MoE 层里的一排小 FFN 之一；分为 routed expert（按路由激活）与 shared expert（全激活）|
| router（路由器） | S1 | MoE 层里给所有专家打分的小模块 |
| top-k / $K_r$ | S1 | 每个 token 只激活 $K_r$ 个得分最高的专家 |
| 路由坍塌（routing collapse） | S1 | 路由器把多数 token 路由到少数专家、其余专家训不动的失效模式 |
| 专家并行（EP, expert parallelism） | S1 | 把专家分片到多张 GPU 的并行方式 |
| 辅助损失负载均衡 | S1 | 在损失函数里加 $L_{bal}=\alpha\sum f_i P_i$ 一类项推动均衡；与 aux-loss-free 对照 |
| 干扰梯度 | S1 | 辅助损失项产生的、混进主任务梯度的额外梯度，扭曲专家特化 |
| sequence-wise balance loss | S1 | DeepSeek-V3 在 aux-loss-free 之外另加的小权重损失项，专门处理单序列内 token 间均衡 |
| 路由分数 $s_{i,t}$ | S2 | router 给 token $t$、专家 $i$ 的打分；DeepSeek-V3 用 $s_{i,t}=\sigma(u_t^\top e_i)$ |
| bias $b_i$ | S2 | 专家 $i$ 的偏置；加到 $s_{i,t}$ 上做 top-k 选择，不进 mixture weight |
| top-k 选择 $\mathrm{Topk}(\{s_{j,t}+b_j\},K_r)$ | S2 | 取加 bias 后分数最高的 $K_r$ 个专家 |
| $g'_{i,t}$ | S2 | 未归一化门控：被选中则取 $s_{i,t}$，未被选中则为 0 |
| $g_{i,t}$ | S2 | 归一化门控权重：$g'_{i,t}/\sum_j g'_{j,t}$ |
| MoE 层输出 $h_t'$ | S2 | shared expert 全激活 + routed expert 按 $g_{i,t}$ 加权 |
| 负载 $c_i$ | S3 | 一个训练 batch 中分给专家 $i$ 的 token 数 |
| 平均负载 $\bar c_i$ | S3 | $\frac{1}{N_r}\sum_j c_j$，所有专家的均值 |
| 偏差 $e_i$ | S3 | $\bar c_i-c_i$，正表示欠载、负表示过载 |
| 固定步长 sign 更新 | S3 | $b_i\leftarrow b_i+\gamma\cdot\mathrm{sign}(e_i)$；规则式，不进反向传播 |
| 步长 $\gamma$（bias update speed） | S3 | bias 每步走多大；原始论文记号 $u$，DeepSeek-V3 报告统一记 $\gamma$ |
| 因果约束 | S3 | 用历史 batch 负载更新 bias，避免用到当前序列的 future token 信息 |
| $\gamma$ 调度 | S5 | DeepSeek-V3 中 $\gamma$ 在前 14.3T token 取 0.001，最后 500B token 取 0（冻结） |
| $\alpha$ | S5 | sequence-wise balance loss 的系数，DeepSeek-V3 取 0.0001 |
| Quantile Balancing | S5 | K3 的改进方法，仍属 aux-loss-free 家族，把 bias 更新从 sign 换成从分位数一步算出；概念页 `quantile-balancing/` |
| sign 函数 | S3 | $\mathrm{sign}(x)=+1$ 若 $x>0$，$-1$ 若 $x<0$，$0$ 若 $x=0$ |
| sigmoid $\sigma$ | S2 | $\sigma(x)=1/(1+e^{-x})$，把任意实数压到 (0,1) |
| $\bar c_i$ vs $\mathrm{Load}_i$ | — | 原始论文记号为 $c_i$ 与 $\bar c_i$；DeepSeek-V3 报告里把负载写为 $\mathrm{Load}_i$，本文统一用 $c_i$ 与 $\bar c_i$（与原始论文一致） |
