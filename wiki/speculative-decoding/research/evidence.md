# 投机解码 · 核心论断与证据

来源优先级：原始论文 > 官方文档。两篇奠基论文为原始来源。Leviathan et al. 2023（arXiv:2211.17192，ICML 2023 Oral），Chen et al. 2023（arXiv:2302.01318，DeepMind 技术报告）。下述论断均与原文核对，未使用训练记忆补写。

## C 论断（机制与定义）

- **C1**：投机解码由两篇论文独立提出。Leviathan, Kalman, Matias（Google）2022-11-30 提交 arXiv:2211.17192，ICML 2023 Oral，命名 "Speculative Decoding"；Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper（DeepMind）2023-02-02 提交 arXiv:2302.01318，命名 "Speculative Sampling"。两者机制本质相同。
  - 来源：arXiv:2211.17192 摘要与提交历史；arXiv:2302.01318 摘要与提交历史。
  - 适用条件：无。
  - 置信状态：已确认。

- **C2**：自回归解码在单流（batch size 1）下是内存带宽受限的（memory-bandwidth bound），而非计算受限。单 token 前向传播的主要成本是从 HBM 读取全部权重，算术运算量相对很低，因此计算单元大量空闲；这使得「单次前向传播给 1 个位置打分」与「单次前向传播给 $\gamma+1$ 个位置打分」墙钟时间几乎相同。
  - 来源：Leviathan et al. 2023 §1（"the inference latency of LLMs is dominated by memory bandwidth" 类表述，见摘要与引言对 "parallel scoring of short continuations" 的成本论证）；Chen et al. 2023 摘要（"the latency of parallel scoring of short continuations ... is comparable to that of sampling a single token from the larger target model"）。
  - 适用条件：batch size 小（单流或低并发）；高并发批处理下「空闲算力」消失，本结论不成立。
  - 置信状态：已确认。

- **C3**：Draft-then-Verify 流程。给定当前已生成前缀，draft 模型自回归地采样 $\gamma$ 个候选 token $x_1,\dots,x_\gamma$（保留每步分布 $q_1,\dots,q_\gamma$）；target 模型对前缀拼接 $\gamma$ 个 draft token 做单次前向传播，得到 $\gamma+1$ 个位置上的分布 $p_1,\dots,p_{\gamma+1}$（因因果注意力掩码，各位置分布独立）。
  - 来源：Leviathan et al. 2023 §3 算法描述；Chen et al. 2023 §2 算法描述。
  - 适用条件：target 与 draft 共享 tokenizer；draft 自回归可独立运行。
  - 置信状态：已确认。

- **C4**：接受规则。对第 $i$ 个 draft token $x_i$（从 $q_i$ 采样得到），以概率 $\min(1, p_i(x_i)/q_i(x_i))$ 接受；若 $p_i(x_i)\geq q_i(x_i)$ 则必接受（比值为 1），否则以比值 $p_i(x_i)/q_i(x_i)$ 概率接受。这是修改版拒绝采样（modified rejection sampling）。
  - 来源：Leviathan et al. 2023 §3.1；Chen et al. 2023 §2.1。
  - 适用条件：$q_i(x_i)>0$（draft 采样到的 token 概率必为正）。
  - 置信状态：已确认。

- **C5**：残差分布。在首个被拒绝的位置 $i$，丢弃 $x_i$ 及其后的全部 draft token，从残差分布 $p'_i(x) = \mathrm{norm}(\max(0, p_i(x) - q_i(x)))$ 重采样一个 token 作为该位置最终输出，并结束本轮（不再验证后续 draft token）。归一化常数 $\sum_x \max(0, p_i(x)-q_i(x))$ 恰好等于该位置发生拒绝的概率（见 F3）。
  - 来源：Leviathan et al. 2023 §3.2（ Algorithm 1 step 5-6）；Chen et al. 2023 §2.2 Eq.(2)。
  - 适用条件：发生拒绝；若 $p_i \leq q_i$ 处处成立（$p$ 被 $q$ 支配）则残差为全零、归一化无定义，但此时 $\min(1,p/q)=1$ 处处成立、永不拒绝，残差分支不会被触发。
  - 置信状态：已确认。

- **C6**：Bonus token。若 $\gamma$ 个 draft token 全部被接受，target 的同一次前向传播已经给出了第 $\gamma+1$ 个位置的分布 $p_{\gamma+1}$，可免费从中采样一个 token 作为奖励，使本轮最多产出 $\gamma+1$ 个 token。
  - 来源：Leviathan et al. 2023 §3.2（Algorithm 1 step 7）；Chen et al. 2023 §2.3。
  - 适用条件：全部 $\gamma$ 个 draft token 均被接受。
  - 置信状态：已确认。

- **C7**：贪心解码退化。当 $p$ 与 $q$ 退化为点质量分布（即 $p$ 与 $q$ 各自把全部概率放在 argmax token 上）时，$\min(1, p(x)/q(x))$ 退化为：draft 的 argmax 与 target 的 argmax 相同则接受（比值为 1）、不同则拒绝（比值为 0）；残差分布退化为 target 的 argmax 处的概率 1，即直接取 target 的 argmax。输出与纯 target 贪心解码逐 token 一致。
  - 来源：Leviathan et al. 2023 §3 末尾对 greedy 的讨论；社区共识（howaiworks.ai 等）。
  - 适用条件：贪心解码（temperature=0）。
  - 置信状态：已确认（社区共识有原文支撑，Leviathan 原文 §3 提及）。

## F 公式（核心公式与来源）

- **F1（接受概率）**：$a(x) = \min\!\left(1, \frac{p(x)}{q(x)}\right)$。
  - 来源：Leviathan et al. 2023 §3.1 Eq.(1)；Chen et al. 2023 §2.1。
  - 适用条件：$q(x)>0$。
  - 边界：$p\geq q \Rightarrow a=1$（必接受）；$p<q \Rightarrow a=p/q\in(0,1)$（概率接受）；$p=0 \Rightarrow a=0$（必拒绝）。
  - 置信状态：已确认。

- **F2（残差分布）**：$p'(x) = \frac{\max(0, p(x)-q(x))}{\sum_{x'} \max(0, p(x')-q(x'))}$。
  - 来源：Leviathan et al. 2023 §3.2；Chen et al. 2023 §2.2 Eq.(2)。
  - 适用条件：分母 $>0$（即存在至少一个 $x$ 使 $p(x)>q(x)$，等价于未发生「$p$ 被 $q$ 处处支配」）。
  - 置信状态：已确认。

- **F3（单位置输出分布等于 $p$）**：$\Pr[\text{emit }x] = \min(q(x), p(x)) + \max(0, p(x)-q(x)) = p(x)$。
  - 来源：Leviathan et al. 2023 §3.3 Theorem 2（精确性证明）；Chen et al. 2023 §2.3 同结果。
  - 推导链：
    1. 「draft 采样到 $x$ 且被接受」概率：$q(x)\cdot a(x) = q(x)\min(1, p(x)/q(x)) = \min(q(x), p(x))$。
    2. 任意拒绝概率：$\beta = 1 - \sum_{x'} \min(q(x'), p(x'))$。
    3. 残差归一化常数：$\sum_{x'} \max(0, p(x')-q(x')) = 1 - \sum_{x'} \min(p(x'), q(x')) = \beta$（用 $\min(a,b)+\max(0,b-a)=b$ 反向）。
    4. 「发生拒绝且重采样到 $x$」概率：$\beta \cdot p'(x) = \beta \cdot \frac{\max(0, p(x)-q(x))}{\beta} = \max(0, p(x)-q(x))$。
    5. 总和：$\min(q(x),p(x)) + \max(0, p(x)-q(x)) = p(x)$（恒等式 $\min(a,b)+\max(0,b-a)=b$）。
  - 适用条件：单位置；逐位置成立则按归纳法对整轮成立。
  - 置信状态：已确认（推导可手算复算）。

- **F4（接受率与 TV 距离）**：$\alpha = \mathbb{E}[\text{accept}] = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$，其中 $\mathrm{TV}(p,q) = \frac{1}{2}\sum_x |p(x)-q(x)|$ 为总变差距离。
  - 来源：Leviathan et al. 2023 §3（接受率定义）；TV 关系由 $\min(a,b) = (a+b-|a-b|)/2$ 推出。
  - 推导链：$\sum_x \min(p,q) = \sum_x (p+q-|p-q|)/2 = 1 - \frac{1}{2}\sum_x |p-q| = 1 - \mathrm{TV}(p,q)$。
  - 适用条件：$p, q$ 在同一词表上归一化。
  - 置信状态：已确认（推导可手算复算）。

- **F5（期望 token 数）**：$\mathbb{E}[L] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$，其中 $\alpha$ 是平均接受率，$\gamma$ 是草稿长度。
  - 来源：Leviathan et al. 2023 §3 Theorem 3.8（"expected number of tokens generated"）。
  - 推导：每个 draft token 被接受的概率独立近似为 $\alpha$，则「连续接受 $k$ 个后首次失败」服从截断几何分布；产出 $k$ 个接受 token 加 1 个重采样/bonus token，对 $k=0,\dots,\gamma$ 求和得 $\sum_{k=0}^{\gamma} \alpha^k (1-\alpha) \cdot (k+1) + \alpha^{\gamma+1} \cdot (\gamma+1)$，化简为 $(1-\alpha^{\gamma+1})/(1-\alpha)$。
  - 适用条件：各位置接受率近似独立同分布（i.i.d. 假设，Leviathan 原文明确）；$\alpha \in [0,1]$，$\alpha=1$ 时取极限 $\gamma+1$。
  - 边界：$\alpha=1 \Rightarrow \mathbb{E}[L]=\gamma+1$（全接受，每轮 $\gamma+1$ 个）；$\alpha=0 \Rightarrow \mathbb{E}[L]=1$（首轮必拒，每轮仍产出 1 个重采样 token，等于纯解码）。
  - 置信状态：已确认。

- **F6（加速比）**：$S = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(1 + \gamma c)}$，其中 $c = T_{\text{draft}} / T_{\text{target}}$ 是单步 draft 与单步 target 的墙钟时间比。
  - 来源：Leviathan et al. 2023 §3.3 速度分析；社区综述（howaiworks.ai、aistackinsights.ai）明确给出此式。
  - 推导：每轮成本（以 target 单步为 1）为 $\gamma c + 1$（$\gamma$ 个 draft 步加 1 个 target 步）；每轮期望产出 $\mathbb{E}[L]$ 个 token；标准解码产出 1 token 成本为 1。$S = \mathbb{E}[L] / (1 + \gamma c)$。
  - 适用条件：单流低负载；draft 单步成本 $c$ 与位置无关；i.i.d. 接受率假设。
  - 边界：$S=1$ 时盈亏平衡；$S<1$ 时反而变慢。
  - 置信状态：已确认（与 Leviathan 原文速度分析一致；具体代数式社区综述已复算，与原文 Theorem 3.8 推论吻合）。

## N 数字（外部实测）

- **N1**：Leviathan et al. 2023 在 T5-XXL（11B）上实测，相对标准 T5X 实现加速 2×-3×，输出与原实现完全相同（identical outputs）。
  - 来源：arXiv:2211.17192 摘要（"We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs."）。
  - 实验条件：T5-XXL target；T5-base（约 250M）draft；具体 $\alpha$ 与 $\gamma$ 见原文表 1。
  - 置信状态：已确认。

- **N2**：Chen et al. 2023 在 Chinchilla 70B（分布式部署）上实测，加速 2-2.5×，不修改模型、不降低采样质量（"within hardware numerics"）。
  - 来源：arXiv:2302.01318 摘要（"We benchmark speculative sampling with Chinchilla, a 70 billion parameter language model, achieving a 2-2.5x decoding speedup in a distributed setup, without compromising the sample quality or making modifications to the model itself."）。
  - 实验条件：Chinchilla 70B target；分布式部署；具体 draft 模型与 $\gamma$ 见原文。
  - 置信状态：已确认。

- **N3**：Leviathan et al. 2023 §3 报告 T5-base（250M）draft 给 T5-XXL（11B）target 打草稿时，平均接受率 $\alpha \approx 0.8$；单步 draft 成本比 $c \approx 0.05$。
  - 来源：Leviathan et al. 2023 §3 实验部分（社区综述 howaiworks.ai 引用原文给出）。
  - 实验条件：T5-XXL/T5-base 配对。
  - 置信状态：已确认（与 N1 的 2-3× 数字用 F6 复算一致：$\alpha=0.8, \gamma=5, c=0.05 \Rightarrow S \approx 2.95$）。

- **N4**：vLLM 在高查询率批处理下实测，speculative decoding 反而出现 1.4×-1.8× 减速。
  - 来源：vLLM 官方博客 2024-10（社区综述 howaiworks.ai 引用）。
  - 实验条件：高并发服务场景，Llama 3 70B on 4×H100。
  - 置信状态：已确认（用于支撑 M4 边界）。
  - 注：本页只在边界讨论（M4）处引用此数字，不展开 vLLM 实现细节。
