# EAGLE-3 投机解码 draft 模型 — 术语表

| 术语 / 缩写 / 符号 | 首次出现位置 | 定义 / 含义 |
|---|---|---|
| EAGLE | S1 | Extrapolation Algorithm for Greater Language-model Efficiency。Li et al. 2024 提出的 draft 模型系列，复用 target 模型 hidden states 作为 draft 输入 |
| EAGLE-1 | S2 | EAGLE 系列初版，3-4 层 decoder，在 second-to-top-layer feature 空间做自回归 |
| EAGLE-2 | S6（边界） | EAGLE 系列第二版，引入动态 draft tree；EAGLE-3 沿用其动态 tree 机制 |
| EAGLE-3 | S3 | EAGLE 系列第三版，单层 decoder + 直接 token 预测 + 多层 feature 融合 + TTT |
| target 模型 | S1 | 投机解码中的大模型，分布 p 是希望最终采样的目标分布 |
| draft 模型 | S1 | 投机解码中预测 γ 个候选 token 的小模型，分布 q |
| speculative decoding / 投机解码 | S1 | draft-then-verify + 拒绝采样的推理加速框架，见前置概念页 |
| 接受率 α | S1 | 单位置平均接受概率，α = Σ_x min(p(x), q(x)) = 1 - TV(p, q) |
| 草稿长度 γ | S1 | 一轮投机解码中 draft 模型生成的候选 token 数 |
| 加速比 S | S1（引用） | 标准解码成本与投机解码成本的比值，S = E[L] / (1 + γc) |
| 单步成本比 c | S1（引用） | 单步 draft 墙钟时间与单步 target 墙钟时间的比 |
| feature / 隐藏状态 / hidden state | S2 | target 模型某一层的输出向量，是 draft 模型的输入 |
| second-to-top-layer feature | S2 | target 模型倒数第二层（lm_head 之前）的输出 feature，EAGLE-1 在此层做自回归 |
| top-layer feature | S3 | target 模型最后一层（lm_head 之前）的输出 feature |
| low / mid / high-level feature | S3 | target 模型低层、中层、高层 feature；EAGLE-3 取三处特定层的输出做融合 |
| feature fusion / 特征融合 | S3 | 拼接 low/mid/high 三层 feature 后过 FC 投影到 hidden size，得到 g |
| g_t | S3, F1 | 第 t 个位置上融合后的 feature，g_t = W_fuse · [l_t; m_t; h_t] ∈ R^k |
| l_t, m_t, h_t | S3, F1 | 第 t 个位置上 target 的 low/mid/high feature，各 ∈ R^k |
| W_fuse / W_E3 | S3, F1 | feature fusion 的无偏投影矩阵 ∈ R^{k×3k}；K3 命名为 W_E3 |
| a_t | S4, F2 | draft 单层 decoder 在第 t 步的输出 ∈ R^k，作为下一步输入的 g 替代 |
| e_t | S4, F2 | 第 t 个 token 的 embedding ∈ R^k |
| DraftLayer / 单层 decoder | S4, F2 | EAGLE-3 的 draft 模型，由单层 transformer decoder 构成 |
| W_lm / lm_head | S4, F2 | target 模型共享的语言模型头，把 a_t 映射为 token 分布 q_t |
| q_t | S4, F2 | draft 在第 t 步输出的 token 分布 |
| p_t | S4（引用） | target 在第 t 步的 token 分布 |
| time-shifted token | S2 | EAGLE-1 引入的、提前一个时间步的 token embedding，用于解决 feature 自回归的不确定性 |
| feature 自回归 | S2 | 在 feature 空间做自回归（EAGLE-1 机制），区别于 token 自回归 |
| 自替代 / self-substitution | S4 | EAGLE-3 推理时后续步骤用 draft 自己上一步的输出 a 替代 target 真实 g 作为输入 |
| train-inference mismatch | S5 | 训练时见到的输入分布与推理时实际输入分布不一致；自替代引入的噪声是 EAGLE-3 的核心 mismatch |
| training-time test / TTT | S5 | EAGLE-3 训练方法：训练时让 draft 见到自己多步输出的近似特征，模拟推理噪声 |
| TTT unroll 长度 / k | S5, F3 | TTT 训练时展开的 draft 步数；K3 用 k = 7 |
| 因果 mask | S5 | TTT 训练时让位置 i 只能看到 ≤ i 的真实 token 与 ≤ i-1 的 draft 预测的注意力 mask |
| L_E3 | S5, F3 | EAGLE-3 论文版训练损失，token-level 负对数似然 L_E3 = -Σ log q(t_{t+i} \| g_{1:t}, a_{t+1:t+i-1}) |
| LK loss / L_LK | S5, F4 | K3 报告版损失，接受率的负对数 L_LK = -log Σ_x min(p(x), q(x)) |
| KL 散度 / KL divergence | S5 | 分布距离代理，minimizing KL 不直接等价于 maximizing 接受率 α |
| TV / 总变差距离 | S5, F5 | TV(p, q) = (1/2) Σ_x \|p(x) - q(x)\|，与 α 的关系 α = 1 - TV(p, q) |
| MTP / multi-token-prediction | S6 | 预训练时让模型一次预测多个未来 token 的训练目标；K3 用其作为 EAGLE-3 draft 的初始化 |
| AttnRes block | S6 | K3 模型的注意力残差块（K3 报告 §2.2），三层 feature 取自 1st、4th、final AttnRes blocks |
| QAT / 量化感知训练 | S6 | 训练时模拟量化精度损失，使模型部署时用真正的量化权重仍保持精度；见 [mxfp4-qat](../../wiki/mxfp4-qat/index.html) |
| MXFP4 | S6 | OCP 微缩浮点 4-bit 格式，K3 用于 MoE 专家权重；见 [mxfp4-qat](../../wiki/mxfp4-qat/index.html) |
| MXFP8 | S6 | OCP 微缩浮点 8-bit 格式，K3 用于 MoE 专家权重的输入激活 |
| W_E3 = [0 0 I] | S6 | K3 的 W_E3 初始化为列拼接的 [0; 0; I]（0 为零矩阵、I 为单位矩阵），使初始 fused feature 等于 high-level feature h_h |
| lossless / 无损 | S1, C10 | 输出分布与纯 target 采样完全相同；EAGLE 系列保持这一性质 |
| draft-then-verify | S1（引用） | 投机解码的一轮五步流程，见前置概念页 |
