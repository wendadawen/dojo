# EAGLE-3 投机解码 draft 模型 — 核心论断与证据

编号规则：C 论断 / F 公式 / N 数字。仅覆盖核心内容。

## C 论断

### C1
- 论断：传统投机解码用独立小语言模型作为 draft，存在「太弱接受率低、太强自身成本高」的两难；EAGLE 系列改为复用 target 模型已算出的隐藏状态作为 draft 输入，避免训练独立小模型。
- 来源定位：Li et al. 2024 (EAGLE-1) §1 Introduction，arXiv:2401.15077。原文：「Applying speculative sampling hinges on finding a draft model that mirrors the original LLM's functionality but with reduced latency, often involving a lower-parameter version from the same LLM series... Training a new, appropriately sized draft model specifically for speculative sampling is not an ideal solution either due to the high cost: TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens.」
- 适用条件：基于 LLaMA / Vicuna / Qwen 等主流 decoder-only Transformer 的投机解码场景。
- 置信状态：已确认。

### C2
- 论断：EAGLE-1 在 feature（second-to-top-layer）层做自回归比在 token 层更规律；但 feature 自回归本身存在不确定性，需要引入 time-shifted token 序列（提前一个时间步的 token embedding）来解决。
- 来源定位：Li et al. 2024 §1, §3。原文摘要：「Firstly, autoregression at the feature (second-to-top-layer) level is more straightforward than at the token level. Secondly, the inherent uncertainty in feature (second-to-top-layer) level autoregression constrains its performance. Based on these insights, we introduce EAGLE... By incorporating a token sequence advanced by one time step, EAGLE effectively resolves the uncertainty, enabling precise second-to-top-layer feature prediction with minimal overhead.」
- 适用条件：target 模型为标准 decoder-only Transformer，能暴露 second-to-top-layer hidden state。
- 置信状态：已确认。

### C3
- 论断：EAGLE-3 做了两项架构改变：（a）放弃 feature 预测、改为直接 token 预测；（b）用 target 的 low/mid/high 三层特征融合替代 second-to-top-layer 单层特征。同时引入 training-time test（TTT）训练方法。
- 来源定位：Li et al. 2025 (EAGLE-3) 摘要与 §3，arXiv:2503.01840。原文摘要：「EAGLE-3, which abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test.」
- 适用条件：EAGLE-3 设置（单层 decoder draft、target 暴露多层 feature）。
- 置信状态：已确认。

### C4
- 论断：EAGLE-3 draft 模型在推理时自回归生成 γ 个 draft token。第一步用 target 真实的融合 feature g；后续步骤因新位置的 target feature 尚未算出，用 draft 自己上一步的输出 a 替代 g 作为输入。这种「自替代」引入噪声，必须靠训练阶段 TTT 来对齐。
- 来源定位：Li et al. 2025 §3.1 Inference Pipeline。原文：「In Step 2, the prefix becomes 'How can I'. Ideally, we would reuse g_how, g_can, and g_I from the target model. However, this is not possible because the token 'I' has not yet been checked by the target model, and we cannot obtain g_I. Instead, we use the output a_I from the draft model in the previous step to replace g_I, and concatenate a_I with the embedding e_do of the sampled result 'do' as the input to the draft model in Step 1.」
- 适用条件：EAGLE-3 推理 pipeline，γ ≥ 2 时「自替代」必然发生。
- 置信状态：已确认。

### C5
- 论断：EAGLE-3 在训练时用 training-time test（TTT）：训练时让 draft 模型见到自己多步输出的近似特征（通过因果 mask 构造模拟推理的噪声环境），让模型学会在自替代噪声下仍能输出有效分布。
- 来源定位：Li et al. 2025 §3.2 Draft Model Training 与 Figure 6。原文：「The training-time test enables the draft model to simulate multi-step autoregressive generation during training, attending to its own previous predictions through custom causal masks. At each simulated position, the model restricts attention to the correct causal prefix, thereby learning to handle its own outputs as inputs—a crucial requirement for robust drafting at inference.」
- 适用条件：EAGLE-3 训练阶段。
- 置信状态：已确认。

### C6
- 论断：EAGLE-3 论文版的训练损失是 token-level 负对数似然 L_E3 = -Σ_{i=1}^{k} log q(t_{t+i} | g_{1:t}, a_{t+1:t+i-1})；K3 报告版改用 LK loss = -log Σ_{x∈V} min(p(x), q(x))，即接受率 α 的负对数。K3 选择 LK loss 是因为最小化常规 KL 散度代理不保证最大化 capacity-limited draft 模型的接受率。
- 来源定位：EAGLE-3 论文损失：Li et al. 2025 §3.2（emergentmind.com 摘要「The draft model in EAGLE-3 outputs a token distribution q at each step... The training objective for a generation prefix of length t is: L_E3 = -Σ_{i=1}^k log q(t_{t+i} | g_{1:t}, a_{t+1:t+i-1})」）。K3 LK loss：K3 报告 §4.1.4 Eq.(16)，arXiv:2503.01840 引用 [104]。原文：「Since minimizing the conventional KL-divergence surrogate does not guarantee maximizing this rate for a capacity-limited draft model, we directly optimize the likelihood-based LK loss [104], the negative logarithm of the acceptance rate itself, L_LK = -log Σ_{x∈V} min(p(x), q(x)), with p and q evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term.」
- 适用条件：EAGLE-3 训练；K3 用 LK loss 变体。
- 置信状态：已确认。

### C7
- 论断：K3 把预训练的 MTP（multi-token-prediction）层 fine-tune 成 EAGLE-3-style draft 模型。MTP 层结构镜像 backbone block、单层 decoder，天然匹配 EAGLE-3 draft 架构。Fine-tune 时 target 冻结、只更新 draft 层和 feature-fusion 投影。TTT 长度为 7 步。
- 来源定位：K3 报告 §4.1.4 Draft Model Fine-Tuning。原文：「Kimi K3 is pre-trained with a multi-token-prediction (MTP) layer that mirrors the structure of a backbone block. As the draft model of EAGLE-3 [71] comprises a single decoder layer whose structure matches the MTP layer, we fine-tune the pre-trained MTP layer into an EAGLE-3-style draft model, with the target model frozen and only the draft layer and its feature-fusion projection updated. Following the training-time test protocol of EAGLE-3, the draft is unrolled for seven steps during training...」
- 适用条件：K3 部署。
- 置信状态：已确认。

### C8
- 论断：K3 的 draft 输入融合 target 的 low/mid/high feature，分别取自第 1、4、final AttnRes blocks 的输出。融合矩阵 WE3 初始化为 [0 0 I]，使初始 fused feature 等于 high-level feature h_h（MTP 层预训练时用的输入），训练中逐渐学到 low/mid feature。
- 来源定位：K3 报告 §4.1.4。原文：「The draft input fuses low-, mid-, and high-level features of the target model, taken from the outputs of the 1st, 4th, and final AttnRes blocks, respectively (§ 2.2). These features are concatenated and projected to the hidden size by a bias-free matrix WE3, initialized as [0 0 I] so that the fused representation coincides at initialization with the high-level feature h_h — the input on which the MTP layer was pre-trained — and gradually learns to incorporate the low- and mid-level features during fine-tuning.」
- 适用条件：K3 部署。
- 置信状态：已确认。

### C9
- 论断：K3 的 draft fine-tuning 沿用后训练 QAT 配置：MoE 专家权重 MXFP4、输入激活 MXFP8、非专家模块保持高精度。这与 target 共享同一 QAT 方案，消除 train-inference mismatch。
- 来源定位：K3 报告 §4.1.4 末段。原文：「Draft fine-tuning follows the post-training QAT configuration (§ 4.1.4), with MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision.」
- 适用条件：K3 部署。
- 置信状态：已确认。

### C10
- 论断：EAGLE-3 不改变投机解码的接受/拒绝规则；输出分布与纯 target 采样完全相同（lossless）。EAGLE-3 只替换 draft 角色，target 模型的 lm_head、采样规则、残差分布都不变。
- 来源定位：投机解码框架见 Leviathan et al. 2023（arXiv:2211.17192）§3 Theorem 2；EAGLE-1 摘要「maintaining the distribution of the generated text」；EAGLE-3 摘要「ensuring lossless performance」。前置概念页 [speculative-decoding](../../wiki/speculative-decoding/index.html) 第 3 章已证明。
- 适用条件：投机解码通用条件（共享 tokenizer、draft 与 target 同词表）。
- 置信状态：已确认。

## F 公式

### F1
- 公式：EAGLE-3 多层特征融合 g_t = W_fuse · [l_t; m_t; h_t]，其中 l_t, m_t, h_t ∈ R^k 是 target 在第 t 个位置上 low/mid/high 三层 feature，[·;·;·] 是拼接，W_fuse ∈ R^{k×3k} 是无偏融合矩阵，g_t ∈ R^k 是融合后的 feature。
- 来源定位：Li et al. 2025 §3.1（emergentmind 摘要：「gt = W_fuse [f^(1)_t; ...; f^(L)_t] ∈ R^k」）。K3 的实现用 W_E3 命名融合矩阵。
- 适用条件：EAGLE-3 推理与训练全程。
- 置信状态：已确认。

### F2
- 公式：EAGLE-3 推理时单步 draft 计算 a_t = DraftLayer([g_{<t}; e_{t-1}])；q_t = softmax(W_lm · a_t)，其中 a_t ∈ R^k 是 draft 单层 decoder 输出，e_{t-1} 是上一步采样 token 的 embedding，W_lm 是 target 模型共享的 lm_head。
- 来源定位：Li et al. 2025 §3.1。原文：「The concatenated vector is then passed through an FC layer to reduce its dimensionality to k, and subsequently inputted into a single layer decoder, producing the output a. Finally, we input a_I into the LM head and sample to obtain the draft token 'do'.」
- 适用条件：EAGLE-3 推理 pipeline 每一步。
- 置信状态：已确认。

### F3
- 公式：EAGLE-3 论文版训练损失 L_E3 = -Σ_{i=1}^{k} log q(t_{t+i} | g_{1:t}, a_{t+1:t+i-1})，其中 k 是 TTT unroll 长度，t_{t+i} 是 ground-truth token，q 是 draft 模型在第 i 步的输出分布。
- 来源定位：Li et al. 2025 §3.2（emergentmind 摘要给出公式）。
- 适用条件：EAGLE-3 训练阶段。
- 置信状态：已确认。

### F4
- 公式：K3 报告版 LK loss：L_LK = -log Σ_{x∈V} min(p(x), q(x))，其中 p 是 target 分布、q 是 draft 分布、V 是词表。这是接受率 α = Σ_x min(p(x), q(x)) 的负对数。
- 来源定位：K3 报告 §4.1.4 Eq.(16)。原文：「L_LK = -log Σ_{x∈V} min(p(x), q(x)), with p and q evaluated at temperature 1 and no auxiliary ground-truth cross-entropy term.」
- 适用条件：K3 draft fine-tuning。
- 置信状态：已确认。

### F5
- 公式：接受率 α = Σ_{x∈V} min(p(x), q(x)) = 1 - TV(p, q)，其中 TV(p, q) = (1/2) Σ_x |p(x) - q(x)| 是总变差距离。α 是单位置平均接受概率。
- 来源定位：Leviathan et al. 2023 §3（α 定义）；α = 1 - TV(p, q) 由 min(a, b) = (a + b - |a - b|)/2 推出。前置概念页 [speculative-decoding](../../wiki/speculative-decoding/index.html) 第 3 章已证明。
- 适用条件：投机解码通用。
- 置信状态：已确认。

## N 数字

### N1
- 数字：EAGLE-1 在 LLaMA2-Chat 70B 上延迟加速 2.7x-3.5x，吞吐翻倍。
- 来源与实验条件：Li et al. 2024 摘要。「For LLaMA2-Chat 70B, EAGLE achieved a latency speedup ratio of 2.7x-3.5x, doubled throughput, while maintaining the distribution of the generated text.」实验设置：MT-bench，贪心解码（temperature=0）。
- 置信状态：已确认。

### N2
- 数字：EAGLE-3 加速比最高 6.5x，相比 EAGLE-2 提升约 1.4x。在 SGLang 框架中，batch size 64 时吞吐提升 1.38x。
- 来源与实验条件：Li et al. 2025 摘要。「The results show that EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2. In the SGLang framework, EAGLE-3 achieves a 1.38x throughput improvement at a batch size of 64.」实验设置：MT-bench（chat 模型）、GSM8K（reasoning 模型）等五个任务。
- 置信状态：已确认。

### N3
- 数字：EAGLE-3 在 Vicuna 13B 上比 vanilla decoding 快 5.6x，比 EAGLE-1 快 1.8x。推理设置：2× RTX 3090 GPU，fp16 精度。
- 来源与实验条件：Hugging Face EAGLE 模型卡 README（yuhuili/EAGLE-Qwen2-72B-Instruct）。「EAGLE-3 is: 5.6 faster than vanilla decoding (13B). 1.8x faster than EAGLE-1 (13B). Inference is conducted on 2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model.」
- 置信状态：已确认。

### N4
- 数字：K3 draft fine-tuning 的 TTT unroll 长度为 7 步。
- 来源与实验条件：K3 报告 §4.1.4。「Following the training-time test protocol of EAGLE-3, the draft is unrolled for seven steps during training; beyond the first step, where the target-side features of the newest position are unavailable, the draft consumes its own outputs from earlier steps, mirroring the recurrent drafting procedure at inference.」
- 置信状态：已确认。

### N5
- 数字：EAGLE-1 训练用 2-4B tokens，而 TinyLLaMA 训练用 3000B tokens。
- 来源与实验条件：Li et al. 2024 §1。「TinyLLaMA is trained on 3,000B tokens, whereas EAGLE is trained on 2-4B tokens.」用于说明 EAGLE 复用 target feature 让训练成本远低于独立训练小模型。
- 置信状态：已确认。
