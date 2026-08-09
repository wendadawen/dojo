# EAGLE-3 投机解码 draft 模型 — 教学大纲

## 1. 页面开头

### 1.1 钩子场景

让一个 70B 模型用投机解码加速推理，draft 模型该是什么？传统答案是一个独立小语言模型（如 LLaMA2-7B 给 LLaMA2-70B 打草稿）。但小模型有「太弱接受率低、太强自身成本高」的两难。EAGLE 系列给出的答案反直觉：不要训练独立小模型，直接复用 target 模型已经算出来的隐藏状态作为 draft 输入。EAGLE-3 把这个思路推到极致：draft 只有单层 decoder，靠 target 的低、中、高三层特征融合 + training-time test 训练，在 Vicuna 13B 上比 vanilla decoding 快 5.6x。

### 1.2 一句话解释

EAGLE-3 是一种单层 decoder 的 draft 模型，用于投机解码；它复用 target 大模型已算出的低、中、高三层隐藏状态作为输入，直接预测 token 分布，并用 training-time test 训练让自己在多步预测中保持接受率不衰减。

### 1.3 要解决的具体问题

构造一个既低成本（参数量远小于 target）又高接受率（分布 q 接近 target 分布 p）的 draft 模型。

### 1.4 学习承诺

读完这一页，你应该能够：

1. 说出 EAGLE 系列如何复用 target 的隐藏状态替代独立小模型 draft，并解释为什么这能同时降低 draft 成本和提高接受率（Q1）。
2. 说出 EAGLE-3 在 EAGLE-1 基础上做了哪两项架构改变，以及每项改变解决的具体问题（Q2）。
3. 写出推理时单层 decoder 自回归生成 γ 个 draft token 的 pipeline，解释为什么后续步骤要用自己的输出替代 target feature（Q3）。
4. 说出 training-time test 如何让 draft 在多步预测中保持接受率，并说明 K3 用 LK loss 比常规 KL 更直接优化接受率的原因（Q4）。
5. 说出 K3 如何把预训练的 MTP 层 fine-tune 成 EAGLE-3 draft 模型（Q5）。

### 1.5 首个具体场景

固定 prefix = "How can I"、γ = 3 个 draft token，跟随 EAGLE-3 推理 pipeline 走一遍三步，看 draft 如何在第一步用 target 真实 feature、后续步骤用自己的输出替代。这个贯穿例子在 S4 展开。

### 1.6 与第一章的过渡

S1 解释传统独立小模型的两难与 EAGLE 的思路转向。

## 2. 章节设计

### S1：为什么独立小模型 draft 有两难——EAGLE 的思路转向

- 主要教学问题：为什么需要新的 draft 方法？传统独立小模型 draft 有什么两难？EAGLE 系列的核心思路是什么？
- 对应范围：Q1；C1, C10；F5
- 正文要点：
  - 复用前置概念页 [speculative-decoding](../../wiki/speculative-decoding/index.html) 的结论：接受率 α = Σ_x min(p, q) 决定加速比，draft 与 target 越接近、α 越接近 1、加速越高。
  - 传统独立小模型 draft 的两难：太弱（如 TinyLLaMA）接受率低、α 趋近 0；太强（如 LLaMA2-7B 给 70B 打草稿）draft 自身成本 c 接近 target，加速比退化。
  - 训练独立小模型的成本：TinyLLaMA 用 3000B tokens 训练（[N5]）。
  - EAGLE 思路：不训练独立小模型，复用 target 已算出的隐藏状态作为 draft 输入。EAGLE-1 训练只需 2-4B tokens（[N5]），因 target feature 已经包含丰富上下文。
  - EAGLE 不改变投机解码框架：接受/拒绝规则、残差分布、bonus token 都不变（[C10]）；EAGLE 系列只替换 draft 角色，输出分布仍与纯 target 采样完全相同（lossless）。
- 讲解材料及职责：
  - 复用 speculative-decoding 的 α 与 γ 公式 → 说明 EAGLE 改变的是 α 的来源，不改变 α 决定加速比的框架。
  - 对照表格（传统独立小模型 vs EAGLE 系列的 draft 输入来源、训练成本、与 target 的关系）→ 澄清两类 draft 的本质区别。
- 前置知识安排：
  - 引用 [speculative-decoding](../../wiki/speculative-decoding/index.html) 接受率 α、γ、draft-then-verify 框架；不重新讲。
- 完成检查：
  - 用一句话说出传统独立小模型 draft 的两难。
  - 用一句话说出 EAGLE 系列的核心思路（复用 target hidden states）。
  - 说出 EAGLE 是否改变了投机解码的接受/拒绝规则。
- 过渡：本章说明 EAGLE 的思路转向——「复用 target feature」——但具体怎么复用？EAGLE-1 给出第一个答案：在 feature 空间做自回归。下一章讲 EAGLE-1 的具体机制。

### S2：EAGLE-1 的核心机制——在 feature 空间做自回归

- 主要教学问题：EAGLE-1 如何复用 target 的 hidden states 做 draft？
- 对应范围：Q1；C2；N1
- 正文要点：
  - 关键观察 1：feature（second-to-top-layer）层自回归比 token 层更规律。token 序列是自然语言的离散变换，feature 序列是 model 内部表示、更连续可预测。
  - 关键观察 2：feature 自回归本身存在不确定性——仅靠历史 feature 预测下一步 feature，模型不知道下一步采了哪个 token。
  - 解决方案：引入 time-shifted token 序列（提前一个时间步的 token embedding）作为额外输入，给 draft 提供「下一步采了什么」的信息。
  - draft 模型结构：3-4 层 decoder，输入 = second-to-top-layer feature + shifted token embedding，输出 = 预测下一步 feature。
  - 输出 feature → 过 target 共享的 lm_head → 得到 token 分布 q → 采样。
  - 性能：EAGLE-1 在 LLaMA2-Chat 70B 上延迟加速 2.7x-3.5x，吞吐翻倍（[N1]）。
- 讲解材料及职责：
  - ASCII 图示展示 EAGLE-1 的 draft 输入与输出 → 澄清「feature 自回归」与「token 自回归」的区别。
  - 对照表格（EAGLE-1 vs vanilla speculative decoding 的 draft 输入、输出、训练成本）→ 澄清 EAGLE-1 与传统独立小模型 draft 的区别。
- 前置知识安排：
  - 引用 [speculative-decoding](../../wiki/speculative-decoding/index.html) 的 draft-then-verify 与拒绝采样；不重新讲。
- 完成检查：
  - 说出 EAGLE-1 在哪一层 feature 做自回归（second-to-top-layer）。
  - 说出 time-shifted token 序列解决什么问题（feature 自回归的不确定性）。
  - 说出 EAGLE-1 的 draft 输出是 feature 还是 token（feature，再过 target lm_head）。
- 过渡：EAGLE-1 在 feature 空间做自回归，但 feature 预测本身有容量瓶颈——EAGLE-3 发现扩大训练数据对 EAGLE-1 提升有限。下一章讲 EAGLE-3 的两项架构改变。

### S3：EAGLE-3 的两项架构改变——直接 token 预测 + 多层特征融合

- 主要教学问题：EAGLE-3 在 EAGLE-1 基础上做了哪两项架构改变？为什么这两项改变能进一步提升接受率？
- 对应范围：Q2；C3；N2, N3
- 正文要点：
  - 改变 1：放弃 feature 预测、改为直接 token 预测。
    - EAGLE-1 的瓶颈：feature 回归任务占用了 draft 容量，扩大训练数据对 EAGLE-1 提升有限（论文 Figure 1 的 scaling law 曲线在 EAGLE 上饱和）。
    - EAGLE-3 让 draft 直接输出 token 分布 q、不再回归 feature。draft 容量全部用于贴近 target 的 token 分布。
  - 改变 2：用 low/mid/high 三层 feature 融合替代 second-to-top-layer 单层 feature。
    - 单层 top-layer feature 主要承载 next-token 预测信息，丢失了低层句法/词法、中层语义信息。
    - 三层 feature 拼接后过 FC 投影到 hidden size，得到融合 feature g（[F1]）。
    - 低层 feature 对采样误差更鲁棒（不随采样噪声快速恶化），高层 feature 定位全局语义。
  - 架构：单层 transformer decoder（EAGLE-1 是 3-4 层，EAGLE-3 减到 1 层）+ feature fusion FC + 共享 target lm_head。
  - 性能：EAGLE-3 在 Vicuna 13B 上比 vanilla 快 5.6x、比 EAGLE-1 快 1.8x（[N3]）；最高加速 6.5x（[N2]）；SGLang batch=64 吞吐 1.38x（[N2]）。
- 讲解材料及职责：
  - 对照表格（EAGLE-1 vs EAGLE-3 的 draft 层数、输入 feature、输出空间、训练方法）→ 让两项改变一目了然。
  - ASCII 图示展示 EAGLE-3 的 feature fusion + 单层 decoder pipeline → 澄清架构。
- 前置知识安排：
  - 引用 S2 的 EAGLE-1 机制作为对比基准。
- 完成检查：
  - 说出 EAGLE-3 的两项架构改变（直接 token 预测 + 多层 feature 融合）。
  - 说出每项改变解决的具体问题（feature 回归容量瓶颈 + top-layer feature 信息局限）。
  - 说出 EAGLE-3 draft 的层数（单层）与 EAGLE-1 的对比。
- 过渡：架构讲完了——但推理时单层 decoder 怎么自回归生成 γ 个 token？为什么后续步骤要用自己的输出替代 target feature？下一章用贯穿例子手算走一遍。

### S4：推理时的自回归 draft——单层 decoder 如何生成 γ 个 draft token

- 主要教学问题：单层 decoder 如何在推理时自回归生成 γ 个 token？为什么第一步用 target 真实 feature、后续步骤要用自己的输出替代 target feature？
- 对应范围：Q3；C4, F1, F2
- 正文要点：
  - 推理 pipeline 三步（论文 Figure 5）：
    - prefill 阶段：target 对 prefix 做一次前向，得到所有位置的 low/mid/high feature；融合得到 g_1, ..., g_t。
    - draft 步骤 1：输入 (g_1, ..., g_t, e_{t+1}) → 单层 decoder → 输出 a_{t+1} → lm_head → q_1 → 采样得到 x_1。
    - draft 步骤 2：新位置 t+1 的 target feature g_{t+1} 尚未算出（target 还没验证），用 draft 上一步输出 a_{t+1} 替代 g_{t+1}；输入 (g_1, ..., g_t, a_{t+1}, e_{t+2}) → 单层 decoder → 输出 a_{t+2} → lm_head → q_2 → 采样得到 x_2。
    - draft 步骤 3：同理用 a_{t+2} 替代 g_{t+2}，继续。
  - 为什么必须替代：新位置的 target feature 在 target 验证前不存在；draft 只能用自己的输出近似。
  - 这个「自替代」引入噪声：a 是 draft 单层 decoder 的近似输出，与真实 g 有偏差。噪声会随 draft 深度累积，接受率随深度衰减。
  - 必须靠训练阶段 TTT 对齐（下一章）。
  - 贯穿手算例子：prefix = "How can I"、γ = 3、hidden size k = 4、词表 V = {I, do, it, now, &lt;other&gt;}。
    - 给出 target 真实 g_how, g_can。
    - 给出简化的 W_a (4×8) 与 W_lm (5×4)，让数字便于手算。
    - 第一步：输入 (g_can, e_I) → a_I → lm_head → q_1 → 采样 "do"。
    - 第二步：用 a_I 替代 g_I，输入 (a_I, e_do) → a_do → lm_head → q_2 → 采样 "it"。
    - 第三步：用 a_do 替代 g_do，输入 (a_do, e_it) → a_it → lm_head → q_3 → 采样 "now"。
    - 对照真实 g_I 与 draft 近似 a_I 的 L2 偏差，说明「自替代」噪声的具体大小。
  - 伪代码：把三步形式化。
- 讲解材料及职责：
  - ASCII 图示展示三步推理 pipeline → 澄清「第一步真实、后续自替代」的时序关系。
  - 数字例子（教学构造）→ 让「自替代」的噪声变成可计算的偏差。
  - 伪代码 → 形式化 pipeline，便于后续对照训练损失。
- 前置知识安排：
  - 引用 S3 的 EAGLE-3 架构；不重新讲。
- 完成检查：
  - 写出推理 pipeline 的三步，标明哪些步骤用 target 真实 feature、哪些用 draft 自身输出。
  - 解释为什么后续步骤必须用 draft 自己的输出替代 target feature。
  - 给定简化数字，手算 a_t 与 q_t 一步。
  - 说出「自替代」噪声如何影响接受率。
- 过渡：推理时 draft 必须自替代、噪声会累积——训练时如何让 draft 在噪声下仍输出有效分布？下一章讲 TTT 与损失函数。

### S5：训练 draft 模型——training-time test 与接受率损失

- 主要教学问题：如何训练 draft 模型让它在多步预测中保持接受率不衰减？为什么 K3 直接用接受率的负对数作为损失比常规 KL 散度更直接？
- 对应范围：Q4；C5, C6, F3, F4
- 正文要点：
  - 训练时的核心问题：若训练时只用 ground-truth g 与 ground-truth token，draft 永远见不到「自己的近似 a 作为输入」的情形；推理时遇到自替代会失配，接受率随深度快速衰减。
  - TTT 解决方案：训练时让 draft 见到自己多步输出的近似特征。具体做法是用因果 mask 构造多轮展开（论文 Figure 6）：
    - 第 1 轮（native training step）：draft 在 ground-truth prefix 上预测一步。
    - 第 2、3 轮（simulated steps）：draft 把第 1 轮的预测当作输入、再预测下一步；用因果 mask 让位置 i 只能看到位置 ≤ i 的真实 token 与 ≤ i-1 的 draft 预测。
    - 这样训练时 draft 就在模拟推理时「自替代」的噪声环境。
  - EAGLE-3 论文版损失 L_E3 = -Σ_{i=1}^{k} log q(t_{t+i} | g_{1:t}, a_{t+1:t+i-1})（[F3]）：每一步的 token-level 负对数似然，k 是 TTT unroll 长度。
  - K3 报告版 LK loss L_LK = -log Σ_{x∈V} min(p(x), q(x))（[F4]）：直接对接受率 α 求负对数。
  - 为什么 LK loss 比常规 KL 更直接：
    - 接受率 α = Σ_x min(p, q) = 1 - TV(p, q)（[F5]），是 draft 与 target 分布相似度的直接度量。
    - KL(p‖q) 或 KL(q‖p) 是分布距离的代理，不直接等价于 α；capacity-limited draft 模型在 KL 下可能学到「平均分布接近」但「具体 token 接受率不高」的解。
    - LK loss 直接对 α 求负对数，梯度指向「最大化每个 token 的 min(p, q)」；这与接受率定义完全对齐。
  - 温度 1、无 ground-truth cross-term：K3 明确 p 与 q 都在 temperature=1 评估、不加额外的 ground-truth cross-entropy 项（[C6]）。
- 讲解材料及职责：
  - ASCII 图示展示 TTT 的因果 mask（论文 Figure 6 简化版）→ 澄清训练时如何让 draft 见到自己的输出。
  - 对照表格（标准训练 vs TTT 的输入、目标、draft 看到的信息）→ 澄清 TTT 与「多步训练」的区别。
  - 公式 L_E3 与 L_LK 并列 → 澄清两者关系：L_E3 是 token-level NLL，L_LK 是接受率直接负对数。
  - 简短推导：从 α = Σ min(p, q) 出发，说明为何 LK loss 直接优化 α。
- 前置知识安排：
  - 引用 [speculative-decoding](../../wiki/speculative-decoding/index.html) 第 3 章 α = 1 - TV(p, q) 的证明；不重新推导。
- 完成检查：
  - 说出 TTT 解决的具体问题（自替代引入的 train-inference mismatch）。
  - 说出 TTT 与「多步训练」的区别（TTT 让 draft 见到自己的输出作为输入，不只是增加训练步数）。
  - 写出 L_E3 与 L_LK 的公式。
  - 说出 LK loss 比常规 KL 更直接优化接受率的原因。
- 过渡：训练机制讲完了——最后一章把所有概念落到 K3 的工程实例上。

### S6：工程实例——K3 的 EAGLE-3 部署与边界

- 主要教学问题：K3 如何把预训练的 MTP 层 fine-tune 成 EAGLE-3 draft 模型？EAGLE-3 的边界在哪里？
- 对应范围：Q5；C7, C8, C9, N4
- 正文要点：
  - K3 的 MTP 层：预训练时让模型一次预测多个未来 token 的层；结构镜像 backbone block、单层 decoder，与 EAGLE-3 draft 架构天然匹配。
  - Fine-tune 流程：target 冻结、只更新 draft 层和 feature-fusion 投影 W_E3（[C7]）。
  - 三层 feature 来源：1st、4th、final AttnRes blocks 的输出（[C8]）。
  - W_E3 初始化为 [0 0 I]：初始 fused feature 等于 high-level feature h_h（MTP 层预训练时用的输入），训练中逐渐学到 low/mid feature（[C8]）。
  - TTT unroll 长度 7 步（[N4]）。
  - QAT 配置：draft 与 target 共享 MXFP4（MoE 专家权重）+ MXFP8（激活）+ 高精度（非专家模块），消除 train-inference mismatch（[C9]）。
  - 边界总结：
    - EAGLE-3 解决：单流自回归解码的延迟。
    - EAGLE-3 不解决：高并发批处理的吞吐（与投机解码边界一致）。
    - EAGLE-3 不改变：投机解码的接受/拒绝规则、输出分布（lossless）。
    - 不展开：EAGLE-2 的动态 draft tree（被 EAGLE-3 沿用，但其剪枝机制不属本页）；tree attention / tree mask 实现；与 Medusa、Lookahead 等其他 draft 方法的对比；vLLM/SGLang 等框架的 API。
- 讲解材料及职责：
  - 对照表格（K3 的 EAGLE-3 配置项：MTP 初始化、W_E3、三层 feature 来源、TTT 长度、QAT 配置）→ 把工程选择一览化。
  - 引用 [mxfp4-qat](../../wiki/mxfp4-qat/index.html) → 不重新讲 QAT 机制。
- 前置知识安排：
  - 引用 [mxfp4-qat](../../wiki/mxfp4-qat/index.html) 的 QAT 配置；不重新讲。
- 完成检查：
  - 说出 K3 用什么作为 EAGLE-3 draft 的初始化（预训练 MTP 层）。
  - 说出 W_E3 初始化为 [0 0 I] 的作用。
  - 说出 K3 的三层 feature 来源（1st、4th、final AttnRes blocks）。
  - 说出 K3 draft fine-tuning 的 QAT 配置。
  - 说出至少两条 EAGLE-3 不展开或边界的内容。

## 3. 讲解顺序

S1（动机）→ S2（EAGLE-1 基线机制）→ S3（EAGLE-3 两项改变）→ S4（推理 pipeline + 手算）→ S5（训练 + TTT + 损失）→ S6（K3 工程实例 + 边界）。

从最小必要前置开始：S1 只依赖 α 与 γ 公式（来自 speculative-decoding）。一次只引入一个新变量：S2 引入 feature 自回归；S3 引入直接 token 预测 + 多层融合；S4 引入自替代；S5 引入 TTT + LK loss；S6 引入 K3 工程选择。

## 4. 贯穿例子

固定 prefix = "How can I"、γ = 3、hidden size k = 4、词表 V = {I, do, it, now, &lt;other&gt;}（5 个 token，包含一个 &lt;other&gt; 兜底）。

教学简化：
- target 的 low/mid/high 三层 feature 已融合给出（不展开 FC 投影）。
- draft 单层 decoder 简化为线性变换 + tanh 激活，省略 attention 子层、MLP、layer norm。
- token embedding 用 one-hot（4 维对应前 4 个 token；&lt;other&gt; 用全零向量）。
- W_a (4×8)、W_lm (5×4) 数字选择让结果便于手算、能展示「第一步真实、后续自替代」的偏差。

数字（构造）：
- g_how = [0.5, 0.3, -0.2, 0.8]
- g_can = [0.4, 0.2, 0.1, 0.6]
- g_I（target 真实，draft 看不到）= [0.45, 0.40, 0.10, 0.20]（仅在最后对照偏差时给出）
- e_I = [0, 1, 0, 0]
- W_a (4×8) 见正文。
- 第一步：a_I = tanh(W_a · [g_can; e_I]) ≈ [0.080, 0.422, 0.139, 0.129]
- q_1 = softmax(W_lm · a_I) ≈ [0.152, 0.318, 0.171, 0.167, 0.193]，最大概率 "do"，采样 "do"。
- 第二步：用 a_I 替代 g_I；e_do = [0, 0, 1, 0]
- a_do = tanh(W_a · [a_I; e_do]) ≈ [0.098, 0.087, 0.336, 0.040]
- q_2 = softmax(W_lm · a_do) ≈ [0.180, 0.176, 0.289, 0.160, 0.195]，最大概率 "it"，采样 "it"。
- 第三步：用 a_do 替代 g_do；e_it = [0, 0, 0, 1]
- a_it = tanh(W_a · [a_do; e_it]) ≈ [0.027, 0.109, 0.024, 0.549]
- q_3 = softmax(W_lm · a_it) ≈ [0.136, 0.160, 0.135, 0.386, 0.184]，最大概率 "now"，采样 "now"。
- 对照偏差：‖g_I - a_I‖_2 ≈ 0.382，可见自替代引入明显偏差；这正是 TTT 要训练 draft 去适应的噪声。

后续章节每次复用：
- S5 复用：在 TTT 训练时，draft 见到 a_I、a_do 等自己的输出作为输入，模拟这里展示的「自替代」噪声。
- S6 复用：K3 的 7 步 TTT unroll 长度对应这里 γ = 3 的推广（γ = 7）。

## 5. 讲解材料职责

| 材料 | 服务教学问题 | 职责 |
|---|---|---|
| speculative-decoding 接受率公式 | S1 | 提供加速比与 α 关系的依据，说明 EAGLE 改变 α 来源 |
| 传统独立小模型 vs EAGLE 对照表 | S1 | 澄清两类 draft 的本质区别 |
| EAGLE-1 ASCII 图示 | S2 | 展示 feature 自回归 + time-shifted token 的输入输出结构 |
| EAGLE-1 vs EAGLE-3 对照表 | S3 | 让两项架构改变一目了然 |
| EAGLE-3 推理 pipeline ASCII 图示 | S4 | 展示三步推理的时序关系，标明真实 g 与自替代 a |
| 贯穿手算例子 | S4 | 让「自替代」的噪声变成可计算的偏差 |
| 推理 pipeline 伪代码 | S4 | 形式化 pipeline，便于对照训练损失 |
| TTT 因果 mask 图示 | S5 | 展示训练时如何让 draft 见到自己的输出 |
| 标准训练 vs TTT 对照表 | S5 | 澄清 TTT 与「多步训练」的区别 |
| L_E3 与 L_LK 公式 | S5 | 澄清论文版与 K3 版损失的关系 |
| K3 配置对照表 | S6 | 把工程选择一览化 |

## 6. 正文与折叠块分工

### 必须放正文

- EAGLE 系列的核心思路（复用 target hidden states）。
- EAGLE-1 的 feature 自回归机制与 time-shifted token 的作用。
- EAGLE-3 的两项架构改变（直接 token 预测、多层 feature 融合）。
- 推理 pipeline 三步的时序关系（第一步真实 g、后续自替代 a）。
- TTT 的核心思想与 LK loss 的形式。
- K3 的 MTP 初始化、W_E3=[0 0 I]、三层 feature 来源、QAT 配置。
- 公式的目的与符号：[F1]-[F5] 的符号定义。
- EAGLE-3 不改变投机解码框架、输出分布 lossless。

### 可放折叠块

- 贯穿例子的完整三步手算（正文只展示第一步与对照偏差，完整三步放折叠块）。
- 推理 pipeline 的伪代码。
- TTT 因果 mask 的具体位置展开（论文 Figure 6 的细节）。
- LK loss 与 KL 散度关系的简短推导（min(p, q) = (p + q - |p - q|)/2 → α = 1 - TV）。

折叠块全部收起时，正文仍能回答全部 5 个学习目标。

## 7. 范围与证据约束

大纲只使用 scope.md 中已纳入范围的内容。无新增学习目标、无新增核心论断、无范围外内容。所有数字来自 evidence.md 的 N1-N5；所有公式来自 F1-F5；所有论断来自 C1-C10。
