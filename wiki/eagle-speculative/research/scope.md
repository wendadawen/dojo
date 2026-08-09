# EAGLE-3 投机解码 draft 模型 — 内容范围

## 1.1 概念含义

- 概念名称：EAGLE-3 投机解码 draft 模型
- 英文名称：EAGLE-3 draft model for speculative decoding
- 常见缩写：EAGLE-3、EAGLE-3-style draft
- 一句话定义：一种只有单层 transformer decoder 的 draft 模型，复用 target 大模型自身的隐藏状态作为输入、直接预测 token 分布，并通过 training-time test 训练，使它在投机解码中作为 draft 角色。
- 正式定义：EAGLE 系列是 Li et al. 2024 提出的 draft 模型方法（arXiv:2401.15077，ICML 2024）。其核心是把 target 模型已算出的隐藏状态作为 draft 模型的输入，从而避免训练一个独立的小语言模型作为 draft。EAGLE-3（Li et al. 2025，arXiv:2503.01840，NeurIPS 2025）在 EAGLE-1 基础上做了两项架构改变：（a）放弃预测 feature、改为直接预测 token；（b）用 target 的低、中、高三层特征融合替代原来的 second-to-top-layer 单层特征；并引入 training-time test（TTT）训练方法，让 draft 在训练时就见到自己多步输出的近似特征，从而在推理多步预测中保持接受率不衰减。本文采用此定义。
- 本文语境：把 EAGLE-3 作为一种「draft 模型架构」讲，它仍然依附于通用投机解码框架（draft-then-verify + 拒绝采样规则），不重新解释整个投机解码。投机解码框架见 [speculative-decoding](../../wiki/speculative-decoding/index.html)。

## 1.2 包括什么

- EAGLE-1 基线机制（feature 层自回归 + time-shifted token）——属本概念因为它定义了「复用 target 隐藏状态」这一 EAGLE 系列核心，是 EAGLE-3 改变的比较基准。
- EAGLE-3 的两项架构改变（直接 token 预测、多层特征融合）——属本概念因为这两项改变定义了 EAGLE-3 与 EAGLE-1 的区别。
- training-time test（TTT）训练方法——属本概念因为它让 draft 在多步预测中保持接受率，是 EAGLE-3 与 EAGLE-1 训练侧的关键差异。
- 推理时 draft 模型的自回归 pipeline——属本概念因为这是 draft 模型生成 γ 个候选 token 的实际机制。
- K3 用 EAGLE-3 的工程实例（MTP 层初始化、WE3=[0 0 I]、LK loss、QAT 配置）——属本概念作为具体工程落地。

## 1.3 不包括什么

- 投机解码的通用接受/拒绝规则与残差分布证明——已在前置概念页 [speculative-decoding](../../wiki/speculative-decoding/index.html) 讲清，本文只引用结果。
- EAGLE-2 的动态 draft tree——属于 EAGLE 系列另一分支改进，本文只在 S6 边界处提及它存在，不展开其剪枝机制。
- Medusa、Lookahead、SpecDec 等其他 draft 方法——是与 EAGLE 并列的方案，本文不展开对比。
- tree attention / tree mask 的实现细节——是验证阶段对多候选并行打分的工程实现，与 draft 模型本身正交。
- vLLM、SGLang、TensorRT-LLM 等框架的 EAGLE-3 集成 API——属部署侧工程，不属概念核心。
- draft 模型训练数据集的选择与配比——属训练数据工程，本文只引用 K3 报告 §4.1.4 的训练配置作为实例。

## 1.4 相邻概念

- 投机解码（speculative decoding）：EAGLE-3 是其 draft 角色的一种实现。区别：投机解码是「draft-then-verify + 拒绝采样」的框架；EAGLE-3 是这个框架里的 draft 角色。纳入本页：仅引用其接受率 α 与 γ 的关系，不重新讲框架。
- Medusa：用一组 MLP 头预测多个未来 token，不依赖 target feature 自回归。区别：Medusa 是「多头并行」而非「自回归」draft。不纳入本页。
- MTP（multi-token-prediction）：预训练时让模型一次预测多个未来 token 的训练目标。区别：MTP 是预训练目标，EAGLE-3 是推理时 draft。K3 把预训练的 MTP 层 fine-tune 成 EAGLE-3 draft——这个 fine-tune 关系纳入 S6 工程实例，MTP 本身不展开。
- QAT（量化感知训练）：与 EAGLE-3 正交的训练侧技术，K3 让 draft 与 target 共享同一 QAT 配置。引用 [mxfp4-qat](../../wiki/mxfp4-qat/index.html)，不展开。

## 2. 学习目标

### Q1：EAGLE 系列如何复用 target 的隐藏状态替代独立小模型 draft？为什么这能同时降低 draft 成本和提高接受率？

- 完成答案：说明传统独立小模型 draft 的两难（太弱接受率低、太强成本高）；说明 EAGLE 用 target 已算出的 feature 作为 draft 输入的本质；说明这种复用为何既降低 draft 自身参数量又让 draft 决策更贴近 target。
- 为什么是核心目标：不理解这一点就看不出 EAGLE 系列与 Medusa/Lookahead/SpecDec 的本质区别，也无法理解 EAGLE-3 的进一步改变为何有效。
- 依赖内容：speculative-decoding 的接受率 α 与 γ、加速比公式；EAGLE-1 的 feature 自回归机制。

### Q2：EAGLE-3 在 EAGLE-1 基础上做了哪两项架构改变？为什么这两项改变能进一步提升接受率？

- 完成答案：说出两项改变——（a）放弃 feature 预测、改为直接 token 预测；（b）用 low/mid/high 三层 feature 融合替代 top-layer feature。说明每项改变解决的具体问题（feature 预测的容量瓶颈、top-layer feature 的信息局限）。
- 为什么是核心目标：EAGLE-3 与 EAGLE-1 的区别全部浓缩在这两项改变里，不掌握就无法区分 EAGLE 系列版本。
- 依赖内容：EAGLE-1 的 feature 预测机制；EAGLE-3 的架构图（论文 Figure 5）。

### Q3：推理时单层 decoder 如何自回归生成 γ 个 draft token？为什么第一步用 target 真实 feature、后续步骤要用自己的输出替代 target feature？

- 完成答案：说明推理 pipeline 的三步——第一步输入 target 真实融合 feature + 上一个采样 token 的 embedding、过单层 decoder 与 lm_head 得到 q_1、采样；后续步骤因新位置的 target feature 尚未算出，只能用 draft 上一步的输出 a 替代 g；说明这种「自替代」会引入噪声、必须靠训练阶段 TTT 来对齐。
- 为什么是核心目标：这是 EAGLE-3 推理时与训练时 TTT 联动的核心机制，不理解就无法解释为什么需要 TTT。
- 依赖内容：EAGLE-3 的单层 decoder 架构；target 的 hidden states 在 prefill 后已固定的时序关系。

### Q4：training-time test 如何让 draft 模型在多步预测中保持接受率？为什么 K3 直接用接受率的负对数作为损失比常规 KL 散度更直接？

- 完成答案：说明 TTT 在训练时让 draft 模型见到自己多步输出（用因果 mask 构造模拟推理的噪声环境）；说明 EAGLE-3 论文版的 token-level NLL 损失与 K3 报告版的 LK loss = -log Σ_x min(p(x), q(x)) 各自形式；说明为什么 KL 散度只是分布距离的代理、不直接最大化接受率，而 LK loss 直接对接受率求负对数。
- 为什么是核心目标：TTT 与 LK loss 是 EAGLE-3 训练侧的两项关键创新，决定 draft 在多步深度预测中是否仍可用。
- 依赖内容：speculative-decoding 的接受率 α = Σ_x min(p(x), q(x))；标准负对数似然与 KL 散度的关系。

### Q5：K3 如何把预训练的 MTP 层 fine-tune 成 EAGLE-3 draft 模型？

- 完成答案：说明 K3 的 MTP 层结构（镜像 backbone block、单层 decoder）天然匹配 EAGLE-3 draft 架构；说明 fine-tune 时 target 冻结、只更新 draft 层和 feature-fusion 投影；说明 WE3 初始化为 [0 0 I] 的作用（初始 fused feature 等于 high-level feature、与 MTP 层预训练输入对齐）；说明 7 步 TTT、low/mid/high feature 取自 1st、4th、final AttnRes blocks；说明 draft 与 target 共享 QAT 配置（MXFP4 权重 + MXFP8 激活，非专家保持高精度）。
- 为什么是核心目标：这是 EAGLE-3 在生产级 MoE 模型上的真实部署实例，把抽象架构与训练机制落到具体工程选择上。
- 依赖内容：EAGLE-3 架构与 TTT；K3 报告 §4.1.4；mxfp4-qat 的 QAT 配置。

## 3. 内容分级

### 3.1 核心内容

- EAGLE-1 的 feature 自回归机制 + time-shifted token → 服务 Q1。
- 传统独立小模型 draft 的两难 → 服务 Q1。
- EAGLE-3 的两项架构改变 → 服务 Q2。
- 推理时单层 decoder 自回归 pipeline（第一步真实 g、后续用 a 替代）→ 服务 Q3。
- TTT 训练机制 + LK loss → 服务 Q4。
- K3 工程实例（MTP 初始化、WE3=[0 0 I]、三层 feature、QAT）→ 服务 Q5。
- 必须讲清的结论：EAGLE-3 不改变投机解码的接受/拒绝规则；EAGLE-3 不取代 target；draft 自身成本远小于 target。

### 3.2 辅助内容

- EAGLE-1 在 LLaMA2-Chat 70B 上 2.7x-3.5x 加速、EAGLE-3 在 13B 上 5.6x 等性能数字 → 消除「EAGLE-3 究竟快多少」的疑问，不参与核心机制论证。
- 单层 decoder 与多层 decoder 的参数量对比 → 澄清「draft 自身成本远小于 target」这一前提。

### 3.3 扩展内容

- EAGLE-2 的动态 draft tree → 排除本页范围（属另一分支改进），S6 边界处提及。
- tree attention / tree mask 实现 → 排除本页范围（属验证侧工程）。
- 与 Medusa、Lookahead 的对比 → 排除本页范围。
- EAGLE-3 在 SGLang、vLLM 等框架的 API → 排除本页范围（属部署侧工程）。

## 4. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页链接 / 生成状态 | 递归深度 |
|---|---|---|---|
| 投机解码（speculative-decoding） | Q1, Q4（接受率 α、γ、draft-then-verify 框架） | [../../wiki/speculative-decoding/index.html](../../wiki/speculative-decoding/index.html) 已生成 | 0 |
| MXFP4 量化感知训练（mxfp4-qat） | Q5（K3 的 QAT 配置） | [../../wiki/mxfp4-qat/index.html](../../wiki/mxfp4-qat/index.html) 已生成 | 0 |

无递归生成的概念页。

## 5. 明确不展开的内容

| 内容 | 与概念的关系 | 不展开原因 |
|---|---|---|
| EAGLE-2 的动态 draft tree | EAGLE 系列另一分支改进 | 不影响 Q1-Q5 的回答；EAGLE-3 也采用 EAGLE-2 的动态 tree，但其剪枝机制不是 EAGLE-3 的核心创新，仅在 S6 边界提及 |
| tree attention / tree mask 实现 | 验证阶段对多候选并行打分的工程实现 | 与 draft 模型本身正交；不影响理解 EAGLE-3 的架构与训练 |
| Medusa、Lookahead、SpecDec 等其他 draft 方法 | 与 EAGLE 并列的 draft 方案 | 不影响理解 EAGLE-3 自身；对比属于综述性内容 |
| vLLM / SGLang / TensorRT-LLM 的 EAGLE-3 API | 部署侧工程 | 不影响理解概念机制；属框架文档 |
| draft 训练数据集的选择与配比 | 训练数据工程 | K3 报告 §4.1.4 未给出具体数据集；EAGLE-3 论文用 ShareGPT 等，但数据选择不影响概念机制 |
| EAGLE-3 的 scaling law（数据规模与加速比关系） | EAGLE-3 论文发现的现象 | 不影响理解架构与训练机制；属经验观察 |

## 6. 常见误解和适用边界

### 6.1 常见误解

| 错误理解 | 正确结论 | 形成原因 | 影响学习目标 |
|---|---|---|---|
| 「EAGLE-3 是一种新的投机解码框架，有自己的接受/拒绝规则」 | EAGLE-3 只替换 draft 角色；接受/拒绝规则仍是原投机解码的拒绝采样，输出分布与纯 target 采样完全相同 | 把 draft 模型与投机解码框架混在一起 | Q1, Q3 |
| 「EAGLE-3 的 draft 是一个独立的小语言模型」 | EAGLE-3 draft 是单层 decoder，复用 target 的 hidden states 作为输入，不是一个独立小 LLM | 把 EAGLE 系列与 SpecDec（独立小模型）混淆 | Q1, Q2 |
| 「直接 token 预测和 EAGLE-1 没区别，反正都是预测下一个 token」 | EAGLE-1 预测的是 second-to-top-layer feature、再过 target lm_head 得到 token；EAGLE-3 直接预测 token 分布、不再回归 feature。差异在于 draft 容量是否被 feature 回归任务占用 | 没有区分 feature 空间与 token 空间 | Q2 |
| 「TTT 只是多训练几步」 | TTT 在训练时让 draft 见到自己多步输出的近似特征、用因果 mask 模拟推理时「自替代」带来的噪声；不只是增加训练步数 | 把 TTT 与「多步训练」混淆 | Q4 |
| 「K3 用 LK loss 是因为 KL 散度算不出来」 | KL 散度完全可以算；LK loss 的选择是因为 KL 只是分布距离的代理、不直接最大化接受率 α = Σ_x min(p, q) | 把工程选择误解为计算便利 | Q4 |
| 「draft 模型用 target 的 hidden states 就等于 target 模型」 | draft 是单层 decoder，hidden states 只是输入；draft 自己仍然有参数、输出分布 q 与 target 分布 p 不同；接受率取决于 ‖p - q‖ | 把「复用 feature」与「参数共享」混淆 | Q1, Q3 |

### 6.2 适用边界

- EAGLE-3 解决的问题：单流自回归解码的延迟（latency per token）。
- EAGLE-3 不解决的问题：高并发批处理的吞吐；与投机解码本身的边界一致（见 [speculative-decoding](../../wiki/speculative-decoding/index.html) 第 4、6 章）。
- EAGLE-3 需要的条件：
  - target 与 draft 共享 tokenizer；
  - target 的 hidden states 可在推理时被 draft 读取（架构上要求 target 暴露 low/mid/high 三层 feature）；
  - draft 单步成本远小于 target 单步成本（c ≪ 1，因 draft 只有单层 decoder）；
  - 训练时能用 TTT 模拟推理噪声环境。
- 条件不满足时：
  - 若 target 不暴露多层 feature → 退回到 EAGLE-1 的 second-to-top-layer 单层 feature；
  - 若训练时无法做 TTT（如算力不足）→ 接受率随 draft 深度衰减更快，深度 draft 不可用；
  - 若 draft 单步成本与 target 接近（理论上不应发生，因 draft 仅单层）→ 加速比退化。
