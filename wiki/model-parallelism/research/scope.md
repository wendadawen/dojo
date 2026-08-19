# 模型并行（model-parallelism）内容范围

## 1. 概念歧义处理

- "模型并行"（model parallelism）在不同语境下含义不同：
  - 广义：一切把单个模型切分到多个加速器的技术（含 TP、PP、EP 及其组合）。
  - 狭义（Megatron-LM 论文语境）：与流水线并行（pipeline-based model parallelism）相对的层内并行（intra-layer model parallelism），即张量并行。
- 本页采用广义含义，并在正文首次出现时说明狭义用法的历史来源（Megatron-LM §1 将两者并列），消除歧义。状态：已裁定。
- 缩写：TP = Tensor Parallelism，PP = Pipeline Parallelism，EP = Expert Parallelism，DP = Data Parallelism。
- "流水线气泡"（pipeline bubble）术语统一，不与"空泡""气泡率"混用。

## 2. 概念含义

### 2.1 简要定义

模型并行是把单个神经网络按权重切开、分布到多块 GPU 上协同完成一次前向/反向计算的技术。两条基本切法：按层内矩阵的维度切（张量并行 TP），按层堆叠的深度切（流水线并行 PP）。

### 2.2 正式定义与来源

- TP：Megatron-LM（Shoeybi et al., 2019, arXiv:1909.08053）提出对 Transformer 层内权重矩阵按行/列切分，self-attention 与 FFN 各需 2 次 all-reduce 完成一层前向（§2.2-2.3，Figure 2-3）。来源：Megatron-LM 论文。
- PP：GPipe（Huang et al., 2018, arXiv:1811.06965）将模型按层序切成多个 cell（stage）放到不同加速器，micro-batch 流水执行（§2，Figure 2-3）。来源：GPipe 论文。
- 流水线气泡：GPipe 训练时 step 时间 = (m + p - 1) × 单 micro-batch 前向+反向时间，气泡占比 = (p - 1)/(m + p - 1)（GPipe §3.1 推导）。来源：GPipe 论文。
- EP：按专家维度切 MoE 层，通信为 all-to-all。来源：本仓库 moe-serving 页已覆盖，本页只给链接不重复展开。
- DP 与本页关系：DP 是按数据切分（每卡一份完整模型），不属于模型并行；正文在"容易混淆"处说明区别，不展开。

### 2.3 本文采用的语境

面向 LLM 推理 serving 的模型并行。训练特有问题（梯度累积、重计算）只在 PP 气泡推导处按需提及。

### 2.4 包括什么

- TP 的切分机制：attention 的 head 切分、FFN 的列/行切分、每层 2 次 all-reduce
- PP 的切分机制：stage 划分、micro-batch、气泡的来源与公式
- TP+PP 组合维度记号（TP × PP = 总卡数）
- EP 的定位与链接（不展开机制）
- 通信量与互联拓扑的关系（为什么 TP 要留在 NVLink 域内、PP 可跨机）——引用 gpu-communication 页

### 2.5 不包括什么

- EP 的机制细节（router、all-to-all 路径）：属 moe-serving 页职责，本页只链接。排除理由：避免跨页重复。
- 张量并行的数学证明（正交性、结果等价性的完整证明）：不影响学习目标，只给验证性说明。
- 训练专用技术（ZeRO、激活重计算）：与推理 serving 的学习目标无关。
- 专家并行的负载均衡：属于 deepseek-moe/moe-serving 职责。

### 2.6 相邻概念

- 数据并行 DP：每卡完整模型+不同数据。区别：模型并行每卡只有部分权重。不纳入本页范围，正文一句话区分。
- Chunked Pipeline Parallelism（CPP）：PP 与输入分块的组合，气泡消除的另一个路径。本页在 PP 气泡一节提及并链接 chunked-prefill 页（该页负责完整机制）。
- 上下文并行/序列并行：与 TP 属同族但本页不展开，标注为扩展内容排除。

## 3. 学习目标

### Q1：一个模型放不进单卡（或单卡算力/带宽不够）时，模型并行提供哪两条基本切分路径，各自切的是什么？

- 完成答案：读者应能说明 TP 按层内矩阵维度切（每卡持有每层的一部分，所有卡参与每一层的计算）、PP 按层堆叠深度切（每卡持有连续的若干完整层，数据像流水线一样流过），以及两者可以组合。
- 为什么是核心目标：这是本页全部后续内容（通信、气泡、组合）的骨架。
- 依赖内容：Transformer 层结构（attention + FFN 堆叠）、GPU 显存概念。

### Q2：张量并行怎么切一个 Transformer 层？一层前向需要哪些通信，为什么？

- 完成答案：读者应能说明 attention 按 head 分组切（每卡一组 head，各自算各自的 attention 再拼回）、FFN 按第一线性层列切/第二线性层行切（两个切法配合使本地可直接求和）、一层内发生 2 次 all-reduce（attention 输出处、FFN 输出处）。
- 为什么是核心目标：理解 TP 的通信特征是理解"TP 留在 NVLink 域内"的基础。
- 依赖内容：矩阵分块乘法（可手算）、attention 的多头结构（standard-attention 页链接）、all-reduce 概念（gpu-communication 页链接）。

### Q3：流水线并行为什么有气泡？气泡大小由什么决定？

- 完成答案：读者应能说明：PP 各 stage 串行依赖，首个 micro-batch 进入时后面的 stage 在等待（填充期），最后 micro-batch 离开时前面的 stage 在等待（排空期）；step 时间为 (m + p - 1) 个 slot，气泡占比 (p - 1)/(m + p - 1)，增加 micro-batch 数 m 可摊薄气泡；给出小例子手算。
- 为什么是核心目标：气泡是 PP 的核心代价，也是 Beyond the Buzz 论文中 CPP、以及 chunked-prefill 页中 PP 气泡消除的铺垫。
- 依赖内容：Q1 的 PP 定义。

### Q4：TP 和 PP 各自适合部署在什么互联拓扑上，为什么推理系统常常组合使用？

- 完成答案：读者应能说明：TP 每层多次 all-reduce、通信频繁且量大，需要高带宽低延迟互联（NVLink/NVSwitch 域内）；PP 通信只发生在 stage 边界（相邻卡传一次激活值）、频率低量小，可容忍跨机低速互联；因此典型部署是 NVLink 域内 TP、跨机 PP（TP × PP = 总卡数），DeepSeek-R1 64 卡 EP × PP = 64 即此类组合。
- 为什么是核心目标：这是 Beyond the Buzz 论文"模型切分策略"维度与 NVLink 域敏感性分析的直接前置。
- 依赖内容：Q2 Q3 的通信特征、gpu-communication 页的互联分层。

## 4. 内容分级

核心内容（直接服务学习目标）：
- TP 的 head 切分与 FFN 行/列切分（Q2）
- 每层 2 次 all-reduce 的位置与原因（Q2）
- PP 的 stage/micro-batch/气泡机制（Q1 Q3）
- 气泡公式 (p-1)/(m+p-1) 及推导（Q3）
- TP/PP 通信量对比与拓扑选择（Q4）
- TP × PP 组合记号（Q4）

辅助内容：
- DP 与模型并行的区别（消除混淆）
- EP 定位与链接（衔接 MoE 模型语境）
- 气泡公式推导的完整展开（折叠块）

扩展内容（排除）：
- 序列并行/上下并行的机制
- ZeRO 与训练显存优化

## 5. 前置知识映射

- Transformer 层结构（attention 多头、FFN 两个线性层）：standard-attention 页已有（覆盖多头结构），正文引用。
- all-reduce 等 collective 通信与通信量：gpu-communication 页已有（覆盖 ring all-reduce 与通信量公式），正文引用。
- MoE 与专家并行：moe-serving 页已有，本页 EP 处引用。
- prefill/decode 与 KV cache：moe-serving 页已有；本页 CPP 提及处引用。
- chunked prefill 机制：本任务同时递归生成 chunked-prefill 页，双向链接。
- 矩阵分块乘法：读者应有的数学常识，正文用可手算小例子自足说明，不单独成页。

## 6. 明确不展开的内容

- EP 的 router 与 all-to-all 细节：moe-serving 页职责，本页链接。
- chunked prefill / piggybacking 完整机制：chunked-prefill 页职责，本页在 CPP 提及处链接。
- 训练显存优化（ZeRO/重计算）：与推理 serving 学习目标无关。
- 3D 并行的数据并行维度：DP 一句话区分即可。

## 7. 常见误解和适用边界

误解 1
- 错误理解：模型并行就是张量并行。
- 正确结论：模型并行是总称，TP 与 PP 是两条正交路径（Megatron-LM 原文将 intra-layer 与 pipeline-based 并列）。
- 形成原因：狭义用法在 Megatron-LM 语境中出现过。
- 影响目标：Q1。

误解 2
- 错误理解：PP 把模型切到多卡后，卡数越多吞吐越高。
- 正确结论：气泡占比 (p-1)/(m+p-1) 随 stage 数 p 增大而上升，p 越大需要越多的 micro-batch 才能摊薄；推理时 micro-batch 受 batch size 限制。
- 形成原因：忽略串行依赖的空转。
- 影响目标：Q3。

误解 3
- 错误理解：TP 卡数越多越好，可以无限细分。
- 正确结论：TP 每层需要 2 次 all-reduce，切得越细通信次数不变但每卡计算量变小，通信占比升高；TP 超过 NVLink 域后 all-reduce 走跨机网络，代价急剧上升。TP 通常限制在单机 8 卡（NVLink 域）内。
- 形成原因：只看显存/算力切分，忽略通信开销。
- 影响目标：Q2 Q4。

适用边界
- 本页讲的是机制与代价结构，不给出"最优 TP/PP 配置"；具体最优配置依赖模型结构、序列长度、延迟约束（这正是 Beyond the Buzz 论文的搜索空间问题，本页链接论文页）。
- 气泡公式的 (p-1)/(m+p-1) 基于 GPipe 的同步训练 step 推导；推理场景中各 micro-batch 迭代结构不同（Sarathi 指出 prefill/decode 混合导致不平衡），公式用于说明数量级与趋势。

## 8. 论断分级标注

- "TP 对 Transformer 层做行/列切分、每层 2 次 all-reduce"：论文明确声称（Megatron-LM §2.2-2.3）。
- "PP 气泡占比 (p-1)/(m+p-1)"：论文明确声称（GPipe §3.1）。
- "TP 适合 NVLink 域内、PP 可跨机"：文献已有结论（Megatron-LM §4.4 讨论通信代价；Pope et al. 2022 讨论 TP 通信需求；各推理系统部署实践）。正文引用时标注来源。
- "TP×PP 组合是主流推理部署形态"：基于证据的推断（多个系统文档与论文的组合记号，DeepSeek 推理报告的 EP×PP 记号）；标注推断并给依据。
- 手算数值例子中自设的维度数字：构造示例，明确标注。
