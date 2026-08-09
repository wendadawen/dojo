# MoE 大模型推理与服务基础 scope：内容范围

## 0. 版本与定位

- 概念名称：MoE 大模型推理与服务基础；英文：Fundamentals of MoE LLM Inference and Serving；无通用缩写
- 本文语境：在线服务（serving）场景下的 MoE LLM 推理机制，面向论文 ExpertPlex（arXiv:2607.18002）解析页的前置知识（该论文 scope 已将本页登记为 depth-1 前置概念页）
- 歧义检查：MoE（Mixture-of-Experts）在 LLM 领域含义唯一主流（稀疏激活的专家混合层）；serving 一词在 ML 系统领域含义明确。无同名冲突，状态：已裁定。EP 在本页只指 expert parallelism（专家并行），不涉及 epoch 等其他含义；正文首次出现时写明全称

## 1. 概念含义

- 一句话定义：把 Transformer 层的 FFN 换成"一排只被选中几个的专家"之后，模型怎么算（路由与稀疏激活）、权重怎么摆（专家并行）、一次请求怎么流（prefill/decode 与 KV cache）、服务好坏怎么量（TTFT/TPOT/goodput）、两阶段怎么摆（PD 合设与分离）
- 正式定义：与来源一致的分件定义——MoE 层（router 为每个 token 选择 top-k 个专家并加权求和，Shazeer 2017 §2 / DeepSeek-V3 §2.1.2）；EP 与 dispatch/combine（ExpertPlex §2.3）；prefill/decode 与 TTFT/TPOT/goodput（DistServe §2.1）
- 本页不是一篇单一概念，而是一组"读懂 MoE 服务系统论文的最小概念链"；链上每个环节单独成立

### 包括什么（每项为何属于本概念）

- Transformer 层 = attention + FFN 的最小结构：MoE 替换的是 FFN，不先讲清被替换对象就无法讲替换动机
- MoE 层结构：router、top-k 稀疏激活、routed/shared expert：概念核心
- 稀疏激活的动机（容量与计算量解耦）：MoE 存在的理由
- 专家并行 EP 与 all-to-all dispatch/combine：MoE 权重超出单卡显存后的必然工程后果，ExpertPlex 的直接前置
- TBO/SBO 通信-计算重叠：EP 通信开销的通行缓解手段，ExpertPlex §2.3 的对照概念
- prefill/decode 两阶段与 KV cache：一切服务指标与 PD 部署决策的对象
- TTFT/TPOT/SLO/goodput：MoE 服务论文的度量语言
- PD 合设（colocation）与 PD 分离（PDD）的基本动机与代价：ExpertPlex 问题域的入口

### 不包括什么（排除理由）

- GPU 执行模型细节（SM/warp/kernel/CUDA Graph/tile）：由另一概念页《GPU 执行模型》负责（ExpertPlex 规划登记为 `wiki/gpu-execution-model/`，与本页同级），本页只用"GPU 是并行做矩阵运算的芯片、显存有限"这类最小表述
- MoE 训练侧机制（辅助损失/bias 均衡的公式与训练动态、反向传播）：本页只服务推理理解；训练侧负载均衡一句话带过并标注
- attention 内部机制（Q/K/V 投影、softmax 打分、MLA/GQA 等变体）：本页只需"attention 跨 token 混合信息并产出 K/V"这一最小功能描述；KV cache 只用到"每层缓存 K/V 张量"
- 具体推理引擎实现（vLLM/SGLang 的调度与 PagedAttention）：属引擎实现层，不影响本页学习目标
- ExpertPlex 自身的架构与局限分析（APK、一侧通信、head-of-line blocking 定量数字）：属论文解析页内容，本页只给到 PDD/colocation 的基本权衡
- 量化（FP8 等）、投机解码：不影响学习目标

### 相邻概念

- 张量并行 TP / 数据并行 DP：容易与 EP 混淆。区别：EP 按"专家"切模型，TP 按矩阵切每层，DP 复制整模型。本页只在需要区分时一句带过，不展开（展开属并行策略概念页/note `wiki/vllm-parallelism/index.html`）
- 稠密（dense）模型：作为对照对象纳入讲解
- 模型蒸馏/剪枝：同样减少计算但机制完全不同，不纳入

## 2. 学习目标

### Q1：为什么把 Transformer 层的 FFN 换成 MoE，能让模型存下多得多的参数而每个 token 的计算量几乎不涨？

- 完成答案：FFN 是标准 Transformer 层的参数大户（Vaswani 配置下约为 attention 的 2 倍）；MoE 把一个 FFN 换成 N 个专家子网络 + 一个 router，router 对每个 token 只选 top-k 个专家计算并加权求和（可手算 8 专家 top-2 的例子），其余专家不计算；因此总参数量随 N 增长而每 token 计算量只随 k 增长；DeepSeek-V3 总参数 671B、每 token 仅激活约 37B（≈5.5%）；shared expert 处理所有 token、routed expert 按路由激活
- 为什么是核心目标：MoE 存在的全部理由；不理解它就无法理解后文一切工程问题的来源
- 依赖内容：S1 的 Transformer 层结构与 FFN 参数占比；F1 公式；N1–N3

### Q2：MoE 模型的专家为什么要切到多张 GPU 上，all-to-all 的 dispatch/combine 在搬运什么，TBO/SBO 怎样掩盖这部分通信开销？

- 完成答案：稀疏激活只省计算不省显存，全部专家权重必须驻留，超出单卡显存后按专家分片（EP）；路由按 token 决定，token 激活要被搬到选中专家所在卡（dispatch）、算完搬回（combine），两者通常用 all-to-all 实现；通信与计算串行会产生等待气泡，TBO 用一个微批的计算掩盖另一个微批的通信，SBO 在同一微批内用共享专家计算掩盖路由专家通信；能复算 2 卡 4 专家 top-2 的 dispatch 计数并指出负载不均
- 为什么是核心目标：EP + all-to-all 是 MoE 服务系统区别于稠密模型服务的核心结构，ExpertPlex 的通信设计全部建立其上
- 依赖内容：Q1 的 top-k 路由；E2 手算例子；N4

### Q3：一次对话请求在 prefill 和 decode 两个阶段分别做什么，KV cache 消除了哪部分重复计算？

- 完成答案：生成是自回归逐 token 的；prefill 并行处理全部输入 token、建立各层 KV cache、产出首 token（吞吐导向）；decode 每步只处理一个新 token、复用前缀 KV cache、逐 token 生成（延迟敏感）；无 KV cache 时每生成一步都要为全部前缀重算 K/V，有 cache 时每步只算 1 个新 token 的 K/V（可手算 4+3 例子的对比）
- 为什么是核心目标：TTFT/TPOT 与 PDD 的对象就是这两个阶段；不理解两阶段差异就无法理解部署决策
- 依赖内容：S1 的 attention 最小功能描述；C9/C10

### Q4：TTFT、TPOT、SLO、goodput 各自衡量什么，为什么"服务快"要拆成两个指标？

- 完成答案：TTFT 是 prefill 阶段时长（首 token 延迟），TPOT 是除首 token 外平均每 token 生成时间；不同应用对两者要求不同（实时聊天重 TTFT，长文生成重 TPOT，人阅读速度约 250 词/分钟使 TPOT 低于它即可）；SLO 是对 TTFT/TPOT 分别设定的目标与达成率要求；goodput 是满足 SLO 达成率（如 90%）前提下系统能承接的最大请求率，区别于不顾延迟的裸吞吐；可从给定时间线手算 TTFT/TPOT/总延迟
- 为什么是核心目标：MoE 服务论文（含 ExpertPlex）用 goodput 作主指标；不会读这组指标就读不懂任何实验结论
- 依赖内容：Q3 的两阶段；C11/C12

### Q5：PD 合设与 PD 分离各自的动机和基本代价是什么？

- 完成答案：合设（colocation）让两阶段共享同一实例和一份权重，省显存、无跨实例 KV 传输，但长 prefill 会干扰短 decode；分离（PDD）把两阶段放到不同 GPU 实例，消除干扰、可按不同目标各自配置资源，代价是每个实例都要存完整模型副本（MoE 权重大使副本很贵）、要按 P:D 配比以部署单元为单位扩容（DeepSeek-V3 报告单元为 32 prefill GPU + 320 decode GPU）、并引入跨实例 KV 传输
- 为什么是核心目标：ExpertPlex 的全部动机是这两条路线的局限；读者必须先知道两条路线是什么
- 依赖内容：Q3 两阶段、Q4 指标、Q2 的权重显存结论；C13/C14、N5

## 3. 内容分级

### 核心内容（缺一则学习目标答不全）

- Transformer 层结构（attention + FFN）与 FFN 参数占比 → Q1
- MoE 层机制：router、top-k、加权和、routed/shared expert、F1 公式、E1 手算 → Q1
- 稀疏激活动机与 V3 数字（671B/37B、256+1 top-8）→ Q1
- EP 动机（权重须全量驻留）、dispatch/combine 与 all-to-all、E2 手算 → Q2
- TBO/SBO 定义与掩盖逻辑、时间线图 → Q2
- prefill/decode 定义、自回归、KV cache 机制、E3 手算 → Q3
- TTFT/TPOT/SLO/goodput 定义、E4 手算 → Q4
- colocation/PDD 动机与基本代价、N5 部署单元数字 → Q5

### 辅助内容（消除理解障碍）

- token/参数/前向计算的零基础解释（服务 Q1–Q3 的阅读连续性）
- FFN 8d² vs attention 4d² 手算（回答"为什么换 FFN 不换 attention"，服务 Q1）
- 负载不均的路由层成因一句话 + 服务端表现（慢卡拖慢整步，服务 Q2 的 E2 观察）
- "激活 5.5% 参数 ≠ 只需 5.5% 显存"误解破除（服务 Q2 前提）
- all-to-all 的一句话定义（每个节点都可能与所有节点互换数据，服务 Q2）
- goodput ≠ throughput 对照（服务 Q4 误解）

### 扩展内容（标记纳入/排除）

- DeepSeek-V3 部署的冗余专家、微批细节：排除（属 ExpertPlex 页背景，不影响本页目标）
- MoE 训练负载均衡（辅助损失、bias 更新）：排除出正文，一句话标注
- MLA 等 attention 变体：排除
- vLLM/SGLang 实现：排除；note `wiki/vllm-parallelism/index.html` 作扩展阅读链接
- Shazeer 2017 / GShard / Switch 的历史脉络：纳入一句话版本（MoE 思想来源与 top-k 取值差异），不展开

## 4. 前置知识映射

读者为完全小白，无任何 GPU 或服务系统背景。逐项检查 `wiki/`：现有 3 篇均为 note（非 concept 流程产物），无可用概念页。

| 前置知识 | 被哪些学习目标依赖 | 处理方式 | 递归深度 |
|---|---|---|---|
| token、参数（权重）、向量、前向计算 | Q1–Q3 | 本页正文内联最小解释（属"从最小必要前置开始"的正文内容，不构成独立概念页） | — |
| attention 的最小功能（跨 token 交换信息、产出 K/V） | Q3 | 本页内联最小功能描述，不展开内部机制 | — |
| GPU 与显存的最小概念（并行矩阵运算芯片、显存有限） | Q2 | 本页内联一句话 + 占位提示：完整讲解由概念页《GPU 执行模型》（规划中，`wiki/gpu-execution-model/`）承担 | 登记不生成（由编排方另行生成，见 ExpertPlex scope §4） |
| 矩阵乘法 | Q1（FFN 参数计数） | 内联一句"把向量变成另一个向量的权重表"，参数计数只需乘法 | — |

depth-2 判定：本页所有前置均内联解决或登记占位，无缺口，无需递归生成。

## 5. 明确不展开的内容

- MoE 训练与负载均衡算法：与概念关系（路由倾斜的成因），不展开原因：本页服务推理理解，训练侧机制不影响任何学习目标；正文一句话标注"这是训练时要处理的问题"
- attention 内部 Q/K/V 计算：与概念关系（KV cache 的 K/V 来自它），不展开原因：KV cache 只需要"每层产出并缓存 K/V 张量"这一功能事实；展开属另一概念（注意力机制）
- GPU kernel、SM、CUDA Graph：与概念关系（TBO/SBO 在 GPU 上调度），不展开原因：由《GPU 执行模型》概念页负责；本页时间线图不依赖 kernel 概念
- chunked prefill、投机解码、量化：不影响学习目标，属独立技术点
- TP/DP/PP 并行策略体系：只在区分 EP 时一句带过；完整体系属并行策略概念，不影响学习目标（本页 EP 可独立讲清）
- ExpertPlex 对 PDD/colocation 的定量局限（部署单元数百 GPU 的弹性/故障分析、head-of-line blocking 数量级）：属论文解析页，本页只给基本权衡，避免与论文页重复

## 6. 常见误解与适用边界

- 误解 1：「MoE 就是多个独立模型投票」。错误理解：专家是各自独立训练/部署的模型。正确结论：专家是同一个网络内部的 FFN 子网络，与 router 一起联合训练、作为一个模型部署。形成原因：「专家」一词的日常含义。影响 Q1
- 误解 2：「每个 token 只激活 5.5% 参数，所以模型只要 37B 的显存」。错误理解：稀疏激活省显存。正确结论：稀疏激活省的是每 token 计算量；全部 671B 权重都必须驻留显存（或分层存储），这正是 EP 存在的原因。影响 Q2
- 误解 3：「TBO/SBO 把通信开销消除了」。错误理解：重叠后通信不再耗时。正确结论：重叠是把通信时间藏在计算之下，通信本身仍占用带宽与硬件资源；能否藏住取决于有没有足够的可并行计算。影响 Q2
- 误解 4：「goodput 就是吞吐量」。错误理解：两者同义。正确结论：goodput 是满足 SLO 达成率前提下的最大请求率；系统裸吞吐再高，若大量请求超 SLO，goodput 仍低。影响 Q4
- 误解 5：「PD 分离后两阶段就各占一半 GPU」。错误理解：资源对半分。正确结论：P:D 配比由两阶段负载不对称决定且以整副本部署单元为最小粒度（DeepSeek-V3 报告单元 32P:320D）。影响 Q5
- 适用边界：
  - 解决：读懂 MoE 服务系统论文所需的最小机制与度量语言
  - 不解决：如何训练 MoE、如何写推理引擎、如何选择具体并行配置
  - 结论成立条件：数字均以 DeepSeek-V3 技术报告（2024-12）、DeepSeek-V3/R1 推理系统概览（2025-03）、DistServe（OSDI'24）的原文为准；不同模型的层数/专家数/top-k 不同
  - 条件不满足时：换模型（如 Mixtral 8 专家 top-2）后 N2/N3 数字失效，但机制描述（F1、dispatch/combine、两阶段）仍成立
