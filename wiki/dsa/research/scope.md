# DSA 内容范围

## 0. 概念歧义处理

缩写 DSA 在不同领域指向不同对象，需先裁定。

| 候选含义 | 领域 | 处置 |
|---|---|---|
| DeepSeek Sparse Attention | LLM 架构 | 已裁定，本页采用 |
| Data Structures and Algorithms | 计算机基础课程 | 排除，与本页领域无关 |
| Digital Signature Algorithm | 密码学 | 排除，与本页领域无关 |
| Dynamic Sparse Attention | 稀疏注意力泛称 | 排除但需在正文声明区别：DSA 是特定实现，不是"动态稀疏注意力"这一类方法的统称 |

裁定依据：用户语境为 LLM 推理框架工程（vLLM、长上下文服务），且要求"工程侧权重更高"。该语境下 DSA 唯一指 DeepSeek Sparse Attention。

同一含义内部还存在一处版本歧义，需并列呈现：

- DSA 首次公开于 DeepSeek-V3.2-Exp 技术报告（2025 年 9 月），作为该实验版本相对 V3.1-Terminus 的唯一架构改动。
- 正式版论文 arXiv:2512.02556《DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models》明确 V3.2 与 V3.2-Exp 架构完全相同（§ DeepSeek-V3.2 Architecture 首段）。

因此 DSA 的机制描述在两个版本间一致，本页以 arXiv:2512.02556 的 LaTeX 源码为公式正本，不区分版本。无"无法消歧"项。

## 1. 概念含义

- 概念名称：DeepSeek 稀疏注意力
- 英文名称：DeepSeek Sparse Attention
- 常见缩写：DSA

### 一句话定义

DSA 是一种可训练的细粒度稀疏注意力机制：先用一个廉价的"lightning indexer"给历史中每个位置打相关性分数，再只对分数最高的 k 个位置做完整注意力，把主注意力的计算复杂度从 $O(L^2)$ 降到 $O(Lk)$。

### 正式定义

按 arXiv:2512.02556 § DeepSeek Sparse Attention，DSA 的原型由两个组件构成：

1. lightning indexer：计算查询 token $\mathbf{h}_t$ 与在它之前的 token $\mathbf{h}_s$ 之间的 index score $I_{t,s}$，决定该查询选哪些 token。
2. fine-grained token selection mechanism：只取 index score 排前 k 的 key-value 条目 $\{\mathbf{c}_s\}$，在查询与这些条目之间做注意力。

### 本页采用的语境

大语言模型解码器的自注意力层，因果（causal）设定，长上下文推理与训练。工程侧以 DeepSeek 官方开源推理实现和 vLLM 的落地为主要材料。

### 包括什么

| 项 | 为什么属于 DSA |
|---|---|
| lightning indexer 的公式、结构与设计取舍 | DSA 两大组件之一，index score 的定义即 DSA 的核心 |
| top-k 细粒度 token 选择 | DSA 两大组件之一，决定稀疏模式 |
| indexer 在 MLA 的 MQA 模式下实例化 | 论文明确 DSA 在 V3.2 中就是这样落地的，且这是 kernel 层能高效的前提 |
| 两阶段续训：dense warm-up + sparse training | indexer 的参数从哪来。稀疏模式不是免费的，必须训出来 |
| 复杂度与推理成本结论 | DSA 的存在理由，也是它的适用边界所在 |
| 工程落地：FP8 index kernel、indexer 独立 KV cache、短序列 dense 回退、DCP 下 top-k 合并 | 用户明确工程侧权重更高；这些是"机制如何真的跑起来"的组成部分 |

### 不包括什么

| 项 | 排除理由 |
|---|---|
| V3.2 的 RL 框架与 agentic 数据合成 | 属于 arXiv:2512.02556 的另外两项贡献，与注意力机制无关 |
| MoE / DeepSeekMoE 路由 | V3.2 的 MoE 部分与 V3.1 相同，不是 DSA 的组成部分 |
| MLA 本身的推导（低秩压缩、吸收技巧） | 独立概念，已有概念页 `wiki/mla/`，引用不内联 |
| FP8 量化的一般原理 | 独立概念，已有概念页 `wiki/quantization-basics/`，引用不内联 |
| V3.2 的完整评测表 | 本页只需要"是否有性能回退"这一条结论，不需要逐 benchmark 复述 |
| NSA 的完整机制 | 相邻概念，只说明关系与关键区别，不展开 |

### 相邻概念

| 概念 | 关键区别 | 是否纳入 |
|---|---|---|
| NSA（Native Sparse Attention） | 同为可训练稀疏注意力。DSA 论文引用它来支撑"kernel 层面每个 KV 条目必须被多个 query 共享"这一约束。DSA 用单一 indexer 打分 + 共享候选集；NSA 是另一篇独立论文的另一套设计 | 仅说明关系与引用点，不展开机制 |
| 滑动窗口注意力 | 稀疏模式是固定的局部窗口，与内容无关；DSA 的选择是学出来的，可以选到很远的位置 | 作为对照点纳入，用于说明"可训练稀疏"的含义 |
| 线性注意力 / KDA | 换掉 softmax 注意力的形式本身，用固定大小状态替代 KV cache；DSA 保留 softmax 注意力，只是少读一些位置，KV cache 仍然全量保留 | 作为对照点纳入，引用 `wiki/linear-attention/`、`wiki/kda/` |
| FlashAttention | 优化的是稠密注意力"怎么算"（分块、访存），仍然读全部位置；DSA 改的是"读哪些位置"。两者正交 | 作为对照点纳入，用于消除"稀疏注意力就是省显存的 kernel 优化"这一误解 |
| MLA | DSA 在 V3.2 中基于 MLA 实例化。MLA 压缩每个位置存什么，DSA 决定读哪些位置 | 作为前置知识引用 `wiki/mla/` |

## 2. 学习目标

### Q1：为什么长上下文场景需要 DSA 这样的机制，滑动窗口或线性注意力为什么不够？

- 完成答案：读者应能说明稠密注意力的代价随序列长度平方增长，且在 decode 阶段每生成一个 token 都要重读整个 KV cache，长上下文下这部分成为瓶颈；固定窗口稀疏虽然便宜但会丢掉窗口外的信息，无法应对"关键信息在很远处"的情况；线性注意力换掉了注意力形式、用固定状态替代 KV cache，是另一条路线，代价是精确检索能力。DSA 的取向是保留 softmax 注意力的精确检索能力，只减少每个 query 实际读取的位置数，且让"读哪些"由模型学出来。
- 为什么是核心目标：不理解这一点，读者会把 DSA 误当成一个 kernel 优化或显存优化，无法判断它适用于什么场景。
- 依赖内容：稠密注意力代价、KV cache 在 decode 中的角色、固定窗口稀疏的局限、线性注意力路线的差异。

### Q2：lightning indexer 如何给一个历史位置打分，为什么它比主注意力便宜得多？

- 完成答案：读者应能写出 index score 公式并解释每个符号；说明它便宜的四个来源——头数少、维度低、用 ReLU 而非 softmax、可以在 FP8 下计算；并说明它便宜但复杂度仍是 $O(L^2)$，只是每对 token 的单价远低于 MLA。
- 为什么是核心目标：indexer 是 DSA 的核心，也是"稀疏为什么能省"的关键；只说"打个分"无法解释省在哪。
- 依赖内容：index score 公式（F1）、indexer 的配置数字（N1、N2）、ReLU 与 softmax 的差别、FP8（引用概念页）。

### Q3：top-k 选择如何接到主注意力上，为什么候选集在所有注意力头之间共享？

- 完成答案：读者应能说明选择后的注意力公式（F2）；说明共享候选集的原因是 kernel 效率——每个 KV 条目必须被多个 query 共享，否则每个头产生各自的不规则访存模式；并说明 DSA 因此实现在 MLA 的 MQA 模式上，一个 latent 向量被该 token 的所有 query 头共享。
- 为什么是核心目标：这是算法设计被硬件约束反向决定的一处，也是理解工程实现的入口。
- 依赖内容：F2、MLA 的 MQA/MHA 模式（引用 `wiki/mla/`、`wiki/mqa-gqa/`）、C4 的 kernel 约束论断。

### Q4：indexer 的参数从哪里来——为什么需要 dense warm-up 和 sparse training 两个阶段？

- 完成答案：读者应能说明 indexer 是新增模块、初始权重无意义，必须先学会"模仿主注意力认为哪些位置重要"；warm-up 阶段保持稠密注意力、冻结除 indexer 外的全部参数，用主注意力分数跨头求和再 L1 归一化得到目标分布，以 KL 散度为损失训练 indexer；sparse training 阶段引入 top-k、放开全部参数，KL 只在被选中的集合上计算；并说明 indexer 的输入从计算图上 detach，indexer 只由 $\mathcal{L}^I$ 驱动、主模型只由语言建模损失驱动。
- 为什么是核心目标：这解释了"可训练稀疏"与"启发式稀疏"的本质差别，也解释了为什么不能把 DSA 直接套到一个未经续训的稠密模型上。
- 依赖内容：F3、F4、两阶段的超参数与 token 量（N3–N8）、KL 散度的含义。

### Q5：DSA 在推理时省下了什么、没省下什么，什么情况下它不起作用？

- 完成答案：读者应能说明主注意力复杂度从 $O(L^2)$ 降到 $O(Lk)$，但 indexer 自身仍是 $O(L^2)$，只是单价低得多；KV cache 并未因 DSA 变小，全部历史位置仍须保留且 indexer 还额外增加一份自己的 key cache；序列长度不超过 k 时选择不产生任何裁剪，此时反而应走稠密路径（官方为短序列 prefill 专门实现 masked MHA 模式，vLLM 中对应 `seq_len <= topk_tokens` 时走 dense MHA）。
- 为什么是核心目标：这是 DSA 的适用边界，也是工程上最容易踩错的判断。
- 依赖内容：C7、C8、C9、N2、vLLM 与官方实现的对应代码位置。

## 3. 内容分级

### 核心内容

| 编号 | 内容 | 对应学习目标 | 必须讲清的结论 |
|---|---|---|---|
| K1 | 稠密注意力在长上下文下的代价结构 | Q1 | 复杂度随 L 平方增长；decode 时每步重读整个 KV cache |
| K2 | 可训练稀疏 vs 固定模式稀疏 | Q1 | 选哪些位置由内容决定而非位置决定，可选到远处 |
| K3 | index score 公式与符号 | Q2 | 每个 indexer 头算一个 ReLU 后的点积，按 $w_{t,j}^I$ 加权求和 |
| K4 | indexer 便宜的四个来源 | Q2 | 头数少、维度低、ReLU 替代 softmax、FP8 |
| K5 | top-k 选择与稀疏注意力输出 | Q3 | 只取 top-k 的 KV 条目参与注意力 |
| K6 | 候选集跨头共享与 MQA 模式实例化 | Q3 | kernel 效率要求 KV 条目被多 query 共享 |
| K7 | dense warm-up 阶段 | Q4 | 冻结主模型，KL 对齐主注意力分布 |
| K8 | sparse training 阶段 | Q4 | 引入 top-k，KL 只在选中集合上，indexer 输入 detach |
| K9 | 复杂度结论与 indexer 的残留 $O(L^2)$ | Q5 | 主注意力 $O(Lk)$；indexer 仍 $O(L^2)$ 但单价低 |
| K10 | KV cache 未减少 + indexer 额外 cache | Q5 | DSA 省计算不省 KV 显存 |
| K11 | 短序列失效与 dense 回退 | Q5 | $L \le k$ 时选择无裁剪作用 |

### 辅助内容

| 编号 | 内容 | 服务对象 |
|---|---|---|
| B1 | FlashAttention 与 DSA 的正交关系 | 消除"稀疏注意力=kernel优化"误解，服务 Q1 |
| B2 | 线性注意力/KDA 路线对照 | 澄清 DSA 不改变注意力形式，服务 Q1 |
| B3 | ReLU 与 softmax 在打分场景的差别 | 服务 K4，解释为什么能省 |
| B4 | indexer 用自己的 RoPE 且非 interleaved | 服务 Q2 的工程侧，解释 indexer 是独立打分通道而非复用主注意力的 q/k |
| B5 | index score 的 FP8 计算链（官方 kernel 注释逐步） | 服务 K4 与 Q5，把"FP8 便宜"落到实际算子 |
| B6 | DCP 下局部 top-k 合并的正确性论证 | 服务 Q3 的工程侧，说明稀疏选择在并行切分下如何保持精确 |
| B7 | NSA 与 DSA 的关系 | 消除"DSA 是 NSA 的重命名"误解 |

### 扩展内容

| 编号 | 内容 | 纳入 |
|---|---|---|
| E1 | V3.2 完整 benchmark 表 | 排除，只取 parity 结论 |
| E2 | H800 上的 token 成本曲线数值 | 部分纳入：只用"成本随位置变化、长序列显著加速"这一定性结论与其成立条件（H800、2 USD/GPU 小时的估算口径），不复述曲线读数 |
| E3 | 后续模型（GLM-5 等）采用 DSA | 排除，属于影响面而非机制 |
| E4 | TileLang / CUDA 双版本算子开源 | 纳入一句，作为 B5 的来源说明 |

## 4. 前置知识映射

| 前置知识 | 被哪些学习目标依赖 | 概念页 | 递归层级 |
|---|---|---|---|
| 标准（稠密）注意力 | Q1、Q2、Q3 | `../standard-attention/index.html` | 已存在，0 层 |
| MLA | Q3、Q5 | `../mla/index.html` | 已存在，0 层 |
| MQA / GQA | Q3 | `../mqa-gqa/index.html` | 已存在，0 层 |
| 量化基础（FP8） | Q2、Q5 | `../quantization-basics/index.html` | 已存在，0 层 |
| RoPE | Q2（B4） | `../rope/index.html` | 已存在，0 层 |
| 线性注意力 | Q1（B2） | `../linear-attention/index.html` | 已存在，0 层 |
| KDA | Q1（B2，作为线性注意力的具体实例对照） | `../kda/index.html` | 已存在，0 层 |

全部前置知识在 `wiki/` 下均已有概念页，无需递归生成，无第 3 层登记项。

## 5. 明确不展开的内容

| 项 | 与 DSA 的关系 | 不展开原因 |
|---|---|---|
| MLA 的低秩压缩推导与吸收技巧 | DSA 实例化在 MLA 之上 | 属于另一独立概念，已有概念页 |
| FP8 的数值格式细节（e4m3、ue8m0 缩放） | indexer 在 FP8 下计算 | 属于另一独立概念，已有概念页；本页只需"低精度使单价更低"这一判断 |
| NSA 的机制细节 | DSA 引用它的 kernel 层约束 | 属于另一独立概念，本页只需要它支撑的那一条约束 |
| V3.2 的 RL 与 agentic 训练 | 同一篇论文的其他贡献 | 不影响任何学习目标的回答 |
| TileLang 语言本身 | 官方算子的实现语言 | 只影响工程实现规模，不改变机制 |
| MTP / 投机解码与 DSA 的交互 | vLLM 实现中确有相关分支 | 属于另一独立概念，不影响本页学习目标 |

## 6. 常见误解和适用边界

### 误解 1：DSA 能减少 KV cache 显存

- 错误理解：既然只读 top-k 个位置，那就只需要存这 k 个位置。
- 正确结论：DSA 不减少 KV cache。每个 query token 选的 top-k 集合不同，且后续 token 可能选到任何历史位置，因此全部历史位置的 KV 条目必须完整保留。此外 lightning indexer 还需要自己的一份 key cache（官方实现中是 FP8 的 `k_cache` 加一份 scale cache；vLLM 中是独立的 `DeepseekV32IndexerCache`），DSA 实际上略微增加了显存占用。
- 形成原因：把"稀疏读取"与"稀疏存储"混为一谈。
- 影响学习目标：Q5。

### 误解 2：DSA 是一种 kernel 优化，和 FlashAttention 同类

- 错误理解：都是让注意力更快的技术。
- 正确结论：FlashAttention 改变稠密注意力"怎么算"（分块、减少 HBM 往返），计算的数学结果与朴素实现一致，仍读全部位置。DSA 改变"读哪些位置"，计算结果与稠密注意力不同，因此必须通过续训让模型适应这个新的稀疏模式。两者正交且可叠加。
- 形成原因：都以"加速注意力"为宣传点。
- 影响学习目标：Q1。

### 误解 3：可以把 DSA 直接加到任意训练好的稠密模型上

- 错误理解：indexer 只是打分，装上去即可用。
- 正确结论：indexer 是新增参数模块，随机初始化时打出的分数没有意义，直接 top-k 会切掉真正重要的位置。DeepSeek 的做法是两阶段续训：先用 KL 散度让 indexer 学会模仿主注意力的分布（1000 步、2.1B token），再引入 top-k 并放开全部参数做稀疏训练（15000 步、943.7B token）。近千亿 token 的续训不是可以省略的步骤。
- 形成原因：把 indexer 当成一个无参数的启发式打分器。
- 影响学习目标：Q4。

### 误解 4：DSA 把注意力复杂度降到线性

- 错误理解：$O(Lk)$ 中 k 是常数，所以是线性的。
- 正确结论：主注意力部分确实变成 $O(Lk)$，在 k 固定时对 L 线性。但 lightning indexer 仍需给每个 query 扫过全部历史位置，复杂度仍是 $O(L^2)$。论文明确写了这一点，其论证是 indexer 的计算量相比 V3.1-Terminus 中的 MLA 小得多，而非把它消掉。因此 DSA 是"降低了平方项的系数并把主项变成线性"，不是整体线性。
- 形成原因：只看主注意力的复杂度结论。
- 影响学习目标：Q5。

### 误解 5：每个注意力头各选自己的 top-k

- 错误理解：不同头关注不同模式，自然应该各选各的。
- 正确结论：DSA 使用单一共享候选集，该 token 的所有 query 头读同一批 KV 条目。原因是 kernel 效率：论文引用 NSA 指出每个 KV 条目必须被多个 query 共享，若每头独立选择会产生多套不规则访存模式。DSA 因此实现在 MLA 的 MQA 模式上，一个 latent 向量被该 token 的全部 query 头共享。
- 形成原因：按"多头注意力各头独立"的直觉外推。
- 影响学习目标：Q3。

### 适用边界

- 解决什么问题：长上下文下主注意力的计算量与访存量；使长序列的训练与推理成本显著下降。
- 不解决什么问题：KV cache 显存占用；短序列场景的效率（序列长度不超过 k 时无裁剪作用）；不改变注意力的数学形式，因此不带来线性注意力那类的常数状态优势。
- 结论成立需要的条件：序列长度显著大于 k（V3.2 中 k=2048）；模型已经过 DSA 续训；有配套的 FP8 index kernel 与稀疏注意力 kernel；KV 条目跨 query 头共享（MQA 模式）。
- 条件不满足时会发生什么：$L \le k$ 时 top-k 选不出任何裁剪，额外的 indexer 计算成为纯开销，此时应走稠密路径——官方为短序列 prefill 专门实现 masked MHA 模式来模拟 DSA，vLLM 中对应 `prefill_max_seq_len <= topk_tokens` 时置 `use_dense_mha=True`；未经续训的模型直接开启 top-k 会显著掉点；缺少共享候选集时 kernel 无法高效实现。
