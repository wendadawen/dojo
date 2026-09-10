# DeepSeek-V4.1-Flash 内容范围

## 2.1 概念含义

- **概念名称**：DeepSeek-V4.1-Flash（官方仓库 `deepseek-ai/DeepSeek-V4.1-Flash`，技术报告标题 *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*）
- **简要定义**：一个 552B backbone + 196B Engram 参数、支持 1M 上下文的多模态 MoE 模型，核心设计目标是压榨长上下文推理的 KV cache 与 prefill 成本。

### 正式定义

由官方材料给出的构成（每项均有来源）：

1. 40 层语言主干 = 20 层因果编码器 + 20 层解码器（CED）；除前两层只用滑动窗口注意力外，每层同时有全局注意力与滑动窗口注意力 <sup>[C1]</sup>
2. 全局 KV 由 CSA2 管理：压缩 + 跨层共享 + Top-K 索引复用，三种静态模式 Full / Reindex / Reuse <sup>[C2]</sup>
3. 主 KV cache 量化到 FP4（MXFP4：E2M1 + 每 16 通道一个 E4M3 scale，无二级 global scale），SWA KV 保持 FP8 <sup>[C3]</sup>
4. 每 token 全局 KV cache 占用 890 字节，约为 DeepSeek-V4-Flash 的 1/4 <sup>[N1]</sup>
5. 每 token 激活 8B（prefill，只跑编码器半栈）/ 16B（decode，全栈）参数 <sup>[N2]</sup>
6. Engram：n-gram 哈希查表的条件记忆，插在层 1 与层 14；DSpark：块式推测解码 draft 头，3 个 MTP 块、block size 5 <sup>[C4][C5]</sup>

### 本文采用的语境

以官方仓库的推理参考实现（`inference/model.py` 等）与 `inference/config.json` 为准；技术报告用于陈述设计动机与宣称数字。两者不一致处按实测记录，不外推。

### 包括什么

- CSA2 的三种模式、压缩器池化、分层稀疏索引器、候选池
- CED 的全局 KV 来源与 SWA 层内计算
- FP4 主 KV 的量化格式与字节账
- 每 token 激活参数量的构成
- Engram 与 DSpark 各自解决的问题（机制细节到"能说明它补什么"为止）

### 不包括什么

- 训练流程（数据配比、head-wise Muon 细节、RL 阶段）：属于训练主题，且本页学习目标不依赖
- 服务端实现（SWA Bounded Replay 的调度、SSD 分层、PD 分离）：属推理系统主题，报告中的 1/8 持久化数字依赖服务实现，本页只登记不推导
- 视觉编码器（32 层 ViT + Aligner）的内部结构：属多模态主题，本页只说明图像 token 参与路由与 Engram 屏蔽

### 相邻概念

| 相邻概念 | 关键区别 | 是否纳入本页 |
|---|---|---|
| DeepSeek-V4-Pro | 1.6T、HCA（压 128）与 CSA（压 4 + Lightning Indexer）逐层交替；V4.1-Flash 是纯 CSA2、无 HCA | 只作对照 |
| DeepSeek-V3.2-Exp 的 DSA | 有 indexer 与 top-k 稀疏选择，但没有跨层 KV 共享、没有压缩、没有候选池 | 作为前置概念引用 |
| MLA | 低秩潜向量压缩 K/V，压缩发生在"每 token 内维"；CSA2 的压缩发生在"跨 token 池化" | 作为前置概念引用 |

## 2.2 学习目标

### Q1：890 字节/token 这个数字是怎么算出来的？

- 完成答案：890 = 主 KV 720 + 索引器 K 170。主 KV 每条目 512 维 FP4（256 B）+ 每 16 通道 1 字节 E4M3 scale（32 B）= 288 B；层 2/8/14 压缩比 2（每条目覆盖 2 个 token）各贡献 144 B/token，层 20 压缩比 1 贡献 288 B/token，合计 720 B；索引器 K 每条目 128 维 FP4（64 B）+ 4 B scale = 68 B，按同样口径贡献 170 B。
- 为什么是核心目标：这是全模型设计的因变量——三种机制（跨层共享、压缩比、FP4）各自贡献多少，只有拆开账才能理解设计取舍。
- 依赖内容：CSA2 的跨层共享结构（哪些层共用一个 cache）、压缩比的定义、FP4 与 scale 的字节布局。

### Q2：CED 为什么能让 prefill 只跑一半的层？

- 完成答案：解码器的全局 KV 不来自各层自己的隐状态，而是由编码器最后一层的隐状态投影而来（报告 Eq. 1）；参考实现里这一条表现为"层 20 的压缩器读取它自己的注意力输入（即层 19 的输出），层 20-39 共用该 cache"。因此 prefill 阶段只需要跑编码器半栈就能拿到解码器的全局 KV；SWA 仍需逐层计算，故有界重放只重算最后 $n_{\text{win}}$ 个 token。
- 为什么是核心目标：CED 是"prefill 8B"与"全局 KV 只存一份"两个结论的共同前提。
- 依赖内容：跨层 KV 共享、滑动窗口注意力、压缩器输入位置。

### Q3：一个 query 到底能看到哪些位置？

- 完成答案：可见集 = 本层滑动窗口内的原始 KV（窗口 128）+ 该层所属 ratio 组的压缩条目中被索引器选中的 Top-512 个；索引器的候选来自建池层（层 20）选出的 2048 个块 × 8 位置 = 16384 个候选位置；不可达的压缩条目在打分阶段就被置为 $-\infty$。
- 为什么是核心目标：这是"稀疏注意力为什么不是近似"的关键——选择规则与可见性规则都可逐条核对。
- 依赖内容：稀疏注意力与 indexer、压缩条目与位置对应关系、候选池机制。

### Q4：8B / 16B 激活参数量是怎么来的？

- 完成答案：单层每 token 激活约 377M（注意力 126.6M + gate 1.97M + 共享专家 35.4M + top-6 路由专家 212.3M + mHC 0.98M + norm）；编码器 20 层约 7.89B（含层 1、14 的 Engram 投影），解码器 20 层约 7.58B；prefill 只跑编码器得 8B，decode 全栈得 15.5B ≈ 16B。
- 为什么是核心目标：把"激活参数"从宣传口径变成可复算的账，并说明 MoE 稀疏性与 CED 半栈各自贡献多少。
- 依赖内容：MoE 路由（top-6 + 共享专家）、Engram 的稀疏查表与稠密投影的区别。

### Q5：Engram 与 DSpark 各自补了什么？

- 完成答案：Engram 用 n-gram 哈希查表给残差流补"可寻址的条件记忆"，每 token 只查 24 行、但投影矩阵是稠密的（每层约 157M 参数被激活）；DSpark 用 3 个 MTP 块做块式推测解码，一次前向产出 5 个草稿位置（block size 5），并用 Markov 头与置信头控制接受率。
- 为什么是核心目标：这两个模块不改变注意力与缓存账，但决定"每 token 激活参数"与"解码吞吐"，是理解整机成本的两块拼图。
- 依赖内容：n-gram 哈希与嵌入表、推测解码、MTP。

## 2.3 内容分级

**核心内容**

| 内容 | 服务目标 | 必须说明的结论 |
|---|---|---|
| 压缩器池化公式与组内位置约定 | Q1、Q3 | 组 $j$ 覆盖 token $[jr,(j+1)r)$，RoPE 位置取 $j \cdot r$ |
| CSA2 三模式的判定条件与运行期共享 | Q1、Q2、Q3 | Full = kv_source ∩ index_source；Reindex 自算 Top-K；Reuse 两者都复用 |
| 跨层共享分组 | Q1、Q2 | 层 2-7 / 8-13 / 14-19 / 20-39 各自共用一个 cache |
| FP4 量化格式与字节账 | Q1 | 每 16 通道一个 E4M3 scale，条目 288 B |
| CED 的全局 KV 来源 | Q2 | 解码器全局 KV 由 $H_{L/2}$ 投影，参考实现由层 20 的压缩器落实 |
| 可见性规则（窗口 + 压缩 + Top-K + 候选池） | Q3 | 不可达压缩条目置 $-\infty$；候选池 2048×8 |
| 激活参数量构成 | Q4 | 单层 377M、半栈 7.89B / 7.58B |
| Engram 的稀疏查表与稠密投影 | Q5 | 24 行查表 vs 157M 投影 |

**辅助内容**

- 报告与参考实现的差异清单（哪些设计只存在于报告/服务实现）：防止读者把报告设计当成代码行为
- 参考实现的 prefill/decode 不一致实测：澄清"逐位一致"不是这类系统的验收标准
- 与 V4-Pro 的对照（HCA vs 纯 CSA2）：解释"pure CSA2"的含义

**扩展内容**

- DSpark 的 Markov 头与置信头的具体公式：纳入但只到机制层，不展开训练
- 视觉路径细节：排除（见 2.5）

## 2.4 前置知识映射

| 前置概念 | 被哪些目标依赖 | 页面 | 状态 |
|---|---|---|---|
| 滑动窗口注意力 | Q2、Q3 | — | **缺失，需递归生成** |
| 注意力汇聚点（attention sink） | Q3 | — | **缺失，需递归生成** |
| 稀疏注意力与 indexer | Q3 | [dsa](../../wiki/dsa/index.html) | 已有 |
| KV cache 与显存布局 | Q1 | [kv-cache](../../wiki/kv-cache/index.html)、[kv-cache-layout](../../wiki/kv-cache-layout/index.html) | 已有 |
| 低秩潜变量注意力（MLA） | Q1 | [mla](../../wiki/mla/index.html) | 已有 |
| 超连接与 Sinkhorn 双随机混合 | Q3、Q4 | [hyper-connections](../../wiki/hyper-connections/index.html) | 已有 |
| MoE 路由（无辅助损失偏置） | Q4 | [deepseek-moe](../../wiki/deepseek-moe/index.html)、[aux-loss-free-routing](../../wiki/aux-loss-free-routing/index.html) | 已有 |
| MXFP4 量化 | Q1 | [mxfp4-qat](../../wiki/mxfp4-qat/index.html)、[mixed-precision-quant](../../wiki/mixed-precision-quant/index.html) | 已有 |
| n-gram 哈希查表 | Q5 | [ngram](../../wiki/ngram/index.html) | 已有 |
| 推测解码 | Q5 | [speculative-decoding](../../wiki/speculative-decoding/index.html) | 已有 |
| RoPE 与 YaRN | Q1、Q3 | [rope](../../wiki/rope/index.html) | 已有 |
| RMSNorm | Q1 | [rmsnorm](../../wiki/rmsnorm/index.html) | 已有 |

缺失项在引用处先写该概念在当前上下文中的最小含义，页面生成后再补链接。

## 2.5 明确不展开的内容

- **训练配方与优化器**：与 KV cache / prefill 成本主线无关；head-wise Muon 属训练主题。
- **服务端 KV 分层与 SWA Bounded Replay 调度**：报告的 1/8 持久化数字依赖服务实现，本页只登记宣称与成立条件，不推导。
- **视觉编码器内部结构**：图像路径只影响 MoE 路由偏置与 Engram 屏蔽，展开会脱离主线。
- **量化 kernel 实现（tilelang）**：属工程实现，本页只用其语义（分块量化 + scale）。

## 2.6 常见误解和适用边界

| 错误理解 | 正确结论 | 形成原因 | 影响目标 |
|---|---|---|---|
| "890 B/token 是每层缓存之和" | 跨层共享后每 token 只存 4 份主 KV（层 2/8/14/20）与 4 份索引器 K；不共享时主 KV 一项就是 8352 B/token | 忽略 CSA2 的跨层共享 | Q1 |
| "压缩比 2 就是每 2 个 token 存一个 KV，和池化一样" | 池化权重是可学习门控的 softmax（每通道一个权重），不是平均；且压缩条目在索引器里另有 FP4 键 | 把压缩等同于平均池化 | Q1、Q3 |
| "Reuse 层复用 Top-K 索引，所以它和 Full 层看到一样的东西" | 复用索引的前提是同一份主 KV；不同 ratio 组的主 KV 不同，跨组不能复用 | 忽略"索引与主 KV 绑定" | Q3 |
| "prefill 8B 说明解码器不参与推理" | decode 阶段仍要跑满 40 层；8B 只是 prefill 阶段的计算量口径 | 把激活参数口径当成结构裁剪 | Q2、Q4 |
| "报告 Eq. 1 说每个解码器层都有自己的 KV 投影权重" | 该式是 CED 的一般形式；与 CSA2 结合后只有解码器的 Full 层（层 20）计算全局 KV，其余解码器层共用 | 未读 CSA2 与 CED 的结合说明 | Q2 |

**适用边界**

- 本页的字节账适用于 1M 上下文、序列长度远大于窗口（$N \gg n_{\text{win}}$）的场景；短序列下压缩条目很少，索引器与候选池几乎不起作用。
- 可见性规则（哪些位置可见）对所有长度成立；但"Top-512 里选得对不对"依赖索引器训练质量，本页只核对选择机制不评价质量。
- 参考实现与报告设计不一致时，本页以参考实现描述"代码实际行为"、以报告描述"设计意图"，两者分别标注。
