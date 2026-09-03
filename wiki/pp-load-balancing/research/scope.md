# 内容范围：PP 负载均衡

## 1. 概念歧义处理

- 概念名称：PP 负载均衡（pipeline parallelism load balancing），目录名 `pp-load-balancing`。
- 歧义状态：已裁定。用户语境中的缩写 PPLB 不是公开标准术语（公开领域 PPLB 指网络领域的 per-packet load balancing 等无关概念），页面不使用 PPLB 作为正式名称。裁定依据：用户 2026-09-03 明确含义为「大模型推理中 PP 各 stage 间的负载均衡，做法含 micro-batch 与 token 维度切分」；公开资料中该概念存在于 pipeline stage balancing、balanced pipeline parallelism（gLLM 论文标题）、token-level pipeline parallelism（TeraPipe）等名义下，与用户指定含义一致。
- 同名/近名概念辨析：
  - EPLB（expert parallelism load balancing）：MoE 专家维度的负载均衡，与 PP 是不同并行维度，排除。
  - PD 分离实例间路由（PPD 路由）：请求该进 P 池还是 D 池的决策，属实例间调度，已有页面 `ppd-disaggregation`，排除并互链。
  - 层维度划分（每 stage 放几层）：同属 PP 负载均衡的一类做法，但用户已学过、明确排除本页（2026-09-03 用户指示「层的划分就没必要，这个之前介绍过了」）。

## 2. 概念含义

- 简要定义：LLM 推理系统采用 PP 部署时，让各流水线 stage 在时间上保持相近利用率、减少气泡的切分与调度机制。
- 本文语境与正式定义：聚焦两个运行时维度——
  1. token 维度切分：把单条请求的输入序列切成片段，让不同片段同时出现在不同 stage（TeraPipe 提出、Mooncake 以 CPP 落地到推理 prefill、SGLang 以 CPP + 动态 chunk 完成时间对齐）。
  2. batch 维度均衡：并发请求组成 micro-batch 时，配平各 micro-batch 的计算量（gLLM 的 token throttling）。
- 权威依据：gLLM（arXiv:2504.14775）§2.4 将流水线气泡归因于 inter-stage 与 inter-batch 两类依赖，将失衡归因于 inter-stage 与 inter-batch 两类不均；TeraPipe（arXiv:2102.07988）§3.2 指出流水线总时延由最慢 stage 决定；Mooncake（arXiv:2407.00079）§5.1 定义 chunked pipeline parallelism。
- 包括什么：
  - token 级流水的可行性依据（自回归依赖性质）——属于它，因为这是序列维度切分的存在理由。
  - 最优切分问题（切点为何前长后短、目标函数、DP 求解）——属于它，是 token 维度均衡的核心。
  - 固定 chunk 的失衡与动态 chunk 对齐——属于它，是 token 维度均衡在长前缀下的补充。
  - micro-batch 填充的极限与 inter-batch 失衡、token throttling——属于它，是 batch 维度均衡的核心。
  - 两维度的组合（TeraPipe §3.4 联合切分）——属于它，回答两类做法的关系。
- 不包括什么：
  - 层维度划分调优（vLLM `VLLM_PP_LAYER_PARTITION` 等）——用户已学过，排除。
  - EPLB——不同并行维度，排除。
  - PD 实例间路由——已有页面覆盖，排除。
  - 训练侧流水线调度（GPipe 1F1B、interleaved、zero-bubble）——不影响推理场景的学习目标，只在背景提一句「训练里还有一类按调度顺序消气泡的做法」。

## 3. 学习目标

### Q1：推理中 PP 的忙闲不均从哪来，代价是什么？

- 完成答案：GPU 空闲来自两类依赖——inter-stage（后级等前级完成）与 inter-batch（并发 micro-batch 数受流水线深度限制）；推理有两个放大器：单条长请求整体进入流水线时下游 stage 长时间空等，以及注意力的增量计算特性使同尺寸 chunk 的处理时间随前缀增长非线性增加、错配在高 PP rank 级联放大。代价：一次迭代的节拍由最慢 stage 决定，气泡占比理论值为 $\frac{K-1}{K-1+M}$（$K$ 为 stage 数、$M$ 为 micro-batch/chunk 数，结论引用模型并行页），实际错配下高于理论值。
- 为什么是核心目标：不理解失衡的来源与代价，就无法判断两类做法各自解决什么。
- 依赖内容：PP 定义与气泡公式（model-parallelism 页）、causal attention 依赖性质（standard-attention / causal-mask 页）、prefill/decode 两阶段（chunked-prefill 页）。

### Q2：一条长请求怎么沿 token 维度拆进流水线，切点为什么前长后短？

- 完成答案：decoder-only 模型中位置 $t$ 的计算只依赖 $\le t$ 的 token，因此同一序列的不同片段可以同时在不同 stage 上计算。位置越靠后注意力交互次数越多，均分序列会使各片段计算量递增（第 $i$ 片段的交互数含其对全部前缀的注意力）；最优切分前长后短。TeraPipe 目标函数 $T^*=\min\{\sum_i t_i+(K-1)\max_j t_j\}$：第一项是单个 stage 的总前向时间，第二项是流水线开销、由最慢片段乘 $K-1$。Mooncake 把该机制以 CPP 形式用于推理 prefill：每 $X$ 个节点组成流水线组，输入切成不超过 `prefill_chunk` 的 chunk，同请求不同 chunk 由不同节点同时处理；跨节点通信只发生在 stage 边界且可与计算重叠；短上下文无显著开销。
- 为什么是核心目标：token 维度切分是本页第一个主体机制。
- 依赖内容：Q1、causal attention、KV cache（chunk 间衔接）。

### Q3：固定大小的 chunk 为什么随前缀增长而失衡，动态 chunk 怎么对齐？

- 完成答案：增量注意力使同尺寸 chunk 的处理时间随前缀长度 $L$ 非线性增加；错配沿流水线传播，在更高 PP rank 上复合放大，固定 chunk 下实际气泡比率超过理论值。SGLang 动态 chunk：把累计运行时间建模为 $L$ 的二次函数，解方程 $\mathrm{Runtime}(L+\Delta L)-\mathrm{Runtime}(L)=\mathrm{Runtime}(\text{初始 chunk})$ 得下一 chunk 大小 $\Delta L$，$\Delta L$ 随 $L$ 增长逐步缩小以对齐各 stage 的执行时间；预测值向下对齐到 $\max(\text{page size},64)$ 的倍数；smooth factor（默认 0.75）缓冲模型误差。
- 为什么是核心目标：固定 chunk 失衡是 token 维度做法在实际系统里必须回答的问题。
- 依赖内容：Q2、chunked prefill（chunk 与 page）。

### Q4：并发请求下 micro-batch 之间的计算量怎么配平？

- 完成答案：Sarathi-Serve 用固定 token budget 混排 chunked prefill 与 decode token，波动来自两方面——错过 decode 与 prefill 混批的机会、decode token 在 batch 间分布不均；缩小预算会惩罚 prefill 速率。gLLM 解耦两者并利用全局信息：decode 侧按 $\#D=\#RD/\#PP_{\text{depth}}$ 把 decode token 均摊到全部 micro-batch（$\#RD$ 为运行中 decode token 总数）；prefill 侧按等待 token 数（分摊到 $\#T$ 次迭代）、KV cache 空闲率、溢出阈值三者动态决定 batch 大小，KV 空闲率低于阈值时暂停 prefill 防止 decode 被抢占重算。
- 为什么是核心目标：batch 维度均衡是本页第二个主体机制。
- 依赖内容：Q1、chunked prefill（token budget）、KV cache（水位）。

### Q5：两类做法怎么组合，各自适用什么场景？

- 完成答案：TeraPipe §3.4 对 batch 维与 token 维联合优化：对每个候选 batch size 跑 token 维 DP 得最优 $T_b$，再决定 batch 维切分使总和最小，归约为一维背包问题。适用边界：batch 足够大时最优解退化为只切 batch 维（GPT3-1B 实验）；切分过细时 GPU 利用率下降（GPT3-1B 单层上单 token 与 256 token 前向时间相同）；短上下文下 CPP 无显著开销也无收益；gLLM 只解决 inter-batch 失衡，inter-stage 失衡留作未来工作；token 维切分依赖单向自回归，双向注意力模型不适用。
- 为什么是核心目标：回答两类做法的关系与边界，防止读者把单一做法当成万能解。
- 依赖内容：Q2、Q3、Q4。

## 4. 内容分级

- 核心内容（缺一则学习目标无法完整回答）：
  - 两类依赖与两类失衡的区分（→Q1、Q4）
  - 长请求独占与注意力增量两个放大器（→Q1、Q3）
  - 气泡公式的引用与「最慢 stage 决定节拍」（→Q1、Q2）
  - 自回归依赖性质与序列内流水（→Q2）
  - 均分序列计算量递增、前长后短、目标函数两项（→Q2）
  - Mooncake CPP 机制与两大收益（→Q2）
  - 固定 chunk 级联错配、动态 chunk 方程与 smooth factor（→Q3）
  - Sarathi 固定预算的波动来源（→Q4）
  - gLLM decode 均摊与 prefill 动态调节公式（→Q4）
  - 联合切分与适用边界（→Q5）
- 辅助内容：
  - TeraPipe DP 递推与复杂度、$t_{\text{fwd}}$ 的线性拟合模型——深化 Q2 的求解细节。
  - SGLang 异步 P2P 与 micro-batching 事件循环——澄清「对齐了 chunk 时间还要工程消除阻塞等待」。
  - SGLang 动态 chunk 三步调参法——澄清 smooth factor 怎么选。
  - 腾讯一念的多阶段流水线 batch 间均衡——工业侧 inter-batch 均衡的佐证（折叠块）。
  - DP 最小可运行实现——验证贯穿示例的最优切分（折叠块）。
- 扩展内容（排除本页）：训练侧调度算法族、PP×TP 拓扑选择、KV 传输与卸载、vLLM/SGLang 部署参数细节。

## 5. 前置知识映射

| 前置概念 | 依赖的学习目标 | 页面状态 |
|---|---|---|
| PP 定义、气泡公式 $\frac{p-1}{m+p-1}$、TP×PP 组合 | Q1、Q2 | 已有：`model-parallelism` |
| chunked prefill、token budget、stall-free、与 PP 组合成 CPP 的提法 | Q1、Q3、Q4 | 已有：`chunked-prefill` |
| KV cache 结构与水位含义 | Q2、Q4 | 已有：`kv-cache` |
| causal attention（位置 $t$ 只依赖 $\le t$） | Q2 | 已有：`standard-attention` / `causal-mask`（写作时按实际内容择一引用） |

递归生成：无缺失项，不需要生成新前置页。

## 6. 明确不展开的内容

- 层维度划分（每 stage 层数怎么定、`VLLM_PP_LAYER_PARTITION` 类调参）：用户已在其他材料学过，明确排除（用户 2026-09-03 指示）。
- EPLB / MoE 专家负载均衡：不同并行维度的均衡问题，属另一概念。
- PD 分离实例间路由（PPD）：`ppd-disaggregation` 页已覆盖；本页 Ch5 只在组合处提 SGLang PP 兼容 PD 分离的事实。
- 训练侧流水线调度（1F1B、interleaved、zero-bubble）：不影响推理场景学习目标的回答，属训练系统独立主题。
- Sarathi-Serve chunked prefill 的 stall-free 机制细节：`chunked-prefill` 页已讲，本页只引用其固定 token budget 作为 gLLM 的对照。

## 7. 常见误解和适用边界

### 常见误解

- M1「均分就是均衡」：均衡对象是各 stage 的执行时间，不是元素个数。token 维度上均分序列使后段片段计算量递增；batch 维度上均分请求数不等于均分计算量（请求长度不一）。形成原因：把「数量相等」直觉迁移到「时间相等」。影响 Q2、Q4。
- M2「气泡只因填充不足，多塞 micro-batch 就能解决」：$M\gg K$ 时气泡率确实趋近 0，但推理 serving 中 batch 有限；单条长请求独占时 batch 维没有切分空间；micro-batch 计算量不等时 inter-batch 失衡仍在。形成原因：只见过训练场景的大 batch 流水线。影响 Q1、Q4。
- M3「动态 chunk 的目的是减少 chunk 数量」：目的是对齐各 stage 执行时间；chunk 随前缀缩小反而使总数增加，尾 chunk 过小会损害性能（SGLang 因此建议初始 chunk 取最优固定值的 2–3 倍）。影响 Q3。
- M4「token throttling 就是 token budget 换个名字」：固定预算给每步总量上限且 prefill/decode 耦合；token throttling 独立调节两类 token 且依赖全局反馈（积压量、KV 空闲率）。影响 Q4。
- M5「TeraPipe 的 DP 应该原样照搬到推理」：TeraPipe 面向同步训练（含反向）；推理 forward-only 场景下 Mooncake 采用简化的 chunk 上限方案、SGLang 采用在线二次模型预测，二者都是工程取舍而非照搬。影响 Q2、Q3。

### 适用边界

- token 维切分依赖单向自回归（causal attention）；双向注意力模型（编码器类）不适用。
- 切分过细时 GPU 利用率下降（TeraPipe §3.2：GPT3-1B 单层上单 token 与 256 token 前向时间相同）——chunk 有下限约束的根源。
- 短上下文下 CPP 无显著开销也无收益（Mooncake 原文限定语）。
- batch 足够大时联合 DP 退化为只切 batch 维（TeraPipe GPT3-1B 实验）。
- gLLM 明确只解决 inter-batch 失衡；inter-stage 失衡（层划分那一轴）留作未来工作——与本页排除层维度划分的边界互为印证。
- 性能数字的成立条件：TeraPipe 5.0×（GPT-3 175B、48 台 p3.16xlarge、对比 SOTA 模型并行训练）；SGLang 3.31×（DeepSeek-V3.1、H20、PP4×TP8 对 TP8、chunked prefill 12K）；gLLM 11%–398%（Qwen2.5-14B/32B、Llama-3.1-100B，基线 vLLM v0.8.1 PP 与 SGLang v0.4.3 TP）。引用时必须带条件。

## 8. 页面元数据约定

- h1 方向：`PP 负载均衡（pipeline parallelism load balancing）：token 切分与 batch 配平`（写作时按 style-guide 定稿）。
- `dojo:type`：concept；`dojo:topics`：并行与通信, 推理系统；`dojo:tag`：load-balancing。
- 语言注意：正文不使用 PPLB 作为正式术语；开头说明该词是内部叫法、公开文献的对应名称。
