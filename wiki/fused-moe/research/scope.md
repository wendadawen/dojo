# FusedMoE 内容范围

## 1. 概念歧义处理

状态：已裁定。

- "FusedMoE" / "fused MoE" 在 vLLM 语境下指 `vllm/model_executor/layers/fused_moe/` 模块族提供的融合 MoE 算子与层实现：文件头自述 "Fused MoE Triton kernels."，历史版本（v0.10.2）中模型使用的层类名为 `FusedMoE`，v0.28.0 中由 `FusedMoEFactory` 组装 Router + RoutedExperts + MoERunner。
- 泛化含义：把 MoE 层中"路由后所有专家的矩阵乘"合并为少数几次内核调用的算子技术；同类实现存在于 SGLang、TensorRT-LLM 等推理系统，思想先例是 MegaBlocks 的块稀疏 MoE 表述（arXiv:2211.15841）。

裁定依据：用户在 vLLM 生态语境下提出该名词；vLLM 是该名词最主流的载体。本页以 vLLM v0.28.0（commit 2cf0a69）源码为参考实现讲机制，其他系统的实现只在边界处提及、不展开。

## 2. 概念含义

- 概念名称：FusedMoE（融合 MoE 算子）
- 英文名称：Fused Mixture of Experts，vLLM 模块名 fused_moe
- 简要定义：把 MoE 层中路由完成后全部专家参与的矩阵乘，组织成一（或两）次分块 GEMM 内核调用，而不是每个专家一次小 GEMM。
- 正式定义（与来源一致）：
  - fused_moe_kernel 计算 A(*, K) 与 B(E, N, K) 的分块乘法得到 C(M, topk, N)，B 为堆叠的全部专家权重，按 expert_ids 为每个 tile 选择专家矩阵（v0.28.0 fused_moe.py 内核 docstring）。
  - fused_experts_impl 流程：专家分配对齐 → 第一次 GEMM(w1) → MoE 激活（默认 silu_and_mul）→ 第二次 GEMM(w2) → moe_sum 沿 topk 求和（v0.28.0 fused_moe.py L1772–L1851）。
- 本文采用的语境：单进程内 MoE 专家计算的组织方式；以未量化 bf16 的 Triton 路径为机制基准。

### 包括什么

- 朴素逐专家循环的问题分析（小 GEMM 形状退化、内核启动次数、gather/scatter）
- moe_align_block_size 的三个数组（sorted_token_ids、expert_ids、num_tokens_post_padded）的语义与构造，含填充上界
- fused_moe_kernel 的 tile 索引逻辑：A 的行还原（除以 top_k）、B 的专家偏移、掩码跳过填充、路由权重乘法位置、网格划分与 L2 分组
- 一次前向的完整数据流与可手算的贯穿示例；朴素实现与融合实现在数值上等价的可运行验证
- FusedMoE 层在 vLLM 中的组成与职责边界：TP 下 w13 列切 / w2 行切、EP 下 expert_ids=-1 写零、调优配置的按形状查找、极小批量的跳过对齐路径
- MegaBlocks 作为"MoE 计算表述为分块/块稀疏 GEMM"的思想先例（动机：丢 token 与填充浪费的权衡）

每项属于本概念的理由：都是"把专家计算合并进分块 GEMM 内核"这一机制的组成部分或直接边界。

### 不包括什么

- 量化路径的内核内部（fp8_w8a8 / int8 / int4 的反量化与 scale 索引）：属于量化实现，只说明这些路径复用同一内核框架
- 路由算法变体（grouped top-k、sigmoid 打分、e_score_correction_bias）：模型结构问题
- 专家并行的 dispatch/combine 通信与 DeepEP 等通信库：并行通信问题，内核只负责本 rank 内计算
- 调优 JSON 的离线生成流程与 benchmark 工具：工程流程
- 其他 MoE 后端（DeepGEMM、CUTLASS、FlashInfer、ROCm AITER）的内部实现：只提到存在
- 性能对比数字（除 MegaBlocks 论文自带的加速数字作为动机旁证）：无本页可核对的实验

排除理由：以上各项均不影响"融合算子如何组织计算"这一核心机制的理解，且各有独立主题。

### 相邻概念

- DeepEP / UltraEP（EP 通信库）：FusedMoE 解决单 rank 内计算组织，通信由 EP 组件承担；不纳入
- EPLB（专家负载均衡）：改变专家摆放与复制，不改变内核机制；不纳入，正文链接概念页
- CUDA Graph：FusedMoE 可被 capture，属于执行调度话题；不纳入
- MegaBlocks（块稀疏 MoE 训练内核）：思想先例，作为动机与对照纳入第 1 章少量篇幅

## 3. 学习目标

### Q1：MoE 层的专家计算为什么不能靠"每个专家一次普通 GEMM"的循环来做？

- 完成答案：能说明朴素循环在推理（尤其 decode）下的三重浪费：每个专家分到的 token 行数极小导致 GEMM 退化成访存受限的小矩阵乘、铺不满 GPU 的 SM；内核启动次数随专家数线性增长；逐专家 gather/scatter 与中间张量的额外访存。
- 为什么是核心目标：不理解朴素实现的浪费，就无法判断"融合"到底融合了什么、收益从哪来。
- 依赖内容：MoE 层结构与 top-k 路由（moe-serving）、GEMM tile 与 SM 执行（gpu-execution-model）。

### Q2：moe_align_block_size 如何把"哪个 token 归哪个专家"编码成三个数组？

- 完成答案：给定 topk_ids 与 block_size，能手算 sorted_token_ids（槽位按专家排序、每段填充到 block_size 倍数）、expert_ids（每个 block 的专家）、num_tokens_post_padded（填充后总数），并说明填充上界公式。
- 为什么是核心目标：这是融合内核正确性的枢纽——保证一个 tile 只含同一专家的 token。
- 依赖内容：Q1 的结论（为什么需要分块对齐）。

### Q3：fused_moe_kernel 如何用这三个数组算出"正确的 token × 对应专家权重"？

- 完成答案：能说明网格如何划分（EM 上取 block、N 上取 block）、每个 program 如何取 A 的行（sorted_token_ids 的值除以 top_k 还原 token 行）、如何偏移 B 到对应专家、如何用掩码跳过填充槽、GEMM2 里 MUL_ROUTED_WEIGHT 在哪一步生效。
- 为什么是核心目标：内核索引逻辑是本概念的核心机制，错一处即结果错误。
- 依赖内容：Q2 的三数组、GEMM 分块计算（gpu-execution-model）。

### Q4：一个 token 从进入 MoE 层到输出，完整经过哪些计算？

- 完成答案：能按顺序列出 router 打分 → top-k 选择与权重 → 对齐 → GEMM1(w13) → silu_and_mul → GEMM2(w2) → moe_sum 加权求和，并对一个 2 维小例子手算出单个 token 的输出。
- 为什么是核心目标：数据流把前面三章的机制串成完整前向，是"学完能复述"的验收。
- 依赖内容：Q2、Q3、SwiGLU 门控激活（swiglu）。

### Q5：FusedMoE 层管什么、不管什么？

- 完成答案：能说出它在 vLLM 中承担单 rank 内专家计算组织（含 TP 切分下的形状、EP 下非本 rank 专家写零），不承担路由算法变体、EP 通信、负载均衡；能说明调优配置按形状查找与极小批量跳过对齐路径的存在。
- 为什么是核心目标：防止把相邻系统的职责误挂到本概念上，明确适用边界。
- 依赖内容：Q4、TP 列切/行切（model-parallelism）、EP 概念（moe-serving）。

## 4. 内容分级

### 核心内容

- 朴素循环的三重浪费（→ Q1）：小 GEMM 形状退化是主因，必须给出"decode 批量下每专家 token 数极少"的量化说明（构造数字）
- 三数组语义与构造、填充上界（→ Q2）
- 内核 tile 索引：A 行还原、B 专家偏移、掩码、网格划分（→ Q3）
- 五步数据流与 w13/w2 形状、激活语义、加权求和位置（→ Q4）
- 贯穿示例：T=4、E=4、topk=2、block_size=2、H=2、I=2 的手算全程（→ Q2/Q3/Q4）
- 职责边界：TP/EP 表现、调优配置、跳过对齐路径（→ Q5）

### 辅助内容

- GROUP_SIZE_M 的 L2 分组（服务 Q3 的网格划分理解，不展开 swizzle 推导）
- cache13 内存复用（服务 Q4 的实现细节理解）
- MegaBlocks 的 drop-vs-pad 权衡（服务 Q1 的历史动机）
- 误解澄清：融合≠整层一个内核、不丢 token≠无填充、EP 通信不在本层（服务各章）

### 扩展内容

- 量化路径与多后端的存在性说明（纳入，一段带链接）
- topk_softmax 融合内核的出处（纳入，一句）
- SGLang/TensorRT-LLM 同类实现（排除，无一手核对材料）

## 5. 前置知识映射

| 前置概念 | 依赖的学习目标 | 概念页状态 |
|---|---|---|
| MoE 层结构、router 与 top-k 路由、专家并行 | Q1、Q5 | wiki/moe-serving（已有 concept 页） |
| GEMM tile、SM、内核启动 | Q1、Q3 | wiki/gpu-execution-model（已有 concept 页） |
| SwiGLU 门控激活（silu 门控乘 up） | Q4 | wiki/swiglu（已有 concept 页） |
| 张量并行列切/行切 | Q5 | wiki/model-parallelism（已有 concept 页） |

全部已有，无需递归生成。

## 6. 明确不展开的内容

- 量化内核内部：不影响未量化路径的机制理解，量化反量化是独立主题（wiki 已有 fp8-block-quant 等页面）
- 路由变体与 DeepSeek 系列的 grouped topk：模型结构差异，不改变内核数据流
- EP 通信（all-to-all、DeepEP）：并行通信主题，本页只需 expert_ids=-1 的写零行为
- 调优流程与性能数字：工程流程与实验，缺少本页可核对来源
- Triton 语言本身的编程教学（tl.load/tl.dot 语法）：理解机制只需"program 处理一个 tile"的执行模型，已在 gpu-execution-model 覆盖

## 7. 常见误解和适用边界

### 常见误解

1. 错误：FusedMoE 把 router、GEMM、求和全部融进一个内核。正确：融合对象是"全部专家的 GEMM 合并为一次调用"；整层仍是多个内核（topk_softmax、对齐、两次 GEMM、激活、求和）接力。影响 Q4/Q5。
2. 错误：不丢 token 是因为没有填充。正确：存在填充（每专家段补到 block 边界，上界 E×(block−1) 个槽），填充槽被掩码跳过、不算错值；不丢指所有 (token, expert) 对都参与计算，与容量截断路由相对。影响 Q2。
3. 错误：FusedMoE 包含专家并行的 all-to-all 通信。正确：单 rank 语义不含通信；EP 的 dispatch/combine 由其他组件承担，内核只对非本 rank 专家写零。影响 Q5。
4. 错误：BLOCK_SIZE_M 越小越好（减少填充）。正确：tile 大小影响 GEMM 效率与 SM 占用，配置按批量从调优表选取（默认 16→128 随 M 增大）；填充浪费有上界且被掩码。影响 Q3/Q5。

### 适用边界

- 概念解决：单进程内 MoE 专家计算的组织——把变长的按专家分组变成固定 tile 的分块 GEMM。
- 不解决：路由质量与负载均衡（EPLB 的地盘）、跨 rank 通信（DeepEP 等）、权重如何量化存储。
- 结论成立条件：以 vLLM v0.28.0 的 Triton 未量化路径为基准；量化路径复用同一框架但内核内有额外反量化步骤；其他后端（DeepGEMM 等）机制不同，本页结论不直接迁移。
- 条件不满足时：极小批量（num_tokens×top_k×4 ≤ 全局专家数）时 vLLM 走跳过对齐的低延迟路径，三数组退化为直通形式；此时"分块对齐"描述不适用，但 GEMM 语义不变。

## 8. 完成条件核对

- 歧义已裁定，无无法消歧项
- 5 个学习目标，完成答案均可在正文章节落地
- 核心内容均有对应目标；前置知识全部映射到已有概念页
- 误解与边界具体可查
- 见 evidence.md / outline.md / glossary.md
