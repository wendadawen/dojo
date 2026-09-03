# PCP 与 DCP 核心论断与证据

来源编号与定位：

- S1 = vLLM 官方文档《Context Parallel Deployment》。本地副本：`/Users/wendadawen/code/HCF-Distributed/vllm/docs/serving/context_parallel_deployment.md`（fork 自上游，与 docs.vllm.ai 最新版一致；上游 latest 对应 v0.28.0）。在线：https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
- S2 = vLLM 官方博客《Efficient Decode Context Parallelism with vLLM for Long Context Workloads》，2026-08-07。https://vllm-project.github.io/2026/08/07/decode-context-parallelism.html
- S3 = vLLM 源码 `vllm/config/parallel.py`（上游 main 分支）。https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py
- S4 = vLLM GitHub RFC #26133《[RFC]: Support Context Parallelism with Fully Sharded KV Cache and Ring Attention》（2025-10）。https://github.com/vllm-project/vllm/issues/26133
- S5 = Helix Parallelism 论文（arXiv:2507.07120，2025-07）。https://arxiv.org/abs/2507.07120
- S6 = Ring Attention 论文（arXiv:2310.01889）。https://arxiv.org/abs/2310.01889
- S7 = vllm-ascend 文档《Context Parallel Guide》。http://vllm-ascend.readthedocs.io/en/main/user_guide/feature_guide/context_parallel.html

## C 论断（定义与机制）

### C1 prefill 与 decode 特性不同、SLO 不同，上下文并行需分别实现

- 内容：prefill 和 decode 呈现完全不同的特性并有不同的 SLO（服务等级目标），因此上下文并行要针对两个阶段分别实现。
- 来源定位：S1 开篇："As prefill and decode present quite different characteristics and have quite different SLO (service level objectives), we need to implement context parallel separately for them."
- 适用条件：LLM 自回归推理的两阶段划分。
- 置信状态：已确认。

### C2 两阶段的目标：PCP 控制 TTFT，DCP 腾出 KV 空间提高吞吐

- 内容：长上下文 prefill 要把计算时间摊到多个 rank 上以控制 TTFT；长上下文 decode 需要更多 KV cache 空间以增大 batch 从而提高吞吐。
- 来源定位：S1："For long context prefill, we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens. For long context decode, we need more space for KV cache to increase the batchsize (and hence the throughput)."
- 适用条件：同 C1。
- 置信状态：已确认。

### C3 decode 每步只算少量 query token、读大量 KV token

- 内容：自回归解码的每步只需计算少量新 query token，却要访问 paged KV cache 中大量历史 KV token；DCP 的核心是如何跨 GPU 分片 KV cache。
- 来源定位：S1 Decode Context Parallel 节："Due to the auto-regressive nature of decoding, every decoding step needs to compute a small amount of query tokens w.r.t. a large number of key/value tokens stored in the paged KV cache. The core of decode context parallel is how to shard the KV cache across GPUs."
- 适用条件：标准自回归 Transformer 解码。
- 置信状态：已确认。

### C4 TP 先沿 head 维切 KV，head 切尽后产生 $t/H$ 倍重复

- 内容：单卡放不下或想容纳更多请求时，先沿 $H$（KV 头数）维分片即普通 TP；$H$ 由模型架构决定，继续增大 tp 后每张 GPU 的 KV cache 重复 $t/H$ 次；再加 DCP 沿 $T$ 维继续分片。
- 来源定位：S1："we can first shard the KV cache along the H dimension, that's the plain tensor parallel sharding... Since H is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for tp_size / H times. Of course, duplication is not good for efficiency. Then we need to add decode context parallel to further shard the KV cache along the T dimension."
- 适用条件：$t > H$ 且整除时重复恰为 $t/H$ 倍。
- 置信状态：已确认。

### C5 GQA 头数少、MLA 有效单头，两者都撞 head 维天花板

- 内容：GQA 模型只存少量 KV 头，TP 最多切到每卡一个头，超过后开始复制；MLA 把 K/V 压成跨 query 头共享的低秩潜向量，有效 KV 头数为 1，普通 TP 下潜向量 KV cache 在每个 TP rank 完整复制。
- 来源定位：S2 §1："GQA models store a small number of KV heads, and TP can only split the KV cache down to one head per GPU; once TP exceeds the number of KV heads, the cache starts duplicating across GPUs. Multi-head latent attention (MLA) models make this even worse: MLA compresses the Key/Value into a single low-rank latent vector shared across all query heads, so it effectively has only one KV head. Under normal TP there is nothing to split by head, meaning the latent KV cache is replicated in full across every TP rank."
- 适用条件：GQA/MLA 架构；MHA（头数多）不必然撞天花板。
- 置信状态：已确认。

### C6 DCP 沿序列维分片，每卡持唯一切片，不浪费容量在副本上

- 内容：DCP 沿序列（token 位置）维分片 KV cache，每张 GPU 持有唯一切片，不把容量浪费在副本上，腾出的显存用于接纳更多请求、跑更大 batch。
- 来源定位：S2 §1："DCP instead shards the KV cache along the sequence dimension, so each GPU holds a unique slice and no capacity is wasted on duplicates. This frees up GPU memory, allowing each GPU to take on more requests and thus run at a larger batch size."
- 适用条件：互联带宽足够的系统（S2 同段："On systems with high-bandwidth GPU-to-GPU interconnects, this helps preserve interactive responsiveness..."）。
- 置信状态：已确认。

### C7 dcp 不增加启动的 GPU 数，只减少 KV 重复

- 内容：`-dcp <size>` 不增加需要启动的 GPU 数量，只减少 KV cache 重复。
- 来源定位：S1："This is as simple as adding -dcp <size> to the command line. Note that size does not increase the number of GPUs we need to launch, but just reduces the KV cache duplication."
- 适用条件：无 PCP 时（DCP 复用 TP rank）。
- 置信状态：已确认。

### C8 dcp 取值范围 $[1, t/H]$，越大重复越少、通信开销越大

- 内容：dcp size 应落在 $[1, \text{tp\_size}/H]$ 范围内；dcp 越大 KV 重复越少，但通信开销越大。理论上可以超过 $t/H$ 继续切分加速 decode，但 decode 的 query token 数有限，多出的 rank 在非 attention 层无明确分工，为简单起见上界取 $t/H$。
- 来源定位：S1："The dcp size should lie in the range of [1, tp_size/H]. With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases. Theoretically, it is possible to extend the dcp size beyond tp_size / H... it's unclear what should we do for the remaining dcp_size - tp_size / H GPUs for non-attention layers. For the sake of simplicity, dcp size is upper bounded by tp_size / H."
- 适用条件：$t/H$ 为整数时边界清晰；非整除时按博客 GQA 约束用整除（见 C16）。
- 置信状态：已确认。

### C9 源码约束：无 PCP 时 $t \bmod d = 0$；有 PCP 时 $d \in \{1, \text{pcp}, t \cdot \text{pcp}\}$；PCP 不支持 DP

- 内容：`_validate_parallel_config` 校验：PCP>1 且 DP>1 报错"PCP does not support data parallelism yet."；PCP=1 时要求 tp 可被 dcp 整除（DCP 复用 TP rank）；PCP>1 时 dcp 必须取 1（禁用）、pcp（跨 PCP 轴）或 tp×pcp（跨完整 TP×PCP 块）之一。
- 来源定位：S3 `_validate_parallel_config`：
  ```python
  if pcp > 1 and self.data_parallel_size > 1:
      raise ValueError("PCP does not support data parallelism yet.")
  if pcp == 1:
      if tp % dcp != 0:
          raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")
  elif dcp not in (1, pcp, tp * pcp):
      raise ValueError("When PCP is enabled, DCP must be disabled, span the PCP "
          "axis, or span the full TP x PCP axis. ...")
  ```
- 适用条件：上游 main 分支当前校验逻辑，随版本演进可能变化。
- 置信状态：已确认。

### C10 博客给出的模型侧约束：MLA 可切到整个 TP 度，GQA 上界为 $t // H$

- 内容：MLA 因有效 KV 头数为 1，序列可切到整个 TP 度（$t \ge d$ 且 $t \bmod d = 0$）；GQA 的切分度被重复因子 $t // \text{num\_kv\_heads}$ 封顶（$(t//H) \ge d$ 且 $(t//H) \bmod d = 0$）。
- 来源定位：S2 §5.3："Because the effective KV-head count is 1, the sequence can be split up to the full TP degree. Constraints: tensor_parallel_size >= decode_context_parallel_size, tensor_parallel_size % decode_context_parallel_size == 0."；S2 §5.4："the sequence-split degree is capped by the duplication factor tp // num_key_value_heads. Constraints: (tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size, (tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0."
- 适用条件：MLA 与 GQA 分别适用各自约束；与 S1 的 $[1, t/H]$ 在整数情形一致。
- 置信状态：已确认。

### C11 KV cache 沿 $T$ 维用交错策略分片，未来 token 自然落位

- 内容：KV cache 在 decode 中会增长，分片策略需精心实现；vLLM 用交错（interleaving）策略沿 $T$ 维分片，使未来 token 的 KV cache 能自然地沿 $T$ 维分片。该策略由 Moonshot 的 Chao Hong 提出，Helix Parallelism 论文有详细解释。
- 来源定位：S1："Note that kv cache can grow during decoding, and the sharding strategy needs to be carefully implemented. We use an interleaving strategy to shard the KV cache along the T dimension, so that kv cache for future tokens can be naturally sharded along the T dimension. This is proposed by Chao Hong from Moonshot, and also explained in details in this paper (arXiv:2507.07120)."
- 源码细节（S3 `cp_kv_cache_interleave_size` docstring）："Store interleave_size tokens on dcp_rank i, then store next interleave_size tokens on dcp_rank i+1. Interleave_size=1: token-level alignment, where token i is stored on dcp_rank i % dcp_world_size. Interleave_size=block_size: block-level alignment..."
- RFC 印证（S4）："During decoding, new Key-Value (KV) pairs are distributed across CP ranks in a round-robin fashion."
- 适用条件：vLLM 实现；interleave size 可配置（token 级或块级）。
- 置信状态：已确认。

### C12 decode 一步的通信节奏：AllGather Q → 本地计算 → AllGather + ReduceScatter

- 内容：标准 DCP 通信模式：每卡只算出 query 的一个片段，但 attention 需要完整 query 向量对任意 key 打分，因此先在 DCP 组内 AllGather 聚齐 query（decode 时 query 只有一个 token，开销小）；各卡用聚齐后的 query 对本地 KV 切片计算 attention；部分结果通过 AllGather 共享每卡的部分输出与 LSE，LSE 值对部分结果重新加权合并（online-softmax 技巧），ReduceScatter 求和并让每卡取回自己的 head 切片。
- 来源定位：S2 §4.1："Standard Decode Context Parallelism keeps the communication pattern simple, following the rhythm AllGather Q → Compute → AllGather + ReduceScatter."；"An all-gather across the DCP group assembles a complete copy of the query on every GPU. This is cheap during decode because the query is a single token."；"AllGather shares each GPU's partial output and LSE; the LSE values reweight and merge the partials (the online-softmax trick), and ReduceScatter sums them while handing each GPU back only its own head-slice."
- 适用条件：标准 ag_rs 后端路径；a2a 后端交换方式不同（见 C14）。
- 置信状态：已确认。

### C13 博客用连续 token 区间示意 DCP 分片

- 内容：每张 GPU 负责同一序列一段 token 位置的 KV cache；博客以 200K 请求为例给出 GPU0 存 0–50K、GPU1 存 50K–100K 的连续区间示意。
- 来源定位：S2 §4："Each GPU is made responsible for the KV cache of a chunk of token positions from the same sequence. For a single 200K-token request, GPU 0 might hold the cache for tokens 0–50K, GPU 1 for tokens 50K–100K, GPU 2 for tokens 100K–150K, and GPU 3 for tokens 150K–200K."
- 适用条件：讲解性示意；vLLM 实际存储按 C11 的交错规则分配。两者是"按 token 位置分片"的概念与"交错落位"的实现关系，页面需同时呈现并说明关系，避免读者以为矛盾。
- 置信状态：已确认（注意与 C11 的关系说明）。

### C14 两种 DCP 通信后端：ag_rs 与 a2a；MLA 可选 query 投影复制

- 内容：源码提供两种 DCP 通信后端——`ag_rs`（AllGather + ReduceScatter，默认）与 `a2a`（All-to-All 交换部分输出与 LSE 后用 Triton kernel 合并，对 MLA 模型每层 NCCL 调用从 3 次降到 2 次）。MLA 另有可选的 query 投影复制：加载时在每个 DCP 组内复制（较小的）query projection，使 decode 跳过 query all-gather（环境变量 `VLLM_DCP_Q_REPLICATE=1`，PR #45964）。
- 来源定位：S3 `dcp_comm_backend` docstring："Communication backend for Decode Context Parallel (DCP). 'ag_rs': AllGather + ReduceScatter (existing behavior); 'a2a': All-to-All exchange of partial outputs + LSE, then combine with Triton kernel. Reduces NCCL calls from 3 to 2 per layer for MLA models."；S3 `dcp_q_replicate` docstring："Replicate the MLA query projection within each DCP group so decode can skip the query all-gather."
- 博客（S2 §4.1）关于 q replicate："As an opt-in alternative for MLA, vLLM #45964 can replicate the (small) query projection within each DCP group at load time so decode skips this query all-gather entirely (VLLM_DCP_Q_REPLICATE=1)."
- a2a 的现状注意：S2 §6（2026-08）将更好的 A2A kernel 列为 future work（"We are also developing better DCP all-to-all (A2A) communication kernels..."），而 S3（main 源码）已有 `a2a` 选项。呈现时说明 a2a 为较新加入的后端选项。
- 适用条件：源码 main 分支；博客成文时 a2a 仍在开发。
- 置信状态：已确认。

### C15 MLA 与 GQA 的 DCP 计算路径差异

- 内容：MLA 路径下 DCP 沿序列维切潜向量 KV cache，每 rank 只存自己那段的潜向量，attention 时各自上投影本地潜向量切片（`k_up` 步骤）重建所需 K/V；GQA 路径下 DCP 把原本会重复的副本填上不同的序列切片，共享的 KV 头跨其 query 头广播（`tensor_broadcast for GQA` 步骤）。
- 来源定位：S2 §5.3："DCP splits the latent KV cache along the sequence dimension, so each rank stores only its chunk of the latent; at attention time each rank up-projects its latent slice (the k_up step) to reconstruct the Keys/Values it needs."；S2 §5.4："DCP takes those would-be-duplicate copies and fills them with different sequence chunks instead, while the shared KV heads are broadcast across their query heads (the 'tensor broadcast for GQA' step)."
- 适用条件：分别对应 MLA/GQA 模型。
- 置信状态：已确认。

### C16 attention 阶段按序列分片、FFN 阶段同一批 GPU 重组为全池

- 内容：DCP 让每张 GPU 都有活干：attention 阶段按序列分片，随后立刻把同一批 GPU 重组成完整池，摊薄 FFN 权重加载。
- 来源定位：S2 §7："DCP puts every GPU to work: sharding the sequence during attention, then immediately reconfiguring those same GPUs to amortize FFN weight loading across the full pool."
- 适用条件：Helix 论文（S5）对同一思想的表述为"attention 阶段 KV 并行、FFN 阶段同一批 GPU 复用为 TP（dense）或 TP×EP（MoE）"。
- 置信状态：已确认。

### C17 PCP 两种策略：partial-Q/full-KV 与 partial-Q/partial-KV（ring attention）

- 内容：prefill 一个 $T$ token 的长请求时，$N$ 张 GPU 把请求切成 $N$ 段，每卡计算一段的 Q/K/V。策略一（partial query, full key/value）：token 长度中等、能容纳全量 K/V 时，聚齐所有 GPU 的 K/V，各卡算自己段的 query 对应的 attention 输出。策略二（partial query, partial key/value）：序列太长、放不下全量 K/V 时，每卡只算一段 Q/K/V，用 ring attention 逐块收发 K/V。
- 来源定位：S1 Prefill Context Parallel 节全文，关键句："Say we have N GPUs, we can split the request into N chunks, and each GPU computes one chunk of the query/key/value tensors."；"Partial query, full key/value: If the request token length is moderately long (we can afford holding the full key/value tensors)... gather the key/value tensors from all GPUs and let each GPU compute the attention output corresponding to the query tokens of its chunk."；"Partial query, partial key/value: If the request token length is too long... use techniques like ring-attention to send/recv key/value tensors chunk by chunk."
- ring attention 出处：S1 引用链接指向 arXiv:2310.01889（S6）。
- 适用条件：S1 标注 "Both approaches are under active development."（成文时）。
- 置信状态：已确认。

### C18 causal mask 造成 prefill 负载不均，2×cp 配对切分均衡负载

- 内容：causal attention 使每个 token 的计算负载不同；为均匀分布负载，把序列切成 $2 \times \text{cp\_world\_size}$ 块，CP rank $i$ 同时分配第 $i$ 块和第 $(2 \times \text{cp\_world\_size} - i - 1)$ 块，使各 rank 计算负载均衡。
- 来源定位：S4（RFC #26133）："Causal attention imposes a varying computational load for each token, as shown in the following figure. To ensure an even workload distribution, tokens should be partitioned across different context parallelism (CP) ranks. Specifically, the sequence is divided into 2 × cp_world_size chunks. Each CP rank i is assigned both the i-th chunk and the (2 × cp_world_size - i - 1)-th chunk. This approach helps balance the compute load among all CP ranks."
- 佐证：RFC #22693 提出同名思想的 DualChunkSwap 策略（"we split the sequence into 2*cp_size parts instead of splitting it into cp_size parts"）。
- 适用条件：这是 vLLM RFC 描述的 CP 负载均衡设计；页面表述为"RFC 给出的配对切分设计"，不对现网实现版本作断言。
- 置信状态：已确认（作为 RFC 设计呈现）。

### C19 PCP 扩展 world size，DCP 不扩展

- 内容：`prefill_context_parallel_size` 是切分 prefill 序列计算的 rank 数，PCP 扩展进程 world size 但不增加 KV cache 分片数；`decode_context_parallel_size` 是切分 decode KV cache 的 rank 数，DCP 不扩展 world size，无 PCP 时复用 TP rank，有 PCP 时跨 PCP 轴或完整 TP×PCP 块。world size 计算为 PP × TP × PCP。
- 来源定位：S3 字段 docstring："Number of ranks that split prefill sequence computation. PCP expands the process world size but does not increase the KV-cache shard count."；"Number of ranks that shard the decode KV cache. DCP does not expand the process world size. Without PCP, DCP reuses TP ranks. With PCP, DCP either spans the PCP axis or the full TP x PCP block."；S3 `__post_init__`：`self.world_size = (self.pipeline_parallel_size * self.tensor_parallel_size * self.prefill_context_parallel_size)`。
- 适用条件：上游 main 源码。
- 置信状态：已确认。

### C20 官方选型建议：先加 tp 到满意，再加 dcp 消重复

- 内容：对 DCP，先把 tp 加大到性能满意，然后加 dcp 减少 KV cache 重复。
- 来源定位：S1："In short, for decode context parallel, try to increase -tp size until you get satisfactory performance, and then add -dcp to reduce the KV cache duplication."
- 适用条件：vLLM 部署经验建议。
- 置信状态：已确认。

### C21 DCP 支持状态：MLA 与 GQA 均已支持约一年；PCP 仍在开发

- 内容：vLLM 的 DCP 支持 MLA 和 GQA 模型，部分 attention backend 还支持 DCP 与 MTP 组合；写博客时 DCP 已支持近一年。PCP 两种策略均在活跃开发中（官方文档），博客将 PCP 列为更长期的 roadmap。
- 来源定位：S1："Decode context parallel is supported in vLLM, for both MLA and GQA models. Some attention backends also support the combination of decode context parallel and MTP (multi-token prediction)."；"Both approaches are under active development."；S2："vLLM has supported DCP for almost a year, but we are writing this blog now to highlight the feature..."；S2 §6："there is a longer roadmap for Prefill Context Parallelism (PCP)."
- 适用条件：以官方文档当前状态为准。
- 置信状态：已确认。

### C22 PCP/DCP 与 PD 分离是不同维度，可组合

- 内容：PD 分离是 prefill 与 decode 使用不同 GPU 池的部署方式；PCP/DCP 是池内单条长请求的跨卡执行方式；两者正交可组合。vLLM 正在加固 P/D 分离支持使 DCP 在分离式部署中稳健（博客 future work）。
- 来源定位：S2 §6："as well as hardening prefill/decode (P/D) disaggregation support to make DCP robust in disaggregated serving deployments."（"robust in disaggregated serving deployments"蕴含 DCP 可与 P/D 分离共存，当前在加固）；S1 的整体结构（PCP/DCP 均为"进入某个资源池后单请求如何跨卡"的机制，不涉及池的划分）。PD 分离的定义见仓库页面 `../ppd-disaggregation/index.html`（论文页）。
- 适用条件：概念辨析性结论，依据来源结构与博客表述。
- 置信状态：已确认（辨析性表述，正文标注依据）。

## F 公式

### F1 KV cache 条目数

- 内容：$H$ 个 KV 头的模型，一条 $T$ token 上下文的请求需要在 KV cache 中存 $H \times T$ 个 key/value 张量。
- 来源定位：S1："For a model with H kv-heads, a request with T tokens in the context needs to store H * T key/value tensors in the KV cache."
- 适用条件：按"头 × token"计条目数；MLA 时 $H$ 取有效值 1。
- 置信状态：已确认。

### F2 重复因子与每卡存储份额

- 内容：TP 规模 $t$、KV 头数 $H$：$t \le H$ 时无重复、每卡存 $H/t$ 个头的 KV；$t > H$ 时每个 KV 头复制到多张卡，KV cache 总副本数为 $t/H$ 份（重复因子 $\max(1, t/H)$）。加 dcp $= d$ 后每卡存储份额为 $\max(1, t/H)/(t \cdot d)$ 份完整 KV（即总副本数降为 $\max(1, t/H)/d$）。
- 来源定位：$t/H$ 重复倍数直接来自 S1（C4、C8）；每卡份额与总副本的换算是 $H \times T$ 计数下的直接算术（推导链：每卡头数 $= H \cdot \lceil t/H \rceil / t$ 在整除时 $= \max(H/t, 1)$，再除以 token 维分片度 $d$）。
- 适用条件：$t$ 与 $H$（或 $t$ 与 $t/H$）整除情形；非整除时按博客约束向下取整（C10）。
- 置信状态：已确认（$t/H$ 部分）；换算部分为算术推导，正文标注推导链。

### F3 LSE 合并公式

- 内容：DCP 组内第 $r$ 卡对本地 KV 切片算得部分输出 $o_r$ 与本地 log-sum-exp $l_r = \ln \sum_{i \in \text{rank } r} e^{s_i}$（$s_i$ 为该 query 对本地各 key 的分数），全局 attention 输出为
  $$o = \frac{\sum_r e^{l_r}\, o_r}{\sum_r e^{l_r}},$$
  与整卡对全部 KV 计算 softmax attention 的结果恒等。
- 来源定位：合并机制与"online-softmax trick"的命名来自 S2（C12 引文："the LSE values reweight and merge the partials (the online-softmax trick)"）；公式本身由 softmax 分子分母按卡分组分解直接得出（推导链：全局 softmax 的分子 $\sum_i e^{s_i} v_i = \sum_r \sum_{i \in r} e^{s_i} v_i = \sum_r e^{l_r} o_r$，分母同理），正文给手算验证。
- 适用条件：精确 attention（非稀疏/近似）；对任意分数值成立。
- 置信状态：已确认（机制来自 S2，恒等式为数学推导并在页面手算验证）。

### F4 causal prefill 的块负载与配对切分

- 内容：序列切成等长 $b$ 的块，第 $j$ 块（0 起）的 query 需 attend 到第 $0..j$ 块的 KV，attention 计算量 $\propto (j+1) b^2$；把 $2N$ 块按下标配对（rank $i$ 取第 $i$ 块与第 $2N-1-i$ 块）时每卡负载 $\propto (i+1)b^2 + (2N-i)b^2 = (2N+1)b^2$，与 $i$ 无关，负载恒定。
- 来源定位：配对规则来自 S4（C18 引文）；负载计算量与"首尾配对和恒定"为等差数列的直接算术（推导链）。
- 适用条件：等长块、causal mask、忽略非 attention 层（其负载本就与位置无关）。
- 置信状态：配对规则已确认；恒等式为算术推导，正文手算验证。

## N 数字

### N1 DCP 实测性能（Kimi K2.6 / 8×B200 / NVFP4）

- 数值：单节点 8×B200 服务 Kimi K2.6（NVFP4），并发从 16 扫到 512。基线 TP 在并发 64 时 KV 使用率达 100% 撞墙，吞吐平台期约 1,863 tok/s/GPU；DCP 支撑到并发 512，达 6,091 tok/s/GPU，KV 使用率 82%。按长度分桶（<32k、32–64k、64–128k、128–200k、200k+）中 DCP 在 200k+ 段仍保持高且稳定的前沿。
- 来源定位：S2 §2.2："It reaches 100% at a concurrency of 64 and hits a wall, and throughput plateaus near 1,863 tok/s/GPU because no additional requests can fit."；"DCP reaches 6,091 tok/s/GPU at c512 while still sitting at just 82% KV usage."；S2 §2.3："DCP keeps a high, stable frontier even in the 200k+ range."
- 实验条件：单节点 8×B200、Kimi K2.6 NVFP4、vLLM、仅改变 decode 阶段 KV 分片方式。
- 置信状态：已确认。

### N2 实验数据集（Mooncake trace 格式 agentic 负载）

- 数值：输入长度中位数约 67k token，输出约 400 token；约 53% 请求 ≥64k（重尾至约 1M），约 47% <64k（其中约 18% <8k）；约 8% 超过 128k，约 3–4% 超过 256k。多轮 agentic 负载（长输入短输出）。
- 来源定位：S2 §2.1："Inputs are centered around a median of ~67k tokens and paired with short ~400-token outputs... roughly half the requests sit at 64k+ (≈53%, with a heavy tail reaching ~1M tokens) and half are short-to-mid (≈47% under 64k, ~18% under 8k). About 8% of requests exceed 128k and ~3–4% exceed 256k."
- 适用条件：该公开 agentic trace，不代表所有业务流量。
- 置信状态：已确认。

### N3 三个官方 case study 的头数与重复倍数

- 数值：DeepSeek-R1 开 MLA 时 KV 头数为 1，`-tp 8` 造成 8 倍 KV 重复，可加 `-dcp 8` 消除。Kimi-K2 架构类似 R1 但参数更多，`-tp 16` 时重复 16 倍，`-dcp 16` 完全消除，或 `-dcp 8` 降到 2 倍（DCP 通信只发生在节点内，开销更小）。Qwen3-235B-A22B 有 4 个 KV 头，`-tp 8` 时重复 2 倍，`-dcp 2` 消除。
- 来源定位：S1 Case study 节："For DeepSeek-R1, we have 1 kv-head when MLA is enabled. The typical single-node deployment with -tp 8 causes 8x KV cache duplication. We can consider adding -dcp 8..."；"For Kimi-K2... When we deploy it with -tp 16, the KV cache duplication is 16x. We can add -dcp 16 to completely remove the KV cache duplication... We can also add -dcp 8 to reduce the KV cache duplication to 2x. Although it still duplicates the KV cache twice, the communication overhead is smaller since the DCP communication only happens inside one node."；"For Qwen3-235B-A22B, we have 4 kv-heads. When we deploy it with -tp 8, the KV cache duplication is 2x. Then we can add -dcp 2 to remove the KV cache duplication."
- 适用条件：所列模型与部署规模。
- 置信状态：已确认。

### N4 a2a 后端对 MLA 的每层 NCCL 调用数

- 数值：a2a 后端把 MLA 模型每层的 NCCL 调用从 3 次降到 2 次。
- 来源定位：S3 `dcp_comm_backend` docstring："Reduces NCCL calls from 3 to 2 per layer for MLA models."
- 适用条件：MLA 模型、a2a 后端、上游 main 源码。
- 置信状态：已确认。

## 冲突与不足记录

- S1 的 dcp 上界 $t/H$ 与 S3 源码校验（仅 $t \bmod d = 0$）并不相同：S1 讲的是有意义的推荐范围（超过 $t/H$ 后非 attention 层无明确分工），S3 是可运行的硬校验。页面按"官方推荐范围 + 源码硬校验"两层呈现，不写成矛盾。
- S2 博客（2026-08）将更好的 A2A kernel 列为 future work，S3（main）已有 `a2a` 选项：时间先后关系，页面标注 a2a 为较新后端。
- S2 的连续区间示意与 S1/S3/S4 的交错分片：前者是概念示意，后者是 vLLM 实际存储分配。页面同时呈现并说明关系（见 C13）。
- vllm-ascend（S7）的 PCP 兼容矩阵（experimental、仅 ModelRunner V2 等）未纳入核心论断：硬件移植与上游 CUDA 版不同步，且页面不展开兼容矩阵。
