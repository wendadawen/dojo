# FusedMoE 核心论断与证据

版本基准：vLLM v0.28.0，commit 2cf0a6915ce544dc493a0990f2ea38d81601128a（N2）。
源文件定位均相对仓库根目录；fused_moe.py = vllm/model_executor/layers/fused_moe/fused_moe.py，其余类推。所有行号按 v0.28.0 原文件。

## C 论断（定义与机制）

### C1 模块与内核命名
- 论断：vLLM 在 `vllm/model_executor/layers/fused_moe/` 实现融合 MoE 算子，fused_moe.py 文件头为 "Fused MoE Triton kernels."，核心 Triton 内核名为 `fused_moe_kernel`。
- 来源定位：fused_moe.py L3、L298–299。
- 适用条件：v0.28.0。
- 置信状态：已确认。

### C2 内核的矩阵语义
- 论断：fused_moe_kernel 计算 A(*, K) 与 B(E, N, K) 的分块乘法得到 C(M, topk, N)；B 是堆叠的全部专家权重，`expert_ids` 决定每个 block 使用哪个专家矩阵。
- 来源定位：fused_moe.py L356–375 内核 docstring："A: The input tensor representing tokens with shape (*, K)… B: The stacked MOE weight tensor with shape (E, N, K)… C: The output cache tensor with shape (M, topk, N)…"；"expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A."
- 适用条件：v0.28.0 未量化与量化共用该内核框架。
- 置信状态：已确认。

### C3 三个对齐数组的语义
- 论断：moe_align_block_size 返回 sorted_token_ids（token 槽位索引按所属专家排序、每段填充到 block_size 的倍数）、expert_ids（每个 block 的专家索引）、num_tokens_post_padded（填充后总槽位数）；填充槽的值为 num_valid_tokens（即 topk_ids.numel()），内核用 `token_mask = offs_token < num_valid_tokens` 跳过。
- 来源定位：moe_align_block_size.py L48–72（Returns 与 Example 段，例中 token 12 为填充、"Tokens 12 are non-existent (padding) and are ignored"）；fused_moe.py L421（token_mask 定义）。
- 适用条件：v0.28.0；极小批量走跳过对齐路径（C14）时三个数组退化为直通形式。
- 置信状态：已确认。

### C4 排序与填充的目的
- 论断：排序加填充保证每个 BLOCK_SIZE_M tile 内的行属于同一专家，这是分块矩阵乘在各专家的 block 间保持一致性的必要条件。
- 来源定位：fused_moe.py L379–383："The sorting of sorted_token_ids by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert."；moe_align_block_size.py L54–57："This function pads the number of tokens that each expert needs to process so that it is divisible by block_size."
- 适用条件：同 C3。
- 置信状态：已确认。

### C5 fused_experts 的五步流程
- 论断：fused_experts_impl 依次执行：准备专家分配（moe_align_block_size）→ 第一次 dispatch_fused_moe_kernel（w1 → intermediate_cache1）→ apply_moe_activation（默认 MoEActivation.SILU，即 silu_and_mul：输入 (·, 2d) 输出 (·, d)）→ 第二次 dispatch_fused_moe_kernel（w2 → intermediate_cache3）→ ops.moe_sum 写出 out_hidden_states。
- 来源定位：fused_moe.py L1772–1783（对齐调用）、L1785–1807（第一次 GEMM）、L1809–1811（激活）、L1824–1846（第二次 GEMM）、L1848–1851（moe_sum）；activation.py L22–24（"Gated activations (gate * activation(up)) expect input of shape [..., 2*d] and produce output of shape [..., d]"）、L230–236（SILU → torch.ops._C.silu_and_mul）。
- 适用条件：未量化路径；量化时在两次 GEMM 前后插入量化输入步骤（L1764–1770、L1813–1819），流程骨架不变。
- 置信状态：已确认。

### C6 权重作用位置与求和
- 论断：默认（apply_router_weight_on_input=False）时第一次 GEMM 不乘路由权重（mul_routed_weight=False）、第二次乘（mul_routed_weight=True，MUL_ROUTED_WEIGHT 分支在 fp32 中乘 topk_weights[offs_token]），第二次 GEMM 的 top_k 参数传 1；moe_sum 将 (M, topk, K) 的中间结果沿 topk 维相加得到 (M, K) 输出。
- 来源定位：fused_moe.py L1785–1807（第一次调用传 `apply_router_weight_on_input`、`top_k_num`）、L1824–1846（第二次传 `not apply_router_weight_on_input`、`1`）、L593–599（"Router (MoE) weight multiplication: This multiplication MUST be performed in float32…"）、L1848–1851（ops.moe_sum(intermediate_cache3.view(...), out_hidden_states)，cache3 形状 (M, topk, K)、out 形状同 hidden_states）。
- 适用条件：默认参数；apply_router_weight_on_input=True 的模型变体把权重乘在输入侧。
- 置信状态：已确认（另有本页可运行代码对合成公式做等价验证）。

### C7 网格划分与 L2 分组
- 论断：内核网格为 cdiv(EM, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)（一维展平）；pid 到 (pid_m, pid_n) 的映射按 GROUP_SIZE_M 分组排序以促进 L2 复用。
- 来源定位：fused_moe.py L836–839（grid 定义）、L385–396（"This is done in a grouped ordering to promote L2 data reuse."）。
- 适用条件：v0.28.0。
- 置信状态：已确认。

### C8 A 的行还原与 B 的专家偏移
- 论断：GEMM1 中 A 的行地址由 `offs_token // top_k * stride_am` 得到（把槽位索引还原为 token 行），B 的地址由 `off_experts * stride_be` 偏移到对应专家；GEMM2 中 A 换成 (M×topk, inter) 的中间缓存、top_k 传 1，槽位索引即行号。
- 来源定位：fused_moe.py L472–479（a_ptrs/b_ptrs 指针运算）、L1740–1741 与 L1747–1751（cache1/cache2/cache3 形状）。
- 适用条件：非 SWAP_AB、非 USE_TD 的默认指针路径。
- 置信状态：已确认。

### C9 越界与无效块的处理
- 论断：`pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` 的 program 直接返回；填充槽通过 token_mask 屏蔽读入（other=0.0）与写回。
- 来源定位：fused_moe.py L404–407（提前返回）、L531–537（带 mask 的 tl.load）、L607–610（带 mask 的写回）。
- 适用条件：v0.28.0。
- 置信状态：已确认。

### C10 EP 下非本 rank 专家写零
- 论断：专家不在当前 EP rank 时 expert_ids 标为 -1，内核对这类 block 调 write_zeros_to_output 写零输出；对齐函数先用全局专家数对齐、返回前把非本 rank 专家标记为 -1。
- 来源定位：fused_moe.py L423–440（"Write back zeros to the output when the expert is not in the current expert parallel rank."）；moe_align_block_size.py L23–27（"In the case of expert_parallel, moe_align_block_size initially considers all experts as valid… Before the function returns it marks the experts_ids that are not in the current GPU rank as -1"）。
- 适用条件：EP 启用且 expert_map 存在。
- 置信状态：已确认。

### C11 调优配置的查找
- 论断：内核配置（BLOCK_SIZE_M/N/K 等）按 (E, N, device, dtype, block_shape) 命名规则查 JSON 文件，找到则按最接近的批量档位取配置，找不到用 get_default_config 启发式并告警；调优文件也可来自用户目录 VLLM_TUNED_CONFIG_FOLDER。
- 来源定位：fused_moe.py L1089–1100（get_config_file_name 命名规则）、L1104–1166（get_moe_configs docstring："The return value will be a dictionary that maps an irregular grid of batch sizes to configurations of the fused_moe kernel. To evaluate the kernel on a given batch size bs, the closest batch size in the grid should be picked"）、L1159–1166（默认配置告警）。
- 适用条件：v0.28.0。
- 置信状态：已确认。

### C12 层的组成与 TP 切分
- 论断：v0.10.2 中模型使用的层类为 `FusedMoE(CustomOp)`；v0.28.0 中由 FusedMoEFactory 组装 Router（token→专家分配）+ RoutedExperts（专家权重，含 MergedColumnParallel 的 gate_up_proj/w13 与 RowParallelLinear 的 down_proj/w2）+ MoERunner（编排前向）；命名沿用 Mixtral 的 w1=gate、w2=down、w3=up 约定。
- 来源定位：v0.10.2 layer.py L741（class FusedMoE(CustomOp)）；v0.28.0 layer.py L149–161（docstring："Creates and configures a complete MoE execution pipeline including: Router… RoutedExperts… MoERunner… The experts contain both MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)."、"Mixtral uses w1, w2, and w3 for gate, up, and down_proj."）。
- 适用条件：版本如标注。
- 置信状态：已确认。

### C13 路由融合内核
- 论断：gating 的 softmax 与 top-k 选择由融合 CUDA 算子 topk_softmax 完成（输入 gating_output [num_tokens, num_experts]，输出 topk_weights 与 topk_indices），该内核改编自 TensorRT-LLM v0.7.1 的 moe_kernels.cu。
- 来源定位：v0.28.0 router/fused_topk_router.py L31–44（vllm_topk_softmax 调 ops.topk_softmax）；v0.10.2 csrc/moe/topk_softmax_kernels.cu L1–2（"Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu"）、L531–560（张量形状注释）。
- 适用条件：CUDA 平台默认路径；ROCm 走 aiter 变体。
- 置信状态：已确认。

### C14 极小批量的跳过对齐路径
- 论断：当 num_tokens × top_k × 4 ≤ global_num_experts（SPARSITY_FACTOR=4 启发式）且无 expert_map 时，跳过 moe_align_block_size，sorted_token_ids 传 None，内核走 naive_block_assignment（每个 block 只放第一个有效槽位、其余置为 num_valid_tokens 屏蔽）。
- 来源定位：fused_moe.py L1552–1558（注释与条件）、L1566–1576（返回 None/直通 expert_ids）、L408–416 与 L898（naive_block_assignment 分支）。
- 适用条件：v0.28.0；量化 wna16 路径除外（L1559–1563）。
- 置信状态：已确认。

### C15 cache13 内存复用
- 论断：intermediate_cache1 与 intermediate_cache3 共享同一块缓冲（按 max(N, K) 分配），因为用到 cache3 时 cache1 已消费完毕。
- 来源定位：fused_moe.py L1733–1741（注释 "We can reuse the memory between these because by the time we need cache3, we're done with cache1"）。
- 适用条件：v0.28.0。
- 置信状态：已确认。

### C16 MegaBlocks 思想先例
- 论断：MegaBlocks（arXiv:2211.15841，Gale 等，2022）指出当时的 MoE 框架为满足软硬件约束而限制动态路由，迫使在"丢弃 token"与"为填充浪费计算和内存"之间权衡；其方案是把 MoE 计算重构为块稀疏运算并开发块稀疏 GPU 内核，从不丢弃 token。
- 来源定位：arXiv:2211.15841 摘要（"These formulations force a tradeoff between model quality and hardware efficiency, as users must choose between dropping tokens from the computation or wasting computation and memory on padding… we reformulate MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels… Our approach never drops tokens"）。
- 适用条件：论文针对训练场景；对推理侧的"组织专家计算"思路同样成立，但不构成 vLLM 内核出自该代码库的证据。
- 置信状态：已确认（仅用于动机与对照，不作为 vLLM 出处声明）。

## F 公式

### F1 填充槽位上界
- 公式：max_num_tokens_padded = topk_ids.numel() + num_experts × (block_size − 1)。
- 来源定位：moe_align_block_size.py L74。
- 适用条件：pad_sorted_ids=False 的默认分支。
- 置信状态：已确认。

### F2 tile 索引公式
- 公式：对第 b 个 M-block（专家 e = expert_ids[b]，槽位 j ∈ [b·BM, (b+1)·BM)），输出元素
  C[sorted_token_ids[j], n] = Σ_{k=0}^{K−1} A[sorted_token_ids[j] // top_k, k] · B[e, n, k]，
  其中 GEMM1 的 A 为 hidden_states、top_k 为路由 top-k；GEMM2 的 A 为 (M×topk, inter) 中间缓存、top_k = 1。
- 来源定位：推导自 fused_moe.py L408–479（指针运算：offs_token 加载、`// top_k` 还原行、off_experts 偏移、K 维累加循环）与 L1740–1751（缓存形状）；C2/C8。
- 适用条件：默认指针路径；掩码槽位（offs_token ≥ num_valid_tokens）的读为 0、写被屏蔽。
- 置信状态：已确认（另有本页可运行代码逐元素验证）。

### F3 token 输出合成公式
- 公式：y_t = Σ_{k=1}^{topk} g_{t,k} · W2_{e_{t,k}} · act(W13_{e_{t,k}} · x_t)，act 默认为 silu 门控（gate 半与 up 半逐元素相乘）。
- 来源定位：由 C5（两次 GEMM 与激活流程）、C6（权重在 GEMM2 乘、沿 topk 求和）、activation.py L22–24（gate×activation(up)，2d→d）组合。
- 适用条件：apply_router_weight_on_input=False 的默认路径；renormalize 等路由细节不影响该合成式。
- 置信状态：已确认（可运行代码对照朴素实现验证等价）。

## N 数字

### N1 默认 BLOCK_SIZE_M 阶梯
- 数值：bf16/fp16 通用默认路径中 M≤32→16、M≤96→32、M≤512→64、否则 128（block_n：M≤64→64 否则 128）。
- 来源定位：fused_moe.py L1366–1380（get_default_config 的通用分支）。
- 实验条件：无调优 JSON 命中时的启发式；fp8/int4 路径另有阶梯。
- 置信状态：已确认。

### N2 版本标识
- 数值：vLLM v0.28.0，commit 2cf0a6915ce544dc493a0990f2ea38d81601128a；layer 类对照版本 v0.10.2。
- 来源定位：git ls-remote refs/tags/v0.28.0；GitHub raw 文件按 tag 获取。
- 置信状态：已确认。

### N3 跳过对齐阈值
- 数值：num_tokens × top_k × 4 ≤ global_num_experts 时跳过对齐（SPARSITY_FACTOR = 4）。
- 来源定位：fused_moe.py L1552–1558。
- 实验条件：expert_map 为 None 且非 wna16 量化。
- 置信状态：已确认。

### N4 MegaBlocks 加速数字
- 数值：训练端到端加速最高 40%（对比 Tutel 训练的 MoE）、2.4×（对比 Megatron-LM 稠密 DNN）。
- 来源定位：arXiv:2211.15841 摘要。
- 实验条件：论文训练场景，非 vLLM 推理路径。
- 置信状态：已确认（仅作历史动机旁证）。
