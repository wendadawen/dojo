<!-- review-meta
round: 5
page: wiki/fused-moe/index.html
reviewed_content_sha256: d9df3a67ec9b326c
-->
# FusedMoE审查记录（第 5 轮）

- 页面版本：`wiki/fused-moe/index.html` 工作树 sha256 `34654a9a930ea860b1a3147a3dd3b605a6c7240700286832cfd8abb7626f3b6e`
- 审查时间：2026-09-14 17:39
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题、常见误解、1. 朴素实现的问题、2. 按专家分块对齐、3. 一次内核调用算完所有专家、4. 一个 token 的完整路径、5. FusedMoE 层的职责边界、来源与范围说明（含全部 details 折叠块与两处图注），并读了 `overview.html`。

## 来源版本核对

- vLLM v0.28.0（commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`）：本轮把 `vllm/model_executor/layers/fused_moe/fused_moe.py`、`moe_align_block_size.py`、`layer.py`、`activation.py`、`router/fused_topk_router.py` 与 `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` 逐一与该仓库 tag `v0.28.0` 的同一路径逐字节比对，全部完全一致——页面标注的版本可作为核对基准。
- 对照版本 v0.10.2：`vllm/model_executor/layers/fused_moe/layer.py` L741 = `class FusedMoE(CustomOp):`；`csrc/moe/topk_softmax_kernels.cu` 存在。
- MegaBlocks：arXiv:2211.15841（Gale、Narayanan、Young、Zaharia）摘要。

## 逐条来源核对（含关键原文/数值）

- C1–C2：`fused_moe.py` L3 = `"""Fused MoE Triton kernels."""`；L298–299 = `@triton.jit` / `def fused_moe_kernel(`；docstring L361 / L364 / L367 = `- A: The input tensor representing tokens with shape (*, K)`、`- B: The stacked MOE weight tensor with shape (E, N, K)`、`- C: The output cache tensor with shape (M, topk, N)`。与正文「$A(*,K)$、$B(E,N,K)$、$C(M,k,N)$，一次调用覆盖全部专家」一致。
- C3–C4：`moe_align_block_size.py` L48–52 Returns「sorted_token_ids … expert_ids … num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size」；L66–68 示例「Then append padding tokens [12, 12, 12, 12] for each block」（12 = `topk_ids.numel()`，即填充值=有效槽位数）；CUDA 侧 `moe_align_sum_kernels.cu` 中 `sorted_token_ids[...] = numel`（注释「Initialize sorted_token_ids with numel」）与 L421 `token_mask = offs_token < num_valid_tokens` 印证「填充槽值=有效槽位数、用掩码识别」；`fused_moe.py` L379–383「The sorting of sorted_token_ids by expert index and padding ensures divisibility by BLOCK_SIZE_M」。
- C5–C6：`fused_moe.py` L1772 `_prepare_expert_assignment(`、L1785/L1824 两次 `dispatch_fused_moe_kernel(`（L1796 传 `apply_router_weight_on_input`，L1835 传 `not apply_router_weight_on_input`）、L1848 `ops.moe_sum(`；L593–599 `if MUL_ROUTED_WEIGHT: … accumulator *= moe_weight[:, None]`；`activation.py` L22–23 注释「Gated activations … expect input of shape [..., 2*d] and produce output of shape [..., d]」、L230–236 `if activation == MoEActivation.SILU: … torch.ops._C.silu_and_mul(output, input)`；`csrc/libtorch_stable/activation_kernels.cu` L303 注释「alpha=1.0, beta=0.0 reduce this to silu(gate) * up」并确认 `silu_and_mul` 对前半（gate）做 silu，故正文「gate 在前、silu(gate)⊙up」成立。
- C7–C9：`fused_moe.py` L836–839 `grid = cdiv(EM, BLOCK_SIZE_M) * cdiv(B.size(1), BLOCK_SIZE_N)`；L385–396 的 GROUP_SIZE_M 分组映射；L405–407 `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return`；L409–410 `offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)`；L423 `off_experts = tl.load(expert_ids_ptr + pid_m)`；L472–473 `offs_token[:, None] // top_k * stride_am`；L477 `off_experts * stride_be`；L531–537 A 载入带 `token_mask`；L607–610 store 带 `c_mask`。
- C10：`fused_moe.py` L423–440 `if off_experts == -1: write_zeros_to_output(...)`，注释「when the expert is not in the current expert parallel rank」；`moe_align_block_size.py` L23–27 注释「Before the function returns it marks the experts_ids that are not in the current GPU rank as -1」。页面表述与该两处一致。
- C11：`fused_moe.py` L1089–1100 配置文件名 `E={E},N={N},device_name=…,dtype=…`；L1141 `config = configs[min(configs.keys(), key=lambda x: abs(x - M))]`（「取最接近批量的档位」）；L1161–1165 `logger.warning_once("Using default MoE config…")`。
- C12：`layer.py` L151–154 docstring「- Router (for token-to-expert assignment) / - RoutedExperts (containing expert weight parameters) / - MoERunner (orchestrates the complete forward pass)」、L156–159「MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)」；v0.10.2 `layer.py` L741 为 `class FusedMoE(CustomOp)`。
- C13：`router/fused_topk_router.py` L31–42（`renormalize: bool = False` … `ops.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, …)` … `return topk_weights, topk_indices`）；`topk_softmax_kernels.cu` L1–2「Adapted from …TensorRT-LLM/blob/v0.7.1/…」，L823–826 形状注释 `[num_tokens, topk]`、`[num_tokens, num_experts]`。
- C14–N3：`fused_moe.py` L1552–1564 `naive_block_assignment = (expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and not (…block_shape[1] > 0))`（L1552 注释点名 SPARSITY_FACTOR），L1566–1576 返回 `(None, topk_ids.view(-1), numel * BLOCK_SIZE_M)`；对应正文「未启用 EP 且 $M\cdot k\times4\le E$ 时跳过对齐、每块只装一个有效槽位」。
- C15：`fused_moe.py` L1733–1734 注释「We can reuse the memory between these because by the time we need cache3, we're done with cache1」+ L1735–1741 cache13 切片。
- C16 / N4：arXiv:2211.15841 摘要「…force a tradeoff between model quality and hardware efficiency: either dropping tokens or wasting computation and memory on padding」「reformulate MoE computation in terms of block-sparse operations」「never drops tokens」「up to 40% … over MoEs trained with the state-of-the-art Tutel library and 2.4x over DNNs trained with … Megatron-LM」。
- F1：`moe_align_block_size.py` L74 `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`；L77–80 `if topk_ids.numel() < num_experts: max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)`——与正文「槽位总数小于专家数时截断到 $M\cdot k\times B_M$ 与上界的较小值」逐字对应。
- F2：`fused_moe.py` L1740–1741 `intermediate_cache1 = cache13[: M*top_k_num*N].view(M, top_k_num, N)` / `intermediate_cache3 = cache13[: M*top_k_num*K].view(M, top_k_num, K)`，配合 L408–479 指针运算。
- N1：`fused_moe.py` L1371–1378 `M<=32→16`、`M<=96→32`、`M<=512→64`、`else 128`（bf16/fp16 通用默认路径，无调优命中时）。
- N2：`v0.28.0` + commit `2cf0a69…`（比对结果见上）；v0.10.2 对照路径均存在。

## 机械验证

- 可运行代码：从页面抽出唯一 Python 代码块（仅 `import math`，无第三方依赖）本地执行，输出与页面「预期输出」块逐行一致：`sorted_token_ids = [0, 4, 1, 2, 6, 8, 7, 8, 3, 5] (分配长度 12)`、`expert_ids = [0, 1, 1, 2, 3]`、`num_tokens_post_padded = 10`、s0–s7 的 cache1/cache2/cache3、`y_t0=[1.3494,0.0] … y_t3=[2.8186,0.7046]`，末行断言通过并打印「两条路径输出逐元素一致（误差 < 1e-9）」。
- 手算复核（独立于代码）：示例 A 的段长 2/3/1/2、补齐后总长 10、`[0,4,1,2,6,8,7,8,3,5]`、`[0,1,1,2,3]`、块 1 的 $C[s_1]=(1,0,1,1)$、$C[s_2]=(0,1,1,0)$、$t_0$ 的 $1.3494$、$t_2$ 的两支路 $0.7\times(5.7154,0.7311)$ 与 $0.3\times(3.5232,1.4621)$ 相加得 $(5.0578,0.9504)$，全部与正文一致。
- 公式：抽出正文与 `dojo:summary` 的全部 394 个 `$…$`/`$$…$$`（先按 HTML 实体解码 `&lt;`/`&gt;`），用本地 `libs/katex.min.js` 以 `throwOnError:true` 渲染，0 例失败；`\operatorname{silu}`、`\odot`、`\big[`/`\Big(`、`\text{…}` 均正常。
- 结构：`.dojo/scripts/validate.py wiki/fused-moe/index.html` 返回 `validation ok`；引用的 `moe-serving`、`gpu-execution-model`、`swiglu`、`model-parallelism`、`eplb` 五个前置页均存在；`overview.html` 与 `index.html` 互链；无越界 Unicode 数学字符，无 `$…$` 出现在 alt；`dojo:topics=推理系统`、`dojo:tag=MoE` 均在词表内。
- 图像读数：图 1 的 12 个框＝分配长度、前 10 个有效（含下标 5、7 两个值为 8 的填充框），下排 5 个块专家号 `0,1,1,2,3` 与第 6 块「未读取」、下括线标注「有效长度 10」与正文三数组读数一致；`.dg-accent` 的 stroke 取 `var(--blue)`，图注称「蓝框」与实际配色相符。

## 问题

- [轻微·表述] 第 4 章折叠块末段（第 459 行）：「两条支路 $0.7\times(5.7154,0.7311)$ 通道经 $w_2[e_0]$（单位阵）直接缩放」｜引文依据：不适用｜问题：「通道」在本页没有定义，句法上读不通，疑为编辑残留；同句另一支路写作「$0.3\times w_2[e_3](1.4621,3.5232)$」先写 $w_2$ 再写缩放系数，两支书写顺序不一致，读者可能误以为「通道」是某个专门术语｜修复要求：删去「通道」，把两支改成同一句式（例如「支路 $s_4$：经 $w_2[e_0]$（单位阵）缩放后乘 0.7；支路 $s_5$：经 $w_2[e_3]$ 交换坐标后乘 0.3」）｜修复：｜复验：
- [轻微·术语] 第 3 章第 324 行：「第 5 号块（槽位 10、11）启动后立即退出」｜引文依据：不适用｜问题：本页把「槽位」定义为 $s=t\cdot k+j$ 且 $0\le s<M\cdot k=8$，此处的 10、11 是 `sorted_token_ids` 数组的下标位置（在分配长度 12 之内），不是槽位编号；与定义冲突，会让人以为存在槽位 10、11｜修复要求：改为「第 5 号块（`sorted_token_ids` 中下标 10、11 两个位置）」或等义表述｜修复：｜复验：
- [轻微·表述] 第 3 章第 304 行末句：「末尾块因超出有效长度被前序检查直接返回」｜引文依据：不适用｜问题：「前序检查」无先行语，指代不明（读者无法判断是哪种检查）｜修复要求：改为「被程序开头的越界检查直接返回」，与第 324 行的「块的起始位置不小于 num_tokens_post_padded 时 program 直接返回」用同一说法｜修复：｜复验：
- [轻微·表述] 第 1、2、3 章末尾的过渡句｜引文依据：不适用｜问题：三章连续以同一句式收尾——「下一章说明 vLLM 如何把这个动态归属预先编码成三个数组」「下一章说明内核如何拿这三个数组……」「下一章把全部步骤串起来，并用构造示例 A 算出真实数值」，构成 style-guide 第 8 节禁止的固定句式｜修复要求：至少改动其中一处，去掉「下一章说明/下一章把……」的引导语，直接以该章留下的问题收束（如第 3 章改为「一个 token 的输出要经过两次 GEMM、一次激活、一次加权求和才最终成形，其数值可用构造示例 A 逐步算出。」）｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布。全部 C1–C16、F1–F3、N1–N4 编号均在本轮核对的 vLLM v0.28.0（commit 2cf0a69，已与 tag `v0.28.0` 逐字节比对）与 MegaBlocks arXiv:2211.15841 摘要中定位到支持片段；代码实际运行输出与页面预期输出逐行一致；394 个公式全部可被 KaTeX 渲染；validate.py 通过。4 条轻微问题只涉及措辞与术语一致性，不影响结论正确性与来源一致性。
