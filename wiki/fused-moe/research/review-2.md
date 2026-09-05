# FusedMoE 审查记录（第 2 轮）

- 页面版本：index.html 工作树哈希 `3dd1704797f8a3e9780ce3e247f4f25d46b64acd`（overview.html `11a0a590f1c9789739f7fa3adaa7fd555e5f8386`）
- 审查时间：2026-09-03 14:41 CST
- 审查者：第 2 轮独立审查者（未参与写作、未参与第 1 轮审查与修复；按规范未读取 `research/` 下任何前序记录，本轮发现无法与第 1 轮条目对号，均按独立观察记录）
- 已完整阅读章节（按顺序）：index.html——标题与导语、核心问题（5 问及全部解答折叠块）、常见误解、第 1 章「朴素实现的问题」及本章问题、第 2 章「按专家分块对齐」及本章问题、第 3 章「一次内核调用算完所有专家」及本章问题、第 4 章「一个 token 的完整路径」及本章问题（含全部折叠块：伪代码、补充推导、示例 A 全量表、可运行代码与预期输出）、第 5 章「FusedMoE 层的职责边界」及本章问题、来源与范围说明；overview.html 全文。

## 核对方法与总体结果

- 外部来源全部重新获取：vLLM v0.28.0（tag 经 `git ls-remote` 验证指向 `2cf0a6915ce544dc493a0990f2ea38d81601128a`）的 fused_moe.py、moe_align_block_size.py、layer.py、activation.py、router/fused_topk_router.py，对照 v0.10.2 的 layer.py 与 csrc/moe/topk_softmax_kernels.cu；另按 4 步核对需要补取了 v0.28.0 的 `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`、`experts/triton_moe.py`、`modular_kernel.py` 等公开源码用于交叉验证 C10/C6；MegaBlocks 摘要取自 arXiv:2211.15841。
- 全部 23 条来源论断（C1–C16、F1–F3、N1–N4）逐条打开到页面标注行号核对，引文依据见下节，全部支持页面表述。
- 页面可运行代码：提取、unescape 后以 Python 实跑，退出码 0（含等价性 assert），实跑输出与页面「预期输出」经 `difflib.unified_diff` 比对**逐行一致**。
- `.dojo/scripts/validate.py`：index.html 与 overview.html 均返回 `validation ok`（含 BARE_MATH_RE、SVG text 数学、ASCII 近似、锚点、本地链接等检查）。
- 结构规范：核心问题 5 问、各章本章问题 2/3/3/2/3 问，两级均带「解答」折叠块；答案独立可读、与正文一致，核心问题答案均指明论证所在章节。overview.html 与 index.html 相互链接。5 个前置概念链接（moe-serving、gpu-execution-model、swiglu、model-parallelism、eplb）均指向已存在的完整页面。
- 手算复核算例：示例 A 三数组（`[0,4,1,2,6,8,7,8,3,5]`/`[0,1,1,2,3]`/10）、分配长度 12、块 1 手算、$t_0$ 与 $t_2$ 全路径、SVG 图中块/专家/填充槽标注均复算一致。

### 逐条核对依据（C/F/N）

| 编号 | 核对位置与原文/关键数值 |
| --- | --- |
| C1 | fused_moe.py L3 `"""Fused MoE Triton kernels."""`；L298–299 `@triton.jit / def fused_moe_kernel(` |
| C2 | fused_moe.py L361–369 docstring："A: …shape (\*, K)"、"B: The stacked MOE weight tensor with shape (E, N, K)"、"C: The output cache tensor with shape (M, topk, N)" |
| C3 | moe_align_block_size.py L48–52 Returns 三数组；L59–70 Example：12 个有效 token、填充值 12、"Tokens 12 are non-existent (padding)"；fused_moe.py L421 `token_mask = offs_token < num_valid_tokens`；CUDA 侧 `sorted_token_ids` 以 `numel` 初始化 |
| C4 | fused_moe.py L379–383 "The sorting of `sorted_token_ids` by expert index and padding ensures divisibility by BLOCK_SIZE_M" |
| C5 | fused_moe.py L1772（对齐）→ L1785（GEMM1）→ L1809（激活）→ L1824（GEMM2）→ L1848（`ops.moe_sum`）；activation.py L22–24 "Gated activations (gate \* activation(up)) expect input of shape [..., 2\*d] and produce output of shape [..., d]"、L230–236 `torch.ops._C.silu_and_mul(output, input)` |
| C6 | L1796/L1835 两次 dispatch 分别传 `apply_router_weight_on_input` 与 `not apply_router_weight_on_input`（"两栏对调"）；L589–599 "This multiplication MUST be performed in float32"；L1848–1851 moe_sum；modular_kernel.py L284 "When True, apply the weights to the activations, before quantization + dispatching"（支持"乘到输入侧"表述） |
| C7 | L836–839 `grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)`；L386–387 "grouped ordering to promote L2 data reuse"；L1387 "Grouping adjacent M-blocks lets them share weight tiles in L2" |
| C8 | L409–410 `offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)`；L423 `off_experts = tl.load(expert_ids_ptr + pid_m)`；L472–478 `a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + …)`、`b_ptrs = b_ptr + off_experts * stride_be + …` |
| C9 | L405–407 `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return`；L421 token_mask；L531–537 读掩码 `other=0.0`；L607–610 `c_mask = token_mask[:, None] & (offs_cn[None, :] < N)` |
| C10 | fused_moe.py L423–440 `if off_experts == -1: write_zeros_to_output(…)`；moe_align_block_size.py L23–27 "…marks the experts_ids that are not in the current GPU rank as -1…"；交叉验证：默认 Triton 后端 experts/triton_moe.py L686 以默认 `ignore_invalid_experts=False` 调用对齐（先全量对齐、再经 expert_map 映射出 -1），与页面描述一致 |
| C11 | L1089–1100 `get_config_file_name(E, N, dtype, block_shape)` → `"E={E},N={N},device_name=…"`；L1115–1118 "the closest batch size in the grid should be picked"；L1159–1166 "Using default MoE config. Performance might be sub-optimal!" |
| C12 | v0.28.0 layer.py L149–161 docstring："Router (for token-to-expert assignment) / RoutedExperts (containing expert weight parameters) / MoERunner (orchestrates the complete forward pass)"、"MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)"、"Mixtral uses w1, w2, and w3…"；v0.10.2 layer.py L740–741 `@CustomOp.register("fused_moe") / class FusedMoE(CustomOp)` |
| C13 | v0.28.0 fused_topk_router.py L31–44 `ops.topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, …)`；v0.10.2 .cu L1–2 "Adapt from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/…"、L531–560 `topk_softmax` → `topkGatingSoftmaxKernelLauncher`（2 的幂专家数时单内核融合 softmax+topK） |
| C14 | L1552–1558 "SPARSITY_FACTOR…Skips moe_align_block_size and activates the `sorted_token_ids is None` path"、`naive_block_assignment = (expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and …)`；内核 L411–416 naive 路径每块仅首个元素有效 |
| C15 | L1733–1741 "We can reuse the memory between these because by the time we need cache3, we're done with cache1" |
| C16 | 摘要："users must choose between dropping tokens from the computation or wasting computation and memory on padding"、"Our approach never drops tokens" |
| F1 | moe_align_block_size.py L74 `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`（默认分支；示例 8+4×1=12 与页面一致） |
| F2 | 由 L408–479 指针运算与 L1740–1741 缓存形状（(M, top\_k, N)/(M, top\_k, K)）推导成立；正文代码实跑逐元素验证通过 |
| F3 | 合成公式由 C5、C6 与 activation.py 门控定义组合；代码实跑 naive 与 fused 输出逐元素一致（误差 <1e-9） |
| N1 | L1366–1378 bf16/fp16 通用默认：M≤32→16、M≤96→32、M≤512→64、否则 128 |
| N2 | `git ls-remote`：`refs/tags/v0.28.0 → 2cf0a6915ce544dc493a0990f2ea38d81601128a`；v0.10.2 → `01efc7ef…` |
| N3 | L1552–1558 `num_tokens * top_k_num * 4 <= global_num_experts`（SPARSITY_FACTOR=4） |
| N4 | 摘要："enabling end-to-end training speedups of up to 40% over MoEs trained with the state-of-the-art Tutel library and 2.4x over DNNs trained with the highly-optimized Megatron-LM framework" |

## 问题

- [重要·技术] index.html §3 首段（"网格沿两个维度展开：$\lceil EM/B_M\rceil\times\lceil N/B_N\rceil$，其中 $EM$ 是排序填充后的槽位总数"）及本章问题 Q1 解答（"$EM$ 是填充后的槽位总数"）｜EM 的定义与源码不符：源码中 `EM = sorted_token_ids.size(0)`，即**分配长度**（按 F1 上界预留，示例 A 为 12），而非"填充后的槽位总数"（页面在第 2 章将 num_tokens_post_padded 定义为"填充后的有效总槽位数"，示例 10）。按页面定义计算网格得 $\lceil 10/2\rceil=5$ 块，与同章下文"分配长度是 12、有效长度是 10，网格会覆盖 6 个块，第 5 号块……启动后立即退出"直接矛盾，读者无法解释第 5 号块的存在｜引文依据：fused_moe.py L824–825 `if sorted_token_ids is not None: EM = sorted_token_ids.size(0)`（sorted_token_ids 由 moe_align_block_size.py L81–83 按 `max_num_tokens_padded` 分配，示例 A 为 12）；页面 L1012 自述"网格会覆盖 6 个块"｜修复要求：将两处 EM 定义改为"sorted_token_ids 的分配长度（按 F1 上界预留、不小于有效长度，示例 A 为 12）"，并可与 L1012 的"分配长度 12/有效长度 10"表述呼应；无需改动其余内容｜修复：｜复验：
- [轻微·可读性] index.html 导语第 1 段（"再用一次分块 GEMM 内核调用算完全部专家的份额——对每一份权重各调用一次，调用次数不再随专家数增长"）｜"每一份权重"指代不明：紧邻上一句用"份"计量专家权重（"散布在 256 份专家权重上"），读者会把"对每一份权重各调用一次"读成"每个专家的权重各一次"，与"调用次数不再随专家数增长"自相矛盾；实际含义是 w13 与 w2 两份权重矩阵各一次（共两次 GEMM 调用）｜引文依据：不适用（可读性）；语义佐证：fused_moe.py L1785/L1824 两次 dispatch_fused_moe_kernel 分别以 w1、w2 为 B 矩阵｜修复要求：将该短语改为"对 w13 与 w2 两份权重各调用一次"或删去，使指代唯一｜修复：｜复验：
- [轻微·技术] index.html §5 正文（"当 $M\cdot k\times 4$ 不超过全局专家数时……vLLM 跳过 moe_align_block_size"）及本章问题 Q3 解答｜跳过对齐的条件缺"未启用专家并行"：源码要求 `expert_map is None` 才走该路径，EP 模式下（大专家数模型的常见部署方式）decode 单请求并不会跳过对齐；overview.html 已正确写为"未启用专家并行且 $M\cdot k\times 4\le E$"，两页不一致｜引文依据：fused_moe.py L1556–1558 `naive_block_assignment = (expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and …)`；overview.html L66"未启用专家并行且 $M\cdot k\times 4\le E$"｜修复要求：在 index.html §5 正文与本章问题 Q3 解答两处补上"未启用专家并行"条件，与 overview 对齐｜修复：｜复验：
- [轻微·来源] index.html「来源与范围说明」N4 条目（"MegaBlocks 训练加速最高 40%、2.4 倍"）｜"2.4 倍"未注明对比对象，易被读成也与 Tutel 对比；摘要中 2.4× 的对比对象是 Megatron-LM 训练的稠密 DNN（正文仅使用了 40% 对比 Tutel 的数字，本身无误）｜引文依据：arXiv:2211.15841 摘要 "up to 40% over MoEs trained with the state-of-the-art Tutel library and 2.4x over DNNs trained with the highly-optimized Megatron-LM framework"｜修复要求：N4 条目改为"训练加速最高 40%（对比 Tutel）、2.4 倍（对比 Megatron-LM 稠密 DNN）"｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复后所有 4 条均已关闭。Python 文件读取确认四处修复全部落盘（"EM 取 sorted_token_ids" 出现在 §3 主体与 Q1 解答；"w13 与 w2 两份权重各调用一次"出现在导语；"未启用专家并行（expert_map 为空）"出现在 §5 正文与 Q3 解答；"对比 Tutel 训练的 MoE，2.4 倍对比 Megatron-LM 训练的稠密 DNN"出现在 N4）；validate.py、代码实跑、headless Chrome 渲染三项复检通过；可进入第 3 轮独立审查。
