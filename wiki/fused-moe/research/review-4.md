<!-- review-meta
round: 4
page: wiki/fused-moe/index.html
reviewed_content_sha256: 0910b1a3257eff03
-->
# FusedMoE 审查记录（第 4 轮）

- 页面版本：index.html 工作树 blob cf6686ad8073d560ebafbfe3fa06f7512d637c34（仓库 HEAD be5c3c8）
- 审查时间：2026-09-14 16:50–16:58
- 审查者：独立子代理（未参与写作，未读取本页 research/ 下任何文件）
- 已完整阅读章节：引言 → 核心问题 → 常见误解 → 1. 朴素实现的问题 → 2. 按专家分块对齐 → 3. 一次内核调用算完所有专家 → 4. 一个 token 的完整路径 → 5. FusedMoE 层的职责边界 → 来源与范围说明（含全部 details 折叠块、两个图注、SVG 内逐标签读数、可运行代码）
- 来源获取方式：raw.githubusercontent.com 拉取 vLLM commit 2cf0a6915ce544dc493a0990f2ea38d81601128a 与 v0.10.2 的 fused_moe.py / moe_align_block_size.py / activation.py / layer.py / router/fused_topk_router.py / csrc 下 topk_softmax_kernels.cu、moe_align_sum_kernels.cu；MegaBlocks 用 WebFetch 抓 arXiv:2211.15841 摘要；页面 Python 代码抽出后本地实跑。

## 来源逐条核对（本轮已核对的引文依据，未列入问题的部分均为通过）

- C1–C2：fused_moe.py L3 `"""Fused MoE Triton kernels."""`、L298–299 `@triton.jit\ndef fused_moe_kernel(`、L356–375 docstring `- A: ... shape (*, K)`、`- B: The stacked MOE weight tensor with shape (E, N, K)`、`- C: ... shape (M, topk, N)` —— 与页面「A 形状 (*,K)、B (E,N,K)、输出 C (M,k,N)」一致。
- C3–C4：moe_align_block_size.py L48–72（Returns 与 Example：例中填充值 12 = `topk_ids.numel()`，与页面「填充槽的值等于有效槽位数」一致）、L54–57 `This function pads the number of tokens that each expert needs to process so that it is divisible by block_size`；fused_moe.py L379–383 `The sorting of sorted_token_ids by expert index and padding ensures divisibility by BLOCK_SIZE_M`。
- C5：fused_moe.py L1772–1851 五步（`_prepare_expert_assignment` → `dispatch_fused_moe_kernel`(w1) → `apply_moe_activation` → `dispatch_fused_moe_kernel`(w2) → `ops.moe_sum`）；activation.py L22–23 `Gated activations (gate * activation(up)) expect input of shape [..., 2*d] and produce output of shape [..., d]`、L230–236 `torch.ops._C.silu_and_mul(output, input)`。
- C6：fused_moe.py L1785–1807（第一次调用 `apply_router_weight_on_input`，top_k=top_k_num）与 L1824–1846（第二次调用 `not apply_router_weight_on_input`，top_k=1）；L589–599 `This multiplication MUST be performed in float32` + `accumulator *= moe_weight[:, None]`；L1848–1851 `ops.moe_sum`。
- C7–C9：L836–839 grid = `triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(B.size(1), BLOCK_SIZE_N)`；L824–835 `EM = sorted_token_ids.size(0)`（小批量再取 min）；L385–396 分组 pid 映射与注释 `This is done in a grouped ordering to promote L2 data reuse`；L404–410 `offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)`；L421 `token_mask = offs_token < num_valid_tokens`；L472–479 `offs_token[:, None] // top_k * stride_am` 与 `off_experts * stride_be`；L531–537 读掩码 `other=0.0`；L607–610 写掩码 `tl.store(c_ptrs, accumulator, mask=c_mask)`。
- C10：L423–440 `if off_experts == -1: write_zeros_to_output(...)`；moe_align_block_size.py L23–27 `Before the function returns it marks the experts_ids that are not in the current GPU rank as -1 so the MoE matmuls could skip those blocks.`
- C11：L1089–1100 文件名含 E、N、device_name、dtype；L1115–1118 `the closest batch size in the grid should be picked`；L1159–1166 `logger.warning_once("Using default MoE config. ...")`。
- C12：v0.28.0 layer.py L99 `def FusedMoEFactory(`、L149–161 docstring（`- Router (for token-to-expert assignment)`、`- RoutedExperts (containing expert weight parameters)`、`- MoERunner (orchestrates the complete forward pass)`、`The experts contain both MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)`、`Mixtral uses w1, w2, and w3 for gate, up, and down_proj`）；v0.10.2 layer.py L740–741 `@CustomOp.register("fused_moe")` / `class FusedMoE(CustomOp):`。
- C13：v0.28.0 router/fused_topk_router.py L26–42 `def vllm_topk_softmax(...)`；csrc 文件头 L1 `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/...`。行号 L531–560 见下方问题 3。
- C14/N3：L1552–1558 `naive_block_assignment = (expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and not (...))`，`SPARSITY_FACTOR is a heuristic margin`；L1566–1576 返回 `(None, topk_ids.view(-1), torch.full((1,), topk_ids.numel() * config["BLOCK_SIZE_M"], ...))`；L834–835 `EM = num_tokens * config["BLOCK_SIZE_M"]`；kernel L408–416 `offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)` —— 与页面「sorted_token_ids 传空、每个块只装一个有效槽位，其余位置屏蔽」一致。
- C15：L1733–1741 `We can reuse the memory between these because by the time we need cache3, we're done with cache1` / `intermediate_cache1 = cache13[: M * top_k_num * N]...`、`intermediate_cache3 = cache13[: M * top_k_num * K]...`。
- C16/N4：arXiv:2211.15841 摘要 `reframe MoE computation in terms of block-sparse operations`、`never drops tokens`、`up to 40% versus MoEs trained with the state-of-the-art Tutel library, and 2.4x versus DNNs trained with Megatron-LM`、`they either drop tokens or waste compute and memory on padding` —— 与页面第 1 章末段及 N4 一致（页面已标注为训练场景、仅作历史动机旁证）。
- F1：moe_align_block_size.py L74 `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`；L77–80 `if topk_ids.numel() < num_experts: max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)` —— 与页面「槽位总数小于专家数时截断到 M·k×B_M 与上界较小值」一致。
- F2：由 L472–479 指针式与 L1740–1751 缓存形状推出，页面公式 $C[s,n]=\sum_q A[\lfloor s/k\rfloor,q]\cdot B[e,n,q]$ 与 `offs_token // top_k`、`off_experts * stride_be` 一致。
- F3：与 L1785–1851 两次调用 + silu_and_mul 语义一致。
- N1：fused_moe.py L1366–1380 `if M <= 32: block_m = 16 elif M <= 96: 32 elif M <= 512: 64 else: 128`（bf16/fp16 通用默认分支）—— 与页面「16 到 128 随批量增大」一致。
- N2：commit 2cf0a6915ce544dc493a0990f2ea38d81601128a（上述文件均按该 commit 拉取成功）。
- 构造示例与手算：抽出页面 `<code class="language-python">` 实跑，输出与页面「预期输出」逐字节一致（`sorted_token_ids = [0, 4, 1, 2, 6, 8, 7, 8, 3, 5] (分配长度 12)`、`expert_ids = [0, 1, 1, 2, 3]`、`num_tokens_post_padded = 10`、8 行 cache1/cache2/cache3、`y_t0=[1.3494, 0.0]`…`y_t3=[2.8186, 0.7046]`、末行 `两条路径输出逐元素一致（误差 < 1e-9）`）。页面第 3 章块 1 手算（C[s1]=(1,0,1,1)、C[s2]=(0,1,1,0)）与第 4 章 t_0 手算（1.7616 / 0.7311 / 1.0570 / 0.2924 / 1.3494）、t_2 合并（5.0578, 0.9504）均复算通过。
- 图注读数：SVG 逐标签核对——槽位序列 0,4,1,2,6,8,7,8,3,5、块序号 0–4 与「—」、专家行 0,1,1,2,3 与「未读取」、有效长度线 120→616 对应 10 个槽、蓝框两处标「填充」值为 8；与正文数组、与 expert_ids 完全一致。第 4 章流程图为 HTML 节点（非 SVG），节点内 $...$ 由 KaTeX 渲染，无 ASCII 近似。
- 链接与功能：moe-serving / gpu-execution-model / swiglu / model-parallelism / eplb 五个前置页均真实存在；overview.html 与 index.html 互链；`python3 .dojo/scripts/validate.py wiki/fused-moe/index.html` 返回 `validation ok`；`dojo:type=concept`、`dojo:topics=推理系统`、`dojo:tag=MoE`、description 与 summary 齐备；summary 内 `$...$` 仅用 \cdot \sum \operatorname \odot \big 等 KaTeX 支持的命令；全页 Unicode 数学字符扫描：`×` 仅出现在两处代码块内（pseudocode），正文/summary/表格/图注无 Unicode 数学符号；无 alt 含 `$...$` 的图片。

## 问题

- [重要·技术] 第 1 章「第二个来源：调度边界随专家数线性增多」（index.html L147）：把朴素循环的内核启动次数写成只由 $E$ 决定，「$E$ 为 256 时一个 MoE 层一次前向就要发起上千次调用」，与本页引言给出的同一 $E$ 场景数字「64 次计算会变成上百个……小内核调用」（L65）以及本节自己给出的伪代码「若 rows 为空：继续下一个专家」（L130–139）都不一致：按页内伪代码，空专家不产生任何启动，启动数由**非空专家数**（≤ E）与批量共同决定。｜引文依据：页面 L65「$E=256$、批 8、top-8（64 槽）→ 上百个」；L147「次数随 $E$ 线性增长；$E$ 为 256 时……上千次调用」；L128–139「若 rows 为空：继续下一个专家」；L76/L161 同句复述。按此伪代码，64 槽落在 256 个专家上仅约 57 个非空专家、每次约 5 次启动 ≈ 285 次（与 L65 的「上百」一致），要到「上千」需批量大到几乎所有专家都分到槽位（如 prefill）。｜修复要求：把三处「次数随 $E$ 线性增长」改为随非空专家数增长（上界为 $E$），并为「上千次」补上成立条件（全部专家都分到槽位的批量），或删去该数字；修后须与 L65 的「上百个」在 $E=256$/批 8 场景下自洽。｜修复：｜复验：

- [轻微·技术] 「来源与范围说明 › 核心论断与来源」C13（index.html L715）：「CUDA 内核在 v0.28.0 位于 csrc/libtorch_stable/moe/topk_softmax_kernels.cu（v0.10.2 位于 csrc/moe/topk_softmax_kernels.cu），文件头 L1–2 …… 、L531–560（张量形状注释）」中的 L531–560 指向的其实是 v0.10.2 的文件。｜引文依据：v0.28.0 csrc/libtorch_stable/moe/topk_softmax_kernels.cu L531–543 为 `VLLM_SHFL_XOR_SYNC_WIDTH` 蝶形 argmax 归约代码，该文件的张量形状注释在 L826 `// [num_tokens, num_experts]`、L864–866 `// [num_tokens, topk]`；v0.10.2 csrc/moe/topk_softmax_kernels.cu L531–535 恰为 `torch::Tensor& topk_weights, // [num_tokens, topk]` … `// [num_tokens, num_experts]`。｜修复要求：把行号归属写明为 v0.10.2，或改用 v0.28.0 的 L826、L864–866。｜修复：｜复验：

- [轻微·格式] 「来源与范围说明」下两个 h3 命名（index.html L705、L721）：使用了「核心论断与来源」「核心公式与来源」。｜引文依据：不适用（规范条款见 guides/concept/style-guide.md §1「来源章节……下的 h3 使用固定命名、同样不编号：`论断与来源（C）`、`公式与来源（F）`……」；全站 75 页使用固定名，仅 4 页用「核心……」变体）。｜修复要求：改为固定的「论断与来源（C）」「公式与来源（F）」，并同步页内对这两节的自指表述（若有）。｜修复：｜复验：

- [轻微·表述] 第 3 章（index.html L304）：「请注意 $EM$ 是分配长度而非 num_tokens_post_padded：……」——面向读者的提醒式元话语。｜引文依据：不适用（guides/concept/check.md §2.12 把「需要注意的是」一类元话语列为须排除的表述；该措辞在本仓库约 95 个页面中仅此一处）。｜修复要求：改为直接陈述，例如「$EM$ 取 sorted_token_ids 的分配长度、不是 num_tokens_post_padded——构造示例 A 中前者 12、后者 10，网格据此覆盖 6 个 M 块……」，不改变其后的数值与结论。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（1 条重要 + 3 条轻微；无阻断，无返回规划事项）

统计：阻断 0 / 重要 1 / 轻微 3