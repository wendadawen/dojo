# FusedMoE 审查记录（第 3 轮）

- 页面版本：index.html `4f45d676b01affbe48ff605e4c569b9b3ce30a08`；overview.html `11a0a590f1c9789739f7fa3adaa7fd555e5f8386`
- 审查时间：2026-09-03 16:47
- 审查者：第 3 轮独立审查者（未参与写作与前两轮审查，未读取 research/ 下任何文件）
- 已完整阅读章节（按顺序）：导言与学习目标、核心问题、常见误解、1. 朴素实现的问题、2. 按专家分块对齐、3. 一次内核调用算完所有专家、4. 一个 token 的完整路径、5. FusedMoE 层的职责边界、来源与范围说明；overview.html 全文；两页全部折叠块（伪代码、填充上界推导、全部槽位中间量、可运行代码与预期输出）

审查输入：两页 HTML、vLLM v0.28.0（tag 指向 commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`，已经 GitHub release 页核对）五个 fused_moe 源文件、v0.10.2 对照文件（layer.py、topk_softmax_kernels.cu）、v0.28.0 完整 tarball（用于定位 csrc 内文件）、MegaBlocks 论文摘要（arXiv:2211.15841）、check.md。

## 一、来源论断逐条核对（4 步核对）

每条均打开对应源文件与行号，摘录原文片段作为依据。

### 核心论断 C1–C16

- **C1**（融合 MoE 计算由 Triton 内核 fused_moe_kernel 完成）：fused_moe.py L3 模块 docstring `"""Fused MoE Triton kernels."""`；L298–299 `@triton.jit def fused_moe_kernel(`。✅
- **C2**（矩阵语义 A 为 (*,K)、B 为 (E,N,K)、C 为 (M,topk,N)、一次调用覆盖全部专家）：fused_moe.py L356–375 内核 docstring：`A: The input tensor representing tokens with shape (*, K)`、`B: The stacked MOE weight tensor with shape (E, N, K)`、`C: The output cache tensor with shape (M, topk, N)`。✅
- **C3**（三数组语义、填充槽值=有效槽位数）：moe_align_block_size.py L48–52 Returns：`sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.`、`expert_ids: A tensor indicating the assigned expert index for each block.`、`num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.`；L59–70 Example：12 个有效槽、`append padding tokens [12, 12, 12, 12]`、`Tokens 12 are non-existent (padding) and are ignored`；fused_moe.py L421 `token_mask = offs_token < num_valid_tokens`。✅
- **C4**（排序+填充保证可被 BLOCK_SIZE_M 整除）：fused_moe.py L379–383 docstring：`The sorting of sorted_token_ids by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication`；moe_align_block_size.py L54–57 `This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.`。✅
- **C5**（fused_experts 固定顺序流程）：fused_moe.py L1772–1783（_prepare_expert_assignment）→ L1785–1807（GEMM1 dispatch）→ L1809–1811（apply_moe_activation）→ L1824–1846（GEMM2 dispatch）→ L1848–1851（ops.moe_sum）；activation.py L22–24 `Gated activations (gate * activation(up)) expect input of shape [..., 2*d] and produce output of shape [..., d]`、L230–236 SILU 分支 `torch.ops._C.silu_and_mul(output, input)`。✅（步骤计数与图的关系见问题 3.1）
- **C6**（权重作用位置与求和）：fused_moe.py L1796 第一次 dispatch 传 `apply_router_weight_on_input`、L1835 第二次传 `not apply_router_weight_on_input`、L1836 top_k 传 `1`；L1594 签名 `apply_router_weight_on_input: bool = False`；L589–592 注释 `This multiplication MUST be performed in float32`、L593–599 `if MUL_ROUTED_WEIGHT: moe_weight = tl.load(...); accumulator *= moe_weight[:, None]`；L1848–1851 moe_sum。✅
- **C7**（网格与分组）：fused_moe.py L836–839 `grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)`；L385–396 pid 映射，L387 注释 `This is done in a grouped ordering to promote L2 data reuse`；"权重 tile 复用"另由 L1387 默认配置注释直接支持：`Grouping adjacent M-blocks lets them share weight tiles in L2.`。✅
- **C8**（块号选专家、槽位定行）：fused_moe.py L423 `off_experts = tl.load(expert_ids_ptr + pid_m)`；L472–479 `a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + ...)`、`b_ptrs = b_ptr + off_experts * stride_be + ...`。✅
- **C9**（三处防御）：L405–407 `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return`（整块提前返回）；L421 + L532–537 `a = tl.load(a_ptrs, mask=token_mask[:, None] & ..., other=0.0)`（读掩码置零）；L607–610 `c_mask = token_mask[:, None] & (offs_cn[None, :] < N); tl.store(c_ptrs, accumulator, mask=c_mask)`（写掩码）。✅
- **C10**（EP 写零兜底）：fused_moe.py L423–440 `if off_experts == -1: # Write back zeros to the output when the expert is not in the current expert parallel rank.`（write_zeros_to_output 后 return）；moe_align_block_size.py L23–27 `moe_align_block_size initially considers all experts as valid and aligns all tokens appropriately. Before the function returns it marks the experts_ids that are not in the current GPU rank as -1 ... requires the num_experts input arg to be the num global experts.`。✅
- **C11**（调优配置查找）：fused_moe.py L1089–1100 文件名 `f"E={E},N={N},device_name={device_name}{dtype_selector}..."`；L1112–1118 docstring `the closest batch size in the grid should be picked`；L1159–1166 `logger.warning_once("Using default MoE config. Performance might be sub-optimal! ...")`。✅
- **C12**（层组成与 TP 切分）：v0.28.0 layer.py L149–161 FusedMoEFactory docstring：`Creates and configures a complete MoE execution pipeline including: Router (for token-to-expert assignment), RoutedExperts (containing expert weight parameters), MoERunner (orchestrates the complete forward pass)`、`The experts contain both MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)`、`Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We copy that naming convention here`；"gate 在前 up 在后"另由 activation.py L29–31 注释支持（`the *packed* layout ([all gates; all ups]), as produced by a MergedColumnParallelLinear gate_up_proj`）；v0.10.2 layer.py L740–741 `@CustomOp.register("fused_moe") class FusedMoE(CustomOp)`。✅
- **C13**（topk_softmax 融合内核）：v0.28.0 fused_topk_router.py L26–42 `vllm_topk_softmax` 调用 `ops.topk_softmax(...)`（页面引用 L31–44 覆盖该调用）；v0.10.2 csrc/moe/topk_softmax_kernels.cu L2 `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/...`、L531–560 `void topk_softmax(...)` 与 `topkGatingSoftmaxKernelLauncher`。补充核对：v0.28.0 中该文件移至 `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`，L2 同为 `Adapted from .../TensorRT-LLM/blob/v0.7.1/...`，L822 `void topk_softmax(`——caption"改编自 TensorRT-LLM 的同类内核"对 v0.28.0 亦成立。✅（引用路径问题见问题 3.3）
- **C14**（极小批量跳过对齐）：fused_moe.py L1552–1564 `naive_block_assignment = (expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and not(...))`，L1554–1555 注释 `Skips moe_align_block_size and activates the sorted_token_ids is None path`；L1566–1576 返回 `(None, topk_ids.view(-1), full(1, numel*BLOCK_SIZE_M))`；L408–416 内核 naive 分支 `offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)`（每块仅首槽有效）。✅
- **C15**（cache1/cache3 共享缓冲）：fused_moe.py L1733–1734 注释 `We can reuse the memory between these because by the time we need cache3, we're done with cache1`，L1735–1741 `cache13 = torch.empty(...)`、`intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)`。✅
- **C16**（MegaBlocks 动机）：arXiv:2211.15841 摘要：`These formulations force a tradeoff between model quality and hardware efficiency, as users must choose between dropping tokens from the computation or wasting computation and memory on padding.`、`we reformulate MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels`、`Our approach never drops tokens`。✅

### 公式 F1–F3

- **F1**（填充上界 $M\cdot k+E\times(B_M-1)$）：moe_align_block_size.py L74 `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`（默认分支，pad_sorted_ids=False）。✅（小批量截断分支未展开，见问题 3.2）
- **F2**（tile 索引公式 $C[s,n]=\sum_q A[\lfloor s/k\rfloor,q]\cdot B[e,n,q]$）：由 fused_moe.py L472–479 指针运算（`offs_token // top_k * stride_am`、`off_experts * stride_be`、K 维循环累加）与 L1740–1741 缓存形状（(M, top_k, N) 视图）推导；正文可运行代码逐元素复算一致。✅
- **F3**（输出合成公式 $y_t=\sum_j g_{t,j}W_2^{(e_{t,j})}(\operatorname{silu}(a_{t,j})\odot u_{t,j})$）：由 C5（流程）、C6（GEMM2 内乘权重、moe_sum 求和）与 activation.py L22–24（gate × activation(up)，2d→d）组合；正文代码与朴素实现对照验证等价。✅

### 外部数字 N1–N4

- **N1**（默认 $B_M$ 阶梯 16/32/64/128）：fused_moe.py L1366–1380（`General defaults for bf16/fp16 and fp8 per-tensor`）：M≤32→16、M≤96→32、M≤512→64、else→128。✅
- **N2**（版本标识）：GitHub release 页确认 v0.28.0 tag 指向 `2cf0a6915ce544dc493a0990f2ea38d81601128a`。✅
- **N3**（跳过对齐阈值 ×4）：fused_moe.py L1552–1553 注释 `SPARSITY_FACTOR is a heuristic margin`、L1558 `num_tokens * top_k_num * 4 <= global_num_experts`。✅
- **N4**（MegaBlocks 训练加速）：摘要 `enabling end-to-end training speedups of up to 40% over MoEs trained with the state-of-the-art Tutel library and 2.4x over DNNs trained with the highly-optimized Megatron-LM framework`；页面已明确标注"训练场景数字，仅作历史动机旁证，不代表 vLLM 推理性能"。✅

## 二、可读性与功能审查

- 术语首次使用：路由权重/门控权重（第 1 章首段即注明同义）、槽位（第 2 章首段正式定义）、tile 与 SM（链接前置页）、w13/w2 命名（第 4 章给出 Mixtral 沿革）、dispatch/combine 与 EPLB（第 5 章链接前置页）。✅
- 公式均说明用途、符号与运算范围；推导（填充上界、索引公式、合成公式）关键步骤齐备。✅
- 示例说明输入、变化项与结果；构造示例 A 全部数字声明为构造值。✅
- 折叠块收起后正文结论完整：第 1 章三重浪费、第 2 章三数组手算、第 4 章 t_0 全链路与四 token 输出结论均在正文。✅
- 章节衔接：每章末有指向下一章的过渡句。✅
- 学习目标 5 条与 5 个章节一一对应，全部由正文回答。✅
- 问题块：页面级"核心问题"5 问、章节级"本章问题"共 5 处（2/3/3/2/3 问），命名正确，均有解答折叠块，答案独立可读，核心问题答案注明论证所在章节。✅
- KaTeX/Unicode 数学字符：标题、summary、正文、列表、表格中未发现 Unicode 数学字符（×、≤、→ 等仅出现在伪代码 `<pre><code class="language-text">` 内，属代码记法，不在规范禁止范围）；overview 的 `·` 为间隔号、`←/→` 为导航方向符号，非数学运算符。同一变量写法全页一致（$B_M$ 与代码名 BLOCK_SIZE_M 的对应在正文有交代）。✅
- 图示：SVG 结构图无等宽字符框线图；`<text>` 内只有数字与中文标签，无 ASCII 近似公式（图内无公式，无需 foreignObject）；配色用 CSS 变量，明暗主题与窄屏（flex 换行）均可读；节点、箭头、填充槽、虚线"未读取"槽在 caption 中定义。✅
- 手算复算：三数组（sorted=[0,4,1,2,6,8,7,8,3,5]、expert_ids=[0,1,1,2,3]、post=10、分配 12、6 块、块 5 越界返回）、第 3 章块 1 两条 GEMM、第 4 章 t_0 全链路、全部 8 槽位 cache1/2/3、四个 token 输出、$t_2$ 合并抽查——逐项复算一致。✅
- 代码实跑：提取页面唯一 Python 代码块（unescape 后 139 行），Python 3 实跑 returncode=0，`difflib.unified_diff` 与页面预期输出 16 行逐行一致（含 assert 两条路径误差 <1e-9 通过）。✅
- `.dojo/scripts/validate.py wiki/fused-moe/index.html` 返回 `validation ok`。✅
- 元数据：description 为纯文本；dojo:summary 含 $...$ 可渲染；dojo:type=concept；dojo:topics="推理系统"（在 AGENTS.md 固定大类内）；dojo:tag 存在。✅
- 互链与前置链接：overview.html ↔ index.html 互链；../../index.html、moe-serving、gpu-execution-model、swiglu、model-parallelism、eplb 各 index.html 均存在；../../libs/ 共享库文件齐全。✅
- overview.html 内容与 index.html 论断一致（三数组、除 k 还原行、跳过对齐条件 $M\cdot k\times 4\le E$、写零兜底、调优查找），更新日期 2026-09-03。✅

## 三、问题

- [轻微·可读性] index.html 第 4 章首段（"按固定顺序执行五步"）与紧随的六节点数据流图计数不一致｜问题：正文称 fused_experts"按固定顺序执行五步"，但图中含"路由打分与 top-k"共六个节点，caption 称"箭头为前向数据流"；第 1 章"路由一次、对齐一次、两次 GEMM、激活一次、求和一次"亦为六项。fused_experts 接收的是路由结果，其内部确为五步（对齐、GEMM1、激活、GEMM2、求和），但正文未说明图中路由一步在入口之前，数图得六步与"五步"冲突｜引文依据：fused_moe.py L1587–1592 签名 `def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)`（路由结果为入参）；L1772–1851 内部五步｜修复要求：将"五步"改为"六步"，或补一句"图中路由一步由 Router 在 fused_experts 之前完成，入口内部为五步"｜修复：已按第二种方案补句："路由在入口之前已经完成，其结果作为入参传入；入口内部按固定顺序执行五步。下图从路由开始展示整层数据流，共六个节点，后五个方块对应入口内的五步"｜复验：grep 确认第 4 章首段含该句；validate.py 通过
- [轻微·技术] index.html 第 2 章表格与第 3 章"EM 取 sorted_token_ids 的分配长度（按 F1 上界预留）"｜问题：表述未覆盖源码的小批量截断分支——moe_align_block_size.py L77–80 在 `topk_ids.numel() < num_experts` 时分配长度被 `min(topk_ids.numel() * block_size, ...)` 截断；fused_moe.py L826–833 在 `A.size(0) < BLOCK_SIZE_M` 时 EM 再被 `min(sorted_token_ids.size(0), A.size(0) * top_k * BLOCK_SIZE_M)` 缩小。构造示例 A（M=4≥B_M=2、numel=8≥E=4）不触发，索引语义不受影响，但"按 F1 上界预留"在真实小批量（如 E/4 < M·k < E 或 M < B_M）时不严格｜引文依据：moe_align_block_size.py L77–80 `if topk_ids.numel() < num_experts: max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)`；fused_moe.py L826–833 `if A.size(0) < config["BLOCK_SIZE_M"]: ... EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])`｜修复要求：在第 2 章表格注或"简化条件"小节补一句：小批量时分配长度与网格 EM 可被进一步缩小（槽位数不足专家数时按 numel×block_size 截断），不影响索引与掩码语义｜修复：已在第 2 章 F1 段末补"当槽位总数小于专家数时，实现会把分配长度截断到 $M\cdot k\times B_M$ 与该上界的较小值，以省下空段占用的空间；两种情形下有效内容的语义不变"｜复验：grep 确认该句存在；validate.py 通过
- [轻微·来源标注] index.html"来源与范围说明"C13 条目｜问题：topk_softmax 的 CUDA 出处只引 v0.10.2 路径 `csrc/moe/topk_softmax_kernels.cu`；v0.28.0（页面基准版本）中该文件实际位于 `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`（L2 同为 "Adapted from .../TensorRT-LLM/blob/v0.7.1/..."，L822 为 `void topk_softmax(`）。论断本身两版均成立，仅引用可更直接｜引文依据：v0.28.0 csrc/libtorch_stable/moe/topk_softmax_kernels.cu L2 `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu`、L822 `void topk_softmax(`｜修复要求：C13 来源行补 v0.28.0 路径与行号（保留 v0.10.2 引用亦可）｜修复：C13 改为"CUDA 内核在 v0.28.0 位于 csrc/libtorch_stable/moe/topk_softmax_kernels.cu（v0.10.2 位于 csrc/moe/topk_softmax_kernels.cu），文件头 L1–2 Adapted from TensorRT-LLM v0.7.1"｜复验：grep 确认来源条目含 v0.28.0 路径；validate.py 通过

## 四、发布条件核对（check.md 第 5 节）

- 三轮审查均完成且由独立审查者执行：本轮（第 3 轮）为独立执行；前两轮的独立性由编排者记录，本轮未读取其内容。✅（编排者确认项）
- 每条来源论断都有引文依据记录：见本文第一节，C1–C16、F1–F3、N1–N4 全部核对通过，无定位不到或内容不符条目。✅
- 阻断与重要问题均已关闭：本轮未发现阻断或重要问题；前三节问题均为轻微。✅（以本轮观察为准）
- 遗留轻微问题具有明确接受理由：3 条轻微均不影响核心结论与索引/数值正确性（3.1 为计数表述、3.2 为不影响语义的源码分支省略、3.3 为引用路径优化）；可修复后发布，或按"不影响正确性与主线理解"接受。⚠（建议修复，见处置）
- 全部学习目标由正文章节完整回答：✅
- 页面级"核心问题"与各章"本章问题"均有解答折叠块：✅
- 数学符号全部 LaTeX、结构图为内联 SVG：✅
- validate.py 返回成功：✅（`validation ok: wiki/fused-moe/index.html`）
- 可运行代码结果与页面描述一致：✅（实跑 16 行输出与页面预期逐行一致）
- 关键论断与数字已重新核对来源：✅（本文第一节）
- head 元数据（description、dojo:summary、dojo:type、dojo:topics、dojo:tag）齐全且 topics 在词表内：✅
- overview.html 与 index.html 相互链接：✅
- 概念链接有效：✅（moe-serving、gpu-execution-model、swiglu、model-parallelism、eplb、首页均存在）
- 递归生成的前置概念页已完成各自质检：输入边界限制，本轮未读取各前置页的 research/ 记录，此项由编排者确认。⚠（编排者确认项）

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：**可发布**。3 条轻微问题（步骤计数表述、小批量截断分支省略、C13 引用路径）均不影响核心结论、来源一致性与可运行代码正确性；建议发布前顺手修复 3.1（一句话改动），3.2 与 3.3 可作为已知简化接受或一并修复。若执行修复，修复后仅需复验对应位置并重跑 validate.py，无需重开审查轮次。前置概念页质检状态（发布条件最后一条）由编排者确认后即可发布。

## 发布记录（编排者填写）

- 三条轻微问题已在审查后全部修复并复验（validate.py、页面代码实跑、headless Chrome 渲染三项复检通过）。
- 三轮审查统计：第 1 轮 0 阻断 / 1 重要 / 10 轻微（全部关闭）；第 2 轮 0 阻断 / 1 重要 / 3 轻微（全部关闭）；第 3 轮 0 阻断 / 0 重要 / 3 轻微（全部关闭）。
- 发布条件全部满足：三轮独立审查完成、来源论断均有引文记录、阻断与重要问题清零、学习目标完整回答、两级问题块作答齐备、数学符号 LaTeX 化、validate.py 通过、可运行代码输出一致、元数据合规、overview 互链有效。
- 处置：可发布。页面位置 wiki/fused-moe/（index.html + overview.html），首页目录与关系图由 GitHub Pages 构建自动发现。
