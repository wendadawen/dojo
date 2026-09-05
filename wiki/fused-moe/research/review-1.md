# FusedMoE 审查记录（第 1 轮）

- 页面版本：index.html 工作树哈希 `7e17b79b8e748ef31acc299d91bd22a28c71d145`（overview.html `84648b80a2facae1fa850392edde0180d6873579`）
- 审查时间：2026-09-03 14:17
- 审查者：编排者派发的独立审查者（未参与写作与前序审查）
- 已完整阅读章节（按顺序）：标题与引言、核心问题（5 题及解答）、常见误解、1 朴素实现的问题（含伪代码折叠块与本章问题）、2 按专家分块对齐（含三数组表、手算三步、SVG 图、上界证明折叠块与本章问题）、3 一次内核调用算完所有专家（含索引公式、示例手算、两次 GEMM 表与本章问题）、4 一个 token 的完整路径（含流程图、权重表、中间量折叠表、可运行代码与预期输出、本章问题）、5 FusedMoE 层的职责边界（含职责表与本章问题）、来源与范围说明（全部小节）；overview.html 全文。

核对方式：vLLM v0.28.0（commit 2cf0a69）与 v0.10.2 源码逐行下载核对；MegaBlocks 摘要经 arXiv 页面核对；页面代码提取、unescape 后实跑并以 difflib 比对预期输出。

## 一、来源论断核对表（每条含核对时看到的原文片段或关键数值）

- C1：fused_moe.py L3 `"""Fused MoE Triton kernels."""`；L299 `def fused_moe_kernel(`。✓
- C2：内核 docstring（L356–375）`A: The input tensor ... shape (*, K)` / `B: The stacked MOE weight tensor with shape (E, N, K)` / `C: The output cache tensor with shape (M, topk, N)`。✓
- C3：moe_align_block_size.py L48–60 Returns 三条（`sorted_token_ids ... sorted token indices according to their allocated expert`、`expert_ids ... for each block`、`num_tokens_post_padded ... total number of tokens after padding`）；Example 中 `append padding tokens [12, 12, 12, 12]`（填充值=有效槽位数 12）；fused_moe.py L421 `token_mask = offs_token < num_valid_tokens`。✓
- C4：fused_moe.py L381 docstring `The sorting of sorted_token_ids by expert index and padding ensures divisibility by BLOCK_SIZE_M`。✓
- C5：fused_moe.py L1772–1851 fused_experts 主体顺序为 align → dispatch(GEMM1) → activation → dispatch(GEMM2, top_k=1) → `ops.moe_sum`（L1848）；activation.py L22–24 `Gated activations (gate * activation(up)) expect input of shape [..., 2*d] and produce output of shape [..., d]`、L236 `torch.ops._C.silu_and_mul(output, input)`。✓
- C6：GEMM2 调用传 `not apply_router_weight_on_input` 与 `1`（L1824、L1827 附近）；L593–599 `This multiplication MUST be performed in float32 ... accumulator *= moe_weight[:, None]`；L1848–1851 moe_sum。✓（默认路径，见问题 8）
- C7：L835–839 `grid = ... cdiv(EM, META["BLOCK_SIZE_M"]) * cdiv(B.size(1), META["BLOCK_SIZE_N"])`；L385–396 `grouped ordering to promote L2 data reuse`、GROUP_SIZE_M。✓（EM 定义见问题 1）
- C8：L421–424 `off_experts = tl.load(expert_ids_ptr + pid_m)`、L423–479 `b_ptr + off_experts * stride_be`、`offs_token[:, None] // top_k * stride_am`。✓
- C9：L406 `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return`；L421 读掩码；L607–610 `c_mask = token_mask[:, None] & (offs_cn[None, :] < N)`。✓
- C10：L424 `if off_experts == -1: write_zeros_to_output(...)`；moe_align_block_size.py L23–27 `Before the function returns it marks the experts_ids that are not in the current GPU rank as -1 ... requires the num_experts input arg to be the num global experts`。✓
- C11：L1089–1166 `get_config_file_name(E, N, dtype, ...)` / `the closest batch size in the grid should be picked` / `logger.warning_once("Using default MoE config. Performance might be sub-optimal! ...")`。✓
- C12：v0.28.0 layer.py L149–161 `Router (for token-to-expert assignment), RoutedExperts (containing expert weight parameters), MoERunner (orchestrates the complete forward pass)`、`MergedColumnParallel weights (gate_up_proj/w13) and RowParallelLinear weights (down_proj/w2)`、`Mixtral uses w1, w2, and w3`；v0.10.2 layer.py L741 `class FusedMoE(CustomOp)`。✓
- C13：fused_topk_router.py L33 `ops.topk_softmax(...)`；v0.10.2 topk_softmax_kernels.cu L1–2 `Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/...`、L531–560 `topkGatingSoftmaxKernelLauncher`。✓
- C14：L1552–1584（原文与页面表述的条件差异见问题 2）。△
- C15：L1733–1738 `We can reuse the memory between these because by the time we need cache3, we're done with cache1`。✓
- C16：arXiv:2211.15841 摘要 `users must choose between dropping tokens from the computation or wasting computation and memory on padding`、`we reformulate MoE computation in terms of block-sparse operations`、`Our approach never drops tokens`。✓
- F1：moe_align_block_size.py L74 `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`（默认分支，pad_sorted_ids=False 已核实为默认值）。✓
- F2：由 L408–479 指针运算（`offs_token // top_k`、`off_experts * stride_be`）与 L1733–1755 缓存形状推导；页面代码实跑逐元素验证通过。✓
- F3：C5+C6+activation.py 组合；代码对照朴素实现，`assert` 通过。✓
- N1：L1366–1380 bf16/fp16 通用默认 `M<=32→16, M<=96→32, M<=512→64, else→128`。✓
- N2：按用户给定 commit 核对版本即 v0.28.0（2cf0a69）。✓
- N3：L1556–1558 `num_tokens * top_k_num * 4 <= global_num_experts`（SPARSITY_FACTOR=4 注释核实）。✓（条件完整性见问题 2）
- N4：摘要 `end-to-end training speedups of up to 40% over ... Tutel library and 2.4x over DNNs trained with ... Megatron-LM`。✓

## 二、机械验证

- 代码实跑：提取 index.html 唯一 `language-python` 代码块，unescape 后以 python3 执行，exit 0，`assert` 通过；实际输出与页面预期输出经 `difflib.unified_diff` 比对**完全一致**（OUTPUT MATCH）。
- `.dojo/scripts/validate.py wiki/fused-moe/`：返回 `validation ok`。
- 问题块计数：页面级核心问题 5 题 / 解答折叠块 5 个；章节级 2+3+3+2+3 = 13 题 / 解答 13 个，全部成对；核心问题答案均指明论证所在章节，答案脱离正文可独立阅读。
- KaTeX/Unicode 数学字符：正文（除代码块与图示 UI 箭头、导航图标外）无裸露 Unicode 数学字符；SVG `<text>` 仅含数字与中文说明，无 `$`、无 ASCII 数学近似；代码块内无 `$`（不会被 auto-render 破坏）。发现 1 处正文裸 `→`（见问题 6）。
- 链接与资源：moe-serving、gpu-execution-model、swiglu、model-parallelism、eplb 五个概念页均存在；libs 下 katex/prism 资源齐全；overview 与 index 互相链接；锚点 h2/h3 全部有 id。
- 说明：`guides/concept/style-guide.md` 不在本轮允许的输入清单内，2.2-12 格式一致性仅经 validate.py 与规范其余各条覆盖。

## 三、问题

- [重要·技术] index.html 第 5 章第 4 段（及页面级误解第 3 条答句、第 5 章本章问题 3 解答、overview.html「关键结论与边界」第 4 条）｜跳过对齐路径的生效条件缺失：页面表述为"当 $M\cdot k\times 4$ 不超过全局专家数时，vLLM 跳过 moe_align_block_size"，但源码该路径还需 `expert_map is None`（即未启用专家并行）且非块状量化路径｜引文依据：fused_moe.py L1556–1561 `naive_block_assignment = expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts and not ((use_int8_w8a16 or use_int4_w4a16) and block_shape is not None ...)`｜修复要求：在第 5 章该句补"且未启用专家并行（expert_map 为空）"；章节问题 3 解答与 overview 同步补充｜修复：index.html 第 5 章两处（条件描述句 + 本章问题 3 解答）已补"未启用专家并行（expert_map 为空）"；overview.html 关键结论第 4 条同步补齐｜复验：复检查相关句子与 overview 第 4 条均含"未启用专家并行"前提；validate.py 与渲染探针全数通过
- [轻微·技术] index.html 第 3 章第 2 段与本章问题 1 解答｜$EM$ 定义不准确：按字面 EM 为 num_tokens_post_padded（示例 10），复算得 5 块，与同章"6 个块"矛盾；实际 $EM$ 取 sorted_token_ids 的分配长度（示例 12）｜引文依据：fused_moe.py L823–824 `EM = sorted_token_ids.size(0)`；moe_align_block_size.py L81–83 分配 `(max_num_tokens_padded,)`=12｜修复要求：将两处定义改为"$EM$ 取 sorted_token_ids 的分配长度（即 F1 上界，示例为 12）"｜修复：第 3 章正文与本章问题 1 解答均改为"$EM$ 取 sorted_token_ids 的分配长度（即 F1 上界 $M\cdot k+E\times(B_M-1)$，示例为 12）"；并附"请注意 $EM$ 取的是分配长度而非 num_tokens_post_padded"的提示句｜复验：grep 确认两处均使用"分配长度"表述；validate.py 通过
- [轻微·技术] index.html 第 2 章三数组表"长度"列｜混淆语义长度与分配长度：sorted_token_ids 标"按下界与上界之间的实际填充结果"（实际分配长度恒为上界 $M\cdot k+E\times(B_M-1)$，示例 12）；expert_ids 标"填充后总长除以 block_size 上取整"（=5，实际分配 6，末尾块提前返回）；"下界/上界"在表中先于 F1 公式出现｜引文依据：moe_align_block_size.py L81–89 `sorted_ids = torch.empty((max_num_tokens_padded,))`、`max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)`｜修复要求：注明该列指有效内容长度并补一句分配长度，或统一以"有效长度/分配长度"两列表述｜修复：将"长度"列拆为"有效长度"与"分配长度"两列；每行分别给 num_tokens_post_padded 与 F1 上界对应值（示例分别 10/12、5/6、1/—）；并明确"末尾块提前返回"｜复验：表格三行两列数值与代码实跑一致
- [轻微·可读性] index.html 引言第 1 段｜"再用一次分块 GEMM 内核调用算完全部专家的份额——对每一份权重各调用一次"分句自相矛盾：字面读作"每份权重各调用一次内核"（$E$ 次调用），与"一次内核调用"及后句"调用次数不再随专家数增长"冲突｜引文依据：不适用｜修复要求：改写为如"每一份专家权重在同一次调用中各由属于自己的块组读取"｜修复：改写为"每份专家权重（w13 与 w2）在各自的一次调用中由属于自己的块组读取，调用次数固定为个位数、不再随专家数增长"｜复验：grep 确认引言段不再含"对每一份权重各调用一次"
- [轻微·可读性] index.html 第 4 章首段与流程图｜"按固定顺序执行五步"与紧随流程图的 6 个节点不对应：fused_experts 的五步（对齐、GEMM1、激活、GEMM2、求和）不含路由（其输入即路由结果），但图首节点为"路由打分与 top-k"｜引文依据：fused_moe.py L1772 签名 `fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)`——路由结果为入参，不在函数内｜修复要求：在图注或首段明确五步划分，说明路由在 fused_experts 之外、图为完整层路径｜修复：首段改写为"其内部按固定顺序执行五步；路由打分与 top-k 选择是 fused_experts 的上游步骤，结果已作为入参传入。下图从路由开始展示整层数据流，后五个方块对应这五步"｜复验：图注中"两次 GEMM 使用同一个 fused_moe_kernel"仍成立，路由作为上游方块存在
- [轻微·格式] index.html 第 4 章本章问题 1 解答｜形状序列以裸 Unicode 箭头连接：违反"正文无 Unicode 数学字符直接出现"｜引文依据：不适用（规范 guides/concept/check.md 2.2-9）｜修复要求：箭头改写为 KaTeX 形式（`$\to$`）｜修复：将"$(M, H)$ → …"全部 5 个箭头改为"$\to$"｜复验：grep 确认 ch4 该段仅在 $...$ 内含箭头
- [轻微·技术] index.html 第 4 章合成公式及"其中"句｜符号碰撞：$g$ 同时表示 gate 半向量与门控权重标量 $g_{t,j}$｜引文依据：不适用（写法一致性要求）｜修复要求：gate 半改用不冲突的符号（如 $a$ 与 $u$），全页及 summary 同步｜修复：gate 半改用 $a_{t,j}$、up 半沿用 $u_{t,j}$；F3 公式改为 $y_t=\sum_j g_{t,j}\cdot W_2^{(e_{t,j})}(\operatorname{silu}(a_{t,j})\odot u_{t,j})$，$(a_{t,j},u_{t,j})$ 为 $W_{13}^{(e_{t,j})}x_t$ 拆为前后 $I$ 维；dojo:summary 与核心问题 Q4 答案同步｜复验：grep 确认 $g$ 仅作为 $g_{t,j}$（路由权重）出现，不再作为 gate 半
- [轻微·技术] index.html 第 3 章两次 GEMM 表"乘路由权重：否/是，fp32"｜未说明这是 `apply_router_weight_on_input=False` 的默认路径；个别模型配置下两栏对调｜引文依据：fused_moe.py L1789 附近 GEMM1 实参 `apply_router_weight_on_input`、L1824 附近 GEMM2 实参 `not apply_router_weight_on_input`｜修复要求：在表格或"简化条件"中注明默认路径前提｜修复：表后追加说明段"两栏'乘路由权重'的取舍由参数 `apply_router_weight_on_input` 决定：默认 False 时权重在 GEMM2 乘入；某些模型置 True 把权重乘到 hidden_states 输入侧再传入，两栏对调<sup>[C6]</sup>"｜复验：grep 确认表格后存在该说明段
- [轻微·可读性] index.html 全页｜"门控权重"与"路由权重"两个称呼混用（同一 topk_weights）｜引文依据：不适用｜修复要求：首次出现"路由权重"处注明"即门控权重（topk_weights）"｜修复：ch1 引入"门控权重"句改为"router 给出 $k$ 个专家与路由权重（即门控权重，对应 topk_weights）"；ch1 模型层定义句改为"router 给出路由权重（即 topk_weights，本页也称门控权重）"；ch2 路由表表头与相关句同步改为"路由权重"；Q4 答案亦同步；dojo:summary 与 overview 保持"路由权重"为主、"门控"仅用于 silu(gate)·up 激活 | 复验：grep 确认 topk_weights 全部语境使用"路由权重"，"门控"仅出现在 silu(gate)·up 激活上下文
- [轻微·格式] overview.html「关键结论与边界」第 1 条｜"（本页附纯 Python 双实现验证）"指称错误：overview 页无代码｜引文依据：不适用｜修复要求：改为"完整说明页附纯 Python 双实现验证"｜修复：已改｜复验：grep 确认
- [轻微·格式] overview.html 链接锚文本｜与目标页标题不一致：`MoE 层与路由`（目标页题为"MoE 大模型推理与服务基础"）、`EPLB`（目标页题为"EPLB 专家并行负载均衡"）｜引文依据：不适用｜修复要求：锚文本与 index.html 统一为目标页全称｜修复：两处锚文本均改为目标页全称｜复验：grep 确认 overview 锚文本与 index 一致

## 四、复算与走查记录（无问题项）

- 构造示例 A 三数组手算复核：$[0,4\,|\,1,2,6,\mathrm{pad}\,|\,7,\mathrm{pad}\,|\,3,5]$、expert_ids $=[0,1,1,2,3]$、post $=10$、分配长度 $12$（$8+4\times1$，pad_sorted_ids 默认 False），与源码语义及页面一致；填充值 8 = num_valid_tokens，与 docstring Example 的 12 一致。
- 第 3 章块 1 手算、第 4 章 $t_0$ 全路径与 $t_2$ 合并核对、silu 数值，均与代码实跑输出一致。
- 学习目标 5 条分别由第 1–5 章完整回答；各章末均有衔接段；折叠块收起后正文结论完整；类比边界已在"来源与范围说明"声明。
- 术语首次使用（槽位、门控权重、dispatch/combine 等）均有解释或前置页承接。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 10
- 处置：修复后重要问题 1 条已关闭、轻微 10 条全部关闭；validate.py、代码实跑、headless Chrome 渲染三项复检通过；页面达到发布条件。
