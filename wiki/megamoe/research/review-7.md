<!-- review-meta
round: 7
page: wiki/megamoe/index.html
reviewed_content_sha256: 846222b7a41d091b
-->
# MegaMoE 审查记录（第 7 轮）

- 页面版本：index.html 402f87e9d71dd01162835827d58bf269e250f3e2（工作树）；overview.html 42ef9429f585f561c9b4b2e4112373a1ca215a3f
- 审查时间：2026-09-14 17:39
- 审查者：编排者派发的独立审查者（子代理，未参与写作，亦未参与前序轮次审查与修复）
- 已完整阅读章节（按顺序）：核心问题 / 常见误解 / 1. MoE 一层的五段执行——通信为什么让 GPU 空转（1.1、1.2、本章问题）/ 2. 一个持久 kernel 里的分工——融合与重叠如何成立（2.1、2.2、本章问题）/ 3. 对称内存——kernel 内跨 rank 读写的地基（3.1、3.2、本章问题）/ 4. 一个 token 的完整旅程——拉取、两层 GEMM、远程写回与归约（4.1、4.2、4.3、本章问题）/ 5. 收益与边界——基准数字、正确性与适用条件（5.1、5.2、5.3、本章问题）/ 来源与范围说明（论断与来源（C）、公式与来源（F）、外部数字与实验条件（N）、构造示例、辅助解释与类比边界、简化条件及其限制）

## 核对所用来源版本

- DeepGEMM main @ 559d79f（2026-07-15，commit message "Public release 26/07 (#377)"，与页内声明一致）：README.md、tests/test_mega_moe.py、csrc/jit_kernels/heuristics/mega_moe.hpp、csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp、deep_gemm/include/deep_gemm/layout/sym_buffer.cuh、deep_gemm/include/deep_gemm/layout/mega_moe.cuh、deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh、deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
- PR #304（2026-04-17 合并）、PR #316（2026-04-24 合并）
- X-Stage 论文 arXiv:2607.23264v1
- AMD Primus 文档 rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/mega-moe.html（标题 "MegaMoE — AMD Primus 26.6"）

## 来源核对要点（引文依据）

- C1/C7（README Mega MoE 节 @559d79f）："Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory."
- C2（README 首段与 News）：首段含 "fused MoE with overlapped communication (Mega MoE)"；News "2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL, faster JIT compilation and more."
- C22（README Usage 代码块 @559d79f）：`get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)`、`transform_weights_for_mega_moe(l1_weights, l2_weights)`、`buffer.x[:num_tokens].copy_(x_fp8)` 等四行、`deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)`——与页内代码块逐字一致。
- C26（README Utilities）："`DG_COMM_KERNEL_DEBUG`: `0` or `1`, zero symmetric buffer before each Mega MoE call for debugging, `0` by default"
- C3/C27/C28（PR #304）："Only FP8 x FP4 MoE is supported"、"Requires PyTorch >= 2.9"、"Mega MoE is still under development and optimizations"、"has nothing to do with internal model release"；Mega MoE 贡献者 8 人；Apr 17 2026 合并。
- N1–N4（PR #316 表一/表二）：V4-Flash 256 专家/6/4096/2048 与 V4-Pro 384 专家/6/7168/3072；四档 batch 耗时 56.5/146.5/1283.1/4855.5 与 108.1/369.6/2818.5/10655.2 us，加速 1.96/1.73/1.56/1.62 与 1.61/1.54/1.50/1.54，互联 batch512 为 266 与 182 GB/s，EP8、8 rank 平均——与页内表逐格一致。N5：zheanxu 评论确认 batch 为 per-rank token（512×8=4096）。N6：两 PR 合并日 04-17、04-24。N9：yiakwy 评论 Hopper 无 TMEM 收益不显著、可换 "Fp4 EP V2 + FP8 DeepGeem + PDL"、sm90 移植推进中。
- F2（heuristics/mega_moe.hpp）：`float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;`
- C24/4.2 表（同文件）：六档 `<= 8.5 / <= 16.5 / <= 32.5 / <= 64.5 / <= 96.5 / else` → block_m `16/32/64/96/128/192`；注释 "Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512" 对应 block_m=96，"Prefill, or large EP decoding" 对应 192，"Really small token-per-expert (e.g. RL long-tail rollout)" 对应 16；`const int block_n = 128;`、`const int gran_k = 32;`
- C12（同文件）：`// Pull: divide token bytes by 2 until <= kPullThreshold`、`constexpr int kPullThreshold = 4096;`
- C9（同文件）：`num_dispatch_threads = 128`、`num_non_epilogue_threads = 128`、epilogue = num_epilogue_warpgroups(1 或 2)×128 → 128/256，合计 384/512。
- C18（scheduler/mega_moe.cuh）：`enum class BlockPhase : uint32_t { ... Linear1 = 1, Linear2 = 2, SharedLinear1 = 3, SharedLinear2 = 4 }`、`// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock`、`result = ptx::atomic_add(global_task_count_ptr, 1u);`（第 309-314、15-45、316-350 行附近）
- C17（scheduler/mega_moe.cuh 第 385 行）：`// Shared expert L1 tasks do not depend on dispatch.`
- C5（.cuh 第 78、1453-1455 行）：`#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)` 与 `#else ... DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");`
- C7（.cuh 第 143-146 行）：`// NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)`、`using a_dtype_t = cutlass::float_e4m3_t;`、`using b_dtype_t = ... float_e2m1_unpacksmem_t;`、`using shared_b_dtype_t = cutlass::float_e4m3_t;`
- C30（.cuh 第 312-322 行）：`kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64; kNumDispatchRegisters = ... 48 : 96; kNumNonEpilogueRegisters = ... 40 : 88; kNumEpilogueRegisters = ... 208 : 160;` 合计 `<= 64512`
- C14（.cuh 第 1052-1071 行）：gate 仅 `__hmin2(bf16_gate, …)`，up 同时 `__hmax2` 与 `__hmin2`；`activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);`（权重乘法在 L1 epilogue，行 1071）
- C15/C16（.cuh 第 1205、1323、1313-1345 行）：`// L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink`、`// Combine: reduce top-k results and write back`、`DG_STATIC_ASSERT(kNumTopk <= 32, …)`（行 334）、`DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp")`（行 1345）、`DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, …)`（行 86）
- C29（.cuh 第 308-311 行）：`kBeforeDispatchPullBarrierTag = 1; kBeforeCombineReduceBarrierTag = 2; kAfterWorkspaceCleanBarrierTag = 3;`
- 4.1 节元数据尺寸（.cuh 第 357-374 行）：`const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]); ... ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value)` → 每专家计数为 uint64（8 字节）；`*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;`（`token_topk_idx` 为 int）→ 每个 token-专家对源索引 4 字节。
- C10/N10（layout/sym_buffer.cuh 第 7、38 行）：`constexpr static uint32_t kNumMaxRanks = 72;`、`int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);`
- C11/C12（.cuh 第 461 行）：`// Round-robin rank selection via iterative min-peeling`
- C13（csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp 第 187 行）：`// NOTES: L1 output and L2 activations are essentially the same tensor.`
- C8（同 .hpp 第 277、309 行）：`const auto num_sms = device_runtime->get_num_sms();`、`.launch_args = LaunchArgs(num_sms, …)`；.cuh 第 55 行 `CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void`
- C21/N7/C4/C23（tests/test_mega_moe.py @559d79f）：`assert torch.equal(fused_stats, baseline_stats)`、`assert torch.equal(fused_y, baseline_y)`、`assert calc_diff(fused_y, baseline_y) < 1e-8`；基线 = deep_ep + `tilelang_ops.swiglu_apply_weight_to_fp8` + `m_grouped_fp8_fp4_gemm_nt_contiguous`（`import_baseline()` 注释 "Load legacy implements from third-party"，变量 `is_legacy_loaded`）；默认 384/6/7168/3072/共享 1/8192；`assert hidden % 128 == 0 and intermediate_hidden % 128 == 0 and shared_intermediate_hidden % 128 == 0`（仅非 bf16 路径）；第 197 行 `(deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)`、第 425 行 `--mma-type … default='fp8xfp4' … 'bf16xbf16'`；`--num-processes` 默认 8、`torch.multiprocessing.spawn`。
- F1/C20/C25（X-Stage arXiv:2607.23264v1）："Combine returns expert outputs to the source rank and aggregates the top-k results using the router weights."；"MegaMoE groups local experts into expert waves to organize locality and execution."；"an estimated speedup ceiling of about 1.5× over the serial baseline, whereas the implementation reaches 1.56×"；"Across 84 configurations, the resulting interleaved scheduler achieves a 1.18× geometric-mean speedup … and a 1.62× maximum speedup over the Expert-Wave baseline"；全文无 "expert wave" 于官方源码的对应词（scheduler/mega_moe.cuh 检索无 "expert wave"，仅有 warmup waves）。
- 来源与范围说明（同名项目，AMD Primus）："MegaMoE is a FlyDSL-based fused MoE layer that replaces Megatron's native MoELayer … Runtime target is EP-only (TP=1) + bf16."——与页内描述一致。
- 机械项：`.dojo/scripts/validate.py wiki/megamoe/index.html` 返回 `validation ok`；页内 9 个概念链接（deepseek-moe、moe-serving、swiglu、gpu-communication、deepep、gpu-execution-model、fp8-block-quant、mxfp4-qat、deepseek-v4-dataflow）在磁盘上均存在；head 的 description / dojo:summary / dojo:type=concept / dojo:topics / dojo:tag 齐备；无 `$...$` 出现在 alt 属性。
- 说明（不计为问题）：来源表中 `heuristics/mega_moe.hpp`、`layout/*.cuh`、`scheduler/mega_moe.cuh`、`sm100_fp8_fp4_mega_moe.hpp` 等是仓库内路径的后缀片段（实际为 `csrc/jit_kernels/…` 与 `deep_gemm/include/deep_gemm/…`）；页内已声明"可在 deepseek-ai/DeepGEMM 对应文件处检索复核"并在正文中约定未带路径的 `.cuh` 指 `sm100_fp8_fp4_mega_moe.cuh`，逐条行号在本轮核对的版本上均可定位，故不列为问题。

## 问题

- [轻微·来源] 第 1 章首段（"MoE 模型把一个大的前馈网络拆成许多小专家，路由网络为每个 token 只选其中几个[C1]"）：[C1] 对应的 README "Mega MoE" 节与 PR #304 只陈述融合/重叠与"多进程 + 对称内存"要求，不含路由机制内容，该句是通用 MoE 背景陈述，与 [C1] 无对应关系｜引文依据：README "Mega MoE" 节原文 "Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel…It requires multi-process launch with symmetric memory."（无路由内容）｜修复要求：删去该句末的 `<sup>[C1]</sup>`（该句为背景陈述，可无引用），或把 [C1] 移至本段其后确实由 README 支持的机制句（如同一章的融合/重叠表述）｜修复：｜复验：
- [轻微·格式] 章节衔接固定句式：第 1、2、3 章末尾均以"……，下一章……"同一套式收束（1 章末"这正是 Mega MoE 的做法，下一章看它如何组织。"；2 章末"这个前提由对称内存提供，下一章专门讲它。"；3 章末"地基就位。下一章跟着贯穿示例的 token，把这层缓冲上发生的每一步走完整。"）｜引文依据：不适用（`guides/concept/style-guide.md` §8："章节顺序存在依赖时，用一至两句说明前一节结论与下一节问题的关系。不使用固定句式"）｜修复要求：把其中至少两处改写为不出现"下一章"套式的过渡句，只保留结论到下一节问题的依赖说明｜修复：｜复验：
- [轻微·可读性] 引言承诺未完全兑现（第 1 段末"它们在首次正式使用的章节还有更完整的说明与概念页链接。"）：rank、SM、warp、TMA 四个词中仅 SM 在 1.2 节给出概念页链接（"GPU 执行模型"）；warp 与 TMA 在首次正式使用处（2.2 节）只重复引言里的一行定义，未再给概念页链接；全页链接集合不含 warp / TMA 主题页｜引文依据：不适用｜修复要求：删除"与概念页链接"五字（保留"更完整的说明"），或为 warp、TMA 在 2.2 节补上真实存在的概念页链接｜修复：｜复验：

## 结论

- 事实与来源：C1–C30、F1–F3、N1–N11 全部条目本轮逐条回到 559d79f 对应文件与 PR/X-Stage/Primus 原文核对通过，未发现来源不支持、编号错配（含 v1/v2 差异）或推断被写成来源结论的情形；两项本页推断（5.1 节 batch=1 收益成因、F3 通信量估算）均已显式标注为推断/估算而非官方结论。
- 数字与算式：基准表逐格与 PR #316 一致；43 KB/86 KB/22 MB、N_exp 表（64/1024/0.125 与 block_m 96/192/16）复算无误；同页多处同一数字（1.50–1.96、266/182 GB/s、top-k 6、hidden 7168）在正文、summary、overview、图注间一致；F1 在正文与 summary 均由 KaTeX 渲染。
- 表述：无第一/第二人称、无调试叙事与临场评价；"本页/本文"的自称符合 style-guide §12；仅第三项（承诺落空）与第二项（过渡套式）为表述层面残留。
- 处置：可发布。3 条轻微问题均不影响正确性与主线理解（分别为：单处背景句的引用多余、三处过渡句套式化、引言一处链接承诺偏宽），可接受并连同下轮轻微清理一并处理。
- 统计：阻断 0 / 重要 0 / 轻微 3