# MegaMoE 审查记录（第 2 轮）

- 页面版本：index.html 110371 字节（工作树，2026-09-03 14:24）；overview.html 6751 字节
- 审查时间：2026-09-03 14:31
- 审查者：独立技术审查者（未参与写作与前序审查）
- 已完整阅读章节（按顺序）：标题与导语 → 核心问题（5 题及解答折叠块）→ 常见误解 → 1. MoE 一层的五段执行（1.1 五段各自做什么 / 1.2 串行执行的两种空闲 / 本章问题）→ 2. 一个持久 kernel 里的分工（2.1 / 2.2 / 本章问题）→ 3. 对称内存（3.1 / 3.2 / 本章问题）→ 4. 一个 token 的完整旅程（4.1 / 4.2 / 4.3 / 本章问题）→ 5. 收益与边界（5.1 / 5.2 / 5.3 / 本章问题）→ 来源与范围说明（C/F/N 表、构造示例、辅助解释与类比边界、简化条件）→ overview.html 全文（问题背景 / 核心机制 / 关键结论与边界）

## 来源核对记录

每条按 check.md 2.2 要求打开存档定位到页面标注位置，摘录原文片段或关键数值。共核对 24 条（≥10 条要求），覆盖正文各章与全部 11 个来源文件。

| 编号 | 核对位置（存档文件:行） | 看到的原文片段或关键数值 | 结果 |
|---|---|---|---|
| C1 | deepgemm-readme.md:114-116 | "Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory." | 相符 |
| C1/C3 | pr304-extracts.md:9-12 | "fusing & overlapping dispatch/linear 1/SwiGLU/linear 2/combine into a single mega-kernel… Only FP8 x FP4 MoE is supported / Requires PyTorch >= 2.9" | 相符 |
| C2 | deepgemm-readme.md:3, 11 | "unified, high-performance tensor core kernel library… fused MoE with overlapped communication (Mega MoE)"；News "2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL…" | 相符 |
| C4 | sm100_bf16_mega_moe.hpp:19,111；test_mega_moe.py:197,425 | `class SM100BF16MegaMoERuntime`、`static void sm100_bf16_mega_moe(`；`(deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)`；`--mma-type` default 'fp8xfp4', help "fp8xfp4 or bf16xbf16" | 相符 |
| C5 | sm100_fp8_fp4_mega_moe.cuh:78,1454-1457 | 行 78 `#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)`；行 1456 `DG_DEVICE_ASSERT(false and "This kernel only support sm_100f")` | 相符 |
| C6 | 同上:80,216-222,848,899 | 行 80 `using Allocator = cute::TMEM::Allocator2Sm;`；行 848 `umma_arrive_multicast_2x1SM`；行 899 `SM100_MMA_MXF8F6F4_2x1SM_SS::fma(`；216-222 为 TMEM 列数断言 | 相符 |
| C7 | 同上:142-146 | "NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)"，`a_dtype_t = float_e4m3_t`、`b_dtype_t = float_e2m1_unpacksmem_t`、`shared_b_dtype_t = float_e4m3_t` | 相符 |
| C8 | sm100_fp8_fp4_mega_moe.hpp:309-311；.cuh:55 | `LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, config.smem_size, 2)`；行 55 `__launch_bounds__(kNumThreads, 1)` | 相符 |
| C9 | .cuh:329,669,735,794,920,927 | 行 329 dispatch 分支起；行 673 "GEMM TMA load warp for tokens with SFA"；行 739 "GEMM TMA load warp for weights"；行 798 "GEMM MMA issue warp (only the leader CTA will run)"；行 924 "Do mainloop by the leader CTA"（scheduler.mainloop）；行 927 起 epilogue 分支 | 相符 |
| C10/N11 | sym_buffer.cuh:7,34-40 | 行 7 `constexpr static uint32_t kNumMaxRanks = 72;`；行 38 `int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);`（map 一次加法） | 相符 |
| C11 | .cuh:336-409,414-555 | 行 357 `atomicAdd_block(shared_storage.expert_token_count…)`、行 366 `ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value)`、行 376 `*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;`；行 461 注释 "Round-robin rank selection via iterative min-peeling"；行 548 `ptx::tma_load_1d(... kNumBytesPerPull)` | 相符 |
| C12 | mega_moe.hpp（heuristics）:208-209 | "// Pull: divide token bytes by 2 until <= kPullThreshold / constexpr int kPullThreshold = 4096;" | 相符 |
| C13 | sm100_fp8_fp4_mega_moe.hpp:187-189 | "// NOTES: L1 output and L2 activations are essentially the same tensor. // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile)..."（tensor_map_l1_output 用 l2_acts 构造） | 相符 |
| C14 | .cuh:1045,1053-1056,1071,1116,1128 | 行 1045 注释 "Apply SwiGLU: silu(gate) * up"；行 1053 `if constexpr (kActivationClamp != ...infinity())` 截断；行 1071 `activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);`（乘路由权重在 L1 epilogue）；行 1116 `__nv_fp8x4_e4m3`；行 1128 "Store SF ... as UE8M0" | 相符 |
| C15 | .cuh:1205,1274-1299 | 行 1205 注释 "L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink"；行 1280 读 `src_metadata`；行 1299 `*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;` | 相符 |
| C16 | .cuh:1313-1318,1323-1325,1404-1447 | 行 1313 "NVLink barrier ... ~4 us"；行 1323 注释 "Combine: reduce top-k results and write back"、行 1325 "~3 us"；1404-1417 float 寄存器累加、1430 `__float22bfloat162_rn`、1445 `tma_store_1d` 写 y | 相符 |
| C17 | scheduler_mega_moe.cuh:386-390；.cuh:1276-1278,1370；layout_mega_moe.cuh:433-435 | 行 387 "// Shared expert L1 tasks do not depend on dispatch."；.cuh 行 1276-1278 `dst_rank_idx = sym_buffer.rank_idx; ... dst_topk_idx = kNumTopk;`（共享专家占第 k+1 槽）；layout 行 434 `Buffer(bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), ...)` | 相符 |
| C18 | scheduler_mega_moe.cuh:15-45,83-89,309-314,316-350 | 行 15 "// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock"；行 83-89 `enum class BlockPhase { None, Linear1, Linear2, SharedLinear1, SharedLinear2 }`；行 312 `ptx::atomic_add(global_task_count_ptr, 1u)`；316-350 为 L1 预热后与 L2 交替的 get_next_task | 相符 |
| C20 | scheduler_mega_moe.cuh 全文（grep）；xstage-extracts.md:13,29 | scheduler 全文无 "expert wave"（仅 "L1 warmup waves" 指 CTA 波次）；xstage §2.2 "MegaMoE groups local experts into expert waves…"、使用注意"DeepGEMM 官方源码…没有 expert wave 一词" | 相符 |
| C21/N7 | test_mega_moe.py:223-235,322-326,411-425 | 基线 `ep_buffer.dispatch` + `gemm_fn`（m_grouped_fp8_fp4_gemm_nt_contiguous）+ `tilelang_ops.swiglu_apply_weight_to_fp8` + `ep_buffer.combine`；行 322 `assert torch.equal(fused_stats, baseline_stats)`、324 `assert torch.equal(fused_y, baseline_y)`、326 `calc_diff(fused_y, baseline_y) < 1e-8`；默认值 384 专家/topk 6/hidden 7168/中间维 3072/共享 1/8192 token；行 146 `assert hidden % 128 == 0 and intermediate_hidden % 128 == 0` | 相符 |
| C23 | test_mega_moe.py:411,443 | `--num-processes ... default: 8`；`torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)` | 相符 |
| C24/F2 | mega_moe.hpp（heuristics）:82,94,194,202 | 行 82 `num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;`；行 94 分支 `<= 64.5 → {2, 96, 16, 128, 2}`（注释 "bsz 512"）；行 98-100 else `→ 192`（"Prefill, or large EP decoding"）；行 85 `<= 8.5 → 16`（"RL long-tail rollout"）；行 194 `block_n = 128`；行 202 `gran_k = 32` | 相符 |
| C25 | xstage-extracts.md:17,19 | "this interpretation predicts at most approximately 1.5x speedup ... below the 1.56x reported by the implementation"；"Across 84 configurations, the resulting interleaved scheduler achieves a 1.18x geometric-mean speedup, a 1.17x median speedup, and a 1.62x maximum speedup over the Expert-Wave baseline" | 相符 |
| C26/C27/C28 | deepgemm-readme.md:180；pr304-extracts.md:28,34-35 | "`DG_COMM_KERNEL_DEBUG`: ... zero symmetric buffer before each Mega MoE call"；"Mega MoE is still under development and optimizations..."、"has nothing to do with internal model release"；Contributors 列 8 人（@LyricZhao @zheanxu @bucket-xv @RayWang96 @interestingLSY @kurisu6912 @xay5421 @yukuai26） | 相符 |
| C29/C30 | .cuh:308-311,313-322,403-409,662-668,1313-1318；layout_mega_moe.cuh:129-156 | 三个 tag `kBeforeDispatchPullBarrierTag=1 / kBeforeCombineReduceBarrierTag=2 / kAfterWorkspaceCleanBarrierTag=3`；行 316-318 `kNumDispatchRegisters = ... 48 : 96; kNumNonEpilogueRegisters = ... 40 : 88; kNumEpilogueRegisters = ... 208 : 160`、行 321 `<= 64512`；layout 129-156 NVLink barrier counter/signals 布局 | 相符 |
| N1-N5 | pr316-extracts.md:15-37 | V4-Flash 表：56.5/146.5/1283.1/4855.5 us，1.96/1.73/1.56/1.62x，互联 1/266/499/529；V4-Pro 表：108.1/369.6/2818.5/10655.2 us，1.61/1.54/1.50/1.54x，互联 1/182/393/417；规格 256/384 专家、top-k=6、hidden 4096/7168、中间维 2048/3072；"All values are averaged across 8 ranks"；zheanxu 评论 "the batch size listed is the number of tokens per rank... 512 × 8 = 4,096" | 与页面 5.1 表逐格一致 |
| N8 | .cuh:86,334,1345 | 行 86 `DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0)`；行 334 `DG_STATIC_ASSERT(kNumTopk <= 32)`；行 1345 `DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp")` | 相符 |
| N9 | pr316-extracts.md:39-47 | yiakwy 评论（2026-08-12，第三方）："For the moments, without TMEM, the performance is not significant, hence MegaMoE can be replaced with Fp4 EP V2 + FP8 DeepGeem + PDL in hopper platform." | 相符（页面已标注第三方陈述） |
| F1 | xstage-extracts.md:9；.cuh:1071 | "Combine returns expert outputs to the source rank and aggregates the top-k results using the router weights."；权重乘法位置见 C14 行 1071 | 相符 |

## 手算复算记录

1. 贯穿示例路由表（第 1 章）：按 E0/E1@rank0、E2/E3@rank1 复算每对 token-专家的所在 rank，跨 rank 对为 t0→E3、t1→E2、t1→E3、t2→E1、t3→E0、t3→E1，共 6/8，页面"6 个……四分之三"正确；路由表四行的"专家所在 rank"列逐格相符。
2. 第 4 章槽位走查：E2 处理 t1（rank0）、t2（本地）；E3 处理 t0、t1（均 rank0）与页面 4.1 文字相符；两图 L1 环形缓冲分组（rank0: E0:t0,t3 / E1:t2,t3；rank1: E2:t1,t2 / E3:t0,t1）与 combine 槽位（t0:E0,E3；t1:E2,E3；t2:E1,E2；t3:E0,E1）逐格复算相符。
3. 通信量估算（V4-Pro，h=7168，k=6）：dispatch 6×7168×1 B=43008 B=42 KB、combine 6×7168×2 B=84 KB 均可复算；512 token/rank dispatch 总量 43008×512=22020096 B，按十进制 22.0 MB、按二进制 21.0 MiB——页面 1.2 正文"约 21.5 MB"不可按任一口径复算（见问题 3）。
4. 每专家期望 token 数分档（第 4 章表）：512×8×6/384=64（≤64.5 档，源码注释 "bsz 512" 对应 block_m=96）、8192×8×6/384=1024（else 档 block_m=192）、1×8×6/384=0.125（≤8.5 档 block_m=16，"RL long-tail"），三行计算与档位结论均与源码相符。

## 其他机械验证

- 概念链接 8 个（deepseek-moe、moe-serving、swiglu、gpu-communication、deepep、gpu-execution-model、fp8-block-quant、mxfp4-qat）全部存在；本地资源（katex/prism）存在；overview.html 与 index.html 相互链接；`.dojo/scripts/validate.py wiki/megamoe` 返回 "validation ok"。
- 页面正文（移除代码块与脚本后）无 Unicode 数学字符直接出现（index.html 0 处；overview.html 1 处，见问题 1）。
- 两级问题块：页面级"核心问题"5 题、各章"本章问题"（1 章 2 题、2 章 2 题、3 章 2 题、4 章 3 题、5 章 2 题）每题均有独立解答折叠块，答案独立可读、与正文一致，核心问题答案均指明论证章节。
- 代码块均标注不可运行原因（"需 sm100 GPU 与多进程环境，本页不运行"、"非可运行代码"），改为静态审查：API 三步代码与 README Usage 逐行一致，伪代码与源码角色分支一致。
- N12 所列代码注释耗时参考页面正文确未引用，页面标注"备考"属实。

## 问题

- [轻微·格式] overview.html「关键结论与边界」第 3 条（"成立条件较硬"列表内）："发布版仅 FP8×FP4 精度组合"中的乘号为 Unicode 数学字符 ×，且与 index.html 全文的 "FP8xFP4"（字母 x）写法不一致｜引文依据：overview.html 原文"发布版仅 FP8×FP4 精度组合"；index.html 对应表述为"仅支持 FP8xFP4 组合"｜修复要求：将 overview.html 该处改为 "FP8xFP4"，与 index.html 统一｜修复：overview.html 该处已改为 "FP8xFP4"，与 index.html 统一｜复验：grep FP8×FP4 两文件均无结果，FP8xFP4 全站一致
- [轻微·格式] index.html 第 3 章图（"combine 缓冲（按 top-k 分槽）"2 处 <text>）与第 4 章图（同标签 2 处、"输出 y" 2 处 <text>）：SVG <text> 内使用 ASCII 数学近似写法（top-k、y），而正文中同一对象均写作 top-$k$、$y$；check.md 第 10 项要求图内公式写在 <foreignObject> 中由 KaTeX 渲染、<text> 内无 ASCII 近似｜引文依据：SVG 原文 `<text ...>combine 缓冲（按 top-k 分槽）</text>`、`<text ...>输出 y</text>`；同图其余公式标签（如 $B_0$、$t_0$）已用 foreignObject+KaTeX｜修复要求：将上述 6 处标签改为 foreignObject+KaTeX（top-$k$、$y$），或改写为不含数学符号的纯文字（如"按路由槽位分槽"、"输出结果"）｜修复：4 处"combine 缓冲（按 top-k 分槽）"的 <text> 去掉括注只留"combine 缓冲"（分槽细节由正文与 figcaption 说明，其中 figcaption 已含"每 token $k$ 个槽位"的 LaTeX 表述）；2 处"输出 y"改为 foreignObject 内"输出 $y$"｜复验：grep "top-k 分槽" 与 "输出 y<" 均为 0；headless Chrome 复测 katexInFO 36、全部标签零重叠
- [轻微·技术] index.html 1.2 节末段："仅 dispatch 方向就要发出约 21.5 MB"，同章问题 2 解答作"约 22 MB（$6\times 7168\times 512\approx 2.2\times 10^{7}$ 字节）"，两处不一致且 21.5 无法按任一单位口径复算：43008 B×512 = 22,020,096 B = 22.0 MB（十进制）或 21.0 MiB（二进制），21.5 为 KB 取二进制、MB 取十进制的混合换算｜引文依据：正文"每 rank 512 个 token 时，仅 dispatch 方向就要发出约 $21.5\,\mathrm{MB}$"；解答"$6\times 7168\times 512\approx 2.2\times 10^{7}$ 字节"；"构造示例"一节亦作"21.5 MB/rank@512"｜修复要求：将 1.2 节正文与"构造示例"一节的 21.5 MB 统一改为"约 22 MB"（与 $2.2\times 10^{7}$ 字节一致），或统一改为"约 21 MiB"并同步两处｜修复：1.2 节正文与"构造示例"节均已改为十进制 43 KB / 86 KB / 22 MB（含计算式与口径标注）。过程备注：第 1 轮对该处正文的修复因两个编辑并行执行被覆盖而实际未生效（第 1 轮记录有误），本轮重修并三处 grep 复验（21.5、42 KB、84 KB 均为 0）｜复验：43008×512=22,020,096 字节 ≈ 22 MB，与 Q2 解答、构造示例三处口径一致
- [轻微·可读性] index.html 5.3 节"硬件"条目：缩写 PDL（Programmatic Dependent Launch）首次出现，无全称、无解释、无概念链接或占位，读者无法得知其指代｜引文依据：原文"可用'FP4 EP v2 + FP8 DeepGEMM + PDL'的组合替代"；第 5 章问题 2 解答中同样出现｜修复要求：在 5.3 首次出现处补全称"（Programmatic Dependent Launch，程序化依赖启动）"或加链接/占位，问题 2 解答处可沿用｜修复：5.3 首次出现处已补全称与一句机制说明（"让后一个 kernel 在前一个结束前就开始的调度机制"）；问题 2 解答沿用缩写（已有全称在前）｜复验：全页 PDL 共 2 处，首次出现带全称
- [轻微·技术] index.html 4.1 节："这两步传的都是几十字节的元数据"与源码数量级不符：每专家计数为 8 字节（uint64 send_value，.cuh 行 364）、每 token-专家对源索引为 4 字节（uint32，行 376），为个位数字节而非几十字节；"元数据小、先行"的定性结论不受影响｜引文依据：.cuh 行 364 `const uint64_t send_value = (1ull << 32) | ...`、行 376 `*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;`（uint32）｜修复要求：将该句改为"几字节量级的元数据"或"每条计数与索引仅 4–8 字节"一类与源码相符的表述｜修复：已改为"每专家一个 8 字节计数、每个 token-专家对一个 4 字节源索引"｜复验：与 .cuh 行 364（uint64_t send_value）、行 376（uint32 写入）字节数一致

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 5
- 处置：修复。5 条轻微问题均不影响来源一致性与主线理解（来源抽查 24 条全部相符，手算除 21.5 MB 口径外全部复算通过，validate.py 通过），逐条修复后即可进入下一轮。
