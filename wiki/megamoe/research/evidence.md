# MegaMoE 核心论断与证据

来源文件（均在 `wiki/megamoe/research/sources/`，拷贝自 DeepGEMM main @ 559d79f，2026-07-15 克隆）：

- S-KERNEL = `sm100_fp8_fp4_mega_moe.cuh`（deep_gemm/include/deep_gemm/impls/，kernel 本体，1460 行）
- S-RT = `sm100_fp8_fp4_mega_moe.hpp`（csrc/jit_kernels/impls/，JIT 启动器，319 行）
- S-HEUR = `mega_moe.hpp`（csrc/jit_kernels/heuristics/，配置启发式）
- S-SCHED = `scheduler_mega_moe.cuh`（源仓库 deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh，任务调度器）
- S-SYM = `sym_buffer.cuh`（deep_gemm/include/deep_gemm/layout/，对称内存）
- S-LAYOUT = `layout_mega_moe.cuh`（源仓库 deep_gemm/include/deep_gemm/layout/mega_moe.cuh，缓冲与 workspace 布局；与 S-SCHED 源文件同名不同目录，存档已改名区分）
- S-TEST = `test_mega_moe.py`（tests/，443 行）
- S-README = `deepgemm-readme.md`
- S-PR304 = `pr304-extracts.md`（PR #304 原文摘录）
- S-PR316 = `pr316-extracts.md`（PR #316 原文与关键评论摘录）
- S-XSTAGE = `xstage-extracts.md`（arXiv:2607.23264 §1/§2.2/§2.4 原文摘录）

版本时间线：PR #304（2026-04-17 合并，commit 7f2a703）引入 Mega MoE；PR #316（2026-04-24 合并）加基准；本地核对版本 main @ 559d79f（2026-07-15）。X-Stage 论文实验用 commit 7f2a703。

## C 论断（定义与机制）

- C1：Mega MoE 把 EP dispatch、Linear-1（FP8×FP4）、SwiGLU、Linear-2（FP8×FP4）、EP combine 融合为单个 mega-kernel，重叠 NVLink 通信与 Tensor Core 计算；要求多进程启动与对称内存。来源：S-README "Mega MoE" 节首句；S-PR304 New features 首条。条件：无。状态：已确认。
- C2：DeepGEMM 自我定位是统一的高性能 Tensor Core kernel 库，Mega MoE 是其中"带通信重叠的融合 MoE"组件；2026-04-16 News 条目宣布 Mega MoE、FP8xFP4 GEMM、FP4 Indexer、PDL 等。来源：S-README 首段与 News 节。状态：已确认。
- C3：发布说明声明当时仅支持 FP8×FP4 MoE、要求 PyTorch >= 2.9。来源：S-PR304（"Only FP8 x FP4 MoE is supported"、"Requires PyTorch >= 2.9"）。条件：PR #304 时点。状态：已确认（后续 main 中出现 bf16 变体，见 C4）。
- C4：当前 main 存在 bf16 路径：源码文件 sm100_bf16_mega_moe.hpp（csrc/jit_kernels/impls/ 目录清单）、测试支持 `--mma-type bf16xbf16`（S-TEST 参数定义）、启动器按 mma_type 分派（S-RT parse_mma_kind）。状态：已确认（仓库结构 + 测试参数）。
- C5：kernel 仅为 sm100（Blackwell）实现：代码以 `#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)` 守卫，else 分支断言 "This kernel only support sm_100f"。来源：S-KERNEL 行 78、行 1454-1457。状态：已确认。
- C6：kernel 使用 Blackwell 专有特性：TMEM（`cutlass::TMEM::Allocator2Sm` 分配、`SM100_TMEM_LOAD` 读取）、2-CTA UMMA（`SM100_MMA_MXF8F6F4_2x1SM_SS::fma`）、2-SM cluster（cluster_size=2、`umma_arrive_multicast_2x1SM`）。来源：S-KERNEL 行 80、216-222、270-271、848、899；S-HEUR cluster_size=2。状态：已确认。
- C7：激活为 FP8 E4M3（`cutlass::float_e4m3_t`）、路由权重为 FP4 E2M1（`float_e2m1_unpacksmem_t`，smem 中解包为 8-bit）；共享专家权重为 FP8 E4M3。来源：S-KERNEL 行 142-146 注释与类型定义。状态：已确认。
- C8：持久 kernel 形态：grid 维度 = num_sms（每 SM 一个 block），块内线程数 = dispatch + 非 epilogue + epilogue 三组线程之和；调度由 kernel 内部完成。来源：S-RT 行 `LaunchArgs(num_sms, ..., config.smem_size, 2)` 与第 2.3 节；S-KERNEL 行 55 `__launch_bounds__(kNumThreads, 1)`。状态：已确认。
- C9：warp 专用化分工：dispatch 线程组（kNumDispatchThreads=128，读路由、计数、拉取）、TMA A 加载 warp（激活+SFA）、TMA B 加载 warp（权重+SFB）、MMA 发射 warp（仅 leader CTA）、调度 warp（scheduler.mainloop，仅 leader CTA 执行）、epilogue 线程组（SwiGLU/量化/远程写回/combine 归约）。来源：S-KERNEL 行 329-599（dispatch）、669-734（TMA A）、735-793（TMA B）、794-919（MMA）、920-926（调度）、927-1453（epilogue）；S-HEUR 行 `num_dispatch_threads = 128`、`num_non_epilogue_threads = 128`。状态：已确认。
- C10：对称内存模型：SymBuffer 持有本 rank 基址与全部 rank 的偏移表，`map(ptr, dst_rank) = ptr + offsets[dst_rank]`，把本 rank 地址翻译为任意 rank 的等价地址；rank 数内嵌于地址表（`num_ranks = sym_buffer_ptrs.size()`），最大 72 rank。来源：S-SYM 行 7-42；S-RT `layout::SymBuffer<>(sym_buffer_ptrs, rank_idx)` 与 num_ranks 推导。状态：已确认。
- C11：dispatch 侧机制分三步：① 每个 rank 的 dispatch 线程组读本地 token 的 topk_idx、统计每专家 token 数，并通过 NVLink 远端原子加把计数累加到专家所属 rank 的 workspace；② 将每个 (token, topk) 的源索引写往目标 rank 的 workspace；③ 经 NVLink barrier 后，接收侧按专家分组、以 round-robin（迭代最小剥离）决定每个槽位从哪个源 rank 拉取，用 TMA 按 kNumBytesPerPull 字节的块把 token 数据从远端对称缓冲搬入本地环形缓冲。来源：S-KERNEL 行 336-409（计数与源索引）、414-555（pull 循环与分块 TMA）、461-509（min-peeling 注释 "Round-robin rank selection via iterative min-peeling"）。状态：已确认。
- C12：pull 粒度：num_bytes_per_pull 从 hidden × 每元素字节数出发反复除 2 直到不超过 4096 字节（kPullThreshold）。来源：S-HEUR `// Pull: divide token bytes by 2 until <= kPullThreshold` 及代码。状态：已确认。
- C13：L1 输出与 L2 输入是同一张量（l2_acts 兼作 L1 输出缓冲），SwiGLU 后宽度减半（BLOCK_N/2），swizzle 模式相应减半。来源：S-RT 原始注释 "NOTES: L1 output and L2 activations are essentially the same tensor. Post-SwiGLU output has half the N width..."；S-KERNEL `L1_OUT_BLOCK_N = BLOCK_N / 2`。状态：已确认。
- C14：L1 epilogue 在片上完成 SwiGLU 与重新量化：从 TMEM 读 gate/up 对 → clamp（activation_clamp）→ silu(gate)*up → 乘该 token-专家对的路由权重 → amax 归约 → 量化为 FP8 E4M3（SF 为 UE8M0）→ 经共享内存 TMA store 写回 L2 输入缓冲。来源：S-KERNEL 行 995-1172（行 1045 注释 "Apply SwiGLU: silu(gate) * up"、行 1053-1057 clamp、行 1100-1126 cast to FP8 E4M3、行 1128-1153 SF UE8M0）。状态：已确认。
- C15：L2 epilogue 远程写回：L2 GEMM 输出从 TMEM 读出转 BF16 写入共享内存后，每个 token 行查 dispatch 阶段记录的源元数据（rank_idx、token_idx、topk_idx），通过 `sym_buffer.map` 直接 NVLink 远程写入源 rank 的 combine_token_buffer 对应 topk 槽。来源：S-KERNEL 行 1205 注释 "L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink"、行 1274-1299（metadata 读取与 `*sym_buffer.map(dst_ptr, dst_rank_idx) = packed`）。状态：已确认。
- C16：combine 归约：全部 L2 写回完成后经 NVLink barrier，epilogue 线程按 token 遍历：每 warp 的 lane 各读一个 topk 槽索引（共享专家占第 kNumTopk 槽），双缓冲 TMA 加载各槽数据，在 float 寄存器中累加，转 BF16 后 TMA store 写入输出 y。来源：S-KERNEL 行 1313-1452（行 1323 注释 "Combine: reduce top-k results and write back"、行 1368-1370 topk 槽读取、行 1404-1421 float 累加、行 1443-1449 TMA store 到 y）。状态：已确认。
- C17：共享专家路径不经 dispatch：其 tensormap 的 token 维用 num_max_tokens_per_rank（本地），调度器中 SharedLinear1 任务"不依赖 dispatch"可先行；combine 时共享专家结果作为第 kNumTopk 槽参与归约。来源：S-RT shared tensormap 构造（token 维 num_max_tokens_per_rank）；S-SCHED 行 386-390 注释 "Shared expert L1 tasks do not depend on dispatch"；S-KERNEL combine 槽位（lane_idx == kNumTopk 时取共享槽）。状态：已确认。
- C18：任务调度：调度单元是 TaskInfo（block_phase ∈ {Linear1, Linear2, SharedLinear1, SharedLinear2}、专家号、M 块号、N 簇号等）；先发布若干 L1 预热波（避免 L1→L2 依赖死锁），之后 L1/L2 交替发布；各 SM 的调度 warp 通过 workspace 中的原子计数器动态认领 task。来源：S-SCHED 行 83-134（TaskInfo/BlockPhase）、行 15-45（warmup waves 与死锁避免注释）、行 316-350（L1/L2 交替逻辑）、行 309-314（atomic_add 认领）。状态：已确认。
- C19：环形缓冲衔接生产消费：L1/L2 数据缓冲按 kNumRingTokens 容量循环使用，配套 full/empty 计数器（ld_acq 自旋等待）；workspace 布局中另有 dispatch 拉取源索引区与 combine 推送源索引区。来源：S-KERNEL 行 525-531（等 L1 empty）、692-704（等 full）；S-LAYOUT 行 53-56、行 113-117（"Dispatch pulling source token-topk"、"Combine push source indices" 注释）。状态：已确认。
- C20：官方源码中没有 "expert wave" 术语；调度以 task 为单位。"expert waves" 是 X-Stage 论文对调度结构的描述（把本地专家分组组织局部性与执行，一个 wave 内先 Linear-1 后 activation+Linear-2，不同 wave 的 Linear-1 无神经网络依赖）。来源：S-SCHED 全文无 wave 于专家粒度的用语（仅 L1 warmup waves 指 CTA 波）；S-XSTAGE §2.2 第三段。条件：引用 wave 概念时必须标注为论文描述。状态：已确认。
- C21：正确性验证：融合实现与基线（DeepEP dispatch/combine + DeepGEMM grouped GEMM + TileLang SwiGLU 的非重叠流水线）对比——无共享专家时 `torch.equal` 逐位一致，有共享专家时差异 < 1e-8；专家接收统计逐位一致。来源：S-TEST 断言代码（`assert torch.equal(fused_stats, baseline_stats)` 等）与基线流程。状态：已确认。
- C22：使用方式三步：`get_symm_buffer_for_mega_moe` 分配对称缓冲（含 group、专家数、每 rank 最大 token 数、topk、hidden、intermediate hidden）→ `transform_weights_for_mega_moe` 把权重转为所需布局 → 拷入输入（x/x_sf/topk_idx/topk_weights）后调用 `fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)`，输出 y 为 BF16。来源：S-README Usage 代码块；S-TEST 完整调用。状态：已确认。
- C23：多进程形态：测试默认 `torch.multiprocessing.spawn` 启动 8 进程（每 rank 一进程），各 rank 用不同随机种子生成数据；`--local-rank-idx` 支持单进程模式（NCU 剖析用）。来源：S-TEST `__main__` 与 init 逻辑。状态：已确认。
- C24：block 配置按"每专家期望 token 数"分六档选择 block_m（16/32/64/96/128/192），档位注释对应 RL 长尾、小批解码、中批、大批、中 EP、prefill/大 EP 等场景；block_n 固定 128；SF 粒度 32（kGranK）。来源：S-HEUR get_block_config_for_mega_moe。状态：已确认。
- C25：X-Stage 论文分析：Mega MoE 公布的时间线把 Combine 视为单一通信阶段，保守的 completion-coupled 解释预测加速上限约 1.5x，低于实测 1.56x，说明发送方在 remote store 发出后即可继续后续工作；论文据此提出跨 wave 交错 Linear-1/Linear-2 的改法，在其 84 组配置上比原调度再快 1.18x（几何均值）/1.62x（最大）。来源：S-XSTAGE §1、§2.2、§2.4。条件：论文实验环境（commit 7f2a703），非官方 main 的实现。状态：已确认（第三方）。
- C26：调试支持：环境变量 DG_COMM_KERNEL_DEBUG=1 时每次调用前把对称缓冲清零。来源：S-README Utilities 节 `DG_COMM_KERNEL_DEBUG` 条目。状态：已确认。
- C27：仍在开发声明："Mega MoE is still under development and optimizations, stay tuned and optimization ideas are welcome!"，且声明该发布与内部模型发布无关。来源：S-PR304 Additional notes。状态：已确认。
- C28：贡献者：Mega MoE 由 LyricZhao（Chenggang Zhao）、zheanxu、bucket-xv、RayWang96、interestingLSY、kurisu6912、xay5421、yukuai26 贡献。来源：S-PR304 Contributors。状态：已确认。
- C29：kernel 内跨 rank 屏障由 NVLink barrier 原语实现（grid sync + 跨 rank 计数信号 + grid sync），dispatch pull 前、combine 归约前、workspace 清理后各有一次。来源：S-KERNEL 行 308-311（barrier tags 定义）、行 403-409、1313-1318；S-LAYOUT 行 129-156（计数器布局）。状态：已确认。
- C30：寄存器在角色间再分配：dispatch/非 epilogue 组用 warpgroup_reg_dealloc 释放寄存器，epilogue 组用 warpgroup_reg_alloc 获取更多（三组配额 48/40/208 或 96/88/160，按每 rank 专家数选择），总和不超过 64512。来源：S-KERNEL 行 313-322。状态：已确认。

## F 公式（公式与来源）

- F1：MoE 路由专家输出聚合：token 的层输出为各被选专家输出按路由权重加权求和（combine 按 router 权重聚合 top-k 结果）。来源：S-XSTAGE §2.2 首段（"Combine returns expert outputs to the source rank and aggregates the top-k results using the router weights"）；与 S-KERNEL combine 归约实现（乘权重在 L1 epilogue 行 1071 完成：`activation_values = silu(gate)*up*weight`，combine 阶段为纯求和）一致。注意：页面写公式时按实现写成 $y=\sum_k g_k\,W_2^{(k)}\,\mathrm{SwiGLU}(W_1^{(k)}x)$，并注明权重乘法位置在 L1 epilogue。状态：已确认。
- F2：每专家期望 token 数：`num_expected_tokens_per_expert = num_tokens * num_ranks * num_topk / num_experts`（num_tokens 为本 rank token 数）。来源：S-HEUR get_block_config_for_mega_moe 首行。状态：已确认。
- F3：单 token dispatch 通信量（本页推导，构造示例用）：每个 token 的激活被送往其 top-k 个专家所在 rank，发送字节约为 $k \times h \times b_{\mathrm{act}}$（$k$ 为 top-k，$h$ 为 hidden，$b_{\mathrm{act}}$ 为激活每元素字节数；FP8 为 1）。来源：由 EP dispatch 语义直接推出；数值代入用 PR #316 规格。状态：已确认（推导，页面标注为推导而非官方公式）。
- F4：对称缓冲 token 池容量上界：`num_max_pool_tokens = align(num_ranks * num_max_tokens_per_rank * min(num_topk, num_experts_per_rank) + num_experts_per_rank * (192 - 1), 384)`（最坏情况接收 token 数 + 每 expert 的 BLOCK_M 对齐 padding，对齐到候选 block_m 的 LCM）。来源：S-LAYOUT get_num_max_pool_tokens（kMaxCandidateBlockM=192、kLCMCandidateBlockM=384）。状态：已确认。

## N 数字（外部数字与实验条件）

- N1：基准配置一（V4-Flash）：256 专家、top-k=6、hidden 4096、intermediate hidden 2048；EP8；batch size = 每 rank token 数；数值为 8 rank 平均。来源：S-PR316 表一。状态：已确认。
- N2：V4-Flash 结果：batch 1 → 56.5 us、5 TFLOPS、显存带宽 1311 GB/s、互联 1 GB/s、1.96x；batch 512 → 146.5 us、1056 TFLOPS、3192 GB/s、266 GB/s、1.73x；batch 8192 → 1283.1 us、1928 TFLOPS、998 GB/s、499 GB/s、1.56x；batch 32768 → 4855.5 us、2038 TFLOPS、794 GB/s、529 GB/s、1.62x。加速相对 legacy（DeepEP+grouped GEMM+TileLang 非重叠基线）。来源：S-PR316 表一 + S-TEST 基线定义。状态：已确认。
- N3：基准配置二（V4-Pro）：384 专家、top-k=6、hidden 7168、intermediate hidden 3072；EP8。来源：S-PR316 表二。状态：已确认。
- N4：V4-Pro 结果：batch 1 → 108.1 us、7 TFLOPS、1758 GB/s、1 GB/s、1.61x；batch 512 → 369.6 us、1098 TFLOPS、4619 GB/s、182 GB/s、1.54x；batch 8192 → 2818.5 us、2304 TFLOPS、1094 GB/s、393 GB/s、1.50x；batch 32768 → 10655.2 us、2438 TFLOPS、692 GB/s、417 GB/s、1.54x。来源：S-PR316 表二。状态：已确认。
- N5：batch size 口径官方确认：表内 batch size 为每 rank token 数，EP8 下 512/rank = 全节点 4096。来源：S-PR316 评论（zheanxu 2026-04-27）。状态：已确认。
- N6：时间线版本：2026-04-16 News 宣布；PR #304 2026-04-17 合并；PR #316 2026-04-24 合并。来源：S-README News、S-PR304、S-PR316。状态：已确认。
- N7：测试默认参数：384 专家、top-k 6、hidden 7168、intermediate hidden 3072、共享专家 1、每 rank 最大 token 8192、mma-type fp8xfp4；FP8 路径要求 hidden、intermediate_hidden、shared_intermediate_hidden 均被 128 整除。来源：S-TEST argparse 默认值与约束注释。状态：已确认。
- N8：kernel 约束：kNumTopk <= 32；kNumTopk + 共享专家数 <= 32（combine 单 warp 遍历约束）；专家数被 rank 数整除。来源：S-KERNEL 行 334、1345、86 静态断言。状态：已确认。
- N9：Hopper（sm90）状况：官方未提供 sm90 的 Mega MoE 收益数据；第三方评论（yiakwy，2026-08-12）称原 PR 面向 Blackwell（NVFP4），代码用 `__CUDA_ARCH__ >= 1000` 守卫并大量使用 TMEM，无 TMEM 时性能不显著，Hopper 可用 FP4 EP v2 + FP8 DeepGEMM + PDL 替代；sm90 移植在推进（链接见摘录）。来源：S-PR316 评论。状态：第三方陈述，页面必须标注来源性质。其中 `__CUDA_ARCH__ >= 1000` 守卫已由 C5 独立确认。
- N10：X-Stage 论文数字：84 组配置上 interleaved 调度相对原 Expert-Wave 调度 1.18x 几何均值 / 1.17x 中位 / 1.62x 最大。来源：S-XSTAGE §1。状态：已确认（第三方，其实验修改版）。
- N11：SymBuffer 最大 rank 数 72。来源：S-SYM 行 7。状态：已确认。
- N12：代码注释中的耗时参考：取 SM offset 约 6.5 us、写源索引约 2 us（512 token）、NVLink barrier 约 4 us、combine 每 token 每 topk 约 3 us。来源：S-KERNEL 行 361、370、1313、1325 注释。状态：已确认（注释值，标注为代码注释）。

## 存在冲突或证据不足、不写入正文的项

- "C/B 比值不超过 6144 FLOPs/Byte 时通信可被计算完全覆盖"：仅见于中文百科词条，未能溯源到 DeepGEMM 官方 PR/README/源码，不采用。
- "在英伟达 GPU 和华为昇腾 NPU 平台上得到验证"（百科）：PR #316 未说明测试硬件型号，源码仅见 sm100 CUDA 路径，不采用。
- "MegaMoE2 已作为 DeepGEMM 组件开源"（百科）：官方无 MegaMoE2 命名，不采用该名称。
- vLLM/SGLang 集成状态：yiakwy 评论提到 sglang 侧 sm90 移植 PR，但非官方发布信息，仅作为 N9 的一部分引用且标注第三方。
