<!-- review-meta
round: 5
page: wiki/megamoe/index.html
reviewed_content_sha256: 441086607619958c
-->
# MegaMoE 审查记录（第 5 轮）

- 页面版本：b91623895383（index.html 工作树哈希 b9162389538361ad3a426c49fbd19dbcd603f58e）
- 审查时间：2026-09-13 20:21
- 审查者：编排者派发的独立审查者（未参与写作，未读取 research/ 下任何文件）
- 已完整阅读章节：引言 / 核心问题 / 常见误解；1（1.1、1.2、本章问题）；2（2.1、2.2、本章问题）；3（3.1、3.2、本章问题）；4（4.1、4.2、4.3、本章问题、两处代码折叠块、X-Stage 补充折叠块）；5（5.1、5.2、5.3、本章问题）；来源与范围说明（C/F/N 三表、构造示例、辅助解释与类比边界、简化条件及其限制）。另读 overview.html。
- 来源获取方式：GitHub API/raw（deepseek-ai/DeepGEMM，commit 559d79f = 2026-07-15）、PR #304/#316 页面与评论、arXiv:2607.23264（WebFetch 抓原文并下载 PDF 全文核对）、AMD Primus 文档站。

## 本轮核对结论（无问题项，记录核对依据）

- 基准表（行 654-661）逐格与 PR #316 表一/表二一致：V4-Flash 56.5/5/1311/1/1.96x、146.5/1056/3192/266/1.73x、1283.1/1928/998/499/1.56x、4855.5/2038/794/529/1.62x；V4-Pro 108.1/7/1758/1/1.61x、369.6/1098/4619/182/1.54x、2818.5/2304/1094/393/1.50x、10655.2/2438/692/417/1.54x。规格（256/384 专家、top-k 6、hidden 4096/7168、中间维 2048/3072）与"EP8、8 rank 平均、batch 为每 rank token 数"均见于 PR #316 正文与 zheanxu 2026-04-27 评论（"512 × 8 = 4,096"）。
- 算式全部可复算：6×7168=43008 B≈43 KB（行 242）；6×7168×2=86016 B≈86 KB；43008×512=22,020,096≈22 MB（行 242、259）；N_exp=512×8×6/384=64、8192×8×6/384=1024、1×8×6/384=0.125（行 475-477）。
- 源码定位（commit 559d79f）逐条核对，除下表 C9 一处外全部命中：行 55 `__launch_bounds__(kNumThreads, 1)`；行 78 `#if (… __CUDA_ARCH__ >= 1000)`；行 142-146 `// NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)` + `shared_b_dtype_t = cutlass::float_e4m3_t`；行 80 `cute::TMEM::Allocator2Sm`；行 848 `umma_arrive_multicast_2x1SM`、行 899 `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`；行 1045 `// Apply SwiGLU: silu(gate) * up`、行 1052-1056 gate 只按 `__hmin2` 上限截断而 up 上下限都截断、行 1071 `__fmul2_rn(__fmul2_rn(gate, up), weights)`；行 1205 `// L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink`；行 1323 `// Combine: reduce top-k results and write back`；行 313-322 寄存器 48/40/208 或 96/88/160 且上限 64512；行 308-311 三个 barrier tag；行 403-409 拉取前 barrier、行 1313-1318 归约前 barrier（`~4 us` 注释在 1313）；行 334 `kNumTopk <= 32`、行 1345 `kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u`、行 86 `kNumExperts % kNumRanks == 0`；行 89 `is_leader_cta = cute::block_rank_in_cluster() == 0`；行 798/924 "only the leader CTA"；行 600-668 dispatch 分支内的 workspace 清理段。
- `layout/sym_buffer.cuh` 行 7 `kNumMaxRanks = 72`，行 34-40 `map()` 即 `offsets[dst_rank_idx] + ptr`（一次加法）；`csrc/jit_kernels/heuristics/mega_moe.hpp` 行 82 `num_tokens * num_ranks * num_topk / num_experts`、行 83-101 六档（block_m 16/32/64/96/128/192，与页面 3 行手算的 96/192/16 全部吻合）、行 194 `block_n = 128`、行 205-206 `num_dispatch_threads = 128`/`num_non_epilogue_threads = 128`、行 208-209 `kPullThreshold = 4096`；`scheduler/mega_moe.cuh` 行 15 deadlock 注释、行 83-88 四相 BlockPhase、行 387 "Shared expert L1 tasks do not depend on dispatch."、行 309-314 原子认领；`tests/test_mega_moe.py` 行 197/425（bf16 分派与 `--mma-type bf16xbf16`）、行 322-326（`torch.equal` 统计与无共享专家逐位、`< 1e-8`）、行 146（`hidden % 128 == 0`）、行 411（`--num-processes` 默认 8）、行 223-262（DeepEP + 分组 GEMM + TileLang SwiGLU 非重叠基线）。
- X-Stage：PDF 全文核对——"predicts at most approximately 1.5× speedup over the serial stage sum, below the 1.56× reported by the implementation"；"1.18× geometric-mean speedup, a 1.17× median speedup, and a 1.62× maximum speedup over the Expert-Wave baseline"（84 configurations）；源码全文无 "expert wave" 一词、仅 `get_num_l1_warmup_waves` 等指 CTA 波次——与行 614 表述一致。
- PR #304 合并 2026-04-17、PR #316 合并 2026-04-24（GitHub API），与 N6 一致；八位贡献者用户名与 C28 的"八位"一致；README（该 commit）行 116/120/121-137/180 与 C1/C3/C22/C26 及行 399-418 代码块逐行一致；AMD Primus "MegaMoE" 技术指南页确存在（04-technical-guides/mega-moe.html，"FlyDSL-based fused MoE layer … Runtime target is EP-only (TP=1) + bf16"），与「来源与范围说明」同名项目说明一致。
- `dojo:topics`=推理系统,并行与通信、`dojo:tag`=MoE 均在词表内；`.dojo/scripts/validate.py wiki/megamoe/index.html` 返回 `validation ok`。

## 问题

- [轻微·图示] 行 237（第 1 章时间线图内底部说明文字）：该 `<text>` 右边界超出 SVG 画布，尾部被裁切。｜引文依据：headless Chrome 实测该 `<text>` 的 getBBox 为 x=110、宽 661（右边界 771），而该 SVG 的 viewBox 宽仅 680（`<svg viewBox="0 0 680 412" …>`，CSS `.diagram svg { max-width:100%; height:auto }`）；渲染截图显示该行在"…计算接续（示意图，非实"处断掉，"测 trace）。"不可见。｜修复要求：把该行文字缩短或拆分，使实测 getBBox 右边界 ≤ 680（也可把"示意图，非实测 trace"移入同一 figure 的 figcaption）；改完重测右边界。｜修复：｜复验：
- [轻微·来源] 行 726（来源表 C9 "warp 专用化六类角色分工" 的 dispatch 区间）：dispatch 区间与源码不符。｜引文依据：`sm100_fp8_fp4_mega_moe.cuh` 中 `if (warp_idx < kNumDispatchWarps) {` 在第 329 行，其分支体直到第 668 行才闭合（第 669 行为 `} else if (warp_idx == kNumDispatchWarps) {`）；行 600-668（`// Clean workspace for the next usage, and also do cumulative stats`，含行 662-668 的 `comm::nvlink_barrier<…, kAfterWorkspaceCleanBarrierTag>`）属于 dispatch 分支，但表中"329-599（dispatch）"未覆盖，其余五段区间（669-734、735-793、794-919、920-926、927-1453）也未覆盖 600-668。｜修复要求：把 dispatch 写作 329-668，或另列 600-668 为 workspace 清理段。｜修复：｜复验：
- [轻微·来源] 行 745（来源表 C28）：括号内的英文姓名在所引来源中定位不到。｜引文依据：PR #304 描述只列出 8 个 GitHub 用户名——`@LyricZhao @zheanxu @bucket-xv @RayWang96 @interestingLSY @kurisu6912 @xay5421 @yukuai26`，无任何英文全名；页面写"（LyricZhao/Chenggang Zhao 等）"，其中"Chenggang Zhao"在 C28 标注的 "PR #304 Contributors" 中不存在。｜修复要求：删除该英文名，或补一条能定位到该姓名的来源。｜修复：｜复验：
- [轻微·表述] 行 457（4.1 节末）：正文出现第一人称。｜引文依据：check.md 第 2.2 条 12 项把"会话指代（我、我们、你）"列为不合格表述；该句为 `rank0 也把"我贡献了哪些 token-专家对"的源索引写了过来`。｜修复要求：改为非人称表述，例如"rank0 也把本 rank 有哪些 token-专家对的源索引写了过来"。｜修复：｜复验：

## 轻微问题的接受理由（若本轮不修）

- 图示裁切：仅影响图内一行说明的可读性，图中"示意图"三字仍可见，"非实测 trace"另有「来源与范围说明」第 788 行与 figcaption 兜底，不改变任何结论；但属可见的断句，建议随手修掉。
- C9 区间：600-668 段描述的是收尾清理，不进入正文数据流论证，定位偏差不影响任何结论的核对。
- C28 英文名与行 457 第一人称：均不影响事实与主线理解，仅降低来源可核性与文字一致性。

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布（阻断与重要均为 0；4 项轻微已列明并给出接受理由，其中行 237 图示裁切建议本轮随手修复）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
