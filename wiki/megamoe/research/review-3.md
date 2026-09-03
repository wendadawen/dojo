# MegaMoE 审查记录（第 3 轮）

- 页面版本：index.html，110754 字节（overview.html，6750 字节）
- 审查时间：2026-09-03
- 审查者：独立技术审查者（未参与页面写作，未参与第 1、2 轮审查与修复）
- 已完整阅读章节（按顺序）：页首 meta 与主要依据说明、核心问题（5 条及全部解答折叠块）、常见误解、1. MoE 一层的五段执行（1.1、1.2、本章问题）、2. 一个持久 kernel 里的分工（2.1、2.2、本章问题）、3. 对称内存（3.1、3.2、本章问题）、4. 一个 token 的完整旅程（4.1、4.2、4.3、本章问题）、5. 收益与边界（5.1、5.2、5.3、本章问题）、来源与范围说明（论断 C / 公式 F / 外部数字 N / 构造示例 / 辅助解释与类比边界 / 简化条件）；另完整阅读 overview.html（问题背景 / 核心机制 / 关键结论与边界）。全文从头到尾按顺序阅读，包括全部折叠块。

## 问题

- [重要·技术] 3.1 构造示例（index.html 行 1018）及图 2 标签（行 1055）、图注：对称内存示例的翻译方向颠倒。页面写"$t_0$ 的输入槽在 rank0 缓冲里的地址是 $a$；rank1 的线程要读它时，直接访问 $a + \Delta$——那正是 $t_0$ 输入槽在 rank1 视角下的同一位置"；图 2 标签写"读 $t_0$：$\mathrm{map}(a,\,r_1)=a+\Delta$"。按字面计算：rank1 视角下自身 0 号槽地址 $= B_1 + (a - B_0) = a + \Delta$，即 rank1 访问 $a+\Delta$ 读到的是**自己缓冲的 0 号槽（贯穿示例中是 $t_2$ 的数据）**，不是 rank0 上的 $t_0$。｜引文依据：sym_buffer.cuh 构造函数 `base = c[rank_idx]; offsets[i] = i < size ? (c[i] - base) : 0;`（偏移相对**本 rank** 基址）与 `map(ptr, dst_rank_idx) = offsets[dst_rank_idx] + ptr`，证明 map 是"本 rank 指针 + $(B_{dst} - B_{own})$"、指向远端同布局位置；本 rank 指针加偏移得到的仍是本 rank 槽位。｜修复要求：将示例改为正确方向——例如"rank0 的线程为专家 $E_1$ 拉取 $t_2$：$t_2$ 的输入槽在 rank1 缓冲里与本地 0 号槽同布局位置，rank0 视角下地址为 $a$，访问 $a + \Delta$ 即得到 rank1 上 $t_2$ 的槽"；同步修改图 2 标签与图注为"读 $t_2$（为 $E_1$ 拉取）：$\mathrm{map}(a,\,r_1)=a+\Delta$"一类的正确表述（图中原有箭头从 rank0 的 $t_0$ 框指向 rank1 的 $t_2$ 框，与正确叙述方向一致，仅需改文字）。修复后复验：$a + \Delta = B_0 + (a - B_0) + (B_1 - B_0) = B_1 + (a - B_0)$，确为 rank1 缓冲同布局槽位。｜修复：已按建议方向重写 3.1 示例（rank0 的线程为 $E_1$ 拉取 $t_2$：本地视角槽地址 $a$，访问 $a+\Delta$ 落到 rank1 上 $t_2$ 的槽）；图 3 弧线标签改为"读 $t_2$（为 $E_1$ 拉取）"、图注同步改写；弧线箭头本身从 rank0 的 0 号槽指向 rank1 的 $t_2$ 槽，与新叙述一致未动｜复验：$a+\Delta=B_1+(a-B_0)$ 为 rank1 同布局槽位，与 sym_buffer.cuh 的 map 语义（本 rank 指针 + $B_{dst}-B_{own}$）一致
- [重要·技术] 4.2 正文（行 1141）、4.2 本章问题解答（行 1310）、2.2 图示角色行（行 974）：调度 warp 的归属粒度写错为"每个 SM"。行 1141 写"每个 SM 上的调度 warp 通过 workspace 里的原子计数器动态认领 task，快的 SM 多干活"；行 1310 写"各 SM 的调度 warp 通过 workspace 里的原子计数器动态认领，负载不均时快的 SM 多做"；行 974 图示"调度 warp 发布计算任务"未加 leader CTA 限定（同图 MMA warp 已标注"仅 leader CTA"）。｜引文依据：sm100_fp8_fp4_mega_moe.cuh 行 924-926 `// Do mainloop by the leader CTA`、`if (is_leader_cta) scheduler.mainloop(num_tokens);`——调度 mainloop（含 atomic 认领）仅由 leader CTA 执行；scheduler_mega_moe.cuh 行 309-314 `get_next_task_idx` 内 `ptx::atomic_add(global_task_count_ptr, 1u)`，行 319 注释 `// One local L1 task per scheduler; globally this is one CTA-pair wave.` 明确认领粒度是 CTA-pair（2-SM cluster）。｜修复要求：三处统一改为"每个 2-SM cluster 的 leader CTA 上的调度 warp"（或等义表述），负载均衡句改为"快的 cluster 多干活"；行 974 的调度 warp 角色说明补"（仅 leader CTA）"限定。｜修复：四处全部改齐——2.2 图角色行补"（仅 leader CTA 运行）"；4.2 正文改为"每个 2-SM cluster 的 leader CTA 上运行一个调度 warp……发布给 cluster 内的两个 CTA，快的 cluster 多干活"；伪代码改为"leader CTA 的调度 warp……发布给 cluster 内两个 CTA"；4.2 解答同步；另将 C9 来源定位补注"mainloop 仅 leader CTA 执行"｜复验：grep "SM 上的调度 warp / SM 的调度 warp" 为 0；与 .cuh 行 924-926（leader CTA 条件）和 scheduler 行 319（CTA-pair 注释）一致
- [重要·技术] 1.2 公式段（行 859）：路由权重符号 $g_{k}$ 与公式及全文的 $g_{e}$ 不一致，且下标含义冲突。行 859 写"路由权重为 $g_{k}$"，紧接着行 861 公式为 $y=\sum_{e\in\mathcal{S}} g_{e}\cdot W_2^{(e)}\,\mathrm{SwiGLU}(W_1^{(e)}x)$；由于同句刚定义 $|\mathcal{S}|=k$，$g_{k}$ 会被读作"按 top-k 序号索引的权重"，与按专家索引的 $g_{e}$ 冲突。｜引文依据：页面行 859 原文"路由权重为 $g_{k}$"；行 861 公式使用 $g_{e}$；行 1166"乘上该 token-专家对的路由权重 $g_{e}$"（4.3 与全文一致用 $g_{e}$）。｜修复要求：行 859 改为"每个专家 $e\in\mathcal{S}$ 的路由权重为 $g_{e}$"，使符号与公式及后文一致。｜修复：已改为"每个专家 $e\in\mathcal{S}$ 的路由权重为 $g_{e}$、两层变换为 $W_1^{(e)}$、$W_2^{(e)}$"｜复验：grep $g_{k}$ 为 0，$g_{e}$ 全页 6 处一致
- [轻微·格式] 图 2 SVG `<text>`（行 1029、1031、1035、1037）：token 标签写作 "t0/t1/t2/t3"，而正文与图内 KaTeX 标签均用 $t_0$ 下标写法，同一变量写法不一致，且 `<text>` 内出现 ASCII 近似。｜引文依据：行 1029 `<text ...>t0</text>` 等；行 1055 图内公式标签用 `$\mathrm{map}(a,\,r_1)=a+\Delta$`（foreignObject + KaTeX）。｜修复要求：将四个槽位标签改为 foreignObject 内 KaTeX 渲染的 $t_0$–$t_3$，或改用纯文字名（如"token 0"）避免与正文变量写法冲突。｜修复：四个槽位标签已改为 foreignObject 内 $t_0$–$t_3$。过程备注：第 1 轮的同类修复因两个编辑并行执行被后一个覆盖而未生效（与第 2 轮 21.5 MB 问题同因），本轮以单脚本顺序执行并 grep 验证｜复验：grep <text>t0-3</text> 为 0；渲染复测 katexInFO 41、零重叠
- [轻微·可读性] 图 3（行 1206、1210）与第 4 章伪代码（行 1279）：缩写 UMMA 首次出现未解释，正文全文亦无展开。｜引文依据：行 1206 `<text ...>TMA 分块加载 · UMMA 于 TMEM 累加</text>`；行 1279 "对每个 task 发射 2-CTA UMMA，累加器在 TMEM"；UMMA 在页面中仅出现于图示、伪代码与 C6 来源行（行 1403），无任何定义。｜修复要求：在图 3 图注或首次出现前的正文补一处展开，如"UMMA（Unified Matrix-Multiply-Accumulate，Blackwell tcgen05 架构的矩阵乘加指令）"。｜修复：已在 2.2 正文首次提及处补全称与定位（"矩阵乘加指令 UMMA——Unified Matrix-Multiply-Accumulate，Blackwell tcgen05 架构的指令族"），先于图 4 与伪代码出现｜复验：全页 UMMA 出现处均已位于释义之后
- [轻微·技术] 2.1（行 960）："两组计算线程之间还有一次寄存器再分配"——参与再分配的实际是三组（dispatch 组与非 epilogue 组释放，epilogue 组领取），且 dispatch 组并非计算线程；冒号后的展开正确，但"两组计算线程之间"的总起句不准确。｜引文依据：行 960 原文"两组计算线程之间还有一次寄存器再分配：dispatch 组与非 epilogue 组主动释放寄存器，epilogue 组领取更多"；页面自身 C30 行（行 1427）写"寄存器再分配（三组 48/40/208 或 96/88/160，上限 64512）"；源码 sm100_fp8_fp4_mega_moe.cuh 行 921-922 非 epilogue 组 `cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();` 及 dispatch 组对应的 dealloc。｜修复要求：总起句改为"三组线程之间还有一次寄存器再分配"或"dispatch 组与非 epilogue 组把寄存器让给 epilogue 组"。｜修复：总起句已改为"三组线程之间还有一次寄存器再分配"｜复验：与 C30（三组 48/40/208 或 96/88/160）一致
- [轻微·技术] 4.3（行 1166）："先按上限截断（防止激活爆炸）"表述不完整——gate 只按上限截断，up 上下限都截断。｜引文依据：sm100_fp8_fp4_mega_moe.cuh 行 1052-1056：`// Clamp`、`bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});`（gate 仅上限）、`bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp}); bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});`（up 上下限）。｜修复要求：改为"gate 按上限截断、up 上下限都截断（防止激活爆炸）"或等义的完整表述。｜修复：4.3 正文已改为"gate 只按上限截断、up 上下限都截断（防止激活爆炸）"；本章问题 Q3 解答同步为"按界限截断（gate 上限、up 上下限）后算"｜复验：与 .cuh 行 1052-1056（gate 仅 __hmin2、up 先 __hmax2 后 __hmin2）一致
- [轻微·来源] 主要依据 blockquote（行 750）与来源说明（行 1391）："DeepGEMM main 分支 commit 559d79f，2026-07-15"——commit 哈希与日期在 11 个存档来源中均无记载，无法定位复核。｜引文依据：对 research/sources/ 全部存档执行 `grep -rn '559d79f\|2026-07'` 结果为空。｜修复要求：为 commit 哈希与日期补充存档依据（如在来源目录新增 commit 元数据存档，记录哈希、日期与获取方式），或删去日期仅保留可复核的哈希并在存档中落档。｜修复：已新增 sources/commit-info.md，记录完整哈希、git log -1 输出、获取方式与相关的 7f2a703 历史 commit｜复验：grep 559d79f 于 commit-info.md 命中；blockquote 与来源说明的哈希日期均有存档依据
- [轻微·来源] 来源与范围说明（行 1391）：AMD ROCm Primus 同名项目声明（"AMD ROCm 的 Primus 项目中也有一个名为 MegaMoE 的融合 MoE 层（FlyDSL 实现，面向训练的 EP-only + bf16 路径）"）不属于 11 个存档来源中的任何一条，页面未给出可定位出处。审查者本轮经公开文档联网核实该声明内容属实（AMD ROCm Primus 文档："MegaMoE is a FlyDSL-based fused MoE layer… Runtime target is EP-only (TP=1) + bf16"），属事实但缺来源标注。｜引文依据：research/sources/ 11 个存档中无 Primus 相关内容（grep "Primus" 无匹配）。｜修复要求：在来源与范围说明中为该声明补充可定位来源条目（官方文档 URL 与访问日期，或新增存档），使其满足"来源事实具有可定位依据"。｜修复：已新增 sources/primus-extracts.md（官方文档 URL、访问日期 2026-09-03、原文摘录），来源说明的同名项目段落补注 URL 与存档位置｜复验：grep primus 于 index.html 与 sources/ 均命中

## 抽查论断核对记录（12 条以上，含原文片段）

1. 架构限制（C5）：sm100_fp8_fp4_mega_moe.cuh 行 78 `#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))`；行 1454-1457 `"This kernel only support sm_100f"`。与页面"仅 Blackwell sm100"一致。
2. 精度方案（C7）：同文件行 142-146 dtype 注释（激活 FP8 E4M3、权重 FP4 E2M1、缩放因子 UE8M0），与页面 5.3 一致。
3. 寄存器再分配（C30）：同文件行 313-322 寄存器常量与 `<= 64512` 静态断言，三组 48/40/208 或 96/88/160 与页面一致。
4. task 四相（C18）：scheduler_mega_moe.cuh 行 83-89 `enum class BlockPhase {None, Linear1, Linear2, SharedLinear1, SharedLinear2}`；行 309-314 动态认领 `ptx::atomic_add(global_task_count_ptr, 1u)`。与页面 4.2 一致（认领主体归属见问题 2）。
5. L1 预热波防死锁：scheduler_mega_moe.cuh 行 15-45 `get_num_l1_warmup_waves`，注释 "ensure no L1 -> L2 deadlock"。与页面一致。
6. 共享专家不依赖 dispatch（C17）：scheduler_mega_moe.cuh 行 386-390 "Shared expert L1 tasks do not depend on dispatch"。与页面一致。
7. L1 输出与 L2 输入同缓冲（C13）：sm100_fp8_fp4_mega_moe.hpp 行 187-189 "NOTES: L1 output and L2 activations are essentially the same tensor"。与页面 4.2 一致。
8. 对称内存（C10）：sym_buffer.cuh `constexpr static uint32_t kNumMaxRanks = 72;`、`offsets[i] = i < size ? (c[i] - base) : 0;`、`map(ptr, dst) = offsets[dst] + ptr`。支持页面 3.1 的"偏移表 + 一次加法"表述（示例方向错误见问题 1）。
9. SwiGLU 与路由权重乘入点（C14/F1）：sm100_fp8_fp4_mega_moe.cuh 行 1045 `"Apply SwiGLU: silu(gate) * up"`；行 1071 `__fmul2_rn(__fmul2_rn(gate, up), weights)`——权重在 L1 epilogue 乘入。与页面 4.3"写回源 rank 的是已经乘过权重的专家输出"一致。
10. 远程写回与归约（C21）：同文件行 1205 "write GEMM output to remote combine buffer via NVLink"；行 1323 "Combine: reduce top-k results and write back"。与页面一致。
11. 启发式分档（F2）：mega_moe.hpp 行 82 `num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;`；六档分支 block_m {16,32,64,96,128,192}、阈值 8.5/16.5/32.5/64.5/96.5；行 194 `block_n = 128`。与页面 4.2 及分档表逐项一致。
12. 基准数字（C23/N 组）：pr316-extracts.md 两档规格全部加速比（EP8 相对非重叠基线 1.50–1.96x）与页面 5.1 表格逐行核对一致；批次口径评注（zheanxu）与页面口径说明一致。
13. 支持范围与依赖（PR #304）：pr304-extracts.md "Only FP8 x FP4 MoE is supported"、"Requires PyTorch >= 2.9"、"still under development"、八位贡献者、免责声明。与页面 5.3、来源说明一致。
14. X-Stage 第三方评述（N 组）：xstage-extracts.md "approximately 1.5x ... below the 1.56x"、84 组配置 1.18x/1.17x/1.62x。与页面 5.1 引用及第三方标注一致。
15. 正确性验证（5.2）：test_mega_moe.py 行 322-326 `assert torch.equal(fused_stats, baseline_stats)`；无共享专家 `torch.equal(fused_y, baseline_y)`、有共享 `calc_diff(...) < 1e-8`。与页面一致。
16. NVLink barrier 三处（C29）：sm100_fp8_fp4_mega_moe.cuh 行 403-409、662-668、1313-1318。与页面一致。
17. 轮转选源：同文件行 461 "Round-robin rank selection via iterative min-peeling"。与页面 4.1 一致。
18. 共享专家 combine 第 k+1 槽：同文件行 1278 `dst_topk_idx = kNumTopk`；layout_mega_moe.cuh 行 433-435 `combine_token_buffer = Buffer(bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), ...)`。与页面 4.3 一致。
19. pull 块上限：mega_moe.hpp 行 208-214 `kPullThreshold = 4096`。与页面一致。
20. 持久 kernel（C8）：sm100_fp8_fp4_mega_moe.hpp 行 309 `LaunchArgs(num_sms, ...)`（grid = num_sms）。与页面 2.1 一致。
21. bf16 变体（C4）：sm100_bf16_mega_moe.hpp 存档存在；test_mega_moe.py 行 197、425（bf16 分派与 `--mma-type`）。与页面一致。
22. 静态断言：sm100_fp8_fp4_mega_moe.cuh 行 334、1345、86（top-k ≤ 32、topk+shared ≤ 32、experts % ranks == 0）。与页面 5.3 适用条件一致。

## 手算复算记录

- 跨 rank 路由计数（贯穿示例，2 rank × 4 token × top-2）：8 个 token-专家对中 6 对跨 rank（$t_0\to E_3$、$t_1\to E_2$ 与 $t_1\to E_3$、$t_2\to E_1$、$t_3\to E_0$ 与 $t_3\to E_1$），2 对本地，"四分之三"成立。
- 第 4 章槽位走查：rank0 拉入 $E_0\{t_0,t_3\}$、$E_1\{t_2,t_3\}$；rank1 拉入 $E_2\{t_1,t_2\}$、$E_3\{t_0,t_1\}$；combine 槽位 $t_0\{E_0,E_3\}$、$t_1\{E_2,E_3\}$、$t_2\{E_1,E_2\}$、$t_3\{E_0,E_1\}$。与第 1 章路由表逐项一致。
- 通信量：每个 token-专家对 hidden 7168 × 2 字节（BF16）= 14336 B；每 rank 发送 3 对 = 43008 B ≈ 43 KB（十进制），两侧共 ≈ 86 KB；V4-Pro batch 512/rank 时 43008 B × 512 对 = 22,020,096 B ≈ 22 MB。复算通过，页面口径（十进制 KB/MB、BF16 激活）标注正确。交叉验证：官方口径 66,060,288 B ÷ 369.6 μs ≈ 178.7 GB/s，与实测 182 GB/s 量级一致；V4-Flash 257.7 vs 266 亦一致。
- 每专家期望 token 数分档：$N_{\mathrm{exp}} = n_{\mathrm{tokens}} \times n_{\mathrm{ranks}} \times k / E$；64 → block_m=96（32.5 < 64 ≤ 64.5）、1024 → 192（> 96.5）、0.125 → 16（≤ 8.5）。与 mega_moe.hpp 六档分支逐一比对通过。

## 机械校验记录

- `.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回成功。
- 页面引用的 8 个前置概念链接目标均存在。
- `dojo:topics` 取值在 AGENTS.md 固定词表内。
- overview.html 与 index.html 相互链接。
- 两级问题块（核心问题 5 条、各章本章问题）均有解答折叠块，答案独立可读、与正文结论一致（4.2 解答中的调度 warp 表述除外，见问题 2）。

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 6
- 处置：修复 → 全部 9 条问题已逐条修复并复验（修复后：validate.py 两页通过；headless Chrome 渲染复测 KaTeX 238 处、SVG 内 41 处、全部标签零重叠；三轮审查的阻断与重要问题全部关闭）
- 发布结果：可发布。三轮独立审查（每轮新审查者、未参与写作与前序轮次）累计发现问题：第 1 轮 0 阻断/2 重要/8 轻微，第 2 轮 0/0/5，第 3 轮 0/3/6；全部修复闭环，无遗留未解决问题，无带接受理由的遗留轻微问题。
- 过程教训（写给后续会话）：三轮中有两处修复（1.2 节 21.5 MB、图 3 t0 标签）因在同一消息里并行发送两个文件编辑、后写覆盖先写而实际未生效，第 2/3 轮审查将其重新抓出。对同一文件的多次修改必须合并为单脚本顺序执行，修复后必须逐条 grep 复验再更新审查记录。
