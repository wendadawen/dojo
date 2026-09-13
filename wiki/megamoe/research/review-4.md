<!-- review-meta
round: 4
page: wiki/megamoe/index.html
reviewed_content_sha256: 600fdd6860b50567
-->
# MegaMoE 审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 a2e9a30558e2fead163b0a344c579702d9ff5e99（overview.html 59100d3df8ec45d1b6b7902557a8f7bba97c873b）
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 常见误解 / 1. MoE 一层的五段执行（1.1、1.2、本章问题）/ 2. 一个持久 kernel 里的分工（2.1、2.2、本章问题）/ 3. 对称内存（3.1、3.2、本章问题）/ 4. 一个 token 的完整旅程（4.1、4.2、4.3、本章问题）/ 5. 收益与边界（5.1、5.2、5.3、本章问题）/ 来源与范围说明（含 C/F/N 表、构造示例、辅助解释与类比边界、简化条件及其限制）
- 外部来源核对方式：DeepGEMM README 原文（WWW/仓库）、PR #304、PR #316、arXiv:2607.23264 经 WebFetch/WebSearch 抓取原文逐条比对；本页 research/sources/ 下的 6 个 .md 摘录逐份通读

## 问题

- [重要·来源] 来源与范围说明（index.html 行 62、711）：页面称 kernel/测试源码"均存档于本页 research/sources/"，第 711 行更明确写"关键文件已拷贝存档于 research/sources/（…存档分别命名为 scheduler_mega_moe.cuh 与 layout_mega_moe.cuh），下列定位均可在存档中复核"；但该目录现存只有 6 个 .md 摘录（commit-info、deepgemm-readme、pr304-extracts、pr316-extracts、primus-extracts、xstage-extracts），C/F/N 表引用的源码快照 sm100_fp8_fp4_mega_moe.cuh、sm100_fp8_fp4_mega_moe.hpp、scheduler_mega_moe.cuh、layout/mega_moe.cuh、layout/sym_buffer.cuh、heuristics/mega_moe.hpp、sm100_bf16_mega_moe.hpp、test_mega_moe.py 均已不在仓库。据此，C4–C20、C29、C30、F1（"权重乘法位置在 L1 epilogue（.cuh 行 1071）为源码事实"部分）、F2、N7、N8、N11 这些以"文件名+行号"为唯一依据的论断，在本仓库内已无任何可复核的引文载体，"可在存档中复核"的承诺不成立。｜引文依据：research/measured.md 登记表列 `sources/sm100_fp8_fp4_mega_moe.cuh | 0.1 KB | 模型源码快照`、`sources/scheduler_mega_moe.cuh | 0.0 KB | 模型源码快照`（表头注"现已从仓库移除"）；`find wiki/megamoe/research/sources -type f` 只返回 6 个 .md。｜修复要求：改述第 62 行与第 711 行，删除"来源均存档于本页 research/sources/""下列定位均可在存档中复核"等与现状不符的表述，改为指向实际存在的 .md 摘录与官方仓库（deepseek-ai/DeepGEMM）外部地址；对只依赖已移除源码的行号级论断，统一改标为"官方仓库 commit 559d79f 源码定位（本仓库不再随页存档）"并在 C 表中给出可外部检索的路径，或无法外链时降级为明确标注的推断。｜修复：已改述第 62 行与第 711 行——删除"来源均存档于本页 research/sources/""关键文件已拷贝存档于 research/sources/""下列定位均可在存档中复核"等表述；第 711 行改为声明所有 .cuh/.hpp/test_mega_moe.py 的文件名与行号均为官方仓库 commit 559d79f 的定位（可在 deepseek-ai/DeepGEMM 对应文件处检索复核，本仓库不再随页存档源码快照），并注明未带路径的 .cuh 指 sm100_fp8_fp4_mega_moe.cuh；C 表中的 scheduler_mega_moe.cuh、layout_mega_moe.cuh 改为官方路径记法 scheduler/mega_moe.cuh、layout/mega_moe.cuh，C4 行尾的"（存档）"删除；指向现存 .md 摘录（README/PR #304/#316/X-Stage/Primus）的定位保留。｜复验：grep 全文已无"来源均存档于""可在存档中复核""（存档）"，亦无指向不存在文件的路径；剩余"存档"仅出现在"本仓库不再随页存档源码快照"这一否定句中；python3 .dojo/scripts/validate.py wiki/megamoe/index.html 返回 validation ok

- [轻微·格式] 来源与范围说明 → 外部数字与实验条件（N）表（行 764–782）：编号从 N9 直接跳到 N11，缺 N10；正文与 C 表引用中亦无 N10，且行 779 的 N12 自注"本页正文未引用，备考"。C 表（C1–C30）与 F 表（F1–F3）编号连续，唯 N 表断号。｜引文依据：不适用（机械项）｜修复要求：恢复编号连续性——将现 N11、N12 顺次改为 N10、N11，或补入 N10 条目并说明；保证 C/F/N 三表编号无断号、与正文上标一一对应。｜修复：N 表将原 N11（SymBuffer 最大 72 rank）改为 N10、原 N12（代码注释耗时参考）改为 N11，正文第 681 行上标 [N11] 同步改为 [N10]；N 表编号变为连续的 N1–N11。｜复验：grep 确认 N 表编号连号无断档，正文 [N1]–[N11] 上标均能在表中找到唯一定义；C 表（C1–C30）与 F 表（F1–F3）编号原已连续，未改动

- [轻微·表述] 正文与折叠块存在临场评价/元话语：行 177"注意两件事："、行 463"一个值得注意的省内存设计"、行 486"这里出现了一个容易忽视的事实"、行 666"三个读数值得停下来看"。check.md 第 2.2 节"表述"维度要求排除元话语与临场评价。｜引文依据：不适用｜修复要求：改为直接陈述（如行 666 直接写"三个读数…"、行 463 写"L1 输出与 L2 输入共用同一缓冲——SwiGLU 后宽度减半正好原地容纳"），删去"值得/值得注意/容易忽视/注意"等导读式评价词，不改变事实内容。｜修复：行 177 删去"注意两件事："；行 463 删去"一个值得注意的省内存设计："，直接陈述"L1 的输出缓冲和 L2 的输入缓冲是同一块内存——…"；行 486 删去"这里出现了一个容易忽视的事实："；行 666 改为"三个读数如下。"。｜复验：grep 全文已无"注意两件事""值得注意""容易忽视""值得停下来看"等导读式评价词，事实内容未变

## 已核对且无问题的项（引文依据）

- 基准数字：页内 5.1 表（行 654–661）与 PR #316 表一/表二逐格一致（V4-Flash 1/512/8192/32768 → 56.5/146.5/1283.1/4855.5 us、1.96/1.73/1.56/1.62x，batch512 互联 266 GB/s；V4-Pro → 108.1/369.6/2818.5/10655.2 us、1.61/1.54/1.50/1.54x，batch512 互联 182 GB/s）。｜依据：PR #316 原文表（WebFetch 抓取一致）；N1/N2/N3/N4。
- batch 口径：行 648"batch size 指每 rank 的 token 数（EP8 下 512/rank 即全节点 4096），全部数值为 8 个 rank 的平均"。｜依据：PR #316 评论 zheanxu"the batch size listed is the number of tokens per rank…512 × 8 = 4,096"（N5）。
- 通信量估算（行 242、259）：$6\times7168=43008$ B≈43 KB、BF16 回传≈86 KB、$43008\times512=22{,}020{,}096$（≈22 MB）——算式自洽且标注为 F3 推导（非官方公式）；该量级代入 V4-Pro batch512（耗时 369.6 us）得 ≈178 GB/s，与表中互联 182 GB/s 同量级，未见矛盾。
- $N_{\mathrm{exp}}$ 表（行 471–480）：$512\times8\times6/384=64$、$8192\times8\times6/384=1024$、$1\times8\times6/384=0.125$ 三项算术全部正确。
- $t_0$ 路径与贯穿示例（行 177、457；图行 517/522/540–551）：8 个专家-路由对中 6 个跨 rank（3/4）、$E_0$–$E_3$ 各专家本地 token 归组与 combine 槽位，均与第 1 章路由表逐项自洽。
- 来源论断（可外部核对部分）：README"fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel""requires multi-process launch with symmetric memory"（C1）；PR #304"Only FP8 x FP4 MoE is supported""Requires PyTorch >= 2.9"（C3）与 Mega MoE 八位贡献者名单 @LyricZhao/@zheanxu/@bucket-xv/@RayWang96/@interestingLSY/@kurisu6912/@xay5421/@yukuai26（C28）；"still under development and optimizations"与"nothing to do with internal model release"（C27）；README Utilities 的 `DG_COMM_KERNEL_DEBUG`"zero symmetric buffer before each Mega MoE call"（C26）；X-Stage §1/§2.2 的 completion-coupled 上限"at most approximately 1.5x…below the 1.56x"、interleaved 调度"84 configurations…1.18x geometric-mean…1.62x maximum"、源码无"expert wave"术语（C20/C25）。
- 术语定义：行 296"UMMA——Unified Matrix-Multiply-Accumulate，Blackwell tcgen05 架构的指令族"经外部检索确认（CUTLASS 将 tcgen05.mma 记作 UMMA，仅单线程发射、累加器在 TMEM）。
- 结构与格式：h2 编号 1–5 连续、来源章 h2 不编号；各章 h3 编号连续且章末均有"本章问题"；页面级"核心问题"5 条、两级问题均有"解答："折叠块且答案指明完整论证章节；4 张结构图均为 HTML/SVG，图内数学一律走 `<foreignObject>`，`<text>` 内无 ASCII 近似；`python3 .dojo/scripts/validate.py wiki/megamoe/index.html` 返回 `validation ok`；index.html ↔ overview.html 互链；引用的 9 个概念页（deepseek-moe、moe-serving、gpu-communication、deepep、swiglu、fp8-block-quant、mxfp4-qat、gpu-execution-model、deepseek-v4-dataflow）全部存在，无"（待生成）"占位。
- 代码：行 399–418 API 块与 README Usage 逐行一致，summary 已注明"本页不运行"及原因（需 sm100 GPU 与多进程环境），属静态核对；行 582–607 伪代码块 summary 已注明"非可运行代码"。

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 2
- 处置：修复。第 1 项（已移除的 research/sources/ 源码路径与"可在存档中复核"的错误承诺）须改述后关闭；第 2、3 项按修复要求处理并在本记录回填修复/复验结果。核心结论（五段融合、持久 kernel、warp 专用化、对称内存、1.50–1.96 倍基准与硬边界）均有现存外部来源与可外部核对的引文依据支撑，页面可用，故无阻断项。
