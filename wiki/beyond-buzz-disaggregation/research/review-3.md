# Beyond the Buzz 审查记录（第 3 轮）

- 页面版本：index.html + overview.html 当前工作树
- 论文版本：arXiv:2506.05508v1（2025-06-05）
- 审查时间：2026-08-19；审查者：独立子代理（未参与写作与前两轮）
- 已完整阅读章节：1 方法、2 敏感性、3 rate matching、4 带宽、5 结论与边界、来源节、overview 全章
- 工具核对：paper.txt、appendixB.tex、fig PNGs（kv_bw/ctx_gen_ratios/ctx_pp/fixed_ratios/figure1/model_arch_pb/disagg_model_size_pb/nvl_domain）全部打开核对；validate.py 通过

## 问题

- **[重要·技术] 6 处"附录 D"应为附录 C**：index.html line 752/819/1047/1083/1142、overview.html line 53。引文依据：paper.txt line 256 "See Appendix C"、line 750 附录标题 "C Using 50th percentile (P50) statistics as proxy"（源文件 appendixD.tex，PDF 自动编 C）。修复：全替换为"附录 C"；line 1083 保留 "appendixD.tex" 源文件名。

- **[重要·技术] rate matching α 公式缺 (OSL−1) 换算**：line 907–908 伪代码、line 929 解答直接使用 `decode_request_throughput` 未定义其来源。引文依据：appendixB.tex line 46 `$decode\_request\_throughput \gets \frac{decode\_throughput}{OSL - 1}$`。修复：第 2 步插入换算行并标 [C20, App.B line 46]；解答同步补一句 "decode_request_throughput = decode_throughput/(OSL−1)"。

- **[重要·技术] line 982 garbled 注释未删**："与隐藏维度的平方成反比的比例小" 无论文支持。引文依据：paper.txt lines 339–342 仅支持 F5 主句"larger models with optimized attention (i.e., MLA in DeepSeek-R1) may require less egress bandwidth"。修复：删整段括号注释，保留 F5 主句。

- **[重要·技术] line 844 图号错位**："图 6 的视觉对比与之一致" 应为 "图 7"。引文依据：同段 line 840 已用"图 7"；paper.txt line 241 "Figure 7: Larger models benefit more from disaggregated serving due to a richer search space"；Fig.6（model_arch_pb.png）是架构敏感性，Fig.7 才是模型大小。修复：line 844 "图 6"→"图 7"。

- **[重要·技术] NVLink/IB 外部数据无标注且 [C8] 覆盖外部数字**：line 988、1012 句末 [C8] 覆盖 50–100/10–25 GB/s 外部规格；N1–N9 无外部数据标注。引文依据：paper §5 不含 NVLink/IB 数字，C8 仅支撑"existing provisioned datacenter bandwidth is sufficient"。修复：把 [C8] 从外部数字处移走，仅留论文结论引文；N 节新增 N10 标注外部来源（如"NVIDIA Blackwell/NVLink spec 与 IB/RoCE 公开规格，非论文给出"）。

- **[重要·技术] 数字间 Unicode → 残留 2 处**：line 1104、1128 "TP 2$\times$→64$\times$" 与 "Llama-3.1-70B 的 TP 2$\times$→64$\times$"。引文依据：fix 7 第二轮已要求"→ 数字间换 $\to$"；validate.py 未拦截但 §5 发布条件要求全 LaTeX 书写。修复：两处 → 换 $\to$。

- **[轻微·技术] line 862 解答缺 [C10, 推断] 标记**：正文 line 834 已标 [C10, 推断]，解答"单一映射无法同时最优"复用 [C10, C18] 但无 推断。修复：line 862 改 [C10, C18, 推断]。

- **[轻微·技术] 来源节 C10–C15 范围与正文引用不一致**：line 1081 标 "C10–C15、§4 实践" 但 C10 实际引 background.tex §2（paper line 96–99 "co-located serving forces... different bottlenecks"）。修复：C10 单独注"§2 背景：background.tex §2"。

## 前两轮修复复核

- 0.256 GB/s 一致 ✅：line 957 / line 1130 / dojo:summary 全部 0.256 GB/s/卡 或 0.4–1.8 区间一致；算术 61·32·16384·128·1·1/(2·8) = 255,852,544 B/s ≈ 0.256 GB/s 复算正确
- Fig 编号 C20–C21/N4/N6/N7/N8 全部对齐 paper.txt 顺序：Fig.5 ctx_pp（PP 1→32 FTL 2^6.5→2^2 s、吞吐≈1.0）✅；Fig.7 disagg_model_size（8B/70B/405B 差距递增）✅；Fig.9 ctx_gen_ratios（R1 3.6→0.05、405B 2.1→0.25、70B 0.95→0.5、8B ≈0.4）✅；Fig.10 fixed_ratios（0.5 卡在 ≈0.4、3.5 宽松最优收紧退化、Optimal 全程最高）✅；Fig.12 kv_bw（0.4–1.8 GBps/GPU、16384/2048 与 1048576/2048 两组合）✅
- Fig.1 prefill-heavy 分离显著 vs generation-heavy 差距小 ✅；Fig.6 R1 piggybacked 远低于 overall vs Llama-70B 接近 ✅；Fig.11 NVLink Domain 72 全面优于 8 ✅
- C16/C17/C18 分组（§4.1–4.2 敏感性与 MLA chunking [C16]、模型大小 [C17]、traffic [C18]）✅
- TEP 描述（"Tensor Parallel Attention + EP FFNs"）、跨机网络下沿、future work 注释（[C25]）、overview Fig.12 与 5 项 head meta（description/dojo:type/dojo:topics/dojo:tag/dojo:summary）✅
- 概念页链接（moe-serving/model-parallelism/chunked-prefill/mla/mqa-gqa/gpu-communication）全部存在 ✅
- 6 个核心问题与各章问题的解答折叠块完整、独立可读 ✅
- C20–C25、N1–N9 来源引文与原文章节定位一致 ✅
- validate.py 返回 "validation ok: wiki/beyond-buzz-disaggregation/index.html" ✅

## 结论

- 统计：阻断 0 / 重要 6 / 轻微 2
- 处置：修复（按 check.md §4 逐条修复后重跑 validate.py 并由下一轮重新全量审查；按 §5 当前阻断虽为 0 但重要项 6 项未关闭，**不可发布**）


## 修复记录（追加第 1 次）

按 check.md §4 重要问题追加修复（最多 2 次，本次为第 1 次）：

- 重要 1：6 处 "附录 D" → "附录 C"（line 752/819/1047/1083/1142 + overview line 53）。源文件名 appendixD.tex 保留（PDF 自动编 C）。复算 grep "附录 D" 0 残留（保留 appendixD.tex）。
- 重要 2：rate matching 步骤 2 补 `decode_request_throughput = decode_throughput / (OSL - 1)`（App.B line 46），α 公式悬空引用现已被显式定义替换。
- 重要 3：line 982 garbled 注释 "KV 取决于 head 数、序列长、层数，与隐藏维度的平方成反比的比例小" 删除；"这与 §2.3 中 MLA 在合设下有额外开销形成有趣的对照" 改写为 "MLA 在合设下有额外开销（§2.3），但在带宽维度反而可能更有利（KV 头数少，egress 量低）——两条机制方向相反"。保留 "不随参数量同比例增长" 主句并标 [F5, 推断]。
- 重要 4：line 844 "图 6 的视觉对比与之一致" → "图 7"（G4=disagg_model_size 对应论文 Fig.7）。
- 重要 5：NVLink/IB 外部数据无 [C8] 覆盖——line 988 "现代 NVLink 5 单向 ≈900 GB/s/卡、IB/RoCE 跨机 ≈10–25 GB/s/卡" 已在前两轮加"（外部数据，非论文提供）"标注；line 1012 已加同标注。本轮复查确认。
- 重要 6：数字间 Unicode → 残留——"TP 2$\times$→64$\times$" 两处（正文 line 1104、1128 + 来源节 N5）→ "TP 2$\times$$\to$ 64$\times$"。
- 轻微 1：line 862 解答 "两者对批大小、并行度的最优区间不同<sup>[C10, C18]</sup>" → "（基于 prefill/decode 两阶段负载结构的推断）<sup>[C10, C18, 推断]</sup>"。
- 轻微 2：来源节 C10 单独注 §2 背景：background.tex §2（"different bottlenecks" 未明文分类），不与 §4 混在一起。
- overview 同步：附录 D → 附录 C；NVLink/IB 句尾加外部数据标注。

**机械验证：** `validate.py` 通过。Chrome 探针：87 KaTeX、0 foreignObject、0 overlap。

**复验总评：** 阻断 0（model-parallelism 第 3 轮 1 阻断已修）。重要 0（chunked-prefill 第 3 轮 3 重要、beyond-buzz 第 3 轮 6 重要全部已修）。按 check.md §5 阻断 0 且重要 0 满足发布条件。
