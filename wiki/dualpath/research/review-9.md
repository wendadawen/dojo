<!-- review-meta
round: 9
page: wiki/dualpath/index.html
reviewed_content_sha256: 095d756a28588c70
-->
# DualPath 审查记录（第 9 轮）

- 页面版本：d6d4564be6315b4cbae1f6be2ad20a5c1c9ca5c4（`git hash-object wiki/dualpath/index.html`，工作树）
- 论文版本：arXiv:2602.21548v2（cs.DC，2026-02-26；PDF 正文与文献表为准，arXiv HTML 的 LaTeXML 版仅作辅助）
- 审查时间：2026-09-14 14:38
- 审查者：独立子代理（未参与写作，未参与前序审查与修复；未读取本页 research/）
- 已完整阅读章节：核心问题（5 条含解答）、贯穿示例、§1 agentic 推理的存储 I/O 瓶颈由三因素叠加（含本章问题）、§2 DualPath 的双路径数据流与块布局（含本章问题）、§3 双路径不引入新瓶颈的 P/D 区间（含本章问题）、§4 CNIC-centric 流量管理（含本章问题）、§5 Adaptive Request Scheduler（含本章问题）、§6 实验（含本章问题）、§7 方法评价（含本章问题）、来源与范围说明

## 核对依据（本轮回源片段）

- Fig.3 左：PDF 图内注释框原文为 `GPU Compute: 28.8x` / `PCIe Bandwidth: 2.0x` / `GPU Memory: 2.4x`，正文 `from NVIDIA Ampere to Blackwell, the I/O-compute ratio decreases by 14.4 ×`。页面「FLOPS 28.8×，PCIe 仅 2.0×，HBM 2.4×，I/O-compute 比下降 14.4×」成立（28.8/2=14.4 自洽）。
- Fig.3 右：页面「5→10 仍有约 12-13% 提升（读图估计）」。像素核对：线性刻度 1x→y=418px、2x→292px、3x→165px（126.5px/单位）；曲线在 batch 5 处 y≈216.5（2.59x）、batch 10 处 y≈175.5（2.92x）→ 比值 1.13（≈+13%）。与页面「约 12-13%」相符，且页面已标注为读图估计。
- Fig.1：PDF 图内文字为 `40% GPU util.` / `80% GPU util.` / `100% util.`（左右各一），与页面图注「PE 节点 storage NIC 100% 饱和、GPU 利用率 40%」「推回 80%」一致。
- Table 1 逐格一致（Qwen2.5-32B FP16 117-267、GPT-OSS-120B 47-95、Qwen3-235B-A22B 39-60、DS-V3.2 660B 13-36、DS-V3 660B 4.8-5.8），正文 `approximately 22 GB/PFLOP for DeepSeek-V3.2`；Table 1 表头 `append length 429, across context lengths (16k–64k). KV-Cache data type defaults to FP8 unless specified.` —— 页面 §1.1 表述与之一致。
- Table 2 逐格一致：32K(60/608/148/28639/17183)、48K(106/474/172/42607/25120)、64K(157/429/176/55958/32721)；正文 `mean number of rounds is 157 ... 32.7k ... 429 ... hit rate of 98.7%`。
- §4.2 四组不等式与汇总区间可复算且与论文 eq.(1)-(9) 一致：(1) 2Bs/g≤B；(2) (Tp+Tc)Dg=Bs/g(1+D/P)≤B→(3) P/D≥s/(g-s)；(4) (Tp+2Tc)Pg=s/g(P/D+2)B≤B→(5) P/D≤(g-2s)/s；(6)(2Tp+Tc)Pg≤B→(7) P/D≤(g-s)/(2s)；(8) P/D≤(M/Bs-3)/2；(9) 汇总。典型配置 (g=8,s=1,M≈500,Bs≈50) → 1/7≤P/D≤7/2（页面逐项数值 0.143/6/3.5/3.5 复算无误）。
- 页面 §3.2 对论文 eq.(1) 的处理：PDF 原文 `2 × Tp × Dg = 2Bs/g ≤ B (1) ... Since s ≤ g always holds in practice, the read direction is always bottleneck-free.` 该式等价于 s≤g/2，与「s≤g」不一致；页面如实标注「疑为原文笔误，原文照录、不据此改写结论」，处理正确。
- 参考文献编号：PDF 正文 `existing GPUs do not support PCIe QoS [39]`，文献表 `[39] Andre Richter, Christian Herber, Thomas Wild, and Andreas Herkersdorf`（Euromicro DSD 2016）。页面「§5 引言段引用 Richter et al. 2016，参考文献 [39]」成立。（注：arXiv HTML 版被 LaTeXML 转成作者-年制，须以 PDF 为准。）
- Fig.13 图内右侧标注 `1.528`（红虚）/`1.184`（蓝实），正文 `improves load balance from 1.53 to 1.18 compared to round robin scheduling`；Fig.14 正文 `as low as 1.06 during the first 5% of the task`。页面 alt/图注「1.528 / 1.184 / 1.06」一致。
- Fig.15：图内三条序列为 `Prompt TPS`（左轴，`1e7` 标注）、`TTFT`（右轴 s）、`Running agents`；TTFT 起始峰值约 22 s（右轴刻度 0/10/20，像素核对峰值略高于 20）。页面图注成立。
- Fig.11：x 轴刻度 0.01/0.05/0.1/…/0.6，像素换算为线性（约 1300px/1.0 APS）；SGL(MC) 菱形簇 x∈[188,264] → APS∈[0.001,0.059]，页面「≈ 0.01–0.06」成立。
- Fig.7（三行 DS 27B/DS 660B/Qwen 32B × 三列 32k/48k/64k）、Fig.8（#Agents 512/1024/2048 × 1p1d/1p2d/2p1d）、Fig.9（Append/Gen Scale x1 x1.5 x2 x3；Basic 随 append 变长 JCT 下降）与页面图注逐项相符。
- 消融 17.21% / 38.19% / 45.62%（DS 660B、64K MAL、1024/2048 agents），大规模 Table 3 `3,167s / 3,201s`、`2P4D 0.4 APS → 44P88D 8.8 APS = 22×`、`1.739s/0.228s/0.039s → 1.847s/0.194s/0.036s`、`scheduler CPU usage remains below 10 cores`，在线 `1.67× for DS 27B, 2.25× for DS 660B`（(1.67+2.25)/2=1.96），离线 `up to 1.87×` / `1.78×` / `1.09–1.85× slower than Oracle`，P/D 敏感性 `1.64× ... up to 2.46×`，append 缩放 `1.82−1.99×`，working set `69 GB at APS 0.1 to 681 GB at APS 0.45` 与 `r times more machine hours and r² times more storage (cost scaling as r³)` —— 全部与原文一致。
- A.1 配置 `qos_max_vls 4` / `qos_high_limit 240` / `qos_vlarb_high 0:192,1:192,2:0,3:192` / `qos_vlarb_low 0:192,1:192,2:64,3:192`、`four lossless RDMA TCs`；A.4 `α ... 3 seconds`、`β ... 5 seconds`、`Compute Quota Threshold is set to 300ms`；A.5 `[1,tokens,bytes]` / `[layer,tokens,bytes]` / trie 每节点一个 Full Block；§5.2 `5-7 μs`（cudaMemcpyAsync）与 `around 1 μs`（RDMA Write）+ doorbell batching —— 页面逐条一致。
- 机械项：`python3 .dojo/scripts/validate.py wiki/dualpath/index.html` → `validation ok`；无 `alt` 含 `$...$`（grep 计数 0）；无「待生成」占位；全部 assets/*.webp|png 存在；外链 wiki 页与锚点（moe-serving #s3-ep-alltoall / #s5-prefill-decode-kvcache / #s7-pdd-colocation、standard-attention、deepseek-moe、mla、dsa、mqa-gqa、gpu-communication）均存在；overview.html 与 index.html 互链；`dojo:type=paper`、`dojo:topics`（推理系统/内存与缓存/并行与通信）与 `dojo:tag` 在词表内；summary/description 公式为合法 KaTeX（`\tfrac`/`\min`/`\to`）。

## 问题

- [轻微·表述] §2.4（“这一说法是页面推断（论文未就此对比）”“这一联系由页面综合两处得出”）、§5.3（“是页面的推断”）、§7 前「本章问题」（“1.05 倍均值是页面为覆盖大多数碎片场景给出的解释”）、「来源与范围说明」（“页面内嵌的 14 张原图…全部图片存于本页 assets/ 目录”“页面所有 C 编号论断”“页面 N 编号论断”“页面第七章”“页面推导假设…”）：以「页面／本页」为主语的自我指代式元话语。｜引文依据：不适用｜修复要求：改为无自我指代的客观表述，例如“这一说法是页面推断（论文未就此对比）”改为“论文未就此对比，此处为推断”；“全部图片存于本页 assets/ 目录”改为“全部图片存于 assets/ 目录”；「来源与范围说明」中同类句子一并去主语化，保留“（推断）”“（读图估计）”等来源标注不改。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 1
- 处置：可发布。本轮逐条回源核对（PDF 正文、图内像素、A.1/A.4/A.5 附录、Table 1-3、Fig.1/3/7-15），未发现与原文不符的事实、数字、公式或引文编号；四组不等式与汇总区间可独立复算；页面已把推断、读图估计、原文疑似笔误（eq.(1) 的 s≤g）三处不确定性显式标注，未把推断写成来源结论。唯一遗留的轻微项为表述层面的自我指代，不影响正确性与主线理解，理由为：该写法在本仓库多数页面沿用，且部分句子（“页面推断”“页面解释”）承担质检规范所要求的“推断必须明确标注”的功能；建议按上条要求顺手去主语化，但可接受为不阻断发布的遗留轻微。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
