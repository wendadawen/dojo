<!-- review-meta
round: 9
page: wiki/strata/index.html
reviewed_content_sha256: c27c4c492437ecec
-->
# Strata 审查记录（第 9 轮）

- 页面版本：5394c9fa507c07dc5c7c884a17ce114cfe5e0801（git hash-object wiki/strata/index.html）
- 论文版本：arXiv:2508.18572v1（cs.DC，2025-08-26 提交；arXiv 仅 v1，comments 仍为 "under peer review"）；发表场次：USENIX OSDI '26 技术场次，Track 1「KV Cache and Long Context」（已核对 usenix.org/conference/osdi26/technical-sessions，该 session 下第一项即 Strata）
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题（5 题）→ 术语表 → 1 瓶颈定位（1.1 容量账 / 1.2 碎片化传输 / 1.3 大页不是出路 / 1.4 调度器假设失效与 delay hit / 本章问题）→ 2 GPU-assisted I/O（2.1 / 2.2 / 本章问题）→ 3 布局解耦与存储层（3.1 / 3.2 / 3.3 / 本章问题）→ 4 Cache-aware 调度（4.1 / 4.2 / 4.3 / 4.4 / 本章问题）→ 5 实验验证（5.1–5.6 / 本章问题）→ 6 方法评价（6.1–6.3 / 本章问题）→ 来源与范围说明

## 核对方法与依据

- 论文原文：拉取 arXiv HTML（arxiv.org/html/2508.18572v1）逐段核对；TeX 版引用位置以 § 号复核。
- 官方材料：USENIX OSDI '26 技术场次页（curl 取原始 HTML，确认 "Track 1 / KV Cache and Long Context"）；arXiv abs 页（确认 v1、提交日期、作者与单位）。
- 图内数值：逐张打开 assets/ 下 14 张 webp 目视核对（非仅比对文字）。
- 机械项：validate.py 返回 "validation ok: wiki/strata/index.html"；全页无 Unicode 数学字符（含 ×÷≈≥≤ 等）；无 alt 含 `$`；概念页 kv-cache / paged-attention / prefix-caching / standard-attention / gpu-execution-model / gpu-communication / kv-cache-layout / dualpath 均存在。

## 已回源确认的关键论断（抽样，均一致）

- 元数据作者与单位（Stanford & NVIDIA / SJTU / CU Boulder / CMU / NVIDIA / NVIDIA & Michigan / NVIDIA / NVIDIA & Stanford）与原文一致。
- F1 `C=λL`、`X=λS`、F2 `X=C·S/L` 对应原文 §3.1；F3 与 kv-cache 页 F1 公式一致，`2×32×8×128×2=131072 B=128 KB/token`、20,000 token≈2.44 GB、占 40 GB 的 6.1% 均可复算。
- 22%（PCIe 5.0）/ 5%（Grace-Hopper）/ 74% stall / 吞吐降 4× / 优化后仍 24%：与 §1、§3.1 一致。
- 页大小 32/16/1（TensorRT/vLLM/SGLang）、75–80% 带宽需 1–2 MB、128 字节粒度、2×1024 thread→~50 GB/s、prefill<5%/decode 10%、2 block 加载 1 block 备份、page-first 磁盘延迟最多降 4×：与 §3.1、§4.2、§5.3.4 一致。
- 调度两处阈值均 100（transient 匹配 §4.3.1；load/compute §4.3.2），且 §4.3.2 明确"硬件/模型相关"而 §4.3.1 未提硬件相关性——页面区分正确。
- 端到端/消融/页大小/cache distance/GH200 全套数字（3.2/2.6/1.9、3.9/2.1/1.9、5/5/3.75、ReviewMT 1.7/2.3/2.3、预热 2.3/2.6/2.5、1.8×/2.3×、93%/2.4%、+42/+76/+95/+11/+12/+8/+3%、40→150 GB/s）逐项与 §5 一致。
- 图注读数：Figure 1（红≈74%、绿≈24%）、Figure 5（2 block 处 HtoD≈50 GB/s、prefill≈95%、decode≈90%）、Figure 10（512 处≈0.93、命中率≈0.976）、Figure 11（Min 段 +42%、Shuffle/Max 段 +76/+95 等）、Figure 14（Strata-IO 40.30→150.50 GB/s）与页面表述相符。
- Figure 7 行序描述（FIFO：A0+A1、B0+B1、C+D0、D1+F、G、Decoding；Strata：A0+B0、A1+B1、C+F、D0+D1、Decoding、G）与图 img-07 逐格一致，颜色映射（橙/绿/紫/蓝/灰）与图例一致。
- Algorithm 1 伪代码（language-text、标"伪代码"）逐步对应原文 Algorithm 1，未标为可运行语言，属静态审查范围，无执行要求。
- DualPath 交叉引用（agentic 命中率≥95%→prefill 退化为 I/O 密集、第二条 KV 加载路径）与 wiki/dualpath 页一致，非无源论断。
- 核心问题 5 题、各章「本章问题」均有 `解答：` 折叠块；核心问题答案均以"完整论证见第 N 章 / 完整数据见第 5 章 / 完整分析见第 6 章"结尾；折叠块全部收起时正文仍能回答全部核心问题。

## 问题

- [轻微·图注] 图 5 图注（第 2 章 2.2 节，正文约 255 行）：图注称"纵轴是同跑场景下的吞吐（数据传输、prefill、decode）与互相干扰百分比"，但该图右轴实际为 Bandwidth (GB/s)（图 img-11 左轴 Normalized Throughput (%)、右轴 Bandwidth (GB/s)），页面凭空多出一个"互相干扰百分比"轴。｜引文依据：img-11 图例与坐标轴标注为 Prefill Throughput / Decode Throughput / HtoD Bandwidth / DtoH Bandwidth，双纵轴标签为 "Normalized Throughput (%)" 与 "Bandwidth (GB/s)"。｜修复要求：把"与互相干扰百分比"改为"与传输带宽（HtoD/DtoH，GB/s）"或删去，使图注轴含义与图上刻度一致。｜修复：｜复验：
- [轻微·图注] 图 14 图注（5.5 节，约 539 行）：图注称"但 Strata-IO-GH（仅 I/O 机制，无调度）仍不敌 Strata-PCIe（完整版，H200 平台）……完整 Strata-GH 接近 Oracle"，这两条是 TTFT 比较结论（出自 Figure 13），而 Figure 14 只画持续带宽，图中不含 Strata-PCIe / Strata-GH / Oracle 三组数据。｜引文依据：img-01 图例仅 SGLang-HiCache-PCIe(10.80) / SGLang-HiCache-GH(19.43) / Strata-IO-PCIe(40.30) / Strata-IO-GH(150.50) 四柱。｜修复要求：图 14 图注只保留带宽读数（40→150 GB/s），把"Strata-IO-GH 不敌 Strata-PCIe”“Strata-GH 接近 Oracle"移到图 13 段落或明确标注"（结论出自 Figure 13）"。｜修复：｜复验：
- [轻微·技术] Figure 10 图注括注（5.4 节，约 521 行）："（论文未明说 2.4% 相对哪个具体对照配置，数字本身见 §5.3.2）"与原文及同页 5 章问题解答（约 558 行"cache 命中率比 Strata-IO 低 2.4%"）不一致——原文 §5.3.2 的 2.4% 就是相对 Strata-IO 的同一比较，并非未指明基线。｜引文依据：原文 §5.3.2："Even at its best-performing setting (page size 512), SGLang-HiCache achieves only 93% of Strata-IO's performance, primarily due to a 2.4% lower cache hit rate."｜修复要求：删除该括注，或改为"2.4% 为相对 Strata-IO 的命中率差（§5.3.2）"，与第 5 章解答保持一致。｜修复：｜复验：
- [轻微·来源] 页尾"辅助解释与类比边界"（约 639 行）：称"OS 分页类比（块-页、token-字节、请求-进程）借自 vLLM 论文，对应关系在 KV cache 与 OS 页之间一致……"，但全页正文（含折叠块）并未出现该 OS 分页类比，此栏描述了页面不存在的内容。｜引文依据：不适用（正文全文检索"类比/操作系统/分页机制"无此映射）。｜修复要求：删去该句，或在正文相应位置真正引入该类比并注明辅助解释边界。｜修复：｜复验：
- [轻微·格式] 1.1 节"一句话账"引文（约 143 行）："40 GB HBM 对 Llama-8B 只够缓存约 0.3M token[N1, N2]"，其中 N2 在来源说明中定义为"页大小 32/16/1 token"，与本条 0.3M token 的容量论断无关。｜引文依据：来源说明 N2 = 页大小 32/16/1 token（§2.2 L14、§3.1 L48）。｜修复要求：该处引文编号改为 [N1]（容量事实），或补上所引的第二条实际支撑。｜修复：｜复验：
- [轻微·表述] 表述维度：正文出现口语化与评价性措辞——1.1 节"一句话账："（口语化 callout），2 节开头"Strata 在数据面提出一个反直觉的杠杆"（"反直觉"为临场评价，论文未作此判断，宜移入第 6 章或删去评价词）。其余段落已通读，未见元话语、以"本页"作主语的自我指代、会话指代（我/我们/你）、调试叙事或 AI 拼接腔；正文与论文声称/本文解读已分开。｜引文依据：不适用。｜修复要求：把"一句话账"改为中性表述（如"容量换算"），删去"反直觉的"或改为可回源的描述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 6
- 处置：无阻断、无重要问题；页面核心论断、公式、实验数字与图注读数已逐条回源，venue 与场次（OSDI '26 Track 1「KV Cache and Long Context」）已对官方场次页确认，validate.py 通过。6 条轻微问题均为图注轴标注/图注越界、来源栏描述空内容、个别引文编号冗余与措辞润色，不影响正确性与主线理解，可按上述要求逐条收敛；不改变结论范围与大纲。建议本轮修复上述轻微项后即可发布，或将轻微项留作明确接受的遗留项。
