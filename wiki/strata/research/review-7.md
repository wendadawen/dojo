<!-- review-meta
round: 7
page: wiki/strata/index.html
reviewed_content_sha256: 5df84e93d03e2b8c
-->
# Strata 审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 8fdd88513da9c828a60877a95fd7cee02f4675dd
- 论文版本：arXiv:2508.18572v1（2025-08-26 提交，TeX 源码；本轮核对用 arxiv.org/html/2508.18572v1 全文与本页 assets 内 14 张原图）
- 审查时间：2026-09-13 21:54
- 审查者：独立子代理（未参与写作，未读取本页 research/ 任何文件）
- 已完整阅读章节：head/meta → 论文信息块 → 导读与阅读前假设 → 核心问题（5 题解答折叠块）→ 术语表 → 1 瓶颈定位（1.1 容量账 / 1.2 分片传输 / 1.3 大页不是出路 / 1.4 调度假设失效）→ 1 本章问题 → 2 GPU-assisted I/O（2.1 机制 / 2.2 干扰控制）→ 2 本章问题 → 3 布局解耦（3.1 两种布局 / 3.2 在途转换 / 3.3 存储层预取）→ 3 本章问题 → 4 Cache-aware 调度（4.1 系统全景 / 4.2 延迟命中 / 4.3 均衡 batch / 4.4 bubble filling）→ 4 本章问题 → 5 实验（5.1 设置 / 5.2 端到端 / 5.3 消融 / 5.4 页大小与 cache distance / 5.5 GH200 / 5.6 短上下文）→ 5 本章问题 → 6 方法评价（6.1 优点 / 6.2 局限 / 6.3 适用场景）→ 6 本章问题 → 来源与范围说明（核对/公式/外部数字/原图对应/构造示例/简化条件）→ 页脚脚本

## 核对方法

- 论文全文（v1 HTML）抓取后逐条回源：§1 引言、§3.1（Little's Law 与 22%/5%/75–80%/1–2MB/页大小 1→1024）、§3.2（delay hit、74%/24%）、§4.1（系统架构、P-D co-location）、§4.2（128 字节、2 block×1024 thread、50 GB/s、<5%/10%、端到端 <5%、ROCm）、§4.2.1（预取、一次算术转换）、§4.3/4.3.1/4.3.2/4.3.3（transient node、阈值 100、Algorithm 1、bubble filling）、§5.1 表 1（平台/模型/基线/数据集全部数字）、§5.2.1/5.2.2/5.2.3（端到端 3.2/2.6/1.9、3.9/2.1/1.9、5/5/3.75、ReviewMT 1.7/2.3/2.3、预热 2.3/2.6/2.5、95% hit）、§5.3.1/5.3.2/5.3.3/5.3.4（1.8×/2.3×、93%/2.4%、42%/76%/95%/11%/12%/8%/3%、4×）、§5.4（40→150 GB/s、Strata-IO-GH 不敌 Strata-PCIe、Oracle）、§6 相关工作、§7 Conclusion 与 on-chip 展望。
- 逐一读原图核对图注：img-14(Fig1)、img-12(Fig2)、img-13(Fig3)、img-06(Fig8)、img-05(Fig9)、img-04(Fig10)、img-03(Fig11)、img-09(Fig12)、img-02(Fig13)、img-01(Fig14)、img-11(Fig5)、img-10(Fig6)、img-07(Fig7 时间线各元素 A0+A1/B0+B1/C+D0/D1+F/G/Decoding 与 Strata 行 A0+B0/A1+B1/C+F/D0+D1/Decoding/G 逐块对齐)。
- 复算：每 token 128 KB（2·32·8·128·2=131072 B）、20,000×128 KB≈2.44 GB、2.44/40≈6.1%、40 GB/128 KB≈0.33M、⌈20000/32⌉=625、32×128 KB=4 MB、4 MB/32=128 KB、页 32 共享 100 token→命中 96 丢 4、1.687/0.420≈4.0×、384/64=6×。
- 运行 `python3 .dojo/scripts/validate.py wiki/strata/index.html` → `validation ok`（含数学字符与结构图检查）；无 `$` 写入 alt；正文/列表/表格无 Unicode 数学字符（仅伪代码用 `←` 作赋值，属代码块）；8 个前置概念页（kv-cache、paged-attention、prefix-caching、standard-attention、gpu-execution-model、gpu-communication、kv-cache-layout、dualpath）均真实存在，无「（待生成）」；overview.html 与 index.html 互链。
- 本轮未发现任何阻断级事实/公式/数字错误；上列数字全部与 v1 原文一致，未发现算式与结论不符、分项之和≠合计、正文与 summary 数字冲突、引文编号错位。

## 问题

- [重要·技术] 阅读前假设段（贯穿示例）："手册缓存约 2.44 GB——已超出单请求显存预算，分层与跨层搬运不可避免。"｜引文依据：§1 原文 "40 GB of GPU High-Bandwidth Memory (HBM) can only hold roughly 0.3M tokens for Llama-8B, which can be quickly consumed by a handful of documents or hundreds of conversation turns."（即 20K token 手册仅占约 6.1%，是"多份文档/多轮对话"才耗尽，不是单请求即超预算）；本页 1.1 亦写"一份手册已经吃掉 0.02M，生产中几十份文档或几百轮对话就耗尽显存"与"占 40 GB HBM 的约 6.1%"，与本句口径相互矛盾｜修复要求：将该分句改为与 1.1 一致的表述，例如"已占掉 GPU 显存预算的相当份额（40 GB HBM 下约 6.1%）"；不得保留"单请求即超出显存预算"这一原文不支持的结论，全文（含 overview.html 若同义）一并统一｜修复：｜复验：
- [轻微·技术] 2.2 节图（原文 Figure 5）说明："横轴是 I/O kernel 占用的 CUDA block 数与每 block 线程数"｜引文依据：原文 Figure 5 横轴仅为 "Number of Blocks"（刻度 0/1/2/4/8/16/32/64/128），线程数固定 1024，见正文 "using only two CUDA blocks of 1024 threads each"｜修复要求：横轴描述改为"CUDA block 数（每 block 1024 线程固定）"，不要写成二维扫描｜修复：｜复验：
- [轻微·技术] 1.2 节："打满 PCIe 5.0 的 75%–80% 需要 1–2 MB 传输[C5]；SSD/NIC 等更高带宽介质的要求更高[C5]。"｜引文依据：§3.1 原文 "achieving high throughput on other media like SSDs or network interfaces often demands even larger transfer sizes"——原文只说这些介质需要更大的传输，并未称其为"更高带宽介质"（SSD/NIC 带宽并不高于 PCIe 5.0 x16 的 64 GB/s）｜修复要求：删去"更高带宽"这一原文未给的定性，改为"SSD/NIC 等介质要达到高吞吐同样（甚至更）依赖大传输"｜修复：｜复验：
- [轻微·技术] 5.2 节核心数字表 NarrativeQA（预热稳态）行的 "vs SGLang-HiCache" 与 "vs TensorRT-HiCache" 两格均填 "—"｜引文依据：§5.2.2 原文 "We report Strata, SGLang-HiCache, and vLLM-LMCache, as TensorRT-HiCache does not support pre-warming."——该设置下 SGLang-HiCache 有报告（Figure 8 第二行 SGLang-HiCache 曲线在列），只有 TensorRT-HiCache 未测｜修复要求：SGLang-HiCache 格不得标 "—"（写"论文正文未给出倍数，见 Figure 8 第二行"或补数）；TensorRT 格可标"未测"，与 5.2 节正文括注一致｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 1 / 轻微 3
- 处置：修复（关闭上列 1 条重要与 3 条轻微后即可发布；无阻断）

## 附：本轮核对通过、未列为问题的重点（供下一轮参考）

- 全部端到端数字（LooGLE 8B 3.2/2.6/1.9、14B 3.9/2.1/1.9、70B 5/5/3.75；ReviewMT 1.7/2.3/2.3；预热 2.3/2.6/2.5；abstract 5×/3.75×）与 §5.2、Figure 8 一致。
- 消融（1.8×/2.3×）、页大小（最优页 512 → 93%、命中低 2.4%）、cache distance（42%/76%/95%/11%/12%/8%/3%）、GH200（40→150 GB/s、Strata-IO-GH 不敌 Strata-PCIe、近 Oracle）与 §5.3/§5.4 及 Figure 10/11/13/14 逐值一致。
- 机制与阈值：transient node 三态、阈值 100"严格大于才推迟"（§4.3.1 "exceeds this value"，与自绘 SVG 的 `>` 一致）、loading_bound 阈值 100 及"硬件/模型相关可分别 profiling"、Algorithm 1 伪代码逐行对齐（含 14–16 行补降级请求、防饿死保留原序）。
- 构造示例算术（128 KB/token、2.44 GB、6.1%、625 页、4 MB、128 KB/单层片段、2 层×4 token 布局算例）自洽，且已由「构造示例」小节显式标注。
- 表述维度：全文无「本页/我们/你」等自我指代与会话指代，无「下面来看/需要注意的是」类元话语，无调试叙事与临场评价，无 AI 拼接腔；伪代码与图注未混入论文声称与本文解读；第 6 章有 callout 声明属分析性判断；来源与范围说明对简化条件（1TB pinned、disk 未纳入、自建 SGLang-HiCache、阈值可调、P-D co-location）逐条交代。
- 页面功能：KaTeX 与 Prism 本地资源齐备、折叠块与目录锚点正常、图注未压线、交互视图（SVG）不依赖脚本即可读。

统计：阻断 0 / 重要 1 / 轻微 3
