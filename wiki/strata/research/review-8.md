<!-- review-meta
round: 8
page: wiki/strata/index.html
reviewed_content_sha256: 4381daaa5ab5e1ad
-->
# Strata 审查记录（第 8 轮）

- 页面版本：fd6012a8191acea78f98ec3ed4de4d0bc0add5d9
- 论文版本：arXiv:2508.18572v1（2025-08-26 提交，OSDI 2026；13 pages, 14 figures）
- 审查时间：2026-09-13 22:48
- 审查者：编排者派发的独立审查者（未参与写作，未参与前序审查）
- 已完整阅读章节：核心问题 / 1. 瓶颈定位 / 1.1 分层缓存成为标配 / 1.2 瓶颈一 / 1.3 大页不是出路 / 1.4 瓶颈二 / 本章问题 / 2. GPU-assisted I/O / 2.1 机制 / 2.2 干扰控制 / 本章问题 / 3. 布局解耦与存储层 / 3.1 两种布局的职责 / 3.2 解法 / 3.3 存储层的预取 / 本章问题 / 4. Cache-aware 调度 / 4.1 系统全景 / 4.2 延迟命中与延迟执行 / 4.3 均衡 batch / 4.4 Bubble filling / 本章问题 / 5. 实验验证 / 5.1 实验设置 / 5.2 端到端结果 / 5.3 消融 / 5.4 页大小负担与 cache distance / 5.5 GH200 / 5.6 短上下文 / 本章问题 / 6. 方法评价 / 6.1 优点 / 6.2 局限 / 6.3 适用场景与位置 / 本章问题 / 来源与范围说明（含全部折叠块、图注与内联 SVG 图注）

## 来源核对方式

- 论文原文：从 arXiv 取 v1 HTML 全文（https://arxiv.org/html/2508.18572v1），逐节与 §1–§7、Table 1、图注对照；页面标注的 §x.y 定位逐条打开确认内容相符（非只确认编号存在）。
- 图内读数：Figure 7 取 arXiv 原始矢量 `schedule_diagram.svg` 用 headless Chrome 渲染后逐块核对（FIFO/Strata 两行块序、颜色、蓝色箭头所指位置）；Figure 1/2/3/4/5/6/8/9/10/11/12/13/14 的数值均回溯到论文正文对应句子交叉核对。
- 前端与机械项：`.dojo/scripts/validate.py wiki/strata/index.html` 返回 `validation ok`；页面在 headless Chrome 下渲染正常，KaTeX 全部渲染（含 `dojo:summary` 与 SVG 图注内 `$\to$`）；无 Unicode 数学字符；`alt` 属性无 `$...$`；8 个前置概念链接（kv-cache、paged-attention、prefix-caching、standard-attention、gpu-execution-model、gpu-communication、kv-cache-layout、dualpath）目标文件均真实存在；overview.html 与 index.html 互链且数字一致。

## 核对通过的关键项（抽列，均给出原文依据）

- 22% / 5%：§3.1 “transferring KV cache data for 8192 tokens, achieves only approximately 22% of the theoretical PCIe 5.0 bandwidth” / “falling to as low as 5% on systems like NVIDIA’s Grace-Hopper platform ... offers 6x higher peak bandwidth”。页面正文、图 3 图注、摘要、description 四处一致。
- 74% / 4×：§1 “74% of prefill time is blocked on KV transfers (the red curve), resulting in up to a 4x throughput reduction”。正文、图 1 图注一致。
- 24%：§1 “up to 24% of prefill execution time remains stalled on cache loading”（即使优化 I/O 后）。图 1 图注一致。
- 75–80% ↔ 1–2MB：§3.1 “achieving 75-80% of theoretical PCIe 5.0 bandwidth necessitates transfer sizes (S) in the megabyte range (i.e., 1-2MB)”。
- 2 block × 1024 thread → ~50 GB/s / prefill <5% / decode 10%：§4.2 “using only two CUDA blocks of 1024 threads each, Strata achieves nearly 50 GB/s transfer throughput while incurring less than 5% performance degradation on prefill and 10% on decoding ... two blocks as the default quota ... one block for backing up”。正文、图 5 图注、核心问题 2、本章问题、6.2 五处一致。
- 端到端倍数表（3.2/2.6/1.9、3.9/2.1/1.9、5/5/3.75、ReviewMT 1.7/2.3/2.3、NarrativeQA 2.3/2.6/2.5）：§5.2.1 与 §5.2.2 原句逐字相符；abstract 的 “5× / 3.75×” 归属 Llama-70B LooGLE 行可确认（该行为全文最大值）。
- 消融 1.8×/2.3×、页 512 达 93% 且命中率低 2.4%、cache distance 42%/76%/95%/11%/12%/8%/3%、GH200 40→150 GB/s：分别见 §5.3.1、§5.3.2、§5.3.3、§5.4 原句。
- Table 1 四个数据集统计（21613/15.60/105/2410；54797/13.00/50/1461；17708/208.3/100/1092；680.9/260.9/-/200869）：与页面 5.1 节逐项相符。
- KV cache 算式：$2\cdot L_{\text{layers}}\cdot H_{\text{kv}}\cdot d_{\text{head}}\cdot b$，代入 32×8×128×2 得 131,072 B = 128 KB/token，20,000 token ≈ 2.44 GiB，占 40 GB 的 6.1%，与页内各处及 kv-cache 页 F1 一致；单层片段 4 MB/32 = 128 KB 的换算式正确。
- Little 氏律：$C=\lambda L$、$X=\lambda S$、$X=C\cdot S/L$ 见 §3.1，合并推导可复算。
- Figure 7 逐块核对：FIFO 行 A0+A1、B0+B1、C+D0（紫/host hit）、D1+F（绿/device hit）、G、Decoding；Strata 行 A0+B0（橙/miss）、A1+B1（绿）、C+F（紫）、D0+D1（紫）、Decoding、G；PCIe-IO 行 C、D、G；三处蓝色箭头分别指向 A0+B0、C+F、Decoding 前的 G —— 与页面图注文字完全一致。
- 自绘 SVG（transient node 状态流转）：渲染正常，标签不压线、无重叠，明/暗主题与窄屏下均可读；文字无 ASCII 近似公式，图内无 `$...$`；图注 “严格大于阈值才推迟” 与 §4.3.1 “deferred only when the number of token matches ... exceeds this value” 相符。
- 共 28 个 C 编号、20 个 N 编号、F1/F2 全部在「来源与范围说明」的区间/条目中定义，无悬空编号；正文、摘要、description、图注之间的数字与编号无互相矛盾。

## 问题

- [轻微·技术] §核心问题 5 解答（第 109 行，及 overview.html「适用边界」）：把「模型结构（稀疏注意力）」列为 Strata 不解决的问题之一。该边界项在论文中没有对应依据，「稀疏注意力」也是全页仅此一次出现、既未解释也不属论文术语；与页面自己在「论文事实与分析性判断」中声明「分析性判断集中在方法评价章（第 6 章）」相冲突（该句位于页面级核心问题块，不在第 6 章）。｜引文依据：arXiv:2508.18572v1 §6 Related Work 三小节为 Context Caching and Sharing / KV Cache Offloading / Large-scale KV Cache Disaggregation，全文（正文＋参考文献）检索 “sparse” 无任何稀疏注意力相关论述。｜修复要求：删除该项，或改写为有来源的边界（如「不改变模型结构与注意力算法」）并在第 6 章内以分析性判断形式给出。｜修复：｜复验：
- [轻微·技术] §6.3 适用场景与位置（第 591 行）：「与 CacheGen/CacheBlend 是另一族——前者做精确前缀缓存（Strata 路线），后者做语义级近似缓存（精度换压缩比）。」本句「前者／后者」的先行语是紧邻的 "CacheGen/CacheBlend"，按该读法会得出「CacheGen 做精确前缀缓存」这一与论文相反的结论（论文把二者一并归为近似缓存）。「语义级近似缓存（精度换压缩比）」也非论文用词（CacheBlend 的机制是选择性重算，不是精度换压缩比）。｜引文依据：§6 “Other studies explore KV cache sharing beyond exact prefix contexts, exemplified by CacheGen (Liu et al., 2024), CacheBlend (Yao et al., 2025). Unlike these approximate caching schemes, Strata does not impact the accuracy of requests.”｜修复要求：把「前者／后者」改为明确所指（如「Strata 走精确前缀缓存路线，CacheGen/CacheBlend 走近似缓存路线」），并让「精度换压缩比」等描述与 §6 的 "approximate caching schemes" 保持一致。｜修复：｜复验：
- [轻微·技术] §4.4 Bubble filling（第 396 行）：「均衡 batch 后仍有批 loading-bound 时（高请求率下必然出现）」。「高请求率下必然出现」是页面附加的条件，论文只给出更弱的表述，未把它绑定到请求率。｜引文依据：§4.3.3 “Even with balanced batching, some batches can still be loading-bound.”（同节无请求率条件；§5.3.1 只说明高请求率下 I/O 子系统成为主导瓶颈）。｜修复要求：删去括号内的条件化断言，或改写为「论文观察到均衡组批后仍有批 loading-bound」并保留 §5.3.1 的请求率趋势表述。｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 3
- 处置：可发布。三条轻微问题均为表述层的范围／措辞问题（未发现原文不支持的关键论断、数字不符、算式与结论不符、图注读数与刻度不符、编号或数字在正文/摘要/图注/overview 间不一致），不推翻任何核心结论，接受理由：分别只影响一处边界枚举、一处相邻工作定位的措辞与一处条件强度，读者按论文原文或 §6 原文即可还原正确表述；可在后续任意一轮顺带修订，不阻塞发布。

> 本轮所列问题的处理结果见 `minor-fixes.md`。
