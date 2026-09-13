<!-- review-meta
round: 8
page: wiki/beyond-buzz-disaggregation/index.html
reviewed_content_sha256: fabc95f9567b9cd5
-->
# Beyond the Buzz 审查记录（第 8 轮）

- 页面版本：883321d6c27c85bcf5b77e1fce49f1b5069a5671
- 论文版本：arXiv:2506.05508v1（2025-06-05 提交；TeX 源码 2025-06-06 打包，NeurIPS 2025 模板）
- 审查时间：2026-09-13 22:43
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复）
- 已完整阅读章节：术语速查、核心问题（5 题解答）、第 1 章 方法：模拟器与设计空间（含本章问题 2 题）、第 2 章 什么条件下分离收益最大（2.1 流量 / 2.2 模型大小 / 2.3 架构，含本章问题 3 题）、第 3 章 配套机制：切分策略与动态 rate matching（3.1 CPP / 3.2 decode 轨迹 / 3.3 rate matching 算法，含本章问题 2 题）、第 4 章 分离的代价：KV cache 传输带宽（4.1 egress / 4.2 ingress / 4.3 四条趋势，含本章问题 3 题）、第 5 章 方法评价：可操作结论与边界（5.1 清单 / 5.2 边界，含本章问题 2 题）、来源与范围说明
- 核对方式：论文原文取自 arXiv v1 PDF（pdftotext 抽取）与 arXiv e-print 源码包（abstract/introduction/background/design_principles/disaggregation_in_practice/system_considerations/related_work/future_work/conclusions/appendixA,B,D.tex）；图内数值用像素测量核对（对论文源图 ctx_pp.pdf=Fig.5、ctx_gen_ratios.pdf=Fig.9、fixed_ratios.pdf=Fig.10、kv_bw.pdf=Fig.12、dynamic_vs_static.pdf=Fig.14、figure1_no_static.pdf=Fig.1、isl_osl_pb.pdf=Fig.8、model_arch_pb.pdf=Fig.6、disagg_model_size_pb.pdf=Fig.7 逐条取色定位曲线端点与坐标刻度换算）

## 核对通过的关键项（本轮确认无误）

- 元数据：作者 13 人、NVIDIA、arXiv:2506.05508v1、2025-06-05 提交均与 arXiv 页面一致。
- 公式 F1/F2 与 system_considerations.tex 的 Eq.(1)/Eq.(2) 逐字符一致；正文（第 292、306 行）、summary（第 7 行）、核心问题第 4 题（第 116 行）三处写法一致。
- 手算 egress：$61\times32\times16384\times128\times1\times1/(2\times8)=255{,}852{,}544\ \text{B/s}\approx0.256$ GB/s/卡，与页面一致。
- rate matching 两步算法与 appendixB.tex Alg.1/Alg.2 逐步一致（含 `decode_request_throughput ← decode_throughput/(OSL−1)`、`num_prefill_gpus ← numerator(α)×G`、`num_decode_gpus ← denominator(α)×G_prefill`、`tolerance=0.03`）；页面引注 "[App.B, line 46]" 经核对确为 appendixB.tex 第 46 行。
- 图读数：Fig.5（FTL 91.5s→2.94s，log2 6.52→1.56；吞吐 1.004→0.989）、Fig.9（R1 3.64→0.08；70B 0.97→0.47；405B 2.08→0.23；8B 0.43→0.37）、Fig.10（0.5 平台 0.42，全区间蓝线最高，3.5 左端 0.971≈蓝 0.984 / 右端 0.045）、Fig.12（蓝 0.37–1.23，红 0.97–1.82，合成 0.37–1.82）均与页面表述吻合；Fig.1/Fig.6/Fig.7/Fig.8/Fig.14 的标题、图例配色、"两条实线"等均与源图一致。
- 引文：'existing provisioned datacenter bandwidth is sufficient to support KV cache transfer without becoming a bottleneck'、'fall short of providing concrete guidance on when and how disaggregation is beneficial'、'largely focused on small-scale testbeds and peak throughput scenarios, without examining the full throughput–interactivity Pareto frontier'、'we also highlight scenarios where disaggregation offers limited benefit—such as serving small-scale models or generation-heavy traffic' 四条均为原文逐字。
- 结构：17 个 <details> 均有对应 <summary>；5 章各有"本章问题"与解答折叠块；前置概念链接 6 个（moe-serving / model-parallelism / chunked-prefill / mla / mqa-gqa / gpu-communication）均真实存在；"Chunked Prefill 页第 5 章"确为该页 "5. 与流水线并行合流：CPP"；dojo:topics 三项与 dojo:tag "推理系统" 均在词表内；`.dojo/scripts/validate.py` 对 index.html 与 overview.html 均返回 validation ok。

## 问题

- [阻断·技术] 第 5 章本章问题第 1 题解答（第 405 行，"对'是否上 PD 分离'的决策…"折叠块）：把分离的模型规模门槛写成"若工作负载属于 prefill-heavy…且模型达到 70B 以上规模，分离值得上"，与论文 introduction 的 ">10B" 以及本页其余全部五处表述（description 第 6 行、dojo:summary 第 7 行、§2.2 第 187 行、§2 本章问题第 2 题 第 212 行、§5.1 清单① 第 369 行、来源表 N1 第 445 行）相矛盾；且同一段落内前句已写 "更大模型（>10B）收益最大"、后句又写 "70B 以上规模…值得上"（同页两处互相矛盾）。｜引文依据：introduction.tex "…disaggregation provides the greatest benefits in prefill-heavy traffic scenarios (i.e., ISL >> OSL) and when serving larger models (e.g., >10B parameters)."；全文（含 Fig.7 与 §4.1）无任何 "70B 门槛" 表述。｜修复要求：将该处 "模型达到 70B 以上规模" 改为与来源一致的 "模型达到 10B 以上规模"（或删去具体阈值，仅保留"更大模型"），并使全页门槛统一为 >10B。｜修复：｜复验：
- [轻微·格式] §4.1 第 300 行（"其一，公式假设 KV 逐层即时传输与计算重叠（$C8$）…"）与 §4 本章问题第 3 题解答 第 355 行（"…KV 逐层即时传输与计算重叠（$C8$）…"）：引文标记写作 `$C8$`，经 KaTeX 渲染为意大利体 "C8"，与全页其余位置的 `<sup>[C8]</sup>` 写法不一致。｜引文依据：不适用｜修复要求：两处 `$C8$` 改为 `<sup>[C8]</sup>`。｜修复：｜复验：
- [轻微·技术] §4.1 符号表（第 295 行）：`$bytes_{element}$` 释义为"每 token 每头 KV 字节数（FP4 下为 0.5 等）"。｜引文依据：system_considerations.tex "$bytes_{{element}}$ indicates the number of KV cache bytes per token"（无"每头"）；公式中已另有 $d_{head}$ 与 $N_{kv\_heads}$ 两个独立因子，且同一括号给出的示例值 0.5（FP4）/ 正文示例 1（FP8）是"每个 KV 元素的字节数"，与"每 token 每头"自相矛盾。｜修复要求：释义改为"每个 KV 元素的字节数"（或去掉"每头"使其与来源一致），保证 0.5/1 取值与公式因子自洽。｜修复：｜复验：
- [轻微·可读性] §5.1 末段（第 380 行）："…；本文侧重分离在什么条件下值得、怎么配、有什么代价。" 以"本文"自称，属页面自我指代，且与全页用"论文"指代原论文的用法相混（同句前半即在说"MoE Serving 页"与本文两个页面）。｜引文依据：不适用｜修复要求：改为无自我指代的说法（如"这份解析侧重…"或直接以主题作主语）。｜修复：｜复验：
- [轻微·可读性] §3 章首（第 226 行）："本章展开第二维，因为它比模型切分更反直觉、更容易做错。" "更反直觉、更容易做错"为无来源支持的临场评价。｜引文依据：不适用｜修复要求：删去该评价，或替换为可核对的中性陈述（如"论文在 §3.2 之后用 Fig.9/Fig.10 单独论证这一维"）。｜修复：｜复验：
- [轻微·可读性] §4.3（第 327 行）："…带宽需求约在 0.4–1.8 GB/s/GPU 之间（…原图注为 "Maximum of egress and ingress bandwidth across various TTLs"）[C8, N8, G9]。具体观察：" 之后直接接图片，无对应条目，"具体观察："冒号悬空。｜引文依据：不适用｜修复要求：把"具体观察："改写为引出图注的完整句子，或补上与 Fig.12 一致的观察条目（如蓝线约 0.4–1.2、红线约 1.0–1.8）。｜修复：｜复验：
- [轻微·技术] §2.3 正文（第 195 行）与 §2 本章问题第 3 题解答（第 219 行）："GQA 没有 down/up 投影这一步…每块的 attention 只用本块自己的 K/V，没有重复计算开销。" ｜引文依据：论文 Fig.4 caption "…processing each chunk independently, using the KV cache from previous chunks but not their outputs"——chunk 的 attention 必须用到前面块的 KV，GQA 与 MLA 的差别只在是否需要重建 K/V，而非"只用本块 K/V"。｜修复要求：改为"每块直接使用已按完整维度存储的 K/V（无需重建）"之类的准确表述。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 6
- 处置：修复
