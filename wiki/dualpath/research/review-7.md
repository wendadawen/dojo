<!-- review-meta
round: 7
page: wiki/dualpath/index.html
reviewed_content_sha256: cedc3b37fd813a8f
-->
# DualPath 审查记录（第 7 轮）

- 页面版本：index.html 工作树哈希 4518f4341f739078b71d1ff450ca44bc110e3f76（git hash-object）
- 论文版本：arXiv:2602.21548v2（2026-02-26；HTML 版与 PDF 版均逐字对照）
- 审查时间：2026-09-13 21:48
- 审查者：独立子代理（未参与写作，未参与前序轮次审查与修复；未读取本页 research/ 下任何文件）
- 已完整阅读章节：核心问题（5 题及解答折叠块）、贯穿示例 callout、§1 agentic 推理的存储 I/O 瓶颈由三因素叠加（含 1.1/1.2/1.3、本章问题）、§2 DualPath 的双路径数据流与块布局（2.1–2.4、本章问题）、§3 双路径不引入新瓶颈的 P/D 区间（3.1–3.3、本章问题）、§4 CNIC-centric 流量管理（4.1–4.5、本章问题）、§5 Adaptive Request Scheduler（5.1–5.6、本章问题）、§6 实验（6.1–6.8、本章问题）、§7 方法评价（7.1–7.4、本章问题）、来源与范围说明（全部小节）；另读 overview.html 并核对互链

核对材料：arXiv:2602.21548v2 正文全文（§1–§10、Appendix A.1–A.5）与 Figure 1/3/4(a)(b)/6/7/8/9/10/11/12/13/14/15 原图（逐张读图）；`.dojo/scripts/validate.py` 返回 "validation ok"；页面 7 个前置概念链接（moe-serving #s5/#s7/#s3、standard-attention、deepseek-moe、mla、dsa、mqa-gqa、gpu-communication）均真实存在且锚点有效。

已核对一致（抽样列出）：Table 1 五个模型比值（117-267 / 47-95 / 39-60 / 13-36 / 4.8-5.8）、trace 157 轮/32.7k/429/98.7%、Fig.3 左 FLOPS 28.8×·PCIe 2.0×·HBM 2.4×、Fig.3 右"30K context + append 300"、I/O-compute 比 14.4×、Fig.1 的 40%/80% GPU 利用率、§4.2 记号与 F-1/F-2（T_p=Bs/(Dg²)、T_c=Bs/(Pg²)）、四条不等式与 F-12 汇总式（逐式复算：2Bs/g≤B、Bs(1+D/P)/g≤B→P/D≥s/(g−s)、(P/D+2)Bs/g≤B→P/D≤(g−2s)/s、(2P/D+1)Bs/g≤B→P/D≤(g−s)/(2s)、(3+2P/D)Bs≤M→P/D≤(M/Bs−3)/2）、g=8/s=1/M≈500/Bs≈50 代入得 1/7≤P/D≤7/2、A.1 四条 IB 配置、cudaMemcpyAsync 5-7 µs vs RDMA Write ~1 µs、Algorithm 1 伪代码、α=3 s/β=5 s/compute quota 300 ms、Z=1.05×公式、§7.2 数据集表、SGL(MC) 基线构成（commit 19089aa）、P/D 默认 2P4D/1P2D/1P1D、1.87×/1.78×/1.09-1.85×、1.82-1.99×、1.64×/2.46×、SLO TTFT≤4 s 与 TPOT≤50 ms、1.67×/2.25×、-17.21%/-38.19%/-45.62%、1.53→1.18 与 1.06（Fig.13 实测 1.528/1.184、Fig.14 首点 1.06，均与图一致）、Table 3（3,167s/3,201s；1.739/1.847、0.228/0.194、0.039/0.036；22×；CPU<10 cores）、working set 69→681 GB 与 r³ 口径（原文确为"r times more machine hours and r² times more storage (cost scaling as r³)"，页面注明不指部署成本，正确）、引文编号 [39]=Richter et al. 2016（PDF 参考文献确为编号制，[39] 正确）、Strata/LayerKV 为 arXiv 无会议字段、KVPR/TailorKV 为 ACL 2025 Findings、PrefillOnly 为 SOSP'25、DistServe OSDI'24、Splitwise ISCA'24、"performance gain is marginal" 原文逐字一致。

## 问题

- [阻断·技术] §3.2 四个不等式中的 F-4、F-6、F-8 所标注的 §4.2 步骤范围与原文不符（3 处），并据此在 F-4 补充折叠块里写出与原文相反的说明。｜引文依据：原文 §4.2 逐条写明——"Write operations include PE path (4) and DE path (5)"；"For DE CNIC, read operations include PE path 8 and DE paths 3/6"；"Write operations include PE paths 7/9 and DE path 7"；同一段还给出"只算 PCIe 侧压力"的理由："the total traffic on the PCIe side is always greater than or equal to the switch-direction traffic ... we only need to compute the pressure on the PCIe side"。页面 F-3 引的 "PE paths (3) and (5)" 与 F-6 的 "DE paths 3/6" 都与原文一致，说明页面确实按原文图号标注，其余三处属错位：F-4 写 "DE path 步骤 (3)(4)"（原文 DE path 5）；F-6 写 "PE path 步骤 (6)(7)"（原文 PE path 8）；F-8 写 "PE path 步骤 (8)(9) 与 DE path 步骤 (6)(7)"（原文 PE paths 7/9 与 DE path 7）。F-4 下方"补充"折叠块据此写"F-4 把两类流量合并成一条不等式，可理解为对 PCIe 侧总压力做单边近似（推断，论文未展开该近似）"——原文并非近似，而是明确把两条 PE CNIC→PE GPU 的写入（PE path 4 与 DE path 5）合并，并说明只计算 PCIe 侧压力的理由，故"论文未展开该近似"不成立。｜修复要求：按原文 §4.2 改正三处步骤范围（F-4 的 DE 侧改为 DE path 5；F-6 的 PE 侧改为 PE path 8；F-8 改为 PE paths 7/9 与 DE path 7），并重写 F-4 的补充折叠块：删去"单边近似 / 论文未展开"，改为原文"只计算 PCIe 侧压力（PCIe 侧流量≥交换机侧流量）"这一理由。｜修复：｜复验：

- [重要·技术] §6.8 的 Figure 15 图注与图中内容不符。｜引文依据：原文 Figure 15 caption 为 "48P96D offline inference metrics. 1e7 is the scaling factor of Prompt TPS."；图内图例只有三条序列 Prompt TPS、TTFT、Running agents，没有"调度延迟"序列，且 TTFT 曲线在任务开始处升至约 22 s。页面图注写"Prompt TPS 在 48P96D 下保持稳定；调度延迟维持低位"，alt 同样写"Prompt TPS、调度延迟等随时间的变化"。｜修复要求：图注与 alt 改为图中实际序列（Prompt TPS、TTFT、Running agents）；若要保留"调度开销低"的结论，改引 §7.6 正文 "scheduler CPU usage remains below 10 cores" 并注明来源，不再挂在 Figure 15 名下。｜修复：｜复验：

- [轻微·表述] 元话语与自我指代。｜引文依据：不适用｜位置：§2.1 末句"本节重点描述它如何与双路径加载耦合。"；§3 引言"本节逐步推导并把典型配置代入验证。"；§3.2 引言"这里把中间过程压缩，列出每条不等式的物理含义与最终形式："；§4.3 末句"本节按实际配置数值描述。"；§7.1"读者能据此判断自己的集群配置是否在覆盖范围内"。｜修复要求：改写为以论文/机制为主语的陈述（如"§4.2 的中间步骤在此压缩，只列出每条不等式的物理含义与最终形式"），或删除该类元话语与"读者"称谓。｜修复：｜复验：

- [轻微·技术] §2.1 把"每层传输的数据量"标注为"（Full Block 维度）"，与同页 §2.4 的块定义矛盾。｜引文依据：原文 §A.5 "A Layer Block is a byte tensor with shape [1,tokens,bytes] and stores one-layer KV-Cache ... a Full Block has shape [layer,tokens,bytes]"；页面 §2.4 亦写"层间传输一律用 Layer Block"。每层传输只能是 Layer Block，不可能是 Full Block 维度。｜修复要求：将"（Full Block 维度）"改为"（Layer Block 维度，$[1,\text{tokens},\text{bytes}]$）"，或删去该括注。｜修复：｜复验：

- [轻微·技术] §5.3 与 §5.6 对 $Z=1.05$ 的口径不一致，且 §5.6 解答对 §A.4 的引用错位。｜引文依据：原文 §A.4 "The short reading queue threshold α … is set to the number of tokens we can read during 3 seconds, and the unfinished token upper limit β … 5 seconds. Those values are profiled in advance."——"profiled in advance" 只针对 α 与 β，不涉及 Z；原文 §6 只给出 $Z=1.05\times(\sum_{r\in R}len_r+\sum_{e\in E}tok_e)/|E|$ 的公式，未解释 1.05 的取法。｜修复要求：§5.3 的"留 5% 余量以容忍 HBM 碎片与估计误差"须标注为本文解释/推断（论文未说明 1.05 的取法）；§5.6 解答删去"Appendix §A.4 也只说「profiled in advance」"这一指向 Z 的引用。｜修复：｜复验：

- [轻微·技术] §4.2 以引号引用的英文与原文词序不符。｜引文依据：原文 §5 "we propose a CNIC–centric data transfer approach which is widely adopted in our production deployment"。页面写作「in our production deployment widely adopted」。｜修复要求：改为原文语序，或改为不加引号的转述。｜修复：｜复验：

- [轻微·表述] §7.1 "agent 数 24× 翻倍 JCT 几乎不变"表述含混（24× 与"翻倍"重复且互相冲突）。｜引文依据：不适用｜修复要求：改为"agent 数 24× 而 JCT 几乎不变"，与 §6.8 的"agent 数翻 24×，JCT 几乎不变"统一。｜修复：｜复验：

## 结论

- 统计：阻断 1 / 重要 1 / 轻微 5
- 处置：修复

补充说明（不影响统计）：§1.2 图注"5→10 仍有约 12-13% 提升"按 Figure 3 右读图约为 15-16%（batch 5 处约 2.55×、batch 10 处约 2.95×），页面已标注"读图估计"，误差在人工读图范围内，本轮不单列为问题；如后续要收紧，可改为"5→10 仍有约 15% 提升（读图估计）"。