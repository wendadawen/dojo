<!-- review-meta
round: 4
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: 0838c5664343349c
-->
# MQA 与 GQA 审查记录（第 4 轮）

- 页面版本：index.html 工作树哈希 f387944458495a982cc0bc3bd853c36a0651c4077a7a8b6355f7fbad1198b973
- 审查时间：2026-09-13
- 审查者：独立子代理（未参与写作，未读取本页 research/ 任何文件）
- 已完整阅读章节：核心问题（含 5 个解答折叠块）、最容易误解、1. 为什么 MHA 推理受内存带宽限制、2. MQA——所有 query 头共享一组 K/V、3. GQA——在 MHA 与 MQA 之间插值、4. 手算对比、5. 边界与后续、来源与范围说明（含全部 4 个折叠块与各章本章问题折叠块）
- 来源核对方式：WebFetch 抓取 arXiv:1911.02150（ar5iv 全文）、arXiv:2305.13245（ar5iv 全文）、arXiv:2405.04434（ar5iv 全文）；NVIDIA 数据手册口径的 N6 数字按公开规格核对

## 已核对通过的关键项（无问题，不列入问题清单）

- 全部 cache 算式可复算且与标注一致：$2\times4\times64=512$、$512\times10=5120$、$\times2=10240\approx10$ KB；GQA-2 $256/2560/5120$；MQA $128/1280/2560$；真实规模 $h{=}128,d_k{=}128,l{=}80,n{=}4096$ 下 $32768\times4096\times80\approx1.07\times10^{10}\to21.5$ GB、GQA-8 $1.34$ GB、MQA $168$ MB；$32768/576\approx1/57$；$21.5\,\text{GB}/168\,\text{MB}\approx128=h$。
- 表 1 数字与 Ainslie 2023 Table 1 一致（MHA-Large 0.37/46.0、MHA-XXL 1.51/47.2、MQA-XXL 0.24/46.6、GQA-8-XXL 0.28/47.1），倍数换算 1.51/0.24=6.3×、1.51/0.28=5.4× 正确；"每 TPUv4 chip"经原文 "We report time per sample per TPUv4 chip" 核实；"仅用于 decoder self-attention 与 cross-attention"经原文核实。
- Shazeer 引文逐句核实：§3 "Multi-query attention is identical except that the different heads share a single set of keys and values."；§3.1 "We have reduced the offensive n/d by a factor of h"；§2.4.1 "the ratio of memory access to arithmetic operations is Θ(n/d+1/b)" 及 "When n≈d or b≈1, the ratio is close to 1, causing memory bandwidth to be a major performance bottleneck"；§2.2 投影形状 $P_k,P_v$ 为 $[h,d,k]/[h,d,v]$。
- Ainslie 引文核实：§2 GQA 定义句、Appendix A 训练稳定性句、§2.1 α 定义句、N3 第 303 行 Figure 6 引文 "increasing the number of groups from MQA only results in modest slowdowns initially...We selected 8 groups as a favorable middle ground." 均为原文。
- DeepSeek-V2 核实：abstract "reduces the KV cache by 93.3%...Compared with DeepSeek 67B"；§2.1.4 Table 1 MLA 行为 $(d_c+d_h^R)l$；配置 $n_h{=}128,d_h{=}128,d_c{=}512,d_h^R{=}64$；§2.1.2 Eq.(9)-(11) 存在。
- 页面链接：../standard-attention/index.html、../mla/index.html 均真实存在；无"（待生成）"占位；无指向 research/ 的路径；overview.html 与 index.html 相互链接。
- `.dojo/scripts/validate.py wiki/mqa-gqa/index.html` 返回 "validation ok"。

## 问题

- [重要·技术] 第 3 章 "uptraining" 第 1 步（index.html:316）：把"保留预训练信息程度"这条理由的出处标为 Ainslie 2023 §2.1，但 §2.1 全文并无该理由，理由句实际在 §3.3｜引文依据：§2.1 原文只有 "The projection matrices for key and value heads are mean pooled into single projection matrices, which we find works better than selecting a single key and value head or randomly initializing new key and value heads from scratch."；页面所引 "the degree to which information is preserved from the pre-trained model" 出自 §3.3 Ablations（Figure 4 讨论）："Intuitively, results are ordered by the degree to which information is preserved from the pre-trained model."｜修复要求：把该理由句的出处改为 §3.3（Figure 4）；§2.1 仅保留"均值池化优于选单头/随机初始化"｜修复：｜复验：
- [重要·技术] 第 1 章性能分析段（index.html:172）："现代 GPU 的算力增长（…15.7→19.5→67 TFLOPS）远快于内存带宽增长（900→2039→3352 GB/s）"与同页 N6 自报数字不符，4.3 与 3.7 之差不足以称"远快于"｜引文依据：页面 N6 "算力增长约 4.3 倍、带宽增长约 3.7 倍"；据页面数字 67/15.7=4.27、3352/900=3.72｜修复要求：删去"远"，改为"快于"或直接并列写出"算力约 4.3 倍、带宽约 3.7 倍"｜修复：｜复验：
- [重要·表述] 全文自我指代（index.html:65 两处、99、138、142、257、475、486、505、513、535、541、564、568）：以"本页/本文"为主语，共 13 处，含页面级核心问题第 5 问"为什么说本页是 MLA 的前置？"及第 5 章小标题，属 check.md 表述维度明列的不合格表述｜引文依据：不适用｜修复要求：逐处改写为不含自指的表述（如"与 MLA 的关系""不展开""此处不展开""来源说明"），核心问题与章节问题同步修改，改写后核心问题答案仍须指向完整论证所在章节｜修复：｜复验：
- [轻微·技术] 第 3 章 uptraining 第 2 步（index.html:317）："Ainslie 2023 §3.1 报告…5% uptraining 已显著提升质量，10% 后收益递减"归错章节，该 α 消融在 §3.3（Figure 5）；§3.1（Experimental setup）只给训练开销｜引文依据：§3.1 仅 "For α=0.05, training took approximately 600 TPUv3 chip-days."；Figure 5 α 消融讨论在 §3.3｜修复要求：5%/10% 结论改标 §3.3（Figure 5）；"600 TPUv3 chip-days" 保留 §3.1｜修复：｜复验：
- [轻微·格式] 第 3 章实验表格单元格（index.html:335、336）：直接出现 Unicode 数学字符 "×"(U+00D7) 与 "−"(U+2212)："速度 6.3×，质量 −0.6" / "速度 5.4×，质量 −0.1"，违反 check.md 第 9 条"表格中无 Unicode 数学字符直接出现"｜引文依据：不适用｜修复要求：改为 KaTeX 书写，如 $6.3\times$、$-0.6$｜修复：｜复验：
- [轻微·技术] overview.html 第 2 节：把 Θ(n/d+1/b) 标为 "Shazeer 2019 §3.1 的性能分析给出"｜引文依据：§2.4.1 原文 "the ratio of memory access to arithmetic operations is Θ(n/d+1/b)"；§3.1 给的是 MQA 式 "Θ(1/d + n/(dh) + 1/b)"｜修复要求：overview 中该式改标 §2.4.1｜修复：｜复验：
- [轻微·技术] 第 1 章折叠块"补充：roofline 模型与'算术强度'的含义"（index.html:174-177）：算术强度/roofline 脊点定义及"增大 batch 只能摊薄 $1/b$ 项…只能靠减少 cache 本身"等判断无来源支撑｜引文依据：不适用｜修复要求：补注来源，或标注为通用背景知识/推断｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 3 / 轻微 4
- 处置：修复

（本轮无阻断项；核心数字、公式与三篇来源的引文均已逐条回源核对通过。3 项重要问题为两处来源章节归属错误与 1 项全文性自我指代，修复后可进入复验。）