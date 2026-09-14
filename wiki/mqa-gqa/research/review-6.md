<!-- review-meta
round: 6
page: wiki/mqa-gqa/index.html
reviewed_content_sha256: 4b0fbbafacf9bd4a
-->
# MQA 与 GQA 审查记录（第 6 轮）

- 页面版本：dfba1d62405f8c49d5e5659a7ae587b89f869dd4
- 审查时间：2026-09-14 17:06
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题 / 最容易误解 / 1. 为什么 MHA 推理受内存带宽限制 / 2. MQA / 3. GQA / 4. 手算对比 / 5. 边界与后续 / 来源与范围说明（含全部折叠块与图注）
- 核对来源：Shazeer 2019 arXiv:1911.02150（ar5iv 全文）、Ainslie 2023 arXiv:2305.13245（ar5iv 全文）、DeepSeek-V2 arXiv:2405.04434（arXiv HTML v5）Table 1 / §2.1.2 / §3.1.2、NVIDIA Data Center GPU Line Card；overview.html 与 index.html 相互核对；validate.py 返回 `validation ok`。

## 问题

- [阻断·技术] §2「代价」段（index.html:253）、页面级「核心问题」Q2 解答（index.html:81）、§2「本章问题」第 3 题解答（index.html:280）：三处把「MQA 在**大模型**上质量退化」写成 Ainslie 2023 §1 与 Appendix A 的来源结论，来源不支持「大模型」这一限定条件｜引文依据：Ainslie 2023 §1 原文 "However, multi-query attention (MQA) can lead to quality degradation and training instability"（无条件、无模型规模限定）；Appendix A 原文 "We find that multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks"（条件是长输入任务，不是大模型）；该论文涉及模型规模的表述仅有 "larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity" 与 "we expect GQA to present a particularly good trade-off for larger models"，均只谈带宽/容量削减，未断言大模型上质量退化。页面自身的 C5 条（来源说明）也只写 "Ainslie 2023 §1 'MQA can lead to quality degradation'"，与正文的「大模型」限定自相矛盾｜修复要求：删去正文三处「大模型」限定——Ainslie 只在 §1 无条件下称 MQA can lead to quality degradation、在 Appendix A 把微调不稳定归因于长输入任务；若确要保留「大模型质量更差」的判断，必须在正文明确标注为「由大模型头数更多、MQA 削减更激进推出」的推断并声明非论文结论｜修复：｜复验：

- [轻微·来源] 「来源与范围说明 · 外部数字与实验条件（N）」N1 条（index.html:541）：把 Shazeer 2019 一手 PDF Table 1 / Table 2 的数字描述为「二手转述报告」，归因不准；且同句「不引用一手 PDF 之外的精确数字」与本条所列数字实际出自一手 PDF 相矛盾｜引文依据：Shazeer 2019 Table 1（WMT14 EnDe test BLEU，beam-1/beam-4）"multi-head (baseline) 27.7 / 28.4"、"multi-query 27.5 / 28.5"；Table 2（TPUv2 微秒/输出 token，Inference enc.+dec.）"multi-head 1.7 + 46"、"multi-query 1.5 + 3.8"——28.5/28.4 与 46µs→3.8µs 均出自一手 PDF｜修复要求：把这句改为准确归因（数字出自 Shazeer 2019 Table 1、Table 2，属一手来源），或改写为「本页不引用 Table 1/2 的精确数字，只取 abstract 的定性结论」，删除「二手转述报告」表述｜修复：｜复验：

## 已核对且无问题的要点（供复验参考）

- C1/C2/C3/C4/C5/C6/C7/C8/C9 各条引文均能在对应来源定位，逐字或近义一致：Shazeer §1 "memory bandwidth necessary to reload the large 'keys' and 'values' tensors"；§2.4.1 "When n≈d or b≈1, the ratio is close to 1" 与 Θ(n/d+1/b)；§3.1 Θ(1/d+n/(dh)+1/b) 与 "We have reduced the offensive n/d by a factor of h"；§3 "the different heads share a single set of keys and values"；Ainslie §2 "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head"、§2.1 均值池化优于选首头/随机初始化、§3.1 "For α=0.05, training took approximately 600 TPUv3 chip-days"、§3.3 Figure 4/5、Figure 6 "We selected 8 groups as a favorable middle ground"、Appendix A。
- Table 1 数值逐格核对：MHA-Large 0.37/46.0、MHA-XXL 1.51/47.2、MQA-XXL 0.24/46.6、GQA-8-XXL 0.28/47.1；速度 1.51/0.24=6.3×、1.51/0.28=5.4×，质量差 0.6 / 0.1，均与页面一致。
- 投影张量形状核对：Shazeer MHA P_k[h,d,k]、P_v[h,d,v]、P_o[h,d,v]，MQA P_k[d,k]、P_v[d,v]、P_o[h,d,v]——页面表格逐格一致。
- 公式复算：手算 512/256/128 元素、5120/2560/1280、10240/5120/2560 字节全部正确；真实规模 2·128·128·4096·80·2≈2.15e10 字节（21.5 GB）、GQA-8 1.34 GB、MQA 168 MB 全部正确。
- NVIDIA 数字核对：V100/A100/H100 SXM FP32 15.7/19.5/67 TFLOPS（67/15.7=4.3×）、带宽 900/2039/3352 GB/s（3352/900=3.7×），与页面一致。
- DeepSeek-V2 核对：Table 1 MHA 2n_h d_h l、GQA 2n_g d_h l、MQA 2d_h l、MLA (d_c+d_h^R)l；§3.1.2 n_h=128,d_h=128,d_c=512,d_h^R=64 → MLA 576、MHA 32768、比值 1/57，页面一致；93.3% 来自 abstract 且对比对象为 DeepSeek 67B，与页面表述一致。
- 链接：../standard-attention/index.html、../mla/index.html 均真实存在；overview.html 与 index.html 相互链接；无「（待生成）」占位。
- 公式渲染：head 中 KaTeX auto-render 已加载；dojo:summary 无公式、为纯文本；点号 / 箭头经交叉核对为全站既有约定（其他 20+ 页面同样使用 →），不当问题计。
- 结构：页面级「核心问题」5 问、5 个章节「本章问题」各 3 问，全部有解答折叠块且答案独立可读；结构图为 HTML 表格/div（非等宽字符框线），明暗主题与窄屏可读；交互视图无脚本时仍可读；img alt 只有 lightbox 的 alt=""，无 `$...$`。
- 表述通读：未发现「本页将…」「下面来看…」式元话语、会话指代（我/我们/你）、调试叙事、临场评价或 AI 拼接腔；推断性内容（roofline 背景、MQA 广播实现、真实规模推算）均显式标注为推断/构造。

## 结论

- 统计：阻断 1 / 重要 0 / 轻微 1
- 处置：修复

统计：阻断 1 / 重要 0 / 轻微 1