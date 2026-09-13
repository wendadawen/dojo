<!-- review-meta
round: 8
page: wiki/dualpath/index.html
reviewed_content_sha256: 66cad9c942832743
-->
# DualPath 审查记录（第 8 轮）

- 页面版本：784d42ea90de2e1697745df2d5e79073a2b3280f
- 论文版本：arXiv:2602.21548v2（2026-02-26）
- 审查时间：2026-09-13 22:44
- 审查者：独立子代理（未参与写作与前序任何轮次）
- 已完整阅读章节：核心问题（5 条）、贯穿示例、1. agentic 推理的存储 I/O 瓶颈由三因素叠加、2. DualPath 的双路径数据流与块布局、3. 双路径不引入新瓶颈的 P/D 区间、4. CNIC-centric 流量管理、5. Adaptive Request Scheduler、6. 实验、7. 方法评价、来源与范围说明；含全部折叠块（本章问题解答、中间步骤补充、Algorithm 1 伪代码）与 14 条图注、14 张图。
- 核对方式：用 pdftotext 抽取 arXiv:2602.21548v2 全文（含 Appendix A.1/A.4/A.5、Table 1/2/3、所有公式与 Figure 标题）逐条回源；图内数值以像素测量核对（Figure 1/3/7/9/11/12/13/14/15）。

## 问题

- [重要·可读性] §1.3（第 162 行）、§7.1（第 687 行）、§7.4（第 770 行）：章节交叉引用比页面实际编号大 1。页面正文章节号为 3=「双路径不引入新瓶颈的 P/D 区间」、4=「CNIC-centric 流量管理」、5=「Adaptive Request Scheduler」，核心问题解答也写明「完整论证在第三章」（第 91 行）。但 §1.3 把「如何让两类闲置资源被利用而不把推理挤爆」（流量隔离，属第四章）写成「后文第五章要解决的」；§7.1 把 P/D 不等式推导（第三章）写成「第四章的不等式推导」；§7.4 同一句内把 P/D 推导写成「第四章」、把 vlarb 权重配置（第四章）写成「第五章」，而同句却正确使用了「第七章的『解决了什么 / 没解决什么』」。引文依据：页面自身标题「3. 双路径不引入新瓶颈的 P/D 区间」「4. CNIC-centric 流量管理」；核心问题解答第 91 行「完整论证在第三章」。修复要求：把三处改为页面实际章节号——第 162 行「第五章」→「第四章」，第 687 行「第四章」→「第三章」，第 770 行「第四章」→「第三章」、「第五章」→「第四章」；并统一约定「第X章」只指页面章节、指论文处一律写「论文 §X」。修复：｜复验：

- [重要·技术] §1.3 图注（第 158 行，Figure 1 左）：把 DE 的 storage NIC 写成「完全空闲」。论文原文为「the SNICs on decode engines remain largely idle」（§3，第三因素段），是「largely」而非「完全」；同页内嵌的 Figure 1 左图在 Decode 侧画有一支指向 KV Persistent Storage 的存储流量箭头，且页内 §4.2 也描述 KV-Cache 会「写到存储后端」、§4.1 描述 decode 期间每积满一个 block 即落盘（即 DE 有写侧存储 I/O）。引文依据：「This design centralizes all storage I/O pressure on the prefill-side SNICs, while the SNICs on decode engines remain largely idle.」（§3）。修复要求：把第 158 行图注与第 77 行解答中「完全空闲 / 完全闲置」改为与原文一致的表述（如「基本空闲，读取带宽未被利用」），或明确限定语为「KV-Cache 读取」。修复：｜复验：

- [轻微·技术] §6.2（第 572 行）与贯穿示例（第 112 行）：把「1.87×」绑定到「64K MAL / 1024 agents」这一具体配置。论文只写「On DS 660B, DualPath achieves up to 1.87× over Basic」（§7.3），未把 1.87× 归给某个 agents 数；页面 §6.2 自身也写「DS 660B：最高 1.87×」（即跨配置上界）。引文依据：「On DS 660B, DualPath achieves up to 1.87× over Basic, and demonstrates performance with Oracle」。修复要求：删去「64K MAL / 1024 agents 下」这一配置限定，或改为「DS 660B 离线实验最高（含 64K MAL 配置）」。修复：｜复验：

- [轻微·格式] 来源与范围说明（第 755 行）：「N-1 至 N-26 全部来自论文正文或 Appendix §A」。正文实际出现的 N 编号仅 N-1/4/6/7/8/12/14/15/18/19/20/26（12 个），无 N-2、N-3、N-5 等，与「26」条不符（对照 C 编号给出 C-1..C-33 的完整分组，N 没有分组清单）。引文依据：不适用。修复要求：改为与页面一致的表述（如「页面 N 编号论断均来自论文正文或 Appendix §A」），或补出 N-1..N-26 的分组清单。修复：｜复验：

- [轻微·格式] 全文自指用词不统一：第 218/739/758/761/770 行用「页面」，第 476/505/764 行用「本文」（「是本文的推断」「本文将其含义解释为」）。在论文解析页中「本文」易被读成所解析的论文，与同句的「论文」混淆。引文依据：不适用。修复要求：统一改用「页面」，把三处「本文」改为「页面」。修复：｜复验：

- [轻微·格式] 无来源支持的编辑性陈述：（a）§1.3 图注（第 158 行）「这是 DualPath 名字的由来——一条路径不够，要两条」，论文未说明命名由来；（b）§2.4（第 216 行）「这一设计沿用业界 prefix cache 的常规做法」，未给来源。引文依据：不适用。修复要求：（a）改为「名字对应两条 KV-Cache 加载路径」并标为页面解释，或删除；（b）标注为辅助解释（页面推断）或删除。修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 4
- 处置：修复

（本轮独立核对通过、未报为问题的关键项，供下游留档：Table 1 五档数值、Table 2/3 全部数字、§4.2 eq.(1)-(9) 与页面 F-1..F-12 逐式相符、§3「s≤g 与 eq.(1) 不一致疑为笔误」的判断成立、§A.1 vlarb 四值、§A.4 的 α/β「3 秒 / 5 秒」与 compute quota 300ms、§5.2 的 5-7μs / 1μs、§7.3/7.4/7.5/7.6 全部加速比与消融百分比、§8.2 的 69→681 GB 与 r/r²/r³、§A.5 的 Layer/Full Block 形状与 trie、外部会议归属（Mooncake FAST'25、PrefetchOnly SOSP'25、KVPR/TailorKV 均 Findings of ACL 2025、Strata/LayerKV 为 @misc、DistServe OSDI'24、Splitwise=51st ISCA）、Figure 3 右「5→10 约 12-13%」读图、Figure 13 的 1.528/1.184、Figure 14 的 1.06、Figure 15 的 TTFT≈22s 均与来源一致；validate.py 返回 success；overview.html 与 index.html 互链、7 个前置概念页与 3 个锚点均存在；全部 alt 无 `$...$`。）