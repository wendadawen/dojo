<!-- review-meta
round: 2
page: wiki/moe-serving/index.html
reviewed_content_sha256: 2d26967c5ac81f95
-->
# MoE 大模型推理与服务基础审查记录（第 2 轮）

- 页面版本：b915dd603d4bfc6c19285e006b8789e0d9f6df46
- 审查时间：2026-09-13 19:00
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节（按顺序）：引言（line 107-109）、开始之前 context-box、核心问题（5 条及解答）、第 1 章「一个字是怎么蹦出来的：token、参数与 Transformer 层」、第 2 章「把一个 FFN 换成一排专家：MoE 与 top-k 稀疏激活」（含补充折叠块）、第 3 章「权重放不下了：专家并行与 all-to-all 搬运」（含代码折叠块）、第 4 章「通信不能干等：TBO 与 SBO 的重叠」、第 5 章「一次请求的两个阶段：prefill、decode 与 KV cache」（含补充折叠块）、第 6 章「服务好不好怎么量：TTFT、TPOT、SLO 与 goodput」、第 7 章「放一起还是分开：PD 合设与 PD 分离」、来源与范围说明（C/F/N/构造示例/辅助解释/简化条件）、页面全部 script。
- 机械验证：`python3 .dojo/scripts/validate.py wiki/moe-serving/index.html` → validation ok；第 3 章 Python 代码实际执行，输出与页面「预期输出」逐行一致（含 卡0=6/卡1=2、均匀随机 1000 token 卡0=1003/卡1=997）；readme 前置页与内联概念链接（gpu-execution-model、megamoe、mooncake、expertplex）均真实存在，页面无 research/ 残留路径、无「（待生成）」占位。
- 外部来源核对（已抓原文）：DeepSeek-V3 Technical Report（arXiv:2412.19437 摘要与 §2.1.2）、DistServe（arXiv:2401.09670 §1/§2.1）、DeepSeek《V3/R1 推理系统概览》（2025-03-01，open-infra-index day 6）、Shazeer et al. 2017（arXiv:1701.06538）、ExpertPlex（arXiv:2607.18002 v2）。

## 问题

- [重要·技术] 第 5 章构造示例（正文 line 509-511；本章问题解答 line 554；简化条件 line 690 又复述一次）：同一示例的「无 cache」与「有 cache」两个计数对应的前向次数不一致——无 cache 用 3 次前向（4+5+6），有 cache 却写成 4 项（4+1+1+1=7）。4 个输入 token 生成 3 个 token 只有 3 次前向（prefill 处理 4 个并产出首 token；随后 2 步 decode 各处理 1 个新 token），因此有 cache 应为 $4+1+1=6$，而非 7；若坚持 7（把最后生成 token 的 K/V 也算上），则无 cache 一侧应为 $4+5+6+7=22$，与页面写的 15 不符。两处计数无法同时成立。｜引文依据：页面原文「不存 cache：第 1 步要为 4 个前缀 token 算 key/value；第 2 步前缀变成 5 个，重算 5 个；第 3 步重算 6 个。合计 $4+5+6=15$ 个 token 次」与「存 cache：第一步算 4 个并入库；之后每步只算新 token 的 1 个。合计 $4+1+1+1=7$ 个 token 次」（前向步骤数 3，合计项数 4）；本章问题解答与简化条件均写「15 对 7」。核对依据为页面自身时间线设定（3 步）与算术，非外部来源。｜修复要求：使两侧对应同一前向次数。取 3 次前向时把正文与本章问题答案改为 $4+1+1=6$、结论改为「15 次对 6 次」，并同步改简化条件里的「15 对 7」；或把设定改为「生成 4 个 token」并把无 cache 改为 22。另需与第 6 章「生成 N 个」（=N 个 decode token）的用法对齐，避免同一短语两处含义不同。｜修复：｜复验：
- [重要·技术] 第 3 章（line 337）与第 7 章（line 631、本章问题解答 line 653、660）：同一个「DeepSeek-V3 的部署单元/配比单元」给出两个不同的 decode 规模，页面未说明二者口径不同。第 3 章称 decode 单元是「18 节点 144 张 GPU（EP144）」，第 7 章称「技术报告的一个配比单元是 32 张 prefill GPU 加 320 张 decode GPU」，于是读者读到两个互相抵触的 P:D 配比（144:32≈4.5:1 与 320:32=10:1），且第 7 章本章问题 2 问「举出 DeepSeek-V3 的部署单元数字」，读者无从判断以哪个为准。两个数字各自有据（前者出自在线系统概览，后者出自技术报告），问题在于页面把两者都叫「部署单元」而不加区分。｜引文依据：页面第 3 章「decode 的一个部署单元是 18 节点 144 张 GPU（EP144，每卡只放 2 个路由专家加 1 个共享专家）（数字 N4）」；第 7 章「DeepSeek-V3 技术报告的一个配比单元是 32 张 prefill GPU 加 320 张 decode GPU（数字 N5）」。外部核对：DeepSeek《V3/R1 推理系统概览》「18 nodes with 32 redundant routed experts; each GPU manages 2 routed experts and 1 shared expert」；DeepSeek-V3 技术报告部署章节「The minimum deployment unit for the decoding stage consists of 40 nodes with 320 GPUs」；ExpertPlex v2 §2.4「A reported DeepSeek-V3 deployment uses 32 GPUs for prefill and 320 GPUs for decode in one unit」。｜修复要求：在第 7 章（或 N5）补一句说明两个数字的口径差异（生产部署按 EP144 运行；技术报告给出的是最小部署单元 EP320），使两处不再读作同一数量的两种说法。｜修复：｜复验：
- [轻微·表述] 全页出现第二人称与元话语。line 107「变成你能对话的在线服务」直接以第二人称称呼读者（style-guide 第 12 节：不直接使用第二人称称呼读者）；line 523/525 的 KV cache 示意图把「你」当作示例 token（「输入 [你][好][今][天]」），同样会被读成称呼读者。元话语：line 107「这一页把这条链完整讲一遍」；line 233「下面会看到 DeepSeek-V3 是 256 选 8」；line 256「还有一个隐患先记下来……需要注意」；line 288「但注意一个容易滑过去的点」；line 500「现在把它展开成一个循环」。｜引文依据：不适用｜修复要求：示例 token 换成不含人称的字（如「今/天/天/气」），line 107 改为无人称直陈；删除「下面会看到」「先记下来」「这一个容易滑过去的点」等元话语，直接陈述结论。｜修复：｜复验：
- [轻微·格式] 正文来源引用未采用规范形式，且 C2、F1 未与正文双向对应。style-guide 第 6 节要求正文用 `<sup>[Cx]</sup>` 上标（可组合 `<sup>[C7, F2, N2]</sup>`），本页全部写成「（论断 C1）」「（数字 N4）」一类正文文字；同仓库 megamoe、mooncake 页用的是 `<sup>[C…]</sup>`。另：C2、F1 只在「来源与范围说明」出现（C2 计 1 次、F1 计 1 次），正文第 1 章的 $8d^2$ 约为 $4d^2$ 的 2 倍（line 183）与第 2 章 MoE 输出公式（line 223）都没有回指引用，双向对应在这两条上缺一半。｜引文依据：不适用｜修复要求：正文引用改为 `<sup>[Cx]</sup>` 形式；在第 1 章 8d²/4d² 与第 2 章公式处补 C2、F1 引用。｜修复：｜复验：
- [轻微·格式] 前置 section 顺序与 style-guide 第 2 节不符。规范固定顺序为 reading-time → blockquote.meta（主要依据）→ 引言 → learning-goals；本页把 blockquote.meta（line 161）放在引言、context-box、learning-goals 之后。同仓库 megamoe、expertplex 页为 reading-time 紧接 blockquote.meta。｜引文依据：不适用｜修复要求：把 blockquote.meta 移到 reading-time 之后、引言之前。｜修复：｜复验：
- [轻微·可读性] 第 4 章 callout（line 475）使用未解释、且页面已声明「不在本页范围」的术语：「把 dispatch、专家计算与 combine 融进一个占满全部 SM 的持久 kernel、由 kernel 内的 warp 分工让两者同时推进」——SM、warp、持久 kernel 均未解释，而「简化条件及其限制」第 (5) 条写明「SM、kernel、CUDA Graph 等执行细节不在本页范围」。同句「只是让它与计算同时进行。 把 dispatch…」句号后多一个空格。｜引文依据：不适用｜修复要求：改成不依赖未解释术语的说法并就地给出最小解释，或删去该句只保留指向 MegaMoE 的链接；删掉多余空格。｜修复：｜复验：
- [轻微·技术] 第 3 章 prefill/decode 单元的「每卡专家数」无法由页面自述的专家总数推出，且未说明原因。页面先说「把 256 个专家分片到多张 GPU」，随后说 prefill EP32 单元「每卡放 9 个路由专家」（256/32=8），decode EP144 单元「每卡只放 2 个」（256/144≈1.8）。DeepSeek 在线系统概览的解释是额外设 32 个冗余路由专家（256+32=288，288/32=9、288/144=2），页面未提，做除法的读者会以为数字有误。｜引文依据：DeepSeek《V3/R1 推理系统概览》「Each deployment unit spans 4 nodes with 32 redundant routed experts; each GPU handles 9 routed experts and 1 shared expert」与「18 nodes with 32 redundant routed experts; each GPU manages 2 routed experts and 1 shared expert」。｜修复要求：补一句「另有 32 个冗余路由专家，故每卡 9（prefill）/2（decode）个」，使数字可由 256 专家推出。｜修复：｜复验：
- [轻微·来源] 两处机制/归因描述没有对应 C/N 编号。第 2 章补充折叠块（line 260）「GShard（2020）把它搬进 Transformer 并用 top-2；Switch Transformer（2021）简化为 top-1」——Switch Transformer 未出现在来源列表，GShard 仅在 C7 中为 all-to-all 引用；第 7 章（line 631）「Mooncake……把全集群闲置的 CPU/DRAM/SSD 组织成全局 KVCache 池，再由 Conductor 按缓存复用与负载同时最优来路由请求」——无 C/N 引用。｜引文依据：不适用｜修复要求：为 GShard/Switch 的 top-k 事实补来源编号或明确标注为背景补充，为 Mooncake 的描述补来源或注明转述出处。｜修复：｜复验：

## 已核对且无问题的条目（供复验参考）

- 671B / 37B / 约 5.5%：`37 ÷ 671 ≈ 0.055` 复算一致；DeepSeek-V3 摘要原文「671B total parameters with 37B activated for each token」。
- 每层 1 shared + 256 routed、top-8、61 层、前 3 层稠密：技术报告 §2.1.2 原文「Each MoE layer consists of 1 shared expert and 256 routed experts…8 experts will be activated for each token」「We substitute all FFNs except for the first three layers with MoE layers」「61」层；匹配 N2/N3。
- sigmoid 亲和度 + 偏置选 top-k：§2.1.2 原文「we introduce a bias term bi for each expert and add it to the corresponding affinity scores si,t to determine the top-K routing」。
- $4d^2$ / $8d^2$ 与 2/3 占比：$d=512,d_{ff}=2048$ 复算得 1 048 576 与 2 097 152，比值恰为 2/3。
- N4 在线部署（prefill 4 节点 32 GPU/EP32、每卡 9 路由+1 共享；decode 18 节点 144 GPU/EP144、每卡 2 路由+1 共享；prefill 双批次重叠）与 N5（技术报告 32 prefill/320 decode）均与来源一致（见上「重要·技术」第 2 条的引文）。
- N6：DistServe 摘要「7.4× more requests or 12.6× tighter SLO…>90% of requests stayed within latency constraints」；goodput 定义「the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%)」；TTFT/TPOT 定义与「250 words/min」原文「TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min)」。
- Shazeer 2017：137B 参数与 noisy top-k 门控属实。
- C8 TBO/SBO 定义：ExpertPlex v2 原文「TBO, which overlaps one microbatch's communication with another's computation, or single-batch overlap (SBO), which overlaps communication with shared-expert computation in the same microbatch」。
- 第 7 章 ExpertPlex 描述「跨阶段共享同一份 MoE 专家，attention 按阶段各自独占整卡」：ExpertPlex v2「Disaggregating attention gives each phase full GPUs instead of GPU partitions」。
- 第 3 章 dispatch 计数表与代码输出一致（6 对 / 2 对；均匀随机 1003/997）。
- 数学符号：全页变量与公式均由 KaTeX 渲染，未发现裸 Unicode 数学字符（`×`、`→` 经 .dojo/scripts/validate.py 明确豁免为普通排版字符）。
- 结构完整：核心问题 5 条、七章各有「本章问题」且每题有解答折叠块、来源章节六个固定 h3 齐备、overview.html 与 index.html 互链、前置概念页真实存在。

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 6
- 处置：修复。两条重要问题（第 5 章 K/V 计数自相矛盾；第 3 章与第 7 章 DeepSeek-V3 部署单元数字口径未区分）须在第 2 轮修复中关闭；六条轻微问题建议一并处理，若保留须给出接受理由。全部学习目标与两级问题块已由正文完整回答，页面结构、公式渲染与本地资源均正常。
