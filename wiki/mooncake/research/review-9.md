<!-- review-meta
round: 9
page: wiki/mooncake/index.html
reviewed_content_sha256: d6da1a74afce17f7
-->
# Mooncake审查记录（第 9 轮）

- 页面版本：761cd59396096ef695dd95d18a81a847ec69e082
- 审查时间：2026-09-14
- 审查者：独立子代理（未参与写作与前序轮次）
- 已完整阅读章节：核心问题、常见误解、1. 为什么把 prefill 与 decode 拆开、2. Mooncake 架构总览：组件、KVCache 池与四步工作流、3. KVCache-centric 调度算法、4. 长上下文 prefill 的多节点与传输重叠、5. 过载场景下的早拒绝与预测、来源与范围说明（含全部折叠块与图注）
- 来源获取：arXiv:2407.00079v4 全文 HTML（含 Figure 8 矢量图 `prefill_global_scheduling.svg`，本地渲染后按刻度测量）

## 核对过的关键项（均一致）

- 图 8 平均 TTFT：像素测量（y 轴刻度 0→1101px、50→924px，3.532px/单位）读得 KVCache-centric 6.26 / cache-aware 14.36 / load-balancing 60.41 / random 92.07，与正文、N8、overview 完全一致；SLO 虚线位于 y≈994px，换算 (1101−994)/3.532≈30.3s，页面「≈30s」正确，且仅前两者低于该线。
- Table 3 拒绝数 4183/3771/3589、trace 23,608 条/1 小时、平均输入 7,590/输出 182、块大小 512（§4.1）、LRU 1,000→50,000 命中率 30%→50%、>50% 块零访问、prefill_chunk 典型 >1,000、TTFT_P90=10×/TBT_P90=5×、真实负载 TTFT 30s / TBT 0.1s/token、8×A800-SXM4-80GB + 800 Gbps RDMA、ArXiv +20%/L-Eval +40%、模拟 50%–525%、真实 +75% 且 TBT ~100% vs 57%、§9 的 ~50%/~90%（papers.cool）：逐条与论文原文一致。
- Algorithm 1 伪代码逐行比对（第 3–31 行、两条分支条件、末尾 TransferKVCache）与论文 Algorithm 1 一致。
- 第 3 章贯穿示例全部算术可复算：A 4.524、B 9.012、C 3.036、纯 LB 4.572，与结论一致；复用率 66.7%、块数 16/20/24 均自洽。
- KVCache 估算复算：2×80×8×128×2=327,680 B=320 KiB/token；12,288×320 KiB≈3.75 GiB；128k≈40 GiB。正确。
- C36（HTTP 429）、C11（swapping/replication）、C33（vLLM 本地复用）、C24（离线拟合模型）、CPP 两点收益、S·T 占用代价、§7.1–7.4 全部定位到原文并核对通过。
- validate.py 通过；前置概念页 kv-cache/paged-attention/pcp-dcp/chunked-prefill/prefix-caching 均存在；无「（待生成）」占位；无 alt 含 `$...$`；无 Unicode 数学字符落在公式定界符之外。

## 问题

- [轻微·格式] 来源与范围说明·公式与来源（F）｜F1｜F1 在「公式与来源（F）」中列出，但正文全篇无 `<sup>[F1]</sup>` 上标引用（F2 有，见第 4.3 节），违反 C/F/N 与正文的双向对应；正文第 2 章 KVCache 大小估算处已使用该公式却未标注｜引文依据：不适用（`grep` 全页仅得 `[F2]` 一处，无 `[F1]`）｜修复要求：在正文使用该公式处补 `<sup>[F1]</sup>`，或将 F1 移出 F 表｜修复：｜复验：
- [轻微·格式] 正文 2.3 节「按四步推进（论文图 4）<sup>[C10]</sup>」与来源栏 C10 行｜C10 来源栏仅写「§1」，但该处正文引用的四步工作流与「论文图 4」位于论文 §3｜引文依据：`Figure 4 : Workflow of inference instances.` 出现在位置 17169，处于 §3（15412–19917）区间；§1 内仅有 Figure 1（位置 3875）｜修复要求：将 C10 来源栏补为「§1；§3（Figure 4）」｜修复：｜复验：
- [轻微·技术] 第 5.4 节端到端结果表第四行｜该行把真实负载端到端实验（§8.1.3，Mooncake-[10P+10D] vs vLLM-[20M]）标为「真实回放 23,000 请求」，但论文 §8.1.3 未给出请求条数；23,000 这一数字出自 §6.2（Figure 8 调度实验）与 §8.2（过载实验）｜引文依据：§8.1.3 原文 “We further utilize 10 prefill instances and 10 decoding instances … to replay real request traces and conduct load tests”，全段无条数；`23,000` 全文仅出现于 §6.2/§8.2 语境｜修复要求：删去该行的「23,000」或改为「真实负载回放」｜修复：｜复验：
- [轻微·表述] 正文 3.3 节｜含元话语/临场评价：「Algorithm 1 末尾还有一步容易被忽略」「这一步看似只是「搬运」，但它的实际作用是……」｜引文依据：不适用｜修复要求：改为直接陈述，如「Algorithm 1 末尾在选定 $p$ 后还有一步……」「该步骤的实际作用是热点缓存的自动复制」｜修复：｜复验：

## 结论

- 统计：阻断 0 / 重要 0 / 轻微 4
- 处置：可发布（4 项轻微问题不推翻核心结论，修复要求明确且可复验）

> 本轮所列问题的处理结果见 `minor-fixes.md`。
