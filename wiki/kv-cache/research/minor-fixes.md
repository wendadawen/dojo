# 轻微问题清理记录（2026-09-14）

汇总 `review-1`–`review-7.md` 中历轮报告的**全部**轻微级问题，逐条对照当前页面后处理。

- 实际修复：**6** 条
- 判定已不存在（历轮修复的副作用或已被后续轮次复报并关闭）：**19** 条
- 有理由不改：**2** 条

## 修复与判定说明

读完 kv-cache/research/ 下全部 7 份 review-*.md（第 1–7 轮），汇总出 25 条去重后的轻微级问题。本轮实际修掉并复验 6 条：①dojo:summary「约 2.4 GB」→「约 2.44 GB」，与正文四处及 overview 统一；②来源章 C4/C5 来源混挂拆分——C4→vLLM §1、C5→SGLang 论文 §1、§3 及 vLLM §4.3，修掉 C4（65%/30% 显存分布）被误挂到 SGLang §3 的归属错误；③核心问题 Q3 首现处为 MHA/GQA/bf16 补中文全称（多头注意力/分组查询注意力/bfloat16 每元素 2 字节）；④统一自称，「本页公式」→「本文公式」，全页 9 处自称均为「本文」；⑤来源章 h3「外部数字与实验条件」补「（N）」，使 6 个 h3 全部符合 style-guide §1 固定命名；⑥第 4 章本章问题答案中与量级判断相矛盾的「几十份长文档」改「十几份长文档」（0.3M÷0.02M=15）。其余 19 条判定为「已不存在」——均为历轮修复的副作用已被消除（2.5 GB 数值、表格 Unicode 箭头、表头「需重算」、HBM/bf16/算术强度首现解释、callout-inline 未定义类、GQA 歧义表述、description「§3.2」、C1 悬空、d_k/d_head 等价说明、§2 的 20,100 取整、§2 机制句的来源链接、「存储行业惯例」归属、HiSparse 来源、「多数开源模型采用 GQA」概括、标签句堆叠、「需要在此澄清」元话语、引言四问对应、构造示例章号「第 2–4 章」、变量 L_layers/H_kv 写法统一），未做重复改动。另有 2 项如实说明不修的理由（见 unfixed）。全程只改被指出的问题位置，保持章节结构与文风；未改动 overview.html（其数字本就与正文一致）；validate.py 通过。
## 不改的条目及理由

- R7 #2 的第二项子要求（把 C5 的 SGLang 位置由「§1、§3」改为「§1」）未采纳：经 WebFetch 核对 arXiv:2312.07104，§3 首段即含「Unlike existing systems that discard the KV cache after a generation request finishes, our system retains the cache for prompts and generation results in a radix tree」，直接支持 C5「完成后丢弃缓存」，删除 §3 会丢掉一条有效且经 R2/R3/R5/R6 多轮验证的引文。本轮只修该问题中真正的缺陷——C4 被误与 C5 合并挂到「SGLang §3」名下（已分列为 C4→vLLM §1、C5→SGLang §1、§3 及 vLLM §4.3），来源分列后 description 与 blockquote 的「SGLang §1、§3」保持自洽。
- R4 #4 的前半项（删去以「本文」为主语的引导句「本文回答四个问题」）不单独处理：该句已被 R5 #10 的修复要求覆盖，现文为「本文按顺序回答：…」并与核心问题四条一一对齐，采纳 R5 版本（R4 与 R5 要求相左，以后轮为准）。
