# prefix-caching 内容范围

## 1. 概念歧义处理

prefix caching（前缀缓存）：跨请求识别公共前缀并复用其 KV cache。与 context caching（上下文缓存，Strata 论文对同一族机制的总称）、prompt caching（OpenAI/Anthropic 的 API 层叫法）基本同义；与 OS 的 page cache、CPU 的 cache 无关。与 block-level sharing 的关系：页粒度共享是前缀缓存的实现基础（分页使共享单元明确）。处理：已裁定——本页以 SGLang RadixAttention（NeurIPS 2024）的 radix tree 实现为主线，vLLM/Mooncake 的哈希实现作并列呈现（两种实现都是主流，Strata 论文 §6 也并列列出）；"context caching / prompt caching"作为同义叫法在开头说明。

## 2. 概念含义

- 名称：prefix caching（前缀缓存）；同义叫法 context caching、prompt caching
- 简要定义：把已完成请求的 KV cache 按前缀留在显存里，后续请求若与之前请求共享前缀（系统提示词、对话历史、文档），匹配到的部分直接复用、跳过 prefill 计算。
- 正式定义（与 SGLang 论文一致）：运行时自动、系统地复用 KV cache：与生成完即丢弃缓存的系统不同，把提示与生成结果的 KV cache 保留在 radix tree 中，支持高效前缀搜索、复用、插入与逐出，配合 LRU 逐出与 cache-aware 调度提升命中率（SGLang §3.2 "automatic and systematic KV cache reuse during runtime... retains the cache for prompts and generation results in a radix tree, enabling efficient prefix search, reuse, insertion, and eviction"）。
- 本文语境：单实例（单机/单副本）显存内与跨内存层级的前缀复用机制本体。
- 包括：共享前缀的来源（system prompt、多轮对话、few-shot、RAG 文档）；radix tree 的结构与匹配/插入/逐出；LRU 从叶子逐出与引用计数；命中率与页大小的关系；两种实现（radix tree vs 哈希前缀页）。
- 不包括：分层存储与传输效率（Strata 页）；近似/语义缓存（不要求精确前缀匹配的另一族工作）；cache-aware 调度策略细节（LPM 等，SGLang 论文内容，本页一句提及）；跨实例全局缓存池（Mooncake）。
- 相邻概念：块级共享（paged-attention 页，实现基础）；KV cache 逐出策略（本页 LRU，限于 radix tree 语境）；delay hit（Strata 页概念——缓存未就绪时的重复计算问题，本页结尾一句引出）。

## 3. 学习目标

### Q1：跨请求到底能复用什么——共享前缀从哪来？

- 完成答案：KV cache 只依赖 token 序列（与查询无关，kv-cache 页结论），所以任何两个请求只要开头 token 序列相同，这段前缀的 KV cache 就是同一份数据，可互相复用。典型来源：system prompt（所有请求共享）、多轮对话历史（新一轮 = 前几轮 + 新消息）、few-shot 示例（同一批示例的多个查询）、RAG/文档问答（同一文档的多个问题）。复用收益：命中部分完全跳过 prefill 计算（省计算）且不重复占显存（省存储）。
- 为什么核心：动机；确认"前缀"是精确 token 序列匹配而非语义相似。
- 依赖内容：kv-cache 页（K/V 与查询无关）、paged-attention 页（页粒度共享）。

### Q2：radix tree 怎么组织与匹配前缀缓存？

- 完成答案：radix tree（基数树）是压缩前缀树：边上标 token 序列段（而非单 token），从根到任一节点的路径拼接 = 一个已缓存序列；新请求沿树逐 token（实际按段）匹配：匹配到最深节点即命中前缀，其下各页已就绪可复用；未命中部分继续 prefill 并把新段插入为该节点的新子边。SGLang 实现中页大小为 1 token、KV cache 以非连续分页布局存储（paged-attention 页机制）。相比朴素 trie（每边一个 token），压缩边降低树深与查找开销。
- 为什么核心：机制主体，Strata 页 HiRadixTree 的直接前置。
- 依赖内容：Q1、paged-attention 页。

### Q3：缓存满了怎么办——LRU 逐出与引用计数？

- 完成答案：显存有限，radix tree 按最近最少使用逐出：先逐出最久未用的叶子，叶子被逐出后其祖先（可能是公共前缀）仍可被其他分支复用，直到祖先自身成为叶子再被逐出——共享前缀因被多分支引用而天然存活更久（SGLang §3.2）。运行中的请求不能被逐出：每节点维护引用计数，计数为 0 才可逐出。缓存与运行请求共用同一内存池，系统可在"多缓存"与"更大批"之间动态权衡。
- 为什么核心：命中率的另一半决定因素——留什么在缓存里。
- 依赖内容：Q2、kv-cache 页（显存是瓶颈）。

### Q4：命中率由什么决定，页大小为什么影响它？

- 完成答案：命中率 = 命中 token / 总 token，由三点决定：负载的共享结构（哪些请求共享多长前缀）、缓存容量（逐出压力）、匹配粒度（页大小）。页粒度匹配：只有整页命中才算——若两请求共享 100 token 但页大小 256，命中 0 页；页 32 则命中 3 页（96 token）。因此页越大、非整页的共享尾部越浪费，命中率越低（Strata 论文实验：页 512 比页 32 低 2.4% 命中率且吞吐只剩 93%）。SGLang 取页 1 使匹配粒度最细。
- 为什么核心：页大小是 prefix caching 与 paged-attention 两页共同的关键自变量；Strata 页"页大小负担"的直接前置。
- 依赖内容：Q2、paged-attention 页（页大小语义）。

### Q5：除了 radix tree，还有什么实现——哈希前缀页？

- 完成答案：vLLM 与 Mooncake 用哈希：每页由 token ID 与前一页的哈希链式计算唯一指纹，指纹相同即同前缀页，用哈希表定位（无需树结构）。两种实现等价于"结构化索引 vs 哈希索引"的选择：radix tree 支持最长前缀匹配与逐出时祖先复用的结构推理，哈希实现简单、可分布式共享指纹。Strata 论文 §6 并列两者并说明自身基于 SGLang 扩展 RadixTree 为 HiRadixTree。
- 为什么核心：读者对照 vLLM 系文献时的消歧点；HiRadixTree 的谱系。
- 依赖内容：Q2。

## 4. 内容分级

核心内容：共享前缀来源与精确匹配语义（Q1）；radix tree 结构与匹配/插入（Q2）；LRU 叶子优先逐出与引用计数（Q3）；页粒度匹配与页大小-命中率关系（Q4）；哈希实现与谱系（Q5）。

辅助内容：SGLang 的 cache-aware 调度一句话（优先调共享前缀的请求，细节归 Strata 页消融的 LPM 讨论）；生成结果也入缓存（多轮场景下上轮输出 = 本轮输入前缀）；context/prompt caching 叫法说明。

扩展内容：排除——CacheGen/CacheBlend 的语义级复用；Mooncake 全局池；LMCache 分层实现；LLM serving API 的 prompt caching 计费（商业层）。

## 5. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页 |
|---|---|---|
| KV cache、两阶段、K/V 与查询无关 | Q1（复用可行性）、Q3（显存压力） | `wiki/kv-cache/`（递归生成，先完成） |
| 分页、页大小、块级共享 | Q2（页粒度匹配）、Q4（页大小） | `wiki/paged-attention/`（递归生成，先完成） |

## 6. 明确不展开的内容

- 分层缓存（CPU/SSD 层）与传输（Strata 页主题；本页结尾过渡一段）。
- delay hit 的机制与缓解（Strata 页主题；结尾一句引出：缓存"正在生成中"时到达的请求会引发重复计算）。
- 近似/语义缓存（CacheGen/CacheBlend：编码压缩与跨文档融合，另一方向）。
- Mooncake/MemServe 全局调度与分布式池。
- LRU 之外的逐出策略研究（LFU/ARC 等缓存文献）。

## 7. 常见误解和适用边界

误解：

1. "前缀缓存是语义相似就复用"——不是：精确 token 序列前缀匹配；哪怕语义相同，token 不同就不能复用（近似缓存是另一族工作）。影响 Q1。
2. "命中即可用，无任何等待"——命中页必须已在显存；在 CPU/SSD 层的命中还要搬运（Strata 页），且"正在计算中"的前缀不可直接复用（delay hit，Strata 页）。影响 Q1、Q3。
3. "缓存越大命中率单调升，与匹配粒度无关"——容量之外页大小决定匹配粒度；共享 100 token、页 256 时容量再大也是 0 命中。影响 Q4。
4. "逐出会丢掉共享前缀"——LRU 从叶子逐出恰恰保护共享祖先：多分支引用的节点只有当所有子分支都被逐出后才轮到它。影响 Q3。

适用边界：机制要求复用部分是前缀（从头开始相同）；对话中轮次追加满足前缀条件、插入/编辑历史则不满足；命中率收益依赖负载共享结构（Strata 页 cache distance 实验：同上下文请求间隔太远则分层缓存收益也受限）；页大小 1 匹配最细但跨层传输效率低（权衡归 Strata 页）。

## 8. 论断分级

- 文献已有结论：RadixAttention 机制、radix tree 压缩边、LRU 叶子优先逐出、引用计数、页 1 token、非连续分页布局、缓存与运行请求共享内存池（SGLang NeurIPS'24 §3.2）；vLLM/Mooncake 哈希实现（Strata 论文 §6 + vLLM 论文 prefix caching 描述）；prompt caching 被 OpenAI/Google 等提供商采用（Strata 论文 §2.3）；页大小影响命中率（Strata 论文 §3.1 实验：页 512 比 32 低 2.4%、按页匹配的机制论述 "cache matching is performed on a per-page basis"）。
- 基于证据的推断：哈希与 radix tree 的优劣对比表述（"结构化索引 vs 哈希索引"）——由两者实现机制推出的分析性对比，标注推断；"生成结果也构成下轮输入的前缀"由自回归对话结构直接得出。
