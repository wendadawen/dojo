# 大纲审查 · 第 1 轮

## 读者

- 顺序错乱｜长页「预填充算得多，解码存得多」｜「KV」先用后讲｜缺 KV cache 是什么。
- 顺序错乱｜长页「沿头维切到头就开始重复」｜张量并行、头维/头数、tp、GQA、MLA 没先讲。
- 顺序错乱｜长页「每张卡只存一段 token 的 KV」｜「注意力」「FFN」没先讲。
- 顺序错乱｜长页「一步解码的通信与合并」｜「AllGather」没先讲。
- 顺序错乱｜长页「PCP：把预填充的序列计算摊到多张卡」｜「Q/K/V」「ring attention」「因果（掩码）」没先讲。
- 顺序错乱｜长页「与 chunked prefill 的区别」｜「chunked prefill」没先讲。
- 顺序错乱｜短页「把一条长请求切到多张卡上」｜短页内「KV cache」没先讲。
- 顺序错乱｜短页「预填充切计算，解码切缓存」｜短页内「ring attention」「LSE」没先讲。
- 多余｜长页「与 chunked prefill 的区别」｜整节专讲别的方法与 PCP 对比。
- 遗漏关键内容：无。短页标题：无。

## 对照资料

- 资料里没有｜dcp 的取值与约束｜漏条件：tp 整除 dcp 只在 PCP 关闭时成立。
- 资料里没有｜与 chunked prefill 的区别｜位置错：deployment「Prefill Context Parallel」内无 chunked prefill 表述，原文在别处。
- 资料里没有｜适用条件与边界｜位置错：「GQA 多一层整除」不在所标范围内，原文在 blog 5.4 与 model.py。
- 资料里没有｜预填充费算力，解码费显存｜位置错：prefill 一侧依据内无 prefill/TTFT 表述，原文在 deployment「Context Parallel Deployment」。
- 遗漏关键内容、抬高次要点：均无。

## 篇幅与冗余

- 重复｜长页「dcp 的取值与约束」与「适用条件与边界」｜dcp 上界、GQA 整除、通信代价三项两节都写。
- 重复｜长页「开头」与「预填充算得多，解码存得多」｜「按两阶段分成两种形态」两处都写，开头越界讲了后节的机制。
- 预算不合理｜长页「每张卡只存一段 token 的 KV」｜核心机制节 440 小于背景节 520/500/480。
- 无关｜长页「沿头维切到头就开始重复」｜整节把张量并行沿头维切 KV 的机制讲完整。
- 无关｜长页「与 chunked prefill 的区别」｜整节 220 专门对比相近概念。
- 重复｜短页「把一条长请求切到多张卡上」与「预填充切计算，解码切缓存」｜「PCP 切预填充、DCP 切解码」定义两处都写。

## 处理

- 读者八条「先用后讲」：全部成立。这些都是基线外词（KV cache、张量并行、注意力头、GQA/MLA、注意力、FFN、AllGather、因果掩码、chunked prefill、Q/K/V、ring attention、LSE）。按 guides/concept.md「基线外词」处理：一
  两句讲得清的（AllGather、LSE、Q/K/V、ring attention、因果负载）在正文第一次出现处用一句话解释；讲清要一整段的作为前置概念链接：KV cache→wiki/kv-cache，张量并行→wiki/model-parallelism，GQA/MLA→wiki/mqa-gqa、wiki/mla，注意力→wiki/standard-attention，因果掩码→wiki/causal-mask，chunked prefill→wiki/chunked-prefill（均已存在）。大纲不改节点，正文落实。
- 读者「多余」（chunked prefill 整节）：成立。删去该章，「chunked prefill」压成「适用条件与边界」里的一两句对比。
- 对照资料「漏条件」（tp 整除 dcp）：成立。改为「无 PCP 时要求 tp 整除 dcp」。
- 对照资料「位置错」（chunked prefill）：成立。依据改指 repo/vllm/docs/configuration/optimization.md Chunked Prefill；「不加卡数」sources 未明说，改为不加卡数的通用表述。
- 对照资料「位置错」（GQA 整除）：成立。依据补 blog 5.4 GQA Backend 与 repo/vllm/vllm/config/model.py L1558-L1587。
- 对照资料「位置错」（预填充费算力）：成立。prefill 一侧依据改指 blog/vllm-context-parallel-deployment.md Context Parallel Deployment。
- 篇幅「重复」（dcp 取值 vs 边界）：成立。「适用条件与边界」删去 dcpp 上界/GQA 整除/通信代价，只留 LSE 要求、PCP 开发状态、chunked 区别、两条误解。
- 篇幅「重复」（开头 vs 预填充算得多）：成立。「开头」只留概览（上下文并行是什么、两种形态），删去两阶段压力的理由。
- 篇幅「预算不合理」：成立。「每张卡只存一段 token 的 KV」预算 440→540，高于背景节。
- 篇幅「无关」（chunked prefill 整节）：成立，同读者「多余」的处理。
- 篇幅「无关」（沿头维切到头）：部分成立。张量并行沿头维切 KV 的机制压成一句并链接 wiki/model-parallelism；本节点保留，因为「沿头维切到头就重复」是 DCP 动机的关键，不是可省的背景。
- 篇幅「重复」（短页定义）：不成立。短页第 1 节回答「是什么」（两种形态各切什么），第 3 节回答「怎么做」（PCP/DCP 的具体动作），第 3 节讲怎么做时必须重述切的是什么，这不算同一事实第二次出现，不删。
