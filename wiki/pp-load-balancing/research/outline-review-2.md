# 大纲审查第 2 轮

## 读者

- 顺序错乱｜长页「固定 chunk 为什么错配、动态 chunk 怎么对齐」｜本节才第一次出现「chunk」，本节「写什么」没交代，前面节点只说「切分／切片」｜此前节点都没讲 chunk｜本节首次用到 chunk 时链接前置页
- 顺序错乱｜长页「把并发请求配平（batch 维度）」｜「decode 均摊」的 decode 没讲过｜节点 4 只链接了 prefill，节点 1、2、3、5 都没讲 decode｜把 decode 也作为前置概念链接成页
- 顺序错乱｜长页「切片怎么求、推理里怎么落地」｜「不选跨机 TP/SP 的动机」里的 TP/SP 没讲过｜此前节点都没讲 TP/SP｜TP/SP 作为前置概念链接成页
- 顺序错乱｜短页「按计算量切分，不是按条数」｜用了「batch 维度均摊 decode」和「缩小后续 chunk」，短页前两节都没讲｜短页要自足｜短页内讲清或链接前置页
- 遗漏关键内容、多余、短页标题｜无

## 对照资料

- 资料里没有（位置错）｜适用条件与边界「短上下文收益小」｜四条依据内都没有，原文在别处｜blog L113；mooncake prefill.tex L42｜补引 blog L113 与 mooncake prefill.tex L42
- 资料里没有（依据缺）｜适用条件与边界「调参」｜原文不在所引范围内｜blog L160｜补引 blog L160-L166
- 资料里没有（找不到）｜适用条件与边界「两条常见误解」｜所引四条内无对应，sources 别处也从未概括为「误解」｜范围内最接近 blog L88｜改标 blog L88 与 blog L160，改写为「注意事项/常见误解」两点
- 资料里没有（说过头）｜按计算量切分，不是按条数「不是 token 条数」｜batch 维度 gLLM 恰按 token 数均摊，「不是」把结论推强｜4.design.tex L92（在该节点所引范围内）｜限定到 token 维度，batch 维度改为按 token 数均摊
- 遗漏关键内容、抬高次要点｜无

## 篇幅与冗余

- 无关｜长页「切片怎么求、推理里怎么落地」｜该节含 Mooncake 的 CPP 机制逐点，并点名「不选跨机 TP/SP 的动机」，属另一系统的做法与多方案对比｜「写什么」有「不选跨机 TP/SP 的动机」，依据 paper/src-2407.00079/data/prefill.tex Multi-node Prefill｜删「不选跨机 TP/SP 的动机」及这组对比，Mooncake 只留与 token 切分落地直接相关的机制
- 无关｜长页「两条路线怎么配合」｜「层划分那一维」是 token 切分、batch 配平之外的第三种做法，被当并列一维写入｜「写什么」有「层划分那一维」｜删「层划分那一维」，本节只留两维配合
- 重复、预算不合理、短页不合格｜无

## 处理

读者：

- chunk 链接：成立。长页「切片怎么求」节点写明链接 [chunked-prefill](wiki/chunked-prefill)。
- decode 链接：成立。「把并发请求配平」节点写明链接 decode 与 KV cache 前置页（[chunked-prefill](wiki/chunked-prefill) 的 decode 段与 [kv-cache](wiki/kv-cache)）。
- TP/SP：成立。该对比已随下面「无关」条一并删去，TP/SP 不再出现。
- 短页自足：成立。短页第三节改为「按执行时间对齐，不是按条数」，chunk 与 decode 均链接前置页，本页各留一句。

对照资料：

- 位置错、依据缺、找不到：全部成立，按改法改依据：短上下文收益补 blog L113 与 mooncake prefill.tex L42；调参补 blog L160-L166；「两条常见误解」改标 blog L88 与 blog L160。
- 说过头：成立。删「不是 token 条数」的说法；短页第三节改为「按执行时间对齐」为均衡对象，token 维度否决等长切片、batch 维度按 token 数均摊分别注明。

篇幅与冗余：

- Mooncake 无关：部分成立。Mooncake 的 CPP 是本概念在推理 prefill 的落地，保留其机制（节点分组、chunk 上限、同请求不同 chunk 同时处理、边界通信与计算重叠）；「不选跨机 TP/SP 的动机」属另一维度的方案对比，删去。
- 层划分：成立。删「层划分那一维」，本节只讲 token 与 batch 两维，inter-stage 仅作为 gLLM 的范围边界一句。

本轮无两位审查者意见相反之处，不需仲裁。
