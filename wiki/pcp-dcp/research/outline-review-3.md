# 大纲审查 · 第 3 轮

## 读者

- 无。

## 对照资料

- 无（逐节点、描述、摘要均找到原文支撑；抬高次要点无：核心机制节点预算 540/560 最高，背景节 460/440）。

## 篇幅与冗余

- 重复｜长页「每张卡只存一段 token 的 KV」与「两个机制为什么要分开」｜DCP 沿 token 位置切开、每卡各持一片的机制两处都写。
- 重复｜长页「dcp 的取值与约束」与「两个机制为什么要分开」｜无 PCP 时 DCP 落在 tp 组这些卡上的事实两处都写。

## 处理

- 篇幅「重复」第一条：成立但只改一处。保留「每张卡只存一段 token 的 KV」的机制描述；「两个机制为什么要分开」删去「真正切开 KV」的重述，只留卡数层面的对比（是否扩展启动的卡数）。
- 篇幅「重复」第二条：成立但只改一处。保留「dcp 的取值与约束」的约束表述（无 PCP 时 tp 整除 dcp）；「两个机制为什么要分开」删去「无 PCP 时复用 tp 的卡」的重述，改为「落在已有的卡上」。
- 节点改后预算 400→320。

## 最后一批改动核对

到轮数上限（3 轮），最后一轮之后所做的改动交仲裁核对。改动为「两个机制为什么要分开」一节：原「PCP 只在预填充生效、扩展启动卡数、KV 仍各卡完整；DCP 不扩卡数、无 PCP 时复用 tp 的卡、真正切开 KV｜400」改为「PCP 扩展启动的卡数，DCP 不扩卡数、落在已有的卡上；一条请求预填充用 PCP、解码用 DCP，各管一段｜320」，依据不变（repo/vllm/vllm/config/parallel.py L143-L146、L380-L384、L940-L944；blog/vllm-dcp-blog.md 6. Future Work）。

- 独立仲裁核对结论：不必改（通过）。仲裁逐句核：三句的位置与推论均有据（repo/vllm/vllm/config/parallel.py L143-L146、L380-L384、L940-L944；blog/vllm-context-parallel-deployment.md），未引入新的重复或遗漏，未违背第 1、2 轮裁定。
- 依据：repo/vllm/vllm/config/parallel.py L143-L146「PCP expands the process world size but does not increase the KV-cache shard count」支撑「PCP 扩展启动的卡数」；L380-L384「DCP does not expand the process world size. Without PCP, DCP reuses TP ranks」支撑「DCP 不扩卡数、落在已有的卡上」；两阶段各用一机制见 blog 6. Future Work 与 guides/concept.md 两阶段划分。
