# 大纲审查 · 第 1 轮

## 读者

- 顺序错乱｜长页「prefill 和 decode：一次请求的两个阶段」｜用了「建 KV cache」这个词｜读到这里我不知道 KV cache 是什么；此前没有节点讲过它｜作为前置概念链接成页，或在节点内先讲清。
- 顺序错乱｜长页「存得多、算得少」｜「V3 的 256 选 8 与 671B/37B」里的「V3」｜读到这里我不知道 V3 指什么；此前没有节点提过｜先用一句话交代 V3 是哪个模型，或换成一般化的例子。
- 顺序错乱｜长页「部署：两阶段合设还是分离」｜「按部署单元扩容」里的「部署单元」｜此前没有节点讲过它｜先在本节用一句话说清部署单元指什么。
- 顺序错乱｜短页「一排专家，每个 token 只走几个」｜「FFN」与「token」｜短页内没有哪一节讲过它们｜在本节先一句话说明 FFN 是每层的前馈网络、token 是什么。
- 顺序错乱｜短页「切到多卡、分成两段跑」｜「all-to-all、prefill、decode、TTFT、TPOT」｜短页内没有一处讲清它们｜在本节各自先用一句话讲清。
- 短页标题｜短页「切到多卡、分成两段跑」｜这一节同时讲了两件事，标题也是两件事｜一节不只讲一件事｜拆成两节。
- 遗漏关键内容、多余：无。

## 对照资料

- 资料里没有（位置错）｜长页「适用条件与边界」｜「top-k 中 k 远小于专家数才省计算」在所引三处内均无对应，sources 别处才有｜正确位置是 paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer（L114「up to thousands of experts, but only need to evaluate a handful of them for every example」）｜改依据位置。
- 遗漏关键内容｜全页｜没有任何节点承担「专家负载均衡」（负载不均让单卡成瓶颈、需辅助损失/无辅助损失偏置、冗余专家与负载均衡器）｜blog/deepseek-v3-r1-inference-system-overview.md Achieving Optimal Load Balancing；paper/src-dsv3/main.tex Auxiliary-Loss-Free Load Balancing；paper/src-shazeer/main.tex Balancing Expert Utilization｜新增一节点承担。
- 抬高次要点：无。

## 篇幅与冗余

- 短页不合格｜短页「切到多卡、分成两段跑」｜一节里塞了两件事：①专家并行；②请求分 prefill/decode 两阶段、用 TTFT/TPOT 度量｜改法：本节只留一件，删去两阶段，标题去掉「分成两段跑」。
- 重复、无关、预算不合理：无。

## 处理

- 读者第 1 条（KV cache）：成立。长页「prefill 和 decode」节点改为链接 KV cache 页（wiki/kv-cache 已存在）。
- 读者第 2 条（V3）：成立。开头改用全名 DeepSeek-V3；「存得多、算得少」节点写全名。
- 读者第 3 条（部署单元）：成立。部署节点补「部署单元（最小整副本组合）」的括注。
- 读者第 4 条（短页 FFN/token）：成立。短页第 1 节「写什么」补「先交代 token 与 FFN」。
- 读者第 5 条（短页 all-to-all/prefill/decode/TTFT/TPOT）：成立。短页第 3 节把 all-to-all、prefill、decode、TTFT、TPOT 各用一句讲清（放进「写什么」要点）。
- 读者第 6 条 + 篇幅第 1 条（短页第 3 节两件事，一方要拆两节、一方要删一件）：两方意见相反，交仲裁。仲裁判两条都「不必改」：拆成四节违反 guides/concept.md「短页」的「固定三节」；concept.md 无「一节只许讲一件事」的条文，且本概念的「怎么做」本身同时含专家并行与两阶段（paper/src-dsv3/main.tex Inference and Deployment 同节既讲 EP32/EP320 又讲 prefilling/decoding）。作者采纳仲裁，短页维持三节、第 3 节内容不变（补上基线外词的通用释义，不属「新增一件事」）。
- 对照资料第 1 条（k 远小于 n 的依据位置）：成立。依据改为 paper/src-shazeer/main.tex The Structure of the Mixture-of-Experts layer。
- 对照资料第 2 条（遗漏负载均衡）：成立。新增长页节点「负载不均：最慢的卡决定整步」，预算 260，从各节点腾出。
- 预算：新增节点后长页合计 5000 以内（改后预算见大纲）。

第 1 轮结束：读者 5 条顺序/短页、对照资料 2 条、篇幅 1 条（与读者第 6 条相反，仲裁判不必改）均已处理。进入第 2 轮重审。
