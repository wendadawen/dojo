# 大纲审查第 1 轮

## 读者

- 顺序错乱｜长页·开头｜「长上下文请求进 PP 流水线时」直接用了「PP」，本节只说「交代 stage、micro-batch、气泡」，没交代 PP｜此前无节点讲过 PP，读到这里缺「把模型按层切到多台设备、逐段传递」这段背景｜作为前置概念链接成页
- 顺序错乱｜长页·把一条请求切进流水线（token 维度）｜「自回归性质」当依据直接用｜此前节点都没讲自回归，读到这里不懂为什么「均分为何前轻后重」｜作为前置概念链接成页
- 顺序错乱｜长页·切片怎么求、推理里怎么落地｜「推理 prefill」直接使用｜此前节点都没讲 prefill，读到这里不懂 Mooncake 那套机制作用在哪个阶段｜作为前置概念链接成页
- 顺序错乱｜长页·固定 chunk 为什么错配、动态 chunk 怎么对齐｜「增量注意力使同尺寸 chunk 时间随前缀增长」当依据直接用｜此前节点都没讲注意力｜作为前置概念链接成页
- 顺序错乱｜长页·把并发请求配平（batch 维度）｜「decode 均摊」「KV 空闲率」直接使用｜此前节点都没讲 decode 与 KV cache｜作为前置概念链接成页
- 顺序错乱｜短页·为什么要它｜「注意力随前缀变重」当依据｜短页第 1 节只交代 PP、stage、气泡，没先讲注意力｜短页内讲清或链接成页
- 顺序错乱｜短页·怎么做｜「均摊 decode」「prefill 并入」「KV 空闲率」「chunk」直接使用｜短页前两节都没讲这些词｜短页内讲清或链接成页
- 短页标题｜短页·它是什么｜标题「它是什么」直接用了「是什么」，未点出概念｜规则要求标题不直接用这三个词｜改成描述该节内容的具体标题
- 短页标题｜短页·为什么要它｜标题「为什么要它」是「为什么需要」的公式化问句｜同上｜改成描述该节内容的具体标题
- 短页标题｜短页·怎么做｜标题「怎么做」直接用了「怎么做」｜同上｜改成描述该节内容的具体标题
- 遗漏关键内容｜全篇｜无
- 多余｜全篇｜无

## 对照资料

- 资料里没有（位置错）｜GPU 为什么空转「两类依赖与两个放大器」｜所引 gLLM L106-L113 只有欠利用现象与 batch token 波动，两类依赖/两类失衡在 L81-L85｜『pipeline bubbles, periods of GPU idle time caused by two types of dependencies: (1) inter-stage dependency … (2) inter-batch dependency』2.background_motivation.tex L81-L85｜gLLM 依据补 L81-L85，L106-L113 只留空转现象
- 资料里没有（位置错）｜适用条件与边界「自回归」｜所引 terapipe L300-L305 讲切片粒度与 GPU 欠利用，未设自回归前提｜『the computation of a self-attention layer SelfAtt(h_t) only depends on the hidden states of previous positions』terapipe.tex L283｜自回归条件改引 terapipe L282-L283
- 资料里没有（位置错）｜短页·为什么要它「注意力随前缀变重」｜所引 gLLM L106-L113 与 blog L111-L117 都无此内容｜『As the prefix sequence length grows, the per-chunk processing time increases non-linearly』blog L141｜该放大源改引 blog L139-L141
- 资料里没有（位置错）｜短页·怎么做「按积压与 KV 空闲率调 prefill」｜所引 gLLM L90-L96 是 decode 节，prefill 公式在 L61-L64｜『#P=max(min(#WP/#T, #MaxP×(KV_free−KV_thresh)/(1−KV_thresh)), #MinP)』4.design.tex L61-L64｜prefill 部分补引 4.design.tex L24-L64
- 遗漏关键内容、抬高次要点｜无

## 篇幅与冗余

- 短页不合格｜短页·怎么做｜一节讲了两件事：token 维度切分与 batch 维度配平｜「写什么」并列了「token 维度切前长后短并随前缀缩小 chunk；batch 维度均摊 decode、按积压与 KV 空闲率调 prefill」｜改为一句主线「均衡每步计算量」，两条路线作为它的实例
- 重复｜长页「把一条请求切进流水线（token 维度）」与「固定 chunk 为什么错配、动态 chunk 怎么对齐」｜同一机制各讲一遍：等量 token 的后段更重、注意力随前缀增长｜前者「均分为何前轻后重」，后者「增量注意力使同尺寸 chunk 时间随前缀增长」｜原因只在 token 维度节讲，动态 chunk 节改为一句回指
- 无关｜长页「两条路线怎么配合」｜含训练场景的「训练侧两维联合切分」，属推理概念之外的另一种做法｜「写什么」有「训练侧两维联合切分折叠」，依据是 TeraPipe（训练）｜删去该折叠要点，本节只做题内组合

## 处理

读者：

- 顺序错乱（长页 5 条、短页 2 条）：成立。这些都是基线外的前置概念，各按指南链接到已有页面：PP → [model-parallelism](wiki/model-parallelism)；自回归/因果 → [causal-mask](wiki/causal-mask)；prefill、decode、chunk → [chunked-prefill](wiki/chunked-prefill)；注意力 → [standard-attention](wiki/standard-attention)；KV cache → [kv-cache](wiki/kv-cache)。大纲各节点的「写什么」里写明首用处链接，本页只留一两句结论。
- 短页标题 3 条：成立。短页三节标题改为描述内容的具体标题：「把各 stage 每步执行时间拉齐」「节拍由最慢的 stage 决定」「按计算量切分，不是按条数」。
- 遗漏、多余：无。

对照资料：

- 位置错 4 条：全部成立，按改法改依据。GPU 为什么空转补 gLLM L81-L85；边界节自回归条件改 terapipe L282-L283；短页「为什么」的放大器二改 blog L139-L141；短页「怎么做」prefill 部分改 gLLM 4.design.tex L24-L64。

篇幅与冗余：

- 短页不合格 1 条：成立。短页「怎么做」改为一句主线——均衡的对象是每步计算量（不是 token 条数），下面两条路线是它的实例，并收在一节内。
- 重复 1 条：成立。把「注意力随前缀增长」这个原因只放在 token 维度一节，动态 chunk 一节改为一句回指后再讲对齐机制。
- 无关 1 条：成立。删去训练侧两维联合切分，本节只讲推理里两条路线的组合。

无两位审查者意见相反之处，本轮不需仲裁。
