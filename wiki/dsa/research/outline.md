# DeepSeek Sparse Attention

- 描述：DeepSeek Sparse Attention（DSA）是一种稀疏注意力：每个查询位置先用一个轻量的闪电索引器给所有前序位置打一个索引分数，只取分数最高的 $k$ 个，注意力只在这 $k$ 个位置上做。
- 摘要：用一个小索引器给每个前序位置打一个分数，取 top-$k$ 后只对它们做注意力，把核心注意力的代价从 $O(L^2)$ 降到 $O(Lk)$；索引器先靠近稠密注意力训练出来，再让模型适配稀疏模式。长上下文的训练和推理效率都因此提升。
- 主题：注意力机制
- 标签：注意力

## 长页

- 开头｜交代注意力、KV cache、隐藏状态、主干（链接或一句话）；长上下文里注意力越来越贵；DSA 是什么、为什么需要、大概怎么工作｜400｜paper/DeepSeek_V3_2.txt L7-L11；paper/DeepSeek_V3_2.txt L13-L17；paper/DeepSeek_V3_2.txt L282-L289
- ### 打分、选择、做注意力｜回答 DSA 由哪两步组成（索引器打分、按分数取前若干位置），只有标题，内容在两个小节｜0｜paper/DeepSeek_V3_2.txt L16-L17
- #### 闪电索引器：给每个前序位置打分｜索引分数的公式与每个符号；查询、键、权重各从哪个隐藏状态投影来；一句话说明 ReLU；为什么它便宜（头少、可用低精度）；公式折叠｜450｜paper/DeepSeek_V3_2.txt L18-L41
- #### 细粒度选择与稀疏注意力｜按索引分数取 top-k 个前序位置，注意力只在这些位置做（公式）；k 是固定的选取个数、远小于序列长度；一个手算例子折叠｜500｜paper/DeepSeek_V3_2.txt L43-L50；paper/DeepSeek_V3_2.txt L283-L285
- ### 接在 MLA 上｜DSA 接在 MLA 上、用它的 MQA 模式，让每个位置的 KV 条目被该位置所有查询头共享，一次选出的 top-k 对全部头都适用（链接 MLA 页）｜200｜paper/DeepSeek_V3_2.txt L124-L130
- ### 索引器怎么训出来｜两阶段：稠密热身（冻结主干、用 KL 逼近主干注意力分布）与稀疏训练（只在选中位置上对齐、单独优化，每查询取 2048 个 KV）；学习率与 token 数折叠｜550｜paper/DeepSeek_V3_2.txt L133-L137；paper/DeepSeek_V3_2.txt L138-L157；paper/DeepSeek_V3_2.txt L158-L175
- ### 核心计算量的变化｜核心注意力从 $O(L^2)$ 降到 $O(Lk)$；索引器仍是 $O(L^2)$ 但计算量小得多；训练和推理在长上下文都变快；成本曲线折叠｜400｜paper/DeepSeek_V3_2.txt L7-L11；paper/DeepSeek_V3_2.txt L18-L41；paper/DeepSeek_V3_2.txt L282-L289
- ### 适用条件与边界｜DSA 由继续训练引入（非从零）；代价：端到端收益依赖配套优化，短序列预填充改用 masked MHA 模拟（两词本节解释）；误解：索引器与主干一起训练｜400｜paper/DeepSeek_V3_2.txt L8-L9；paper/DeepSeek_V3_2.txt L447-L448；paper/DeepSeek_V3_2.txt L286-L287；paper/DeepSeek_V3_2.txt L170-L172

## 短页

- ### 只算一小撮位置｜DSA 是什么：一种稀疏注意力（注意力是什么一句话带过、链接），把每个位置的注意力限制在少数前序位置上，不在全部前序位置上做｜200｜paper/DeepSeek_V3_2.txt L13-L17；paper/DeepSeek_V3_2.txt L43-L50
- ### 长上下文里注意力太贵｜为什么需要：注意力要对每个前序位置算一遍，代价随序列长度平方增长，长上下文里成为瓶颈｜250｜paper/DeepSeek_V3_2.txt L282-L289
- ### 打分、取 top-k、只算这些｜怎么做：索引器给每个前序位置一个分数（多头、可用低精度），取分数最高的 $k$ 个 KV（KV 一句话带过、链接），注意力只在这 $k$ 个上做｜300｜paper/DeepSeek_V3_2.txt L18-L41；paper/DeepSeek_V3_2.txt L43-L50
