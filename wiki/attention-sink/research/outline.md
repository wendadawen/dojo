# Attention Sink

- 描述：Attention sink 是自回归 Transformer 的注意力现象：序列开头少数几个 token 分到很大的注意力权重，与它们的语义无关，模型把它们当作注意力分数的默认去处。
- 摘要：注意力分数必须和为 $1$，没有该关注的东西时，多余的分量被倾倒到序列开头的几个 token 上，这就是 attention sink。窗口注意力在长文本上崩溃，是因为挤掉了它们；在缓存里保住它们的 KV，再配上最近 token 的 KV，就能在远超训练长度的文本上持续生成。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 长页

- 开头｜attention sink 是什么，页面怎么用它；先用几句交代自回归、查询键值、SoftMax、KV cache，并说明微调；无需微调、直接用于已训练模型｜400｜paper/src-v4/text/0_abstract.tex L5-L7；paper/src-v4/text/1_introduction.tex L13；paper/src-v4/text/1_introduction.tex L21-L25；paper/src-v4/text/3_method.tex L11-L14；paper/src-v4/text/3_method.tex L34-L36
- ### 窗口注意力的崩溃点｜只有标题，内容在下一小节｜0｜paper/src-v4/text/1_introduction.tex L17
- #### 滑窗省显存却崩在 cache 上｜dense 的缓存与耗时随长度涨、滑窗把开销固定住，超 cache 困惑度急剧上升；先说明预训练与困惑度；实测条件折叠｜250｜paper/src-v4/text/1_introduction.tex L2；paper/src-v4/text/1_introduction.tex L13-L20；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17；paper/src-v4/text/3_method.tex L8-L9
- ### 初始 token 上到底发生了什么｜只有标题，内容在两个小节｜0｜paper/src-v4/text/3_method.tex L20-L28
- #### 证据：除最底两层外各层各头都集中关注开头｜先说明层与头；注意力图、换行符实验、长序列时首个 token 拿走过半；统计口径折叠｜300｜paper/src-v4/text/3_method.tex L11-L14；paper/src-v4/text/3_method.tex L20-L22；paper/src-v4/text/6_appendix.tex L41-L45；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17；paper/src-v4/figtab/attention_visualization.tex L5
- #### 为什么是初始 token｜SoftMax 要求权重和为 1，多余分量得有去处，sink 就是那个去处，开头 token 更容易被训练成它（资料的解释）；SoftMax 公式折叠｜400｜paper/src-v4/text/1_introduction.tex L22；paper/src-v4/text/3_method.tex L11-L16；paper/src-v4/text/3_method.tex L23-L28
- ### 保住 sink 的滚动 KV cache｜cache 分 sink 与滚动窗口；先说明位置编码，位置取缓存内序号，RoPE、ALiBi 各一句；算法与自设位置编号例子折叠｜350｜paper/src-v4/text/3_method.tex L32-L39；paper/src-v4/text/2_related.tex L6
- ### 四个就够｜初始 token 个数的消融，1 个或 2 个恢复不全，4 个之后收益很小，为什么是 4 个；cache 变大困惑度不一定变低；两张表折叠｜250｜paper/src-v4/text/3_method.tex L29；paper/src-v4/text/4_experiment.tex L37-L39；paper/src-v4/figtab/start_size_and_linebreak.tex L26-L42；paper/src-v4/figtab/cache_size.tex L5-L18
- ### 换模型、换长度都成立｜四个模型族，20K 文本上困惑度不崩，400 万 token 仍稳定；实验条件折叠｜200｜paper/src-v4/text/4_experiment.tex L3-L8；paper/src-v4/figtab/nll_curve.tex L5-L11；paper/src-v4/figtab/long_lm.tex L5-L6
- ### 适用条件与边界｜一条前提、一个代价、一条误解；长文档任务做不了，答案超出 cache 准确率归零｜300｜paper/src-v4/text/6_appendix.tex L2-L13；paper/src-v4/text/6_appendix.tex L26-L31；paper/src-v4/text/3_method.tex L34-L38；paper/src-v4/figtab/kv_lines.tex L3-L21；paper/src-v4/text/1_introduction.tex L2；paper/src-v4/text/1_introduction.tex L31

## 短页

- ### 开头的几个 token 拿走了注意力｜是什么：序列开头少数 token 的注意力权重很大，与语义无关；顺带说明查询键值、SoftMax、KV cache｜300｜paper/src-v4/text/1_introduction.tex L13；paper/src-v4/text/1_introduction.tex L21-L23；paper/src-v4/text/3_method.tex L11-L22
- ### 分数必须和为 1｜为什么需要：权重和为 1，多余分量得有去处，开头 token 更容易被选为去处；先说明窗口注意力与困惑度，挤掉它困惑度急剧上升｜300｜paper/src-v4/text/1_introduction.tex L17；paper/src-v4/text/1_introduction.tex L21-L22；paper/src-v4/text/3_method.tex L8-L9；paper/src-v4/text/3_method.tex L12；paper/src-v4/text/3_method.tex L23-L27；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L17
- ### 缓存的头几个位置留给 sink｜怎么做：cache 开头留几个初始 token，其余给最近 token；先说明位置编码，位置按 cache 内序号编号｜300｜paper/src-v4/text/3_method.tex L32-L36；paper/src-v4/figtab/start_size_and_linebreak.tex L26-L42
