# Attention Sink

- 描述：Attention sink 是自回归 Transformer 的注意力现象：序列开头少数几个 token 拿走了远高于其语义分量的注意力，模型把它们当作注意力分数的默认去处。
- 摘要：注意力分数必须和为 $1$，没有该关注的东西时，多余的分量被倾倒到序列开头的几个 token 上，这就是 attention sink。窗口注意力在长文本上崩溃，是因为挤掉了它们；保住它们的 KV，就能在远超训练长度的文本上流式生成。
- 主题：注意力机制, 内存与缓存
- 标签：注意力

## 长页

- 开头｜attention sink 是什么，为什么需要这个概念，页面怎么用它；一两句交代自回归生成与 KV cache，方法无需微调、直接用于已训练模型｜400｜paper/src-v4/text/0_abstract.tex L1-L11；paper/src-v4/text/1_introduction.tex L21-L25
- ### 窗口注意力的崩溃点｜只有标题，内容在下一小节｜0｜paper/src-v4/text/1_introduction.tex L17
- #### 滑窗省显存却崩在 cache 上｜dense 的缓存与耗时随长度涨、滑窗把开销固定住，超 cache 困惑度爆，另有一条带重算的滑窗；实测数字折叠｜250｜paper/src-v4/text/1_introduction.tex L13-L20；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L10；paper/src-v4/text/3_method.tex L8-L9
- ### 初始 token 上到底发生了什么｜只有标题，内容在两个小节｜0｜paper/src-v4/text/3_method.tex L20-L28
- #### 证据：除最底两层外各层各头都集中关注开头｜注意力图、换行符实验、长序列时首个 token 拿走过半、统计方式；换行符表与统计口径折叠｜500｜paper/src-v4/text/3_method.tex L11-L14；paper/src-v4/text/3_method.tex L20-L22；paper/src-v4/text/6_appendix.tex L41-L45；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L10
- #### 为什么是初始 token｜SoftMax 要求权重和为 1，多余分量得有去处，sink 就是那个去处，开头 token 最易被训练成它；SoftMax 公式与推导折叠｜250｜paper/src-v4/text/3_method.tex L23-L28
- ### 保住 sink 的滚动 KV cache｜无需微调，cache 分 sink 与滚动窗口，位置按 cache 内重排；位置取缓存内序号，RoPE、ALiBi 各一句；算法与自设位置编号例子折叠｜350｜paper/src-v4/text/3_method.tex L32-L39；paper/src-v4/text/3_method.tex L34-L35；paper/src-v4/text/0_abstract.tex L7；paper/src-v4/text/2_related.tex L6
- ### 四个就够｜初始 token 个数的消融，1 个或 2 个恢复不全，4 个之后收益很小；cache 变大困惑度不一定变低；两张表折叠｜250｜paper/src-v4/text/4_experiment.tex L35-L40；paper/src-v4/figtab/start_size_and_linebreak.tex L26-L30；paper/src-v4/figtab/cache_size.tex L5-L11
- ### 训练时留一个 sink｜指定一个可学 sink token 就不必用多个初始 token，160M 三组对照；Zero Sink 的做法与数字折叠｜300｜paper/src-v4/text/3_method.tex L40-L50；paper/src-v4/figtab/pretrain_fix.tex L6-L7；paper/src-v4/text/3_method.tex L29-L31
- ### 换模型、换长度、换任务都成立｜四个模型族，20K 文本困惑度只有它贴着带重算的滑窗，400 万 token，问答对齐逐条喂的基线；实验条件折叠｜350｜paper/src-v4/text/4_experiment.tex L3-L10；paper/src-v4/figtab/nll_curve.tex L5-L11；paper/src-v4/figtab/long_lm.tex L5-L6；paper/src-v4/text/4_experiment.tex L27-L33
- ### 每 token 最多快 22.2 倍｜开销随 cache 线性涨、带重算的滑窗二次涨，差距随 cache 放大，显存相当；效率实验条件与曲线折叠｜250｜paper/src-v4/text/4_experiment.tex L42-L46；paper/src-v4/figtab/efficiency.tex L4-L5
- ### 适用条件与边界｜一条前提、一个代价、一条误解；长文档任务做不了，答案超出 cache 准确率归零｜300｜paper/src-v4/text/6_appendix.tex L2-L13；paper/src-v4/text/6_appendix.tex L26-L31；paper/src-v4/text/3_method.tex L34-L38

## 短页

- ### 开头的几个 token 拿走了注意力｜是什么：序列开头少数 token 的注意力分数远高于语义分量；顺带说明 KV cache 与注意力分数｜300｜paper/src-v4/text/0_abstract.tex L1-L11；paper/src-v4/text/3_method.tex L23-L28
- ### 分数必须和为 1｜为什么需要：分数和为 1，多余分量得有去处，开头 token 最易被选为去处；窗口注意力挤掉它就崩｜300｜paper/src-v4/text/3_method.tex L23-L28；paper/src-v4/figtab/start_size_and_linebreak.tex L7-L10
- ### 缓存的头几个位置留给 sink｜怎么做：cache 开头留几个初始 token，其余给最近 token，位置按 cache 内重排；预训练前置 sink token｜300｜paper/src-v4/text/3_method.tex L32-L39；paper/src-v4/text/3_method.tex L40-L50
