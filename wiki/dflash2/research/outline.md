# DFlash 2

- 描述：DFlash 2 是在块扩散草稿器 DFlash 上增加路径选择器与两抽头动态卷积的第二代草稿器，一次前向并行猜出一整块 token 并从中串出一条连贯路径，解码无损、输出与目标模型一致。
- 摘要：块内每个位置保留 top-16 候选，轻量选择器给相邻候选对打分、一次并行算完，再沿分数贪心走出一条路径；骨干里插入两抽头动态卷积，让每个位置读到前一位的表示，压住块末端的准确率衰减。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 长页

- 开头｜讲清投机解码的起草与验证、块扩散草稿器为何叫块扩散（一次前向并行填整块），解释草稿器、目标模型、token、接受长度，并链接投机解码页；两个组件与发布形态｜350｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L34-L52；model-card/qwen3.8-27b-dflash2.md L23-L36；model-card/qwen3.8-27b-dflash2.md L81-L92
- ### 两个剩余问题｜说明块扩散草稿器逐位置独立预测留下的两处余量，各自对应一个组件｜150｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L49；blog/inco-ai-dflash2-keep-drafting-parallel.txt L145-L152
- #### 选择空间：答案已经在候选里｜top-1 正确率与 top-16 正确率的差距、完美选择器（oracle，总能从 top-16 里挑对）能到多远；逐位置 Recall 全表折叠｜220｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L145-L161
- #### 后缀衰减：连 oracle 也在掉｜准确率沿块下降、连完美选择也掉，候选池本身在后段枯竭｜230｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L152-L156；blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L236
- ### 路径选择器｜说明路径选择器做什么：把逐位置独立挑换成在候选之间串一条连贯路径；打分与代价在下两节｜100｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L194-L197
- #### 相邻对打分｜打分为何只算相邻对（连贯性主要取决于前一位）、公式的两项、256 维嵌入与上下文门、低秩双线性；手算走出一条路径的过程折叠｜450｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L196-L207
- #### 为什么便宜｜打分全并行、串行只在走路径、拒绝采样一句讲清保分布；与串行修正头一句对比；选择器单项接受长度对照折叠｜300｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L209-L227
- ### 两抽头动态卷积｜说明后缀衰减出在草稿模型的骨干（即串起各层的 transformer 主体）上，用短卷积（链接 depthwise-conv 页）专门补局部位｜120｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L307；blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L237
- #### 诊断：注意力的两份工作失衡｜深度帮后段但对早期也无差别加容量、块内注意力份额沿层下降（注意力链接成页）；层数对照与注意力份额数据折叠｜350｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L239-L262
- #### 卷积怎么做｜两抽头公式、动态系数（学习基核加隐状态修正，每 16 个特征通道共享一个修正）、首位读上一个已验证 token、块内无状态；逐位置 Recall 对照表折叠｜450｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L355；blog/inco-ai-dflash2-keep-drafting-parallel.txt L248-L255
- ### 组合效果｜两个组件合用的接受长度提升与总开销；另两个目标模型上的方向与逐基准、逐位置数据表折叠｜300｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L357-L390；blog/inco-ai-dflash2-keep-drafting-parallel.txt L393-L432
- ### 适用条件与边界｜收益随并发上升收窄、因任务而异、需为每个目标配套草稿器、引擎集成尚在分支（vLLM、llama.cpp 走 PR 分支）；并发×任务表折叠｜350｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L52；blog/inco-ai-dflash2-keep-drafting-parallel.txt L54-L99；model-card/qwen3.8-27b-dflash2.md L67-L127

## 短页

- ### 它是第二代的块扩散草稿器｜讲清 DFlash 2 是什么：草稿器逐 token（词元）猜、目标模型验证，块扩散草稿器一次前向猜整块，DFlash 2 加了选路径与卷积两个组件，解码无损｜220｜model-card/qwen3.8-27b-dflash2.md L23-L36；blog/inco-ai-dflash2-keep-drafting-parallel.txt L43-L49
- ### 逐位置独立挑不够｜讲清为什么需要：独立挑只取每个位置最可能的 token，块内不连贯、块末端准确率又在掉｜260｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L145-L161；blog/inco-ai-dflash2-keep-drafting-parallel.txt L230-L236
- ### 选路径补卷积，并行照旧｜讲清怎么做：保留 top-16 候选、给相邻对并行打分再走一条路径，并给草稿模型的层插入两抽头卷积｜360｜blog/inco-ai-dflash2-keep-drafting-parallel.txt L194-L227；blog/inco-ai-dflash2-keep-drafting-parallel.txt L303-L355
