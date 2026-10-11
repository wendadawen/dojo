# Pipeline Parallelism

- 描述：流水线并行把网络按层堆叠切成连续的若干分区、各放在一块加速器上，一个 mini-batch 拆成更小的 micro-batch 依次流过各分区，不同分区同时处理不同的 micro-batch。
- 摘要：把整摞层切成连续分区放到多卡，再把一个 mini-batch 拆成 micro-batch 流水执行。只在分区边界传激活、通信量小，代价是分区造成的气泡。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 长页

- 开头｜说清流水线并行是什么：按层堆叠切成连续分区放多卡、数据依次流过；一句话交代层、加速器、MLP，多头注意力链接成页｜320｜paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L132-L141
- ### 把全部层切成连续分区｜按层的先后切成 K 个连续分区、各放一卡；边界处自动插入通信、只传激活；分区算法让各分区代价尽量均衡｜330｜paper/gpipe-src-v1/main.tex L132-L141；paper/gpipe-src-v1/main.tex L181
- #### 把一个 mini-batch 拆成 micro-batch 流水｜说清 mini-batch 与 micro-batch；不同分区并行处理不同 micro-batch；梯度在 mini-batch 末累计后更新一次｜380｜paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L141-L143
- #### 气泡从哪来、占多少｜分区造成各卡空转（气泡）；把时间切成 slot、数清每个分区空着的 slot 数，得气泡占比 $(K-1)/(M+K-1)$；计数推导折叠｜360｜paper/gpipe-src-v1/main.tex L175-L183
- #### 手算一遍气泡占比｜取 K=3、M=2 与 M=12 两组代入比例算占比；数字与算式折叠｜90｜自行构造
- #### micro-batch 要多到多少｜M≥4K 时气泡可忽略；M=1 时没有流水、任一时刻只有一块卡在算｜240｜paper/gpipe-src-v1/main.tex L175-L183；paper/gpipe-src-v1/main.tex L242
- ### 切完每块卡的激活显存怎么变｜分区只存边界激活；反向（从输出往回算梯度）时各自重算这一段，于是峰值激活显存随每块卡的层数与 micro-batch 大小走；实验数字折叠｜320｜paper/gpipe-src-v1/main.tex L175-L179；paper/gpipe-src-v1/main.tex L212-L218
- ### 适用条件与边界｜每层要放得下一卡，气泡随分区数上升；通信量小，因此可跨没有高速互联的设备；与张量并行对比一句并链过去；两者正交｜320｜paper/gpipe-src-v1/main.tex L363；paper/gpipe-src-v1/main.tex L179；paper/gpipe-src-v1/main.tex L181；paper/megatron-src-v1/intro.tex L7

## 短页

- ### 把整摞层切给不同的卡｜是什么：按层堆叠切成连续分区、各卡持一段、数据依次流过；一句话交代层、加速器；末尾点名张量并行并链接｜340｜paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L132-L141；paper/megatron-src-v1/intro.tex L7
- ### 一块卡装不下整个模型｜为什么需要：模型参数超过单卡显存，把模型按层切开放在多卡上是一条路｜240｜paper/gpipe-src-v1/main.tex L110；paper/gpipe-src-v1/main.tex L127
- ### 拆小批次，缩小气泡｜怎么做：把 mini-batch 拆成 micro-batch 依次流过各分区；梯度在 mini-batch 末累计；micro-batch 越多气泡越小｜330｜paper/gpipe-src-v1/main.tex L128；paper/gpipe-src-v1/main.tex L143；paper/gpipe-src-v1/main.tex L179；paper/gpipe-src-v1/main.tex L242
