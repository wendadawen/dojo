# 模型并行

- 描述：模型并行把单个神经网络按权重切分到多块加速器上协同完成一次计算，两条基本路径是按层内矩阵维度切的张量并行（TP）和按层堆叠深度切的流水线并行（PP）。
- 摘要：模型参数大到一块卡放不下时，按层内矩阵维度切（TP）或按层堆叠深度切（PP）。TP 每层通信、要求高带宽互联；PP 只在分区边界传激活、但有气泡。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 长页

- 开头｜模型并行是什么、与数据并行的区别；先交代 MLP、多头注意力、all-reduce 三个基线外词，再点出两条基本切法｜300｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L127-L128
- ### 张量并行：把一层里的矩阵切到多卡｜本章回答一层的 MLP 与注意力两块各怎么切，一层前向与反向各要几次 all-reduce｜100｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### MLP 的列切与行切｜两条切法的取舍在于哪一步能本地做：列切让 GeLU 本地应用，行切让第二段本地部分积；切完在输出处一次 all-reduce｜430｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### 小矩阵走一遍切分｜用一个 $1\times4$ 输入、$4\times4$ 权重切到两卡的小例子，验证切分前后结果相同；算式与对比全过程折叠｜100｜自行构造
- #### 注意力按 head 切｜Q/K/V 列切后每个 head 整落一卡，输出投影行切直接消费｜300｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### 一层两次 all-reduce，反向对称｜一层前向共两次 all-reduce、反向两次；两个算子共轭，一个前向恒等、反向全部归约，另一个反过来｜300｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- ### 流水线并行：把整摞层切开｜按层堆叠切成连续分区，每块卡只负责一段｜100｜paper/gpipe-src-v1/main.tex L132-L183
- #### 按层切、micro-batch 流水｜分区数与 micro-batch 数的含义、mini-batch 与 micro-batch 的关系；梯度在 mini-batch 末累计、只更新一次｜400｜paper/gpipe-src-v1/main.tex L127-L128；paper/gpipe-src-v1/main.tex L141-L143；paper/gpipe-src-v1/main.tex L142；paper/gpipe-src-v1/main.tex L367-L368
- #### 气泡从哪来、占多少｜分区造成各卡空转（气泡）；先把流水线按时间切成 slot、数清总数，得气泡占比 $(p-1)/(m+p-1)$，计数推导折叠｜350｜paper/gpipe-src-v1/main.tex L175-L183
- #### 手算一遍气泡占比｜取 $p=4$、$m=4$ 与 $m=16$ 两组数按 slot 图数出占比；数字与算式折叠｜100｜自行构造
- #### micro-batch 要多到多少｜$m\ge 4p$ 时气泡可忽略，$m=1$ 就没有流水｜200｜paper/gpipe-src-v1/main.tex L175-L183；paper/gpipe-src-v1/main.tex L242
- ### 切完每块卡的激活显存怎么变｜分区后每块卡只存边界激活，反向时重算各自的层，峰值激活显存降到与层数、micro-batch 数有关的一档；实验数字折叠｜300｜paper/gpipe-src-v1/main.tex L175-L183；paper/gpipe-src-v1/main.tex L212-L217
- ### TP 与 PP 各适合什么互联｜TP 每层都 all-reduce、只能待在高速域内，PP 只在分区边界传激活、可跨机｜350｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L175-L183；paper/gpipe-src-v1/main.tex L358
- ### 两者组合起来用｜TP 和 PP 正交互补，可以乘起来叠用；超过单机显存的超大模型要叠 intra-layer、inter-layer 与跨节点三种｜200｜paper/megatron-src-v1/modelpar.tex L70；paper/megatron-src-v1/discussion.tex L5
- ### 适用条件与边界｜PP 要求一层放得下一卡、各分区算量均衡，气泡随分区数上升；另两条边界：模型并行不止张量并行一路，分区也不是越多越好｜300｜paper/gpipe-src-v1/main.tex L363；paper/gpipe-src-v1/main.tex L175-L183；paper/megatron-src-v1/background.tex L30；paper/gpipe-src-v1/main.tex L242

## 短页

- ### 把权重分到多块卡上｜是什么：模型并行把权重分到多卡；一句话给 TP、PP 定义；和数据并行的区别｜220｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L127-L128
- ### 一块卡装不下只好切开算｜为什么需要：模型参数和优化器状态超过单卡显存，把权重分到多卡是一条路｜180｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction
- ### 按维度切或按层切｜怎么做：TP 按注意力头和 MLP 矩阵切、切完 all-reduce，PP 按层切、用 micro-batch 流水，两条可组合｜250｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/gpipe-src-v1/main.tex L127-L128；paper/megatron-src-v1/discussion.tex L5
