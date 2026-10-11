# Tensor Parallelism

- 描述：张量并行把一层里的权重矩阵按维度切到多块加速器上，各卡各算一部分，每层前向、反向各两次 all-reduce 把结果合并，合起来算出同一层。
- 摘要：单卡放不下一层时，把层里的权重矩阵按列或按行切到多卡。切法让 GeLU 等逐元素运算不必通信，代价是每层前向、反向各两次 all-reduce。
- 主题：并行与通信, 训练与优化
- 标签：并行与通信

## 长页

- 开头｜说清张量并行是什么；区别数据并行；交代层、模型并行、加速器、显存、前向反向、MLP、all-reduce，多头注意力链接成页｜300｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- ### 把一层的矩阵切到多卡｜一个层由多头注意力和两层 MLP 组成；把两块各自的权重矩阵切开、所有卡一起算同一层｜90｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### MLP 的列切与行切｜两条切法的取舍在哪一步能本地做：列切让 GeLU（逐元素非线性）本地施加、激活前不通信，行切让第二段直接吃本地激活；输出处一次 all-reduce｜440｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### 小矩阵切分示例｜用一个 4 维输入、两段 4×4 权重切两卡的小例子手算，验证切开与不切开结果相同；算式与对比全过程折叠｜90｜自行构造
- #### 注意力按 head 切｜Q/K/V 三处按列切使一个 head 整落一卡、注意力本身不通信；输出投影按行切、与 MLP 共用同一处 all-reduce｜320｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- #### 一层两次 all-reduce，反向对称｜一层前向两次 all-reduce、反向两次；两个算子共轭，一个前向恒等反向归约、另一个反过来，可接自动求导（框架自动算梯度）｜320｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- ### 切分后每块卡的显存｜模型并行区里的权重、激活及其优化器状态按卡切开；区外部分（dropout、层归一化、残差连接）不切分、各卡重复算一份；整个模型能装下比单卡容量更大的规模｜320｜paper/megatron-src-v1/intro.tex Introduction；paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
- ### 适用条件与边界｜每层都通信、通信量大，适合放在高速互联域内（卡间带宽高的一层互联，TP 属 SPMD 一路）；与流水线并行对比一句并链过去；两者正交｜320｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/megatron-src-v1/intro.tex Introduction；paper/megatron-src-v1/abs.tex L3；paper/megatron-src-v1/discussion.tex Conclusion and Future Work；paper/gpipe-src-v1/main.tex L358

## 短页

- ### 把一层里的矩阵切到多卡｜是什么：把一层内权重矩阵按维度切开、各卡合算同一层；一句话交代层、权重矩阵、加速器；与数据并行的区别一句；末尾点名流水线并行｜360｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/modelpar.tex Model Parallel Transformers；paper/megatron-src-v1/intro.tex L7
- ### 一块卡装不下整个模型｜为什么需要：参数与优化器状态超过单卡显存，把权重切到多卡是一条路｜240｜paper/megatron-src-v1/background.tex Data and Model Parallelism in Deep Learning；paper/megatron-src-v1/intro.tex Introduction
- ### 按列或按行切，输出处 all-reduce｜怎么做：MLP 按列或按行切、注意力按 head 切，逐元素算子本地施加，输出处一次 all-reduce 合并｜340｜paper/megatron-src-v1/modelpar.tex Model Parallel Transformers
