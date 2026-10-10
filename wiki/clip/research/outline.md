# CLIP

- 描述：CLIP 用一批（图像，文本）对同时训练图像编码器和文本编码器，让配对的图文向量在同一个空间里靠近，从而把图像分类任务写成一段文本，不训练就能对新的类别做零样本分类。
- 摘要：把「一批图像各配哪段文本」当作预训练任务，两个编码器拼出一个多模态向量空间；测试时把类别名写成提示文本，用文本编码器现生成一个分类器，不用该数据集的任何训练样本。
- 主题：多模态
- 标签：视觉与多模态

## 长页

- 开头｜交代嵌入、编码器、相似度、对比、预训练五个基线外词；CLIP 是什么（双编码器做图文对比预训练）；为什么需要；测试时大致怎么工作｜340｜paper/clip_paper.tex Introduction and Motivating Work；paper/clip_paper.tex Natural Language Supervision；paper/clip_paper.tex Using CLIP for Zero-Shot Transfer
- ### 训练数据：从网络收集四亿图文对｜已有图文数据集为什么不够用，怎么用五十万查询词搜集并做类别均衡得到四亿对，词数与公开网页语料 WebText 相当｜240｜paper/clip_paper.tex Creating a Sufficiently Large Dataset
- ### 预训练目标：为什么用对比｜为什么从预测整段文本换成只判断哪段文本配哪张图，只有标题，内容在小节｜0｜paper/clip_paper.tex Selecting an Efficient Pre-Training Method
- #### 相似度矩阵与双向交叉熵｜一批 N 对形成 N×N 相似度矩阵，最大化对角、最小化其余，损失是双向交叉熵（链接前置页），温度是缩放相似度分数的可学习标量；三对图文的手算例子与伪代码折叠｜380｜paper/clip_paper.tex Selecting an Efficient Pre-Training Method
- ### 两个编码器与训练规模｜本章列出图像端、文本端和训练配置，只有标题，内容在三个小节｜0｜paper/clip_paper.tex Choosing and Scaling a Model；paper/clip_paper.tex Training
- #### 图像编码器｜ResNet（卷积网络，用一个小窗口在图上滑动逐处加权求和；残差连接见前置页）的三个改法与 ViT（链接前置页）两种选择，以及怎么按宽度、深度、分辨率同时放大｜200｜paper/clip_paper.tex Choosing and Scaling a Model
- #### 文本编码器｜层数、宽度、注意力头数、子词词表规模、最长长度、取哪个位置的输出当文本向量；两端输出各经一个线性投影映到同一维度，得到多模态嵌入空间｜180｜paper/clip_paper.tex Choosing and Scaling a Model
- #### 训练配置与规模｜轮数、批量、温度的初始值与上限、最大模型的天数与卡数；实验条件与数字折叠｜200｜paper/clip_paper.tex Training；paper/clip_paper.tex Model Hyperparameters
- ### 零样本分类：用类别名生成分类器｜本章讲测试时怎么用，只有标题，内容在两个小节｜0｜paper/clip_paper.tex Using CLIP for Zero-Shot Transfer
- #### 分类器怎么算出来｜图像向量与每个类别的文本向量算余弦相似度、乘温度、用 softmax 归一成概率；等价于一个无偏置线性分类器；算一次可缓存；三类的例子折叠｜340｜paper/clip_paper.tex Using CLIP for Zero-Shot Transfer
- #### 提示工程与模板集成｜为什么裸类名不够、默认模板是什么、按任务的模板怎么写、多模板在向量空间集成；各提升多少折叠｜220｜paper/clip_paper.tex Prompt Engineering and Ensembling
- ### 零样本分类的效果｜本章列结果，只有标题，内容在三个小节｜0｜paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance
- #### 零样本分类的水平｜零样本的准确率接近一个在当时已偏弱的有监督基线，一句结论，数字折叠｜150｜paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance
- #### 偏弱的任务｜卫星图、计数、专门领域等失败任务，以及为什么会这样｜160｜paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Limitations
- #### 随规模变化与鲁棒性｜平均误差随算力的 log-log 线性、分布迁移下的表现为随何变化、与人类的对比；数字折叠｜180｜paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Robustness to Natural Distribution Shift；paper/clip_paper.tex Comparison to Human Performance
- ### 表示质量：线性探针｜编码器输出直接训一个线性分类器就有效，说明预训练学到了通用的、可迁移的表示；条件与数字折叠｜150｜paper/clip_paper.tex Representation Learning；paper/clip_paper.tex Linear-probe evaluation
- ### 适用条件与边界｜什么条件下成立、代价是什么、常见误解；评测集与 CLIP 共同演化；数字折叠｜340｜paper/clip_paper.tex Limitations；paper/clip_paper.tex Robustness to Natural Distribution Shift；repo/CLIP/model-card.md Model Use

## 短页

- ### 两个编码器，一个共同的向量空间｜CLIP 是什么：一个图像编码器加一个文本编码器（各自把输入变成向量），把一批图文对里配对的图文拉近（对比训练），得到共同的向量空间｜300｜paper/clip_paper.tex Natural Language Supervision；paper/clip_paper.tex Using CLIP for Zero-Shot Transfer
- ### 换一套概念就要重新标数据｜为什么需要：有监督模型只认训练时定好的类别，换一套概念要重标数据；原始文本是现成、更宽的监督｜320｜paper/clip_paper.tex Introduction and Motivating Work
- ### 用类别名现生成分类器｜怎么做：把每个类别的名字写成一段提示文本，编码后与图像向量比相似度，取最像的类别｜320｜paper/clip_paper.tex Using CLIP for Zero-Shot Transfer
