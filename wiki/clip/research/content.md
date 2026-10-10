# CLIP

- 描述：CLIP 用一批（图像，文本）对同时训练图像编码器和文本编码器（把图像、文本各变成一个向量），让配对的图文向量在同一个空间里靠近，从而把图像分类任务写成一段文本，不训练就能对新的类别做零样本分类。
- 摘要：把「一批图像各配哪段文本」当作预训练任务，两个编码器拼出一个多模态向量空间；测试时把类别名写成提示文本，用文本编码器现生成一个分类器，不用该数据集的任何训练样本。
- 主题：多模态
- 标签：视觉与多模态

## 资料

- Alec Radford, Jong Wook Kim, Chris Hallacy 等. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020v1, 2021. [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- OpenAI. CLIP: official implementation. 2021. [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

## 短页

### 两个编码器，一个共同的向量空间

CLIP 是一个图像编码器加一个文本编码器，两者各自把输入变成一个向量。训练时给它一批（图像，文本）对，让配对的图与文在同一个向量空间里靠近、不配对的离开，这种训练叫对比训练；训练完得到的是一个把图和文放在一起比较的向量空间。

依据：paper/clip_paper.tex Natural Language Supervision；paper/clip_paper.tex Selecting an Efficient Pre-Training Method

### 换一套概念就要重新标数据

常见的图像模型是在一批带标签的图上训练一个分类器，标签集在训练时就固定下来。要它识别一套新的概念，就得重新标注数据、重新训练。文本是另一种监督（训练时告诉模型正确答案的信号）：图像在网络上往往配着说明文字，这些文字描述的是图上有什么，可以直接当作监督信号用。它比人工标签更容易扩到很大规模，能覆盖的视觉概念也更宽。

依据：paper/clip_paper.tex Introduction and Motivating Work

### 用类别名现生成分类器

测试时，把要分的每个类别的名字写成一段提示文本，用文本编码器编成向量；图像也编成一个向量。二者算余弦相似度（两个向量方向越接近，这个值越大），得分最高的那个类别就是预测。整个分类器由类别名现生成，换一套类别只改写文本、不用重新训练。

依据：paper/clip_paper.tex Using CLIP for Zero-Shot Transfer

## 长页

图像、文本这类原始输入先要变成向量（一列实数）才能被模型处理，做这件事的网络叫编码器。两个向量方向接近时夹角小，余弦相似度就大。CLIP 是一个图像编码器加一个文本编码器：在一批图文对里，配对的图与文在同一个向量空间里被拉近，不配对的被推远，这种训练叫对比预训练。它把图像分类换成一段文本——把类别的名字写成一句话，用文本编码器编成向量，与图像向量最像的那个类别就是答案，不用这个数据集的任何训练样本。之所以不直接训练一个有监督分类器（用带答案的样本训练、只输出这些答案的模型），是因为那样得到的模型只会识别训练时定好的类别，换一套概念就要重新标数据；而图像配的文字在网络上本来就有，是一种现成、更宽的监督。

依据：paper/clip_paper.tex Introduction and Motivating Work；paper/clip_paper.tex Natural Language Supervision；paper/clip_paper.tex Selecting an Efficient Pre-Training Method；paper/clip_paper.tex Using CLIP for Zero-Shot Transfer

### 训练数据：从网络收集四亿图文对

训练一个图像编码器要大量图文对。已有的公开图文数据集偏小：高质量的那批只有十万量级，规模上亿的那批元数据稀疏、质量不齐，按英文描述筛一遍只剩千万量级。

CLIP 自己从公开网络收集了四亿对（图像，文本）：用五十万个查询词去搜文本里含这些词的图文对，每个查询最多取两万对，并把各个查询取到的对的数量拉平到同一水平，得到的数据集叫 WIT。

依据：paper/clip_paper.tex Creating a Sufficiently Large Dataset

### 预训练目标：为什么用对比

::: 从预测整段文本到只判断配对
早期做法让图像编码器与文本编码器联合预测图像对应的说明文字，逐词生成。一个 63M 参数（模型的参数指其中可学习的权重，M 表示百万）的语言模型，算力已达到它配的 ResNet-50 图像编码器的两倍，识别图像类别仍比一个简单得多的预测词袋（忽略词序、只看出现了哪些词）的基线慢 3 倍；把目标换成只判断哪段文本配哪张图，又快 4 倍，省下的是把描述的确切措辞都预测出来那部分工作，而同一批数据里措辞千差万别，这部分很难学。
依据：paper/clip_paper.tex Selecting an Efficient Pre-Training Method
:::

#### 相似度矩阵与双向交叉熵

一批取 $N$ 对（图像，文本）。$N$ 个图像向量与 $N$ 个文本向量两两算相似度，排成一个 $N\times N$ 的矩阵，第 $i$ 行第 $j$ 列是图像 $i$ 与文本 $j$ 的相似度，对角线上的 $N$ 项是真正配对的那几对。训练把对角线上的相似度推高、其余 $N^2-N$ 项压低。

做法是把每一行、每一列各当作一次分类：第 $i$ 行判断文本 $1,\dots,N$ 里哪一段配图像 $i$，正确答案是第 $i$ 段，列方向对称。相似度先乘 $1/\tau$ 缩放，其中 $\tau$ 叫温度，是缩放相似度分数的一个可学习标量，它本身作为一个参数在训练中直接优化，不作为超参数（需要人为设定、不靠训练学出来的量）调。再经 softmax 归一成一组和为 1 的概率（对每个分数取指数，再除以所有指数的和），分数越高概率越大。对每一行、每一列算交叉熵（见[交叉熵](wiki/cross-entropy)），两个方向各算一次再取平均，就是对称交叉熵损失。

依据：paper/clip_paper.tex Selecting an Efficient Pre-Training Method

::: 三对图文的相似度矩阵
取一批 3 对图文，$3\times3$ 相似度矩阵、行是图像、列是文本：
$$
\mathbf{S} = \begin{pmatrix} 0.9 & 0.2 & 0.1 \\ 0.1 & 0.8 & 0.3 \\ 0.2 & 0.3 & 0.7 \end{pmatrix}
$$
温度 $\tau = 0.1$，缩放后 $z_{ij} = s_{ij}/\tau$。
依据：自行构造

1. 第 1 行（图像 1）的分数是 $0.9/0.1=9$、$0.2/0.1=2$、$0.1/0.1=1$。
2. softmax：$e^9=8103$、$e^2=7.39$、$e^1=2.72$，分母 $8113.19$，得 $0.999$、$0.0009$、$0.0003$。
3. 正确类是第 1 列，交叉熵 $-\ln(0.999)\approx 0.001$，这一对几乎不产生损失。
4. 把第 1 行的 $s_{11}$ 改成 $0.1$、$s_{12}$ 改成 $0.9$：分数是 $1$、$9$、$1$，正确类概率降到约 $0.0003$，交叉熵升到约 $8.0$。
5. 三行各算一个交叉熵取平均，得到图像到文本方向的损失；三列各算一个取平均，得到文本到图像方向的损失；两者再取平均是总损失。
依据：自行构造
:::

### 两个编码器与训练规模

#### 图像编码器

图像端有两种选择。一种是 ResNet，卷积网络用一个小窗口在图上滑动、逐处加权求和来抽特征，[残差连接](wiki/residual-connection)让它能叠得很深。CLIP 在 ResNet-50 上做了三处改动：换成改进的残差块结构、加抗混叠的模糊下采样（缩小图像前先做一次低通模糊，以免相邻像素混在一起）、把最后取平均的那层换成注意力池化（按内容对各位置加权求和）。另一种是 [Vision Transformer](wiki/vit)：把图切成固定大小的小块、每块映成一个向量排成序列，交给由标准注意力层叠成的 Transformer。放大模型时，ResNet 版把算力同时分给宽度、层数和输入分辨率，而不是只加其中一维。

依据：paper/clip_paper.tex Choosing and Scaling a Model

#### 文本编码器

文本端是一个 Transformer（由[注意力](wiki/standard-attention)层叠成），基础配置是 12 层、宽 512、8 个注意力头（把注意力拆成多组并行计算）。文本先按子词切分，词表按子词而非整词建（一个词可由多个子词拼成），规模 49152，最长长度截到 76。序列两端加 [SOS]（句首）与 [EOS]（句尾）两个标记，取最高层 [EOS] 位置的激活（该位置输出的那一列数）当整段文本的表示。两端编码器的输出各经一个线性投影映到同一维度，得到图和文可直接比较的多模态嵌入空间（多模态指这里同时处理图像和文本两类输入）。放大时只把文本端的宽度按图像端宽度增长的比例同步放大，不加层数。

依据：paper/clip_paper.tex Choosing and Scaling a Model；paper/clip_paper.tex Selecting an Efficient Pre-Training Method

#### 训练配置与规模

CLIP 一共训练了 5 个 ResNet 和 3 个 Vision Transformer，用 Adam 优化器（按每个参数梯度的一阶、二阶矩自适应调整步长的优化方法）训练。批量（一次并行处理多少对图文）很大，温度有一个初始值和一个上限以防止训练不稳。

依据：paper/clip_paper.tex Training

::: 训练配置与规模
- 批量 32768；共训练 32 轮；温度从 0.07 起，上限 100。
- 权重衰减（每步把权重按比例缩小一点，抑制权重过大）0.2；Adam 的 $\beta_1$（一阶矩的衰减系数）取 0.9。
- RN50x64（算力约为 ResNet-50 的 64 倍，按宽度、深度、分辨率同比例放大）：592 张 V100 显卡，训练 18 天。
- ViT-L/14：256 张 V100，训练 12 天。
- ViT-L/14（Vision Transformer 的一种规模）另在 336 像素分辨率上多训练 1 轮，得到 ViT-L/14@336px。
依据：paper/clip_paper.tex Training；paper/clip_paper.tex Model Hyperparameters
:::

### 零样本分类：用类别名生成分类器

#### 分类器怎么算出来

拿到一个数据集，把它的全部类别名当作候选文本，用文本编码器各编成一个向量；图像也用图像编码器编成一个向量。图像向量与每个类别向量算余弦相似度，再乘缩放系数 $1/\tau$（即除以温度 $\tau$），经 softmax 归一成一组和为 1 的概率，概率最高的类别就是预测。两端输出都做过归一化（把每个向量的长度化为 1），余弦相似度就等于点积、对图像向量是线性的，这一步等价于一个无偏置的线性分类器：类别向量充当它的权重，图像向量是输入，相似度是打分，softmax 是接在打分之后的分类头。这些权重由文本生成，换一批类别名就换一组权重，所以分类器不用训练。同一个数据集的类别向量只需算一次并缓存，之后的预测复用它，生成分类器的开销被摊到全部预测上。

依据：paper/clip_paper.tex Using CLIP for Zero-Shot Transfer

::: 三个类别的一次分类
一个图像向量与三个类别文本向量算相似度，温度 $\tau=0.1$。
依据：自行构造

1. 「一只猫」0.8、「一只狗」0.3、「一辆车」-0.1。
2. 除以 $\tau$：$0.8/0.1=8$、$0.3/0.1=3$、$-0.1/0.1=-1$。
3. softmax：$e^8=2981$、$e^3=20.09$、$e^{-1}=0.368$，分母 $3001.4$，得 $0.993$、$0.0067$、$0.0001$。
4. 取概率最大的一类，判为「一只猫」。
依据：自行构造
:::

#### 提示工程与模板集成

类别名直接当文本用往往不够。数据集里的类名有时忽略上下文，一词多义会让模型分不清是哪个意思，例如 crane 既指鹤也指起重机。另外，能与图配上的文本在预训练数据里通常是一整句话，很少只是一个词，单独一个类别名和训练分布不一致。用一个模板包住类名能补上这两点，默认模板是「A photo of a {label}.」（一张{标签}的照片）。按任务换成更贴切的模板还能再提高，比如在宠物数据集上用「一张{标签}的照片，一种宠物」，在识别图中文字（OCR）的数据集上给要识别的内容加引号。把多个模板编出的文本向量做平均，相当于集成多个分类器；在向量空间里平均，缓存一套平均后的向量就行，开销与用单个模板一样。

依据：paper/clip_paper.tex Prompt Engineering and Ensembling

::: 提示与集成各提高多少
- ImageNet 上，用默认模板比只用类名高 1.3 个点。
- 再对 80 个模板做集成，ImageNet 上比单个默认模板再高 3.5 个点。
- 两者合起来，ImageNet 上提高接近 5 个点。
- 在 36 个数据集上平均，提示与集成提高接近 5 个点。
依据：paper/clip_paper.tex Prompt Engineering and Ensembling
:::

### 零样本分类的效果

#### 零样本分类的水平

零样本 CLIP 的准确率大致接近一个在当时已经偏弱的有监督基线：在一个普通的 ResNet-50 图像特征（编码器输出的那个向量）上训练出来的线性分类器（逻辑回归，对特征做线性打分、取最高分的类）。

依据：paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Initial Comparison to Visual N-Grams；paper/clip_paper.tex Limitations

::: ImageNet 上的具体数
- 零样本 CLIP 在 ImageNet 上 76.2%，与原始 ResNet-50 相当。
- 在这 27 个数据集上，它赢过 ResNet-50 特征上的逻辑回归 16 个。
依据：paper/clip_paper.tex Initial Comparison to Visual N-Grams；paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance
:::

#### 偏弱的任务

零样本 CLIP 在一批专门、抽象或系统的任务上低于有监督基线：卫星图与病理切片这类专业图像、合成场景里的计数、交通标志识别、估计与最近车辆的距离。细粒度分类上有强有弱，动作识别反而更强。

细粒度上的强弱差别，很可能主要来自各任务在预训练数据里拿到的监督量不同；动作识别强，推测是自然语言对动词的监督比 ImageNet 那种以名词为主的监督更宽。

依据：paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Limitations

::: 弱项与强弱对比的具体数
- 卫星图、病理切片、合成计数、交通标志、距离估计这几类，都低于有监督基线。
- 细粒度分类两头分化：一组高 20 个点以上，另一组低 10 个点以上。
- 视频动作识别高 14.5 与 7.7 个点。
依据：paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance
:::

#### 随规模变化与鲁棒性

零样本误差随模型算力平稳下降，平均误差落在一条 log-log 直线附近，单个数据集上则起伏更大。分布迁移（测试数据的分布与训练分布不同）下它比同等准确率的有监督模型更稳：在一个大规模图像分类数据集 ImageNet 上的准确率，与它在自然分布迁移数据集上的准确率之间的差距被缩小；一旦把模型适配到 ImageNet 训练集，ImageNet 准确率上升，迁移集上的平均准确率反而略降。在一个 37 类猫狗品种的数据集上，零样本 CLIP 的准确率高于只看了 1 到 2 个样本的人类。

依据：paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Robustness to Natural Distribution Shift；paper/clip_paper.tex Comparison to Human Performance

::: 规模、鲁棒性与人类对比的数字
- 5 个 ResNet CLIP 模型在 36 个数据集上的 39 项评测，平均误差随算力跨 44 倍范围呈 log-log 线性。
- 分布迁移差距最多缩小 75%；适配 ImageNet 后 ImageNet 准确率从约 76% 升到 85.4%，提高 9.2 个点。
- Oxford IIT Pets：零样本人类 53.7%，零样本 CLIP 93.5%；人类看 1 到 2 个样本后升到 75.7%。
依据：paper/clip_paper.tex Analysis of Zero-Shot CLIP Performance；paper/clip_paper.tex Robustness to Natural Distribution Shift；paper/clip_paper.tex Comparison to Human Performance
:::

### 表示质量：线性探针

把图像编码器末端、投影到多模态空间之前的那列特征取出来，在它上面训练一个线性分类器（冻结编码器，只学这一层），可以测出这套表示本身的好坏。预训练得到的表示在一批与预训练无关的下游数据集上都能用，说明对比预训练学到的是通用、可迁移的表示，不只是为了某一套类别。

依据：paper/clip_paper.tex Representation Learning

::: 为什么用线性分类器来测表示
微调（在预训练表示上继续训练、整体权重都可再更新）会让表示适配到每个数据集，掩盖预训练表示本身的不足；线性分类器可变的部分少，更能暴露这些问题，也便于在很多模型之间做比较。
依据：paper/clip_paper.tex Representation Learning
:::

### 适用条件与边界

零样本要起作用，依赖要分的概念在预训练数据里有对应图文对。预训练数据里几乎没有像手写数字的图，它的表现就不如一个在原始像素上训练的分类器。它的水平有限：在有训练集的数据集上，它大致只与一个偏弱的有监督基线相当，要追上整体最好水平还要多得多的算力，以现有硬件不可行。它不解决数据效率问题，只是用能扩到几亿样本的监督源来补。它的数据从公开网络抓取，未经筛选与整理，模型因此学到不少社会偏见；这份数据不公开。

在细粒度、计数以及预训练里少见的任务上它偏弱；对预训练里很少出现的新任务，表现可能接近随机。它也不能生成新输出，只能从给定分类器已有的概念里选一个。一条误解是它已经解决泛化（在训练时没见过的数据上也照样好用）：它其实绕开了这个问题，指望数据足够大以致一切都在分布内，手写数字说明这个假设很容易被打破。另一条误解是评测集可以当中立标尺：主要结果的 27 个数据集是与 CLIP 的开发共同演化出来的，不是中立基准。

依据：paper/clip_paper.tex Limitations；repo/CLIP/model-card.md Data

::: 边界上的几个数
- 要追上整体最好水平，估计还要约 1000 倍算力。
- 手写数字 MNIST 上零样本只到 88%。
依据：paper/clip_paper.tex Limitations
:::
