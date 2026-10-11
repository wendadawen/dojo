# Residual Connection

- 描述：残差连接把若干层的输入一路直连、加到这些层的输出上，让中间层只学一个相对输入的小修正，从而能训练很深的网络。
- 摘要：把要拟合的映射记为 $\mathcal{H}(x)$，让层去学残差 $\mathcal{F}(x)=\mathcal{H}(x)-x$，输出就是 $\mathcal{F}(x)+x$；当恒等接近最优时，把残差推向零可能比让一叠非线性层拟合恒等映射容易。
- 主题：模型结构
- 标签：网络结构

## 资料

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv preprint, v1, 2015. [https://arxiv.org/abs/1512.03385v1](https://arxiv.org/abs/1512.03385v1)
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. deep-residual-networks: official models. 2016. [https://github.com/KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

## 短页

### 把输入加到输出上

神经网络由一层一层叠成，一层把输入向量算成输出向量。残差连接让其中若干层不直接学最终结果，而是学一个残差：把这几个层的输入 $x$ 送进去，它们算出一个 $\mathcal{F}(x)$，这个相对于输入算出的修正量就叫残差。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

同时 $x$ 从一条直连的通路原样走到出口。两条路在出口逐元素相加，得到 $\mathcal{F}(x)+x$，再对每个分量做一次非线性变换，得到这几个层的输出。这样的一组层和它的捷径合起来叫一个残差块。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

$\mathcal{F}$ 就是这几层要学的残差函数，直连通路叫捷径，它不做变换、不带参数。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

### 加深为什么反而更差

训练网络就是不断调参数，让网络的输出与目标的误差变小。把网络加深时会出现一个反常现象：训练误差不再下降，反而上升。
依据：paper/src-v1/residual_v1_arxiv_release.tex Introduction

这跟过拟合不一样：过拟合是训练误差还在降、另一批数据上的误差却升了，而这里训练误差本身就在升。为什么会这样当时还不清楚，一个猜想是深层的普通网络收敛很慢。
依据：paper/src-v1/residual_v1_arxiv_release.tex Introduction；paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification

残差连接针对的就是这个问题：它把「这几层直接学出结果」换成「在输入的基础上只学一点修正」。于是「这几层什么都不做」对应残差学成接近零，而把这几层待调的参数（权重）推向零，比让多层非线性层精确地拟合恒等映射（输出等于输入的变换）容易。
依据：paper/src-v1/residual_v1_arxiv_release.tex Residual Learning

### 加一条捷径，必要时对齐维度

做法是让这几层只学残差 $\mathcal{F}$，再把输入经一条捷径加到输出上，加法逐元素进行，相加之后还要过一次非线性函数（对每个分量做一次非线性变换）。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

捷径做的是恒等映射：把输入原样送到出口，不做任何变换。维度相同时它不引入参数，也不增加计算。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

相加要求两边维度相同；如果这几层改变了维度，就给捷径补零，或用一个学出来的线性变换把输入投到目标维度，让两边对得齐。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Network Architectures

## 长页

神经网络由一层一层叠成，每一层把输入向量变成输出向量。训练就是不断调这些层的参数，让网络输出和目标的误差变小。层数叫深度，近期证据表明深度很重要，大规模图像分类上最好的结果都用了很深的网络。但把普通网络继续加深时，出现了一个反常现象：训练误差不降反升，这叫退化。残差连接针对的就是它。它的做法是让若干层不直接拟合目标映射，而是只学这个映射与输入的差，再把输入原样加到结果上，相加后再过一次非线性函数，得到输出；这一组层连同捷径合起来叫一个残差块。把要拟合的映射记作 $\mathcal{H}(x)$，让这些层去学残差 $\mathcal{F}(x)=\mathcal{H}(x)-x$，输入经捷径加回这个残差，得到 $\mathcal{F}(x)+x$。这条从输入直通到输出的加法通路叫捷径连接。
依据：paper/src-v1/residual_v1_arxiv_release.tex Introduction；paper/src-v1/residual_v1_arxiv_release.tex Residual Learning；paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

### 加深反而更差

加深本应无害，甚至更好。把一个浅层网络和它的深层版本对照：深层版本可以把浅层学到的那些层照搬过来，再把多出来的层设成什么都不做，于是它至少不该比浅层差。可是实际训练时，深层版本的训练误差反而更高。这就是退化：深度增加后准确率先饱和再下降，而且训练误差本身也上升。
依据：paper/src-v1/residual_v1_arxiv_release.tex Introduction

退化不是过拟合。过拟合是训练误差还在下降、验证误差（在另一批留出的数据上算的误差）却开始上升，模型只是把训练集记得太熟。

退化里训练误差自己就升了，所以问题出在优化，模型没能把深层版本训好。
依据：paper/src-v1/residual_v1_arxiv_release.tex Introduction

一个可能是梯度在加深时消失（反向传回来的梯度越传越小），但这组对照网络都用了批归一化（一种把每层输入重新定尺度和中心的处理），前向信号的方差不为零，反向的梯度范数也正常，所以不像梯度消失。更深的原因当时还不清楚，一个猜想是深层的普通网络收敛很慢。
依据：paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification

::: 普通网络与残差网络的误差数字
在一个大规模图像分类数据集（ImageNet）的验证集上，比较 18 层与 34 层网络，残差网就在对应的普通网络上加捷径、层数相同。
| 网络 | 18 层 | 34 层 |
|---|---|---|
| 普通 | 27.94 | 28.54 |
| 残差 | 27.88 | 25.03 |
表里是 top-1 错误率（%），即模型给出的第一候选答案就错的那部分样本占的百分比，越低越好。普通网络加深后错误率上升，残差网络加深后错误率下降，方向与训练误差那边的现象一致。
依据：paper/src-v1/residual_v1_arxiv_release.tex L302-L311；paper/src-v1/residual_v1_arxiv_release.tex L320；paper/src-v1/residual_v1_arxiv_release.tex L292；paper/src-v1/residual_v1_arxiv_release.tex L322
:::

### 残差块怎么写

#### 把目标从 $\mathcal{H}(x)$ 改成 $\mathcal{F}(x)+x$

残差连接改的是这几层要学什么。原本希望一叠层直接拟合目标映射 $\mathcal{H}(x)$；改成让它们拟合残差 $\mathcal{F}(x)=\mathcal{H}(x)-x$，再把输入加回去，得到 $y=\mathcal{F}(x,\{W_i\})+x$，$y$ 再过一次非线性函数才是这几层的输出。这里 $x$ 是这几层的输入，$\{W_i\}$ 是堆叠各层里待学的权重，$\mathcal{F}$ 是这些层构成的非线性函数，$+x$ 就是把输入原样送过去的那条捷径。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Residual Learning

这一步只是把 $\mathcal{H}$ 重写成 $\mathcal{F}+x$，$\mathcal{F}$ 与 $\mathcal{H}$ 一样能逼近要拟合的映射（这个假设当时仍未决），换的只是好不好学。
依据：paper/src-v1/residual_v1_arxiv_release.tex Residual Learning

#### 捷径连接的代价与维度匹配

捷径做的是恒等映射：把输入原样送到出口，不做任何变换。维度相同时它不引入参数，也不增加计算。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

这套说法对全连接层（每个输入分量都与每个输出分量相连的一层）成立，对卷积层（对图像按局部窗口滑动做加权求和的一类层）也成立，只是加法改成两张特征图（卷积层输出的多通道二维网格，通道是其中一维）按通道、逐位置相加。如果 $\mathcal{F}$ 改变了维度，两边对不齐，就要把捷径也对齐：一种做法是给多出来的位置补零，另一种是用一个学出来的线性变换 $W_s$（投影）把输入投到目标维度，此时 $y=\mathcal{F}(x,\{W_i\})+W_s x$。实验里只把投影用在维度对不齐的地方，其余都用恒等，因为恒等已经够解决退化，而且更省内存和计算。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Network Architectures；paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification

### 为什么它让深网好训

退化提示：多出来的那些层不容易用一叠非线性层逼近恒等映射。残差形式换了一个参照——如果要拟合的映射离恒等不远，网络只需在恒等的基础上学一个小的残差 $\mathcal{F}(x)$，优化的起点就已经接近正确，只要修正这个小量，比直接拟合整个映射容易。这种把问题改写、让优化更好做的想法，可能有助于给问题做预条件（换一个等价的问题表述，让同一个优化过程更容易走）。
依据：paper/src-v1/residual_v1_arxiv_release.tex Residual Learning

有一个实验和这个看法一致：在一个小图像数据集（CIFAR-10）上，量每一层卷积（卷积窗口大小是 3×3）在批归一化之后、非线性之前的输出，看它的标准差，作为这一层改动信号的大小。残差网络的响应大体比同规模的普通网络小，这与「学到的残差函数整体比非残差函数更接近零」相符；而且层数越多，单层响应越小，对照的是 20、56、110 层的残差网络。
依据：paper/src-v1/residual_v1_arxiv_release.tex CIFAR-10 and Analysis

::: 手算：为什么「什么都不做」对应残差为零
用一个最简的残差块演示算术：这块只把输入乘一个权重 $w$，没有非线性、没有偏置（每层里可加的一个可调常数），算 $y=wx$。输入取 $x=2$。
1. 普通形式里，块直接算出输出 $y=wx$。要让块什么都不做，即输出仍等于输入 $2$，需要 $w=1$。
2. 残差形式里，块算的是残差 $\mathcal{F}(x)=wx$，输出是 $y=\mathcal{F}(x)+x=(w+1)x$。要让块什么都不做，需要 $w=0$。
3. 取 $w=0$，得 $y=2$；取 $w=0.1$，残差 $\mathcal{F}(x)=0.2$，输出 $y=2.2$。
4. 于是普通形式里「什么都不做」对应权重恰好等于 $1$，残差形式里对应权重等于 $0$。
依据：自行构造
:::

### 适用条件与边界

残差连接是解决退化的一个可行办法，不是唯一的解法。实验里换用投影捷径也能解决退化，代价是额外的内存与计算；它和门控型捷径也不同，后者的开关由数据决定、会关闭，残差连接里的捷径则一直直通，再叠加学到的残差。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification；paper/src-v1/residual_v1_arxiv_release.tex Related Work

实验中的 $\mathcal{F}$ 由两三层组成；若只有一层，就没观察到优势，因为它变成一个线性层。
依据：paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts

边界在深度收益上。在一个小数据集上把网络加到 1202 层，训练误差能压到 0.1% 以下，说明优化不再困难；但测试误差（在另一批数据上评估得到的误差）反而比 110 层的网络差。一种解释是过拟合——数据集太小、这么大的模型对它来说过大了。这是推测，不是已证明的结论。
依据：paper/src-v1/residual_v1_arxiv_release.tex CIFAR-10 and Analysis
