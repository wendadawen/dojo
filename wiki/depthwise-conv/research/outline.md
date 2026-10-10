# 深度可分离卷积

- 描述：深度可分离卷积把标准卷积一步做的空间滤波和跨通道组合拆成两步，先逐通道滤波（depthwise），再用 $1\times1$ 卷积跨通道组合（pointwise），乘加量降为原来的约 $1/N+1/D_K^2$。
- 摘要：一步卷积同时做空间滤波和跨通道组合，拆成逐通道滤波加 $1\times1$ 组合后，成本按 $1/N+1/D_K^2$ 比例缩小；$3\times3$ 时省 8 到 9 倍。
- 主题：模型结构
- 标签：网络结构

## 长页

- 开头｜交代卷积网络与通道，点出标准卷积一步同时做的滤波（邻域加权求和）与组合（跨通道混合）两件事，给出定义与动机，说明先看清一步再拆开算账｜250｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Depthwise Separable Convolution
- ### 一步卷积做了哪两件事｜只有标题，两个小节分别讲一步里的两件事和它的成本｜0｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 一步完成滤波与组合｜标准卷积在一个输出位置上对邻域和输入通道求加权和，把这件事指认成空间滤波与跨通道组合两件；公式、符号定义与配对示意折叠｜250｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 一步要算多少｜成本 $D_K^2MND_F^2$ 里每个符号是什么、怎么由各分量乘起来（推导折叠），引出贯穿本页的构造数字（折叠）｜250｜paper/src-v1/main.tex Depthwise Separable Convolution
- ### 拆成两步｜只有标题，内容在两个小节｜0｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 逐通道滤波｜每通道一个 $D_K\times D_K$ 核、只在本通道邻域上加权，输出通道数与输入相同；小例子折叠｜250｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 跨通道组合与两步总成本｜$1\times1$ 卷积在每个位置把各通道线性组合成新通道，补上 depthwise 缺的那一步；两步顺序与总成本公式折叠｜250｜paper/src-v1/main.tex Depthwise Separable Convolution
- ### 省了多少｜只看拆分的比值，本章回答省多少和什么时候最省｜0｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 缩减比怎么除出来｜总成本除以标准成本、逐项约分得 $1/N+1/D_K^2$；推导两行与沿用前面那组构造数字的实算折叠｜300｜paper/src-v1/main.tex Depthwise Separable Convolution
- #### 什么时候省得多｜$D_K=3$、$N$ 大时比值由 $1/9$ 主导、省 8 到 9 倍；$N$ 小或核接近 $1\times1$ 时省得少｜200｜paper/src-v1/main.tex Depthwise Separable Convolution
- ### 适用条件与边界｜收益成立的前提和代价；两条常见误解各一两句：把 depthwise 单独当完整替代、以为缩减比是固定的 8 到 9 倍｜250｜paper/src-v1/main.tex Depthwise Separable Convolution

## 短页

- ### 一步卷积拆成两步｜是什么：标准卷积一步同时做空间滤波（邻域加权求和）与跨通道组合，深度可分离卷积把它拆成逐通道滤波和 $1\times1$ 组合两步；一句释义卷积与通道｜300｜paper/src-v1/main.tex Depthwise Separable Convolution
- ### 为什么值得拆｜为什么需要：网络要小要快，每步只做一件事，成本降为约 $1/N+1/D_K^2$（$N$ 通道、$D_K$ 核边长），$3\times3$ 省 8 到 9 倍｜300｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Depthwise Separable Convolution
- ### 逐通道滤波与跨通道组合｜怎么做：先每通道一个 $D_K\times D_K$ 核只在本通道滤波，输出通道数不变，再用 $1\times1$ 卷积把各通道线性组合成新通道；只做第一步不够｜300｜paper/src-v1/main.tex Depthwise Separable Convolution
