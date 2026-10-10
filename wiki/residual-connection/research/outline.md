# 残差连接

- 描述：残差连接把若干层的输入一路直连、加到这些层的输出上，让中间层只学一个相对输入的小修正，深层网络因此可以加深而不退化。
- 摘要：把要拟合的映射记为 $\mathcal{H}(x)$，让层去学残差 $\mathcal{F}(x)=\mathcal{H}(x)-x$，输出就是 $\mathcal{F}(x)+x$；当恒等接近最优时，把残差推向零可能比让一摞非线性层拟合恒等映射容易。
- 主题：模型结构
- 标签：网络结构

## 长页

- 开头｜交代层是输入到输出的映射、训练是调参数让输出误差变小、按说越深越好，点出退化现象和残差形式 $\mathcal{F}(x)+x$ 这一主张｜360｜paper/src-v1/residual_v1_arxiv_release.tex Introduction；paper/src-v1/residual_v1_arxiv_release.tex Residual Learning；paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts
- ### 加深反而更差｜退化：加深时训练误差也升，不是过拟合（过拟合是训练误差仍降、验证误差却升）；构造论证说明是优化问题；数字表折叠｜420｜paper/src-v1/residual_v1_arxiv_release.tex Introduction；paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification
- ### 残差块怎么写｜本章讲残差块的定义与实现，内容在两小节，只有标题｜0｜paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts
- #### 把目标从 $\mathcal{H}(x)$ 改成 $\mathcal{F}(x)+x$｜把拟合目标重写成残差 $\mathcal{F}=\mathcal{H}-x$、输出 $y=\mathcal{F}(x)+x$；解释各符号｜460｜paper/src-v1/residual_v1_arxiv_release.tex Residual Learning；paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts
- #### 捷径连接的代价与维度匹配｜捷径是恒等、无参数、只做逐元素相加；要求同维，维度不匹配时补零（零填充）或用线性变换（投影）对齐｜380｜paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Network Architectures；paper/src-v1/residual_v1_arxiv_release.tex ImageNet Classification
- ### 为什么它让深网好训｜为何学残差比拟合恒等容易；实测残差响应大体更小、层越深越倾向少改动信号；手算例子折叠｜520｜paper/src-v1/residual_v1_arxiv_release.tex Residual Learning；paper/src-v1/residual_v1_arxiv_release.tex CIFAR-10 and Analysis
- ### 适用条件与边界｜恒等捷径是充分非唯一；与门控型捷径（多一个会关闭的开关）的区别；单层 $\mathcal{F}$ 未见优势；极深时收益下降、作者推测是过拟合（限小数据集）｜440｜paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Related Work；paper/src-v1/residual_v1_arxiv_release.tex CIFAR-10 and Analysis

## 短页

- ### 把输入加到输出上｜是什么：层把输入映射成输出；$x$ 经若干层映射成残差 $\mathcal{F}(x)$，又经直连到出口，逐元素相加得 $\mathcal{F}(x)+x$｜330｜paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts
- ### 加深为什么反而更差｜为什么需要：训练是调参数让输出误差变小；加深时训练误差先饱和后上升，不是过拟合（训练降、验证升）；残差形式让什么都不做对应学近零的 $\mathcal{F}$｜330｜paper/src-v1/residual_v1_arxiv_release.tex Introduction；paper/src-v1/residual_v1_arxiv_release.tex Residual Learning
- ### 加一条无参数的捷径｜怎么做：相加后过非线性；捷径无参数、不增加计算；要求同维，维度变化时补零（零填充）或用学出来的线性变换（投影）对齐｜300｜paper/src-v1/residual_v1_arxiv_release.tex Identity Mapping by Shortcuts；paper/src-v1/residual_v1_arxiv_release.tex Network Architectures
