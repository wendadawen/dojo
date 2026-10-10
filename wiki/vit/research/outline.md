# Vision Transformer

- 描述：Vision Transformer 把图像切成固定大小的小块，每块线性映射成一个向量，排成序列后交给标准的 Transformer 编码器，用监督学习做图像分类。
- 摘要：把 $H\times W$ 的图像切成 $N=HW/P^2$ 个 $P\times P$ 的小块当作 token，用标准 Transformer 编码器处理；在 JFT-300M 这类大数据上预训练后，它在多个图像分类基准上匹配或超过当时最好的卷积网络。
- 主题：模型结构, 注意力机制
- 标签：视觉与多模态

## 长页

- 开头｜ViT 是什么、为什么需要、大概怎么工作：把图像切成小块、线性映射成序列，交给标准 Transformer 编码器做分类；链接前置｜240｜paper/src-v2/03_method.tex L12；paper/src-v2/01_introduction.tex L15-L18；paper/src-v2/01_introduction.tex L23-L27；paper/src-v2/05_conclusion.tex L4-L7；paper/src-v2/03_method.tex L27-L29
- ### 把图像变成一段序列｜在标准 Transformer 上只做最少修改：把二维图像重排成小块序列并线性投影；本章只有标题，内容在两个小节｜0｜paper/src-v2/01_introduction.tex L15；paper/src-v2/03_method.tex L21-L24
- #### 分块与线性嵌入｜把图像切成 N 个 P×P 小块、展平乘矩阵映到 D 维；N 即序列长度；小块投影的折叠例子｜400｜paper/src-v2/03_method.tex L23-L25；paper/src-v2/03_method.tex L39-L40；paper/src-v2/04_experiments.tex L31-L32
- #### 分类 token 与位置嵌入｜位置嵌入是什么、怎么加、为何要加（自注意力对顺序不敏感）；分类 token 是什么、为何用它的输出；例子延续投影折叠｜380｜paper/src-v2/03_method.tex L27-L33
- ### 编码器与分类头｜编码器各块的组成与顺序，公式折叠；分类头把分类 token 输出映成类别分数；自注意力作前置概念链接｜420｜paper/src-v2/03_method.tex L35-L44；paper/src-v2/03_method.tex L27-L29
- ### 与卷积网络差在归纳偏置｜一两句解释卷积网络的归纳偏置与「归纳偏置」一词；ViT 只有 MLP 局部、自注意力全局，二维结构只在两处注入｜220｜paper/src-v2/03_method.tex Inductive bias；paper/src-v2/03_method.tex Fine-tuning and Higher Resolution；paper/src-v2/05_conclusion.tex L4
- ### 预训练与高分辨率微调｜一句解释预训练与微调；换掉分类头为零初始化线性层；提高分辨率导致序列变长，对位置嵌入二维插值｜280｜paper/src-v2/03_method.tex Fine-tuning and Higher Resolution；paper/src-v2/04_experiments.tex Setup；paper/src-v2/03_method.tex L27-L29
- ### 数据规模决定成败｜中等数据不如卷积网络、大数据上胜过归纳偏置（一句解释过拟合）；90M 以上反超，JFT 上在多数任务超过 BiT-L；数字折叠｜380｜paper/src-v2/01_introduction.tex L20-L27；paper/src-v2/04_experiments.tex Pre-training Data Requirements；paper/src-v2/04_experiments.tex Comparison to State of the Art
- ### 计算代价与规模权衡｜同等性能省约 2–4 倍预训练算力，在尝试范围内似乎未饱和；改用更小的小块更贵；模型与算力数字折叠｜340｜paper/src-v2/04_experiments.tex Comparison to State of the Art；paper/src-v2/04_experiments.tex Scaling Study；paper/src-v2/06_appendix.tex L138-L171
- ### 适用条件与边界｜成立条件是有大规模预训练数据；代价是从零训练需强正则；两条误解各一句并指向前节：把二维结构当成建进了模型、任何数据量都胜过卷积网络｜300｜paper/src-v2/04_experiments.tex Setup；paper/src-v2/01_introduction.tex L20-L25；paper/src-v2/04_experiments.tex Pre-training Data Requirements

## 短页

- ### 把图片当成一串小块｜是什么：把图像当作小块组成的序列、用原本处理文本的标准 Transformer 编码器处理（注意力链接前置页）；一两句解释小块、位置嵌入、分类 token｜280｜paper/src-v2/03_method.tex L12；paper/src-v2/03_method.tex L23-L33
- ### 不靠卷积也能做视觉｜为什么需要：注意力（链接前置页）多与卷积网络结合或替换其部分结构；先在大数据上训练再迁移（预训练）后，纯 Transformer 追平或超过卷积网络｜300｜paper/src-v2/01_introduction.tex L10-L13；paper/src-v2/01_introduction.tex L23-L25；paper/src-v2/05_conclusion.tex L4-L7
- ### 切块、嵌入、编码、分类｜怎么做：切块展平、线性投影到 D 并加位置嵌入与分类 token、过 L 层编码器、取分类 token 输出经分类头（把向量映成各类别分数）｜300｜paper/src-v2/03_method.tex L23-L33；paper/src-v2/03_method.tex L35-L44
