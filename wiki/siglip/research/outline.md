# SigLIP

- 描述：SigLIP 把语言-图像对比预训练里的 softmax 对比损失换成逐对的 sigmoid 损失，每个图文对各算一个二分类损失，不需要在整批相似度上归一化。
- 摘要：用逐对 sigmoid 损失 $-\frac{1}{|\mathcal{B}|}\sum_{i,j}\log\sigma\big(z_{ij}\,(t\,\mathbf{x}_i\!\cdot\!\mathbf{y}_j+b)\big)$ 替代 softmax 对比损失。配对取 $z_{ij}=+1$、其余取 $-1$。损失不再依赖全局归一化，批量大小与损失定义解耦，可掉大批量、可分块、更省显存。
- 主题：多模态, 训练与优化
- 标签：视觉与多模态

## 长页

- 开头｜一句说明损失函数是训练时被最小化的目标量；交代图文对比预训练的场景（图像/文本编码器、配对为正），SigLIP 是什么、为什么需要、大致怎么工作｜510｜paper/arXiv-2303.15343v4.txt L36-L50；paper/arXiv-2303.15343v4.txt L107-L122；paper/arXiv-2303.15343v4.txt L128-L142
- ### 对比学习与 softmax 损失｜本章回答换掉的是什么，只有标题，内容在两个小节｜0｜paper/arXiv-2303.15343v4.txt L48-L50
- #### 相似度与正负对｜两个编码器各把图像、文本映成同维向量并归一化，点积当相似度，一句解释 softmax｜300｜paper/arXiv-2303.15343v4.txt L219-L229；paper/arXiv-2303.15343v4.txt L620-L626
- #### softmax 对比损失与它的代价｜损失两个方向各做一次 softmax、数值稳定要减最大值、把批量大小写进定义；需要整批相似度矩阵因而占显存，损失公式折叠｜380｜paper/arXiv-2303.15343v4.txt L107-L115；paper/arXiv-2303.15343v4.txt L588-L630；paper/arXiv-2303.15343v4.txt L669-L671
- ### 逐对的 sigmoid 损失｜本章讲损失怎么从整批归一化改成逐对二分类，只有标题，内容在三个小节｜0｜paper/arXiv-2303.15343v4.txt L631-L639
- #### 损失函数与标签｜配对 +1、非配对 −1，取 logistic 损失后平均；σ 是 logistic 函数（[交叉熵](wiki/cross-entropy)同族）；公式折叠｜480｜paper/arXiv-2303.15343v4.txt L640-L655；paper/arXiv-2303.15343v4.txt L183-L198
- #### 温度与偏置项｜logit 乘可学习温度再加可学习偏置；$b$ 取 −10 让训练从接近先验起步；两组对比与推导折叠｜440｜paper/arXiv-2303.15343v4.txt L655-L665；paper/arXiv-2303.15343v4.txt L1472-L1523
- #### 一个批量手算一轮｜四对图文的小批量，列相似度、标签、logit 与每对损失，全过程折叠｜60｜自行构造
- ### 分块实现｜本章回答为什么能分块、分块怎么做，只有标题，内容在两个小节｜0｜paper/arXiv-2303.15343v4.txt L666-L676
- #### 逐对独立带来的可分块性｜损失是逐对独立项之和，因此跨设备不必 all-gather 整个相似度矩阵，一句解释 all-gather｜420｜paper/arXiv-2303.15343v4.txt L666-L709；paper/arXiv-2303.15343v4.txt L586-L587
- #### 分块的一次轮换｜三设备、批量 12 的示意，本设备只保留 4×4 分块，文本嵌入逐轮换到相邻设备，全过程折叠｜240｜paper/arXiv-2303.15343v4.txt L235-L585；paper/arXiv-2303.15343v4.txt L666-L709
- ### 训练性质：批大小、负样本比与噪声｜小批量时明显优于 softmax、批量增大差距收窄并在 32k 附近饱和；正负不平衡不致命、难负样本（与正样本相近的负样本）更有价值；对噪声更鲁棒，数字折叠｜400｜paper/arXiv-2303.15343v4.txt L131-L142；paper/arXiv-2303.15343v4.txt L857-L877；paper/arXiv-2303.15343v4.txt L931-L936；paper/arXiv-2303.15343v4.txt L1393-L1471；paper/arXiv-2303.15343v4.txt L1524-L1545
- ### 适用条件与边界｜成立条件（非配对的图文被一律当负样本，这个假设并不精确）与代价，以及一条常见误读：批量越大越好｜380｜paper/arXiv-2303.15343v4.txt L224-L229；paper/arXiv-2303.15343v4.txt L708-L709；paper/arXiv-2303.15343v4.txt L814-L820；paper/arXiv-2303.15343v4.txt L1549-L1560

## 短页

- ### 逐对判断的对比损失｜是什么：一句解释损失函数、一句解释对比预训练与 softmax、一句解释 sigmoid，再说用 sigmoid 逐对损失替代 softmax｜360｜paper/arXiv-2303.15343v4.txt L11-L17；paper/arXiv-2303.15343v4.txt L631-L639
- ### softmax 要在整批上归一｜为什么需要：softmax 损失要在整批相似度上归一，需整批相似度矩阵、显存大、数值不稳，还把批量大小写进定义；逐对独立可绕开｜380｜paper/arXiv-2303.15343v4.txt L107-L122；paper/arXiv-2303.15343v4.txt L588-L630；paper/arXiv-2303.15343v4.txt L669-L671
- ### 一对一个 logit 独立算｜怎么做：每对算 logit（相似度乘温度加偏置）、配对 +1 其余 −1，取 sigmoid 损失求平均｜250｜paper/arXiv-2303.15343v4.txt L183-L198；paper/arXiv-2303.15343v4.txt L655-L665
