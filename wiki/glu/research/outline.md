# GLU

- 描述：GLU 是一种逐维度的乘性门控单元：把输入做两个线性投影，其中一个过 sigmoid 压到 0 到 1 之间当门，再与另一个逐元素相乘，让每个维度各自决定放行多少。
- 摘要：$h(\mathbf{X}) = (\mathbf{X}\mathbf{W}+\mathbf{b}) \otimes \sigma(\mathbf{X}\mathbf{V}+\mathbf{c})$，一条线性投影乘上一道 sigmoid 门，既保留非线性又给梯度留一条不被压的线性通路。
- 主题：模型结构, 训练与优化
- 标签：模型架构

## 长页

- 开头｜概览：GLU 是按输入逐维度放行或压低的乘性门控单元；为什么需要：门控让网络自己控制哪些信息往下传，并减少梯度消失；补一句 sigmoid 的定义｜230｜paper/src-v3/main.tex L107；paper/src-v3/main.tex L166-L178；paper/src-v3/main.tex L203-L232
- ### 定义｜两头各一个线性投影，一头过 sigmoid 得门、逐元素乘另一头；门的取值在 0 到 1，门趋 0 或趋 1 时输出分别退化成什么｜235｜paper/src-v3/main.tex L166-L178
- ### 一个手算的小例子｜取 2 维输入和 2×2 权重，逐步算出值分支、门分支与逐元素乘积；例子全过程折叠｜150｜自行构造
- ### 它为什么给梯度留了一条线性通路｜sigmoid 门把它的导数乘进梯度；GLU 的导数拆成哪两项、哪一项不带 σ 的导数；推导折叠｜300｜paper/src-v3/main.tex L218-L232
- ### 和 tanh 门控的区别｜先一句讲清 tanh（实数映到 -1 到 1、两端趋平）；tanh 门控的导数两项都带已趋 0 的导数，与上一节 GLU 那条通路相对｜80｜paper/src-v3/main.tex L215；paper/src-v3/main.tex L476-L481
- ### 适用条件与边界｜线性通路只在门被激活时成立，门饱和到 0 时也断；门控本身要增加参数；输入序列更长时收益递减｜200｜paper/src-v3/main.tex L231；paper/src-v3/main.tex L476-L499；paper/src-v3/main.tex L542-L553；paper/src-v3/main.tex L572-L582

## 短页

- ### 逐维度的乘性门｜GLU 是什么：门逐维度缩放，缩放系数由输入算出的 sigmoid 决定，让每个维度各自放行多少；补 sigmoid 的取值｜250｜paper/src-v3/main.tex L166-L178
- ### 让梯度有一条不被压的路｜为什么需要：门控要按输入逐维度调节，又不能让梯度在传递时被逐层压没；GLU 的门只做缩放，不把带回导数的因子套进去｜300｜paper/src-v3/main.tex L203-L232
- ### 两道投影相乘，一道当门｜怎么做：给出公式，门全为 1 或全为 0 时各自退化成什么｜250｜paper/src-v3/main.tex L166-L178
