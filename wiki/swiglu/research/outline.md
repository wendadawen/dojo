# SwiGLU

- 描述：SwiGLU 是 Transformer 前馈层的一种写法，把前馈层的激活函数换成一道由 Swish 算出的门，逐元素缩放另一条线性投影。
- 摘要：$\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV$，两条线性投影逐元素相乘，一条先过 Swish 当门，用同一参数量换更好的结果。
- 主题：模型结构
- 标签：模型架构

## 长页

- 开头｜先交代前馈层、激活函数、Swish；再讲 SwiGLU 是什么、为何需要、大致怎么工作｜250｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants
- ### 前馈层里换上一道门｜前馈层两条线性变换与中间的 ReLU；GLU 改为两投影逐元素相乘、一个过 sigmoid，链接它；SwiGLU 只是把门换成 Swish｜280｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants
- ### SwiGLU 的公式｜给出公式与三个权重矩阵，逐项解释符号；门由 Swish 算出｜280｜paper/src-v1/main.tex Gated Linear Units (GLU) and Variants
- ### 手算一遍 SwiGLU｜取一维输入和两组权重，逐步算出值分支、门分支与乘积；全过程折叠｜150｜自行构造
- ### 隐层宽度为什么要乘 2/3｜比 ReLU 前馈层多一块权重，隐层宽度从 3072 降到 2048 才与它同参数；2/3 的来历折叠｜250｜paper/src-v1/main.tex Gated Linear Units (GLU) and Variants；paper/src-v1/main.tex Model Architecture
- ### 适用条件与边界｜困惑度、下游任务各一句解释；SwiGLU 与 GEGLU（门用 GELU 的同类变体）困惑度并列最好、下游大多更好，前提是参数量与计算量对齐；提升未必超出波动｜240｜paper/src-v1/main.tex Pre-Training and Perplexity Results；paper/src-v1/main.tex Fine-Tuning

## 短页

- ### 门换成 Swish 的前馈层｜SwiGLU 是什么：前馈层的一条线性投影过 Swish 当门，逐元素缩放另一条；补前馈层、Swish 的定义｜280｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Gated Linear Units (GLU) and Variants
- ### 为什么换掉 ReLU｜前馈层中间通常套 ReLU（max(0,·)）；换成由 Swish 算出的门后，困惑度与下游结果更好；补困惑度、下游任务各一句｜260｜paper/src-v1/main.tex Introduction；paper/src-v1/main.tex Pre-Training and Perplexity Results；paper/src-v1/main.tex Fine-Tuning
- ### 两道投影相乘，一道过 Swish｜怎么做：写出 SwiGLU 公式，说明它有三块权重、比常规前馈层多一块｜250｜paper/src-v1/main.tex Gated Linear Units (GLU) and Variants
