# Hadamard Transform

- 描述：Hadamard Transform 是用正负号组合混合坐标的正交变换，用于在量化前分散集中在少数坐标上的大值，配对变换权重在实数运算下保持原计算。
- 摘要：长度不变，坐标分布改变；先分散大值、同步变换权重，再量化，二次幂维度可用递归加减实现。
- 主题：数学基础, 推理系统
- 标签：量化

## 长页

- 开头｜界定量化中的 Hadamard 是变换而非量化器，链接量化前置页，解释权重和激活，概述目的与工作方式｜180｜paper/quarot-v2-src/intro.tex L5-L8；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations
- ### 正负号组成的正交换基｜定义未归一化矩阵及归一化变换，说明保范数和逆变换；自行构造的矩阵与正交推导折叠，与逐元素乘积只区别一句｜400｜paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices
- ### 分散大值，让共享网格变细｜解释大值撑宽共享步长及随机符号预处理；自行构造的孤立大值、随机二阶矩推导和非零误差量化算例折叠｜500｜paper/quarot-v2-src/intro.tex L5-L8；paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations
- ### 同步变换权重，保持原计算｜解释线性层，用列向量给出量化前实数运算下的配对式及权重离线处理；自行构造的抵消与逆变换误差推导折叠｜500｜paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 1a: Weight Modification；paper/quarot-v2-src/method.tex Stage 2a: Weight Quantization
- ### 递归加减，不做稠密矩阵乘｜解释二次幂维度的递归加减、最终归一化和复杂度；自行构造的递归矩阵、逐轮手算及验证代码折叠｜400｜paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices
- ### 适用条件与边界｜说明不保证峰值或误差下降、不能穿过任意非线性、二次幂实现的维度限制及运行时与浮点代价；自行构造反例折叠｜350｜paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/background.tex Computational Invariance；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 1a: Weight Modification

## 短页

- ### 保持长度的坐标混合｜先链接量化前置页，定义归一化 Hadamard 变换，解释权重和激活，说明它不是量化器｜220｜paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations
- ### 大值会撑宽量化网格｜解释共享步长被离群值撑宽和变换的目的，不承诺所有输入峰值下降｜230｜paper/quarot-v2-src/intro.tex L5-L8；paper/quarot-v2-src/method.tex Stage 2b: Online Quantization Operations
- ### 先换基、配对权重、再量化｜解释线性层，给出量化前实数运算下的配对式；解释固定随机符号和二次幂递归实现，区分量化前后｜400｜paper/quarot-v2-src/background.tex Orthogonal, Rotation and Hadamard Matrices；paper/quarot-v2-src/method.tex Stage 1b: Rotate FFN activations；paper/quarot-v2-src/method.tex Stage 2a: Weight Quantization
