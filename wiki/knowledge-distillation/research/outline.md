# 知识蒸馏

- 描述：知识蒸馏是把一个已训练大型模型的知识迁移到小型模型的方法，做法是让小模型去拟合大模型给出的输出概率分布，而不是只拟合真实标签。
- 摘要：在 softmax 里加热度 $T$，大模型输出的软目标里带着类与类之间的相似结构；小模型拟合这些软目标，常能与大模型的泛化能力接近，而规模小得多、便于部署。
- 主题：训练与优化, 概率与统计
- 标签：训练

## 长页

- 开头｜先交代大模型的训练与部署之别、模型输出的概率分布，讲清 softmax 和它的输入 logit；蒸馏是什么、为什么需要、大概怎么做｜550｜paper/src-v1/introduction11.tex Introduction；paper/src-v1/abstract11.tex L3-L20；paper/src-v1/distillation11.tex Distillation
- ### 软目标：为什么概率分布比标签信息多｜类别概率的相对比例编码类间相似结构，硬标签只有正确类别；分布较平时软目标每个样本信息更多、梯度方差更小｜650｜paper/src-v1/introduction11.tex Introduction
- ### 温度与两个目标函数｜softmax 加热度把分布变软；迁移集（训练学生的数据）上用软目标训练，推理回 $T=1$；软目标与真实标签两项交叉熵加权相加，乘 $T^2$；例子折叠｜700｜paper/src-v1/distillation11.tex Distillation；paper/src-v1/introduction11.tex Introduction
- #### 匹配 logits 是蒸馏的特例｜单个 logit 的交叉熵梯度，高温一阶近似、去均值后化简为最小化 $\tfrac12(z_i-v_i)^2$；低温对很负 logits 关注少得多；推导折叠｜620｜paper/src-v1/distillation11.tex Matching logits is a special case of distillation
- ### 软目标传了什么：几个实验｜MNIST 大小模型与只拟合软目标的错误数、缺数字 3 靠偏置修正；3% 数据下软目标仍接近全量准确率；蒸馏把多个模型平均的提升转移到单模型；条件与数字折叠｜600｜paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST；paper/src-v1/regularizer11.tex Soft Targets as Regularizers；paper/src-v1/speech11.tex Results
- ### 适用条件与边界｜需要先有训练好的大模型与迁移集；教师在新数据上表现好时学生也更好；学生远小于教师时中间温度 2.5 到 4 更好；一条误解：把知识等同于参数｜420｜paper/src-v1/introduction11.tex Introduction；paper/src-v1/distillation11.tex Matching logits is a special case of distillation；paper/src-v1/prelim_mnist11.tex Preliminary experiments on MNIST

## 短页

- ### 学生模仿老师的输出分布｜是什么：把已训练大模型的知识迁到小模型，学生不只学真实标签，而是模仿老师输出的概率分布｜300｜paper/src-v1/distillation11.tex Distillation；paper/src-v1/introduction11.tex Introduction
- ### 部署要省，真实标签信息不足｜为什么需要：大模型训练算力足、部署受延迟与资源限制；老师概率分布里错误类别的相对大小是真实标签没有的，模仿它在新数据上也往往更好｜350｜paper/src-v1/introduction11.tex Introduction
- ### 温度与软目标｜怎么做：先用一句话讲清 softmax；给老师 softmax 加热度得软目标，用同温度训练学生，推理回温度 1；有标签时加真实标签交叉熵（链接前置页）｜300｜paper/src-v1/distillation11.tex Distillation
