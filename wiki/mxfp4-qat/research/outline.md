# MXFP4 量化感知训练

- 描述：MXFP4 量化感知训练是把模型权重编码成 4-bit 微缩浮点格式 MXFP4，并在训练的每一步前向里模拟这种量化，让模型学会适应由此带来的精度损失的训练方法。
- 摘要：MXFP4 用 32 个权重共享一个 8-bit 幂次尺度、每个权重存 4-bit 浮点元素；QAT 在训练前向先量化再反量化，反向用直通估计器把梯度传回原权重。
- 主题：训练与优化
- 标签：量化

## 长页

- 开头｜先说部署要把权重压低位宽引出量化，交代 MoE 是什么并链接 wiki/fusedmoe，再说这个方法是什么、为什么需要、大概怎么做｜300｜paper/k3-tech-report.txt L923-L928；standard/ocp-mx-spec-v1.0.txt L266-L296；paper/jacob-2018-qat.txt L254-L286
- ### MXFP4：块共享尺度的 4-bit 浮点｜一个 MXFP4 块是什么样：k 个元素共享一个尺度，元素和尺度的位宽；只有标题，内容在三个小节｜0｜standard/ocp-mx-spec-v1.0.txt L306-L324；paper/rouhani-2023-mx.txt L77-L100
- #### E2M1 元素：块内的可表示值｜每个权重存成 4-bit 浮点元素 E2M1 的编码、能表示哪些值、按最近值舍入（超范围截断、低于最小次正规归零）；数值表折叠｜300｜standard/ocp-mx-spec-v1.0.txt L423-L452；paper/rouhani-2023-mx.txt L127-L133
- #### 共享尺度与反量化｜32 个元素共用的 E8M0 尺度是什么、为什么由它承担动态范围，反量化公式，每权重分摊位宽；四元素块算例折叠｜300｜standard/ocp-mx-spec-v1.0.txt L492-L506；paper/rouhani-2023-mx.txt L77-L100；standard/ocp-mx-spec-v1.0.txt L266-L296
- #### 块尺度怎么定｜块的尺度取成块内最大幅值向下的 2 的幂、再除以元素格式能表示的最大 2 的幂，说明按块共享为什么优于每张量一个尺度；换算步骤折叠｜300｜paper/rouhani-2023-mx.txt L136-L160；standard/ocp-mx-spec-v1.0.txt L548-L570；paper/rouhani-2023-mx.txt L42-L48
- ### 量化为什么让权重失真｜量化把权重换成最近的可表示值，与原来不同；训练后量化时模型没见过这种偏差｜300｜paper/jacob-2018-qat.txt L254-L286；standard/ocp-mx-spec-v1.0.txt L423-L452
- ### QAT：训练时把量化算进去｜前向只量化点积输入、权重另存高精度主副本，先量化再反量化算损失，反向用直通估计器（舍入不可导、反向作恒等）传梯度；单权重算例与代码折叠｜500｜paper/jacob-2018-qat.txt L254-L315；paper/rouhani-2023-mx.txt L165-L179；paper/bengio-2013-ste.txt L400-L430
- ### 把 QAT 贯穿整个后训练｜监督微调和强化学习两个阶段都做 QAT；强化学习里边生成边训练，生成与训练方案不一致会造成训练与推理不一致，共享同一方案消除｜300｜paper/k3-tech-report.txt L923-L928；paper/k3-tech-report.txt L815-L823
- ### 量化哪些组件｜只把占显存大头的专家权重量化到 MXFP4，非专家组件（注意力等）保持高精度；配套的激活格式与配置折叠｜300｜paper/k3-tech-report.txt L923-L928；repo/kimi-k3-hf/config.json L202-L240
- ### 适用条件与边界｜量化省显存、仍有小的精度损失；误解：QAT 能把精度恢复到量化前，它只是让模型适应这个损失｜260｜paper/rouhani-2023-mx.txt L398-L410；paper/jacob-2018-qat.txt L254-L286；paper/k3-tech-report.txt L923-L928

## 短页

- ### 一种 4-bit 浮点格式加训练时的模拟｜是什么：MXFP4 把 32 个权重一组共享一个 8-bit 幂次尺度、每个权重存 4-bit 浮点元素；QAT 是让模型在训练中适应这种格式的方法｜300｜standard/ocp-mx-spec-v1.0.txt L306-L324；paper/jacob-2018-qat.txt L254-L286
- ### 显存能省下去，精度会掉下来｜为什么需要：MoE 用多个并行专家子网络替代前馈层，权重占显存大头、压到 4-bit 大幅省显存；训练后量化的模型没见过舍入误差，QAT 让它在训练时见到它｜300｜paper/k3-tech-report.txt L923-L928；paper/jacob-2018-qat.txt L254-L286
- ### 前向量化、反向直通、全程同方案｜怎么做：前向先量化再反量化得到近似权重、用它算损失；反向用直通估计器（把舍入在反向当作恒等函数、梯度原样穿过）把梯度传回原权重；训练全程保持与推理一致的量化方案｜300｜paper/jacob-2018-qat.txt L254-L286；paper/bengio-2013-ste.txt L400-L430；paper/k3-tech-report.txt L923-L928
