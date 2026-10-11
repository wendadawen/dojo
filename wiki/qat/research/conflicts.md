# 冲突

- 位置：长页第 1 章「前向模拟量化」关于激活、残差相加这两类非点积运算是否量化。两处说法：paper/rouhani-2023-mx.txt L173-L174「Vector operations (e.g., layernorm, Softmax, GELU, and residual add) are performed in a scalar floating-point format like Bfloat16 or FP32」把这类非点积运算留在高精度；paper/jacob-2018-qat.txt L290-L294「Activations are quantized at points where they would be during inference, e.g. after the activation function is applied … or after a bypass connection adds or concatenates the outputs of several layers together」把激活函数输出与残差相加处也量化。页面采用：第 1 章限定在「被量化的模块里」，对做点积的运算把两个输入量化、非点积运算仍高精度做（按 rouhani 的 MX 计算流写），不作无条件的「一律」陈述。
