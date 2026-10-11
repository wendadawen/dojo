# QAT

- 描述：量化感知训练（QAT）是在训练的前向里按低位宽格式对权重先量化再反量化、用近似权重算损失，反向用直通估计器把梯度传回原始高精度权重，让模型适应量化误差的训练方法。
- 摘要：把权重量化留到训练之后，模型没见过舍入误差。QAT 在训练前向插入「先量化再反量化」，反向用直通估计器（把量化当作恒等函数）传梯度，只更新另存的高精度主副本。
- 主题：训练与优化
- 标签：量化

## 资料

- Benoit Jacob 等. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. arXiv:1712.05877. [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- Yoshua Bengio, Nicholas Léonard, Aaron Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432, 2013. [https://arxiv.org/abs/1308.3432](https://arxiv.org/abs/1308.3432)
- Bita Darvish Rouhani 等. Microscaling Data Formats for Deep Learning. arXiv:2310.10537v3, 2023. [https://arxiv.org/abs/2310.10537](https://arxiv.org/abs/2310.10537)
- Kimi Team. Kimi K3: Open Frontier Intelligence. 技术报告, 2026. [https://github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

## 短页

### 训练时就把量化算进去

QAT 是一种训练方法。它把量化（见[量化基础](wiki/quantization-basics)）放进训练的每一步里：训练时按一种低位宽格式（把数值用很少的位数表示）把权重（模型里通过训练学到的数值参数）换成低位宽的值、再换回高精度（即原始精度）参与计算，让模型在训练里就见到量化带来的误差。
依据：paper/jacob-2018-qat.txt L254-L286

### 模型没见过舍入误差

把量化留到训练完成后做，模型训练全程用高精度权重。换成低位宽时，权重只能取最近的可表示值，与原来不同；模型从没见过这个误差，这种误差小模型上造成的精度下降比大模型更多。
依据：paper/jacob-2018-qat.txt L254-L286；paper/rouhani-2023-mx.txt L165-L179

### 前向量化、反向直通

训练每走一步，先把权重编码成低位宽再还原，得到近似权重，用它算损失（衡量输出与目标差距）；算梯度时，量化这一步没有可用的导数，用直通估计器（在算梯度这一步把量化当作恒等函数）把梯度原样传回；权重另存一份高精度副本，更新只改这份副本。
依据：paper/jacob-2018-qat.txt L254-L286；paper/bengio-2013-ste.txt L400-L410

## 长页

把模型的权重（训练学到的数值参数）[量化](wiki/quantization-basics)（把数值换成低位宽表示的值）留到训练完成之后再做，叫训练后量化（post-training quantization）：训练全程用高精度（即全精度，如 16 位）权重，训练完成后才把权重换成低位宽，模型从没见过换值带来的误差，小模型上精度下降比大模型更多。量化感知训练（quantization-aware training，QAT）把量化提前到训练里：训练时按目标低位宽格式（把数值用很少的位数表示的格式）把权重先量化再反量化（换回高精度），用近似权重算损失（衡量输出与目标差距、被训练优化的那个量）；反向（从损失算出梯度）用直通估计器（把量化这一步在反向当作恒等函数）把梯度传回原始高精度权重，让模型在训练中就适应这个误差。
依据：paper/jacob-2018-qat.txt L254-L286；paper/bengio-2013-ste.txt L400-L410

### 前向模拟量化

QAT 在训练的前向（模型由输入算出输出的那一步）里把量化算进去。在被量化的模块里，对做点积的运算（矩阵乘这类），两个输入都先按各自的目标低位宽格式编码再反量化，得到近似值再算；归一化（把一批数值按均值方差缩放）、Softmax（把一组数变成和为 1 的占比）、激活函数（逐元素的非线性）、残差相加（把输入直接加回输出）这类非点积运算仍在高精度里做。权重另外保留一份高精度的主副本，每一步训练都更新它。把一个权重先按格式编码、再解回数值，得到的是它的近似权重 $\hat{w}$。前向用 $\hat{w}$ 而不是原始的 $w$ 参与计算，损失因此建立在这套近似权重上。
依据：paper/jacob-2018-qat.txt L254-L286；paper/rouhani-2023-mx.txt L165-L179

### 反向用直通估计器

反向传播（从损失出发逐层算出每个权重的梯度、据此更新权重）时，量化这一步把 $w$ 换成离散的 $\hat{w}$，是阶梯状、除跳变点外导数处处为零的函数，梯度无法穿过。直通估计器专门处理这种情形：反向时把这一步当作恒等函数，令梯度原样穿过，损失 $L$ 对 $w$ 的梯度 $\partial L/\partial w$ 直接取上游（反向传播中更靠近损失的那一侧）传给 $\hat{w}$ 的梯度，更新仍然作用在原始高精度权重 $w$ 上。它是偏的估计器（算出的梯度偏离真实值），但在单层上方向与真实梯度一致。
依据：paper/bengio-2013-ste.txt L167-L168；paper/bengio-2013-ste.txt L400-L415；paper/jacob-2018-qat.txt L306-L311；paper/jacob-2018-qat.txt L275-L286

::: 单权重走一遍前向量化、反向直通
以 [MXFP4](wiki/mxfp4) 的块为例（块是共用同一尺度的若干个权重），取块 $[0.8, 0.3, -0.1, 0.55]$，块大小缩到 4 便于手算，看其中一个权重 $w = 0.8$ 前向量化、反向梯度怎么走。
依据：自行构造

1. 块内最大幅值 $0.8$，向下取 2 的幂得 $0.5$，除以元素能表示的最大 2 的幂 $4$ 得共享尺度 $s_b = 0.125$。
2. $w$ 除以尺度得 $0.8 / 0.125 = 6.4$，按最近值取元素得 $6.0$。
3. 反量化得近似权重 $\hat{w} = 0.125 \times 6.0 = 0.75$，与原始 $w = 0.8$ 相差 $0.05$。
4. 前向用 $\hat{w} = 0.75$ 算损失。反向时直通估计器让量化这一步的导数取 $1$，上游梯度 $1$ 原样传回原始权重。
5. 优化器（按梯度更新权重的那一步）用这个梯度更新 $w$，下一轮前向会重新量化更新后的 $w$。
依据：自行构造

```python
import math

# E2M1 是 MXFP4 的 4 位浮点元素（1 位符号、2 位指数、1 位尾数），下列为它的正数可表示值
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1(v):
    sign = -1 if v < 0 else 1
    a = abs(v)
    return sign * min(E2M1, key=lambda r: (abs(r - a), r))


def scale_of(block):
    # 块内最大幅值向下的 2 的幂，除以 E2M1 的最大 2 的幂 4
    return 2.0 ** (math.floor(math.log2(max(abs(v) for v in block))) - 2)


block = [0.8, 0.3, -0.1, 0.55]
s_b = scale_of(block)
print("共享尺度", s_b)

for w in block:
    print("权重", w, "量化后", s_b * e2m1(w / s_b))

print("上游梯度 1 经直通估计器传回的梯度", 1.0)
```

```text
共享尺度 0.125
权重 0.8 量化后 0.75
权重 0.3 量化后 0.25
权重 -0.1 量化后 -0.125
权重 0.55 量化后 0.5
上游梯度 1 经直通估计器传回的梯度 1.0
```
:::

### 贯穿整个后训练

QAT 从[监督微调](wiki/sft)（用带标注的数据继续训练模型）阶段起就做，贯穿整个后训练（预训练之后的所有训练阶段），监督微调和强化学习两个阶段都做。强化学习（让模型按奖励试错改进）阶段里，模型一边自己生成数据（这一步叫 rollout）、一边拿这些数据更新自己；如果生成时用一套量化方案、训练更新时用另一套，这就是训练与推理不匹配（推理指用训练好的模型算输出）。让生成与训练共享同一套量化方案，两边看到的是同一种权重，这个不匹配就被消掉。
依据：paper/k3-tech-report.txt L815-L823；paper/k3-tech-report.txt L922-L928

### 量化哪些组件

训练时不是所有组件都压到低位宽。专家权重（混合专家（MoE）层里每个输入只激活少数几个的并行子网络；见[混合专家](wiki/fusedmoe)）占显存（显卡上的内存）的大部分，压到低位宽；非专家组件（注意力这类模块，见[标准注意力](wiki/standard-attention)）保持高精度。激活值（层里传下去的中间数值）另用一种更宽的格式。
依据：paper/k3-tech-report.txt L212；paper/k3-tech-report.txt L428；paper/k3-tech-report.txt L922-L928

### 适用条件与边界

训练后量化的精度对小模型不够时，QAT 让模型在训练里适应量化带来的误差，把精度拉回接近全精度，损失仍比全精度略高一点。它只改训练流程，不改推理时权重的存储格式：训练里的量化只是模拟，部署的仍是按目标低位宽格式编码的权重（例如 [MXFP4](wiki/mxfp4)）。
依据：paper/jacob-2018-qat.txt L256-L261；paper/jacob-2018-qat.txt L557-L559；paper/rouhani-2023-mx.txt L398-L410；paper/k3-tech-report.txt L922-L928
