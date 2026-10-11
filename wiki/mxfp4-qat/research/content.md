# MXFP4 QAT

- 描述：MXFP4 量化感知训练是把模型的权重（训练学到的数值参数）编码成 4-bit 微缩浮点格式 MXFP4，并在训练里模拟这种量化，让模型学会适应由此带来的精度损失的训练方法。
- 摘要：MXFP4 用 32 个权重共享一个 8-bit 幂次尺度、每个权重存 4-bit 浮点元素；量化感知训练（QAT）在训练前向先量化再反量化，反向用直通估计器把梯度传回原权重。
- 主题：训练与优化
- 标签：量化

## 资料

- Bita Darvish Rouhani 等. Microscaling Data Formats for Deep Learning. arXiv:2310.10537v3, 2023. [https://arxiv.org/abs/2310.10537](https://arxiv.org/abs/2310.10537)
- Open Compute Project. OCP Microscaling Formats (MX) Specification v1.0. 2023. [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- Benoit Jacob 等. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. arXiv:1712.05877. [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- Yoshua Bengio, Nicholas Léonard, Aaron Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432, 2013. [https://arxiv.org/abs/1308.3432](https://arxiv.org/abs/1308.3432)
- Kimi Team. Kimi K3: Open Frontier Intelligence. 技术报告, 2026. [https://github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- Moonshot AI. Kimi-K3 模型配置. 2026. [https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)

## 短页

### 一种 4-bit 浮点格式加训练时的模拟

MXFP4 是一种 4-bit 浮点格式。它把权重（模型里通过训练学到的数值参数）每 32 个分成一组，一组里的 32 个数共享一个尺度，尺度是一个 8-bit 的无符号指数、取值是 2 的整数次幂；每个数本身用 4 位存，这 4 位常写成一个 4 位浮点元素。读一个权重时，把它的 4 位元素解出的值与这组的共享尺度相乘，就还原出近似值。量化感知训练（Quantization-Aware Training，QAT）是在训练过程中就让模型适应这种量化的方法：它不把权重量化留到训练之后，而是在训练里就按上面的方式编码权重。
依据：standard/ocp-mx-spec-v1.0.txt L306-L324；standard/ocp-mx-spec-v1.0.txt L495-L506；paper/jacob-2018-qat.txt L254-L286

### 显存下降与精度下降

大模型的权重占显存（GPU 上的内存）的主要部分，把每个权重从 16 位浮点压到 4 位，权重占的显存随之下降。混合专家（MoE）模型每层有许多并行的专家子网络，每个 token（模型处理的最小单位）只用到其中少数几个；没被用到的专家权重也必须常驻，专家权重因此占了参数的大部分，是最值得压缩的部分。但 4 位能表示的值很少，原来精确的权重只能换成最近的那个，产生误差。训练后量化的模型在训练全程用全精度权重，只在训练完成后才换成 4 位，它从没见过这个误差；QAT 让模型在训练时就见到，于是能在这个误差下继续工作。
依据：paper/k3-tech-report.txt L923-L928；paper/k3-tech-report.txt L104；paper/jacob-2018-qat.txt L254-L286

### 前向量化、反向直通、全程同方案

训练的前向（模型由输入算出输出的那一步）里插入一步「先量化再反量化」：把权重按 MXFP4 编码再还原，得到近似权重，用它参与计算和损失（衡量输出与目标差距、被训练优化的那个量）。这样每次前向用的都是模型部署时真正会用到的权重。反向传播（从损失出发逐层算出每个权重的梯度、据此更新权重）时，量化和反量化这一步没有可用的导数，用直通估计器处理：把这一步在反向时当作恒等函数，梯度原样穿过，直接更新原始的高精度权重。训练全程还要让前向用的量化方案与部署时一致，模型适应的是它部署时真正会遇到的偏差。
依据：paper/jacob-2018-qat.txt L254-L286；paper/bengio-2013-ste.txt L400-L430；paper/k3-tech-report.txt L923-L928

## 长页

部署一个大模型时，权重（模型里通过训练学到的数值参数）占用的显存（GPU 上的内存）是主要成本之一。把每个权重从 16 位浮点压到 4 位，权重占的显存随之下降。混合专家（MoE）模型的每一层把前馈网络（层里参数最多的一块）换成许多并行的专家子网络，每个 token（模型处理的最小单位）只激活其中少数几个（这种层的结构见[混合专家](wiki/fusedmoe)）。稀疏激活省的是计算，不省存储：没有被激活的专家权重也必须常驻显存，专家权重因此占了这种模型参数的大部分，成为量化的主要对象。
依据：paper/k3-tech-report.txt L923-L928；paper/k3-tech-report.txt L104；repo/kimi-k3-hf/config.json L11

直接把权重压到 4 位会引入误差：4 位能表示的数很少，原来的权重只能换成最近的那个。把量化留到训练完成之后再做的做法叫训练后量化（post-training quantization）：训练全程用全精度权重，只在训练完成后才换成低位宽，模型从没见过这个误差，小模型上的精度下降比大模型更多。MXFP4 量化感知训练（quantization-aware training，简称 QAT）把量化提前到训练里：权重按 MXFP4 这个 4 位浮点格式编码，训练中用近似权重算损失（衡量输出与目标差距、被训练优化的那个量），再把梯度传回原始权重，让模型在训练中就适应这个误差。
依据：paper/jacob-2018-qat.txt L254-L286；standard/ocp-mx-spec-v1.0.txt L306-L324

### MXFP4：块共享尺度的 4-bit 浮点

#### E2M1 元素：块内的可表示值

每个权重存成一个 4 位浮点元素，这个元素叫 E2M1：1 位符号、2 位指数、1 位尾数，指数偏置为 1。它和常见的浮点数一样用「符号乘上一个二进制小数」表示数值，只是位数很少。正数能表示的就只有有限几个值：0.5 之外，还有 1、1.5、2、3、4、6，最大幅值是 6。把一个权重放进 E2M1 时，按最近的值舍入；如果它超过了可表示的范围，就截断到最大幅值并保留符号；如果它舍入后小于最小可表示正值 0.5，就归零。
依据：standard/ocp-mx-spec-v1.0.txt L423-L452

::: E2M1 正数能表示的值与编码
E2M1 的编码写成一个符号位加「指数两位 尾数一位」，下表列出全部正数取值（负数同幅值取负）。
依据：standard/ocp-mx-spec-v1.0.txt L443-L452

| 编码 | 数值 |
|---|---|
| 0 00 1 | 0.5 |
| 0 01 0 | 1.0 |
| 0 01 1 | 1.5 |
| 0 10 0 | 2.0 |
| 0 10 1 | 3.0 |
| 0 11 0 | 4.0 |
| 0 11 1 | 6.0 |

指数为零、尾数为一的那一档是次正规数，所以 0.5 与 1.0 之间没有别的可表示值。
依据：standard/ocp-mx-spec-v1.0.txt L443-L452
:::

#### 共享尺度与反量化

一个权重只用 4 位元素表示不够——元素能覆盖的数值范围太窄，直接用它存权重，稍有大小差别的数就覆盖不到。MXFP4 的做法是把连续的 32 个权重分成一个块，整块共享一个尺度。尺度叫 E8M0：8 位无符号指数，偏置 127，取值是 2 的整数次幂，能覆盖 $2^{-127}$ 到 $2^{127}$（最大的编码 255 保留给 NaN，不参与取值），动态范围由它承担；块里的元素只需表示各自的相对大小。反量化时把元素解出的值乘上这个共享尺度：第 $i$ 个权重的近似值 $\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$，其中 $s_b$ 是块的共享尺度，$q_i$ 是该权重的 E2M1 编码，$\mathrm{FP4}(q_i)$ 是把编码解出的数值。一个块存 $k$ 个元素，每个 $d$ 位，加上 $e$ 位的尺度，共 $e + kd$ 位；分摊到每个权重是 $d + e/k$ 位。
依据：standard/ocp-mx-spec-v1.0.txt L275-L287；standard/ocp-mx-spec-v1.0.txt L289-L294；standard/ocp-mx-spec-v1.0.txt L492-L506；standard/ocp-mx-spec-v1.0.txt L306-L324；paper/rouhani-2023-mx.txt L42-L48

::: 每权重分摊的存储位宽
MXFP4 的块大小 32、元素 4 位、尺度 8 位：一个块存 32 个元素，每个 4 位，再加一个 8 位的尺度，共 $32 \times 4 + 8 = 136$ 位，分摊到每个权重是 $136 / 32 = 4.25$ 位；相比 16 位权重是约 $16 / 4.25 \approx 3.76$ 倍。
依据：自行构造
:::

#### 块尺度怎么定

块的尺度取成块内最大幅值向下的那个 2 的整数次幂，再除以元素格式能表示的最大 2 的幂。对 E2M1，最大正规值是 6，其中含的最大 2 的幂是 4，所以这个除数是 4。这样定出尺度后，块内最大幅值的权重除以尺度后正好落在元素能表示的最大档附近，其余权重按块内相对大小落进各个档位。尺度按块共享，而不是整个权重数组共用一个，是因为一个数组级的尺度无法同时覆盖数值范围差别很大的多组权重；把尺度缩到小块内，块内数值差异的范围由每块自己的尺度承担，元素就只负责块内的相对差异。
依据：paper/rouhani-2023-mx.txt L136-L160；standard/ocp-mx-spec-v1.0.txt L548-L570；standard/ocp-mx-spec-v1.0.txt L443-L452；paper/rouhani-2023-mx.txt L42-L48

::: 从一个块定出尺度并换算
设一个 4 元素的教学块（块大小缩到 4 便于手算）为 $[0.5, -0.25, 1.0, 0.5]$，按下述步骤换算。
依据：自行构造

1. 块内最大幅值是 $1.0$，向下取 2 的整数次幂得 $1.0$，再除以元素格式的最大 2 的幂 $4$，得到共享尺度 $s_b = 0.25$。
2. 每个元素除以尺度：$0.5 / 0.25 = 2$，$-0.25 / 0.25 = -1$，$1.0 / 0.25 = 4$，$0.5 / 0.25 = 2$。
3. 这四个结果都在 E2M1 的可表示值里，按最近值取元素编码得 $2, -1, 4, 2$。
4. 反量化乘回尺度：得到 $0.5, -0.25, 1.0, 0.5$，与原块完全相同，这一块没有损失。
5. 换一个块 $[0.3, 0.1, 0.45, 0.2]$：最大幅值 $0.45$ 向下取幂得 $0.25$，除以 $4$ 得 $s_b = 0.0625$。各自除以尺度得 $4.8, 1.6, 7.2, 3.2$；按最近值取元素，$4.8$ 取 $4$、$1.6$ 取 $1.5$、$3.2$ 取 $3$，$7.2$ 超过最大档 $6$ 被截断到 $6$。
6. 反量化得 $0.25, 0.09375, 0.375, 0.1875$，与原来的 $0.3, 0.1, 0.45, 0.2$ 都不同，这一块产生了损失。
依据：自行构造
:::

### 量化为什么让权重失真

量化把每个权重换成 E2M1 能表示的最近的那个值，这个替换使权重与它原来精确的值有了偏差。偏差的大小取决于该值离最近的可表示档位有多远：恰好落在档位上就没有偏差，落在两个档位之间就按最近的那一档取值，差值就是偏差。
依据：standard/ocp-mx-spec-v1.0.txt L423-L452

### QAT：训练时把量化算进去

量化感知训练的做法是在训练的前向（模型由输入算出输出的那一步）里就把量化算进去。在被量化的那些模块里，对做点积的运算（矩阵乘这类），两个输入都先按 MX 格式编码再反量化，得到近似值再算；归一化（把一批数值按均值方差缩放）、Softmax（把一组数变成和为 1 的占比）、激活函数（逐元素的非线性）、残差相加（把输入直接加回输出）这类非点积运算仍在高精度里做；权重另外保留一份高精度的主副本，每一步训练都更新它。把一个权重先按格式编码、再解回数值，得到的是它的近似权重 $\hat{w}$。前向用 $\hat{w}$ 而不是原始的 $w$ 参与计算，损失因此建立在这套近似权重上。反向传播（从损失出发逐层算出每个权重的梯度、据此更新权重）时，量化这一步把 $w$ 换成离散的 $\hat{w}$，是阶梯状、导出为零的函数，梯度无法直接穿过。直通估计器专门处理这种情形：反向时把这一步当作恒等函数，令梯度原样穿过，$\partial L/\partial w$ 直接取上游传给 $\hat{w}$ 的梯度，更新仍然作用在原始高精度权重 $w$ 上。
依据：paper/jacob-2018-qat.txt L254-L315；paper/rouhani-2023-mx.txt L165-L179；paper/bengio-2013-ste.txt L400-L430

::: 单权重走一遍前向量化与反向直通
取块 $[0.8, 0.3, -0.1, 0.55]$，看其中一个权重 $w = 0.8$ 的前向量化与反向梯度。块大小缩到 4 便于手算。
依据：自行构造

1. 块内最大幅值 $0.8$，向下取 2 的幂得 $0.5$，除以 $4$ 得共享尺度 $s_b = 0.125$。
2. $w$ 除以尺度得 $0.8 / 0.125 = 6.4$，按最近值取元素得 $6.0$。
3. 反量化得近似权重 $\hat{w} = 0.125 \times 6.0 = 0.75$，与原始 $w = 0.8$ 相差 $0.05$。
4. 前向用 $\hat{w} = 0.75$ 算损失。假设损失对 $\hat{w}$ 的上游梯度是 $1$，直通估计器让这一步的导数取 $1$，于是传给原始权重的梯度 $\partial L/\partial w$ 也是 $1$。
5. 优化器用这个梯度更新 $w$，下一轮前向会重新量化更新后的 $w$。
依据：自行构造

```python
import math

# E2M1 正数可表示值，含 0
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

### 把 QAT 贯穿整个后训练

做法是把量化感知训练贯穿整个后训练（预训练之后的所有训练阶段；预训练指从大规模文本里学通用能力的那个阶段），监督微调（用带标注的数据继续训练）和强化学习（模型边生成、边按奖励信号更新）两个阶段都做，而不是只在最后补一段微调。强化学习阶段多一层要求：模型一边自己生成数据（这一步叫 rollout）、一边拿这些数据更新自己。如果生成时用一套量化方案、训练更新时用另一套，训练拿到的目标来自前一套方案下的分布，参数却按后一套方案更新；这种不一致叫训练与推理不匹配。让 rollout 和训练共享同一套量化方案，两边看到的是同一种权重，这个不一致就被消掉。
依据：paper/k3-tech-report.txt L923-L928；paper/k3-tech-report.txt L815-L823

### 量化哪些组件

不是所有组件都压到 4 位。占参数的大部分、又只被少量 token 激活的专家权重压到 MXFP4；[注意力](wiki/standard-attention)和共享专家（每个 token 都经过、不属于前面按 token 稀疏激活的那批专家）这类每个 token 都要经过的组件保持高精度，激活值（层里传下去的中间数值）另用一种更宽的格式。这样收益主要来自专家权重，又不让高频路径上的数值也承担 4 位的误差。
依据：paper/k3-tech-report.txt L923-L928；paper/k3-tech-report.txt L104；repo/kimi-k3-hf/config.json L202-L240

::: 官方配置里的量化项
发布模型的 config.json 里，量化由 format、group_size、num_bits 等字段给出，ignore 列出保持高精度、不量化的模块。
依据：repo/kimi-k3-hf/config.json L202-L240

| 字段 | 值 |
|---|---|
| format | mxfp4-pack-quantized |
| group_size | 32 |
| num_bits | 4 |
| type | float |
| strategy | group |
| symmetric | true |
| scale_dtype | torch.uint8 |

忽略列表中的模块：self_attn、shared_experts、mlp 的 gate/up/gate_up/down 投影、lm_head、vision_tower、mm_projector。
依据：repo/kimi-k3-hf/config.json L202-L240
:::

### 适用条件与边界

MXFP4 量化感知训练省的是显存：权重仍要为每个 token 做同样的乘加，只是读取的量下降。它让模型适应量化带来的精度损失，但不把精度补回量化前；用 MXFP4 权重配 MXFP6 激活（另一种 6 位的微缩浮点格式）训练的语言模型，损失仍略高于全精度，只是差距很小。它按块共享尺度：块越大，块内数值差异越难兼顾；块越小，每个块那个 8 位尺度分摊到每个权重上的开销越大，块大小是这两者之间的取舍。
依据：paper/rouhani-2023-mx.txt L42-L48；paper/rouhani-2023-mx.txt L398-L410；paper/jacob-2018-qat.txt L254-L286；standard/ocp-mx-spec-v1.0.txt L548-L570；standard/ocp-mx-spec-v1.0.txt L275-L287；standard/ocp-mx-spec-v1.0.txt L306-L321
