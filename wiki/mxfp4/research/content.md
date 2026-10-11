# MXFP4

- 描述：MXFP4 是一种 4-bit 浮点数据格式，把连续若干个权重分为一块、整块共享一个 8-bit 幂次尺度，每个数存成 4-bit 浮点元素。
- 摘要：权重从 16 位压到 4 位能省显存，但低位宽元素能覆盖的数值范围太窄。MXFP4 让每 $32$ 个数共享一个 8-bit 幂次尺度、每个数存 4-bit 浮点元素 E2M1，块内各数只需表示相对大小。
- 主题：内存与缓存, 训练与优化
- 标签：量化

## 资料

- Bita Darvish Rouhani 等. Microscaling Data Formats for Deep Learning. arXiv:2310.10537v3, 2023. [https://arxiv.org/abs/2310.10537](https://arxiv.org/abs/2310.10537)
- Open Compute Project. OCP Microscaling Formats (MX) Specification v1.0. 2023. [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- Kimi Team. Kimi K3: Open Frontier Intelligence. 技术报告, 2026. [https://github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- Moonshot AI. Kimi-K3 模型配置. 2026. [https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)

## 短页

### 一种 4-bit 浮点格式

MXFP4 是一种 4-bit 浮点格式。它把模型的权重（模型里通过训练学到的数值参数）每连续 $k$ 个数分成一块，一块里的数共享一个尺度；尺度是一个 8 位无符号指数（偏置 127）、取值是 2 的整数次幂（可小于 1）。每个数本身用 4 位存，这 4 位是一个 4 位浮点元素，能表示的值只有几档，最大是 6。读一个权重时，把它的元素解出的值与这组的共享尺度相乘，就还原出近似值。
依据：standard/ocp-mx-spec-v1.0.txt L306-L324；standard/ocp-mx-spec-v1.0.txt L443-L452；standard/ocp-mx-spec-v1.0.txt L492-L506；standard/ocp-mx-spec-v1.0.txt L289-L294；paper/rouhani-2023-mx.txt L77-L100

### 位宽压下去，误差从哪里来

部署大模型时，权重占用显存（GPU 上的内存）；把每个权重从 16 位压到 4 位能降低这部分显存。但 4 位元素能表示的正值只有 0.5、1、1.5、2、3、4、6 这几个，原来精确的权重除以块尺度后只能换成最近的那个元素值，还原出来的值与原来不同，这就是误差。MXFP4 用一个块共享的尺度来减小这种误差：块内数值的大小差异由这个尺度承担，块里的元素只需要表示各自的相对大小。
依据：repo/kimi-k3-hf/config.json L11；paper/rouhani-2023-mx.txt L20-L30；standard/ocp-mx-spec-v1.0.txt L423-L452；paper/rouhani-2023-mx.txt L42-L48

### 定尺度、编码

把一个块里的数编码成 MXFP4，分两步。第一步定尺度：取块内最大幅值（绝对值）向下的那个 2 的整数次幂，再除以元素格式能表示的最大 2 的幂，得到这一块的共享尺度；元素的最大值是 6，其中含的最大 2 的幂是 4，所以这个除数是 4。第二步编码：块里每个数除以这个尺度，结果按最近的值取一个 4 位元素编码；超过元素能表示范围的，截断到最大幅值并保留符号。
依据：standard/ocp-mx-spec-v1.0.txt L548-L570；paper/rouhani-2023-mx.txt L136-L160；standard/ocp-mx-spec-v1.0.txt L423-L452

## 长页

部署一个大模型时，权重（模型里通过训练学到的数值参数）占用显存（GPU 上的内存）；把每个权重从 16 位浮点压到 4 位能降低这部分显存。但 4 位能表示的值有限，直接用它存权重会损失精度。MXFP4 是一种 4-bit 浮点格式：它把连续 $k$ 个权重分成一个块，整块共享一个尺度（一个取值是 2 的整数次幂的数），块里的每个权重只存一个 4 位浮点元素；读一个权重时，把元素解出的值与这组的共享尺度相乘。
依据：repo/kimi-k3-hf/config.json L11；paper/rouhani-2023-mx.txt L20-L30；standard/ocp-mx-spec-v1.0.txt L266-L296；standard/ocp-mx-spec-v1.0.txt L306-L324；standard/ocp-mx-spec-v1.0.txt L492-L506；paper/rouhani-2023-mx.txt L42-L48

### E2M1 元素：块内一个数怎么存

每个权重存成一个 4 位浮点元素，这个元素叫 E2M1：1 位符号、2 位指数、1 位尾数，指数偏置为 1。它和常见的浮点数一样用「符号乘上一个二进制小数」表示数值，只是位数很少。正数能表示的就只有有限几个值：0.5 之外，还有 1、1.5、2、3、4、6，最大幅值是 6。把一个权重除以块尺度后得到的商放进 E2M1 时，按最近的值舍入；如果它超过了可表示的范围，就截断到最大幅值并保留符号；如果它舍入后小于最小可表示正值 0.5，就归零。
依据：standard/ocp-mx-spec-v1.0.txt L423-L452

::: E2M1 正数能表示的值与编码
E2M1 的编码写成一个符号位加「指数两位 尾数一位」，负数同幅值取负。
依据：standard/ocp-mx-spec-v1.0.txt L443-L452

| 编码 | 数值 |
|---|---|
| 0 00 0 | 0 |
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

### 共享尺度与反量化

一个权重只用 4 位元素表示不够：元素能表示的数值范围有限。MXFP4 的做法是把连续的 32 个权重分成一个块，整块共享一个尺度。尺度叫 E8M0：8 位无符号指数，偏置 127，取值是 2 的整数次幂，能覆盖 $2^{-127}$ 到 $2^{127}$（最大的编码 255 保留给 NaN，不参与取值）。反量化时把元素解出的值乘上这个共享尺度：第 $i$ 个权重的近似值 $\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$，其中 $s_b$ 是块的共享尺度，$q_i$ 是该权重的 E2M1 编码，$\mathrm{FP4}(q_i)$ 是把编码解出的数值。一个块存 $k$ 个元素，每个 $d$ 位，加上 $w$ 位的尺度，共 $w + kd$ 位；分摊到每个权重是 $d + w/k$ 位。
依据：standard/ocp-mx-spec-v1.0.txt L266-L296；standard/ocp-mx-spec-v1.0.txt L306-L324；standard/ocp-mx-spec-v1.0.txt L492-L506；standard/ocp-mx-spec-v1.0.txt L443-L452；standard/ocp-mx-spec-v1.0.txt L289-L294；paper/rouhani-2023-mx.txt L42-L48

::: 每权重分摊的存储位宽
MXFP4 的块大小 32、元素 4 位、尺度 8 位：一个块存 32 个元素，每个 4 位，再加一个 8 位的尺度，共 $32 \times 4 + 8 = 136$ 位，分摊到每个权重是 $136 / 32 = 4.25$ 位；相比 16 位权重是约 $16 / 4.25 \approx 3.76$ 倍。
依据：自行构造
:::

### 块尺度怎么定

块的尺度取成块内最大幅值向下的那个 2 的整数次幂，再除以元素格式能表示的最大 2 的幂。对 E2M1，最大正规值是 6，其中含的最大 2 的幂是 4，所以这个除数是 4。这样定出尺度后，块内最大幅值的权重除以尺度后落到元素可表示区间的上端，超过元素最大值 6 的会被截断到 6；其余权重按块内相对大小落进各个档位。尺度按块共享，而不是整份权重共用一个，是因为低位宽元素能覆盖的数值范围有限，整份权重共用一个尺度无法覆盖其中相差很大的数值。
依据：standard/ocp-mx-spec-v1.0.txt L548-L570；paper/rouhani-2023-mx.txt L136-L160；standard/ocp-mx-spec-v1.0.txt L443-L452；paper/rouhani-2023-mx.txt L42-L48

::: 从一个块定出尺度并换算
设一个 4 元素的教学块（块大小缩到 4 便于手算）为 $[0.5, -0.25, 1.0, 0.5]$。
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

[量化](wiki/quantization-basics)把每个权重换成块共享尺度与 E2M1 可表示值相乘得到的值，这个替换使权重与它原来精确的值有了偏差。偏差的大小取决于该值离最近的可表示档位有多远：恰好落在档位上就没有偏差，落在两个档位之间就按最近的那一档取值，差值就是偏差。
依据：standard/ocp-mx-spec-v1.0.txt L266-L296；standard/ocp-mx-spec-v1.0.txt L289-L294；standard/ocp-mx-spec-v1.0.txt L423-L452

### 适用条件与边界

MXFP4 的块大小固定为 32。用低位宽表示权重会带来精度损失，训练时把量化算进去可以让模型适应这个损失（见 [QAT](wiki/qat)），但不把精度补回量化前。
依据：standard/ocp-mx-spec-v1.0.txt L306-L324；paper/k3-tech-report.txt L922-L928；paper/rouhani-2023-mx.txt L398-L410
