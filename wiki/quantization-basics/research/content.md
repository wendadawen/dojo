# 量化基础

- 描述：量化基础是把模型的浮点权重与激活（可学参数与逐层流动的中间结果）换成低精度整数网格上的值、以少量误差换取存储、访存与计算下降的方法。
- 摘要：把权重和激活从浮点换成 8 位或 4 位整数，误差来自网格太粗、舍入和裁剪；缩放因子的取法、粒度和是否重训决定误差能减到多小。
- 主题：训练与优化, 推理系统
- 标签：量化

## 资料

- Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer. A Survey of Quantization Methods for Efficient Neural Network Inference. arXiv:2103.13630, 2021. [https://arxiv.org/abs/2103.13630](https://arxiv.org/abs/2103.13630)
- Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)

## 短页

### 低精度网格上的最近值

量化把张量（多维数组）里的浮点值换成位数更少的整数值。模型的权重（可学的参数）和激活（计算中逐层产生的中间结果）都存成张量。位数少，能表示的不同值就少，它们等间距地排在一条网格上；每个浮点值被舍到网格上最近的格点，还原时用同一个映射换回实数，得到原来的近似值。
依据：paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations

### 位宽、存储与吞吐

每个权重和激活占几个比特由位宽决定：16 位浮点每个数占 16 比特，换到 8 位整数每个数占 8 比特，换到 4 位占 4 比特。占的比特少了，把参数存下来和读进来的开销随之下降。专用硬件上低精度乘加的吞吐更高、能耗更低。代价是精度：能表示的级变少，同一个数换过去会偏离原来的值。
依据：paper/gholami2021-src/_s0_chapter.tex Introduction；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations；paper/gholami2021-src/_s0_chapter.tex Simulated and Integer-only Quantization

### 怎么把浮点换成整数网格

先定两个参数：缩放因子是相邻两个整数值之间隔的实数距离，零点是对应实数 0 的那个整数值，有了它实数 0 才能精确表示。取缩放因子有两种方式：对称，把整数范围取成关于 0 对称，缩放因子由张量里最大的绝对值定出；非对称，用观测到的最小值和最大值一起定出范围，缩放因子是范围跨度除以（整数格点的个数减一）。定好之后，每个实数除以缩放因子、加上零点、舍到最近的整数，超出整数范围的值裁到端点。粒度决定一个缩放因子覆盖多少个数：整张张量共用一个最粗，把若干个数分成一组、每组一个更细；粒度越细，一个缩放因子覆盖的范围越小、误差越小，要存和要算的缩放因子也越多。
依据：paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization Granularity

## 长页

模型的权重是可学的参数，激活是前向计算里逐层流动的中间结果，前向计算指用当前权重从输入逐层算到输出，两者都存成张量，也就是多维数组。量化把张量里的浮点值换成位数更少的整数值：位数少，能表示的不同值就少，它们等间距地排在一条网格上，每个值被舍到网格上最近的格点，还原时用同一个映射换回实数。这样每个权重和激活占的比特数下降，把参数存下来、读进来和算它们的开销随之下降，代价是每个数会偏离原来的值一点。
依据：paper/jacob2018-src/main.tex Quantization scheme；paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations；paper/gholami2021-src/_s0_chapter.tex Introduction

### 量化要解决什么

模型规模的增长把权重和激活的存储、延迟和能耗推到部署瓶颈。参数以浮点存储时每个数占的比特多：32 位浮点占 32 比特，16 位占 16 比特。换到低位宽整数能直接减少占用的比特，换到 8 位整数每个数占 8 比特，换到 4 位占 4 比特。从 32 位浮点换到低位宽整数，存储和延迟最好情况下能降到 1/16，实际常做到 1/4 到 1/8。更低的精度在专用硬件上还有吞吐和能耗上的好处。代价是精度：位宽越低，每个数偏离原来的值越多。
依据：paper/gholami2021-src/_s0_chapter.tex L6；paper/gholami2021-src/_s0_chapter.tex Introduction；paper/gholami2021-src/_s0_chapter.tex Problem Setup and Notations；paper/gholami2021-src/_s0_chapter.tex Simulated and Integer-only Quantization

### 均匀量化：缩放因子与整数网格

均匀量化是量化里最常用的一种：网格上相邻两个整数值等间距。它把实数 $r$ 用两个参数映射到整数 $q$：缩放因子 $S$，相邻格点之间的距离；零点 $Z$，实数 $0$ 对应的那个整数值。网格等间距的好处是能只用整数乘加直接实现；网格不等间距的量化叫非均匀量化，同样位宽下可能更贴合数据的分布，但不容易在通用硬件上高效实现。
依据：paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Non-Uniform Quantization；paper/jacob2018-src/main.tex Quantization scheme

#### 仿射映射与反量化

量化把实数映射到整数，反量化把整数还原成实数，两者是一对仿射关系

$$r = S(q - Z),$$

$S$ 决定网格的步长，$Z$ 让实数 $0$ 精确对应到整数 $Z$。
依据：paper/jacob2018-src/main.tex Quantization scheme

从实数到整数是反过来的：$q = \mathrm{round}(r/S) + Z$，再裁到整数网格的两个端点之间。$\mathrm{round}(\cdot)$ 表示舍入到最近的整数。缩放因子由这个张量里观测到的取值范围 $[\alpha, \beta]$ 定出，$S = (\beta - \alpha)/(q_{\max} - q_{\min})$，其中 $q_{\min}$、$q_{\max}$ 是网格的两个端点，$\alpha$、$\beta$ 是观测到的最小值和最大值；零点 $Z$ 取使 $\alpha$ 映射到 $q_{\min}$ 的那个整数，$Z = q_{\min} - \mathrm{round}(\alpha/S)$。
依据：paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Uniform Quantization

范围在什么时候定有两种做法：静态量化在推理（用训练好的模型算输出）前用一批样本跑一遍、统计出范围就固定下来，之后不再变；动态量化每来一个输入现算一次范围。动态往往更准，因为每个输入都有自己的范围，但要实时统计，开销更大。
依据：paper/gholami2021-src/_s0_chapter.tex Range Calibration Algorithms: Static vs Dynamic Quantization

::: 非对称量化的一个算例
取 $r = [-1.0, 0.6, 2.4, 6.0]$，用 3 位非对称量化，网格端点 $q_{\min} = 0$、$q_{\max} = 7$。
依据：自行构造

1. 观测范围 $\alpha = -1$、$\beta = 6$，$S = (6 - (-1))/(7 - 0) = 1$。
2. 零点 $Z = q_{\min} - \mathrm{round}(\alpha/S) = 0 - \mathrm{round}(-1/1) = 0 - (-1) = 1$。
3. 每个数算 $r/S + Z$，得 $[0, 1.6, 3.4, 7]$，舍入到最近整数得 $q = [0, 2, 3, 7]$。
4. 反量化 $\hat{r} = S(q - Z)$，得 $[-1, 1, 2, 6]$。
5. 误差 $\hat{r} - r = [0, 0.4, -0.4, 0]$，都不超过半个格距。
依据：自行构造
:::

#### 对称量化

对称量化把零点固定为 0，网格关于 0 对称。整数范围有两种取法：一种只用到对称的范围 $[-2^{b-1}+1, 2^{b-1}-1]$，$b$ 位时 8 位取 $[-127, 127]$、4 位取 $[-7, 7]$，缩放因子 $S = \max(|r|)/(2^{b-1}-1)$；另一种用满整数范围 $[-2^{b-1}, 2^{b-1}-1]$，8 位即 $[-128, 127]$，缩放因子 $S = 2\max(|r|)/(2^b - 1)$。用满范围的那一种更准，因为它多用一个格点。取 $[-127, 127]$ 的做法避开了 $-128$，有实现上的好处：两个数相乘时就不会出现 $(-128)\times(-128)$，乘积的绝对值始终小于 $2^{14}$，于是可以先把两个乘积累加在一个 16 位中间累加器里，再往 32 位累加器放。
依据：paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/appendix.tex ARM NEON details

反量化只剩一次乘法 $\hat{r} = Sq$。权重常用对称量化，一个原因是反量化没有减零点这一步：零点固定在 0，推理时省掉它带来的计算开销，实现也更直接。分布在 0 的一侧更重时，比如经过一种输出非负的激活函数（把负数截成 0、只留非负输出的那种）之后的激活，取值都落在 $[0, \max]$，对称的负数那一半格点用不上，这时非对称的网格更贴合。
依据：paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/main.tex Learning quantization ranges

### 量化误差从哪来

量化把每个实数换成网格上最近的格点，换成别的值就产生误差。网格太粗是第一处：位宽低，能表示的格点就少，相邻格点之间的距离大，每个数被舍到的位置离它更远。舍入是第二处：不在格点上的数只能舍到最近的格点，误差来自这一步。裁剪是第三处：超出观测范围的值被裁到范围端点，误差等于它超出端点的部分。
依据：paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/main.tex Training with simulated quantization

离群值会把误差放大。整张张量共用一个缩放因子时，张量里那个远大于其余元素的值会把范围拉大、$S$ 随之变大，其余元素除以 $S$ 之后都集中在 0 附近，只能落到少数几个格点上，分辨率被压低、误差变大。
依据：paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/jacob2018-src/main.tex Training with simulated quantization

::: 离群值把正常值限制到少数几个格点
没有离群值时，$r = [1.5, -2.5, 0.6, 3.5]$，4 位对称量化，网格取 $-7$ 到 $7$。
依据：自行构造

1. $S = \max(|r|)/7 = 3.5/7 = 0.5$。
2. $r/S = [3, -5, 1.2, 7]$，舍入得 $q = [3, -5, 1, 7]$。
3. 反量化 $\hat{r} = Sq = [1.5, -2.5, 0.5, 3.5]$，误差 $[0, 0, -0.1, 0]$，都不超过半个格距 $S/2 = 0.25$。
4. 加入一个离群值，$r = [1.5, -2.5, 0.6, 3.5, 14.0]$，仍用 4 位对称量化，最大绝对值变成 $14$。
5. $S = 14/7 = 2$，$r/S = [0.75, -1.25, 0.3, 1.75, 7]$，舍入得 $q = [1, -1, 0, 2, 7]$。
6. 反量化 $\hat{r} = [2, -2, 0, 4, 14]$，误差 $[0.5, 0.5, -0.6, 0.5, 0]$：前四个正常值的误差从不超过 $0.25$ 涨到 $0.5$ 上下。
依据：自行构造
:::

### 用粒度把误差限制在局部

一个缩放因子覆盖张量里的一片数，这一片的取值范围决定网格覆盖的范围与疏密。粒度就是把张量切成多大的片、每片各用一个缩放因子。张量按层组织：一层里做一次矩阵乘或一次卷积，这一步里有若干路并行的输出，每一路对应一组权重，这样的一路叫一个通道。
依据：paper/gholami2021-src/_s0_chapter.tex Quantization Granularity

整层共用一个缩放因子时缩放因子数量最少，但层里某个通道的取值范围远大于同层其余通道时，会把这层的网格拉宽，其余通道的分辨率被压低。按组是把若干个通道合成一组、每组一个缩放因子，在分辨率和开销之间折中。按通道是每个通道一个，分辨率比整层共用一个更好，代价是缩放因子的数量增到通道数那么多。再细一档是子通道，把一个通道内部任意分组，每组一个，分辨率最好，但要处理的缩放因子最多，开销也最大。
依据：paper/gholami2021-src/_s0_chapter.tex Quantization Granularity；paper/gholami2021-src/_s0_chapter.tex Sub-channelwise Quantization

::: 分成两个通道后离群值只影响自己那一档
把 $[1.5, -2.5, 0.6, 3.5, 14.0]$ 看成两个通道：通道 A 是 $[1.5, -2.5, 0.6, 3.5]$，通道 B 是 $[14.0]$。
依据：自行构造

1. 整张张量共用一个缩放因子时 $S = 2$，通道 A 的误差是 $[0.5, 0.5, -0.6, 0.5]$。
2. 改成按通道各用一个缩放因子：通道 A 的 $S = 3.5/7 = 0.5$，误差回到 $[0, 0, -0.1, 0]$；通道 B 的 $S = 14/7 = 2$，误差为 $0$。
3. 同一个离群值，只影响它所在通道 B 的网格，通道 A 的分辨率不再被它压低。
依据：自行构造
:::

### 训练后量化与量化感知训练

量化在训练完成后做：把已经训练好的浮点权重换成低位宽整数，再拿去部署。这条路叫训练后量化（PTQ，post-training quantization），它不需要重新训练，换完就能用，低位宽时精度下降大。
依据：paper/gholami2021-src/_s0_chapter.tex Post-Training Quantization

另一条路是量化感知训练（QAT，quantization-aware training）。训练本来分两步：前向用当前的权重算出预测和损失（损失是一个衡量预测和目标差多少的数），反向从损失出发算出梯度，再按梯度更新权重。QAT 在训练的前向里就模拟量化：把权重按前面的方法量化再还原成近似权重，用这个近似权重算前向和损失；每次梯度更新之后，再把权重投影回去。模型因此在训练里见到量化带来的偏差。
依据：paper/jacob2018-src/main.tex Training with simulated quantization；paper/gholami2021-src/_s0_chapter.tex Quantization-Aware Training；paper/gholami2021-src/_s0_chapter.tex Fine-tuning Methods

反向时，量化里的舍入不可导，照常算梯度这一段的梯度几乎处处是 0，训练传不过去。QAT 用一个直通估计器（STE，straight-through estimator）处理这一步：反向传播经过量化时，把梯度按原样传给量化前的权重，相当于把量化当成恒等映射。这样梯度能传过去，权重得以更新；代价是要重新训练。
依据：paper/gholami2021-src/_s0_chapter.tex Quantization-Aware Training

两条路的区别在是否重训：PTQ 不重训，快，可以在数据受限或无标注时用，低位宽时精度损失大；QAT 要重训，训练里见过量化偏差，低位宽时的精度损失小。
依据：paper/gholami2021-src/_s0_chapter.tex Post-Training Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization-Aware Training

### 适用条件与边界

均匀仿射量化对正的缩放因子 $S$ 成立。对称量化适合大致关于 0 对称的分布。观测范围外的值被裁到端点，误差等于超出量，所以范围取得准不准直接决定这部分误差。均匀量化之所以是通用做法，是因为它简单，而且能高效地映射到硬件。
依据：paper/gholami2021-src/_s0_chapter.tex Uniform Quantization；paper/gholami2021-src/_s0_chapter.tex Symmetric and Asymmetric Quantization；paper/gholami2021-src/_s0_chapter.tex Non-Uniform Quantization；paper/jacob2018-src/main.tex Quantization scheme

精度和开销是一对：位宽越低，网格越粗、误差越大。同一份误差可以用更细的粒度或 QAT 换回来，但更细的粒度要多存、多算缩放因子，粒度细到子通道时开销尤其大，QAT 则要重新训练。它带来的速度收益依硬件而定，受片上内存、带宽和缓存层次等因素影响。
依据：paper/gholami2021-src/_s0_chapter.tex Quantization Granularity；paper/gholami2021-src/_s0_chapter.tex Hardware Aware Quantization；paper/gholami2021-src/_s0_chapter.tex Quantization-Aware Training；paper/gholami2021-src/_s0_chapter.tex Mixed-Precision Quantization
