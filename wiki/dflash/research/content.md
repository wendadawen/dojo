# DFlash

- 描述：DFlash 是一种投机解码方法：用一个块扩散草稿器一次前向并行猜出整块草稿 token，并靠目标模型的隐藏特征条件化，再由目标模型一次验证。
- 摘要：把目标模型若干层的隐藏特征拼成上下文特征，注入轻量块扩散草稿器的每一层 KV，一次前向并行猜出整块；验证仍由目标模型完成，输出因此无损。
- 主题：推理系统, 模型结构
- 标签：推理加速

## 资料

- Jian Chen, Yesheng Liang, Zhijian Liu. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv:2602.06036v2, 2026. [https://arxiv.org/abs/2602.06036](https://arxiv.org/abs/2602.06036)
- z-lab. DFlash: official implementation. 2026. [https://github.com/z-lab/dflash](https://github.com/z-lab/dflash)

## 短页

### 条件化的并行草稿器

自回归生成一次前向只产出一个 token（token 是文本切分后的最小生成单位；一次前向指把输入过一遍网络、算出输出的那次计算）。投机解码把一次生成拆成两步：一个小的草稿模型先猜出后面一段 token，再由大的目标模型一次前向并行验证这段猜测，只保留其中正确的那段。DFlash 是一个用块扩散起草的投机解码框架：块扩散指把序列分成块，一次并行填补一块里多个被遮住的位置，而不是一个接一个地猜。它还把目标模型的隐藏特征当条件——目标模型每一层都把每个位置表示成一个向量，DFlash 用这些向量作条件，让草稿贴近目标模型的判断。草稿猜得越准、越长，目标模型一轮验证能接受的 token 越多，整段生成就越快。
依据：paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Inference

### 起草环节限制了加速

投机解码本是为绕开逐 token 生成这一点，但它的草稿模型同样一次前向只猜一个 token，时间又花在了起草上：草稿越长起草越慢，为了不拖慢，草稿模型只能做得很浅（甚至只有一层），层数少的网络又猜不准，接受长度很快饱和，于是加速只有 2 到 3 倍。DFlash 换个做法：块扩散一次并行出整块，起草成本不再随草稿长度线性增长，草稿模型就能做得更深、猜得更准。
依据：paper/src-v2/sections/preliminaries.tex Diffusion Drafting；paper/src-v2/sections/intro.tex Introduction

### 目标特征加块扩散

第一，预填充（prefill，模型把整段提示一次读完、算出第一个 token 的那次前向）时，从目标模型若干层取出隐藏特征，拼起来过一个投影层（把向量做一次线性变换、压到另一维度）压成一段目标上下文特征。第二，把这段特征注入草稿器每一层注意力的 KV 缓存（注意力是让每个位置读取前面位置的层，它把算好的键 K 和值 V 缓存下来），当作条件。第三，草稿器一次前向并行填出整块被遮位置，再交目标模型一次验证。
依据：paper/src-v2/sections/method.tex Inference

## 长页

大语言模型逐 token 生成（token 是文本切分后的最小生成单位）：先算出第一个 token，把它接回输入，再算下一个；每个 token 都要等上一个算完，延迟高，也让 GPU 利用率低。投机解码为绕开这一点，让一个小的草稿模型先猜出后面一段 token，再由大的目标模型一次前向（把输入过一遍网络、算出输出的那次计算）并行验证，只保留猜对的那段；被接受的 token 与目标模型自己生成的结果一致，输出不变。麻烦在于草稿模型自己也是自回归的，猜得越多越慢，能拿到的加速有限。DFlash 换用一个块扩散草稿器来起草：块扩散把序列分成块、一次并行填补一块里多个被遮位置，草稿器一次前向就并行给出整块 token；它还把目标模型的隐藏特征压成目标上下文特征当条件——目标模型每一层把每个位置表示成一个向量，这些向量里含有对后续 token 的提示。
依据：paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Inference

### 串行起草限制了加速

投机解码每一轮分两步：草稿模型先猜出若干 token，目标模型再对这段猜测做一次前向验证。每生成一个 token 的平均延迟，等于一轮起草加验证的总时间，除以这一轮被接受的长度。被接受的长度里包含一个顺带多产出的 token：目标模型在最后一个被接受位置之后，会自己算出下一个 token，这个 token 叫 bonus token，所以接受长度的上限比草稿长度多一。

对自回归草稿模型，起草一轮要串行跑若干次前向，次数等于每轮的草稿 token 数，耗时随之线性增长。为了不拖慢整体，它只能做得很浅，浅网络猜不准，接受长度很快饱和，加速因此受限。块扩散草稿器不同：它一次前向就把整块位置并行填出，一轮起草的耗时几乎不随草稿长度变化。起草开销小了，草稿模型就能做得更深、更准，接受长度随之上升。
依据：paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/preliminaries.tex Diffusion Drafting

::: 平均延迟与加速比的推导
1. 设目标模型自回归生成时每 token 的延迟为 $L_{\text{target}}$，投机解码每 token 的平均延迟为 $L$，则加速比 $\eta=L_{\text{target}}/L$。
2. 一轮的耗时由起草 $T_{\text{draft}}$ 与验证 $T_{\text{verify}}$ 组成，产出 $\tau$ 个 token，故 $L=\dfrac{T_{\text{draft}}+T_{\text{verify}}}{\tau}$；$\tau\in[1,\gamma+1]$ 是每轮期望接受的 token 数，含 bonus token，$\gamma$ 是每轮的草稿 token 数。
3. 加速要么靠更大的 $\tau$，要么靠更小的 $T_{\text{draft}}$。
4. 自回归起草：$T_{\text{draft}}=\gamma\cdot t_{\text{step}}$，$t_{\text{step}}$ 是单次前向的延迟，起草成本随 $\gamma$ 线性增长。
5. 扩散起草：$T_{\text{draft}}=t_{\text{parallel}}$，$t_{\text{parallel}}$ 是整块生成的延迟；规模相近时并行一次远快于串行多次，即 $t_{\text{parallel}}\ll\gamma\cdot t_{\text{step}}$。
依据：paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；paper/src-v2/sections/preliminaries.tex Diffusion Drafting
:::

::: 草稿层数与加速的消融
在 Math500（数学评测集）上比 3、5、8 层草稿器：
| 草稿层数 | 加速 | 接受长度 |
|---|---|---|
| 3 层 | 4.69 倍 | 5.64 |
| 5 层 | 4.71 倍 | 5.99 |
| 8 层 | 4.64 倍 | 6.33 |
层数越多接受长度越高，但起草延迟也越高，两者相抵后综合加速以 5 层最好。
依据：paper/src-v2/sections/exp.tex Number of Draft Layers
:::

### 用目标模型的隐藏特征条件化草稿

直接拿一个没有任何条件的扩散模型当草稿器，起草虽快，但猜不准：在几个数学任务上，它的加速低于有目标特征的草稿器。原因是它拿不到目标模型对当前上下文的理解，只能完全从零预测未来 token。

目标模型自己的隐藏特征里就有这类信息。大模型逐层把每个位置的输入表示成一个向量，这些向量里含有远距离依赖和任务相关的语义，还隐含编码了未来若干 token 的信息。让草稿器直接用上这些向量，比让它自己从头推断更准。

DFlash 在预填充阶段把这些向量取出来。预填充是目标模型把整段提示一次读完、算出第一个 token 的那次前向，此时每个位置都有一份完整的隐藏特征。DFlash 从固定的一组、由浅到深均匀取出的若干层里各取一份隐藏特征，把它们按层拼起来，过一个轻量投影层融合跨层信息，压成一段紧凑的目标上下文特征，用它来条件化草稿模型。
依据：paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/appendix.tex Diffusion Drafter without Target Feature

::: 无目标特征块扩散草稿器的加速
一个 5 层块扩散草稿器，不接收任何目标上下文特征，在几个数学任务（GSM8K、Math500、AIME24、AIME25 都是数学评测集）上的加速；温度指采样温度，0 表示每步取概率最大的：
| 温度 | GSM8K | Math500 | AIME24 | AIME25 |
|---|---|---|---|---|
| 0 | 2.83 倍 | 3.73 倍 | 3.43 倍 | 3.35 倍 |
| 1 | 2.76 倍 | 3.31 倍 | 2.66 倍 | 2.65 倍 |
缺了目标特征，草稿质量受限，加速整体只有约 2.7 到 3.7 倍。
依据：paper/src-v2/sections/appendix.tex Diffusion Drafter without Target Feature
:::

### 把特征注入每一层的 KV 缓存

已有方法把目标上下文特征和草稿器的词嵌入（把 token 编号映成向量的表）拼在一起，只作为输入送进草稿器。这样草稿器越深，这份条件被逐层摊薄，多加一层草稿带来的接受长度收益越来越小。

DFlash 把这份特征当成一段常驻的上下文，直接注入草稿器每一层注意力的键和值（K/V）里。注意力在位置之间传信息，靠的是每个位置的查询（Q）去和所有位置的键（K）算相关度，再按相关度加权取值（V）。DFlash 让草稿 token 产生查询，而把目标特征投影成额外的键和值条目，与草稿自己的键值拼在一起。于是每一层都能看到这份条件，且它不随层数被摊薄，接受长度能随草稿层数有效增长。

这样注入之后，目标特征只作为额外的 KV 条目存在，它只被动地被查，不主动输出：不参与草稿器注意力的 [查询投影与输出投影](wiki/standard-attention)，也不经过前馈网络（一层里逐个位置各算各的那部分）。多出来的参数只有那个用来压出特征的共享投影矩阵。
依据：paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/appendix.tex KV Injection Mechanism and Memory Overhead；paper/src-v2/sections/exp.tex KV Injection vs. Input Fusion

::: 注入公式与显存开销
1. 目标上下文特征由各层隐藏特征拼接后过投影与归一化得到：$\mathbf{H}_t=\mathrm{RMSNorm}\!\left(W_c[\mathbf{H}^{(l_1)};\ldots;\mathbf{H}^{(l_5)}]\right)$；$\mathbf{H}^{(l_i)}$ 是目标模型第 $l_i$ 层的隐藏特征，$W_c$ 是融合投影，各草稿层共用这段特征；RMSNorm 是一种归一化（把向量按各分量的均方根缩放），$[\,;\,]$ 表示把各层向量沿特征维拼起来。
2. 第 $i$ 层草稿：查询只来自草稿 token，$\mathbf{Q}_i=W_i^Q\mathbf{H}_d$；键和值各自把目标特征与草稿 token 拼在一起，$\mathbf{K}_i=[W_i^K\mathbf{H}_t;\,W_i^K\mathbf{H}_d]_{\mathrm{seq}}$，$\mathbf{V}_i=[W_i^V\mathbf{H}_t;\,W_i^V\mathbf{H}_d]_{\mathrm{seq}}$，下标 seq 表示沿序列维拼接；$\mathbf{H}_d$ 是草稿 token 的隐藏特征，$W_i^Q,W_i^K,W_i^V$ 是第 $i$ 层的投影矩阵。
3. 显存：唯一新增的带参组件是 $W_c\in\mathbb{R}^{D\times 5D}$，$D$ 是隐藏维度。取 $D=2048$ 与 BF16 权重（16 位浮点），$W_c$ 约 42 MB，相对约 70 GB 的目标模型可忽略；批量 1、序列长 2048 时，投影的输入与输出约 40 MB 与 8 MB，块 16 解码时临时激活小于 400 KB。
依据：paper/src-v2/sections/appendix.tex KV Injection Mechanism and Memory Overhead
:::

::: KV 注入与输入融合的消融
在块扩散起草下，每层注入 KV（KV）比只在输入层融合（输入）更好，三个任务的接受长度与加速都更高；HumanEval 是代码评测集，MT-Bench 是对话评测集。
| 起草方式 | 注入位置 | GSM8K 接受/加速 | HumanEval 接受/加速 | MT-Bench 接受/加速 |
|---|---|---|---|---|
| 自回归 | 输入 | 4.2 / 2.1 倍 | 4.3 / 2.2 倍 | 3.1 / 1.4 倍 |
| 自回归 | KV | 4.8 / 2.4 倍 | 4.6 / 2.3 倍 | 3.4 / 1.5 倍 |
| 块扩散 | 输入 | 3.5 / 2.9 倍 | 3.5 / 2.9 倍 | 2.6 / 2.0 倍 |
| 块扩散 | KV | 4.2 / 3.3 倍 | 4.0 / 3.2 倍 | 3.0 / 2.2 倍 |
依据：paper/src-v2/sections/exp.tex KV Injection vs. Input Fusion
:::

### 训练：随机锚点、分块掩码与位置加权

草稿器的训练目标是让它的块预测对齐一个冻结的目标模型：把提示加响应过一遍目标模型，取出隐藏特征注入草稿器当键值，再在响应上构造被遮的块让它并行还原；训练只更新草稿器自己的层。
依据：paper/src-v2/sections/method.tex Training

#### 随机锚点构块

标准的块扩散训练把响应均分成块，在每块里随机遮一些位置让模型填回。DFlash 不用均分，而是为投机解码专门设计：从响应里随机取若干 token 当锚点，每个锚点作为一块的第一个位置，遮掉它后面的若干位置（块大小是一块的位置总数、含锚点，这里取 16，块首是锚点，其后 15 个位置被遮），让草稿器并行预测这些被遮 token。

这样构块与推理时的情形一致：推理时草稿器总是以一个目标模型产出的干净 token（已确定、未经草稿猜测的 token）为起点。随机化锚点位置还让草稿器见到更多样的目标上下文特征，相当于做了数据增广。消融（对有无某个设计做对照的实验）显示它同时提高了接受长度和加速。

训练时把所有块拼成一条序列一次处理，用一个稀疏注意力掩码（决定哪些位置能互相看见）控制可见范围：同一块内的 token 互相可见，也能看到注入的目标特征；不同块之间不可见，避免信息跨块泄漏，保证每块的条件只有它自己的锚点。用这种掩码能把多个块放进一次前向与反向（反向传播：按损失回传梯度去更新参数的那一步；损失是衡量预测分布与真实分布差别的量）。
依据：paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Random Sampling of Masked Blocks

::: 随机锚点与标准构块的消融
消融（3 层草稿、块 16、5 个目标特征层、训练数据 100K）：
| 构块方式 | Math500 加速 | Math500 接受长度 |
|---|---|---|
| 标准均分 | 4.13 倍 | 4.94 |
| 随机锚点 | 4.69 倍 | 5.64 |
依据：paper/src-v2/sections/appendix.tex Random Sampling of Masked Blocks
:::

#### 位置加权与共享嵌入

一块里靠前位置的预测一旦出错，它后面的位置全部作废，所以靠前的位置更重要。DFlash 相应地把交叉熵损失（衡量预测分布与真实分布差别的损失）按位置加权，越靠前权重越大：位置 k 的权重随 k 指数衰减。这让训练把重点放在更关键的前几个位置上，收敛更快、接受长度更高。

另一个省参数的设计是共享：草稿器不自己训练词嵌入表和输出头（把隐藏特征映成词表上分数的层），而是和目标模型共用，并在训练中冻结。这样可训练参数更少，草稿器更像一个紧贴目标模型表示空间的轻量适配器。

长上下文训练上，DFlash 把每条序列的被遮块数量固定、每个 epoch（把训练数据过一遍）重新随机锚点位置，把训练成本控制住。
依据：paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Training Implementation

::: 损失权重与训练超参
1. 位置权重 $w_k=\exp\!\left(-\dfrac{k-1}{\gamma}\right)$，$k$ 是 token 在块内的位置，$\gamma$ 是控制衰减率的超参。
2. 超参：6 个 epoch；AdamW（一种常用优化器），学习率 $6\times10^{-4}$（每步更新的幅度），梯度裁剪阈值 1.0（限制单步梯度的大小），学习率按余弦曲线从峰值衰减、开头用 0.04 的比例预热；序列长 3072，代码模型 Qwen3-Coder 取 4096，每条序列随机采 512 个锚点位置；衰减率 $\gamma$ 取块 16 为 7、块 10 为 5、块 8 为 4。
3. 训练可在线或离线：在线每一步现算目标隐藏特征；离线预计算并缓存，训练时载入以省算力。
依据：paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Training Implementation
:::

### 一次前向猜一整块

DFlash 的起草在推理时一次完成。当前已经确定的最后一个 token 作为这一块的起点，它正好是上一轮验证时目标模型顺带产出的 bonus token；它后面若干位置先留成占位。草稿器把这些位置拼成一条与已确定前缀错开的位置序列（位置编号接着前缀往下排），连同目标特征做一次前向，一次就给出所有待定位置的概率分布，每个位置取概率最大的（或按采样取一个），整块草稿 token 一次得到。

接下来照投机解码的常规做法验证：把这块候选交给目标模型一次前向，得到每个位置之后的下一个 token；从块首的待定位置开始逐个比对，找最长的、与目标一致的前缀，接受这一段，再取目标模型在该前缀之后算出的下一个 token 作为 bonus。已接受的 token 连同 bonus 并入输出，起点前进这么多，进入下一轮。
依据：paper/src-v2/sections/method.tex Inference；paper/src-v2/sections/preliminaries.tex Speculative Decoding Speedup；repo/dflash/dflash/model.py L234-L249；repo/dflash/dflash/model.py L260-L268；repo/dflash/dflash/model.py L282-L294

#### 跟着走一遍

块内被接受的草稿越长，本轮新增的 token 越多。
依据：自行构造

::: 跟着走一遍：一次起草加一次验证
设块大小 4，已生成到位置 10，位置 10 的 token 是 5，它来自上一轮目标模型，作这一块的锚点；位置 11、12、13 待定。
1. 把位置 10 到 13 拼成候选块：位置 10 是锚点 5，位置 11、12、13 是待定位置。
2. 草稿器一次前向，给三个待定位置各取一个 token：7、8、9。
3. 目标模型对这条候选块一次前向，得到每个位置之后的下一个 token 预测：位置 10 后是 7，位置 11 后是 8，位置 12 后是 5。
4. 从待定位置开头逐个比对：第一个 7 与预测一致、第二个 8 与预测一致，第三个 9 与预测 5 不一致，最长一致前缀长度为 2。
5. 接受 7、8，再取目标模型在位置 12 之后的预测 5 作 bonus，本轮共新增 3 个 token，含 bonus 的接受长度为 3；起点从 10 前进到 13。
依据：自行构造

```python
draft = [7, 8, 9]          # 草稿器给出的三个待定位置
target_next = [7, 8, 5]    # 目标模型在位置 10、11、12 之后预测的 token

accepted = 0
while accepted < len(draft) and draft[accepted] == target_next[accepted]:
    accepted += 1
bonus = target_next[accepted]
new_tokens = draft[:accepted] + [bonus]
print("一致前缀长度", accepted)
print("本轮新增 token", new_tokens)
print("下一轮起点", 10 + accepted + 1)
```
```text
一致前缀长度 2
本轮新增 token [7, 8, 5]
下一轮起点 13
```
依据：自行构造
:::

### 适用条件与边界

无损来自验证这一步，不是草稿器：草稿 token 只是候选，只有目标模型前向认可的那段前缀会进入输出，其余丢弃，所以并行起草不影响输出。代价是每轮都要跑一次目标模型前向做验证；块太大时，在计算受限的情形下（如大批量）验证开销会涨上来。

起草成本几乎不随块大小变，是在中等块大小的范围内成立。

符号 $\gamma$ 在两处含义不同：延迟公式里的 $\gamma$ 是每轮草稿的 token 数，损失衰减公式里的 $\gamma$ 是控制衰减率的超参，同一草稿器里这个超参按训练块大小取不同值。

训练块大小与推理块大小可以不一致，但只在一个方向上成立：训练块偏大的草稿器能用到更小的推理块，接受长度接近同块训练；反过来，训练块偏小的草稿器换到大块会下降。块 8 的草稿器在 Math500 上有 35.7% 的块被整块接受，说明块 8 常没被用满。

在 4K 上下文上训出的草稿器，用到 16K 时接受长度会衰减；用少量长样本微调后，接受长度大体保持甚至更高，说明取出的目标特征在长上下文下仍有代表性。
依据：paper/src-v2/sections/preliminaries.tex Diffusion Drafting；paper/src-v2/sections/intro.tex Introduction；paper/src-v2/sections/method.tex Training；paper/src-v2/sections/appendix.tex Training Implementation；paper/src-v2/sections/exp.tex Training-Inference Time Block Size；paper/src-v2/sections/exp.tex Long Context Adaptation
