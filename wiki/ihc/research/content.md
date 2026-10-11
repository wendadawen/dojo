# iHC

- 描述：iHC 是 Hyper-Connections 的一个变体：残差流扩成 n 条，流间的残差混合映射在 Hy4 实现里等价于恒等。
- 摘要：把残差流从一条扩成 $n$ 条，每层的输入由 $n$ 条流汇合而得，层的输出再分发回各条流；iHC 中流间的残差混合映射等价于恒等，省掉它的矩阵乘法。
- 主题：模型结构
- 标签：模型架构

## 资料

- Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou. Hyper-Connections. arXiv:2409.19606v3, 2024. [https://arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606)
- DeepSeek-AI. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. arXiv:2606.19348v1, 2026. [https://arxiv.org/abs/2606.19348](https://arxiv.org/abs/2606.19348)
- Tencent Hunyuan. Hy4-preview. 2026. [https://huggingface.co/tencent/Hy4-preview](https://huggingface.co/tencent/Hy4-preview)
- Tencent Hunyuan. Hy4-preview: official implementation. 2026. [https://github.com/Tencent-Hunyuan/Hy4-preview](https://github.com/Tencent-Hunyuan/Hy4-preview)
- Hugging Face. transformers，src/transformers/models/hy_v4/（modeling 与 configuration），commit cb33194ad6152bd9fad6305378d92db385dd7b32，2026. [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

## 短页

### 流间取恒等的那一族

残差连接是一条贯穿所有层的通路，每层的输出不覆盖输入，而是加回输入。Hyper-Connections（HC）把这条单线的通路扩成 $n$ 条平行的流，每层用几组可学的线性映射把各条流汇成层的输入、给层的输出分配权重，并让流与流之间互相混合。iHC 是 HC 的变体：Hy4 实现里，流间混合的那组映射等价于恒等。腾讯混元的 Hy4 preview 用的就是 iHC，残差流是 4 条。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；repo/hy4-transformers-modeling.py L603-L641；repo/hy4-transformers-configuration.py L47-L52；blog/hy4-model-card.md L63-L84

### 连接强度预先固定

残差连接只有一条宽度固定的通路，层的输出与输入之间的连接强度是预先固定的，不由模型学习。HC 把这个连接强度改成可学的。
依据：paper/hc-2409.19606v3.txt L113-L115；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections

### 汇入、分发、取恒等

1. 复制：把网络输入复制成 $n$ 条流，每一条一开始都相同。
2. 汇入：每层用输入映射把 $n$ 条流汇成这一层的输入，送进层算出 $d$ 维输出，$d$ 是隐藏维度。
3. 分发：用输出映射把层的输出分发回 $n$ 条流，再加到流上，构成下一层的状态。
4. 取恒等：流间混合的映射等价于恒等，每条流在层与层之间保持自己，这一组 $n \times n$ 的混合乘法省掉了。输入、输出两组权重由当前状态计算得到。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；repo/hy4-transformers-modeling.py L603-L641

## 长页

残差连接在现代网络里广泛使用，如 Transformer（由自注意力层和前馈层堆叠而成的序列模型）：每层的输出加回它的输入，$\mathbf{h}^{\ell+1} = \mathbf{h}^{\ell} + \mathcal{T}_\ell(\mathbf{h}^{\ell})$，$\mathbf{h}^{\ell}$ 是第 $\ell$ 层的输入向量，$\mathcal{T}_\ell$ 是这一层的计算，$\mathbf{h}^{\ell+1}$ 是它的输出、也是下一层的输入。Hyper-Connections（HC）把这条单线的通路扩宽成 $n$ 条平行的流，状态不再是一个向量，而是 $n \times d$ 的矩阵，$d$ 是隐藏维度。每层不再直接把输出加回输入，而是用几组可学的线性映射把各条流汇成层的输入、给层的输出分配权重，并让流与流之间互相混合。iHC 是 HC 的变体：腾讯混元的 Hy4 preview 模型用它，残差流是 4 条，流间混合的那组映射在实现里等价于恒等。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/hc-2409.19606v3.txt L120-L175；paper/src-v3/app.tex Pre-Norm Residual Connection；repo/hy4-transformers-modeling.py L603-L607；repo/hy4-transformers-configuration.py L47-L52；blog/hy4-model-card.md L63-L84

### 残差流与它的两条通路

残差连接把主干串成一条通路，这条通路叫残差流。它有两种接法。预归一化（Pre-Norm）把归一化放在层内部、加回之前，写成 $\mathbf{h}^{\ell+1} = \mathbf{h}^{\ell} + \mathcal{T}_\ell(\text{Norm}(\mathbf{h}^{\ell}))$，$\text{Norm}$ 是归一化（把向量的数值尺度调到固定范围）。它缓解了梯度消失（反向传播时梯度逐层变小），但深层的特征高度相似，后加的层贡献变小，这叫表示塌缩。后归一化（Post-Norm）把归一化放在加回之后，写成 $\mathbf{h}^{\ell+1} = \text{Norm}(\mathbf{h}^{\ell} + \mathcal{T}_\ell(\mathbf{h}^{\ell}))$，它减弱了隐藏状态（层间传递的向量）对后续层的影响，缓解了塌缩，梯度消失却又出现。两种接法里，层的输出与输入之间的连接强度都是预先固定的，不由模型学习。
依据：paper/src-v3/iclr2025_conference.tex Introduction；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/src-v3/app.tex Post-Norm Residual Connection；paper/hc-2409.19606v3.txt L68-L76；paper/hc-2409.19606v3.txt L113-L115

### 把残差扩成 n 条流

HC 把残差流从一条扩成 $n$ 条。设第 $\ell$ 层之前的状态是矩阵 $X_\ell \in \mathbb{R}^{n \times d}$，它的第 $i$ 行是一条流 $\mathbf{x}_{\ell,i} \in \mathbb{R}^{d}$，$n$ 叫扩张率。HC 引入三组线性映射：

- 输入映射 $A_\ell \in \mathbb{R}^{1 \times n}$，把 $n$ 条流汇成这一层的输入 $\mathbf{a}_\ell = A_\ell X_\ell \in \mathbb{R}^d$；
- 残差映射 $B_\ell \in \mathbb{R}^{n \times n}$，在流与流之间做线性混合；
- 输出映射 $C_\ell \in \mathbb{R}^{n \times 1}$，把层的输出分发回 $n$ 条流。

一层的更新写成 $X_{\ell+1} = B_\ell X_\ell + C_\ell \mathcal{T}_\ell(A_\ell X_\ell)$。式中的 $\mathcal{T}_\ell$ 已把前置归一化并进层里，即上一节的 $\mathcal{T}_\ell(\text{Norm}(\cdot))$ 在这里写作 $\mathcal{T}_\ell(\cdot)$。$C_\ell$ 与层输出的乘积按列向量与行向量相乘理解：$C_\ell$ 是 $n \times 1$，层的输出 $\in \mathbb{R}^d$ 当作 $1 \times d$ 的行向量，相乘得到 $n \times d$，即层的输出乘上 $n$ 个权重后分到各条流。层的输入和输出都还是 $d$ 维，加宽残差流不动层内部的结构，$n$ 与 $d$ 因此是两条独立的轴，而 $n$ 通常远小于 $d$。普通的残差连接是 $n=1$ 的特例。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections；paper/src-v3/app.tex Pre-Norm Residual Connection；paper/hc-2409.19606v3.txt L182-L192；paper/hc-2409.19606v3.txt L294-L318；paper/dsv4-2606.19348v1.txt L355-L397

::: $n=1$ 时 HC 退回预归一化
1. 输入映射 $A_\ell$ 是 $1 \times 1$ 的标量，取 1，于是 $\mathbf{a}_\ell = \mathbf{x}_{\ell,1}$，层收到原样的输入。
2. 残差映射 $B_\ell$ 是 $1 \times 1$ 的标量，取 1，于是流只保留自己，没有别的流可混合。
3. 输出映射 $C_\ell$ 是 $1 \times 1$ 的标量，取 1，于是层的输出原样加回。
4. 三步合起来是 $X_{\ell+1} = X_\ell + \mathcal{T}_\ell(X_\ell)$，正是预归一化的写法，可见预归一化是 HC 在 $n=1$、三组权重都固定为 1 时的特例。
依据：paper/src-v3/iclr2025_conference.tex Residual Connections as Non-trainable Hyper-Connections
:::

#### 一层的前向顺序

网络入口把输入向量复制成 $n$ 份，$X_0$ 的每一行都是这个输入向量。每一层依次做三步：用 $A_\ell$ 把 $n$ 条流汇成层的输入，层计算，用 $C_\ell$ 把层的输出分发回各条流并加到 $B_\ell X_\ell$ 上；$B_\ell$ 是否取恒等，区别只在各条流的原有内容在加上层输出之前先不先被混合。Transformer 的一个块里有两个这样的 HC，自注意力外一个，前馈外一个。HC 原式在末端把 $X_L$ 按行求和汇成一条向量，$L$ 是总层数，再进最终的归一化层与输出层。
依据：paper/src-v3/iclr2025_conference.tex Static Hyper-Connections；paper/hc-2409.19606v3.txt L182-L192；repo/hy4-transformers-modeling.py L667-L716；paper/dsv4-2606.19348v1.txt L365-L385

::: 手算一层：n=2、d=2
设 $n=2$、$d=2$，输入映射 $A=(0.6,\ 0.4)$，输出映射 $C=(0.9,\ 1.5)^\top$，两条流 $X=\begin{pmatrix}1.0 & 2.0\\0.5 & -0.5\end{pmatrix}$。层取逐元素的 $\mathcal{T}(\mathbf{v})=0.5\mathbf{v}+0.1$，只为让数字好算。对照用的残差映射取 $B=\begin{pmatrix}0.7 & 0.3\\0.2 & 0.8\end{pmatrix}$。
依据：自行构造

1. 汇入：$AX=(0.6\times1.0+0.4\times0.5,\ 0.6\times2.0+0.4\times(-0.5))=(0.8,\ 1.0)$。
2. 过层：$\mathcal{T}(0.8,\ 1.0)=(0.5,\ 0.6)$。
3. 分发：$C\mathcal{T}(AX)$ 的第 1 行是 $0.9\times(0.5,\ 0.6)=(0.45,\ 0.54)$，第 2 行是 $1.5\times(0.5,\ 0.6)=(0.75,\ 0.90)$。
4. $B=I$：下一层的状态是 $X$ 加上第 3 步的结果，即 $\begin{pmatrix}1.45 & 2.54\\1.25 & 0.40\end{pmatrix}$，每条流都是自己加上一份层输出。
5. 一般的 $B$：$BX=\begin{pmatrix}0.85 & 1.25\\0.60 & 0.00\end{pmatrix}$，加上第 3 步的结果得 $\begin{pmatrix}1.30 & 1.79\\1.35 & 0.90\end{pmatrix}$，两条流的原有内容先被混合过。
依据：自行构造

```python
A = [0.6, 0.4]
C = [0.9, 1.5]
X = [[1.0, 2.0], [0.5, -0.5]]
B = [[0.7, 0.3], [0.2, 0.8]]
I = [[1.0, 0.0], [0.0, 1.0]]

a = [sum(A[i] * X[i][j] for i in range(2)) for j in range(2)]
t = [0.5 * v + 0.1 for v in a]
print("汇入", [round(v, 2) for v in a], "层输出", [round(v, 2) for v in t])
for name, M in (("B=I", I), ("一般 B", B)):
    new = [[sum(M[i][k] * X[k][j] for k in range(2)) + C[i] * t[j] for j in range(2)] for i in range(2)]
    print(name, [[round(v, 2) for v in row] for row in new])
```

```text
汇入 [0.8, 1.0] 层输出 [0.5, 0.6]
B=I [[1.45, 2.54], [1.25, 0.4]]
一般 B [[1.3, 1.79], [1.35, 0.9]]
```
:::

#### 映射的参数从哪来

三组映射的权重不是固定常数，而是由当前状态计算得到：先对状态做归一化，经线性层得到未约束的参数，再加一组与输入无关的静态偏置，动态部分乘一个初值很小的可学因子，权重因此既随输入变化，又保留一份基线。线性层之后的激活，HC 的原始形式用 tanh，并把静态部分初始化成让 HC 起步时等价于预归一化；DeepSeek-V4（另一个采用 HC 类连接的模型）里的写法和 Hy4 实现改用 Sigmoid，把输入映射压成 0 到 1 之间的非负数，以避免信号相消，输出映射在 Sigmoid 之后再乘 2。
依据：paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections；paper/src-v3/iclr2025_conference.tex Initialization；paper/dsv4-2606.19348v1.txt L393-L395；paper/dsv4-2606.19348v1.txt L398-L424；repo/hy4-transformers-modeling.py L603-L641

::: 输入、输出映射的参数公式
记 $n$ 条流拉平并归一化后的行向量为 $\hat{X}_\ell = \mathrm{RMSNorm}(\mathrm{vec}(X_\ell))$，$\mathrm{vec}$ 是拉平，$\mathrm{RMSNorm}$ 是均方根归一化。
依据：paper/dsv4-2606.19348v1.txt L398-L424

1. 输入映射：$\tilde{A}_\ell = \alpha_\ell^{\text{pre}} \cdot (\hat{X}_\ell W_\ell^{\text{pre}}) + S_\ell^{\text{pre}}$。
2. 输出映射：$\tilde{C}_\ell = \alpha_\ell^{\text{post}} \cdot (\hat{X}_\ell W_\ell^{\text{post}})^\top + S_\ell^{\text{post}}$。
3. 其中 $W$ 是可学的映射参数，$S$ 是可学的静态偏置，$\alpha$ 是初始化为小值的可学门控因子。
4. 再过 Sigmoid：$A_\ell = \sigma(\tilde{A}_\ell)$，$C_\ell = 2\sigma(\tilde{C}_\ell)$，$\sigma$ 是 Sigmoid。
依据：paper/dsv4-2606.19348v1.txt L398-L424
:::

### Identity Hyper-Connections

Hy4 实现里，一层只算两组权重 pre 和 post，pre 是输入映射 $A_\ell$ 那一组，post 是输出映射 $C_\ell$ 那一组。读入时 $n$ 条流由 pre 汇成层的输入，层的输出乘 post 分发回所有流，再直接加回原状态：更新式是 $X_{\ell+1} = X_\ell + C_\ell \mathcal{T}_\ell(A_\ell X_\ell)$，相当于 $B_\ell X_\ell$ 里的 $B_\ell$ 取单位矩阵，每条流在层与层之间保持自己，省下一次 $n \times n$ 的混合乘法。

与它最容易混淆的是 DeepSeek-V4 的 mHC：HC 多层堆叠时训练常出现数值不稳定，mHC 为此把 $B_\ell$ 约束成双随机矩阵（各行各列非负且和为 1），并用 Sinkhorn-Knopp 迭代把未约束的矩阵投影进去。iHC 没有这组映射，也就没有投影迭代。
依据：repo/hy4-transformers-modeling.py L603-L641；repo/hy4-transformers-modeling.py L643-L664；repo/hy4-transformers-modeling.py L667-L716；paper/dsv4-2606.19348v1.txt L380-L383；paper/dsv4-2606.19348v1.txt L386-L431

::: iHC 里两组权重的算法
均方根归一化只是乘一个标量，与线性映射的先后可以交换，结果同「先归一化再线性映射」。
依据：repo/hy4-transformers-modeling.py L603-L641

1. 拉平：把 $n$ 条流拉平成一个向量 `flat`，算 `linear(flat, fn)` 得到一组结果 `mixes`，`fn` 是这层线性映射的可学参数。
2. 归一化：把 `flat` 整体做一次均方根归一化得到每位置一个缩放系数，再与上面的 `mixes` 逐元素相乘。
3. 拆成两组：`mixes.split(n)`，各得 $n$ 个未约束的 logit，前 $n$ 个给输入侧、后 $n$ 个给输出侧。
4. 输入侧：`pre = sigmoid(pre_logits * pre_scale + pre_b) + eps`，`pre_scale` 是缩放、`pre_b` 是偏置（对应 DeepSeek-V4 形式里的门控因子与静态偏置），sigmoid 的值在 0 到 1 之间，`eps` 是再加上的一个很小的常数。
5. 输出侧：`post = magnitude * sigmoid(post_logits * post_scale + post_b) + eps`，`magnitude` 是幅度系数，Hy4 配置里取 2。
6. 汇入：`out = Σ pre_i * stream_i`，即用 $n$ 个 $\text{pre}_i$ 把 $n$ 条流加权求和，作为这一层的输入。
7. 分发：层算完后 `hidden = post ⊗ layer_out + residual`，$\otimes$ 表示列向量与行向量相乘（外积），$n$ 个 post 权重都乘同一个层输出，再加回原状态，没有 $B$ 这一项。
8. 末端：HC 原式按行求和汇合 $n$ 条流，Hy4 实现改用同样方式由 `hc_fn` 算出的一组权重加权求和，再进最终的归一化。
依据：repo/hy4-transformers-modeling.py L603-L641；repo/hy4-transformers-modeling.py L643-L664；repo/hy4-transformers-modeling.py L667-L716；repo/hy4-transformers-configuration.py L47-L52
:::

### 效果与代价

实验用 OLMo 系列语言模型，基线是同一模型不加 HC，预训练 500B 个 token（训练文本的最小单位）。1B 稠密模型上，动态 HC（权重由当前状态计算得到的 HC 原始形式）的结果是：$n=1$ 不如基线，$n>1$ 好过基线，$n=8$ 相对 $n=4$ 的额外收益很小。V2 验证损失 2.781 对 2.778，V3 验证损失 2.514 对 2.516，下游平均准确率 63.8 对 62.8，V2、V3 是两套验证集。OLMoE-1B-7B 是总参数 7B、激活 1.3B 参数的混合专家（MoE）模型；在它上面，$n=4$ 的动态 HC 收敛快 1.8 倍，ARC-Challenge（一个问答类评测集）准确率比基线高 6 分。OLMo-1B、$n=4$ 时，HC 新增参数占总参数的 0.03349%，浮点运算量增加 0.200%。
依据：paper/src-v3/iclr2025_conference.tex Ablation Study；paper/src-v3/iclr2025_conference.tex MoE Models；paper/hc-2409.19606v3.txt L409-L422；paper/hc-2409.19606v3.txt L465-L470；paper/hc-2409.19606v3.txt L168-L170；paper/hc-2409.19606v3.txt L431-L433；paper/hc-2409.19606v3.txt L987-L1027

::: 1B 稠密模型按 n 的消融
OLMo-1B 及它的动态 HC 变体，预训练 500B token，报告 V2、V3 两套验证集上的损失（越低越好）和下游平均准确率（越高越好）。
依据：paper/hc-2409.19606v3.txt L409-L422

1. 基线 OLMo-1B：V2 损失 2.811，V3 损失 2.544，下游平均 62.5。
2. $n=1$：2.819，2.556，62.3，三项都不如基线。
3. $n=2$：2.802，2.534，63.0。
4. $n=4$：2.781，2.514，63.8。
5. $n=8$：2.778，2.516，62.8，V2 损失比 $n=4$ 低，V3 损失和下游平均比 $n=4$ 差。
依据：paper/hc-2409.19606v3.txt L409-L422
:::

### 适用条件与边界

动态 HC 的训练显存占用增加：实测 OLMo-1B 在 $n=4$ 时多 26.1%，OLMo-7B 多 28.28%。推理时，激活（前向计算中保存的中间结果）所占显存主要由 KV cache（缓存已处理位置的注意力键值）决定，KV cache 不受 HC 额外激活的影响。iHC 并不是流间完全不交流：各条流仍通过共同的汇入和分发间接相关。本页资料里没有 iHC 与 HC 的对照实验，Hy4 的 iHC 开销也没有数字。
依据：paper/hc-2409.19606v3.txt L1029-L1057；paper/src-v3/app.tex Parameters, Computation and Memory Footprint Analysis；repo/hy4-transformers-modeling.py L603-L641

::: 训练显存占用的实测
在 8 张 GPU 上测得的训练显存占用，每档的增量是相对它自己那一档的基线（同一个模型不加 HC），微批大小是每张 GPU 一次处理的 token 数，各档不同。
依据：paper/hc-2409.19606v3.txt L1044-L1057

1. OLMo-1B 基线：41.11 GB；加 $n=2$：47.56 GB，多 15.7%；加 $n=4$：51.86 GB，多 26.1%。微批 16384。
2. OLMo-7B 基线：26.27 GB；加 $n=4$：33.70 GB，多 28.28%。微批 2048，与 1B 不同，绝对值不能直接比。
3. OLMoE-1B-7B 基线：31.59 GB；加 $n=4$：34.65 GB，多 9.7%。微批 4096。
依据：paper/hc-2409.19606v3.txt L1044-L1057
:::
