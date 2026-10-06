# FlashAttention

- 描述：FlashAttention 把注意力按块算完，不把 N×N 的中间矩阵写回显存，在结果与标准注意力完全相同的前提下减少显存读写。
- 摘要：标准注意力要把 $N\times N$ 的矩阵 $S$、$P$ 写进显存，慢的是读写，不是计算。FlashAttention 分块计算并逐块修正 softmax 的归一化，反向时重算而不存中间矩阵，精确得到 $\mathrm{softmax}(QK^\top)V$，额外显存只随 $N$ 线性增长。
- 主题：注意力机制，内存与缓存
- 标签：注意力

## 资料

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135v2, 2022. [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

## 短页

### 少读写，算得一样

FlashAttention 是一种计算注意力的方法。它得到的结果与标准注意力完全相同，只是计算过程中不再把 $N\times N$ 的大矩阵写回显存。所有步骤在一次 kernel 调用里完成。
依据：paper/src-v2/src/intro.tex L31-L41

### 慢的是搬数据，不是做乘法

GPU 的计算速度已经远超显存的读写速度，注意力里的 softmax 这类操作受显存读写限制：耗时由读写的数据量决定，不由乘法的次数决定。标准注意力要把两个 $N\times N$ 的矩阵写进显存再读回来，读写量随 $N$ 平方增长，所以许多只减少乘法次数的近似方法，并没有带来实际的加速。
依据：paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/background.tex Standard Attention Implementation；paper/src-v2/src/intro.tex L12-L14

### 分块算、边算边修正、反向重算

把输入切成小块，每次只在片上存储里算一块。softmax 需要整行的最大值和求和，所以每块额外记下这两个量，处理下一块时修正已有的结果。反向时不存 $S$、$P$，只存输出和这两个量，需要时在片上重新算出来。
依据：paper/src-v2/src/algo.tex Tiling；paper/src-v2/src/algo.tex Recomputation

## 长页

标准注意力对长序列又慢又占显存，原因不在乘法太多，而在中间结果在显存与计算单元之间来回搬运。FlashAttention 是一种精确的注意力算法：把输入切成块，每次在片上存储里算完一块，不把 $N\times N$ 的中间矩阵写回显存。它得到的输出与标准注意力一致，额外显存只随序列长度线性增长。
依据：paper/src-v2/src/intro.tex L31-L41

### 注意力慢在读写，不在计算

输入 $Q$、$K$、$V$ 都是 $N\times d$ 的矩阵，$N$ 是序列长度，$d$ 是头维度，通常 $N$ 远大于 $d$。标准注意力分三步：先算 $S=QK^\top$，再对 $S$ 的每一行做 softmax 得到 $P$，最后算 $O=PV$，其中 $S$、$P$ 都是 $N\times N$ 的矩阵，$O$ 是 $N\times d$ 的输出。
依据：paper/src-v2/src/background.tex Standard Attention Implementation

GPU 的存储分两级。高带宽显存（HBM）容量大、速度慢，片上存储（SRAM）容量小、速度快，快约一个数量级，容量却小几个数量级。GPU 上一次运算称为一个 kernel：它从显存读入输入，在片上算完，再把输出写回显存。
依据：paper/src-v2/src/background.tex Hardware Performance

一个操作的耗时由什么决定，取决于算术强度，即每读写一个字节所做的运算次数。耗时主要由运算次数决定的，称为受计算限制，例如大矩阵乘法；耗时主要由读写次数决定的，称为受访存限制，例如逐元素运算和 softmax 这类归约（把一行的多个数合成一个数的操作）。随着计算速度相对显存速度越来越快，越来越多的操作变成受访存限制。
依据：paper/src-v2/src/background.tex Hardware Performance

标准注意力把 $S$ 和 $P$ 都完整写进显存，再读回来参与下一步，占用 $O(N^2)$ 的显存。其中 softmax 受访存限制，读写量随 $N$ 平方增长，这就是它慢的原因。
依据：paper/src-v2/src/background.tex Standard Attention Implementation

### 分块计算 softmax

要避免写出 $S$ 和 $P$，就要让 softmax 一次只处理一部分。难点在于 softmax 把一行里所有元素耦合在一起：每个元素要除以整行的和。
依据：paper/src-v2/src/algo.tex Tiling

#### 数值稳定的 softmax 与分块合并

为了数值稳定，softmax 先减去这一行的最大值。对向量 $x$，记 $m(x)$ 为最大值，$f(x)$ 为各元素减去最大值后取指数的向量，$\ell(x)$ 为 $f(x)$ 的各元素之和，则 $\mathrm{softmax}(x)=f(x)/\ell(x)$。
依据：paper/src-v2/src/algo.tex Tiling

把一行拆成两块 $x^{(1)}$、$x^{(2)}$ 后，只要每块记下自己的 $m$ 和 $\ell$，就能合并成整行的结果：整行的最大值取两块最大值中的较大者；整行的求和量是各块求和量乘上一个修正系数再相加，系数是该块的最大值与整行最大值之差的指数，用来把不同基准下的指数统一到同一个最大值上。
依据：paper/src-v2/src/algo.tex Tiling

$$m=\max(m^{(1)},m^{(2)}),\qquad \ell=e^{m^{(1)}-m}\ell^{(1)}+e^{m^{(2)}-m}\ell^{(2)}$$

输出 $O$ 是各行 softmax 与 $V$ 的乘积。显存里保存的是已经除过求和量的输出，所以每处理完一块，已有的输出也要更新：先乘上旧的求和量还原成未归一化的值，按新的最大值重新缩放，加上这一块的贡献（这一块未归一化的指数与 $V$ 的乘积，同样按新的最大值缩放），再除以新的求和量。
依据：paper/src-v2/src/algo.tex Efficient Attention Algorithm

#### 一个能手算的例子

只要每块保留最大值和求和量，分块合并的结果就与整行直接算完全一样。
依据：自行构造

::: 四个数、两块的手算
取一行 $x=(1,2,3,4)$，$V$ 取 $(10,20,30,40)$ 的四个标量，分成两块 $(1,2)$ 和 $(3,4)$。块 1 的最大值是 2，求和量 $e^{-1}+1\approx1.3679$；块 2 的最大值是 4，求和量也是 $e^{-1}+1\approx1.3679$。合并时整行最大值取 4，块 1 的修正系数是 $e^{2-4}$，所以整行求和量 $=e^{-2}\times1.3679+1.3679\approx1.553$，与直接对整行算得的 $\ell$ 相同。每块各自的输出 $o_k$ 是该块内 softmax 与 $V$ 的乘积，即 $o_1\approx17.3106$、$o_2\approx37.3106$。按块合并时，用 $\ell^{(1)}e^{m^{(1)}-m}o_1$ 与 $\ell^{(2)}e^{m^{(2)}-m}o_2$ 相加，再除以整行求和量，得到的 $O$ 与整行直接算的 $34.9265$ 相同。

```python
import math

x = [1.0, 2.0, 3.0, 4.0]
v = [10.0, 20.0, 30.0, 40.0]

def block(xs, vs):
    m = max(xs)
    e = [math.exp(a - m) for a in xs]
    l = sum(e)
    o = sum(a * b for a, b in zip(e, vs)) / l
    return m, l, o

full_m, full_l, full_o = block(x, v)
m1, l1, o1 = block(x[:2], v[:2])
m2, l2, o2 = block(x[2:], v[2:])
m = max(m1, m2)
l = math.exp(m1 - m) * l1 + math.exp(m2 - m) * l2
o = (l1 * math.exp(m1 - m) * o1 + l2 * math.exp(m2 - m) * o2) / l
print(round(l, 4), round(full_l, 4))
print(round(o, 4), round(full_o, 4))
```

```text
1.553 1.553
34.9265 34.9265
```
依据：自行构造
:::

### 完整的前向算法

外层循环遍历 $K$、$V$ 的块，内层循环遍历 $Q$ 的块。$S=QK^\top$ 的每一行里，第 $j$ 段对应 $K$ 的第 $j$ 块，所以遍历 $K$、$V$ 的块，就是在逐段合并同一行的 softmax。每一步把当前的 $K$、$V$ 块和 $Q$ 块载入片上，在片上算出这一小块的 $S$，更新输出和两个统计量，再写回显存。块的大小由片上存储的容量决定，一块 $K$、$V$ 要能放进片上存储。整个过程放进一次 kernel 调用：载入输入，在片上完成矩阵乘、softmax 等全部步骤，再把结果写回显存，中间的 $S$、$P$ 不出片。算法返回精确的 $\mathrm{softmax}(QK^\top)V$，除输入输出外额外只需要 $O(N)$ 的显存，用来存每行的最大值和求和量。
依据：paper/src-v2/src/algo.tex Efficient Attention Algorithm；paper/src-v2/src/proofs.tex L5-L20；paper/src-v2/src/proofs.tex L116-L139

#### 算法的完整步骤

::: 前向算法的逐行描述
设片上存储大小为 $M$，取 $K$、$V$ 的块大小 $B_c=\lceil M/(4d)\rceil$，$Q$ 的块大小 $B_r=\min(\lceil M/(4d)\rceil,d)$。在显存里把输出 $O$ 和求和量 $\ell$ 初始化为 0，最大值 $m$ 初始化为负无穷。把 $Q$ 切成 $T_r=\lceil N/B_r\rceil$ 块，把 $K$、$V$ 切成 $T_c=\lceil N/B_c\rceil$ 块。

对每个 $K$、$V$ 块 $j$：载入 $K_j$、$V_j$。对每个 $Q$ 块 $i$：载入 $Q_i$、$O_i$、$\ell_i$、$m_i$；在片上算 $S_{ij}=Q_iK_j^\top$；算这一小块每行的最大值 $\tilde m_{ij}$、减最大值后取指数的矩阵 $\tilde P_{ij}$、每行的和 $\tilde\ell_{ij}$；合并得到新的最大值 $m_i^{\mathrm{new}}=\max(m_i,\tilde m_{ij})$ 和新的求和量 $\ell_i^{\mathrm{new}}=e^{m_i-m_i^{\mathrm{new}}}\ell_i+e^{\tilde m_{ij}-m_i^{\mathrm{new}}}\tilde\ell_{ij}$；更新输出（$\mathrm{diag}(v)$ 表示以向量 $v$ 为对角线的矩阵，左乘它等于按行缩放）$O_i\leftarrow\mathrm{diag}(\ell_i^{\mathrm{new}})^{-1}\left(\mathrm{diag}(\ell_i)e^{m_i-m_i^{\mathrm{new}}}O_i+e^{\tilde m_{ij}-m_i^{\mathrm{new}}}\tilde P_{ij}V_j\right)$；把 $O_i$、$\ell_i$、$m_i$ 写回显存。第一块处理时 $m=-\infty$、$\ell=0$，旧值项的系数 $e^{m_i-m_i^{\mathrm{new}}}$ 为 0，更新结果就是这一块自己的值。

遍历完所有块后返回 $O$。
依据：paper/src-v2/src/algo.tex Efficient Attention Algorithm
:::

### 反向：重算代替存储

反向传播要用到注意力矩阵 $S$ 和 $P$ 来计算对 $Q$、$K$、$V$ 的梯度。标准做法是前向时把它们存下来，占 $O(N^2)$ 显存。FlashAttention 只存输出 $O$ 和 softmax 的统计量（最大值 $m$ 与求和量 $\ell$），反向时从 $Q$、$K$、$V$ 的块在片上重新算出 $S$，再用已存的整行统计量得到 $P$：$P_{ij}=\mathrm{diag}(\ell_i)^{-1}\exp(S_{ij}-m_i)$。这类似选择性的梯度检查点，即不存中间值、反向时重新计算。
依据：paper/src-v2/src/algo.tex Recomputation；paper/src-v2/src/algo_details.tex L293

重算会增加运算次数，但避免了把 $S$、$P$ 从显存读回来，反向反而更快。
依据：paper/src-v2/src/algo.tex Recomputation

把作用于同一输入的多个操作融合成一个 kernel，输入就只需从显存读一次，这叫 kernel 融合。但在训练时，反向需要的中间值仍要写回显存，朴素融合的效果因此下降。FlashAttention 用重算避免存储这些中间值。
依据：paper/src-v2/src/background.tex L48-L55；paper/src-v2/src/algo.tex Recomputation

### 读写次数的分析

设片上存储的大小为 $M$，且 $d\le M\le Nd$。标准注意力需要 $\Theta(Nd+N^2)$ 次显存读写，FlashAttention 需要 $\Theta(N^2d^2M^{-1})$ 次。$d$ 通常在 64 到 128，$M$ 约 100KB，$d^2$ 远小于 $M$，所以后者少得多。FlashAttention 的运算量其实更大，因为反向要重算，但读写少很多，速度反而更快，说明读写次数是决定注意力速度的主要因素。
依据：paper/src-v2/src/theory.tex Analysis: IO Complexity

#### 两个读写次数的量级

标准注意力三步各自读写：算 $S$ 时读 $Q$、$K$ 写 $S$，是 $\Theta(Nd+N^2)$；算 $P$ 时读 $S$ 写 $P$，是 $\Theta(N^2)$；算 $O$ 时读 $P$、$V$ 写 $O$，是 $\Theta(Nd+N^2)$。合计 $\Theta(Nd+N^2)$。

FlashAttention 中 $K$、$V$ 的每个元素只从显存读一次，而 $Q$ 和 $O$ 要被整个遍历 $T_c$ 次，每次读写 $\Theta(Nd)$，合计 $\Theta(NdT_c)$。块大小受片上存储限制，取 $B_c=\Theta(M/d)$，所以 $T_c=\Theta(Nd/M)$，读写次数为 $\Theta(N^2d^2M^{-1})$，与片上存储容量成反比：存储越大，块越大，遍历越少。
依据：paper/src-v2/src/theory.tex Analysis: IO Complexity；paper/src-v2/src/proofs.tex L85-L139

#### 为什么渐近上不能再少

不存在一个精确注意力算法，能在 $[d,Nd]$ 范围内所有的片上存储大小 $M$ 上同时做到 $o(N^2d^2M^{-1})$ 次显存读写。证明用反证法：若存在，则在 $M=\Theta(Nd)$ 时读写次数是 $o(Nd)$，而输入 $Q$、$K$、$V$ 和输出 $O$ 本身就至少要 $Nd$ 次读写，二者矛盾。按每个 $M$ 分别给出下界，论文留作未来工作。
依据：paper/src-v2/src/theory.tex Analysis: IO Complexity；paper/src-v2/src/proofs.tex L149-L165

### 适用条件与边界

FlashAttention 通过减少显存读写来加速。它的结果是精确的，不是近似：输出与标准注意力一致。它的实现需要为每一种新的注意力实现写一个专门的 CUDA kernel。CUDA 是编写 GPU 程序的底层语言，比 PyTorch 这类高层框架低得多，写起来工程量大，而且实现不一定能迁移到不同的 GPU 架构上。
依据：paper/src-v2/src/discussion.tex L1-L8；paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/algo.tex Efficient Attention Algorithm
