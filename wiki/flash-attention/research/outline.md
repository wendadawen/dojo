# FlashAttention

- 描述：FlashAttention 把注意力按块算完，不把 N×N 的中间矩阵写回显存，在结果与标准注意力完全相同的前提下减少显存读写。
- 摘要：标准注意力要把 $N\times N$ 的矩阵 $S$、$P$ 写进显存，慢的是读写，不是计算。FlashAttention 分块计算并逐块修正 softmax 的归一化，反向时重算而不存中间矩阵，精确得到 $\mathrm{softmax}(QK^\top)V$，额外显存只随 $N$ 线性增长。
- 主题：注意力机制，内存与缓存
- 标签：注意力

## 长页

- 开头｜说明 FlashAttention 是什么、大概怎么做：精确的注意力，只是不把大矩阵写回显存｜正文｜250｜paper/src-v2/src/intro.tex L31-L41
- ### 注意力慢在读写，不在计算｜先定义标准注意力的三步计算和 Q、K、V、N、d，再讲 GPU 两级存储与受访存限制，说明标准注意力慢在把 S、P 写回显存｜正文｜550｜paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/background.tex Standard Attention Implementation
- ### 分块计算 softmax｜讲 softmax 为什么能分块、每块要多记哪两个量｜正文｜250｜paper/src-v2/src/algo.tex Tiling
- #### 数值稳定的 softmax 与分块合并｜先讲 softmax 减最大值的写法，再讲两块结果怎么用最大值和求和量合并，以及输出怎么随之重新缩放｜公式｜250｜paper/src-v2/src/algo.tex Tiling；paper/src-v2/src/algo.tex Efficient Attention Algorithm
- #### 一个能手算的例子｜用四个数分成两块算 softmax，对照整体直接算，正文只写它说明的道理｜折叠块｜400｜自行构造
- ### 完整的前向算法｜讲两层循环怎样逐块更新输出和两个统计量，块大小由片上容量定，整个过程放进一次 kernel，结果精确、额外显存线性｜正文｜500｜paper/src-v2/src/algo.tex Efficient Attention Algorithm
- #### 算法的完整步骤｜给出算法的逐行描述｜折叠块｜500｜paper/src-v2/src/algo.tex Efficient Attention Algorithm
- ### 反向：重算代替存储｜讲反向为何需要 S 和 P、只存输出和两个统计量为何就能重算、重算为何反而更快，以及朴素融合为何不够用｜正文｜600｜paper/src-v2/src/algo.tex Recomputation；paper/src-v2/src/background.tex L48-L55
- ### 读写次数的分析｜给出两种做法各需要多少次显存读写，说明读写次数而非计算量决定运行时间：计算量更大、读写更少，反而更快｜正文｜400｜paper/src-v2/src/theory.tex Analysis: IO Complexity
- #### 两个读写次数的量级｜先逐步数出标准做法与分块做法各自的读写来自哪几步，再给出量级，说明它为什么与片上存储容量成反比｜公式｜250｜paper/src-v2/src/theory.tex Analysis: IO Complexity；paper/src-v2/src/proofs.tex L85-L115
- #### 为什么渐近上不能再少｜讲对所有片上存储容量，没有精确算法能在读写次数上做到渐近更好｜正文｜200｜paper/src-v2/src/theory.tex Analysis: IO Complexity
- ### 适用条件与边界｜讲每种注意力都要写专门的 kernel，收益依赖受访存限制，并澄清结果是精确的、不是近似｜正文｜250｜paper/src-v2/src/discussion.tex L1-L8；paper/src-v2/src/background.tex Hardware Performance；paper/src-v2/src/algo.tex Efficient Attention Algorithm

## 短页

- ### 少读写，算得一样｜一两段说清 FlashAttention 是什么：精确的注意力，只是不把大矩阵写回显存｜正文｜250｜paper/src-v2/src/intro.tex L31-L41
- ### 慢的是搬数据，不是做乘法｜讲标准注意力受显存读写限制，所以单纯减少乘法次数并不加速｜正文｜250｜paper/src-v2/src/background.tex Hardware Performance
- ### 分块算、边算边修正、反向重算｜讲分块、逐块修正 softmax 的归一化、反向时重算三件事｜正文｜350｜paper/src-v2/src/algo.tex Tiling；paper/src-v2/src/algo.tex Recomputation
