# Per-Head Muon 核心论断与证据

来源优先级：K3 技术报告原文 > Muon 原始博客（Keller Jordan）> 后续技术报告（Moonlight arXiv:2502.16982）。本页核心论断全部来自 K3 报告 §2.5 与 §5.2.2。

## C 论断（事实性）

### C1
- 论断内容：Muon 对注意力 Q/K/V 投影矩阵做 Newton-Schulz 正交化时，若把整个投影矩阵当单一矩阵处理，全矩阵正交化会把所有头当作一个耦合块，大梯度/动量尺度的头主导共享的更新方向，小尺度头获得"未充分正交化"的更新。
- 来源定位：K3 技术报告 §2.5 "Per-Head Muon"，原文："The intuition is that full-matrix orthogonalization treats all heads as a single coupled block, so heads with larger gradient or momentum scales dominate the shared update direction, while smaller-scale heads receive insufficiently normalized updates"。
- 适用条件：Q/K/V 投影权重沿头维度堆叠为单一矩阵、对该矩阵整体做 NS 正交化。
- 置信状态：已确认。

### C2
- 论断内容：Per-Head Muon 把 Q/K/V 投影的动量矩阵沿头维度切分，对每个头块单独做 Newton-Schulz 正交化，均衡各头更新尺度。
- 来源定位：K3 技术报告 §2.5，原文："instead of applying Newton–Schulz orthogonalization to the full Q, K, and V projection matrices, we partition their momentum matrices along the head dimension and orthogonalize each head's block separately ... per-head orthogonalization equalizes the update scale across heads"。
- 适用条件：注意力投影权重存在可沿头维度切分的块结构。
- 置信状态：已确认。

### C3
- 论断内容：按头正交化在实践中带来更均衡的学习动力学、提升大尺度训练稳定性，并略降优化器开销（NS 在更小的 per-head 高瘦块上比在全投影矩阵上便宜）。
- 来源定位：K3 技术报告 §2.5，原文："this design yields more balanced learning dynamics across heads and improves training stability at larger scales. It also slightly reduces optimizer overhead, as Newton–Schulz iterations on tall per-head blocks are cheaper than on the full projection matrix"。
- 适用条件：K3 的大尺度训练配置；具体开销节省量与头数、head_dim、model_dim 相关，报告未给数值。
- 置信状态：已确认（定性结论；数值未公开）。

### C4
- 论断内容：分布式优化器把参数均匀分片到各 DP rank，而 Muon 的 NS 正交化需要完整参数矩阵，朴素做法是每步对所有 rank 做全参数 all-gather，造成大内存占用并使通信成为大尺度下的主要瓶颈。
- 来源定位：K3 技术报告 §5.2.2 "P2P-based Muon orthogonalization"，原文："The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton–Schulz orthogonalization in Muon requires the full parameter matrix, necessitating a communication step to gather complete parameters before each update. The naive approach performs an all-gather over the entire parameter buffer on every rank, which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale"。
- 适用条件：ZeRO 式参数分片 + Muon NS 正交化。
- 置信状态：已确认。

### C5
- 论断内容：K3 采用 P2P 通信方案：每个 rank 只向持有自己所需分片的 owner rank 取回本地负责的参数分片，消除全参数缓冲区、降低内存与通信量；并把通信与计算按 model-chunk 粒度流水化隐藏开销。
- 来源定位：K3 技术报告 §5.2.2，原文："each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume. Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead"。
- 适用条件：DP 参数分片布局；具体通信量节省取决于分片与参数布局。
- 置信状态：已确认。

## F 公式（机制性）

### F1
- 公式内容：设多头注意力投影的动量矩阵 $M \in \mathbb{R}^{(H\cdot d_h)\times d}$ 是各头块沿行方向堆叠 $M = [M_1; M_2; \dots; M_H]$（$M_h \in \mathbb{R}^{d_h\times d}$），原版 Muon 对整个 $M$ 做 Newton-Schulz 正交化得到 $\mathrm{Ortho}(M)$；Per-Head Muon 对每个 $M_h$ 单独正交化，更新由 $\mathrm{Ortho}(M_1), \dots, \mathrm{Ortho}(M_H)$ 纵向拼接组成。
- 来源定位：C2 的形式化表达；切分方式由 K3 §2.5 "partition their momentum matrices along the head dimension" 直接给出，记号为本页教学标注。
- 适用条件：$M$ 的行按头顺序堆叠；每头块 $M_h$ 行数为 head_dim。
- 置信状态：已确认（切分维度与方向来自来源；具体记号为本页定义）。

### F2
- 公式内容：Newton-Schulz 正交化对矩阵 $X$ 的作用等价于把 $X = U S V^\top$ 的所有奇异值拉平为 1，得到半正交矩阵 $U V^\top$（记号 $\mathrm{Ortho}(X) \approx U V^\top$）。
- 来源定位：Muon 原始博客（Keller Jordan）"Newton-Schulz iteration" 段，原文将正交化目标定义为 $\arg\min_O \|O - G\|_F$ s.t. $O$ 半正交，等价于 SVD 后取 $U V^\top$。K3 §2.5 默认沿用 Muon 的 NS 正交化定义。
- 适用条件：$X$ 已按 Frobenius 范数归一化以保证奇异值在 $[0,1]$；NS 迭代步数足够（Moonlight 实现为 5 步）。
- 置信状态：已确认（NS 等价 SVD 取 $UV^\top$ 来自 Muon 原始博客；K3 沿用 Muon NS）。

### F3（教学示例，非来源公式）
- 公式内容：教学示例。设 $H=2, d_h=1, d=2$，两头动量行向量平行同向、尺度差 10 倍：$M_1=[3,4]$（范数 5），$M_2=[0.3,0.4]$（范数 0.5）。全矩阵 $M=[3,4;0.3,0.4]$ 是秩 1 矩阵，唯一非零奇异值 $\sigma_1=\sqrt{5^2+0.5^2}=\sqrt{25.25}\approx 5.025$，正交化结果 $\mathrm{Ortho}(M)=UV^\top \approx [0.995\cdot \hat{u}; 0.0995\cdot \hat{u}]$（$\hat u=[0.6,0.8]$），两头行块范数比约为 10:1，小尺度头行块范数 $\approx 0.0995$ 远小于 1；按头正交化后 $\mathrm{Ortho}(M_1)=\mathrm{Ortho}(M_2)=[0.6,0.8]$，两头行块范数都为 1。
- 来源定位：本页教学构造，数字人为构造；机制依据为 F1、F2。
- 适用条件：两头平行同向的极端情形，用于展示"尺度耦合"被压低的最简形态；非平行情形方向也会被耦合，本例不展示方向耦合。
- 置信状态：教学示例（机制正确，数字便于手算）。

## N 数字（外部数字）

本页无外部实验数字。K3 报告对 per-head Muon 只给定性结论（C3），未给"per-head 单独贡献"的消融数值；Moonlight 报告的 ~2× 计算效率是整体 Muon 相对 AdamW 的结果，不归因于 per-head，本页不引用以避免误导。
