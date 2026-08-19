# 模型并行（model-parallelism）核心论断与证据

## C 论断（机制与结论）

- C1：模型并行把单个模型切分到多块 GPU，两条基本路径为张量并行（TP，按层内矩阵维度切）与流水线并行（PP，按层堆叠深度切）；Megatron-LM 将层内并行与流水线并行并列为两类模型并行。来源：Megatron-LM §1（"This approach is orthogonal to pipeline-based model parallelism as advocated by approaches such as GPipe"）；GPipe §1。适用条件：通用。置信：已确认。
- C2：TP 对 MLP 块的第一个 GEMM 按权重矩阵列切分、第二个 GEMM 按行切分；该组合使 GeLU 可本地逐块应用、第二个 GEMM 本地乘加，前向只需在 MLP 输出处 1 次 all-reduce。来源：Megatron-LM §2.2 MLP 段（"we partition the first GEMM in this column parallel fashion and split the second GEMM along its rows ... requires only a single all-reduce operation in the forward pass"）。适用条件：标准两层 MLP + GeLU 结构。置信：已确认。
- C3：TP 对 self-attention 的 Q/K/V 投影按列切分，使每个 attention head 的计算完整落在一张卡上；输出线性层按行切分直接消费本地 attention 输出；attention 块前向 1 次 all-reduce。来源：Megatron-LM §2.2 Self-Attention 段（"partitioning the GEMMs associated with key (K), query (Q), and value (V) in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU"）。适用条件：多头注意力。置信：已确认。
- C4：一个标准 Transformer 层的 TP 前向共 2 次 all-reduce（attention 一次 + MLP 一次）；反向各 2 次。来源：Megatron-LM §2.2（"using only two all-reduces in the forward path and two in the backward path"）。适用条件：Megatron-LM 切分方案。置信：已确认。
- C5：PP 把模型按层序切成连续分区（stage/cell），micro-batch 依次流入各分区流水执行；相邻分区之间只需传递一次激活张量。来源：GPipe §2（"consecutive groups of layers can be partitioned into cells. Each cell is then placed on a separate accelerator ... micro-batches"）；"we only need to pass activation tensors at the partition boundaries"。适用条件：通用。置信：已确认。
- C6：PP 引入气泡：一个 step 内各分区有空闲等待时间；气泡时间占比为 $(p-1)/(m+p-1)$，$p$ 为分区数、$m$ 为 micro-batch 数；GPipe 实验发现 $m \ge 4p$ 时气泡可忽略。来源：GPipe §2.2 Performance Optimization 段（"This bubble time is O((K-1)/(M+K-1)) amortized over the number of micro-steps M. ... bubble overhead to be negligible when M ≥ 4 × K"）。适用条件：分区负载均衡、同步执行。置信：已确认。
- C7：PP 的通信只需在分区边界传激活张量，因此不需要高速互联也能有效扩展。来源：GPipe §2.2（"GPipe also introduces low communication overhead ... We can achieve efficient scaling performance even on accelerators without high-speed interconnects"）。适用条件：通用。置信：已确认。
- C8：TP 需要高带宽互联才能有效扩展：TP 可在 DGX A100 的 8 GPU NVLink 上部署、Pope et al. 展示在专用 TPUv4 pod 上扩到 256 设备，但缺乏高速互联（hyper-cluster）时大规模 TP 性能差。来源：Sarathi §1（"tensor-parallelism can enable deployment of an LLM on up to 8 GPUs ... Pope et al. show that tensor parallelism can be scaled up to 256 devices on specialized TPUv4 pods. However, tensor-parallelism at such a large scale can result in poor performance when hyper-clusters are unavailable"）。适用条件：A100 级互联与 TPU pod 语境。置信：已确认。
- C9：TP 与 PP 可正交组合，总卡数 = TP × PP（× EP）；DeepSeek-R1 在 64 卡上用 EP × PP = 64 的组合是此类组合记号的实例。来源：Beyond the Buzz §4 图 5 说明（"EP × PP = 64"）。前半句（正交组合）为推断：依据 Megatron-LM §2.4 与各系统组合维度记号的通行用法。置信：前半推断（组合正交性，依据充分）、后半已确认（EP×PP=64 出现在 Beyond the Buzz 原文）。
- C10：模型并行与数据并行（DP）不同：DP 每卡持有一份完整模型、处理不同数据；模型并行每卡只持有部分模型、协作处理同一份数据。来源：通用教材级结论；正文作概念区分，不引具体实验。适用条件：通用。置信：已确认。

## F 公式

- F1：TP 后 MLP 两段本地计算：$Y = [Y_1, Y_2] = [\mathrm{GeLU}(XA_1), \mathrm{GeLU}(XA_2)]$，第二 GEMM 行切后本地部分积 $XA_1A_1' + XA_2A_2'$ 需 all-reduce 求和得最终输出。来源：Megatron-LM Eq.(2)(3) 与 Figure 3a。置信：已确认。
- F2：GPipe 气泡占比 $\frac{p-1}{m+p-1}$。来源：GPipe §2.2。置信：已确认。推导以 micro-batch slot 图讲清 $(m+p-1)$ 的来源：首个 micro-batch 走完全程需要 $p$ 个 slot，其后每个 micro-batch 每 slot 出一个，共 $m$ 个 micro-batch 排空需要额外 $p-1$ 个 slot。

## N 数字

- N1：GPipe 气泡在 $m \ge 4p$ 时可忽略（$m$ 为 micro-batch 数，$p$ 为分区数）。来源：GPipe §2.2。适用条件：GPipe 训练实验。置信：已确认。
- N2：TP 单层前向 2 次 all-reduce、反向 2 次；多层堆叠后随层数线性增长。来源：Megatron-LM §2.2。适用条件：Megatron-LM 方案。置信：已确认。

## 原文图候选

本页为概念页，机制图自绘（TP 的层内切分示意、PP 的流水线 slot 图），不使用论文原图。

## 构造示例（不入 C/F/N）

- 手算 TP FFN 切分：设 $X$ 为 $1\times 4$ 输入、$A$ 为 $4\times 4$、两卡列切 $A=[A_1, A_2]$（各 $4\times 2$）、第二矩阵 $B=[B_1; B_2]$ 行切（各 $2\times 4$），验证 $\mathrm{GeLU}(XA)B = [\mathrm{GeLU}(XA_1), \mathrm{GeLU}(XA_2)] \cdot [B_1; B_2]$ 两侧分别等于本地部分积之和。数字自设，标注构造示例。
- 手算 PP 气泡：$p=4$ 个 stage、$m=4$ 个 micro-batch，总 slot 数 $m+p-1=7$，气泡 slot 每 stage $p-1=3$，占比 $3/7\approx 0.43$；$m$ 提高到 16 时占比 $3/19\approx 0.16$。数字自设，标注构造示例（比例结构来自 F2）。
