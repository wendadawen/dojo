# evidence.md：FlashKDA 与 KDA Context Parallelism 核心论断与证据

核心论断编号（C 论断 / F 公式 / N 数字）只覆盖核心内容。来源：K3 报告 §5.1.1、§5.1.2、§5.4.2。所有论断已确认，无冲突或证据不足项。

## C 论断

### C1：KDA 串行状态与 GPU 并行偏好冲突，四 regime 瓶颈不同

- **论断内容**：KDA 的串行状态更新（$S_t$ 依赖 $S_{t-1}$）与 GPU 偏好宽而均匀的并行相冲突；这个冲突在训练/prefill、长 prefill、跨设备、解码四个 regime 下表现成不同瓶颈。
- **来源定位**：K3 报告 §5.1.1，原文 "The serial dependence of the KDA state is at odds with the GPU's preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime. We design a dedicated kernel for each regime."
- **适用条件**：KDA 递归形式成立（$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$）。
- **置信状态**：已确认。

### C2：FlashKDA 重叠 intra-chunk 计算与 cross-chunk 状态传播

- **论断内容**：FlashKDA 是 CUTLASS-based 的 chunkwise kernel，把工作分解成 token-parallel stages 和 head-parallel recurrence，各自独立调度，重叠 intra-chunk 计算与 cross-chunk 状态传播；服务训练和 prefill；是 flash-linear-attention 的后端。
- **来源定位**：K3 报告 §5.1.1，原文 "We therefore develop FlashKDA [14], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation. The kernel decomposes the work into token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently, and substantially outperforms the Triton reference implementation. FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention [139]."
- **适用条件**：chunkwise 形式（chunk 内并行、chunk 间递归）成立。
- **置信状态**：已确认。

### C3：设备内 CP 把序列切到单 rank 的 SMs，无跨设备通信

- **论断内容**：TP 只切 head 不缩短递归；纯 TP 下长 prefill 每个 rank 持少量 head、SM 空闲。关键观察：每段的 state transition 可独立于入状态计算、之后精确合并。SM 级 CP planner 把序列切到单 rank 的 SMs，并行算 segment transition 再合并恢复每段精确初始状态；与 KCP 不同，完全设备内、无跨设备通信。
- **来源定位**：K3 报告 §5.1.1，原文 "Tensor parallelism partitions heads across devices but never shortens the recurrence, so under pure TP deployment, prefilling an ultra-long sequence leaves most SMs idle when each rank holds only a few heads. The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward. An automatic SM-level context-parallel (CP) planner [142, 139] therefore partitions the sequence across the SMs of a single rank, evaluates the segment transitions in parallel, and merges them to recover each segment's exact initial state. In contrast to the cross-device KCP of §5.1.2, this parallelism is entirely intra-device and incurs no cross-device communication."
- **适用条件**：段转移可独立于入状态计算并可精确合并（KDA 状态转移的线性性赋予）。
- **置信状态**：已确认。

### C4：KDA 的 $M_t$ 使本地段效果依赖入状态，不能直接求和；KCP 分解为 $M+\tilde S$，prefix scan + 一次 all-gather，通信量固定

- **论断内容**：vanilla 线性注意力递归是纯加性，各 rank 算本地状态后直接求和即可恢复入状态；KDA 的 delta rule 有 token-dependent $M_t = I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$ 作用于入状态后再加写入，本地段效果依赖进入该段的状态，不能从 $S=0$ 单独确定。KCP 把每段效果分解成"作用于入状态的累积转移 $M$"+"从零生成的状态 $\tilde S$"两个本地可算量；各 rank 一次 all-gather 交换，prefix scan 重组入状态；交换的是固定大小状态，通信量与序列长度无关，达成线性计算扩展。
- **来源定位**：K3 报告 §5.1.2，原文 "Prior context-parallel methods exploit the additive recurrence of vanilla linear attention by computing, on each rank, the state that the local tokens generate from S = 0 and summing these local states over the preceding ranks to recover the incoming state [114, 113]. This direct summation, however, is insufficient for KDA. Recall from Eq. 1 that KDA updates its state as $S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$, where $M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$. KDA's delta rule applies the token-dependent matrix $M_t$ to the incoming state before adding the current write. Consequently, the effect of a local sequence segment depends on the state entering that segment and cannot be determined from the state computed with S = 0 alone." 以及 "we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities, a cumulative transition acting on the incoming state and a state generated locally from zero" 与 "KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling."
- **适用条件**：KDA 递归成立；状态 $S$ 固定大小。
- **置信状态**：已确认。

### C5：解码缓存投影输入而非状态，片上重建接受 token 状态

- **论断内容**：解码每步原地更新状态；MTP 投机解码若拒绝部分 draft token，状态已越过最后接受 token、无法平凡回滚。存状态快照能支持回滚但状态流量在大 batch 下占主导。关键观察：任意接受前缀后的状态由 draft token 的投影输入完全决定，而投影输入比状态小得多。方案：只缓存投影输入，片上重建接受 token 状态，写回已验证和 bonus token 状态；与 ReplaySSM 并发提出；投影缓存不离开 decode 阶段，前缀缓存和 PD 分离与非投机服务同载荷。
- **来源定位**：K3 报告 §5.4.2，原文 "KDA decoding presents a distinct set of challenges: the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step. This in-place update becomes problematic in MTP-based speculative decoding: if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back. Maintaining a state snapshot for each draft position would enable rollback, but would also multiply state traffic — a cost that dominates at the large batch sizes typical of online serving. The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself. We therefore cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [25]." 以及 "Because the projection caches never leave the decode stage, prefix caching and prefill–decode disaggregation operate on the same payload as in non-speculative serving."
- **适用条件**：KDA 更新确定性（给定相同输入序列状态唯一）；MTP 投机解码。
- **置信状态**：已确认。

## F 公式

### F1：KDA 单步递归（Eq.1，引自 KDA 页）

- **公式**：$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，其中 $M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$。
- **来源定位**：K3 报告 §2.1.1 Eq.1；KDA 概念页 [F1]。
- **适用条件**：单头 KDA；多头加 $h$ 上标。
- **置信状态**：已确认。本页引用不重推。

### F2：KCP 段效果分解（Eq.17）

- **公式**：对进入第 $(i+1)$ 个 context-parallel rank 的任意入状态，经 $t$ 个本地 token 后的状态为
$$S_t^{[i+1]} = \tilde S_t^{[i+1]} + M_{t\leftarrow 1}^{[i+1]} S_T^{[i]},$$
其中累积转移 $M_{t\leftarrow 1}^{[i+1]} := \prod_{r\leftarrow 1}^{t} M_r \in \mathbb{R}^{d_k \times d_k}$，$\tilde S_t^{[i+1]}$ 是同一递归从 $S=0$ 起的状态。展开到跨 rank：
$$S_T^{[i]} = \tilde S_T^{[i]} + \sum_{j=1}^{i} \left(\prod_{l=j+1}^{i} M_{T\leftarrow 1}^{[l]}\right) \tilde S_T^{[j]}.$$
- **来源定位**：K3 报告 §5.1.2 Eq.17。
- **适用条件**：KDA 递归 Eq.1 成立；rank 内 token 顺序处理。
- **置信状态**：已确认。第一项 $\tilde S$ 是本地从零生成，第二项是入状态经各 rank 本地 $M$ 累积传播。两项在 $S_T^{[i]}$ 可用前都可本地算出，是各 rank all-gather 交换的片段。

### F3：vanilla 线性注意力加性递归（对照用）

- **公式**：$s_i = s_{i-1} + \phi(k_i) v_i^\top$（无 $M_t$ 作用）。
- **来源定位**：K3 报告 §5.1.2 所述 "additive recurrence of vanilla linear attention"；线性注意力概念页。
- **适用条件**：vanilla 线性注意力（无 delta rule 擦除、无 forget gate）。
- **置信状态**：已确认。用于对照说明"为何可直接求和而 KDA 不可"。

## N 数字

本页无外部性能数字（K3 报告 §5.1/§5.4.2 未给出可定位的加速比/带宽基准表）。涉及 KDA 配置数字（$d_k=d_v=128$、状态 $S\in\mathbb{R}^{128\times128}$ 约 32KB BF16）引自 KDA 概念页 [N5]，本页只复用其结论性引用，不重新登记为新数字。
