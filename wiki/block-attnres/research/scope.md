# Block AttnRes：内容范围

## 1.1 概念含义

- **概念名称**：Block Attention Residuals（块注意力残差）
- **英文名称**：Block Attention Residuals；常见缩写：Block AttnRes
- **一句话定义**：把网络深度方向的"等权累加残差"替换为"块级 softmax 加权检索"——每层用可学习 pseudo-query 对若干块级表征做注意力，选择性汇总此前块的信息，把逐层残差流变成块间可寻址的深度记忆。
- **正式定义**：设网络共 $L$ 层，分为 $N$ 个 block、每个 block $S=L/N$ 层。第 $l$ 层的输出为 $f_l(h_l)$。Full Attention Residuals（先定义它的完整形式）对第 $l$ 层定义可学习 pseudo-query $q_l=w_l\in\mathbb{R}^d$，keys 与 values 取 $k_i=v_i=h_1\ (i=0)$ 与 $k_i=v_i=f_i(h_i)\ (1\le i\le l-1)$，权重与输出按
  $$\alpha_{i\to l}=\frac{\phi(q_l,k_i)}{\sum_{j=0}^{l-1}\phi(q_l,k_j)},\qquad h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i,\qquad \phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k)).$$
  Block Attention Residuals 在此基础上把每个 block 内的层输出先求和成单表征 $b_n=\sum_{j\in B_n}f_j(h_j)$，再把 attention 作用在 block 级表征 $\{b_0,b_1,\dots,b_{n-1},b_n^{i-1}\}$ 上（$b_0=h_1$ 为 embedding，$b_n^{i-1}$ 为当前 block 前 $i-1$ 层的 partial sum）；内存从 $O(Ld)$ 降到 $O(Nd)$。该定义与 Kimi K3 Technical Report §2.2 Eq.(8)(9)(10) 一致；K3 的具体配置（$N=8$、$S=12$）与 HuggingFace 官方 `config.json` 中 `attn_res_block_size=12` 一致。
- **本文采用的语境**：深度神经网络（特指 Transformer 风格的堆叠主干）中"如何让信息跨越很多层流动"这一问题的解决方案。语境限定在层间信息流机制，不展开 token 维度或 channel 维度的混合。

### 包括什么

| 纳入项 | 纳入理由 |
|---|---|
| 标准残差在深度方向的瓶颈（RNN-over-depth 类比） | 概念的动机来源，回答"为什么需要它" |
| Full AttnRes 的定义：pseudo-query、keys/values、softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$ | 概念的核心公式（Eq.8、Eq.9），缺则无法回答"是什么" |
| Block AttnRes 的分块、块内求和 $b_n$、块间 attention、候选集合 $[b_0,\dots,b_{n-1},b_n^{i-1}]$ | 把 $O(Ld)$ 降到 $O(Nd)$ 的具体机制（Eq.10） |
| K3 的具体配置：$N=8$、$S=12$、9 个候选来源、每层 attention 前/MLP 前各加权、模型末尾第三次加权 | 把抽象公式落到一个真实模型上，避免概念悬空 |
| softmax kernel 中 RMSNorm 的作用：防止幅值大的层主导权重 | 公式中的一个关键设计选择，影响机制结论 |
| Full AttnRes 与 Block AttnRes 的对比 | 区分两个层次，避免把分块优化当成机制本身 |
| 可手算的小块数字例子（$N=3$、$S=2$、$d=2$） | 让公式可追踪，验证机制而非黑箱接受 |

### 不包括什么

| 排除项 | 排除理由 |
|---|---|
| KDA、Gated MLA、Stable LatentMoE 等 K3 其他模块的内部机制 | 各自是独立概念，本页只在配置表中使用它们的存在，不展开 |
| Pipeline parallelism 下的 cache-based pipeline communication | 工程优化，不影响 Block AttnRes 的机制理解 |
| `online softmax` 的两阶段 kernel 实现（prefill/decode 优化） | 工程实现细节，影响吞吐但不改变数学定义 |
| 训练时 checkpointing 与激活保存策略 | 工程细节，不影响概念 |
| AttnRes 在 K3 之外的其他模型（如 nano-kpu 的 block size=2）的应用 | 扩展内容，不纳入本页范围 |
| 序列方向的标准 attention 与 Flash Attention | 独立概念；本页 attention 只用作"深度方向"的类比 |
| MTP / EAGLE-3 草稿模型如何复用 AttnRes 中间特征 | 属于推理系统范畴，不影响概念本身 |

### 相邻概念

| 相邻概念 | 关键区别 | 是否纳入 |
|---|---|---|
| 标准残差连接 $h_{l+1}=h_l+F(h_l)$ | 等权累加，所有历史压成单一流；AttnRes 用注意力加权选择性检索 | 不纳入，作为前置概念引用 `wiki/residual-connection/` |
| 序列方向的自注意力 | 作用在 token 维度（同一层、不同位置）；AttnRes 作用在深度维度（同一位置、不同层） | 不纳入，仅在动机处作为类比一句话提及 |
| DenseNet 稠密连接 | 把前层输出拼接而非加权求和，通道数随层数增长；AttnRes 用 softmax 加权且通道数不变 | 不纳入，仅在边界处一句话提及 |
| Highway Network 门控捷径 | 捷径上有标量门控；AttnRes 用 softmax 向量权重且对所有来源重新分配 | 不纳入，仅在边界处一句话提及 |
| Memory-efficient AttnRes（§5.2.2）与 Block AttnRes kernel（§5.4.2） | 是 Block AttnRes 在 K3 训练/推理时的工程优化，不改变公式 | 不纳入正文，在来源说明中标注 |

## 1.2 学习目标

### Q1：标准残差在深度方向上为什么是瓶颈？AttnRes 用什么思路替代？

- **完成答案**：读者应能说明标准残差把所有前层信息等权压进单一流 $h_l$，深网络中早期信息会被层层稀释或覆盖，类似 RNN 在时间维度的瓶颈；AttnRes 把"沿深度做累加"替换为"沿深度做 attention"——每层用可学习 pseudo-query 对此前所有层的输出做 softmax 加权检索，让模型按内容选择要回看哪些层。
- **为什么是核心目标**：不理解这个动机就无法理解为什么公式里要用 softmax 加权而不是简单的加法。
- **依赖内容**：标准残差连接的公式与"等权累加"性质（前置概念页 `wiki/residual-connection/`）；softmax 加权的基本直觉。

### Q2：Full AttnRes 的公式是什么？pseudo-query、keys/values、softmax kernel 各代表什么？

- **完成答案**：读者应能写出 Eq.(8)(9)：$q_l=w_l\in\mathbb{R}^d$ 是层 $l$ 自带的可学习 pseudo-query（无输入依赖），keys 与 values 取自 embedding $h_1$ 与此前各层输出 $f_i(h_i)$；权重 $\alpha_{i\to l}=\mathrm{softmax}_i\,\phi(q_l,k_i)$，输出 $h_l=\sum_i\alpha_{i\to l}v_i$；softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$ 让 pseudo-query 与归一化后的 key 做内积再取 exp，RMSNorm 防止幅值大的层主导 softmax。
- **为什么是核心目标**：这是概念的正式定义，符号不理解则 Block AttnRes 的分块无从展开。
- **依赖内容**：softmax 的基本定义；RMSNorm 的"按均方根归一化"作用（页面内会给出最小说明）；向量内积。

### Q3：Block AttnRes 如何分块、块内求和、块间 attention？内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源是什么？

- **完成答案**：读者应能说明 Block AttnRes 把 $L$ 层划分为 $N$ 个 block、每个 block $S=L/N$ 层；block $n$ 内的层输出求和成单表征 $b_n=\sum_{j\in B_n}f_j(h_j)$（$b_0=h_1$ 为 embedding）；对 block $n$ 的第 $i$ 层，候选集合为 $[b_0,\dots,b_{n-1}]$（$i=1$）或 $[b_0,\dots,b_{n-1},b_n^{i-1}]$（$i\ge 2$），其中 $b_n^{i-1}$ 是当前 block 的 partial sum；attention 仍按 Eq.(8)(9) 计算，但作用在 block 级表征上。内存从 $O(Ld)$ 降到 $O(Nd)$ 的来源是：只需保留 $N$ 个 block 级表征而非 $L$ 个层输出，且跨 pipeline stage 通信量同比例下降。
- **为什么是核心目标**：Block AttnRes 是 K3 实际采用的形态，也是概念名中"Block"的来源，必须讲清分块带来的变化与代价。
- **依赖内容**：Q2 的 Full AttnRes 公式；求和与 attention 的可交换性（block 内求和后再 attention 等价于对块内层输出做均匀加权的 attention，本页只陈述结论不作推导）。

### Q4：K3 的具体配置是什么——8 块×12 层、9 个候选来源、加权三次的位置在哪里？

- **完成答案**：读者应能说出 K3 主干 93 层按 `attn_res_block_size=12` 分为 8 个 block（最后一个 block 因 93÷12 非整除而是 partial block），加上 embedding 共 9 个 block 级表征；对最后一个 block 内的层（$i\ge 2$），候选来源 = 8 个块快照（$b_0$ 即 embedding + 7 个完整 block 的 $b_1\dots b_7$）+ 当前 block 的 partial sum $b_8^{i-1}$ = 9 个。K3 在每个 attention 子层前与每个 MLP 子层前各做一次 AttnRes 加权（两套独立可学习参数），模型末尾 final norm 前再做第三次（output AttnRes）。
- **为什么是核心目标**：把抽象公式落到一个真实模型，让读者区分"机制本身"与"机制在 K3 中的具体实例化"。
- **依赖内容**：Q3 的分块机制；K3 主干 93 层、`attn_res_block_size=12`（来源：官方 `config.json`）；K3 每个 decoder layer 含一个 attention 子层 + 一个 LatentMoE 子层（来源：K3 报告 §2、Figure 2）。

### Q5：softmax kernel 为什么用 $\exp(q^\top\mathrm{RMSNorm}(k))$——RMSNorm 防大值主导的具体含义和边界？

- **完成答案**：读者应能说明若 key 直接做内积 $q^\top k$，幅值大的层（如某个 $f_i(h_i)$ 模长很大）会让 $q^\top k_i$ 远大于其他项，softmax 后权重几乎集中在该层，等于"屏蔽"了其他来源；RMSNorm 把每个 key 归一化到单位 RMS 后再内积，让 pseudo-query 按方向而非幅值选择来源，权重更平滑。边界：RMSNorm 不改变 key 的方向，只去掉幅值；若所有 key 方向接近，softmax 仍会接近均匀；RMSNorm 也不保证数值稳定（exp 仍可能溢出，工程实现通常配合减最大值）。
- **为什么是核心目标**：公式中 RMSNorm 是一个看似细节但影响机制结论的设计选择，不讲清会被读者误以为是"装饰性归一化"。
- **依赖内容**：Q2 的 softmax kernel；RMSNorm 的定义（页面内给最小说明）；softmax 对大值的敏感性。

## 1.3 内容分级

### 核心内容

| 核心内容 | 对应目标 | 必须讲清的结论 |
|---|---|---|
| 标准残差的深度瓶颈（RNN-over-depth 类比） | Q1 | 等权累加把历史压成单一流，深网络中早期信息被稀释 |
| AttnRes 的核心思路：用 attention 替代等权累加 | Q1 | 沿深度做 softmax 加权检索 |
| Full AttnRes 公式 Eq.(8)(9) | Q2 | pseudo-query $q_l=w_l$、keys=values、softmax kernel |
| softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$ | Q2、Q5 | 内积 + exp，RMSNorm 防大值主导 |
| Block AttnRes 的分块与块内求和 $b_n$ | Q3 | $L$ 层分 $N$ 块，块内求和成单表征 |
| 块间 attention 的候选集合 $[b_0,\dots,b_{n-1},b_n^{i-1}]$ | Q3 | Eq.(10) 的两种情况 |
| 内存从 $O(Ld)$ 降到 $O(Nd)$ | Q3 | 只保留 $N$ 个块级表征 |
| K3 的 $N=8$、$S=12$、9 个候选来源 | Q4 | 配置数值与候选集合的具体实例 |
| K3 加权三次的位置：attention 前、MLP 前、模型末尾 | Q4 | 每层两次 + 末尾一次 |
| Full vs Block AttnRes 的对比 | Q2、Q3 | Block 是 Full 的工程可行化，公式形式相同 |
| RMSNorm 防大值主导的机制 | Q5 | 按方向而非幅值选择 |

### 辅助内容

| 辅助内容 | 服务的核心内容或误解 |
|---|---|
| 与序列方向 attention 的类比（"沿深度做 attention"） | 服务 Q1：用读者熟悉的序列 attention 类比深度方向 |
| 标准残差与 AttnRes 的对照表 | 服务 Q1、Q2：让两种残差形式并排对比 |
| 小块数字例子 $N=3$、$S=2$、$d=2$ 的逐步追踪 | 服务 Q2、Q3、Q5：让公式可手算 |
| 论文 §2.2 中 "$N\approx 8$ recovers most of the benefit" 的经验结论 | 服务 Q3、Q4：解释为什么 K3 选 $N=8$ |
| 论文 §2.2 末尾 "final output layer aggregates all N block representations" | 服务 Q4：解释模型末尾第三次加权 |

### 扩展内容

| 扩展内容 | 纳入或排除 |
|---|---|
| AttnRes 原论文 [57]（Kimi Team, 2026 preprint）中关于 $N$ 的消融实验 | 排除：原 preprint 未在本页参考资料中完整获取，仅引用 K3 报告 §2.2 的转述 |
| nano-kpu 设计中 block size=2 的应用 | 排除：另一模型实例，不影响 K3 的理解 |
| Block AttnRes kernel 的两阶段 schedule（inter-block + intra-block） | 排除：工程实现，不影响数学定义 |
| Pipeline parallelism 下的 cache-based pipeline communication | 排除：分布式系统细节 |

## 1.4 前置知识映射

| 前置知识 | 被哪些学习目标依赖 | 概念页链接或生成状态 | 递归深度 |
|---|---|---|---|
| 残差连接（$h_{l+1}=h_l+F(h_l)$、等权累加、退化问题） | Q1、Q2、Q3 | 已有：[`wiki/residual-connection/`](../../residual-connection/index.html) | 0 |
| softmax（把任意实数分数转成概率分布） | Q2、Q3、Q5 | 未生成；页面内给最小说明（一段话 + 一个小检查），不递归生成 | — |
| RMSNorm（按均方根归一化） | Q2、Q5 | 未生成；页面内给最小说明（一段话 + 一个小检查），不递归生成 | — |
| 自注意力（序列方向的 $Q,K,V$） | Q1（作为类比） | 未生成；页面只用一句话类比，不展开 | — |

说明：softmax 与 RMSNorm 不作为独立概念页生成，原因有二：(1) 它们在本文中只作为公式零件使用，最小说明即可支撑理解；(2) plan.md 规定递归深度最多 2 层，但这两个概念在本页的用途足够局部，不需要独立页面。若读者要深入学习，页面会在文末来源说明中给出标准出处。

## 1.5 明确不展开的内容

| 不展开项 | 与概念的关系 | 不展开的原因 |
|---|---|---|
| KDA / Gated MLA / Stable LatentMoE 的内部机制 | K3 的其他主干模块，与 AttnRes 同级 | 各自是独立概念；本页只在配置表中使用它们的存在 |
| Block AttnRes kernel 的两阶段 schedule（§5.4.2） | Block AttnRes 在 K3 中的工程实现 | 影响吞吐但不改变数学定义；属于工程优化范畴 |
| Memory-efficient AttnRes 的 checkpointing 策略（§5.2.2） | Block AttnRes 训练时的显存优化 | 影响训练显存但不改变公式 |
| Pipeline parallelism 下的 cache-based pipeline communication | Block AttnRes 跨 stage 的通信优化 | 属于分布式系统细节 |
| AttnRes 原论文 [57] 的完整消融实验 | Block size $N$ 的选择依据 | 原 preprint 未完整获取；本页只引用 K3 报告 §2.2 的转述（"$N\approx 8$ recovers most of the benefit"） |
| MTP / EAGLE-3 草稿模型如何复用 AttnRes 中间特征 | AttnRes 输出的下游应用 | 属于推理系统范畴，不影响概念本身 |

## 1.6 常见误解和适用边界

### 常见误解

| 误解 | 正确结论 | 形成原因 | 影响目标 |
|---|---|---|---|
| "AttnRes 替代了标准残差，K3 不再有 $h_{l+1}=h_l+F(h_l)$" | K3 仍保留标准残差的等权累加（每层 prefix_sum 逐层加 attn 输出与 MLP 输出）；AttnRes 是叠加在等权累加之上的"块级加权选择"，把送入 norm 的输入从"当前残差流"替换为"块级加权检索结果" | 把"替代等权累加"误读为"取消等权累加" | Q1、Q4 |
| "Block AttnRes 的 9 个候选是固定的，对每一层都一样" | 候选数随 block index 增长：对 block $n$ 的第 1 层有 $n$ 个候选（$[b_0,\dots,b_{n-1}]$），第 $i\ge 2$ 层有 $n+1$ 个；9 是最后一个 block 内 $i\ge 2$ 层的最大值 | 把 K3 实现中"预分配 9 个槽位"误当成"每层都用 9 个候选" | Q3、Q4 |
| "pseudo-query $q_l$ 是该层输入的函数" | $q_l=w_l$ 是层 $l$ 自带的可学习参数向量，不依赖该层输入；它与 key 的内积决定权重，但不消费 $h_l$ | 把 pseudo-query 误当成标准 attention 中的 $Q=W_Q x$ | Q2 |
| "RMSNorm 在 kernel 里只是为了数值稳定" | RMSNorm 的主要作用是按方向而非幅值选择来源，防止幅值大的层主导 softmax；数值稳定是副作用（且 exp 仍可能溢出，工程上还要减最大值） | 把"归一化"笼统理解为"防溢出" | Q5 |
| "Block AttnRes 就是把层分组后做组间 attention，块内还是标准残差" | 块内层输出会累加成 partial sum $b_n^{i-1}$ 作为"当前流"参与 attention；块内并非单纯标准残差，而是"标准残差累加 + 块内 partial sum 作为额外候选" | 把"块内求和"误读为"块内回到标准残差" | Q3 |

### 适用边界

- **AttnRes 解决的问题**：深度方向的信息稀释——让深网络每层能按内容回看此前块级表征，而不是只能依赖被等权累加压扁的当前残差流。
- **AttnRes 不解决的问题**：不解决 token 维度的长程依赖（由 KDA / Gated MLA 负责）、不解决 channel 维度的稀疏混合（由 Stable LatentMoE 负责）、不解决梯度消失（标准残差连接已经基本解决，见前置概念页）。
- **Block AttnRes 成立的条件**：网络深度 $L$ 足够大，使得 $O(Ld)$ 的内存/通信成为瓶颈；若 $L$ 很小（如 $L<20$），Full AttnRes 的开销本就可承受，分块带来的节省有限。
- **$N\approx 8$ 的经验结论**：来自 K3 报告 §2.2 对原 AttnRes 工作 [57] 的转述，在 K3 的模型尺度下成立；不保证对所有模型尺度都是最优。
- **K3 加权三次的具体位置**：每层 attention 前、MLP 前、模型末尾 final norm 前；这是 K3 的实例化（基于官方源码核对，参见 `wiki/kimi-k3-dataflow/`），AttnRes 公式本身（Eq.8-10）不规定加权次数，原 preprint [57] 的设计可由模型自行决定。

## 1.7 完成条件自检

- 概念歧义已处理：AttnRes 与 Block AttnRes 在论文中明确区分（§2.2 分两段），无歧义。
- 5 个学习目标互不重复：Q1 动机、Q2 Full 公式、Q3 Block 机制与内存、Q4 K3 实例化、Q5 RMSNorm 设计选择。
- 每项核心内容对应至少一个学习目标。
- 每项前置知识映射到概念页或说明不递归生成的理由。
- 误解和边界具体可查（5 条误解 + 5 条边界）。
