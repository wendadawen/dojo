# Block AttnRes：核心论断与证据

来源优先级：原始论文与标准 > 权威教材与同行评审综述 > 对应版本的官方文档 > 固定版本或 commit 的官方源码。本页主要来源为 K3 技术报告（§2.2、§5.2.2、§5.4.2）与 HuggingFace 官方 `config.json`；AttnRes 原始 preprint [57] 未完整获取，相关结论只用 K3 报告的转述，并标注。

## C 论断（机制与结论）

### C1
- **论断内容**：标准残差连接把所有前层信息等权压进单一流 $h_l$，深网络中类似 RNN 在时间维度的瓶颈；AttnRes 把"沿深度做累加"替换为"沿深度做 attention"。
- **来源定位**：K3 报告 §2.2 第 379-383 行："Standard residual connections [43] compress all prior information into a single state $h_l$ over depth — a bottleneck reminiscent of RNNs over time. ... Attention Residuals (AttnRes) [57] applies the same methodology to depth: each layer selectively retrieves representations from all preceding layers rather than accumulating them uniformly."
- **适用条件**：深网络（$L$ 较大）下成立；浅网络中瓶颈不明显。
- **置信状态**：已确认。

### C2
- **论断内容**：Full AttnRes 对每层 $l$ 定义可学习 pseudo-query $q_l=w_l\in\mathbb{R}^d$，keys 与 values 取 $k_i=v_i=h_1\ (i=0)$ 与 $k_i=v_i=f_i(h_i)\ (1\le i\le l-1)$；权重与输出按 Eq.(9) 计算，softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$。
- **来源定位**：K3 报告 §2.2 Eq.(8)(9)，第 385-396 行。
- **适用条件**：网络深度 $L<100$ 时 $O(L^2 d)$ 算力可承受；实际开销瓶颈是 $O(Ld)$ 内存与跨 stage 通信。
- **置信状态**：已确认。

### C3
- **论断内容**：softmax kernel 中 RMSNorm 的作用是防止幅值大的层主导权重。
- **来源定位**：K3 报告 §2.2 第 390-392 行："The attention weights follow a softmax kernel $\phi(q,k)=\exp(q^\top\mathrm{RMSNorm}(k))$ [55, 146], where the RMSNorm prevents layers with large-magnitude outputs from dominating the weights."
- **适用条件**：key 来自不同层、幅值可能差异较大时成立；若所有 key 幅值接近则 RMSNorm 不改变结果。
- **置信状态**：已确认。

### C4
- **论断内容**：Block AttnRes 把 $L$ 层分为 $N$ 个 block、每个 $S=L/N$ 层；block $n$ 内的层输出求和成单表征 $b_n=\sum_{j\in B_n}f_j(h_j)$，$b_0=h_1$ 为 embedding；对 block $n$ 的第 $i$ 层，候选集合按 Eq.(10) 取 $[b_0,\dots,b_{n-1}]$（$i=1$）或 $[b_0,\dots,b_{n-1},b_n^{i-1}]$（$i\ge 2$），keys 与权重仍按 Eq.(8)(9)。
- **来源定位**：K3 报告 §2.2 Eq.(10)，第 401-413 行。
- **适用条件**：$L$ 可被近似均分为 $N$ 个 block；最后一个 block 可为 partial block。
- **置信状态**：已确认。

### C5
- **论断内容**：Block AttnRes 把内存与通信开销从 $O(Ld)$ 降到 $O(Nd)$。
- **来源定位**：K3 报告 §2.2 第 415-417 行："Under Block AttnRes, memory and communication overhead drop from $O(Ld)$ to $O(Nd)$."
- **适用条件**：只比较"层输出保留"与"块级表征保留"的内存；不含其他激活（如 attention 中间态）。
- **置信状态**：已确认。

### C6
- **论断内容**：$N\approx 8$ 在多数模型尺度下能恢复 Full AttnRes 的大部分收益；K3 把 93 层划分为 8 个 block、每个 12 层，最后一个 block 为 partial block，加上 embedding 共 9 个 block 级表征。
- **来源定位**：K3 报告 §2.2 第 418-419 行："Empirically, $N\approx 8$ recovers most of the benefit across model scales [57]; for Kimi K3, we partition its layers into 8 blocks with 12-layer size, giving a partial final block and 9 total blocks when counting the embedding layer."
- **适用条件**：$N\approx 8$ 是经验结论（来自 [57]，本页未获取原 preprint 完整内容，仅引用 K3 报告转述）；9 个 block 级表征是 K3 的具体实例。
- **置信状态**：已确认（K3 报告原文）；$N\approx 8$ 的最优性证据不足（原 preprint 未获取），仅作为经验引用。

### C7
- **论断内容**：模型末尾的输出层会聚合所有 $N$ 个 block 表征。
- **来源定位**：K3 报告 §2.2 第 414 行："The final output layer then aggregates all N block representations."
- **适用条件**：K3 的具体实例化；公式 Eq.(8-10) 不强制要求末尾聚合，是 K3 的设计选择。
- **置信状态**：已确认。

### C8
- **论断内容**：K3 在每个 attention 子层前与每个 MLP（LatentMoE）子层前各做一次 AttnRes 加权（两套独立可学习参数），模型末尾 final norm 前再做第三次（output AttnRes）。
- **来源定位**：K3 报告 §2 第 196-198 行："Attention Residuals (AttnRes) use learned pseudo-queries (w) to derive attention weights (α) over the embedding and preceding block outputs" + §2 第 209-210 行："AttnRes [57] enable each module to selectively retrieve representations from the embedding, the current block, and preceding blocks"。K3 实现细节（每层 attention 前、MLP 前、末尾 final norm 前）来自 `wiki/kimi-k3-dataflow/` 对官方源码 `modeling_kimi_linear.py` 的核对（该页面 2026-08-07 按源码修订）。本页未直接复核源码，标注为间接来源。
- **适用条件**：K3 的具体实例化；AttnRes 公式本身不规定加权次数。
- **置信状态**：K3 报告原文已确认"each module"（即 attention 模块与 MLP 模块各一次）+ 末尾聚合；具体"attention 前 / MLP 前 / final norm 前"三次的位置来自 dataflow 页面对源码的核对，标注为间接证据，需在页面中明确说明。

### C9
- **论断内容**：K3 主干 93 层、`attn_res_block_size=12`、hidden_size=7168、num_attention_heads=96。
- **来源定位**：HuggingFace 官方 `config.json`（`huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`）：`num_hidden_layers=93`、`attn_res_block_size=12`、`hidden_size=7168`、`num_attention_heads=96`；K3 报告 Table 1（第 742 行 #Layers=93）。
- **适用条件**：K3 Instruct 模型版本。
- **置信状态**：已确认。

## F 公式（核心公式与来源）

### F1
- **公式**：Full AttnRes 的 keys/values 定义
  $$k_i=v_i=\begin{cases}h_1 & i=0\\ f_i(h_i) & 1\le i\le l-1\end{cases}$$
- **来源**：K3 报告 §2.2 Eq.(8)，第 385-389 行。
- **置信状态**：已确认。

### F2
- **公式**：softmax kernel 与权重
  $$\phi(q_l,k_i)=\exp(q_l^\top\mathrm{RMSNorm}(k_i)),\qquad \alpha_{i\to l}=\frac{\phi(q_l,k_i)}{\sum_{j=0}^{l-1}\phi(q_l,k_j)},\qquad h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i.$$
- **来源**：K3 报告 §2.2 Eq.(9)，第 393-396 行。
- **置信状态**：已确认。

### F3
- **公式**：Block AttnRes 的块内求和
  $$b_n=\sum_{j\in B_n}f_j(h_j),\qquad b_n^i=\sum_{j\in B_n,\,j\le i}f_j(h_j),\qquad b_0=h_1.$$
- **来源**：K3 报告 §2.2 第 401-404 行（$b_n$ 与 $b_n^i$ 的文字描述；公式形式由文字推出）。
- **置信状态**：已确认（$b_n$ 为 block 完整求和；$b_n^i$ 为 partial sum，K3 报告原文用 $b_n^i$ 表示"partial sum over the first $i$ layers of the block"，本页采用此记号）。

### F4
- **公式**：Block AttnRes 的候选集合
  $$V=\begin{cases}[b_0,b_1,\dots,b_{n-1}]^\top & i=1\ (\text{block } n\text{ 的第一层})\\ [b_0,b_1,\dots,b_{n-1},b_n^{i-1}]^\top & i\ge 2\ (\text{block } n\text{ 的后续层})\end{cases}$$
- **来源**：K3 报告 §2.2 Eq.(10)，第 407-413 行。
- **置信状态**：已确认。

### F5
- **公式**：RMSNorm 定义（页面内最小说明用）
  $$\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2+\epsilon}}\approx\frac{x}{\|x\|_2/\sqrt d}.$$
- **来源**：Zhang & Sennrich 2019（RMSNorm 原文）；K3 报告引用 [146]。
- **置信状态**：已确认。本页只用定义，不展开推导。

## N 数字（外部数字与实验条件）

### N1
- **数字**：K3 主干 93 层（69 KDA + 24 MLA），分 8 个 block、每个 12 层，最后一个 block 为 partial block；加上 embedding 共 9 个 block 级表征。
- **来源**：K3 报告 §2.2 第 418-419 行 + Table 1 第 742、757 行；HuggingFace `config.json`：`num_hidden_layers=93`、`attn_res_block_size=12`、`linear_attn_config.full_attn_layers`（24 个 MLA layer 索引）、`linear_attn_config.kda_layers`（69 个 KDA layer 索引）。
- **实验条件**：K3 Instruct 模型版本。
- **置信状态**：已确认。

### N2
- **数字**：$N\approx 8$ 在多数模型尺度下能恢复 Full AttnRes 的大部分收益。
- **来源**：K3 报告 §2.2 第 418 行转述 [57]。
- **实验条件**：原 preprint [57] 的实验设置（本页未获取原文）。
- **置信状态**：K3 报告转述已确认；原 preprint 的具体实验数据未核对，标注为间接证据。

### N3
- **数字**：K3 主干 hidden_size=7168、num_attention_heads=96；block size=12 层。
- **来源**：HuggingFace `config.json`。
- **实验条件**：K3 Instruct 模型版本。
- **置信状态**：已确认。

## 来源清单

1. **K3 技术报告**：Moonshot AI. *Kimi K3: Open Frontier Intelligence*. Technical Report, 2026-07-28. 本页引用 §2.2（Eq.8-10，第 377-419 行）、§2 概述（第 196-210 行）、§5.2.2（第 1376-1381 行）、§5.4.2（第 1624-1637 行）、Table 1（第 742-757 行）。文件位置：`/tmp/kimi-k3-research/k3-report.txt`。
2. **HuggingFace 官方 config.json**：`https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`（访问日期 2026-08-09）。关键字段：`attn_res_block_size=12`、`num_hidden_layers=93`、`hidden_size=7168`、`num_attention_heads=96`、`linear_attn_config.full_attn_layers`（24 个 MLA）、`linear_attn_config.kda_layers`（69 个 KDA）。
3. **AttnRes 原 preprint**：[57] Kimi Team. *Attention Residuals*. Preprint, 2026. **本页未获取原文完整内容**；相关结论（如 $N\approx 8$ 的经验结论）只用 K3 报告 §2.2 的转述，标注为间接证据。
4. **K3 前向数据流 note**：`wiki/kimi-k3-dataflow/index.html`（2026-08-07 按官方源码 `modeling_kimi_linear.py` 修订）。用于核对 K3 加权三次的具体位置（attention 前、MLP 前、final norm 前）与 9 个候选来源的工程实现。本页标注为间接来源，未直接复核源码。
5. **残差连接概念页**：`wiki/residual-connection/index.html`。前置概念，本页引用其"等权累加"性质与"退化问题"动机。
6. **RMSNorm**：Zhang & Sennrich 2019（K3 报告引用 [146]）。本页只用定义，不展开推导。

## 不确定项与处理

- **C8 的间接证据**：K3 报告原文只说"each module"（即 attention 模块与 MLP 模块各一次）+ 末尾聚合（C7），具体"attention 前 / MLP 前 / final norm 前"三次的位置来自 dataflow 页面对源码的核对。本页在正文与来源说明中明确标注："K3 报告原文确认 each module 各加权一次 + 末尾聚合；具体三次位置（attention 前 / MLP 前 / final norm 前）来自 `wiki/kimi-k3-dataflow/` 对官方源码的核对，本页未直接复核源码。"
- **$N\approx 8$ 的最优性**：原 preprint [57] 未获取，K3 报告只转述结论。本页引用时标注"经验结论，来自 K3 报告对 [57] 的转述，原 preprint 实验数据未核对"。
- **partial block 的层数**：K3 报告说"8 blocks with 12-layer size, giving a partial final block"，但未明说最后一个 block 有几层。由 `num_hidden_layers=93` 与 `attn_res_block_size=12` 推算：$93 = 7\times 12 + 9$，即前 7 个 block 各 12 层、第 8 个 block 9 层。本页按此推算标注。
