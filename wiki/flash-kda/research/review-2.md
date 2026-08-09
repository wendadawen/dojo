# FlashKDA 与 KDA Context Parallelism 独立审查（第二轮）

- 审查者：独立上下文（AI 模拟小白读者 + 对照来源核查）
- 页面版本：wiki/flash-kda/index.html（69780 bytes, 2026-08-09 16:01）+ overview.html（7786 bytes, 2026-08-09 16:01）
- 时间：2026-08-09
- 审查依据：guides/concept/check.md（段A盲读 + 段B对照来源）
- 来源：/tmp/kimi-k3-research/k3-report.txt §5.1（§5.1.1 FlashKDA + 设备内 CP、§5.1.2 KCP）+ §5.4.2（KDA 解码 kernel）+ §2.1.1（Eq.1）

## 段 A 盲读笔记（小白读者卡点）

按页面顺序阅读，记录理解主线上的卡点：

- §1（行 690）：$S_t \in \mathbb{R}^{d_k \times d_v}$（K3 config.json 中 $d_k=d_v=128$，约 32KB）。$d_k, d_v$ 首现未说明是 key/value 维度；"约 32KB"未说明是单 head 状态还是全部 head 状态（K3 有 96 head，读者可能困惑）。
- §2（行 739）：CUTLASS 首现未展开缩写，到行 770 才补"NVIDIA 的高性能 kernel 模板库"。
- §2（行 741-744 表格）："intra-chunk 的位置-位置 attention"首现未解释"位置-位置"含义。
- §4（行 830）：vanilla 线性注意力递归 $s_i = s_{i-1} + \phi(k_i) v_i^\top$，$\phi$ 首现未定义，读者不知道这是 key 的特征映射。
- §4（行 862）：$M_{t\leftarrow 1}^{[i+1]} := \prod_{r\leftarrow 1}^{t} M_r$，$\leftarrow$ 记号未说明含义（矩阵乘法顺序）。
- §5（行 990）：投影输入列出 $q,k,v,\beta,\alpha$，但本页之前 $q$ 未出现且未说用途；且页面称"它们是 KDA 更新 Eq.1 的全部输入"，而 Eq.1（$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$）只用 $k,v,\beta,\alpha$，$q$ 不在其中。

段 A 结束时逐题核对学习目标：
- 目标1（冲突根源 + 四 regime 瓶颈）：§1 章节完整回答，清晰。
- 目标2（FlashKDA 重叠）：§2 回答，机制清晰。
- 目标3（设备内 CP 单 rank 无跨设备通信）：§3 回答。
- 目标4（KCP 不能直接求和 + $M+\tilde S$ 分解 + 通信量与序列长度无关）：§4 回答，手算验证完整可复算。
- 目标5（解码回滚 + 投影缓存）：§5 回答。

## 段 B 对照来源核查

### 1. 定义与机制

逐条对照 k3-report.txt：

- 页面 §1 Eq.1（$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$）——来源行 1200-1201（§5.1.2）与行 234（§2.1.1 Eq.1）**完全一致**。
- §1 引用"四 regime 瓶颈不同"（[C1]）——来源行 1169-1170 **完全一致**。
- §2 FlashKDA 重叠机制、token-parallel + head-parallel 分解、CUTLASS-based、flash-linear-attention 后端、服务训练和 prefill（[C2]）——来源行 1174-1178 **完全一致**。
- §2 "substantially outperforms the Triton reference implementation"——来源行 1177 **一致**（页面行 1057 标注为定性结论不数值化）。
- §3 TP 切 head 不缩短递归（[C3]）——来源行 1180-1181 **完全一致**。
- §3 段转移可独立于入状态计算、SM 级 CP planner、无跨设备通信（[C3]）——来源行 1182-1190 **完全一致**。
- §4 vanilla 线性注意力"additive recurrence"可直接求和（[C4]）——来源行 1197-1199 **一致**（来源未给具体公式，页面 $s_i = s_{i-1} + \phi(k_i) v_i^\top$ 是对"additive recurrence"的具体化，$\phi$ 未定义见问题 1）。
- §4 KDA 直接求和失效、$M_t$ 作用于入状态（[C4]）——来源行 1200-1203 **完全一致**。
- §4 KCP 分解 Eq.17（$S_t^{[i+1]} = \tilde S_t^{[i+1]} + M_{t\leftarrow 1}^{[i+1]} S_T^{[i]}$）——来源行 1204-1218 **完全一致**，符号记法对应。
- §4 prefix scan + 一次 all-gather + 固定通信量（[C4]）——来源行 1247-1254 **完全一致**。
- §5 解码瓶颈从"利用并行"转为"管理原地更新状态"（[C5]）——来源行 1609-1611 **完全一致**。
- §5 MTP 拒绝 draft token 状态无法回滚（[C5]）——来源行 1611-1613 **完全一致**。
- §5 状态快照方案流量代价（[C5]）——来源行 1613-1614 **完全一致**。
- §5 投影输入决定接受前缀状态、缓存投影输入、片上重建、ReplaySSM 并发（[C5]）——来源行 1615-1618 **完全一致**。
- §5 融合 kernel 覆盖短卷积、输入归一化、门控、KDA 递归、输出归一化——来源行 1618-1620 **完全一致**。
- §5 验证延迟亚线性增长、低于状态缓存基线——来源行 1620-1621 **一致**。
- §5 投影缓存不离开 decode 阶段、PD 分离载荷不变（[C5]）——来源行 1621-1622 **完全一致**。

### 2. 公式与推导

手算验证（§4 行 911-954，2 rank × 2 token KDA 递归）：

- $d_k=d_v=2$，$\alpha=0.5$（标量，$\mathrm{Diag}(\alpha)=0.5I$），$\beta=1$，单位向量 $k$。
- $M_1=M_3=\begin{pmatrix}0.5&0\\0&1\end{pmatrix}$，$M_2=M_4=\begin{pmatrix}1&0\\0&0.5\end{pmatrix}$——**复算正确**（$M_t = I - k_t k_t^\top \cdot 0.5I$）。
- ground truth $S_4=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$——**逐步复算正确**（$S_1\to S_2\to S_3\to S_4$ 每步代入验证）。
- KCP 分解：$M累积_1=M累积_2=0.5I$，$\tilde S_1=\begin{pmatrix}1&2\\3&4\end{pmatrix}$，$\tilde S_2=\begin{pmatrix}5&6\\7&8\end{pmatrix}$——**复算正确**。
- prefix scan 重组：$S_T^{[2]}=\tilde S_2 + 0.5I\cdot S_T^{[1]}=\begin{pmatrix}5.5&7\\8.5&10\end{pmatrix}$——**与 ground truth 完全一致**。
- 误用直接求和对照：$\tilde S_2+\tilde S_1=\begin{pmatrix}6&8\\10&12\end{pmatrix}\neq S_4$——**正确**，展示"direct summation insufficient"的数值体现。

折叠块（行 941-952）完整 4 步 ground truth + 3 步 KCP 分解的逐矩阵代入——全部可复算，与展开外的摘要一致。

注：$\beta=1$ 在来源定义域 $(0,1)$ 边界（来源行 238 "$\beta_t \in (0,1)$"），但页面教学简化第 (1) 条已明确"教学构造...不代表真实 K3 数值"，可接受。

### 3. 可运行代码

页面无可运行代码（教学简化第 (2) 条说明"四套方案都是 kernel/系统级机制...可运行代码需要 CUTLASS/分布式框架，超出教学职责"）。手算例子已足够验证 KCP 分解。可接受。

### 4. 事实与推断

- 96 head：来源行 751（K2→K3 对照表 "Attention Heads 64→96"）**支持**。
- 69 KDA + 24 Gated MLA：来源行 757（"69 KDA + 24 MLA"）**支持**；行 226 "An additional Gated MLA layer is placed" 支持 24 而非 23。
- $\alpha$ 范围 $(e^{-5},1)$：来源行 322, 329, 338-340（"scaled sigmoid... $g_{min}=-5$... $\alpha_{t,j} > e^{-5} \approx 6.7\times 10^{-3}$"）**支持**。
- $d_k=d_v=128$：来源未明确写数值，页面标注来自 config.json，无法在本审查中直接核对；32KB = $128\times 128\times 2$ bytes（bf16）计算自洽。
- 132 块 SM（H100）：来源未明确（行 2182 只说"NVIDIA Hopper GPU"），页面写"以 H100 为例"作为背景，H100 SXM 确有 132 SM，可接受。
- "substantially outperforms Triton" 与"verification latency sub-linear"为来源原文定性结论，页面只复述不数值化——**正确**。

### 5. 前置知识引用

- KDA 概念页（../../wiki/kda/index.html）——链接有效，页面存在。
- GPU 执行模型页（../../wiki/gpu-execution-model/index.html）——链接有效，页面存在。
- 线性注意力页（../../wiki/linear-attention/index.html）——链接有效，页面存在（本审查范围内）。

### 6. 教学简化

- 手算例子参数简化（$d_k=d_v=2$、$\alpha=0.5$ 标量、$\beta=1$、单位 $k$）已标记为教学构造，说明不代表真实 K3 数值（实际 $d_k=128$、$\alpha$ 由 scaled sigmoid、$k$ 经 L2Norm 非坐标轴向量、96 head）——**正确**。
- "投影输入比状态小"只陈述定性结论，不引入未经证实的精确比例数字——**正确**。
- 四条教学解释与类比边界（行 1047-1050）均说明失效边界——**正确**。
- 无伪代码/可运行代码的简化已说明理由——**正确**。

### 7. 页面功能

- KaTeX 公式渲染：delimiters 配置正确（`$$...$$` display、`$...$` inline）。
- 折叠块：行 940-952 的 details 标签正确，summary 清晰，收起后正文仍有手算摘要完整。
- 目录锚点：h2/h3 有 id，scroll-margin-top 避开顶部导航。
- 来源引用标记 [C1]-[C5]、[F1]-[F3] 在文末"核心论断与来源"完整列出原文——**清晰可定位**。

## 问题

- [重要·盲读] §4 行 830：vanilla 线性注意力递归 $s_i = s_{i-1} + \phi(k_i) v_i^\top$ 中 $\phi$ 首现未定义，读者不知道这是 key 的特征映射；来源只说"additive recurrence"未提 $\phi$，页面具体化时未解释符号。：在公式下方加一句"$\phi$ 是 key 的特征映射（feature map），详见线性注意力页；本页只需用'纯加性、无矩阵作用于入状态'这一性质"；或改用不含 $\phi$ 的简记（如 $k_i' v_i^\top$ 并注明 $k_i'$ 是经特征映射后的 key）。｜ 修复：采用第一种方案——在 §4 vanilla 线性注意力递归公式 $s_i = s_{i-1} + \phi(k_i) v_i^\top$ 下方加一句"$\phi$ 是 key 的特征映射（feature map），详见线性注意力页；本页只需用'纯加性、无矩阵作用于入状态'这一性质"。validate.py 通过。 ｜ 复验：
- [重要·技术] §5 行 990：页面写"投影输入指 draft token 经过 $W_{q/k/v/\beta/\alpha}$ 投影后的 $q,k,v,\beta,\alpha$——它们是 KDA 更新 Eq.1 的全部输入"。但 Eq.1（$S_t = M_t S_{t-1} + \beta_t k_t v_t^\top$，$M_t := I - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$）只用 $k,v,\beta,\alpha$，$q$ 不在其中（$q$ 用于输出 $o_t = S_t^\top q_t$，见来源 §2.1.1 Eq.1 第二式）。来源 §5.4.2 只说"projected inputs"未列举具体项，页面具体化时把 $q$ 错误归入"Eq.1 的全部输入"。：改为"它们是 KDA 前向的全部投影输入（$k,v,\beta,\alpha$ 用于状态更新 Eq.1，$q$ 用于输出计算 $o_t = S_t^\top q_t$）"。｜ 修复：已将 §5 投影输入缓存段"它们是 KDA 更新 Eq.1 的全部输入"改为"它们是 KDA 前向的全部投影输入（$k,v,\beta,\alpha$ 用于状态更新 Eq.1，$q$ 用于输出计算 $o_t = S_t^\top q_t$）"，区分状态更新输入与输出计算输入。validate.py 通过。 ｜ 复验：
- [轻微·盲读] §1 行 690：$d_k, d_v$ 首现未明确说明是 key/value 的维度。：首次出现时加"$d_k$（key 维度）、$d_v$（value 维度）"。｜ 修复： ｜ 复验：
- [轻微·盲读] §2 行 741-744 表格："intra-chunk 的位置-位置 attention"首现未解释"位置-位置"含义。：改为"intra-chunk 的注意力计算（chunk 内所有位置对的注意力）"或加一句解释。｜ 修复： ｜ 复验：
- [轻微·盲读] §4 行 862：$M_{t\leftarrow 1}^{[i+1]} := \prod_{r\leftarrow 1}^{t} M_r$ 中 $\leftarrow$ 记号未说明含义（矩阵乘法不可交换，顺序重要）。：首次出现时加一句"$\leftarrow$ 表示按文档顺序从右向左连乘"。｜ 修复： ｜ 复验：
- [轻微·盲读] §2 行 739：CUTLASS 首次出现未展开缩写，行 770 才补"（NVIDIA 的高性能 kernel 模板库）"。：把行 770 的解释提前到行 739 首次出现处。｜ 修复： ｜ 复验：
- [轻微·盲读] §1 行 690："约 32KB"未说明是单 head 状态（K3 有 96 head，读者可能困惑 32KB 是单 head 还是全部）。：加一句"单 head 状态约 32KB（$128\times 128\times 2$ bytes，bf16）"。｜ 修复： ｜ 复验：

## 结论

- 统计：阻断 0 / 重要 2 / 轻微 5
- 处置：进入修复

来源对照全部通过（Eq.1、四 regime 论断、FlashKDA、设备内 CP、KCP 分解 Eq.17、prefix scan/all-gather、解码投影缓存/ReplaySSM/融合 kernel/PD 分离载荷均与 k3-report.txt §5.1+§5.4.2+§2.1.1 完全一致）；手算例子 100% 可复算，ground truth 与 KCP 分解重组结果完全一致；96 head、69 KDA+24 MLA、$\alpha$ 范围 $(e^{-5},1)$ 均有来源支持。两个重要问题分别为 $\phi$ 符号未定义（盲读卡点）和 $q$ 错误归入 Eq.1 输入（技术表述错误），均不导致核心结论失效但造成明显误解或认知跳步。
