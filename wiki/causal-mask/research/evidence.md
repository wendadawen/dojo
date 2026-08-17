# 因果掩码（Causal Mask）核心论断与证据

核心论断编号：C 论断 / F 公式 / N 数字。只覆盖核心内容。来源选择优先级：原始论文与标准 > 权威教材与同行评审综述 > 对应版本的官方文档 > 固定版本或 commit 的官方源码。

## C 论断（可证实/证伪的完整句子）

### C1：因果掩码是 decoder 自注意力的遮挡规则——位置 $t$ 只能 attend 到位置 $1..t$，看不到未来 token

- 论断内容：在 decoder 的自注意力中，位置 $t$ 的 query 只能与位置 $1,2,\dots,t$ 的 key/value 交互，不能与 $t+1,t+2,\dots$ 交互。这一规则由掩码实现，目的是保留自回归性质（位置 $t$ 的预测只能依赖位置 $<t$ 的已知输出）。
- 来源定位：Vaswani et al. 2017《Attention Is All You Need》§3.2.3 "Applications of Attention in our Model"——"We need to modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$."；以及同节"self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position."
- 适用条件：decoder 自注意力（保留自回归性质）。
- 置信状态：已确认（论文原文直接支持）。

### C2：实现方式是把注意力分数矩阵中"未来位置"对应的元素置 $-\infty$（在 softmax 之前），softmax 后这些位置权重为 $0$

- 论断内容：在 $\operatorname{softmax}$ 的输入（即注意力分数 $QK^\top/\sqrt{d_k}$）上，把所有"非法连接"（query 位置 $i$ 到 key 位置 $j>i$）对应的元素置 $-\infty$。由于 $e^{-\infty}=0$，softmax 后这些位置权重为 $0$，剩余（合法）位置权重重新归一化到和为 $1$。
- 来源定位：Vaswani et al. 2017 §3.2.3 脚注——"In addition to self-attention layers, we also mask out (set to $-\infty$) all entries in the input to the softmax which correspond to illegal connections."；论文 Figure 2 左图 "Scale Dot-Product Attention" 中的 "Mask (opt.)" 方框对应这一操作。
- 适用条件：softmax 归一化在掩码之后进行；掩码施加于 softmax 输入（分数），不是施加于 softmax 输出（权重）。
- 置信状态：已确认（论文原文 + 图直接支持）。

### C3：因果掩码是一个严格上三角为 $-\infty$、对角线及以下为 $0$ 的矩阵 $M$，与注意力分数相加后即完成遮挡

- 论断内容：$M$ 是一个 $n\times n$ 矩阵（$n$ 为序列长度），$M_{ij}=-\infty$ 当 $j>i$（严格上三角，对应未来位置），$M_{ij}=0$ 当 $j\le i$（对角线及以下，对应可见位置）。把 $M$ 加到注意力分数上等价于把未来位置的分数置 $-\infty$。
- 来源定位：Vaswani et al. 2017 §3.2.3（C2 引用）的等价实现表述；标准 Transformer 实现（PyTorch `nn.Transformer`、Hugging Face `mask` 参数、fairseq 等）统一采用上三角 $-\infty$ 形式。本文以论文 §3.2.3 的"set to $-\infty$ in the input to the softmax"为准，上三角形式为其标准等价实现。
- 适用条件：序列长度为 $n$ 的 decoder 自注意力。
- 置信状态：已确认（论文语义的直接等价实现；标准教材与官方实现一致）。

### C4：因果掩码是自回归生成的基础——它保证 decoder 在训练时一次并行计算所有位置，同时每个位置只依赖其前缀

- 论断内容：自回归生成逐 token 预测下一个 token，每步只依赖已生成的前缀。训练时若不并行，则需逐位置串行计算（$O(n)$ 顺序步骤）；因果掩码允许一次喂入整条目标序列、用单个矩阵运算并行计算所有位置的表示，同时通过掩码保证每个位置只看到前缀——既保留自回归性质又获得训练并行度。
- 来源定位：Vaswani et al. 2017 §3.1 "Model Architecture"——"At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next."；§3.2.3 掩码机制（C1/C2）使训练并行成为可能。论文 §3.2.4 "Why Self-Attention"对比 RNN 的顺序约束 $O(n)$ 与自注意力的 $O(1)$ 顺序操作，因果掩码是 decoder 侧获得这一并行度的关键。
- 适用条件：decoder 训练阶段；推理阶段用 KV-cache 逐 token 增长（见 C5）。
- 置信状态：已确认（论文直接支持自回归性质与并行性，因果掩码为其实现手段）。

### C5：推理时因果掩码隐含在 KV-cache 的结构中——cache 只存过去 token 的 key/value，新 query 与 cache 交互天然只看过去

- 论断内容：推理（生成）阶段逐 token 进行，每生成一个新 token，把它的 key/value 追加到 KV-cache，下一个 query 与整个 cache 交互得到下一个 token。由于 cache 中只有已生成的（过去的）token，query 不可能"看到"未来——因果掩码的约束由 cache 的内容范围天然保证，推理时无需显式构造 $n\times n$ 上三角矩阵。
- 来源定位：Vaswani et al. 2017 §3.1 自回归生成描述（C4 引用）的推理侧推论；KV-cache 是自回归 Transformer 推理的标准实现（原始论文未单独命名 KV-cache，但其自回归生成语义直接蕴含"缓存过去 token 复用"）。本文以论文自回归语义为依据，KV-cache 为其工程实现的等价表述。
- 适用条件：自回归推理（逐 token 生成）。
- 置信状态：已确认（自回归语义的直接推论；KV-cache 是标准实现，所有主流推理框架一致）。

### C6：因果掩码打破排列对称性——使位置 $t$ 的可见 token 集合恰为 $\{1,\dots,t\}$，不同位置可见集合不同，因此即使不加任何显式位置编码，不同位置的注意力输出也会不同

- 论断内容：在没有位置编码的双向注意力中，交换输入顺序会让每个位置的输出也作同样交换（排列等变），无法区分"甲乙"和"乙甲"；因果掩码限制每个位置只看到前缀，使位置 $t$ 的可见集合恰为 $\{1,\dots,t\}$，不同位置可见集合不同，从而打破排列对称性——位置信息被结构隐式提供。这是 NoPE（不施加任何显式位置编码）仍能区分词序的结构前提。
- 来源定位：Vaswani et al. 2017 §3.2.3 因果掩码定义（C1）的推论；NoPE 论文（Kazemnejad et al., NeurIPS 2023, arXiv:2305.19466）摘要论证因果掩码使 NoPE 可表示位置——"explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences"。本文以因果掩码的可见集合性质为依据，NoPE 完整论证见 `wiki/nope/`。
- 适用条件：decoder-only / 因果注意力（双向注意力下不成立——见 C7）。
- 置信状态：已确认（因果掩码可见集合性质是直接推论；NoPE 论文摘要支持"因果掩码使 NoPE 可工作"的结论）。

### C7：因果掩码只用于 decoder 自注意力；encoder 用双向注意力（无因果掩码），每个位置看到全部输入

- 论断内容：encoder 的职责是构建输入序列的完整上下文表示，每个位置需要看到全部输入（双向）；decoder 的职责是自回归生成，每个位置只能看到前缀（因果）。因此因果掩码只在 decoder 自注意力中使用，不施加于 encoder 自注意力。
- 来源定位：Vaswani et al. 2017 §3.2.3——"self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position"（decoder 因果）；§3.2 encoder 自注意力无掩码限制——"Each position in the encoder can attend to all positions in the previous layer of the encoder."
- 适用条件：标准 encoder-decoder Transformer；decoder-only Transformer（如 GPT 系列）所有自注意力层均用因果掩码。
- 置信状态：已确认（论文直接支持）。

## F 公式

### F1：带因果掩码的缩放点积注意力

- 公式：$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}+M\right)V$，其中 $M_{ij}=\begin{cases}-\infty & j>i\\0 & j\le i\end{cases}$。
- 来源定位：Vaswani et al. 2017 §3.2.1 Eq.(1) 缩放点积注意力 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ 加上 §3.2.3 的掩码（C2/C3）。$M$ 的形式（严格上三角 $-\infty$）由 C3 给出。
- 适用条件：单头、序列长度 $n$、$Q,K,V\in\mathbb{R}^{n\times d}$（$d_k=d_v=d$），decoder 自注意力。
- 置信状态：已确认（标准注意力公式 + 论文掩码语义的等价合并）。

### F2：位置 $t$ 的注意力输出（因果形式，用于手算例子）

- 公式：$o_t=\sum_{j=1}^{t}\alpha_{t,j}\,v_j$，其中 $\alpha_{t,j}=\dfrac{\exp(s_{t,j})}{\sum_{i=1}^{t}\exp(s_{t,i})}$，$s_{t,j}=\dfrac{q_t^\top k_j}{\sqrt{d_k}}$（被掩的 $j>t$ 项 $\exp(-\infty)=0$ 不参与）。
- 来源定位：F1 在单位置 $t$ 上的展开；求和上界为 $t$（而非 $n$）由 C1 的可见规则决定。
- 适用条件：单头、因果掩码下位置 $t$ 的输出。
- 置信状态：已确认（F1 的直接展开）。

## N 数字

本文不引用任何外部实测数字（如精度、速度、显存占用）。所有数字均为构造示例（$3$-token 序列、$v_1=1,v_2=2,v_3=3$、注意力分数全为 $0$），用于手算验证机制，不代表真实模型或工程推荐值。构造示例的数字与目的登记在文末"构造示例"小节。
