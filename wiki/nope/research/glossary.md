# NoPE 术语表

登记全文首次出现的术语、缩写和符号。保证全文含义一致。

| 术语/缩写/符号 | 首次出现 | 定义或含义 |
|---|---|---|
| NoPE | S1 钩子 | No Position Encoding，无位置编码；在注意力计算中不施加任何显式位置编码的做法 |
| 位置编码（PE） | S1 | 给注意力注入位置信息的显式机制的总称；NoPE 是其方案集合中"什么都不做"的选项 |
| 显式位置编码 | S2 | 由人工设计的、直接注入位置信息的机制（位置嵌入、距离偏置、旋转等）；NoPE 去掉的正是这些 |
| 绝对位置编码（APE） | S1 对照表 | 给输入嵌入加一个与绝对位置绑定的向量 |
| 相对位置编码 | S1 对照表 | 给注意力分数加一个与两个位置距离绑定的偏置（T5 相对 PE 为代表） |
| T5 相对 PE | S4 | T5 模型采用的相对位置编码方案；NoPE 用 SGD 训练时学到的注意力模式主要类似它 |
| ALiBi | S1 对照表 | 给注意力分数加一个与距离成正比的线性衰减偏置 |
| RoPE（旋转位置编码） | S1 对照表 | 根据绝对位置旋转 query 和 key 的位置编码；NoPE 不旋转 |
| 注意力（attention） | S1 | 用 query 与 key 的匹配程度对 value 加权求和的机制 |
| query / key / value（$q,k,v$） | S3 公式 | 注意力的三组向量：query 发起匹配、key 被匹配、value 被加权取出 |
| 因果掩码（causal mask） | S3 | decoder-only 中限制位置 t 只能 attend 到位置 1..t、看不到未来 token 的遮挡规则 |
| decoder-only | S3 | 仅含解码器、自回归预测下一个 token 的 Transformer 结构；NoPE 的结论在此成立 |
| 排列等变 | S1 | 交换输入顺序时输出也作同样交换、但每个位置的输出不变的性质；双向注意力无位置信息时退化为排列等变导致无法区分顺序 |
| 长度泛化（length generalization） | S4 | 在短训练长度的序列上训练、在更长的序列上测试（外推）的能力 |
| 外推（extrapolation） | S4 | 测试序列长度超过训练时见过的长度 |
| 频率基数 | S4 教学解释 | RoPE 中决定旋转角随位置增长速率的参数；外推时常需重缩放 |
| YaRN | S5 | 一种在扩展上下文时调整 RoPE 的方法；K3 因用 NoPE 无需使用 |
| Kimi K3 | S5 | 采用 KDA + Gated MLA 混合注意力、对所有 MLA 层用 NoPE 的大模型 |
| KDA（Kernelized Drift Attention） | S5 | K3 中提供位置敏感、近因感知序列混合的循环注意力层；为 MLA 层的 NoPE 补足位置信息 |
| MLA（Multi-head Latent Attention） | S5 | 将 key–value 压缩为低维潜向量的多头注意力；K3 在其上用 NoPE |
| Gated MLA | S5 | K3 在 MLA 输出上加输入相关全秩门控的变体 |
| 循环门控与衰减 | S5 | KDA 中随每步更新、并用衰减因子压缩历史的机制；是 KDA 携带位置感的来源 |
| 衰减因子 $\alpha$ | S5 公式 | $\alpha_t^h=\exp(g_t^h)\in(e^{g_{\min}},1)^{d_k}$；控制历史信息随距离衰减的强度 |
| $g_{\min}$ | S5 数字 | K3 中 KDA 衰减对数的下界，固定为 $-5$ |
| $o_t$ | S3 公式 | 位置 $t$ 的注意力输出 |
| $\alpha_{t,i}$ | S3 公式 | 位置 $t$ 对位置 $i$ 的注意力权重（softmax 归一化） |
| SGD | S4 | 随机梯度下降；NoPE 在 SGD 训练下学到的模式主要类似 T5 相对 PE |
