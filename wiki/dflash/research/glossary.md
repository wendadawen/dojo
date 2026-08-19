# DFlash 术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| 投机解码（speculative decoding） | 开头 | 小草稿模型先猜一段、大目标模型一次前向并行验证的无损加速框架 |
| 草稿模型（draft model）$\mathcal{M}_d$ | 开头 | 投机解码中负责猜的模型 |
| 目标模型（target model）$\mathcal{M}_t$ | 开头 | 投机解码中负责验证的模型 |
| $T_{\text{draft}}$ / $T_{\text{verify}}$ | 第 1 章 | 一轮中起草 / 验证的耗时 |
| $\tau$ | 第 1 章 | 每轮期望接受 token 数（含 bonus token），上限 $\gamma+1$ |
| bonus token | 第 1 章 | 验证前向顺带多产出的一个 token |
| $\gamma$（延迟公式） | 第 1 章 | 每轮草稿 token 数（投机预算） |
| $t_{\text{step}}$ / $t_{\text{parallel}}$ | 第 1 章 | 单次前向的延迟 / 块并行生成的延迟 |
| 块扩散（block diffusion） | 第 2 章 | 块间自回归、块内并行去噪的生成范式；DFlash 取单步形态 |
| 块大小（block size） | 第 2 章 | 草稿块内的 token 数（实验 16/10/8） |
| 单步并行预测 | 第 2 章 | 一次前向同时预测整块所有被遮位置 |
| target 上下文特征 $H_t$ | 第 2 章 | target 若干层隐藏态拼接投影后的条件特征 |
| $H^{(l_i)}$ | 第 2 章 | target 第 $l_i$ 层隐藏态 |
| $W_c$ | 第 2 章 | 特征融合投影（$D\times 5D$，各 draft 层共享） |
| KV 注入（KV injection） | 第 2 章 | 把 $H_t$ 的 K/V 投影注入 draft 每层注意力的机制 |
| $H_d$ | 第 2 章 | draft token 的隐藏态 |
| 输入融合（input fusion） | 第 2 章 | EAGLE 式做法：target 特征与 token embedding 拼接后只进输入 |
| anchor token | 第 3 章 | 训练构块时作为块首的干净 token，其后位置被遮 |
| 稀疏注意力掩码 | 第 3 章 | 训练时块内双向、块间禁止的注意力遮挡 |
| $w_k$ | 第 3 章 | 块内第 $k$ 个位置的损失权重 |
| $\gamma$（loss decay 超参） | 第 3 章 | 指数衰减率（块 16 取 7、块 10 取 5、块 8 取 4），与延迟公式的 $\gamma$ 不同物，页面各自就地声明 |
| Flex Attention | 第 3 章 | PyTorch 自定义注意力掩码机制，训练实现载体 |
| LM head | 第 3 章 | 隐藏态到词表 logits 的输出层 |
| EAGLE-3 | 开头/第 1 章 | feature 级自回归起草 + 树验证的 SOTA 对比方法 |
| 树大小（tree size） | 第 4 章 | EAGLE-3 树验证的草稿 token 预算（16/60） |
| MTP（multi-token prediction） | 第 4 章 | target 模型自带的多 token 预测头路径（对比项） |
| thinking 模式 | 第 4 章 | Qwen3 输出长思考链的模式 |
| Spec-v2 | 第 4 章 | SGLang 的投机解码 V2 引擎（重叠调度） |
| 接受长度 / 加速比 | 第 4 章 | $\tau$ 与对自回归基线的吞吐倍数 |
| DFlash 2 | 第 5 章 | Inco AI 的后续迭代（加选择器与卷积），非本论文内容 |
