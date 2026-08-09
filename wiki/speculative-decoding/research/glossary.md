# 投机解码 · 术语表

| 术语 / 缩写 / 符号 | 首次出现位置 | 定义或含义 |
|---|---|---|
| 投机解码 / Speculative Decoding | S1 钩子 | 用小 draft 模型生成候选 token、大 target 模型并行验证的推理加速方法；不改变输出分布。 |
| Speculative Sampling | 页面开头注释 | Chen et al. 2023 对同一机制的命名；本页统一用「投机解码」。 |
| target 模型 / $p$ | S1 | 大模型，其分布 $p(x\|{\cdot})$ 是希望最终采样的目标分布。 |
| draft 模型 / $q$ | S2 步骤 1 | 小而快的模型，分布 $q(x\|{\cdot})$，用于生成候选 token。 |
| 草稿长度 / $\gamma$ | S2 步骤 1 | 一轮中 draft 模型自回归生成的候选 token 数；正整数。 |
| 自回归解码 | S1 首段 | 每次用一个 token 喂回模型得到下一个 token 的分布的串行生成方式。 |
| KV cache | S1 首段（最小衔接） | 缓存已计算的注意力键值对，避免历史 token 重复计算。本页假设其已存在，不展开。 |
| 内存带宽受限 / memory-bandwidth bound | S1 | 单步解码的瓶颈是从 HBM 读取全部权重，算术运算量相对很低、计算单元空闲。 |
| 单次前向传播 / forward pass | S2 步骤 2 | target 模型对前缀 + $\gamma$ 个 draft token 做一次推理，因因果注意力同时得到 $\gamma+1$ 个位置的分布。 |
| 因果注意力 / causal attention | S2 步骤 2 | 注意力掩码使位置 $i$ 的分布只依赖前 $i$ 个 token；标准 Transformer 注意力的固有性质。见 [standard-attention](../../wiki/standard-attention/index.html)。 |
| 接受概率 / $a(x)$ | S2 步骤 3 / F1 | $a(x) = \min(1, p(x)/q(x))$；$p\geq q$ 时为 1（必接受），$p<q$ 时为 $p/q$（概率接受）。 |
| 接受 / accept | S2 步骤 3 | 以概率 $a(x)$ 保留 draft token 作为该位置最终输出。 |
| 拒绝 / reject | S2 步骤 4 | 以概率 $1-a(x)$ 丢弃 draft token，从残差分布重采样替代。 |
| 残差分布 / $p'(x)$ | S2 步骤 4 / F2 | $p'(x) = \mathrm{norm}(\max(0, p(x)-q(x)))$；首个拒绝处用于重采样一个 token。 |
| 归一化常数 / $\beta$ | S3 路径 B / F3 推导 | $\sum_x \max(0, p(x)-q(x))$；恰好等于单位置发生拒绝的概率。 |
| bonus token | S2 步骤 5 / C6 | 全部 $\gamma$ 个 draft token 被接受时，从 $p_{\gamma+1}$ 免费采样的额外 token。 |
| 修改版拒绝采样 / modified rejection sampling | S3 首段 | Leviathan/Chen 提出的、带残差重采样的拒绝采样变体；区别于经典拒绝采样（后者拒绝时直接从 $p$ 重采样）。 |
| 单位置输出分布 / $\Pr[\text{emit }x]$ | S3 / F3 | 单位上一个 token 被最终采出的概率；分解为路径 A（draft 采到 + 接受）+ 路径 B（拒绝 + 残差重采样）。 |
| 路径 A | S3 路径 A | draft 采样到 $x$ 且被接受；概率 $\min(q(x), p(x))$。 |
| 路径 B | S3 路径 B | 某个 token 被拒、从残差重采样到 $x$；概率 $\max(0, p(x)-q(x))$。 |
| 接受率 / $\alpha$ | S3 / F4 / S4 | 平均为单位置接受概率；$\alpha = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q)$。 |
| 总变差距离 / $\mathrm{TV}(p,q)$ | S3 / F4 | $\mathrm{TV}(p,q) = \frac{1}{2}\sum_x \|p(x)-q(x)\|$；两个分布的差异度量；$\alpha = 1 - \mathrm{TV}$。 |
| 期望 token 数 / $\mathbb{E}[L]$ | S4 / F5 | 一轮期望产出的 token 数；$\mathbb{E}[L] = (1-\alpha^{\gamma+1})/(1-\alpha)$。 |
| 加速比 / $S$ | S4 / F6 | 相对纯 target 解码的墙钟时间加速；$S = (1-\alpha^{\gamma+1})/[(1-\alpha)(1+\gamma c)]$。 |
| 单步成本比 / $c$ | S4 / F6 | $c = T_{\text{draft}}/T_{\text{target}}$；draft 单步墙钟时间与 target 单步之比。 |
| 贪心解码 / greedy decoding | S3 末尾 / C7 | 每步取 argmax 的解码方式；本页在 C7 处说明投机解码在该情形下退化为 argmax 匹配。 |
| 点质量分布 | S3 / C7 | 把全部概率放在一个 token 上的分布；贪心解码下 $p$ 与 $q$ 均为点质量。 |
| HBM | S1 | GPU 板载高带宽显存；权重存放处，带宽是单步解码的瓶颈。见 [gpu-execution-model](../../wiki/gpu-execution-model/index.html)。 |
| SM / Tensor Core | S1 | GPU 计算单元；算术运算在其上执行，单步解码时大量空闲。见 [gpu-execution-model](../../wiki/gpu-execution-model/index.html)。 |
| T5-XXL | S6 / N1 | Leviathan et al. 2023 实测的 target 模型，11B 参数。 |
| T5-base | S4 复算 / N3 | Leviathan et al. 2023 实测中作为 draft 的模型，约 250M 参数。 |
| Chinchilla 70B | S6 / N2 | Chen et al. 2023 实测的 target 模型，70B 参数，分布式部署。 |
| EAGLE-3 | S6 | K3 用作 draft 模型的具体架构；本页只占位引用，不展开。概念页占位链接 `../../wiki/eagle-speculative/index.html`。 |
| vLLM | S4 / N4 | 开源推理框架；其高查询率批处理下的减速数据用于支撑 M4 边界。 |
