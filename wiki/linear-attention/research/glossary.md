# 线性注意力术语表

| 术语 / 缩写 / 符号 | 首次出现位置 | 定义或含义 |
|---|---|---|
| 线性注意力 (Linear Attention) | S1 钩子 | 把 softmax 注意力的指数相似度换成可分解为 φ(q)·φ(k) 的核函数，使复杂度从 O(N²) 降到 O(N) 的注意力机制 |
| softmax 注意力 (softmax attention) | S1 | 标准 Transformer 注意力，相似度函数为 exp(q·k/√d)；本文的对比对象 |
| N | S1 | 序列长度（token 数）；也写作 L（Performer 论文） |
| d | S1 | 每个注意力头的维度（head dimension）；Q、K、V 每行的长度 |
| d' | S2 | 特征映射 φ 的输出维度；通常取 d'=d |
| Q, K, V | S1 | 查询、键、值矩阵，形状均为 N×d |
| Q_i, K_i, V_i | S1 | 第 i 个 token 的查询、键、值向量（行向量），形状 d |
| QK^T | S1 | 查询与键的内积矩阵，形状 N×N；softmax 注意力 O(N²) 瓶颈所在 |
| V' | S1 | 注意力输出，形状 N×d |
| √d | S1 | 缩放因子，用于稳定 softmax 数值；本文公式中常省略以简化 |
| softmax | S1 | 把任意实数分数转成和为 1 的正数概率的函数；本文首次使用时一句话提示 |
| 因果掩码 (causal mask) | S3 | 让第 i 个 query 只能看到前 i 个 key 的掩码，用于自回归生成 |
| 自回归 (autoregressive) | S1 | 逐 token 生成序列的方式；每步根据已生成内容预测下一 token |
| KV cache | S1 | 自回归推理时缓存历史 K、V 的内存；softmax 注意力下随 N 线性增长 |
| sim(q, k) | S2 | 广义注意力中的相似度函数；softmax 注意力下为 exp(q·k/√d) |
| 核函数 (kernel function) | S2 | 可表示为 φ(q)·φ(k) 的相似度函数；本文特指有限维正特征映射 |
| 特征映射 φ | S2 | 把 q 映射到 φ(q) 的函数，使 sim(q,k)=φ(q)^T φ(k)；必须保证 sim 非负 |
| 非负约束 | S2 | sim(q,k) 必须非负的要求，因为广义注意力分母作归一化用 |
| 结合律重排 | S2 | (φ(Q)φ(K)^T)V = φ(Q)(φ(K)^T V)，由矩阵乘法结合律得到；先聚合再查询 |
| φ(K)^T V | S2 | keys 特征映射与 values 的聚合，形状 d'×d，与 N 无关 |
| elu(x) | S4 | exponential linear unit；x≥0 时为 x，x<0 时为 e^x-1；值域 (-1, +∞) |
| φ(x)=elu(x)+1 | S4 | Katharopoulos 2020 选用的特征映射；保证 φ(x)>0 满足非负约束 |
| softmax 核 | S4 | sim(q,k)=exp(q·k) 的核函数；没有有限维正特征映射，无法精确线性化 |
| Performer | S4 | Choromanski et al. 2020 提出的线性注意力变体；用随机特征近似 softmax 核 |
| FAVOR+ | S4 | Performer 使用的正交正随机特征算法；本文不展开细节 |
| 随机特征 (random features) | S4 | 用随机投影 ω 构造 φ 使 E[φ(q)·φ(k)]≈exp(q·k) 的方法 |
| s_i | S3 | 第 i 步的 attention memory，形状 d'×d；递推 s_i = s_{i-1} + φ(K_i)V_i^T |
| z_i | S3 | 第 i 步的 normalizer memory，形状 d'；递推 z_i = z_{i-1} + φ(K_i) |
| s_0, z_0 | S3 | 递归初始状态，均为 0 |
| RNN (Recurrent Neural Network) | S3 | 循环神经网络；linear attention 在因果掩码下可写成 RNN 形式（固定状态 + 递推更新） |
| "Transformers are RNNs" | S3 | Katharopoulos 2020 论文标题；特指因果掩码下 linear attention 可重写为带 s_i、z_i 两个隐藏状态的 RNN |
| LSTM | S3 | 长短期记忆网络；本文仅用于澄清 s_i 不等同于 LSTM 细胞状态 c_t |
| 表达力 (expressivity) | S4 | 模型可表达的函数族范围；linear attention 表达力严格弱于 softmax |
| O(N²) / O(N) | S1 / S2 | 渐近复杂度记号；N 为序列长度 |
| W_Q, W_K, W_V | S3 折叠 | query/key/value 投影矩阵；完整 RNN 形式中出现 |
| f_l | S3 折叠 | linear attention 层后接的非线性激活（论文用 softmax over heads）；本文不展开 |
| 残差连接 | S3 折叠 | 把输入 x_i 加到注意力输出上的机制；完整 RNN 形式 y_i = f_l(V'_i + x_i) 中出现 |
