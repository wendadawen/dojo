# RoPE 旋转位置编码 · 术语表

登记全文所有首次出现的术语、缩写和符号。保证全文含义一致，防止同一对象出现多种记号或术语漂移。

## 术语

| 术语 | 首次出现 | 定义或含义 |
|---|---|---|
| 位置编码 | S1 | 给注意力注入位置信息的方法 |
| 自注意力 / softmax 注意力 | S1 | 用 query 和 key 做匹配、按匹配程度对 value 加权求和的机制 |
| 排列等变 | S1 | 交换输入顺序后每个位置输出值不变的性质；自注意力在无位置信息时的特性 |
| 绝对位置编码（APE） | S1 | 把位置向量加到 token 嵌入上的方案 |
| 相对位置编码 | S1 | 在注意力分数上加与相对距离绑定偏置的方案（含 T5 相对偏置、ALiBi） |
| NoPE | S1 | 不施加任何显式位置编码的方案；K3 MLA 层使用 |
| 旋转位置编码（RoPE） | 开头 | 按 token 绝对位置旋转 Q/K 的位置编码 |
| 旋转矩阵 | S2 | 2 维平面上按角度旋转的 2×2 矩阵 `R_m` |
| 频率 | S2 | 单位位置旋转的角度，记 θ；2 维对在位置 m 的旋转角为 mθ |
| 几何级数 | S4 | θ_i = base^{-2i/d} 形成的等比数列 |
| 多尺度 | S4 | 不同 i 的 θ_i 编码不同尺度的位置信息 |
| 分块对角矩阵 | S4 | d/2 个 2×2 块沿对角线排列构成的 d×d 矩阵 `R_m^d` |
| 远程衰减 | S5 | 期望内积随 |m−n| 增大趋于 0 的性质 |
| 局部性偏置（locality bias） | S5 | 远距离 token 注意力衰减的隐式效果 |
| 长度外推 | S6 | 把训练时见过的最大长度外的位置应用到 RoPE |
| Position Interpolation（PI） | S6 | 缩放位置 m 以适配更长上下文的 RoPE 扩展方法 |
| YaRN | S6 | 按频率分组缩放的 RoPE 扩展方法 |
| 矩阵吸收 | S6 | MLA 推理优化中把 W^UK 从 key 侧搬到 query 侧的技巧 |
| 因果掩码 | S6 | decoder-only 注意力中上三角 mask，阻止看到未来 token |
| KDA | S6 | Kimi K3 中的递归注意力变体，承担大部分序列混合 |

## 缩写

| 缩写 | 全称 | 首次出现 |
|---|---|---|
| RoPE | Rotary Position Embedding | 开头 |
| APE | Absolute Position Encoding | S1 |
| NoPE | No Position Encoding | S1 |
| PI | Position Interpolation | S6 |
| YaRN | Yet another RoPE extensioN | S6 |
| MLA | Multi-head Latent Attention | S6 |
| KDA | Kimi Distributed Attention（K3 中的递归注意力变体） | S6 |
| K3 | Kimi K3 | S6 |

## 符号

| 符号 | 首次出现 | 含义 |
|---|---|---|
| `m`、`n` | S2 | query 与 key 的绝对位置（整数索引） |
| `q`、`k`、`v` | S2 | 单个 query、key、value 向量 |
| `q_m`、`k_n` | S2 | 位置 m 的 query、位置 n 的 key |
| `q₀`、`q₁` | S2 | 2 维 query 的两个分量 |
| `R_m` | S2 | 位置 m 的 2 维旋转矩阵 |
| `R_m^d` | S4 | 位置 m 的 d 维分块对角旋转矩阵 |
| `θ` | S2 | 单个 2 维对的频率 |
| `θ_i` | S4 | 第 i 个 2 维对的频率，i ∈ {0, 1, …, d/2−1} |
| `base` | S4 | 频率公式的底数，默认 10000 |
| `d` | S4 | query/key 向量的维度（偶数） |
| `R_m^{(i)}` | S4 | `R_m^d` 的第 i 个 2×2 块 |
| `Q`、`K`、`V` | S5 | 注意力层中的 query、key、value 矩阵（多个 token 拼起来） |
| `E[·]` | S5 | 期望 |
| `Σ` | S5 | 求和符号 |
