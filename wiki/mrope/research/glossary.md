# glossary：MRoPE

| 术语/符号 | 首现 | 含义 |
|---|---|---|
| MRoPE / 多模态旋转位置编码 | 开头 | 把位置 id 拆成三个分量的 RoPE 扩展 |
| 1D-RoPE | 第 1 章 | 原始一维 RoPE（引 rope 页） |
| 位置 id / position id | 第 1 章 | 送进旋转的整数坐标；MRoPE 下为 $(t,h,w)$ 三元组 |
| temporal / height / width 分量 | 第 2 章 | 三元组的时间/高/宽三个数（$t,h,w$） |
| grid_thw | 第 2 章 | 视觉输入的 (时间,高,宽) patch 网格尺寸 |
| spatial_merge_size / merge | 第 2 章 | 空间合并因子（2×2 相邻 patch 合一），网格到 token 数与位置量都除以它 |
| 跨模态衔接 | 第 2 章 | 新模态位置从前一模态最大位置 id + 1 开始（C7） |
| 推进量 | 第 4 章 | 一个模态块结束后位置轴前进的数值 $\max(h,w)/\text{merge}$ |
| mrope_section | 第 3 章 | 三分量在频率槽位上的配额，如 $[11,11,10]$ |
| 频率槽位 | 第 3 章 | rotary_dim/2 个旋转频率的下标位置 |
| 分段排布 / chunked | 第 3 章 | 每个分量独占一段连续槽位（qwen2_vl 实现） |
| 交错排布 / interleaved | 第 3 章 | 三分量以 3 为步长交替占用槽位（qwen4_exp 实现，mrope_interleaved=true） |
| rotary_dim | 第 3 章 | 头维中参与旋转的维数（如 $256\times 0.25=64$） |
| 长序列外推 | 第 4 章 | 推理长度超过训练最大长度的能力 |
| 位置轴预算 | 第 4 章 | max_position_embeddings 约束的位置 id 范围，区别于序列长度（token 数）预算 |
