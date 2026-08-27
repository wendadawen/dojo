# glossary.md：深度可分离卷积

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| $D_F$ | 第 1 章 | 输入特征图的方形宽高 |
| $M$ / $N$ | 第 1 章 | 输入 / 输出通道数 |
| $D_K$ | 第 1 章 | 卷积核的方形边长 |
| 滤波（filtering） | 第 1 章 | 在单通道空间邻域上做加权求和 |
| 组合（combining） | 第 1 章 | 把各通道的滤波结果做跨通道线性混合 |
| depthwise 卷积 | 第 2 章 | 每个输入通道配一个 $D_K\times D_K\times1$ 核、只滤波不跨通道 |
| pointwise 卷积 | 第 2 章 | $1\times1\times M\times N$ 核、只做跨通道线性组合 |
| groups（卷积分组） | 第 4 章 | 卷积实现参数：每组内输入输出通道绑定独立核；groups=C 即逐通道（depthwise） |
| 因果卷积（causal） | 第 4 章 | 只看当前与历史位置的卷积（1D 时序上右 padding 实现） |
| KDA | 第 4 章 | Kimi Delta Attention，使用短深度卷积的线性注意力（链接已有页） |
