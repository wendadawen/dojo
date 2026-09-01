# 逐层混合精度量化：术语表

| 术语/符号 | 首次出现 | 含义 |
|---|---|---|
| 逐层混合精度量化 | 开头 | 推理量化中按层/张量族分别指定比特档位的策略；本文主题 |
| MIX-STQ1_0 | 开头 | 该策略在 Hy4 preview 轻量版中的实例名 |
| mixed-precision training | 开头 | 训练侧同名异义概念（FP16/BF16 与 FP32 主权重混用），本文排除 |
| bpw | 开头 | 平均每权重占用 bit 数 |
| 均匀分配 | 第 1 章 | 所有层用同一量化档位的做法 |
| UD-IQ1_M | 第 1 章 | 社区已有的重要性混合方案，本文对照基线 |
| imatrix（重要性矩阵） | 第 2 章 | 用校准文本前向统计的逐权重重要性，llama.cpp 生态标准产物 |
| 校准数据/校准集 | 第 2 章 | 用于统计敏感度的一批代表性文本 |
| STQ1_0 | 第 1 章 | Sherry 稀疏三值量化在 llama.cpp 的格式，1.3125 bpw，见 Sherry 概念页 |
| IQ2_XXS / IQ1_M / IQ3_XXS / IQ4_XS | 第 1/3 章 | llama.cpp 低比特量化格式，2.0625 / 1.75 bpw 档位等 |
| Q5_K / Q6_K / Q8_0 / Q4_K | 第 3 章 | llama.cpp 较高比特格式族名称 |
| 路由专家（routed experts） | 第 3 章 | MoE 中按输入路由激活的 FFN 权重，Hy4 参数主体 |
| gate/up/down 投影 | 第 3 章 | 专家 FFN 的三个线性投影张量族 |
| 残差流（residual stream） | 第 3 章 | 层输出逐层累加的主干表示；直接写入者的误差会被后续层看到 |
| MLA | 第 3 章 | Hy4 使用的注意力变体，其分量张量在配方中取 Q8_0 |
| DSA indexer | 第 3 章 | 决定每个 query 可见 token 子集的索引张量族 |
| iHC | 第 3 章 | Hy4 架构组件名，其 *_fn、router、norms、sink 保持 F32 |
| n_expert | 第 3 章 | llama.cpp GGUF 元数据中的专家数量字段 |
| 自动提档 | 第 3 章 | llama.cpp 量化器对部分张量自动升到更高精度的内置逻辑 |
| PTQ | 开头 | 训练后量化 |
| 保留率（retention） | 第 4 章 | 量化模型评测分数相对 BF16 的百分比 |
