# 资料冲突

- S-NIAH-3 的检索目标｜paper/src-v3/main.tex L408-L430 表 Table 2 表头写 S-NIAH-3 是「uuid in haystack」；paper/src-v3/main.tex Experiment Continued 的 In-context retrieval 段写 S-NIAH-3 是「word-based needle in haystack」｜页面按表格写「更长的串（uuid）」
- 参数 $\alpha,\beta$ 的产生方式｜paper/src-v3/main.tex Token mixer block. 写「$\alpha, \beta$ use linear projection only」；repo/GatedDeltaNet/lit_gpt/gated_delta_net.py L157-L165 在投影后对 β 取 sigmoid、对 α 用 Mamba2 的参数化（softplus）｜页面按正文写「由线性投影得到、取值落在 $(0,1)$ 的标量」，具体非线性不写在正文
- 状态矩阵的形状｜paper/src-v3/main.tex Mamba2: Linear Attention with decay 写 $\rmS_t\in\mathbb{R}^{d_v\times d_k}$（值维乘键维）；repo/GatedDeltaNet/lit_gpt/gated_delta_net.py L224 与 gated_delta_rule_ops/chunk.py L109 按（键维乘值维）分配状态，是论文 $\rmS$ 的转置｜页面按论文统一写 $d_v\times d_k$

