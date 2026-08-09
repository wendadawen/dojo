# MXFP4 量化感知训练 glossary：术语表

| 术语 / 缩写 / 符号 | 首次出现 | 定义或含义 |
|---|---|---|
| MXFP4 | 页面开头 | OCP Microscaling Formats 定义的 4-bit 浮点格式；元素为 E2M1，每 32 元素一组共享 E8M0 scale |
| QAT | 页面开头 | Quantization-Aware Training，量化感知训练；训练前向模拟量化、反向用 STE 传梯度 |
| MoE | S1 | Mixture-of-Experts，专家混合层；router 为每个 token 选 top-k 个专家计算并加权求和（详见 wiki/moe-serving） |
| 路由专家 / routed expert | S1 | MoE 层中按 router 决定是否激活的专家子网络；K3 共 896 个 |
| 共享专家 / shared expert | S1 | MoE 层中每个 token 都计算的专家；K3 有 2 个；不量化 |
| 稀疏激活 | S1 | 每 token 只激活少量专家（K3 为 16/896），省计算不省显存 |
| FFN | S1 | Feed-Forward Network；MoE 中每个专家是一个 3 投影 FFN（gate/up/down） |
| gate_proj / up_proj / down_proj | S1 | DeepSeek-MoE 风格专家的三个投影；config.json `ignore` 列表中的 `mlp.(gate\|up\|gate_up\|down)_proj` 即指（顶层 dense）FFN 投影，不匹配路由专家路径 |
| BF16 | S1 | Brain Float 16，16-bit 浮点；K3 训练精度（`dtype: "bfloat16"`） |
| E2M1 | S2 | 4-bit 浮点元素编码：1 符号 + 2 指数 + 1 尾数，bias=1；正数正常可表示值 {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}，最大幅度 6 |
| E8M0 | S2 | 8-bit 无符号指数 scale（无尾数），表示 $s_b = 2^{e_b}$；范围 $2^{-127}$ 到 $2^{128}$ |
| 块 / block | S2 | MXFP 中共享一个 scale 的元素组；MXFP4 块大小为 32（`group_size: 32`） |
| $s_b$ | S2 | 块共享 scale，$s_b = 2^{e_b}$（E8M0） |
| $q_i$ | S2 | 第 $i$ 个元素的 E2M1 量化值 |
| $\hat{x}_i$ | S2 | 第 $i$ 个权重的反量化近似值，$\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$ |
| 有效位宽 | S2 | 每权重分摊的存储位宽；MXFP4 $\approx 4 + 8/32 = 4.25$ bit |
| PTQ | S3 | Post-Training Quantization，训练后量化；训练全程全精度、推理时才量化 |
| 伪量化 / fake-quant | S3 | QAT 前向的"先量化再反量化"算子，得到近似权重 $\hat{w}$ 用于算损失 |
| $\hat{w}$ | S3 | QAT 前向中 $w$ 经伪量化后的近似权重；前向用 $\hat{w}$ 不是 $w$ |
| STE | S3 | Straight-Through Estimator，直通估计器；令 $\partial\hat{w}/\partial w \equiv 1$，使梯度绕过不可导的量化 |
| $w$ | S3 | QAT 中原始高精度权重；梯度更新发生在 $w$ 上 |
| rollout | S4 | RL 中模型的推理生成；rollout 输出经奖励塑形成为训练 target |
| train-inference mismatch | S4 | 训练分布与推理分布不一致；RL 中 rollout 量化方案与训练不同时产生 |
| MXFP8 | S4 | OCP 8-bit 微缩浮点（E4M3 元素，块 32，共享 E8M0 scale）；K3 用于激活量化 |
| draft model | S4 | 投机解码中的小模型；K3 用 EAGLE-3 风格单层 draft，沿用同一 QAT 配置 |
| 选择性量化 | S5 | 只量化专家权重与激活、非专家组件保持高精度的策略 |
| config.json | S5 | HuggingFace `moonshotai/Kimi-K3` 仓库的模型配置文件；`quantization_config` 字段记录量化格式与 ignore 列表 |
| `mxfp4-pack-quantized` | S5 | config.json 中量化格式名；表示 MXFP4 打包量化 |
| `ignore` | S5 | config.json 中不量化的模块路径正则列表 |
| NVFP4 | 边界（术语表） | NVIDIA FP4 格式，块大小 16、FP8 scale；与 MXFP4（块 32、E8M0 scale）不同；本页不展开 |
| BFP | 边界（术语表） | Block Floating Point，全块共享一个指数、元素无独立指数；MXFP 元素有独立指数位，表达力更强；本页不展开 |

符号一致性：$s_b$ 始终指块共享 scale；$q_i$ 始终指 E2M1 量化值；$\hat{x}_i$ 与 $\hat{w}$ 都指反量化近似值（$\hat{w}$ 用于单权重 QAT 场景）。全文不混用。
