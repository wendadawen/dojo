# MXFP4 量化感知训练 evidence：核心论断与证据

编号：C 论断 / F 公式 / N 数字。来源优先级：原始论文与标准 > 官方 config.json（固定版本）> K3 技术报告。核心论断处于已确认状态才进入生产。

## C 论断

### C1
- 论断：K3 把 MoE 专家权重量化到 MXFP4，激活用 MXFP8，从 SFT 阶段起做量化感知训练，覆盖 SFT + RL，使模型适应量化精度损失。
- 来源：K3 技术报告 §4.1.4 "Deployment-Aware Post-Training"，原文 "we quantize the MoE expert weights — which dominate the model's parameter memory — to MXFP4 [103], with activations computed in MXFP8 ... We perform quantization-aware training (QAT) [49] throughout the entire post-training stage, covering both SFT and RL, so that the model adapts to quantization-induced precision loss."
- 适用条件：K3 的后训练部署配置
- 置信状态：已确认

### C2
- 论断：K3 在 RL 中让 rollout 与训练共享同一量化方案，消除 train-inference mismatch。
- 来源：K3 报告 §4.1.4 "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch."
- 适用条件：RL 阶段，rollout 与训练都使用 MXFP4 权重 + MXFP8 激活
- 置信状态：已确认

### C3
- 论断：K3 的非专家组件（attention 投影、latent MoE 投影、共享专家、MoE router）保持高精度，不量化。
- 来源：K3 报告 §4.1.4 "all non-expert components (attention projections, latent MoE projections, shared experts, and MoE routers) remain in higher precision."
- 适用条件：K3 选择性量化策略
- 置信状态：已确认

### C4
- 论断：draft model（EAGLE-3 风格）沿用与 target model 相同的 QAT 配置——MoE 专家权重 MXFP4、输入激活 MXFP8、非专家保持高精度。
- 来源：K3 报告 §4.1.4 "Draft fine-tuning follows the post-training QAT configuration (§ 4.1.4), with MoE expert weights in MXFP4 and their input activations in MXFP8, while non-expert modules remain in higher precision."
- 适用条件：K3 draft model 微调
- 置信状态：已确认

### C5
- 论断：MXFP4 是 OCP Microscaling Formats v1.0 定义的 4-bit 浮点格式，元素编码为 E2M1，块大小 32，共享 scale 为 E8M0（8-bit 无符号指数，幂次）。
- 来源：Rouhani et al. 2023 "Microscaling Data Formats for Deep Learning"（arXiv:2310.10537）；OCP Microscaling Formats (MX) Specification v1.0 (September 2023)
- 适用条件：OCP MX 格式族
- 置信状态：已确认

### C6
- 论断：QAT 在训练前向插入"先量化再反量化"的伪量化算子、反向用直通估计器（STE）令 $\partial\hat{w}/\partial w \equiv 1$。
- 来源：Jacob et al. 2018 "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018, pp. 2704–2713)，K3 报告引用 [49]
- 适用条件：QAT 方法的一般定义
- 置信状态：已确认（K3 引用 [49] 即此方法；K3 报告未细化 STE 细节，但 [49] 与 QAT 通行定义一致）

### C7
- 论断：MXFP4 中 32 个元素共享一个 E8M0 scale，scale 为 $s_b = 2^{e_b}$，$e_b$ 由 8-bit 无符号指数表示，覆盖范围 $2^{-127}$ 到 $2^{128}$。
- 来源：Rouhani et al. 2023；OCP MX v1.0 spec
- 适用条件：OCP MXFP4
- 置信状态：已确认

### C8
- 论断：K3 官方 config.json 的 quantization_config 字段格式为 "mxfp4-pack-quantized"，块大小 32，4-bit 浮点元素，对称量化，按 group 策略，且显式忽略 self_attn、shared_experts、mlp 的 (gate|up|gate_up|down)_proj、lm_head、vision_tower、mm_projector。
- 来源：HuggingFace `moonshotai/Kimi-K3` 仓库 `config.json`（WebFetch 于 2026-08-09 获取）
- 适用条件：K3 官方发布权重
- 置信状态：已确认（与 K3 报告 §4.1.4 文字描述一致）

### C9
- 论断：E2M1（1 符号 + 2 指数 + 1 尾数，bias=1）的正数可表示正常值为 {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}，最大可表示幅度为 6。
- 来源：OCP MX v1.0 spec；Rouhani et al. 2023；pychop 文档（MXFP4_E2M1）
- 适用条件：OCP MXFP4 E2M1 编码
- 置信状态：已确认

## F 公式

### F1
- 公式：$\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$，其中 $s_b = 2^{e_b}$（E8M0 块共享 scale），$q_i$ 为 E2M1 元素量化后的值，$\hat{x}_i$ 为反量化近似权重。
- 来源：Rouhani et al. 2023；OCP MX v1.0
- 适用条件：MXFP4 反量化
- 置信状态：已确认

### F2
- 公式（STE 约束）：$\frac{\partial \hat{w}}{\partial w} \equiv 1$（直通估计器把量化算子导数当 1）。
- 来源：Jacob et al. 2018；Bengio et al. 2013（STE 通行定义，K3 报告未细化但 [49] 即 QAT 原始来源）
- 适用条件：QAT 反向
- 置信状态：已确认

### F3（推导链，由 F1 推出）
- 公式：MXFP4 每权重有效位宽 $\approx 4 + 8/32 = 4.25$ bit（4 bit 元素 + 8 bit 共享 scale 摊到 32 个元素）；相对 BF16（16 bit）的理论压缩比 $\approx 16 / 4.25 \approx 3.76\times$。
- 来源：由 F1 与 OCP MX 块结构直接推出
- 适用条件：忽略对齐 padding 与元数据
- 置信状态：已确认

## N 数字

### N1
- 数字：K3 有 896 个路由专家（`num_experts: 896`），每 token 激活 16 个（`num_experts_per_token: 16`），另有 2 个共享专家。
- 来源：HuggingFace `moonshotai/Kimi-K3` config.json
- 适用条件：K3 官方配置
- 置信状态：已确认

### N2
- 数字：K3 `hidden_size = 7168`、`moe_intermediate_size = 3072`、`num_experts = 896`。
- 来源：config.json
- 适用条件：K3 官方配置
- 置信状态：已确认

### N3（教学示例，由 N1/N2 + 标准 3 投影 MoE 专家结构推出）
- 数字：每个路由专家参数量 $\approx 3 \times 7168 \times 3072 \approx 66.06\mathrm{M}$；896 专家共 $\approx 59.19\mathrm{B}$ 参数；BF16 下 $\approx 118.38$ GB；MXFP4 下元素 $29.60$ GB + 共享 scale $1.85$ GB $\approx 31.44$ GB；压缩比 $\approx 3.76\times$。
- 来源：由 N1/N2 config 与 DeepSeek-MoE 风格 3 投影专家（gate/up/down，由 config ignore 列表 `mlp.(gate|up|gate_up|down)_proj` 印证）推出
- 适用条件：教学估算，忽略 padding 与 latent MoE 结构内部细节；用于展示数量级而非精确部署
- 置信状态：已确认（手算已用代码复算，见 draft-check.md）

### N4
- 数字：MXFP4 块大小为 32（`group_size: 32`），与 OCP 规范一致。
- 来源：config.json `quantization_config.config_groups.group_0.weights.group_size`；OCP MX v1.0
- 适用条件：K3 部署
- 置信状态：已确认

### N5（教学示例手算）
- 数字：教学块 scale $s_b = 0.25$ 时：
  - 块 1 = [0.50, -0.25, 1.00, 0.75]，归一化后 [2.0, -1.0, 4.0, 3.0]，均落在 E2M1 可表示值内，量化无损。
  - 块 2 = [0.30, 0.10, 0.45, 0.20]，归一化后 [1.2, 0.4, 1.8, 0.8]，E2M1 最近邻舍入为 [1.0, 0.5, 2.0, 0.75]，反量化为 [0.25, 0.125, 0.50, 0.1875]，误差 [0.05, -0.025, -0.05, 0.0125]。
  - 单权重 QAT：$w=0.80$、$s_b=0.25$，归一化 $3.2$ → E2M1 最近邻 $3.0$ → $\hat{w}=0.75$，前向偏差 $0.05$；STE 把上游梯度 $g=1$ 直传为 $\partial L/\partial w = 1$。
- 来源：教学示例（人为构造，便于手算）；E2M1 可表示值集合由 C9 给出
- 适用条件：教学缩写为 4 元素块（真实块大小 32 见 N4）；scale 给定（真实由 OCP 规则按组内最大幅度选 power-of-two）
- 置信状态：已确认（代码已实跑，输出与手算一致，见 draft-check.md 与代码 C1）

### N6
- 数字：K3 激活量化用 MXFP8（E4M3 元素，8 bit，块大小 32，共享 E8M0 scale）。
- 来源：K3 报告 §4.1.4 "activations computed in MXFP8"
- 适用条件：K3 部署
- 置信状态：已确认（MXFP8 元素编码细节来自 OCP MX v1.0；K3 未细化但与规范一致）

## 来源清单

- [K3] Kimi K3 Technical Report, §4.1.4 Deployment-Aware Post-Training（约 §4.1.4 行 922–954）
- [config.json] HuggingFace `moonshotai/Kimi-K3` 仓库 `config.json`，`quantization_config` 字段（WebFetch 2026-08-09）
- [Rouhani 2023 / OCP MX] Bita Darvish Rouhani et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537 (2023)；OCP Microscaling Formats (MX) Specification v1.0 (September 2023). https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- [Jacob 2018] Benoit Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018, pp. 2704–2713.（K3 引用 [49]）
- [EAGLE-3] Yuhui Li et al. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.（K3 引用 [71]）
