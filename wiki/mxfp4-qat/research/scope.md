# MXFP4 量化感知训练 scope：内容范围

## 0. 版本与定位

- 概念名称：MXFP4 量化感知训练；英文：MXFP4 Quantization-Aware Training（MXFP4 QAT）；常见缩写：MXFP4 QAT
- 本文语境：大模型（尤其 MoE 大模型）后训练与部署场景下，把权重压到 4-bit 微缩浮点格式并让模型在训练中适应这一精度损失的方法；事实来源以 Kimi K3 技术报告 §4.1.4 与 HuggingFace 官方 `config.json` 为准
- 歧义检查：
  - "MXFP4" 中的 "MX" 指 OCP Microscaling（微缩浮点格式族），不是 Mixed-precision（混合精度）或其它缩写。状态：已裁定（OCP Microscaling Formats v1.0，Rouhani et al. 2023）
  - "QAT"（Quantization-Aware Training，量化感知训练）与 PTQ（Post-Training Quantization，训练后量化）是两类不同方法，不合并。状态：已裁定（Jacob et al. 2018）
  - "MXFP4" 与 "NVFP4"（NVIDIA FP4，块大小 16、FP8 scale）是两种相关但不同的 4-bit 浮点格式；本文只讲 MXFP4，NVFP4 仅在边界处一句区分。状态：已裁定（OCP spec vs NVIDIA Blackwell 文档）

## 1. 概念含义

- 一句话定义：把模型权重（在 K3 中是 MoE 专家权重）编码到 4-bit 微缩浮点格式 MXFP4，并在整个后训练阶段让前向模拟这种量化、反向用直通估计器传梯度，使模型在部署时用真正的 4-bit 权重仍能保持训练时的精度。
- 正式定义：
  - MXFP4 格式（OCP Microscaling Formats v1.0，Rouhani et al. 2023）：元素为 E2M1（1 符号 + 2 指数 + 1 尾数，共 4 bit），每 32 个元素为一组共享一个 E8M0（8-bit 无符号指数，幂次 scale）的 block-scale；反量化 $\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$
  - QAT（Jacob et al. 2018）：训练前向插入"先量化再反量化"的伪量化算子得到近似权重 $\hat{w}$，用 $\hat{w}$ 计算损失；反向用直通估计器（STE）令 $\partial\hat{w}/\partial w \equiv 1$，使梯度绕过不可导的量化步骤
  - K3 部署（§4.1.4）：MoE 专家权重量化到 MXFP4，激活用 MXFP8；QAT 贯穿 SFT 与 RL 全后训练阶段；RL 中 rollout 与训练共享同一量化方案；非专家组件保持高精度
- 本页讲一组紧密耦合的内容：MXFP4 格式是什么 → 为什么量化会让精度变差 → QAT 怎么在前向/反向模拟量化 → K3 怎么把它贯穿到 RL 且不产生 train-inference mismatch

### 包括什么（每项为何属于本概念）

- MoE 专家权重为什么是量化首选 → Q1 的动机前提
- MXFP4 块结构（E2M1 元素 + E8M0 共享 scale）、反量化公式 → Q2 的机制对象
- 量化为什么引入误差（粗离散级 + 舍入）→ Q3 的"为什么需要 QAT"
- QAT 的前向伪量化与反向 STE → Q3 的核心机制
- PTQ 与 QAT 的对照 → Q3 的方法定位
- K3 把 QAT 贯穿 SFT 与 RL、rollout 与训练共享量化方案 → Q4 的部署核心
- K3 选择性量化（专家 MXFP4、激活 MXFP8、非专家高精度）与 config.json 证据 → Q5 的工程判断
- draft model 沿用同一 QAT 配置 → Q4 的延伸（投机解码 draft 与 target 一致性）

### 不包括什么（排除理由）

- 通用"什么是量化"基础（scale、zero-point、对称/非对称、per-tensor vs per-channel、INT8 量化）：由前置概念页 `quantization-basics` 负责，本页只引用其结论，不重新推导。前置页未生成 → 占位
- QAT 在 CV 模型 / 量化到整数推理（Jacob 2018 原文场景）的细节：本文只取 QAT 的核心机制（伪量化前向 + STE 反向），不展开整数算术 only 推理
- MXFP4 硬件实现（Tensor Core 指令、TMA、scale 张量布局）：属 GPU 执行模型与硬件实现层，不影响学习目标
- NVFP4（块大小 16、FP8 scale）的精度差异、OAS/MBS 等 MXFP 精度增强方法：属格式族细化，不影响"K3 用 MXFP4 + QAT"的主线
- RL 算法本身（PPO 目标、优势函数、KL 散度）：本页只用"RL 中 rollout 产出训练 target、训练据此更新参数"这一最小机制；不展开 RL 算法
- MoE 路由、专家并行、PD 分离等服务系统问题：由 `moe-serving` 概念页负责，本页只在动机处引用"MoE 权重大"这一结论
- K3 的其它训练创新（AttnRes、SiTU-GLU、KDA、LK loss、EAGLE-3 draft 结构）：与本概念无依赖，不展开

### 相邻概念

- INT4 / INT8 量化：与 MXFP4 同为低位宽量化，但用整数元素而非浮点元素；本页只在"为什么用浮点而非整数"边界处一句区分
- Block Floating Point（BFP）：与 MXFP 都用共享 scale，但 BFP 全张量/全块共享一个指数且元素无独立指数，MXFP 元素仍有自己的指数位 → MXFP 表达力更强。不展开
- NVFP4：块大小 16、FP8 scale，NVIDIA Blackwell 原生；与 MXFP4（块 32、E8M0 scale）不同；只在边界一句区分
- GPTQ / AWQ / SmoothQuant 等 PTQ 方法：训练后、无需重训的量化路径，与 QAT 是对照关系；本页在 PTQ vs QAT 处一句带过，不展开具体算法

## 2. 学习目标

### Q1：为什么 MoE 模型的"专家权重"是量化压缩的首选对象，量化到 MXFP4 能省多少显存？

- 完成答案：MoE 模型专家数量多（K3 共 896 个路由专家，config.json `num_experts`）、每个专家是一个 3 投影 FFN（gate/up/down），所有专家权重必须全量驻留显存（稀疏激活只省计算不省显存）；按 config `hidden_size=7168`、`moe_intermediate_size=3072` 手算，896 专家共约 59.2B 参数；BF16 下约 118.4 GB，MXFP4（每元素 4 bit + 每 32 元素共享 1 字节 E8M0 scale）下约 31.4 GB，压缩比约 3.76×；非专家组件（attention、shared expert、router）参数量远小且频繁参与每 token 计算，不是显存瓶颈
- 为什么是核心目标：不看清"专家权重主导显存"就读不懂"为什么只量化专家、为什么量化值得付出 QAT 成本"
- 依赖内容：S1 的 MoE 专家结构与显存特性；F1 反量化公式；N1–N3 数字

### Q2：MXFP4 一个权重值是怎么编码的，为什么"32 个一组共享一个 scale"比"每个值独立 4-bit"表达力强？

- 完成答案：MXFP4 把每 32 个连续权重组成一个块；块内共享一个 E8M0 scale（8-bit 无符号指数，取值 $s_b = 2^{e_b}$，覆盖 $2^{-127}$ 到 $2^{128}$ 的极宽范围）；每个权重存为 4-bit E2M1 浮点元素（1 符号 + 2 指数 + 1 尾数，正数可表示值约 {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}）；反量化 $\hat{x}_i = s_b \cdot \mathrm{FP4}(q_i)$；每个权重的有效位宽约 $4 + 8/32 = 4.25$ bit；共享 scale 让整块的动态范围由 scale 承担（接近 FP32 的范围），元素位宽只承担块内相对精度，因此即便只有 4 bit 也能保持可观表达力；可手算一个 4 元素块（教学缩写）的量化与反量化
- 为什么是核心目标：MXFP4 格式是 QAT 模拟的对象，不看清"块内共享 scale + 窄元素"就读不懂"为什么 4-bit 还能用"
- 依赖内容：S2 的格式结构与图示；F1 反量化公式；E1 手算

### Q3：量化感知训练（QAT）在前向和反向分别做了什么，它和训练后量化（PTQ）的关键差别在哪？

- 完成答案：QAT 前向用伪量化——先把高精度权重 $w$ 按 MXFP4 规则量化再反量化得到近似权重 $\hat{w}$，用 $\hat{w}$（不是 $w$）算前向与损失；反向用直通估计器（STE）把 $\partial\hat{w}/\partial w$ 当作 1，让梯度绕过不可导的量化步骤直传到 $w$，于是梯度更新发生在原始高精度权重上；PTQ 在训练时全程用全精度权重、推理时才量化，模型从未见过量化引入的离散误差；QAT 让模型在训练中每个前向都见到该误差，于是权重会朝"在量化后仍能减小损失"的方向移动；可手算单权重 $w=0.80$、scale $s_b=0.25$ 的 fake-quant 前向（$\hat{w}=0.75$、偏差 $0.05$）与 STE 反向（梯度 $g=1$ 直传）
- 为什么是核心目标：QAT 的全部机制就是这两件事；不理解它就读不懂 K3 为什么能"压到 4-bit 还不掉点"
- 依赖内容：S3 的伪量化与 STE 机制；F2 STE 约束；E2 手算；代码 C1
- 依赖前置：`quantization-basics`（伪量化、STE 的最小说明；本页引用其结论）

### Q4：K3 为什么把 QAT 贯穿 SFT 和 RL 两个阶段，且 RL 中 rollout 与训练必须共享同一量化方案？

- 完成答案：SFT 阶段做 QAT 让模型适应量化精度损失；RL 阶段模型一边自己生成数据（rollout，即推理）一边据此更新参数（训练），rollout 的输出经奖励塑形成为训练 target；若 rollout 用量化方案 A、训练用量化方案 B，则 target 来自分布 A、梯度优化分布 B，两者不一致即 train-inference mismatch，会使 RL 训练崩坏或收敛到错误方向；K3 让 rollout 与训练共享同一 MXFP4 量化方案，消除该 mismatch；draft model（EAGLE-3 风格）也沿用同一 QAT 配置，保证 target 与 draft 的量化行为一致
- 为什么是核心目标：RL 量化的核心难点就在 train-inference match；不理解它就读不懂"为什么不能训练 FP、推理量化"
- 依赖内容：S4 的 K3 部署与 RL 一致性；N4；C2 图示
- 内联最小 RL 框架：rollout 产出训练 target；不展开 RL 算法

### Q5：K3 量化了哪些组件、不量化哪些，为什么这样划分？

- 完成答案：量化对象——MoE 路由专家权重 → MXFP4；专家输入激活 → MXFP8；非量化对象——attention 投影、latent MoE 投影、共享专家、MoE router、lm_head、vision tower、mm_projector 保持高精度；`config.json` 的 `quantization_config` 字段证据：`format: "mxfp4-pack-quantized"`、`group_size: 32`、`num_bits: 4`、`type: "float"`、`strategy: "group"`、`symmetric: true`，`ignore` 列表匹配 `self_attn`、`shared_experts`、`mlp.(gate|up|gate_up|down)_proj`、`lm_head`、`vision_tower`、`mm_projector`；动机：路由专家占参数绝大部分但稀疏激活（每 token 仅 16/896 激活），量化收益最大、对每 token 计算的质量影响相对小；非专家组件频繁参与每个 token 计算、对精度更敏感且参数占比小，保持高精度既保质量又几乎不增显存
- 为什么是核心目标：选择性量化是 K3 部署的核心工程判断，也是读懂 config.json 的钥匙
- 依赖内容：S5 的选择性量化；N5 config 证据；C3 表格

## 3. 内容分级

### 核心内容（缺一则学习目标答不全）

- MoE 专家权重大且需全量驻留 → Q1
- MXFP4 块结构（E2M1 + E8M0 + 块大小 32）与反量化公式 F1 → Q2
- 量化误差来源（粗离散级 + 舍入）→ Q3
- QAT 前向伪量化与反向 STE、F2、E2 手算 → Q3
- PTQ vs QAT 对照 → Q3
- K3 把 QAT 贯穿 SFT+RL、rollout/训练共享方案 → Q4
- draft model 沿用 QAT 配置 → Q4
- K3 选择性量化与 config.json 证据 → Q5

### 辅助内容（不直接构成核心答案但消除障碍）

- 4-bit 浮点 vs 4-bit 整数的区别（为什么用浮点）
- E2M1 可表示值集合的直观来源
- "有效位宽 4.25 bit" 的算法
- 内存压缩比的 3.76× 与 BF16 的对照

### 扩展内容（与概念相关但不影响学习目标，本页排除）

- NVFP4 与 MXFP4 的精度对比：排除（属格式族细化，不影响主线）
- OAS / MBS 等 MXFP 精度增强方法：排除（属研究前沿）
- 量化对推理速度（不只是显存）的定量影响：排除（属硬件实现层）
- K3 之外其它模型的 MXFP4 部署经验：排除（本页只以 K3 为来源）

## 4. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | 概念页链接 / 生成状态 | 递归层级 |
|---|---|---|---|
| quantization-basics（量化基础：scale、伪量化、STE、对称量化） | Q3 | `wiki/quantization-basics/` 未生成 → 占位（编排者另行安排生成） | depth-1（占位，不递归生成） |
| MoE 大模型推理与服务基础（MoE 层结构、router、专家、稀疏激活） | Q1、Q5 | `wiki/moe-serving/index.html`（已存在） | depth-0（已存在） |
| Transformer 层与 FFN 结构 | Q1 | `wiki/moe-serving/index.html` 内已包含最小说明 | depth-0 |
| RL 中 rollout 与训练 target 的最小关系 | Q4 | 无独立概念页，正文内联一句话框架 | 不递归 |

说明：用户（编排者）指定 quantization-basics 未生成则占位、不递归生成。本文正文首次依赖它时给出占位链接与最小衔接，不内联大段背景。

## 5. 明确不展开的内容

- OCP Microscaling 规范的完整位级编码（subnormal、NaN、Inf 的处理）：与"K3 用 MXFP4 + QAT"主线无关，只在术语表标注 E2M1/E8M0 含义
- MXFP4 scale 的精确选取算法（floor/ceil 规则的边界情况）：本页用"按组内最大幅度选 power-of-two，使归一化值落入 E2M1 表示范围"这一表述，具体 floor/ceil 规则引 OCP spec，不手算推导（避免引入未核对细节）
- EAGLE-3 / LK loss 的完整算法：与 QAT 主线无关，只在 Q4 提"draft 沿用同一 QAT 配置"
- MoE router 的 sigmoid / noaux_tc top-k 选择：由 `moe-serving` 负责
- RL 的奖励设计、任务合成：与量化机制无关

## 6. 常见误解和适用边界

### 误解

1. 误解：MXFP4 就是"4-bit 整数量化"。
   正确：MXFP4 元素是 4-bit 浮点（E2M1：1 符号 + 2 指数 + 1 尾数），有独立指数位；整数量化元素是定点整数、无独立指数。
   形成原因：两者都是"4-bit 量化"，名字像。
   影响：Q2。

2. 误解：QAT 让模型"恢复"量化损失的精度。
   正确：QAT 不恢复精度——权重仍以 4-bit 部署；QAT 让模型权重在训练中朝"在 4-bit 下仍能减小损失"的方向调整，从而补偿（部分）精度损失带来的质量下降，补偿程度取决于训练数据与 QAT 配置。
   形成原因："感知"一词暗示"消除"。
   影响：Q3。

3. 误解：MXFP4 每 32 元素共享 scale，所以等价于"32 个值共用一个数"。
   正确：共享的只是 scale（动态范围），每个元素仍独立存 4-bit 浮点值（含自己的符号、指数、尾数），块内相对差异由元素承担。
   形成原因：把 block-scale 与"块内所有值相等"混淆。
   影响：Q2。

4. 误解：训练用 FP、推理用 MXFP4 的 PTQ 与 QAT 差别不大，反正都量化。
   正确：PTQ 训练时从未见过量化误差，推理时量化引入分布外（out-of-distribution）的权重，精度下降通常显著大于 QAT；RL 中 PTQ 还会引入 rollout（推理分布）与训练分布的 mismatch。QAT 的核心收益就是"训练时已模拟该误差"。
   形成原因：把"是否量化"当成唯一变量，忽略"训练时是否模拟量化"。
   影响：Q3、Q4。

5. 误解：既然 MXFP4 好，应该把所有组件都量化到 4-bit。
   正确：K3 只量化 MoE 专家权重，非专家组件（attention、shared expert、router）保持高精度。原因：非专家频繁参与每 token 计算、对精度更敏感、参数占比小，量化收益不抵质量损失。
   形成原因：把"4-bit 压缩"当成无差别最优。
   影响：Q5。

### 适用边界

- 概念解决什么：把 MoE 大模型权重压到 4-bit 浮点部署、且训练时让模型适应这一精度损失
- 不解决什么：不解决"4-bit 在所有硬件上都比 8-bit 快"（取决于硬件原生支持）；不解决"任何模型都能无损压到 4-bit"（QAT 能补偿多少取决于模型与训练）
- 结论成立条件：MXFP4 硬件原生支持或高效模拟、QAT 贯穿足够长训练、量化对象选择合理（专家权重大且稀疏激活）
- 条件不满足时：硬件不支持时 4-bit 反而慢；QAT 训练不足时精度下降显著；量化了精度敏感组件时质量崩坏
- "train-inference mismatch 消除"成立条件：rollout 与训练用同一量化方案、同一 scale 选择、同一数值路径；任一不一致即引入 mismatch
