# MLA 核心论断与证据

编号约定：C 论断 / F 公式 / N 数字。来源优先级：DeepSeek-V2 论文（MLA 定义）> K3 报告 §2.1.2（K3 改动）> K3 config.json（数值）。

## C 论断（机制与结论）

### C1：MLA 通过低秩 KV 联合压缩把每 token 缓存从 2 n_h d_h 降到 d_c

- 论断：MLA 把每 token 的 K 和 V 压成一个 d_c 维潜向量 c_t^{KV}，推理时只缓存 c_t^{KV}（外加一个共享的解耦 RoPE key）；K 和 V 在注意力计算时由学习到的上投影 W^{UK}、W^{UV} 重建。
- 来源：DeepSeek-V2 §2.1.2 Eq.(9)(10)(11)；§2.1.4 Table 1 MLA 行。
- 适用条件：d_c ≪ d_h n_h，且 W^{UK}/W^{UV} 在推理时可被吸收（见 C4）。
- 置信：已确认。

### C2：MLA 的注意力仍是全局 token-to-token softmax 注意力

- 论断：MLA 不改注意力结构，每个 query 仍对所有前序 token 求 softmax 加权和；压缩作用于存储，不作用于注意力连接图。
- 来源：DeepSeek-V2 §2.1.3 Eq.(18) 与 §2.1.1 Eq.(7) 形式相同（仅分母由 d_h 改为 d_h + d_h^R、k 由拼接 [k^C; k^R] 构成）。
- 适用条件：始终成立。
- 置信：已确认。

### C3：MLA 的解耦 RoPE 是因为 RoPE 与矩阵吸收不兼容

- 论断：若对 k_t^C = W^{UK} c_t^{KV} 直接应用 RoPE，则 RoPE 矩阵 R 位置敏感，W^{UK} 无法被吸收进 W^Q；解耦方案把 RoPE 单独放在共享 key k_t^R 上，content 部分保持可吸收。
- 来源：DeepSeek-V2 §2.1.3 第一段（"RoPE is incompatible with the low-rank KV compression"），Eq.(14)–(17)。
- 适用条件：使用 RoPE 的 MLA 配置；K3 改用 NoPE 后此冲突不存在（见 C5）。
- 置信：已确认。

### C4：推理时 W^{UK} 可吸进 W^Q、W^{UV} 可吸进 W^O，从而不显式重建 K/V

- 论断：注意力的 q^T k^C 项可改写为 q^T W^{UK} c_t^{KV} = (W^{UK T} q)^T c_t^{KV}，把 W^{UK} 吸进 W^Q 后 query 直接与 c_t^{KV} 做内积；同理 W^{UV} 吸进 W^O。RoPE 部分（k_t^R）不能吸收，仍需单独缓存。
- 来源：DeepSeek-V2 §2.1.2 末段（"W^{UK} and W^{UV} can be absorbed into W^Q and W^O, respectively, during inference"）。
- 适用条件：MLA 不对 k_t^C 应用 RoPE；推理时 W^Q、W^O 可与 W^{UK}、W^{UV} 合并存储（合并后矩阵形状不变大太多）。
- 置信：已确认。

### C5：K3 对所有 MLA 层使用 NoPE（不施加 RoPE）

- 论断：K3 §2.1.2 第二段明确——MLA 层不施加位置编码，与 KDA 层形成"全局无位置内容交互 + 位置敏感近邻混合"的分工；同时免去扩展上下文时调整 RoPE 频率基（如 YaRN）的需要。
- 来源：K3 报告 §2.1.2 第二段；config.json `mla_use_nope: true`。
- 适用条件：K3 hybrid 架构（KDA + MLA 混合）；纯 MLA 模型不适用。
- 置信：已确认。

### C6：K3 给 MLA 加 input-dependent full-rank output gate

- 论断：K3 在 MLA 末加 gate $y_t = W^O[\mathrm{Sigmoid}(W^g x_t) \odot \tilde o_t]$（Eq.(7)），W^g 满秩，允许每 token 调制从全局注意力读出的通道。
- 来源：K3 报告 §2.1.2 Eq.(7) 及其后一段（"full rank, matching the new parameterization used by KDA"）。
- 适用条件：K3 配置；DeepSeek-V2 原始 MLA 无此 gate。
- 置信：已确认。

### C7：MLA 的 KV cache 是 (d_c + d_h^R) l 元素，相比 MHA 的 2 n_h d_h l 大幅减少

- 论断：DeepSeek-V2 配置下 MLA 每 token cache = (512+64)×60 = 34560，MHA = 2×128×128×60 = 1966080，比值约 1/57；Table 1 给出四种机制对照。
- 来源：DeepSeek-V2 §2.1.4 Table 1；§2.1.2 末段（d_c = 4 d_h = 512、d_h^R = d_h/2 = 64）。
- 适用条件：DeepSeek-V2 配置；其他配置按公式重算。
- 置信：已确认。

### C8：DeepSeek-V2 报告 KV cache 减少 93.3% 是相对 DeepSeek 67B（GQA baseline），不是相对同配置 MHA

- 论断：摘要中"reduces the KV cache by 93.3%"的对照对象是 DeepSeek 67B（GQA 架构，65 层、8 KV head、d_h=128），不是同配置 MHA。同配置 MHA 下 MLA 的减少比例约 98.2%（按 Table 1 公式算）。
- 来源：DeepSeek-V2 摘要与 §3.2.3 Table 5（推理效率对比 vs DeepSeek 67B）。
- 适用条件：对比基线需明确。
- 置信：已确认（区分两个 baseline 避免误用）。

## F 公式（核心公式与来源）

- F1：标准 MHA 的 q/k/v（DeepSeek-V2 Eq.(1)(2)(3)），q_t = W^Q h_t, k_t = W^K h_t, v_t = W^V h_t；切分头 Eq.(4)(5)(6)；注意力 Eq.(7) o_{t,i} = sum_j softmax(q_{t,i}^T k_{j,i} / sqrt(d_h)) v_{j,i}；输出 Eq.(8) u_t = W^O [o_{t,1};...;o_{t,n_h}]。来源：§2.1.1。
- F2：KV 联合压缩 Eq.(9) c_t^{KV} = W^{DKV} h_t；Eq.(10) k_t^C = W^{UK} c_t^{KV}；Eq.(11) v_t^C = W^{UV} c_t^{KV}。来源：§2.1.2。
- F3：Query 压缩 Eq.(12) c_t^Q = W^{DQ} h_t；Eq.(13) q_t^C = W^{UQ} c_t^Q。来源：§2.1.2。
- F4：解耦 RoPE Eq.(14) q_t^R = RoPE(W^{QR} c_t^Q)；Eq.(15) k_t^R = RoPE(W^{KR} h_t)；Eq.(16) q_{t,i} = [q_{t,i}^C; q_{t,i}^R]；Eq.(17) k_{t,i} = [k_{t,i}^C; k_t^R]；Eq.(18) o_{t,i} = sum_j softmax(q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R)) v_{j,i}^C；Eq.(19) u_t = W^O [o_{t,1};...;o_{t,n_h}]。来源：§2.1.3。
- F5：矩阵吸收代数变形——q^T k^C = q^T W^{UK} c_t^{KV} = (W^{UK T} q)^T c_t^{KV}，故 W^{UK} 吸进 W^Q；W^{UV} 同理吸进 W^O。来源：§2.1.2 末段（论文给出结论，代数变形由结合律直接得到）。
- F6：K3 output gate Eq.(7) y_t = W^O [Sigmoid(W^g x_t) ⊙ õ_t]。来源：K3 报告 §2.1.2 Eq.(7)。

## N 数字（外部数值与实验条件）

- N1：DeepSeek-V2 MLA 配置（§2.1.2 末段 + §2.1.4）：d = 5120、n_h = 128、d_h = 128、l = 60、d_c = 4 d_h = 512、d_c' = 1536、d_h^R = d_h / 2 = 64。
- N2：DeepSeek-V2 每 token KV cache（§2.1.4 Table 1）：MHA = 2 n_h d_h l = 32768 l、GQA = 2 n_g d_h l、MQA = 2 d_h l = 256 l、MLA = (d_c + d_h^R) l = 576 l ≈ 4.5 d_h l。l = 60 时 MLA = 34560 元素。
- N3：DeepSeek-V2 vs DeepSeek 67B 摘要数字（摘要 + §3.2.3）：训练成本 -42.5%、KV cache -93.3%、最大生成吞吐 5.76×。注意：93.3% 的 baseline 是 DeepSeek 67B（GQA），不是同配置 MHA。
- N4：K3 config.json MLA 字段：q_lora_rank = 1536（即 d_c'）、kv_lora_rank = 512（即 d_c）、qk_nope_head_dim = 128（即 d_h，content 部分每头维度）、qk_rope_head_dim = 64（即 d_h^R）、v_head_dim = 128（v 每头维度）、mla_use_output_gate = true、mla_use_nope = true、num_attention_heads = 96（n_h）、hidden_size = 7168（d）、num_hidden_layers = 93。
- N5：K3 MLA 层位（config.json linear_attn_config.full_attn_layers）：[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,93]，共 24 个 MLA 层；其余 69 层为 KDA。本文不展开调度，但记录此数字供 Q3 真实数值计算使用。
- N6：K3 每 token 每 MLA 层 KV cache：d_c + d_h^R = 512 + 64 = 576 元素（注意：mla_use_nope=true 时 d_h^R=64 实际不施加 RoPE，但 config 仍保留 qk_rope_head_dim=64，按公式 cache 含此部分；K3 §2.1.2 第二段说明 NoPE 表示 query/key 不施加位置编码，但解耦分支的参数结构保留，本页按 config 数值计算 cache）。

  说明（待 draft-check 阶段进一步核实）：NoPE 是否让 k_t^R 也免缓存这一点在 K3 报告中未明确。保守起见，本页在 K3 cache 计算时同时给出"按 DeepSeek-V2 公式 (d_c + d_h^R) l"和"若 NoPE 下不再需要解耦 key 则为 d_c l"两种说法，并标注 NoPE 对 cache 的影响在报告中未明确。
