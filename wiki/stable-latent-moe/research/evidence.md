# Stable LatentMoE 核心论断与证据

来源优先级：K3 报告 §2.3-2.3.3（原始论文级）> K3 官方 `config.json`（固定版本官方文档）> DeepSeekMoE 论文（被引用的来源）。本页核心论断全部来自前两类。

## C 论断（事实/机制陈述）

### C1
- **论断**：LatentMoE 通过分离全模型宽度与路由专家宽度，使扩大专家池不再让每个被选中的路由专家接收全宽 $d$ 维 token 表示——共享专家走全宽，路由专家在紧凑隐空间 $\ell$ 操作。
- **来源定位**：K3 报告 §2.3 第 425-428 行。
- **适用条件**：LatentMoE 架构（K3 报告引用 [32]）。
- **置信状态**：已确认。

### C2
- **论断**：K3 把路由专家池扩大到 896、每 token 激活 16，对应稀疏度 56。
- **来源定位**：K3 报告 §2.3 第 428 行；`config.json` 的 `num_experts=896`、`num_experts_per_tok=16`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### C3
- **论断**：路由路径由 $W_\downarrow$、门控多分支专家 FFN、$W_\uparrow$ 串联，是近四个连续矩阵乘法的链。
- **来源定位**：K3 报告 §2.3 第 429-431 行。
- **适用条件**：LatentMoE 路由分支结构。
- **置信状态**：已确认。

### C4
- **论断**：上述结构在 2.8T 参数规模下产生路由分支内部激活爆炸。
- **来源定位**：K3 报告 §2.3 第 431-432 行。
- **适用条件**：K3 规模（2.8T 参数）。
- **置信状态**：已确认。

### C5
- **论断**：平衡近 $10^3$ 个专家的负载超出了已有无辅助损失方法的适用范围。
- **来源定位**：K3 报告 §2.3 第 432-433 行。
- **适用条件**：专家数近千。
- **置信状态**：已确认。

### C6
- **论断**：Stable LatentMoE 由三件稳定化组成——Normalized LatentMoE（RMSNorm 插在路由聚合后、上投影前）、SiTU-GLU（有界激活）、Quantile Balancing（负载均衡）。
- **来源定位**：K3 报告 §2.3 第 461-463 行。
- **适用条件**：K3 实现。
- **置信状态**：已确认。

### C7
- **论断**：K3 沿用 DeepSeekMoE 的 shared+routed 组织——共享专家无路由、所有 token 必经，路由专家参与 Top-k。
- **来源定位**：K3 报告 §2.3 第 464 行（"the layer follows the shared- and routed-expert organization of DeepSeekMoE [23]"）；DeepSeekMoE 论文 §3.2 公式 $h_t^l = \sum_{i=1}^{K_s}\mathrm{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t}\mathrm{FFN}_i(u_t^l) + u_t^l$。
- **适用条件**：DeepSeekMoE 及其后续（含 K3）。
- **置信状态**：已确认。

### C8
- **论断**：K3 每层固定 $N_s = 2$ 个全宽共享专家。
- **来源定位**：K3 报告 §2.3 第 478 行；`config.json` 的 `num_shared_experts=2`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### C9
- **论断**：原始 LatentMoE 直接把 $W_\uparrow$ 作用于聚合表示 $u$，$u$ 的尺度随选中专家与路由权重变化；K3 在 $u$ 与 $W_\uparrow$ 之间插入 RMSNorm，降低路由分支对尺度变化的敏感度。
- **来源定位**：K3 报告 §2.3.1 第 480-484 行。
- **适用条件**：Normalized LatentMoE（K3 改造）。
- **置信状态**：已确认。

### C10
- **论断**：RMSNorm 除了稳定训练，还持续改善验证损失与下游基准。
- **来源定位**：K3 报告 §2.3.1 第 484-485 行。
- **适用条件**：K3 训练配置。
- **置信状态**：已确认（K3 报告自称，未提供独立复现）。

### C11
- **论断**：SiTU-GLU 的两分支乘积有上界 $\beta_1\beta_2$；K3 取 $\beta_1=4, \beta_2=25$，上界为 100。
- **来源定位**：K3 报告 §2.3.2 第 497-502 行（Eq. 12）与第 541 行；`config.json` 的 `activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`。
- **适用条件**：SiTU-GLU 定义。
- **置信状态**：已确认。

### C12
- **论断**：Quantile Balancing 从路由分数的 $(1-k/n)$ 分位数设置每个专家的 bias，使每专家获得目标负载 $q = mk/n$。
- **来源定位**：K3 报告 §2.3.3 第 547-596 行（Eq. 13-14）。
- **适用条件**：QB 实现。
- **置信状态**：已确认。

### C13
- **论断**：QB 在 DeepSeek-V3 无辅助损失路由（auxiliary-loss-free，bias 加在 Top-k 选择分数上但不进入 mixture 权重）基础上改进 bias 更新规则。
- **来源定位**：K3 报告 §2.3.3 第 547-557 行。
- **适用条件**：K3 路由。
- **置信状态**：已确认。

## F 公式（核心公式）

### F1（Eq. 11，Stable LatentMoE 层输出）
- **公式**：

$$u = \sum_{i \in T_k(x)} p_i \, E_i^{\text{routed}}(W_\downarrow x), \qquad y = \sum_{j=1}^{N_s} E_j^{\text{shared}}(x) + W_\uparrow \, \mathrm{RMSNorm}(u).$$

- **来源定位**：K3 报告 §2.3 第 467-474 行（Eq. 11）。
- **适用条件**：Stable LatentMoE 层。
- **置信状态**：已确认。

### F2（DeepSeekMoE 组织公式，被引用）
- **公式**：

$$h_t^l = \sum_{i=1}^{K_s} \mathrm{FFN}_i(u_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t} \, \mathrm{FFN}_i(u_t^l) + u_t^l.$$

- **来源定位**：DeepSeekMoE 论文 §3.2（arXiv:2401.06066）。
- **适用条件**：DeepSeekMoE 架构。
- **置信状态**：已确认。本文只引用结果（shared 无路由、routed Top-k），不展开 $g_{i,t}$ 的 softmax 与 Top-k 定义。

### F3（SiTU-GLU，被引用）
- **公式**：

$$\mathrm{SiTU\text{-}GLU}(x) = \beta_1 \tanh\!\left(\frac{W_g x}{\beta_1}\right) \odot \mathrm{Sigmoid}(W_g x) \odot \beta_2 \tanh\!\left(\frac{W_u x}{\beta_2}\right).$$

- **来源定位**：K3 报告 §2.3.2 第 499-502 行（Eq. 12）。
- **适用条件**：SiTU-GLU 定义。
- **置信状态**：已确认。本文只引用结论（有界、上界 $\beta_1\beta_2$、近原点近似 Swish），完整机制见 `wiki/situ-glu/`。

### F4（QB 更新，被引用）
- **公式**：

$$\tilde{b}_j^{(t+1)} \leftarrow -\mathrm{quantile}_{1-k/n}\!\left(s_{:,j}^{(t)} - \alpha^{(t)}\right), \qquad b_j^{(t+1)} \leftarrow \tilde{b}_j^{(t+1)} - \mathrm{mean}\!\left(\tilde{b}^{(t+1)}\right) \cdot \mathbf{1}.$$

- **来源定位**：K3 报告 §2.3.3 第 586-589 行（Eq. 14）。
- **适用条件**：QB 实现。
- **置信状态**：已确认。本文只引用结论（从分位数设 bias），完整推导见 `wiki/quantile-balancing/`（占位）。

## N 数字（外部数字）

### N1
- **数字**：$d = 7168$（全模型宽度，`hidden_size`）。
- **来源定位**：`config.json` 的 `text_config.hidden_size`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### N2
- **数字**：$\ell = 3584$（路由隐空间宽度，`routed_expert_hidden_size`）。
- **来源定位**：`config.json` 的 `text_config.routed_expert_hidden_size`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### N3
- **数字**：`moe_intermediate_size = 3072`（路由专家 FFN 中间维度）。
- **来源定位**：`config.json` 的 `text_config.moe_intermediate_size`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### N4
- **数字**：$N = 896$（路由专家总数，`num_experts`）；$k = 16$（每 token 激活，`num_experts_per_token`）；$N_s = 2$（共享专家数，`num_shared_experts`）。
- **来源定位**：`config.json` 与 K3 报告 §2.3 第 428 行。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### N5
- **数字**：稀疏度 $N/k = 896/16 = 56$。
- **来源定位**：K3 报告 §2.3 第 428 行（"a sparsity of 56"）。
- **适用条件**：K3 配置。
- **置信状态**：已确认。

### N6
- **数字**：K3 总参数规模 2.8T。
- **来源定位**：K3 报告 §2.3 第 431 行（"the 2.8-trillion-parameter scale"）。
- **适用条件**：K3 模型整体。
- **置信状态**：已确认。

### N7
- **数字**：SiTU-GLU 上界 $\beta_1\beta_2 = 4 \times 25 = 100$。
- **来源定位**：K3 报告 §2.3.2 第 541 行与 Fig. 4 标注；`config.json` 的 `activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`。
- **适用条件**：K3 的 SiTU-GLU 配置。
- **置信状态**：已确认。

### N8
- **数字**：`latent_moe_use_norm = true`、`first_k_dense_replace = 1`、`moe_layer_freq = 1`、`num_hidden_layers = 93`、`moe_router_activation_func = "sigmoid"`、`topk_method = "noaux_tc"`（auxiliary-loss-free）。
- **来源定位**：`config.json`。
- **适用条件**：K3 配置。
- **置信状态**：已确认。
