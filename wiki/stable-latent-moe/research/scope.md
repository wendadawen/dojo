# Stable LatentMoE 内容范围

## 1. 概念歧义处理

概念名 "Stable LatentMoE" 由 Kimi K3 技术报告（§2.3）提出，是 LatentMoE 的稳定化变体，无同名缩写或跨领域歧义。

相关名称需要区分：

- **LatentMoE**：原始架构（K3 报告引用 [32]），将路由专家操作空间从全模型宽度 $d$ 压缩到紧凑隐空间 $\ell$。本文只介绍其最小必要结构，作为 Stable LatentMoE 的对照基线，不展开原始论文的训练动力学。
- **DeepSeekMoE**：共享专家 + 路由专家的组织方式（K3 报告引用 [23]），K3 沿用。本文只引用其"shared 必经 + routed Top-k"的组织公式，不展开细粒度专家切分等独立内容。
- **Stable LatentMoE**：本文主题 = LatentMoE 架构 + 三件稳定化（Normalized LatentMoE / SiTU-GLU / Quantile Balancing）。

状态：已裁定。本文采用 K3 报告 §2.3 的定义。

## 2. 概念含义

### 2.1 定义

- **概念名称**：Stable LatentMoE（稳定隐空间混合专家）
- **英文名称**：Stable LatentMoE
- **一句话定义**：一种把路由专家放进紧凑隐空间、再用三件稳定化（路由分支归一化、有界激活、分位数负载均衡）修复极端稀疏下两大失败模式的 MoE 层结构。
- **正式定义**（K3 报告 Eq. 11）：对输入 $x \in \mathbb{R}^d$，共享专家走全宽路径直接处理 $x$；路由路径先把 $x$ 投影到 $z = W_\downarrow x \in \mathbb{R}^\ell$，在隐空间 $\ell$ 上分发给选中的路由专家，聚合后再经 RMSNorm 与上投影 $W_\uparrow$ 回到 $\mathbb{R}^d$，最后与共享分支相加：

$$u = \sum_{i \in T_k(x)} p_i \, E_i^{\text{routed}}(W_\downarrow x), \qquad y = \sum_{j=1}^{N_s} E_j^{\text{shared}}(x) + W_\uparrow \, \mathrm{RMSNorm}(u).$$

其中 $E_j^{\text{shared}}: \mathbb{R}^d \to \mathbb{R}^d$、$E_i^{\text{routed}}: \mathbb{R}^\ell \to \mathbb{R}^\ell$、$p_i$ 是路由权重、$N_s$ 是共享专家数。

### 2.2 本文语境

本文以 Kimi K3 的实现为正本。K3 配置（来自 `config.json`）：

| 量 | 含义 | K3 取值 |
|---|---|---|
| $d$ | 全模型宽度（`hidden_size`） | 7168 |
| $\ell$ | 路由隐空间宽度（`routed_expert_hidden_size`） | 3584 |
| `moe_intermediate_size` | 路由专家 FFN 中间维度 | 3072 |
| `intermediate_size` | dense FFN / 共享专家 FFN 中间维度 | 33792 |
| $N$（`num_experts`） | 路由专家总数 | 896 |
| $k$（`num_experts_per_tok`） | 每 token 激活路由专家数 | 16 |
| $N_s$（`num_shared_experts`） | 共享专家数 | 2 |
| `latent_moe_use_norm` | 是否插入 RMSNorm | true |
| 稀疏度 $N/k$ | 路由专家总数 ÷ 每 token 激活数 | 56 |

### 2.3 包括什么

- **LatentMoE 的最小结构**：$W_\downarrow$ / 紧凑隐空间 $\ell$ / 路由专家 / $W_\uparrow$ / 共享专家走全宽。属于本概念的核心架构前提。
- **两个失败模式**：路由分支连续矩阵乘法链导致的激活爆炸；近千专家下无辅助损失负载均衡失效。属于 Stable LatentMoE 要解决的问题。
- **三件稳定化**：Normalized LatentMoE（RMSNorm 插入位置与作用）、SiTU-GLU（有界激活的作用，引用 `wiki/situ-glu/`）、Quantile Balancing（分位数负载均衡，引用占位）。这是 "Stable" 的全部内容。
- **K3 的关键参数取值**：上表所列。用于让读者把抽象结构与实际规模对应。
- **DeepSeekMoE 的 shared+routed 组织公式**：仅引用结果（共享专家无路由、路由专家 Top-k），作为 K3 层结构的来源。不展开细粒度切分。

### 2.4 不包括什么

- **LatentMoE 原始论文 [32] 的训练动力学分析**：排除。原始论文的损失景观分析不影响理解 K3 的稳定化改造，且与本文主题重叠会冲淡主线。
- **DeepSeekMoE 的细粒度专家切分策略**：排除。细粒度切分（把每个专家切小、增加激活数）是 DeepSeekMoE 的另一独立贡献，K3 沿用的只是 shared+routed 组织，不展开切分。
- **Quantile Balancing 的完整推导与直方图估计实现**：排除。QB 在本文只作为"三件稳定化之一"被引用，其完整机制（Top-(k+1) 截断、分位数更新公式、全批量直方图估计）属于独立概念页 `wiki/quantile-balancing/`（占位）。
- **SiTU-GLU 的 softcap 推导与极限情形**：排除。SiTU-GLU 在本文只引用其结论（有界、上界 $\beta_1\beta_2 = 100$、近原点近似 Swish），完整机制见 `wiki/situ-glu/`。
- **K3 的注意力层、Block AttnRes、KDA 线性注意力**：排除。本文只讲 FFN/MoE 层，注意力结构属于其他独立概念。
- **专家并行、All-to-All 通信的工程实现**：排除。属于 `wiki/moe-serving/` 与 `wiki/gpu-communication/` 的范围。
- **K3 的训练超参数（学习率、batch 大小、训练 token 数）**：排除。不影响理解层结构。

### 2.5 相邻概念

| 相邻概念 | 关键区别 | 是否纳入 |
|---|---|---|
| MoE（混合专家）基础 | MoE 是路由 + 专家的通用框架；Stable LatentMoE 是其中一种具体的层结构 | 不纳入，引用 `wiki/moe-serving/` |
| DeepSeekMoE | 提出了 shared+routed 组织；Stable LatentMoE 沿用此组织，再加隐空间分离与三件稳定化 | 只引用其组织公式 |
| LatentMoE | 提出了隐空间分离；Stable LatentMoE = LatentMoE + 三件稳定化 | 纳入最小必要结构作为对照 |
| SiTU-GLU | 一种有界激活函数；是 Stable LatentMoE 的稳定化组件之一 | 引用 `wiki/situ-glu/`，不重复 |
| Quantile Balancing | 一种负载均衡方法；是 Stable LatentMoE 的稳定化组件之一 | 引用占位，不展开 |
| RMSNorm | 一种归一化；在 Stable LatentMoE 中被插入到特定位置 | 只讲插入位置与作用，不讲 RMSNorm 本身 |

## 3. 学习目标

### Q1：为什么 LatentMoE 要把路由专家放进比全模型宽度更窄的隐空间？这解决了什么问题，代价是什么？

- **完成答案**：读者应能说明——扩大专家池与激活专家数会膨胀路由通信与专家权重流量；LatentMoE 让共享专家保留全宽处理通用变换、路由专家在宽度 $\ell < d$ 的隐空间操作，使扩大路由专家数不再线性增加全宽流量；代价是路由分支多出 $W_\downarrow$ 与 $W_\uparrow$ 两次投影，形成更长的矩阵乘法链。
- **为什么是核心目标**：不理解动机就无法判断三件稳定化在修复什么。
- **依赖内容**：MoE 基础（路由 + 专家）、shared+routed 组织、$W_\downarrow$/$W_\uparrow$ 投影。

### Q2：Stable LatentMoE 的层结构是什么？共享分支、路由分支、$W_\downarrow$、$W_\uparrow$、RMSNorm 各自的位置和职责是什么？

- **完成答案**：读者应能画出/复述 Eq. 11 的数据流——共享专家 $E_j^{\text{shared}}$ 直接吃 $x \in \mathbb{R}^d$；路由路径先 $W_\downarrow$ 把 $x$ 投到 $z \in \mathbb{R}^\ell$，分发到 Top-k 路由专家 $E_i^{\text{routed}}$ 得到加权聚合 $u$，RMSNorm 作用于 $u$，$W_\uparrow$ 把归一化后的 $u$ 投回 $\mathbb{R}^d$，最后两支相加。
- **为什么是核心目标**：这是本文的概念定义本身。
- **依赖内容**：DeepSeekMoE 组织公式、RMSNorm 的作用（不展开 RMSNorm 本身）、Top-k 路由。

### Q3：极端稀疏下的两个失败模式是什么？Stable LatentMoE 用哪三件稳定化分别修复哪一个？

- **完成答案**：读者应能说明——（a）路由分支 $W_\downarrow \to$ 多分支 FFN $\to W_\uparrow$ 近四个连续矩阵乘法链条件数差，在 2.8T 参数规模下产生激活爆炸；修复方法：Normalized LatentMoE（在聚合后、上投影前插 RMSNorm）+ SiTU-GLU（有界激活，上界 $\beta_1\beta_2 = 100$）。（b）平衡近千个专家的负载超出了无辅助损失方法的适用范围；修复方法：Quantile Balancing（从路由分数分位数设置专家 bias）。
- **为什么是核心目标**：这是 "Stable" 一词的全部含义。
- **依赖内容**：LatentMoE 的路由分支结构、SiTU-GLU 结论、Quantile Balancing 结论。

### Q4：RMSNorm 插在路由聚合之后、上投影之前，为什么这个位置是必要的？它解决了什么具体问题？

- **完成答案**：读者应能说明——聚合表示 $u$ 的尺度随选中专家与路由权重变化，原始 LatentMoE 直接把 $W_\uparrow$ 作用于 $u$，把这种尺度变化透传到与共享分支的相加；RMSNorm 把 $u$ 的尺度归一化后再交给 $W_\uparrow$，降低路由分支对尺度变化的敏感度，并附带改善验证损失与下游基准。
- **为什么是核心目标**：Normalized LatentMoE 是 K3 相对原始 LatentMoE 的核心结构改造，需要单独说清。
- **依赖内容**：Eq. 11 中 $u$ 与 $W_\uparrow$ 的位置、RMSNorm 的基本作用。

### Q5：Stable LatentMoE 的适用边界是什么？它依赖哪些独立概念，不能单独推出什么结论？

- **完成答案**：读者应能说明——Stable LatentMoE 的结论在 K3 的规模与配置下成立（$N=896, k=16, \ell=3584, d=7168$）；它依赖 SiTU-GLU、Quantile Balancing、RMSNorm 三个独立概念；它不推出"任何 MoE 都应该用 LatentMoE"、不推出"SiTU-GLU 一定优于 SwiGLU"、不推出"QB 一定优于辅助损失"，只是 K3 在其规模下的选择。
- **为什么是核心目标**：避免把单一模型的工程选择推广成一般性结论。
- **依赖内容**：三件稳定化的成立条件、K3 的具体配置。

## 4. 内容分级

### 4.1 核心内容（缺少后至少一个学习目标无法回答）

| 内容 | 服务的学习目标 | 必须讲清的结论 |
|---|---|---|
| LatentMoE 的动机：扩大专家池的流量代价 | Q1 | 全宽路由下通信与权重流量随路由倍数线性增长 |
| LatentMoE 的结构：$W_\downarrow$、$\ell$、路由专家、$W_\uparrow$、共享专家走全宽 | Q1, Q2 | Eq. 11 的两个分支与各自职责 |
| DeepSeekMoE 的 shared+routed 组织公式（引用） | Q2 | 共享专家无路由必经、路由专家 Top-k |
| Eq. 11 完整公式与符号 | Q2 | $x, z, u, y, W_\downarrow, W_\uparrow, E^{\text{shared}}, E^{\text{routed}}, p_i, T_k, N_s$ |
| 失败模式一：路由分支激活爆炸 | Q3 | 连续矩阵乘法链 + 2.8T 规模 → 激活爆炸 |
| 失败模式二：近千专家负载失衡 | Q3 | 无辅助损失方法的固定步长更新在大专家池下失效 |
| Normalized LatentMoE：RMSNorm 插入位置 | Q3, Q4 | 插在聚合后、上投影前 |
| SiTU-GLU 作为有界激活（引用） | Q3 | 上界 $\beta_1\beta_2 = 100$，近原点近似 Swish |
| Quantile Balancing 作为负载均衡（引用占位） | Q3 | 从路由分数分位数设置 bias |
| RMSNorm 位置的必要性：归一化 $u$ 的尺度 | Q4 | $u$ 的尺度随专家与权重变化，归一化后透传到 $W_\uparrow$ |
| 适用边界与依赖概念 | Q5 | 结论成立的规模条件、依赖的三个独立概念、不能推出的结论 |

### 4.2 辅助内容（消除关键理解障碍）

| 内容 | 服务的核心内容 |
|---|---|
| K3 的具体配置数字（$d=7168, \ell=3584, N=896, k=16, N_s=2$） | 让抽象结构与实际规模对应，避免空谈 |
| 稀疏度 $N/k = 56$ 的含义 | 说明"极端稀疏"到底有多稀疏 |
| 路由分支矩阵乘法链的图示 | 让"近四个连续矩阵乘法"具象化 |
| 原 LatentMoE 与 Normalized LatentMoE 的对照 | 说明 RMSNorm 插入前后的差异 |

### 4.3 扩展内容（不纳入本页范围）

| 内容 | 是否纳入 | 原因 |
|---|---|---|
| LatentMoE 原始论文 [32] 的损失景观分析 | 排除 | 不影响理解稳定化改造 |
| SiTU-GLU 的 softcap 推导与极限情形 | 排除 | 独立概念页 `wiki/situ-glu/` 已覆盖 |
| Quantile Balancing 的完整推导与直方图实现 | 排除 | 独立概念页 `wiki/quantile-balancing/`（占位）将覆盖 |
| K3 的训练超参数与实验结果 | 排除 | 不影响理解层结构 |

## 5. 前置知识映射

| 前置概念 | 被哪些学习目标依赖 | wiki/ 状态 | 递归深度 |
|---|---|---|---|
| MoE 基础（路由 + 专家 + Top-k） | Q1, Q2 | 已有：`wiki/moe-serving/index.html` | 0 |
| SiTU-GLU（有界激活） | Q3 | 已有：`wiki/situ-glu/index.html` | 0 |
| DeepSeekMoE（shared+routed 组织） | Q2 | 未生成，占位 | 1（登记不生成） |
| LatentMoE 原始架构 | Q1, Q2 | 未生成，本文内联最小必要结构 | 1（本文内联） |
| Quantile Balancing | Q3 | 未生成，占位 | 1（登记不生成） |
| RMSNorm | Q4 | 未生成，本文只讲插入位置与作用，不讲 RMSNorm 本身 | 1（登记不生成） |

注：DeepSeekMoE、LatentMoE、Quantile Balancing、RMSNorm 四个概念在 `wiki/` 下均无对应概念页。按 plan.md 第 2.4 节，递归深度上限为 2 层；本页自身是第 1 层（被 K3 总览页引用），其前置概念属第 2 层。但任务要求只生成 Stable LatentMoE 一页，前置缺失页面登记占位，正文保留阅读所需最小衔接，不内联大段背景。

## 6. 明确不展开的内容

| 不展开的内容 | 与概念的关系 | 不展开的原因 |
|---|---|---|
| LatentMoE 原始论文的训练动力学 | 是 Stable LatentMoE 的对照基线 | 不影响理解三件稳定化修复什么 |
| DeepSeekMoE 的细粒度专家切分 | K3 沿用 shared+routed 组织，未沿用切分 | 切分是独立贡献，与本页主线无关 |
| SiTU-GLU 的 softcap 极限与硬截断对比 | 是 SiTU-GLU 的内部细节 | 已在 `wiki/situ-glu/` 覆盖 |
| Quantile Balancing 的直方图估计实现 | 是 QB 在大规模下的工程实现 | 属独立概念页 `wiki/quantile-balancing/`（占位） |
| K3 的注意力结构（Block AttnRes / KDA） | 与 MoE 层并列，非 MoE 内部 | 属其他独立概念 |
| 专家并行与 All-to-All 通信 | 是 MoE 的工程部署 | 属 `wiki/moe-serving/` 与 `wiki/gpu-communication/` |

## 7. 常见误解和适用边界

### 7.1 常见误解

| 误解 | 正确结论 | 形成原因 | 影响目标 |
|---|---|---|---|
| "LatentMoE 就是用更窄的专家 FFN" | LatentMoE 是在路由分支两端加 $W_\downarrow$/$W_\uparrow$ 投影，让路由专家在隐空间 $\ell$ 操作；专家 FFN 本身的中间维度（`moe_intermediate_size=3072`）是另一回事 | 把"隐空间宽度"与"专家 FFN 中间维度"混为一谈 | Q2 |
| "RMSNorm 是 LayerNorm 的一种，所以位置无所谓" | 位置是 K3 的核心改造——必须在路由聚合之后、上投影之前，才能切断 $u$ 的尺度变化向 $W_\uparrow$ 的透传 | 把"RMSNorm 是什么"与"为什么要插在这里"混为一谈 | Q4 |
| "SiTU-GLU 一定优于 SwiGLU" | SiTU-GLU 是为了修复 LatentMoE 路由分支在 2.8T 规模下的激活爆炸；在其它架构/规模下 SwiGLU 仍是有效选择 | 把单一模型的工程选择推广成一般结论 | Q3, Q5 |
| "Quantile Balancing 是 K3 发明的全新负载均衡" | QB 是在 DeepSeek-V3 无辅助损失路由（bias 更新）基础上改进 bias 更新规则；不是从零发明 | 不了解 DeepSeek-V3 的 auxiliary-loss-free 路由 | Q3 |
| "896 个专家全部参与每个 token" | 每 token 只激活 $k=16$ 个路由专家 + 2 个共享专家；稀疏度 $N/k=56$ 表示总池/激活之比 | 把"专家总数"与"每 token 激活数"混为一谈 | Q2 |
| "共享专家也走隐空间" | 共享专家走全宽 $d$ 直接处理 $x$，不经过 $W_\downarrow$/$W_\uparrow$；只有路由分支走隐空间 | 没有区分两个分支 | Q2 |

### 7.2 适用边界

- **成立条件**：Stable LatentMoE 的结论在 K3 的规模与配置下成立——$N=896$ 路由专家、$k=16$ 每 token 激活、$\ell=3584$ 隐空间宽度、$d=7168$ 全宽、2.8T 参数、bfloat16 训练。
- **不解决什么**：不解决"如何选择 $N$ 与 $k$"、"如何选择 $\ell$"、"如何在更小规模下判断是否需要 LatentMoE"。
- **条件不满足时**：在更小规模或更小稀疏度下，普通 MoE 或 DeepSeekMoE 可能已足够稳定，三件稳定化的边际收益可能不抵额外开销；但本文不提供规模阈值，因为 K3 报告未给出。
- **不能推出**：不能从本文推出"任何 MoE 都应该用 LatentMoE"、"SiTU-GLU 一定优于 SwiGLU"、"QB 一定优于辅助损失"、"RMSNorm 插在任何位置都有效"。
