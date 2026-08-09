# LatentMoE 内容范围

## 1. 概念歧义处理

- "LatentMoE"目前在两处出现：
  - NVIDIA 论文（Elango et al., 2026, arXiv:2601.18089）提出的通用 LatentMoE 架构（"project to latent dimension ℓ for routing and computation, project back to d"）。
  - Kimi K3 技术报告（§2.3）的 Stable LatentMoE，是 LatentMoE + 三件稳定化（Normalized LatentMoE/RMSNorm、SiTU-GLU、Quantile Balancing）的合称。
- 两者主结构一致：路由专家放进紧凑隐空间，共享专家保留全宽。区别在于 K3 在此基础上加了三件稳定化以应对极端稀疏（N=896, k=16）下的两个失败模式。
- 裁定：本页讲 LatentMoE 这一通用架构概念（以 NVIDIA 论文为主源），K3 的 Stable LatentMoE 作为已生成的衍生概念页引用（`wiki/stable-latent-moe/`）。状态：已裁定。

## 2.1 概念含义

- 概念名称：LatentMoE（隐空间 MoE、潜在空间 MoE）
- 英文名称：Latent Mixture-of-Experts / LatentMoE
- 一句话定义：LatentMoE 是一种 MoE 架构，把路由专家的输入输出从模型全宽 $d$ 投影到更窄的隐空间 $\ell$（$\ell < d$）再计算，使扩大专家池时路由通信与专家权重流量不再随路由倍数线性增长。
- 正式定义：见 evidence.md C1，与 NVIDIA LatentMoE 论文 Figure 1(b) 与 §2 一致。
- 本文采用的语境：以 MoE 层为基本单位，讲清"全宽路径 vs 隐空间路径"的分工，以及 LatentMoE 与标准 MoE 的结构差异。
- 包括什么：
  - down-projection $W_\downarrow$、up-projection $W_\uparrow$ 与隐空间 $\ell$ 的引入（属于核心定义）。
  - 共享专家保留全宽 $d$、路由专家放进隐空间 $\ell$ 的分工（属于核心定义）。
  - 扩大专家池时通信与显存带宽随路由倍数线性增长的问题，以及 LatentMoE 如何缓解（属于动机）。
  - reinvestment 策略：把节省的预算投入更多专家与更高 top-k（属于核心机制）。
  - 压缩比 $d/\ell$ 与"特征秩下限"边界（属于边界）。
- 不包括什么：
  - Stable LatentMoE 的三件稳定化（RMSNorm 位置、SiTU-GLU、Quantile Balancing）——属于 `wiki/stable-latent-moe/`，本页只引用。
  - 路由打分细节（sigmoid/softmax、aux-loss-free 偏置）——属于 `wiki/deepseek-moe/` 与 `wiki/moe-serving/`。
  - MoE 服务部署的工程细节（EP、dispatch/combine、TBO/SBO）——属于 `wiki/moe-serving/`。
  - 训练侧负载均衡损失的完整推导——属于训练侧，不影响理解架构本身。
- 相邻概念：
  - 标准 top-K MoE：LatentMoE 的改造对象，本页第 1 章简述其"全宽路由"特征后引出问题，完整机制见 `wiki/moe-serving/`。
  - DeepSeekMoE 的 shared+routed 组织：LatentMoE 沿用这一组织，本页引用 `wiki/deepseek-moe/`。
  - Stable LatentMoE：LatentMoE + 稳定化，本页引用 `wiki/stable-latent-moe/`。

## 2.2 学习目标

### Q1：标准 MoE 在扩大专家池时，为什么路由通信与专家权重流量会随路由倍数线性增长？LatentMoE 用什么结构变化缓解？

- 完成答案：读者应能说明标准 MoE 中每个被选中的路由专家都接收完整 $d$ 维 token、专家权重也在 $d$ 维空间，因此 top-k 与专家数 $N$ 增大时 dispatch 通信与专家权重读取量同步增长；LatentMoE 在路由分支两端加 $W_\downarrow$（$d \to \ell$）和 $W_\uparrow$（$\ell \to d$），把路由专家的输入输出与专家权重都放进 $\ell$ 维隐空间，路由部分的开销不再随 $d$ 走，而是随更小的 $\ell$ 走。
- 为什么是核心目标：不理解这个动机，就无法判断 LatentMoE 何时有用、何时不必要。
- 依赖内容：标准 MoE 的 top-k 与加权和结构、dispatch/combine 通信、显存带宽与专家权重读取。

### Q2：LatentMoE 层的共享分支与路由分支各自走什么宽度、经过哪些投影？写出完整层公式。

- 完成答案：读者应能写出 $u = \sum_{i \in T_k(x)} p_i E_i^{\text{routed}}(W_\downarrow x)$，$y = \sum_j E_j^{\text{shared}}(x) + W_\uparrow u$（K3 在 $W_\uparrow$ 前加 RMSNorm 是稳定化，本页只引用）；指出共享专家输入输出都在 $d$、不经过 $W_\downarrow$/$W_\uparrow$，路由专家输入输出都在 $\ell$。
- 为什么是核心目标：这是 LatentMoE 的核心机制，后续讨论都依赖它。
- 依赖内容：shared+routed 组织（来自 DeepSeekMoE）、矩阵投影、加权和。

### Q3：LatentMoE 把路由宽度从 $d$ 压到 $\ell$ 后，节省的开销以什么比例缩小？这笔节省通常被 reinvest 到哪里？

- 完成答案：读者应能说明路由 dispatch 通信量与路由专家权重大小都大致按 $d/\ell$ 比例缩小；节省的预算通常被同时投入到更大的专家数 $N$ 与更高的 top-k（按同一因子放大），以提升专家组合多样性，而不显著抬高服务成本。手算 $d=4096, \ell=1024$ 时 $d/\ell = 4$ 的含义。
- 为什么是核心目标：这是 LatentMoE 区别于"单纯压缩"的关键——它不是让模型更小，而是把同样的服务预算换成更多专家。
- 依赖内容：压缩比、组合数 $\binom{N}{k}$、服务成本（通信 + 显存带宽，不是 FLOPs）。

### Q4：LatentMoE 的压缩比 $d/\ell$ 能不能无限放大？受什么约束？

- 完成答案：读者应能指出存在任务相关的"特征秩"下限，把 $\ell$ 压到低于这个下限会损害模型质量；同时 down/up-projection 本身有计算成本，压缩比不等于整体加速比。NVIDIA 实验显示压缩比到 4 时质量仍可保持，再压缩需配合 reinvestment 才不损失。
- 为什么是核心目标：避免读者把 LatentMoE 误读成"压得越窄越好"。
- 依赖内容：特征秩、投影计算成本、reinvestment 的必要性。

### Q5：LatentMoE 与 Stable LatentMoE 是什么关系？

- 完成答案：读者应能说明 Stable LatentMoE（K3）= LatentMoE 主结构 + 三件稳定化（Normalized LatentMoE 在 $W_\uparrow$ 前加 RMSNorm、SiTU-GLU 有界激活、Quantile Balancing 负载均衡），用于在 N=896, k=16 的极端稀疏与 2.8T 规模下保持训练稳定；稳定化的完整机制见 `wiki/stable-latent-moe/`。
- 为什么是核心目标：澄清两者关系，避免读者以为本页要重复 Stable LatentMoE 页的内容。
- 依赖内容：LatentMoE 主结构、K3 的配置数字、稳定化要解决的问题。

## 2.3 内容分级

### 核心内容（缺一不可，对应学习目标）

- 标准 MoE 的全宽路由开销（对应 Q1）：必须讲清 dispatch 通信与专家权重读取都随 top-k 与 $d$ 线性增长。
- LatentMoE 的层结构与公式（对应 Q2）：必须给出 $W_\downarrow$、$W_\uparrow$、共享/路由分支的分工与完整公式。
- 压缩比 $d/\ell$ 与 reinvestment（对应 Q3）：必须给出 $d/\ell$ 比例的含义与 reinvestment 策略（同时放大 $N$ 与 $k$）。
- 压缩下限与投影成本（对应 Q4）：必须讲清特征秩下限与投影自身成本。
- 与 Stable LatentMoE 的关系（对应 Q5）：必须点明继承与叠加关系。

### 辅助内容（消除关键理解障碍）

- 标准 MoE 的"每 token 激活 $k$ 个、加权求和"回顾：从 `wiki/moe-serving/` 引用，不重复推导。
- shared+routed 组织回顾：从 `wiki/deepseek-moe/` 引用。
- 一个可手算的小例子：展示 $d/\ell = 2$ 时通信与权重流量的比例。

### 扩展内容（不影响学习目标）

- NVIDIA LatentMoE 论文的五条设计原则：作为来源支撑，正文点出与核心机制直接相关的几条，不全文复述。
- Nemotron-3 Super/Ultra 的具体配置：作为真实数字对照，放在来源说明或折叠块。
- 与 MLA（多头潜在注意力）的"投影到潜在空间"思想类比：本页不展开，仅点一句类比关系。

## 2.4 前置知识映射

- **标准 top-K MoE**（router、top-k、加权和）：已有概念页 `wiki/moe-serving/index.html`，正文首次引用。被 Q1、Q2 依赖。
- **DeepSeekMoE 的 shared+routed 组织**（共享专家恒激活、路由专家 top-k）：已有概念页 `wiki/deepseek-moe/index.html`，正文首次引用。被 Q2 依赖。
- **Stable LatentMoE**（K3 的稳定化版本）：已有概念页 `wiki/stable-latent-moe/index.html`，Q5 引用。注意：stable-latent-moe 页面已讲了 LatentMoE 的基本结构（$W_\downarrow$、$W_\uparrow$、隐空间 $\ell$），本页是更基础的通用概念页，避免重复——本页讲通用动机与 reinvestment，stable-latent-moe 讲 K3 的具体稳定化。
- **MoE 服务部署**（EP、dispatch/combine、all-to-all、显存带宽）：已有概念页 `wiki/moe-serving/index.html`，Q1 引用以说明"通信与显存带宽"指什么。

递归深度：本页所有前置知识均已有概念页，无需递归生成。

## 2.5 明确不展开的内容

- Stable LatentMoE 的三件稳定化机制（RMSNorm 位置推导、SiTU-GLU 的 softcap、Quantile Balancing 的对偶线性规划）：属于 `wiki/stable-latent-moe/`，本页只在 Q5 点明继承关系，不展开。
- 路由打分的具体实现（sigmoid/softmax、aux-loss-free 偏置）：属于 `wiki/deepseek-moe/` 与 `wiki/moe-serving/`，本页只用"router 给出门控权重 $p_i$"的抽象形式。
- MoE 服务部署的工程细节（TBO/SBO、PD 分离、goodput）：属于 `wiki/moe-serving/`，本页只在 Q1 用"dispatch 通信与专家权重读取"这一最小表述。
- 训练侧负载均衡损失的完整推导：不影响理解 LatentMoE 架构本身。
- MoE 训练动态与专家专门化的实证分析：超出本页范围。

## 2.6 常见误解和适用边界

### 常见误解

1. **误解**：LatentMoE 就是把整个模型压窄。
   **正确**：只压窄路由分支；共享专家与 router 仍在全宽 $d$ 工作。
   **形成原因**：名称里的"latent"容易被理解为"整个隐空间都压窄"。
   **影响**：Q2。

2. **误解**：压缩比 $d/\ell = 4$ 意味着整个 MoE 层快 4 倍。
   **正确**：只压缩了路由部分的通信与权重流量；down/up-projection 本身有计算成本，整个 MoE 层的加速比小于 $d/\ell$。
   **形成原因**：把局部压缩比当成整体加速比。
   **影响**：Q3、Q4。

3. **误解**：LatentMoE 是 K3 发明的。
   **正确**：LatentMoE 作为通用架构由 NVIDIA 论文（Elango et al., 2026）系统提出；K3 的 Stable LatentMoE 是在此基础上的稳定化变体。
   **形成原因**：K3 报告里的 Stable LatentMoE 名字更响，容易混淆。
   **影响**：Q5。

4. **误解**：压得越窄越好。
   **正确**：存在任务相关的特征秩下限，低于此下限会损害质量；reinvestment（同时增加 $N$ 与 $k$）是维持质量的关键，单纯压缩会掉点。
   **形成原因**：只看到压缩带来的节省，没看到质量代价。
   **影响**：Q4。

### 适用边界

- LatentMoE 解决的问题：路由部分随 top-k 与 $d$ 线性增长的通信与显存带宽开销。
- LatentMoE 不解决的问题：路由器本身的打分计算（仍在 $d$ 维）、共享专家的开销（仍在 $d$ 维）、训练侧负载均衡。
- 成立条件：$\ell$ 不低于任务特征秩下限；reinvestment 同时放大 $N$ 与 $k$ 以维持非线性容量。
- 条件不满足时：$\ell$ 过低会损害质量；不 reinvestment 只压缩会让模型容量下降。
- 不推出的结论：LatentMoE 不保证在所有任务上都比标准 MoE 更好——收益依赖任务特征秩与 reinvestment 配置。
